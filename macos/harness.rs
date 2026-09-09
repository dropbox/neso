#[cfg(target_os = "macos")]
mod macos_harness {
    use candle_metal_kernels::metal::{
        create_command_buffer, Buffer, CommandQueue, CommandSemaphore, ComputePipeline, Device,
    };
    use candle_metal_kernels::RESOURCE_OPTIONS;
    use half::f16;
    use objc2_metal::MTLSize;
    use std::error::Error;
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const BLOCK_SIZE: usize = 256;
    const FA2_BLOCK_M: usize = 8;
    const FA2_HEAD_DIM: usize = 64;
    #[cfg(target_arch = "aarch64")]
    const FA2_THREADS: usize = 256;
    #[cfg(target_arch = "x86_64")]
    const FA2_THREADS: usize = 416;
    const VECTOR_ADD_METALLIB: &[u8] = include_bytes!(concat!(
        env!("NESO_MACOS_KERNEL_DIR"),
        "/vector_add.metallib"
    ));
    const SCALE_METALLIB: &[u8] =
        include_bytes!(concat!(env!("NESO_MACOS_KERNEL_DIR"), "/scale.metallib"));
    #[cfg(target_arch = "aarch64")]
    const FLASH_ATTENTION_METALLIB: &[u8] = include_bytes!(concat!(
        env!("NESO_MACOS_KERNEL_DIR"),
        "/flash_attention_fwd_simd.metallib"
    ));
    #[cfg(target_arch = "x86_64")]
    const FLASH_ATTENTION_METALLIB: &[u8] = include_bytes!(concat!(
        env!("NESO_MACOS_KERNEL_DIR"),
        "/flash_attention_fwd_scalar.metallib"
    ));

    fn upload(device: &Device, values: &[f32]) -> Result<Buffer, Box<dyn Error>> {
        Ok(device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(values),
            RESOURCE_OPTIONS,
        )?)
    }

    fn upload_f16(device: &Device, values: &[f32]) -> Result<Buffer, Box<dyn Error>> {
        let values: Vec<f16> = values.iter().copied().map(f16::from_f32).collect();
        Ok(device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(values.as_slice()),
            RESOURCE_OPTIONS,
        )?)
    }

    fn allocate(device: &Device, count: usize) -> Result<Buffer, Box<dyn Error>> {
        Ok(device.new_buffer(count * size_of::<f32>(), RESOURCE_OPTIONS)?)
    }

    fn download(buffer: &Buffer, count: usize) -> Vec<f32> {
        let values = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count) };
        values.to_vec()
    }

    fn pipeline(
        device: &Device,
        metallib: &[u8],
        name: &str,
    ) -> Result<ComputePipeline, Box<dyn Error>> {
        let library = device.new_library_with_data(metallib)?;
        let function = library.get_function(name, None)?;
        Ok(device.new_compute_pipeline_state_with_function(&function)?)
    }

    fn groups(count: usize) -> usize {
        count.div_ceil(BLOCK_SIZE)
    }

    fn dispatch(
        queue: &CommandQueue,
        pipeline: &ComputePipeline,
        buffers: &[&Buffer],
        scalars: &[(usize, &[u8])],
        grid_x: usize,
        threads_per_group: usize,
    ) -> Result<(), Box<dyn Error>> {
        let command_buffer = create_command_buffer(queue, Arc::new(CommandSemaphore::new()))?;
        {
            let encoder = command_buffer.compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                encoder.set_buffer(index, Some(buffer), 0);
            }
            for (index, bytes) in scalars {
                encoder.set_bytes_directly(*index, bytes.len(), bytes.as_ptr().cast());
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: grid_x,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group,
                    height: 1,
                    depth: 1,
                },
            );
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if let Some(error) = command_buffer.error() {
            return Err(error.into_owned().into());
        }
        Ok(())
    }

    fn dispatch_fa2(
        queue: &CommandQueue,
        pipeline: &ComputePipeline,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        output: &Buffer,
        sequence_length: usize,
    ) -> Result<(), Box<dyn Error>> {
        let n = (sequence_length as i32).to_ne_bytes();
        let stride = (FA2_HEAD_DIM as i32).to_ne_bytes();
        let scale = (1.0 / (FA2_HEAD_DIM as f32).sqrt()).to_ne_bytes();
        dispatch(
            queue,
            pipeline,
            &[q, k, v, output],
            &[(4, &n), (5, &stride), (6, &stride), (7, &scale)],
            sequence_length / FA2_BLOCK_M,
            FA2_THREADS,
        )
    }

    fn fa2_values(sequence_length: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let count = sequence_length * FA2_HEAD_DIM;
        let q = (0..count)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
            .collect();
        let k = (0..count)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
            .collect();
        let v = (0..count)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        (q, k, v)
    }

    fn check_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        let (index, max_error) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            max_error <= tolerance,
            "mismatch at {index}: got {}, expected {}, error {max_error}",
            actual[index],
            expected[index]
        );
    }

    fn test() -> Result<(), Box<dyn Error>> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let queue = device.new_command_queue()?;
        let add = pipeline(&device, VECTOR_ADD_METALLIB, "vector_add")?;
        let scale = pipeline(&device, SCALE_METALLIB, "scale")?;
        let fa2 = pipeline(&device, FLASH_ATTENTION_METALLIB, "flash_attention_fwd")?;
        let count = 65_537usize;
        let count_bytes = (count as i32).to_ne_bytes();
        let x: Vec<f32> = (0..count).map(|i| (i as f32 - 1000.0) * 0.125).collect();
        let y: Vec<f32> = (0..count).map(|i| (i % 97) as f32 * -0.25).collect();
        let x_gpu = upload(&device, &x)?;
        let y_gpu = upload(&device, &y)?;
        let output = allocate(&device, count)?;

        dispatch(
            &queue,
            &add,
            &[&x_gpu, &y_gpu, &output],
            &[(3, &count_bytes)],
            groups(count),
            BLOCK_SIZE,
        )?;
        let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + y).collect();
        check_close(&download(&output, count), &expected, 1e-6);
        println!("PASS vector_add ({count} elements, masked tail)");

        let factor = -1.75f32;
        let factor_bytes = factor.to_ne_bytes();
        dispatch(
            &queue,
            &scale,
            &[&x_gpu, &output],
            &[(2, &factor_bytes), (3, &count_bytes)],
            groups(count),
            BLOCK_SIZE,
        )?;
        let expected: Vec<f32> = x.iter().map(|x| x * factor).collect();
        check_close(&download(&output, count), &expected, 1e-6);
        println!("PASS scale ({count} elements, scalar argument)");

        let sequence_length = 64;
        let fa2_count = sequence_length * FA2_HEAD_DIM;
        let q = vec![0.0; fa2_count];
        let k = vec![0.0; fa2_count];
        let (_, _, v) = fa2_values(sequence_length);
        let q_gpu = upload_f16(&device, &q)?;
        let k_gpu = upload_f16(&device, &k)?;
        let v_gpu = upload_f16(&device, &v)?;
        let fa2_output = allocate(&device, fa2_count)?;
        dispatch_fa2(
            &queue,
            &fa2,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &fa2_output,
            sequence_length,
        )?;
        let means: Vec<f32> = (0..FA2_HEAD_DIM)
            .map(|column| {
                (0..sequence_length)
                    .map(|row| f16::from_f32(v[row * FA2_HEAD_DIM + column]).to_f32())
                    .sum::<f32>()
                    / sequence_length as f32
            })
            .collect();
        let expected: Vec<f32> = (0..sequence_length)
            .flat_map(|_| means.iter().copied())
            .collect();
        check_close(&download(&fa2_output, fa2_count), &expected, 2e-3);
        println!("PASS flash_attention_2 (f16, N={sequence_length}, d={FA2_HEAD_DIM})");
        Ok(())
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn measure(
        mut dispatch: impl FnMut() -> Result<(), Box<dyn Error>>,
        iterations: usize,
    ) -> Result<Duration, Box<dyn Error>> {
        for _ in 0..10 {
            dispatch()?;
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            dispatch()?;
            samples.push(start.elapsed());
        }
        Ok(median(samples))
    }

    fn bench() -> Result<(), Box<dyn Error>> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let queue = device.new_command_queue()?;
        let add = pipeline(&device, VECTOR_ADD_METALLIB, "vector_add")?;
        let scale = pipeline(&device, SCALE_METALLIB, "scale")?;
        let fa2 = pipeline(&device, FLASH_ATTENTION_METALLIB, "flash_attention_fwd")?;
        let count = 16 * 1024 * 1024usize;
        let count_bytes = (count as i32).to_ne_bytes();
        let x = upload(&device, &vec![1.0; count])?;
        let y = upload(&device, &vec![2.0; count])?;
        let output = allocate(&device, count)?;
        let iterations = 50;

        let add_time = measure(
            || {
                dispatch(
                    &queue,
                    &add,
                    &[&x, &y, &output],
                    &[(3, &count_bytes)],
                    groups(count),
                    BLOCK_SIZE,
                )
            },
            iterations,
        )?;
        println!(
            "vector_add  {:8.3} ms  {:8.2} GB/s",
            add_time.as_secs_f64() * 1e3,
            count as f64 * 12.0 / add_time.as_secs_f64() / 1e9
        );

        let factor_bytes = 1.25f32.to_ne_bytes();
        let scale_time = measure(
            || {
                dispatch(
                    &queue,
                    &scale,
                    &[&x, &output],
                    &[(2, &factor_bytes), (3, &count_bytes)],
                    groups(count),
                    BLOCK_SIZE,
                )
            },
            iterations,
        )?;
        println!(
            "scale       {:8.3} ms  {:8.2} GB/s",
            scale_time.as_secs_f64() * 1e3,
            count as f64 * 8.0 / scale_time.as_secs_f64() / 1e9
        );

        let max_sequence_length = 512;
        let (q, k, v) = fa2_values(max_sequence_length);
        let q = upload_f16(&device, &q)?;
        let k = upload_f16(&device, &k)?;
        let v = upload_f16(&device, &v)?;
        let fa2_output = allocate(&device, max_sequence_length * FA2_HEAD_DIM)?;
        for sequence_length in [128usize, 256, 512] {
            let fa2_time = measure(
                || dispatch_fa2(&queue, &fa2, &q, &k, &v, &fa2_output, sequence_length),
                20,
            )?;
            let flops = 4.0 * sequence_length as f64 * sequence_length as f64 * FA2_HEAD_DIM as f64;
            println!(
                "flash_attn2 f16 N={sequence_length:4} d={FA2_HEAD_DIM}  {:8.3} ms  {:8.2} GFLOP/s",
                fa2_time.as_secs_f64() * 1e3,
                flops / fa2_time.as_secs_f64() / 1e9
            );
        }
        Ok(())
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        match std::env::args().nth(1).as_deref() {
            Some("test") => test(),
            Some("bench") => bench(),
            _ => Err("usage: neso-macos-kernels <test|bench>".into()),
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos_harness::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Neso Metal kernel harness only runs on macOS");
    std::process::exit(1);
}
