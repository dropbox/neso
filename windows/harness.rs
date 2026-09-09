#[cfg(target_os = "windows")]
mod windows_harness {
    use candle_d3d12_kernels::{BufferBinding, Gpu, GpuBuffer, ID3D12PipelineState};
    use half::f16;
    use std::error::Error;
    use std::time::{Duration, Instant};

    const BLOCK_SIZE: u32 = 256;
    const FA2_BLOCK_M: usize = 8;
    const FA2_HEAD_DIM: usize = 64;
    const VECTOR_ADD_DXIL: &[u8] =
        include_bytes!(concat!(env!("NESO_WINDOWS_KERNEL_DIR"), "/vector_add.dxil"));
    const SCALE_DXIL: &[u8] =
        include_bytes!(concat!(env!("NESO_WINDOWS_KERNEL_DIR"), "/scale.dxil"));
    const FLASH_ATTENTION_DXIL: &[u8] = include_bytes!(concat!(
        env!("NESO_WINDOWS_KERNEL_DIR"),
        "/flash_attention_fwd.dxil"
    ));

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn upload(gpu: &Gpu, values: &[f32]) -> Result<GpuBuffer, Box<dyn Error>> {
        let buffer = gpu.create_buffer((values.len() * 4) as u64)?;
        gpu.upload_to_buffer(&bytes(values), &buffer)?;
        Ok(buffer)
    }

    fn upload_f16(gpu: &Gpu, values: &[f32]) -> Result<GpuBuffer, Box<dyn Error>> {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let buffer = gpu.create_buffer(data.len() as u64)?;
        gpu.upload_to_buffer(&data, &buffer)?;
        Ok(buffer)
    }

    fn download(gpu: &Gpu, buffer: &GpuBuffer, count: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        let data = gpu.download_buffer(buffer, (count * 4) as u64)?;
        Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn groups(count: usize) -> u32 {
        (count as u32).div_ceil(BLOCK_SIZE)
    }

    fn dispatch_fa2(
        gpu: &Gpu,
        pipeline: &ID3D12PipelineState,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        output: &GpuBuffer,
        buffer_count: usize,
        sequence_length: usize,
    ) -> Result<(), Box<dyn Error>> {
        let grid_x = sequence_length / FA2_BLOCK_M;
        gpu.dispatch_uav_only(
            pipeline,
            &[
                sequence_length as u32,
                FA2_HEAD_DIM as u32,
                FA2_HEAD_DIM as u32,
                (1.0 / (FA2_HEAD_DIM as f32).sqrt()).to_bits(),
                grid_x as u32,
                1,
                1,
            ],
            &[
                BufferBinding::structured_f16(q, buffer_count as u32),
                BufferBinding::structured_f16(k, buffer_count as u32),
                BufferBinding::structured_f16(v, buffer_count as u32),
                BufferBinding::structured_f32(output, buffer_count as u32),
            ],
            [grid_x as u32, 1, 1],
        )?;
        Ok(())
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
        let gpu = Gpu::new(0)?;
        let add = gpu.create_compute_pso(VECTOR_ADD_DXIL)?;
        let scale = gpu.create_compute_pso(SCALE_DXIL)?;
        let fa2 = gpu.create_compute_pso(FLASH_ATTENTION_DXIL)?;

        // Deliberately not divisible by BLOCK_SIZE, exercising the generated mask.
        let count = 65_537usize;
        let x: Vec<f32> = (0..count).map(|i| (i as f32 - 1000.0) * 0.125).collect();
        let y: Vec<f32> = (0..count).map(|i| (i % 97) as f32 * -0.25).collect();
        let x_gpu = upload(&gpu, &x)?;
        let y_gpu = upload(&gpu, &y)?;
        let output = gpu.create_buffer((count * 4) as u64)?;
        let dispatch_groups = groups(count);

        gpu.dispatch_uav_only(
            &add,
            &[count as u32, dispatch_groups, 1, 1],
            &[
                BufferBinding::structured_f32(&x_gpu, count as u32),
                BufferBinding::structured_f32(&y_gpu, count as u32),
                BufferBinding::structured_f32(&output, count as u32),
            ],
            [dispatch_groups, 1, 1],
        )?;
        let actual = download(&gpu, &output, count)?;
        let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + y).collect();
        check_close(&actual, &expected, 1e-6);
        println!("PASS vector_add ({count} elements, masked tail)");

        let factor = -1.75f32;
        gpu.dispatch_uav_only(
            &scale,
            &[factor.to_bits(), count as u32, dispatch_groups, 1, 1],
            &[
                BufferBinding::structured_f32(&x_gpu, count as u32),
                BufferBinding::structured_f32(&output, count as u32),
            ],
            [dispatch_groups, 1, 1],
        )?;
        let actual = download(&gpu, &output, count)?;
        let expected: Vec<f32> = x.iter().map(|x| x * factor).collect();
        check_close(&actual, &expected, 1e-6);
        println!("PASS scale ({count} elements, scalar argument)");

        let sequence_length = 64;
        let fa2_count = sequence_length * FA2_HEAD_DIM;
        let q = vec![0.0; fa2_count];
        let k = vec![0.0; fa2_count];
        let (_, _, v) = fa2_values(sequence_length);
        let q_gpu = upload_f16(&gpu, &q)?;
        let k_gpu = upload_f16(&gpu, &k)?;
        let v_gpu = upload_f16(&gpu, &v)?;
        let fa2_output = gpu.create_buffer((fa2_count * 4) as u64)?;
        dispatch_fa2(
            &gpu,
            &fa2,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &fa2_output,
            fa2_count,
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
        check_close(&download(&gpu, &fa2_output, fa2_count)?, &expected, 2e-3);
        println!("PASS flash_attention_2 (f16, N={sequence_length}, d={FA2_HEAD_DIM})");
        Ok(())
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn measure(mut dispatch: impl FnMut(), iterations: usize) -> Duration {
        for _ in 0..10 {
            dispatch();
        }
        median(
            (0..iterations)
                .map(|_| {
                    let start = Instant::now();
                    dispatch();
                    start.elapsed()
                })
                .collect(),
        )
    }

    fn bench() -> Result<(), Box<dyn Error>> {
        let gpu = Gpu::new(0)?;
        let add = gpu.create_compute_pso(VECTOR_ADD_DXIL)?;
        let scale = gpu.create_compute_pso(SCALE_DXIL)?;
        let fa2 = gpu.create_compute_pso(FLASH_ATTENTION_DXIL)?;
        let count = 16 * 1024 * 1024usize;
        let dispatch_groups = groups(count);
        let x = upload(&gpu, &vec![1.0; count])?;
        let y = upload(&gpu, &vec![2.0; count])?;
        let output = gpu.create_buffer((count * 4) as u64)?;
        let iterations = 50;

        let add_time = measure(
            || {
                gpu.dispatch_uav_only(
                    &add,
                    &[count as u32, dispatch_groups, 1, 1],
                    &[
                        BufferBinding::structured_f32(&x, count as u32),
                        BufferBinding::structured_f32(&y, count as u32),
                        BufferBinding::structured_f32(&output, count as u32),
                    ],
                    [dispatch_groups, 1, 1],
                )
                .unwrap();
            },
            iterations,
        );
        let add_gbps = count as f64 * 12.0 / add_time.as_secs_f64() / 1e9;
        println!(
            "vector_add  {:8.3} ms  {:8.2} GB/s",
            add_time.as_secs_f64() * 1e3,
            add_gbps
        );

        let factor = 1.25f32;
        let scale_time = measure(
            || {
                gpu.dispatch_uav_only(
                    &scale,
                    &[factor.to_bits(), count as u32, dispatch_groups, 1, 1],
                    &[
                        BufferBinding::structured_f32(&x, count as u32),
                        BufferBinding::structured_f32(&output, count as u32),
                    ],
                    [dispatch_groups, 1, 1],
                )
                .unwrap();
            },
            iterations,
        );
        let scale_gbps = count as f64 * 8.0 / scale_time.as_secs_f64() / 1e9;
        println!(
            "scale       {:8.3} ms  {:8.2} GB/s",
            scale_time.as_secs_f64() * 1e3,
            scale_gbps
        );

        let max_sequence_length = 2048;
        let buffer_count = max_sequence_length * FA2_HEAD_DIM;
        let (q, k, v) = fa2_values(max_sequence_length);
        let q = upload_f16(&gpu, &q)?;
        let k = upload_f16(&gpu, &k)?;
        let v = upload_f16(&gpu, &v)?;
        let fa2_output = gpu.create_buffer((buffer_count * 4) as u64)?;
        for sequence_length in [128usize, 256, 512, 1024, 2048] {
            let fa2_time = measure(
                || {
                    dispatch_fa2(
                        &gpu,
                        &fa2,
                        &q,
                        &k,
                        &v,
                        &fa2_output,
                        buffer_count,
                        sequence_length,
                    )
                    .unwrap();
                },
                20,
            );
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
            _ => Err("usage: neso-windows-kernels <test|bench>".into()),
        }
    }
}

#[cfg(target_os = "windows")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    windows_harness::main()
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("the Neso kernel harness only runs on Windows");
    std::process::exit(1);
}
