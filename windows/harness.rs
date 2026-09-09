#[cfg(target_os = "windows")]
mod windows_harness {
    use candle_d3d12_kernels::{BufferBinding, Gpu, GpuBuffer};
    use std::error::Error;
    use std::time::{Duration, Instant};

    const BLOCK_SIZE: u32 = 256;
    const VECTOR_ADD_DXIL: &[u8] =
        include_bytes!(concat!(env!("NESO_WINDOWS_KERNEL_DIR"), "/vector_add.dxil"));
    const SCALE_DXIL: &[u8] =
        include_bytes!(concat!(env!("NESO_WINDOWS_KERNEL_DIR"), "/scale.dxil"));

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
