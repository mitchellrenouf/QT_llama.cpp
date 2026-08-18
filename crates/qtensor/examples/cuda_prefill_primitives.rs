#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use qtensor::cuda::{CudaBuffer, CudaDevice};
    use qtensor::quant::f32_to_f16;
    use std::time::Instant;

    const COLS: usize = 32;
    const ROWS: usize = 64;
    const BATCH: usize = 128;
    let device = CudaDevice::new(0)?;

    // Each Q4 block has scale 1 and alternating exactly representable values.
    let mut weights = vec![0u8; ROWS * 18];
    for row in 0..ROWS {
        let offset = row * 18;
        weights[offset..offset + 2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let packed = if row % 2 == 0 { 0x98 } else { 0x87 };
        weights[offset + 2..offset + 18].fill(packed);
    }
    let input: Vec<f32> = (0..BATCH * COLS)
        .map(|index| ((index % COLS) as f32 - 15.5) / 16.0)
        .collect();
    let d_weights = CudaBuffer::from_host(&weights)?;
    let d_input = CudaBuffer::from_host(&input)?;
    let mut d_output = CudaBuffer::alloc(BATCH * ROWS)?;

    device.gemm_q4_0(&d_weights, &d_input, &mut d_output, ROWS, COLS, BATCH);
    device.sync()?;
    let mut output = vec![0.0f32; BATCH * ROWS];
    d_output.copy_to_host(&mut output)?;
    for token in 0..BATCH {
        let values = &input[token * COLS..(token + 1) * COLS];
        let even_expected: f32 = values[16..].iter().sum();
        let odd_expected = -values[..16].iter().sum::<f32>();
        assert!((output[token * ROWS] - even_expected).abs() < 1e-4);
        assert!((output[token * ROWS + 1] - odd_expected).abs() < 1e-4);
    }

    let iterations = 1_000;
    let started = Instant::now();
    for _ in 0..iterations {
        device.gemm_q4_0(&d_weights, &d_input, &mut d_output, ROWS, COLS, BATCH);
    }
    device.sync()?;
    let batched = started.elapsed();

    let mut d_single_in = CudaBuffer::alloc(COLS)?;
    let mut d_single_out = CudaBuffer::alloc(ROWS)?;
    let started = Instant::now();
    for _ in 0..iterations {
        for token in 0..BATCH {
            device.copy_from_host_async(
                &mut d_single_in,
                &input[token * COLS..(token + 1) * COLS],
            )?;
            device.gemv_q4_0(&d_weights, &d_single_in, &mut d_single_out, ROWS, COLS);
        }
    }
    device.sync()?;
    let sequential = started.elapsed();

    println!(
        "batch={BATCH} iterations={iterations} batched_ms={:.3} sequential_ms={:.3} speedup={:.2}x",
        batched.as_secs_f64() * 1_000.0,
        sequential.as_secs_f64() * 1_000.0,
        sequential.as_secs_f64() / batched.as_secs_f64(),
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_prefill_primitives requires --features cuda");
}
