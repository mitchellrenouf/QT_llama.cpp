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
    let mut d_single_in = CudaBuffer::alloc(COLS)?;

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

    // Fused batched Q/K/V projection must match the established single-token
    // CUDA path before it is allowed into model prefill.
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const Q_ROWS: usize = HEADS * HEAD_DIM;
    const KV_ROWS: usize = KV_HEADS * HEAD_DIM;
    let mut d_qkv = CudaBuffer::alloc(BATCH * (Q_ROWS + 2 * KV_ROWS))?;
    device.gemm_q4_0_qkv(
        &d_weights, &d_weights, &d_weights, &d_input, &mut d_qkv,
        Q_ROWS, KV_ROWS, COLS, BATCH,
    );
    device.sync()?;
    let mut qkv = vec![0.0f32; d_qkv.len()];
    d_qkv.copy_to_host(&mut qkv)?;
    let mut d_single_qkv = CudaBuffer::alloc(Q_ROWS + 2 * KV_ROWS)?;
    for token in 0..BATCH {
        device.copy_from_host_async(
            &mut d_single_in,
            &input[token * COLS..(token + 1) * COLS],
        )?;
        device.gemv_q4_0_qkv(
            &d_weights, &d_weights, &d_weights, &d_single_in,
            &mut d_single_qkv, Q_ROWS, KV_ROWS, COLS,
        );
        device.sync()?;
        let mut expected = vec![0.0f32; Q_ROWS + 2 * KV_ROWS];
        d_single_qkv.copy_to_host(&mut expected)?;
        let actual = &qkv[token * expected.len()..(token + 1) * expected.len()];
        assert!(actual.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    let norm = vec![1.0f32; HEAD_DIM];
    let d_norm = CudaBuffer::from_host(&norm)?;
    let mut d_k_cache = CudaBuffer::alloc(BATCH * KV_ROWS)?;
    let mut d_v_cache = CudaBuffer::alloc(BATCH * KV_ROWS)?;
    device.qkv_postprocess_batch(
        &mut d_qkv, &d_norm, &d_norm, &mut d_k_cache, &mut d_v_cache,
        0, 0, HEADS, KV_HEADS, HEAD_DIM, 10_000.0, BATCH,
    );
    let mut d_attention = CudaBuffer::alloc(BATCH * Q_ROWS)?;
    device.attention_prefill(
        &d_qkv, &d_k_cache, &d_v_cache, &mut d_attention, 0, BATCH,
        HEADS, KV_HEADS, HEAD_DIM, 1.0 / (HEAD_DIM as f32).sqrt(), None,
    );
    device.sync()?;
    let mut attention = vec![0.0f32; d_attention.len()];
    d_attention.copy_to_host(&mut attention)?;
    assert!(attention.iter().all(|value| value.is_finite()));

    let iterations = 1_000;
    let started = Instant::now();
    for _ in 0..iterations {
        device.gemm_q4_0(&d_weights, &d_input, &mut d_output, ROWS, COLS, BATCH);
    }
    device.sync()?;
    let batched = started.elapsed();

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
