#[cfg(feature = "cuda")]
fn main() -> mrml_tensor::anyhow::Result<()> {
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};
    use mrml_tensor::quant::f32_to_f16;
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
        &d_weights, &d_weights, &d_weights, &d_input, &mut d_qkv, Q_ROWS, KV_ROWS, COLS, BATCH,
    );
    device.sync()?;
    let mut qkv = vec![0.0f32; d_qkv.len()];
    d_qkv.copy_to_host(&mut qkv)?;
    let mut d_single_qkv = CudaBuffer::alloc(Q_ROWS + 2 * KV_ROWS)?;
    for token in 0..BATCH {
        device.copy_from_host_async(&mut d_single_in, &input[token * COLS..(token + 1) * COLS])?;
        device.gemv_q4_0_qkv(
            &d_weights,
            &d_weights,
            &d_weights,
            &d_single_in,
            &mut d_single_qkv,
            Q_ROWS,
            KV_ROWS,
            COLS,
        );
        device.sync()?;
        let mut expected = vec![0.0f32; Q_ROWS + 2 * KV_ROWS];
        d_single_qkv.copy_to_host(&mut expected)?;
        let actual = &qkv[token * expected.len()..(token + 1) * expected.len()];
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
    }

    let norm = vec![1.0f32; HEAD_DIM];
    let d_norm = CudaBuffer::from_host(&norm)?;
    let mut d_k_cache = CudaBuffer::alloc(BATCH * KV_ROWS)?;
    let mut d_v_cache = CudaBuffer::alloc(BATCH * KV_ROWS)?;
    device.qkv_postprocess_batch(
        &mut d_qkv,
        &d_norm,
        &d_norm,
        &mut d_k_cache,
        &mut d_v_cache,
        0,
        0,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        10_000.0,
        BATCH,
        BATCH,
        0,
        0,
    );
    let mut d_attention = CudaBuffer::alloc(BATCH * Q_ROWS)?;
    device.attention_prefill(
        &d_qkv,
        &d_k_cache,
        &d_v_cache,
        &mut d_attention,
        0,
        BATCH,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        None,
        BATCH,
        0,
        0,
    );
    device.sync()?;
    let mut attention = vec![0.0f32; d_attention.len()];
    d_attention.copy_to_host(&mut attention)?;
    assert!(attention.iter().all(|value| value.is_finite()));

    let router_weights: Vec<f32> = (0..128 * COLS)
        .map(|index| ((index % 19) as f32 - 9.0) / 19.0)
        .collect();
    let d_router_weights = CudaBuffer::from_host(&router_weights)?;
    let mut d_router_logits = CudaBuffer::alloc(BATCH * 128)?;
    let mut d_router_ids = CudaBuffer::alloc(BATCH * 8)?;
    let mut d_router_probs = CudaBuffer::alloc(BATCH * 8)?;
    device.moe_router_batch(
        &d_router_weights,
        &d_input,
        &mut d_router_logits,
        &mut d_router_ids,
        &mut d_router_probs,
        COLS,
        128,
        BATCH,
    );
    device.sync()?;
    let mut router_ids = vec![0i32; BATCH * 8];
    let mut router_probs = vec![0.0f32; BATCH * 8];
    d_router_ids.copy_to_host(&mut router_ids)?;
    d_router_probs.copy_to_host(&mut router_probs)?;
    let mut d_single_logits = CudaBuffer::alloc(128)?;
    let mut d_single_ids = CudaBuffer::alloc(8)?;
    let mut d_single_probs = CudaBuffer::alloc(8)?;
    for token in 0..BATCH {
        device.copy_from_host_async(&mut d_single_in, &input[token * COLS..(token + 1) * COLS])?;
        device.moe_router(
            &d_router_weights,
            &d_single_in,
            &mut d_single_logits,
            &mut d_single_ids,
            &mut d_single_probs,
            COLS,
            128,
        );
        device.sync()?;
        let mut expected_ids = vec![0i32; 8];
        let mut expected_probs = vec![0.0f32; 8];
        d_single_ids.copy_to_host(&mut expected_ids)?;
        d_single_probs.copy_to_host(&mut expected_probs)?;
        assert_eq!(&router_ids[token * 8..(token + 1) * 8], expected_ids);
        assert!(
            router_probs[token * 8..(token + 1) * 8]
                .iter()
                .zip(&expected_probs)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }

    let mut expert_gate_up = vec![0u8; 128 * 2 * COLS * 18];
    let mut expert_down = vec![0u8; 128 * COLS * 18];
    for block in expert_gate_up
        .chunks_exact_mut(18)
        .chain(expert_down.chunks_exact_mut(18))
    {
        block[..2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
        block[2..].fill(0x98);
    }
    let d_expert_gate_up = CudaBuffer::from_host(&expert_gate_up)?;
    let d_expert_down = CudaBuffer::from_host(&expert_down)?;
    let mut d_expert_act = CudaBuffer::alloc(BATCH * 8 * COLS)?;
    let mut d_expert_out = CudaBuffer::alloc(BATCH * COLS)?;
    device.moe_topk_batch_q4_0(
        &d_expert_gate_up,
        &d_expert_down,
        &d_router_ids,
        &d_router_probs,
        None,
        &d_input,
        &mut d_expert_act,
        &mut d_expert_out,
        COLS,
        COLS,
        8,
        BATCH,
    );
    device.sync()?;
    let mut expert_out = vec![0.0f32; BATCH * COLS];
    d_expert_out.copy_to_host(&mut expert_out)?;
    let mut d_single_act = CudaBuffer::alloc(8 * COLS)?;
    let mut d_single_moe = CudaBuffer::alloc(COLS)?;
    for token in 0..BATCH {
        device.copy_from_host_async(&mut d_single_in, &input[token * COLS..(token + 1) * COLS])?;
        let token_ids = CudaBuffer::from_host(&router_ids[token * 8..(token + 1) * 8])?;
        let token_probs = CudaBuffer::from_host(&router_probs[token * 8..(token + 1) * 8])?;
        device.moe_topk_q4_0(
            &d_expert_gate_up,
            &d_expert_down,
            &token_ids,
            &token_probs,
            None,
            &d_single_in,
            &mut d_single_act,
            &mut d_single_moe,
            COLS,
            COLS,
            8,
        );
        device.sync()?;
        let mut expected = vec![0.0f32; COLS];
        d_single_moe.copy_to_host(&mut expected)?;
        assert!(
            expert_out[token * COLS..(token + 1) * COLS]
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
    }

    let mut d_geglu = CudaBuffer::alloc(BATCH * ROWS)?;
    device.gemm_q4_0_geglu(
        &d_weights,
        &d_weights,
        &d_input,
        &mut d_geglu,
        ROWS,
        COLS,
        BATCH,
    );
    device.sync()?;
    let mut geglu = vec![0.0f32; d_geglu.len()];
    d_geglu.copy_to_host(&mut geglu)?;
    let mut d_single_geglu = CudaBuffer::alloc(ROWS)?;
    for token in 0..BATCH {
        device.copy_from_host_async(&mut d_single_in, &input[token * COLS..(token + 1) * COLS])?;
        device.gemv_q4_0_geglu(
            &d_weights,
            &d_weights,
            &d_single_in,
            &mut d_single_geglu,
            ROWS,
            COLS,
        );
        device.sync()?;
        let mut expected = vec![0.0f32; ROWS];
        d_single_geglu.copy_to_host(&mut expected)?;
        assert!(
            geglu[token * ROWS..(token + 1) * ROWS]
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
    }

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
            device
                .copy_from_host_async(&mut d_single_in, &input[token * COLS..(token + 1) * COLS])?;
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

    // Use a Gemma-sized projection as the performance gate. The tiny case
    // above is intentionally convenient for correctness, but mostly measures
    // launch overhead and cannot reveal whether batched weights are reused.
    const MODEL_DIM: usize = 2816;
    const MODEL_ROWS: usize = 2816;
    const MODEL_BATCH: usize = 128;
    const MODEL_ITERS: usize = 100;
    let model_row_bytes = MODEL_DIM / 32 * 18;
    let mut model_weights = vec![0u8; MODEL_ROWS * model_row_bytes];
    for block in model_weights.chunks_exact_mut(18) {
        block[..2].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
        block[2..].fill(0x98);
    }
    let model_input: Vec<f32> = (0..MODEL_BATCH * MODEL_DIM)
        .map(|index| ((index % 127) as f32 - 63.0) / 64.0)
        .collect();
    let d_model_weights = CudaBuffer::from_host(&model_weights)?;
    let d_model_input = CudaBuffer::from_host(&model_input)?;
    let mut d_model_output = CudaBuffer::alloc(MODEL_BATCH * MODEL_ROWS)?;
    for _ in 0..10 {
        device.gemm_q4_0(
            &d_model_weights,
            &d_model_input,
            &mut d_model_output,
            MODEL_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..MODEL_ITERS {
        device.gemm_q4_0(
            &d_model_weights,
            &d_model_input,
            &mut d_model_output,
            MODEL_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
    }
    device.sync()?;
    let model_elapsed = started.elapsed();
    println!(
        "gemma_projection rows={MODEL_ROWS} cols={MODEL_DIM} batch={MODEL_BATCH} iterations={MODEL_ITERS} total_ms={:.3} per_iteration_ms={:.3}",
        model_elapsed.as_secs_f64() * 1_000.0,
        model_elapsed.as_secs_f64() * 1_000.0 / MODEL_ITERS as f64,
    );

    const MODEL_Q_ROWS: usize = 4096;
    const MODEL_KV_ROWS: usize = 1024;
    const MODEL_FFN_ROWS: usize = 2112;
    let mut q_weights = vec![0u8; MODEL_Q_ROWS * model_row_bytes];
    let mut kv_weights = vec![0u8; MODEL_KV_ROWS * model_row_bytes];
    let mut ffn_weights = vec![0u8; MODEL_FFN_ROWS * model_row_bytes];
    for block in q_weights
        .chunks_exact_mut(18)
        .chain(kv_weights.chunks_exact_mut(18))
        .chain(ffn_weights.chunks_exact_mut(18))
    {
        block[..2].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
        block[2..].fill(0x98);
    }
    let d_q_weights = CudaBuffer::from_host(&q_weights)?;
    let d_kv_weights = CudaBuffer::from_host(&kv_weights)?;
    let d_ffn_weights = CudaBuffer::from_host(&ffn_weights)?;
    let mut d_model_qkv = CudaBuffer::alloc(MODEL_BATCH * (MODEL_Q_ROWS + 2 * MODEL_KV_ROWS))?;
    let mut d_model_geglu = CudaBuffer::alloc(MODEL_BATCH * MODEL_FFN_ROWS)?;
    for _ in 0..10 {
        device.gemm_q4_0_qkv(
            &d_q_weights,
            &d_kv_weights,
            &d_kv_weights,
            &d_model_input,
            &mut d_model_qkv,
            MODEL_Q_ROWS,
            MODEL_KV_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
        device.gemm_q4_0_geglu(
            &d_ffn_weights,
            &d_ffn_weights,
            &d_model_input,
            &mut d_model_geglu,
            MODEL_FFN_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..MODEL_ITERS {
        device.gemm_q4_0_qkv(
            &d_q_weights,
            &d_kv_weights,
            &d_kv_weights,
            &d_model_input,
            &mut d_model_qkv,
            MODEL_Q_ROWS,
            MODEL_KV_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
    }
    device.sync()?;
    let qkv_elapsed = started.elapsed();
    let started = Instant::now();
    for _ in 0..MODEL_ITERS {
        device.gemm_q4_0_geglu(
            &d_ffn_weights,
            &d_ffn_weights,
            &d_model_input,
            &mut d_model_geglu,
            MODEL_FFN_ROWS,
            MODEL_DIM,
            MODEL_BATCH,
        );
    }
    device.sync()?;
    let geglu_elapsed = started.elapsed();
    println!(
        "gemma_qkv batch={MODEL_BATCH} iterations={MODEL_ITERS} per_iteration_ms={:.3}",
        qkv_elapsed.as_secs_f64() * 1_000.0 / MODEL_ITERS as f64,
    );
    println!(
        "gemma_geglu batch={MODEL_BATCH} iterations={MODEL_ITERS} per_iteration_ms={:.3}",
        geglu_elapsed.as_secs_f64() * 1_000.0 / MODEL_ITERS as f64,
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_prefill_primitives requires --features cuda");
}
