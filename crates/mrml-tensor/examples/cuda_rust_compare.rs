#![no_std]
#![no_main]

use mrml_runtime::{Instant, Vector as Vec, mrml_println as println};

macro_rules! vec {
    ($value:expr; $length:expr) => {{
        let mut values = Vec::new();
        values.resize($length, $value);
        values
    }};
}

fn application_main() -> mrml_tensor::error::Result<()> {
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};

    const ELEMENTS: usize = 1 << 22;
    const ITERATIONS: usize = 1_000;
    let backend = "rust-ptx";
    let device = CudaDevice::new(0)?;
    let a = vec![1.25f32; ELEMENTS];
    let b = vec![2.5f32; ELEMENTS];
    let d_a = CudaBuffer::from_host(&a)?;
    let d_b = CudaBuffer::from_host(&b)?;
    let mut d_out = CudaBuffer::alloc(ELEMENTS)?;

    for _ in 0..20 {
        device.add(&d_a, &d_b, &mut d_out);
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.add(&d_a, &d_b, &mut d_out);
    }
    device.sync()?;
    let elapsed = started.elapsed().as_secs_f64();

    let mut output = vec![0.0; ELEMENTS];
    d_out.copy_to_host(&mut output)?;
    assert!(
        output
            .iter()
            .all(|value| (*value - 3.75).abs() < f32::EPSILON)
    );
    println!(
        "backend={backend} elements={ELEMENTS} iterations={ITERATIONS} total_ms={:.3} kernel_us={:.3} bandwidth_gbps={:.2}",
        elapsed * 1e3,
        elapsed * 1e6 / ITERATIONS as f64,
        ELEMENTS as f64 * 12.0 * ITERATIONS as f64 / elapsed / 1e9
    );

    const DIM: usize = 5_376;
    const BATCH: usize = 32;
    let values: Vec<f32> = (0..DIM * BATCH)
        .map(|i| mrml_math::sin(i as f32 * 0.001))
        .collect();
    let weights = vec![1.0f32; DIM];
    let d_values = CudaBuffer::from_host(&values)?;
    let d_weights = CudaBuffer::from_host(&weights)?;
    let mut d_norm = CudaBuffer::alloc(values.len())?;
    for _ in 0..20 {
        device.rms_norm_batch(&d_values, Some(&d_weights), &mut d_norm, DIM, BATCH, 1e-6);
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.rms_norm_batch(&d_values, Some(&d_weights), &mut d_norm, DIM, BATCH, 1e-6);
    }
    device.sync()?;
    println!(
        "backend={backend} rms_norm_batch_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    let d_gate = CudaBuffer::from_host(&values)?;
    let d_up = CudaBuffer::from_host(&values)?;
    let mut d_activation = CudaBuffer::alloc(values.len())?;
    for _ in 0..20 {
        device.geglu(&d_gate, &d_up, &mut d_activation);
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.geglu(&d_gate, &d_up, &mut d_activation);
    }
    device.sync()?;
    println!(
        "backend={backend} geglu_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    const EXPERTS: usize = 128;
    let router_weights: Vec<f32> = (0..EXPERTS * DIM)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.0001)
        .collect();
    let d_router_weights = CudaBuffer::from_host(&router_weights)?;
    let mut d_logits = CudaBuffer::alloc(EXPERTS * BATCH)?;
    let mut d_ids = CudaBuffer::alloc(8 * BATCH)?;
    let mut d_probabilities = CudaBuffer::alloc(8 * BATCH)?;
    for _ in 0..20 {
        device.moe_router_batch(
            &d_router_weights,
            &d_values,
            &mut d_logits,
            &mut d_ids,
            &mut d_probabilities,
            DIM,
            EXPERTS,
            BATCH,
        );
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.moe_router_batch(
            &d_router_weights,
            &d_values,
            &mut d_logits,
            &mut d_ids,
            &mut d_probabilities,
            DIM,
            EXPERTS,
            BATCH,
        );
    }
    device.sync()?;
    println!(
        "backend={backend} moe_router_batch_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    let mut d_attn_res = CudaBuffer::alloc(DIM * BATCH)?;
    let mut d_shared = CudaBuffer::alloc(DIM * BATCH)?;
    let mut d_moe = CudaBuffer::alloc(DIM * BATCH)?;
    let mut d_router = CudaBuffer::alloc(DIM * BATCH)?;
    let mut d_output = CudaBuffer::alloc(DIM * BATCH)?;
    for _ in 0..20 {
        device.prepare_ffn_batch(
            &d_values,
            &d_values,
            &d_weights,
            &d_weights,
            &d_weights,
            &d_weights,
            &mut d_attn_res,
            &mut d_shared,
            &mut d_moe,
            &mut d_router,
            DIM,
            BATCH,
        );
        device.finish_ffn_batch(
            &d_attn_res,
            &mut d_shared,
            &mut d_moe,
            &d_weights,
            &d_weights,
            &d_weights,
            &mut d_output,
            1.0,
            DIM,
            BATCH,
        );
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.prepare_ffn_batch(
            &d_values,
            &d_values,
            &d_weights,
            &d_weights,
            &d_weights,
            &d_weights,
            &mut d_attn_res,
            &mut d_shared,
            &mut d_moe,
            &mut d_router,
            DIM,
            BATCH,
        );
        device.finish_ffn_batch(
            &d_attn_res,
            &mut d_shared,
            &mut d_moe,
            &d_weights,
            &d_weights,
            &d_weights,
            &mut d_output,
            1.0,
            DIM,
            BATCH,
        );
    }
    device.sync()?;
    println!(
        "backend={backend} prepare_finish_ffn_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    const VOCAB: usize = 262_144;
    const PARTITIONS: usize = 128;
    const TOP_K: usize = 40;
    let vocab_logits: Vec<f32> = (0..VOCAB)
        .map(|i| ((i * 7919 % 65521) as f32) * 0.001)
        .collect();
    let vocab_valid = vec![1u8; VOCAB];
    let recent = vec![0i32; 32];
    let d_vocab_logits = CudaBuffer::from_host(&vocab_logits)?;
    let d_vocab_valid = CudaBuffer::from_host(&vocab_valid)?;
    let d_recent = CudaBuffer::from_host(&recent)?;
    let mut d_top_scores = CudaBuffer::alloc(PARTITIONS * TOP_K)?;
    let mut d_top_ids = CudaBuffer::alloc(PARTITIONS * TOP_K)?;
    for _ in 0..10 {
        device.vocab_topk(
            &d_vocab_logits,
            &d_vocab_valid,
            &d_recent,
            &mut d_top_scores,
            &mut d_top_ids,
            VOCAB,
            32,
            10,
            TOP_K,
            PARTITIONS,
        )
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.vocab_topk(
            &d_vocab_logits,
            &d_vocab_valid,
            &d_recent,
            &mut d_top_scores,
            &mut d_top_ids,
            VOCAB,
            32,
            10,
            TOP_K,
            PARTITIONS,
        )
    }
    device.sync()?;
    println!(
        "backend={backend} vocab_topk_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    const HEAD_DIM: usize = 256;
    const HEADS: usize = 16;
    const KV_HEADS: usize = 8;
    const Q_ROWS: usize = HEADS * HEAD_DIM;
    const KV_ROWS: usize = KV_HEADS * HEAD_DIM;
    let qkv_row_bytes = (DIM / 32) * 18;
    let qkv_weights = vec![0u8; Q_ROWS * qkv_row_bytes];
    let kv_weights = vec![0u8; KV_ROWS * qkv_row_bytes];
    let d_wq = CudaBuffer::from_host(&qkv_weights)?;
    let d_wk = CudaBuffer::from_host(&kv_weights)?;
    let d_wv = CudaBuffer::from_host(&kv_weights)?;
    let qkv_input = vec![0.01f32; DIM];
    let d_qkv_input = CudaBuffer::from_host(&qkv_input)?;
    let mut d_qkv_projection = CudaBuffer::alloc(Q_ROWS + 2 * KV_ROWS)?;
    for _ in 0..10 {
        device.gemv_q4_0_qkv(
            &d_wq,
            &d_wk,
            &d_wv,
            &d_qkv_input,
            &mut d_qkv_projection,
            Q_ROWS,
            KV_ROWS,
            DIM,
        )
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..200 {
        device.gemv_q4_0_qkv(
            &d_wq,
            &d_wk,
            &d_wv,
            &d_qkv_input,
            &mut d_qkv_projection,
            Q_ROWS,
            KV_ROWS,
            DIM,
        )
    }
    device.sync()?;
    println!(
        "backend={backend} qkv_projection_decode_ms={:.3}",
        started.elapsed().as_secs_f64() * 1e3 / 200.0
    );
    let qkv_dim = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let qkv_values: Vec<f32> = (0..BATCH * qkv_dim)
        .map(|i| mrml_math::sin(i as f32 * 0.001))
        .collect();
    let qkv_norm = vec![1.0f32; HEAD_DIM];
    let mut d_qkv = CudaBuffer::from_host(&qkv_values)?;
    let d_qkv_norm = CudaBuffer::from_host(&qkv_norm)?;
    let mut d_k_cache = CudaBuffer::alloc(BATCH * KV_HEADS * HEAD_DIM)?;
    let mut d_v_cache = CudaBuffer::alloc(BATCH * KV_HEADS * HEAD_DIM)?;
    for _ in 0..10 {
        device.qkv_postprocess_batch(
            &mut d_qkv,
            &d_qkv_norm,
            &d_qkv_norm,
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
        )
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        device.qkv_postprocess_batch(
            &mut d_qkv,
            &d_qkv_norm,
            &d_qkv_norm,
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
        )
    }
    device.sync()?;
    println!(
        "backend={backend} qkv_postprocess_batch_us={:.3}",
        started.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64
    );

    const EXP_DIM: usize = 704;
    const ACTIVE: usize = 8;
    const MOE_EXPERTS: usize = 2;
    const MOE_BATCH: usize = 32;
    let moe_gate_bytes = MOE_EXPERTS * 2 * EXP_DIM * (DIM / 32) * 18;
    let moe_down_bytes = MOE_EXPERTS * DIM * (EXP_DIM / 32) * 18;
    let d_moe_gate = CudaBuffer::from_host(&vec![0u8; moe_gate_bytes])?;
    let d_moe_down = CudaBuffer::from_host(&vec![0u8; moe_down_bytes])?;
    let moe_ids: Vec<i32> = (0..MOE_BATCH * ACTIVE)
        .map(|i| (i % MOE_EXPERTS) as i32)
        .collect();
    let moe_weights = vec![0.125f32; MOE_BATCH * ACTIVE];
    let d_moe_ids = CudaBuffer::from_host(&moe_ids)?;
    let d_moe_weights = CudaBuffer::from_host(&moe_weights)?;
    let moe_input: Vec<f32> = (0..MOE_BATCH * DIM)
        .map(|i| mrml_math::sin(i as f32 * 0.001))
        .collect();
    let d_moe_input = CudaBuffer::from_host(&moe_input)?;
    let mut d_moe_act = CudaBuffer::alloc(MOE_BATCH * ACTIVE * EXP_DIM)?;
    let mut d_moe_output = CudaBuffer::alloc(MOE_BATCH * DIM)?;
    for _ in 0..5 {
        device.moe_topk_batch_q4_0(
            &d_moe_gate,
            &d_moe_down,
            &d_moe_ids,
            &d_moe_weights,
            None,
            &d_moe_input,
            &mut d_moe_act,
            &mut d_moe_output,
            DIM,
            EXP_DIM,
            ACTIVE,
            MOE_BATCH,
        )
    }
    device.sync()?;
    let started = Instant::now();
    for _ in 0..100 {
        device.moe_topk_batch_q4_0(
            &d_moe_gate,
            &d_moe_down,
            &d_moe_ids,
            &d_moe_weights,
            None,
            &d_moe_input,
            &mut d_moe_act,
            &mut d_moe_output,
            DIM,
            EXP_DIM,
            ACTIVE,
            MOE_BATCH,
        )
    }
    device.sync()?;
    println!(
        "backend={backend} moe_experts_batch_ms={:.3}",
        started.elapsed().as_secs_f64() * 1e3 / 100.0
    );

    const ATTN_TOKENS: usize = 1024;
    let attn_q: Vec<f32> = (0..HEADS * HEAD_DIM)
        .map(|i| mrml_math::sin(i as f32 * 0.003))
        .collect();
    let attn_cache: Vec<u16> = (0..ATTN_TOKENS * KV_HEADS * HEAD_DIM)
        .map(|i| mrml_tensor::quant::f32_to_f16(mrml_math::sin(i as f32 * 0.0001)))
        .collect();
    let d_attn_q = CudaBuffer::from_host(&attn_q)?;
    let d_attn_k = CudaBuffer::from_host(&attn_cache)?;
    let d_attn_v = CudaBuffer::from_host(&attn_cache)?;
    let mut d_attn_output = CudaBuffer::alloc(HEADS * HEAD_DIM)?;
    for keys in [32usize, 64, 128, 256, 1024] {
        for _ in 0..10 {
            device.attention_causal(
                &d_attn_q,
                &d_attn_k,
                &d_attn_v,
                &mut d_attn_output,
                keys - 1,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                1.0 / mrml_math::sqrt(HEAD_DIM as f32),
                Some(keys),
                ATTN_TOKENS,
                0,
                0,
            )
        }
        device.sync()?;
        let iterations = if keys < 256 { 1000 } else { 200 };
        let started = Instant::now();
        for _ in 0..iterations {
            device.attention_causal(
                &d_attn_q,
                &d_attn_k,
                &d_attn_v,
                &mut d_attn_output,
                keys - 1,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                1.0 / mrml_math::sqrt(HEAD_DIM as f32),
                Some(keys),
                ATTN_TOKENS,
                0,
                0,
            )
        }
        device.sync()?;
        println!(
            "backend={backend} attention_decode_{keys}_ms={:.3}",
            started.elapsed().as_secs_f64() * 1e3 / iterations as f64
        );
    }
    Ok(())
}

mrml_runtime::mrml_entrypoint!(application_main);
