use qtensor::ops::*;

#[test]
fn test_rms_norm() {
    let x = vec![2.0f32, 2.0, 2.0, 2.0];
    let mut out = vec![0.0f32; 4];
    rms_norm(&x, None, 1e-6, &mut out);

    for v in out {
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_swiglu() {
    let gate = vec![0.0f32, 2.0, -2.0];
    let up = vec![1.0f32, 3.0, 4.0];
    let mut out = vec![0.0f32; 3];

    swiglu(&gate, &up, &mut out);

    assert_eq!(out[0], 0.0);
    assert!((out[1] - 5.28478).abs() < 1e-3);
}

#[test]
fn test_mat_mul_f32() {
    let a = vec![
        1.0f32, 2.0,
        3.0, 4.0,
    ];
    let b = vec![
        5.0f32, 6.0,
        7.0, 8.0,
    ];
    let mut c = vec![0.0f32; 4];

    mat_mul_f32(&a, &b, &mut c, 2, 2, 2);

    assert_eq!(c[0], 1.0 * 5.0 + 2.0 * 7.0);
    assert_eq!(c[1], 1.0 * 6.0 + 2.0 * 8.0);
    assert_eq!(c[2], 3.0 * 5.0 + 4.0 * 7.0);
    assert_eq!(c[3], 3.0 * 6.0 + 4.0 * 8.0);
}

#[test]
fn test_prequantized_mat_vec_matches_convenience_path() {
    use qtensor::quant::{f32_to_f16, quantize_f32_to_q8_0};

    let n_rows = 73;
    let n_cols = 64;
    let mut weights = vec![0u8; n_rows * (n_cols / 32) * 18];
    for (block_idx, block) in weights.chunks_exact_mut(18).enumerate() {
        block[..2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = ((block_idx * 17 + i * 29) & 0xff) as u8;
        }
    }
    let input: Vec<f32> = (0..n_cols).map(|i| (i as f32 * 0.31).sin()).collect();

    let mut expected = vec![0.0; n_rows];
    mat_vec_mul_q4_0(&weights, &input, &mut expected, n_rows, n_cols);

    let mut input_q8 = vec![0u8; q8_0_size(n_cols)];
    quantize_f32_to_q8_0(&input, &mut input_q8);
    let mut actual = vec![0.0; n_rows];
    mat_vec_mul_q4_0_q8_0(&weights, &input_q8, &mut actual, n_rows, n_cols);

    assert_eq!(actual, expected);
}

#[test]
fn test_batched_rope_matches_per_head_rope() {
    let n_heads = 8;
    let head_dim = 64;
    let original: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| (i as f32 * 0.071).cos())
        .collect();
    let mut expected = original.clone();
    for head in expected.chunks_exact_mut(head_dim) {
        rope_1d(head, 137, head_dim, 10_000.0, 1.0);
    }

    let mut actual = original;
    rope_1d_batched(&mut actual, 137, n_heads, head_dim, 10_000.0, 1.0);

    assert_eq!(actual, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_ops() {
    use qtensor::cuda::{CudaBuffer, CudaDevice};

    if !CudaDevice::is_available() {
        println!("Skipping CUDA tests: No CUDA device found");
        return;
    }

    let dev = CudaDevice::new(0).expect("Failed to create CudaDevice");

    let x_host = vec![2.0f32; 128];
    let d_x = CudaBuffer::from_host(&x_host).unwrap();
    let mut d_out = CudaBuffer::alloc(128).unwrap();

    dev.rms_norm(&d_x, None, &mut d_out, 1e-6);
    dev.sync().unwrap();

    let mut out_host = vec![0.0f32; 128];
    d_out.copy_to_host(&mut out_host).unwrap();

    for val in out_host {
        assert!((val - 1.0).abs() < 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_q8_gemv_matches_reference() {
    use qtensor::cuda::{CudaBuffer, CudaDevice};
    use qtensor::quant::{f16_to_f32, f32_to_f16};

    if !CudaDevice::is_available() {
        return;
    }

    let (rows, cols) = (73, 64);
    let row_bytes = cols / 32 * 34;
    let mut weights = vec![0u8; rows * row_bytes];
    for (block_index, block) in weights.chunks_exact_mut(34).enumerate() {
        block[..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = block_index.wrapping_mul(17).wrapping_add(i * 11) as u8;
        }
    }
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.031).sin()).collect();
    let expected: Vec<f32> = weights
        .chunks_exact(row_bytes)
        .map(|row| {
            let mut dot = 0.0;
            for block in 0..cols / 32 {
                let off = block * 34;
                let d = f16_to_f32(u16::from_le_bytes([row[off], row[off + 1]]));
                for i in 0..32 {
                    dot += row[off + 2 + i] as i8 as f32 * input[block * 32 + i] * d;
                }
            }
            30.0 * (dot / 30.0).tanh()
        })
        .collect();

    let dev = CudaDevice::new(0).unwrap();
    let d_weights = CudaBuffer::from_host(&weights).unwrap();
    let d_input = CudaBuffer::from_host(&input).unwrap();
    let mut d_output = CudaBuffer::alloc(rows).unwrap();
    dev.gemv_q8_0(&d_weights, &d_input, &mut d_output, rows, cols);
    dev.sync().unwrap();
    let mut actual = vec![0.0; rows];
    d_output.copy_to_host(&mut actual).unwrap();

    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() < 1e-3, "row {index}: {actual} != {expected}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_vocab_topk_contains_exact_global_topk() {
    use qtensor::cuda::{CudaBuffer, CudaDevice};

    if !CudaDevice::is_available() {
        return;
    }

    const N: usize = 4096;
    const K: usize = 40;
    const PARTITIONS: usize = 8;
    let logits: Vec<f32> = (0..N)
        .map(|i| ((i * 7919 % 65521) as f32) * 0.001 + i as f32 * 1e-7)
        .collect();
    let valid: Vec<u8> = (0..N).map(|i| u8::from(i % 17 != 0)).collect();
    let recent = [31i32, 777, 2049, 4001];
    let mut recent_padded = [0i32; 32];
    recent_padded[..recent.len()].copy_from_slice(&recent);

    let dev = CudaDevice::new(0).unwrap();
    let d_logits = CudaBuffer::from_host(&logits).unwrap();
    let d_valid = CudaBuffer::from_host(&valid).unwrap();
    let d_recent = CudaBuffer::from_host(&recent_padded).unwrap();
    let mut d_scores = CudaBuffer::alloc(PARTITIONS * K).unwrap();
    let mut d_ids = CudaBuffer::alloc(PARTITIONS * K).unwrap();
    dev.vocab_topk(
        &d_logits, &d_valid, &d_recent, &mut d_scores, &mut d_ids,
        N, recent.len(), 10, K, PARTITIONS,
    );
    let mut scores = vec![0.0f32; PARTITIONS * K];
    let mut ids = vec![0i32; PARTITIONS * K];
    d_scores.copy_to_host(&mut scores).unwrap();
    d_ids.copy_to_host(&mut ids).unwrap();

    let mut actual: Vec<_> = scores.into_iter().zip(ids).collect();
    actual.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    actual.truncate(K);
    let mut expected: Vec<_> = logits.iter().copied().enumerate()
        .filter(|(id, _)| valid[*id] != 0)
        .map(|(id, mut score)| {
            if recent.contains(&(id as i32)) { score -= 1.8; }
            (score, id as i32)
        })
        .collect();
    expected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    expected.truncate(K);
    assert_eq!(actual, expected);
}
