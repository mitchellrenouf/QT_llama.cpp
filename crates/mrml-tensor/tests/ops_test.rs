use mrml_tensor::ops::*;

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
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let mut c = vec![0.0f32; 4];

    mat_mul_f32(&a, &b, &mut c, 2, 2, 2);

    assert_eq!(c[0], 1.0 * 5.0 + 2.0 * 7.0);
    assert_eq!(c[1], 1.0 * 6.0 + 2.0 * 8.0);
    assert_eq!(c[2], 3.0 * 5.0 + 4.0 * 7.0);
    assert_eq!(c[3], 3.0 * 6.0 + 4.0 * 8.0);
}

#[test]
fn test_prequantized_mat_vec_matches_convenience_path() {
    use mrml_tensor::quant::{f32_to_f16, quantize_f32_to_q8_0};

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
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};

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
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};
    use mrml_tensor::quant::{f16_to_f32, f32_to_f16};

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
        assert!(
            (actual - expected).abs() < 1e-3,
            "row {index}: {actual} != {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_vocab_topk_contains_exact_global_topk() {
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};

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
        &d_logits,
        &d_valid,
        &d_recent,
        &mut d_scores,
        &mut d_ids,
        N,
        recent.len(),
        10,
        K,
        PARTITIONS,
    );
    let mut scores = vec![0.0f32; PARTITIONS * K];
    let mut ids = vec![0i32; PARTITIONS * K];
    d_scores.copy_to_host(&mut scores).unwrap();
    d_ids.copy_to_host(&mut ids).unwrap();

    let mut actual: Vec<_> = scores.into_iter().zip(ids).collect();
    actual.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    actual.truncate(K);
    let mut expected: Vec<_> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(id, _)| valid[*id] != 0)
        .map(|(id, mut score)| {
            if recent.contains(&(id as i32)) {
                score -= 1.8;
            }
            (score, id as i32)
        })
        .collect();
    expected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    expected.truncate(K);
    assert_eq!(actual, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_qkv_postprocess_matches_cpu() {
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};
    use mrml_tensor::ops::{rms_norm_inplace, rope_1d_batched};

    if !CudaDevice::is_available() {
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (4, 2, 64);
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let mut expected: Vec<f32> = (0..q_dim + 2 * kv_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();
    let q_norm: Vec<f32> = (0..head_dim).map(|i| 0.8 + i as f32 * 0.001).collect();
    let k_norm: Vec<f32> = (0..head_dim).map(|i| 1.1 - i as f32 * 0.001).collect();
    for head in expected[..q_dim].chunks_exact_mut(head_dim) {
        rms_norm_inplace(head, Some(&q_norm), 1e-6);
    }
    rope_1d_batched(&mut expected[..q_dim], 37, n_heads, head_dim, 10_000.0, 1.0);
    for head in expected[q_dim..q_dim + kv_dim].chunks_exact_mut(head_dim) {
        rms_norm_inplace(head, Some(&k_norm), 1e-6);
    }
    rope_1d_batched(
        &mut expected[q_dim..q_dim + kv_dim],
        37,
        n_kv_heads,
        head_dim,
        10_000.0,
        1.0,
    );
    for head in expected[q_dim + kv_dim..].chunks_exact_mut(head_dim) {
        rms_norm_inplace(head, None, 1e-6);
    }

    let original: Vec<f32> = (0..q_dim + 2 * kv_dim)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();
    let dev = CudaDevice::new(0).unwrap();
    let mut d_qkv = CudaBuffer::from_host(&original).unwrap();
    let d_q_norm = CudaBuffer::from_host(&q_norm).unwrap();
    let d_k_norm = CudaBuffer::from_host(&k_norm).unwrap();
    let mut d_k_cache = CudaBuffer::alloc(3 * kv_dim).unwrap();
    let mut d_v_cache = CudaBuffer::alloc(3 * kv_dim).unwrap();
    dev.qkv_postprocess(
        &mut d_qkv,
        &d_q_norm,
        &d_k_norm,
        &mut d_k_cache,
        &mut d_v_cache,
        37,
        1,
        n_heads,
        n_kv_heads,
        head_dim,
        10_000.0,
        0,
        0,
    );
    let mut actual = vec![0.0; expected.len()];
    d_qkv.copy_to_host(&mut actual).unwrap();
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 2e-4, "index {i}: {a} != {e}");
    }
    let mut k_cache = vec![0u16; 3 * kv_dim];
    let mut v_cache = vec![0u16; 3 * kv_dim];
    d_k_cache.copy_to_host(&mut k_cache).unwrap();
    d_v_cache.copy_to_host(&mut v_cache).unwrap();
    for (stored, expected) in k_cache[kv_dim..2 * kv_dim]
        .iter()
        .zip(&actual[q_dim..q_dim + kv_dim])
    {
        assert!((mrml_tensor::quant::f16_to_f32(*stored) - expected).abs() < 1e-3);
    }
    for (stored, expected) in v_cache[kv_dim..2 * kv_dim]
        .iter()
        .zip(&actual[q_dim + kv_dim..])
    {
        assert!((mrml_tensor::quant::f16_to_f32(*stored) - expected).abs() < 1e-3);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_qkv_postprocess_quantized_cache_formats() {
    use mrml_tensor::cuda::{CudaBuffer,CudaDevice};
    use mrml_tensor::quant::f16_to_f32;
    if !CudaDevice::is_available(){return}
    let(n_heads,n_kv_heads,head_dim)=(4usize,2usize,64usize);let q_dim=n_heads*head_dim;let kv_dim=n_kv_heads*head_dim;
    let original:Vec<f32>=(0..q_dim+2*kv_dim).map(|i|(i as f32*0.017).sin()).collect();let norm=vec![1.0f32;head_dim];let dev=CudaDevice::new(0).unwrap();let d_norm=CudaBuffer::from_host(&norm).unwrap();
    for format in [1,2]{let block_bytes=if format==1{34}else{18};let cache_bytes=n_kv_heads*(head_dim/32)*block_bytes;let mut d_qkv=CudaBuffer::from_host(&original).unwrap();let mut d_k=CudaBuffer::alloc(cache_bytes.div_ceil(2)).unwrap();let mut d_v=CudaBuffer::alloc(cache_bytes.div_ceil(2)).unwrap();dev.qkv_postprocess(&mut d_qkv,&d_norm,&d_norm,&mut d_k,&mut d_v,37,0,n_heads,n_kv_heads,head_dim,10_000.0,format,format);let mut transformed=vec![0.0f32;original.len()];d_qkv.copy_to_host(&mut transformed).unwrap();for(cache,expected)in[(&d_k,&transformed[q_dim..q_dim+kv_dim]),(&d_v,&transformed[q_dim+kv_dim..])]{let mut words=vec![0u16;cache.len()];cache.copy_to_host(&mut words).unwrap();let bytes:Vec<u8>=words.into_iter().flat_map(u16::to_le_bytes).collect();for head in 0..n_kv_heads{for block in 0..head_dim/32{let offset=(head*(head_dim/32)+block)*block_bytes;let scale=f16_to_f32(u16::from_le_bytes([bytes[offset],bytes[offset+1]]));for lane in 0..32{let quant=if format==1{bytes[offset+2+lane]as i8 as i32}else{let packed=bytes[offset+2+(lane&15)];if lane<16{(packed&15)as i32-8}else{(packed>>4)as i32-8}};let actual=scale*quant as f32;let target=expected[head*head_dim+block*32+lane];let tolerance=if format==1{0.02}else{0.2};assert!((actual-target).abs()<tolerance,"format={format} head={head} block={block} lane={lane}: {actual} != {target}")}}}}
    }
}

#[cfg(feature="cuda")]
#[test]
fn test_cuda_causal_attention_matches_cpu(){use mrml_tensor::cuda::{CudaBuffer,CudaDevice};use mrml_tensor::quant::f32_to_f16;if !CudaDevice::is_available(){return}let(n_heads,n_kv_heads,head_dim,tokens)=(4usize,2usize,64usize,8usize);let q:Vec<f32>=(0..n_heads*head_dim).map(|i|(i as f32*0.013).sin()).collect();let keys:Vec<f32>=(0..tokens*n_kv_heads*head_dim).map(|i|(i as f32*0.007).cos()).collect();let values:Vec<f32>=(0..tokens*n_kv_heads*head_dim).map(|i|(i as f32*0.011).sin()).collect();let scale=1.0/(head_dim as f32).sqrt();let mut expected=vec![0.0f32;n_heads*head_dim];for head in 0..n_heads{let kv_head=head/(n_heads/n_kv_heads);let mut scores=Vec::new();for token in 4..tokens{let mut dot=0.0;for d in 0..head_dim{dot+=q[head*head_dim+d]*mrml_tensor::quant::f16_to_f32(f32_to_f16(keys[(token*n_kv_heads+kv_head)*head_dim+d]))}scores.push(dot*scale)}let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);let mut sum=0.0;for score in &mut scores{*score=(*score-max).exp();sum+=*score}for d in 0..head_dim{for(token,score)in(4..tokens).zip(&scores){expected[head*head_dim+d]+=score/sum*mrml_tensor::quant::f16_to_f32(f32_to_f16(values[(token*n_kv_heads+kv_head)*head_dim+d]))}}}let dev=CudaDevice::new(0).unwrap();let d_q=CudaBuffer::from_host(&q).unwrap();let d_k=CudaBuffer::from_host(&keys.iter().copied().map(f32_to_f16).collect::<Vec<_>>()).unwrap();let d_v=CudaBuffer::from_host(&values.iter().copied().map(f32_to_f16).collect::<Vec<_>>()).unwrap();let mut d_out=CudaBuffer::alloc(expected.len()).unwrap();dev.attention_causal(&d_q,&d_k,&d_v,&mut d_out,tokens-1,n_heads,n_kv_heads,head_dim,scale,Some(4),tokens,0,0);let mut actual=vec![0.0;expected.len()];d_out.copy_to_host(&mut actual).unwrap();for(i,(a,e))in actual.iter().zip(&expected).enumerate(){assert!((a-e).abs()<2e-4,"index {i}: {a} != {e}")}}

#[cfg(feature="cuda")]
#[test]
fn test_cuda_moe_topk_matches_dequantized_cpu(){use mrml_tensor::cuda::{CudaBuffer,CudaDevice};use mrml_tensor::quant::{dequantize_q4_0,quantize_f32_to_q4_0};if !CudaDevice::is_available(){return}const DIM:usize=64;const EXP:usize=32;const EXPERTS:usize=2;const ACTIVE:usize=2;let row_bytes=(DIM/32)*18;let down_row_bytes=(EXP/32)*18;let mut gate_up=vec![0u8;EXPERTS*2*EXP*row_bytes];let mut down=vec![0u8;EXPERTS*DIM*down_row_bytes];for expert in 0..EXPERTS{for row in 0..2*EXP{let values:Vec<f32>=(0..DIM).map(|i|(((expert*131+row*17+i*7)%101)as f32-50.0)*0.002).collect();let offset=(expert*2*EXP+row)*row_bytes;quantize_f32_to_q4_0(&values,&mut gate_up[offset..offset+row_bytes])}for row in 0..DIM{let values:Vec<f32>=(0..EXP).map(|i|(((expert*97+row*19+i*11)%89)as f32-44.0)*0.003).collect();let offset=(expert*DIM+row)*down_row_bytes;quantize_f32_to_q4_0(&values,&mut down[offset..offset+down_row_bytes])}}let input:Vec<f32>=(0..DIM).map(|i|(i as f32*0.031).sin()).collect();let ids=[0i32,1];let weights=[0.65f32,0.35];let scales=[1.0f32,0.8];let mut expected=vec![0.0f32;DIM];for slot in 0..ACTIVE{let expert=ids[slot]as usize;let mut act=vec![0.0f32;EXP];for row in 0..EXP{let mut gw=vec![0.0;DIM];let mut uw=vec![0.0;DIM];let go=(expert*2*EXP+row)*row_bytes;let uo=(expert*2*EXP+EXP+row)*row_bytes;dequantize_q4_0(&gate_up[go..go+row_bytes],&mut gw);dequantize_q4_0(&gate_up[uo..uo+row_bytes],&mut uw);let gate: f32=gw.iter().zip(&input).map(|(a,b)|a*b).sum();let up:f32=uw.iter().zip(&input).map(|(a,b)|a*b).sum();act[row]=0.5*gate*(1.0+(0.7978845608*gate*(1.0+0.044715*gate*gate)).tanh())*up}for row in 0..DIM{let mut dw=vec![0.0;EXP];let offset=(expert*DIM+row)*down_row_bytes;dequantize_q4_0(&down[offset..offset+down_row_bytes],&mut dw);expected[row]+=dw.iter().zip(&act).map(|(a,b)|a*b).sum::<f32>()*weights[slot]*scales[expert]}}let dev=CudaDevice::new(0).unwrap();let d_gu=CudaBuffer::from_host(&gate_up).unwrap();let d_down=CudaBuffer::from_host(&down).unwrap();let d_ids=CudaBuffer::from_host(&ids).unwrap();let d_weights=CudaBuffer::from_host(&weights).unwrap();let d_scales=CudaBuffer::from_host(&scales).unwrap();let d_input=CudaBuffer::from_host(&input).unwrap();let mut d_act=CudaBuffer::alloc(ACTIVE*EXP).unwrap();let mut d_out=CudaBuffer::alloc(DIM).unwrap();dev.moe_topk_q4_0(&d_gu,&d_down,&d_ids,&d_weights,Some(&d_scales),&d_input,&mut d_act,&mut d_out,DIM,EXP,ACTIVE);let mut actual=vec![0.0;DIM];d_out.copy_to_host(&mut actual).unwrap();for(i,(a,e))in actual.iter().zip(&expected).enumerate(){assert!((a-e).abs()<2e-4,"index {i}: {a} != {e}")}}
