use mrml_tensor::quant::{dequantize_q4_k_to_f32, dequantize_q6_k_to_f32};

#[test]
fn test_q4_k_dequantize() {
    let mut raw_block = vec![0u8; 144];
    // scale d = 1.0 (fp16: 0x3C00)
    raw_block[0] = 0x00;
    raw_block[1] = 0x3C;
    // dmin = 0.0
    raw_block[2] = 0x00;
    raw_block[3] = 0x00;
    // scales: all 1
    for i in 4..16 {
        raw_block[i] = 1;
    }
    // quants: all 2
    for i in 16..144 {
        raw_block[i] = 0x22;
    }

    let mut out = vec![0.0f32; 256];
    dequantize_q4_k_to_f32(&raw_block, &mut out);

    assert_eq!(out.len(), 256);
    assert!(out[0] >= 0.0);
}

#[test]
fn test_q6_k_dequantize() {
    let mut raw_block = vec![0u8; 210];
    // scale d = 1.0 (fp16: 0x3C00) at offset 208..209
    raw_block[208] = 0x00;
    raw_block[209] = 0x3C;
    // scales: all 1
    for i in 192..208 {
        raw_block[i] = 1;
    }

    let mut out = vec![0.0f32; 256];
    dequantize_q6_k_to_f32(&raw_block, &mut out);

    assert_eq!(out.len(), 256);
}
