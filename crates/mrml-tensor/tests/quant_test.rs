use mrml_tensor::quant::*;

#[test]
fn test_q8_0_quant_dequant_roundtrip() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let mut q8_bytes = vec![0u8; (64 / 32) * 34];
    quantize_f32_to_q8_0(&original, &mut q8_bytes);

    let mut restored = vec![0.0f32; 64];
    dequantize_q8_0(&q8_bytes, &mut restored);

    for i in 0..64 {
        let diff = (original[i] - restored[i]).abs();
        assert!(diff < 0.05, "Mismatch at index {}: original={}, restored={}", i, original[i], restored[i]);
    }
}

#[test]
fn test_q4_0_dequantize() {
    let mut block = vec![0u8; 18];
    block[0] = 0x00; // fp16 1.0 LSB
    block[1] = 0x3C; // fp16 1.0 MSB

    for i in 0..16 {
        block[2 + i] = 0x88; // low nibble = 8 (val=0), high nibble = 8 (val=0)
    }

    let mut out = vec![0.0f32; 32];
    dequantize_q4_0(&block, &mut out);

    for i in 0..32 {
        assert_eq!(out[i], 0.0f32);
    }
}

#[test]
fn test_vec_dot_q4_q8() {
    let mut q4_block = vec![0u8; 18];
    q4_block[0] = 0x00;
    q4_block[1] = 0x3C; // d = 1.0
    for i in 0..16 {
        q4_block[2 + i] = 0x99; // low nibble = 9 (value = +1), high nibble = 9 (value = +1)
    }

    let activations = vec![1.0f32; 32];
    let mut q8_block = vec![0u8; 34];
    quantize_f32_to_q8_0(&activations, &mut q8_block);

    let dot = vec_dot_q4_0_q8_0(&q4_block, &q8_block, 32);
    assert!((dot - 32.0).abs() < 0.5, "Expected ~32.0, got {}", dot);
}

#[test]
fn test_vec_dot_q8_q8_matches_dequantized_reference() {
    let x: Vec<f32> = (0..96).map(|i| (i as f32 * 0.13).sin() * 3.0).collect();
    let y: Vec<f32> = (0..96).map(|i| (i as f32 * 0.07).cos() * 2.0).collect();
    let mut qx = vec![0u8; 3 * 34];
    let mut qy = vec![0u8; 3 * 34];
    quantize_f32_to_q8_0(&x, &mut qx);
    quantize_f32_to_q8_0(&y, &mut qy);

    let mut dx = vec![0.0; 96];
    let mut dy = vec![0.0; 96];
    dequantize_q8_0(&qx, &mut dx);
    dequantize_q8_0(&qy, &mut dy);
    let expected: f32 = dx.iter().zip(&dy).map(|(a, b)| a * b).sum();
    let actual = vec_dot_q8_0_q8_0(&qx, &qy, 96);

    assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
}
