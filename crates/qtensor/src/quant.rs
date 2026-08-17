/// Fast bitwise conversion from IEEE 754 half-precision float (f16) to single-precision float (f32)
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ff;
            let exp_f32 = (127 - 15 + 1 - e) as u32;
            f32::from_bits((sign << 31) | (exp_f32 << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xff << 23)) // Inf
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13)) // NaN
        }
    } else {
        let exp_f32 = (exp + (127 - 15)) << 23;
        let mant_f32 = mant << 13;
        f32::from_bits((sign << 31) | exp_f32 | mant_f32)
    }
}

/// Fast bitwise conversion from single-precision float (f32) to half-precision float (f16)
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7fffff) as u32;

    if exp == 255 {
        if mant == 0 {
            (sign << 15) | 0x7c00
        } else {
            (sign << 15) | 0x7e00
        }
    } else {
        let exp16 = exp - 127 + 15;
        if exp16 >= 31 {
            (sign << 15) | 0x7c00
        } else if exp16 <= 0 {
            0
        } else {
            let mant16 = (mant >> 13) as u16;
            (sign << 15) | ((exp16 as u16) << 10) | mant16
        }
    }
}

/// Fast conversion from bfloat16 (bf16) to f32
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Fast conversion from f32 to bfloat16 (bf16)
#[inline]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Block structure for Q4_0 quantization (32 weights per 18-byte block)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: u16,          // fp16 delta scale as raw u16
    pub qs: [u8; 16],    // 32 nibbles (4-bit weights)
}

/// Block structure for Q8_0 quantization (32 weights per 34-byte block)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: u16,          // fp16 delta scale as raw u16
    pub qs: [i8; 32],    // 32 8-bit weights
}

/// Dequantize a contiguous buffer of Q4_0 blocks to F32 output
pub fn dequantize_q4_0(data: &[u8], output: &mut [f32]) {
    assert!(data.len() % 18 == 0, "Q4_0 data size must be multiple of 18 bytes");
    let n_blocks = data.len() / 18;
    assert!(output.len() >= n_blocks * 32, "Output buffer too small for dequantize_q4_0");

    for b in 0..n_blocks {
        let block_offset = b * 18;
        let d_raw = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
        let d = f16_to_f32(d_raw);

        let out_offset = b * 32;
        let qs = &data[block_offset + 2..block_offset + 18];

        for i in 0..16 {
            let byte = qs[i];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;

            output[out_offset + i] = (q0 as f32) * d;
            output[out_offset + i + 16] = (q1 as f32) * d;
        }
    }
}

/// Dequantize a contiguous buffer of Q8_0 blocks to F32 output
pub fn dequantize_q8_0(data: &[u8], output: &mut [f32]) {
    assert!(data.len() % 34 == 0, "Q8_0 data size must be multiple of 34 bytes");
    let n_blocks = data.len() / 34;
    assert!(output.len() >= n_blocks * 32, "Output buffer too small for dequantize_q8_0");

    for b in 0..n_blocks {
        let block_offset = b * 34;
        let d_raw = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
        let d = f16_to_f32(d_raw);

        let out_offset = b * 32;
        let qs = &data[block_offset + 2..block_offset + 34];

        for i in 0..32 {
            let q = qs[i] as i8;
            output[out_offset + i] = (q as f32) * d;
        }
    }
}

/// Quantize a slice of F32 activations into Q8_0 blocks for fast matrix multiplication
pub fn quantize_f32_to_q8_0(src: &[f32], dst: &mut [u8]) {
    assert!(src.len() % 32 == 0, "Source length must be multiple of 32");
    let n_blocks = src.len() / 32;
    assert!(dst.len() >= n_blocks * 34, "Destination buffer too small for Q8_0");

    for b in 0..n_blocks {
        let src_block = &src[b * 32..(b + 1) * 32];
        let dst_offset = b * 34;

        // Find max absolute value for scale
        let mut max_val = 0.0f32;
        for &val in src_block {
            let a = val.abs();
            if a > max_val {
                max_val = a;
            }
        }

        let scale = max_val / 127.0f32;
        let d_raw = f32_to_f16(scale);
        let id = if scale != 0.0 { 1.0f32 / scale } else { 0.0 };

        let d_bytes = d_raw.to_le_bytes();
        dst[dst_offset] = d_bytes[0];
        dst[dst_offset + 1] = d_bytes[1];

        for i in 0..32 {
            let v = (src_block[i] * id).round();
            let q = v.clamp(-128.0, 127.0) as i8;
            dst[dst_offset + 2 + i] = q as u8;
        }
    }
}

/// Fast dot product between Q4_0 row (weights) and Q8_0 column (activations)
#[inline]
pub fn vec_dot_q4_0_q8_0(w_q4: &[u8], a_q8: &[u8], n_elements: usize) -> f32 {
    let n_blocks = n_elements / 32;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let w_off = b * 18;
        let a_off = b * 34;

        let d_w_raw = u16::from_le_bytes([w_q4[w_off], w_q4[w_off + 1]]);
        let d_a_raw = u16::from_le_bytes([a_q8[a_off], a_q8[a_off + 1]]);

        let d_w = f16_to_f32(d_w_raw);
        let d_a = f16_to_f32(d_a_raw);

        let w_qs = &w_q4[w_off + 2..w_off + 18];
        let a_qs = &a_q8[a_off + 2..a_off + 34];

        let mut block_sum = 0i32;

        for i in 0..16 {
            let byte = w_qs[i];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;

            let a0 = a_qs[i] as i8 as i32;
            let a1 = a_qs[i + 16] as i8 as i32;

            block_sum += q0 * a0 + q1 * a1;
        }

        sum += (block_sum as f32) * (d_w * d_a);
    }

    sum
}

/// Dequantize F16 buffer to F32
pub fn dequantize_f16_to_f32(src: &[u8], dst: &mut [f32]) {
    assert!(src.len() >= dst.len() * 2);
    for i in 0..dst.len() {
        let raw = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        dst[i] = f16_to_f32(raw);
    }
}

/// Dequantize BF16 buffer to F32
pub fn dequantize_bf16_to_f32(src: &[u8], dst: &mut [f32]) {
    assert!(src.len() >= dst.len() * 2);
    for i in 0..dst.len() {
        let raw = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        dst[i] = bf16_to_f32(raw);
    }
}
