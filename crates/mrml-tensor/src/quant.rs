#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "std")]
#[inline]
fn round(value: f32) -> f32 {
    value.round()
}

#[cfg(not(feature = "std"))]
#[inline]
fn round(value: f32) -> f32 {
    mrml_math::round(value)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2")
    }
}

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
    pub d: u16,       // fp16 delta scale as raw u16
    pub qs: [u8; 16], // 32 nibbles (4-bit weights)
}

/// Block structure for Q8_0 quantization (32 weights per 34-byte block)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: u16,       // fp16 delta scale as raw u16
    pub qs: [i8; 32], // 32 8-bit weights
}

/// Dequantize a contiguous buffer of Q4_0 blocks to F32 output
pub fn dequantize_q4_0(data: &[u8], output: &mut [f32]) {
    assert!(
        data.len() % 18 == 0,
        "Q4_0 data size must be multiple of 18 bytes"
    );
    let n_blocks = data.len() / 18;
    assert!(
        output.len() >= n_blocks * 32,
        "Output buffer too small for dequantize_q4_0"
    );

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
    assert!(
        data.len() % 34 == 0,
        "Q8_0 data size must be multiple of 34 bytes"
    );
    let n_blocks = data.len() / 34;
    assert!(
        output.len() >= n_blocks * 32,
        "Output buffer too small for dequantize_q8_0"
    );

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
    assert!(
        dst.len() >= n_blocks * 34,
        "Destination buffer too small for Q8_0"
    );

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
            let v = round(src_block[i] * id);
            let q = v.clamp(-128.0, 127.0) as i8;
            dst[dst_offset + 2 + i] = q as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn avx_vnni_dpwssd(acc: __m256i, lhs: __m256i, rhs: __m256i) -> __m256i {
    let mut result = acc;
    core::arch::asm!(
        // VEX.256.66.0F38.W0 52 /r: vpdpwssd ymm0, ymm1, ymm2.
        // The mnemonic currently selects the EVEX/AVX-512VL encoding in LLVM's
        // integrated assembler, so spell out the AVX-VNNI encoding.
        ".byte 0xc4, 0xe2, 0x75, 0x52, 0xc2",
        inout("ymm0") result,
        in("ymm1") lhs,
        in("ymm2") rhs,
        options(pure, nomem, nostack),
    );
    result
}

/// Fast dot product between Q4_0 row (weights) and Q8_0 column (activations) with AVX2 SIMD
#[inline]
pub fn vec_dot_q4_0_q8_0(w_q4: &[u8], a_q8: &[u8], n_elements: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx_vnni_enabled() {
            unsafe {
                return vec_dot_q4_0_q8_0_avx_vnni(w_q4, a_q8, n_elements);
            }
        }
        if avx2_enabled() {
            unsafe {
                return vec_dot_q4_0_q8_0_avx2(w_q4, a_q8, n_elements);
            }
        }
    }

    vec_dot_q4_0_q8_0_scalar(w_q4, a_q8, n_elements)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q4_0_q8_0_avx2(w_q4: &[u8], a_q8: &[u8], n_elements: usize) -> f32 {
    let n_blocks = n_elements / 32;
    let mut sum = 0.0f32;

    let mask_low = _mm_set1_epi8(0x0F);
    let offset_eight = _mm_set1_epi8(8);

    for b in 0..n_blocks {
        let w_off = b * 18;
        let a_off = b * 34;

        let d_w_raw =
            u16::from_le_bytes([*w_q4.get_unchecked(w_off), *w_q4.get_unchecked(w_off + 1)]);
        let d_a_raw =
            u16::from_le_bytes([*a_q8.get_unchecked(a_off), *a_q8.get_unchecked(a_off + 1)]);

        let d_w = f16_to_f32(d_w_raw);
        let d_a = f16_to_f32(d_a_raw);

        let w_ptr = w_q4.as_ptr().add(w_off + 2);
        let a_ptr = a_q8.as_ptr().add(a_off + 2);

        let q4_128 = _mm_loadu_si128(w_ptr as *const __m128i);
        let q8_low_128 = _mm_loadu_si128(a_ptr as *const __m128i);
        let q8_high_128 = _mm_loadu_si128(a_ptr.add(16) as *const __m128i);

        let q4_low = _mm_sub_epi8(_mm_and_si128(q4_128, mask_low), offset_eight);
        let q4_high = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi16(q4_128, 4), mask_low),
            offset_eight,
        );

        // Sign-extend 8-bit to 16-bit
        let w_low_lo = _mm_cvtepi8_epi16(q4_low);
        let w_low_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q4_low, 8));
        let a_low_lo = _mm_cvtepi8_epi16(q8_low_128);
        let a_low_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q8_low_128, 8));

        let w_high_lo = _mm_cvtepi8_epi16(q4_high);
        let w_high_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q4_high, 8));
        let a_high_lo = _mm_cvtepi8_epi16(q8_high_128);
        let a_high_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q8_high_128, 8));

        let p_low_lo = _mm_madd_epi16(w_low_lo, a_low_lo);
        let p_low_hi = _mm_madd_epi16(w_low_hi, a_low_hi);
        let p_high_lo = _mm_madd_epi16(w_high_lo, a_high_lo);
        let p_high_hi = _mm_madd_epi16(w_high_hi, a_high_hi);

        let sum_low = _mm_add_epi32(p_low_lo, p_low_hi);
        let sum_high = _mm_add_epi32(p_high_lo, p_high_hi);
        let sum_all = _mm_add_epi32(sum_low, sum_high);

        let mut acc = [0i32; 4];
        _mm_storeu_si128(acc.as_mut_ptr() as *mut __m128i, sum_all);
        let block_sum = acc[0] + acc[1] + acc[2] + acc[3];

        sum += (block_sum as f32) * (d_w * d_a);
    }

    sum
}

#[inline]
fn vec_dot_q4_0_q8_0_scalar(w_q4: &[u8], a_q8: &[u8], n_elements: usize) -> f32 {
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

/// Quantize F32 values into GGML Q4_0 blocks (32 values per 18-byte block).
pub fn quantize_f32_to_q4_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % 32, 0);
    assert!(dst.len() >= src.len() / 32 * 18);
    for (block_index, values) in src.chunks_exact(32).enumerate() {
        let max_abs = values
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 8.0 } else { 0.0 };
        let offset = block_index * 18;
        dst[offset..offset + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for index in 0..16 {
            let low = (round(values[index] * inverse) as i32).clamp(-8, 7) + 8;
            let high = (round(values[index + 16] * inverse) as i32).clamp(-8, 7) + 8;
            dst[offset + 2 + index] = low as u8 | ((high as u8) << 4);
        }
    }
}

/// AVX-VNNI is opt-in until all supported Rust/LLVM assemblers are known to
/// emit the VEX form. Some toolchains incorrectly encode these intrinsics as
/// AVX-512VL, which faults on otherwise AVX-VNNI-capable processors.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx_vnni_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avxvnni")
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use mrml_runtime::Vector;

    #[test]
    fn dispatched_dots_match_scalar() {
        let values: Vector<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect();
        let other: Vector<f32> = (0..256)
            .map(|i| ((i * 19 % 89) as f32 - 44.0) / 11.0)
            .collect();
        let mut q4 = Vector::new();
        let mut q8_a = Vector::new();
        let mut q8_b = Vector::new();
        q4.resize(values.len() / 32 * 18, 0u8);
        q8_a.resize(values.len() / 32 * 34, 0u8);
        q8_b.resize(other.len() / 32 * 34, 0u8);
        quantize_f32_to_q4_0(&values, &mut q4);
        quantize_f32_to_q8_0(&values, &mut q8_a);
        quantize_f32_to_q8_0(&other, &mut q8_b);
        let q4_scalar = vec_dot_q4_0_q8_0_scalar(&q4, &q8_b, values.len());
        let q8_scalar = vec_dot_q8_0_q8_0_scalar(&q8_a, &q8_b, values.len());
        assert!((vec_dot_q4_0_q8_0(&q4, &q8_b, values.len()) - q4_scalar).abs() < 1e-3);
        assert!((vec_dot_q8_0_q8_0(&q8_a, &q8_b, values.len()) - q8_scalar).abs() < 1e-3);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vec_dot_q4_0_q8_0_avx_vnni(w_q4: &[u8], a_q8: &[u8], n_elements: usize) -> f32 {
    let mut sum = 0.0f32;
    let mask = _mm_set1_epi8(0x0f);
    for block in 0..n_elements / 32 {
        let w_off = block * 18;
        let a_off = block * 34;
        let scale = f16_to_f32(u16::from_le_bytes([
            *w_q4.get_unchecked(w_off),
            *w_q4.get_unchecked(w_off + 1),
        ])) * f16_to_f32(u16::from_le_bytes([
            *a_q8.get_unchecked(a_off),
            *a_q8.get_unchecked(a_off + 1),
        ]));
        let packed = _mm_loadu_si128(w_q4.as_ptr().add(w_off + 2) as *const __m128i);
        let low = _mm_sub_epi8(_mm_and_si128(packed, mask), _mm_set1_epi8(8));
        let high = _mm_sub_epi8(
            _mm_and_si128(_mm_srli_epi16(packed, 4), mask),
            _mm_set1_epi8(8),
        );
        let q8_low = _mm_loadu_si128(a_q8.as_ptr().add(a_off + 2) as *const __m128i);
        let q8_high = _mm_loadu_si128(a_q8.as_ptr().add(a_off + 18) as *const __m128i);
        let mut dot = avx_vnni_dpwssd(
            _mm256_setzero_si256(),
            _mm256_cvtepi8_epi16(low),
            _mm256_cvtepi8_epi16(q8_low),
        );
        dot = avx_vnni_dpwssd(
            dot,
            _mm256_cvtepi8_epi16(high),
            _mm256_cvtepi8_epi16(q8_high),
        );
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, dot);
        sum += lanes.iter().sum::<i32>() as f32 * scale;
    }
    sum
}

/// Dot product between two Q8_0 vectors. This is used by the tied Q8_0 token
/// embedding/output matrix and mirrors llama.cpp's Q8_0 x Q8_0 decode path.
#[inline]
pub fn vec_dot_q8_0_q8_0(x: &[u8], y: &[u8], n_elements: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx_vnni_enabled() {
        return unsafe { vec_dot_q8_0_q8_0_avx_vnni(x, y, n_elements) };
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        return unsafe { vec_dot_q8_0_q8_0_avx2(x, y, n_elements) };
    }

    vec_dot_q8_0_q8_0_scalar(x, y, n_elements)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vec_dot_q8_0_q8_0_avx_vnni(x: &[u8], y: &[u8], n_elements: usize) -> f32 {
    let mut sum = 0.0f32;
    for block in 0..n_elements / 32 {
        let x_off = block * 34;
        let y_off = block * 34;
        let scale = f16_to_f32(u16::from_le_bytes([
            *x.get_unchecked(x_off),
            *x.get_unchecked(x_off + 1),
        ])) * f16_to_f32(u16::from_le_bytes([
            *y.get_unchecked(y_off),
            *y.get_unchecked(y_off + 1),
        ]));
        let x_low = _mm_loadu_si128(x.as_ptr().add(x_off + 2) as *const __m128i);
        let x_high = _mm_loadu_si128(x.as_ptr().add(x_off + 18) as *const __m128i);
        let y_low = _mm_loadu_si128(y.as_ptr().add(y_off + 2) as *const __m128i);
        let y_high = _mm_loadu_si128(y.as_ptr().add(y_off + 18) as *const __m128i);
        let mut dot = avx_vnni_dpwssd(
            _mm256_setzero_si256(),
            _mm256_cvtepi8_epi16(x_low),
            _mm256_cvtepi8_epi16(y_low),
        );
        dot = avx_vnni_dpwssd(
            dot,
            _mm256_cvtepi8_epi16(x_high),
            _mm256_cvtepi8_epi16(y_high),
        );
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, dot);
        sum += lanes.iter().sum::<i32>() as f32 * scale;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q8_0_q8_0_avx2(x: &[u8], y: &[u8], n_elements: usize) -> f32 {
    let mut accum = _mm256_setzero_ps();
    for block in 0..n_elements / 32 {
        let off = block * 34;
        let dx = f16_to_f32(u16::from_le_bytes([
            *x.get_unchecked(off),
            *x.get_unchecked(off + 1),
        ]));
        let dy = f16_to_f32(u16::from_le_bytes([
            *y.get_unchecked(off),
            *y.get_unchecked(off + 1),
        ]));
        let xv = _mm256_loadu_si256(x.as_ptr().add(off + 2) as *const __m256i);
        let yv = _mm256_loadu_si256(y.as_ptr().add(off + 2) as *const __m256i);

        let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
        let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
        let y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yv));
        let y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yv, 1));
        let products =
            _mm256_add_epi32(_mm256_madd_epi16(x_lo, y_lo), _mm256_madd_epi16(x_hi, y_hi));
        let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(products), _mm256_set1_ps(dx * dy));
        accum = _mm256_add_ps(accum, scaled);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accum);
    lanes.iter().sum()
}

#[inline]
fn vec_dot_q8_0_q8_0_scalar(x: &[u8], y: &[u8], n_elements: usize) -> f32 {
    let mut sum = 0.0f32;
    for block in 0..n_elements / 32 {
        let off = block * 34;
        let dx = f16_to_f32(u16::from_le_bytes([x[off], x[off + 1]]));
        let dy = f16_to_f32(u16::from_le_bytes([y[off], y[off + 1]]));
        let mut block_sum = 0i32;
        for i in 0..32 {
            block_sum += (x[off + 2 + i] as i8 as i32) * (y[off + 2 + i] as i8 as i32);
        }
        sum += block_sum as f32 * (dx * dy);
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

/// Dequantize Q4_K buffer to F32 (block size: 256 weights per 144-byte block)
pub fn dequantize_q4_k_to_f32(src: &[u8], dst: &mut [f32]) {
    let n_blocks = dst.len() / 256;
    for b in 0..n_blocks {
        let b_src = &src[b * 144..(b + 1) * 144];
        let d_raw = u16::from_le_bytes([b_src[0], b_src[1]]);
        let dmin_raw = u16::from_le_bytes([b_src[2], b_src[3]]);
        let d = f16_to_f32(d_raw);
        let dmin = f16_to_f32(dmin_raw);

        let scales = &b_src[4..16];
        let qs = &b_src[16..144];

        let out = &mut dst[b * 256..(b + 1) * 256];
        for i in 0..32 {
            let sc = (scales[i / 4] & 0x3F) as f32;
            let m = ((scales[i / 4] >> 6) | ((scales[i / 4 + 4] & 0x03) << 2)) as f32;
            let dl = d * sc;
            let ml = dmin * m;

            let byte = qs[i];
            let q0 = (byte & 0x0F) as f32;
            let q1 = ((byte >> 4) & 0x0F) as f32;

            out[i * 2] = dl * q0 - ml;
            out[i * 2 + 1] = dl * q1 - ml;
        }
        for i in 32..128 {
            let sc = (scales[i / 16] & 0x3F) as f32;
            let m = ((scales[i / 16 + 4] >> 2) & 0x0F) as f32;
            let dl = d * sc;
            let ml = dmin * m;

            let byte = qs[i];
            let q0 = (byte & 0x0F) as f32;
            let q1 = ((byte >> 4) & 0x0F) as f32;

            out[i * 2] = dl * q0 - ml;
            out[i * 2 + 1] = dl * q1 - ml;
        }
    }
}

/// Dequantize Q6_K buffer to F32 (block size: 256 weights per 210-byte block)
pub fn dequantize_q6_k_to_f32(src: &[u8], dst: &mut [f32]) {
    let n_blocks = dst.len() / 256;
    for b in 0..n_blocks {
        let b_src = &src[b * 210..(b + 1) * 210];
        let ql = &b_src[0..128];
        let qh = &b_src[128..192];
        let scales = &b_src[192..208];
        let d_raw = u16::from_le_bytes([b_src[208], b_src[209]]);
        let d = f16_to_f32(d_raw);

        let out = &mut dst[b * 256..(b + 1) * 256];
        for i in 0..128 {
            let sc = scales[i / 16] as i8 as f32;
            let l_byte = ql[i];
            let h_byte = qh[i / 2];

            let h_val = if i % 2 == 0 {
                h_byte & 0x0F
            } else {
                (h_byte >> 4) & 0x0F
            };
            let q0 = ((l_byte & 0x0F) | ((h_val & 0x03) << 4)) as i32 - 32;
            let q1 = (((l_byte >> 4) & 0x0F) | (((h_val >> 2) & 0x03) << 4)) as i32 - 32;

            out[i * 2] = d * sc * (q0 as f32);
            out[i * 2 + 1] = d * sc * (q1 as f32);
        }
    }
}
