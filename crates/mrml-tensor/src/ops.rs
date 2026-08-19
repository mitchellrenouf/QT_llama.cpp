use crate::quant::{quantize_f32_to_q8_0, vec_dot_q4_0_q8_0};
use mrml_runtime::Vector;

macro_rules! math_fn {
    ($name:ident, $method:ident, $native:path) => {
        #[cfg(feature = "std")]
        #[inline]
        fn $name(value: f32) -> f32 {
            value.$method()
        }
        #[cfg(not(feature = "std"))]
        #[inline]
        fn $name(value: f32) -> f32 {
            $native(value)
        }
    };
}
math_fn!(sqrt, sqrt, mrml_math::sqrt);
math_fn!(cos, cos, mrml_math::cos);
math_fn!(sin, sin, mrml_math::sin);
math_fn!(tanh, tanh, mrml_math::tanh);
math_fn!(exp, exp, mrml_math::exp);

#[cfg(feature = "std")]
#[inline]
fn pow(value: f32, exponent: f32) -> f32 {
    value.powf(exponent)
}
#[cfg(not(feature = "std"))]
#[inline]
fn pow(value: f32, exponent: f32) -> f32 {
    mrml_math::pow(value, exponent)
}

/// In-place or out-of-place RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let dim = x.len();

    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }

    let mean_sq = sum_sq / (dim as f32);
    let scale = 1.0f32 / sqrt(mean_sq + eps);

    if let Some(w) = weight {
        let w_slice = if w.len() >= dim { &w[..dim] } else { w };
        for i in 0..dim.min(w_slice.len()) {
            out[i] = x[i] * scale * w_slice[i];
        }
        for i in w_slice.len()..dim {
            out[i] = x[i] * scale;
        }
    } else {
        for i in 0..dim {
            out[i] = x[i] * scale;
        }
    }
}

/// In-place RMS Normalization: x = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm_inplace(x: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let dim = x.len();
    if dim == 0 {
        return;
    }

    let mut sum_sq = 0.0f32;
    for &val in x.iter() {
        sum_sq += val * val;
    }

    let mean_sq = sum_sq / (dim as f32);
    let scale = 1.0f32 / sqrt(mean_sq + eps);

    if let Some(w) = weight {
        let w_slice = if w.len() >= dim { &w[..dim] } else { w };
        for i in 0..dim.min(w_slice.len()) {
            x[i] = x[i] * scale * w_slice[i];
        }
        for i in w_slice.len()..dim {
            x[i] = x[i] * scale;
        }
    } else {
        for i in 0..dim {
            x[i] = x[i] * scale;
        }
    }
}

/// Rotary Positional Embedding (RoPE) for query/key vectors
pub fn rope_1d(vec: &mut [f32], pos: usize, head_dim: usize, freq_base: f32, freq_scale: f32) {
    let half_dim = head_dim / 2;
    let theta_base = freq_base;

    for i in 0..half_dim {
        let theta = (pos as f32) * freq_scale / pow(theta_base, (2 * i) as f32 / head_dim as f32);
        let cos_th = cos(theta);
        let sin_th = sin(theta);

        let v0 = vec[i];
        let v1 = vec[i + half_dim];

        vec[i] = v0 * cos_th - v1 * sin_th;
        vec[i + half_dim] = v0 * sin_th + v1 * cos_th;
    }
}

/// GELU approximate activation (Gemma): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu_approx(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608f32;
    0.5f32 * x * (1.0f32 + tanh(sqrt_2_over_pi * (x + 0.044715f32 * x * x * x)))
}

/// GeGLU forward elementwise: out = gelu_approx(gate) * up
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());

    for i in 0..gate.len() {
        out[i] = gelu_approx(gate[i]) * up[i];
    }
}

/// SiLU activation: x / (1 + exp(-x))
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + exp(-x))
}

/// SwiGLU forward elementwise: out = silu(gate) * up
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());

    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Softmax over a vector of logits
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    let mut max_val = logits[0];
    for &val in logits.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum_exp = 0.0f32;
    for val in logits.iter_mut() {
        let e = exp(*val - max_val);
        *val = e;
        sum_exp += e;
    }

    if sum_exp > 0.0 {
        let inv_sum = 1.0f32 / sum_exp;
        for val in logits.iter_mut() {
            *val *= inv_sum;
        }
    }
}

/// Quantized Matrix-Vector Multiplication: y = W * x (where W is Q4_0 and x is F32)
/// Using Q8_0 activation quantization with Rayon thread pool
pub fn mat_vec_mul_q4_0(
    w_q4_bytes: &[u8],
    x_f32: &[f32],
    y_out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    if w_q4_bytes.is_empty() {
        return;
    }
    assert_eq!(x_f32.len(), n_cols);
    assert_eq!(y_out.len(), n_rows);

    let mut x_q8 = Vector::new();
    x_q8.resize(q8_0_size(n_cols), 0u8);
    quantize_f32_to_q8_0(x_f32, &mut x_q8);

    mat_vec_mul_q4_0_q8_0(w_q4_bytes, &x_q8, y_out, n_rows, n_cols);
}

/// Apply identical RoPE frequencies to a contiguous set of attention heads.
/// Trigonometric values depend on position and dimension, not on the head, so
/// compute them once instead of once per head.
pub fn rope_1d_batched(
    vec: &mut [f32],
    pos: usize,
    n_heads: usize,
    head_dim: usize,
    freq_base: f32,
    freq_scale: f32,
) {
    assert_eq!(vec.len(), n_heads * head_dim);
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let theta = (pos as f32) * freq_scale / pow(freq_base, (2 * i) as f32 / head_dim as f32);
        let cos_th = cos(theta);
        let sin_th = sin(theta);

        for head in vec.chunks_exact_mut(head_dim) {
            let v0 = head[i];
            let v1 = head[i + half_dim];
            head[i] = v0 * cos_th - v1 * sin_th;
            head[i + half_dim] = v0 * sin_th + v1 * cos_th;
        }
    }
}

/// Number of bytes required to hold `n_cols` Q8_0 activations.
#[inline]
pub fn q8_0_size(n_cols: usize) -> usize {
    n_cols.div_ceil(32) * 34
}

/// Matrix-vector multiplication using an already quantized activation vector.
///
/// Transformer projections commonly reuse the same input (Q/K/V and gate/up).
/// Keeping quantization outside this function avoids doing identical work and
/// allocating an identical temporary for every projection.
pub fn mat_vec_mul_q4_0_q8_0(
    w_q4_bytes: &[u8],
    x_q8: &[u8],
    y_out: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    if w_q4_bytes.is_empty() {
        return;
    }
    assert_eq!(y_out.len(), n_rows);
    assert!(x_q8.len() >= q8_0_size(n_cols));

    let row_bytes = n_cols.div_ceil(32) * 18;

    if n_rows <= 64 {
        for (r, y) in y_out.iter_mut().enumerate() {
            let row_start = r * row_bytes;
            if row_start + row_bytes <= w_q4_bytes.len() {
                let row_slice = &w_q4_bytes[row_start..row_start + row_bytes];
                *y = vec_dot_q4_0_q8_0(row_slice, &x_q8, n_cols);
            }
        }
    } else {
        let output_address = y_out.as_mut_ptr() as usize;
        crate::parallel::for_each_range(y_out.len(), 64, |start, end| {
            for r in start..end {
                let row_start = r * row_bytes;
                if row_start + row_bytes <= w_q4_bytes.len() {
                    let row_slice = &w_q4_bytes[row_start..row_start + row_bytes];
                    // SAFETY: each worker owns a disjoint row range.
                    unsafe {
                        *(output_address as *mut f32).add(r) =
                            vec_dot_q4_0_q8_0(row_slice, &x_q8, n_cols);
                    }
                }
            }
        });
    }
}

/// Dense F32 Matrix-Matrix Multiplication: C = A * B
pub fn mat_mul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_row[p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
