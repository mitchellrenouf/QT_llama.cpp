use mrml_tensor::ops::{
    mat_vec_mul_q4_0, mat_vec_mul_q4_0_q8_0, q8_0_size, rope_1d, rope_1d_batched,
};
use mrml_tensor::quant::{f16_to_f32, f32_to_f16, quantize_f32_to_q8_0, vec_dot_q8_0_q8_0};
use mrml_tensor::{KvCacheFormat, KvCacheRow};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn elapsed_best(mut f: impl FnMut(), rounds: usize) -> Duration {
    (0..rounds)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .unwrap()
}

fn main() {
    let (rows, cols) = (1024, 2816);
    let mut weights = vec![0u8; rows * (cols / 32) * 18];
    for (b, block) in weights.chunks_exact_mut(18).enumerate() {
        block[..2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = (b.wrapping_mul(17).wrapping_add(i * 13)) as u8;
        }
    }
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin()).collect();
    let mut outputs = [vec![0.0f32; rows], vec![0.0; rows], vec![0.0; rows]];

    let repeated = elapsed_best(
        || {
            for output in &mut outputs {
                mat_vec_mul_q4_0(&weights, &input, output, rows, cols);
            }
            black_box(&outputs);
        },
        5,
    );

    let shared = elapsed_best(
        || {
            let mut q8 = vec![0u8; q8_0_size(cols)];
            quantize_f32_to_q8_0(&input, &mut q8);
            for output in &mut outputs {
                mat_vec_mul_q4_0_q8_0(&weights, &q8, output, rows, cols);
            }
            black_box(&outputs);
        },
        5,
    );

    let scores: Vec<(f32, i32)> = (0..262_144)
        .map(|i| (((i as f32 * 0.73).sin()), i))
        .collect();
    let full_sort = elapsed_best(
        || {
            let mut values = scores.clone();
            values.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            values.truncate(40);
            black_box(values);
        },
        5,
    );
    let top_k = elapsed_best(
        || {
            let mut values = scores.clone();
            values.select_nth_unstable_by(40, |a, b| b.0.total_cmp(&a.0));
            values.truncate(40);
            values.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            black_box(values);
        },
        5,
    );

    let heads = 16;
    let head_dim = 256;
    let rope_input: Vec<f32> = (0..heads * head_dim)
        .map(|i| (i as f32 * 0.017).cos())
        .collect();
    let per_head_rope = elapsed_best(
        || {
            let mut values = rope_input.clone();
            for head in values.chunks_exact_mut(head_dim) {
                rope_1d(head, 4096, head_dim, 10_000.0, 1.0);
            }
            black_box(values);
        },
        10,
    );
    let batched_rope = elapsed_best(
        || {
            let mut values = rope_input.clone();
            rope_1d_batched(&mut values, 4096, heads, head_dim, 10_000.0, 1.0);
            black_box(values);
        },
        10,
    );

    let vocab_rows = 4096;
    let q8_row_bytes = cols / 32 * 34;
    let mut q8_table = vec![0u8; vocab_rows * q8_row_bytes];
    for (block_index, block) in q8_table.chunks_exact_mut(34).enumerate() {
        block[..2].copy_from_slice(&f32_to_f16(0.015).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = block_index.wrapping_mul(31).wrapping_add(i * 7) as u8;
        }
    }
    let float_vocab = elapsed_best(
        || {
            let mut sum = 0.0f32;
            for row in q8_table.chunks_exact(q8_row_bytes) {
                let mut dot = 0.0f32;
                for block in 0..cols / 32 {
                    let off = block * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([row[off], row[off + 1]]));
                    for i in 0..32 {
                        dot += row[off + 2 + i] as i8 as f32 * input[block * 32 + i] * scale;
                    }
                }
                sum += dot;
            }
            black_box(sum);
        },
        3,
    );
    let mut input_q8 = vec![0u8; q8_row_bytes];
    quantize_f32_to_q8_0(&input, &mut input_q8);
    let quant_vocab = elapsed_best(
        || {
            let sum: f32 = q8_table
                .chunks_exact(q8_row_bytes)
                .map(|row| vec_dot_q8_0_q8_0(row, &input_q8, cols))
                .sum();
            black_box(sum);
        },
        3,
    );

    println!(
        "shared projection quantization: {:?} -> {:?} ({:.2}x)",
        repeated,
        shared,
        repeated.as_secs_f64() / shared.as_secs_f64()
    );
    println!(
        "vocabulary top-40: {:?} -> {:?} ({:.2}x)",
        full_sort,
        top_k,
        full_sort.as_secs_f64() / top_k.as_secs_f64()
    );
    println!(
        "batched RoPE: {:?} -> {:?} ({:.2}x)",
        per_head_rope,
        batched_rope,
        per_head_rope.as_secs_f64() / batched_rope.as_secs_f64()
    );
    println!(
        "Q8 vocabulary dot: {:?} -> {:?} ({:.2}x)",
        float_vocab,
        quant_vocab,
        float_vocab.as_secs_f64() / quant_vocab.as_secs_f64()
    );

    let kv_dim = 1024;
    let context = 8192;
    let head_dim = 256;
    let kv_values: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.031).sin()).collect();
    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.021).cos()).collect();
    let mut query_q8 = vec![0; head_dim / 32 * 34];
    quantize_f32_to_q8_0(&query, &mut query_q8);
    for format in [KvCacheFormat::F32, KvCacheFormat::Q8, KvCacheFormat::Q4] {
        let cache: Vec<_> = (0..context)
            .map(|_| KvCacheRow::from_f32(&kv_values, format))
            .collect();
        let bytes = match format {
            KvCacheFormat::F32 => 4 * kv_dim,
            KvCacheFormat::Q8 => 34 * kv_dim / 32,
            KvCacheFormat::Q4 => 18 * kv_dim / 32,
        };
        let scan = elapsed_best(
            || {
                let mut out = vec![0.0f32; head_dim];
                for row in &cache {
                    let score = row.dot_head(&query, &query_q8, 0);
                    row.add_head_scaled(&mut out, 0, score * 0.0001);
                }
                black_box(out);
            },
            5,
        );
        println!(
            "{format:?} KV attention scan (8k): {:?}, {:.1} MiB",
            scan,
            (bytes * context) as f64 / 1_048_576.0
        );
    }
}
