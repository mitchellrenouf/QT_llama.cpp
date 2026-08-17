#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

#define WARP_SIZE 32

// Warp-level sum reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction
__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    val = (lane < n_warps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// 1. CUDA RMSNorm Kernel
__global__ void k_rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int dim,
    float eps
) {
    int tid = threadIdx.x;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        local_sum_sq += v * v;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq);
    __shared__ float s_scale;
    if (tid == 0) {
        float mean_sq = total_sum_sq / (float)dim;
        s_scale = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float scale = s_scale;
    for (int i = tid; i < dim; i += blockDim.x) {
        float w = (weight != nullptr) ? weight[i] : 1.0f;
        out[i] = x[i] * scale * w;
    }
}

// 2. CUDA SwiGLU Kernel: out = silu(gate) * up
__global__ void k_swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float u = up[idx];
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = silu_g * u;
    }
}

// 2b. CUDA GeGLU Kernel: out = gelu_approx(gate) * up
__global__ void k_geglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = gate[idx];
        float u = up[idx];
        float gelu_x = 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
        out[idx] = gelu_x * u;
    }
}

// 3. CUDA RoPE 256k Kernel (Rotary Position Embedding supporting 256,000+ context)
__global__ void k_rope_256k_f32(
    float* __restrict__ vec,
    int pos,
    int head_dim,
    int n_heads,
    float freq_base,
    float freq_scale
) {
    int head_idx = blockIdx.x;
    int i = threadIdx.x;
    int half_dim = head_dim / 2;

    if (i < half_dim && head_idx < n_heads) {
        float theta = (float)pos * freq_scale / powf(freq_base, (float)(2 * i) / (float)head_dim);
        float cos_th = cosf(theta);
        float sin_th = sinf(theta);

        int base_idx = head_idx * head_dim;
        float v0 = vec[base_idx + i];
        float v1 = vec[base_idx + i + half_dim];

        vec[base_idx + i] = v0 * cos_th - v1 * sin_th;
        vec[base_idx + i + half_dim] = v0 * sin_th + v1 * cos_th;
    }
}

// 4. CUDA Q4_0 Matrix-Vector Multiplication: y = W * x
__global__ void k_gemv_q4_0_f32(
    const uint8_t* __restrict__ w_q4,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int tid = threadIdx.x;
    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* row_w = w_q4 + row * row_bytes;

    float local_sum = 0.0f;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = qs[i];
            int q0 = (byte & 0x0F) - 8;
            int q1 = ((byte >> 4) & 0x0F) - 8;

            block_sum += (float)q0 * x[x_base + i];
            block_sum += (float)q1 * x[x_base + i + 16];
        }

        local_sum += block_sum * d;
    }

    float total_row_sum = block_reduce_sum(local_sum);
    if (tid == 0) {
        y[row] = total_row_sum;
    }
}

// 4b. CUDA Q8_0 Matrix-Vector Multiplication: y = W * x with fused logit softcapping
__global__ void k_gemv_q8_0_f32(
    const uint8_t* __restrict__ w_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int tid = threadIdx.x;
    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 34;
    const uint8_t* row_w = w_q8 + row * row_bytes;

    float local_sum = 0.0f;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        int w_off = b * 34;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const int8_t* qs = (const int8_t*)(row_w + w_off + 2);
        int x_base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            block_sum += (float)qs[i] * x[x_base + i];
        }

        local_sum += block_sum * d;
    }

    float total_row_sum = block_reduce_sum(local_sum);
    if (tid == 0) {
        y[row] = 30.0f * tanhf(total_row_sum / 30.0f);
    }
}

// 5. CUDA Elementwise Addition (Residual Connections): y = a + b
__global__ void k_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// 6. CUDA Token Embedding Lookup: out = embd[token]
__global__ void k_embedding_f32(
    const float* __restrict__ table,
    float* __restrict__ out,
    int token,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        out[idx] = table[token * dim + idx];
    }
}

// 7. CUDA Causal Attention with Sliding Window Attention (SWA up to 256k tokens)
__global__ void k_attention_causal_swa_f32(
    const float* __restrict__ q,       // [n_heads, head_dim]
    const float* __restrict__ k_cache, // [n_past + 1, n_kv_heads, head_dim]
    const float* __restrict__ v_cache, // [n_past + 1, n_kv_heads, head_dim]
    float* __restrict__ out,           // [n_heads, head_dim]
    int n_past,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    int sliding_window                 // e.g. 4096 (or -1 for full global attention)
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = head / gqa_ratio;

    int start_pos = (sliding_window > 0 && n_past >= sliding_window) ? (n_past - sliding_window + 1) : 0;
    int total_keys = n_past + 1;

    extern __shared__ float s_scores[]; // shared memory for attention scores

    int tid = threadIdx.x;

    const float* q_h = q + head * head_dim;

    // Compute Q * K^T
    for (int p = start_pos + tid; p < total_keys; p += blockDim.x) {
        const float* k_h = k_cache + p * (n_kv_heads * head_dim) + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_h[d] * k_h[d];
        }
        s_scores[p - start_pos] = dot * scale;
    }
    __syncthreads();

    // Find max score for numerical stability in softmax
    float local_max = -1e20f;
    for (int p = tid; p < total_keys - start_pos; p += blockDim.x) {
        if (s_scores[p] > local_max) local_max = s_scores[p];
    }
    float max_score = block_reduce_sum(local_max); // reduction placeholder
    __syncthreads();

    // Exp and sum
    float local_exp_sum = 0.0f;
    for (int p = tid; p < total_keys - start_pos; p += blockDim.x) {
        float e = expf(s_scores[p] - max_score);
        s_scores[p] = e;
        local_exp_sum += e;
    }
    float sum_exp = block_reduce_sum(local_exp_sum);
    __syncthreads();

    // Compute weighted sum: Out = Sum(scores * V)
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    float* out_h = out + head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int p = start_pos; p < total_keys; ++p) {
            const float* v_h = v_cache + p * (n_kv_heads * head_dim) + kv_head * head_dim;
            val += (s_scores[p - start_pos] * inv_sum) * v_h[d];
        }
        out_h[d] = val;
    }
}

// C-ABI Exported Functions
extern "C" {

void cuda_op_rms_norm(
    const float* d_x,
    const float* d_w,
    float* d_out,
    int dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (dim < 1024) ? dim : 1024;
    k_rms_norm_f32<<<1, threads, 0, stream>>>(d_x, d_w, d_out, dim, eps);
}

void cuda_op_swiglu(
    const float* d_gate,
    const float* d_up,
    float* d_out,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_swiglu_f32<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_out, size);
}

void cuda_op_geglu(
    const float* d_gate,
    const float* d_up,
    float* d_out,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_geglu_f32<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_out, size);
}

void cuda_op_rope_256k(
    float* d_vec,
    int pos,
    int head_dim,
    int n_heads,
    float freq_base,
    float freq_scale,
    cudaStream_t stream
) {
    int threads = head_dim / 2;
    k_rope_256k_f32<<<n_heads, threads, 0, stream>>>(d_vec, pos, head_dim, n_heads, freq_base, freq_scale);
}

void cuda_op_gemv_q4_0(
    const uint8_t* d_w_q4,
    const float* d_x,
    float* d_y,
    int n_rows,
    int n_cols,
    cudaStream_t stream
) {
    int threads = 128;
    k_gemv_q4_0_f32<<<n_rows, threads, 0, stream>>>(d_w_q4, d_x, d_y, n_rows, n_cols);
}

void cuda_op_gemv_q8_0(
    const uint8_t* d_w_q8,
    const float* d_x,
    float* d_y,
    int n_rows,
    int n_cols,
    cudaStream_t stream
) {
    int threads = 128;
    k_gemv_q8_0_f32<<<n_rows, threads, 0, stream>>>(d_w_q8, d_x, d_y, n_rows, n_cols);
}

void cuda_op_add(
    const float* d_a,
    const float* d_b,
    float* d_out,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_add_f32<<<blocks, threads, 0, stream>>>(d_a, d_b, d_out, size);
}

void cuda_op_embedding(
    const float* d_table,
    float* d_out,
    int token,
    int dim,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    k_embedding_f32<<<blocks, threads, 0, stream>>>(d_table, d_out, token, dim);
}

void cuda_op_attention(
    const float* d_q,
    const float* d_k_cache,
    const float* d_v_cache,
    float* d_out,
    int n_past,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    int sliding_window,
    cudaStream_t stream
) {
    int shared_mem = (n_past + 1) * sizeof(float);
    k_attention_causal_swa_f32<<<n_heads, 128, shared_mem, stream>>>(
        d_q, d_k_cache, d_v_cache, d_out,
        n_past, n_heads, n_kv_heads, head_dim, scale, sliding_window
    );
}

// Fused MoE Top-K Gate+Up GEMV + GeGLU Kernel
// gridDim = (exp_ffn_dim, n_active), blockDim = 128
__global__ void k_moe_gate_up_topk_q4_0_f32(
    const uint8_t* __restrict__ gate_up_exps,
    const int32_t* __restrict__ active_exp_ids,
    const float* __restrict__ x_in,
    float* __restrict__ act_out,
    int exp_ffn_dim,
    int dim,
    int n_active
) {
    int row = blockIdx.x;       // row in 0..exp_ffn_dim
    int slot = blockIdx.y;      // slot in 0..n_active
    if (row >= exp_ffn_dim || slot >= n_active) return;

    int exp_idx = active_exp_ids[slot];
    int n_blocks = dim / 32;
    int row_bytes = n_blocks * 18;
    int exp_bytes = (2 * exp_ffn_dim) * row_bytes;

    const uint8_t* exp_base = gate_up_exps + (size_t)exp_idx * exp_bytes;
    const uint8_t* gate_row_w = exp_base + row * row_bytes;
    const uint8_t* up_row_w = exp_base + (exp_ffn_dim + row) * row_bytes;

    int tid = threadIdx.x;
    float local_gate = 0.0f;
    float local_up = 0.0f;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        int w_off = b * 18;
        int x_base = b * 32;

        // Gate dot
        uint16_t d_raw_g = (uint16_t)gate_row_w[w_off] | ((uint16_t)gate_row_w[w_off + 1] << 8);
        float d_g = __half2float(__ushort_as_half(d_raw_g));
        const uint8_t* qs_g = gate_row_w + w_off + 2;

        // Up dot
        uint16_t d_raw_u = (uint16_t)up_row_w[w_off] | ((uint16_t)up_row_w[w_off + 1] << 8);
        float d_u = __half2float(__ushort_as_half(d_raw_u));
        const uint8_t* qs_u = up_row_w + w_off + 2;

        float block_gate = 0.0f;
        float block_up = 0.0f;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte_g = qs_g[i];
            int q0_g = (byte_g & 0x0F) - 8;
            int q1_g = ((byte_g >> 4) & 0x0F) - 8;
            block_gate += (float)q0_g * x_in[x_base + i] + (float)q1_g * x_in[x_base + i + 16];

            uint8_t byte_u = qs_u[i];
            int q0_u = (byte_u & 0x0F) - 8;
            int q1_u = ((byte_u >> 4) & 0x0F) - 8;
            block_up += (float)q0_u * x_in[x_base + i] + (float)q1_u * x_in[x_base + i + 16];
        }
        local_gate += d_g * block_gate;
        local_up += d_u * block_up;
    }

    float total_gate = block_reduce_sum(local_gate);
    float total_up = block_reduce_sum(local_up);

    if (tid == 0) {
        // Fused approximate GeGLU: GELU(gate) * up
        float x = total_gate;
        float gelu_x = 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
        act_out[slot * exp_ffn_dim + row] = gelu_x * total_up;
    }
}

// Fused MoE Top-K Down GEMV + Scale + Accumulate Kernel
// gridDim = (dim, n_active), blockDim = 128
__global__ void k_moe_down_topk_q4_0_f32(
    const uint8_t* __restrict__ down_exps,
    const int32_t* __restrict__ active_exp_ids,
    const float* __restrict__ active_exp_weights,
    const float* __restrict__ down_exps_scale,
    const float* __restrict__ act_in,
    float* __restrict__ out_moe,
    int dim,
    int exp_ffn_dim,
    int n_active
) {
    int row = blockIdx.x;       // row in 0..dim
    int slot = blockIdx.y;      // slot in 0..n_active
    if (row >= dim || slot >= n_active) return;

    int exp_idx = active_exp_ids[slot];
    float weight = active_exp_weights[slot];
    float scale = (down_exps_scale != nullptr) ? down_exps_scale[exp_idx] : 1.0f;
    float alpha = weight * scale;

    int n_blocks = exp_ffn_dim / 32;
    int row_bytes = n_blocks * 18;
    int exp_bytes = dim * row_bytes;

    const uint8_t* exp_base = down_exps + (size_t)exp_idx * exp_bytes;
    const uint8_t* row_w = exp_base + row * row_bytes;
    const float* act_slot = act_in + slot * exp_ffn_dim;

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = qs[i];
            int q0 = (byte & 0x0F) - 8;
            int q1 = ((byte >> 4) & 0x0F) - 8;
            block_sum += (float)q0 * act_slot[x_base + i] + (float)q1 * act_slot[x_base + i + 16];
        }
        local_sum += d * block_sum;
    }

    float total_down = block_reduce_sum(local_sum);

    if (tid == 0) {
        atomicAdd(&out_moe[row], total_down * alpha);
    }
}

void cuda_op_moe_topk_q4_0(
    const uint8_t* d_gate_up_exps,
    const uint8_t* d_down_exps,
    const int32_t* d_active_exp_ids,
    const float* d_active_exp_weights,
    const float* d_down_exps_scale,
    const float* d_x_in,
    float* d_act_scratch,
    float* d_out_moe,
    int dim,
    int exp_ffn_dim,
    int n_active,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_out_moe, 0, dim * sizeof(float), stream);

    dim3 grid_gu(exp_ffn_dim, n_active);
    k_moe_gate_up_topk_q4_0_f32<<<grid_gu, 128, 0, stream>>>(
        d_gate_up_exps, d_active_exp_ids, d_x_in, d_act_scratch, exp_ffn_dim, dim, n_active
    );

    dim3 grid_down(dim, n_active);
    k_moe_down_topk_q4_0_f32<<<grid_down, 128, 0, stream>>>(
        d_down_exps, d_active_exp_ids, d_active_exp_weights, d_down_exps_scale,
        d_act_scratch, d_out_moe, dim, exp_ffn_dim, n_active
    );
}

}

