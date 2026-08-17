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
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float half_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, 16);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float block_reduce_max(float val) {
    __shared__ float shared_max[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) shared_max[wid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (lane < n_warps) ? shared_max[lane] : -3.402823466e+38F;
    if (wid == 0) {
        val = warp_reduce_max(val);
        if (lane == 0) shared_max[0] = val;
    }
    __syncthreads();
    return shared_max[0];
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
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int rows_per_block = (blockDim.x / WARP_SIZE) * 2;
    int row = blockIdx.x * rows_per_block + warp * 2 + (lane >> 4);
    bool active = row < n_rows;

    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* row_w = active ? w_q4 + (size_t)row * row_bytes : w_q4;

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        uint8_t byte = active ? qs[sublane] : 0;
        int q0 = (byte & 0x0F) - 8;
        int q1 = (byte >> 4) - 8;
        local_sum += active ? d * ((float)q0 * x[x_base + sublane] + (float)q1 * x[x_base + sublane + 16]) : 0.0f;
    }

    float total_row_sum = half_warp_reduce_sum(local_sum);
    if (active && sublane == 0) {
        y[row] = total_row_sum;
    }
}

// Fused Q/K/V projection. A single launch writes a contiguous result so the
// host needs one synchronization/copy instead of three per transformer layer.
__global__ void k_gemv_q4_0_qkv_f32(
    const uint8_t* __restrict__ w_q,
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    const float* __restrict__ x,
    float* __restrict__ y,
    int q_rows,
    int kv_rows,
    int n_cols
) {
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int total_rows = q_rows + 2 * kv_rows;
    int rows_per_block = (blockDim.x / WARP_SIZE) * 2;
    int out_row = blockIdx.x * rows_per_block + warp * 2 + (lane >> 4);
    bool active = out_row < total_rows;

    const uint8_t* matrix;
    int row;
    if (!active || out_row < q_rows) {
        matrix = w_q;
        row = out_row;
    } else if (out_row < q_rows + kv_rows) {
        matrix = w_k;
        row = out_row - q_rows;
    } else {
        matrix = w_v;
        row = out_row - q_rows - kv_rows;
    }

    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* row_w = active ? matrix + (size_t)row * row_bytes : matrix;
    float local_sum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));
        const uint8_t* qs = row_w + w_off + 2;
        uint8_t byte = active ? qs[sublane] : 0;
        int q0 = (byte & 0x0F) - 8;
        int q1 = (byte >> 4) - 8;
        local_sum += active ? d * ((float)q0 * x[b * 32 + sublane] + (float)q1 * x[b * 32 + sublane + 16]) : 0.0f;
    }
    float total = half_warp_reduce_sum(local_sum);
    if (active && sublane == 0) y[out_row] = total;
}

// Normalize projected heads, apply RoPE, and append K/V directly to the
// persistent device cache. This keeps Q/K/V off the host between projection
// and attention.
__global__ void k_qkv_postprocess_f32(
    float* __restrict__ qkv,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int pos,
    int cache_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float freq_base
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    bool is_q = head < n_heads;
    bool is_k = head >= n_heads && head < n_heads + n_kv_heads;
    int local_head = is_q ? head : head - n_heads;
    float* src = is_q ? qkv + local_head * head_dim
        : is_k ? qkv + q_dim + local_head * head_dim
        : qkv + q_dim + kv_dim + (head - n_heads - n_kv_heads) * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float value = src[i];
        sum_sq += value * value;
    }
    float total = block_reduce_sum(sum_sq);
    __shared__ float norm_scale;
    if (tid == 0) norm_scale = rsqrtf(total / (float)head_dim + 1e-6f);
    __syncthreads();

    if (is_q || is_k) {
        const float* weights = is_q ? q_norm : k_norm;
        for (int i = tid; i < head_dim / 2; i += blockDim.x) {
            float a = src[i] * norm_scale * weights[i];
            float b = src[i + head_dim / 2] * norm_scale * weights[i + head_dim / 2];
            float theta = (float)pos / powf(freq_base, (float)(2 * i) / (float)head_dim);
            float s, c;
            sincosf(theta, &s, &c);
            float r0 = a * c - b * s;
            float r1 = a * s + b * c;
            src[i] = r0;
            src[i + head_dim / 2] = r1;
            if (is_k) {
                size_t dst = (size_t)cache_pos * kv_dim + (size_t)local_head * head_dim;
                k_cache[dst + i] = r0;
                k_cache[dst + i + head_dim / 2] = r1;
            }
        }
    } else {
        int v_head = head - n_heads - n_kv_heads;
        size_t dst = (size_t)cache_pos * kv_dim + (size_t)v_head * head_dim;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float value = src[i] * norm_scale;
            src[i] = value;
            v_cache[dst + i] = value;
        }
    }
}

__global__ void k_gemv_q4_0_geglu_f32(
    const uint8_t* __restrict__ w_gate,
    const uint8_t* __restrict__ w_up,
    const float* __restrict__ x,
    float* __restrict__ act,
    int n_rows,
    int n_cols
) {
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int rows_per_block = (blockDim.x / WARP_SIZE) * 2;
    int row = blockIdx.x * rows_per_block + warp * 2 + (lane >> 4);
    bool active = row < n_rows;
    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* gate_row = active ? w_gate + (size_t)row * row_bytes : w_gate;
    const uint8_t* up_row = active ? w_up + (size_t)row * row_bytes : w_up;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        int off = b * 18;
        uint16_t dg_raw = (uint16_t)gate_row[off] | ((uint16_t)gate_row[off + 1] << 8);
        uint16_t du_raw = (uint16_t)up_row[off] | ((uint16_t)up_row[off + 1] << 8);
        float dg = __half2float(__ushort_as_half(dg_raw));
        float du = __half2float(__ushort_as_half(du_raw));
        uint8_t bg = active ? gate_row[off + 2 + sublane] : 0;
        uint8_t bu = active ? up_row[off + 2 + sublane] : 0;
        float x0 = x[b * 32 + sublane];
        float x1 = x[b * 32 + sublane + 16];
        gate_sum += active ? dg * ((float)((bg & 0x0F) - 8) * x0 + (float)((bg >> 4) - 8) * x1) : 0.0f;
        up_sum += active ? du * ((float)((bu & 0x0F) - 8) * x0 + (float)((bu >> 4) - 8) * x1) : 0.0f;
    }
    float gate = half_warp_reduce_sum(gate_sum);
    float up = half_warp_reduce_sum(up_sum);
    if (active && sublane == 0) {
        float gelu = 0.5f * gate * (1.0f + tanhf(0.7978845608f * gate * (1.0f + 0.044715f * gate * gate)));
        act[row] = gelu * up;
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
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int rows_per_block = (blockDim.x / WARP_SIZE) * 2;
    int row = blockIdx.x * rows_per_block + warp * 2 + (lane >> 4);
    bool active = row < n_rows;

    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 34;
    const uint8_t* row_w = active ? w_q8 + (size_t)row * row_bytes : w_q8;

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 34;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const int8_t* qs = (const int8_t*)(row_w + w_off + 2);
        int x_base = b * 32;

        local_sum += active ? d * ((float)qs[sublane] * x[x_base + sublane] + (float)qs[sublane + 16] * x[x_base + sublane + 16]) : 0.0f;
    }

    float total_row_sum = half_warp_reduce_sum(local_sum);
    if (active && sublane == 0) {
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

__global__ void k_moe_router_logits_f32(
    const float* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ logits,
    int dim,
    int n_experts
) {
    int expert = blockIdx.x;
    if (expert >= n_experts) return;
    const float* row = weights + (size_t)expert * dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += row[i] * input[i];
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) logits[expert] = sum;
}

__global__ void k_moe_router_top8_f32(
    const float* __restrict__ logits,
    int32_t* __restrict__ ids,
    float* __restrict__ probabilities,
    int n_experts
) {
    __shared__ float scores[128];
    __shared__ int32_t indices[128];
    int tid = threadIdx.x;
    if (tid < n_experts) {
        scores[tid] = logits[tid];
        indices[tid] = tid;
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) {
            int best = i;
            for (int j = i + 1; j < n_experts; ++j) {
                if (scores[j] > scores[best]) best = j;
            }
            float score = scores[i]; scores[i] = scores[best]; scores[best] = score;
            int32_t id = indices[i]; indices[i] = indices[best]; indices[best] = id;
        }
        float max_score = scores[0];
        float total = 0.0f;
        for (int i = 0; i < 8; ++i) {
            ids[i] = indices[i];
            probabilities[i] = expf(scores[i] - max_score);
            total += probabilities[i];
        }
        float inv = total > 0.0f ? 1.0f / total : 0.0f;
        for (int i = 0; i < 8; ++i) probabilities[i] *= inv;
    }
}

// Keep the post-attention residual and all three FFN inputs on device.  The
// unweighted RMS scale of attn_res is shared by the dense, MoE, and router
// inputs, so this replaces four host-side passes and three HtoD copies.
__global__ void k_prepare_ffn_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ attn_proj,
    const float* __restrict__ post_attn_norm,
    const float* __restrict__ ffn_norm,
    const float* __restrict__ pre_ffw_norm_2,
    const float* __restrict__ router_scale,
    float* __restrict__ attn_res,
    float* __restrict__ shared_in,
    float* __restrict__ moe_in,
    float* __restrict__ router_in,
    int dim
) {
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = attn_proj[i];
        sum_sq += v * v;
    }
    float total = block_reduce_sum(sum_sq);
    __shared__ float inv_proj;
    if (threadIdx.x == 0) inv_proj = rsqrtf(total / (float)dim + 1.0e-6f);
    __syncthreads();

    sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = hidden[i] + attn_proj[i] * inv_proj * post_attn_norm[i];
        attn_res[i] = v;
        sum_sq += v * v;
    }
    total = block_reduce_sum(sum_sq);
    __shared__ float inv_res;
    if (threadIdx.x == 0) inv_res = rsqrtf(total / (float)dim + 1.0e-6f);
    __syncthreads();

    float router_factor = inv_res * rsqrtf((float)dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = attn_res[i];
        shared_in[i] = v * inv_res * ffn_norm[i];
        moe_in[i] = v * inv_res * pre_ffw_norm_2[i];
        router_in[i] = v * router_factor * router_scale[i];
    }
}

// Normalize dense and expert outputs, combine them, apply the final FFN norm,
// and update the layer residual without returning intermediate tensors to CPU.
__global__ void k_finish_ffn_f32(
    const float* __restrict__ attn_res,
    float* __restrict__ dense,
    float* __restrict__ moe,
    const float* __restrict__ post_ffw_norm_1,
    const float* __restrict__ post_ffw_norm_2,
    const float* __restrict__ post_ffw_norm,
    float* __restrict__ hidden_out,
    float layer_scale,
    int dim
) {
    float dense_sq = 0.0f;
    float moe_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dense_sq += dense[i] * dense[i];
        moe_sq += moe[i] * moe[i];
    }
    float dense_total = block_reduce_sum(dense_sq);
    __shared__ float inv_dense;
    if (threadIdx.x == 0) inv_dense = rsqrtf(dense_total / (float)dim + 1.0e-6f);
    __syncthreads();
    float moe_total = block_reduce_sum(moe_sq);
    __shared__ float inv_moe;
    if (threadIdx.x == 0) inv_moe = rsqrtf(moe_total / (float)dim + 1.0e-6f);
    __syncthreads();

    float combined_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = dense[i] * inv_dense * post_ffw_norm_1[i]
                + moe[i] * inv_moe * post_ffw_norm_2[i];
        moe[i] = v;
        combined_sq += v * v;
    }
    float combined_total = block_reduce_sum(combined_sq);
    __shared__ float inv_combined;
    if (threadIdx.x == 0) inv_combined = rsqrtf(combined_total / (float)dim + 1.0e-6f);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        hidden_out[i] = (attn_res[i] + moe[i] * inv_combined * post_ffw_norm[i]) * layer_scale;
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
    float max_score = block_reduce_max(local_max);
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
    int rows_per_block = 2 * threads / WARP_SIZE;
    int blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    k_gemv_q4_0_f32<<<blocks, threads, 0, stream>>>(d_w_q4, d_x, d_y, n_rows, n_cols);
}

// Exact top-K candidates per disjoint vocabulary partition. The union of each
// partition's top K necessarily contains the global top K, while avoiding a
// full-vocabulary device-to-host copy.
__global__ void k_vocab_topk_f32(
    const float* __restrict__ logits,
    const uint8_t* __restrict__ valid,
    const int32_t* __restrict__ recent,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_ids,
    int vocab_size,
    int n_recent,
    int generated_count,
    int k
) {
    extern __shared__ unsigned char scratch[];
    float* warp_scores = reinterpret_cast<float*>(scratch);
    int32_t* warp_ids = reinterpret_cast<int32_t*>(warp_scores + 32);
    int32_t* selected = warp_ids + 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int partition_start = (int)(((long long)vocab_size * blockIdx.x) / gridDim.x);
    const int partition_end = (int)(((long long)vocab_size * (blockIdx.x + 1)) / gridDim.x);

    for (int rank = 0; rank < k; ++rank) {
        float best = -3.402823466e+38F;
        int32_t best_id = -1;
        for (int id = partition_start + threadIdx.x; id < partition_end; id += blockDim.x) {
            if (!valid[id]) continue;
            if (generated_count < 4 && (id == 1 || id == 2 || id == 105 || id == 106 || id == 107)) continue;
            bool already_selected = false;
            for (int r = 0; r < rank; ++r) already_selected |= selected[r] == id;
            if (already_selected) continue;
            float score = logits[id];
            for (int r = 0; r < n_recent; ++r) score -= recent[r] == id ? 1.8f : 0.0f;
            if (score > best || (score == best && id < best_id)) {
                best = score;
                best_id = id;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_score = __shfl_down_sync(0xffffffff, best, offset);
            int32_t other_id = __shfl_down_sync(0xffffffff, best_id, offset);
            if (other_score > best || (other_score == best && other_id >= 0 && (best_id < 0 || other_id < best_id))) {
                best = other_score;
                best_id = other_id;
            }
        }
        if (lane == 0) {
            warp_scores[warp] = best;
            warp_ids[warp] = best_id;
        }
        __syncthreads();
        if (warp == 0) {
            int n_warps = blockDim.x >> 5;
            best = lane < n_warps ? warp_scores[lane] : -3.402823466e+38F;
            best_id = lane < n_warps ? warp_ids[lane] : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_score = __shfl_down_sync(0xffffffff, best, offset);
                int32_t other_id = __shfl_down_sync(0xffffffff, best_id, offset);
                if (other_score > best || (other_score == best && other_id >= 0 && (best_id < 0 || other_id < best_id))) {
                    best = other_score;
                    best_id = other_id;
                }
            }
            if (lane == 0) {
                selected[rank] = best_id;
                out_scores[blockIdx.x * k + rank] = best;
                out_ids[blockIdx.x * k + rank] = best_id;
            }
        }
        __syncthreads();
    }
}

void cuda_op_gemv_q4_0_qkv(
    const uint8_t* d_w_q,
    const uint8_t* d_w_k,
    const uint8_t* d_w_v,
    const float* d_x,
    float* d_y,
    int q_rows,
    int kv_rows,
    int n_cols,
    cudaStream_t stream
) {
    int threads = 128;
    int rows_per_block = 2 * threads / WARP_SIZE;
    int total_rows = q_rows + 2 * kv_rows;
    int blocks = (total_rows + rows_per_block - 1) / rows_per_block;
    k_gemv_q4_0_qkv_f32<<<blocks, threads, 0, stream>>>(
        d_w_q, d_w_k, d_w_v, d_x, d_y, q_rows, kv_rows, n_cols
    );
}

void cuda_op_qkv_postprocess(
    float* d_qkv,
    const float* d_q_norm,
    const float* d_k_norm,
    float* d_k_cache,
    float* d_v_cache,
    int pos,
    int cache_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float freq_base,
    cudaStream_t stream
) {
    int blocks = n_heads + 2 * n_kv_heads;
    int threads = head_dim >= 512 ? 256 : 128;
    k_qkv_postprocess_f32<<<blocks, threads, 0, stream>>>(
        d_qkv, d_q_norm, d_k_norm, d_k_cache, d_v_cache,
        pos, cache_pos, n_heads, n_kv_heads, head_dim, freq_base
    );
}

void cuda_op_gemv_q4_0_geglu(
    const uint8_t* d_w_gate,
    const uint8_t* d_w_up,
    const float* d_x,
    float* d_act,
    int n_rows,
    int n_cols,
    cudaStream_t stream
) {
    int threads = 128;
    int rows_per_block = 2 * threads / WARP_SIZE;
    int blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    k_gemv_q4_0_geglu_f32<<<blocks, threads, 0, stream>>>(
        d_w_gate, d_w_up, d_x, d_act, n_rows, n_cols
    );
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
    int rows_per_block = 2 * threads / WARP_SIZE;
    int blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    k_gemv_q8_0_f32<<<blocks, threads, 0, stream>>>(d_w_q8, d_x, d_y, n_rows, n_cols);
}

void cuda_op_vocab_topk(
    const float* d_logits,
    const uint8_t* d_valid,
    const int32_t* d_recent,
    float* d_scores,
    int32_t* d_ids,
    int vocab_size,
    int n_recent,
    int generated_count,
    int k,
    int partitions,
    cudaStream_t stream
) {
    const int threads = 256;
    const int shared = (32 * (sizeof(float) + sizeof(int32_t))) + k * sizeof(int32_t);
    k_vocab_topk_f32<<<partitions, threads, shared, stream>>>(
        d_logits, d_valid, d_recent, d_scores, d_ids,
        vocab_size, n_recent, generated_count, k
    );
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

void cuda_op_moe_router(
    const float* d_weights,
    const float* d_input,
    float* d_logits,
    int32_t* d_ids,
    float* d_probabilities,
    int dim,
    int n_experts,
    cudaStream_t stream
) {
    k_moe_router_logits_f32<<<n_experts, 256, 0, stream>>>(
        d_weights, d_input, d_logits, dim, n_experts
    );
    k_moe_router_top8_f32<<<1, 128, 0, stream>>>(
        d_logits, d_ids, d_probabilities, n_experts
    );
}

void cuda_op_prepare_ffn(
    const float* d_hidden, const float* d_attn_proj,
    const float* d_post_attn_norm, const float* d_ffn_norm,
    const float* d_pre_ffw_norm_2, const float* d_router_scale,
    float* d_attn_res, float* d_shared_in, float* d_moe_in,
    float* d_router_in, int dim, cudaStream_t stream
) {
    k_prepare_ffn_f32<<<1, 256, 0, stream>>>(
        d_hidden, d_attn_proj, d_post_attn_norm, d_ffn_norm,
        d_pre_ffw_norm_2, d_router_scale, d_attn_res, d_shared_in,
        d_moe_in, d_router_in, dim
    );
}

void cuda_op_finish_ffn(
    const float* d_attn_res, float* d_dense, float* d_moe,
    const float* d_post_ffw_norm_1, const float* d_post_ffw_norm_2,
    const float* d_post_ffw_norm, float* d_hidden_out,
    float layer_scale, int dim, cudaStream_t stream
) {
    k_finish_ffn_f32<<<1, 256, 0, stream>>>(
        d_attn_res, d_dense, d_moe, d_post_ffw_norm_1,
        d_post_ffw_norm_2, d_post_ffw_norm, d_hidden_out,
        layer_scale, dim
    );
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
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (2 * blockDim.x / WARP_SIZE) + warp * 2 + (lane >> 4);
    int slot = blockIdx.y;      // slot in 0..n_active
    bool active = row < exp_ffn_dim && slot < n_active;

    int exp_idx = active ? active_exp_ids[slot] : 0;
    int n_blocks = dim / 32;
    int row_bytes = n_blocks * 18;
    int exp_bytes = (2 * exp_ffn_dim) * row_bytes;

    const uint8_t* exp_base = gate_up_exps + (size_t)exp_idx * exp_bytes;
    const uint8_t* gate_row_w = exp_base + (active ? row : 0) * row_bytes;
    const uint8_t* up_row_w = exp_base + (exp_ffn_dim + (active ? row : 0)) * row_bytes;

    float local_gate = 0.0f;
    float local_up = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
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

        uint8_t byte_g = active ? qs_g[sublane] : 0;
        uint8_t byte_u = active ? qs_u[sublane] : 0;
        float x0 = x_in[x_base + sublane];
        float x1 = x_in[x_base + sublane + 16];
        local_gate += active ? d_g * ((float)((byte_g & 0x0F) - 8) * x0 + (float)((byte_g >> 4) - 8) * x1) : 0.0f;
        local_up += active ? d_u * ((float)((byte_u & 0x0F) - 8) * x0 + (float)((byte_u >> 4) - 8) * x1) : 0.0f;
    }

    float total_gate = half_warp_reduce_sum(local_gate);
    float total_up = half_warp_reduce_sum(local_up);

    if (active && sublane == 0) {
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
    int lane = threadIdx.x & 31;
    int sublane = lane & 15;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (2 * blockDim.x / WARP_SIZE) + warp * 2 + (lane >> 4);
    int slot = blockIdx.y;      // slot in 0..n_active
    bool active = row < dim && slot < n_active;

    int exp_idx = active ? active_exp_ids[slot] : 0;
    float weight = active ? active_exp_weights[slot] : 0.0f;
    float scale = (down_exps_scale != nullptr) ? down_exps_scale[exp_idx] : 1.0f;
    float alpha = weight * scale;

    int n_blocks = exp_ffn_dim / 32;
    int row_bytes = n_blocks * 18;
    int exp_bytes = dim * row_bytes;

    const uint8_t* exp_base = down_exps + (size_t)exp_idx * exp_bytes;
    const uint8_t* row_w = exp_base + (active ? row : 0) * row_bytes;
    const float* act_slot = act_in + (active ? slot : 0) * exp_ffn_dim;

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        uint8_t byte = active ? qs[sublane] : 0;
        local_sum += active ? d * (
            (float)((byte & 0x0F) - 8) * act_slot[x_base + sublane]
            + (float)((byte >> 4) - 8) * act_slot[x_base + sublane + 16]
        ) : 0.0f;
    }

    float total_down = half_warp_reduce_sum(local_sum);

    if (active && sublane == 0) {
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

    const int threads = 128;
    const int rows_per_block = 2 * threads / WARP_SIZE;
    dim3 grid_gu((exp_ffn_dim + rows_per_block - 1) / rows_per_block, n_active);
    k_moe_gate_up_topk_q4_0_f32<<<grid_gu, threads, 0, stream>>>(
        d_gate_up_exps, d_active_exp_ids, d_x_in, d_act_scratch, exp_ffn_dim, dim, n_active
    );

    dim3 grid_down((dim + rows_per_block - 1) / rows_per_block, n_active);
    k_moe_down_topk_q4_0_f32<<<grid_down, threads, 0, stream>>>(
        d_down_exps, d_active_exp_ids, d_active_exp_weights, d_down_exps_scale,
        d_act_scratch, d_out_moe, dim, exp_ffn_dim, n_active
    );
}

}
