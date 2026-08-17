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
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    if (row >= n_rows) return;

    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* row_w = w_q4 + row * row_bytes;

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        int packed_lane = lane & 15;
        uint8_t byte = qs[packed_lane];
        int q = lane < 16 ? ((byte & 0x0F) - 8) : (((byte >> 4) & 0x0F) - 8);
        local_sum += (float)q * x[x_base + lane] * d;
    }

    float total_row_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
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
    int warp = threadIdx.x >> 5;
    int out_row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    int total_rows = q_rows + 2 * kv_rows;
    if (out_row >= total_rows) return;

    const uint8_t* matrix;
    int row;
    if (out_row < q_rows) {
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
    const uint8_t* row_w = matrix + (size_t)row * row_bytes;
    float local_sum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));
        const uint8_t* qs = row_w + w_off + 2;
        int packed_lane = lane & 15;
        uint8_t byte = qs[packed_lane];
        int q = lane < 16 ? ((byte & 0x0F) - 8) : (((byte >> 4) & 0x0F) - 8);
        local_sum += (float)q * x[b * 32 + lane] * d;
    }
    float total = warp_reduce_sum(local_sum);
    if (lane == 0) y[out_row] = total;
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
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    if (row >= n_rows) return;
    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 18;
    const uint8_t* gate_row = w_gate + (size_t)row * row_bytes;
    const uint8_t* up_row = w_up + (size_t)row * row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        int off = b * 18;
        uint16_t dg_raw = (uint16_t)gate_row[off] | ((uint16_t)gate_row[off + 1] << 8);
        uint16_t du_raw = (uint16_t)up_row[off] | ((uint16_t)up_row[off + 1] << 8);
        float dg = __half2float(__ushort_as_half(dg_raw));
        float du = __half2float(__ushort_as_half(du_raw));
        int packed_lane = lane & 15;
        uint8_t bg = gate_row[off + 2 + packed_lane];
        uint8_t bu = up_row[off + 2 + packed_lane];
        int qg = lane < 16 ? ((bg & 0x0F) - 8) : (((bg >> 4) & 0x0F) - 8);
        int qu = lane < 16 ? ((bu & 0x0F) - 8) : (((bu >> 4) & 0x0F) - 8);
        float xv = x[b * 32 + lane];
        gate_sum += dg * (float)qg * xv;
        up_sum += du * (float)qu * xv;
    }
    float gate = warp_reduce_sum(gate_sum);
    float up = warp_reduce_sum(up_sum);
    if (lane == 0) {
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
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    if (row >= n_rows) return;

    int n_blocks = n_cols / 32;
    int row_bytes = n_blocks * 34;
    const uint8_t* row_w = w_q8 + row * row_bytes;

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 34;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const int8_t* qs = (const int8_t*)(row_w + w_off + 2);
        int x_base = b * 32;

        local_sum += (float)qs[lane] * x[x_base + lane] * d;
    }

    float total_row_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
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
    int rows_per_block = threads / WARP_SIZE;
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
    int rows_per_block = threads / WARP_SIZE;
    int total_rows = q_rows + 2 * kv_rows;
    int blocks = (total_rows + rows_per_block - 1) / rows_per_block;
    k_gemv_q4_0_qkv_f32<<<blocks, threads, 0, stream>>>(
        d_w_q, d_w_k, d_w_v, d_x, d_y, q_rows, kv_rows, n_cols
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
    int rows_per_block = threads / WARP_SIZE;
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
    int rows_per_block = threads / WARP_SIZE;
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
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    int slot = blockIdx.y;      // slot in 0..n_active
    if (row >= exp_ffn_dim || slot >= n_active) return;

    int exp_idx = active_exp_ids[slot];
    int n_blocks = dim / 32;
    int row_bytes = n_blocks * 18;
    int exp_bytes = (2 * exp_ffn_dim) * row_bytes;

    const uint8_t* exp_base = gate_up_exps + (size_t)exp_idx * exp_bytes;
    const uint8_t* gate_row_w = exp_base + row * row_bytes;
    const uint8_t* up_row_w = exp_base + (exp_ffn_dim + row) * row_bytes;

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

        int packed_lane = lane & 15;
        uint8_t byte_g = qs_g[packed_lane];
        uint8_t byte_u = qs_u[packed_lane];
        int q_g = lane < 16 ? ((byte_g & 0x0F) - 8) : (((byte_g >> 4) & 0x0F) - 8);
        int q_u = lane < 16 ? ((byte_u & 0x0F) - 8) : (((byte_u >> 4) & 0x0F) - 8);
        float xv = x_in[x_base + lane];
        local_gate += d_g * (float)q_g * xv;
        local_up += d_u * (float)q_u * xv;
    }

    float total_gate = warp_reduce_sum(local_gate);
    float total_up = warp_reduce_sum(local_up);

    if (lane == 0) {
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
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
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

    float local_sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        int w_off = b * 18;
        uint16_t d_raw = (uint16_t)row_w[w_off] | ((uint16_t)row_w[w_off + 1] << 8);
        float d = __half2float(__ushort_as_half(d_raw));

        const uint8_t* qs = row_w + w_off + 2;
        int x_base = b * 32;

        int packed_lane = lane & 15;
        uint8_t byte = qs[packed_lane];
        int q = lane < 16 ? ((byte & 0x0F) - 8) : (((byte >> 4) & 0x0F) - 8);
        local_sum += d * (float)q * act_slot[x_base + lane];
    }

    float total_down = warp_reduce_sum(local_sum);

    if (lane == 0) {
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
    const int rows_per_block = threads / WARP_SIZE;
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

