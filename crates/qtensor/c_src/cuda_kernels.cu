#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

#define WARP_SIZE 32

// Block reduction for sum across a thread block
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
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

// 3. CUDA RoPE Kernel
__global__ void k_rope_f32(
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
// Each block computes one or more output rows
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
        
        // Convert fp16 to fp32 scale
        __half d_half;
        memcpy(&d_half, &d_raw, sizeof(__half));
        float d = __half2float(d_half);

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

// C-ABI Exported Host Functions
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

void cuda_op_rope(
    float* d_vec,
    int pos,
    int head_dim,
    int n_heads,
    float freq_base,
    float freq_scale,
    cudaStream_t stream
) {
    int threads = head_dim / 2;
    k_rope_f32<<<n_heads, threads, 0, stream>>>(d_vec, pos, head_dim, n_heads, freq_base, freq_scale);
}

void cuda_op_gemv_q4_0(
    const uint8_t* d_w_q4,
    const float* d_x,
    float* d_y,
    int n_rows,
    int n_cols,
    cudaStream_t stream
) {
    int threads = (n_cols / 32 < 256) ? (n_cols / 32) : 256;
    if (threads < 32) threads = 32;
    k_gemv_q4_0_f32<<<n_rows, threads, 0, stream>>>(d_w_q4, d_x, d_y, n_rows, n_cols);
}

}
