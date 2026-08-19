#![no_std]
#![feature(abi_ptx, asm_experimental_arch, stdarch_nvptx)]

use core::arch::{
    asm, global_asm,
    nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _block_idx_z, _grid_dim_x, _thread_idx_x},
};

global_asm!(".shared .align 4 .b8 rust_rms_scratch[64];");
global_asm!(".shared .align 4 .b8 rust_vocab_scratch[16384];");
global_asm!(".extern .shared .align 4 .b8 rust_dynamic_scratch[];");

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

#[inline(always)]
unsafe fn shfl_down(value: f32, delta: u32, clamp: u32) -> f32 {
    let output: f32;
    asm!("shfl.sync.down.b32 {output}, {value}, {delta}, {clamp}, 0xffffffff;",
        output = out(reg32) output, value = in(reg32) value,
        delta = in(reg32) delta, clamp = in(reg32) clamp);
    output
}
#[inline(always)]
unsafe fn shfl_index(value: f32, index: u32) -> f32 {
    let output;
    asm!("shfl.sync.idx.b32 {o}, {v}, {i}, 31, 0xffffffff;",o=out(reg32)output,v=in(reg32)value,i=in(reg32)index);
    output
}
#[inline(always)]
unsafe fn ptx_max(a: f32, b: f32) -> f32 {
    let output;
    asm!("max.f32 {o}, {a}, {b};",o=out(reg32)output,a=in(reg32)a,b=in(reg32)b);
    output
}
#[inline(always)]
unsafe fn warp_max(mut value: f32) -> f32 {
    value = ptx_max(value, shfl_down(value, 16, 31));
    value = ptx_max(value, shfl_down(value, 8, 31));
    value = ptx_max(value, shfl_down(value, 4, 31));
    value = ptx_max(value, shfl_down(value, 2, 31));
    value = ptx_max(value, shfl_down(value, 1, 31));
    shfl_index(value, 0)
}
#[inline(always)]
unsafe fn round_i32(value: f32) -> i32 {
    let output;
    asm!("cvt.rni.s32.f32 {o}, {v};",o=out(reg32)output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn atomic_add(address: *mut f32, value: f32) {
    let discarded: f32;
    asm!("atom.global.add.f32 {o}, [{a}], {v};",o=out(reg32)discarded,a=in(reg64)address as u64,v=in(reg32)value);
}

#[inline(always)]
unsafe fn half_warp_sum(mut value: f32) -> f32 {
    value += shfl_down(value, 8, 0x100f);
    value += shfl_down(value, 4, 0x100f);
    value += shfl_down(value, 2, 0x100f);
    value += shfl_down(value, 1, 0x100f);
    value
}

#[inline(always)]
unsafe fn warp_sum(mut value: f32) -> f32 {
    value += shfl_down(value, 16, 31);
    value += shfl_down(value, 8, 31);
    value += shfl_down(value, 4, 31);
    value += shfl_down(value, 2, 31);
    value += shfl_down(value, 1, 31);
    value
}
#[inline(always)]
unsafe fn ptx_ex2(value: f32) -> f32 {
    let output;
    asm!("ex2.approx.f32 {o}, {v};",o=out(reg32) output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn ptx_lg2(value: f32) -> f32 {
    let output;
    asm!("lg2.approx.f32 {o}, {v};",o=out(reg32) output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn ptx_rsqrt(value: f32) -> f32 {
    let output;
    asm!("rsqrt.approx.f32 {o}, {v};",o=out(reg32) output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn ptx_sin(value: f32) -> f32 {
    let output;
    asm!("sin.approx.f32 {o}, {v};",o=out(reg32) output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn ptx_cos(value: f32) -> f32 {
    let output;
    asm!("cos.approx.f32 {o}, {v};",o=out(reg32) output,v=in(reg32)value);
    output
}
#[inline(always)]
unsafe fn ptx_div(a: f32, b: f32) -> f32 {
    let output;
    asm!("div.approx.f32 {o}, {a}, {b};",o=out(reg32)output,a=in(reg32)a,b=in(reg32)b);
    output
}
#[inline(always)]
unsafe fn ptx_div_i32(a: i32, b: i32) -> i32 {
    let output;
    asm!("div.s32 {o}, {a}, {b};",o=out(reg32)output,a=in(reg32)a,b=in(reg32)b);
    output
}
#[inline(always)]
unsafe fn ptx_rem_i32(a: i32, b: i32) -> i32 {
    let output;
    asm!("rem.s32 {o}, {a}, {b};",o=out(reg32)output,a=in(reg32)a,b=in(reg32)b);
    output
}
#[inline(always)]
unsafe fn shared_store(index: u32, value: f32) {
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_rms_scratch; add.u32 {a}, {a}, {o}; st.shared.f32 [{a}], {v};",a=out(reg32)address,o=in(reg32)offset,v=in(reg32)value);
}
#[inline(always)]
unsafe fn shared_load(index: u32) -> f32 {
    let output;
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_rms_scratch; add.u32 {a}, {a}, {o}; ld.shared.f32 {v}, [{a}];",a=out(reg32)address,o=in(reg32)offset,v=out(reg32)output);
    output
}
// Keep the barrier out of line. If this is inlined after a divergent work loop,
// LLVM can clone `bar.sync` onto each loop exit. CUDA then has different warps
// waiting at different barrier instructions even though both use barrier ID 0.
#[inline(never)]
unsafe fn block_sync() {
    asm!("bar.sync 0;", options(nostack));
}
#[inline(always)]
unsafe fn vocab_store_f32(index: u32, value: f32) {
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_vocab_scratch; add.u32 {a}, {a}, {o}; st.shared.f32 [{a}], {v};",a=out(reg32)address,o=in(reg32)offset,v=in(reg32)value);
}
#[inline(always)]
unsafe fn vocab_load_f32(index: u32) -> f32 {
    let output;
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_vocab_scratch; add.u32 {a}, {a}, {o}; ld.shared.f32 {v}, [{a}];",a=out(reg32)address,o=in(reg32)offset,v=out(reg32)output);
    output
}
#[inline(always)]
unsafe fn vocab_store_i32(index: u32, value: i32) {
    let mut address: u32;
    let offset = 8192 + index * 4;
    asm!("mov.u32 {a}, rust_vocab_scratch; add.u32 {a}, {a}, {o}; st.shared.u32 [{a}], {v};",a=out(reg32)address,o=in(reg32)offset,v=in(reg32)value);
}
#[inline(always)]
unsafe fn vocab_load_i32(index: u32) -> i32 {
    let output;
    let mut address: u32;
    let offset = 8192 + index * 4;
    asm!("mov.u32 {a}, rust_vocab_scratch; add.u32 {a}, {a}, {o}; ld.shared.u32 {v}, [{a}];",a=out(reg32)address,o=in(reg32)offset,v=out(reg32)output);
    output
}
#[inline(always)]
unsafe fn block_sum(mut value: f32) -> f32 {
    let tid = _thread_idx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    value = warp_sum(value);
    if lane == 0 {
        shared_store(warp as u32, value)
    }
    block_sync();
    if warp == 0 {
        let partial = if lane < (_block_dim_x() as usize / 32) {
            shared_load(lane as u32)
        } else {
            0.0
        };
        let total = warp_sum(partial);
        if lane == 0 {
            shared_store(0, total)
        }
    }
    block_sync();
    shared_load(0)
}
#[inline(always)]
unsafe fn block_max(mut value: f32) -> f32 {
    let tid = _thread_idx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    value = warp_max(value);
    if lane == 0 {
        shared_store(warp as u32, value)
    }
    block_sync();
    if warp == 0 {
        let partial = if lane < (_block_dim_x() as usize / 32) {
            shared_load(lane as u32)
        } else {
            f32::NEG_INFINITY
        };
        let total = warp_max(partial);
        if lane == 0 {
            shared_store(0, total)
        }
    }
    block_sync();
    shared_load(0)
}
#[inline(always)]
unsafe fn dynamic_store(index: u32, value: f32) {
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_dynamic_scratch; add.u32 {a}, {a}, {o}; st.shared.f32 [{a}], {v};",a=out(reg32)address,o=in(reg32)offset,v=in(reg32)value);
}
#[inline(always)]
unsafe fn dynamic_load(index: u32) -> f32 {
    let output;
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_dynamic_scratch; add.u32 {a}, {a}, {o}; ld.shared.f32 {v}, [{a}];",a=out(reg32)address,o=in(reg32)offset,v=out(reg32)output);
    output
}
#[inline(always)]
unsafe fn attention_store(index: u32, value: f32) {
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_dynamic_scratch; add.u32 {a}, {a}, {o}; st.shared.f32 [{a}], {v};",a=out(reg32)address,o=in(reg32)offset,v=in(reg32)value);
}
#[inline(always)]
unsafe fn attention_load(index: u32) -> f32 {
    let output;
    let mut address: u32;
    let offset = index * 4;
    asm!("mov.u32 {a}, rust_dynamic_scratch; add.u32 {a}, {a}, {o}; ld.shared.f32 {v}, [{a}];",a=out(reg32)address,o=in(reg32)offset,v=out(reg32)output);
    output
}

#[inline(always)]
unsafe fn load_cache(
    cache: *const u16,
    mut token: i32,
    head: usize,
    index: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: i32,
    format: i32,
) -> f32 {
    if capacity > 0 {
        if capacity & (capacity - 1) == 0 {
            token &= capacity - 1
        } else if token >= capacity {
            token = ptx_rem_i32(token, capacity)
        }
    }
    if format == 0 {
        return f16_to_f32(
            *cache.add(token as usize * n_kv_heads * head_dim + head * head_dim + index),
        );
    }
    let bytes = cache as *const u8;
    let block_bytes = if format == 1 { 34 } else { 18 };
    let blocks = head_dim / 32;
    let block = bytes.add(
        (token as usize * n_kv_heads + head) * blocks * block_bytes + (index / 32) * block_bytes,
    );
    let scale = f16_to_f32(*(block as *const u16));
    let lane = index & 31;
    if format == 1 {
        scale * (*block.add(2 + lane) as i8 as f32)
    } else {
        let packed = *block.add(2 + (lane & 15));
        let q = if lane < 16 {
            (packed & 15) as i32 - 8
        } else {
            (packed >> 4) as i32 - 8
        };
        scale * q as f32
    }
}
#[inline(always)]
unsafe fn cache_token(token: i32, capacity: i32) -> usize {
    if capacity <= 0 {
        token as usize
    } else if capacity & (capacity - 1) == 0 {
        (token & (capacity - 1)) as usize
    } else {
        ptx_rem_i32(token, capacity) as usize
    }
}
#[inline(always)]
unsafe fn cache_token_pow2(token: i32, capacity: i32) -> usize {
    (token & (capacity - 1)) as usize
}
#[inline(always)]
unsafe fn fast_exp(value: f32) -> f32 {
    ptx_ex2(value * 1.4426950408889634)
}
#[inline(always)]
unsafe fn fast_tanh(value: f32) -> f32 {
    // Using exp(2*x) directly produces inf/inf for the large positive gate
    // activations seen in Gemma's GeGLU. This form has the same value but its
    // exponential is always in [0, 1], so tanh saturates instead of becoming
    // NaN.
    let magnitude = if value < 0.0 { -value } else { value };
    let e = fast_exp(-2.0 * magnitude);
    let saturated = ptx_div(1.0 - e, 1.0 + e);
    if value < 0.0 { -saturated } else { saturated }
}
#[inline(always)]
unsafe fn global_index() -> usize {
    (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize
}

#[inline(always)]
unsafe fn f16_to_f32(bits: u16) -> f32 {
    let output;
    asm!("cvt.f32.f16 {o}, {v};",o=out(reg32)output,v=in(reg16)bits);
    output
}

#[inline(always)]
unsafe fn f32_to_f16(value: f32) -> u16 {
    let output;
    asm!("cvt.rn.f16.f32 {o}, {v};",o=out(reg16)output,v=in(reg32)value);
    output
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemm_q4_0_f32(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    batch: i32,
) {
    const TOKEN_TILE: usize = 8;
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let rows_per_block = (_block_dim_x() >> 5) * 2;
    let row = (_block_idx_x() * rows_per_block + warp * 2 + (lane >> 4)) as usize;
    let token_start = _block_idx_y() as usize * TOKEN_TILE;
    let active = row < rows as usize;
    let blocks = cols as usize / 32;
    let row_weights = if active {
        weights.add(row * blocks * 18)
    } else {
        weights
    };
    let mut sums = [0.0f32; TOKEN_TILE];
    let mut block = 0usize;
    while block < blocks {
        let offset = block * 18;
        let scale = f16_to_f32(
            *row_weights.add(offset) as u16 | ((*row_weights.add(offset + 1) as u16) << 8),
        );
        let packed = if active {
            *row_weights.add(offset + 2 + sublane)
        } else {
            0
        };
        let q0 = ((packed & 15) as i32 - 8) as f32 * scale;
        let q1 = ((packed >> 4) as i32 - 8) as f32 * scale;
        let mut tile = 0usize;
        while tile < TOKEN_TILE {
            let token = token_start + tile;
            if active && token < batch as usize {
                let x = input.add(token * cols as usize + block * 32);
                sums[tile] += q0 * *x.add(sublane) + q1 * *x.add(sublane + 16);
            }
            tile += 1;
        }
        block += 1;
    }
    let mut tile = 0usize;
    while tile < TOKEN_TILE {
        let sum = half_warp_sum(sums[tile]);
        let token = token_start + tile;
        if active && token < batch as usize && sublane == 0 {
            *output.add(token * rows as usize + row) = sum;
        }
        tile += 1;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemm_q4_0_qkv_f32(
    wq: *const u8,
    wk: *const u8,
    wv: *const u8,
    input: *const f32,
    output: *mut f32,
    q_rows: i32,
    kv_rows: i32,
    cols: i32,
    batch: i32,
) {
    const TOKEN_TILE: usize = 8;
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let out_row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let total = q_rows as usize + 2 * kv_rows as usize;
    let active = out_row < total;
    let (matrix, row) = if out_row < q_rows as usize {
        (wq, out_row)
    } else if out_row < (q_rows + kv_rows) as usize {
        (wk, out_row - q_rows as usize)
    } else {
        (wv, out_row - (q_rows + kv_rows) as usize)
    };
    let blocks = cols as usize / 32;
    let row_weights = if active {
        matrix.add(row * blocks * 18)
    } else {
        wq
    };
    let token_start = _block_idx_y() as usize * TOKEN_TILE;
    let mut sums = [0.0f32; TOKEN_TILE];
    let mut block = 0usize;
    while block < blocks {
        let offset = block * 18;
        let scale = f16_to_f32(
            *row_weights.add(offset) as u16 | ((*row_weights.add(offset + 1) as u16) << 8),
        );
        let packed = if active {
            *row_weights.add(offset + 2 + sublane)
        } else {
            0
        };
        let q0 = ((packed & 15) as i32 - 8) as f32 * scale;
        let q1 = ((packed >> 4) as i32 - 8) as f32 * scale;
        let mut tile = 0;
        while tile < TOKEN_TILE {
            let token = token_start + tile;
            if active && token < batch as usize {
                let x = input.add(token * cols as usize + block * 32);
                sums[tile] += q0 * *x.add(sublane) + q1 * *x.add(sublane + 16)
            }
            tile += 1
        }
        block += 1
    }
    let mut tile = 0;
    while tile < TOKEN_TILE {
        let sum = half_warp_sum(sums[tile]);
        let token = token_start + tile;
        if active && token < batch as usize && sublane == 0 {
            *output.add(token * total + out_row) = sum
        }
        tile += 1
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemv_q4_0_qkv_f32(
    wq: *const u8, wk: *const u8, wv: *const u8, input: *const f32, output: *mut f32,
    q_rows: i32, kv_rows: i32, cols: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let rows_per_block = (_block_dim_x() >> 5) * 2;
    let out_row = (_block_idx_x() * rows_per_block + warp * 2 + (lane >> 4)) as usize;
    let total = q_rows as usize + 2 * kv_rows as usize;
    let active = out_row < total;
    let (matrix, row) = if out_row < q_rows as usize {
        (wq, out_row)
    } else if out_row < (q_rows + kv_rows) as usize {
        (wk, out_row - q_rows as usize)
    } else {
        (wv, out_row - (q_rows + kv_rows) as usize)
    };
    let blocks = cols as usize / 32;
    let row_weights = if active { matrix.add(row * blocks * 18) } else { wq };
    let mut sum = 0.0f32;
    let mut block = 0usize;
    while block < blocks {
        let offset = block * 18;
        let scale = f16_to_f32(*row_weights.add(offset) as u16 | ((*row_weights.add(offset + 1) as u16) << 8));
        let packed = if active { *row_weights.add(offset + 2 + sublane) } else { 0 };
        let q0 = ((packed & 15) as i32 - 8) as f32;
        let q1 = ((packed >> 4) as i32 - 8) as f32;
        sum += scale * (q0 * *input.add(block * 32 + sublane) + q1 * *input.add(block * 32 + sublane + 16));
        block += 1;
    }
    let total_sum = half_warp_sum(sum);
    if active && sublane == 0 { *output.add(out_row) = total_sum; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemv_q4_0_f32(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let rows_per_block = (_block_dim_x() >> 5) * 2;
    let row = (_block_idx_x() * rows_per_block + warp * 2 + (lane >> 4)) as usize;
    let active = row < rows as usize;
    let blocks = cols as usize / 32;
    let row_weights = if active {
        weights.add(row * blocks * 18)
    } else {
        weights
    };
    let mut sum = 0.0f32;
    let mut block = 0usize;
    while block < blocks {
        let offset = block * 18;
        let scale = f16_to_f32(
            *row_weights.add(offset) as u16 | ((*row_weights.add(offset + 1) as u16) << 8),
        );
        let packed = if active {
            *row_weights.add(offset + 2 + sublane)
        } else {
            0
        };
        let q0 = ((packed & 15) as i32 - 8) as f32;
        let q1 = ((packed >> 4) as i32 - 8) as f32;
        sum += scale
            * (q0 * *input.add(block * 32 + sublane) + q1 * *input.add(block * 32 + sublane + 16));
        block += 1;
    }
    let total = half_warp_sum(sum);
    if active && sublane == 0 {
        *output.add(row) = total;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemv_q8_0_f32(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let active = row < rows as usize;
    let blocks = cols as usize / 32;
    let row_weights = if active {
        weights.add(row * blocks * 34)
    } else {
        weights
    };
    let mut sum = 0.0;
    let mut block = 0;
    while block < blocks {
        let offset = block * 34;
        let scale = f16_to_f32(
            *row_weights.add(offset) as u16 | ((*row_weights.add(offset + 1) as u16) << 8),
        );
        let q0 = if active {
            *row_weights.add(offset + 2 + sublane) as i8 as f32
        } else {
            0.0
        };
        let q1 = if active {
            *row_weights.add(offset + 18 + sublane) as i8 as f32
        } else {
            0.0
        };
        sum += scale
            * (q0 * *input.add(block * 32 + sublane) + q1 * *input.add(block * 32 + sublane + 16));
        block += 1
    }
    let total = half_warp_sum(sum);
    if active && sublane == 0 {
        *output.add(row) = 30.0 * fast_tanh(total / 30.0)
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemm_q4_0_geglu_f32(
    wgate: *const u8,
    wup: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    batch: i32,
) {
    const TOKEN_TILE: usize = 8;
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let active = row < rows as usize;
    let blocks = cols as usize / 32;
    let gate_row = if active {
        wgate.add(row * blocks * 18)
    } else {
        wgate
    };
    let up_row = if active {
        wup.add(row * blocks * 18)
    } else {
        wup
    };
    let token_start = _block_idx_y() as usize * TOKEN_TILE;
    let mut gate_sums = [0.0f32; TOKEN_TILE];
    let mut up_sums = [0.0f32; TOKEN_TILE];
    let mut block = 0;
    while block < blocks {
        let off = block * 18;
        let dg = f16_to_f32(*gate_row.add(off) as u16 | ((*gate_row.add(off + 1) as u16) << 8));
        let du = f16_to_f32(*up_row.add(off) as u16 | ((*up_row.add(off + 1) as u16) << 8));
        let bg = if active {
            *gate_row.add(off + 2 + sublane)
        } else {
            0
        };
        let bu = if active {
            *up_row.add(off + 2 + sublane)
        } else {
            0
        };
        let g0 = ((bg & 15) as i32 - 8) as f32 * dg;
        let g1 = ((bg >> 4) as i32 - 8) as f32 * dg;
        let u0 = ((bu & 15) as i32 - 8) as f32 * du;
        let u1 = ((bu >> 4) as i32 - 8) as f32 * du;
        let mut tile = 0;
        while tile < TOKEN_TILE {
            let token = token_start + tile;
            if active && token < batch as usize {
                let x = input.add(token * cols as usize + block * 32);
                let x0 = *x.add(sublane);
                let x1 = *x.add(sublane + 16);
                gate_sums[tile] += g0 * x0 + g1 * x1;
                up_sums[tile] += u0 * x0 + u1 * x1
            }
            tile += 1
        }
        block += 1
    }
    let mut tile = 0;
    while tile < TOKEN_TILE {
        let gate = half_warp_sum(gate_sums[tile]);
        let up = half_warp_sum(up_sums[tile]);
        let token = token_start + tile;
        if active && token < batch as usize && sublane == 0 {
            let gelu = 0.5
                * gate
                * (1.0 + fast_tanh(0.7978845608 * gate * (1.0 + 0.044715 * gate * gate)));
            *output.add(token * rows as usize + row) = gelu * up
        }
        tile += 1
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_gemv_q4_0_geglu_f32(
    wgate: *const u8,
    wup: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sublane = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let active = row < rows as usize;
    let blocks = cols as usize / 32;
    let row_bytes = blocks * 18;
    let gate = wgate.add(if active { row * row_bytes } else { 0 });
    let up = wup.add(if active { row * row_bytes } else { 0 });
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;
    let mut block = 0usize;
    while block < blocks {
        let offset = block * 18;
        let gate_scale = f16_to_f32(
            *gate.add(offset) as u16 | ((*gate.add(offset + 1) as u16) << 8),
        );
        let up_scale = f16_to_f32(
            *up.add(offset) as u16 | ((*up.add(offset + 1) as u16) << 8),
        );
        let gate_packed = if active { *gate.add(offset + 2 + sublane) } else { 0 };
        let up_packed = if active { *up.add(offset + 2 + sublane) } else { 0 };
        let x0 = *input.add(block * 32 + sublane);
        let x1 = *input.add(block * 32 + sublane + 16);
        gate_sum += gate_scale
            * (((gate_packed & 15) as i32 - 8) as f32 * x0
                + ((gate_packed >> 4) as i32 - 8) as f32 * x1);
        up_sum += up_scale
            * (((up_packed & 15) as i32 - 8) as f32 * x0
                + ((up_packed >> 4) as i32 - 8) as f32 * x1);
        block += 1;
    }
    let gate_value = half_warp_sum(gate_sum);
    let up_value = half_warp_sum(up_sum);
    if active && sublane == 0 {
        let gelu = 0.5
            * gate_value
            * (1.0
                + fast_tanh(
                    0.7978845608
                        * gate_value
                        * (1.0 + 0.044715 * gate_value * gate_value),
                ));
        *output.add(row) = gelu * up_value;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_add_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    size: i32,
) {
    let mut index = (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize;
    let stride = (_block_dim_x() * _grid_dim_x()) as usize;
    while index < size as usize {
        *out.add(index) = *a.add(index) + *b.add(index);
        index += stride;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_embedding_f32(
    table: *const f32,
    out: *mut f32,
    token: i32,
    dim: i32,
) {
    let i = global_index();
    if i < dim as usize {
        *out.add(i) = *table.add(token as usize * dim as usize + i)
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_swiglu_f32(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    size: i32,
) {
    let i = global_index();
    if i < size as usize {
        let g = *gate.add(i);
        *out.add(i) = g / (1.0 + fast_exp(-g)) * *up.add(i)
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_geglu_f32(
    gate: *const f32,
    up: *const f32,
    out: *mut f32,
    size: i32,
) {
    let i = global_index();
    if i < size as usize {
        let x = *gate.add(i);
        let gelu = 0.5 * x * (1.0 + fast_tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)));
        *out.add(i) = gelu * *up.add(i)
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_rope_f32(
    vec: *mut f32,
    pos: i32,
    head_dim: i32,
    n_heads: i32,
    freq_base: f32,
    freq_scale: f32,
) {
    let head = _block_idx_x() as usize;
    let i = _thread_idx_x() as usize;
    let half = head_dim as usize / 2;
    if head < n_heads as usize && i < half {
        let exponent = ptx_div((2 * i) as f32, head_dim as f32);
        let theta = ptx_div(
            pos as f32 * freq_scale,
            ptx_ex2(exponent * ptx_lg2(freq_base)),
        );
        let c = ptx_cos(theta);
        let s = ptx_sin(theta);
        let base = head * head_dim as usize;
        let a = *vec.add(base + i);
        let b = *vec.add(base + i + half);
        *vec.add(base + i) = a * c - b * s;
        *vec.add(base + i + half) = a * s + b * c;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_rms_norm_f32(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    dim: i32,
    batch: i32,
    eps: f32,
) {
    let token = _block_idx_x() as usize;
    if token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    let offset = token * dim as usize;
    let mut sum = 0.0;
    let mut i = tid;
    while i < dim as usize {
        let v = *x.add(offset + i);
        sum += v * v;
        i += _block_dim_x() as usize
    }
    sum = warp_sum(sum);
    if lane == 0 {
        shared_store(warp as u32, sum)
    }
    block_sync();
    if warp == 0 {
        let partial = if lane < (_block_dim_x() as usize / 32) {
            shared_load(lane as u32)
        } else {
            0.0
        };
        let total = warp_sum(partial);
        if lane == 0 {
            shared_store(0, ptx_rsqrt(ptx_div(total, dim as f32) + eps))
        }
    }
    block_sync();
    let scale = shared_load(0);
    i = tid;
    while i < dim as usize {
        let w = if weight.is_null() {
            1.0
        } else {
            *weight.add(i)
        };
        *out.add(offset + i) = *x.add(offset + i) * scale * w;
        i += _block_dim_x() as usize
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_router_logits_f32(
    weights: *const f32,
    input: *const f32,
    logits: *mut f32,
    dim: i32,
    n_experts: i32,
    batch: i32,
) {
    let expert = _block_idx_x() as usize;
    let token = _block_idx_y() as usize;
    if expert >= n_experts as usize || token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    let mut sum = 0.0;
    let mut i = tid;
    while i < dim as usize {
        sum += *weights.add(expert * dim as usize + i) * *input.add(token * dim as usize + i);
        i += _block_dim_x() as usize
    }
    sum = warp_sum(sum);
    if lane == 0 {
        shared_store(warp as u32, sum)
    }
    block_sync();
    if warp == 0 {
        let partial = if lane < (_block_dim_x() as usize / 32) {
            shared_load(lane as u32)
        } else {
            0.0
        };
        let total = warp_sum(partial);
        if lane == 0 {
            *logits.add(token * n_experts as usize + expert) = total
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_router_top8_f32(
    logits: *const f32,
    ids: *mut i32,
    probabilities: *mut f32,
    n_experts: i32,
    batch: i32,
) {
    let token = _block_idx_x() as usize;
    if token >= batch as usize || _thread_idx_x() != 0 {
        return;
    }
    let row = logits.add(token * n_experts as usize);
    let mut selected = [-1i32; 8];
    let mut scores = [f32::NEG_INFINITY; 8];
    let mut rank = 0;
    while rank < 8 {
        let mut best = f32::NEG_INFINITY;
        let mut best_id = -1;
        let mut expert = 0;
        while expert < n_experts {
            let mut used = false;
            let mut prior = 0;
            while prior < rank {
                if selected[prior] == expert {
                    used = true
                }
                prior += 1
            }
            let score = *row.add(expert as usize);
            if !used && score > best {
                best = score;
                best_id = expert
            }
            expert += 1
        }
        selected[rank] = best_id;
        scores[rank] = best;
        *ids.add(token * 8 + rank) = best_id;
        rank += 1
    }
    let max = scores[0];
    let mut total = 0.0;
    rank = 0;
    while rank < 8 {
        let value = fast_exp(scores[rank] - max);
        scores[rank] = value;
        total += value;
        rank += 1
    }
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    rank = 0;
    while rank < 8 {
        *probabilities.add(token * 8 + rank) = scores[rank] * inv;
        rank += 1
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_prepare_ffn_f32(
    hidden: *const f32,
    attn: *const f32,
    post_attn_norm: *const f32,
    ffn_norm: *const f32,
    pre_ffw_norm_2: *const f32,
    router_scale: *const f32,
    attn_res: *mut f32,
    shared: *mut f32,
    moe: *mut f32,
    router: *mut f32,
    dim: i32,
    batch: i32,
) {
    let token = _block_idx_x() as usize;
    if token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let offset = token * dim as usize;
    let mut sum = 0.0;
    let mut i = tid;
    while i < dim as usize {
        let v = *attn.add(offset + i);
        sum += v * v;
        i += _block_dim_x() as usize
    }
    let inv_proj = ptx_rsqrt(ptx_div(block_sum(sum), dim as f32) + 1e-6);
    sum = 0.0;
    i = tid;
    while i < dim as usize {
        let v = *hidden.add(offset + i) + *attn.add(offset + i) * inv_proj * *post_attn_norm.add(i);
        *attn_res.add(offset + i) = v;
        sum += v * v;
        i += _block_dim_x() as usize
    }
    let inv_res = ptx_rsqrt(ptx_div(block_sum(sum), dim as f32) + 1e-6);
    let router_factor = inv_res * ptx_rsqrt(dim as f32);
    i = tid;
    while i < dim as usize {
        let v = *attn_res.add(offset + i);
        *shared.add(offset + i) = v * inv_res * *ffn_norm.add(i);
        *moe.add(offset + i) = v * inv_res * *pre_ffw_norm_2.add(i);
        *router.add(offset + i) = v * router_factor * *router_scale.add(i);
        i += _block_dim_x() as usize
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_finish_ffn_f32(
    attn_res: *const f32,
    dense: *mut f32,
    moe: *mut f32,
    post1: *const f32,
    post2: *const f32,
    post: *const f32,
    output: *mut f32,
    layer_scale: f32,
    dim: i32,
    batch: i32,
) {
    let token = _block_idx_x() as usize;
    if token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let offset = token * dim as usize;
    let mut dense_sq = 0.0;
    let mut moe_sq = 0.0;
    let mut i = tid;
    while i < dim as usize {
        let d = *dense.add(offset + i);
        let m = *moe.add(offset + i);
        dense_sq += d * d;
        moe_sq += m * m;
        i += _block_dim_x() as usize
    }
    let inv_dense = ptx_rsqrt(ptx_div(block_sum(dense_sq), dim as f32) + 1e-6);
    let inv_moe = ptx_rsqrt(ptx_div(block_sum(moe_sq), dim as f32) + 1e-6);
    let mut combined_sq = 0.0;
    i = tid;
    while i < dim as usize {
        let value = *dense.add(offset + i) * inv_dense * *post1.add(i)
            + *moe.add(offset + i) * inv_moe * *post2.add(i);
        *moe.add(offset + i) = value;
        combined_sq += value * value;
        i += _block_dim_x() as usize
    }
    let inv_combined = ptx_rsqrt(ptx_div(block_sum(combined_sq), dim as f32) + 1e-6);
    i = tid;
    while i < dim as usize {
        *output.add(offset + i) = (*attn_res.add(offset + i)
            + *moe.add(offset + i) * inv_combined * *post.add(i))
            * layer_scale;
        i += _block_dim_x() as usize
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_vocab_topk_f32(
    logits: *const f32,
    valid: *const u8,
    recent: *const i32,
    out_scores: *mut f32,
    out_ids: *mut i32,
    vocab_size: i32,
    n_recent: i32,
    generated_count: i32,
    k: i32,
    partitions: i32,
) {
    const CAPACITY: u32 = 2048;
    let partition = _block_idx_x();
    if partition >= partitions as u32 {
        return;
    }
    let tid = _thread_idx_x();
    let start = (vocab_size as i64 * partition as i64 / partitions as i64) as i32;
    let end = (vocab_size as i64 * (partition + 1) as i64 / partitions as i64) as i32;
    let count = end - start;
    let mut local = tid;
    while local < CAPACITY {
        let id = start + local as i32;
        let mut score = f32::NEG_INFINITY;
        let mut stored = -1;
        if local < count as u32
            && *valid.add(id as usize) != 0
            && !(generated_count < 4 && (id == 1 || id == 2 || id == 105 || id == 106))
        {
            score = *logits.add(id as usize);
            let mut r = 0;
            while r < n_recent {
                if *recent.add(r as usize) == id {
                    score -= 1.8
                }
                r += 1
            }
            stored = id
        }
        vocab_store_f32(local, score);
        vocab_store_i32(local, stored);
        local += _block_dim_x()
    }
    block_sync();
    let mut size = 2u32;
    while size <= CAPACITY {
        let mut stride = size >> 1;
        while stride > 0 {
            let mut index = tid;
            while index < CAPACITY {
                let other = index ^ stride;
                if other > index {
                    let a_score = vocab_load_f32(index);
                    let b_score = vocab_load_f32(other);
                    let a_id = vocab_load_i32(index);
                    let b_id = vocab_load_i32(other);
                    let ascending = index & size == 0;
                    let a_after_b = a_score > b_score
                        || (a_score == b_score && a_id >= 0 && (b_id < 0 || a_id < b_id));
                    if a_after_b == ascending {
                        vocab_store_f32(index, b_score);
                        vocab_store_f32(other, a_score);
                        vocab_store_i32(index, b_id);
                        vocab_store_i32(other, a_id)
                    }
                }
                index += _block_dim_x()
            }
            block_sync();
            stride >>= 1
        }
        size <<= 1
    }
    let mut rank = tid;
    while rank < k as u32 {
        let source = CAPACITY - 1 - rank;
        let output = partition * k as u32 + rank;
        *out_scores.add(output as usize) = vocab_load_f32(source);
        *out_ids.add(output as usize) = vocab_load_i32(source);
        rank += _block_dim_x()
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_vocab_topk_generic_f32(
    logits: *const f32,
    valid: *const u8,
    recent: *const i32,
    out_scores: *mut f32,
    out_ids: *mut i32,
    vocab_size: i32,
    n_recent: i32,
    generated_count: i32,
    k: i32,
    partitions: i32,
) {
    let partition = _block_idx_x() as i32;
    if partition >= partitions || _thread_idx_x() != 0 {
        return;
    }
    let start = (vocab_size as i64 * partition as i64 / partitions as i64) as i32;
    let end = (vocab_size as i64 * (partition + 1) as i64 / partitions as i64) as i32;
    let output = (partition * k) as usize;
    let mut rank = 0;
    while rank < k {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_id = -1;
        let mut id = start;
        while id < end {
            let mut selected = false;
            let mut previous = 0;
            while previous < rank {
                if *out_ids.add(output + previous as usize) == id {
                    selected = true;
                    break;
                }
                previous += 1;
            }
            if !selected
                && *valid.add(id as usize) != 0
                && !(generated_count < 4 && (id == 1 || id == 2 || id == 105 || id == 106))
            {
                let mut score = *logits.add(id as usize);
                let mut r = 0;
                while r < n_recent {
                    if *recent.add(r as usize) == id {
                        score -= 1.8;
                    }
                    r += 1;
                }
                if score > best_score || (score == best_score && (best_id < 0 || id < best_id)) {
                    best_score = score;
                    best_id = id;
                }
            }
            id += 1;
        }
        *out_scores.add(output + rank as usize) = best_score;
        *out_ids.add(output + rank as usize) = best_id;
        rank += 1;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_qkv_postprocess(
    qkv: *mut f32,
    q_norm: *const f32,
    k_norm: *const f32,
    k_cache: *mut u16,
    v_cache: *mut u16,
    start_pos: i32,
    cache_start: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    freq_base: f32,
    batch: i32,
    cache_capacity: i32,
    k_format: i32,
    v_format: i32,
) {
    let head = _block_idx_x() as usize;
    let token = _block_idx_y() as usize;
    if token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let q_dim = n_heads as usize * head_dim as usize;
    let kv_dim = n_kv_heads as usize * head_dim as usize;
    let total_dim = q_dim + 2 * kv_dim;
    let is_q = head < n_heads as usize;
    let is_k = head >= n_heads as usize && head < (n_heads + n_kv_heads) as usize;
    let local_head = if is_q { head } else { head - n_heads as usize };
    let base = qkv.add(token * total_dim);
    let src = if is_q {
        base.add(local_head * head_dim as usize)
    } else if is_k {
        base.add(q_dim + local_head * head_dim as usize)
    } else {
        base.add(q_dim + kv_dim + (head - (n_heads + n_kv_heads) as usize) * head_dim as usize)
    };
    let mut sum = 0.0;
    let mut i = tid;
    while i < head_dim as usize {
        let value = *src.add(i);
        sum += value * value;
        i += _block_dim_x() as usize
    }
    let norm = ptx_rsqrt(ptx_div(block_sum(sum), head_dim as f32) + 1e-6);
    let pos = start_pos + token as i32;
    if is_q || is_k {
        let weights = if is_q { q_norm } else { k_norm };
        i = tid;
        while i < head_dim as usize / 2 {
            let a = *src.add(i) * norm * *weights.add(i);
            let b = *src.add(i + head_dim as usize / 2)
                * norm
                * *weights.add(i + head_dim as usize / 2);
            let exponent = ptx_div((2 * i) as f32, head_dim as f32);
            let theta = ptx_div(pos as f32, ptx_ex2(exponent * ptx_lg2(freq_base)));
            let c = ptx_cos(theta);
            let s = ptx_sin(theta);
            *src.add(i) = a * c - b * s;
            *src.add(i + head_dim as usize / 2) = a * s + b * c;
            i += _block_dim_x() as usize
        }
    } else {
        i = tid;
        while i < head_dim as usize {
            *src.add(i) *= norm;
            i += _block_dim_x() as usize
        }
    }
    block_sync();
    if !is_q {
        let cache_head = if is_k {
            local_head
        } else {
            head - (n_heads + n_kv_heads) as usize
        };
        let mut cache_pos = cache_start + token as i32;
        if cache_capacity > 0 {
            if cache_capacity & (cache_capacity - 1) == 0 {
                cache_pos &= cache_capacity - 1
            } else if cache_pos >= cache_capacity {
                cache_pos %= cache_capacity
            }
        }
        let cache = if is_k { k_cache } else { v_cache };
        let format = if is_k { k_format } else { v_format };
        if format == 0 {
            let dst = cache.add(cache_pos as usize * kv_dim + cache_head * head_dim as usize);
            i = tid;
            while i < head_dim as usize {
                *dst.add(i) = f32_to_f16(*src.add(i));
                i += _block_dim_x() as usize
            }
        } else {
            let lane = tid & 31;
            let warp = tid >> 5;
            let warps = _block_dim_x() as usize / 32;
            let blocks = head_dim as usize / 32;
            let block_bytes = if format == 1 { 34 } else { 18 };
            let cache_bytes = cache as *mut u8;
            let head_base = cache_bytes.add(
                (cache_pos as usize * n_kv_heads as usize + cache_head) * blocks * block_bytes,
            );
            let mut block = warp;
            while block < blocks {
                let value = *src.add(block * 32 + lane);
                let max_abs = warp_max(value.abs());
                let mut scale = ptx_div(max_abs, if format == 1 { 127.0 } else { 7.0 });
                if scale == 0.0 {
                    scale = 1.0
                }
                let dst = head_base.add(block * block_bytes);
                if lane == 0 {
                    *(dst as *mut u16) = f32_to_f16(scale)
                }
                let lower = if format == 1 { -127 } else { -8 };
                let mut quant = round_i32(ptx_div(value, scale));
                if quant < lower {
                    quant = lower
                }
                let upper_limit = if format == 1 { 127 } else { 7 };
                if quant > upper_limit {
                    quant = upper_limit
                }
                if format == 1 {
                    *dst.add(2 + lane) = quant as i8 as u8
                } else if lane < 16 {
                    let upper = *src.add(block * 32 + lane + 16);
                    let mut q1 = round_i32(ptx_div(upper, scale));
                    if q1 < -8 {
                        q1 = -8
                    }
                    if q1 > 7 {
                        q1 = 7
                    }
                    *dst.add(2 + lane) = ((quant + 8) | ((q1 + 8) << 4)) as u8
                }
                block += warps
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_attention(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    out: *mut f32,
    cache_start: i32,
    batch: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    q_stride: i32,
    scale: f32,
    sliding_window: i32,
    cache_capacity: i32,
    _k_format: i32,
    _v_format: i32,
) {
    let head = _block_idx_x() as usize;
    let token = _block_idx_y() as usize;
    if head >= n_heads as usize || token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let n_past = cache_start + token as i32;
    let start = if sliding_window > 0 && n_past >= sliding_window {
        n_past - sliding_window + 1
    } else {
        0
    };
    let count = n_past - start + 1;
    let kv_head = ptx_div_i32(head as i32, ptx_div_i32(n_heads, n_kv_heads)) as usize;
    let qh = q.add(token * q_stride as usize + head * head_dim as usize);
    let mut p = tid as i32;
    while p < count {
        let mut dot = 0.0;
        let mut d = 0;
        let cache_row = k_cache.add(
            (cache_token_pow2(start + p, cache_capacity) * n_kv_heads as usize + kv_head)
                * head_dim as usize,
        );
        while d + 1 < head_dim as usize {
            let packed = *(cache_row.add(d) as *const u32);
            dot += *qh.add(d) * f16_to_f32(packed as u16)
                + *qh.add(d + 1) * f16_to_f32((packed >> 16) as u16);
            d += 2
        }
        attention_store(p as u32, dot * scale);
        p += _block_dim_x() as i32
    }
    block_sync();
    let mut local_max = f32::NEG_INFINITY;
    p = tid as i32;
    while p < count {
        local_max = ptx_max(local_max, attention_load(p as u32));
        p += _block_dim_x() as i32
    }
    let maximum = block_max(local_max);
    let mut local_sum = 0.0;
    p = tid as i32;
    while p < count {
        let value = fast_exp(attention_load(p as u32) - maximum);
        attention_store(p as u32, value);
        local_sum += value;
        p += _block_dim_x() as i32
    }
    let inv = ptx_div(1.0, block_sum(local_sum));
    let target = out.add(token * n_heads as usize * head_dim as usize + head * head_dim as usize);
    let mut d = tid * 2;
    while d + 1 < head_dim as usize {
        let mut value0 = 0.0;
        let mut value1 = 0.0;
        p = 0;
        while p < count {
            let cache_row = v_cache.add(
                (cache_token_pow2(start + p, cache_capacity) * n_kv_heads as usize + kv_head)
                    * head_dim as usize,
            );
            let probability = attention_load(p as u32) * inv;
            value0 += probability * f16_to_f32(*cache_row.add(d));
            value1 += probability * f16_to_f32(*cache_row.add(d + 1));
            p += 1
        }
        *target.add(d) = value0;
        *target.add(d + 1) = value1;
        d += _block_dim_x() as usize * 2
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_attention_streaming(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    out: *mut f32,
    cache_start: i32,
    batch: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    q_stride: i32,
    scale: f32,
    sliding_window: i32,
    cache_capacity: i32,
    k_format: i32,
    v_format: i32,
) {
    let head = _block_idx_x() as usize;
    let token = _block_idx_y() as usize;
    if head >= n_heads as usize || token >= batch as usize {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let n_past = cache_start + token as i32;
    let start = if sliding_window > 0 && n_past >= sliding_window {
        n_past - sliding_window + 1
    } else {
        0
    };
    let count = n_past - start + 1;
    let kv_head = ptx_div_i32(head as i32, ptx_div_i32(n_heads, n_kv_heads)) as usize;
    let qh = q.add(token * q_stride as usize + head * head_dim as usize);
    let target = out.add(token * n_heads as usize * head_dim as usize + head * head_dim as usize);
    let mut d = tid;
    while d < head_dim as usize {
        *target.add(d) = 0.0;
        d += _block_dim_x() as usize;
    }
    block_sync();
    let mut maximum = f32::NEG_INFINITY;
    let mut normalizer = 0.0;
    let mut p = 0;
    while p < count {
        let mut partial = 0.0;
        d = tid;
        while d < head_dim as usize {
            partial += *qh.add(d)
                * load_cache(
                    k_cache,
                    start + p,
                    kv_head,
                    d,
                    n_kv_heads as usize,
                    head_dim as usize,
                    cache_capacity,
                    k_format,
                );
            d += _block_dim_x() as usize;
        }
        let score = block_sum(partial) * scale;
        if tid == 0 {
            let next_maximum = ptx_max(maximum, score);
            let rescale = fast_exp(maximum - next_maximum);
            let probability = fast_exp(score - next_maximum);
            shared_store(0, next_maximum);
            shared_store(1, rescale);
            shared_store(2, probability);
            shared_store(3, normalizer * rescale + probability);
        }
        block_sync();
        maximum = shared_load(0);
        let rescale = shared_load(1);
        let probability = shared_load(2);
        normalizer = shared_load(3);
        d = tid;
        while d < head_dim as usize {
            let value = load_cache(
                v_cache,
                start + p,
                kv_head,
                d,
                n_kv_heads as usize,
                head_dim as usize,
                cache_capacity,
                v_format,
            );
            *target.add(d) = *target.add(d) * rescale + value * probability;
            d += _block_dim_x() as usize;
        }
        block_sync();
        p += 1;
    }
    d = tid;
    while d < head_dim as usize {
        *target.add(d) = ptx_div(*target.add(d), normalizer);
        d += _block_dim_x() as usize;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_gate_up_q4(
    gate_up: *const u8,
    ids: *const i32,
    input: *const f32,
    act: *mut f32,
    exp_dim: i32,
    dim: i32,
    n_active: i32,
    batch: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sub = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let slot = _block_idx_y() as usize;
    let token = _block_idx_z() as usize;
    let active = row < exp_dim as usize && slot < n_active as usize && token < batch as usize;
    let expert = if active {
        *ids.add(token * n_active as usize + slot) as usize
    } else {
        0
    };
    let blocks = dim as usize / 32;
    let row_bytes = blocks * 18;
    let base = gate_up.add(expert * 2 * exp_dim as usize * row_bytes);
    let gate = base.add(if active { row * row_bytes } else { 0 });
    let up = base.add((exp_dim as usize + if active { row } else { 0 }) * row_bytes);
    let x = input.add(token * dim as usize);
    let mut gs = 0.0;
    let mut us = 0.0;
    let mut block = 0;
    while block < blocks {
        let off = block * 18;
        let gd = f16_to_f32(*gate.add(off) as u16 | ((*gate.add(off + 1) as u16) << 8));
        let ud = f16_to_f32(*up.add(off) as u16 | ((*up.add(off + 1) as u16) << 8));
        let g = if active { *gate.add(off + 2 + sub) } else { 0 };
        let u = if active { *up.add(off + 2 + sub) } else { 0 };
        let x0 = *x.add(block * 32 + sub);
        let x1 = *x.add(block * 32 + sub + 16);
        gs += gd * (((g & 15) as i32 - 8) as f32 * x0 + ((g >> 4) as i32 - 8) as f32 * x1);
        us += ud * (((u & 15) as i32 - 8) as f32 * x0 + ((u >> 4) as i32 - 8) as f32 * x1);
        block += 1
    }
    let gv = half_warp_sum(gs);
    let uv = half_warp_sum(us);
    if active && sub == 0 {
        let gelu = 0.5 * gv * (1.0 + fast_tanh(0.7978845608 * gv * (1.0 + 0.044715 * gv * gv)));
        *act.add((token * n_active as usize + slot) * exp_dim as usize + row) = gelu * uv
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_gate_up_q4_gemma4_26b(
    gate_up: *const u8,
    ids: *const i32,
    input: *const f32,
    act: *mut f32,
) {
    const DIM: usize = 2_816;
    const EXP_DIM: usize = 704;
    const ACTIVE: usize = 8;
    const BLOCKS: usize = DIM / 32;
    const ROW_BYTES: usize = BLOCKS * 18;
    let lane = _thread_idx_x() & 31;
    let sub = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let slot = _block_idx_y() as usize;
    let active = row < EXP_DIM && slot < ACTIVE;
    let expert = if active { *ids.add(slot) as usize } else { 0 };
    let base = gate_up.add(expert * 2 * EXP_DIM * ROW_BYTES);
    let gate = base.add(if active { row * ROW_BYTES } else { 0 });
    let up = base.add((EXP_DIM + if active { row } else { 0 }) * ROW_BYTES);
    let mut gs = 0.0f32;
    let mut us = 0.0f32;
    let mut block = 0usize;
    while block < BLOCKS {
        let off = block * 18;
        let gd = f16_to_f32(*gate.add(off) as u16 | ((*gate.add(off + 1) as u16) << 8));
        let ud = f16_to_f32(*up.add(off) as u16 | ((*up.add(off + 1) as u16) << 8));
        let g = if active { *gate.add(off + 2 + sub) } else { 0 };
        let u = if active { *up.add(off + 2 + sub) } else { 0 };
        let x0 = *input.add(block * 32 + sub);
        let x1 = *input.add(block * 32 + sub + 16);
        gs += gd * (((g & 15) as i32 - 8) as f32 * x0 + ((g >> 4) as i32 - 8) as f32 * x1);
        us += ud * (((u & 15) as i32 - 8) as f32 * x0 + ((u >> 4) as i32 - 8) as f32 * x1);
        block += 1;
    }
    let gv = half_warp_sum(gs);
    let uv = half_warp_sum(us);
    if active && sub == 0 {
        let gelu = 0.5 * gv * (1.0 + fast_tanh(0.7978845608 * gv * (1.0 + 0.044715 * gv * gv)));
        *act.add(slot * EXP_DIM + row) = gelu * uv;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_down_q4(
    down: *const u8,
    ids: *const i32,
    weights: *const f32,
    scales: *const f32,
    act: *const f32,
    out: *mut f32,
    dim: i32,
    exp_dim: i32,
    n_active: i32,
    batch: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sub = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let slot = _block_idx_y() as usize;
    let token = _block_idx_z() as usize;
    let active = row < dim as usize && slot < n_active as usize && token < batch as usize;
    let expert = if active {
        *ids.add(token * n_active as usize + slot) as usize
    } else {
        0
    };
    let mut alpha = if active {
        *weights.add(token * n_active as usize + slot)
    } else {
        0.0
    };
    if !scales.is_null() {
        alpha *= *scales.add(expert)
    }
    let blocks = exp_dim as usize / 32;
    let row_bytes = blocks * 18;
    let row_w =
        down.add(expert * dim as usize * row_bytes + if active { row * row_bytes } else { 0 });
    let input = act.add((token * n_active as usize + slot) * exp_dim as usize);
    let mut sum = 0.0;
    let mut block = 0;
    while block < blocks {
        let off = block * 18;
        let d = f16_to_f32(*row_w.add(off) as u16 | ((*row_w.add(off + 1) as u16) << 8));
        let q = if active { *row_w.add(off + 2 + sub) } else { 0 };
        sum += d
            * (((q & 15) as i32 - 8) as f32 * *input.add(block * 32 + sub)
                + ((q >> 4) as i32 - 8) as f32 * *input.add(block * 32 + sub + 16));
        block += 1
    }
    let value = half_warp_sum(sum);
    if active && sub == 0 {
        atomic_add(out.add(token * dim as usize + row), value * alpha)
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_down_q4_combined(
    down: *const u8,
    ids: *const i32,
    weights: *const f32,
    scales: *const f32,
    act: *const f32,
    out: *mut f32,
    dim: i32,
    exp_dim: i32,
    n_active: i32,
    batch: i32,
) {
    let lane = _thread_idx_x() & 31;
    let sub = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let token = _block_idx_z() as usize;
    let active_row = row < dim as usize && token < batch as usize;
    let blocks = exp_dim as usize / 32;
    let row_bytes = blocks * 18;
    let mut combined = 0.0f32;
    let mut slot = 0usize;
    while slot < n_active as usize {
        let expert = if active_row { *ids.add(token * n_active as usize + slot) as usize } else { 0 };
        let mut alpha = if active_row { *weights.add(token * n_active as usize + slot) } else { 0.0 };
        if !scales.is_null() {
            alpha *= *scales.add(expert);
        }
        let row_w = down.add(expert * dim as usize * row_bytes + if active_row { row * row_bytes } else { 0 });
        let input = act.add((token * n_active as usize + slot) * exp_dim as usize);
        let mut sum = 0.0f32;
        let mut block = 0usize;
        while block < blocks {
            let off = block * 18;
            let d = f16_to_f32(*row_w.add(off) as u16 | ((*row_w.add(off + 1) as u16) << 8));
            let q = if active_row { *row_w.add(off + 2 + sub) } else { 0 };
            sum += d
                * (((q & 15) as i32 - 8) as f32 * *input.add(block * 32 + sub)
                    + ((q >> 4) as i32 - 8) as f32 * *input.add(block * 32 + sub + 16));
            block += 1;
        }
        let value = half_warp_sum(sum);
        if sub == 0 {
            combined += value * alpha;
        }
        slot += 1;
    }
    if active_row && sub == 0 {
        *out.add(token * dim as usize + row) = combined;
    }
}

// Gemma 4 26B decode uses this exact top-8 expert shape. Keeping the dimensions
// out of kernel parameters lets LLVM fold the row strides and loop bounds that
// otherwise sit in the hottest projection of every generated token.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rust_cuda_moe_down_q4_gemma4_26b(
    down: *const u8,
    ids: *const i32,
    weights: *const f32,
    scales: *const f32,
    act: *const f32,
    out: *mut f32,
) {
    const DIM: usize = 2_816;
    const EXP_DIM: usize = 704;
    const ACTIVE: usize = 8;
    const BLOCKS: usize = EXP_DIM / 32;
    const ROW_BYTES: usize = BLOCKS * 18;

    let lane = _thread_idx_x() & 31;
    let sub = (lane & 15) as usize;
    let warp = _thread_idx_x() >> 5;
    let row = (_block_idx_x() * ((_block_dim_x() >> 5) * 2) + warp * 2 + (lane >> 4)) as usize;
    let active_row = row < DIM;
    let mut combined = 0.0f32;
    let mut slot = 0usize;
    while slot < ACTIVE {
        let expert = if active_row { *ids.add(slot) as usize } else { 0 };
        let mut alpha = if active_row { *weights.add(slot) } else { 0.0 };
        if !scales.is_null() {
            alpha *= *scales.add(expert);
        }
        let row_w = down.add(expert * DIM * ROW_BYTES + if active_row { row * ROW_BYTES } else { 0 });
        let input = act.add(slot * EXP_DIM);
        let mut sum = 0.0f32;
        let mut block = 0usize;
        while block < BLOCKS {
            let off = block * 18;
            let d = f16_to_f32(*row_w.add(off) as u16 | ((*row_w.add(off + 1) as u16) << 8));
            let q = if active_row { *row_w.add(off + 2 + sub) } else { 0 };
            sum += d
                * (((q & 15) as i32 - 8) as f32 * *input.add(block * 32 + sub)
                    + ((q >> 4) as i32 - 8) as f32 * *input.add(block * 32 + sub + 16));
            block += 1;
        }
        let value = half_warp_sum(sum);
        if sub == 0 {
            combined += value * alpha;
        }
        slot += 1;
    }
    if active_row && sub == 0 {
        *out.add(row) = combined;
    }
}
