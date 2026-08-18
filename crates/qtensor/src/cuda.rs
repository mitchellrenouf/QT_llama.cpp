use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type CudaStream = *mut c_void;
type CudaGraphHandle = *mut c_void;
type CudaGraphExecHandle = *mut c_void;

#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

#[allow(dead_code)]
extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaRuntimeGetVersion(version: *mut i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind, stream: CudaStream) -> i32;
    fn cudaStreamCreate(stream: *mut CudaStream) -> i32;
    fn cudaStreamDestroy(stream: CudaStream) -> i32;
    fn cudaStreamSynchronize(stream: CudaStream) -> i32;
    fn cudaStreamBeginCapture(stream: CudaStream, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraphHandle) -> i32;
    fn cudaGraphInstantiate(exec: *mut CudaGraphExecHandle, graph: CudaGraphHandle,
        error_node: *mut *mut c_void, log_buffer: *mut i8, buffer_size: usize) -> i32;
    fn cudaGraphLaunch(exec: CudaGraphExecHandle, stream: CudaStream) -> i32;
    fn cudaGraphDestroy(graph: CudaGraphHandle) -> i32;
    fn cudaGraphExecDestroy(exec: CudaGraphExecHandle) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cuda_pool_alloc(pointer: *mut *mut c_void, bytes: usize) -> i32;
    fn cuda_pool_release(pointer: *mut c_void, bytes: usize);

    // Exported custom kernel launches from cuda_kernels.cu
    fn cuda_op_rms_norm(d_x: *const f32, d_w: *const f32, d_out: *mut f32, dim: i32, eps: f32, stream: CudaStream);
    fn cuda_op_rms_norm_batch(d_x: *const f32, d_w: *const f32, d_out: *mut f32, dim: i32, batch: i32, eps: f32, stream: CudaStream);
    fn cuda_op_swiglu(d_gate: *const f32, d_up: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_geglu(d_gate: *const f32, d_up: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_rope_256k(d_vec: *mut f32, pos: i32, head_dim: i32, n_heads: i32, freq_base: f32, freq_scale: f32, stream: CudaStream);
    fn cuda_op_gemv_q4_0(d_w_q4: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream);
    fn cuda_op_gemm_q4_0(d_w_q4: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, batch: i32, stream: CudaStream);
    fn cuda_op_gemv_q4_0_qkv(
        d_w_q: *const u8, d_w_k: *const u8, d_w_v: *const u8,
        d_x: *const f32, d_y: *mut f32, q_rows: i32, kv_rows: i32,
        n_cols: i32, stream: CudaStream,
    );
    fn cuda_op_gemm_q4_0_qkv(
        d_w_q: *const u8, d_w_k: *const u8, d_w_v: *const u8,
        d_x: *const f32, d_y: *mut f32, q_rows: i32, kv_rows: i32,
        n_cols: i32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_qkv_postprocess(
        d_qkv: *mut f32, d_q_norm: *const f32, d_k_norm: *const f32,
        d_k_cache: *mut u16, d_v_cache: *mut u16, pos: i32, cache_pos: i32,
        n_heads: i32, n_kv_heads: i32, head_dim: i32, freq_base: f32,
        stream: CudaStream,
    );
    fn cuda_op_qkv_postprocess_batch(
        d_qkv: *mut f32, d_q_norm: *const f32, d_k_norm: *const f32,
        d_k_cache: *mut u16, d_v_cache: *mut u16, start_pos: i32,
        cache_start: i32, n_heads: i32, n_kv_heads: i32, head_dim: i32,
        freq_base: f32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_gemv_q4_0_geglu(
        d_w_gate: *const u8, d_w_up: *const u8, d_x: *const f32,
        d_act: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream,
    );
    fn cuda_op_gemv_q8_0(d_w_q8: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream);
    fn cuda_op_vocab_topk(
        d_logits: *const f32, d_valid: *const u8, d_recent: *const i32,
        d_scores: *mut f32, d_ids: *mut i32, vocab_size: i32,
        n_recent: i32, generated_count: i32, k: i32, partitions: i32,
        stream: CudaStream,
    );
    fn cuda_op_add(d_a: *const f32, d_b: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_embedding(d_table: *const f32, d_out: *mut f32, token: i32, dim: i32, stream: CudaStream);
    fn cuda_op_moe_router(
        d_weights: *const f32, d_input: *const f32, d_logits: *mut f32,
        d_ids: *mut i32, d_probabilities: *mut f32, dim: i32,
        n_experts: i32, stream: CudaStream,
    );
    fn cuda_op_prepare_ffn(
        d_hidden: *const f32, d_attn_proj: *const f32,
        d_post_attn_norm: *const f32, d_ffn_norm: *const f32,
        d_pre_ffw_norm_2: *const f32, d_router_scale: *const f32,
        d_attn_res: *mut f32, d_shared_in: *mut f32, d_moe_in: *mut f32,
        d_router_in: *mut f32, dim: i32, stream: CudaStream,
    );
    fn cuda_op_prepare_ffn_batch(
        h: *const f32, a: *const f32, pan: *const f32, ffn: *const f32,
        pfn: *const f32, rs: *const f32, ar: *mut f32, si: *mut f32,
        mi: *mut f32, ri: *mut f32, dim: i32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_finish_ffn(
        d_attn_res: *const f32, d_dense: *mut f32, d_moe: *mut f32,
        d_post_ffw_norm_1: *const f32, d_post_ffw_norm_2: *const f32,
        d_post_ffw_norm: *const f32, d_hidden_out: *mut f32,
        layer_scale: f32, dim: i32, stream: CudaStream,
    );
    fn cuda_op_finish_ffn_batch(
        ar: *const f32, dense: *mut f32, moe: *mut f32, p1: *const f32,
        p2: *const f32, pf: *const f32, out: *mut f32, scale: f32,
        dim: i32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_attention(
        d_q: *const f32,
        d_k_cache: *const u16,
        d_v_cache: *const u16,
        d_out: *mut f32,
        n_past: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        sliding_window: i32,
        stream: CudaStream,
    );
    fn cuda_op_moe_router_batch(
        d_weights: *const f32, d_input: *const f32, d_logits: *mut f32,
        d_ids: *mut i32, d_probabilities: *mut f32, dim: i32,
        n_experts: i32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_gemm_q4_0_geglu(
        d_w_gate: *const u8, d_w_up: *const u8, d_x: *const f32,
        d_act: *mut f32, n_rows: i32, n_cols: i32, batch: i32,
        stream: CudaStream,
    );
    fn cuda_op_attention_prefill(
        d_q: *const f32, d_k_cache: *const u16, d_v_cache: *const u16,
        d_out: *mut f32, cache_start: i32, batch: i32, n_heads: i32,
        n_kv_heads: i32, head_dim: i32, q_stride: i32, scale: f32, sliding_window: i32,
        stream: CudaStream,
    );
    fn cuda_op_moe_topk_q4_0(
        d_gate_up_exps: *const u8,
        d_down_exps: *const u8,
        d_active_exp_ids: *const i32,
        d_active_exp_weights: *const f32,
        d_down_exps_scale: *const f32,
        d_x_in: *const f32,
        d_act_scratch: *mut f32,
        d_out_moe: *mut f32,
        dim: i32,
        exp_ffn_dim: i32,
        n_active: i32,
        stream: CudaStream,
    );
    fn cuda_op_moe_topk_batch_q4_0(
        d_gate_up_exps: *const u8, d_down_exps: *const u8,
        d_active_exp_ids: *const i32, d_active_exp_weights: *const f32,
        d_down_exps_scale: *const f32, d_x_in: *const f32,
        d_act_scratch: *mut f32, d_out_moe: *mut f32, dim: i32,
        exp_ffn_dim: i32, n_active: i32, batch: i32, stream: CudaStream,
    );
    fn cuda_op_ffn_compute_launches(
        gate: *const u8, up: *const u8, down: *const u8,
        shared_in: *const f32, dense_act: *mut f32, dense_out: *mut f32,
        router_weights: *const f32, router_in: *const f32, router_logits: *mut f32,
        expert_ids: *mut i32, expert_weights: *mut f32,
        gate_up_exps: *const u8, down_exps: *const u8, down_scales: *const f32,
        moe_in: *const f32, moe_act: *mut f32, moe_out: *mut f32,
        dim: i32, ffn_dim: i32, exp_dim: i32, n_experts: i32, n_active: i32,
        stream: CudaStream,
    );
}

/// RAII wrapper for GPU Device Memory
pub struct CudaBuffer<T> {
    ptr: *mut T,
    len: usize,
    device_id: i32,
    allocation: CudaAllocation,
    _marker: PhantomData<T>,
}

enum CudaAllocation {
    Owned,
    Pooled,
    Arena(Arc<CudaArenaAllocation>),
}

struct CudaArenaAllocation {
    ptr: *mut c_void,
    bytes: usize,
    device_id: i32,
}

unsafe impl Send for CudaArenaAllocation {}
unsafe impl Sync for CudaArenaAllocation {}

impl Drop for CudaArenaAllocation {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaSetDevice(self.device_id);
                cudaFree(self.ptr);
            }
        }
    }
}

#[derive(Clone)]
pub struct CudaArena {
    allocation: Arc<CudaArenaAllocation>,
    next: Arc<AtomicUsize>,
}

impl CudaArena {
    pub fn new(device_id: i32, bytes: usize) -> Result<Self> {
        unsafe { cudaSetDevice(device_id) };
        let mut ptr = ptr::null_mut();
        let status = unsafe { cudaMalloc(&mut ptr, bytes) };
        if status != 0 || ptr.is_null() {
            return Err(anyhow!("CUDA arena allocation of {bytes} bytes failed with code {status}"));
        }
        Ok(Self {
            allocation: Arc::new(CudaArenaAllocation { ptr, bytes, device_id }),
            next: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let bytes = len.checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("CUDA arena allocation size overflow"))?;
        let aligned = (bytes + 255) & !255;
        let offset = self.next.fetch_add(aligned, Ordering::Relaxed);
        if offset + aligned > self.allocation.bytes {
            self.next.fetch_sub(aligned, Ordering::Relaxed);
            return Err(anyhow!("CUDA arena exhausted: requested {bytes} bytes, {} remain",
                self.allocation.bytes.saturating_sub(offset)));
        }
        let ptr = unsafe { (self.allocation.ptr as *mut u8).add(offset) as *mut T };
        Ok(CudaBuffer {
            ptr,
            len,
            device_id: self.allocation.device_id,
            allocation: CudaAllocation::Arena(self.allocation.clone()),
            _marker: PhantomData,
        })
    }

    pub fn used_bytes(&self) -> usize { self.next.load(Ordering::Relaxed) }
    pub fn capacity_bytes(&self) -> usize { self.allocation.bytes }
}

pub fn upload_if_full<T: Copy>(
    device: Option<&CudaDevice>,
    data: &[T],
    expected_len: usize,
) -> Option<CudaBuffer<T>> {
    if device.is_some() && data.len() == expected_len {
        CudaBuffer::from_host_on(0, data).ok()
    } else {
        None
    }
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    pub fn alloc_on(device_id: i32, len: usize) -> Result<Self> {
        unsafe { cudaSetDevice(device_id) };
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        let bytes = len * std::mem::size_of::<T>();
        let res = unsafe { cudaMalloc(&mut raw_ptr, bytes) };
        if res != 0 || raw_ptr.is_null() {
            return Err(anyhow!("cudaMalloc failed on GPU {} with code {}", device_id, res));
        }

        Ok(Self {
            ptr: raw_ptr as *mut T,
            len,
            device_id,
            allocation: CudaAllocation::Owned,
            _marker: PhantomData,
        })
    }

    pub fn alloc_pooled_on(device_id: i32, len: usize) -> Result<Self> {
        unsafe { cudaSetDevice(device_id) };
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        let bytes = len * std::mem::size_of::<T>();
        let res = unsafe { cuda_pool_alloc(&mut raw_ptr, bytes) };
        if res != 0 || raw_ptr.is_null() {
            return Err(anyhow!("CUDA pooled allocation failed on GPU {} with code {}", device_id, res));
        }
        Ok(Self { ptr: raw_ptr as *mut T, len, device_id, allocation: CudaAllocation::Pooled, _marker: PhantomData })
    }

    pub fn alloc(len: usize) -> Result<Self> {
        Self::alloc_on(0, len)
    }

    pub fn from_host(slice: &[T]) -> Result<Self> {
        Self::from_host_on(0, slice)
    }

    pub fn from_host_on(device_id: i32, slice: &[T]) -> Result<Self> {
        let mut buf = Self::alloc_on(device_id, slice.len())?;
        buf.copy_from_host(slice)?;
        Ok(buf)
    }

    pub fn copy_from_host(&mut self, slice: &[T]) -> Result<()> {
        assert_eq!(self.len, slice.len());
        unsafe { cudaSetDevice(self.device_id) };
        let bytes = slice.len() * std::mem::size_of::<T>();
        let res = unsafe {
            cudaMemcpy(
                self.ptr as *mut c_void,
                slice.as_ptr() as *const c_void,
                bytes,
                CudaMemcpyKind::HostToDevice,
            )
        };
        if res != 0 {
            return Err(anyhow!("cudaMemcpy (HtoD) failed with code {}", res));
        }
        Ok(())
    }

    pub fn copy_from_host_at(&mut self, offset: usize, slice: &[T]) -> Result<()> {
        assert!(offset + slice.len() <= self.len);
        unsafe { cudaSetDevice(self.device_id) };
        let bytes = std::mem::size_of_val(slice);
        let dst = unsafe { self.ptr.add(offset) };
        let res = unsafe {
            cudaMemcpy(
                dst as *mut c_void,
                slice.as_ptr() as *const c_void,
                bytes,
                CudaMemcpyKind::HostToDevice,
            )
        };
        if res != 0 {
            return Err(anyhow!("cudaMemcpy offset HtoD failed with code {}", res));
        }
        Ok(())
    }

    pub fn copy_to_host(&self, slice: &mut [T]) -> Result<()> {
        assert_eq!(self.len, slice.len());
        unsafe { cudaSetDevice(self.device_id) };
        let bytes = slice.len() * std::mem::size_of::<T>();
        let res = unsafe {
            cudaMemcpy(
                slice.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                bytes,
                CudaMemcpyKind::DeviceToHost,
            )
        };
        if res != 0 {
            return Err(anyhow!("cudaMemcpy (DtoH) failed with code {}", res));
        }
        Ok(())
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaSetDevice(self.device_id);
                match &self.allocation {
                    CudaAllocation::Pooled => cuda_pool_release(
                        self.ptr as *mut c_void, self.len * std::mem::size_of::<T>()),
                    CudaAllocation::Owned => { cudaFree(self.ptr as *mut c_void); }
                    CudaAllocation::Arena(owner) => { let _ = owner; }
                }
            };
        }
    }
}

/// CUDA Device Management and Kernel Execution
pub struct CudaDevice {
    device_id: i32,
    stream: CudaStream,
}

pub struct CudaGraphExec {
    handle: CudaGraphExecHandle,
    device_id: i32,
}

unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cudaSetDevice(self.device_id);
                cudaGraphExecDestroy(self.handle);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub multiprocessors: i32,
    pub runtime_version: i32,
    pub total_memory: usize,
}

impl CudaDeviceInfo {
    pub fn compute_capability(self) -> i32 {
        self.compute_major * 10 + self.compute_minor
    }

    pub fn is_blackwell(self) -> bool {
        self.compute_major >= 10
    }
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_ffn_compute_for_capture(
        &self, gate: &CudaBuffer<u8>, up: &CudaBuffer<u8>, down: &CudaBuffer<u8>,
        shared_in: &CudaBuffer<f32>, dense_act: &mut CudaBuffer<f32>, dense_out: &mut CudaBuffer<f32>,
        router_weights: &CudaBuffer<f32>, router_in: &CudaBuffer<f32>, router_logits: &mut CudaBuffer<f32>,
        expert_ids: &mut CudaBuffer<i32>, expert_weights: &mut CudaBuffer<f32>,
        gate_up_exps: &CudaBuffer<u8>, down_exps: &CudaBuffer<u8>, down_scales: Option<&CudaBuffer<f32>>,
        moe_in: &CudaBuffer<f32>, moe_act: &mut CudaBuffer<f32>, moe_out: &mut CudaBuffer<f32>,
        dim: usize, ffn_dim: usize, exp_dim: usize,
    ) {
        unsafe {
            cuda_op_ffn_compute_launches(
                gate.as_ptr(), up.as_ptr(), down.as_ptr(), shared_in.as_ptr(),
                dense_act.as_mut_ptr(), dense_out.as_mut_ptr(), router_weights.as_ptr(),
                router_in.as_ptr(), router_logits.as_mut_ptr(), expert_ids.as_mut_ptr(),
                expert_weights.as_mut_ptr(), gate_up_exps.as_ptr(), down_exps.as_ptr(),
                down_scales.map_or(ptr::null(), CudaBuffer::as_ptr), moe_in.as_ptr(),
                moe_act.as_mut_ptr(), moe_out.as_mut_ptr(), dim as i32, ffn_dim as i32,
                exp_dim as i32, 128, 8, self.stream,
            );
        }
    }

    pub fn capture<F>(&self, launches: F) -> Result<CudaGraphExec>
    where F: FnOnce() -> Result<()> {
        unsafe { cudaSetDevice(self.device_id) };
        // Thread-local capture prevents an invalid capture from poisoning work
        // submitted by another inference/test thread.
        let status = unsafe { cudaStreamBeginCapture(self.stream, 1) };
        if status != 0 { return Err(anyhow!("cudaStreamBeginCapture failed with code {status}")); }
        if let Err(error) = launches() {
            let mut discarded = ptr::null_mut();
            unsafe {
                cudaStreamEndCapture(self.stream, &mut discarded);
                if !discarded.is_null() { cudaGraphDestroy(discarded); }
            }
            return Err(error);
        }
        let mut graph = ptr::null_mut();
        let status = unsafe { cudaStreamEndCapture(self.stream, &mut graph) };
        if status != 0 || graph.is_null() {
            return Err(anyhow!("cudaStreamEndCapture failed with code {status}"));
        }
        let mut exec = ptr::null_mut();
        let status = unsafe {
            cudaGraphInstantiate(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0)
        };
        unsafe { cudaGraphDestroy(graph) };
        if status != 0 || exec.is_null() {
            return Err(anyhow!("cudaGraphInstantiate failed with code {status}"));
        }
        Ok(CudaGraphExec { handle: exec, device_id: self.device_id })
    }

    pub fn launch_graph(&self, graph: &CudaGraphExec) -> Result<()> {
        unsafe { cudaSetDevice(self.device_id) };
        let status = unsafe { cudaGraphLaunch(graph.handle, self.stream) };
        if status != 0 { return Err(anyhow!("cudaGraphLaunch failed with code {status}")); }
        Ok(())
    }

    pub fn device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        // cudaDevAttrMultiProcessorCount=16, ComputeCapabilityMajor=75,
        // ComputeCapabilityMinor=76. These ABI values are stable in cudart.
        unsafe { cudaSetDevice(device_id) };
        let mut major = 0;
        let mut minor = 0;
        let mut multiprocessors = 0;
        let mut runtime_version = 0;
        let attrs = [
            (75, &mut major),
            (76, &mut minor),
            (16, &mut multiprocessors),
        ];
        for (attr, output) in attrs {
            let status = unsafe { cudaDeviceGetAttribute(output, attr, device_id) };
            if status != 0 {
                return Err(anyhow!("cudaDeviceGetAttribute({attr}) failed with code {status}"));
            }
        }
        let status = unsafe { cudaRuntimeGetVersion(&mut runtime_version) };
        if status != 0 {
            return Err(anyhow!("cudaRuntimeGetVersion failed with code {status}"));
        }
        let (_, total_memory) = Self::get_memory_info(device_id)?;
        Ok(CudaDeviceInfo {
            device_id,
            compute_major: major,
            compute_minor: minor,
            multiprocessors,
            runtime_version,
            total_memory,
        })
    }

    pub fn count() -> usize {
        let mut count = 0;
        let res = unsafe { cudaGetDeviceCount(&mut count) };
        if res == 0 && count > 0 {
            count as usize
        } else {
            0
        }
    }

    pub fn is_available() -> bool {
        Self::count() > 0
    }

    pub fn get_memory_info(device_id: i32) -> Result<(usize, usize)> {
        unsafe {
            cudaSetDevice(device_id);
            let mut free = 0usize;
            let mut total = 0usize;
            let res = cudaMemGetInfo(&mut free, &mut total);
            if res != 0 {
                return Err(anyhow!("cudaMemGetInfo failed for GPU {}", device_id));
            }
            Ok((free, total))
        }
    }

    pub fn new(device_id: i32) -> Result<Self> {
        unsafe { cudaSetDevice(device_id) };
        let mut stream: CudaStream = ptr::null_mut();
        let res = unsafe { cudaStreamCreate(&mut stream) };
        if res != 0 {
            return Err(anyhow!("cudaStreamCreate failed on GPU {} with code {}", device_id, res));
        }
        Ok(Self { device_id, stream })
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { cudaSetDevice(self.device_id) };
        let res = unsafe { cudaStreamSynchronize(self.stream) };
        if res != 0 {
            return Err(anyhow!("cudaStreamSynchronize failed on GPU {} with code {}", self.device_id, res));
        }
        Ok(())
    }

    pub fn rms_norm(
        &self,
        d_x: &CudaBuffer<f32>,
        d_weight: Option<&CudaBuffer<f32>>,
        d_out: &mut CudaBuffer<f32>,
        eps: f32,
    ) {
        unsafe { cudaSetDevice(self.device_id) };
        let w_ptr = d_weight.map(|w| w.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            cuda_op_rms_norm(
                d_x.as_ptr(),
                w_ptr,
                d_out.as_mut_ptr(),
                d_x.len() as i32,
                eps,
                self.stream,
            );
        }
    }

    pub fn rms_norm_batch(
        &self,
        d_x: &CudaBuffer<f32>,
        d_weight: Option<&CudaBuffer<f32>>,
        d_out: &mut CudaBuffer<f32>,
        dim: usize,
        batch: usize,
        eps: f32,
    ) {
        assert_eq!(d_x.len(), dim * batch);
        assert_eq!(d_out.len(), dim * batch);
        let w_ptr = d_weight.map(|weight| weight.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_rms_norm_batch(
                d_x.as_ptr(), w_ptr, d_out.as_mut_ptr(), dim as i32,
                batch as i32, eps, self.stream,
            );
        }
    }

    pub fn swiglu(
        &self,
        d_gate: &CudaBuffer<f32>,
        d_up: &CudaBuffer<f32>,
        d_out: &mut CudaBuffer<f32>,
    ) {
        assert_eq!(d_gate.len(), d_up.len());
        assert_eq!(d_gate.len(), d_out.len());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_swiglu(
                d_gate.as_ptr(),
                d_up.as_ptr(),
                d_out.as_mut_ptr(),
                d_gate.len() as i32,
                self.stream,
            );
        }
    }

    pub fn geglu(
        &self,
        d_gate: &CudaBuffer<f32>,
        d_up: &CudaBuffer<f32>,
        d_out: &mut CudaBuffer<f32>,
    ) {
        assert_eq!(d_gate.len(), d_up.len());
        assert_eq!(d_gate.len(), d_out.len());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_geglu(
                d_gate.as_ptr(),
                d_up.as_ptr(),
                d_out.as_mut_ptr(),
                d_gate.len() as i32,
                self.stream,
            );
        }
    }

    pub fn rope_256k(
        &self,
        d_vec: &mut CudaBuffer<f32>,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        freq_base: f32,
        freq_scale: f32,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_rope_256k(
                d_vec.as_mut_ptr(),
                pos as i32,
                head_dim as i32,
                n_heads as i32,
                freq_base,
                freq_scale,
                self.stream,
            );
        }
    }

    pub fn gemv_q4_0(
        &self,
        d_w_q4: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemv_q4_0(
                d_w_q4.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                self.stream,
            );
        }
    }

    pub fn gemm_q4_0(
        &self,
        d_w_q4: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
        batch: usize,
    ) {
        assert_eq!(d_x.len(), n_cols * batch);
        assert_eq!(d_y.len(), n_rows * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemm_q4_0(
                d_w_q4.as_ptr(), d_x.as_ptr(), d_y.as_mut_ptr(),
                n_rows as i32, n_cols as i32, batch as i32, self.stream,
            );
        }
    }

    pub fn copy_from_host_async<T>(&self, dst: &mut CudaBuffer<T>, src: &[T]) -> Result<()> {
        assert_eq!(dst.len(), src.len());
        unsafe { cudaSetDevice(self.device_id) };
        let res = unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                std::mem::size_of_val(src),
                CudaMemcpyKind::HostToDevice,
                self.stream,
            )
        };
        if res != 0 {
            return Err(anyhow!("cudaMemcpyAsync HtoD failed with code {}", res));
        }
        Ok(())
    }

    pub fn copy_from_host_at_async<T>(
        &self,
        dst: &mut CudaBuffer<T>,
        offset: usize,
        src: &[T],
    ) -> Result<()> {
        assert!(offset + src.len() <= dst.len());
        unsafe { cudaSetDevice(self.device_id) };
        let res = unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr().add(offset) as *mut c_void,
                src.as_ptr() as *const c_void,
                std::mem::size_of_val(src),
                CudaMemcpyKind::HostToDevice,
                self.stream,
            )
        };
        if res != 0 {
            return Err(anyhow!("cudaMemcpyAsync offset HtoD failed with code {}", res));
        }
        Ok(())
    }

    pub fn gemv_q4_0_qkv(
        &self,
        d_w_q: &CudaBuffer<u8>,
        d_w_k: &CudaBuffer<u8>,
        d_w_v: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>,
        q_rows: usize,
        kv_rows: usize,
        n_cols: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemv_q4_0_qkv(
                d_w_q.as_ptr(), d_w_k.as_ptr(), d_w_v.as_ptr(), d_x.as_ptr(),
                d_y.as_mut_ptr(), q_rows as i32, kv_rows as i32, n_cols as i32,
                self.stream,
            );
        }
    }

    pub fn gemm_q4_0_qkv(
        &self, d_w_q: &CudaBuffer<u8>, d_w_k: &CudaBuffer<u8>,
        d_w_v: &CudaBuffer<u8>, d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>, q_rows: usize, kv_rows: usize,
        n_cols: usize, batch: usize,
    ) {
        assert_eq!(d_x.len(), n_cols * batch);
        assert_eq!(d_y.len(), (q_rows + 2 * kv_rows) * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemm_q4_0_qkv(
                d_w_q.as_ptr(), d_w_k.as_ptr(), d_w_v.as_ptr(), d_x.as_ptr(),
                d_y.as_mut_ptr(), q_rows as i32, kv_rows as i32, n_cols as i32,
                batch as i32, self.stream,
            );
        }
    }

    pub fn qkv_postprocess(
        &self,
        qkv: &mut CudaBuffer<f32>,
        q_norm: &CudaBuffer<f32>,
        k_norm: &CudaBuffer<f32>,
        k_cache: &mut CudaBuffer<u16>,
        v_cache: &mut CudaBuffer<u16>,
        pos: usize,
        cache_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        freq_base: f32,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_qkv_postprocess(
                qkv.as_mut_ptr(), q_norm.as_ptr(), k_norm.as_ptr(),
                k_cache.as_mut_ptr(), v_cache.as_mut_ptr(), pos as i32,
                cache_pos as i32, n_heads as i32, n_kv_heads as i32,
                head_dim as i32, freq_base, self.stream,
            );
        }
    }

    pub fn qkv_postprocess_batch(
        &self, qkv: &mut CudaBuffer<f32>, q_norm: &CudaBuffer<f32>,
        k_norm: &CudaBuffer<f32>, k_cache: &mut CudaBuffer<u16>,
        v_cache: &mut CudaBuffer<u16>, start_pos: usize, cache_start: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, freq_base: f32,
        batch: usize,
    ) {
        assert_eq!(qkv.len(), batch * (n_heads + 2 * n_kv_heads) * head_dim);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_qkv_postprocess_batch(
                qkv.as_mut_ptr(), q_norm.as_ptr(), k_norm.as_ptr(),
                k_cache.as_mut_ptr(), v_cache.as_mut_ptr(), start_pos as i32,
                cache_start as i32, n_heads as i32, n_kv_heads as i32,
                head_dim as i32, freq_base, batch as i32, self.stream,
            );
        }
    }

    pub fn gemv_q4_0_geglu(
        &self,
        d_w_gate: &CudaBuffer<u8>,
        d_w_up: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_act: &mut CudaBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemv_q4_0_geglu(
                d_w_gate.as_ptr(), d_w_up.as_ptr(), d_x.as_ptr(), d_act.as_mut_ptr(),
                n_rows as i32, n_cols as i32, self.stream,
            );
        }
    }

    pub fn gemv_q8_0(
        &self,
        d_w_q8: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemv_q8_0(
                d_w_q8.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                self.stream,
            );
        }
    }

    pub fn add(
        &self,
        d_a: &CudaBuffer<f32>,
        d_b: &CudaBuffer<f32>,
        d_out: &mut CudaBuffer<f32>,
    ) {
        assert_eq!(d_a.len(), d_b.len());
        assert_eq!(d_a.len(), d_out.len());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_add(
                d_a.as_ptr(),
                d_b.as_ptr(),
                d_out.as_mut_ptr(),
                d_a.len() as i32,
                self.stream,
            );
        }
    }

    pub fn embedding(
        &self,
        d_table: &CudaBuffer<f32>,
        d_out: &mut CudaBuffer<f32>,
        token: usize,
        dim: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_embedding(
                d_table.as_ptr(),
                d_out.as_mut_ptr(),
                token as i32,
                dim as i32,
                self.stream,
            );
        }
    }

    pub fn attention_causal(
        &self,
        d_q: &CudaBuffer<f32>,
        d_k_cache: &CudaBuffer<u16>,
        d_v_cache: &CudaBuffer<u16>,
        d_out: &mut CudaBuffer<f32>,
        n_past: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        sliding_window: Option<usize>,
    ) {
        let sw = sliding_window.map(|w| w as i32).unwrap_or(-1);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_attention(
                d_q.as_ptr(),
                d_k_cache.as_ptr(),
                d_v_cache.as_ptr(),
                d_out.as_mut_ptr(),
                n_past as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                scale,
                sw,
                self.stream,
            );
        }
    }

    pub fn gemm_q4_0_geglu(
        &self, d_w_gate: &CudaBuffer<u8>, d_w_up: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>, d_act: &mut CudaBuffer<f32>,
        n_rows: usize, n_cols: usize, batch: usize,
    ) {
        assert_eq!(d_x.len(), n_cols * batch);
        assert_eq!(d_act.len(), n_rows * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_gemm_q4_0_geglu(
                d_w_gate.as_ptr(), d_w_up.as_ptr(), d_x.as_ptr(),
                d_act.as_mut_ptr(), n_rows as i32, n_cols as i32,
                batch as i32, self.stream,
            );
        }
    }

    pub fn attention_prefill(
        &self, d_q: &CudaBuffer<f32>, d_k_cache: &CudaBuffer<u16>,
        d_v_cache: &CudaBuffer<u16>, d_out: &mut CudaBuffer<f32>,
        cache_start: usize, batch: usize, n_heads: usize, n_kv_heads: usize,
        head_dim: usize, scale: f32, sliding_window: Option<usize>,
    ) {
        let q_stride = d_q.len() / batch;
        assert!(q_stride >= n_heads * head_dim);
        assert_eq!(d_out.len(), batch * n_heads * head_dim);
        let window = sliding_window.map(|value| value as i32).unwrap_or(-1);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_attention_prefill(
                d_q.as_ptr(), d_k_cache.as_ptr(), d_v_cache.as_ptr(),
                d_out.as_mut_ptr(), cache_start as i32, batch as i32,
                n_heads as i32, n_kv_heads as i32, head_dim as i32, q_stride as i32,
                scale, window, self.stream,
            );
        }
    }

    pub fn moe_topk_q4_0(
        &self,
        d_gate_up_exps: &CudaBuffer<u8>,
        d_down_exps: &CudaBuffer<u8>,
        d_active_exp_ids: &CudaBuffer<i32>,
        d_active_exp_weights: &CudaBuffer<f32>,
        d_down_exps_scale: Option<&CudaBuffer<f32>>,
        d_x_in: &CudaBuffer<f32>,
        d_act_scratch: &mut CudaBuffer<f32>,
        d_out_moe: &mut CudaBuffer<f32>,
        dim: usize,
        exp_ffn_dim: usize,
        n_active: usize,
    ) {
        let scale_ptr = d_down_exps_scale.map(|b| b.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_moe_topk_q4_0(
                d_gate_up_exps.as_ptr(),
                d_down_exps.as_ptr(),
                d_active_exp_ids.as_ptr(),
                d_active_exp_weights.as_ptr(),
                scale_ptr,
                d_x_in.as_ptr(),
                d_act_scratch.as_mut_ptr(),
                d_out_moe.as_mut_ptr(),
                dim as i32,
                exp_ffn_dim as i32,
                n_active as i32,
                self.stream,
            );
        }
    }

    pub fn moe_router(
        &self,
        d_weights: &CudaBuffer<f32>,
        d_input: &CudaBuffer<f32>,
        d_logits: &mut CudaBuffer<f32>,
        d_ids: &mut CudaBuffer<i32>,
        d_probabilities: &mut CudaBuffer<f32>,
        dim: usize,
        n_experts: usize,
    ) {
        assert!(d_weights.len() >= dim * n_experts);
        assert!(d_input.len() >= dim);
        assert!(d_logits.len() >= n_experts);
        assert!(d_ids.len() >= 8);
        assert!(d_probabilities.len() >= 8);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_moe_router(
                d_weights.as_ptr(), d_input.as_ptr(), d_logits.as_mut_ptr(),
                d_ids.as_mut_ptr(), d_probabilities.as_mut_ptr(), dim as i32,
                n_experts as i32, self.stream,
            );
        }
    }

    pub fn moe_topk_batch_q4_0(
        &self, gate_up: &CudaBuffer<u8>, down: &CudaBuffer<u8>,
        ids: &CudaBuffer<i32>, weights: &CudaBuffer<f32>,
        scales: Option<&CudaBuffer<f32>>, input: &CudaBuffer<f32>,
        act: &mut CudaBuffer<f32>, output: &mut CudaBuffer<f32>,
        dim: usize, exp_dim: usize, n_active: usize, batch: usize,
    ) {
        assert_eq!(ids.len(), batch * n_active);
        assert_eq!(weights.len(), batch * n_active);
        assert_eq!(input.len(), batch * dim);
        assert_eq!(act.len(), batch * n_active * exp_dim);
        assert_eq!(output.len(), batch * dim);
        let scale_ptr = scales.map(|value| value.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_moe_topk_batch_q4_0(
                gate_up.as_ptr(), down.as_ptr(), ids.as_ptr(), weights.as_ptr(),
                scale_ptr, input.as_ptr(), act.as_mut_ptr(), output.as_mut_ptr(),
                dim as i32, exp_dim as i32, n_active as i32, batch as i32,
                self.stream,
            );
        }
    }

    pub fn moe_router_batch(
        &self, weights: &CudaBuffer<f32>, input: &CudaBuffer<f32>,
        logits: &mut CudaBuffer<f32>, ids: &mut CudaBuffer<i32>,
        probabilities: &mut CudaBuffer<f32>, dim: usize,
        n_experts: usize, batch: usize,
    ) {
        assert_eq!(input.len(), dim * batch);
        assert_eq!(logits.len(), n_experts * batch);
        assert_eq!(ids.len(), 8 * batch);
        assert_eq!(probabilities.len(), 8 * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_moe_router_batch(
                weights.as_ptr(), input.as_ptr(), logits.as_mut_ptr(),
                ids.as_mut_ptr(), probabilities.as_mut_ptr(), dim as i32,
                n_experts as i32, batch as i32, self.stream,
            );
        }
    }

    pub fn prepare_ffn(
        &self, hidden: &CudaBuffer<f32>, attn_proj: &CudaBuffer<f32>,
        post_attn_norm: &CudaBuffer<f32>, ffn_norm: &CudaBuffer<f32>,
        pre_ffw_norm_2: &CudaBuffer<f32>, router_scale: &CudaBuffer<f32>,
        attn_res: &mut CudaBuffer<f32>, shared_in: &mut CudaBuffer<f32>,
        moe_in: &mut CudaBuffer<f32>, router_in: &mut CudaBuffer<f32>, dim: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_prepare_ffn(
                hidden.as_ptr(), attn_proj.as_ptr(), post_attn_norm.as_ptr(),
                ffn_norm.as_ptr(), pre_ffw_norm_2.as_ptr(), router_scale.as_ptr(),
                attn_res.as_mut_ptr(), shared_in.as_mut_ptr(), moe_in.as_mut_ptr(),
                router_in.as_mut_ptr(), dim as i32, self.stream,
            );
        }
    }

    pub fn prepare_ffn_batch(
        &self, hidden: &CudaBuffer<f32>, attn: &CudaBuffer<f32>,
        pan: &CudaBuffer<f32>, ffn: &CudaBuffer<f32>, pfn: &CudaBuffer<f32>,
        router_scale: &CudaBuffer<f32>, attn_res: &mut CudaBuffer<f32>,
        shared: &mut CudaBuffer<f32>, moe: &mut CudaBuffer<f32>,
        router: &mut CudaBuffer<f32>, dim: usize, batch: usize,
    ) {
        assert_eq!(hidden.len(), dim * batch);
        assert_eq!(attn.len(), dim * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_prepare_ffn_batch(
                hidden.as_ptr(), attn.as_ptr(), pan.as_ptr(), ffn.as_ptr(),
                pfn.as_ptr(), router_scale.as_ptr(), attn_res.as_mut_ptr(),
                shared.as_mut_ptr(), moe.as_mut_ptr(), router.as_mut_ptr(),
                dim as i32, batch as i32, self.stream,
            );
        }
    }

    pub fn finish_ffn(
        &self, attn_res: &CudaBuffer<f32>, dense: &mut CudaBuffer<f32>,
        moe: &mut CudaBuffer<f32>, post_ffw_norm_1: &CudaBuffer<f32>,
        post_ffw_norm_2: &CudaBuffer<f32>, post_ffw_norm: &CudaBuffer<f32>,
        hidden_out: &mut CudaBuffer<f32>, layer_scale: f32, dim: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_finish_ffn(
                attn_res.as_ptr(), dense.as_mut_ptr(), moe.as_mut_ptr(),
                post_ffw_norm_1.as_ptr(), post_ffw_norm_2.as_ptr(),
                post_ffw_norm.as_ptr(), hidden_out.as_mut_ptr(), layer_scale,
                dim as i32, self.stream,
            );
        }
    }

    pub fn finish_ffn_batch(
        &self, attn_res: &CudaBuffer<f32>, dense: &mut CudaBuffer<f32>,
        moe: &mut CudaBuffer<f32>, p1: &CudaBuffer<f32>, p2: &CudaBuffer<f32>,
        pf: &CudaBuffer<f32>, output: &mut CudaBuffer<f32>, scale: f32,
        dim: usize, batch: usize,
    ) {
        assert_eq!(attn_res.len(), dim * batch);
        assert_eq!(output.len(), dim * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_finish_ffn_batch(
                attn_res.as_ptr(), dense.as_mut_ptr(), moe.as_mut_ptr(),
                p1.as_ptr(), p2.as_ptr(), pf.as_ptr(), output.as_mut_ptr(),
                scale, dim as i32, batch as i32, self.stream,
            );
        }
    }

    pub fn vocab_topk(
        &self,
        logits: &CudaBuffer<f32>,
        valid: &CudaBuffer<u8>,
        recent: &CudaBuffer<i32>,
        scores: &mut CudaBuffer<f32>,
        ids: &mut CudaBuffer<i32>,
        vocab_size: usize,
        n_recent: usize,
        generated_count: usize,
        k: usize,
        partitions: usize,
    ) {
        assert!(valid.len() >= vocab_size);
        assert!(recent.len() >= n_recent);
        assert!(scores.len() >= k * partitions);
        assert!(ids.len() >= k * partitions);
        unsafe {
            cudaSetDevice(self.device_id);
            cuda_op_vocab_topk(
                logits.as_ptr(), valid.as_ptr(), recent.as_ptr(),
                scores.as_mut_ptr(), ids.as_mut_ptr(), vocab_size as i32,
                n_recent as i32, generated_count as i32, k as i32,
                partitions as i32, self.stream,
            );
        }
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                cudaSetDevice(self.device_id);
                cudaStreamDestroy(self.stream);
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn arena_returns_stable_aligned_non_overlapping_views() -> Result<()> {
        let _serial = CUDA_TEST_LOCK.lock().unwrap();
        if !CudaDevice::is_available() { return Ok(()); }
        let arena = CudaArena::new(0, 4096)?;
        let first = arena.alloc::<u8>(257)?;
        let second = arena.alloc::<f32>(64)?;
        assert_eq!((first.as_ptr() as usize) & 255, 0);
        assert_eq!((second.as_ptr() as usize) & 255, 0);
        assert!(second.as_ptr() as usize >= first.as_ptr() as usize + 512);
        assert_eq!(arena.used_bytes(), 768);
        assert_eq!(arena.capacity_bytes(), 4096);
        Ok(())
    }

    #[test]
    fn captured_kernel_graph_replays_with_stable_arena_buffers() -> Result<()> {
        let _serial = CUDA_TEST_LOCK.lock().unwrap();
        if !CudaDevice::is_available() { return Ok(()); }
        let device = CudaDevice::new(0)?;
        let arena = CudaArena::new(0, 4096)?;
        let mut a = arena.alloc::<f32>(64)?;
        let mut b = arena.alloc::<f32>(64)?;
        let mut output = arena.alloc::<f32>(64)?;
        a.copy_from_host(&vec![1.25; 64])?;
        b.copy_from_host(&vec![2.5; 64])?;
        let graph = device.capture(|| {
            // Captured sections must contain launches only. The ordinary Rust
            // wrappers select a device defensively before each launch, which
            // CUDA intentionally rejects while a stream is being captured.
            unsafe {
                cuda_op_add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), 64, device.stream);
            }
            Ok(())
        })?;
        device.launch_graph(&graph)?;
        device.launch_graph(&graph)?;
        device.sync()?;
        let mut actual = vec![0.0; 64];
        output.copy_to_host(&mut actual)?;
        assert!(actual.iter().all(|value| (*value - 3.75).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn moe_router_matches_cpu_top8_and_softmax() -> Result<()> {
        let _serial = CUDA_TEST_LOCK.lock().unwrap();
        if !CudaDevice::is_available() {
            return Ok(());
        }
        const DIM: usize = 257;
        const EXPERTS: usize = 128;
        let input: Vec<f32> = (0..DIM)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.003)
            .collect();
        let weights: Vec<f32> = (0..EXPERTS)
            .flat_map(|e| {
                (0..DIM).map(move |i| {
                    (((e * 29 + i * 17) % 113) as f32 - 56.0) * 0.002
                        + e as f32 * 0.00001
                })
            })
            .collect();

        let mut expected: Vec<(f32, i32)> = (0..EXPERTS)
            .map(|e| {
                let dot = input.iter().zip(&weights[e * DIM..(e + 1) * DIM])
                    .map(|(x, w)| x * w).sum();
                (dot, e as i32)
            })
            .collect();
        expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let max = expected[0].0;
        let denom: f32 = expected[..8].iter().map(|x| (x.0 - max).exp()).sum();

        let dev = CudaDevice::new(0)?;
        let d_weights = CudaBuffer::from_host_on(0, &weights)?;
        let d_input = CudaBuffer::from_host_on(0, &input)?;
        let mut d_logits = CudaBuffer::alloc_on(0, EXPERTS)?;
        let mut d_ids = CudaBuffer::alloc_on(0, 8)?;
        let mut d_probs = CudaBuffer::alloc_on(0, 8)?;
        dev.moe_router(&d_weights, &d_input, &mut d_logits, &mut d_ids, &mut d_probs, DIM, EXPERTS);
        dev.sync()?;
        let mut ids = [0i32; 8];
        let mut probs = [0.0f32; 8];
        d_ids.copy_to_host(&mut ids)?;
        d_probs.copy_to_host(&mut probs)?;

        for i in 0..8 {
            assert_eq!(ids[i], expected[i].1);
            let probability = (expected[i].0 - max).exp() / denom;
            assert!((probs[i] - probability).abs() < 2e-5,
                "probability {i}: GPU {} CPU {probability}", probs[i]);
        }
        Ok(())
    }

    #[test]
    fn fused_ffn_residual_pipeline_matches_cpu() -> Result<()> {
        let _serial = CUDA_TEST_LOCK.lock().unwrap();
        if !CudaDevice::is_available() { return Ok(()); }
        const DIM: usize = 257;
        let values = |seed: usize| -> Vec<f32> {
            (0..DIM).map(|i| (((i * seed + 19) % 97) as f32 - 48.0) * 0.007).collect()
        };
        let hidden = values(13);
        let projection = values(29);
        let post_attn: Vec<f32> = values(17).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let ffn_norm: Vec<f32> = values(23).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let pre_moe: Vec<f32> = values(31).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let router_scale: Vec<f32> = values(37).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let post_1: Vec<f32> = values(41).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let post_2: Vec<f32> = values(43).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let post: Vec<f32> = values(47).into_iter().map(|x| 1.0 + x * 0.1).collect();
        let dense = values(53);
        let moe = values(59);

        let dev = CudaDevice::new(0)?;
        let d_hidden = CudaBuffer::from_host_on(0, &hidden)?;
        let d_projection = CudaBuffer::from_host_on(0, &projection)?;
        let d_post_attn = CudaBuffer::from_host_on(0, &post_attn)?;
        let d_ffn_norm = CudaBuffer::from_host_on(0, &ffn_norm)?;
        let d_pre_moe = CudaBuffer::from_host_on(0, &pre_moe)?;
        let d_router_scale = CudaBuffer::from_host_on(0, &router_scale)?;
        let mut d_attn_res = CudaBuffer::alloc_on(0, DIM)?;
        let mut d_shared = CudaBuffer::alloc_on(0, DIM)?;
        let mut d_moe_in = CudaBuffer::alloc_on(0, DIM)?;
        let mut d_router = CudaBuffer::alloc_on(0, DIM)?;
        dev.prepare_ffn(
            &d_hidden, &d_projection, &d_post_attn, &d_ffn_norm, &d_pre_moe,
            &d_router_scale, &mut d_attn_res, &mut d_shared, &mut d_moe_in,
            &mut d_router, DIM,
        );
        dev.sync()?;

        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / DIM as f32 + 1e-6).sqrt().recip();
        let projection_inv = rms(&projection);
        let expected_res: Vec<f32> = (0..DIM)
            .map(|i| hidden[i] + projection[i] * projection_inv * post_attn[i]).collect();
        let res_inv = rms(&expected_res);
        let mut actual_res = vec![0.0; DIM];
        let mut actual_shared = vec![0.0; DIM];
        let mut actual_moe_in = vec![0.0; DIM];
        let mut actual_router = vec![0.0; DIM];
        d_attn_res.copy_to_host(&mut actual_res)?;
        d_shared.copy_to_host(&mut actual_shared)?;
        d_moe_in.copy_to_host(&mut actual_moe_in)?;
        d_router.copy_to_host(&mut actual_router)?;
        for i in 0..DIM {
            assert!((actual_res[i] - expected_res[i]).abs() < 2e-5);
            assert!((actual_shared[i] - expected_res[i] * res_inv * ffn_norm[i]).abs() < 2e-5);
            assert!((actual_moe_in[i] - expected_res[i] * res_inv * pre_moe[i]).abs() < 2e-5);
            let expected_router = expected_res[i] * res_inv / (DIM as f32).sqrt() * router_scale[i];
            assert!((actual_router[i] - expected_router).abs() < 2e-5);
        }

        let mut d_dense = CudaBuffer::from_host_on(0, &dense)?;
        let mut d_moe = CudaBuffer::from_host_on(0, &moe)?;
        let d_post_1 = CudaBuffer::from_host_on(0, &post_1)?;
        let d_post_2 = CudaBuffer::from_host_on(0, &post_2)?;
        let d_post = CudaBuffer::from_host_on(0, &post)?;
        let mut d_out = CudaBuffer::alloc_on(0, DIM)?;
        dev.finish_ffn(
            &d_attn_res, &mut d_dense, &mut d_moe, &d_post_1, &d_post_2,
            &d_post, &mut d_out, 0.75, DIM,
        );
        dev.sync()?;
        let dense_inv = rms(&dense);
        let moe_inv = rms(&moe);
        let combined: Vec<f32> = (0..DIM).map(|i|
            dense[i] * dense_inv * post_1[i] + moe[i] * moe_inv * post_2[i]
        ).collect();
        let combined_inv = rms(&combined);
        let mut actual_out = vec![0.0; DIM];
        d_out.copy_to_host(&mut actual_out)?;
        for i in 0..DIM {
            let expected = (expected_res[i] + combined[i] * combined_inv * post[i]) * 0.75;
            assert!((actual_out[i] - expected).abs() < 3e-5,
                "output {i}: GPU {} CPU {expected}", actual_out[i]);
        }
        Ok(())
    }
}
