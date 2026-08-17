use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

pub type CudaStream = *mut c_void;

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
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind, stream: CudaStream) -> i32;
    fn cudaStreamCreate(stream: *mut CudaStream) -> i32;
    fn cudaStreamDestroy(stream: CudaStream) -> i32;
    fn cudaStreamSynchronize(stream: CudaStream) -> i32;
    fn cudaDeviceSynchronize() -> i32;

    // Exported custom kernel launches from cuda_kernels.cu
    fn cuda_op_rms_norm(d_x: *const f32, d_w: *const f32, d_out: *mut f32, dim: i32, eps: f32, stream: CudaStream);
    fn cuda_op_swiglu(d_gate: *const f32, d_up: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_geglu(d_gate: *const f32, d_up: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_rope_256k(d_vec: *mut f32, pos: i32, head_dim: i32, n_heads: i32, freq_base: f32, freq_scale: f32, stream: CudaStream);
    fn cuda_op_gemv_q4_0(d_w_q4: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream);
    fn cuda_op_gemv_q4_0_qkv(
        d_w_q: *const u8, d_w_k: *const u8, d_w_v: *const u8,
        d_x: *const f32, d_y: *mut f32, q_rows: i32, kv_rows: i32,
        n_cols: i32, stream: CudaStream,
    );
    fn cuda_op_gemv_q4_0_geglu(
        d_w_gate: *const u8, d_w_up: *const u8, d_x: *const f32,
        d_act: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream,
    );
    fn cuda_op_gemv_q8_0(d_w_q8: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream);
    fn cuda_op_add(d_a: *const f32, d_b: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_embedding(d_table: *const f32, d_out: *mut f32, token: i32, dim: i32, stream: CudaStream);
    fn cuda_op_attention(
        d_q: *const f32,
        d_k_cache: *const f32,
        d_v_cache: *const f32,
        d_out: *mut f32,
        n_past: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        sliding_window: i32,
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
}

/// RAII wrapper for GPU Device Memory
pub struct CudaBuffer<T> {
    ptr: *mut T,
    len: usize,
    device_id: i32,
    _marker: PhantomData<T>,
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
            _marker: PhantomData,
        })
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
                cudaFree(self.ptr as *mut c_void);
            };
        }
    }
}

/// CUDA Device Management and Kernel Execution
pub struct CudaDevice {
    device_id: i32,
    stream: CudaStream,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
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
        d_k_cache: &CudaBuffer<f32>,
        d_v_cache: &CudaBuffer<f32>,
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
