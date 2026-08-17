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
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind, stream: CudaStream) -> i32;
    fn cudaStreamCreate(stream: *mut CudaStream) -> i32;
    fn cudaStreamDestroy(stream: CudaStream) -> i32;
    fn cudaStreamSynchronize(stream: CudaStream) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;

    // Exported custom kernel launches from cuda_kernels.cu
    fn cuda_op_rms_norm(d_x: *const f32, d_w: *const f32, d_out: *mut f32, dim: i32, eps: f32, stream: CudaStream);
    fn cuda_op_swiglu(d_gate: *const f32, d_up: *const f32, d_out: *mut f32, size: i32, stream: CudaStream);
    fn cuda_op_rope(d_vec: *mut f32, pos: i32, head_dim: i32, n_heads: i32, freq_base: f32, freq_scale: f32, stream: CudaStream);
    fn cuda_op_gemv_q4_0(d_w_q4: *const u8, d_x: *const f32, d_y: *mut f32, n_rows: i32, n_cols: i32, stream: CudaStream);
}

/// RAII wrapper for GPU Device Memory
pub struct CudaBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        let bytes = len * std::mem::size_of::<T>();
        let res = unsafe { cudaMalloc(&mut raw_ptr, bytes) };
        if res != 0 || raw_ptr.is_null() {
            return Err(anyhow!("cudaMalloc failed with code {}", res));
        }

        Ok(Self {
            ptr: raw_ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    pub fn from_host(slice: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(slice.len())?;
        buf.copy_from_host(slice)?;
        Ok(buf)
    }

    pub fn copy_from_host(&mut self, slice: &[T]) -> Result<()> {
        assert_eq!(self.len, slice.len());
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

    pub fn copy_to_host(&self, slice: &mut [T]) -> Result<()> {
        assert_eq!(self.len, slice.len());
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
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr as *mut c_void) };
        }
    }
}

/// CUDA Device Management and Kernel Execution
pub struct CudaDevice {
    stream: CudaStream,
}

impl CudaDevice {
    pub fn is_available() -> bool {
        let mut count = 0;
        let res = unsafe { cudaGetDeviceCount(&mut count) };
        res == 0 && count > 0
    }

    pub fn new() -> Result<Self> {
        let mut stream: CudaStream = ptr::null_mut();
        let res = unsafe { cudaStreamCreate(&mut stream) };
        if res != 0 {
            return Err(anyhow!("cudaStreamCreate failed with code {}", res));
        }
        Ok(Self { stream })
    }

    pub fn sync(&self) -> Result<()> {
        let res = unsafe { cudaStreamSynchronize(self.stream) };
        if res != 0 {
            return Err(anyhow!("cudaStreamSynchronize failed with code {}", res));
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
            cuda_op_swiglu(
                d_gate.as_ptr(),
                d_up.as_ptr(),
                d_out.as_mut_ptr(),
                d_gate.len() as i32,
                self.stream,
            );
        }
    }

    pub fn rope(
        &self,
        d_vec: &mut CudaBuffer<f32>,
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        freq_base: f32,
        freq_scale: f32,
    ) {
        unsafe {
            cuda_op_rope(
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
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}
