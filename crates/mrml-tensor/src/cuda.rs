#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case)]

use crate::anyhow::{Result, anyhow};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub type CudaStream = *mut c_void;
type CudaGraphHandle = *mut c_void;
type CudaGraphExecHandle = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuContext = *mut c_void;
type CuDevicePtr = u64;

static RUST_PTX_MODULE: OnceLock<usize> = OnceLock::new();
static RUST_FUNCTIONS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
static CUDA_DRIVER: OnceLock<CudaDriverApi> = OnceLock::new();
static CUDA_ALLOCATION_POOL: OnceLock<Mutex<HashMap<(i32, usize), Vec<usize>>>> = OnceLock::new();
const RUST_CUDA_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rust_cuda_kernels.ptx"));

struct CudaDriverApi {
    init: unsafe extern "C" fn(u32) -> i32,
    device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    device_get_count: unsafe extern "C" fn(*mut i32) -> i32,
    device_get_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    driver_get_version: unsafe extern "C" fn(*mut i32) -> i32,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, i32) -> i32,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> i32,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> i32,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> i32,
    memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> i32,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> i32,
    memcpy_dtod: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> i32,
    memcpy_htod_async: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CudaStream) -> i32,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CudaStream) -> i32,
    memcpy_dtod_async: unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize, CudaStream) -> i32,
    memset_d8_async: unsafe extern "C" fn(CuDevicePtr, u8, usize, CudaStream) -> i32,
    stream_create: unsafe extern "C" fn(*mut CudaStream, u32) -> i32,
    stream_destroy: unsafe extern "C" fn(CudaStream) -> i32,
    stream_synchronize: unsafe extern "C" fn(CudaStream) -> i32,
    stream_begin_capture: unsafe extern "C" fn(CudaStream, i32) -> i32,
    stream_end_capture: unsafe extern "C" fn(CudaStream, *mut CudaGraphHandle) -> i32,
    graph_instantiate: unsafe extern "C" fn(
        *mut CudaGraphExecHandle,
        CudaGraphHandle,
        *mut *mut c_void,
        *mut i8,
        usize,
    ) -> i32,
    graph_launch: unsafe extern "C" fn(CudaGraphExecHandle, CudaStream) -> i32,
    graph_destroy: unsafe extern "C" fn(CudaGraphHandle) -> i32,
    graph_exec_destroy: unsafe extern "C" fn(CudaGraphExecHandle) -> i32,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> i32,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> i32,
    launch_kernel: unsafe extern "C" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CudaStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}

static CUDA_PRIMARY_CONTEXTS: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();

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

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const i8, flags: i32) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const i8) -> *mut c_void;
}

fn cuda_driver_library() -> Result<usize> {
    static LIBRARY: OnceLock<usize> = OnceLock::new();
    if let Some(library) = LIBRARY.get() {
        return Ok(*library);
    }
    #[cfg(windows)]
    let library = unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr()) };
    #[cfg(unix)]
    let library = unsafe { dlopen(c"libcuda.so.1".as_ptr(), 2) };
    if library.is_null() {
        return Err(anyhow!("NVIDIA CUDA driver library is not installed"));
    }
    let _ = LIBRARY.set(library as usize);
    Ok(*LIBRARY.get().unwrap())
}

unsafe fn cuda_driver_symbol(name: &str) -> Result<*mut c_void> {
    let library = cuda_driver_library()? as *mut c_void;
    let name = CString::new(name).map_err(|_| anyhow!("invalid CUDA driver symbol"))?;
    #[cfg(windows)]
    let function = GetProcAddress(library, name.as_ptr());
    #[cfg(unix)]
    let function = dlsym(library, name.as_ptr());
    if function.is_null() {
        Err(anyhow!("CUDA driver symbol {name:?} is unavailable"))
    } else {
        Ok(function)
    }
}

pub fn clear_cuda_allocation_pool() {
    let allocations = CUDA_ALLOCATION_POOL
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("CUDA allocation pool mutex poisoned")
        .drain()
        .flat_map(|((device, _), pointers)| {
            pointers.into_iter().map(move |pointer| (device, pointer))
        })
        .collect::<Vec<_>>();
    for (device, pointer) in allocations {
        unsafe {
            cudaSetDevice(device);
            cudaFree(pointer as *mut c_void);
        }
    }
}

fn pooled_cuda_alloc(device_id: i32, bytes: usize) -> (*mut c_void, i32) {
    if let Some(pointer) = CUDA_ALLOCATION_POOL
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("CUDA allocation pool mutex poisoned")
        .get_mut(&(device_id, bytes))
        .and_then(Vec::pop)
    {
        return (pointer as *mut c_void, 0);
    }
    let mut pointer = ptr::null_mut();
    let status = unsafe { cudaMalloc(&mut pointer, bytes) };
    (pointer, status)
}

fn pooled_cuda_release(device_id: i32, pointer: *mut c_void, bytes: usize) {
    CUDA_ALLOCATION_POOL
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("CUDA allocation pool mutex poisoned")
        .entry((device_id, bytes))
        .or_default()
        .push(pointer as usize);
}

fn cuda_driver() -> Result<&'static CudaDriverApi> {
    if let Some(api) = CUDA_DRIVER.get() {
        return Ok(api);
    }
    let api = unsafe {
        CudaDriverApi {
            init: std::mem::transmute(cuda_driver_symbol("cuInit")?),
            device_get: std::mem::transmute(cuda_driver_symbol("cuDeviceGet")?),
            device_get_count: std::mem::transmute(cuda_driver_symbol("cuDeviceGetCount")?),
            device_get_attribute: std::mem::transmute(cuda_driver_symbol("cuDeviceGetAttribute")?),
            driver_get_version: std::mem::transmute(cuda_driver_symbol("cuDriverGetVersion")?),
            primary_ctx_retain: std::mem::transmute(cuda_driver_symbol(
                "cuDevicePrimaryCtxRetain",
            )?),
            ctx_set_current: std::mem::transmute(cuda_driver_symbol("cuCtxSetCurrent")?),
            mem_get_info: std::mem::transmute(cuda_driver_symbol("cuMemGetInfo_v2")?),
            mem_alloc: std::mem::transmute(cuda_driver_symbol("cuMemAlloc_v2")?),
            mem_free: std::mem::transmute(cuda_driver_symbol("cuMemFree_v2")?),
            memcpy_htod: std::mem::transmute(cuda_driver_symbol("cuMemcpyHtoD_v2")?),
            memcpy_dtoh: std::mem::transmute(cuda_driver_symbol("cuMemcpyDtoH_v2")?),
            memcpy_dtod: std::mem::transmute(cuda_driver_symbol("cuMemcpyDtoD_v2")?),
            memcpy_htod_async: std::mem::transmute(cuda_driver_symbol("cuMemcpyHtoDAsync_v2")?),
            memcpy_dtoh_async: std::mem::transmute(cuda_driver_symbol("cuMemcpyDtoHAsync_v2")?),
            memcpy_dtod_async: std::mem::transmute(cuda_driver_symbol("cuMemcpyDtoDAsync_v2")?),
            memset_d8_async: std::mem::transmute(cuda_driver_symbol("cuMemsetD8Async")?),
            stream_create: std::mem::transmute(cuda_driver_symbol("cuStreamCreate")?),
            stream_destroy: std::mem::transmute(cuda_driver_symbol("cuStreamDestroy_v2")?),
            stream_synchronize: std::mem::transmute(cuda_driver_symbol("cuStreamSynchronize")?),
            stream_begin_capture: std::mem::transmute(cuda_driver_symbol(
                "cuStreamBeginCapture_v2",
            )?),
            stream_end_capture: std::mem::transmute(cuda_driver_symbol("cuStreamEndCapture")?),
            graph_instantiate: std::mem::transmute(cuda_driver_symbol("cuGraphInstantiate_v2")?),
            graph_launch: std::mem::transmute(cuda_driver_symbol("cuGraphLaunch")?),
            graph_destroy: std::mem::transmute(cuda_driver_symbol("cuGraphDestroy")?),
            graph_exec_destroy: std::mem::transmute(cuda_driver_symbol("cuGraphExecDestroy")?),
            module_load_data: std::mem::transmute(cuda_driver_symbol("cuModuleLoadData")?),
            module_get_function: std::mem::transmute(cuda_driver_symbol("cuModuleGetFunction")?),
            launch_kernel: std::mem::transmute(cuda_driver_symbol("cuLaunchKernel")?),
        }
    };
    let _ = CUDA_DRIVER.set(api);
    Ok(CUDA_DRIVER.get().unwrap())
}

const CUDA_DRIVER_ERROR: i32 = 999;

unsafe fn cuda_set_device_raw(device_id: i32) -> i32 {
    let Ok(api) = cuda_driver() else {
        return CUDA_DRIVER_ERROR;
    };
    let init = (api.init)(0);
    if init != 0 {
        return init;
    }
    let context = {
        let mut contexts = CUDA_PRIMARY_CONTEXTS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("CUDA primary-context mutex poisoned");
        if let Some(context) = contexts.get(&device_id) {
            *context as CuContext
        } else {
            let mut device = 0;
            let status = (api.device_get)(&mut device, device_id);
            if status != 0 {
                return status;
            }
            let mut context = ptr::null_mut();
            let status = (api.primary_ctx_retain)(&mut context, device);
            if status != 0 {
                return status;
            }
            contexts.insert(device_id, context as usize);
            context
        }
    };
    (api.ctx_set_current)(context)
}

unsafe fn cudaGetDeviceCount(count: *mut i32) -> i32 {
    let Ok(api) = cuda_driver() else {
        return CUDA_DRIVER_ERROR;
    };
    let status = (api.init)(0);
    if status == 0 {
        (api.device_get_count)(count)
    } else {
        status
    }
}

unsafe fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32 {
    cuda_driver()
        .map(|api| (api.device_get_attribute)(value, attr, device))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaRuntimeGetVersion(version: *mut i32) -> i32 {
    cuda_driver()
        .map(|api| (api.driver_get_version)(version))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32 {
    cuda_driver()
        .map(|api| (api.mem_get_info)(free, total))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32 {
    let Ok(api) = cuda_driver() else {
        return CUDA_DRIVER_ERROR;
    };
    let mut pointer = 0;
    let status = (api.mem_alloc)(&mut pointer, size);
    if status == 0 {
        *dev_ptr = pointer as usize as *mut c_void;
    }
    status
}

unsafe fn cudaFree(dev_ptr: *mut c_void) -> i32 {
    cuda_driver()
        .map(|api| (api.mem_free)(dev_ptr as usize as CuDevicePtr))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
) -> i32 {
    let Ok(api) = cuda_driver() else {
        return CUDA_DRIVER_ERROR;
    };
    match kind {
        CudaMemcpyKind::HostToHost => {
            ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), count);
            0
        }
        CudaMemcpyKind::HostToDevice => (api.memcpy_htod)(dst as usize as CuDevicePtr, src, count),
        CudaMemcpyKind::DeviceToHost => (api.memcpy_dtoh)(dst, src as usize as CuDevicePtr, count),
        CudaMemcpyKind::DeviceToDevice => (api.memcpy_dtod)(
            dst as usize as CuDevicePtr,
            src as usize as CuDevicePtr,
            count,
        ),
        CudaMemcpyKind::Default => CUDA_DRIVER_ERROR,
    }
}

unsafe fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
    stream: CudaStream,
) -> i32 {
    let Ok(api) = cuda_driver() else {
        return CUDA_DRIVER_ERROR;
    };
    match kind {
        CudaMemcpyKind::HostToHost => {
            ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), count);
            0
        }
        CudaMemcpyKind::HostToDevice => {
            (api.memcpy_htod_async)(dst as usize as CuDevicePtr, src, count, stream)
        }
        CudaMemcpyKind::DeviceToHost => {
            (api.memcpy_dtoh_async)(dst, src as usize as CuDevicePtr, count, stream)
        }
        CudaMemcpyKind::DeviceToDevice => (api.memcpy_dtod_async)(
            dst as usize as CuDevicePtr,
            src as usize as CuDevicePtr,
            count,
            stream,
        ),
        CudaMemcpyKind::Default => CUDA_DRIVER_ERROR,
    }
}

unsafe fn cudaMemsetAsync(dst: *mut c_void, value: i32, count: usize, stream: CudaStream) -> i32 {
    cuda_driver()
        .map(|api| (api.memset_d8_async)(dst as usize as CuDevicePtr, value as u8, count, stream))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaStreamCreate(stream: *mut CudaStream) -> i32 {
    cuda_driver()
        .map(|api| (api.stream_create)(stream, 0))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaStreamDestroy(stream: CudaStream) -> i32 {
    cuda_driver()
        .map(|api| (api.stream_destroy)(stream))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaStreamSynchronize(stream: CudaStream) -> i32 {
    cuda_driver()
        .map(|api| (api.stream_synchronize)(stream))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaStreamBeginCapture(stream: CudaStream, mode: i32) -> i32 {
    cuda_driver()
        .map(|api| (api.stream_begin_capture)(stream, mode))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraphHandle) -> i32 {
    cuda_driver()
        .map(|api| (api.stream_end_capture)(stream, graph))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaGraphInstantiate(
    exec: *mut CudaGraphExecHandle,
    graph: CudaGraphHandle,
    error_node: *mut *mut c_void,
    log_buffer: *mut i8,
    buffer_size: usize,
) -> i32 {
    cuda_driver()
        .map(|api| (api.graph_instantiate)(exec, graph, error_node, log_buffer, buffer_size))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaGraphLaunch(exec: CudaGraphExecHandle, stream: CudaStream) -> i32 {
    cuda_driver()
        .map(|api| (api.graph_launch)(exec, stream))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaGraphDestroy(graph: CudaGraphHandle) -> i32 {
    cuda_driver()
        .map(|api| (api.graph_destroy)(graph))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

unsafe fn cudaGraphExecDestroy(exec: CudaGraphExecHandle) -> i32 {
    cuda_driver()
        .map(|api| (api.graph_exec_destroy)(exec))
        .unwrap_or(CUDA_DRIVER_ERROR)
}

fn rust_ptx_function(name: &str) -> Result<CuFunction> {
    let api = cuda_driver()?;
    let functions = RUST_FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(function) = functions.lock().unwrap().get(name).copied() {
        return Ok(function as CuFunction);
    }
    let module = if let Some(module) = RUST_PTX_MODULE.get() {
        *module as CuModule
    } else {
        let mut module = ptr::null_mut();
        let mut image = Vec::with_capacity(RUST_CUDA_PTX.len() + 1);
        image.extend_from_slice(RUST_CUDA_PTX);
        image.push(0);
        let init = unsafe { (api.init)(0) };
        let loaded =
            unsafe { (api.module_load_data)(&mut module, image.as_ptr() as *const c_void) };
        if init != 0 || loaded != 0 || module.is_null() {
            return Err(anyhow!(
                "loading Rust CUDA PTX failed (cuInit={init}, cuModuleLoadData={loaded})"
            ));
        }
        let _ = RUST_PTX_MODULE.set(module as usize);
        module
    };
    let symbol = CString::new(name).map_err(|_| anyhow!("invalid CUDA kernel name"))?;
    let mut function = ptr::null_mut();
    let status = unsafe { (api.module_get_function)(&mut function, module, symbol.as_ptr()) };
    if status != 0 || function.is_null() {
        return Err(anyhow!(
            "Rust CUDA kernel '{name}' lookup failed with code {status}"
        ));
    }
    functions
        .lock()
        .unwrap()
        .insert(name.to_string(), function as usize);
    Ok(function)
}

unsafe fn launch_rust_kernel(
    name: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: CudaStream,
    args: &mut [*mut c_void],
) -> Result<()> {
    launch_rust_kernel_shared(name, grid, block, 0, stream, args)
}

unsafe fn launch_rust_kernel_shared(
    name: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    stream: CudaStream,
    args: &mut [*mut c_void],
) -> Result<()> {
    let status = (cuda_driver()?.launch_kernel)(
        rust_ptx_function(name)?,
        grid.0,
        grid.1,
        grid.2,
        block.0,
        block.1,
        block.2,
        shared_bytes,
        stream,
        args.as_mut_ptr(),
        ptr::null_mut(),
    );
    if status != 0 {
        Err(anyhow!(
            "Rust CUDA kernel '{name}' launch failed with code {status}"
        ))
    } else {
        Ok(())
    }
}

unsafe fn launch_rust_add(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    size: i32,
    stream: CudaStream,
) -> Result<()> {
    let function = rust_ptx_function("rust_cuda_add_f32")?;
    let mut a_arg = a;
    let mut b_arg = b;
    let mut out_arg = out;
    let mut size_arg = size;
    let mut args = [
        &mut a_arg as *mut _ as *mut c_void,
        &mut b_arg as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
        &mut size_arg as *mut _ as *mut c_void,
    ];
    let blocks = (size.max(0) as u32).div_ceil(256).max(1);
    let status = (cuda_driver()?.launch_kernel)(
        function,
        blocks,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        ptr::null_mut(),
    );
    if status != 0 {
        Err(anyhow!("Rust CUDA add launch failed with code {status}"))
    } else {
        Ok(())
    }
}

unsafe fn launch_rust_gemm_q4(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let function = rust_ptx_function("rust_cuda_gemm_q4_0_f32")?;
    let mut weights_arg = weights;
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut batch_arg = batch;
    let mut args = [
        &mut weights_arg as *mut _ as *mut c_void,
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut rows_arg as *mut _ as *mut c_void,
        &mut cols_arg as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    let grid_x = (rows.max(0) as u32).div_ceil(8).max(1);
    let grid_y = (batch.max(0) as u32).div_ceil(8).max(1);
    let status = (cuda_driver()?.launch_kernel)(
        function,
        grid_x,
        grid_y,
        1,
        128,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        ptr::null_mut(),
    );
    if status != 0 {
        Err(anyhow!(
            "Rust CUDA Q4 GEMM launch failed with code {status}"
        ))
    } else {
        Ok(())
    }
}

unsafe fn launch_rust_gemv_q4(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    stream: CudaStream,
) -> Result<()> {
    let function = rust_ptx_function("rust_cuda_gemv_q4_0_f32")?;
    let mut weights_arg = weights;
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut args = [
        &mut weights_arg as *mut _ as *mut c_void,
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut rows_arg as *mut _ as *mut c_void,
        &mut cols_arg as *mut _ as *mut c_void,
    ];
    let grid_x = (rows.max(0) as u32).div_ceil(16).max(1);
    let status = (cuda_driver()?.launch_kernel)(
        function,
        grid_x,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        ptr::null_mut(),
    );
    if status != 0 {
        Err(anyhow!(
            "Rust CUDA Q4 GEMV launch failed with code {status}"
        ))
    } else {
        Ok(())
    }
}

unsafe fn launch_rust_qkv_q4(
    wq: *const u8,
    wk: *const u8,
    wv: *const u8,
    input: *const f32,
    output: *mut f32,
    q_rows: i32,
    kv_rows: i32,
    cols: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let mut wq_arg = wq;
    let mut wk_arg = wk;
    let mut wv_arg = wv;
    let mut input_arg = input;
    let mut output_arg = output;
    let mut q_arg = q_rows;
    let mut kv_arg = kv_rows;
    let mut cols_arg = cols;
    let mut batch_arg = batch;
    let mut args = [
        &mut wq_arg as *mut _ as *mut c_void,
        &mut wk_arg as *mut _ as *mut c_void,
        &mut wv_arg as *mut _ as *mut c_void,
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut q_arg as *mut _ as *mut c_void,
        &mut kv_arg as *mut _ as *mut c_void,
        &mut cols_arg as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    let total = (q_rows + 2 * kv_rows).max(0) as u32;
    launch_rust_kernel(
        "rust_cuda_gemm_q4_0_qkv_f32",
        (
            total.div_ceil(16).max(1),
            (batch.max(0) as u32).div_ceil(8).max(1),
            1,
        ),
        (256, 1, 1),
        stream,
        &mut args,
    )
}

unsafe fn launch_rust_qkv_q4_gemv(
    wq: *const u8,
    wk: *const u8,
    wv: *const u8,
    input: *const f32,
    output: *mut f32,
    q_rows: i32,
    kv_rows: i32,
    cols: i32,
    stream: CudaStream,
) -> Result<()> {
    let (mut a0, mut a1, mut a2, mut a3, mut a4) = (wq, wk, wv, input, output);
    let (mut qr, mut kr, mut c) = (q_rows, kv_rows, cols);
    let mut args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut qr as *mut _ as *mut c_void,
        &mut kr as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void,
    ];
    let total = (q_rows + 2 * kv_rows).max(0) as u32;
    launch_rust_kernel(
        "rust_cuda_gemv_q4_0_qkv_f32",
        (total.div_ceil(8).max(1), 1, 1),
        (128, 1, 1),
        stream,
        &mut args,
    )
}

unsafe fn launch_rust_geglu_q4(
    wgate: *const u8,
    wup: *const u8,
    input: *const f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let mut gate_arg = wgate;
    let mut up_arg = wup;
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut batch_arg = batch;
    let mut args = [
        &mut gate_arg as *mut _ as *mut c_void,
        &mut up_arg as *mut _ as *mut c_void,
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut rows_arg as *mut _ as *mut c_void,
        &mut cols_arg as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    if batch == 1 {
        // Decode does not need the eight-token accumulator arrays used by the
        // prefill GEMM. The dedicated GEMV materially reduces register pressure.
        launch_rust_kernel(
            "rust_cuda_gemv_q4_0_geglu_f32",
            ((rows.max(0) as u32).div_ceil(8).max(1), 1, 1),
            (128, 1, 1),
            stream,
            &mut args[..6],
        )
    } else {
        launch_rust_kernel(
            "rust_cuda_gemm_q4_0_geglu_f32",
            (
                (rows.max(0) as u32).div_ceil(16).max(1),
                (batch.max(0) as u32).div_ceil(8).max(1),
                1,
            ),
            (256, 1, 1),
            stream,
            &mut args,
        )
    }
}

unsafe fn launch_rust_moe_router(
    weights: *const f32,
    input: *const f32,
    logits: *mut f32,
    ids: *mut i32,
    probabilities: *mut f32,
    dim: i32,
    n_experts: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let mut weights_arg = weights;
    let mut input_arg = input;
    let mut logits_arg = logits;
    let mut dim_arg = dim;
    let mut experts_arg = n_experts;
    let mut batch_arg = batch;
    let mut logits_args = [
        &mut weights_arg as *mut _ as *mut c_void,
        &mut input_arg as *mut _ as *mut c_void,
        &mut logits_arg as *mut _ as *mut c_void,
        &mut dim_arg as *mut _ as *mut c_void,
        &mut experts_arg as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    launch_rust_kernel(
        "rust_cuda_moe_router_logits_f32",
        (n_experts as u32, batch as u32, 1),
        (256, 1, 1),
        stream,
        &mut logits_args,
    )?;
    let mut ids_arg = ids;
    let mut probabilities_arg = probabilities;
    let mut top_args = [
        &mut logits_arg as *mut _ as *mut c_void,
        &mut ids_arg as *mut _ as *mut c_void,
        &mut probabilities_arg as *mut _ as *mut c_void,
        &mut experts_arg as *mut _ as *mut c_void,
        &mut batch_arg as *mut _ as *mut c_void,
    ];
    launch_rust_kernel(
        "rust_cuda_moe_router_top8_f32",
        (batch as u32, 1, 1),
        (1, 1, 1),
        stream,
        &mut top_args,
    )
}

unsafe fn launch_rust_prepare_ffn(
    hidden: *const f32,
    attn: *const f32,
    pan: *const f32,
    ffn: *const f32,
    pfn: *const f32,
    router_scale: *const f32,
    attn_res: *mut f32,
    shared: *mut f32,
    moe: *mut f32,
    router: *mut f32,
    dim: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8, mut a9) = (
        hidden,
        attn,
        pan,
        ffn,
        pfn,
        router_scale,
        attn_res,
        shared,
        moe,
        router,
    );
    let mut d = dim;
    let mut b = batch;
    let mut args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut a5 as *mut _ as *mut c_void,
        &mut a6 as *mut _ as *mut c_void,
        &mut a7 as *mut _ as *mut c_void,
        &mut a8 as *mut _ as *mut c_void,
        &mut a9 as *mut _ as *mut c_void,
        &mut d as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
    ];
    launch_rust_kernel(
        "rust_cuda_prepare_ffn_f32",
        (batch as u32, 1, 1),
        (512, 1, 1),
        stream,
        &mut args,
    )
}

unsafe fn launch_rust_finish_ffn(
    attn_res: *const f32,
    dense: *mut f32,
    moe: *mut f32,
    p1: *const f32,
    p2: *const f32,
    pf: *const f32,
    output: *mut f32,
    scale: f32,
    dim: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6) =
        (attn_res, dense, moe, p1, p2, pf, output);
    let mut s = scale;
    let mut d = dim;
    let mut b = batch;
    let mut args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut a5 as *mut _ as *mut c_void,
        &mut a6 as *mut _ as *mut c_void,
        &mut s as *mut _ as *mut c_void,
        &mut d as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
    ];
    launch_rust_kernel(
        "rust_cuda_finish_ffn_f32",
        (batch as u32, 1, 1),
        (512, 1, 1),
        stream,
        &mut args,
    )
}

unsafe fn launch_rust_qkv_postprocess(
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
    stream: CudaStream,
) -> Result<()> {
    let (mut a0, mut a1, mut a2, mut a3, mut a4) = (qkv, q_norm, k_norm, k_cache, v_cache);
    let (mut p, mut cp, mut nh, mut nkh, mut hd, mut fb, mut b, mut cap, mut kf, mut vf) = (
        start_pos,
        cache_start,
        n_heads,
        n_kv_heads,
        head_dim,
        freq_base,
        batch,
        cache_capacity,
        k_format,
        v_format,
    );
    let mut args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut p as *mut _ as *mut c_void,
        &mut cp as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void,
        &mut nkh as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void,
        &mut fb as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut cap as *mut _ as *mut c_void,
        &mut kf as *mut _ as *mut c_void,
        &mut vf as *mut _ as *mut c_void,
    ];
    let threads = if head_dim >= 512 { 256 } else { 128 };
    launch_rust_kernel(
        "rust_cuda_qkv_postprocess",
        ((n_heads + 2 * n_kv_heads) as u32, batch as u32, 1),
        (threads, 1, 1),
        stream,
        &mut args,
    )
}

unsafe fn launch_rust_attention(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    out: *mut f32,
    cache_start: i32,
    batch: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    q_stride: i32,
    scale: f32,
    window: i32,
    capacity: i32,
    k_format: i32,
    v_format: i32,
    stream: CudaStream,
) -> Result<()> {
    let (mut a0, mut a1, mut a2, mut a3) = (q, k, v, out);
    let (mut cs, mut b, mut nh, mut nkh, mut hd, mut qs, mut s, mut w, mut cap, mut kf, mut vf) = (
        cache_start,
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
        q_stride,
        scale,
        window,
        capacity,
        k_format,
        v_format,
    );
    let mut args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut cs as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void,
        &mut nkh as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void,
        &mut qs as *mut _ as *mut c_void,
        &mut s as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
        &mut cap as *mut _ as *mut c_void,
        &mut kf as *mut _ as *mut c_void,
        &mut vf as *mut _ as *mut c_void,
    ];
    let max_keys = if window > 0 {
        (cache_start + batch).min(window)
    } else {
        cache_start + batch
    };
    if max_keys <= 8192
        && capacity > 0
        && capacity & (capacity - 1) == 0
        && head_dim & 1 == 0
        && k_format == 0
        && v_format == 0
    {
        let shared_bytes = max_keys as u32 * std::mem::size_of::<f32>() as u32;
        launch_rust_kernel_shared(
            "rust_cuda_attention",
            (n_heads as u32, batch as u32, 1),
            (128, 1, 1),
            shared_bytes,
            stream,
            &mut args,
        )
    } else {
        launch_rust_kernel(
            "rust_cuda_attention_streaming",
            (n_heads as u32, batch as u32, 1),
            (128, 1, 1),
            stream,
            &mut args,
        )
    }
}

unsafe fn launch_rust_moe_topk(
    gate_up: *const u8,
    down: *const u8,
    ids: *const i32,
    weights: *const f32,
    scales: *const f32,
    input: *const f32,
    act: *mut f32,
    out: *mut f32,
    dim: i32,
    exp_dim: i32,
    n_active: i32,
    batch: i32,
    stream: CudaStream,
) -> Result<()> {
    if batch != 1 {
        let clear = cudaMemsetAsync(
            out as *mut c_void,
            0,
            batch as usize * dim as usize * std::mem::size_of::<f32>(),
            stream,
        );
        if clear != 0 {
            return Err(anyhow!(
                "clearing Rust CUDA MoE output failed with code {clear}"
            ));
        }
    }
    let (mut a0, mut a1, mut a2, mut a3) = (gate_up, ids, input, act);
    let (mut ed, mut d, mut na, mut b) = (exp_dim, dim, n_active, batch);
    let mut gate_args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut ed as *mut _ as *mut c_void,
        &mut d as *mut _ as *mut c_void,
        &mut na as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
    ];
    if batch == 1 && dim == 2_816 && exp_dim == 704 && n_active == 8 {
        launch_rust_kernel(
            "rust_cuda_moe_gate_up_q4_gemma4_26b",
            ((exp_dim as u32).div_ceil(8), n_active as u32, 1),
            (128, 1, 1),
            stream,
            &mut gate_args[..4],
        )?;
    } else {
        launch_rust_kernel(
            "rust_cuda_moe_gate_up_q4",
            ((exp_dim as u32).div_ceil(8), n_active as u32, batch as u32),
            (128, 1, 1),
            stream,
            &mut gate_args,
        )?;
    }
    let (mut d0, mut d1, mut d2, mut d3, mut d4, mut d5) = (down, ids, weights, scales, act, out);
    let mut down_args = [
        &mut d0 as *mut _ as *mut c_void,
        &mut d1 as *mut _ as *mut c_void,
        &mut d2 as *mut _ as *mut c_void,
        &mut d3 as *mut _ as *mut c_void,
        &mut d4 as *mut _ as *mut c_void,
        &mut d5 as *mut _ as *mut c_void,
        &mut d as *mut _ as *mut c_void,
        &mut ed as *mut _ as *mut c_void,
        &mut na as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
    ];
    if batch == 1 && dim == 2_816 && exp_dim == 704 && n_active == 8 {
        return launch_rust_kernel(
            "rust_cuda_moe_down_q4_gemma4_26b",
            ((dim as u32).div_ceil(8), 1, 1),
            (128, 1, 1),
            stream,
            &mut down_args[..6],
        );
    }
    let (kernel, grid_y) = if batch == 1 {
        ("rust_cuda_moe_down_q4_combined", 1)
    } else {
        ("rust_cuda_moe_down_q4", n_active as u32)
    };
    launch_rust_kernel(
        kernel,
        ((dim as u32).div_ceil(8), grid_y, batch as u32),
        (128, 1, 1),
        stream,
        &mut down_args,
    )
}

// CUDA's current device is thread-local. Decode used to enter the runtime for
// every buffer operation and kernel launch even though inference stays on one
// device. ggml avoids that repeated runtime/driver dispatch by remembering the
// active device on the submitting thread. Keep the existing call sites behind
// this shim so allocations, copies, graphs, and launches share the policy.
thread_local! {
    static ACTIVE_CUDA_DEVICE: Cell<i32> = const { Cell::new(-1) };
}

#[allow(non_snake_case)]
unsafe fn cudaSetDevice(device: i32) -> i32 {
    ACTIVE_CUDA_DEVICE.with(|active| {
        if active.get() == device {
            return 0;
        }
        let status = cuda_set_device_raw(device);
        if status == 0 {
            active.set(device);
        }
        status
    })
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
            return Err(anyhow!(
                "CUDA arena allocation of {bytes} bytes failed with code {status}"
            ));
        }
        Ok(Self {
            allocation: Arc::new(CudaArenaAllocation {
                ptr,
                bytes,
                device_id,
            }),
            next: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("CUDA arena allocation size overflow"))?;
        let aligned = (bytes + 255) & !255;
        let offset = self.next.fetch_add(aligned, Ordering::Relaxed);
        if offset + aligned > self.allocation.bytes {
            self.next.fetch_sub(aligned, Ordering::Relaxed);
            return Err(anyhow!(
                "CUDA arena exhausted: requested {bytes} bytes, {} remain",
                self.allocation.bytes.saturating_sub(offset)
            ));
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

    pub fn used_bytes(&self) -> usize {
        self.next.load(Ordering::Relaxed)
    }
    pub fn capacity_bytes(&self) -> usize {
        self.allocation.bytes
    }
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
            return Err(anyhow!(
                "cudaMalloc failed on GPU {} with code {}",
                device_id,
                res
            ));
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
        let bytes = len * std::mem::size_of::<T>();
        let (raw_ptr, res) = pooled_cuda_alloc(device_id, bytes);
        if res != 0 || raw_ptr.is_null() {
            return Err(anyhow!(
                "CUDA pooled allocation failed on GPU {} with code {}",
                device_id,
                res
            ));
        }
        Ok(Self {
            ptr: raw_ptr as *mut T,
            len,
            device_id,
            allocation: CudaAllocation::Pooled,
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
                match &self.allocation {
                    CudaAllocation::Pooled => pooled_cuda_release(
                        self.device_id,
                        self.ptr as *mut c_void,
                        self.len * std::mem::size_of::<T>(),
                    ),
                    CudaAllocation::Owned => {
                        cudaFree(self.ptr as *mut c_void);
                    }
                    CudaAllocation::Arena(owner) => {
                        let _ = owner;
                    }
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
        &self,
        gate: &CudaBuffer<u8>,
        up: &CudaBuffer<u8>,
        down: &CudaBuffer<u8>,
        shared_in: &CudaBuffer<f32>,
        dense_act: &mut CudaBuffer<f32>,
        dense_out: &mut CudaBuffer<f32>,
        router_weights: &CudaBuffer<f32>,
        router_in: &CudaBuffer<f32>,
        router_logits: &mut CudaBuffer<f32>,
        expert_ids: &mut CudaBuffer<i32>,
        expert_weights: &mut CudaBuffer<f32>,
        gate_up_exps: &CudaBuffer<u8>,
        down_exps: &CudaBuffer<u8>,
        down_scales: Option<&CudaBuffer<f32>>,
        moe_in: &CudaBuffer<f32>,
        moe_act: &mut CudaBuffer<f32>,
        moe_out: &mut CudaBuffer<f32>,
        dim: usize,
        ffn_dim: usize,
        exp_dim: usize,
    ) -> Result<()> {
        unsafe {
            launch_rust_geglu_q4(
                gate.as_ptr(),
                up.as_ptr(),
                shared_in.as_ptr(),
                dense_act.as_mut_ptr(),
                ffn_dim as i32,
                dim as i32,
                1,
                self.stream,
            )?;
            launch_rust_gemv_q4(
                down.as_ptr(),
                dense_act.as_ptr(),
                dense_out.as_mut_ptr(),
                dim as i32,
                ffn_dim as i32,
                self.stream,
            )?;
            launch_rust_moe_router(
                router_weights.as_ptr(),
                router_in.as_ptr(),
                router_logits.as_mut_ptr(),
                expert_ids.as_mut_ptr(),
                expert_weights.as_mut_ptr(),
                dim as i32,
                128,
                1,
                self.stream,
            )?;
            launch_rust_moe_topk(
                gate_up_exps.as_ptr(),
                down_exps.as_ptr(),
                expert_ids.as_ptr(),
                expert_weights.as_ptr(),
                down_scales.map_or(ptr::null(), CudaBuffer::as_ptr),
                moe_in.as_ptr(),
                moe_act.as_mut_ptr(),
                moe_out.as_mut_ptr(),
                dim as i32,
                exp_dim as i32,
                8,
                1,
                self.stream,
            )?;
        }
        Ok(())
    }

    pub fn capture<F>(&self, launches: F) -> Result<CudaGraphExec>
    where
        F: FnOnce() -> Result<()>,
    {
        unsafe { cudaSetDevice(self.device_id) };
        // Thread-local capture prevents an invalid capture from poisoning work
        // submitted by another inference/test thread.
        let status = unsafe { cudaStreamBeginCapture(self.stream, 1) };
        if status != 0 {
            return Err(anyhow!("cudaStreamBeginCapture failed with code {status}"));
        }
        if let Err(error) = launches() {
            let mut discarded = ptr::null_mut();
            unsafe {
                cudaStreamEndCapture(self.stream, &mut discarded);
                if !discarded.is_null() {
                    cudaGraphDestroy(discarded);
                }
            }
            return Err(error);
        }
        let mut graph = ptr::null_mut();
        let status = unsafe { cudaStreamEndCapture(self.stream, &mut graph) };
        if status != 0 || graph.is_null() {
            return Err(anyhow!("cudaStreamEndCapture failed with code {status}"));
        }
        let mut exec = ptr::null_mut();
        let status =
            unsafe { cudaGraphInstantiate(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0) };
        unsafe { cudaGraphDestroy(graph) };
        if status != 0 || exec.is_null() {
            return Err(anyhow!("cudaGraphInstantiate failed with code {status}"));
        }
        Ok(CudaGraphExec {
            handle: exec,
            device_id: self.device_id,
        })
    }

    pub fn launch_graph(&self, graph: &CudaGraphExec) -> Result<()> {
        unsafe { cudaSetDevice(self.device_id) };
        let status = unsafe { cudaGraphLaunch(graph.handle, self.stream) };
        if status != 0 {
            return Err(anyhow!("cudaGraphLaunch failed with code {status}"));
        }
        Ok(())
    }

    pub fn device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        // cudaDevAttrMultiProcessorCount=16, ComputeCapabilityMajor=75,
        // ComputeCapabilityMinor=76. These ABI values are stable in the CUDA driver API.
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
                return Err(anyhow!(
                    "cudaDeviceGetAttribute({attr}) failed with code {status}"
                ));
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
            return Err(anyhow!(
                "cudaStreamCreate failed on GPU {} with code {}",
                device_id,
                res
            ));
        }
        Ok(Self { device_id, stream })
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { cudaSetDevice(self.device_id) };
        let res = unsafe { cudaStreamSynchronize(self.stream) };
        if res != 0 {
            return Err(anyhow!(
                "cudaStreamSynchronize failed on GPU {} with code {}",
                self.device_id,
                res
            ));
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
            {
                let mut x = d_x.as_ptr();
                let mut weight = w_ptr;
                let mut out = d_out.as_mut_ptr();
                let mut dim = d_x.len() as i32;
                let mut batch = 1i32;
                let mut epsilon = eps;
                let mut args = [
                    &mut x as *mut _ as *mut c_void,
                    &mut weight as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut batch as *mut _ as *mut c_void,
                    &mut epsilon as *mut _ as *mut c_void,
                ];
                launch_rust_kernel(
                    "rust_cuda_rms_norm_f32",
                    (1, 1, 1),
                    (256, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA RMS norm kernel failed");
            }
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
        let w_ptr = d_weight
            .map(|weight| weight.as_ptr())
            .unwrap_or(ptr::null());
        unsafe {
            cudaSetDevice(self.device_id);
            {
                let mut x = d_x.as_ptr();
                let mut weight = w_ptr;
                let mut out = d_out.as_mut_ptr();
                let mut dim_arg = dim as i32;
                let mut batch_arg = batch as i32;
                let mut epsilon = eps;
                let mut args = [
                    &mut x as *mut _ as *mut c_void,
                    &mut weight as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                    &mut dim_arg as *mut _ as *mut c_void,
                    &mut batch_arg as *mut _ as *mut c_void,
                    &mut epsilon as *mut _ as *mut c_void,
                ];
                launch_rust_kernel(
                    "rust_cuda_rms_norm_f32",
                    (batch as u32, 1, 1),
                    (256, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA batched RMS norm kernel failed");
            }
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
            {
                let mut gate = d_gate.as_ptr();
                let mut up = d_up.as_ptr();
                let mut out = d_out.as_mut_ptr();
                let mut size = d_gate.len() as i32;
                let mut args = [
                    &mut gate as *mut _ as *mut c_void,
                    &mut up as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                    &mut size as *mut _ as *mut c_void,
                ];
                launch_rust_kernel(
                    "rust_cuda_swiglu_f32",
                    ((d_gate.len() as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA SwiGLU kernel failed");
            }
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
            {
                let mut gate = d_gate.as_ptr();
                let mut up = d_up.as_ptr();
                let mut out = d_out.as_mut_ptr();
                let mut size = d_gate.len() as i32;
                let mut args = [
                    &mut gate as *mut _ as *mut c_void,
                    &mut up as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                    &mut size as *mut _ as *mut c_void,
                ];
                launch_rust_kernel(
                    "rust_cuda_geglu_f32",
                    ((d_gate.len() as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA GeGLU kernel failed");
            }
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
            {
                let mut vec = d_vec.as_mut_ptr();
                let mut pos_arg = pos as i32;
                let mut dim = head_dim as i32;
                let mut heads = n_heads as i32;
                let mut base = freq_base;
                let mut scale = freq_scale;
                let mut args = [
                    &mut vec as *mut _ as *mut c_void,
                    &mut pos_arg as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut heads as *mut _ as *mut c_void,
                    &mut base as *mut _ as *mut c_void,
                    &mut scale as *mut _ as *mut c_void,
                ];
                launch_rust_kernel(
                    "rust_cuda_rope_f32",
                    (n_heads as u32, 1, 1),
                    ((head_dim / 2) as u32, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA RoPE kernel failed");
            }
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
            launch_rust_gemv_q4(
                d_w_q4.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                self.stream,
            )
            .expect("Rust CUDA Q4 GEMV kernel failed");
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
            launch_rust_gemm_q4(
                d_w_q4.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA Q4 GEMM kernel failed");
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
            return Err(anyhow!(
                "cudaMemcpyAsync offset HtoD failed with code {}",
                res
            ));
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
            launch_rust_qkv_q4_gemv(
                d_w_q.as_ptr(),
                d_w_k.as_ptr(),
                d_w_v.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                q_rows as i32,
                kv_rows as i32,
                n_cols as i32,
                self.stream,
            )
            .expect("Rust CUDA fused QKV GEMV kernel failed");
        }
    }

    pub fn gemm_q4_0_qkv(
        &self,
        d_w_q: &CudaBuffer<u8>,
        d_w_k: &CudaBuffer<u8>,
        d_w_v: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_y: &mut CudaBuffer<f32>,
        q_rows: usize,
        kv_rows: usize,
        n_cols: usize,
        batch: usize,
    ) {
        assert_eq!(d_x.len(), n_cols * batch);
        assert_eq!(d_y.len(), (q_rows + 2 * kv_rows) * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_qkv_q4(
                d_w_q.as_ptr(),
                d_w_k.as_ptr(),
                d_w_v.as_ptr(),
                d_x.as_ptr(),
                d_y.as_mut_ptr(),
                q_rows as i32,
                kv_rows as i32,
                n_cols as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA fused QKV GEMM kernel failed");
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
        k_format: i32,
        v_format: i32,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_qkv_postprocess(
                qkv.as_mut_ptr(),
                q_norm.as_ptr(),
                k_norm.as_ptr(),
                k_cache.as_mut_ptr(),
                v_cache.as_mut_ptr(),
                pos as i32,
                cache_pos as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                freq_base,
                1,
                0,
                k_format,
                v_format,
                self.stream,
            )
            .expect("Rust CUDA QKV postprocessing failed");
        }
    }

    pub fn qkv_postprocess_batch(
        &self,
        qkv: &mut CudaBuffer<f32>,
        q_norm: &CudaBuffer<f32>,
        k_norm: &CudaBuffer<f32>,
        k_cache: &mut CudaBuffer<u16>,
        v_cache: &mut CudaBuffer<u16>,
        start_pos: usize,
        cache_start: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        freq_base: f32,
        batch: usize,
        cache_capacity: usize,
        k_format: i32,
        v_format: i32,
    ) {
        assert_eq!(qkv.len(), batch * (n_heads + 2 * n_kv_heads) * head_dim);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_qkv_postprocess(
                qkv.as_mut_ptr(),
                q_norm.as_ptr(),
                k_norm.as_ptr(),
                k_cache.as_mut_ptr(),
                v_cache.as_mut_ptr(),
                start_pos as i32,
                cache_start as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                freq_base,
                batch as i32,
                cache_capacity as i32,
                k_format,
                v_format,
                self.stream,
            )
            .expect("Rust CUDA batched QKV postprocessing failed");
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
            launch_rust_geglu_q4(
                d_w_gate.as_ptr(),
                d_w_up.as_ptr(),
                d_x.as_ptr(),
                d_act.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                1,
                self.stream,
            )
            .expect("Rust CUDA fused Q4 GeGLU GEMV kernel failed");
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
            let mut weights = d_w_q8.as_ptr();
            let mut input = d_x.as_ptr();
            let mut output = d_y.as_mut_ptr();
            let mut rows = n_rows as i32;
            let mut cols = n_cols as i32;
            let mut args = [
                &mut weights as *mut _ as *mut c_void,
                &mut input as *mut _ as *mut c_void,
                &mut output as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
            ];
            launch_rust_kernel(
                "rust_cuda_gemv_q8_0_f32",
                ((n_rows as u32).div_ceil(8), 1, 1),
                (128, 1, 1),
                self.stream,
                &mut args,
            )
            .expect("Rust CUDA Q8 GEMV kernel failed");
        }
    }

    pub fn add(&self, d_a: &CudaBuffer<f32>, d_b: &CudaBuffer<f32>, d_out: &mut CudaBuffer<f32>) {
        assert_eq!(d_a.len(), d_b.len());
        assert_eq!(d_a.len(), d_out.len());
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_add(
                d_a.as_ptr(),
                d_b.as_ptr(),
                d_out.as_mut_ptr(),
                d_a.len() as i32,
                self.stream,
            )
            .expect("Rust CUDA add kernel failed");
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
            let mut table = d_table.as_ptr();
            let mut out = d_out.as_mut_ptr();
            let mut token_arg = token as i32;
            let mut dim_arg = dim as i32;
            let mut args = [
                &mut table as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
                &mut token_arg as *mut _ as *mut c_void,
                &mut dim_arg as *mut _ as *mut c_void,
            ];
            launch_rust_kernel(
                "rust_cuda_embedding_f32",
                ((dim as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                self.stream,
                &mut args,
            )
            .expect("Rust CUDA embedding kernel failed");
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
        cache_capacity: usize,
        k_format: i32,
        v_format: i32,
    ) {
        let sw = sliding_window.map(|w| w as i32).unwrap_or(-1);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_attention(
                d_q.as_ptr(),
                d_k_cache.as_ptr(),
                d_v_cache.as_ptr(),
                d_out.as_mut_ptr(),
                n_past as i32,
                1,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                (n_heads * head_dim) as i32,
                scale,
                sw,
                cache_capacity as i32,
                k_format,
                v_format,
                self.stream,
            )
            .expect("Rust CUDA causal attention failed");
        }
    }

    pub fn gemm_q4_0_geglu(
        &self,
        d_w_gate: &CudaBuffer<u8>,
        d_w_up: &CudaBuffer<u8>,
        d_x: &CudaBuffer<f32>,
        d_act: &mut CudaBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
        batch: usize,
    ) {
        assert_eq!(d_x.len(), n_cols * batch);
        assert_eq!(d_act.len(), n_rows * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_geglu_q4(
                d_w_gate.as_ptr(),
                d_w_up.as_ptr(),
                d_x.as_ptr(),
                d_act.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA fused Q4 GeGLU GEMM kernel failed");
        }
    }

    pub fn attention_prefill(
        &self,
        d_q: &CudaBuffer<f32>,
        d_k_cache: &CudaBuffer<u16>,
        d_v_cache: &CudaBuffer<u16>,
        d_out: &mut CudaBuffer<f32>,
        cache_start: usize,
        batch: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        sliding_window: Option<usize>,
        cache_capacity: usize,
        k_format: i32,
        v_format: i32,
    ) {
        let q_stride = d_q.len() / batch;
        assert!(q_stride >= n_heads * head_dim);
        assert_eq!(d_out.len(), batch * n_heads * head_dim);
        let window = sliding_window.map(|value| value as i32).unwrap_or(-1);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_attention(
                d_q.as_ptr(),
                d_k_cache.as_ptr(),
                d_v_cache.as_ptr(),
                d_out.as_mut_ptr(),
                cache_start as i32,
                batch as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                q_stride as i32,
                scale,
                window,
                cache_capacity as i32,
                k_format,
                v_format,
                self.stream,
            )
            .expect("Rust CUDA prefill attention failed");
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
            launch_rust_moe_topk(
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
                1,
                self.stream,
            )
            .expect("Rust CUDA MoE expert computation failed");
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
            launch_rust_moe_router(
                d_weights.as_ptr(),
                d_input.as_ptr(),
                d_logits.as_mut_ptr(),
                d_ids.as_mut_ptr(),
                d_probabilities.as_mut_ptr(),
                dim as i32,
                n_experts as i32,
                1,
                self.stream,
            )
            .expect("Rust CUDA MoE router failed");
        }
    }

    pub fn moe_topk_batch_q4_0(
        &self,
        gate_up: &CudaBuffer<u8>,
        down: &CudaBuffer<u8>,
        ids: &CudaBuffer<i32>,
        weights: &CudaBuffer<f32>,
        scales: Option<&CudaBuffer<f32>>,
        input: &CudaBuffer<f32>,
        act: &mut CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        dim: usize,
        exp_dim: usize,
        n_active: usize,
        batch: usize,
    ) {
        assert_eq!(ids.len(), batch * n_active);
        assert_eq!(weights.len(), batch * n_active);
        assert_eq!(input.len(), batch * dim);
        assert_eq!(act.len(), batch * n_active * exp_dim);
        assert_eq!(output.len(), batch * dim);
        let scale_ptr = scales.map(|value| value.as_ptr()).unwrap_or(ptr::null());
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_moe_topk(
                gate_up.as_ptr(),
                down.as_ptr(),
                ids.as_ptr(),
                weights.as_ptr(),
                scale_ptr,
                input.as_ptr(),
                act.as_mut_ptr(),
                output.as_mut_ptr(),
                dim as i32,
                exp_dim as i32,
                n_active as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA batched MoE expert computation failed");
        }
    }

    pub fn moe_router_batch(
        &self,
        weights: &CudaBuffer<f32>,
        input: &CudaBuffer<f32>,
        logits: &mut CudaBuffer<f32>,
        ids: &mut CudaBuffer<i32>,
        probabilities: &mut CudaBuffer<f32>,
        dim: usize,
        n_experts: usize,
        batch: usize,
    ) {
        assert_eq!(input.len(), dim * batch);
        assert_eq!(logits.len(), n_experts * batch);
        assert_eq!(ids.len(), 8 * batch);
        assert_eq!(probabilities.len(), 8 * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_moe_router(
                weights.as_ptr(),
                input.as_ptr(),
                logits.as_mut_ptr(),
                ids.as_mut_ptr(),
                probabilities.as_mut_ptr(),
                dim as i32,
                n_experts as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA batched MoE router failed");
        }
    }

    pub fn prepare_ffn(
        &self,
        hidden: &CudaBuffer<f32>,
        attn_proj: &CudaBuffer<f32>,
        post_attn_norm: &CudaBuffer<f32>,
        ffn_norm: &CudaBuffer<f32>,
        pre_ffw_norm_2: &CudaBuffer<f32>,
        router_scale: &CudaBuffer<f32>,
        attn_res: &mut CudaBuffer<f32>,
        shared_in: &mut CudaBuffer<f32>,
        moe_in: &mut CudaBuffer<f32>,
        router_in: &mut CudaBuffer<f32>,
        dim: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_prepare_ffn(
                hidden.as_ptr(),
                attn_proj.as_ptr(),
                post_attn_norm.as_ptr(),
                ffn_norm.as_ptr(),
                pre_ffw_norm_2.as_ptr(),
                router_scale.as_ptr(),
                attn_res.as_mut_ptr(),
                shared_in.as_mut_ptr(),
                moe_in.as_mut_ptr(),
                router_in.as_mut_ptr(),
                dim as i32,
                1,
                self.stream,
            )
            .expect("Rust CUDA FFN preparation failed");
        }
    }

    pub fn prepare_ffn_batch(
        &self,
        hidden: &CudaBuffer<f32>,
        attn: &CudaBuffer<f32>,
        pan: &CudaBuffer<f32>,
        ffn: &CudaBuffer<f32>,
        pfn: &CudaBuffer<f32>,
        router_scale: &CudaBuffer<f32>,
        attn_res: &mut CudaBuffer<f32>,
        shared: &mut CudaBuffer<f32>,
        moe: &mut CudaBuffer<f32>,
        router: &mut CudaBuffer<f32>,
        dim: usize,
        batch: usize,
    ) {
        assert_eq!(hidden.len(), dim * batch);
        assert_eq!(attn.len(), dim * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_prepare_ffn(
                hidden.as_ptr(),
                attn.as_ptr(),
                pan.as_ptr(),
                ffn.as_ptr(),
                pfn.as_ptr(),
                router_scale.as_ptr(),
                attn_res.as_mut_ptr(),
                shared.as_mut_ptr(),
                moe.as_mut_ptr(),
                router.as_mut_ptr(),
                dim as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA batched FFN preparation failed");
        }
    }

    pub fn finish_ffn(
        &self,
        attn_res: &CudaBuffer<f32>,
        dense: &mut CudaBuffer<f32>,
        moe: &mut CudaBuffer<f32>,
        post_ffw_norm_1: &CudaBuffer<f32>,
        post_ffw_norm_2: &CudaBuffer<f32>,
        post_ffw_norm: &CudaBuffer<f32>,
        hidden_out: &mut CudaBuffer<f32>,
        layer_scale: f32,
        dim: usize,
    ) {
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_finish_ffn(
                attn_res.as_ptr(),
                dense.as_mut_ptr(),
                moe.as_mut_ptr(),
                post_ffw_norm_1.as_ptr(),
                post_ffw_norm_2.as_ptr(),
                post_ffw_norm.as_ptr(),
                hidden_out.as_mut_ptr(),
                layer_scale,
                dim as i32,
                1,
                self.stream,
            )
            .expect("Rust CUDA FFN finalization failed");
        }
    }

    pub fn finish_ffn_batch(
        &self,
        attn_res: &CudaBuffer<f32>,
        dense: &mut CudaBuffer<f32>,
        moe: &mut CudaBuffer<f32>,
        p1: &CudaBuffer<f32>,
        p2: &CudaBuffer<f32>,
        pf: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        scale: f32,
        dim: usize,
        batch: usize,
    ) {
        assert_eq!(attn_res.len(), dim * batch);
        assert_eq!(output.len(), dim * batch);
        unsafe {
            cudaSetDevice(self.device_id);
            launch_rust_finish_ffn(
                attn_res.as_ptr(),
                dense.as_mut_ptr(),
                moe.as_mut_ptr(),
                p1.as_ptr(),
                p2.as_ptr(),
                pf.as_ptr(),
                output.as_mut_ptr(),
                scale,
                dim as i32,
                batch as i32,
                self.stream,
            )
            .expect("Rust CUDA batched FFN finalization failed");
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
            let max_partition = vocab_size.div_ceil(partitions);
            {
                let mut logits_arg = logits.as_ptr();
                let mut valid_arg = valid.as_ptr();
                let mut recent_arg = recent.as_ptr();
                let mut scores_arg = scores.as_mut_ptr();
                let mut ids_arg = ids.as_mut_ptr();
                let mut vocab_arg = vocab_size as i32;
                let mut recent_count = n_recent as i32;
                let mut generated = generated_count as i32;
                let mut k_arg = k as i32;
                let mut partition_arg = partitions as i32;
                let mut args = [
                    &mut logits_arg as *mut _ as *mut c_void,
                    &mut valid_arg as *mut _ as *mut c_void,
                    &mut recent_arg as *mut _ as *mut c_void,
                    &mut scores_arg as *mut _ as *mut c_void,
                    &mut ids_arg as *mut _ as *mut c_void,
                    &mut vocab_arg as *mut _ as *mut c_void,
                    &mut recent_count as *mut _ as *mut c_void,
                    &mut generated as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                    &mut partition_arg as *mut _ as *mut c_void,
                ];
                let kernel = if max_partition <= 2048 {
                    "rust_cuda_vocab_topk_f32"
                } else {
                    "rust_cuda_vocab_topk_generic_f32"
                };
                let threads = if max_partition <= 2048 { 256 } else { 1 };
                launch_rust_kernel(
                    kernel,
                    (partitions as u32, 1, 1),
                    (threads, 1, 1),
                    self.stream,
                    &mut args,
                )
                .expect("Rust CUDA vocabulary top-k failed");
            }
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
        if !CudaDevice::is_available() {
            return Ok(());
        }
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
        if !CudaDevice::is_available() {
            return Ok(());
        }
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
                launch_rust_add(
                    a.as_ptr(),
                    b.as_ptr(),
                    output.as_mut_ptr(),
                    64,
                    device.stream,
                )?;
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
                    (((e * 29 + i * 17) % 113) as f32 - 56.0) * 0.002 + e as f32 * 0.00001
                })
            })
            .collect();

        let mut expected: Vec<(f32, i32)> = (0..EXPERTS)
            .map(|e| {
                let dot = input
                    .iter()
                    .zip(&weights[e * DIM..(e + 1) * DIM])
                    .map(|(x, w)| x * w)
                    .sum();
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
        dev.moe_router(
            &d_weights,
            &d_input,
            &mut d_logits,
            &mut d_ids,
            &mut d_probs,
            DIM,
            EXPERTS,
        );
        dev.sync()?;
        let mut ids = [0i32; 8];
        let mut probs = [0.0f32; 8];
        d_ids.copy_to_host(&mut ids)?;
        d_probs.copy_to_host(&mut probs)?;

        for i in 0..8 {
            assert_eq!(ids[i], expected[i].1);
            let probability = (expected[i].0 - max).exp() / denom;
            assert!(
                (probs[i] - probability).abs() < 2e-5,
                "probability {i}: GPU {} CPU {probability}",
                probs[i]
            );
        }
        Ok(())
    }

    #[test]
    fn fused_ffn_residual_pipeline_matches_cpu() -> Result<()> {
        let _serial = CUDA_TEST_LOCK.lock().unwrap();
        if !CudaDevice::is_available() {
            return Ok(());
        }
        const DIM: usize = 257;
        let values = |seed: usize| -> Vec<f32> {
            (0..DIM)
                .map(|i| (((i * seed + 19) % 97) as f32 - 48.0) * 0.007)
                .collect()
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
            &d_hidden,
            &d_projection,
            &d_post_attn,
            &d_ffn_norm,
            &d_pre_moe,
            &d_router_scale,
            &mut d_attn_res,
            &mut d_shared,
            &mut d_moe_in,
            &mut d_router,
            DIM,
        );
        dev.sync()?;

        let rms = |x: &[f32]| {
            (x.iter().map(|v| v * v).sum::<f32>() / DIM as f32 + 1e-6)
                .sqrt()
                .recip()
        };
        let projection_inv = rms(&projection);
        let expected_res: Vec<f32> = (0..DIM)
            .map(|i| hidden[i] + projection[i] * projection_inv * post_attn[i])
            .collect();
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
            &d_attn_res,
            &mut d_dense,
            &mut d_moe,
            &d_post_1,
            &d_post_2,
            &d_post,
            &mut d_out,
            0.75,
            DIM,
        );
        dev.sync()?;
        let dense_inv = rms(&dense);
        let moe_inv = rms(&moe);
        let combined: Vec<f32> = (0..DIM)
            .map(|i| dense[i] * dense_inv * post_1[i] + moe[i] * moe_inv * post_2[i])
            .collect();
        let combined_inv = rms(&combined);
        let mut actual_out = vec![0.0; DIM];
        d_out.copy_to_host(&mut actual_out)?;
        for i in 0..DIM {
            let expected = (expected_res[i] + combined[i] * combined_inv * post[i]) * 0.75;
            assert!(
                (actual_out[i] - expected).abs() < 3e-5,
                "output {i}: GPU {} CPU {expected}",
                actual_out[i]
            );
        }
        Ok(())
    }
}
