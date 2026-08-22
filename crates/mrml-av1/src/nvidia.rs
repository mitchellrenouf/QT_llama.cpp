//! Runtime-loaded NVIDIA video codec backend.
//!
//! No SDK library is linked into the executable. The display driver's NVDEC,
//! NVENC, and CUDA libraries are loaded at runtime, which preserves the CPU
//! fallback and allows builds on machines without an NVIDIA SDK installation.

use core::ffi::{CStr, c_void};

#[cfg(unix)]
use mrml_linux::DynamicLibrary;
#[cfg(windows)]
use mrml_windows::DynamicLibrary;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvidiaCapabilities {
    pub cuda_driver: bool,
    pub nvdec: bool,
    pub nvenc: bool,
    /// Maximum NVENC API version reported by the installed display driver.
    /// The major version occupies the low 8 bits and the minor version the
    /// high 8 bits, matching NVIDIA's runtime ABI.
    pub nvenc_api_version: Option<u32>,
}

/// Owns loaded driver libraries for the lifetime of any function pointers
/// obtained from them. Session-specific ABI structures live in decoder and
/// encoder modules rather than leaking SDK types into the portable API.
#[allow(dead_code)] // Library ownership keeps resolved session symbols valid.
pub struct NvidiaBackend {
    cuda: Option<DynamicLibrary>,
    nvdec: Option<DynamicLibrary>,
    nvenc: Option<DynamicLibrary>,
    capabilities: NvidiaCapabilities,
}

impl NvidiaBackend {
    /// Probe the installed display driver. Absence is a normal result and does
    /// not prevent use of the software codec.
    pub fn probe() -> Self {
        let cuda = open_first(CUDA_LIBRARIES);
        let nvdec = open_first(NVDEC_LIBRARIES);
        let nvenc = open_first(NVENC_LIBRARIES);
        let cuda_driver = initialize_cuda(&cuda);
        let decode_symbols = [
            c"cuvidGetDecoderCaps" as &CStr,
            c"cuvidCreateDecoder",
            c"cuvidDecodePicture",
            c"cuvidMapVideoFrame64",
            c"cuvidUnmapVideoFrame64",
            c"cuvidDestroyDecoder",
        ];
        let encode_symbols = [
            c"NvEncodeAPIGetMaxSupportedVersion" as &CStr,
            c"NvEncodeAPICreateInstance",
        ];
        let nvenc_api_version = query_nvenc_api_version(&nvenc);
        let capabilities = NvidiaCapabilities {
            cuda_driver,
            nvdec: cuda_driver && has_all(&nvdec, &decode_symbols),
            nvenc: cuda_driver && nvenc_api_version.is_some() && has_all(&nvenc, &encode_symbols),
            nvenc_api_version,
        };
        Self {
            cuda,
            nvdec,
            nvenc,
            capabilities,
        }
    }

    pub fn capabilities(&self) -> NvidiaCapabilities {
        self.capabilities
    }

    #[allow(dead_code)]
    pub(crate) fn nvdec_symbol(&self, name: &CStr) -> Option<*mut c_void> {
        self.nvdec.as_ref()?.symbol(name)
    }

    #[allow(dead_code)]
    pub(crate) fn nvenc_symbol(&self, name: &CStr) -> Option<*mut c_void> {
        self.nvenc.as_ref()?.symbol(name)
    }

    #[allow(dead_code)]
    pub(crate) fn cuda_symbol(&self, name: &CStr) -> Option<*mut c_void> {
        self.cuda.as_ref()?.symbol(name)
    }
}

fn initialize_cuda(library: &Option<DynamicLibrary>) -> bool {
    library
        .as_ref()
        .is_some_and(DynamicLibrary::initialize_cuda_driver)
}

fn query_nvenc_api_version(library: &Option<DynamicLibrary>) -> Option<u32> {
    library.as_ref()?.nvenc_max_supported_version()
}

fn has(library: &Option<DynamicLibrary>, symbol: &CStr) -> bool {
    library
        .as_ref()
        .and_then(|lib| lib.symbol(symbol))
        .is_some()
}

fn has_all(library: &Option<DynamicLibrary>, symbols: &[&CStr]) -> bool {
    symbols.iter().all(|symbol| has(library, symbol))
}

fn open_first(names: &[&CStr]) -> Option<DynamicLibrary> {
    names.iter().find_map(|name| DynamicLibrary::open(name))
}

#[cfg(windows)]
const CUDA_LIBRARIES: &[&CStr] = &[c"nvcuda.dll"];
#[cfg(windows)]
const NVDEC_LIBRARIES: &[&CStr] = &[c"nvcuvid.dll"];
#[cfg(windows)]
const NVENC_LIBRARIES: &[&CStr] = &[c"nvEncodeAPI64.dll"];

#[cfg(unix)]
const CUDA_LIBRARIES: &[&CStr] = &[c"libcuda.so.1", c"libcuda.so"];
#[cfg(unix)]
const NVDEC_LIBRARIES: &[&CStr] = &[c"libnvcuvid.so.1", c"libnvcuvid.so"];
#[cfg(unix)]
const NVENC_LIBRARIES: &[&CStr] = &[c"libnvidia-encode.so.1", c"libnvidia-encode.so"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probing_is_non_fatal_without_nvidia_driver() {
        let backend = NvidiaBackend::probe();
        let caps = backend.capabilities();
        assert!(!caps.nvdec || caps.cuda_driver);
        assert!(!caps.nvenc || caps.cuda_driver);
        assert!(!caps.nvenc || caps.nvenc_api_version.is_some());
    }

    #[test]
    fn absent_library_bootstrap_is_non_fatal() {
        assert!(!initialize_cuda(&None));
        assert_eq!(query_nvenc_api_version(&None), None);
    }
}
