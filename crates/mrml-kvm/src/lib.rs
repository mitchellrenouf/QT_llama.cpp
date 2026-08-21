#![no_std]

mod abi;
#[cfg(target_os = "linux")]
mod native;

pub use abi::{
    KVM_API_VERSION, KVM_MEMORY_REGION_BYTES, KvmError, KvmMemoryRegion, MAX_KVM_MEMORY_SLOTS,
    MRML_KVM_HYPERCALL, decode_run_page,
};
#[cfg(target_os = "linux")]
pub use native::{
    KvmGuestMemory, KvmLaunchLayout, KvmLoadedHandoff, KvmLoadedImage, KvmPageTableStore,
    KvmSystem, PreparedKvmGuest, map_loaded_handoff, map_loaded_pe,
};
