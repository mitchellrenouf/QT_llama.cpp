#![no_std]

mod abi;
#[cfg(target_os = "linux")]
mod native;

pub use abi::{
    decode_run_page, KvmError, KvmMemoryRegion, KVM_API_VERSION, KVM_MEMORY_REGION_BYTES,
    MAX_KVM_MEMORY_SLOTS, MRML_KVM_HYPERCALL,
};
#[cfg(target_os = "linux")]
pub use native::{KvmBackend, KvmGuestMemory, KvmSystem, KvmVcpu, KvmVm};
