#![no_std]

mod abi;

pub use abi::{
    decode_run_page, KvmError, KvmMemoryRegion, KVM_API_VERSION, KVM_MEMORY_REGION_BYTES,
    MAX_KVM_MEMORY_SLOTS, MRML_KVM_HYPERCALL,
};
