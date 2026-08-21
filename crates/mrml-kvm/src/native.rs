use core::ffi::{c_char, c_int, c_ulong, c_void};
use core::ptr::NonNull;
use core::slice;
use mrml_kernel::VmExit;

use crate::{decode_run_page, KvmError, KvmMemoryRegion, KVM_API_VERSION};

const O_RDWR: c_int = 2;
const O_CLOEXEC: c_int = 0x80000;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
const KVM_GET_API_VERSION: c_ulong = 0xae00;
const KVM_CREATE_VM: c_ulong = 0xae01;
const KVM_GET_VCPU_MMAP_SIZE: c_ulong = 0xae04;
const KVM_CREATE_VCPU: c_ulong = 0xae41;
const KVM_SET_USER_MEMORY_REGION: c_ulong = 0x4020_ae46;
const KVM_RUN: c_ulong = 0xae80;
const MIN_RUN_BYTES: usize = 88;
const MAX_RUN_BYTES: usize = 1024 * 1024;

#[link(name = "c")]
unsafe extern "C" {
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(file: c_int) -> c_int;
    fn ioctl(file: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        file: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
}

struct OwnedFd(c_int);

impl Drop for OwnedFd {
    fn drop(&mut self) {
        unsafe {
            close(self.0);
        }
    }
}

pub struct KvmSystem {
    file: OwnedFd,
    run_bytes: usize,
}

impl KvmSystem {
    pub fn open() -> Result<Self, KvmError> {
        let file = unsafe { open(b"/dev/kvm\0".as_ptr().cast(), O_RDWR | O_CLOEXEC) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        let file = OwnedFd(file);
        if unsafe { ioctl(file.0, KVM_GET_API_VERSION) } != KVM_API_VERSION {
            return Err(KvmError::ApiVersion);
        }
        let run_bytes = unsafe { ioctl(file.0, KVM_GET_VCPU_MMAP_SIZE) };
        if run_bytes < 0 {
            return Err(KvmError::SystemCall);
        }
        if run_bytes < MIN_RUN_BYTES as c_int || run_bytes as usize > MAX_RUN_BYTES {
            return Err(KvmError::InvalidRunSize(run_bytes));
        }
        Ok(Self {
            file,
            run_bytes: run_bytes as usize,
        })
    }

    pub fn create_vm(&self) -> Result<KvmVm, KvmError> {
        let file = unsafe { ioctl(self.file.0, KVM_CREATE_VM, 0 as c_ulong) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(KvmVm {
            file: OwnedFd(file),
            run_bytes: self.run_bytes,
        })
    }
}

pub struct KvmVm {
    file: OwnedFd,
    run_bytes: usize,
}

impl KvmVm {
    pub fn register_memory(&self, region: KvmMemoryRegion) -> Result<(), KvmError> {
        let encoded = region.encode();
        if unsafe { ioctl(self.file.0, KVM_SET_USER_MEMORY_REGION, encoded.as_ptr()) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }

    pub fn create_vcpu(&self, id: u32) -> Result<KvmVcpu, KvmError> {
        let file = unsafe { ioctl(self.file.0, KVM_CREATE_VCPU, id as c_ulong) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        let file = OwnedFd(file);
        let run = unsafe {
            mmap(
                core::ptr::null_mut(),
                self.run_bytes,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.0,
                0,
            )
        };
        if run == MAP_FAILED {
            return Err(KvmError::SystemCall);
        }
        let run = NonNull::new(run.cast()).ok_or(KvmError::SystemCall)?;
        Ok(KvmVcpu {
            file,
            run,
            run_bytes: self.run_bytes,
        })
    }
}

pub struct KvmVcpu {
    file: OwnedFd,
    run: NonNull<u8>,
    run_bytes: usize,
}

impl KvmVcpu {
    pub fn run(&mut self) -> Result<VmExit, KvmError> {
        if unsafe { ioctl(self.file.0, KVM_RUN) } < 0 {
            return Err(KvmError::SystemCall);
        }
        let bytes = unsafe { slice::from_raw_parts(self.run.as_ptr(), self.run_bytes) };
        decode_run_page(bytes)
    }

    pub const fn run_mapping_bytes(&self) -> usize {
        self.run_bytes
    }
}

impl Drop for KvmVcpu {
    fn drop(&mut self) {
        unsafe {
            munmap(self.run.as_ptr().cast(), self.run_bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers_match_x86_64_kvm_uapi() {
        assert_eq!(KVM_GET_API_VERSION, 0xae00);
        assert_eq!(KVM_CREATE_VM, 0xae01);
        assert_eq!(KVM_SET_USER_MEMORY_REGION, 0x4020_ae46);
        assert_eq!(KVM_CREATE_VCPU, 0xae41);
        assert_eq!(KVM_RUN, 0xae80);
    }

    #[test]
    fn live_api_version_when_kvm_is_available() {
        match KvmSystem::open() {
            Ok(system) => assert!((MIN_RUN_BYTES..=MAX_RUN_BYTES).contains(&system.run_bytes)),
            Err(KvmError::SystemCall) => {}
            Err(error) => panic!("available KVM failed validation: {error:?}"),
        }
    }
}
