use core::ffi::{c_char, c_int, c_ulong, c_void};
use core::ptr::NonNull;
use core::slice;
use mrml_kernel::{VmBackend, VmExit, PAGE_SIZE};

use crate::{decode_run_page, KvmError, KvmMemoryRegion, KVM_API_VERSION};

const O_RDWR: c_int = 2;
const O_CLOEXEC: c_int = 0x80000;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
const KVM_GET_API_VERSION: c_ulong = 0xae00;
const KVM_CREATE_VM: c_ulong = 0xae01;
const KVM_GET_VCPU_MMAP_SIZE: c_ulong = 0xae04;
const KVM_CREATE_VCPU: c_ulong = 0xae41;
const KVM_SET_USER_MEMORY_REGION: c_ulong = 0x4020_ae46;
const KVM_RUN: c_ulong = 0xae80;
const KVM_INTERRUPT: c_ulong = 0x4004_ae86;
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

    pub fn create_backend<const N: usize>(&self, vcpu_id: u32) -> Result<KvmBackend<N>, KvmError> {
        let vm = self.create_vm()?;
        let vcpu = vm.create_vcpu(vcpu_id)?;
        Ok(KvmBackend {
            vcpu,
            memory: KvmGuestMemory::new(),
            vm,
            vcpu_id,
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

struct HostRegion {
    guest_address: u64,
    bytes: usize,
    host: NonNull<u8>,
    readonly: bool,
}

impl HostRegion {
    fn allocate(guest_address: u64, bytes: usize, readonly: bool) -> Result<Self, KvmError> {
        let bytes_u64 = u64::try_from(bytes).map_err(|_| KvmError::MemoryOverflow)?;
        KvmMemoryRegion::new(0, guest_address, bytes_u64, PAGE_SIZE, readonly)?;
        let host = unsafe {
            mmap(
                core::ptr::null_mut(),
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if host == MAP_FAILED {
            return Err(KvmError::SystemCall);
        }
        Ok(Self {
            guest_address,
            bytes,
            host: NonNull::new(host.cast()).ok_or(KvmError::SystemCall)?,
            readonly,
        })
    }

    fn end(&self) -> u64 {
        self.guest_address + self.bytes as u64
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        unsafe {
            munmap(self.host.as_ptr().cast(), self.bytes);
        }
    }
}

pub struct KvmGuestMemory<const N: usize> {
    regions: [Option<HostRegion>; N],
    count: usize,
}

impl<const N: usize> KvmGuestMemory<N> {
    pub const fn new() -> Self {
        Self {
            regions: [const { None }; N],
            count: 0,
        }
    }

    fn allocate(
        &mut self,
        guest_address: u64,
        bytes: usize,
        readonly: bool,
    ) -> Result<(u32, u64), KvmError> {
        if self.count == N || self.count >= crate::MAX_KVM_MEMORY_SLOTS as usize {
            return Err(KvmError::MemoryTableFull);
        }
        let region = HostRegion::allocate(guest_address, bytes, readonly)?;
        if self.regions[..self.count]
            .iter()
            .flatten()
            .any(|existing| guest_address < existing.end() && existing.guest_address < region.end())
        {
            return Err(KvmError::MemoryOverlap);
        }
        let slot = self.count as u32;
        let host = region.host.as_ptr() as u64;
        self.regions[self.count] = Some(region);
        self.count += 1;
        Ok((slot, host))
    }

    fn rollback_last(&mut self) {
        if self.count != 0 {
            self.count -= 1;
            self.regions[self.count] = None;
        }
    }

    pub fn read(&self, guest_address: u64, output: &mut [u8]) -> Result<(), KvmError> {
        let (region, offset) = self.locate(guest_address, output.len())?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                region.host.as_ptr().add(offset),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        Ok(())
    }

    pub fn write(&mut self, guest_address: u64, input: &[u8]) -> Result<(), KvmError> {
        let (region, offset) = self.locate(guest_address, input.len())?;
        if region.readonly {
            return Err(KvmError::ReadOnlyMemory);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                region.host.as_ptr().add(offset),
                input.len(),
            )
        };
        Ok(())
    }

    fn locate(&self, guest_address: u64, bytes: usize) -> Result<(&HostRegion, usize), KvmError> {
        if bytes == 0 {
            return Err(KvmError::UnmappedMemory);
        }
        let end = guest_address
            .checked_add(bytes as u64)
            .ok_or(KvmError::MemoryOverflow)?;
        let region = self.regions[..self.count]
            .iter()
            .flatten()
            .find(|region| guest_address >= region.guest_address && end <= region.end())
            .ok_or(KvmError::UnmappedMemory)?;
        Ok((region, (guest_address - region.guest_address) as usize))
    }
}

impl<const N: usize> Default for KvmGuestMemory<N> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct KvmBackend<const N: usize> {
    vcpu: KvmVcpu,
    memory: KvmGuestMemory<N>,
    vm: KvmVm,
    vcpu_id: u32,
}

impl<const N: usize> KvmBackend<N> {
    pub fn map_memory(
        &mut self,
        guest_address: u64,
        bytes: usize,
        readonly: bool,
    ) -> Result<(), KvmError> {
        let (slot, host) = self.memory.allocate(guest_address, bytes, readonly)?;
        let region = KvmMemoryRegion::new(slot, guest_address, bytes as u64, host, readonly)?;
        if let Err(error) = self.vm.register_memory(region) {
            self.memory.rollback_last();
            return Err(error);
        }
        Ok(())
    }

    pub fn guest_memory(&self) -> &KvmGuestMemory<N> {
        &self.memory
    }
}

impl<const N: usize> VmBackend for KvmBackend<N> {
    type Error = KvmError;

    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        if vcpu != self.vcpu_id {
            return Err(KvmError::InvalidVcpu);
        }
        self.vcpu.run()
    }

    fn read_guest(&self, guest_address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.memory.read(guest_address, output)
    }

    fn write_guest(&mut self, guest_address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.memory.write(guest_address, input)
    }

    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        if vcpu != self.vcpu_id {
            return Err(KvmError::InvalidVcpu);
        }
        let interrupt = u32::from(vector);
        if unsafe { ioctl(self.vcpu.file.0, KVM_INTERRUPT, &interrupt) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
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

    #[test]
    fn owned_guest_memory_is_bounded_and_readonly() {
        let mut memory = KvmGuestMemory::<2>::new();
        memory.allocate(0x1000, 0x2000, false).unwrap();
        memory.write(0x1fff, &[1, 2]).unwrap();
        let mut output = [0u8; 2];
        memory.read(0x1fff, &mut output).unwrap();
        assert_eq!(output, [1, 2]);
        assert_eq!(
            memory.read(0x2fff, &mut output),
            Err(KvmError::UnmappedMemory)
        );
        assert_eq!(
            memory.allocate(0x2000, 0x1000, false),
            Err(KvmError::MemoryOverlap)
        );
        memory.allocate(0x4000, 0x1000, true).unwrap();
        assert_eq!(memory.write(0x4000, &[1]), Err(KvmError::ReadOnlyMemory));
    }
}
