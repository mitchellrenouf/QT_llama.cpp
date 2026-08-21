use core::ffi::{c_char, c_void};
use core::ptr::NonNull;
use core::slice;

use mrml_kernel::VmExit;

use crate::{GuestRange, WHP_EXIT_CONTEXT_BYTES, WhpError, decode_exit_context};

mod launch;
pub use launch::{PreparedWhpGuest, WhpLaunchLayout};

type Handle = *mut c_void;
type Hresult = i32;
type GetCapability = unsafe extern "system" fn(u32, *mut c_void, u32, *mut u32) -> Hresult;
type CreatePartition = unsafe extern "system" fn(*mut Handle) -> Hresult;
type DeletePartition = unsafe extern "system" fn(Handle) -> Hresult;
type SetPartitionProperty = unsafe extern "system" fn(Handle, u32, *const c_void, u32) -> Hresult;
type SetupPartition = unsafe extern "system" fn(Handle) -> Hresult;
type MapGpaRange = unsafe extern "system" fn(Handle, *mut c_void, u64, u64, u32) -> Hresult;
type UnmapGpaRange = unsafe extern "system" fn(Handle, u64, u64) -> Hresult;
type CreateVirtualProcessor = unsafe extern "system" fn(Handle, u32, u32) -> Hresult;
type DeleteVirtualProcessor = unsafe extern "system" fn(Handle, u32) -> Hresult;
type RunVirtualProcessor = unsafe extern "system" fn(Handle, u32, *mut c_void, u32) -> Hresult;
type SetVirtualProcessorRegisters =
    unsafe extern "system" fn(Handle, u32, *const u32, u32, *const RegisterValue) -> Hresult;
type RequestInterrupt = unsafe extern "system" fn(Handle, *const InterruptControl, u32) -> Hresult;

const PROCESSOR_COUNT: u32 = 0x1fff;
const LOCAL_APIC_EMULATION_MODE: u32 = 0x1005;
const XAPIC_EMULATION: u32 = 1;
const CAPABILITY_HYPERVISOR_PRESENT: u32 = 0;
const MEM_COMMIT_RESERVE: u32 = 0x3000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 4;
const MAX_MAPPINGS: usize = 32;
const REG_RCX: u32 = 1;
const REG_RDX: u32 = 2;
const REG_RSP: u32 = 4;
const REG_RIP: u32 = 0x10;
const REG_RFLAGS: u32 = 0x11;
const REG_ES: u32 = 0x12;
const REG_CS: u32 = 0x13;
const REG_SS: u32 = 0x14;
const REG_DS: u32 = 0x15;
const REG_CR0: u32 = 0x1000;
const REG_CR3: u32 = 0x1002;
const REG_CR4: u32 = 0x1003;
const REG_EFER: u32 = 0x2001;
const CR0_LONG_MODE: u64 = (1 << 0) | (1 << 16) | (1 << 31);
const CR4_PAE: u64 = 1 << 5;
const EFER_LONG_MODE_NX: u64 = (1 << 8) | (1 << 11);

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: Handle) -> i32;
    fn VirtualAlloc(
        address: *mut c_void,
        size: usize,
        allocation: u32,
        protection: u32,
    ) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
}

struct Library(Handle);

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.0);
        }
    }
}

#[derive(Clone, Copy)]
struct Api {
    get_capability: GetCapability,
    create_partition: CreatePartition,
    delete_partition: DeletePartition,
    set_partition_property: SetPartitionProperty,
    setup_partition: SetupPartition,
    map_gpa_range: MapGpaRange,
    unmap_gpa_range: UnmapGpaRange,
    create_vp: CreateVirtualProcessor,
    delete_vp: DeleteVirtualProcessor,
    run_vp: RunVirtualProcessor,
    set_registers: SetVirtualProcessorRegisters,
    request_interrupt: RequestInterrupt,
}

pub struct WhpSystem {
    _library: Library,
    api: Api,
}

impl WhpSystem {
    pub fn open() -> Result<Self, WhpError> {
        let module = unsafe { LoadLibraryW(wide_name().as_ptr()) };
        if module.is_null() {
            return Err(WhpError::PlatformUnavailable);
        }
        let library = Library(module);
        let api = unsafe {
            Api {
                get_capability: resolve(module, b"WHvGetCapability\0")?,
                create_partition: resolve(module, b"WHvCreatePartition\0")?,
                delete_partition: resolve(module, b"WHvDeletePartition\0")?,
                set_partition_property: resolve(module, b"WHvSetPartitionProperty\0")?,
                setup_partition: resolve(module, b"WHvSetupPartition\0")?,
                map_gpa_range: resolve(module, b"WHvMapGpaRange\0")?,
                unmap_gpa_range: resolve(module, b"WHvUnmapGpaRange\0")?,
                create_vp: resolve(module, b"WHvCreateVirtualProcessor\0")?,
                delete_vp: resolve(module, b"WHvDeleteVirtualProcessor\0")?,
                run_vp: resolve(module, b"WHvRunVirtualProcessor\0")?,
                set_registers: resolve(module, b"WHvSetVirtualProcessorRegisters\0")?,
                request_interrupt: resolve(module, b"WHvRequestInterrupt\0")?,
            }
        };
        Ok(Self {
            _library: library,
            api,
        })
    }

    pub fn prepare_partition(&self) -> Result<PreparedWhpPartition<'_>, WhpError> {
        if !self.hypervisor_present()? {
            return Err(WhpError::PlatformUnavailable);
        }
        let mut partition = core::ptr::null_mut();
        check(unsafe { (self.api.create_partition)(&mut partition) })?;
        let partition = NonNull::new(partition).ok_or(WhpError::PlatformUnavailable)?;
        let mut guard = PartitionGuard {
            partition,
            api: self.api,
            vp: false,
        };
        let processor_count = 1u32;
        check(unsafe {
            (self.api.set_partition_property)(
                partition.as_ptr(),
                PROCESSOR_COUNT,
                (&processor_count as *const u32).cast(),
                4,
            )
        })?;
        check(unsafe {
            (self.api.set_partition_property)(
                partition.as_ptr(),
                LOCAL_APIC_EMULATION_MODE,
                (&XAPIC_EMULATION as *const u32).cast(),
                4,
            )
        })?;
        check(unsafe { (self.api.setup_partition)(partition.as_ptr()) })?;
        check(unsafe { (self.api.create_vp)(partition.as_ptr(), 0, 0) })?;
        guard.vp = true;
        let prepared = PreparedWhpPartition {
            partition,
            api: self.api,
            mappings: [const { None }; MAX_MAPPINGS],
            _system: core::marker::PhantomData,
        };
        core::mem::forget(guard);
        Ok(prepared)
    }

    pub fn hypervisor_present(&self) -> Result<bool, WhpError> {
        let mut present = 0u32;
        let mut written = 0u32;
        check(unsafe {
            (self.api.get_capability)(
                CAPABILITY_HYPERVISOR_PRESENT,
                (&mut present as *mut u32).cast(),
                4,
                &mut written,
            )
        })?;
        if written != 4 || present > 1 {
            return Err(WhpError::PlatformUnavailable);
        }
        Ok(present == 1)
    }
}

pub struct PreparedWhpPartition<'system> {
    partition: NonNull<c_void>,
    api: Api,
    mappings: [Option<OwnedMapping>; MAX_MAPPINGS],
    _system: core::marker::PhantomData<&'system WhpSystem>,
}

impl PreparedWhpPartition<'_> {
    pub(crate) fn map_zeroed(&mut self, range: GuestRange) -> Result<usize, WhpError> {
        self.map_initialized(range, &[])
    }

    pub fn map_initialized(
        &mut self,
        range: GuestRange,
        contents: &[u8],
    ) -> Result<usize, WhpError> {
        let size = usize::try_from(range.size()).map_err(|_| WhpError::MemoryOverflow)?;
        if contents.len() > size {
            return Err(WhpError::MemoryOverflow);
        }
        let slot = self
            .mappings
            .iter()
            .position(Option::is_none)
            .ok_or(WhpError::MemoryTableFull)?;
        if self
            .mappings
            .iter()
            .flatten()
            .any(|entry| overlaps(entry.range, range))
        {
            return Err(WhpError::MemoryOverlap);
        }
        let address = NonNull::new(unsafe {
            VirtualAlloc(
                core::ptr::null_mut(),
                size,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            )
        })
        .ok_or(WhpError::PlatformUnavailable)?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                contents.as_ptr(),
                address.as_ptr().cast(),
                contents.len(),
            );
        }
        let mapping = OwnedMapping { address, range };
        check(unsafe {
            (self.api.map_gpa_range)(
                self.partition.as_ptr(),
                address.as_ptr(),
                range.guest_address(),
                range.size(),
                range.permissions().bits(),
            )
        })?;
        self.mappings[slot] = Some(mapping);
        Ok(slot)
    }

    pub fn read_mapping(
        &self,
        id: usize,
        offset: usize,
        output: &mut [u8],
    ) -> Result<(), WhpError> {
        let mapping = self
            .mappings
            .get(id)
            .and_then(Option::as_ref)
            .ok_or(WhpError::MalformedExit)?;
        let end = offset
            .checked_add(output.len())
            .ok_or(WhpError::MemoryOverflow)?;
        if end > mapping.range.size() as usize {
            return Err(WhpError::MemoryOverflow);
        }
        let source = unsafe {
            slice::from_raw_parts(
                mapping.address.as_ptr().cast::<u8>().add(offset),
                output.len(),
            )
        };
        output.copy_from_slice(source);
        Ok(())
    }

    pub(crate) fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), WhpError> {
        let (mapping, offset) = self.locate(address, output.len())?;
        let source = unsafe {
            slice::from_raw_parts(
                mapping.address.as_ptr().cast::<u8>().add(offset),
                output.len(),
            )
        };
        output.copy_from_slice(source);
        Ok(())
    }

    pub(crate) fn write_guest(&mut self, address: u64, input: &[u8]) -> Result<(), WhpError> {
        let (mapping, offset) = self.locate(address, input.len())?;
        if mapping.range.permissions().bits() & 2 == 0 {
            return Err(WhpError::ReadOnlyMemory);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                mapping.address.as_ptr().cast::<u8>().add(offset),
                input.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn mutable_guest(
        &mut self,
        address: u64,
        bytes: usize,
    ) -> Result<&mut [u8], WhpError> {
        let (mapping, offset) = self.locate(address, bytes)?;
        if mapping.range.permissions().bits() & 2 == 0 {
            return Err(WhpError::ReadOnlyMemory);
        }
        Ok(unsafe {
            slice::from_raw_parts_mut(mapping.address.as_ptr().cast::<u8>().add(offset), bytes)
        })
    }

    fn locate(&self, address: u64, bytes: usize) -> Result<(&OwnedMapping, usize), WhpError> {
        if bytes == 0 {
            return Err(WhpError::UnmappedMemory);
        }
        let end = address
            .checked_add(bytes as u64)
            .ok_or(WhpError::MemoryOverflow)?;
        let mapping = self
            .mappings
            .iter()
            .flatten()
            .find(|mapping| {
                address >= mapping.range.guest_address()
                    && end <= mapping.range.guest_address() + mapping.range.size()
            })
            .ok_or(WhpError::UnmappedMemory)?;
        Ok((mapping, (address - mapping.range.guest_address()) as usize))
    }

    pub fn run(&mut self) -> Result<VmExit, WhpError> {
        let mut context = [0u8; WHP_EXIT_CONTEXT_BYTES];
        check(unsafe {
            (self.api.run_vp)(
                self.partition.as_ptr(),
                0,
                context.as_mut_ptr().cast(),
                context.len() as u32,
            )
        })?;
        decode_exit_context(&context)
    }

    pub(crate) fn inject_interrupt(&mut self, vector: u8) -> Result<(), WhpError> {
        if vector < 32 || vector == u8::MAX {
            return Err(WhpError::InvalidInterrupt);
        }
        let control = InterruptControl {
            options: 0,
            destination: 0,
            vector: u32::from(vector),
        };
        check(unsafe {
            (self.api.request_interrupt)(
                self.partition.as_ptr(),
                &control,
                core::mem::size_of::<InterruptControl>() as u32,
            )
        })
    }

    pub fn configure_long_mode(
        &mut self,
        entry: u64,
        stack_pointer: u64,
        page_table: u64,
        handoff: u64,
        handoff_bytes: u64,
    ) -> Result<(), WhpError> {
        if !canonical(entry)
            || !canonical(stack_pointer)
            || !canonical(handoff)
            || !page_table.is_multiple_of(4096)
            || stack_pointer & 0xf != 8
            || handoff_bytes == 0
            || handoff.checked_add(handoff_bytes).is_none()
        {
            return Err(WhpError::InvalidRegisterState);
        }
        let names = [
            REG_RIP, REG_RSP, REG_RFLAGS, REG_RCX, REG_RDX, REG_CR0, REG_CR3, REG_CR4, REG_EFER,
            REG_CS, REG_SS, REG_DS, REG_ES,
        ];
        let mut values = [RegisterValue::zero(); 13];
        values[0] = RegisterValue::scalar(entry);
        values[1] = RegisterValue::scalar(stack_pointer);
        values[2] = RegisterValue::scalar(2);
        values[3] = RegisterValue::scalar(handoff);
        values[4] = RegisterValue::scalar(handoff_bytes);
        values[5] = RegisterValue::scalar(CR0_LONG_MODE);
        values[6] = RegisterValue::scalar(page_table);
        values[7] = RegisterValue::scalar(CR4_PAE);
        values[8] = RegisterValue::scalar(EFER_LONG_MODE_NX);
        values[9] = RegisterValue::segment(0x8, 0xa09b);
        for value in &mut values[10..] {
            *value = RegisterValue::segment(0x10, 0x8093);
        }
        check(unsafe {
            (self.api.set_registers)(
                self.partition.as_ptr(),
                0,
                names.as_ptr(),
                names.len() as u32,
                values.as_ptr(),
            )
        })
    }
}

impl Drop for PreparedWhpPartition<'_> {
    fn drop(&mut self) {
        for mapping in &mut self.mappings {
            if let Some(value) = mapping.take() {
                let _ = unsafe {
                    (self.api.unmap_gpa_range)(
                        self.partition.as_ptr(),
                        value.range.guest_address(),
                        value.range.size(),
                    )
                };
                drop(value);
            }
        }
        let _ = unsafe { (self.api.delete_vp)(self.partition.as_ptr(), 0) };
        let _ = unsafe { (self.api.delete_partition)(self.partition.as_ptr()) };
    }
}

struct PartitionGuard {
    partition: NonNull<c_void>,
    api: Api,
    vp: bool,
}
impl Drop for PartitionGuard {
    fn drop(&mut self) {
        if self.vp {
            let _ = unsafe { (self.api.delete_vp)(self.partition.as_ptr(), 0) };
        }
        let _ = unsafe { (self.api.delete_partition)(self.partition.as_ptr()) };
    }
}

struct OwnedMapping {
    address: NonNull<c_void>,
    range: GuestRange,
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct RegisterValue {
    low: u64,
    high: u64,
}

#[repr(C)]
struct InterruptControl {
    options: u64,
    destination: u32,
    vector: u32,
}

impl RegisterValue {
    const fn zero() -> Self {
        Self { low: 0, high: 0 }
    }

    const fn scalar(value: u64) -> Self {
        Self {
            low: value,
            high: 0,
        }
    }

    const fn segment(selector: u16, attributes: u16) -> Self {
        // WHV_X64_SEGMENT_REGISTER: Base, Limit, Selector, Attributes.
        Self {
            low: 0,
            high: 0xfffff | ((selector as u64) << 32) | ((attributes as u64) << 48),
        }
    }
}
impl Drop for OwnedMapping {
    fn drop(&mut self) {
        let _ = unsafe { VirtualFree(self.address.as_ptr(), 0, MEM_RELEASE) };
    }
}

fn overlaps(left: GuestRange, right: GuestRange) -> bool {
    left.guest_address() < right.guest_address() + right.size()
        && right.guest_address() < left.guest_address() + left.size()
}

const fn canonical(address: u64) -> bool {
    let top = address >> 48;
    if address & (1 << 47) == 0 {
        top == 0
    } else {
        top == 0xffff
    }
}

fn check(result: Hresult) -> Result<(), WhpError> {
    if result < 0 {
        Err(WhpError::SystemCall(result))
    } else {
        Ok(())
    }
}

unsafe fn resolve<T: Copy>(module: Handle, name: &[u8]) -> Result<T, WhpError> {
    let pointer = unsafe { GetProcAddress(module, name.as_ptr().cast()) };
    if pointer.is_null() {
        return Err(WhpError::MissingPlatformFunction);
    }
    if core::mem::size_of::<T>() != core::mem::size_of::<*mut c_void>() {
        return Err(WhpError::MissingPlatformFunction);
    }
    Ok(unsafe { core::mem::transmute_copy(&pointer) })
}

const fn wide_name() -> [u16; 18] {
    [
        87, 105, 110, 72, 118, 80, 108, 97, 116, 102, 111, 114, 109, 46, 100, 108, 108, 0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MapPermissions;

    #[test]
    fn register_values_match_whp_union_layout() {
        assert_eq!(core::mem::size_of::<RegisterValue>(), 16);
        assert_eq!(core::mem::align_of::<RegisterValue>(), 16);
        let code = RegisterValue::segment(8, 0xa09b);
        assert_eq!(code.low, 0);
        assert_eq!(code.high, 0xa09b_0008_000f_ffff);
        assert_eq!(core::mem::size_of::<InterruptControl>(), 16);
    }

    #[test]
    fn only_x64_canonical_addresses_are_accepted() {
        assert!(canonical(0x0000_7fff_ffff_ffff));
        assert!(canonical(0xffff_8000_0000_0000));
        assert!(!canonical(0x0000_8000_0000_0000));
        assert!(!canonical(0xffff_0000_0000_0000));
    }

    #[test]
    fn installed_platform_exports_are_complete() {
        let system = WhpSystem::open().unwrap();
        assert!(system.hypervisor_present().is_ok());
    }

    #[test]
    fn live_partition_lifecycle_when_hypervisor_is_present() {
        let system = WhpSystem::open().unwrap();
        if system.hypervisor_present().unwrap() {
            let mut partition = system.prepare_partition().unwrap();
            let range = GuestRange::new(0x10_0000, 0x1000, MapPermissions::read_write()).unwrap();
            partition.map_zeroed(range).unwrap();
            partition.write_guest(0x10_0000, &[0x5a]).unwrap();
            let mut value = [0u8; 1];
            partition.read_guest(0x10_0000, &mut value).unwrap();
            assert_eq!(value, [0x5a]);
            assert_eq!(
                partition.inject_interrupt(31),
                Err(WhpError::InvalidInterrupt)
            );
            assert_eq!(
                partition.inject_interrupt(255),
                Err(WhpError::InvalidInterrupt)
            );
        }
    }
}
