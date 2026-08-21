use core::ffi::{c_char, c_void};
use core::ptr::NonNull;
use core::slice;

use mrml_kernel::VmExit;

use crate::{GuestRange, WHP_EXIT_CONTEXT_BYTES, WhpError, decode_exit_context};

type Handle = *mut c_void;
type Hresult = i32;
type CreatePartition = unsafe extern "system" fn(*mut Handle) -> Hresult;
type DeletePartition = unsafe extern "system" fn(Handle) -> Hresult;
type SetPartitionProperty = unsafe extern "system" fn(Handle, u32, *const c_void, u32) -> Hresult;
type SetupPartition = unsafe extern "system" fn(Handle) -> Hresult;
type MapGpaRange = unsafe extern "system" fn(Handle, *mut c_void, u64, u64, u32) -> Hresult;
type UnmapGpaRange = unsafe extern "system" fn(Handle, u64, u64) -> Hresult;
type CreateVirtualProcessor = unsafe extern "system" fn(Handle, u32, u32) -> Hresult;
type DeleteVirtualProcessor = unsafe extern "system" fn(Handle, u32) -> Hresult;
type RunVirtualProcessor = unsafe extern "system" fn(Handle, u32, *mut c_void, u32) -> Hresult;

const PROCESSOR_COUNT: u32 = 0x1fff;
const MEM_COMMIT_RESERVE: u32 = 0x3000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 4;
const MAX_MAPPINGS: usize = 32;

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
    create_partition: CreatePartition,
    delete_partition: DeletePartition,
    set_partition_property: SetPartitionProperty,
    setup_partition: SetupPartition,
    map_gpa_range: MapGpaRange,
    unmap_gpa_range: UnmapGpaRange,
    create_vp: CreateVirtualProcessor,
    delete_vp: DeleteVirtualProcessor,
    run_vp: RunVirtualProcessor,
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
                create_partition: resolve(module, b"WHvCreatePartition\0")?,
                delete_partition: resolve(module, b"WHvDeletePartition\0")?,
                set_partition_property: resolve(module, b"WHvSetPartitionProperty\0")?,
                setup_partition: resolve(module, b"WHvSetupPartition\0")?,
                map_gpa_range: resolve(module, b"WHvMapGpaRange\0")?,
                unmap_gpa_range: resolve(module, b"WHvUnmapGpaRange\0")?,
                create_vp: resolve(module, b"WHvCreateVirtualProcessor\0")?,
                delete_vp: resolve(module, b"WHvDeleteVirtualProcessor\0")?,
                run_vp: resolve(module, b"WHvRunVirtualProcessor\0")?,
            }
        };
        Ok(Self {
            _library: library,
            api,
        })
    }

    pub fn prepare_partition(&self) -> Result<PreparedWhpPartition, WhpError> {
        let mut partition = core::ptr::null_mut();
        check(unsafe { (self.api.create_partition)(&mut partition) })?;
        let partition = NonNull::new(partition).ok_or(WhpError::PlatformUnavailable)?;
        let mut guard = PartitionGuard {
            partition,
            api: self.api,
            setup: false,
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
        check(unsafe { (self.api.setup_partition)(partition.as_ptr()) })?;
        guard.setup = true;
        check(unsafe { (self.api.create_vp)(partition.as_ptr(), 0, 0) })?;
        guard.vp = true;
        let prepared = PreparedWhpPartition {
            partition,
            api: self.api,
            mappings: [const { None }; MAX_MAPPINGS],
        };
        core::mem::forget(guard);
        Ok(prepared)
    }
}

pub struct PreparedWhpPartition {
    partition: NonNull<c_void>,
    api: Api,
    mappings: [Option<OwnedMapping>; MAX_MAPPINGS],
}

impl PreparedWhpPartition {
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
            .ok_or(WhpError::MemoryOverflow)?;
        if self
            .mappings
            .iter()
            .flatten()
            .any(|entry| overlaps(entry.range, range))
        {
            return Err(WhpError::MalformedExit);
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
}

impl Drop for PreparedWhpPartition {
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
    setup: bool,
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
impl Drop for OwnedMapping {
    fn drop(&mut self) {
        let _ = unsafe { VirtualFree(self.address.as_ptr(), 0, MEM_RELEASE) };
    }
}

fn overlaps(left: GuestRange, right: GuestRange) -> bool {
    left.guest_address() < right.guest_address() + right.size()
        && right.guest_address() < left.guest_address() + left.size()
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
