#![no_std]

use core::ffi::c_void;

pub type Status = usize;
pub type Handle = *mut c_void;
pub const SUCCESS: Status = 0;
pub const BUFFER_TOO_SMALL: Status = (1usize << (usize::BITS - 1)) | 5;
pub const LOAD_ERROR: Status = (1usize << (usize::BITS - 1)) | 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

pub const GOP_GUID: Guid = Guid {
    data1: 0x9042_a9de,
    data2: 0x23dc,
    data3: 0x4a38,
    data4: [0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a],
};
pub const RNG_GUID: Guid = Guid {
    data1: 0x3152_bca5,
    data2: 0xeade,
    data3: 0x433d,
    data4: [0x86, 0x2e, 0xc0, 0x1c, 0xdc, 0x29, 0x1f, 0x44],
};

#[repr(C)]
pub struct TableHeader {
    pub signature: u64,
    pub revision: u32,
    pub header_size: u32,
    pub crc32: u32,
    pub reserved: u32,
}

pub type GetMemoryMap = unsafe extern "efiapi" fn(
    *mut usize,
    *mut MemoryDescriptor,
    *mut usize,
    *mut usize,
    *mut u32,
) -> Status;
pub type ExitBootServices = unsafe extern "efiapi" fn(Handle, usize) -> Status;
pub type LocateProtocol =
    unsafe extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> Status;

#[repr(C)]
pub struct BootServices {
    pub header: TableHeader,
    pub raise_tpl: usize,
    pub restore_tpl: usize,
    pub allocate_pages: usize,
    pub free_pages: usize,
    pub get_memory_map: GetMemoryMap,
    pub allocate_pool: usize,
    pub free_pool: usize,
    pub create_event: usize,
    pub set_timer: usize,
    pub wait_for_event: usize,
    pub signal_event: usize,
    pub close_event: usize,
    pub check_event: usize,
    pub install_protocol_interface: usize,
    pub reinstall_protocol_interface: usize,
    pub uninstall_protocol_interface: usize,
    pub handle_protocol: usize,
    pub reserved: usize,
    pub register_protocol_notify: usize,
    pub locate_handle: usize,
    pub locate_device_path: usize,
    pub install_configuration_table: usize,
    pub load_image: usize,
    pub start_image: usize,
    pub exit: usize,
    pub unload_image: usize,
    pub exit_boot_services: ExitBootServices,
    pub get_next_monotonic_count: usize,
    pub stall: usize,
    pub set_watchdog_timer: usize,
    pub connect_controller: usize,
    pub disconnect_controller: usize,
    pub open_protocol: usize,
    pub close_protocol: usize,
    pub open_protocol_information: usize,
    pub protocols_per_handle: usize,
    pub locate_handle_buffer: usize,
    pub locate_protocol: LocateProtocol,
}

#[repr(C)]
pub struct SystemTable {
    pub header: TableHeader,
    pub firmware_vendor: *mut u16,
    pub firmware_revision: u32,
    pub console_in_handle: Handle,
    pub con_in: *mut c_void,
    pub console_out_handle: Handle,
    pub con_out: *mut c_void,
    pub standard_error_handle: Handle,
    pub std_err: *mut c_void,
    pub runtime_services: *mut c_void,
    pub boot_services: *mut BootServices,
    pub table_entry_count: usize,
    pub configuration_table: *mut c_void,
}

#[repr(C)]
pub struct MemoryDescriptor {
    pub kind: u32,
    pub physical_start: u64,
    pub virtual_start: u64,
    pub pages: u64,
    pub attributes: u64,
}

#[repr(C)]
pub struct GraphicsOutputModeInformation {
    pub version: u32,
    pub horizontal_resolution: u32,
    pub vertical_resolution: u32,
    pub pixel_format: u32,
    pub pixel_bitmask: [u32; 4],
    pub pixels_per_scan_line: u32,
}

#[repr(C)]
pub struct GraphicsOutputMode {
    pub max_mode: u32,
    pub mode: u32,
    pub info: *mut GraphicsOutputModeInformation,
    pub info_size: usize,
    pub framebuffer_base: u64,
    pub framebuffer_size: usize,
}

#[repr(C)]
pub struct GraphicsOutputProtocol {
    pub query_mode: usize,
    pub set_mode: usize,
    pub blt: usize,
    pub mode: *mut GraphicsOutputMode,
}

pub type GetRng =
    unsafe extern "efiapi" fn(*mut RngProtocol, *const Guid, usize, *mut u8) -> Status;

#[repr(C)]
pub struct RngProtocol {
    pub get_info: usize,
    pub get_rng: GetRng,
}

pub const fn status_is_error(status: Status) -> bool {
    status >> (usize::BITS - 1) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layouts_match_x86_64_uefi() {
        assert_eq!(core::mem::size_of::<TableHeader>(), 24);
        assert_eq!(core::mem::size_of::<MemoryDescriptor>(), 40);
        assert_eq!(core::mem::offset_of!(BootServices, get_memory_map), 56);
        assert_eq!(core::mem::offset_of!(BootServices, exit_boot_services), 232);
        assert_eq!(core::mem::offset_of!(BootServices, locate_protocol), 320);
        assert_eq!(core::mem::offset_of!(SystemTable, boot_services), 96);
    }
}
