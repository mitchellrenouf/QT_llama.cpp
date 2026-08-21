#![no_std]

pub mod tpm;

use core::ffi::c_void;

pub type Status = usize;
pub type Handle = *mut c_void;
pub const SUCCESS: Status = 0;
pub const BUFFER_TOO_SMALL: Status = (1usize << (usize::BITS - 1)) | 5;
pub const LOAD_ERROR: Status = (1usize << (usize::BITS - 1)) | 1;
pub const NOT_FOUND: Status = (1usize << (usize::BITS - 1)) | 14;

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
pub const LOADED_IMAGE_GUID: Guid = Guid {
    data1: 0x5b1b_31a1,
    data2: 0x9562,
    data3: 0x11d2,
    data4: [0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};
pub const SIMPLE_FILE_SYSTEM_GUID: Guid = Guid {
    data1: 0x964e_5b22,
    data2: 0x6459,
    data3: 0x11d2,
    data4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};
pub const FILE_INFO_GUID: Guid = Guid {
    data1: 0x0957_6e92,
    data2: 0x6d3f,
    data3: 0x11d2,
    data4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};
pub const TCG2_PROTOCOL_GUID: Guid = Guid {
    data1: 0x607f_766c,
    data2: 0x7455,
    data3: 0x42be,
    data4: [0x93, 0x0b, 0xe4, 0xd7, 0x6d, 0xb2, 0x72, 0x0f],
};
pub const GLOBAL_VARIABLE_GUID: Guid = Guid {
    data1: 0x8be4_df61,
    data2: 0x93ca,
    data3: 0x11d2,
    data4: [0xaa, 0x0d, 0x00, 0xe0, 0x98, 0x03, 0x2b, 0x8c],
};
pub const ACPI_20_TABLE_GUID: Guid = Guid {
    data1: 0x8868_e871,
    data2: 0xe4f1,
    data3: 0x11d3,
    data4: [0xbc, 0x22, 0x00, 0x80, 0xc7, 0x3c, 0x88, 0x81],
};
pub const ACPI_TABLE_GUID: Guid = Guid {
    data1: 0xeb9d_2d30,
    data2: 0x2d88,
    data3: 0x11d3,
    data4: [0x9a, 0x16, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d],
};

impl Guid {
    pub const fn equals(&self, other: &Self) -> bool {
        self.data1 == other.data1
            && self.data2 == other.data2
            && self.data3 == other.data3
            && self.data4[0] == other.data4[0]
            && self.data4[1] == other.data4[1]
            && self.data4[2] == other.data4[2]
            && self.data4[3] == other.data4[3]
            && self.data4[4] == other.data4[4]
            && self.data4[5] == other.data4[5]
            && self.data4[6] == other.data4[6]
            && self.data4[7] == other.data4[7]
    }
}

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
pub type HandleProtocol =
    unsafe extern "efiapi" fn(Handle, *const Guid, *mut *mut c_void) -> Status;
pub type AllocatePages = unsafe extern "efiapi" fn(u32, u32, usize, *mut u64) -> Status;
pub type Tcg2HashLogExtendEvent =
    unsafe extern "efiapi" fn(*mut Tcg2Protocol, u64, u64, u64, *mut Tcg2Event) -> Status;
pub type Tcg2SubmitCommand =
    unsafe extern "efiapi" fn(*mut Tcg2Protocol, u32, *mut u8, u32, *mut u8) -> Status;

#[repr(C)]
pub struct Tcg2Protocol {
    pub get_capability: usize,
    pub get_event_log: usize,
    pub hash_log_extend_event: Tcg2HashLogExtendEvent,
    pub submit_command: Tcg2SubmitCommand,
    pub get_active_pcr_banks: usize,
    pub set_active_pcr_banks: usize,
    pub get_result_of_set_active_pcr_banks: usize,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Tcg2EventHeader {
    pub header_size: u32,
    pub header_version: u16,
    pub pcr_index: u32,
    pub event_type: u32,
}

#[repr(C, packed)]
pub struct Tcg2Event {
    pub size: u32,
    pub header: Tcg2EventHeader,
    pub event: [u8; 32],
}

#[repr(C)]
pub struct BootServices {
    pub header: TableHeader,
    pub raise_tpl: usize,
    pub restore_tpl: usize,
    pub allocate_pages: AllocatePages,
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
    pub handle_protocol: HandleProtocol,
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

pub type GetVariable =
    unsafe extern "efiapi" fn(*const u16, *const Guid, *mut u32, *mut usize, *mut c_void) -> Status;

#[repr(C)]
pub struct RuntimeServices {
    pub header: TableHeader,
    pub get_time: usize,
    pub set_time: usize,
    pub get_wakeup_time: usize,
    pub set_wakeup_time: usize,
    pub set_virtual_address_map: usize,
    pub convert_pointer: usize,
    pub get_variable: GetVariable,
    pub get_next_variable_name: usize,
    pub set_variable: usize,
}

#[repr(C)]
pub struct LoadedImageProtocol {
    pub revision: u32,
    pub parent_handle: Handle,
    pub system_table: *mut SystemTable,
    pub device_handle: Handle,
    pub file_path: *mut c_void,
    pub reserved: *mut c_void,
    pub load_options_size: u32,
    pub load_options: *mut c_void,
    pub image_base: *mut c_void,
    pub image_size: u64,
    pub image_code_type: u32,
    pub image_data_type: u32,
    pub unload: usize,
}

pub type OpenVolume =
    unsafe extern "efiapi" fn(*mut SimpleFileSystemProtocol, *mut *mut FileProtocol) -> Status;
#[repr(C)]
pub struct SimpleFileSystemProtocol {
    pub revision: u64,
    pub open_volume: OpenVolume,
}
pub type FileOpen = unsafe extern "efiapi" fn(
    *mut FileProtocol,
    *mut *mut FileProtocol,
    *const u16,
    u64,
    u64,
) -> Status;
pub type FileClose = unsafe extern "efiapi" fn(*mut FileProtocol) -> Status;
pub type FileRead = unsafe extern "efiapi" fn(*mut FileProtocol, *mut usize, *mut c_void) -> Status;
pub type FileGetInfo =
    unsafe extern "efiapi" fn(*mut FileProtocol, *const Guid, *mut usize, *mut c_void) -> Status;
#[repr(C)]
pub struct FileProtocol {
    pub revision: u64,
    pub open: FileOpen,
    pub close: FileClose,
    pub delete: usize,
    pub read: FileRead,
    pub write: usize,
    pub get_position: usize,
    pub set_position: usize,
    pub get_info: FileGetInfo,
    pub set_info: usize,
    pub flush: usize,
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
    pub runtime_services: *mut RuntimeServices,
    pub boot_services: *mut BootServices,
    pub table_entry_count: usize,
    pub configuration_table: *mut ConfigurationTable,
}

#[repr(C)]
pub struct ConfigurationTable {
    pub vendor_guid: Guid,
    pub vendor_table: *mut c_void,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcpiError {
    Missing,
    TooManyConfigurationTables,
    NullTable,
    BadSignature,
    BadChecksum,
    BadLength,
}

/// Locates ACPI 2.0 preferentially, falling back to ACPI 1.0. The returned
/// physical pointer is admitted only after the complete applicable RSDP
/// checksum validates.
pub unsafe fn find_acpi_root(system: &SystemTable) -> Result<u64, AcpiError> {
    if system.table_entry_count > 1024 {
        return Err(AcpiError::TooManyConfigurationTables);
    }
    if system.table_entry_count != 0 && system.configuration_table.is_null() {
        return Err(AcpiError::NullTable);
    }
    let entries = unsafe {
        core::slice::from_raw_parts(system.configuration_table, system.table_entry_count)
    };
    for wanted in [&ACPI_20_TABLE_GUID, &ACPI_TABLE_GUID] {
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.vendor_guid.equals(wanted))
        {
            if entry.vendor_table.is_null() {
                return Err(AcpiError::NullTable);
            }
            let rsdp = entry.vendor_table.cast::<u8>();
            let first = unsafe { core::slice::from_raw_parts(rsdp, 20) };
            if &first[..8] != b"RSD PTR " {
                return Err(AcpiError::BadSignature);
            }
            if checksum(first) != 0 {
                return Err(AcpiError::BadChecksum);
            }
            if wanted.equals(&ACPI_20_TABLE_GUID) {
                let length_bytes = unsafe { core::slice::from_raw_parts(rsdp.add(20), 4) };
                let length = u32::from_le_bytes(length_bytes.try_into().unwrap()) as usize;
                if !(36..=4096).contains(&length) {
                    return Err(AcpiError::BadLength);
                }
                let complete = unsafe { core::slice::from_raw_parts(rsdp, length) };
                if checksum(complete) != 0 {
                    return Err(AcpiError::BadChecksum);
                }
            }
            return Ok(entry.vendor_table as u64);
        }
    }
    Err(AcpiError::Missing)
}

fn checksum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte))
}

#[repr(C)]
pub struct MemoryDescriptor {
    pub kind: u32,
    pub physical_start: u64,
    pub virtual_start: u64,
    pub pages: u64,
    pub attributes: u64,
}

pub const MAX_NORMALIZED_REGIONS: usize = 128;
const PAGE_SIZE: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NormalizedMemoryKind {
    Free = 0,
    Kernel = 1,
    Firmware = 2,
    Mmio = 3,
    Acpi = 4,
    Reserved = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormalizedRegion {
    pub start: u64,
    pub pages: u64,
    pub kind: NormalizedMemoryKind,
}

impl NormalizedRegion {
    pub const EMPTY: Self = Self {
        start: 0,
        pages: 0,
        kind: NormalizedMemoryKind::Reserved,
    };

    pub const fn end(self) -> u64 {
        self.start + self.pages * PAGE_SIZE
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizeError {
    Empty,
    BadDescriptorSize,
    Truncated,
    Unaligned,
    ZeroPages,
    Overflow,
    Unsorted,
    Overlap,
    TooManyRegions,
    FramebufferConflict,
}

/// Converts an untrusted variable-stride UEFI map into the fixed MRML memory
/// model. Only conventional memory becomes free; boot-service and loader pages
/// remain reserved until a later kernel phase can prove they are reclaimable.
pub fn normalize_memory_map(
    bytes: &[u8],
    descriptor_size: usize,
    framebuffer_base: u64,
    framebuffer_bytes: u64,
    output: &mut [NormalizedRegion; MAX_NORMALIZED_REGIONS],
) -> Result<usize, NormalizeError> {
    if bytes.is_empty() {
        return Err(NormalizeError::Empty);
    }
    if descriptor_size < core::mem::size_of::<MemoryDescriptor>() {
        return Err(NormalizeError::BadDescriptorSize);
    }
    if bytes.len() % descriptor_size != 0 {
        return Err(NormalizeError::Truncated);
    }
    if framebuffer_base % PAGE_SIZE != 0 || framebuffer_bytes == 0 {
        return Err(NormalizeError::FramebufferConflict);
    }
    let framebuffer_end_unaligned = framebuffer_base
        .checked_add(framebuffer_bytes)
        .ok_or(NormalizeError::Overflow)?;
    let framebuffer_end = framebuffer_end_unaligned
        .checked_add(PAGE_SIZE - 1)
        .map(|end| end & !(PAGE_SIZE - 1))
        .ok_or(NormalizeError::Overflow)?;
    let mut count = 0usize;
    let mut previous_start = 0u64;
    let mut previous_end = 0u64;
    for (index, descriptor) in bytes.chunks_exact(descriptor_size).enumerate() {
        let kind = u32::from_le_bytes(descriptor[..4].try_into().unwrap());
        let start = u64::from_le_bytes(descriptor[8..16].try_into().unwrap());
        let pages = u64::from_le_bytes(descriptor[24..32].try_into().unwrap());
        if start % PAGE_SIZE != 0 {
            return Err(NormalizeError::Unaligned);
        }
        if pages == 0 {
            return Err(NormalizeError::ZeroPages);
        }
        let end = pages
            .checked_mul(PAGE_SIZE)
            .and_then(|length| start.checked_add(length))
            .ok_or(NormalizeError::Overflow)?;
        if index != 0 && start < previous_start {
            return Err(NormalizeError::Unsorted);
        }
        if index != 0 && start < previous_end {
            return Err(NormalizeError::Overlap);
        }
        previous_start = start;
        previous_end = end;
        let normalized_kind = match kind {
            7 => NormalizedMemoryKind::Free,
            5 | 6 => NormalizedMemoryKind::Firmware,
            9 | 10 => NormalizedMemoryKind::Acpi,
            11 | 12 => NormalizedMemoryKind::Mmio,
            _ => NormalizedMemoryKind::Reserved,
        };
        if count != 0
            && output[count - 1].kind == normalized_kind
            && output[count - 1].end() == start
        {
            output[count - 1].pages = output[count - 1]
                .pages
                .checked_add(pages)
                .ok_or(NormalizeError::Overflow)?;
            continue;
        }
        let slot = output
            .get_mut(count)
            .ok_or(NormalizeError::TooManyRegions)?;
        *slot = NormalizedRegion {
            start,
            pages,
            kind: normalized_kind,
        };
        count += 1;
    }
    overlay_framebuffer(output, count, framebuffer_base, framebuffer_end)
}

fn overlay_framebuffer(
    output: &mut [NormalizedRegion; MAX_NORMALIZED_REGIONS],
    count: usize,
    framebuffer_start: u64,
    framebuffer_end: u64,
) -> Result<usize, NormalizeError> {
    let framebuffer = NormalizedRegion {
        start: framebuffer_start,
        pages: (framebuffer_end - framebuffer_start) / PAGE_SIZE,
        kind: NormalizedMemoryKind::Mmio,
    };
    let mut rebuilt = [NormalizedRegion::EMPTY; MAX_NORMALIZED_REGIONS];
    let mut rebuilt_count = 0usize;
    let mut inserted = false;
    for region in output[..count].iter().copied() {
        if !inserted && framebuffer_end <= region.start {
            push_region(&mut rebuilt, &mut rebuilt_count, framebuffer)?;
            inserted = true;
        }
        let overlaps = region.start < framebuffer_end && region.end() > framebuffer_start;
        if !overlaps {
            push_region(&mut rebuilt, &mut rebuilt_count, region)?;
            continue;
        }
        if inserted || framebuffer_start < region.start || framebuffer_end > region.end() {
            return Err(NormalizeError::FramebufferConflict);
        }
        if region.start < framebuffer_start {
            push_region(
                &mut rebuilt,
                &mut rebuilt_count,
                NormalizedRegion {
                    start: region.start,
                    pages: (framebuffer_start - region.start) / PAGE_SIZE,
                    kind: region.kind,
                },
            )?;
        }
        push_region(&mut rebuilt, &mut rebuilt_count, framebuffer)?;
        if framebuffer_end < region.end() {
            push_region(
                &mut rebuilt,
                &mut rebuilt_count,
                NormalizedRegion {
                    start: framebuffer_end,
                    pages: (region.end() - framebuffer_end) / PAGE_SIZE,
                    kind: region.kind,
                },
            )?;
        }
        inserted = true;
    }
    if !inserted {
        push_region(&mut rebuilt, &mut rebuilt_count, framebuffer)?;
    }
    output[..rebuilt_count].copy_from_slice(&rebuilt[..rebuilt_count]);
    Ok(rebuilt_count)
}

fn push_region(
    output: &mut [NormalizedRegion; MAX_NORMALIZED_REGIONS],
    count: &mut usize,
    region: NormalizedRegion,
) -> Result<(), NormalizeError> {
    if *count != 0
        && output[*count - 1].kind == region.kind
        && output[*count - 1].end() == region.start
    {
        output[*count - 1].pages = output[*count - 1]
            .pages
            .checked_add(region.pages)
            .ok_or(NormalizeError::Overflow)?;
        return Ok(());
    }
    let slot = output
        .get_mut(*count)
        .ok_or(NormalizeError::TooManyRegions)?;
    *slot = region;
    *count += 1;
    Ok(())
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

pub fn parse_root_digest(encoded: &[u8]) -> Option<[u8; 64]> {
    if encoded.len() != 128 {
        return None;
    }
    let mut output = [0u8; 64];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = hex(encoded[index * 2])?
            .checked_mul(16)?
            .checked_add(hex(encoded[index * 2 + 1])?)?;
    }
    (!output.iter().all(|byte| *byte == 0)).then_some(output)
}

pub fn parse_nonzero_version(encoded: &[u8]) -> Option<u64> {
    if encoded.is_empty() || encoded.len() > 20 {
        return None;
    }
    let mut value = 0u64;
    for byte in encoded {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as u64)?;
    }
    (value != 0).then_some(value)
}

fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layouts_match_x86_64_uefi() {
        assert_eq!(core::mem::size_of::<TableHeader>(), 24);
        assert_eq!(core::mem::size_of::<MemoryDescriptor>(), 40);
        assert_eq!(core::mem::offset_of!(BootServices, get_memory_map), 56);
        assert_eq!(core::mem::offset_of!(BootServices, allocate_pages), 40);
        assert_eq!(core::mem::offset_of!(BootServices, handle_protocol), 152);
        assert_eq!(core::mem::offset_of!(BootServices, exit_boot_services), 232);
        assert_eq!(core::mem::offset_of!(BootServices, locate_protocol), 320);
        assert_eq!(core::mem::offset_of!(SystemTable, boot_services), 96);
        assert_eq!(core::mem::size_of::<Tcg2EventHeader>(), 14);
        assert_eq!(core::mem::offset_of!(Tcg2Event, header), 4);
        assert_eq!(core::mem::offset_of!(Tcg2Event, event), 18);
        assert_eq!(core::mem::size_of::<Tcg2Protocol>(), 56);
        assert_eq!(core::mem::offset_of!(RuntimeServices, get_variable), 72);
        assert_eq!(core::mem::size_of::<RuntimeServices>(), 96);
        assert_eq!(
            core::mem::offset_of!(LoadedImageProtocol, device_handle),
            24
        );
        assert_eq!(core::mem::offset_of!(FileProtocol, read), 32);
        assert_eq!(core::mem::offset_of!(FileProtocol, get_info), 64);
    }

    #[test]
    fn embedded_policy_text_is_exact_and_nonzero() {
        let mut digest = [b'0'; 128];
        digest[127] = b'f';
        assert_eq!(parse_root_digest(&digest).unwrap()[63], 15);
        assert!(parse_root_digest(&digest[..127]).is_none());
        digest[0] = b'g';
        assert!(parse_root_digest(&digest).is_none());
        assert_eq!(
            parse_nonzero_version(b"18446744073709551615"),
            Some(u64::MAX)
        );
        assert!(parse_nonzero_version(b"0").is_none());
        assert!(parse_nonzero_version(b"18446744073709551616").is_none());
    }

    #[test]
    fn acpi_root_requires_both_checksums() {
        let mut rsdp = [0u8; 36];
        rsdp[..8].copy_from_slice(b"RSD PTR ");
        rsdp[15] = 2;
        rsdp[20..24].copy_from_slice(&36u32.to_le_bytes());
        rsdp[8] = 0u8.wrapping_sub(checksum(&rsdp[..20]));
        rsdp[32] = 0u8.wrapping_sub(checksum(&rsdp));
        let mut entry = ConfigurationTable {
            vendor_guid: ACPI_20_TABLE_GUID,
            vendor_table: rsdp.as_mut_ptr().cast(),
        };
        let system = SystemTable {
            header: TableHeader {
                signature: 0,
                revision: 0,
                header_size: 0,
                crc32: 0,
                reserved: 0,
            },
            firmware_vendor: core::ptr::null_mut(),
            firmware_revision: 0,
            console_in_handle: core::ptr::null_mut(),
            con_in: core::ptr::null_mut(),
            console_out_handle: core::ptr::null_mut(),
            con_out: core::ptr::null_mut(),
            standard_error_handle: core::ptr::null_mut(),
            std_err: core::ptr::null_mut(),
            runtime_services: core::ptr::null_mut(),
            boot_services: core::ptr::null_mut(),
            table_entry_count: 1,
            configuration_table: &mut entry,
        };
        assert_eq!(unsafe { find_acpi_root(&system) }, Ok(rsdp.as_ptr() as u64));
        let mut corrupt = rsdp;
        corrupt[35] ^= 1;
        let mut corrupt_entry = ConfigurationTable {
            vendor_guid: ACPI_20_TABLE_GUID,
            vendor_table: corrupt.as_mut_ptr().cast(),
        };
        let corrupt_system = SystemTable {
            configuration_table: &mut corrupt_entry,
            ..system
        };
        assert_eq!(
            unsafe { find_acpi_root(&corrupt_system) },
            Err(AcpiError::BadChecksum)
        );
    }

    fn descriptor(kind: u32, start: u64, pages: u64) -> [u8; 40] {
        let mut bytes = [0u8; 40];
        bytes[..4].copy_from_slice(&kind.to_le_bytes());
        bytes[8..16].copy_from_slice(&start.to_le_bytes());
        bytes[24..32].copy_from_slice(&pages.to_le_bytes());
        bytes
    }

    #[test]
    fn normalizes_merges_and_requires_mmio_framebuffer() {
        let descriptors = [
            descriptor(7, 0x1000, 2),
            descriptor(7, 0x3000, 1),
            descriptor(11, 0xa0000, 4),
        ];
        let mut bytes = [0u8; 120];
        for (index, descriptor) in descriptors.iter().enumerate() {
            bytes[index * 40..index * 40 + 40].copy_from_slice(descriptor);
        }
        let mut output = [NormalizedRegion::EMPTY; MAX_NORMALIZED_REGIONS];
        assert_eq!(
            normalize_memory_map(&bytes, 40, 0xa0000, 4096, &mut output),
            Ok(2)
        );
        assert_eq!(output[0].pages, 3);
        assert_eq!(output[0].kind, NormalizedMemoryKind::Free);
        assert_eq!(output[1].kind, NormalizedMemoryKind::Mmio);

        let mut output = [NormalizedRegion::EMPTY; MAX_NORMALIZED_REGIONS];
        assert_eq!(
            normalize_memory_map(&bytes, 40, 0xb0000, 4096, &mut output),
            Ok(3)
        );
        assert_eq!(output[2].start, 0xb0000);
        assert_eq!(output[2].kind, NormalizedMemoryKind::Mmio);
    }
}
