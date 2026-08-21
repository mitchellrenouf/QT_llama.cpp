#![no_std]
#![no_main]

use core::arch::global_asm;
use core::cell::UnsafeCell;
use mrml_kernel::{
    ArtifactKind, FramebufferInfo, HANDOFF_HEADER_BYTES, HANDOFF_REGION_BYTES, MAX_HANDOFF_REGIONS,
    MAX_KERNEL_IMAGE_BYTES, MemoryKind, MemoryRegion, PeImage, PhysAddr, PixelFormat,
    SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact, TrustRoot,
    arch::x86_64::{
        PagePermissions, PageTableBuildError, PageTableBuilder, PageTableStore, VirtAddr,
    },
    encode_handoff,
};
use mrml_uefi::tpm::{NvCounterError, TpmTransport, enforce_version};
use mrml_uefi::*;

const MAP_BYTES: usize = 64 * 1024;

struct FirmwareBuffer(UnsafeCell<[u8; MAP_BYTES]>);
unsafe impl Sync for FirmwareBuffer {}
static MEMORY_MAP: FirmwareBuffer = FirmwareBuffer(UnsafeCell::new([0; MAP_BYTES]));
struct RegionBuffer(UnsafeCell<[NormalizedRegion; MAX_NORMALIZED_REGIONS]>);
unsafe impl Sync for RegionBuffer {}
static REGIONS: RegionBuffer = RegionBuffer(UnsafeCell::new(
    [NormalizedRegion::EMPTY; MAX_NORMALIZED_REGIONS],
));
const FILE_INFO_BYTES: usize = 1024;
struct FileInfoBuffer(UnsafeCell<[u8; FILE_INFO_BYTES]>);
unsafe impl Sync for FileInfoBuffer {}
static FILE_INFO: FileInfoBuffer = FileInfoBuffer(UnsafeCell::new([0; FILE_INFO_BYTES]));
const HANDOFF_BYTES: usize = HANDOFF_HEADER_BYTES + MAX_HANDOFF_REGIONS * HANDOFF_REGION_BYTES;
const PAGE_BYTES: usize = 4096;
#[repr(C, align(4096))]
struct HandoffBuffer(UnsafeCell<[u8; PAGE_BYTES]>);
unsafe impl Sync for HandoffBuffer {}
static HANDOFF: HandoffBuffer = HandoffBuffer(UnsafeCell::new([0; PAGE_BYTES]));
const _: () = assert!(HANDOFF_BYTES <= PAGE_BYTES);
const _: () = assert!(core::mem::align_of::<HandoffBuffer>() == PAGE_BYTES);
const _: () = assert!(core::mem::size_of::<HandoffBuffer>() == PAGE_BYTES);
const KERNEL_STACK_PAGES: usize = 16;
const KERNEL_STACK_GUARD_PAGES: usize = 1;
const KERNEL_PATH: &[u16] = &[
    92, 69, 70, 73, 92, 77, 82, 77, 76, 92, 75, 69, 82, 78, 69, 76, 46, 83, 73, 71, 78, 69, 68, 0,
];

#[derive(Clone, Copy)]
struct Framebuffer {
    base: u64,
    size: usize,
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
}

#[derive(Clone, Copy)]
struct Transition {
    root: u64,
    stack_top: u64,
}

struct FirmwarePageTables {
    services: *const BootServices,
}

struct FirmwareTpm {
    protocol: *mut Tcg2Protocol,
}

impl TpmTransport for FirmwareTpm {
    fn submit(&mut self, command: &[u8], response: &mut [u8]) -> Result<usize, NvCounterError> {
        if command.len() > 128 || response.len() < 10 {
            return Err(NvCounterError::Transport);
        }
        let protocol = unsafe { self.protocol.as_mut() }.ok_or(NvCounterError::Transport)?;
        let mut input = [0u8; 128];
        input[..command.len()].copy_from_slice(command);
        check(unsafe {
            (protocol.submit_command)(
                protocol,
                command.len() as u32,
                input.as_mut_ptr(),
                response.len() as u32,
                response.as_mut_ptr(),
            )
        })
        .map_err(|_| NvCounterError::Transport)?;
        let declared = u32::from_be_bytes(
            response[2..6]
                .try_into()
                .map_err(|_| NvCounterError::MalformedResponse)?,
        ) as usize;
        if !(10..=response.len()).contains(&declared) {
            return Err(NvCounterError::MalformedResponse);
        }
        Ok(declared)
    }
}

impl PageTableStore for FirmwarePageTables {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        let services = unsafe { self.services.as_ref() }.ok_or(PageTableBuildError::Storage)?;
        let mut address = 0u64;
        check(unsafe { (services.allocate_pages)(0, 2, 1, &mut address) })
            .map_err(|_| PageTableBuildError::Storage)?;
        let frame = PhysAddr::new(address).map_err(|_| PageTableBuildError::Storage)?;
        unsafe { core::ptr::write_bytes(address as *mut u8, 0, 4096) };
        Ok(frame)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        Ok(unsafe { *((table.get() as *const u64).add(index)) })
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        unsafe { *((table.get() as *mut u64).add(index)) = value };
        Ok(())
    }
}

global_asm!(
    r#"
    .section .text
    .p2align 12
    .global mrml_activate_address_space
mrml_activate_address_space:
    cli
    mov r11, rcx
    mov r12, rdx
    mov r10, qword ptr [rsp + 40]
    mov ecx, 0xc0000080
    rdmsr
    or eax, 0x800
    wrmsr
    mov rax, cr0
    or rax, 0x10000
    mov cr0, rax
    mov cr3, r11
    mov rsp, r12
    mov rcx, r9
    mov rdx, r10
    jmp r8
    .global mrml_activate_address_space_end
mrml_activate_address_space_end:
    "#
);

unsafe extern "efiapi" {
    fn mrml_activate_address_space(
        root: u64,
        stack_top: u64,
        entry: u64,
        handoff: *const u8,
        handoff_length: usize,
    ) -> !;
    static mrml_activate_address_space_end: u8;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

#[unsafe(export_name = "efi_main")]
pub unsafe extern "efiapi" fn efi_main(image: Handle, table: *mut SystemTable) -> Status {
    // SAFETY: UEFI invokes this entry point with a live system table. Every
    // pointer obtained below is checked for null before it is dereferenced and
    // is used only while boot services remain active.
    match unsafe { boot(image, table) } {
        Ok(()) => SUCCESS,
        Err(status) => status,
    }
}

unsafe fn boot(image: Handle, table: *mut SystemTable) -> Result<(), Status> {
    let system = unsafe { table.as_mut() }.ok_or(LOAD_ERROR)?;
    let services = unsafe { system.boot_services.as_mut() }.ok_or(LOAD_ERROR)?;
    let secure_boot = detect_secure_boot(system.runtime_services)?;
    if require_secure_boot() && !secure_boot {
        return Err(LOAD_ERROR);
    }

    let mut gop_pointer = core::ptr::null_mut();
    check(unsafe {
        (services.locate_protocol)(&GOP_GUID, core::ptr::null_mut(), &mut gop_pointer)
    })?;
    let gop = unsafe { (gop_pointer as *mut GraphicsOutputProtocol).as_mut() }.ok_or(LOAD_ERROR)?;
    let mode = unsafe { gop.mode.as_mut() }.ok_or(LOAD_ERROR)?;
    let info = unsafe { mode.info.as_ref() }.ok_or(LOAD_ERROR)?;
    if !matches!(info.pixel_format, 0 | 1)
        || info.horizontal_resolution == 0
        || info.vertical_resolution == 0
        || info.pixels_per_scan_line < info.horizontal_resolution
        || mode.framebuffer_base == 0
        || mode.framebuffer_size == 0
    {
        return Err(LOAD_ERROR);
    }
    let framebuffer = Framebuffer {
        base: mode.framebuffer_base,
        size: mode.framebuffer_size,
        width: info.horizontal_resolution,
        height: info.vertical_resolution,
        stride: info.pixels_per_scan_line,
        format: info.pixel_format,
    };
    let pixel_format = if framebuffer.format == 0 {
        PixelFormat::RedGreenBlueReserved
    } else {
        PixelFormat::BlueGreenRedReserved
    };
    FramebufferInfo::new(
        framebuffer.base,
        framebuffer.size as u64,
        framebuffer.width,
        framebuffer.height,
        framebuffer.stride,
        pixel_format,
    )
    .map_err(|_| LOAD_ERROR)?;
    paint(framebuffer, [0x80, 0x00, 0x00]);

    let mut rng_pointer = core::ptr::null_mut();
    check(unsafe {
        (services.locate_protocol)(&RNG_GUID, core::ptr::null_mut(), &mut rng_pointer)
    })?;
    let rng = unsafe { (rng_pointer as *mut RngProtocol).as_mut() }.ok_or(LOAD_ERROR)?;
    let mut entropy = [0u8; 32];
    check(unsafe { (rng.get_rng)(rng, core::ptr::null(), entropy.len(), entropy.as_mut_ptr()) })?;
    if entropy.iter().all(|byte| *byte == 0) {
        return Err(LOAD_ERROR);
    }
    paint(framebuffer, [0x80, 0x40, 0x00]);

    let kernel_file = unsafe { load_kernel_file(image, services, framebuffer) }?;
    if kernel_file
        .address
        .checked_add(kernel_file.length as u64)
        .is_none()
    {
        return Err(LOAD_ERROR);
    }
    let kernel_bytes = unsafe {
        core::slice::from_raw_parts(kernel_file.address as *const u8, kernel_file.length)
    };
    let root_digest = embedded_kernel_root().ok_or(LOAD_ERROR)?;
    let minimum_version = embedded_minimum_version().ok_or(LOAD_ERROR)?;
    let signed = SignedArtifact::decode(kernel_bytes).map_err(|_| LOAD_ERROR)?;
    let verified = signed
        .verify_executable(
            &TrustRoot::new(ArtifactKind::Kernel, root_digest, minimum_version),
            ArtifactKind::Kernel,
        )
        .map_err(|_| LOAD_ERROR)?;
    let measured_boot = measure_kernel(services, signed.payload())?;
    if require_tpm_measurement() && !measured_boot {
        return Err(LOAD_ERROR);
    }
    if verified.image().image_size() == 0 {
        return Err(LOAD_ERROR);
    }
    paint(framebuffer, [0x20, 0x40, 0x80]);
    let image_size = verified.image().image_size() as usize;
    let image_pages = image_size
        .checked_add(4095)
        .map(|value| value / 4096)
        .ok_or(LOAD_ERROR)?;
    let image_allocation_bytes = image_pages.checked_mul(4096).ok_or(LOAD_ERROR)?;
    let mut image_address = 0u64;
    check(unsafe { (services.allocate_pages)(0, 1, image_pages, &mut image_address) })?;
    if image_address == 0 || image_address % 4096 != 0 {
        return Err(LOAD_ERROR);
    }
    let image_allocation = unsafe {
        core::slice::from_raw_parts_mut(image_address as *mut u8, image_allocation_bytes)
    };
    image_allocation.fill(0);
    let image_destination = &mut image_allocation[..image_size];
    let kernel_entry = verified
        .image()
        .materialize_at(image_destination, image_address)
        .map_err(|_| LOAD_ERROR)?;
    let transition = prepare_transition(services, &verified.image(), image_address, framebuffer)?;
    let kernel_version = verified.artifact().version();
    let kernel_measurement = *verified.artifact().digest();
    paint(framebuffer, [0x00, 0x60, 0x20]);

    paint(framebuffer, [0x14, 0x21, 0x35]);

    let acpi_root = unsafe { find_acpi_root(system) }.map_err(|_| LOAD_ERROR)?;
    if acpi_root == 0 {
        return Err(LOAD_ERROR);
    }
    let rollback_protected = enforce_rollback_counter(services, kernel_version)?;
    if require_rollback_counter() && !rollback_protected {
        return Err(LOAD_ERROR);
    }

    let (map_size, descriptor_size) = unsafe { exit_boot_services(image, services) }?;
    let map_bytes = unsafe { core::slice::from_raw_parts(MEMORY_MAP.0.get().cast(), map_size) };
    let regions = unsafe { &mut *REGIONS.0.get() };
    let region_count = normalize_memory_map(
        map_bytes,
        descriptor_size,
        framebuffer.base,
        framebuffer.size as u64,
        regions,
    )
    .map_err(|_| LOAD_ERROR)?;
    if region_count == 0 {
        return Err(LOAD_ERROR);
    }
    enter_kernel(
        transition,
        kernel_entry,
        kernel_version,
        kernel_measurement,
        measured_boot,
        secure_boot,
        rollback_protected,
        entropy,
        acpi_root,
        framebuffer,
        &regions[..region_count],
    )?;
    halt()
}

fn embedded_kernel_root() -> Option<[u8; 64]> {
    parse_root_digest(option_env!("MRML_KERNEL_ROOT_DIGEST_HEX")?.as_bytes())
}

fn embedded_minimum_version() -> Option<u64> {
    parse_nonzero_version(option_env!("MRML_KERNEL_MIN_VERSION")?.as_bytes())
}

fn require_tpm_measurement() -> bool {
    matches!(option_env!("MRML_REQUIRE_TPM"), Some("1"))
}

fn require_secure_boot() -> bool {
    matches!(option_env!("MRML_REQUIRE_SECURE_BOOT"), Some("1"))
}

fn require_rollback_counter() -> bool {
    matches!(option_env!("MRML_REQUIRE_ROLLBACK"), Some("1"))
}

fn configured_nv_index() -> Result<Option<u32>, Status> {
    let Some(encoded) = option_env!("MRML_TPM_NV_INDEX_HEX") else {
        return Ok(None);
    };
    let bytes = encoded.as_bytes();
    if bytes.len() != 8 {
        return Err(LOAD_ERROR);
    }
    let mut value = 0u32;
    for byte in bytes {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return Err(LOAD_ERROR),
        };
        value = value.checked_mul(16).ok_or(LOAD_ERROR)? | u32::from(digit);
    }
    Ok(Some(value))
}

fn enforce_rollback_counter(services: &BootServices, version: u64) -> Result<bool, Status> {
    let Some(index) = configured_nv_index()? else {
        return Ok(false);
    };
    let mut protocol_pointer = core::ptr::null_mut();
    check(unsafe {
        (services.locate_protocol)(
            &TCG2_PROTOCOL_GUID,
            core::ptr::null_mut(),
            &mut protocol_pointer,
        )
    })?;
    let mut transport = FirmwareTpm {
        protocol: protocol_pointer.cast(),
    };
    enforce_version(&mut transport, index, version, 1024).map_err(|_| LOAD_ERROR)?;
    Ok(true)
}

fn detect_secure_boot(runtime: *mut RuntimeServices) -> Result<bool, Status> {
    const SECURE_BOOT: &[u16] = &[83, 101, 99, 117, 114, 101, 66, 111, 111, 116, 0];
    const SETUP_MODE: &[u16] = &[83, 101, 116, 117, 112, 77, 111, 100, 101, 0];
    let runtime = unsafe { runtime.as_ref() }.ok_or(LOAD_ERROR)?;
    Ok(read_boolean_variable(runtime, SECURE_BOOT)? == Some(true)
        && read_boolean_variable(runtime, SETUP_MODE)? == Some(false))
}

fn read_boolean_variable(runtime: &RuntimeServices, name: &[u16]) -> Result<Option<bool>, Status> {
    if name.last().copied() != Some(0) {
        return Err(LOAD_ERROR);
    }
    let mut attributes = 0u32;
    let mut size = 1usize;
    let mut value = 0u8;
    let status = unsafe {
        (runtime.get_variable)(
            name.as_ptr(),
            &GLOBAL_VARIABLE_GUID,
            &mut attributes,
            &mut size,
            (&mut value as *mut u8).cast(),
        )
    };
    if status == NOT_FOUND {
        return Ok(None);
    }
    check(status)?;
    if size != 1 || value > 1 {
        return Err(LOAD_ERROR);
    }
    Ok(Some(value == 1))
}

fn measure_kernel(services: &BootServices, payload: &[u8]) -> Result<bool, Status> {
    if payload.is_empty() {
        return Err(LOAD_ERROR);
    }
    let mut protocol_pointer = core::ptr::null_mut();
    let status = unsafe {
        (services.locate_protocol)(
            &TCG2_PROTOCOL_GUID,
            core::ptr::null_mut(),
            &mut protocol_pointer,
        )
    };
    if status == NOT_FOUND {
        return Ok(false);
    }
    check(status)?;
    let protocol = unsafe { (protocol_pointer as *mut Tcg2Protocol).as_mut() }.ok_or(LOAD_ERROR)?;
    const DESCRIPTION: &[u8] = b"MRML authenticated kernel PE";
    let mut event = Tcg2Event {
        size: (core::mem::size_of::<u32>()
            + core::mem::size_of::<Tcg2EventHeader>()
            + DESCRIPTION.len()) as u32,
        header: Tcg2EventHeader {
            header_size: core::mem::size_of::<Tcg2EventHeader>() as u32,
            header_version: 1,
            pcr_index: 11,
            event_type: 0x0000_000d,
        },
        event: [0; 32],
    };
    event.event[..DESCRIPTION.len()].copy_from_slice(DESCRIPTION);
    check(unsafe {
        (protocol.hash_log_extend_event)(
            protocol,
            0,
            payload.as_ptr() as u64,
            payload.len() as u64,
            &mut event,
        )
    })?;
    Ok(true)
}

struct LoadedFile {
    address: u64,
    length: usize,
}

unsafe fn load_kernel_file(
    image: Handle,
    services: &BootServices,
    framebuffer: Framebuffer,
) -> Result<LoadedFile, Status> {
    let mut loaded_pointer = core::ptr::null_mut();
    check(unsafe { (services.handle_protocol)(image, &LOADED_IMAGE_GUID, &mut loaded_pointer) })?;
    let loaded =
        unsafe { (loaded_pointer as *mut LoadedImageProtocol).as_ref() }.ok_or(LOAD_ERROR)?;
    paint(framebuffer, [0x80, 0x80, 0x00]);
    if loaded.device_handle.is_null() {
        return Err(LOAD_ERROR);
    }
    paint(framebuffer, [0x40, 0x80, 0x00]);
    let mut filesystem_pointer = core::ptr::null_mut();
    check(unsafe {
        (services.handle_protocol)(
            loaded.device_handle,
            &SIMPLE_FILE_SYSTEM_GUID,
            &mut filesystem_pointer,
        )
    })?;
    let filesystem = unsafe { (filesystem_pointer as *mut SimpleFileSystemProtocol).as_mut() }
        .ok_or(LOAD_ERROR)?;
    paint(framebuffer, [0x80, 0x00, 0x80]);
    let mut root = core::ptr::null_mut();
    check(unsafe { (filesystem.open_volume)(filesystem, &mut root) })?;
    let root = unsafe { root.as_mut() }.ok_or(LOAD_ERROR)?;
    paint(framebuffer, [0x00, 0x80, 0x80]);
    let mut file_pointer = core::ptr::null_mut();
    let open_status = unsafe { (root.open)(root, &mut file_pointer, KERNEL_PATH.as_ptr(), 1, 0) };
    let _ = unsafe { (root.close)(root) };
    check(open_status)?;
    paint(framebuffer, [0x00, 0x00, 0x80]);
    let file = unsafe { file_pointer.as_mut() }.ok_or(LOAD_ERROR)?;
    let result = unsafe { read_file_pages(file, services) };
    let close_status = unsafe { (file.close)(file) };
    let loaded = result?;
    check(close_status)?;
    Ok(loaded)
}

unsafe fn read_file_pages(
    file: &mut FileProtocol,
    services: &BootServices,
) -> Result<LoadedFile, Status> {
    let mut info_size = 0usize;
    let status =
        unsafe { (file.get_info)(file, &FILE_INFO_GUID, &mut info_size, core::ptr::null_mut()) };
    if status != BUFFER_TOO_SMALL || !(80..=FILE_INFO_BYTES).contains(&info_size) {
        return Err(LOAD_ERROR);
    }
    let info = unsafe { &mut *FILE_INFO.0.get() };
    check(unsafe {
        (file.get_info)(
            file,
            &FILE_INFO_GUID,
            &mut info_size,
            info.as_mut_ptr().cast(),
        )
    })?;
    if info_size < 80 {
        return Err(LOAD_ERROR);
    }
    let file_length_u64 = u64::from_le_bytes(info[8..16].try_into().map_err(|_| LOAD_ERROR)?);
    let maximum = MAX_KERNEL_IMAGE_BYTES as usize + SIGNED_ARTIFACT_OVERHEAD_BYTES;
    let file_length = usize::try_from(file_length_u64).map_err(|_| LOAD_ERROR)?;
    if file_length == 0 || file_length > maximum {
        return Err(LOAD_ERROR);
    }
    let pages = file_length
        .checked_add(4095)
        .map(|bytes| bytes / 4096)
        .ok_or(LOAD_ERROR)?;
    let mut address = 0u64;
    check(unsafe { (services.allocate_pages)(0, 2, pages, &mut address) })?;
    if address == 0 || address % 4096 != 0 {
        return Err(LOAD_ERROR);
    }
    let mut total = 0usize;
    while total < file_length {
        let mut amount = file_length - total;
        check(unsafe { (file.read)(file, &mut amount, (address as *mut u8).add(total).cast()) })?;
        if amount == 0 || amount > file_length - total {
            return Err(LOAD_ERROR);
        }
        total += amount;
    }
    let mut extra = 0u8;
    let mut extra_size = 1usize;
    check(unsafe { (file.read)(file, &mut extra_size, (&mut extra as *mut u8).cast()) })?;
    if extra_size != 0 {
        return Err(LOAD_ERROR);
    }
    Ok(LoadedFile {
        address,
        length: file_length,
    })
}

fn enter_kernel(
    transition: Transition,
    kernel_entry: u64,
    kernel_version: u64,
    kernel_measurement: [u8; 64],
    measured_boot: bool,
    secure_boot: bool,
    rollback_protected: bool,
    entropy: [u8; 32],
    acpi_root: u64,
    framebuffer: Framebuffer,
    regions: &[NormalizedRegion],
) -> Result<(), Status> {
    let placeholder = MemoryRegion::new(
        PhysAddr::new(0).map_err(|_| LOAD_ERROR)?,
        1,
        MemoryKind::Reserved,
    )
    .map_err(|_| LOAD_ERROR)?;
    let mut kernel_regions = [placeholder; MAX_NORMALIZED_REGIONS];
    for (destination, source) in kernel_regions.iter_mut().zip(regions) {
        let kind = match source.kind {
            NormalizedMemoryKind::Free => MemoryKind::Free,
            NormalizedMemoryKind::Kernel => MemoryKind::Kernel,
            NormalizedMemoryKind::Firmware => MemoryKind::Firmware,
            NormalizedMemoryKind::Mmio => MemoryKind::Mmio,
            NormalizedMemoryKind::Acpi => MemoryKind::Acpi,
            NormalizedMemoryKind::Reserved => MemoryKind::Reserved,
        };
        *destination = MemoryRegion::new(
            PhysAddr::new(source.start).map_err(|_| LOAD_ERROR)?,
            source.pages,
            kind,
        )
        .map_err(|_| LOAD_ERROR)?;
    }
    let pixel_format = if framebuffer.format == 0 {
        PixelFormat::RedGreenBlueReserved
    } else {
        PixelFormat::BlueGreenRedReserved
    };
    let framebuffer_info = FramebufferInfo::new(
        framebuffer.base,
        framebuffer.size as u64,
        framebuffer.width,
        framebuffer.height,
        framebuffer.stride,
        pixel_format,
    )
    .map_err(|_| LOAD_ERROR)?;
    let handoff_page = unsafe { &mut *HANDOFF.0.get() };
    handoff_page.fill(0);
    let handoff = &mut handoff_page[..HANDOFF_BYTES];
    let handoff_length = encode_handoff(
        kernel_version,
        entropy,
        kernel_measurement,
        secure_boot,
        measured_boot,
        rollback_protected,
        acpi_root,
        framebuffer_info,
        &kernel_regions[..regions.len()],
        handoff,
    )
    .map_err(|_| LOAD_ERROR)?;
    if kernel_entry == 0 {
        return Err(LOAD_ERROR);
    }
    // SAFETY: the authenticated PE parser proved the entry lies in an
    // executable section, materialization completed before boot-services exit,
    // and the canonical handoff remains in static loader memory.
    unsafe {
        mrml_activate_address_space(
            transition.root,
            transition.stack_top,
            kernel_entry,
            handoff.as_ptr(),
            handoff_length,
        )
    }
}

fn prepare_transition(
    services: &BootServices,
    image: &PeImage<'_>,
    image_address: u64,
    framebuffer: Framebuffer,
) -> Result<Transition, Status> {
    let stack_allocation_pages = KERNEL_STACK_PAGES
        .checked_add(KERNEL_STACK_GUARD_PAGES)
        .ok_or(LOAD_ERROR)?;
    let mut stack_allocation_base = 0u64;
    check(unsafe {
        (services.allocate_pages)(0, 2, stack_allocation_pages, &mut stack_allocation_base)
    })?;
    let stack_base = stack_allocation_base
        .checked_add((KERNEL_STACK_GUARD_PAGES as u64) * 4096)
        .ok_or(LOAD_ERROR)?;
    let stack_bytes = (KERNEL_STACK_PAGES as u64)
        .checked_mul(4096)
        .ok_or(LOAD_ERROR)?;
    let stack_end = stack_base.checked_add(stack_bytes).ok_or(LOAD_ERROR)?;
    let stack_top = stack_end.checked_sub(8).ok_or(LOAD_ERROR)?;
    if stack_allocation_base == 0
        || !stack_allocation_base.is_multiple_of(4096)
        || !stack_base.is_multiple_of(4096)
        || !stack_end.is_multiple_of(16)
    {
        return Err(LOAD_ERROR);
    }
    let stack_allocation_bytes = stack_allocation_pages.checked_mul(4096).ok_or(LOAD_ERROR)?;
    unsafe { core::ptr::write_bytes(stack_allocation_base as *mut u8, 0, stack_allocation_bytes) };
    unsafe { *(stack_top as *mut u64) = 0 };

    let store = FirmwarePageTables {
        services: services as *const BootServices,
    };
    let mut tables = PageTableBuilder::new(store).map_err(|_| LOAD_ERROR)?;
    for index in 0..image.load_region_count() {
        let region = image.load_region(index).map_err(|_| LOAD_ERROR)?;
        let start = image_address
            .checked_add(region.virtual_address() as u64)
            .ok_or(LOAD_ERROR)?;
        let permissions = match (region.writable(), region.executable()) {
            (true, false) => PagePermissions::KERNEL_READ_WRITE,
            (false, true) => PagePermissions::KERNEL_READ_EXECUTE,
            (false, false) => PagePermissions::KERNEL_READ,
            (true, true) => return Err(LOAD_ERROR),
        };
        map_identity(&mut tables, start, region.pages() as u64, permissions)?;
    }
    map_identity(
        &mut tables,
        stack_base,
        KERNEL_STACK_PAGES as u64,
        PagePermissions::KERNEL_READ_WRITE,
    )?;
    let handoff_address = HANDOFF.0.get() as u64;
    if !handoff_address.is_multiple_of(PAGE_BYTES as u64) {
        return Err(LOAD_ERROR);
    }
    map_identity(
        &mut tables,
        handoff_address,
        1,
        PagePermissions::KERNEL_READ,
    )?;
    map_containing_identity(
        &mut tables,
        framebuffer.base,
        framebuffer.size as u64,
        PagePermissions::KERNEL_READ_WRITE,
    )?;
    let trampoline_start = mrml_activate_address_space as *const () as usize as u64;
    let trampoline_end = core::ptr::addr_of!(mrml_activate_address_space_end) as u64;
    if !trampoline_start.is_multiple_of(PAGE_BYTES as u64)
        || trampoline_end <= trampoline_start
        || trampoline_end
            > trampoline_start
                .checked_add(PAGE_BYTES as u64)
                .ok_or(LOAD_ERROR)?
    {
        return Err(LOAD_ERROR);
    }
    map_identity(
        &mut tables,
        trampoline_start,
        1,
        PagePermissions::KERNEL_READ_EXECUTE,
    )?;
    Ok(Transition {
        root: tables.root().get(),
        stack_top,
    })
}

fn map_containing_identity(
    tables: &mut PageTableBuilder<FirmwarePageTables>,
    address: u64,
    length: u64,
    permissions: PagePermissions,
) -> Result<(), Status> {
    if length == 0 {
        return Err(LOAD_ERROR);
    }
    let start = address & !4095;
    let end = address.checked_add(length).ok_or(LOAD_ERROR)?;
    let rounded_end = end.checked_add(4095).ok_or(LOAD_ERROR)? & !4095;
    map_identity(tables, start, (rounded_end - start) / 4096, permissions)
}

fn map_identity(
    tables: &mut PageTableBuilder<FirmwarePageTables>,
    start: u64,
    pages: u64,
    permissions: PagePermissions,
) -> Result<(), Status> {
    if start % 4096 != 0 || pages == 0 {
        return Err(LOAD_ERROR);
    }
    for page in 0..pages {
        let address = start
            .checked_add(page.checked_mul(4096).ok_or(LOAD_ERROR)?)
            .ok_or(LOAD_ERROR)?;
        tables
            .map_page(
                VirtAddr::new(address).map_err(|_| LOAD_ERROR)?,
                PhysAddr::new(address).map_err(|_| LOAD_ERROR)?,
                permissions,
            )
            .map_err(|_| LOAD_ERROR)?;
    }
    Ok(())
}

unsafe fn exit_boot_services(
    image: Handle,
    services: &BootServices,
) -> Result<(usize, usize), Status> {
    let map = MEMORY_MAP.0.get().cast::<MemoryDescriptor>();
    let mut last_status = LOAD_ERROR;
    for _ in 0..3 {
        let mut map_size = MAP_BYTES;
        let mut map_key = 0usize;
        let mut descriptor_size = 0usize;
        let mut descriptor_version = 0u32;
        check(unsafe {
            (services.get_memory_map)(
                &mut map_size,
                map,
                &mut map_key,
                &mut descriptor_size,
                &mut descriptor_version,
            )
        })?;
        if descriptor_size < core::mem::size_of::<MemoryDescriptor>()
            || map_size == 0
            || map_size > MAP_BYTES
            || map_size % descriptor_size != 0
        {
            return Err(LOAD_ERROR);
        }
        last_status = unsafe { (services.exit_boot_services)(image, map_key) };
        if !status_is_error(last_status) {
            return Ok((map_size, descriptor_size));
        }
    }
    Err(last_status)
}

fn check(status: Status) -> Result<(), Status> {
    if status_is_error(status) {
        Err(status)
    } else {
        Ok(())
    }
}

fn paint(framebuffer: Framebuffer, rgb: [u8; 3]) {
    let pixels = framebuffer
        .stride
        .checked_mul(framebuffer.height)
        .unwrap_or(0) as usize;
    let bytes = pixels.checked_mul(4).unwrap_or(0).min(framebuffer.size);
    if bytes == 0 {
        return;
    }
    let encoded = if framebuffer.format == 0 {
        [rgb[0], rgb[1], rgb[2], 0]
    } else {
        [rgb[2], rgb[1], rgb[0], 0]
    };
    // SAFETY: GOP guarantees the framebuffer allocation described by the
    // active mode. Length is capped to both calculated geometry and the
    // firmware-reported allocation.
    let framebuffer =
        unsafe { core::slice::from_raw_parts_mut(framebuffer.base as *mut u8, bytes) };
    for pixel in framebuffer.chunks_exact_mut(4) {
        pixel.copy_from_slice(&encoded);
    }
}

fn halt() -> ! {
    loop {
        core::hint::spin_loop();
    }
}
