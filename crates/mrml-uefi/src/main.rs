#![no_std]
#![no_main]

use core::cell::UnsafeCell;
use mrml_kernel::{
    EarlyKernelContext, FramebufferInfo, MemoryKind, MemoryRegion, PhysAddr, PixelFormat,
};
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

#[derive(Clone, Copy)]
struct Framebuffer {
    base: u64,
    size: usize,
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
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

    paint(framebuffer, [0x14, 0x21, 0x35]);

    let acpi_root = unsafe { find_acpi_root(system) }.map_err(|_| LOAD_ERROR)?;
    if acpi_root == 0 {
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
    enter_kernel(entropy, acpi_root, framebuffer, &regions[..region_count])?;
    halt()
}

fn enter_kernel(
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
    let context = EarlyKernelContext::new(
        entropy,
        acpi_root,
        framebuffer_info,
        &kernel_regions[..regions.len()],
    )
    .map_err(|_| LOAD_ERROR)?;
    // SAFETY: GOP supplied this allocation and boot services have exited. The
    // kernel surface checks the declared size and every rendered rectangle.
    let framebuffer_bytes =
        unsafe { core::slice::from_raw_parts_mut(framebuffer.base as *mut u8, framebuffer.size) };
    context
        .render_booted(framebuffer_bytes)
        .map_err(|_| LOAD_ERROR)
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
