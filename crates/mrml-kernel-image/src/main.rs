#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
#[cfg(feature = "production-policy")]
use mrml_kernel::BootPolicy;
use mrml_kernel::arch::x86_64::{InterruptGate, install_fail_stop_tables};
#[cfg(not(feature = "fault-probe"))]
use mrml_kernel::{
    BootHandoff, Color, EarlyKernelContext, FramebufferSurface, HANDOFF_HEADER_BYTES,
    HANDOFF_REGION_BYTES, MAX_HANDOFF_REGIONS, MemoryKind, MemoryRegion, PhysAddr,
};

#[cfg(not(feature = "fault-probe"))]
const MAX_HANDOFF_BYTES: usize = HANDOFF_HEADER_BYTES + MAX_HANDOFF_REGIONS * HANDOFF_REGION_BYTES;

static GDT: [u64; 8] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0x00cf_9300_0000_ffff,
    0x00af_9b00_0000_ffff,
];
static mut IDT: [InterruptGate; 256] = [InterruptGate::MISSING; 256];

global_asm!(
    r#"
    .section .text
    .global mrml_exception_fail_stop
mrml_exception_fail_stop:
    cli
1:
    hlt
    jmp 1b
    "#
);

unsafe extern "C" {
    fn mrml_exception_fail_stop() -> !;
}

#[cfg(feature = "fault-probe")]
#[used]
static RELOCATION_ANCHOR: unsafe extern "C" fn() -> ! = mrml_exception_fail_stop;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

/// Standalone PE32+ kernel entry. The loader calls this only after firmware
/// services have exited and after authenticating both this image and handoff.
///
/// # Safety
///
/// `bytes` must address exactly `length` readable bytes that remain mapped for
/// this non-returning call. The loader must already have installed the final
/// W^X image mappings, a writable kernel stack, and a read-only handoff mapping.
#[unsafe(export_name = "efi_main")]
pub unsafe extern "efiapi" fn kernel_entry(bytes: *const u8, length: usize) -> usize {
    unsafe { install_descriptor_tables() };
    #[cfg(feature = "fault-probe")]
    unsafe {
        let _ = (bytes, length);
        // Keep one absolute pointer live so the minimal diagnostic PE exercises
        // the same DIR64 high-half relocation path as the full kernel.
        let _ = core::ptr::addr_of!(RELOCATION_ANCHOR).read_volatile();
        asm!("ud2", options(noreturn));
    }
    #[cfg(not(feature = "fault-probe"))]
    unsafe {
        run_kernel(bytes, length)
    }
}

#[cfg(not(feature = "fault-probe"))]
unsafe fn run_kernel(bytes: *const u8, length: usize) -> ! {
    if bytes.is_null() || !(HANDOFF_HEADER_BYTES..=MAX_HANDOFF_BYTES).contains(&length) {
        halt();
    }
    let encoded = unsafe { core::slice::from_raw_parts(bytes, length) };
    let placeholder = match MemoryRegion::new(
        match PhysAddr::new(0) {
            Ok(value) => value,
            Err(_) => halt(),
        },
        1,
        MemoryKind::Reserved,
    ) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    let mut regions = [placeholder; MAX_HANDOFF_REGIONS];
    let mut region_count = 0usize;
    let handoff = match BootHandoff::decode(encoded, |region| {
        if region_count < regions.len() {
            regions[region_count] = region;
            region_count += 1;
        }
    }) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    if region_count != handoff.region_count() {
        halt();
    }
    #[cfg(feature = "production-policy")]
    {
        let minimum_version = match embedded_minimum_version() {
            Some(value) => value,
            None => halt(),
        };
        if BootPolicy::production(minimum_version)
            .validate(handoff.evidence())
            .is_err()
        {
            halt();
        }
    }
    let framebuffer = handoff.framebuffer();
    let _early = match EarlyKernelContext::new(
        *handoff.evidence().entropy(),
        handoff.acpi_root(),
        framebuffer,
        &regions[..region_count],
    ) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    let framebuffer_bytes = unsafe {
        core::slice::from_raw_parts_mut(
            framebuffer.base().get() as *mut u8,
            framebuffer.byte_length() as usize,
        )
    };
    let mut surface = match FramebufferSurface::new(framebuffer, framebuffer_bytes) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    if surface
        .fill_rectangle(
            0,
            0,
            framebuffer.width(),
            framebuffer.height(),
            Color {
                red: 0x0b,
                green: 0x3b,
                blue: 0x5a,
            },
        )
        .is_err()
    {
        halt();
    }
    let _ = surface.fill_rectangle(
        0,
        0,
        framebuffer.width().min(96),
        framebuffer.height().min(12),
        Color {
            red: 0xff,
            green: 0xc8,
            blue: 0x57,
        },
    );
    halt()
}

#[cfg(all(not(feature = "fault-probe"), feature = "production-policy"))]
fn embedded_minimum_version() -> Option<u64> {
    let bytes = option_env!("MRML_KERNEL_MIN_VERSION")?.as_bytes();
    if bytes.is_empty() || bytes.len() > 20 {
        return None;
    }
    let mut value = 0u64;
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as u64)?;
    }
    (value != 0).then_some(value)
}

unsafe fn install_descriptor_tables() {
    let handler = mrml_exception_fail_stop as *const () as usize as u64;
    if unsafe {
        install_fail_stop_tables(
            core::ptr::addr_of!(GDT).cast::<u64>(),
            GDT.len(),
            core::ptr::addr_of_mut!(IDT).cast::<InterruptGate>(),
            256,
            handler,
            0x38,
        )
    }
    .is_err()
    {
        halt();
    }
}

fn halt() -> ! {
    loop {
        unsafe { asm!("cli", "hlt", options(nomem, nostack)) };
    }
}
