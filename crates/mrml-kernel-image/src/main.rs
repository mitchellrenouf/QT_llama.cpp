#![no_std]
#![no_main]

use mrml_kernel::{
    BootHandoff, Color, FramebufferSurface, HANDOFF_HEADER_BYTES, HANDOFF_REGION_BYTES,
    MAX_HANDOFF_REGIONS,
};

const MAX_HANDOFF_BYTES: usize = HANDOFF_HEADER_BYTES + MAX_HANDOFF_REGIONS * HANDOFF_REGION_BYTES;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

/// Standalone PE32+ kernel entry. The loader calls this only after firmware
/// services have exited and after authenticating both this image and handoff.
#[unsafe(export_name = "efi_main")]
pub unsafe extern "efiapi" fn kernel_entry(bytes: *const u8, length: usize) -> usize {
    if bytes.is_null() || !(HANDOFF_HEADER_BYTES..=MAX_HANDOFF_BYTES).contains(&length) {
        halt();
    }
    let encoded = unsafe { core::slice::from_raw_parts(bytes, length) };
    let handoff = match BootHandoff::decode(encoded, |_| {}) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    let framebuffer = handoff.framebuffer();
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

fn halt() -> ! {
    loop {
        core::hint::spin_loop();
    }
}
