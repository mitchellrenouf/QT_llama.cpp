#![no_std]
#![no_main]

use core::arch::asm;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

/// Minimal independently signed service entry used to prove that the kernel
/// can enter a separate user PE under its own CR3. Production services replace
/// the deliberate breakpoint with the pointer-free user-call ABI.
#[unsafe(export_name = "efi_main")]
pub extern "efiapi" fn service_entry() -> usize {
    unsafe { asm!("int3", options(noreturn)) }
}

fn halt() -> ! {
    unsafe {
        asm!(
            "cli",
            "2:",
            "hlt",
            "jmp 2b",
            options(noreturn, nomem, nostack)
        )
    }
}
