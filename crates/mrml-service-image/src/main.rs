#![no_std]
#![no_main]

use core::arch::{asm, global_asm};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

// This entry is assembly rather than a Rust ABI function because r12-r14 are
// intentional launch registers, not compiler-owned callee-saved temporaries.
#[cfg(not(feature = "preemption-probe"))]
global_asm!(
    r#"
            .section .text
            .global efi_main
        efi_main:
            mov eax, 2
            xor edi, edi
            xor esi, esi
            xor edx, edx
            xor r10d, r10d
            xor r8d, r8d
            xor r9d, r9d
            int 0x80
            cmp rax, 0
            jne 3f
            cmp rdx, 4
            jne 3f
            cmp r10d, 0x676e6970
            jne 3f
            int3
        3:
            ud2

            // Fix the secondary entry ABI exactly; assembly fails if the
            // receiver ever grows beyond its reserved 128-byte window.
            .org efi_main + 128, 0xcc
            .global mrml_service_sender
        mrml_service_sender:
            mov eax, 1
            mov rdi, r13
            mov rsi, r14
            mov edx, 4
            mov r10d, 0x676e6970
            xor r8d, r8d
            xor r9d, r9d
            int 0x80
            xor eax, eax
            xor edi, edi
            xor esi, esi
            xor edx, edx
            xor r10d, r10d
            xor r8d, r8d
            xor r9d, r9d
            int 0x80
            ud2
    "#
);

#[cfg(feature = "preemption-probe")]
global_asm!(
    r#"
            .section .text
            .global efi_main
        efi_main:
        1:
            pause
            jmp 1b

            .org efi_main + 128, 0xcc
            .global mrml_service_sender
        mrml_service_sender:
            int3
        2:
            pause
            jmp 2b
    "#
);

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
