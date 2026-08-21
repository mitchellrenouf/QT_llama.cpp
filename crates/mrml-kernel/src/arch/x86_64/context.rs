use super::{TrapFrame, VirtAddr};
use crate::{PAGE_SIZE, PhysAddr, TaskId, UserCallFrame};
use core::arch::{asm, global_asm};

global_asm!(
    r#"
    .section .text
    .global mrml_x86_enter_user
mrml_x86_enter_user:
    mov rdx, rdi
    mov rax, qword ptr [rdx + 144]
    mov cr3, rax
    push 0x1b
    push qword ptr [rdx + 128]
    push qword ptr [rdx + 136]
    push 0x23
    push qword ptr [rdx + 120]
    mov ax, 0x1b
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov r15, qword ptr [rdx + 0]
    mov r14, qword ptr [rdx + 8]
    mov r13, qword ptr [rdx + 16]
    mov r12, qword ptr [rdx + 24]
    mov r11, qword ptr [rdx + 32]
    mov r10, qword ptr [rdx + 40]
    mov r9, qword ptr [rdx + 48]
    mov r8, qword ptr [rdx + 56]
    mov rsi, qword ptr [rdx + 72]
    mov rbp, qword ptr [rdx + 80]
    mov rbx, qword ptr [rdx + 88]
    mov rcx, qword ptr [rdx + 104]
    mov rax, qword ptr [rdx + 112]
    mov rdi, qword ptr [rdx + 64]
    mov rdx, qword ptr [rdx + 96]
    iretq
    ud2

    .global mrml_x86_enter_user_on_stack
mrml_x86_enter_user_on_stack:
    mov rsp, rsi
    jmp mrml_x86_enter_user
    "#
);

unsafe extern "sysv64" {
    fn mrml_x86_enter_user(context: *const u64) -> !;
    fn mrml_x86_enter_user_on_stack(context: *const u64, stack: u64) -> !;
}

pub const USER_DATA_SELECTOR: u16 = 0x1b;
pub const USER_CODE_SELECTOR: u16 = 0x23;
pub const USER_INITIAL_RFLAGS: u64 = 0x202;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextError {
    KernelPageTable,
    InvalidEntry,
    InvalidStack,
    InvalidSelectors,
    InvalidFlags,
    DuplicateTask,
    MissingTask,
    TableFull,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct UserContext {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rax: u64,
    rip: u64,
    rsp: u64,
    rflags: u64,
    page_table: PhysAddr,
}

impl UserContext {
    pub fn new(page_table: PhysAddr, entry: u64, stack: u64) -> Result<Self, ContextError> {
        if page_table.get() == 0 {
            return Err(ContextError::KernelPageTable);
        }
        validate_user_address(entry).map_err(|_| ContextError::InvalidEntry)?;
        validate_user_address(stack).map_err(|_| ContextError::InvalidStack)?;
        if stack == 0 || !stack.is_multiple_of(16) {
            return Err(ContextError::InvalidStack);
        }
        Ok(Self {
            r15: 0,
            r14: 0,
            r13: 0,
            r12: 0,
            r11: 0,
            r10: 0,
            r9: 0,
            r8: 0,
            rdi: 0,
            rsi: 0,
            rbp: 0,
            rbx: 0,
            rdx: 0,
            rcx: 0,
            rax: 0,
            rip: entry,
            rsp: stack,
            rflags: USER_INITIAL_RFLAGS,
            page_table,
        })
    }

    pub fn from_trap(page_table: PhysAddr, frame: &TrapFrame) -> Result<Self, ContextError> {
        if frame.cs != u64::from(USER_CODE_SELECTOR) || frame.ss != u64::from(USER_DATA_SELECTOR) {
            return Err(ContextError::InvalidSelectors);
        }
        validate_user_address(frame.rip).map_err(|_| ContextError::InvalidEntry)?;
        validate_user_address(frame.rsp).map_err(|_| ContextError::InvalidStack)?;
        validate_user_flags(frame.rflags)?;
        if page_table.get() == 0 {
            return Err(ContextError::KernelPageTable);
        }
        Ok(Self {
            r15: frame.r15,
            r14: frame.r14,
            r13: frame.r13,
            r12: frame.r12,
            r11: frame.r11,
            r10: frame.r10,
            r9: frame.r9,
            r8: frame.r8,
            rdi: frame.rdi,
            rsi: frame.rsi,
            rbp: frame.rbp,
            rbx: frame.rbx,
            rdx: frame.rdx,
            rcx: frame.rcx,
            rax: frame.rax,
            rip: frame.rip,
            rsp: frame.rsp,
            rflags: frame.rflags,
            page_table,
        })
    }

    pub fn from_user_call(
        page_table: PhysAddr,
        frame: &UserCallFrame,
    ) -> Result<Self, ContextError> {
        frame.validate_return().map_err(|error| match error {
            crate::SyscallError::InvalidPrivilege => ContextError::InvalidSelectors,
            crate::SyscallError::InvalidInstruction => ContextError::InvalidEntry,
            crate::SyscallError::InvalidStack => ContextError::InvalidStack,
            _ => ContextError::InvalidFlags,
        })?;
        if page_table.get() == 0 {
            return Err(ContextError::KernelPageTable);
        }
        Ok(Self {
            r15: frame.r15,
            r14: frame.r14,
            r13: frame.r13,
            r12: frame.r12,
            r11: frame.r11,
            r10: frame.r10,
            r9: frame.r9,
            r8: frame.r8,
            rdi: frame.rdi,
            rsi: frame.rsi,
            rbp: frame.rbp,
            rbx: frame.rbx,
            rdx: frame.rdx,
            rcx: frame.rcx,
            rax: frame.rax,
            rip: frame.rip,
            rsp: frame.rsp,
            rflags: frame.rflags,
            page_table,
        })
    }

    pub fn complete_message(&mut self, payload: &[u8]) {
        let mut words = [0u64; 3];
        for (index, byte) in payload.iter().take(24).enumerate() {
            words[index / 8] |= u64::from(*byte) << ((index % 8) * 8);
        }
        self.rax = 0;
        self.rdx = payload.len().min(24) as u64;
        self.r10 = words[0];
        self.r8 = words[1];
        self.r9 = words[2];
    }

    pub const fn instruction_pointer(&self) -> u64 {
        self.rip
    }

    pub const fn stack_pointer(&self) -> u64 {
        self.rsp
    }

    pub const fn flags(&self) -> u64 {
        self.rflags
    }

    pub const fn page_table(&self) -> PhysAddr {
        self.page_table
    }
}

/// Installs a validated task address space and restores its complete CPL3
/// context. This function cannot return.
///
/// # Safety
///
/// The context's page-table root must map this transition code until the CR3
/// write and must map `rip` user-executable and `rsp` user-writable. The active
/// TSS must contain a mapped kernel-only `RSP0`, and interrupts must remain
/// disabled until `IRETQ` applies the context's validated flags.
pub unsafe fn enter_user_context(context: &UserContext) -> ! {
    unsafe { mrml_x86_enter_user((context as *const UserContext).cast::<u64>()) }
}

/// Moves to a kernel-only transition stack shared by the old and new roots
/// before installing CR3 and restoring the user context.
///
/// # Safety
///
/// `transition_stack` must be a nonzero canonical, 16-byte-aligned stack top
/// mapped writable/supervisor in both address spaces. The remaining safety
/// requirements of [`enter_user_context`] also apply.
pub unsafe fn enter_user_context_on_stack(context: &UserContext, transition_stack: u64) -> ! {
    if transition_stack == 0
        || !transition_stack.is_multiple_of(16)
        || ((transition_stack << 16) as i64 >> 16) as u64 != transition_stack
    {
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
    unsafe {
        mrml_x86_enter_user_on_stack(
            (context as *const UserContext).cast::<u64>(),
            transition_stack,
        )
    }
}

pub struct UserContextTable<const TASKS: usize> {
    entries: [Option<(TaskId, UserContext)>; TASKS],
}

impl<const TASKS: usize> UserContextTable<TASKS> {
    pub const fn new() -> Self {
        Self {
            entries: [None; TASKS],
        }
    }

    pub fn bind(&mut self, task: TaskId, context: UserContext) -> Result<(), ContextError> {
        if self.entries.iter().flatten().any(|entry| entry.0 == task) {
            return Err(ContextError::DuplicateTask);
        }
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.is_none())
            .ok_or(ContextError::TableFull)?;
        *entry = Some((task, context));
        Ok(())
    }

    pub fn get(&self, task: TaskId) -> Result<&UserContext, ContextError> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| entry.0 == task)
            .map(|entry| &entry.1)
            .ok_or(ContextError::MissingTask)
    }

    pub fn replace(&mut self, task: TaskId, context: UserContext) -> Result<(), ContextError> {
        let entry = self
            .entries
            .iter_mut()
            .flatten()
            .find(|entry| entry.0 == task)
            .ok_or(ContextError::MissingTask)?;
        entry.1 = context;
        Ok(())
    }

    pub fn revoke(&mut self, task: TaskId) -> Result<UserContext, ContextError> {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.is_some_and(|entry| entry.0 == task))
            .ok_or(ContextError::MissingTask)?;
        Ok(entry.take().unwrap().1)
    }
}

impl<const TASKS: usize> Default for UserContextTable<TASKS> {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_user_address(address: u64) -> Result<(), ()> {
    if address == 0 || address >= 1 << 47 {
        return Err(());
    }
    // `VirtAddr` performs the architecture's canonicality check. Round down
    // solely for validation because instruction and stack pointers need not be
    // page aligned.
    VirtAddr::new(address & !(PAGE_SIZE - 1)).map_err(|_| ())?;
    Ok(())
}

fn validate_user_flags(flags: u64) -> Result<(), ContextError> {
    const ALLOWED: u64 = 2
        | (1 << 0)
        | (1 << 2)
        | (1 << 4)
        | (1 << 6)
        | (1 << 7)
        | (1 << 8)
        | (1 << 9)
        | (1 << 10)
        | (1 << 11)
        | (1 << 18)
        | (1 << 21);
    if flags & 2 == 0 || flags & !ALLOWED != 0 {
        Err(ContextError::InvalidFlags)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelScheduler, Priority};

    fn user_frame() -> TrapFrame {
        let mut frame = TrapFrame {
            r15: 15,
            r14: 14,
            r13: 13,
            r12: 12,
            r11: 11,
            r10: 10,
            r9: 9,
            r8: 8,
            rdi: 7,
            rsi: 6,
            rbp: 5,
            rbx: 4,
            rdx: 3,
            rcx: 2,
            rax: 1,
            vector: 14,
            error: 7,
            rip: 0x400123,
            cs: u64::from(USER_CODE_SELECTOR),
            rflags: USER_INITIAL_RFLAGS,
            rsp: 0x7fff_ffff_f000,
            ss: u64::from(USER_DATA_SELECTOR),
        };
        frame.rflags |= 1 << 9;
        frame
    }

    #[test]
    fn initial_context_is_ring_three_and_fail_closed() {
        let root = PhysAddr::new(0x20_0000).unwrap();
        let context = UserContext::new(root, 0x40_0000, 0x7fff_ffff_f000).unwrap();
        assert_eq!(context.page_table(), root);
        assert_eq!(context.flags(), USER_INITIAL_RFLAGS);
        assert_eq!(
            UserContext::new(PhysAddr::new(0).unwrap(), 0x40_0000, 0x7fff_ffff_f000),
            Err(ContextError::KernelPageTable)
        );
        assert_eq!(
            UserContext::new(root, 1 << 47, 0x7fff_ffff_f000),
            Err(ContextError::InvalidEntry)
        );
        assert_eq!(
            UserContext::new(root, 0x40_0000, 0x7fff_ffff_eff8),
            Err(ContextError::InvalidStack)
        );
    }

    #[test]
    fn trap_conversion_rejects_privilege_and_flag_forgery() {
        let root = PhysAddr::new(0x20_0000).unwrap();
        let frame = user_frame();
        let context = UserContext::from_trap(root, &frame).unwrap();
        assert_eq!(context.instruction_pointer(), frame.rip);
        assert_eq!(context.rax, 1);
        let mut forged = frame;
        forged.cs = 0x38;
        assert_eq!(
            UserContext::from_trap(root, &forged),
            Err(ContextError::InvalidSelectors)
        );
        forged = frame;
        forged.rflags |= 3 << 12;
        assert_eq!(
            UserContext::from_trap(root, &forged),
            Err(ContextError::InvalidFlags)
        );
        forged = frame;
        forged.rflags |= 1 << 63;
        assert_eq!(
            UserContext::from_trap(root, &forged),
            Err(ContextError::InvalidFlags)
        );
    }

    #[test]
    fn user_call_conversion_preserves_the_exact_post_interrupt_context() {
        let root = PhysAddr::new(0x30_0000).unwrap();
        let frame = crate::UserCallFrame {
            r15: 15,
            r14: 14,
            r13: 13,
            r12: 12,
            r11: 11,
            r10: 10,
            r9: 9,
            r8: 8,
            rdi: 7,
            rsi: 6,
            rbp: 5,
            rbx: 4,
            rdx: 0,
            rcx: 3,
            rax: 2,
            rip: 0x40_1000,
            cs: u64::from(USER_CODE_SELECTOR),
            rflags: USER_INITIAL_RFLAGS,
            rsp: 0x7000_0000,
            ss: u64::from(USER_DATA_SELECTOR),
        };
        let context = UserContext::from_user_call(root, &frame).unwrap();
        assert_eq!(context.page_table(), root);
        assert_eq!(context.instruction_pointer(), frame.rip);
        assert_eq!(context.stack_pointer(), frame.rsp);
        assert_eq!(context.r15, 15);
        assert_eq!(context.rax, 2);
    }

    #[test]
    fn context_binding_tracks_generational_task_identity() {
        let root = PhysAddr::new(0x20_0000).unwrap();
        let context = UserContext::new(root, 0x40_0000, 0x7fff_ffff_f000).unwrap();
        let mut scheduler = KernelScheduler::<1>::new(100, 1).unwrap();
        let old = scheduler.create(Priority::NORMAL).unwrap();
        let mut contexts = UserContextTable::<1>::new();
        contexts.bind(old, context).unwrap();
        assert_eq!(
            contexts.bind(old, context),
            Err(ContextError::DuplicateTask)
        );
        assert_eq!(contexts.revoke(old), Ok(context));
        scheduler.start();
        scheduler.terminate_current().unwrap();
        let replacement = scheduler.create(Priority::NORMAL).unwrap();
        assert_ne!(old, replacement);
        assert_eq!(contexts.get(old), Err(ContextError::MissingTask));
        contexts.bind(replacement, context).unwrap();
        assert_eq!(contexts.get(replacement), Ok(&context));
    }

    #[test]
    fn user_context_layout_matches_transition_assembly() {
        assert_eq!(core::mem::size_of::<UserContext>(), 19 * 8);
        assert_eq!(core::mem::offset_of!(UserContext, r15), 0);
        assert_eq!(core::mem::offset_of!(UserContext, rdi), 64);
        assert_eq!(core::mem::offset_of!(UserContext, rdx), 96);
        assert_eq!(core::mem::offset_of!(UserContext, rax), 112);
        assert_eq!(core::mem::offset_of!(UserContext, rip), 120);
        assert_eq!(core::mem::offset_of!(UserContext, rsp), 128);
        assert_eq!(core::mem::offset_of!(UserContext, rflags), 136);
        assert_eq!(core::mem::offset_of!(UserContext, page_table), 144);
    }
}
