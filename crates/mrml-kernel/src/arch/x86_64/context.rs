use super::{TrapFrame, VirtAddr};
use crate::{PAGE_SIZE, PhysAddr, TaskId};

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
}
