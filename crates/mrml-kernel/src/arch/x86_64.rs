use crate::{PAGE_SIZE, PhysAddr};

mod acpi;
mod address_space;
mod ap_online;
mod ap_trampoline;
mod context;
mod descriptors;
mod local_apic;
mod page_table;
mod pe_mapping;
mod privilege_stack;
mod service_space;
mod topology;
mod trap;
pub use acpi::{AcpiError, AcpiMemory, copy_madt};
pub use address_space::{AddressSpace, AddressSpaceError, Mapping, MappingId};
pub use ap_online::{ApOnlineError, ApOnlineTable};
pub use ap_trampoline::{
    ActiveApTrampolinePage, ApTrampolineError, ApTrampolineImage, ApTrampolinePage,
    InstalledApTrampoline, TrampolinePermissions,
};
pub use context::{
    ContextError, USER_CODE_SELECTOR, USER_DATA_SELECTOR, USER_INITIAL_RFLAGS, UserContext,
    UserContextTable, enter_user_context, enter_user_context_on_stack,
};
pub use descriptors::{
    AlignedTaskState, CPU_GDT_ENTRIES, CPU_TSS_SELECTOR, CpuDescriptorState, DescriptorError,
    InterruptGate, TaskStateSegment, install_exception_tables, install_external_interrupt_gate,
    install_fail_stop_tables, install_user_call_gate, load_task_register, task_state_descriptor,
    write_task_state_descriptor,
};
pub use local_apic::{
    ApStartupTiming, ApicIpi, LocalApicController, LocalApicError, LocalApicTimer, TimerDivide,
};
pub use page_table::{
    ActiveLeaf, ActivePageTables, PageTableBuildError, PageTableBuilder, PageTableStore,
};
pub use pe_mapping::{PeMappingError, map_pe_image};
pub use privilege_stack::{
    CpuPrivilegeStacks, EARLY_STACK_PAGES, MAX_X86_64_CPUS, PRIVILEGE_STACK_ARENA_PAGES,
    PerCpuPrivilegeStacks, PrivilegeStackError, PrivilegeStackLayout,
};
pub use service_space::{ServiceAddressSpace, ServiceSpaceError};
pub use topology::{
    ApStartupTable, ApStartupToken, ApState, TopologyError, X86Cpu, X86CpuTopology,
};
pub use trap::{HardwareTrapFrame, TrapDisposition, TrapError, TrapFrame};

const MAX_PHYSICAL_ADDRESS: u64 = (1u64 << 52) - PAGE_SIZE;
const ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VirtAddr(u64);

impl VirtAddr {
    pub const fn new(address: u64) -> Result<Self, PageError> {
        let high = address >> 48;
        let sign = (address >> 47) & 1;
        if (sign == 0 && high != 0) || (sign == 1 && high != 0xffff) {
            return Err(PageError::NonCanonical);
        }
        if !address.is_multiple_of(PAGE_SIZE) {
            return Err(PageError::Unaligned);
        }
        Ok(Self(address))
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn pml4_index(self) -> usize {
        ((self.0 >> 39) & 0x1ff) as usize
    }
    pub const fn pdpt_index(self) -> usize {
        ((self.0 >> 30) & 0x1ff) as usize
    }
    pub const fn directory_index(self) -> usize {
        ((self.0 >> 21) & 0x1ff) as usize
    }
    pub const fn table_index(self) -> usize {
        ((self.0 >> 12) & 0x1ff) as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagePermissions(u8);

impl PagePermissions {
    pub const KERNEL_READ: Self = Self(0);
    pub const KERNEL_READ_WRITE: Self = Self(1 << 0);
    pub const KERNEL_READ_EXECUTE: Self = Self(1 << 1);
    pub const KERNEL_LOW_READ_WRITE: Self = Self((1 << 0) | (1 << 3));
    pub const KERNEL_LOW_READ_EXECUTE: Self = Self((1 << 1) | (1 << 3));
    pub const KERNEL_SHARED_READ: Self = Self(1 << 3);
    pub const KERNEL_SHARED_READ_WRITE: Self = Self((1 << 0) | (1 << 3));
    pub const KERNEL_MMIO_READ_WRITE: Self = Self((1 << 0) | (1 << 3));
    pub const USER_READ: Self = Self(1 << 2);
    pub const USER_READ_WRITE: Self = Self((1 << 2) | (1 << 0));
    pub const USER_READ_EXECUTE: Self = Self((1 << 2) | (1 << 1));

    pub const fn writable(self) -> bool {
        self.0 & (1 << 0) != 0
    }

    pub const fn executable(self) -> bool {
        self.0 & (1 << 1) != 0
    }

    pub const fn user(self) -> bool {
        self.0 & (1 << 2) != 0
    }

    pub const fn low_supervisor_mmio(self) -> bool {
        self.0 & (1 << 3) != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageError {
    NonCanonical,
    Unaligned,
    PhysicalAddressTooLarge,
    WritableExecutable,
    InvalidEntry,
}

/// A hardware-format 4 KiB leaf entry. Constructors enforce W^X and set NX
/// unless execution was explicitly requested.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    pub const fn leaf(frame: PhysAddr, permissions: PagePermissions) -> Result<Self, PageError> {
        if frame.get() > MAX_PHYSICAL_ADDRESS {
            return Err(PageError::PhysicalAddressTooLarge);
        }
        if permissions.writable() && permissions.executable() {
            return Err(PageError::WritableExecutable);
        }
        let mut bits = frame.get() | 1;
        if permissions.writable() {
            bits |= 1 << 1;
        }
        if permissions.user() {
            bits |= 1 << 2;
        }
        if !permissions.executable() {
            bits |= 1 << 63;
        }
        Ok(Self(bits))
    }

    pub const fn table(frame: PhysAddr, user: bool) -> Result<Self, PageError> {
        if frame.get() > MAX_PHYSICAL_ADDRESS {
            return Err(PageError::PhysicalAddressTooLarge);
        }
        Ok(Self(frame.get() | 1 | (1 << 1) | ((user as u64) << 2)))
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn frame(self) -> Result<PhysAddr, PageError> {
        if self.0 & 1 == 0 {
            return Err(PageError::InvalidEntry);
        }
        match PhysAddr::new(self.0 & ADDRESS_MASK) {
            Ok(address) => Ok(address),
            Err(_) => Err(PageError::InvalidEntry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_addresses_must_be_canonical_and_page_aligned() {
        assert!(VirtAddr::new(0x0000_7fff_ffff_f000).is_ok());
        assert!(VirtAddr::new(0xffff_8000_0000_0000).is_ok());
        assert_eq!(
            VirtAddr::new(0x0000_8000_0000_0000),
            Err(PageError::NonCanonical)
        );
        assert_eq!(VirtAddr::new(1), Err(PageError::Unaligned));
    }

    #[test]
    fn leaf_entries_are_nx_by_default_and_never_writable_executable() {
        let frame = PhysAddr::new(0x2000).unwrap();
        let data = PageTableEntry::leaf(frame, PagePermissions::USER_READ_WRITE).unwrap();
        assert_ne!(data.bits() & (1 << 63), 0);
        assert_ne!(data.bits() & (1 << 1), 0);
        let code = PageTableEntry::leaf(frame, PagePermissions::USER_READ_EXECUTE).unwrap();
        assert_eq!(code.bits() & (1 << 63), 0);
        assert_eq!(code.bits() & (1 << 1), 0);
        let forged = PagePermissions((1 << 0) | (1 << 1));
        assert_eq!(
            PageTableEntry::leaf(frame, forged),
            Err(PageError::WritableExecutable)
        );
        assert_eq!(data.frame(), Ok(frame));
    }
}
