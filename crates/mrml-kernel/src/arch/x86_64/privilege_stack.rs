use crate::PAGE_SIZE;

pub const PRIVILEGE_STACK_ARENA_PAGES: u64 = 32;
pub const MAX_X86_64_CPUS: usize = 256;
const EARLY_STACK_PAGES: u64 = 6;
const ENTRY_STACK_PAGES: u64 = 16;
const DOUBLE_FAULT_STACK_PAGES: u64 = 8;
const ENTRY_GUARD_PAGE: u64 = EARLY_STACK_PAGES;
const ENTRY_STACK_PAGE: u64 = ENTRY_GUARD_PAGE + 1;
const DOUBLE_FAULT_GUARD_PAGE: u64 = ENTRY_STACK_PAGE + ENTRY_STACK_PAGES;
const DOUBLE_FAULT_STACK_PAGE: u64 = DOUBLE_FAULT_GUARD_PAGE + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivilegeStackError {
    InvalidBase,
    InvalidSize,
    Overflow,
    InvalidCpuCount,
    InvalidStride,
    InvalidCpu,
    NonCanonicalVirtualRange,
    InvalidPhysicalRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuPrivilegeStacks {
    cpu: u16,
    physical: PrivilegeStackLayout,
    virtual_layout: PrivilegeStackLayout,
}

impl CpuPrivilegeStacks {
    pub const fn cpu(self) -> u16 {
        self.cpu
    }
    pub const fn physical(self) -> PrivilegeStackLayout {
        self.physical
    }
    pub const fn virtual_layout(self) -> PrivilegeStackLayout {
        self.virtual_layout
    }
}

/// Bounded, arithmetic-only allocator for CPU-private privilege-stack arenas.
/// Physical and virtual arenas use the same stride but occupy independent
/// address namespaces. Requiring at least one complete arena per stride makes
/// aliasing between any two admitted CPU indices impossible by construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerCpuPrivilegeStacks<const CPUS: usize> {
    physical_base: u64,
    virtual_base: u64,
    stride_pages: u64,
}

impl<const CPUS: usize> PerCpuPrivilegeStacks<CPUS> {
    pub fn new(
        physical_base: u64,
        virtual_base: u64,
        stride_pages: u64,
    ) -> Result<Self, PrivilegeStackError> {
        if CPUS == 0 || CPUS > MAX_X86_64_CPUS {
            return Err(PrivilegeStackError::InvalidCpuCount);
        }
        if stride_pages < PRIVILEGE_STACK_ARENA_PAGES {
            return Err(PrivilegeStackError::InvalidStride);
        }
        PrivilegeStackLayout::new(physical_base, PRIVILEGE_STACK_ARENA_PAGES)?;
        PrivilegeStackLayout::new(virtual_base, PRIVILEGE_STACK_ARENA_PAGES)?;
        let last = (CPUS - 1) as u64;
        let last_offset = last
            .checked_mul(stride_pages)
            .and_then(|pages| pages.checked_mul(PAGE_SIZE))
            .ok_or(PrivilegeStackError::Overflow)?;
        let physical_last = physical_base
            .checked_add(last_offset)
            .ok_or(PrivilegeStackError::Overflow)?;
        let virtual_last = virtual_base
            .checked_add(last_offset)
            .ok_or(PrivilegeStackError::Overflow)?;
        let physical_layout =
            PrivilegeStackLayout::new(physical_last, PRIVILEGE_STACK_ARENA_PAGES)?;
        let physical_end = physical_layout.double_fault_top()?;
        if physical_base >> 52 != 0 || (physical_end - 1) >> 52 != 0 {
            return Err(PrivilegeStackError::InvalidPhysicalRange);
        }
        let virtual_layout = PrivilegeStackLayout::new(virtual_last, PRIVILEGE_STACK_ARENA_PAGES)?;
        let virtual_end = virtual_layout.double_fault_top()?;
        if !canonical(virtual_base) || !canonical(virtual_end - 1) {
            return Err(PrivilegeStackError::NonCanonicalVirtualRange);
        }
        Ok(Self {
            physical_base,
            virtual_base,
            stride_pages,
        })
    }

    pub fn cpu(&self, cpu: usize) -> Result<CpuPrivilegeStacks, PrivilegeStackError> {
        if cpu >= CPUS {
            return Err(PrivilegeStackError::InvalidCpu);
        }
        let offset = (cpu as u64)
            .checked_mul(self.stride_pages)
            .and_then(|pages| pages.checked_mul(PAGE_SIZE))
            .ok_or(PrivilegeStackError::Overflow)?;
        Ok(CpuPrivilegeStacks {
            cpu: cpu as u16,
            physical: PrivilegeStackLayout::new(
                self.physical_base
                    .checked_add(offset)
                    .ok_or(PrivilegeStackError::Overflow)?,
                PRIVILEGE_STACK_ARENA_PAGES,
            )?,
            virtual_layout: PrivilegeStackLayout::new(
                self.virtual_base
                    .checked_add(offset)
                    .ok_or(PrivilegeStackError::Overflow)?,
                PRIVILEGE_STACK_ARENA_PAGES,
            )?,
        })
    }
}

/// Fixed launch arena containing an early stack and two independently guarded
/// privilege stacks. Platform launchers must leave both guard pages unmapped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivilegeStackLayout {
    base: u64,
}

const fn canonical(address: u64) -> bool {
    ((address << 16) as i64 >> 16) as u64 == address
}

impl PrivilegeStackLayout {
    pub fn new(base: u64, pages: u64) -> Result<Self, PrivilegeStackError> {
        if base == 0 || !base.is_multiple_of(PAGE_SIZE) {
            return Err(PrivilegeStackError::InvalidBase);
        }
        if pages != PRIVILEGE_STACK_ARENA_PAGES {
            return Err(PrivilegeStackError::InvalidSize);
        }
        base.checked_add(
            pages
                .checked_mul(PAGE_SIZE)
                .ok_or(PrivilegeStackError::Overflow)?,
        )
        .ok_or(PrivilegeStackError::Overflow)?;
        Ok(Self { base })
    }

    pub const fn early_base(self) -> u64 {
        self.base
    }
    pub const fn early_pages(self) -> u64 {
        EARLY_STACK_PAGES
    }
    pub fn early_top(self) -> Result<u64, PrivilegeStackError> {
        self.offset(EARLY_STACK_PAGES)?
            .checked_sub(8)
            .ok_or(PrivilegeStackError::Overflow)
    }
    pub fn entry_guard(self) -> Result<u64, PrivilegeStackError> {
        self.offset(ENTRY_GUARD_PAGE)
    }
    pub fn entry_base(self) -> Result<u64, PrivilegeStackError> {
        self.offset(ENTRY_STACK_PAGE)
    }
    pub const fn entry_pages(self) -> u64 {
        ENTRY_STACK_PAGES
    }
    pub fn entry_top(self) -> Result<u64, PrivilegeStackError> {
        self.offset(ENTRY_STACK_PAGE + ENTRY_STACK_PAGES)
    }
    pub fn double_fault_guard(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_GUARD_PAGE)
    }
    pub fn double_fault_base(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_STACK_PAGE)
    }
    pub const fn double_fault_pages(self) -> u64 {
        DOUBLE_FAULT_STACK_PAGES
    }
    pub fn double_fault_top(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_STACK_PAGE + DOUBLE_FAULT_STACK_PAGES)
    }

    fn offset(self, pages: u64) -> Result<u64, PrivilegeStackError> {
        self.base
            .checked_add(
                pages
                    .checked_mul(PAGE_SIZE)
                    .ok_or(PrivilegeStackError::Overflow)?,
            )
            .ok_or(PrivilegeStackError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_has_two_absent_guards_and_disjoint_stacks() {
        let layout = PrivilegeStackLayout::new(0x40_0000, 32).unwrap();
        assert_eq!(layout.early_top(), Ok(0x40_5ff8));
        assert_eq!(layout.entry_guard(), Ok(0x40_6000));
        assert_eq!(layout.entry_base(), Ok(0x40_7000));
        assert_eq!(layout.entry_top(), Ok(0x41_7000));
        assert_eq!(layout.double_fault_guard(), Ok(0x41_7000));
        assert_eq!(layout.double_fault_base(), Ok(0x41_8000));
        assert_eq!(layout.double_fault_top(), Ok(0x42_0000));
    }

    #[test]
    fn layout_rejects_ambiguous_or_wrapping_arenas() {
        assert_eq!(
            PrivilegeStackLayout::new(0x40_0001, 32),
            Err(PrivilegeStackError::InvalidBase)
        );
        assert_eq!(
            PrivilegeStackLayout::new(0x40_0000, 31),
            Err(PrivilegeStackError::InvalidSize)
        );
        assert_eq!(
            PrivilegeStackLayout::new(!(PAGE_SIZE - 1), 32),
            Err(PrivilegeStackError::Overflow)
        );
    }

    #[test]
    fn per_cpu_arenas_are_indexed_disjoint_and_guarded() {
        let set = PerCpuPrivilegeStacks::<4>::new(
            0x40_0000,
            0xffff_8001_6000_0000,
            PRIVILEGE_STACK_ARENA_PAGES + 2,
        )
        .unwrap();
        for cpu in 0..4 {
            let current = set.cpu(cpu).unwrap();
            assert_eq!(current.cpu(), cpu as u16);
            assert_eq!(
                current.physical().entry_guard().unwrap() + 0xffff_8001_5fc0_0000,
                current.virtual_layout().entry_guard().unwrap()
            );
            if cpu != 0 {
                let previous = set.cpu(cpu - 1).unwrap();
                assert!(
                    previous.physical().double_fault_top().unwrap()
                        <= current.physical().early_base()
                );
                assert!(
                    previous.virtual_layout().double_fault_top().unwrap()
                        <= current.virtual_layout().early_base()
                );
            }
        }
        assert_eq!(set.cpu(4), Err(PrivilegeStackError::InvalidCpu));
    }

    #[test]
    fn per_cpu_arenas_reject_counts_strides_wrap_and_noncanonical_ranges() {
        assert_eq!(
            PerCpuPrivilegeStacks::<0>::new(0x40_0000, 0xffff_8001_6000_0000, 32),
            Err(PrivilegeStackError::InvalidCpuCount)
        );
        assert_eq!(
            PerCpuPrivilegeStacks::<257>::new(0x40_0000, 0xffff_8001_6000_0000, 32),
            Err(PrivilegeStackError::InvalidCpuCount)
        );
        assert_eq!(
            PerCpuPrivilegeStacks::<2>::new(0x40_0000, 0xffff_8001_6000_0000, 31),
            Err(PrivilegeStackError::InvalidStride)
        );
        assert_eq!(
            PerCpuPrivilegeStacks::<2>::new(0x40_0000, 0xffff_8001_6000_0000, u64::MAX / PAGE_SIZE,),
            Err(PrivilegeStackError::Overflow)
        );
        assert_eq!(
            PerCpuPrivilegeStacks::<1>::new(0x40_0000, 0x0000_8000_0000_0000, 32),
            Err(PrivilegeStackError::NonCanonicalVirtualRange)
        );
        assert_eq!(
            PerCpuPrivilegeStacks::<1>::new(1u64 << 52, 0xffff_8001_6000_0000, 32),
            Err(PrivilegeStackError::InvalidPhysicalRange)
        );
    }
}
