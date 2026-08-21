use crate::PAGE_SIZE;

pub const PRIVILEGE_STACK_ARENA_PAGES: u64 = 16;
const EARLY_STACK_PAGES: u64 = 6;
const PRIVILEGE_STACK_PAGES: u64 = 4;
const ENTRY_GUARD_PAGE: u64 = EARLY_STACK_PAGES;
const ENTRY_STACK_PAGE: u64 = ENTRY_GUARD_PAGE + 1;
const DOUBLE_FAULT_GUARD_PAGE: u64 = ENTRY_STACK_PAGE + PRIVILEGE_STACK_PAGES;
const DOUBLE_FAULT_STACK_PAGE: u64 = DOUBLE_FAULT_GUARD_PAGE + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrivilegeStackError {
    InvalidBase,
    InvalidSize,
    Overflow,
}

/// Fixed launch arena containing an early stack and two independently guarded
/// privilege stacks. Platform launchers must leave both guard pages unmapped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivilegeStackLayout {
    base: u64,
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
        PRIVILEGE_STACK_PAGES
    }
    pub fn entry_top(self) -> Result<u64, PrivilegeStackError> {
        self.offset(ENTRY_STACK_PAGE + PRIVILEGE_STACK_PAGES)
    }
    pub fn double_fault_guard(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_GUARD_PAGE)
    }
    pub fn double_fault_base(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_STACK_PAGE)
    }
    pub const fn double_fault_pages(self) -> u64 {
        PRIVILEGE_STACK_PAGES
    }
    pub fn double_fault_top(self) -> Result<u64, PrivilegeStackError> {
        self.offset(DOUBLE_FAULT_STACK_PAGE + PRIVILEGE_STACK_PAGES)
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
        let layout = PrivilegeStackLayout::new(0x40_0000, 16).unwrap();
        assert_eq!(layout.early_top(), Ok(0x40_5ff8));
        assert_eq!(layout.entry_guard(), Ok(0x40_6000));
        assert_eq!(layout.entry_base(), Ok(0x40_7000));
        assert_eq!(layout.entry_top(), Ok(0x40_b000));
        assert_eq!(layout.double_fault_guard(), Ok(0x40_b000));
        assert_eq!(layout.double_fault_base(), Ok(0x40_c000));
        assert_eq!(layout.double_fault_top(), Ok(0x41_0000));
    }

    #[test]
    fn layout_rejects_ambiguous_or_wrapping_arenas() {
        assert_eq!(
            PrivilegeStackLayout::new(0x40_0001, 16),
            Err(PrivilegeStackError::InvalidBase)
        );
        assert_eq!(
            PrivilegeStackLayout::new(0x40_0000, 15),
            Err(PrivilegeStackError::InvalidSize)
        );
        assert_eq!(
            PrivilegeStackLayout::new(!(PAGE_SIZE - 1), 16),
            Err(PrivilegeStackError::Overflow)
        );
    }
}
