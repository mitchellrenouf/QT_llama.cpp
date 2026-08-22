use core::arch::asm;

use super::{ADDRESS_MASK, Mapping, PageError, PagePermissions, PageTableEntry, VirtAddr};
use crate::{PAGE_SIZE, PhysAddr};

const LEAF_ACCESSED_DIRTY: u64 = (1 << 5) | (1 << 6);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageTableBuildError {
    Storage,
    InvalidIntermediate,
    AlreadyMapped,
    NotMapped,
    MappingMismatch,
    AddressOverflow,
    Page(PageError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveLeaf {
    physical: PhysAddr,
    writable: bool,
    executable: bool,
    user: bool,
}

impl ActiveLeaf {
    pub const fn physical(self) -> PhysAddr {
        self.physical
    }
    pub const fn writable(self) -> bool {
        self.writable
    }
    pub const fn executable(self) -> bool {
        self.executable
    }
    pub const fn user(self) -> bool {
        self.user
    }
}

/// Minimal storage contract for physical 4 KiB page-table frames. A platform
/// implementation must return a new zeroed frame on every allocation and make
/// reads/writes address the direct physical table contents.
pub trait PageTableStore {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError>;
    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError>;
    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError>;
}

/// Monotonic page-table storage backed by a caller-owned, identity-mapped
/// physical arena. This is used after firmware allocation is unavailable.
pub struct PreallocatedPageTableStore {
    base: PhysAddr,
    capacity: usize,
    allocated: usize,
}

impl PreallocatedPageTableStore {
    /// Opens a zeroable arena for exclusive page-table construction.
    ///
    /// # Safety
    ///
    /// Every page in `[base, base + pages * PAGE_SIZE)` must be identity
    /// mapped, supervisor-writable, exclusively owned by the caller, and not
    /// observable by another CPU for the lifetime of this store.
    pub unsafe fn new(base: PhysAddr, pages: u64) -> Result<Self, PageTableBuildError> {
        let capacity = usize::try_from(pages).map_err(|_| PageTableBuildError::Storage)?;
        if capacity == 0
            || capacity > 256
            || !base.get().is_multiple_of(PAGE_SIZE)
            || pages
                .checked_mul(PAGE_SIZE)
                .and_then(|bytes| base.get().checked_add(bytes))
                .is_none_or(|end| end >> 52 != 0)
        {
            return Err(PageTableBuildError::Storage);
        }
        Ok(Self {
            base,
            capacity,
            allocated: 0,
        })
    }

    pub const fn allocated_pages(&self) -> usize {
        self.allocated
    }

    fn issued_table(&self, table: PhysAddr) -> Result<usize, PageTableBuildError> {
        let offset = table
            .get()
            .checked_sub(self.base.get())
            .ok_or(PageTableBuildError::Storage)?;
        if !offset.is_multiple_of(PAGE_SIZE) {
            return Err(PageTableBuildError::Storage);
        }
        let page = usize::try_from(offset / PAGE_SIZE).map_err(|_| PageTableBuildError::Storage)?;
        if page >= self.allocated {
            return Err(PageTableBuildError::Storage);
        }
        Ok(page)
    }
}

impl PageTableStore for PreallocatedPageTableStore {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        if self.allocated == self.capacity {
            return Err(PageTableBuildError::Storage);
        }
        let address = self
            .base
            .get()
            .checked_add((self.allocated as u64) * PAGE_SIZE)
            .ok_or(PageTableBuildError::Storage)?;
        let frame = PhysAddr::new(address).map_err(|_| PageTableBuildError::Storage)?;
        for index in 0..512 {
            unsafe { (address as *mut u64).add(index).write_volatile(0) };
        }
        self.allocated += 1;
        Ok(frame)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        self.issued_table(table)?;
        Ok(unsafe { (table.get() as *const u64).add(index).read_volatile() })
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        self.issued_table(table)?;
        unsafe { (table.get() as *mut u64).add(index).write_volatile(value) };
        Ok(())
    }
}

pub struct PageTableBuilder<S> {
    store: S,
    root: PhysAddr,
}

impl<S: PageTableStore> PageTableBuilder<S> {
    pub fn new(mut store: S) -> Result<Self, PageTableBuildError> {
        let root = store.allocate_zeroed()?;
        Ok(Self { store, root })
    }

    /// Creates a fresh lower half while sharing only the current kernel's
    /// supervisor upper-half hierarchy.
    ///
    /// # Safety
    ///
    /// The current CR3 and every referenced upper-half table must remain live
    /// and immutable while the returned address space can run. The caller must
    /// exclude concurrent mutation during this snapshot.
    pub unsafe fn new_with_current_supervisor_half(store: S) -> Result<Self, PageTableBuildError> {
        let source: u64;
        unsafe { asm!("mov {}, cr3", out(reg) source, options(nomem, nostack, preserves_flags)) };
        let source = PhysAddr::new(source & ADDRESS_MASK)
            .map_err(|_| PageTableBuildError::InvalidIntermediate)?;
        Self::new_with_supervisor_half_from(store, &IdentityMappedPageTables, source)
    }

    /// Creates a private copy of every live supervisor mapping, including
    /// lower-half identity mappings used during early boot.
    ///
    /// # Safety
    ///
    /// Every table reachable from the current CR3 must be identity mapped and
    /// stable for the bounded clone. Current mappings must not be user-visible
    /// or use huge-page entries.
    pub unsafe fn clone_current_supervisor_mappings(
        mut store: S,
    ) -> Result<Self, PageTableBuildError> {
        let source: u64;
        unsafe { asm!("mov {}, cr3", out(reg) source, options(nomem, nostack, preserves_flags)) };
        let source = PhysAddr::new(source & ADDRESS_MASK)
            .map_err(|_| PageTableBuildError::InvalidIntermediate)?;
        let root = clone_supervisor_table(&mut store, &IdentityMappedPageTables, source, 4)?;
        Ok(Self { store, root })
    }

    fn new_with_supervisor_half_from<R: PageTableStore>(
        mut store: S,
        source_store: &R,
        source: PhysAddr,
    ) -> Result<Self, PageTableBuildError> {
        let root = store.allocate_zeroed()?;
        for index in 256..512 {
            let entry = source_store.read(source, index)?;
            if entry & 1 == 0 {
                if entry != 0 {
                    return Err(PageTableBuildError::InvalidIntermediate);
                }
                continue;
            }
            if entry & (1 << 2) != 0
                || entry & (1 << 7) != 0
                || entry & !((1 << 63) | 0xfff | ADDRESS_MASK) != 0
                || entry & ADDRESS_MASK == 0
            {
                return Err(PageTableBuildError::InvalidIntermediate);
            }
            store.write(root, index, entry)?;
        }
        Ok(Self { store, root })
    }
    pub const fn root(&self) -> PhysAddr {
        self.root
    }
    pub const fn store(&self) -> &S {
        &self.store
    }
    pub const fn from_existing_root(store: S, root: PhysAddr) -> Self {
        Self { store, root }
    }
    pub fn into_store(self) -> S {
        self.store
    }

    pub fn map(&mut self, mapping: Mapping) -> Result<(), PageTableBuildError> {
        for page in 0..mapping.pages() {
            let offset = page
                .checked_mul(PAGE_SIZE)
                .ok_or(PageTableBuildError::AddressOverflow)?;
            let virtual_address = mapping
                .virtual_start()
                .get()
                .checked_add(offset)
                .ok_or(PageTableBuildError::AddressOverflow)?;
            let physical_address = mapping
                .physical_start()
                .get()
                .checked_add(offset)
                .ok_or(PageTableBuildError::AddressOverflow)?;
            self.map_page(
                VirtAddr::new(virtual_address).map_err(PageTableBuildError::Page)?,
                PhysAddr::new(physical_address)
                    .map_err(|_| PageTableBuildError::AddressOverflow)?,
                mapping.permissions(),
            )?;
        }
        Ok(())
    }

    pub fn map_page(
        &mut self,
        virtual_address: VirtAddr,
        physical_address: PhysAddr,
        permissions: PagePermissions,
    ) -> Result<(), PageTableBuildError> {
        let user = permissions.user();
        let pdpt = self.descend(self.root, virtual_address.pml4_index(), user)?;
        let directory = self.descend(pdpt, virtual_address.pdpt_index(), user)?;
        let table = self.descend(directory, virtual_address.directory_index(), user)?;
        let index = virtual_address.table_index();
        if self.store.read(table, index)? & 1 != 0 {
            return Err(PageTableBuildError::AlreadyMapped);
        }
        let leaf = PageTableEntry::leaf(physical_address, permissions)
            .map_err(PageTableBuildError::Page)?;
        self.store.write(table, index, leaf.bits())
    }

    /// Replaces leaf permissions only when every page still matches the exact
    /// expected physical frame and old permissions. Logical validation is
    /// completed for the full range before the first write.
    pub fn protect(
        &mut self,
        expected: Mapping,
        final_permissions: PagePermissions,
    ) -> Result<(), PageTableBuildError> {
        for page in 0..expected.pages() {
            let (virtual_address, physical_address) = mapping_page(expected, page)?;
            let (table, index) = self.leaf_location(virtual_address)?;
            let current = self.store.read(table, index)?;
            if current & 1 == 0 {
                return Err(PageTableBuildError::NotMapped);
            }
            let wanted = PageTableEntry::leaf(physical_address, expected.permissions())
                .map_err(PageTableBuildError::Page)?
                .bits();
            if current & !LEAF_ACCESSED_DIRTY != wanted {
                return Err(PageTableBuildError::MappingMismatch);
            }
            PageTableEntry::leaf(physical_address, final_permissions)
                .map_err(PageTableBuildError::Page)?;
        }
        for page in 0..expected.pages() {
            let (virtual_address, physical_address) = mapping_page(expected, page)?;
            let (table, index) = self.leaf_location(virtual_address)?;
            let final_entry = PageTableEntry::leaf(physical_address, final_permissions)
                .map_err(PageTableBuildError::Page)?;
            let accessed_dirty = self.store.read(table, index)? & LEAF_ACCESSED_DIRTY;
            self.store
                .write(table, index, final_entry.bits() | accessed_dirty)?;
        }
        Ok(())
    }

    pub fn unmap_exact(&mut self, expected: Mapping) -> Result<(), PageTableBuildError> {
        for page in 0..expected.pages() {
            let (virtual_address, physical_address) = mapping_page(expected, page)?;
            let (table, index) = self.leaf_location(virtual_address)?;
            let current = self.store.read(table, index)?;
            if current & 1 == 0 {
                return Err(PageTableBuildError::NotMapped);
            }
            let wanted = PageTableEntry::leaf(physical_address, expected.permissions())
                .map_err(PageTableBuildError::Page)?
                .bits();
            if current & !LEAF_ACCESSED_DIRTY != wanted {
                return Err(PageTableBuildError::MappingMismatch);
            }
        }
        for page in 0..expected.pages() {
            let (virtual_address, _) = mapping_page(expected, page)?;
            let (table, index) = self.leaf_location(virtual_address)?;
            self.store.write(table, index, 0)?;
        }
        Ok(())
    }

    fn leaf_location(
        &self,
        virtual_address: VirtAddr,
    ) -> Result<(PhysAddr, usize), PageTableBuildError> {
        let mut table = self.root;
        for index in [
            virtual_address.pml4_index(),
            virtual_address.pdpt_index(),
            virtual_address.directory_index(),
        ] {
            let entry = self.store.read(table, index)?;
            if entry & 1 == 0 {
                return Err(PageTableBuildError::NotMapped);
            }
            if entry & (1 << 7) != 0 || entry & !((1 << 63) | 0xfff | ADDRESS_MASK) != 0 {
                return Err(PageTableBuildError::InvalidIntermediate);
            }
            table = PhysAddr::new(entry & ADDRESS_MASK)
                .map_err(|_| PageTableBuildError::InvalidIntermediate)?;
        }
        Ok((table, virtual_address.table_index()))
    }

    fn descend(
        &mut self,
        parent: PhysAddr,
        index: usize,
        user: bool,
    ) -> Result<PhysAddr, PageTableBuildError> {
        let current = self.store.read(parent, index)?;
        if current & 1 == 0 {
            let child = self.store.allocate_zeroed()?;
            let entry = PageTableEntry::table(child, user).map_err(PageTableBuildError::Page)?;
            self.store.write(parent, index, entry.bits())?;
            return Ok(child);
        }
        if current & (1 << 7) != 0 || current & !((1 << 63) | 0xfff | ADDRESS_MASK) != 0 {
            return Err(PageTableBuildError::InvalidIntermediate);
        }
        if user && current & (1 << 2) == 0 {
            self.store.write(parent, index, current | (1 << 2))?;
        }
        PhysAddr::new(current & ADDRESS_MASK).map_err(|_| PageTableBuildError::InvalidIntermediate)
    }
}

fn clone_supervisor_table<S: PageTableStore, R: PageTableStore>(
    destination: &mut S,
    source_store: &R,
    source: PhysAddr,
    level: u8,
) -> Result<PhysAddr, PageTableBuildError> {
    if !(1..=4).contains(&level) {
        return Err(PageTableBuildError::InvalidIntermediate);
    }
    let clone = destination.allocate_zeroed()?;
    for index in 0..512 {
        let entry = source_store.read(source, index)?;
        if entry & 1 == 0 {
            if entry != 0 {
                return Err(PageTableBuildError::InvalidIntermediate);
            }
            continue;
        }
        if entry & (1 << 2) != 0
            || entry & !((1 << 63) | 0xfff | ADDRESS_MASK) != 0
            || entry & ADDRESS_MASK == 0
        {
            return Err(PageTableBuildError::InvalidIntermediate);
        }
        if level == 1 {
            destination.write(clone, index, entry)?;
            continue;
        }
        if entry & (1 << 7) != 0 {
            return Err(PageTableBuildError::InvalidIntermediate);
        }
        let child = PhysAddr::new(entry & ADDRESS_MASK)
            .map_err(|_| PageTableBuildError::InvalidIntermediate)?;
        let child_clone = clone_supervisor_table(destination, source_store, child, level - 1)?;
        destination.write(clone, index, (entry & !ADDRESS_MASK) | child_clone.get())?;
    }
    Ok(clone)
}

struct IdentityMappedPageTables;

impl PageTableStore for IdentityMappedPageTables {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        Err(PageTableBuildError::Storage)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        Ok(unsafe { (table.get() as *const u64).add(index).read_volatile() })
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        if index >= 512 {
            return Err(PageTableBuildError::Storage);
        }
        unsafe { (table.get() as *mut u64).add(index).write_volatile(value) };
        Ok(())
    }
}

pub struct ActivePageTables {
    root: PhysAddr,
}

impl ActivePageTables {
    /// Opens the current CR3 for exact-match leaf protection without allocating
    /// new page-table frames.
    ///
    /// # Safety
    ///
    /// Every page-table frame reachable from the current CR3 must be mapped at
    /// its identical supervisor-writable virtual address and owned exclusively
    /// by this CPU for the duration of each mutation. Interrupts and concurrent
    /// address-space mutation must be excluded by the caller.
    pub unsafe fn current() -> Result<Self, PageTableBuildError> {
        let root: u64;
        unsafe { asm!("mov {}, cr3", out(reg) root, options(nomem, nostack, preserves_flags)) };
        let root = PhysAddr::new(root & ADDRESS_MASK)
            .map_err(|_| PageTableBuildError::InvalidIntermediate)?;
        Ok(Self { root })
    }

    pub const fn root(&self) -> PhysAddr {
        self.root
    }

    pub fn leaf(&self, address: VirtAddr) -> Result<Option<ActiveLeaf>, PageTableBuildError> {
        let tables = PageTableBuilder::from_existing_root(IdentityMappedPageTables, self.root);
        let (table, index) = tables.leaf_location(address)?;
        let bits = tables.store.read(table, index)?;
        if bits & 1 == 0 {
            return Ok(None);
        }
        let allowed = ADDRESS_MASK | (1 << 63) | LEAF_ACCESSED_DIRTY | 0b111;
        if bits & !allowed != 0 {
            return Err(PageTableBuildError::MappingMismatch);
        }
        let physical =
            PhysAddr::new(bits & ADDRESS_MASK).map_err(|_| PageTableBuildError::MappingMismatch)?;
        Ok(Some(ActiveLeaf {
            physical,
            writable: bits & (1 << 1) != 0,
            executable: bits & (1 << 63) == 0,
            user: bits & (1 << 2) != 0,
        }))
    }

    /// Applies an exact-match permission transition and invalidates every
    /// affected local translation before returning.
    ///
    /// # Safety
    ///
    /// The mapping and identity-map ownership requirements from
    /// [`Self::current`] must still hold.
    pub unsafe fn protect(
        &mut self,
        expected: Mapping,
        final_permissions: PagePermissions,
    ) -> Result<(), PageTableBuildError> {
        let mut tables = PageTableBuilder::from_existing_root(IdentityMappedPageTables, self.root);
        tables.protect(expected, final_permissions)?;
        for page in 0..expected.pages() {
            let address = expected
                .virtual_start()
                .get()
                .checked_add(
                    page.checked_mul(PAGE_SIZE)
                        .ok_or(PageTableBuildError::AddressOverflow)?,
                )
                .ok_or(PageTableBuildError::AddressOverflow)?;
            unsafe { asm!("invlpg [{}]", in(reg) address, options(nostack, preserves_flags)) };
        }
        Ok(())
    }

    /// Removes only the exact expected mapping and invalidates its local TLB
    /// entries. The safety requirements from [`Self::current`] apply.
    ///
    /// # Safety
    ///
    /// The caller must exclusively own all affected leaf entries, ensure the
    /// complete active hierarchy is identity mapped, and prevent concurrent
    /// page-table mutation for the duration of this operation.
    pub unsafe fn unmap_exact(&mut self, expected: Mapping) -> Result<(), PageTableBuildError> {
        let mut tables = PageTableBuilder::from_existing_root(IdentityMappedPageTables, self.root);
        tables.unmap_exact(expected)?;
        invalidate_mapping(expected)
    }
}

fn invalidate_mapping(mapping: Mapping) -> Result<(), PageTableBuildError> {
    for page in 0..mapping.pages() {
        let address = mapping
            .virtual_start()
            .get()
            .checked_add(
                page.checked_mul(PAGE_SIZE)
                    .ok_or(PageTableBuildError::AddressOverflow)?,
            )
            .ok_or(PageTableBuildError::AddressOverflow)?;
        unsafe { asm!("invlpg [{}]", in(reg) address, options(nostack, preserves_flags)) };
    }
    Ok(())
}

fn mapping_page(mapping: Mapping, page: u64) -> Result<(VirtAddr, PhysAddr), PageTableBuildError> {
    let offset = page
        .checked_mul(PAGE_SIZE)
        .ok_or(PageTableBuildError::AddressOverflow)?;
    let virtual_address = mapping
        .virtual_start()
        .get()
        .checked_add(offset)
        .ok_or(PageTableBuildError::AddressOverflow)?;
    let physical_address = mapping
        .physical_start()
        .get()
        .checked_add(offset)
        .ok_or(PageTableBuildError::AddressOverflow)?;
    Ok((
        VirtAddr::new(virtual_address).map_err(PageTableBuildError::Page)?,
        PhysAddr::new(physical_address).map_err(|_| PageTableBuildError::AddressOverflow)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(4096))]
    struct IdentityArena([[u64; 512]; 5]);

    struct Tables {
        pages: [[u64; 512]; 8],
        used: usize,
    }
    impl Tables {
        fn slot(frame: PhysAddr) -> Result<usize, PageTableBuildError> {
            let page = frame.get() / PAGE_SIZE;
            usize::try_from(page.checked_sub(1).ok_or(PageTableBuildError::Storage)?)
                .map_err(|_| PageTableBuildError::Storage)
        }
    }
    impl PageTableStore for Tables {
        fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
            if self.used == self.pages.len() {
                return Err(PageTableBuildError::Storage);
            }
            self.pages[self.used].fill(0);
            self.used += 1;
            PhysAddr::new(self.used as u64 * PAGE_SIZE).map_err(|_| PageTableBuildError::Storage)
        }
        fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
            self.pages
                .get(Self::slot(table)?)
                .and_then(|page| page.get(index))
                .copied()
                .ok_or(PageTableBuildError::Storage)
        }
        fn write(
            &mut self,
            table: PhysAddr,
            index: usize,
            value: u64,
        ) -> Result<(), PageTableBuildError> {
            *self
                .pages
                .get_mut(Self::slot(table)?)
                .and_then(|page| page.get_mut(index))
                .ok_or(PageTableBuildError::Storage)? = value;
            Ok(())
        }
    }

    #[test]
    fn builds_hardware_tables_and_rejects_duplicate_leaves() {
        let store = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        let mut builder = PageTableBuilder::new(store).unwrap();
        let address = VirtAddr::new(0x4000).unwrap();
        builder
            .map_page(
                address,
                PhysAddr::new(0x9000).unwrap(),
                PagePermissions::USER_READ_EXECUTE,
            )
            .unwrap();
        assert_eq!(
            builder.map_page(
                address,
                PhysAddr::new(0xa000).unwrap(),
                PagePermissions::USER_READ
            ),
            Err(PageTableBuildError::AlreadyMapped)
        );
        let store = builder.store();
        let pml4 = store.read(builder.root(), address.pml4_index()).unwrap();
        assert_ne!(pml4 & (1 << 2), 0);
        let pdpt = store
            .read(
                PhysAddr::new(pml4 & ADDRESS_MASK).unwrap(),
                address.pdpt_index(),
            )
            .unwrap();
        let directory = store
            .read(
                PhysAddr::new(pdpt & ADDRESS_MASK).unwrap(),
                address.directory_index(),
            )
            .unwrap();
        let leaf = store
            .read(
                PhysAddr::new(directory & ADDRESS_MASK).unwrap(),
                address.table_index(),
            )
            .unwrap();
        assert_eq!(leaf & ADDRESS_MASK, 0x9000);
        assert_eq!(leaf & (1 << 63), 0);
        assert_eq!(leaf & (1 << 1), 0);
    }

    #[test]
    fn preallocated_store_zeros_and_confines_issued_tables() {
        let mut arena = IdentityArena([[u64::MAX; 512]; 5]);
        let base = PhysAddr::new(arena.0.as_mut_ptr() as u64).unwrap();
        let store = unsafe { PreallocatedPageTableStore::new(base, 4) }.unwrap();
        let mut builder = PageTableBuilder::new(store).unwrap();
        builder
            .map_page(
                VirtAddr::new(0x4000).unwrap(),
                PhysAddr::new(0x20_0000).unwrap(),
                PagePermissions::USER_READ_EXECUTE,
            )
            .unwrap();
        assert_eq!(builder.store().allocated_pages(), 4);
        assert_eq!(arena.0[0][511], 0);
        assert_eq!(arena.0[4][0], u64::MAX);
        let mut store = builder.into_store();
        assert_eq!(store.allocate_zeroed(), Err(PageTableBuildError::Storage));
        assert_eq!(
            store.read(PhysAddr::new(base.get() + 4 * PAGE_SIZE).unwrap(), 0),
            Err(PageTableBuildError::Storage)
        );
        assert_eq!(
            store.write(PhysAddr::new(base.get() + 4 * PAGE_SIZE).unwrap(), 0, 0),
            Err(PageTableBuildError::Storage)
        );
    }

    #[test]
    fn fresh_root_copies_only_valid_supervisor_upper_half() {
        let mut source = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        let source_root = source.allocate_zeroed().unwrap();
        let kernel_entry = PageTableEntry::table(PhysAddr::new(0x20_0000).unwrap(), false)
            .unwrap()
            .bits();
        source.write(source_root, 300, kernel_entry).unwrap();
        let destination = Tables {
            pages: [[u64::MAX; 512]; 8],
            used: 0,
        };
        let builder =
            PageTableBuilder::new_with_supervisor_half_from(destination, &source, source_root)
                .unwrap();
        assert_eq!(builder.store().read(builder.root(), 255), Ok(0));
        assert_eq!(builder.store().read(builder.root(), 300), Ok(kernel_entry));

        source
            .write(source_root, 300, kernel_entry | (1 << 2))
            .unwrap();
        let destination = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        assert!(matches!(
            PageTableBuilder::new_with_supervisor_half_from(destination, &source, source_root),
            Err(PageTableBuildError::InvalidIntermediate)
        ));
    }

    #[test]
    fn supervisor_clone_is_private_and_retains_low_identity_mapping() {
        let source_store = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        let address = VirtAddr::new(0x20_0000).unwrap();
        let mapping = Mapping::new(
            address,
            PhysAddr::new(0x20_0000).unwrap(),
            1,
            PagePermissions::KERNEL_LOW_READ_EXECUTE,
        )
        .unwrap();
        let mut source = PageTableBuilder::new(source_store).unwrap();
        source.map(mapping).unwrap();
        let source_root = source.root();
        let source_store = source.into_store();
        let mut destination = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        let cloned_root =
            clone_supervisor_table(&mut destination, &source_store, source_root, 4).unwrap();
        let cloned = PageTableBuilder::from_existing_root(destination, cloned_root);
        let (table, index) = cloned.leaf_location(address).unwrap();
        assert_eq!(
            cloned.store().read(table, index).unwrap() & ADDRESS_MASK,
            0x20_0000
        );
    }

    #[test]
    fn protection_requires_exact_old_mapping_before_any_write() {
        let store = Tables {
            pages: [[0; 512]; 8],
            used: 0,
        };
        let mut builder = PageTableBuilder::new(store).unwrap();
        let mapping = Mapping::new(
            VirtAddr::new(0x8000).unwrap(),
            PhysAddr::new(0x8000).unwrap(),
            1,
            PagePermissions::KERNEL_MMIO_READ_WRITE,
        )
        .unwrap();
        builder.map(mapping).unwrap();
        let root = builder.root();
        let mut store = builder.into_store();
        let address = mapping.virtual_start();
        let pml4 =
            PhysAddr::new(store.read(root, address.pml4_index()).unwrap() & ADDRESS_MASK).unwrap();
        let pdpt =
            PhysAddr::new(store.read(pml4, address.pdpt_index()).unwrap() & ADDRESS_MASK).unwrap();
        let directory =
            PhysAddr::new(store.read(pdpt, address.directory_index()).unwrap() & ADDRESS_MASK)
                .unwrap();
        let leaf = store.read(directory, address.table_index()).unwrap();
        store
            .write(directory, address.table_index(), leaf | LEAF_ACCESSED_DIRTY)
            .unwrap();
        let mut builder = PageTableBuilder::from_existing_root(store, root);
        builder
            .protect(mapping, PagePermissions::KERNEL_LOW_READ_EXECUTE)
            .unwrap();
        assert_eq!(
            builder.protect(mapping, PagePermissions::KERNEL_LOW_READ_EXECUTE),
            Err(PageTableBuildError::MappingMismatch)
        );
        let executable = Mapping::new(
            VirtAddr::new(0x8000).unwrap(),
            PhysAddr::new(0x8000).unwrap(),
            1,
            PagePermissions::KERNEL_LOW_READ_EXECUTE,
        )
        .unwrap();
        assert_eq!(
            builder.protect(executable, PagePermissions::KERNEL_MMIO_READ_WRITE),
            Ok(())
        );
        assert_eq!(builder.unmap_exact(mapping), Ok(()));
        assert_eq!(
            builder.unmap_exact(mapping),
            Err(PageTableBuildError::NotMapped)
        );
    }
}
