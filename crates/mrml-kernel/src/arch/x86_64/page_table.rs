use super::{ADDRESS_MASK, Mapping, PageError, PagePermissions, PageTableEntry, VirtAddr};
use crate::{PAGE_SIZE, PhysAddr};

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

pub struct PageTableBuilder<S> {
    store: S,
    root: PhysAddr,
}

impl<S: PageTableStore> PageTableBuilder<S> {
    pub fn new(mut store: S) -> Result<Self, PageTableBuildError> {
        let root = store.allocate_zeroed()?;
        Ok(Self { store, root })
    }
    pub const fn root(&self) -> PhysAddr {
        self.root
    }
    pub const fn store(&self) -> &S {
        &self.store
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
            if current != wanted {
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
            self.store.write(table, index, final_entry.bits())?;
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
    }
}
