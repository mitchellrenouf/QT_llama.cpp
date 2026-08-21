use mrml_kernel::arch::x86_64::{
    AddressSpace, Mapping, PagePermissions, PageTableBuildError, PageTableBuilder, PageTableStore,
    VirtAddr,
};
use mrml_kernel::{
    BootHandoff, MAX_PE_SECTIONS, PAGE_SIZE, PeImage, PhysAddr, VerifiedExecutable, VmBackend,
    VmExit,
};

use crate::{GuestRange, MapPermissions, WhpError};

use super::{PreparedWhpPartition, WhpSystem};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhpLaunchLayout {
    table_physical: u64,
    table_pages: u64,
    image_physical: u64,
    image_virtual: u64,
    handoff_physical: u64,
    handoff_virtual: u64,
    stack_physical: u64,
    stack_virtual: u64,
    stack_pages: u64,
    user: bool,
}

impl WhpLaunchLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table_physical: u64,
        table_pages: u64,
        image_physical: u64,
        image_virtual: u64,
        handoff_physical: u64,
        handoff_virtual: u64,
        stack_physical: u64,
        stack_virtual: u64,
        stack_pages: u64,
        user: bool,
    ) -> Result<Self, WhpError> {
        if table_pages < 4
            || stack_pages == 0
            || [
                table_physical,
                image_physical,
                handoff_physical,
                stack_physical,
            ]
            .iter()
            .any(|value| *value == 0 || !value.is_multiple_of(PAGE_SIZE))
            || [image_virtual, handoff_virtual, stack_virtual]
                .iter()
                .any(|value| *value == 0 || !value.is_multiple_of(PAGE_SIZE))
        {
            return Err(WhpError::InvalidMapping);
        }
        Ok(Self {
            table_physical,
            table_pages,
            image_physical,
            image_virtual,
            handoff_physical,
            handoff_virtual,
            stack_physical,
            stack_virtual,
            stack_pages,
            user,
        })
    }
}

pub struct PreparedWhpGuest<'system> {
    partition: PreparedWhpPartition<'system>,
    entry: u64,
    root: PhysAddr,
}

impl PreparedWhpGuest<'_> {
    pub const fn entry(&self) -> u64 {
        self.entry
    }
    pub const fn page_table_root(&self) -> PhysAddr {
        self.root
    }
    pub fn run(&mut self) -> Result<VmExit, WhpError> {
        self.partition.run()
    }
    pub fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), WhpError> {
        self.partition.read_guest(address, output)
    }
    pub fn inject_interrupt(&mut self, vector: u8) -> Result<(), WhpError> {
        self.partition.inject_interrupt(vector)
    }
}

impl VmBackend for PreparedWhpGuest<'_> {
    type Error = WhpError;

    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        if vcpu != 0 {
            return Err(WhpError::InvalidVcpu);
        }
        self.partition.run()
    }

    fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.partition.read_guest(address, output)
    }

    fn write_guest(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.partition.write_guest(address, input)
    }

    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        if vcpu != 0 {
            return Err(WhpError::InvalidVcpu);
        }
        self.partition.inject_interrupt(vector)
    }
}

impl WhpSystem {
    pub fn prepare_guest<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        BootHandoff::decode(handoff, |_| {}).map_err(WhpError::Handoff)?;
        let image_bytes = page_bytes(executable.image().image_size() as u64)?;
        let handoff_bytes = page_bytes(handoff.len() as u64)?;
        let table_bytes = layout
            .table_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        let stack_bytes = layout
            .stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        validate_ranges(&[
            (layout.table_physical, table_bytes),
            (layout.image_physical, image_bytes),
            (layout.handoff_physical, handoff_bytes),
            (layout.stack_physical, stack_bytes),
        ])?;
        validate_ranges(&[
            (layout.image_virtual, image_bytes),
            (layout.handoff_virtual, handoff_bytes),
            (layout.stack_virtual, stack_bytes),
        ])?;

        let mut partition = self.prepare_partition()?;
        partition.map_zeroed(range(
            layout.table_physical,
            table_bytes,
            MapPermissions::read_write(),
        )?)?;
        partition.map_zeroed(range(
            layout.image_physical,
            image_bytes,
            MapPermissions::read_write(),
        )?)?;
        partition.map_initialized(
            range(
                layout.handoff_physical,
                handoff_bytes,
                MapPermissions::read_only(),
            )?,
            handoff,
        )?;
        partition.map_zeroed(range(
            layout.stack_physical,
            stack_bytes,
            MapPermissions::read_write(),
        )?)?;

        let image = executable.image();
        let destination =
            partition.mutable_guest(layout.image_physical, image.image_size() as usize)?;
        let entry = image
            .materialize_at(destination, layout.image_virtual)
            .map_err(WhpError::Pe)?;

        let stack = layout
            .stack_virtual
            .checked_add(stack_bytes)
            .and_then(|end| end.checked_sub(8))
            .ok_or(WhpError::MemoryOverflow)?;
        partition.write_guest(layout.stack_physical + stack_bytes - 8, &0u64.to_le_bytes())?;

        let root = build_page_tables(&mut partition, image, layout, handoff_bytes)?;
        partition.configure_long_mode(
            entry,
            stack,
            root.get(),
            layout.handoff_virtual,
            handoff.len() as u64,
        )?;
        Ok(PreparedWhpGuest {
            partition,
            entry,
            root,
        })
    }
}

fn build_page_tables(
    partition: &mut PreparedWhpPartition<'_>,
    image: &PeImage<'_>,
    layout: WhpLaunchLayout,
    handoff_bytes: u64,
) -> Result<PhysAddr, WhpError> {
    let store = WhpPageTableStore::new(partition, layout.table_physical, layout.table_pages)?;
    let mut tables = PageTableBuilder::new(store).map_err(|_| WhpError::InvalidRegisterState)?;
    map_pe(
        &mut tables,
        image,
        layout.image_physical,
        layout.image_virtual,
        layout.user,
    )?;
    let handoff_permissions = if layout.user {
        PagePermissions::USER_READ
    } else {
        PagePermissions::KERNEL_READ
    };
    tables
        .map(
            Mapping::new(
                VirtAddr::new(layout.handoff_virtual).map_err(|_| WhpError::InvalidMapping)?,
                PhysAddr::new(layout.handoff_physical).map_err(|_| WhpError::InvalidMapping)?,
                handoff_bytes / PAGE_SIZE,
                handoff_permissions,
            )
            .map_err(|_| WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::PageTable)?;
    let stack_permissions = if layout.user {
        PagePermissions::USER_READ_WRITE
    } else {
        PagePermissions::KERNEL_READ_WRITE
    };
    tables
        .map(
            Mapping::new(
                VirtAddr::new(layout.stack_virtual).map_err(|_| WhpError::InvalidMapping)?,
                PhysAddr::new(layout.stack_physical).map_err(|_| WhpError::InvalidMapping)?,
                layout.stack_pages,
                stack_permissions,
            )
            .map_err(|_| WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::PageTable)?;
    Ok(tables.root())
}

pub(super) struct WhpPageTableStore<'a, 'system> {
    partition: &'a mut PreparedWhpPartition<'system>,
    start: u64,
    next: u64,
    end: u64,
}

impl<'a, 'system> WhpPageTableStore<'a, 'system> {
    pub(super) fn new(
        partition: &'a mut PreparedWhpPartition<'system>,
        start: u64,
        pages: u64,
    ) -> Result<Self, WhpError> {
        let end = start
            .checked_add(
                pages
                    .checked_mul(PAGE_SIZE)
                    .ok_or(WhpError::MemoryOverflow)?,
            )
            .ok_or(WhpError::MemoryOverflow)?;
        Ok(Self {
            partition,
            start,
            next: start,
            end,
        })
    }

    fn entry_address(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512
            || table.get() < self.start
            || table.get() >= self.next
            || !table.get().is_multiple_of(PAGE_SIZE)
        {
            return Err(PageTableBuildError::Storage);
        }
        table
            .get()
            .checked_add(index as u64 * 8)
            .ok_or(PageTableBuildError::AddressOverflow)
    }
}

impl PageTableStore for WhpPageTableStore<'_, '_> {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        let following = self
            .next
            .checked_add(PAGE_SIZE)
            .ok_or(PageTableBuildError::AddressOverflow)?;
        if following > self.end {
            return Err(PageTableBuildError::Storage);
        }
        let address = PhysAddr::new(self.next).map_err(|_| PageTableBuildError::Storage)?;
        self.partition
            .write_guest(self.next, &[0u8; PAGE_SIZE as usize])
            .map_err(|_| PageTableBuildError::Storage)?;
        self.next = following;
        Ok(address)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        let mut bytes = [0u8; 8];
        self.partition
            .read_guest(self.entry_address(table, index)?, &mut bytes)
            .map_err(|_| PageTableBuildError::Storage)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        let address = self.entry_address(table, index)?;
        self.partition
            .write_guest(address, &value.to_le_bytes())
            .map_err(|_| PageTableBuildError::Storage)
    }
}

fn map_pe<S: PageTableStore>(
    tables: &mut PageTableBuilder<S>,
    image: &PeImage<'_>,
    physical: u64,
    virtual_base: u64,
    user: bool,
) -> Result<(), WhpError> {
    let mut validated = AddressSpace::<{ MAX_PE_SECTIONS + 1 }>::new();
    for index in 0..image.load_region_count() {
        validated
            .map(pe_mapping(image, physical, virtual_base, index, user)?)
            .map_err(|_| WhpError::InvalidMapping)?;
    }
    for index in 0..image.load_region_count() {
        tables
            .map(pe_mapping(image, physical, virtual_base, index, user)?)
            .map_err(|_| WhpError::PageTable)?;
    }
    Ok(())
}

fn pe_mapping(
    image: &PeImage<'_>,
    physical: u64,
    virtual_base: u64,
    index: usize,
    user: bool,
) -> Result<Mapping, WhpError> {
    let region = image.load_region(index).map_err(WhpError::Pe)?;
    let permissions = match (user, region.writable(), region.executable()) {
        (true, true, false) => PagePermissions::USER_READ_WRITE,
        (true, false, true) => PagePermissions::USER_READ_EXECUTE,
        (true, false, false) => PagePermissions::USER_READ,
        (false, true, false) => PagePermissions::KERNEL_READ_WRITE,
        (false, false, true) => PagePermissions::KERNEL_READ_EXECUTE,
        (false, false, false) => PagePermissions::KERNEL_READ,
        (_, true, true) => return Err(WhpError::InvalidMapping),
    };
    let offset = region.virtual_address() as u64;
    Mapping::new(
        VirtAddr::new(
            virtual_base
                .checked_add(offset)
                .ok_or(WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::InvalidMapping)?,
        PhysAddr::new(
            physical
                .checked_add(offset)
                .ok_or(WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::InvalidMapping)?,
        region.pages() as u64,
        permissions,
    )
    .map_err(|_| WhpError::InvalidMapping)
}

fn range(start: u64, bytes: u64, permissions: MapPermissions) -> Result<GuestRange, WhpError> {
    GuestRange::new(start, bytes, permissions)
}

fn page_bytes(bytes: u64) -> Result<u64, WhpError> {
    if bytes == 0 {
        return Err(WhpError::EmptyMemory);
    }
    bytes
        .checked_add(PAGE_SIZE - 1)
        .ok_or(WhpError::MemoryOverflow)
        .map(|value| value / PAGE_SIZE * PAGE_SIZE)
}

fn validate_ranges(ranges: &[(u64, u64)]) -> Result<(), WhpError> {
    for (index, &(start, bytes)) in ranges.iter().enumerate() {
        let end = start.checked_add(bytes).ok_or(WhpError::MemoryOverflow)?;
        for &(other, other_bytes) in &ranges[..index] {
            let other_end = other
                .checked_add(other_bytes)
                .ok_or(WhpError::MemoryOverflow)?;
            if start < other_end && other < end {
                return Err(WhpError::MemoryOverlap);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_layout_rejects_aliases_and_weak_arenas() {
        assert_eq!(
            WhpLaunchLayout::new(
                0x1000,
                3,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                false
            ),
            Err(WhpError::InvalidMapping)
        );
        assert_eq!(
            validate_ranges(&[(0x1000, 0x2000), (0x2000, 0x1000)]),
            Err(WhpError::MemoryOverlap)
        );
        assert!(
            WhpLaunchLayout::new(
                0x1000,
                8,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn backend_rejects_nonexistent_virtual_processors() {
        let system = WhpSystem::open().unwrap();
        if !system.hypervisor_present().unwrap() {
            return;
        }
        let partition = system.prepare_partition().unwrap();
        let mut guest = PreparedWhpGuest {
            partition,
            entry: 0x20_0000,
            root: PhysAddr::new(0x10_0000).unwrap(),
        };
        assert_eq!(VmBackend::run(&mut guest, 1), Err(WhpError::InvalidVcpu));
        assert_eq!(
            VmBackend::inject_interrupt(&mut guest, 1, 48),
            Err(WhpError::InvalidVcpu)
        );
    }
}
