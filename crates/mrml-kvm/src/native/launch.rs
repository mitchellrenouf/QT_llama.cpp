use mrml_kernel::arch::x86_64::{Mapping, PagePermissions, VirtAddr};
use mrml_kernel::{BootHandoff, PhysAddr, VerifiedExecutable, VmBackend, VmExit, PAGE_SIZE};

use super::{map_loaded_handoff, map_loaded_pe, KvmBackend, KvmError, KvmSystem};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmLaunchLayout {
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

impl KvmLaunchLayout {
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
    ) -> Result<Self, KvmError> {
        if table_pages < 4
            || stack_pages == 0
            || [
                table_physical,
                image_physical,
                handoff_physical,
                stack_physical,
            ]
            .iter()
            .any(|address| *address == 0 || address % PAGE_SIZE != 0)
            || [image_virtual, handoff_virtual, stack_virtual]
                .iter()
                .any(|address| *address == 0 || address % PAGE_SIZE != 0)
        {
            return Err(KvmError::InvalidMapping);
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

pub struct PreparedKvmGuest<const N: usize> {
    backend: KvmBackend<N>,
    entry: u64,
    root: PhysAddr,
}

impl<const N: usize> PreparedKvmGuest<N> {
    pub const fn entry(&self) -> u64 {
        self.entry
    }
    pub const fn page_table_root(&self) -> PhysAddr {
        self.root
    }
}

impl<const N: usize> VmBackend for PreparedKvmGuest<N> {
    type Error = KvmError;
    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        self.backend.run(vcpu)
    }
    fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.backend.read_guest(address, output)
    }
    fn write_guest(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.backend.write_guest(address, input)
    }
    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        self.backend.inject_interrupt(vcpu, vector)
    }
}

impl KvmSystem {
    pub fn prepare_guest<const N: usize>(
        &self,
        vcpu_id: u32,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: KvmLaunchLayout,
    ) -> Result<PreparedKvmGuest<N>, KvmError> {
        if N < 4 {
            return Err(KvmError::MemoryTableFull);
        }
        BootHandoff::decode(handoff, |_| {}).map_err(KvmError::Handoff)?;
        let image_bytes = page_bytes(executable.image().image_size() as u64)?;
        let handoff_bytes = page_bytes(handoff.len() as u64)?;
        let table_bytes = layout
            .table_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let stack_bytes = layout
            .stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
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

        let mut backend = self.create_backend::<N>(vcpu_id)?;
        backend.vm.create_irqchip()?;
        backend.map_memory(layout.table_physical, table_bytes as usize, false)?;
        backend.map_memory(layout.image_physical, image_bytes as usize, false)?;
        backend.map_memory(layout.handoff_physical, handoff_bytes as usize, false)?;
        backend.map_memory(layout.stack_physical, stack_bytes as usize, false)?;
        let loaded_image = backend.memory.load_verified_executable(
            executable,
            layout.image_physical,
            layout.image_virtual,
        )?;
        let loaded_handoff = backend
            .memory
            .load_boot_handoff(handoff, layout.handoff_physical)?;
        let stack_physical_end = layout
            .stack_physical
            .checked_add(stack_bytes)
            .ok_or(KvmError::MemoryOverflow)?;
        backend
            .memory
            .write(stack_physical_end - 8, &0u64.to_le_bytes())?;
        let stack = layout
            .stack_virtual
            .checked_add(stack_bytes)
            .and_then(|end| end.checked_sub(8))
            .ok_or(KvmError::MemoryOverflow)?;
        let mut tables = backend.page_tables(layout.table_physical, layout.table_pages)?;
        map_loaded_pe(&mut tables, executable, loaded_image, layout.user)?;
        map_loaded_handoff(
            &mut tables,
            loaded_handoff,
            layout.handoff_virtual,
            layout.user,
        )?;
        let stack_permissions = if layout.user {
            PagePermissions::USER_READ_WRITE
        } else {
            PagePermissions::KERNEL_READ_WRITE
        };
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(layout.stack_virtual).map_err(|_| KvmError::InvalidMapping)?,
                    PhysAddr::new(layout.stack_physical).map_err(|_| KvmError::InvalidMapping)?,
                    layout.stack_pages,
                    stack_permissions,
                )
                .map_err(|_| KvmError::InvalidMapping)?,
            )
            .map_err(|_| KvmError::PageTable)?;
        let root = tables.root();
        drop(tables.into_store());
        backend.vcpu.configure_long_mode_entry(
            loaded_image.entry(),
            stack,
            root.get(),
            layout.handoff_virtual,
            loaded_handoff.bytes() as u64,
        )?;
        Ok(PreparedKvmGuest {
            backend,
            entry: loaded_image.entry(),
            root,
        })
    }
}

fn page_bytes(bytes: u64) -> Result<u64, KvmError> {
    if bytes == 0 {
        return Err(KvmError::EmptyMemory);
    }
    bytes
        .checked_add(PAGE_SIZE - 1)
        .ok_or(KvmError::MemoryOverflow)
        .map(|value| value / PAGE_SIZE * PAGE_SIZE)
}

fn validate_ranges(ranges: &[(u64, u64)]) -> Result<(), KvmError> {
    for (index, &(start, bytes)) in ranges.iter().enumerate() {
        let end = start.checked_add(bytes).ok_or(KvmError::MemoryOverflow)?;
        for &(other_start, other_bytes) in &ranges[..index] {
            let other_end = other_start
                .checked_add(other_bytes)
                .ok_or(KvmError::MemoryOverflow)?;
            if start < other_end && other_start < end {
                return Err(KvmError::MemoryOverlap);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_layout_rejects_small_table_arenas_and_overlaps() {
        assert_eq!(
            KvmLaunchLayout::new(
                0x1000,
                3,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                true
            ),
            Err(KvmError::InvalidMapping)
        );
        assert_eq!(
            validate_ranges(&[(0x1000, 0x2000), (0x2000, 0x1000)]),
            Err(KvmError::MemoryOverlap)
        );
        assert!(KvmLaunchLayout::new(
            0x1000,
            8,
            0x10000,
            0x140000000,
            0x20000,
            0x200000,
            0x30000,
            0x300000,
            2,
            true
        )
        .is_ok());
    }
}
