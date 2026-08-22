use super::{
    AddressSpace, AddressSpaceError, Mapping, MappingId, PagePermissions, PageTableBuildError,
    PageTableBuilder, PageTableStore, PeMappingError, VirtAddr, map_pe_image,
};
use crate::{
    ArtifactKind, MAX_PE_SECTIONS, PAGE_SIZE, PeAllocatedRegion, PeAllocationError, PeError,
    PeImage, PhysAddr, ServiceLaunch, VerifiedExecutable,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceSpaceError {
    WrongArtifact,
    HandoffMismatch,
    InvalidKernelMapping,
    InvalidStack,
    Overflow,
    Address(AddressSpaceError),
    Image(PeMappingError),
    Tables(PageTableBuildError),
    Pe(PeError),
    Allocation(PeAllocationError),
}

/// Final mapping policy for one authenticated user service. Each instance can
/// materialize only a fresh root; it contains supervisor kernel mappings, one
/// service PE, and one stack whose immediately preceding page is absent.
pub struct ServiceAddressSpace<const MAPPINGS: usize> {
    space: AddressSpace<MAPPINGS>,
    entry: u64,
    stack_top: u64,
    lower_guard: u64,
}

impl<const MAPPINGS: usize> ServiceAddressSpace<MAPPINGS> {
    /// Reconstructs the W^X mapping plan for a service authenticated and
    /// materialized by the boot loader. The descriptor fixes the complete
    /// allocation, preferred virtual base, entry, stack, and table arena.
    pub fn from_handoff(
        service: ServiceLaunch,
        executable: &VerifiedExecutable<'_>,
        kernel_mappings: &[Mapping],
    ) -> Result<Self, ServiceSpaceError> {
        if executable.artifact().kind() != ArtifactKind::ServiceImage
            || executable.artifact().version() != service.version()
            || executable.artifact().digest() != &service.measurement()
        {
            return Err(ServiceSpaceError::HandoffMismatch);
        }
        Self::from_handoff_image(service, executable.image(), kernel_mappings)
    }

    fn from_handoff_image(
        service: ServiceLaunch,
        image: &PeImage<'_>,
        kernel_mappings: &[Mapping],
    ) -> Result<Self, ServiceSpaceError> {
        let expected_bytes = service
            .image_pages()
            .checked_mul(PAGE_SIZE)
            .ok_or(ServiceSpaceError::Overflow)?;
        if u64::from(image.image_size()) > expected_bytes
            || image.image_base() != service.image_virtual()
            || service.entry()
                != service
                    .image_virtual()
                    .checked_add(u64::from(image.entry_rva()))
                    .ok_or(ServiceSpaceError::Overflow)?
        {
            return Err(ServiceSpaceError::HandoffMismatch);
        }
        let mut allocations: [Option<PeAllocatedRegion>; MAX_PE_SECTIONS + 1] =
            [None; MAX_PE_SECTIONS + 1];
        let allocation_count = image.load_region_count();
        for (index, slot) in allocations[..allocation_count].iter_mut().enumerate() {
            let load = image.load_region(index).map_err(ServiceSpaceError::Pe)?;
            *slot = Some(
                PeAllocatedRegion::from_contiguous_image(
                    load,
                    service.image_physical(),
                    service.image_pages(),
                )
                .map_err(ServiceSpaceError::Allocation)?,
            );
        }
        let stack_bytes = service
            .stack_pages()
            .checked_mul(PAGE_SIZE)
            .ok_or(ServiceSpaceError::Overflow)?;
        let stack_base = service
            .stack_top()
            .checked_sub(stack_bytes)
            .ok_or(ServiceSpaceError::InvalidStack)?;
        Self::build(
            image,
            service.image_virtual(),
            &allocations,
            allocation_count,
            kernel_mappings,
            stack_base,
            service.stack_physical(),
            service.stack_pages(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executable: &VerifiedExecutable<'_>,
        load_base: u64,
        allocations: &[Option<PeAllocatedRegion>],
        allocation_count: usize,
        kernel_mappings: &[Mapping],
        stack_base: u64,
        stack_physical: PhysAddr,
        stack_pages: u64,
    ) -> Result<Self, ServiceSpaceError> {
        if executable.artifact().kind() != ArtifactKind::ServiceImage {
            return Err(ServiceSpaceError::WrongArtifact);
        }
        Self::build(
            executable.image(),
            load_base,
            allocations,
            allocation_count,
            kernel_mappings,
            stack_base,
            stack_physical,
            stack_pages,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        image: &PeImage<'_>,
        load_base: u64,
        allocations: &[Option<PeAllocatedRegion>],
        allocation_count: usize,
        kernel_mappings: &[Mapping],
        stack_base: u64,
        stack_physical: PhysAddr,
        stack_pages: u64,
    ) -> Result<Self, ServiceSpaceError> {
        if stack_pages == 0 || stack_base < PAGE_SIZE || !stack_base.is_multiple_of(PAGE_SIZE) {
            return Err(ServiceSpaceError::InvalidStack);
        }
        let stack_bytes = stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(ServiceSpaceError::Overflow)?;
        let stack_top = stack_base
            .checked_add(stack_bytes)
            .ok_or(ServiceSpaceError::Overflow)?;
        if stack_top >= 1 << 47 {
            return Err(ServiceSpaceError::InvalidStack);
        }
        let lower_guard = stack_base - PAGE_SIZE;
        let entry = load_base
            .checked_add(u64::from(image.entry_rva()))
            .ok_or(ServiceSpaceError::Overflow)?;
        if entry == 0 || entry >= 1 << 47 {
            return Err(ServiceSpaceError::WrongArtifact);
        }

        let mut space = AddressSpace::new();
        for mapping in kernel_mappings.iter().copied() {
            if mapping.permissions().user() || mapping.virtual_start().get() < 0xffff_8000_0000_0000
            {
                return Err(ServiceSpaceError::InvalidKernelMapping);
            }
            space.map(mapping).map_err(ServiceSpaceError::Address)?;
        }
        let mut image_ids: [Option<MappingId>; MAX_PE_SECTIONS + 1] = [None; MAX_PE_SECTIONS + 1];
        map_pe_image(
            &mut space,
            load_base,
            true,
            allocations,
            allocation_count,
            &mut image_ids,
        )
        .map_err(ServiceSpaceError::Image)?;
        let stack = Mapping::new(
            VirtAddr::new(stack_base).map_err(|_| ServiceSpaceError::InvalidStack)?,
            stack_physical,
            stack_pages,
            PagePermissions::USER_READ_WRITE,
        )
        .map_err(ServiceSpaceError::Address)?;
        space.map(stack).map_err(ServiceSpaceError::Address)?;
        if space.mappings().any(|mapping| {
            let start = mapping.virtual_start().get();
            let end = start + mapping.pages() * PAGE_SIZE;
            lower_guard >= start && lower_guard < end
        }) {
            return Err(ServiceSpaceError::InvalidStack);
        }
        Ok(Self {
            space,
            entry,
            stack_top,
            lower_guard,
        })
    }

    pub const fn entry(&self) -> u64 {
        self.entry
    }

    pub const fn stack_top(&self) -> u64 {
        self.stack_top
    }

    pub const fn lower_guard(&self) -> u64 {
        self.lower_guard
    }

    pub fn build_page_tables<S: PageTableStore>(
        &self,
        store: S,
    ) -> Result<PageTableBuilder<S>, ServiceSpaceError> {
        let mut tables = PageTableBuilder::new(store).map_err(ServiceSpaceError::Tables)?;
        for mapping in self.space.mappings() {
            tables.map(mapping).map_err(ServiceSpaceError::Tables)?;
        }
        Ok(tables)
    }

    /// Builds the isolated user half while retaining the live kernel's
    /// supervisor-only upper half for safe trap and syscall entry.
    ///
    /// # Safety
    ///
    /// The current kernel page tables must satisfy the ownership and lifetime
    /// contract of [`PageTableBuilder::new_with_current_supervisor_half`].
    pub unsafe fn build_page_tables_with_current_kernel<S: PageTableStore>(
        &self,
        store: S,
    ) -> Result<PageTableBuilder<S>, ServiceSpaceError> {
        let mut tables = unsafe { PageTableBuilder::new_with_current_supervisor_half(store) }
            .map_err(ServiceSpaceError::Tables)?;
        for mapping in self.space.mappings() {
            tables.map(mapping).map_err(ServiceSpaceError::Tables)?;
        }
        Ok(tables)
    }

    pub fn mappings(&self) -> impl Iterator<Item = Mapping> + '_ {
        self.space.mappings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeLoadRegion;

    fn minimal_pe() -> [u8; 1024] {
        let mut pe = [0u8; 1024];
        pe[0..2].copy_from_slice(&0x5a4du16.to_le_bytes());
        pe[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        pe[0x80..0x84].copy_from_slice(&0x0000_4550u32.to_le_bytes());
        pe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        pe[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        pe[0x96..0x98].copy_from_slice(&2u16.to_le_bytes());
        let optional = 0x98;
        pe[optional..optional + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        pe[optional + 16..optional + 20].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 24..optional + 32].copy_from_slice(&0x0040_0000u64.to_le_bytes());
        pe[optional + 32..optional + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 36..optional + 40].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 56..optional + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[optional + 60..optional + 64].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 70..optional + 72].copy_from_slice(&0x100u16.to_le_bytes());
        let section = optional + 240;
        pe[section..section + 5].copy_from_slice(b".text");
        pe[section + 8..section + 12].copy_from_slice(&1u32.to_le_bytes());
        pe[section + 12..section + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[section + 16..section + 20].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 20..section + 24].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 36..section + 40].copy_from_slice(&0x6000_0000u32.to_le_bytes());
        pe[0x200] = 0xc3;
        pe
    }

    #[test]
    fn service_plan_keeps_kernel_supervisor_and_lower_guard_absent() {
        let encoded = minimal_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let allocations = [
            Some(PeAllocatedRegion::test_value(
                PeLoadRegion::test_value(0, 1, false, false),
                PhysAddr::new(0x20_0000).unwrap(),
            )),
            Some(PeAllocatedRegion::test_value(
                PeLoadRegion::test_value(0x1000, 1, false, true),
                PhysAddr::new(0x21_0000).unwrap(),
            )),
        ];
        let kernel = Mapping::new(
            VirtAddr::new(0xffff_8000_0040_0000).unwrap(),
            PhysAddr::new(0x10_0000).unwrap(),
            1,
            PagePermissions::KERNEL_READ_EXECUTE,
        )
        .unwrap();
        let plan = ServiceAddressSpace::<8>::build(
            &image,
            0x40_0000,
            &allocations,
            2,
            &[kernel],
            0x70_0000,
            PhysAddr::new(0x30_0000).unwrap(),
            2,
        )
        .unwrap();
        assert_eq!(plan.entry(), 0x40_1000);
        assert_eq!(plan.stack_top(), 0x70_2000);
        assert_eq!(plan.lower_guard(), 0x6f_f000);
        assert!(plan.mappings().all(|mapping| {
            let start = mapping.virtual_start().get();
            let end = start + mapping.pages() * PAGE_SIZE;
            !(plan.lower_guard() >= start && plan.lower_guard() < end)
        }));
        assert_eq!(
            ServiceAddressSpace::<8>::build(
                &image,
                0x40_0000,
                &allocations,
                2,
                &[Mapping::new(
                    VirtAddr::new(0x5000).unwrap(),
                    PhysAddr::new(0x40_0000).unwrap(),
                    1,
                    PagePermissions::KERNEL_MMIO_READ_WRITE,
                )
                .unwrap()],
                0x70_0000,
                PhysAddr::new(0x30_0000).unwrap(),
                2,
            )
            .err(),
            Some(ServiceSpaceError::InvalidKernelMapping)
        );
    }

    #[test]
    fn handoff_reconstructs_contiguous_image_with_section_permissions() {
        let encoded = minimal_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let mut materialized = [0u8; 8192];
        let entry = image.materialize(&mut materialized).unwrap();
        let service = ServiceLaunch::new(
            PhysAddr::new(0x10_0000).unwrap(),
            1024,
            PhysAddr::new(0x20_0000).unwrap(),
            2,
            image.image_base(),
            entry,
            PhysAddr::new(0x30_0000).unwrap(),
            2,
            0x7000_2000,
            PhysAddr::new(0x40_0000).unwrap(),
            8,
            1,
            [1; 64],
        )
        .unwrap();
        let plan = ServiceAddressSpace::<8>::from_handoff_image(service, &image, &[]).unwrap();
        assert_eq!(plan.entry(), entry);
        assert_eq!(plan.stack_top(), 0x7000_2000);
        assert!(plan.mappings().any(|mapping| {
            mapping.virtual_start().get() == image.image_base() + 0x1000
                && mapping.physical_start().get() == 0x20_1000
                && mapping.permissions() == PagePermissions::USER_READ_EXECUTE
        }));

        let wrong_entry = ServiceLaunch::new(
            service.artifact_physical(),
            service.artifact_length(),
            service.image_physical(),
            service.image_pages(),
            service.image_virtual(),
            service.entry() + 1,
            service.stack_physical(),
            service.stack_pages(),
            service.stack_top(),
            service.table_physical(),
            service.table_pages(),
            service.version(),
            service.measurement(),
        )
        .unwrap();
        assert_eq!(
            ServiceAddressSpace::<8>::from_handoff_image(wrong_entry, &image, &[]).err(),
            Some(ServiceSpaceError::HandoffMismatch)
        );
    }
}
