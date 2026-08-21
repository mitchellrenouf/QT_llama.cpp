use super::{AddressSpace, AddressSpaceError, Mapping, MappingId, PagePermissions, VirtAddr};
use crate::{MAX_PE_SECTIONS, PeAllocatedRegion};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeMappingError {
    InvalidCount,
    MissingAllocation,
    OutputTooSmall,
    Address(AddressSpaceError),
    RollbackFailure,
}

/// Installs a validated PE physical plan into an address space. The caller
/// must materialize bytes under separate writable/NX aliases that are removed
/// before invoking this function; these are the final executable mappings.
pub fn map_pe_image<const MAPPINGS: usize>(
    space: &mut AddressSpace<MAPPINGS>,
    load_base: u64,
    user: bool,
    allocations: &[Option<PeAllocatedRegion>],
    count: usize,
    output: &mut [Option<MappingId>],
) -> Result<usize, PeMappingError> {
    if count == 0 || count > MAX_PE_SECTIONS + 1 || count > allocations.len() {
        return Err(PeMappingError::InvalidCount);
    }
    if output.len() < count {
        return Err(PeMappingError::OutputTooSmall);
    }
    output.fill(None);
    let mut prepared = [None; MAX_PE_SECTIONS + 1];
    for (index, slot) in prepared[..count].iter_mut().enumerate() {
        let allocation = allocations[index].ok_or(PeMappingError::MissingAllocation)?;
        let load = allocation.load();
        let virtual_address = load_base
            .checked_add(load.virtual_address() as u64)
            .ok_or(PeMappingError::Address(AddressSpaceError::Overflow))?;
        let permissions = match (user, load.writable(), load.executable()) {
            (true, true, false) => PagePermissions::USER_READ_WRITE,
            (true, false, true) => PagePermissions::USER_READ_EXECUTE,
            (true, false, false) => PagePermissions::USER_READ,
            (false, true, false) => PagePermissions::KERNEL_READ_WRITE,
            (false, false, true) => PagePermissions::KERNEL_READ_EXECUTE,
            (false, false, false) => PagePermissions::KERNEL_READ,
            (_, true, true) => {
                return Err(PeMappingError::Address(AddressSpaceError::InvalidMapping));
            }
        };
        *slot = Some(
            Mapping::new(
                VirtAddr::new(virtual_address)
                    .map_err(|_| PeMappingError::Address(AddressSpaceError::InvalidMapping))?,
                allocation.physical_start(),
                load.pages() as u64,
                permissions,
            )
            .map_err(PeMappingError::Address)?,
        );
    }
    for index in 0..count {
        let mapping = prepared[index].ok_or(PeMappingError::MissingAllocation)?;
        match space.map(mapping) {
            Ok(id) => output[index] = Some(id),
            Err(error) => {
                for rollback in (0..index).rev() {
                    let id = output[rollback]
                        .take()
                        .ok_or(PeMappingError::RollbackFailure)?;
                    space
                        .unmap(id)
                        .map_err(|_| PeMappingError::RollbackFailure)?;
                }
                return Err(PeMappingError::Address(error));
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PeLoadRegion, PhysAddr};

    fn allocation(rva: u32, physical: u64, executable: bool) -> Option<PeAllocatedRegion> {
        Some(PeAllocatedRegion::test_value(
            PeLoadRegion::test_value(rva, 1, false, executable),
            PhysAddr::new(physical).unwrap(),
        ))
    }

    #[test]
    fn maps_complete_user_image_with_final_permissions() {
        let allocations = [allocation(0, 0, false), allocation(0x1000, 0x1000, true)];
        let mut output = [None; 2];
        let mut space = AddressSpace::<2>::new();
        assert_eq!(
            map_pe_image(
                &mut space,
                0x1_4000_0000,
                true,
                &allocations,
                2,
                &mut output
            ),
            Ok(2)
        );
        assert!(
            !space
                .get(output[0].unwrap())
                .unwrap()
                .permissions()
                .executable()
        );
        assert!(
            space
                .get(output[1].unwrap())
                .unwrap()
                .permissions()
                .executable()
        );
    }

    #[test]
    fn rolls_back_every_mapping_after_late_conflict() {
        let allocations = [allocation(0, 0, false), allocation(0x1000, 0x1000, true)];
        let base = 0x1_4000_0000;
        let mut space = AddressSpace::<3>::new();
        space
            .map(
                Mapping::new(
                    VirtAddr::new(base + 0x1000).unwrap(),
                    PhysAddr::new(0x9000).unwrap(),
                    1,
                    PagePermissions::USER_READ,
                )
                .unwrap(),
            )
            .unwrap();
        let mut output = [None; 2];
        assert!(matches!(
            map_pe_image(&mut space, base, true, &allocations, 2, &mut output),
            Err(PeMappingError::Address(AddressSpaceError::VirtualOverlap))
        ));
        assert_eq!(output, [None, None]);
        assert!(
            space
                .map(
                    Mapping::new(
                        VirtAddr::new(base).unwrap(),
                        PhysAddr::new(0).unwrap(),
                        1,
                        PagePermissions::USER_READ,
                    )
                    .unwrap()
                )
                .is_ok()
        );
    }
}
