use core::array;

use super::{MAX_PHYSICAL_ADDRESS, PagePermissions, VirtAddr};
use crate::{PAGE_SIZE, PhysAddr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressSpaceError {
    Empty,
    Overflow,
    CrossesCanonicalBoundary,
    WrongPrivilegeHalf,
    VirtualOverlap,
    PhysicalAlias,
    Full,
    InvalidMapping,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mapping {
    virtual_start: VirtAddr,
    physical_start: PhysAddr,
    pages: u64,
    permissions: PagePermissions,
}

impl Mapping {
    pub fn new(
        virtual_start: VirtAddr,
        physical_start: PhysAddr,
        pages: u64,
        permissions: PagePermissions,
    ) -> Result<Self, AddressSpaceError> {
        if pages == 0 {
            return Err(AddressSpaceError::Empty);
        }
        let length = pages
            .checked_mul(PAGE_SIZE)
            .ok_or(AddressSpaceError::Overflow)?;
        virtual_start
            .get()
            .checked_add(length)
            .ok_or(AddressSpaceError::Overflow)?;
        physical_start
            .get()
            .checked_add(length)
            .ok_or(AddressSpaceError::Overflow)?;
        let virtual_last = virtual_start
            .get()
            .checked_add(length - PAGE_SIZE)
            .ok_or(AddressSpaceError::Overflow)?;
        let virtual_last =
            VirtAddr::new(virtual_last).map_err(|_| AddressSpaceError::CrossesCanonicalBoundary)?;
        let physical_last = physical_start
            .get()
            .checked_add(length - PAGE_SIZE)
            .ok_or(AddressSpaceError::Overflow)?;
        if physical_last > MAX_PHYSICAL_ADDRESS {
            return Err(AddressSpaceError::Overflow);
        }
        let user_half = virtual_start.get() < 1u64 << 47 && virtual_last.get() < 1u64 << 47;
        let kernel_half = virtual_start.get() >= 0xffff_8000_0000_0000
            && virtual_last.get() >= 0xffff_8000_0000_0000;
        if (permissions.user() && !user_half) || (!permissions.user() && !kernel_half) {
            return Err(AddressSpaceError::WrongPrivilegeHalf);
        }
        Ok(Self {
            virtual_start,
            physical_start,
            pages,
            permissions,
        })
    }

    pub const fn virtual_start(self) -> VirtAddr {
        self.virtual_start
    }
    pub const fn physical_start(self) -> PhysAddr {
        self.physical_start
    }
    pub const fn pages(self) -> u64 {
        self.pages
    }
    pub const fn permissions(self) -> PagePermissions {
        self.permissions
    }
    fn virtual_end(self) -> u64 {
        self.virtual_start.get() + self.pages * PAGE_SIZE
    }
    fn physical_end(self) -> u64 {
        self.physical_start.get() + self.pages * PAGE_SIZE
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappingId {
    slot: u32,
    generation: u32,
}

#[derive(Clone, Copy)]
struct Slot {
    mapping: Option<Mapping>,
    generation: u32,
}

impl Slot {
    const EMPTY: Self = Self {
        mapping: None,
        generation: 1,
    };
}

/// Fixed-capacity mapping policy. Physical aliasing is prohibited entirely so
/// a frame cannot be made executable through one virtual address and writable
/// through another.
pub struct AddressSpace<const MAPPINGS: usize> {
    slots: [Slot; MAPPINGS],
}

impl<const MAPPINGS: usize> AddressSpace<MAPPINGS> {
    pub fn new() -> Self {
        Self {
            slots: array::from_fn(|_| Slot::EMPTY),
        }
    }

    pub fn map(&mut self, mapping: Mapping) -> Result<MappingId, AddressSpaceError> {
        for existing in self.slots.iter().filter_map(|slot| slot.mapping) {
            if overlaps(
                mapping.virtual_start.get(),
                mapping.virtual_end(),
                existing.virtual_start.get(),
                existing.virtual_end(),
            ) {
                return Err(AddressSpaceError::VirtualOverlap);
            }
            if overlaps(
                mapping.physical_start.get(),
                mapping.physical_end(),
                existing.physical_start.get(),
                existing.physical_end(),
            ) {
                return Err(AddressSpaceError::PhysicalAlias);
            }
        }
        let (index, slot) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.mapping.is_none() && slot.generation != 0)
            .ok_or(AddressSpaceError::Full)?;
        slot.mapping = Some(mapping);
        Ok(MappingId {
            slot: index as u32,
            generation: slot.generation,
        })
    }

    pub fn get(&self, id: MappingId) -> Result<Mapping, AddressSpaceError> {
        self.slots
            .get(id.slot as usize)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.mapping)
            .ok_or(AddressSpaceError::InvalidMapping)
    }

    pub fn unmap(&mut self, id: MappingId) -> Result<Mapping, AddressSpaceError> {
        let slot = self
            .slots
            .get_mut(id.slot as usize)
            .filter(|slot| slot.generation == id.generation && slot.mapping.is_some())
            .ok_or(AddressSpaceError::InvalidMapping)?;
        let mapping = slot
            .mapping
            .take()
            .ok_or(AddressSpaceError::InvalidMapping)?;
        slot.generation = slot.generation.checked_add(1).unwrap_or(0);
        Ok(mapping)
    }
}

impl<const MAPPINGS: usize> Default for AddressSpace<MAPPINGS> {
    fn default() -> Self {
        Self::new()
    }
}

const fn overlaps(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_mapping(virtual_address: u64, physical_address: u64) -> Mapping {
        Mapping::new(
            VirtAddr::new(virtual_address).unwrap(),
            PhysAddr::new(physical_address).unwrap(),
            1,
            PagePermissions::USER_READ_WRITE,
        )
        .unwrap()
    }

    #[test]
    fn separates_user_and_kernel_halves() {
        assert_eq!(
            Mapping::new(
                VirtAddr::new(0xffff_8000_0000_0000).unwrap(),
                PhysAddr::new(0).unwrap(),
                1,
                PagePermissions::USER_READ,
            ),
            Err(AddressSpaceError::WrongPrivilegeHalf)
        );
        assert_eq!(
            Mapping::new(
                VirtAddr::new(0x4000).unwrap(),
                PhysAddr::new(0).unwrap(),
                1,
                PagePermissions::KERNEL_READ,
            ),
            Err(AddressSpaceError::WrongPrivilegeHalf)
        );
    }

    #[test]
    fn rejects_virtual_overlap_and_physical_aliases() {
        let mut space = AddressSpace::<3>::new();
        space.map(user_mapping(0x1000, 0x1000)).unwrap();
        assert_eq!(
            space.map(user_mapping(0x1000, 0x2000)),
            Err(AddressSpaceError::VirtualOverlap)
        );
        assert_eq!(
            space.map(user_mapping(0x2000, 0x1000)),
            Err(AddressSpaceError::PhysicalAlias)
        );
    }

    #[test]
    fn stale_mapping_ids_cannot_unmap_reused_slots() {
        let mut space = AddressSpace::<1>::new();
        let old = space.map(user_mapping(0x1000, 0x1000)).unwrap();
        space.unmap(old).unwrap();
        let replacement = space.map(user_mapping(0x2000, 0x2000)).unwrap();
        assert_ne!(old, replacement);
        assert_eq!(space.unmap(old), Err(AddressSpaceError::InvalidMapping));
        assert_eq!(
            space.get(replacement).unwrap().virtual_start().get(),
            0x2000
        );
    }
}
