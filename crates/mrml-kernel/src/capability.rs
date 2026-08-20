use core::array;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rights(u16);

impl Rights {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const MAP: Self = Self(1 << 3);
    pub const SIGNAL: Self = Self(1 << 4);
    pub const DELEGATE: Self = Self(1 << 5);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, requested: Self) -> bool {
        self.0 & requested.0 == requested.0
    }

    const fn intersect(self, limit: Self) -> Self {
        Self(self.0 & limit.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capability {
    slot: u32,
    generation: u32,
}

impl Capability {
    pub(crate) const fn token(self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }

    pub(crate) const fn from_token(token: u64) -> Self {
        Self {
            slot: token as u32,
            generation: (token >> 32) as u32,
        }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    generation: u32,
    object: ObjectId,
    rights: Rights,
    occupied: bool,
}

impl Entry {
    const EMPTY: Self = Self {
        generation: 1,
        object: ObjectId(0),
        rights: Rights::NONE,
        occupied: false,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityError {
    SpaceFull,
    Invalid,
    PermissionDenied,
    DelegationDenied,
}

/// A fixed-capacity capability space. Handles carry generations so a revoked
/// slot cannot be reused through a stale handle (the ABA problem).
pub struct CapabilitySpace<const N: usize> {
    entries: [Entry; N],
}

impl<const N: usize> CapabilitySpace<N> {
    pub fn new() -> Self {
        Self {
            entries: array::from_fn(|_| Entry::EMPTY),
        }
    }

    pub fn insert(
        &mut self,
        object: ObjectId,
        rights: Rights,
    ) -> Result<Capability, CapabilityError> {
        let (slot, entry) = self
            .entries
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| !entry.occupied)
            .ok_or(CapabilityError::SpaceFull)?;
        entry.object = object;
        entry.rights = rights;
        entry.occupied = true;
        Ok(Capability {
            slot: slot as u32,
            generation: entry.generation,
        })
    }

    pub fn authorize(
        &self,
        capability: Capability,
        requested: Rights,
    ) -> Result<ObjectId, CapabilityError> {
        let entry = self.entry(capability)?;
        entry
            .rights
            .contains(requested)
            .then_some(entry.object)
            .ok_or(CapabilityError::PermissionDenied)
    }

    pub fn derive(
        &self,
        capability: Capability,
        requested: Rights,
        destination: &mut Self,
    ) -> Result<Capability, CapabilityError> {
        let source = self.entry(capability)?;
        if !source.rights.contains(Rights::DELEGATE) {
            return Err(CapabilityError::DelegationDenied);
        }
        if !source.rights.contains(requested) {
            return Err(CapabilityError::PermissionDenied);
        }
        destination.insert(source.object, source.rights.intersect(requested))
    }

    pub fn revoke(&mut self, capability: Capability) -> Result<(), CapabilityError> {
        let entry = self
            .entries
            .get_mut(capability.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == capability.generation)
            .ok_or(CapabilityError::Invalid)?;
        entry.occupied = false;
        entry.object = ObjectId(0);
        entry.rights = Rights::NONE;
        entry.generation = entry.generation.wrapping_add(1).max(1);
        Ok(())
    }

    fn entry(&self, capability: Capability) -> Result<&Entry, CapabilityError> {
        self.entries
            .get(capability.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == capability.generation)
            .ok_or(CapabilityError::Invalid)
    }
}

impl<const N: usize> Default for CapabilitySpace<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rights_attenuate_and_revoked_handles_never_revalidate() {
        let mut source = CapabilitySpace::<2>::new();
        let rights = Rights::READ.union(Rights::WRITE).union(Rights::DELEGATE);
        let root = source.insert(ObjectId(7), rights).unwrap();
        let mut child = CapabilitySpace::<2>::new();
        let read = source.derive(root, Rights::READ, &mut child).unwrap();
        assert_eq!(child.authorize(read, Rights::READ), Ok(ObjectId(7)));
        assert_eq!(
            child.authorize(read, Rights::WRITE),
            Err(CapabilityError::PermissionDenied)
        );
        child.revoke(read).unwrap();
        let replacement = child.insert(ObjectId(8), Rights::READ).unwrap();
        assert_ne!(read, replacement);
        assert_eq!(
            child.authorize(read, Rights::READ),
            Err(CapabilityError::Invalid)
        );
    }
}
