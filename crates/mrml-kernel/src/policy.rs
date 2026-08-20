use core::array;

use crate::{DirectoryGrant, GrantMode, VmRole};

pub const MAX_VM_NAME: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyError {
    InvalidName,
    DirectoryTableFull,
    DeviceTableFull,
    DuplicateDirectory,
    DuplicateDevice,
    UnsafeDeviceIsolation,
    InvalidDeviceAddress,
}

#[derive(Clone, Eq, PartialEq)]
pub struct VmName {
    bytes: [u8; MAX_VM_NAME],
    length: u8,
}

impl VmName {
    pub fn new(name: &str) -> Result<Self, PolicyError> {
        if name.is_empty()
            || name.len() > MAX_VM_NAME
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(PolicyError::InvalidName);
        }
        let mut bytes = [0; MAX_VM_NAME];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        Ok(Self {
            bytes,
            length: name.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.length as usize])
            .expect("VmName contains validated ASCII")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceAddress {
    pub segment: u16,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl DeviceAddress {
    pub const fn new(segment: u16, bus: u8, device: u8, function: u8) -> Result<Self, PolicyError> {
        if device > 31 || function > 7 {
            return Err(PolicyError::InvalidDeviceAddress);
        }
        Ok(Self {
            segment,
            bus,
            device,
            function,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceGrant {
    pub address: DeviceAddress,
    pub iommu_group: u32,
}

impl DeviceGrant {
    /// Device passthrough is rejected unless the host established an isolated
    /// IOMMU group and a reliable function-level or bus reset path.
    pub const fn new(
        address: DeviceAddress,
        iommu_group: u32,
        isolated_group: bool,
        reset_supported: bool,
    ) -> Result<Self, PolicyError> {
        if !isolated_group || !reset_supported {
            return Err(PolicyError::UnsafeDeviceIsolation);
        }
        Ok(Self {
            address,
            iommu_group,
        })
    }
}

/// Fixed-capacity per-VM launch policy. Empty tables grant no host filesystem
/// or device access. Host backends must translate entries into opened handles
/// and hypervisor/IOMMU assignments rather than exposing ambient host APIs.
pub struct VmPolicy<const DIRECTORIES: usize, const DEVICES: usize> {
    name: VmName,
    role: VmRole,
    directories: [Option<DirectoryGrant>; DIRECTORIES],
    devices: [Option<DeviceGrant>; DEVICES],
}

impl<const DIRECTORIES: usize, const DEVICES: usize> VmPolicy<DIRECTORIES, DEVICES> {
    pub fn new(name: VmName, role: VmRole) -> Self {
        Self {
            name,
            role,
            directories: array::from_fn(|_| None),
            devices: [None; DEVICES],
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub const fn role(&self) -> VmRole {
        self.role
    }

    pub fn add_directory(&mut self, grant: DirectoryGrant) -> Result<(), PolicyError> {
        if self
            .directories
            .iter()
            .flatten()
            .any(|existing| existing.path() == grant.path())
        {
            return Err(PolicyError::DuplicateDirectory);
        }
        let slot = self
            .directories
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(PolicyError::DirectoryTableFull)?;
        *slot = Some(grant);
        Ok(())
    }

    pub fn add_device(&mut self, grant: DeviceGrant) -> Result<(), PolicyError> {
        if self
            .devices
            .iter()
            .flatten()
            .any(|existing| existing.address == grant.address)
        {
            return Err(PolicyError::DuplicateDevice);
        }
        let slot = self
            .devices
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(PolicyError::DeviceTableFull)?;
        *slot = Some(grant);
        Ok(())
    }

    pub fn directory_mode(&self, canonical_path: &str) -> Option<GrantMode> {
        self.directories
            .iter()
            .flatten()
            .find(|grant| grant.path() == canonical_path)
            .map(DirectoryGrant::mode)
    }

    pub fn devices(&self) -> impl Iterator<Item = DeviceGrant> + '_ {
        self.devices.iter().flatten().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_policy_is_deny_by_default_and_write_is_explicit() {
        let mut policy = VmPolicy::<2, 1>::new(VmName::new("tool-1").unwrap(), VmRole::Tool);
        assert_eq!(policy.directory_mode("/workspace"), None);
        policy
            .add_directory(DirectoryGrant::new("/workspace", GrantMode::ReadOnly).unwrap())
            .unwrap();
        assert_eq!(
            policy.directory_mode("/workspace"),
            Some(GrantMode::ReadOnly)
        );
        assert_eq!(policy.directory_mode("/workspace/private"), None);
        assert_eq!(
            policy.add_directory(DirectoryGrant::new("/workspace", GrantMode::ReadWrite).unwrap()),
            Err(PolicyError::DuplicateDirectory)
        );
    }

    #[test]
    fn device_passthrough_requires_isolation_and_reset() {
        let address = DeviceAddress::new(0, 1, 0, 0).unwrap();
        assert_eq!(
            DeviceGrant::new(address, 7, false, true),
            Err(PolicyError::UnsafeDeviceIsolation)
        );
        assert_eq!(
            DeviceGrant::new(address, 7, true, false),
            Err(PolicyError::UnsafeDeviceIsolation)
        );
        assert!(DeviceGrant::new(address, 7, true, true).is_ok());
        assert!(DeviceAddress::new(0, 0, 32, 0).is_err());
    }

    #[test]
    fn names_have_no_path_or_protocol_metacharacters() {
        assert!(VmName::new("inference_0").is_ok());
        assert!(VmName::new("../tool").is_err());
        assert!(VmName::new("tool\nadmin").is_err());
    }
}
