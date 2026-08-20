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
    InvalidDeviceAddress,
    EmptyTopology,
    MissingDevice,
    DuplicateTopologyDevice,
    IncompleteIommuGroup,
    VmTableFull,
    DuplicateVmName,
    DeviceAlreadyAssigned,
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
    segment: u16,
    bus: u8,
    device: u8,
    function: u8,
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

    pub const fn segment(self) -> u16 {
        self.segment
    }
    pub const fn bus(self) -> u8 {
        self.bus
    }
    pub const fn device(self) -> u8 {
        self.device
    }
    pub const fn function(self) -> u8 {
        self.function
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceGrant {
    address: DeviceAddress,
    iommu_group: u32,
}

impl DeviceGrant {
    pub const fn address(self) -> DeviceAddress {
        self.address
    }
    pub const fn iommu_group(self) -> u32 {
        self.iommu_group
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDevice {
    address: DeviceAddress,
    iommu_group: u32,
    reset_supported: bool,
}

impl HostDevice {
    pub const fn new(address: DeviceAddress, iommu_group: u32, reset_supported: bool) -> Self {
        Self {
            address,
            iommu_group,
            reset_supported,
        }
    }
}

pub struct IommuTopology<'a> {
    devices: &'a [HostDevice],
}

impl<'a> IommuTopology<'a> {
    pub fn new(devices: &'a [HostDevice]) -> Result<Self, PolicyError> {
        if devices.is_empty() {
            return Err(PolicyError::EmptyTopology);
        }
        for (index, device) in devices.iter().enumerate() {
            if devices[index + 1..]
                .iter()
                .any(|other| other.address == device.address)
            {
                return Err(PolicyError::DuplicateTopologyDevice);
            }
        }
        Ok(Self { devices })
    }

    pub fn grant(
        &self,
        address: DeviceAddress,
        assigned: &[DeviceAddress],
    ) -> Result<DeviceGrant, PolicyError> {
        let target = self
            .devices
            .iter()
            .find(|device| device.address == address)
            .ok_or(PolicyError::MissingDevice)?;
        let complete = self
            .devices
            .iter()
            .filter(|device| device.iommu_group == target.iommu_group)
            .all(|device| device.reset_supported && assigned.contains(&device.address));
        if !complete {
            return Err(PolicyError::IncompleteIommuGroup);
        }
        Ok(DeviceGrant {
            address,
            iommu_group: target.iommu_group,
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

/// Complete launch policy. Device ownership is globally exclusive, including
/// at IOMMU-group granularity, across every VM in a launch.
pub struct SystemPolicy<const VMS: usize, const DIRECTORIES: usize, const DEVICES: usize> {
    vms: [Option<VmPolicy<DIRECTORIES, DEVICES>>; VMS],
}

impl<const VMS: usize, const DIRECTORIES: usize, const DEVICES: usize>
    SystemPolicy<VMS, DIRECTORIES, DEVICES>
{
    pub fn new() -> Self {
        Self {
            vms: array::from_fn(|_| None),
        }
    }

    pub fn add_vm(&mut self, policy: VmPolicy<DIRECTORIES, DEVICES>) -> Result<(), PolicyError> {
        for existing in self.vms.iter().flatten() {
            if existing.name() == policy.name() {
                return Err(PolicyError::DuplicateVmName);
            }
            for requested in policy.devices() {
                if existing.devices().any(|assigned| {
                    assigned.address == requested.address
                        || assigned.iommu_group == requested.iommu_group
                }) {
                    return Err(PolicyError::DeviceAlreadyAssigned);
                }
            }
        }
        let slot = self
            .vms
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(PolicyError::VmTableFull)?;
        *slot = Some(policy);
        Ok(())
    }

    pub fn vms(&self) -> impl Iterator<Item = &VmPolicy<DIRECTORIES, DEVICES>> {
        self.vms.iter().flatten()
    }
}

impl<const VMS: usize, const DIRECTORIES: usize, const DEVICES: usize> Default
    for SystemPolicy<VMS, DIRECTORIES, DEVICES>
{
    fn default() -> Self {
        Self::new()
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
        let companion = DeviceAddress::new(0, 1, 0, 1).unwrap();
        let devices = [
            HostDevice::new(address, 7, true),
            HostDevice::new(companion, 7, true),
        ];
        let topology = IommuTopology::new(&devices).unwrap();
        assert_eq!(
            topology.grant(address, &[address]),
            Err(PolicyError::IncompleteIommuGroup)
        );
        assert!(topology.grant(address, &[address, companion]).is_ok());
        let unsafe_devices = [HostDevice::new(address, 7, false)];
        let unsafe_topology = IommuTopology::new(&unsafe_devices).unwrap();
        assert_eq!(
            unsafe_topology.grant(address, &[address]),
            Err(PolicyError::IncompleteIommuGroup)
        );
        assert!(DeviceAddress::new(0, 0, 32, 0).is_err());
    }

    #[test]
    fn names_have_no_path_or_protocol_metacharacters() {
        assert!(VmName::new("inference_0").is_ok());
        assert!(VmName::new("../tool").is_err());
        assert!(VmName::new("tool\nadmin").is_err());
    }

    #[test]
    fn system_policy_prevents_duplicate_vm_names_and_device_ownership() {
        let address = DeviceAddress::new(0, 1, 0, 0).unwrap();
        let devices = [HostDevice::new(address, 7, true)];
        let grant = IommuTopology::new(&devices)
            .unwrap()
            .grant(address, &[address])
            .unwrap();
        let mut first = VmPolicy::<0, 1>::new(VmName::new("gpu").unwrap(), VmRole::Device);
        first.add_device(grant).unwrap();
        let mut system = SystemPolicy::<2, 0, 1>::new();
        system.add_vm(first).unwrap();
        assert_eq!(
            system.add_vm(VmPolicy::new(VmName::new("gpu").unwrap(), VmRole::Tool)),
            Err(PolicyError::DuplicateVmName)
        );
        let mut second = VmPolicy::<0, 1>::new(VmName::new("gpu-2").unwrap(), VmRole::Device);
        second.add_device(grant).unwrap();
        assert_eq!(
            system.add_vm(second),
            Err(PolicyError::DeviceAlreadyAssigned)
        );
    }
}
