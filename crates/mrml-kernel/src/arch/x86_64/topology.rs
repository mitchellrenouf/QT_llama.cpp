use core::array;

use super::InstalledApTrampoline;

use super::MAX_X86_64_CPUS;

const MADT_HEADER_BYTES: usize = 44;
const LOCAL_APIC_ENTRY_BYTES: usize = 8;
const LOCAL_X2APIC_ENTRY_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopologyError {
    Truncated,
    WrongSignature,
    NonCanonicalLength,
    BadChecksum,
    MalformedEntry,
    ReservedFlags,
    InvalidApicId,
    InvalidLocalApicAddress,
    DuplicateLocalApicOverride,
    DuplicateApicId,
    DuplicateFirmwareId,
    TooManyCpus,
    NoEnabledCpu,
    MissingBootstrapCpu,
    InvalidCpu,
    InvalidState,
    InvalidStartupVector,
    StaleStartup,
    GenerationExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct X86Cpu {
    apic_id: u32,
    firmware_id: u32,
    x2apic: bool,
}

impl X86Cpu {
    pub const fn apic_id(self) -> u32 {
        self.apic_id
    }
    pub const fn firmware_id(self) -> u32 {
        self.firmware_id
    }
    pub const fn uses_x2apic_id(self) -> bool {
        self.x2apic
    }
}

pub struct X86CpuTopology {
    cpus: [Option<X86Cpu>; MAX_X86_64_CPUS],
    count: usize,
    local_apic_address: u64,
    address_overridden: bool,
}

impl X86CpuTopology {
    /// Parses one complete ACPI Multiple APIC Description Table copied into
    /// trusted bounded memory. Only processors marked enabled are admitted.
    pub fn parse_madt(input: &[u8]) -> Result<Self, TopologyError> {
        if input.len() < MADT_HEADER_BYTES {
            return Err(TopologyError::Truncated);
        }
        if &input[..4] != b"APIC" {
            return Err(TopologyError::WrongSignature);
        }
        let encoded_length = read_u32(input, 4) as usize;
        if encoded_length != input.len() || encoded_length < MADT_HEADER_BYTES {
            return Err(TopologyError::NonCanonicalLength);
        }
        if input.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) != 0 {
            return Err(TopologyError::BadChecksum);
        }
        if read_u32(input, 40) & !1 != 0 {
            return Err(TopologyError::ReservedFlags);
        }
        let mut topology = Self {
            cpus: array::from_fn(|_| None),
            count: 0,
            local_apic_address: u64::from(read_u32(input, 36)),
            address_overridden: false,
        };
        let mut offset = MADT_HEADER_BYTES;
        while offset < input.len() {
            let header = input
                .get(offset..offset + 2)
                .ok_or(TopologyError::MalformedEntry)?;
            let entry_type = header[0];
            let length = usize::from(header[1]);
            if length < 2 {
                return Err(TopologyError::MalformedEntry);
            }
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= input.len())
                .ok_or(TopologyError::MalformedEntry)?;
            let entry = &input[offset..end];
            match entry_type {
                0 => {
                    if length != LOCAL_APIC_ENTRY_BYTES {
                        return Err(TopologyError::MalformedEntry);
                    }
                    topology.admit(
                        u32::from(entry[3]),
                        u32::from(entry[2]),
                        read_u32(entry, 4),
                        false,
                    )?;
                }
                9 => {
                    if length != LOCAL_X2APIC_ENTRY_BYTES
                        || entry[2..4].iter().any(|byte| *byte != 0)
                    {
                        return Err(TopologyError::MalformedEntry);
                    }
                    topology.admit(
                        read_u32(entry, 4),
                        read_u32(entry, 12),
                        read_u32(entry, 8),
                        true,
                    )?;
                }
                5 => {
                    if length != 12 || entry[2..4].iter().any(|byte| *byte != 0) {
                        return Err(TopologyError::MalformedEntry);
                    }
                    if topology.address_overridden {
                        return Err(TopologyError::DuplicateLocalApicOverride);
                    }
                    topology.local_apic_address = read_u64(entry, 4);
                    topology.address_overridden = true;
                }
                _ => {}
            }
            offset = end;
        }
        if topology.count == 0 {
            return Err(TopologyError::NoEnabledCpu);
        }
        if topology.local_apic_address == 0
            || !topology.local_apic_address.is_multiple_of(4096)
            || topology.local_apic_address >> 52 != 0
        {
            return Err(TopologyError::InvalidLocalApicAddress);
        }
        Ok(topology)
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn cpu(&self, index: usize) -> Result<X86Cpu, TopologyError> {
        self.cpus
            .get(index)
            .and_then(|cpu| *cpu)
            .ok_or(TopologyError::InvalidCpu)
    }

    pub const fn local_apic_address(&self) -> u64 {
        self.local_apic_address
    }

    pub fn index_of_apic(&self, apic_id: u32) -> Result<usize, TopologyError> {
        self.cpus[..self.count]
            .iter()
            .position(|cpu| cpu.is_some_and(|cpu| cpu.apic_id == apic_id))
            .ok_or(TopologyError::MissingBootstrapCpu)
    }

    fn admit(
        &mut self,
        apic_id: u32,
        firmware_id: u32,
        flags: u32,
        x2apic: bool,
    ) -> Result<(), TopologyError> {
        if flags & !0b11 != 0 {
            return Err(TopologyError::ReservedFlags);
        }
        if flags & 1 == 0 {
            return Ok(());
        }
        if (!x2apic && apic_id >= 0xff) || (x2apic && apic_id == u32::MAX) {
            return Err(TopologyError::InvalidApicId);
        }
        if self.cpus[..self.count]
            .iter()
            .flatten()
            .any(|cpu| cpu.apic_id == apic_id)
        {
            return Err(TopologyError::DuplicateApicId);
        }
        if self.cpus[..self.count]
            .iter()
            .flatten()
            .any(|cpu| cpu.firmware_id == firmware_id)
        {
            return Err(TopologyError::DuplicateFirmwareId);
        }
        let slot = self
            .cpus
            .get_mut(self.count)
            .ok_or(TopologyError::TooManyCpus)?;
        *slot = Some(X86Cpu {
            apic_id,
            firmware_id,
            x2apic,
        });
        self.count += 1;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApState {
    Offline,
    InitSent,
    StartupSent,
    Online,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApStartupToken {
    slot: u16,
    generation: u32,
}

impl ApStartupToken {
    #[cfg(test)]
    pub(crate) fn from_parts(slot: u16, generation: u32) -> Result<Self, TopologyError> {
        if usize::from(slot) >= MAX_X86_64_CPUS {
            return Err(TopologyError::InvalidCpu);
        }
        if generation == 0 {
            return Err(TopologyError::StaleStartup);
        }
        Ok(Self { slot, generation })
    }

    pub const fn slot(self) -> u16 {
        self.slot
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy)]
struct ApSlot {
    apic_id: u32,
    generation: u32,
    state: ApState,
}

pub struct ApStartupTable<const CPUS: usize> {
    slots: [Option<ApSlot>; CPUS],
    count: usize,
}

impl<const CPUS: usize> ApStartupTable<CPUS> {
    pub fn new(topology: &X86CpuTopology, bsp_apic_id: u32) -> Result<Self, TopologyError> {
        if CPUS == 0 || CPUS > MAX_X86_64_CPUS || topology.len() > CPUS {
            return Err(TopologyError::TooManyCpus);
        }
        topology.index_of_apic(bsp_apic_id)?;
        let mut table = Self {
            slots: array::from_fn(|_| None),
            count: topology.len(),
        };
        for index in 0..topology.len() {
            let cpu = topology.cpu(index)?;
            table.slots[index] = Some(ApSlot {
                apic_id: cpu.apic_id,
                generation: 1,
                state: if cpu.apic_id == bsp_apic_id {
                    ApState::Online
                } else {
                    ApState::Offline
                },
            });
        }
        Ok(table)
    }

    pub fn state(&self, index: usize) -> Result<ApState, TopologyError> {
        self.slot(index).map(|slot| slot.state)
    }

    pub fn begin(&mut self, index: usize) -> Result<ApStartupToken, TopologyError> {
        let slot = self.slot_mut(index)?;
        if slot.state != ApState::Offline {
            return Err(TopologyError::InvalidState);
        }
        slot.state = ApState::InitSent;
        Ok(ApStartupToken {
            slot: index as u16,
            generation: slot.generation,
        })
    }

    pub fn startup_sent(
        &mut self,
        token: ApStartupToken,
        vector: u8,
    ) -> Result<u64, TopologyError> {
        if vector == 0 {
            return Err(TopologyError::InvalidStartupVector);
        }
        let slot = self.token_slot_mut(token)?;
        if slot.state != ApState::InitSent {
            return Err(TopologyError::InvalidState);
        }
        slot.state = ApState::StartupSent;
        Ok(u64::from(vector) << 12)
    }

    pub fn startup_sent_with_image(
        &mut self,
        token: ApStartupToken,
        image: InstalledApTrampoline,
    ) -> Result<u64, TopologyError> {
        let physical = self.startup_sent(token, image.startup_vector())?;
        if physical != image.physical() {
            return Err(TopologyError::InvalidStartupVector);
        }
        Ok(physical)
    }

    pub fn destination(&mut self, token: ApStartupToken) -> Result<u32, TopologyError> {
        let slot = self.token_slot_mut(token)?;
        if !matches!(slot.state, ApState::InitSent | ApState::StartupSent) {
            return Err(TopologyError::InvalidState);
        }
        Ok(slot.apic_id)
    }

    pub fn acknowledge(
        &mut self,
        token: ApStartupToken,
        observed_apic_id: u32,
    ) -> Result<(), TopologyError> {
        let slot = self.token_slot_mut(token)?;
        if slot.state != ApState::StartupSent || slot.apic_id != observed_apic_id {
            return Err(TopologyError::InvalidState);
        }
        slot.state = ApState::Online;
        Ok(())
    }

    pub fn fail(&mut self, token: ApStartupToken) -> Result<(), TopologyError> {
        let slot = self.token_slot_mut(token)?;
        if !matches!(slot.state, ApState::InitSent | ApState::StartupSent) {
            return Err(TopologyError::InvalidState);
        }
        slot.state = ApState::Failed;
        slot.generation = slot
            .generation
            .checked_add(1)
            .filter(|generation| *generation != 0)
            .ok_or(TopologyError::GenerationExhausted)?;
        Ok(())
    }

    /// Rearms a failed slot after higher-level policy has authorized a retry.
    /// The generation was advanced at failure time, so every prior startup
    /// acknowledgement remains permanently stale.
    pub fn rearm_failed(&mut self, index: usize) -> Result<(), TopologyError> {
        let slot = self.slot_mut(index)?;
        if slot.state != ApState::Failed {
            return Err(TopologyError::InvalidState);
        }
        slot.state = ApState::Offline;
        Ok(())
    }

    fn slot(&self, index: usize) -> Result<&ApSlot, TopologyError> {
        if index >= self.count {
            return Err(TopologyError::InvalidCpu);
        }
        self.slots[index].as_ref().ok_or(TopologyError::InvalidCpu)
    }

    fn slot_mut(&mut self, index: usize) -> Result<&mut ApSlot, TopologyError> {
        if index >= self.count {
            return Err(TopologyError::InvalidCpu);
        }
        self.slots[index].as_mut().ok_or(TopologyError::InvalidCpu)
    }

    fn token_slot_mut(&mut self, token: ApStartupToken) -> Result<&mut ApSlot, TopologyError> {
        let slot = self.slot_mut(usize::from(token.slot))?;
        if slot.generation != token.generation {
            return Err(TopologyError::StaleStartup);
        }
        Ok(slot)
    }
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap())
}

fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn madt(entries: &[u8]) -> [u8; 96] {
        let mut table = [0u8; 96];
        let length = MADT_HEADER_BYTES + entries.len();
        table[..4].copy_from_slice(b"APIC");
        table[4..8].copy_from_slice(&(length as u32).to_le_bytes());
        table[8] = 5;
        table[36..40].copy_from_slice(&0xfee0_0000u32.to_le_bytes());
        table[MADT_HEADER_BYTES..length].copy_from_slice(entries);
        let sum = table[..length]
            .iter()
            .fold(0u8, |sum, byte| sum.wrapping_add(*byte));
        table[9] = table[9].wrapping_sub(sum);
        table
    }

    #[test]
    fn parses_enabled_legacy_and_x2apic_cpus() {
        let entries = [
            0, 8, 7, 2, 1, 0, 0, 0, 9, 16, 0, 0, 0x34, 0x12, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 8, 8,
            3, 2, 0, 0, 0,
        ];
        let table = madt(&entries);
        let topology =
            X86CpuTopology::parse_madt(&table[..MADT_HEADER_BYTES + entries.len()]).unwrap();
        assert_eq!(topology.len(), 2);
        assert_eq!(topology.cpu(0).unwrap().apic_id(), 2);
        assert_eq!(topology.cpu(1).unwrap().apic_id(), 0x1234);
        assert!(topology.cpu(1).unwrap().uses_x2apic_id());
        assert_eq!(topology.cpu(1).unwrap().firmware_id(), 9);
        assert_eq!(topology.local_apic_address(), 0xfee0_0000);
    }

    #[test]
    fn validates_local_apic_override_and_rejects_ambiguous_mmio() {
        let entries = [
            0, 8, 0, 1, 1, 0, 0, 0, 5, 12, 0, 0, 0, 0, 0xe0, 0xfe, 0, 0, 0, 0,
        ];
        let table = madt(&entries);
        let topology = X86CpuTopology::parse_madt(&table[..64]).unwrap();
        assert_eq!(topology.local_apic_address(), 0xfee0_0000);

        let duplicate = madt(&[
            0, 8, 0, 1, 1, 0, 0, 0, 5, 12, 0, 0, 0, 0, 0xe0, 0xfe, 0, 0, 0, 0, 5, 12, 0, 0, 0, 0,
            0xe0, 0xfe, 0, 0, 0, 0,
        ]);
        assert_eq!(
            X86CpuTopology::parse_madt(&duplicate[..76]).err(),
            Some(TopologyError::DuplicateLocalApicOverride)
        );
        let unaligned = madt(&[
            0, 8, 0, 1, 1, 0, 0, 0, 5, 12, 0, 0, 1, 0, 0xe0, 0xfe, 0, 0, 0, 0,
        ]);
        assert_eq!(
            X86CpuTopology::parse_madt(&unaligned[..64]).err(),
            Some(TopologyError::InvalidLocalApicAddress)
        );
    }

    #[test]
    fn rejects_checksum_lengths_flags_duplicates_and_empty_topology() {
        let mut bad_checksum = madt(&[0, 8, 0, 1, 1, 0, 0, 0]);
        bad_checksum[20] ^= 1;
        assert_eq!(
            X86CpuTopology::parse_madt(&bad_checksum[..52]).err(),
            Some(TopologyError::BadChecksum)
        );
        let duplicate = madt(&[0, 8, 0, 1, 1, 0, 0, 0, 0, 8, 2, 1, 1, 0, 0, 0]);
        assert_eq!(
            X86CpuTopology::parse_madt(&duplicate[..60]).err(),
            Some(TopologyError::DuplicateApicId)
        );
        let reserved = madt(&[0, 8, 0, 1, 4, 0, 0, 0]);
        assert_eq!(
            X86CpuTopology::parse_madt(&reserved[..52]).err(),
            Some(TopologyError::ReservedFlags)
        );
        let disabled = madt(&[0, 8, 0, 1, 2, 0, 0, 0]);
        assert_eq!(
            X86CpuTopology::parse_madt(&disabled[..52]).err(),
            Some(TopologyError::NoEnabledCpu)
        );
    }

    #[test]
    fn ap_startup_is_ordered_and_stale_tokens_fail() {
        let entries = [0, 8, 0, 1, 1, 0, 0, 0, 0, 8, 1, 2, 1, 0, 0, 0];
        let table = madt(&entries);
        let topology = X86CpuTopology::parse_madt(&table[..60]).unwrap();
        let mut startup = ApStartupTable::<2>::new(&topology, 1).unwrap();
        assert_eq!(startup.state(0), Ok(ApState::Online));
        assert_eq!(startup.state(1), Ok(ApState::Offline));
        let token = startup.begin(1).unwrap();
        assert_eq!(startup.destination(token), Ok(2));
        assert_eq!(
            startup.startup_sent(token, 0),
            Err(TopologyError::InvalidStartupVector)
        );
        assert_eq!(startup.startup_sent(token, 8), Ok(0x8000));
        assert_eq!(
            startup.acknowledge(token, 3),
            Err(TopologyError::InvalidState)
        );
        startup.fail(token).unwrap();
        assert_eq!(startup.state(1), Ok(ApState::Failed));
        assert_eq!(startup.fail(token), Err(TopologyError::StaleStartup));
        startup.rearm_failed(1).unwrap();
        let replacement = startup.begin(1).unwrap();
        assert_ne!(replacement, token);
        assert_eq!(
            startup.startup_sent(token, 8),
            Err(TopologyError::StaleStartup)
        );
        assert_eq!(startup.startup_sent(replacement, 8), Ok(0x8000));
    }

    #[test]
    fn startup_image_is_bound_to_the_sipi_vector() {
        let entries = [0, 8, 0, 1, 1, 0, 0, 0, 0, 8, 1, 2, 1, 0, 0, 0];
        let table = madt(&entries);
        let topology = X86CpuTopology::parse_madt(&table[..60]).unwrap();
        let mut startup = ApStartupTable::<2>::new(&topology, 1).unwrap();
        let token = startup.begin(1).unwrap();
        struct Page(bool);
        impl super::super::ApTrampolinePage for Page {
            fn permissions(&self, _: u64) -> Option<super::super::TrampolinePermissions> {
                Some(super::super::TrampolinePermissions {
                    readable: true,
                    writable: !self.0,
                    executable: self.0,
                })
            }
            fn write_page(&mut self, _: u64, _: &[u8; 4096]) -> bool {
                true
            }
            fn protect_read_execute(&mut self, _: u64) -> bool {
                self.0 = true;
                true
            }
            fn revoke_and_zero(&mut self, _: u64) -> bool {
                true
            }
            fn rearm_read_write_and_zero(&mut self, _: u64) -> bool {
                self.0 = false;
                true
            }
        }
        let image =
            super::super::ApTrampolineImage::new(0x8000, 0x20_0000, 0x1000, 0x9ff8, 1, 1).unwrap();
        let mut page = Page(false);
        let installed = image.install(&mut page).unwrap();
        assert_eq!(
            startup.startup_sent_with_image(token, installed),
            Ok(0x8000)
        );
    }
}
