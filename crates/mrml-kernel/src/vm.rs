use crate::{Capability, CapabilityError, CapabilitySpace, ObjectId, PhysAddr, Rights, PAGE_SIZE};

pub const HYPERCALL_BYTES: usize = 64;
pub const MAX_GUEST_REGIONS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmError {
    EmptyRegion,
    Unaligned,
    Overflow,
    RegionTableFull,
    GuestOverlap,
    HostAlias,
    WritableExecutable,
    Unmapped,
    PermissionDenied,
    MalformedHypercall,
    UnsupportedHypercall,
    Replay,
    SequenceExhausted,
    Capability(CapabilityError),
    WrongObject,
    VmTableFull,
    StaleVm,
    InvalidVmState,
    ExitBudgetExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmId(u64);

impl VmId {
    const fn new(index: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | index as u64)
    }

    pub const fn token(self) -> u64 {
        self.0
    }
    const fn index(self) -> usize {
        self.0 as u32 as usize
    }
    const fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmState {
    Created,
    Loaded,
    Running,
    Stopped,
    Failed,
}

#[derive(Clone, Copy)]
struct VmSlot {
    generation: u32,
    state: VmState,
    exits: u64,
    exit_budget: u64,
}

/// Fixed-capacity VM lifecycle table. IDs include a nonzero generation so a
/// handle retained after destruction cannot address a newly created VM.
pub struct VmTable<const N: usize> {
    slots: [Option<VmSlot>; N],
    generations: [u32; N],
}

impl<const N: usize> VmTable<N> {
    pub const fn new() -> Self {
        Self {
            slots: [None; N],
            generations: [0; N],
        }
    }

    pub fn create(&mut self, exit_budget: u64) -> Result<VmId, VmError> {
        if exit_budget == 0 {
            return Err(VmError::ExitBudgetExceeded);
        }
        let index = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(VmError::VmTableFull)?;
        let generation = self.generations[index]
            .checked_add(1)
            .ok_or(VmError::SequenceExhausted)?;
        self.generations[index] = generation;
        self.slots[index] = Some(VmSlot {
            generation,
            state: VmState::Created,
            exits: 0,
            exit_budget,
        });
        Ok(VmId::new(index as u32, generation))
    }

    pub fn state(&self, id: VmId) -> Result<VmState, VmError> {
        Ok(self.slot(id)?.state)
    }

    pub fn mark_loaded(&mut self, id: VmId) -> Result<(), VmError> {
        self.transition(id, VmState::Created, VmState::Loaded)
    }

    pub fn start(&mut self, id: VmId) -> Result<(), VmError> {
        let slot = self.slot_mut(id)?;
        if !matches!(slot.state, VmState::Loaded | VmState::Stopped) {
            return Err(VmError::InvalidVmState);
        }
        slot.exits = 0;
        slot.state = VmState::Running;
        Ok(())
    }

    pub fn account_exit(&mut self, id: VmId) -> Result<u64, VmError> {
        let slot = self.slot_mut(id)?;
        if slot.state != VmState::Running {
            return Err(VmError::InvalidVmState);
        }
        slot.exits = slot
            .exits
            .checked_add(1)
            .ok_or(VmError::ExitBudgetExceeded)?;
        if slot.exits > slot.exit_budget {
            slot.state = VmState::Failed;
            return Err(VmError::ExitBudgetExceeded);
        }
        Ok(slot.exit_budget - slot.exits)
    }

    pub fn stop(&mut self, id: VmId) -> Result<(), VmError> {
        self.transition(id, VmState::Running, VmState::Stopped)
    }

    pub fn fail(&mut self, id: VmId) -> Result<(), VmError> {
        self.slot_mut(id)?.state = VmState::Failed;
        Ok(())
    }

    pub fn destroy(&mut self, id: VmId) -> Result<(), VmError> {
        self.slot(id)?;
        self.slots[id.index()] = None;
        Ok(())
    }

    fn transition(&mut self, id: VmId, from: VmState, to: VmState) -> Result<(), VmError> {
        let slot = self.slot_mut(id)?;
        if slot.state != from {
            return Err(VmError::InvalidVmState);
        }
        slot.state = to;
        Ok(())
    }

    fn slot(&self, id: VmId) -> Result<&VmSlot, VmError> {
        let slot = self
            .slots
            .get(id.index())
            .and_then(Option::as_ref)
            .ok_or(VmError::StaleVm)?;
        (slot.generation == id.generation())
            .then_some(slot)
            .ok_or(VmError::StaleVm)
    }

    fn slot_mut(&mut self, id: VmId) -> Result<&mut VmSlot, VmError> {
        let slot = self
            .slots
            .get_mut(id.index())
            .and_then(Option::as_mut)
            .ok_or(VmError::StaleVm)?;
        (slot.generation == id.generation())
            .then_some(slot)
            .ok_or(VmError::StaleVm)
    }
}

impl<const N: usize> Default for VmTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestAccess {
    Read,
    Write,
    Execute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestRegion {
    guest_start: u64,
    host_start: PhysAddr,
    pages: u64,
    writable: bool,
    executable: bool,
}

impl GuestRegion {
    pub fn new(
        guest_start: u64,
        host_start: PhysAddr,
        pages: u64,
        writable: bool,
        executable: bool,
    ) -> Result<Self, VmError> {
        if pages == 0 {
            return Err(VmError::EmptyRegion);
        }
        if guest_start % PAGE_SIZE != 0 {
            return Err(VmError::Unaligned);
        }
        if writable && executable {
            return Err(VmError::WritableExecutable);
        }
        let length = pages.checked_mul(PAGE_SIZE).ok_or(VmError::Overflow)?;
        guest_start.checked_add(length).ok_or(VmError::Overflow)?;
        host_start
            .get()
            .checked_add(length)
            .ok_or(VmError::Overflow)?;
        Ok(Self {
            guest_start,
            host_start,
            pages,
            writable,
            executable,
        })
    }

    fn length(self) -> u64 {
        self.pages * PAGE_SIZE
    }
}

pub struct GuestMemory<const N: usize> {
    regions: [Option<GuestRegion>; N],
    count: usize,
}

impl<const N: usize> GuestMemory<N> {
    pub const fn new() -> Self {
        Self {
            regions: [None; N],
            count: 0,
        }
    }

    pub fn map(&mut self, region: GuestRegion) -> Result<(), VmError> {
        if N > MAX_GUEST_REGIONS || self.count == N {
            return Err(VmError::RegionTableFull);
        }
        let guest_end = region.guest_start + region.length();
        let host_end = region.host_start.get() + region.length();
        for existing in self.regions[..self.count].iter().flatten().copied() {
            if overlaps(
                region.guest_start,
                guest_end,
                existing.guest_start,
                existing.guest_start + existing.length(),
            ) {
                return Err(VmError::GuestOverlap);
            }
            if overlaps(
                region.host_start.get(),
                host_end,
                existing.host_start.get(),
                existing.host_start.get() + existing.length(),
            ) {
                return Err(VmError::HostAlias);
            }
        }
        let at = self.regions[..self.count].partition_point(|entry| {
            entry.is_some_and(|value| value.guest_start < region.guest_start)
        });
        self.regions.copy_within(at..self.count, at + 1);
        self.regions[at] = Some(region);
        self.count += 1;
        Ok(())
    }

    pub fn translate(&self, guest: u64, length: u64, access: GuestAccess) -> Result<u64, VmError> {
        if length == 0 {
            return Err(VmError::Unmapped);
        }
        let end = guest.checked_add(length).ok_or(VmError::Overflow)?;
        let region = self.regions[..self.count]
            .iter()
            .flatten()
            .copied()
            .find(|region| {
                guest >= region.guest_start && end <= region.guest_start + region.length()
            })
            .ok_or(VmError::Unmapped)?;
        if (access == GuestAccess::Write && !region.writable)
            || (access == GuestAccess::Execute && !region.executable)
        {
            return Err(VmError::PermissionDenied);
        }
        region
            .host_start
            .get()
            .checked_add(guest - region.guest_start)
            .ok_or(VmError::Overflow)
    }
}

impl<const N: usize> Default for GuestMemory<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum HypercallOperation {
    Yield = 1,
    ToolRequest = 2,
    GpuRequest = 3,
    Shutdown = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hypercall {
    operation: HypercallOperation,
    sequence: u64,
    capability: Capability,
    guest_address: u64,
    length: u32,
    argument: u64,
}

impl Hypercall {
    pub fn decode(input: &[u8]) -> Result<Self, VmError> {
        if input.len() != HYPERCALL_BYTES
            || &input[..8] != b"MRMLHC01"
            || input[10..16].iter().any(|byte| *byte != 0)
            || input[52..].iter().any(|byte| *byte != 0)
        {
            return Err(VmError::MalformedHypercall);
        }
        let operation = match read_u16(input, 8) {
            1 => HypercallOperation::Yield,
            2 => HypercallOperation::ToolRequest,
            3 => HypercallOperation::GpuRequest,
            4 => HypercallOperation::Shutdown,
            _ => return Err(VmError::UnsupportedHypercall),
        };
        let length = read_u32(input, 40);
        if matches!(
            operation,
            HypercallOperation::ToolRequest | HypercallOperation::GpuRequest
        ) && length == 0
        {
            return Err(VmError::MalformedHypercall);
        }
        let capability = read_u64(input, 24);
        let guest_address = read_u64(input, 32);
        let argument = read_u64(input, 44);
        if matches!(operation, HypercallOperation::Yield)
            && (capability != 0 || guest_address != 0 || length != 0 || argument != 0)
        {
            return Err(VmError::MalformedHypercall);
        }
        if matches!(operation, HypercallOperation::Shutdown)
            && (guest_address != 0 || length != 0 || argument != 0)
        {
            return Err(VmError::MalformedHypercall);
        }
        Ok(Self {
            operation,
            sequence: read_u64(input, 16),
            capability: Capability::from_token(capability),
            guest_address,
            length,
            argument,
        })
    }

    pub const fn operation(self) -> HypercallOperation {
        self.operation
    }
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
    pub const fn guest_address(self) -> u64 {
        self.guest_address
    }
    pub const fn length(self) -> u32 {
        self.length
    }
    pub const fn argument(self) -> u64 {
        self.argument
    }

    #[cfg(test)]
    fn encode(self) -> [u8; HYPERCALL_BYTES] {
        let mut output = [0u8; HYPERCALL_BYTES];
        output[..8].copy_from_slice(b"MRMLHC01");
        output[8..10].copy_from_slice(&(self.operation as u16).to_le_bytes());
        output[16..24].copy_from_slice(&self.sequence.to_le_bytes());
        output[24..32].copy_from_slice(&self.capability.token().to_le_bytes());
        output[32..40].copy_from_slice(&self.guest_address.to_le_bytes());
        output[40..44].copy_from_slice(&self.length.to_le_bytes());
        output[44..52].copy_from_slice(&self.argument.to_le_bytes());
        output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedHypercall {
    Yield,
    Tool {
        host_address: u64,
        length: u32,
        argument: u64,
    },
    Gpu {
        host_address: u64,
        length: u32,
        argument: u64,
    },
    Shutdown,
}

/// A bounded, backend-independent description of why a virtual CPU stopped.
/// Platform adapters must reject exits they cannot faithfully translate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmExit {
    Hypercall {
        descriptor_address: u64,
    },
    GuestMemoryFault {
        guest_address: u64,
        access: GuestAccess,
    },
    Io {
        port: u16,
        size: u8,
        write: bool,
        value: u32,
    },
    Halted,
    Interrupted,
    Unknown {
        reason: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmRunError<E> {
    Backend(E),
    Policy(VmError),
}

/// Narrow contract implemented by Hyper-V and KVM adapters. Guest memory is
/// copied through fixed-size caller-owned buffers; no backend-owned pointer is
/// allowed to cross into the policy core.
pub trait VmBackend {
    type Error;

    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error>;
    fn read_guest(&self, guest_address: u64, output: &mut [u8]) -> Result<(), Self::Error>;
    fn write_guest(&mut self, guest_address: u64, input: &[u8]) -> Result<(), Self::Error>;
    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error>;
}

pub fn decode_hypercall_exit<B: VmBackend>(
    backend: &B,
    exit: VmExit,
) -> Result<Option<Hypercall>, VmRunError<B::Error>> {
    let VmExit::Hypercall { descriptor_address } = exit else {
        return Ok(None);
    };
    let mut wire = [0u8; HYPERCALL_BYTES];
    backend
        .read_guest(descriptor_address, &mut wire)
        .map_err(VmRunError::Backend)?;
    Hypercall::decode(&wire)
        .map(Some)
        .map_err(VmRunError::Policy)
}

pub struct HypercallRouter {
    next_sequence: u64,
    tool_object: ObjectId,
    gpu_object: ObjectId,
    control_object: ObjectId,
}

impl HypercallRouter {
    pub const fn new(
        tool_object: ObjectId,
        gpu_object: ObjectId,
        control_object: ObjectId,
    ) -> Self {
        Self {
            next_sequence: 1,
            tool_object,
            gpu_object,
            control_object,
        }
    }

    pub fn route<const C: usize, const M: usize>(
        &mut self,
        call: Hypercall,
        capabilities: &CapabilitySpace<C>,
        memory: &GuestMemory<M>,
    ) -> Result<RoutedHypercall, VmError> {
        if call.sequence != self.next_sequence {
            return Err(VmError::Replay);
        }
        let routed = match call.operation {
            HypercallOperation::Yield => RoutedHypercall::Yield,
            HypercallOperation::ToolRequest => {
                require_object(
                    capabilities,
                    call.capability,
                    Rights::SIGNAL,
                    self.tool_object,
                )?;
                RoutedHypercall::Tool {
                    host_address: memory.translate(
                        call.guest_address,
                        call.length as u64,
                        GuestAccess::Read,
                    )?,
                    length: call.length,
                    argument: call.argument,
                }
            }
            HypercallOperation::GpuRequest => {
                require_object(
                    capabilities,
                    call.capability,
                    Rights::SIGNAL,
                    self.gpu_object,
                )?;
                RoutedHypercall::Gpu {
                    host_address: memory.translate(
                        call.guest_address,
                        call.length as u64,
                        GuestAccess::Read,
                    )?,
                    length: call.length,
                    argument: call.argument,
                }
            }
            HypercallOperation::Shutdown => {
                require_object(
                    capabilities,
                    call.capability,
                    Rights::SIGNAL,
                    self.control_object,
                )?;
                RoutedHypercall::Shutdown
            }
        };
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(VmError::SequenceExhausted)?;
        Ok(routed)
    }
}

fn require_object<const N: usize>(
    capabilities: &CapabilitySpace<N>,
    capability: Capability,
    rights: Rights,
    expected: ObjectId,
) -> Result<(), VmError> {
    let object = capabilities
        .authorize(capability, rights)
        .map_err(VmError::Capability)?;
    (object == expected)
        .then_some(())
        .ok_or(VmError::WrongObject)
}

const fn overlaps(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}
fn read_u16(input: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(input[at..at + 2].try_into().unwrap())
}
fn read_u32(input: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(input[at..at + 4].try_into().unwrap())
}
fn read_u64(input: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(input[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemoryBackend {
        wire: [u8; HYPERCALL_BYTES],
        base: u64,
    }

    impl VmBackend for MemoryBackend {
        type Error = ();

        fn run(&mut self, _vcpu: u32) -> Result<VmExit, Self::Error> {
            Ok(VmExit::Hypercall {
                descriptor_address: self.base,
            })
        }

        fn read_guest(&self, guest_address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
            if guest_address != self.base || output.len() != self.wire.len() {
                return Err(());
            }
            output.copy_from_slice(&self.wire);
            Ok(())
        }

        fn write_guest(&mut self, _guest_address: u64, _input: &[u8]) -> Result<(), Self::Error> {
            Err(())
        }

        fn inject_interrupt(&mut self, _vcpu: u32, _vector: u8) -> Result<(), Self::Error> {
            Err(())
        }
    }

    #[test]
    fn guest_memory_rejects_aliases_crossings_and_wx() {
        assert_eq!(
            GuestRegion::new(0, PhysAddr::new(0).unwrap(), 1, true, true),
            Err(VmError::WritableExecutable)
        );
        let mut memory = GuestMemory::<3>::new();
        memory
            .map(GuestRegion::new(0x1000, PhysAddr::new(0x9000).unwrap(), 2, true, false).unwrap())
            .unwrap();
        assert_eq!(memory.translate(0x1ff0, 32, GuestAccess::Read), Ok(0x9ff0));
        assert_eq!(
            memory.translate(0x2ff0, 32, GuestAccess::Read),
            Err(VmError::Unmapped)
        );
        assert_eq!(
            memory.map(
                GuestRegion::new(0x4000, PhysAddr::new(0xa000).unwrap(), 1, false, true).unwrap()
            ),
            Err(VmError::HostAlias)
        );
    }

    #[test]
    fn routing_requires_exact_sequence_capability_object_and_mapping() {
        let tool = ObjectId(11);
        let gpu = ObjectId(12);
        let control = ObjectId(13);
        let mut capabilities = CapabilitySpace::<4>::new();
        let tool_cap = capabilities.insert(tool, Rights::SIGNAL).unwrap();
        let gpu_cap = capabilities.insert(gpu, Rights::SIGNAL).unwrap();
        let mut memory = GuestMemory::<1>::new();
        memory
            .map(GuestRegion::new(0x2000, PhysAddr::new(0x8000).unwrap(), 1, false, false).unwrap())
            .unwrap();
        let mut router = HypercallRouter::new(tool, gpu, control);
        let call = Hypercall {
            operation: HypercallOperation::ToolRequest,
            sequence: 1,
            capability: tool_cap,
            guest_address: 0x2080,
            length: 32,
            argument: 7,
        };
        let encoded = call.encode();
        let decoded = Hypercall::decode(&encoded).unwrap();
        assert_eq!(
            router.route(decoded, &capabilities, &memory),
            Ok(RoutedHypercall::Tool {
                host_address: 0x8080,
                length: 32,
                argument: 7
            })
        );
        assert_eq!(
            router.route(decoded, &capabilities, &memory),
            Err(VmError::Replay)
        );
        let wrong = Hypercall {
            operation: HypercallOperation::ToolRequest,
            sequence: 2,
            capability: gpu_cap,
            ..call
        };
        assert_eq!(
            router.route(wrong, &capabilities, &memory),
            Err(VmError::WrongObject)
        );
    }

    #[test]
    fn decoder_rejects_reserved_bytes_unknown_operations_and_empty_requests() {
        let cap = Capability::from_token(1u64 << 32);
        let call = Hypercall {
            operation: HypercallOperation::GpuRequest,
            sequence: 1,
            capability: cap,
            guest_address: 0x1000,
            length: 1,
            argument: 0,
        };
        let mut encoded = call.encode();
        encoded[63] = 1;
        assert_eq!(
            Hypercall::decode(&encoded),
            Err(VmError::MalformedHypercall)
        );
        encoded = call.encode();
        encoded[8..10].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            Hypercall::decode(&encoded),
            Err(VmError::UnsupportedHypercall)
        );
        encoded = call.encode();
        encoded[40..44].fill(0);
        assert_eq!(
            Hypercall::decode(&encoded),
            Err(VmError::MalformedHypercall)
        );

        let yield_call = Hypercall {
            operation: HypercallOperation::Yield,
            sequence: 1,
            capability: cap,
            guest_address: 0,
            length: 0,
            argument: 0,
        };
        assert_eq!(
            Hypercall::decode(&yield_call.encode()),
            Err(VmError::MalformedHypercall)
        );
    }

    #[test]
    fn failed_authorization_does_not_consume_sequence() {
        let tool = ObjectId(1);
        let wrong = ObjectId(2);
        let mut capabilities = CapabilitySpace::<2>::new();
        let wrong_cap = capabilities.insert(wrong, Rights::SIGNAL).unwrap();
        let right_cap = capabilities.insert(tool, Rights::SIGNAL).unwrap();
        let mut memory = GuestMemory::<1>::new();
        memory
            .map(GuestRegion::new(0x1000, PhysAddr::new(0x8000).unwrap(), 1, false, false).unwrap())
            .unwrap();
        let mut router = HypercallRouter::new(tool, ObjectId(3), ObjectId(4));
        let base = Hypercall {
            operation: HypercallOperation::ToolRequest,
            sequence: 1,
            capability: wrong_cap,
            guest_address: 0x1000,
            length: 1,
            argument: 0,
        };
        assert_eq!(
            router.route(base, &capabilities, &memory),
            Err(VmError::WrongObject)
        );
        assert_eq!(
            router.route(
                Hypercall {
                    capability: right_cap,
                    ..base
                },
                &capabilities,
                &memory
            ),
            Ok(RoutedHypercall::Tool {
                host_address: 0x8000,
                length: 1,
                argument: 0
            })
        );
    }

    #[test]
    fn backend_exit_copies_and_decodes_a_fixed_descriptor() {
        let call = Hypercall {
            operation: HypercallOperation::Yield,
            sequence: 1,
            capability: Capability::from_token(0),
            guest_address: 0,
            length: 0,
            argument: 0,
        };
        let backend = MemoryBackend {
            wire: call.encode(),
            base: 0x4000,
        };
        assert_eq!(
            decode_hypercall_exit(
                &backend,
                VmExit::Hypercall {
                    descriptor_address: 0x4000
                }
            ),
            Ok(Some(call))
        );
        assert_eq!(decode_hypercall_exit(&backend, VmExit::Halted), Ok(None));
        assert_eq!(
            decode_hypercall_exit(
                &backend,
                VmExit::Hypercall {
                    descriptor_address: 0x5000
                }
            ),
            Err(VmRunError::Backend(()))
        );
    }

    #[test]
    fn lifecycle_rejects_stale_ids_and_invalid_transitions() {
        let mut table = VmTable::<1>::new();
        let first = table.create(2).unwrap();
        assert_eq!(table.start(first), Err(VmError::InvalidVmState));
        table.mark_loaded(first).unwrap();
        table.start(first).unwrap();
        assert_eq!(table.account_exit(first), Ok(1));
        table.stop(first).unwrap();
        table.destroy(first).unwrap();
        let replacement = table.create(1).unwrap();
        assert_ne!(first.token(), replacement.token());
        assert_eq!(table.state(first), Err(VmError::StaleVm));
    }

    #[test]
    fn exit_budget_fail_stops_a_running_vm() {
        let mut table = VmTable::<1>::new();
        let id = table.create(1).unwrap();
        table.mark_loaded(id).unwrap();
        table.start(id).unwrap();
        assert_eq!(table.account_exit(id), Ok(0));
        assert_eq!(table.account_exit(id), Err(VmError::ExitBudgetExceeded));
        assert_eq!(table.state(id), Ok(VmState::Failed));
        assert_eq!(table.start(id), Err(VmError::InvalidVmState));
    }
}
