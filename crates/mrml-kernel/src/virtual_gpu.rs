use core::array;
use mrml_crypto::hmac_sha256;

pub const MAX_DISPATCH_BUFFERS: usize = 16;
const MAX_BLOCK_THREADS: u64 = 1024;
const MAX_SHARED_MEMORY: u32 = 96 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuError {
    ZeroAllocation,
    QuotaExceeded,
    BufferTableFull,
    InvalidBuffer,
    RangeOverflow,
    OutOfBounds,
    InvalidKernel,
    InvalidGrid,
    TooManyBuffers,
    ExcessiveSharedMemory,
    MalformedCommand,
    CommandBufferTooSmall,
    UnsupportedCommand,
    InvalidQueueKey,
    AuthenticationFailed,
    WrongSession,
    Replay,
    SequenceExhausted,
}

pub const GPU_QUEUE_MESSAGE_BYTES: usize = 80;
const GPU_QUEUE_AUTHENTICATED_BYTES: usize = 48;

/// Authenticated producer state for an untrusted shared-memory transport.
/// The transport may copy or modify bytes but cannot mint accepted commands
/// without the per-session key held by the guest and isolated GPU service.
pub struct GpuQueueSender {
    session: u64,
    next_sequence: u64,
    key: [u8; 32],
}

impl GpuQueueSender {
    pub fn new(session: u64, key: [u8; 32]) -> Result<Self, GpuError> {
        validate_queue_identity(session, &key)?;
        Ok(Self {
            session,
            next_sequence: 1,
            key,
        })
    }

    pub fn encode(
        &mut self,
        request_id: u64,
        command: ResourceCommand,
        output: &mut [u8],
    ) -> Result<(), GpuError> {
        let output = output
            .get_mut(..GPU_QUEUE_MESSAGE_BYTES)
            .ok_or(GpuError::CommandBufferTooSmall)?;
        let sequence = self.next_sequence;
        let next = sequence.checked_add(1).ok_or(GpuError::SequenceExhausted)?;
        output.fill(0);
        output[..4].copy_from_slice(b"MRGQ");
        output[4] = 1;
        output[8..16].copy_from_slice(&self.session.to_le_bytes());
        output[16..24].copy_from_slice(&sequence.to_le_bytes());
        command.encode(request_id, &mut output[24..48])?;
        let tag = queue_tag(&self.key, &output[..GPU_QUEUE_AUTHENTICATED_BYTES]);
        output[48..].copy_from_slice(&tag);
        self.next_sequence = next;
        Ok(())
    }
}

/// Authenticated consumer state. Sequence advances only after the tag,
/// session, canonical command, and exact expected sequence all validate.
pub struct GpuQueueReceiver {
    session: u64,
    next_sequence: u64,
    key: [u8; 32],
}

impl GpuQueueReceiver {
    pub fn new(session: u64, key: [u8; 32]) -> Result<Self, GpuError> {
        validate_queue_identity(session, &key)?;
        Ok(Self {
            session,
            next_sequence: 1,
            key,
        })
    }

    pub fn decode(&mut self, input: &[u8]) -> Result<(u64, ResourceCommand), GpuError> {
        if input.len() != GPU_QUEUE_MESSAGE_BYTES
            || &input[..4] != b"MRGQ"
            || input[4] != 1
            || input[5..8].iter().any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let expected_tag = queue_tag(&self.key, &input[..GPU_QUEUE_AUTHENTICATED_BYTES]);
        if !constant_time_equal(&expected_tag, &input[48..80]) {
            return Err(GpuError::AuthenticationFailed);
        }
        let session = u64::from_le_bytes(input[8..16].try_into().unwrap());
        if session != self.session {
            return Err(GpuError::WrongSession);
        }
        let sequence = u64::from_le_bytes(input[16..24].try_into().unwrap());
        if sequence != self.next_sequence {
            return Err(GpuError::Replay);
        }
        let decoded = ResourceCommand::decode(&input[24..48])?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(GpuError::SequenceExhausted)?;
        Ok(decoded)
    }
}

fn validate_queue_identity(session: u64, key: &[u8; 32]) -> Result<(), GpuError> {
    if session == 0 || key.iter().all(|byte| *byte == 0) {
        return Err(GpuError::InvalidQueueKey);
    }
    Ok(())
}

fn queue_tag(key: &[u8; 32], authenticated: &[u8]) -> [u8; 32] {
    hmac_sha256(key, &[b"MRML-VGPU-QUEUE-v1\0", authenticated])
}

fn constant_time_equal(expected: &[u8; 32], candidate: &[u8]) -> bool {
    if candidate.len() != expected.len() {
        return false;
    }
    let mut difference = 0u8;
    for index in 0..expected.len() {
        difference |= expected[index] ^ candidate[index];
    }
    difference == 0
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferId {
    slot: u32,
    generation: u32,
}

impl BufferId {
    const fn token(self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }

    const fn from_token(token: u64) -> Result<Self, GpuError> {
        if token >> 32 == 0 {
            return Err(GpuError::MalformedCommand);
        }
        Ok(Self {
            slot: token as u32,
            generation: (token >> 32) as u32,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceCommand {
    Allocate { bytes: u64 },
    Free { buffer: BufferId },
}

impl ResourceCommand {
    pub const WIRE_LENGTH: usize = 24;

    /// Fixed-size MRVG resource command. Capability IPC supplies transport
    /// authentication and replay protection; this layer rejects alternate
    /// encodings so signed/audited requests have a single representation.
    pub fn encode(self, request_id: u64, output: &mut [u8]) -> Result<(), GpuError> {
        if request_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        let output = output
            .get_mut(..Self::WIRE_LENGTH)
            .ok_or(GpuError::CommandBufferTooSmall)?;
        output.fill(0);
        output[..4].copy_from_slice(b"MRVG");
        output[4] = 1;
        output[5] = match self {
            Self::Allocate { .. } => 1,
            Self::Free { .. } => 2,
        };
        output[8..16].copy_from_slice(&request_id.to_le_bytes());
        let value = match self {
            Self::Allocate { bytes } => bytes,
            Self::Free { buffer } => buffer.token(),
        };
        output[16..24].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn decode(input: &[u8]) -> Result<(u64, Self), GpuError> {
        if input.len() != Self::WIRE_LENGTH
            || &input[..4] != b"MRVG"
            || input[4] != 1
            || input[6..8].iter().any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let request_id = u64::from_le_bytes(input[8..16].try_into().unwrap());
        if request_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        let value = u64::from_le_bytes(input[16..24].try_into().unwrap());
        let command = match input[5] {
            1 if value != 0 => Self::Allocate { bytes: value },
            2 => Self::Free {
                buffer: BufferId::from_token(value)?,
            },
            1 => return Err(GpuError::MalformedCommand),
            _ => return Err(GpuError::UnsupportedCommand),
        };
        Ok((request_id, command))
    }
}

#[derive(Clone, Copy)]
struct Buffer {
    generation: u32,
    bytes: u64,
    occupied: bool,
}

impl Buffer {
    const EMPTY: Self = Self {
        generation: 1,
        bytes: 0,
        occupied: false,
    };
}

/// Per-guest GPU resource accounting. Device pointers never cross this API;
/// the host GPU service resolves opaque generational IDs inside its own CUDA
/// context after every range has been checked.
pub struct VirtualGpuSession<const BUFFERS: usize> {
    buffers: [Buffer; BUFFERS],
    quota: u64,
    allocated: u64,
}

impl<const BUFFERS: usize> VirtualGpuSession<BUFFERS> {
    pub fn new(quota: u64) -> Self {
        Self {
            buffers: array::from_fn(|_| Buffer::EMPTY),
            quota,
            allocated: 0,
        }
    }

    pub fn allocate(&mut self, bytes: u64) -> Result<BufferId, GpuError> {
        if bytes == 0 {
            return Err(GpuError::ZeroAllocation);
        }
        let allocated = self
            .allocated
            .checked_add(bytes)
            .ok_or(GpuError::QuotaExceeded)?;
        if allocated > self.quota {
            return Err(GpuError::QuotaExceeded);
        }
        let (slot, buffer) = self
            .buffers
            .iter_mut()
            .enumerate()
            .find(|(_, buffer)| !buffer.occupied && buffer.generation != 0)
            .ok_or(GpuError::BufferTableFull)?;
        buffer.bytes = bytes;
        buffer.occupied = true;
        self.allocated = allocated;
        Ok(BufferId {
            slot: slot as u32,
            generation: buffer.generation,
        })
    }

    pub fn free(&mut self, id: BufferId) -> Result<u64, GpuError> {
        let buffer = self.buffer_mut(id)?;
        let bytes = buffer.bytes;
        buffer.bytes = 0;
        buffer.occupied = false;
        buffer.generation = buffer.generation.checked_add(1).unwrap_or(0);
        self.allocated -= bytes;
        Ok(bytes)
    }

    pub fn validate_access(&self, access: BufferAccess) -> Result<(), GpuError> {
        let buffer = self.buffer(access.buffer)?;
        let end = access
            .offset
            .checked_add(access.length)
            .ok_or(GpuError::RangeOverflow)?;
        if access.length == 0 || end > buffer.bytes {
            return Err(GpuError::OutOfBounds);
        }
        Ok(())
    }

    pub fn validate_dispatch(&self, dispatch: &Dispatch) -> Result<(), GpuError> {
        for access in dispatch.accesses() {
            self.validate_access(access)?;
        }
        Ok(())
    }

    pub const fn allocated_bytes(&self) -> u64 {
        self.allocated
    }

    fn buffer(&self, id: BufferId) -> Result<&Buffer, GpuError> {
        self.buffers
            .get(id.slot as usize)
            .filter(|buffer| buffer.occupied && buffer.generation == id.generation)
            .ok_or(GpuError::InvalidBuffer)
    }

    fn buffer_mut(&mut self, id: BufferId) -> Result<&mut Buffer, GpuError> {
        self.buffers
            .get_mut(id.slot as usize)
            .filter(|buffer| buffer.occupied && buffer.generation == id.generation)
            .ok_or(GpuError::InvalidBuffer)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelId(u8);

impl KernelId {
    /// IDs 0..28 correspond to the original MRML PTX kernels embedded in the
    /// host build. Guests cannot provide names, modules, PTX, or machine code.
    pub const fn new(id: u8) -> Result<Self, GpuError> {
        if id >= 28 {
            return Err(GpuError::InvalidKernel);
        }
        Ok(Self(id))
    }
    pub const fn get(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferAccess {
    buffer: BufferId,
    offset: u64,
    length: u64,
    mode: BufferMode,
}

impl BufferAccess {
    pub const fn new(buffer: BufferId, offset: u64, length: u64, mode: BufferMode) -> Self {
        Self {
            buffer,
            offset,
            length,
            mode,
        }
    }
    pub const fn buffer(self) -> BufferId {
        self.buffer
    }
    pub const fn offset(self) -> u64 {
        self.offset
    }
    pub const fn length(self) -> u64 {
        self.length
    }
    pub const fn mode(self) -> BufferMode {
        self.mode
    }
}

pub struct Dispatch {
    kernel: KernelId,
    grid: [u32; 3],
    block: [u32; 3],
    shared_memory: u32,
    accesses: [Option<BufferAccess>; MAX_DISPATCH_BUFFERS],
    access_count: u8,
}

impl Dispatch {
    pub const WIRE_LENGTH: usize = 48 + MAX_DISPATCH_BUFFERS * 32;

    pub fn new(
        kernel: KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory: u32,
        accesses: &[BufferAccess],
    ) -> Result<Self, GpuError> {
        if grid.contains(&0) || block.contains(&0) {
            return Err(GpuError::InvalidGrid);
        }
        let threads = (block[0] as u64)
            .checked_mul(block[1] as u64)
            .and_then(|value| value.checked_mul(block[2] as u64))
            .ok_or(GpuError::InvalidGrid)?;
        if threads > MAX_BLOCK_THREADS {
            return Err(GpuError::InvalidGrid);
        }
        if shared_memory > MAX_SHARED_MEMORY {
            return Err(GpuError::ExcessiveSharedMemory);
        }
        if accesses.len() > MAX_DISPATCH_BUFFERS {
            return Err(GpuError::TooManyBuffers);
        }
        let mut stored = [None; MAX_DISPATCH_BUFFERS];
        for (slot, access) in stored.iter_mut().zip(accesses) {
            *slot = Some(*access);
        }
        Ok(Self {
            kernel,
            grid,
            block,
            shared_memory,
            accesses: stored,
            access_count: accesses.len() as u8,
        })
    }

    pub const fn kernel(&self) -> KernelId {
        self.kernel
    }
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }
    pub const fn block(&self) -> [u32; 3] {
        self.block
    }
    pub const fn shared_memory(&self) -> u32 {
        self.shared_memory
    }
    pub fn accesses(&self) -> impl Iterator<Item = BufferAccess> + '_ {
        self.accesses[..self.access_count as usize]
            .iter()
            .flatten()
            .copied()
    }

    /// Canonical fixed-size dispatch representation. Unused access slots and
    /// all reserved bytes are zero, preventing alternate encodings from
    /// reaching the host CUDA executor.
    pub fn encode(&self, request_id: u64, output: &mut [u8]) -> Result<(), GpuError> {
        if request_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        let output = output
            .get_mut(..Self::WIRE_LENGTH)
            .ok_or(GpuError::CommandBufferTooSmall)?;
        output.fill(0);
        output[..4].copy_from_slice(b"MRGD");
        output[4] = 1;
        output[5] = self.kernel.get();
        output[6] = self.access_count;
        output[8..16].copy_from_slice(&request_id.to_le_bytes());
        for axis in 0..3 {
            let grid_offset = 16 + axis * 4;
            let block_offset = 28 + axis * 4;
            output[grid_offset..grid_offset + 4].copy_from_slice(&self.grid[axis].to_le_bytes());
            output[block_offset..block_offset + 4].copy_from_slice(&self.block[axis].to_le_bytes());
        }
        output[40..44].copy_from_slice(&self.shared_memory.to_le_bytes());
        for (index, access) in self.accesses().enumerate() {
            let offset = 48 + index * 32;
            output[offset..offset + 8].copy_from_slice(&access.buffer.token().to_le_bytes());
            output[offset + 8..offset + 16].copy_from_slice(&access.offset.to_le_bytes());
            output[offset + 16..offset + 24].copy_from_slice(&access.length.to_le_bytes());
            output[offset + 24] = match access.mode {
                BufferMode::Read => 1,
                BufferMode::Write => 2,
                BufferMode::ReadWrite => 3,
            };
        }
        Ok(())
    }

    pub fn decode(input: &[u8]) -> Result<(u64, Self), GpuError> {
        if input.len() != Self::WIRE_LENGTH
            || &input[..4] != b"MRGD"
            || input[4] != 1
            || input[7] != 0
            || input[44..48].iter().any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let request_id = u64::from_le_bytes(input[8..16].try_into().unwrap());
        let access_count = input[6] as usize;
        if request_id == 0 || access_count > MAX_DISPATCH_BUFFERS {
            return Err(GpuError::MalformedCommand);
        }
        let kernel = KernelId::new(input[5])?;
        let grid = core::array::from_fn(|axis| read_u32(input, 16 + axis * 4));
        let block = core::array::from_fn(|axis| read_u32(input, 28 + axis * 4));
        let shared_memory = read_u32(input, 40);
        let mut accesses = [None; MAX_DISPATCH_BUFFERS];
        for (index, slot) in accesses[..access_count].iter_mut().enumerate() {
            let offset = 48 + index * 32;
            if input[offset + 25..offset + 32]
                .iter()
                .any(|byte| *byte != 0)
            {
                return Err(GpuError::MalformedCommand);
            }
            let mode = match input[offset + 24] {
                1 => BufferMode::Read,
                2 => BufferMode::Write,
                3 => BufferMode::ReadWrite,
                _ => return Err(GpuError::MalformedCommand),
            };
            *slot = Some(BufferAccess::new(
                BufferId::from_token(read_u64(input, offset))?,
                read_u64(input, offset + 8),
                read_u64(input, offset + 16),
                mode,
            ));
        }
        if input[48 + access_count * 32..]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let access_values = core::array::from_fn::<_, MAX_DISPATCH_BUFFERS, _>(|index| {
            accesses[index].unwrap_or(BufferAccess::new(
                BufferId {
                    slot: 0,
                    generation: 1,
                },
                0,
                0,
                BufferMode::Read,
            ))
        });
        let dispatch = Self::new(
            kernel,
            grid,
            block,
            shared_memory,
            &access_values[..access_count],
        )?;
        Ok((request_id, dispatch))
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

    #[test]
    fn quotas_ranges_and_stale_buffer_ids_are_enforced() {
        let mut session = VirtualGpuSession::<2>::new(1024);
        let buffer = session.allocate(768).unwrap();
        assert_eq!(session.allocate(257), Err(GpuError::QuotaExceeded));
        assert_eq!(
            session.validate_access(BufferAccess::new(buffer, 700, 69, BufferMode::Read)),
            Err(GpuError::OutOfBounds)
        );
        assert!(
            session
                .validate_access(BufferAccess::new(buffer, 700, 68, BufferMode::Read))
                .is_ok()
        );
        session.free(buffer).unwrap();
        let replacement = session.allocate(32).unwrap();
        assert_ne!(buffer, replacement);
        assert_eq!(
            session.validate_access(BufferAccess::new(buffer, 0, 1, BufferMode::Read)),
            Err(GpuError::InvalidBuffer)
        );
    }

    #[test]
    fn only_embedded_kernels_and_bounded_launch_shapes_are_accepted() {
        assert!(KernelId::new(27).is_ok());
        assert_eq!(KernelId::new(28), Err(GpuError::InvalidKernel));
        let kernel = KernelId::new(0).unwrap();
        assert!(Dispatch::new(kernel, [1, 1, 1], [32, 1, 1], 0, &[]).is_ok());
        assert!(matches!(
            Dispatch::new(kernel, [1, 1, 1], [1024, 2, 1], 0, &[]),
            Err(GpuError::InvalidGrid)
        ));
        assert!(matches!(
            Dispatch::new(kernel, [1, 1, 1], [32, 1, 1], MAX_SHARED_MEMORY + 1, &[]),
            Err(GpuError::ExcessiveSharedMemory)
        ));
    }

    #[test]
    fn resource_commands_have_one_exact_bounded_encoding() {
        let command = ResourceCommand::Free {
            buffer: BufferId {
                slot: 4,
                generation: 2,
            },
        };
        let mut wire = [0u8; ResourceCommand::WIRE_LENGTH];
        command.encode(7, &mut wire).unwrap();
        assert_eq!(ResourceCommand::decode(&wire), Ok((7, command)));
        assert_eq!(
            ResourceCommand::decode(&wire[..wire.len() - 1]),
            Err(GpuError::MalformedCommand)
        );
        wire[6] = 1;
        assert_eq!(
            ResourceCommand::decode(&wire),
            Err(GpuError::MalformedCommand)
        );
    }

    #[test]
    fn authenticated_queue_rejects_tampering_replay_and_cross_session_use() {
        let key = [7; 32];
        let command = ResourceCommand::Allocate { bytes: 4096 };
        let mut sender = GpuQueueSender::new(11, key).unwrap();
        let mut receiver = GpuQueueReceiver::new(11, key).unwrap();
        let mut wire = [0u8; GPU_QUEUE_MESSAGE_BYTES];
        sender.encode(9, command, &mut wire).unwrap();

        let mut tampered = wire;
        tampered[47] ^= 1;
        assert_eq!(
            receiver.decode(&tampered),
            Err(GpuError::AuthenticationFailed)
        );
        assert_eq!(receiver.decode(&wire), Ok((9, command)));
        assert_eq!(receiver.decode(&wire), Err(GpuError::Replay));

        let mut other = GpuQueueReceiver::new(12, key).unwrap();
        assert_eq!(other.decode(&wire), Err(GpuError::WrongSession));
        let mut wrong_key = GpuQueueReceiver::new(11, [8; 32]).unwrap();
        assert_eq!(wrong_key.decode(&wire), Err(GpuError::AuthenticationFailed));
        assert!(matches!(
            GpuQueueSender::new(11, [0; 32]),
            Err(GpuError::InvalidQueueKey)
        ));
    }

    #[test]
    fn dispatch_wire_is_canonical_bounded_and_pointer_free() {
        let accesses = [
            BufferAccess::new(
                BufferId {
                    slot: 2,
                    generation: 4,
                },
                64,
                128,
                BufferMode::Read,
            ),
            BufferAccess::new(
                BufferId {
                    slot: 3,
                    generation: 5,
                },
                0,
                256,
                BufferMode::Write,
            ),
        ];
        let dispatch = Dispatch::new(
            KernelId::new(7).unwrap(),
            [2, 3, 4],
            [32, 2, 1],
            1024,
            &accesses,
        )
        .unwrap();
        let mut wire = [0u8; Dispatch::WIRE_LENGTH];
        dispatch.encode(19, &mut wire).unwrap();
        let (request, decoded) = Dispatch::decode(&wire).unwrap();
        assert_eq!(request, 19);
        assert_eq!(decoded.kernel(), KernelId::new(7).unwrap());
        assert_eq!(decoded.grid(), [2, 3, 4]);
        assert_eq!(decoded.block(), [32, 2, 1]);
        let mut decoded_accesses = decoded.accesses();
        assert_eq!(decoded_accesses.next(), Some(accesses[0]));
        assert_eq!(decoded_accesses.next(), Some(accesses[1]));
        assert_eq!(decoded_accesses.next(), None);

        wire[48 + 25] = 1;
        assert_eq!(
            Dispatch::decode(&wire).err(),
            Some(GpuError::MalformedCommand)
        );
        let mut wire = [0u8; Dispatch::WIRE_LENGTH];
        dispatch.encode(19, &mut wire).unwrap();
        wire[48 + 24] = 0;
        assert_eq!(
            Dispatch::decode(&wire).err(),
            Some(GpuError::MalformedCommand)
        );
        let mut wire = [0u8; Dispatch::WIRE_LENGTH];
        dispatch.encode(19, &mut wire).unwrap();
        wire[48 + 2 * 32] = 1;
        assert_eq!(
            Dispatch::decode(&wire).err(),
            Some(GpuError::MalformedCommand)
        );
    }
}
