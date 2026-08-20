use core::array;

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
}
