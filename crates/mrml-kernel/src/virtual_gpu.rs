use crate::{ArtifactKind, PAGE_SIZE, VerifiedArtifact};
use core::{
    array,
    sync::atomic::{AtomicU64, Ordering},
};
use mrml_crypto::{Sha3_512, hmac_sha256};

pub const MAX_DISPATCH_BUFFERS: usize = 16;
pub const MAX_DISPATCH_SCALARS: usize = 16;
pub const MAX_BATCH_DISPATCHES: usize = 32;
pub const MAX_GPU_CONTROL_BYTES: usize = 64 * 1024;
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
    TooManyScalars,
    ExcessiveSharedMemory,
    MalformedCommand,
    CommandBufferTooSmall,
    UnsupportedCommand,
    InvalidQueueKey,
    AuthenticationFailed,
    WrongSession,
    Replay,
    SequenceExhausted,
    DuplicateRequest,
    DispatchTableFull,
    InvalidDispatch,
    InvalidDeadline,
    InvalidQueueCapacity,
    QueueFull,
    QueueEmpty,
    InvalidQueueState,
    ReservationPending,
    InvalidReservation,
    InvalidQueueMapping,
    InvalidBatchCapacity,
    EmptyBatch,
    BatchFull,
    ControlBufferTableFull,
    ControlBufferTooLarge,
    InvalidControlBuffer,
    ControlBufferChanged,
    UntrustedKernelBundle,
    UnsupportedKernelSchema,
    InvalidKernelSchema,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchId {
    slot: u32,
    generation: u32,
}

impl DispatchId {
    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy)]
struct InFlightDispatch {
    generation: u32,
    request_id: u64,
    deadline: u64,
    occupied: bool,
}

impl InFlightDispatch {
    const EMPTY: Self = Self {
        generation: 1,
        request_id: 0,
        deadline: 0,
        occupied: false,
    };
}

/// Fixed-capacity watchdog state for host-executed GPU work. Request IDs are
/// never handles: completion and cancellation require a generational ID minted
/// when the dispatch enters this table.
pub struct DispatchTable<const N: usize> {
    entries: [InFlightDispatch; N],
}

impl<const N: usize> DispatchTable<N> {
    pub const fn new() -> Self {
        Self {
            entries: [InFlightDispatch::EMPTY; N],
        }
    }

    pub fn begin(
        &mut self,
        request_id: u64,
        now: u64,
        deadline: u64,
    ) -> Result<DispatchId, GpuError> {
        if request_id == 0 || deadline <= now {
            return Err(GpuError::InvalidDeadline);
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.occupied && entry.request_id == request_id)
        {
            return Err(GpuError::DuplicateRequest);
        }
        let (slot, entry) = self
            .entries
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| !entry.occupied && entry.generation != 0)
            .ok_or(GpuError::DispatchTableFull)?;
        entry.request_id = request_id;
        entry.deadline = deadline;
        entry.occupied = true;
        Ok(DispatchId {
            slot: slot as u32,
            generation: entry.generation,
        })
    }

    pub fn complete(&mut self, id: DispatchId) -> Result<u64, GpuError> {
        self.remove(id)
    }

    pub fn cancel(&mut self, id: DispatchId) -> Result<u64, GpuError> {
        self.remove(id)
    }

    /// Invalidates every expired ID before invoking `expired`, so callbacks
    /// that initiate device reset cannot race a stale completion into a reused
    /// slot. Returns the number of invalidated dispatches.
    pub fn expire<F>(&mut self, now: u64, mut expired: F) -> usize
    where
        F: FnMut(DispatchId, u64),
    {
        let mut count = 0;
        for (slot, entry) in self.entries.iter_mut().enumerate() {
            if entry.occupied && entry.deadline <= now {
                let id = DispatchId {
                    slot: slot as u32,
                    generation: entry.generation,
                };
                let request_id = entry.request_id;
                invalidate_dispatch(entry);
                count += 1;
                expired(id, request_id);
            }
        }
        count
    }

    pub fn contains(&self, id: DispatchId) -> bool {
        self.entries
            .get(id.slot as usize)
            .is_some_and(|entry| entry.occupied && entry.generation == id.generation)
    }

    fn remove(&mut self, id: DispatchId) -> Result<u64, GpuError> {
        let entry = self
            .entries
            .get_mut(id.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == id.generation)
            .ok_or(GpuError::InvalidDispatch)?;
        let request_id = entry.request_id;
        invalidate_dispatch(entry);
        Ok(request_id)
    }
}

impl<const N: usize> Default for DispatchTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

fn invalidate_dispatch(entry: &mut InFlightDispatch) {
    entry.occupied = false;
    entry.request_id = 0;
    entry.deadline = 0;
    entry.generation = entry.generation.checked_add(1).unwrap_or(0);
}

pub const GPU_QUEUE_MESSAGE_BYTES: usize = 80;
pub const MAX_GPU_QUEUE_SLOTS: usize = 256;
const GPU_QUEUE_AUTHENTICATED_BYTES: usize = 48;

/// Architecture-neutral physical layout for two unidirectional shared rings.
/// Command and completion memory are deliberately disjoint so neither endpoint
/// receives write access to the other endpoint's publication area.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSharedQueueLayout {
    command_base: u64,
    completion_base: u64,
    pages_per_ring: u64,
    slots: u16,
}

impl GpuSharedQueueLayout {
    pub fn new(command_base: u64, completion_base: u64, slots: usize) -> Result<Self, GpuError> {
        if slots == 0
            || slots > MAX_GPU_QUEUE_SLOTS
            || command_base == 0
            || completion_base == 0
            || !command_base.is_multiple_of(PAGE_SIZE)
            || !completion_base.is_multiple_of(PAGE_SIZE)
        {
            return Err(GpuError::InvalidQueueMapping);
        }
        let payload = (slots as u64)
            .checked_mul(GPU_QUEUE_MESSAGE_BYTES as u64)
            .ok_or(GpuError::RangeOverflow)?;
        let bytes = (core::mem::size_of::<GpuSharedRingIndices>() as u64)
            .checked_add(payload)
            .ok_or(GpuError::RangeOverflow)?;
        let pages_per_ring = bytes
            .checked_add(PAGE_SIZE - 1)
            .ok_or(GpuError::RangeOverflow)?
            / PAGE_SIZE;
        let length = pages_per_ring
            .checked_mul(PAGE_SIZE)
            .ok_or(GpuError::RangeOverflow)?;
        let command_end = command_base
            .checked_add(length)
            .ok_or(GpuError::RangeOverflow)?;
        let completion_end = completion_base
            .checked_add(length)
            .ok_or(GpuError::RangeOverflow)?;
        if command_base < completion_end && completion_base < command_end {
            return Err(GpuError::InvalidQueueMapping);
        }
        Ok(Self {
            command_base,
            completion_base,
            pages_per_ring,
            slots: slots as u16,
        })
    }

    pub const fn command_base(self) -> u64 {
        self.command_base
    }

    pub const fn completion_base(self) -> u64 {
        self.completion_base
    }

    pub const fn pages_per_ring(self) -> u64 {
        self.pages_per_ring
    }

    pub const fn slots(self) -> u16 {
        self.slots
    }
}

/// A monotonic position and its derived physical slot. Positions do not wrap;
/// exhaustion requires establishing a new authenticated GPU session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRingTicket {
    position: u64,
    slot: u16,
}

impl GpuRingTicket {
    pub const fn position(self) -> u64 {
        self.position
    }

    pub const fn slot(self) -> u16 {
        self.slot
    }
}

/// Producer-side ownership state for a future shared-page SPSC queue. A slot
/// is reserved, filled, and only then committed. At most one unpublished slot
/// exists, making the required release-store boundary explicit to adapters.
pub struct GpuRingProducer<const N: usize> {
    published: u64,
    reserved: Option<GpuRingTicket>,
}

impl<const N: usize> GpuRingProducer<N> {
    pub const fn new() -> Result<Self, GpuError> {
        if N == 0 || N > MAX_GPU_QUEUE_SLOTS {
            return Err(GpuError::InvalidQueueCapacity);
        }
        Ok(Self {
            published: 0,
            reserved: None,
        })
    }

    pub fn reserve(&mut self, consumed: u64) -> Result<GpuRingTicket, GpuError> {
        if self.reserved.is_some() {
            return Err(GpuError::ReservationPending);
        }
        if consumed > self.published {
            return Err(GpuError::InvalidQueueState);
        }
        if self.published - consumed >= N as u64 {
            return Err(GpuError::QueueFull);
        }
        let position = self
            .published
            .checked_add(1)
            .ok_or(GpuError::SequenceExhausted)?;
        let ticket = ring_ticket::<N>(position);
        self.reserved = Some(ticket);
        Ok(ticket)
    }

    pub fn commit(&mut self, ticket: GpuRingTicket) -> Result<u64, GpuError> {
        if self.reserved != Some(ticket) || ticket.position != self.published + 1 {
            return Err(GpuError::InvalidReservation);
        }
        self.published = ticket.position;
        self.reserved = None;
        Ok(self.published)
    }

    pub const fn published(&self) -> u64 {
        self.published
    }
}

/// Consumer-side ownership state. The adapter must acquire-load `published`,
/// copy the indicated immutable slot, authenticate it, then release `ticket`.
pub struct GpuRingConsumer<const N: usize> {
    consumed: u64,
}

/// Cache-line-separated indices suitable for a coherently mapped shared page.
/// The producer writes slot bytes before `publish`; the consumer calls
/// `published` before reading them. The reverse ordering protects slot reuse.
/// Platform code must provide naturally aligned, cache-coherent shared memory.
#[repr(C, align(64))]
pub struct GpuSharedRingIndices {
    published: AtomicU64,
    _published_padding: [u8; 56],
    consumed: AtomicU64,
    _consumed_padding: [u8; 56],
}

impl GpuSharedRingIndices {
    pub const fn new() -> Self {
        Self {
            published: AtomicU64::new(0),
            _published_padding: [0; 56],
            consumed: AtomicU64::new(0),
            _consumed_padding: [0; 56],
        }
    }

    pub fn published(&self) -> u64 {
        self.published.load(Ordering::Acquire)
    }

    pub fn consumed(&self) -> u64 {
        self.consumed.load(Ordering::Acquire)
    }

    /// Release-publishes a filled slot only if the shared counter still has
    /// the producer's expected value. Any external mutation fails closed.
    pub fn publish(&self, previous: u64, ticket: GpuRingTicket) -> Result<(), GpuError> {
        if ticket.position != previous.checked_add(1).ok_or(GpuError::SequenceExhausted)? {
            return Err(GpuError::InvalidReservation);
        }
        self.published
            .compare_exchange(
                previous,
                ticket.position,
                Ordering::Release,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| GpuError::InvalidQueueState)
    }

    /// Release-publishes completion only in exact FIFO order. This makes all
    /// preceding slot reads happen before the producer may observe reuse.
    pub fn consume(&self, previous: u64, ticket: GpuRingTicket) -> Result<(), GpuError> {
        if ticket.position != previous.checked_add(1).ok_or(GpuError::SequenceExhausted)? {
            return Err(GpuError::InvalidReservation);
        }
        self.consumed
            .compare_exchange(
                previous,
                ticket.position,
                Ordering::Release,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| GpuError::InvalidQueueState)
    }
}

impl Default for GpuSharedRingIndices {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> GpuRingConsumer<N> {
    pub const fn new() -> Result<Self, GpuError> {
        if N == 0 || N > MAX_GPU_QUEUE_SLOTS {
            return Err(GpuError::InvalidQueueCapacity);
        }
        Ok(Self { consumed: 0 })
    }

    pub fn acquire(&self, published: u64) -> Result<GpuRingTicket, GpuError> {
        if published < self.consumed || published - self.consumed > N as u64 {
            return Err(GpuError::InvalidQueueState);
        }
        if published == self.consumed {
            return Err(GpuError::QueueEmpty);
        }
        Ok(ring_ticket::<N>(self.consumed + 1))
    }

    pub fn release(&mut self, ticket: GpuRingTicket) -> Result<u64, GpuError> {
        if ticket != ring_ticket::<N>(self.consumed + 1) {
            return Err(GpuError::InvalidReservation);
        }
        self.consumed = ticket.position;
        Ok(self.consumed)
    }

    pub const fn consumed(&self) -> u64 {
        self.consumed
    }
}

fn ring_ticket<const N: usize>(position: u64) -> GpuRingTicket {
    GpuRingTicket {
        position,
        slot: ((position - 1) % N as u64) as u16,
    }
}

/// Fixed-capacity command transport between a guest-facing VMM endpoint and
/// the isolated GPU service. Messages are copied into kernel-owned slots so an
/// untrusted producer cannot mutate a command after it has been admitted. The
/// authenticated receiver still validates the session, sequence, tag, and
/// canonical command after dequeue.
pub struct GpuCommandRing<const N: usize> {
    slots: [[u8; GPU_QUEUE_MESSAGE_BYTES]; N],
    read: usize,
    write: usize,
    count: usize,
}

impl<const N: usize> GpuCommandRing<N> {
    pub const fn new() -> Result<Self, GpuError> {
        if N == 0 || N > MAX_GPU_QUEUE_SLOTS {
            return Err(GpuError::InvalidQueueCapacity);
        }
        Ok(Self {
            slots: [[0; GPU_QUEUE_MESSAGE_BYTES]; N],
            read: 0,
            write: 0,
            count: 0,
        })
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub const fn is_full(&self) -> bool {
        self.count == N
    }

    /// Copies exactly one authenticated wire message into an owned slot.
    /// Admission never overwrites unread work.
    pub fn enqueue(&mut self, message: &[u8]) -> Result<(), GpuError> {
        let message: &[u8; GPU_QUEUE_MESSAGE_BYTES] =
            message.try_into().map_err(|_| GpuError::MalformedCommand)?;
        if self.is_full() {
            return Err(GpuError::QueueFull);
        }
        self.slots[self.write].copy_from_slice(message);
        self.write = (self.write + 1) % N;
        self.count += 1;
        Ok(())
    }

    /// Copies one owned slot to the consumer and erases it before reuse.
    pub fn dequeue(&mut self, output: &mut [u8; GPU_QUEUE_MESSAGE_BYTES]) -> Result<(), GpuError> {
        if self.is_empty() {
            return Err(GpuError::QueueEmpty);
        }
        output.copy_from_slice(&self.slots[self.read]);
        self.slots[self.read].fill(0);
        self.read = (self.read + 1) % N;
        self.count -= 1;
        Ok(())
    }
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCompletionStatus {
    Success,
    Rejected,
    TimedOut,
    DeviceReset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCompletion {
    request_id: u64,
    dispatch: DispatchId,
    status: GpuCompletionStatus,
}

impl GpuCompletion {
    pub fn new(
        request_id: u64,
        dispatch: DispatchId,
        status: GpuCompletionStatus,
    ) -> Result<Self, GpuError> {
        if request_id == 0 || dispatch.generation == 0 {
            return Err(GpuError::InvalidDispatch);
        }
        Ok(Self {
            request_id,
            dispatch,
            status,
        })
    }

    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    pub const fn dispatch(self) -> DispatchId {
        self.dispatch
    }

    pub const fn status(self) -> GpuCompletionStatus {
        self.status
    }
}

/// GPU-service producer for the independently authenticated completion ring.
pub struct GpuCompletionSender {
    session: u64,
    next_sequence: u64,
    key: [u8; 32],
}

impl GpuCompletionSender {
    pub fn new(session: u64, key: [u8; 32]) -> Result<Self, GpuError> {
        validate_queue_identity(session, &key)?;
        Ok(Self {
            session,
            next_sequence: 1,
            key,
        })
    }

    pub fn encode(&mut self, completion: GpuCompletion, output: &mut [u8]) -> Result<(), GpuError> {
        let output = output
            .get_mut(..GPU_QUEUE_MESSAGE_BYTES)
            .ok_or(GpuError::CommandBufferTooSmall)?;
        let sequence = self.next_sequence;
        let next = sequence.checked_add(1).ok_or(GpuError::SequenceExhausted)?;
        output.fill(0);
        output[..4].copy_from_slice(b"MRGC");
        output[4] = 1;
        output[5] = match completion.status {
            GpuCompletionStatus::Success => 1,
            GpuCompletionStatus::Rejected => 2,
            GpuCompletionStatus::TimedOut => 3,
            GpuCompletionStatus::DeviceReset => 4,
        };
        output[8..16].copy_from_slice(&self.session.to_le_bytes());
        output[16..24].copy_from_slice(&sequence.to_le_bytes());
        output[24..32].copy_from_slice(&completion.request_id.to_le_bytes());
        output[32..36].copy_from_slice(&completion.dispatch.slot.to_le_bytes());
        output[36..40].copy_from_slice(&completion.dispatch.generation.to_le_bytes());
        let tag = completion_tag(&self.key, &output[..GPU_QUEUE_AUTHENTICATED_BYTES]);
        output[48..].copy_from_slice(&tag);
        self.next_sequence = next;
        Ok(())
    }
}

/// Guest-facing consumer for authenticated, strictly ordered completions.
pub struct GpuCompletionReceiver {
    session: u64,
    next_sequence: u64,
    key: [u8; 32],
}

impl GpuCompletionReceiver {
    pub fn new(session: u64, key: [u8; 32]) -> Result<Self, GpuError> {
        validate_queue_identity(session, &key)?;
        Ok(Self {
            session,
            next_sequence: 1,
            key,
        })
    }

    pub fn decode(&mut self, input: &[u8]) -> Result<GpuCompletion, GpuError> {
        if input.len() != GPU_QUEUE_MESSAGE_BYTES
            || &input[..4] != b"MRGC"
            || input[4] != 1
            || input[6..8].iter().any(|byte| *byte != 0)
            || input[40..48].iter().any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let expected_tag = completion_tag(&self.key, &input[..GPU_QUEUE_AUTHENTICATED_BYTES]);
        if !constant_time_equal(&expected_tag, &input[48..]) {
            return Err(GpuError::AuthenticationFailed);
        }
        if read_u64(input, 8) != self.session {
            return Err(GpuError::WrongSession);
        }
        if read_u64(input, 16) != self.next_sequence {
            return Err(GpuError::Replay);
        }
        let status = match input[5] {
            1 => GpuCompletionStatus::Success,
            2 => GpuCompletionStatus::Rejected,
            3 => GpuCompletionStatus::TimedOut,
            4 => GpuCompletionStatus::DeviceReset,
            _ => return Err(GpuError::MalformedCommand),
        };
        let completion = GpuCompletion::new(
            read_u64(input, 24),
            DispatchId {
                slot: read_u32(input, 32),
                generation: read_u32(input, 36),
            },
            status,
        )?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(GpuError::SequenceExhausted)?;
        Ok(completion)
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

fn completion_tag(key: &[u8; 32], authenticated: &[u8]) -> [u8; 32] {
    hmac_sha256(key, &[b"MRML-VGPU-COMPLETION-v1\0", authenticated])
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
pub struct ControlBufferId {
    slot: u32,
    generation: u32,
}

impl ControlBufferId {
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

#[derive(Clone, Copy)]
struct ControlBuffer {
    generation: u32,
    length: u32,
    digest: [u8; 64],
    occupied: bool,
}

impl ControlBuffer {
    const EMPTY: Self = Self {
        generation: 1,
        length: 0,
        digest: [0; 64],
        occupied: false,
    };
}

/// Kernel-owned admission records for immutable batch/control descriptors.
/// Shared bytes must match the sealed length and SHA3-512 digest on every use,
/// closing mutation-after-validation without copying a maximum-size graph into
/// privileged memory.
pub struct ControlBufferTable<const N: usize> {
    entries: [ControlBuffer; N],
}

impl<const N: usize> ControlBufferTable<N> {
    pub const fn new() -> Self {
        Self {
            entries: [ControlBuffer::EMPTY; N],
        }
    }

    pub fn seal(&mut self, bytes: &[u8]) -> Result<ControlBufferId, GpuError> {
        if bytes.is_empty() || bytes.len() > MAX_GPU_CONTROL_BYTES {
            return Err(GpuError::ControlBufferTooLarge);
        }
        let (slot, entry) = self
            .entries
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| !entry.occupied && entry.generation != 0)
            .ok_or(GpuError::ControlBufferTableFull)?;
        entry.length = bytes.len() as u32;
        entry.digest = Sha3_512::digest(bytes);
        entry.occupied = true;
        Ok(ControlBufferId {
            slot: slot as u32,
            generation: entry.generation,
        })
    }

    pub fn verify(&self, id: ControlBufferId, bytes: &[u8]) -> Result<(), GpuError> {
        let entry = self.entry(id)?;
        if bytes.len() != entry.length as usize {
            return Err(GpuError::ControlBufferChanged);
        }
        let candidate = Sha3_512::digest(bytes);
        let mut difference = 0u8;
        for (expected, candidate) in entry.digest.iter().zip(candidate.iter()) {
            difference |= *expected ^ *candidate;
        }
        if difference != 0 {
            return Err(GpuError::ControlBufferChanged);
        }
        Ok(())
    }

    pub fn release(&mut self, id: ControlBufferId) -> Result<(), GpuError> {
        let entry = self
            .entries
            .get_mut(id.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == id.generation)
            .ok_or(GpuError::InvalidControlBuffer)?;
        entry.length = 0;
        entry.digest.fill(0);
        entry.occupied = false;
        entry.generation = entry.generation.checked_add(1).unwrap_or(0);
        Ok(())
    }

    fn entry(&self, id: ControlBufferId) -> Result<&ControlBuffer, GpuError> {
        self.entries
            .get(id.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == id.generation)
            .ok_or(GpuError::InvalidControlBuffer)
    }
}

impl<const N: usize> Default for ControlBufferTable<N> {
    fn default() -> Self {
        Self::new()
    }
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
    SubmitBatch { control: ControlBufferId },
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
            Self::SubmitBatch { .. } => 3,
        };
        output[8..16].copy_from_slice(&request_id.to_le_bytes());
        let value = match self {
            Self::Allocate { bytes } => bytes,
            Self::Free { buffer } => buffer.token(),
            Self::SubmitBatch { control } => control.token(),
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
            3 => Self::SubmitBatch {
                control: ControlBufferId::from_token(value)?,
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

/// Proof that the release-signed CUDA bundle is byte-for-byte identical to
/// the PTX embedded in the host GPU service. Construction requires an artifact
/// verification result that callers cannot forge through this module.
pub struct VerifiedGpuKernelBundle {
    version: u64,
    digest: [u8; 64],
}

impl VerifiedGpuKernelBundle {
    pub fn admit(artifact: &VerifiedArtifact, embedded_bundle: &[u8]) -> Result<Self, GpuError> {
        if artifact.kind() != ArtifactKind::CudaKernelBundle || embedded_bundle.is_empty() {
            return Err(GpuError::UntrustedKernelBundle);
        }
        let digest = Sha3_512::digest(embedded_bundle);
        let mut difference = 0u8;
        for (verified, embedded) in artifact.digest().iter().zip(digest.iter()) {
            difference |= *verified ^ *embedded;
        }
        if difference != 0 {
            return Err(GpuError::UntrustedKernelBundle);
        }
        Ok(Self {
            version: artifact.version(),
            digest,
        })
    }

    pub const fn version(&self) -> u64 {
        self.version
    }

    pub const fn digest(&self) -> &[u8; 64] {
        &self.digest
    }

    pub const fn permits(&self, _: KernelId) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarKind {
    U32,
    I32,
    F32,
}

/// One explicitly typed, pointer-free CUDA scalar argument. Floating-point
/// values retain their canonical IEEE-754 bit representation for validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarArg {
    kind: ScalarKind,
    bits: u32,
}

impl ScalarArg {
    pub const fn u32(value: u32) -> Self {
        Self {
            kind: ScalarKind::U32,
            bits: value,
        }
    }

    pub const fn i32(value: i32) -> Self {
        Self {
            kind: ScalarKind::I32,
            bits: value as u32,
        }
    }

    pub const fn f32_bits(bits: u32) -> Self {
        Self {
            kind: ScalarKind::F32,
            bits,
        }
    }

    pub const fn kind(self) -> ScalarKind {
        self.kind
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dispatch {
    kernel: KernelId,
    grid: [u32; 3],
    block: [u32; 3],
    shared_memory: u32,
    accesses: [Option<BufferAccess>; MAX_DISPATCH_BUFFERS],
    access_count: u8,
    scalars: [Option<ScalarArg>; MAX_DISPATCH_SCALARS],
    scalar_count: u8,
}

/// Proof that a dispatch matches a kernel-specific executor ABI. Construction
/// is deliberately fail-closed: kernels without a complete schema cannot be
/// handed to the host CUDA launcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedKernelLaunch {
    dispatch: Dispatch,
    element_count: u32,
}

impl ValidatedKernelLaunch {
    pub const fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }

    pub const fn element_count(&self) -> u32 {
        self.element_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchedDispatch {
    request_id: u64,
    dispatch: Dispatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedGpuDispatch {
    request_id: u64,
    dispatch_id: DispatchId,
    dispatch: Dispatch,
}

impl PreparedGpuDispatch {
    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    pub const fn dispatch_id(self) -> DispatchId {
        self.dispatch_id
    }

    pub const fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }
}

#[derive(Clone, Copy)]
pub struct PreparedGpuBatch<const N: usize> {
    entries: [Option<PreparedGpuDispatch>; N],
    count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSubmitError<E> {
    Rejected,
    Uncertain(E),
}

/// Narrow service boundary for lowering one already validated batch. `false`
/// means the service definitely rejected it before any GPU-visible action;
/// `Err` means acceptance is uncertain and requires watchdog/reset recovery.
pub trait GpuBatchExecutor<const N: usize> {
    type Error;

    fn submit(&mut self, batch: &ValidatedGpuBatch<N>) -> Result<bool, Self::Error>;
}

impl<const N: usize> PreparedGpuBatch<N> {
    fn new() -> Self {
        Self {
            entries: [None; N],
            count: 0,
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = PreparedGpuDispatch> + '_ {
        self.entries[..self.count].iter().flatten().copied()
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Executor-visible batch whose signed kernel bundle and every kernel-specific
/// ABI have been validated. The inner prepared identities remain bound to the
/// watchdog entries minted before this proof was constructed.
pub struct ValidatedGpuBatch<const N: usize> {
    prepared: PreparedGpuBatch<N>,
    launches: [Option<ValidatedKernelLaunch>; N],
}

impl<const N: usize> ValidatedGpuBatch<N> {
    pub const fn prepared(&self) -> &PreparedGpuBatch<N> {
        &self.prepared
    }

    pub fn entries(
        &self,
    ) -> impl Iterator<Item = (PreparedGpuDispatch, ValidatedKernelLaunch)> + '_ {
        self.prepared.entries().zip(
            self.launches[..self.prepared.count]
                .iter()
                .flatten()
                .copied(),
        )
    }

    pub const fn len(&self) -> usize {
        self.prepared.count
    }

    pub const fn is_empty(&self) -> bool {
        self.prepared.count == 0
    }
}

impl VerifiedGpuKernelBundle {
    /// Convert watchdog-bound work into the only batch type accepted by the
    /// executor. Validation is all-or-nothing and has no GPU-visible effects.
    pub fn validate_batch<const N: usize>(
        &self,
        prepared: &PreparedGpuBatch<N>,
    ) -> Result<ValidatedGpuBatch<N>, GpuError> {
        if prepared.is_empty() {
            return Err(GpuError::EmptyBatch);
        }
        let mut launches = [None; N];
        for (index, entry) in prepared.entries().enumerate() {
            if !self.permits(entry.dispatch.kernel) {
                return Err(GpuError::UntrustedKernelBundle);
            }
            launches[index] = Some(entry.dispatch.validate_executor_schema()?);
        }
        Ok(ValidatedGpuBatch {
            prepared: *prepared,
            launches,
        })
    }
}

impl BatchedDispatch {
    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    pub const fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }
}

/// Kernel-validated coarse submission unit for the isolated GPU service. The
/// fixed capacity prevents attacker-controlled allocation; preserving order
/// lets the service submit one CUDA graph/stream sequence per doorbell.
pub struct GpuDispatchBatch<const N: usize> {
    entries: [Option<BatchedDispatch>; N],
    count: usize,
}

impl<const N: usize> GpuDispatchBatch<N> {
    pub const WIRE_HEADER_BYTES: usize = 16;

    pub const fn new() -> Result<Self, GpuError> {
        if N == 0 || N > MAX_BATCH_DISPATCHES {
            return Err(GpuError::InvalidBatchCapacity);
        }
        Ok(Self {
            entries: [None; N],
            count: 0,
        })
    }

    pub fn push<const BUFFERS: usize>(
        &mut self,
        request_id: u64,
        dispatch: Dispatch,
        session: &VirtualGpuSession<BUFFERS>,
    ) -> Result<(), GpuError> {
        if request_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        if self.count == N {
            return Err(GpuError::BatchFull);
        }
        if self.entries[..self.count]
            .iter()
            .flatten()
            .any(|entry| entry.request_id == request_id)
        {
            return Err(GpuError::DuplicateRequest);
        }
        session.validate_dispatch(&dispatch)?;
        self.entries[self.count] = Some(BatchedDispatch {
            request_id,
            dispatch,
        });
        self.count += 1;
        Ok(())
    }

    pub fn entries(&self) -> impl Iterator<Item = BatchedDispatch> + '_ {
        self.entries[..self.count].iter().flatten().copied()
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn validate_ready(&self) -> Result<(), GpuError> {
        if self.is_empty() {
            Err(GpuError::EmptyBatch)
        } else {
            Ok(())
        }
    }

    pub fn encoded_len(&self) -> Result<usize, GpuError> {
        self.validate_ready()?;
        Self::WIRE_HEADER_BYTES
            .checked_add(
                self.count
                    .checked_mul(Dispatch::WIRE_LENGTH)
                    .ok_or(GpuError::RangeOverflow)?,
            )
            .ok_or(GpuError::RangeOverflow)
    }

    /// Canonical exact-length batch representation for a sealed control
    /// buffer. Each member retains its independent request ID and canonical
    /// dispatch encoding; there are no offsets, pointers, or unused entries.
    pub fn encode(&self, batch_id: u64, output: &mut [u8]) -> Result<usize, GpuError> {
        if batch_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        let length = self.encoded_len()?;
        let output = output
            .get_mut(..length)
            .ok_or(GpuError::CommandBufferTooSmall)?;
        output.fill(0);
        output[..4].copy_from_slice(b"MRGB");
        output[4] = 2;
        output[6..8].copy_from_slice(&(self.count as u16).to_le_bytes());
        output[8..16].copy_from_slice(&batch_id.to_le_bytes());
        for (index, entry) in self.entries().enumerate() {
            let start = Self::WIRE_HEADER_BYTES + index * Dispatch::WIRE_LENGTH;
            entry.dispatch.encode(
                entry.request_id,
                &mut output[start..start + Dispatch::WIRE_LENGTH],
            )?;
        }
        Ok(length)
    }

    pub fn decode<const BUFFERS: usize>(
        input: &[u8],
        session: &VirtualGpuSession<BUFFERS>,
    ) -> Result<(u64, Self), GpuError> {
        if input.len() < Self::WIRE_HEADER_BYTES
            || &input[..4] != b"MRGB"
            || input[4] != 2
            || input[5] != 0
        {
            return Err(GpuError::MalformedCommand);
        }
        let count = u16::from_le_bytes(input[6..8].try_into().unwrap()) as usize;
        let batch_id = read_u64(input, 8);
        if count == 0 || count > N || count > MAX_BATCH_DISPATCHES || batch_id == 0 {
            return Err(GpuError::MalformedCommand);
        }
        let expected = Self::WIRE_HEADER_BYTES
            .checked_add(
                count
                    .checked_mul(Dispatch::WIRE_LENGTH)
                    .ok_or(GpuError::RangeOverflow)?,
            )
            .ok_or(GpuError::RangeOverflow)?;
        if input.len() != expected {
            return Err(GpuError::MalformedCommand);
        }
        let mut batch = Self::new()?;
        for index in 0..count {
            let start = Self::WIRE_HEADER_BYTES + index * Dispatch::WIRE_LENGTH;
            let (request_id, dispatch) =
                Dispatch::decode(&input[start..start + Dispatch::WIRE_LENGTH])?;
            batch.push(request_id, dispatch, session)?;
        }
        Ok((batch_id, batch))
    }
}

impl<const N: usize> DispatchTable<N> {
    /// Mints watchdog identities for an entire validated batch. Admission is
    /// transactional: any failure invalidates every identity minted during
    /// this call before returning an error.
    pub fn begin_batch<const M: usize>(
        &mut self,
        batch: &GpuDispatchBatch<M>,
        now: u64,
        deadline: u64,
    ) -> Result<PreparedGpuBatch<M>, GpuError> {
        batch.validate_ready()?;
        let mut prepared = PreparedGpuBatch::new();
        for entry in batch.entries() {
            let dispatch_id = match self.begin(entry.request_id, now, deadline) {
                Ok(id) => id,
                Err(error) => {
                    for admitted in prepared.entries() {
                        let _ = self.cancel(admitted.dispatch_id);
                    }
                    return Err(error);
                }
            };
            prepared.entries[prepared.count] = Some(PreparedGpuDispatch {
                request_id: entry.request_id,
                dispatch_id,
                dispatch: entry.dispatch,
            });
            prepared.count += 1;
        }
        Ok(prepared)
    }

    /// Cancels all still-live identities when a service rejects a prepared
    /// graph before accepting ownership. Already completed IDs are ignored.
    pub fn cancel_batch<const M: usize>(&mut self, batch: &PreparedGpuBatch<M>) {
        for entry in batch.entries() {
            if self.contains(entry.dispatch_id) {
                let _ = self.cancel(entry.dispatch_id);
            }
        }
    }
}

pub fn submit_gpu_batch<E, const WATCHDOG: usize, const BATCH: usize>(
    executor: &mut E,
    watchdog: &mut DispatchTable<WATCHDOG>,
    batch: &ValidatedGpuBatch<BATCH>,
) -> Result<(), GpuSubmitError<E::Error>>
where
    E: GpuBatchExecutor<BATCH>,
{
    match executor.submit(batch) {
        Ok(true) => Ok(()),
        Ok(false) => {
            watchdog.cancel_batch(batch.prepared());
            Err(GpuSubmitError::Rejected)
        }
        Err(error) => Err(GpuSubmitError::Uncertain(error)),
    }
}

impl Dispatch {
    const SCALAR_WIRE_OFFSET: usize = 48 + MAX_DISPATCH_BUFFERS * 32;
    pub const WIRE_LENGTH: usize = Self::SCALAR_WIRE_OFFSET + MAX_DISPATCH_SCALARS * 8;

    pub fn new(
        kernel: KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory: u32,
        accesses: &[BufferAccess],
    ) -> Result<Self, GpuError> {
        Self::new_with_scalars(kernel, grid, block, shared_memory, accesses, &[])
    }

    pub fn new_with_scalars(
        kernel: KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory: u32,
        accesses: &[BufferAccess],
        scalars: &[ScalarArg],
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
        if scalars.len() > MAX_DISPATCH_SCALARS {
            return Err(GpuError::TooManyScalars);
        }
        let mut stored = [None; MAX_DISPATCH_BUFFERS];
        for (slot, access) in stored.iter_mut().zip(accesses) {
            *slot = Some(*access);
        }
        let mut stored_scalars = [None; MAX_DISPATCH_SCALARS];
        for (slot, scalar) in stored_scalars.iter_mut().zip(scalars) {
            *slot = Some(*scalar);
        }
        Ok(Self {
            kernel,
            grid,
            block,
            shared_memory,
            accesses: stored,
            access_count: accesses.len() as u8,
            scalars: stored_scalars,
            scalar_count: scalars.len() as u8,
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
    pub fn scalars(&self) -> impl Iterator<Item = ScalarArg> + '_ {
        self.scalars[..self.scalar_count as usize]
            .iter()
            .flatten()
            .copied()
    }

    /// Validate the pointer-free dispatch against the exact ABI and launch
    /// geometry supported by the service executor. Schemas are enabled one at
    /// a time; an otherwise valid kernel ID is not sufficient for execution.
    pub fn validate_executor_schema(&self) -> Result<ValidatedKernelLaunch, GpuError> {
        match self.kernel.get() {
            0 => self.validate_gemm_q4_0_f32_schema(),
            3 => self.validate_quantized_gemv_schema(18, 16, 256),
            4 => self.validate_quantized_gemv_schema(34, 8, 128),
            7 | 9 | 10 => self.validate_three_f32_elementwise_schema(),
            8 => self.validate_embedding_f32_schema(),
            12 => self.validate_rms_norm_f32_schema(),
            _ => Err(GpuError::UnsupportedKernelSchema),
        }
    }

    fn validate_quantized_gemv_schema(
        &self,
        quantized_block_bytes: u64,
        rows_per_grid_block: u32,
        threads_per_block: u32,
    ) -> Result<ValidatedKernelLaunch, GpuError> {
        if self.access_count != 3
            || self.scalar_count != 2
            || self.shared_memory != 0
            || self.block != [threads_per_block, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let weights = self.accesses[0].ok_or(GpuError::InvalidKernelSchema)?;
        let input = self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?;
        let output = self.accesses[2].ok_or(GpuError::InvalidKernelSchema)?;
        let rows = self.positive_i32_scalar(0)?;
        let columns = self.positive_i32_scalar(1)?;
        if columns % 32 != 0
            || weights.mode != BufferMode::Read
            || input.mode != BufferMode::Read
            || output.mode != BufferMode::Write
            || weights.offset % 2 != 0
            || input.offset % 4 != 0
            || output.offset % 4 != 0
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let weight_bytes = u64::from(rows)
            .checked_mul(u64::from(columns / 32))
            .and_then(|value| value.checked_mul(quantized_block_bytes))
            .ok_or(GpuError::InvalidKernelSchema)?;
        let input_bytes = u64::from(columns)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        let output_bytes = u64::from(rows)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        if weights.length != weight_bytes
            || input.length != input_bytes
            || output.length != output_bytes
            || self.grid != [rows.div_ceil(rows_per_grid_block), 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(ValidatedKernelLaunch {
            dispatch: *self,
            element_count: rows,
        })
    }

    fn validate_gemm_q4_0_f32_schema(&self) -> Result<ValidatedKernelLaunch, GpuError> {
        if self.access_count != 3
            || self.scalar_count != 3
            || self.shared_memory != 0
            || self.block != [128, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let weights = self.accesses[0].ok_or(GpuError::InvalidKernelSchema)?;
        let input = self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?;
        let output = self.accesses[2].ok_or(GpuError::InvalidKernelSchema)?;
        if weights.mode != BufferMode::Read
            || input.mode != BufferMode::Read
            || output.mode != BufferMode::Write
            || weights.offset % 2 != 0
            || input.offset % 4 != 0
            || output.offset % 4 != 0
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let rows = self.positive_i32_scalar(0)?;
        let columns = self.positive_i32_scalar(1)?;
        let batch = self.positive_i32_scalar(2)?;
        if columns % 32 != 0 {
            return Err(GpuError::InvalidKernelSchema);
        }
        let weight_bytes = u64::from(rows)
            .checked_mul(u64::from(columns / 32))
            .and_then(|value| value.checked_mul(18))
            .ok_or(GpuError::InvalidKernelSchema)?;
        let input_bytes = u64::from(columns)
            .checked_mul(u64::from(batch))
            .and_then(|value| value.checked_mul(4))
            .ok_or(GpuError::InvalidKernelSchema)?;
        let output_elements = rows
            .checked_mul(batch)
            .ok_or(GpuError::InvalidKernelSchema)?;
        let output_bytes = u64::from(output_elements)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        if weights.length != weight_bytes
            || input.length != input_bytes
            || output.length != output_bytes
            || self.grid != [rows.div_ceil(8), batch.div_ceil(8), 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(ValidatedKernelLaunch {
            dispatch: *self,
            element_count: output_elements,
        })
    }

    fn positive_i32_scalar(&self, index: usize) -> Result<u32, GpuError> {
        let scalar = self.scalars[index].ok_or(GpuError::InvalidKernelSchema)?;
        if scalar.kind != ScalarKind::I32 || scalar.bits == 0 || scalar.bits > i32::MAX as u32 {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(scalar.bits)
    }

    fn nonnegative_i32_scalar(&self, index: usize) -> Result<u32, GpuError> {
        let scalar = self.scalars[index].ok_or(GpuError::InvalidKernelSchema)?;
        if scalar.kind != ScalarKind::I32 || scalar.bits > i32::MAX as u32 {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(scalar.bits)
    }

    fn validate_three_f32_elementwise_schema(&self) -> Result<ValidatedKernelLaunch, GpuError> {
        if self.access_count != 3
            || self.scalar_count != 0
            || self.shared_memory != 0
            || self.block != [256, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let a = self.accesses[0].ok_or(GpuError::InvalidKernelSchema)?;
        let b = self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?;
        let output = self.accesses[2].ok_or(GpuError::InvalidKernelSchema)?;
        if a.mode != BufferMode::Read
            || b.mode != BufferMode::Read
            || output.mode != BufferMode::Write
            || a.length == 0
            || a.length != b.length
            || a.length != output.length
            || a.offset % 4 != 0
            || b.offset % 4 != 0
            || output.offset % 4 != 0
            || a.length % 4 != 0
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let elements = a.length / 4;
        let element_count = u32::try_from(elements).map_err(|_| GpuError::InvalidKernelSchema)?;
        let blocks = element_count.div_ceil(256).max(1);
        if self.grid != [blocks, 1, 1] {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(ValidatedKernelLaunch {
            dispatch: *self,
            element_count,
        })
    }

    fn validate_embedding_f32_schema(&self) -> Result<ValidatedKernelLaunch, GpuError> {
        if self.access_count != 2
            || self.scalar_count != 2
            || self.shared_memory != 0
            || self.block != [256, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let table = self.accesses[0].ok_or(GpuError::InvalidKernelSchema)?;
        let output = self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?;
        let token = self.nonnegative_i32_scalar(0)?;
        let dimension = self.positive_i32_scalar(1)?;
        let row_bytes = u64::from(dimension)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        if table.mode != BufferMode::Read
            || output.mode != BufferMode::Write
            || table.offset % 4 != 0
            || output.offset % 4 != 0
            || table.length == 0
            || table.length % row_bytes != 0
            || output.length != row_bytes
            || u64::from(token) >= table.length / row_bytes
            || self.grid != [dimension.div_ceil(256), 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(ValidatedKernelLaunch {
            dispatch: *self,
            element_count: dimension,
        })
    }

    fn validate_rms_norm_f32_schema(&self) -> Result<ValidatedKernelLaunch, GpuError> {
        if !(self.access_count == 2 || self.access_count == 3)
            || self.scalar_count != 3
            || self.shared_memory != 0
            || self.block != [256, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        let input = self.accesses[0].ok_or(GpuError::InvalidKernelSchema)?;
        let (weight, output) = if self.access_count == 3 {
            (
                Some(self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?),
                self.accesses[2].ok_or(GpuError::InvalidKernelSchema)?,
            )
        } else {
            (None, self.accesses[1].ok_or(GpuError::InvalidKernelSchema)?)
        };
        let dimension = self.positive_i32_scalar(0)?;
        let batch = self.positive_i32_scalar(1)?;
        let epsilon_arg = self.scalars[2].ok_or(GpuError::InvalidKernelSchema)?;
        let epsilon = f32::from_bits(epsilon_arg.bits);
        if epsilon_arg.kind != ScalarKind::F32 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(GpuError::InvalidKernelSchema);
        }
        let elements = dimension
            .checked_mul(batch)
            .ok_or(GpuError::InvalidKernelSchema)?;
        let tensor_bytes = u64::from(elements)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        let weight_bytes = u64::from(dimension)
            .checked_mul(4)
            .ok_or(GpuError::InvalidKernelSchema)?;
        if input.mode != BufferMode::Read
            || output.mode != BufferMode::Write
            || input.offset % 4 != 0
            || output.offset % 4 != 0
            || input.length != tensor_bytes
            || output.length != tensor_bytes
            || self.grid != [batch, 1, 1]
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        if let Some(weight) = weight
            && (weight.mode != BufferMode::Read
                || weight.offset % 4 != 0
                || weight.length != weight_bytes)
        {
            return Err(GpuError::InvalidKernelSchema);
        }
        Ok(ValidatedKernelLaunch {
            dispatch: *self,
            element_count: elements,
        })
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
        output[4] = 2;
        output[5] = self.kernel.get();
        output[6] = self.access_count;
        output[7] = self.scalar_count;
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
        for (index, scalar) in self.scalars().enumerate() {
            let offset = Self::SCALAR_WIRE_OFFSET + index * 8;
            output[offset] = match scalar.kind {
                ScalarKind::U32 => 1,
                ScalarKind::I32 => 2,
                ScalarKind::F32 => 3,
            };
            output[offset + 4..offset + 8].copy_from_slice(&scalar.bits.to_le_bytes());
        }
        Ok(())
    }

    pub fn decode(input: &[u8]) -> Result<(u64, Self), GpuError> {
        if input.len() != Self::WIRE_LENGTH
            || &input[..4] != b"MRGD"
            || input[4] != 2
            || input[44..48].iter().any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let request_id = u64::from_le_bytes(input[8..16].try_into().unwrap());
        let access_count = input[6] as usize;
        let scalar_count = input[7] as usize;
        if request_id == 0
            || access_count > MAX_DISPATCH_BUFFERS
            || scalar_count > MAX_DISPATCH_SCALARS
        {
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
        if input[48 + access_count * 32..Self::SCALAR_WIRE_OFFSET]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(GpuError::MalformedCommand);
        }
        let mut scalars = [None; MAX_DISPATCH_SCALARS];
        for (index, slot) in scalars[..scalar_count].iter_mut().enumerate() {
            let offset = Self::SCALAR_WIRE_OFFSET + index * 8;
            if input[offset + 1..offset + 4].iter().any(|byte| *byte != 0) {
                return Err(GpuError::MalformedCommand);
            }
            let kind = match input[offset] {
                1 => ScalarKind::U32,
                2 => ScalarKind::I32,
                3 => ScalarKind::F32,
                _ => return Err(GpuError::MalformedCommand),
            };
            *slot = Some(ScalarArg {
                kind,
                bits: read_u32(input, offset + 4),
            });
        }
        if input[Self::SCALAR_WIRE_OFFSET + scalar_count * 8..]
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
        let scalar_values = core::array::from_fn::<_, MAX_DISPATCH_SCALARS, _>(|index| {
            scalars[index].unwrap_or(ScalarArg::u32(0))
        });
        let dispatch = Self::new_with_scalars(
            kernel,
            grid,
            block,
            shared_memory,
            &access_values[..access_count],
            &scalar_values[..scalar_count],
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
    use crate::{TrustRoot, artifact_statement};
    use mrml_crypto::{
        LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES,
        lamport_public_key, lamport_sign,
    };

    struct MockExecutor(Result<bool, u8>);

    impl<const N: usize> GpuBatchExecutor<N> for MockExecutor {
        type Error = u8;

        fn submit(&mut self, _: &ValidatedGpuBatch<N>) -> Result<bool, Self::Error> {
            self.0
        }
    }

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
        assert_eq!(
            Dispatch::new_with_scalars(
                kernel,
                [1, 1, 1],
                [32, 1, 1],
                0,
                &[],
                &[ScalarArg::u32(0); MAX_DISPATCH_SCALARS + 1],
            ),
            Err(GpuError::TooManyScalars)
        );
    }

    #[test]
    fn executor_accepts_only_the_exact_add_f32_abi() {
        let id = |slot| BufferId {
            slot,
            generation: 1,
        };
        let accesses = [
            BufferAccess::new(id(0), 0, 4096, BufferMode::Read),
            BufferAccess::new(id(1), 0, 4096, BufferMode::Read),
            BufferAccess::new(id(2), 0, 4096, BufferMode::Write),
        ];
        let dispatch = Dispatch::new(
            KernelId::new(7).unwrap(),
            [4, 1, 1],
            [256, 1, 1],
            0,
            &accesses,
        )
        .unwrap();
        let launch = dispatch.validate_executor_schema().unwrap();
        assert_eq!(launch.element_count(), 1024);
        assert_eq!(launch.dispatch(), &dispatch);
        for kernel in [9, 10] {
            let activation = Dispatch::new(
                KernelId::new(kernel).unwrap(),
                [4, 1, 1],
                [256, 1, 1],
                0,
                &accesses,
            )
            .unwrap();
            assert_eq!(
                activation
                    .validate_executor_schema()
                    .unwrap()
                    .element_count(),
                1024
            );
        }

        let wrong_mode = [
            accesses[0],
            accesses[1],
            BufferAccess::new(id(2), 0, 4096, BufferMode::ReadWrite),
        ];
        let wrong_mode = Dispatch::new(
            KernelId::new(7).unwrap(),
            [4, 1, 1],
            [256, 1, 1],
            0,
            &wrong_mode,
        )
        .unwrap();
        assert_eq!(
            wrong_mode.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );

        let wrong_grid = Dispatch::new(
            KernelId::new(7).unwrap(),
            [5, 1, 1],
            [256, 1, 1],
            0,
            &accesses,
        )
        .unwrap();
        assert_eq!(
            wrong_grid.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );
        let unsupported = Dispatch::new(
            KernelId::new(1).unwrap(),
            [1, 1, 1],
            [128, 1, 1],
            0,
            &accesses,
        )
        .unwrap();
        assert_eq!(
            unsupported.validate_executor_schema(),
            Err(GpuError::UnsupportedKernelSchema)
        );
        let prepared = PreparedGpuBatch {
            entries: [Some(PreparedGpuDispatch {
                request_id: 9,
                dispatch_id: DispatchId {
                    slot: 0,
                    generation: 1,
                },
                dispatch: unsupported,
            })],
            count: 1,
        };
        let bundle = VerifiedGpuKernelBundle {
            version: 1,
            digest: [3; 64],
        };
        assert_eq!(
            bundle.validate_batch(&prepared).map(|_| ()),
            Err(GpuError::UnsupportedKernelSchema)
        );
    }

    #[test]
    fn executor_binds_q4_gemm_dimensions_to_ranges_and_geometry() {
        let id = |slot| BufferId {
            slot,
            generation: 1,
        };
        let accesses = [
            BufferAccess::new(id(0), 0, 288, BufferMode::Read),
            BufferAccess::new(id(1), 0, 1024, BufferMode::Read),
            BufferAccess::new(id(2), 0, 512, BufferMode::Write),
        ];
        let scalars = [ScalarArg::i32(16), ScalarArg::i32(32), ScalarArg::i32(8)];
        let dispatch = Dispatch::new_with_scalars(
            KernelId::new(0).unwrap(),
            [2, 1, 1],
            [128, 1, 1],
            0,
            &accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            dispatch.validate_executor_schema().unwrap().element_count(),
            128
        );

        let wrong_columns = [ScalarArg::i32(16), ScalarArg::i32(31), ScalarArg::i32(8)];
        let wrong_columns = Dispatch::new_with_scalars(
            KernelId::new(0).unwrap(),
            [2, 1, 1],
            [128, 1, 1],
            0,
            &accesses,
            &wrong_columns,
        )
        .unwrap();
        assert_eq!(
            wrong_columns.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );

        let short_output = [
            accesses[0],
            accesses[1],
            BufferAccess::new(id(2), 0, 508, BufferMode::Write),
        ];
        let short_output = Dispatch::new_with_scalars(
            KernelId::new(0).unwrap(),
            [2, 1, 1],
            [128, 1, 1],
            0,
            &short_output,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            short_output.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );

        let wrong_type = [ScalarArg::u32(16), ScalarArg::i32(32), ScalarArg::i32(8)];
        let wrong_type = Dispatch::new_with_scalars(
            KernelId::new(0).unwrap(),
            [2, 1, 1],
            [128, 1, 1],
            0,
            &accesses,
            &wrong_type,
        )
        .unwrap();
        assert_eq!(
            wrong_type.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );
    }

    #[test]
    fn executor_distinguishes_q4_and_q8_gemv_storage_and_tiling() {
        let id = |slot| BufferId {
            slot,
            generation: 1,
        };
        let scalars = [ScalarArg::i32(32), ScalarArg::i32(64)];
        let q4_accesses = [
            BufferAccess::new(id(0), 0, 1152, BufferMode::Read),
            BufferAccess::new(id(1), 0, 256, BufferMode::Read),
            BufferAccess::new(id(2), 0, 128, BufferMode::Write),
        ];
        let q4 = Dispatch::new_with_scalars(
            KernelId::new(3).unwrap(),
            [2, 1, 1],
            [256, 1, 1],
            0,
            &q4_accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(q4.validate_executor_schema().unwrap().element_count(), 32);

        let q8_accesses = [
            BufferAccess::new(id(0), 0, 2176, BufferMode::Read),
            q4_accesses[1],
            q4_accesses[2],
        ];
        let q8 = Dispatch::new_with_scalars(
            KernelId::new(4).unwrap(),
            [4, 1, 1],
            [128, 1, 1],
            0,
            &q8_accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(q8.validate_executor_schema().unwrap().element_count(), 32);

        let q8_with_q4_storage = Dispatch::new_with_scalars(
            KernelId::new(4).unwrap(),
            [4, 1, 1],
            [128, 1, 1],
            0,
            &q4_accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            q8_with_q4_storage.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );
    }

    #[test]
    fn executor_bounds_f32_embedding_token_and_row() {
        let id = |slot| BufferId {
            slot,
            generation: 1,
        };
        let accesses = [
            BufferAccess::new(id(0), 0, 5120, BufferMode::Read),
            BufferAccess::new(id(1), 0, 512, BufferMode::Write),
        ];
        let scalars = [ScalarArg::i32(9), ScalarArg::i32(128)];
        let dispatch = Dispatch::new_with_scalars(
            KernelId::new(8).unwrap(),
            [1, 1, 1],
            [256, 1, 1],
            0,
            &accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            dispatch.validate_executor_schema().unwrap().element_count(),
            128
        );

        let past_end = [ScalarArg::i32(10), ScalarArg::i32(128)];
        let past_end = Dispatch::new_with_scalars(
            KernelId::new(8).unwrap(),
            [1, 1, 1],
            [256, 1, 1],
            0,
            &accesses,
            &past_end,
        )
        .unwrap();
        assert_eq!(
            past_end.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );

        let wrong_output = [
            accesses[0],
            BufferAccess::new(id(1), 0, 508, BufferMode::Write),
        ];
        let wrong_output = Dispatch::new_with_scalars(
            KernelId::new(8).unwrap(),
            [1, 1, 1],
            [256, 1, 1],
            0,
            &wrong_output,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            wrong_output.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );
    }

    #[test]
    fn executor_validates_weighted_and_unweighted_rms_norm() {
        let id = |slot| BufferId {
            slot,
            generation: 1,
        };
        let scalars = [
            ScalarArg::i32(128),
            ScalarArg::i32(2),
            ScalarArg::f32_bits(1.0e-6f32.to_bits()),
        ];
        let unweighted_accesses = [
            BufferAccess::new(id(0), 0, 1024, BufferMode::Read),
            BufferAccess::new(id(2), 0, 1024, BufferMode::Write),
        ];
        let unweighted = Dispatch::new_with_scalars(
            KernelId::new(12).unwrap(),
            [2, 1, 1],
            [256, 1, 1],
            0,
            &unweighted_accesses,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            unweighted
                .validate_executor_schema()
                .unwrap()
                .element_count(),
            256
        );

        let weighted_accesses = [
            unweighted_accesses[0],
            BufferAccess::new(id(1), 0, 512, BufferMode::Read),
            unweighted_accesses[1],
        ];
        let weighted = Dispatch::new_with_scalars(
            KernelId::new(12).unwrap(),
            [2, 1, 1],
            [256, 1, 1],
            0,
            &weighted_accesses,
            &scalars,
        )
        .unwrap();
        assert!(weighted.validate_executor_schema().is_ok());

        let invalid_epsilon = [
            scalars[0],
            scalars[1],
            ScalarArg::f32_bits(f32::NAN.to_bits()),
        ];
        let invalid_epsilon = Dispatch::new_with_scalars(
            KernelId::new(12).unwrap(),
            [2, 1, 1],
            [256, 1, 1],
            0,
            &weighted_accesses,
            &invalid_epsilon,
        )
        .unwrap();
        assert_eq!(
            invalid_epsilon.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );

        let short_weight = [
            unweighted_accesses[0],
            BufferAccess::new(id(1), 0, 508, BufferMode::Read),
            unweighted_accesses[1],
        ];
        let short_weight = Dispatch::new_with_scalars(
            KernelId::new(12).unwrap(),
            [2, 1, 1],
            [256, 1, 1],
            0,
            &short_weight,
            &scalars,
        )
        .unwrap();
        assert_eq!(
            short_weight.validate_executor_schema(),
            Err(GpuError::InvalidKernelSchema)
        );
    }

    #[test]
    fn kernel_bundle_token_requires_signed_bytes_matching_embedded_ptx() {
        let bundle = b"original embedded MRML PTX bundle";
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(29).wrapping_add(3) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        let root = TrustRoot::new(ArtifactKind::CudaKernelBundle, Sha3_512::digest(&public), 5);
        let statement = artifact_statement(
            ArtifactKind::CudaKernelBundle,
            5,
            bundle.len() as u64,
            Sha3_512::digest(bundle),
        );
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        lamport_sign(&private, &statement, &mut signature).unwrap();
        let artifact = root.verify(5, bundle, &public, &signature).unwrap();
        let admitted = VerifiedGpuKernelBundle::admit(&artifact, bundle).unwrap();
        assert_eq!(admitted.version(), 5);
        assert!(admitted.permits(KernelId::new(27).unwrap()));
        assert!(matches!(
            VerifiedGpuKernelBundle::admit(&artifact, b"changed PTX"),
            Err(GpuError::UntrustedKernelBundle)
        ));

        let wrong_root = TrustRoot::new(ArtifactKind::VmImage, Sha3_512::digest(&public), 5);
        let wrong_statement = artifact_statement(
            ArtifactKind::VmImage,
            5,
            bundle.len() as u64,
            Sha3_512::digest(bundle),
        );
        lamport_sign(&private, &wrong_statement, &mut signature).unwrap();
        let wrong_artifact = wrong_root.verify(5, bundle, &public, &signature).unwrap();
        assert!(matches!(
            VerifiedGpuKernelBundle::admit(&wrong_artifact, bundle),
            Err(GpuError::UntrustedKernelBundle)
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

        let mut controls = ControlBufferTable::<1>::new();
        let control = controls.seal(b"canonical batch descriptor").unwrap();
        let submit = ResourceCommand::SubmitBatch { control };
        submit.encode(8, &mut wire).unwrap();
        assert_eq!(ResourceCommand::decode(&wire), Ok((8, submit)));
    }

    #[test]
    fn control_buffers_are_typed_sealed_bounded_and_generational() {
        let mut controls = ControlBufferTable::<1>::new();
        assert_eq!(controls.seal(&[]), Err(GpuError::ControlBufferTooLarge));
        let bytes = b"bounded canonical dispatch batch";
        let id = controls.seal(bytes).unwrap();
        assert!(controls.verify(id, bytes).is_ok());
        assert_eq!(
            controls.verify(id, b"bounded canonical dispatch batcH"),
            Err(GpuError::ControlBufferChanged)
        );
        assert_eq!(
            controls.seal(b"second"),
            Err(GpuError::ControlBufferTableFull)
        );
        controls.release(id).unwrap();
        assert_eq!(
            controls.verify(id, bytes),
            Err(GpuError::InvalidControlBuffer)
        );
        let replacement = controls.seal(bytes).unwrap();
        assert_ne!(replacement, id);
        assert_eq!(controls.release(id), Err(GpuError::InvalidControlBuffer));
    }

    #[test]
    fn command_ring_is_bounded_fifo_and_wraps_without_overwrite() {
        assert!(matches!(
            GpuCommandRing::<0>::new(),
            Err(GpuError::InvalidQueueCapacity)
        ));
        assert!(matches!(
            GpuCommandRing::<257>::new(),
            Err(GpuError::InvalidQueueCapacity)
        ));
        let mut ring = GpuCommandRing::<2>::new().unwrap();
        let first = [1u8; GPU_QUEUE_MESSAGE_BYTES];
        let second = [2u8; GPU_QUEUE_MESSAGE_BYTES];
        let third = [3u8; GPU_QUEUE_MESSAGE_BYTES];
        let mut output = [0u8; GPU_QUEUE_MESSAGE_BYTES];

        assert_eq!(ring.dequeue(&mut output), Err(GpuError::QueueEmpty));
        assert!(ring.enqueue(&first).is_ok());
        assert!(ring.enqueue(&second).is_ok());
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.enqueue(&third), Err(GpuError::QueueFull));
        ring.dequeue(&mut output).unwrap();
        assert_eq!(output, first);
        assert!(ring.enqueue(&third).is_ok());
        ring.dequeue(&mut output).unwrap();
        assert_eq!(output, second);
        ring.dequeue(&mut output).unwrap();
        assert_eq!(output, third);
        assert!(ring.is_empty());
        assert_eq!(ring.enqueue(&first[..79]), Err(GpuError::MalformedCommand));
    }

    #[test]
    fn shared_ring_ownership_rejects_lapping_forgery_and_reuse() {
        let mut producer = GpuRingProducer::<2>::new().unwrap();
        let mut consumer = GpuRingConsumer::<2>::new().unwrap();
        assert_eq!(consumer.acquire(0), Err(GpuError::QueueEmpty));

        let first = producer.reserve(consumer.consumed()).unwrap();
        assert_eq!(
            first,
            GpuRingTicket {
                position: 1,
                slot: 0
            }
        );
        assert_eq!(
            producer.reserve(consumer.consumed()),
            Err(GpuError::ReservationPending)
        );
        assert_eq!(producer.commit(first), Ok(1));
        let second = producer.reserve(consumer.consumed()).unwrap();
        assert_eq!(
            second,
            GpuRingTicket {
                position: 2,
                slot: 1
            }
        );
        producer.commit(second).unwrap();
        assert_eq!(
            producer.reserve(consumer.consumed()),
            Err(GpuError::QueueFull)
        );

        assert_eq!(consumer.acquire(producer.published()), Ok(first));
        assert_eq!(consumer.release(second), Err(GpuError::InvalidReservation));
        consumer.release(first).unwrap();
        let third = producer.reserve(consumer.consumed()).unwrap();
        assert_eq!(
            third,
            GpuRingTicket {
                position: 3,
                slot: 0
            }
        );
        producer.commit(third).unwrap();
        assert_eq!(consumer.acquire(producer.published()), Ok(second));
        consumer.release(second).unwrap();
        assert_eq!(consumer.acquire(producer.published()), Ok(third));
        consumer.release(third).unwrap();
        assert_eq!(consumer.acquire(2), Err(GpuError::InvalidQueueState));
        assert_eq!(consumer.acquire(6), Err(GpuError::InvalidQueueState));
        assert_eq!(producer.reserve(4), Err(GpuError::InvalidQueueState));
    }

    #[test]
    fn shared_indices_publish_and_consume_only_in_exact_order() {
        assert_eq!(core::mem::align_of::<GpuSharedRingIndices>(), 64);
        assert_eq!(core::mem::size_of::<GpuSharedRingIndices>(), 128);
        let indices = GpuSharedRingIndices::new();
        let first = GpuRingTicket {
            position: 1,
            slot: 0,
        };
        let second = GpuRingTicket {
            position: 2,
            slot: 1,
        };
        assert_eq!(
            indices.publish(0, second),
            Err(GpuError::InvalidReservation)
        );
        assert!(indices.publish(0, first).is_ok());
        assert_eq!(indices.published(), 1);
        assert_eq!(indices.publish(0, first), Err(GpuError::InvalidQueueState));
        assert_eq!(
            indices.consume(0, second),
            Err(GpuError::InvalidReservation)
        );
        assert!(indices.consume(0, first).is_ok());
        assert_eq!(indices.consumed(), 1);
        assert_eq!(indices.consume(0, first), Err(GpuError::InvalidQueueState));
    }

    #[test]
    fn shared_queue_layout_is_page_bounded_disjoint_and_overflow_safe() {
        let layout = GpuSharedQueueLayout::new(0x1000, 0x3000, 64).unwrap();
        assert_eq!(layout.command_base(), 0x1000);
        assert_eq!(layout.completion_base(), 0x3000);
        assert_eq!(layout.pages_per_ring(), 2);
        assert_eq!(layout.slots(), 64);
        assert_eq!(
            GpuSharedQueueLayout::new(0x1000, 0x2000, 64),
            Err(GpuError::InvalidQueueMapping)
        );
        assert_eq!(
            GpuSharedQueueLayout::new(0x1001, 0x4000, 1),
            Err(GpuError::InvalidQueueMapping)
        );
        assert_eq!(
            GpuSharedQueueLayout::new(0x1000, 0x4000, 0),
            Err(GpuError::InvalidQueueMapping)
        );
        assert_eq!(
            GpuSharedQueueLayout::new(u64::MAX - (PAGE_SIZE - 1), 0x1000, 1),
            Err(GpuError::RangeOverflow)
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
    fn completion_queue_binds_dispatch_and_rejects_replay_and_cross_direction() {
        let key = [9; 32];
        let mut dispatches = DispatchTable::<2>::new();
        let dispatch = dispatches.begin(44, 100, 200).unwrap();
        let completion = GpuCompletion::new(44, dispatch, GpuCompletionStatus::Success).unwrap();
        let mut sender = GpuCompletionSender::new(17, key).unwrap();
        let mut receiver = GpuCompletionReceiver::new(17, key).unwrap();
        let mut wire = [0u8; GPU_QUEUE_MESSAGE_BYTES];
        sender.encode(completion, &mut wire).unwrap();

        let mut tampered = wire;
        tampered[24] ^= 1;
        assert_eq!(
            receiver.decode(&tampered),
            Err(GpuError::AuthenticationFailed)
        );
        let decoded = receiver.decode(&wire).unwrap();
        assert_eq!(decoded, completion);
        assert_eq!(dispatches.complete(decoded.dispatch()), Ok(44));
        assert_eq!(receiver.decode(&wire), Err(GpuError::Replay));

        let mut command_receiver = GpuQueueReceiver::new(17, key).unwrap();
        assert_eq!(
            command_receiver.decode(&wire),
            Err(GpuError::MalformedCommand)
        );
        let mut command_sender = GpuQueueSender::new(17, key).unwrap();
        command_sender
            .encode(1, ResourceCommand::Allocate { bytes: 1 }, &mut wire)
            .unwrap();
        let mut completion_receiver = GpuCompletionReceiver::new(17, key).unwrap();
        assert_eq!(
            completion_receiver.decode(&wire),
            Err(GpuError::MalformedCommand)
        );

        let invalid = GpuCompletion::new(
            44,
            DispatchId {
                slot: 0,
                generation: 0,
            },
            GpuCompletionStatus::Success,
        );
        assert_eq!(invalid, Err(GpuError::InvalidDispatch));
    }

    #[test]
    fn dispatch_batches_are_bounded_ordered_and_session_validated() {
        assert!(matches!(
            GpuDispatchBatch::<0>::new(),
            Err(GpuError::InvalidBatchCapacity)
        ));
        assert!(matches!(
            GpuDispatchBatch::<33>::new(),
            Err(GpuError::InvalidBatchCapacity)
        ));
        let mut session = VirtualGpuSession::<2>::new(4096);
        let buffer = session.allocate(4096).unwrap();
        let dispatch = Dispatch::new(
            KernelId::new(7).unwrap(),
            [4, 1, 1],
            [256, 1, 1],
            0,
            &[
                BufferAccess::new(buffer, 0, 4096, BufferMode::Read),
                BufferAccess::new(buffer, 0, 4096, BufferMode::Read),
                BufferAccess::new(buffer, 0, 4096, BufferMode::Write),
            ],
        )
        .unwrap();
        let mut batch = GpuDispatchBatch::<2>::new().unwrap();
        assert_eq!(batch.validate_ready(), Err(GpuError::EmptyBatch));
        assert!(batch.push(10, dispatch, &session).is_ok());
        assert_eq!(
            batch.push(10, dispatch, &session),
            Err(GpuError::DuplicateRequest)
        );
        assert!(batch.push(11, dispatch, &session).is_ok());
        assert_eq!(batch.push(12, dispatch, &session), Err(GpuError::BatchFull));
        assert!(batch.validate_ready().is_ok());
        let mut entries = batch.entries();
        assert_eq!(entries.next().map(BatchedDispatch::request_id), Some(10));
        assert_eq!(entries.next().map(BatchedDispatch::request_id), Some(11));
        assert_eq!(entries.next(), None);

        let mut wire = [0u8; GpuDispatchBatch::<2>::WIRE_HEADER_BYTES + 2 * Dispatch::WIRE_LENGTH];
        let length = batch.encode(77, &mut wire).unwrap();
        assert_eq!(length, wire.len());
        let (batch_id, decoded) = GpuDispatchBatch::<2>::decode(&wire, &session).unwrap();
        assert_eq!(batch_id, 77);
        assert_eq!(decoded.len(), 2);
        let mut decoded_entries = decoded.entries();
        assert_eq!(decoded_entries.next().unwrap().request_id(), 10);
        assert_eq!(decoded_entries.next().unwrap().request_id(), 11);
        assert_eq!(decoded_entries.next(), None);
        assert_eq!(
            GpuDispatchBatch::<2>::decode(&wire[..wire.len() - 1], &session).map(|_| ()),
            Err(GpuError::MalformedCommand)
        );
        let mut noncanonical = wire;
        noncanonical[5] = 1;
        assert_eq!(
            GpuDispatchBatch::<2>::decode(&noncanonical, &session).map(|_| ()),
            Err(GpuError::MalformedCommand)
        );

        let mut controls = ControlBufferTable::<1>::new();
        let control = controls.seal(&wire).unwrap();
        assert!(controls.verify(control, &wire).is_ok());

        let mut too_small = DispatchTable::<1>::new();
        assert!(matches!(
            too_small.begin_batch(&batch, 100, 200),
            Err(GpuError::DispatchTableFull)
        ));
        let recovered = too_small.begin(99, 100, 200).unwrap();
        assert!(too_small.contains(recovered));

        let mut watchdog = DispatchTable::<2>::new();
        let prepared = watchdog.begin_batch(&batch, 100, 200).unwrap();
        assert_eq!(prepared.len(), 2);
        for entry in prepared.entries() {
            assert!(watchdog.contains(entry.dispatch_id()));
        }
        watchdog.cancel_batch(&prepared);
        for entry in prepared.entries() {
            assert!(!watchdog.contains(entry.dispatch_id()));
        }

        let mut rejected_watchdog = DispatchTable::<2>::new();
        let rejected_batch = rejected_watchdog.begin_batch(&batch, 100, 200).unwrap();
        let bundle = VerifiedGpuKernelBundle {
            version: 1,
            digest: [7; 64],
        };
        let validated_rejected = bundle.validate_batch(&rejected_batch).unwrap();
        assert_eq!(validated_rejected.len(), 2);
        assert_eq!(validated_rejected.entries().count(), 2);
        assert_eq!(
            submit_gpu_batch(
                &mut MockExecutor(Ok(false)),
                &mut rejected_watchdog,
                &validated_rejected
            ),
            Err(GpuSubmitError::Rejected)
        );
        for entry in rejected_batch.entries() {
            assert!(!rejected_watchdog.contains(entry.dispatch_id()));
        }

        let mut uncertain_watchdog = DispatchTable::<2>::new();
        let uncertain_batch = uncertain_watchdog.begin_batch(&batch, 100, 200).unwrap();
        let validated_uncertain = bundle.validate_batch(&uncertain_batch).unwrap();
        assert_eq!(
            submit_gpu_batch(
                &mut MockExecutor(Err(7)),
                &mut uncertain_watchdog,
                &validated_uncertain
            ),
            Err(GpuSubmitError::Uncertain(7))
        );
        for entry in uncertain_batch.entries() {
            assert!(uncertain_watchdog.contains(entry.dispatch_id()));
        }

        let mut accepted_watchdog = DispatchTable::<2>::new();
        let accepted_batch = accepted_watchdog.begin_batch(&batch, 100, 200).unwrap();
        let validated_accepted = bundle.validate_batch(&accepted_batch).unwrap();
        assert!(
            submit_gpu_batch(
                &mut MockExecutor(Ok(true)),
                &mut accepted_watchdog,
                &validated_accepted
            )
            .is_ok()
        );
        for entry in accepted_batch.entries() {
            assert!(accepted_watchdog.contains(entry.dispatch_id()));
        }

        session.free(buffer).unwrap();
        let mut stale_batch = GpuDispatchBatch::<1>::new().unwrap();
        assert_eq!(
            stale_batch.push(12, dispatch, &session),
            Err(GpuError::InvalidBuffer)
        );
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
        let scalars = [
            ScalarArg::u32(17),
            ScalarArg::i32(-9),
            ScalarArg::f32_bits(1.5f32.to_bits()),
        ];
        let dispatch = Dispatch::new_with_scalars(
            KernelId::new(7).unwrap(),
            [2, 3, 4],
            [32, 2, 1],
            1024,
            &accesses,
            &scalars,
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
        assert!(decoded.scalars().eq(scalars));

        let mut old_version = wire;
        old_version[4] = 1;
        assert_eq!(
            Dispatch::decode(&old_version),
            Err(GpuError::MalformedCommand)
        );

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
        let mut wire = [0u8; Dispatch::WIRE_LENGTH];
        dispatch.encode(19, &mut wire).unwrap();
        wire[Dispatch::SCALAR_WIRE_OFFSET + 1] = 1;
        assert_eq!(Dispatch::decode(&wire), Err(GpuError::MalformedCommand));
        let mut wire = [0u8; Dispatch::WIRE_LENGTH];
        dispatch.encode(19, &mut wire).unwrap();
        wire[Dispatch::SCALAR_WIRE_OFFSET + scalars.len() * 8] = 1;
        assert_eq!(Dispatch::decode(&wire), Err(GpuError::MalformedCommand));
    }

    #[test]
    fn dispatch_watchdog_rejects_duplicates_and_stale_completions() {
        let mut table = DispatchTable::<2>::new();
        let first = table.begin(41, 100, 120).unwrap();
        assert_eq!(table.begin(41, 100, 130), Err(GpuError::DuplicateRequest));
        assert_eq!(table.begin(42, 100, 100), Err(GpuError::InvalidDeadline));
        let second = table.begin(42, 100, 140).unwrap();
        assert_eq!(table.begin(43, 100, 150), Err(GpuError::DispatchTableFull));

        let mut expired_id = None;
        assert_eq!(
            table.expire(120, |id, request| {
                expired_id = Some((id, request));
            }),
            1
        );
        assert_eq!(expired_id, Some((first, 41)));
        assert!(!table.contains(first));
        assert_eq!(table.complete(first), Err(GpuError::InvalidDispatch));

        let replacement = table.begin(43, 120, 150).unwrap();
        assert_ne!(replacement, first);
        assert_eq!(table.cancel(second), Ok(42));
        assert_eq!(table.complete(second), Err(GpuError::InvalidDispatch));
        assert_eq!(table.complete(replacement), Ok(43));
    }
}
