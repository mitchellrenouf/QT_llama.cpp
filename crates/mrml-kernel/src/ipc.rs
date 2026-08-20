use crate::{Capability, CapabilityError, CapabilitySpace, ObjectId, Rights};

pub const MAX_INLINE_PAYLOAD: usize = 4096;
pub const MAX_CAPABILITIES: usize = 8;
pub const WIRE_HEADER_LENGTH: usize = 24;
pub const MAX_WIRE_MESSAGE: usize = WIRE_HEADER_LENGTH + MAX_CAPABILITIES * 8 + MAX_INLINE_PAYLOAD;
const WIRE_MAGIC: [u8; 4] = *b"MRIP";
const WIRE_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IpcError {
    PayloadTooLarge,
    TooManyCapabilities,
    Unauthorized,
    SequenceExhausted,
    BufferTooSmall,
    Malformed,
    Replay,
}

impl From<CapabilityError> for IpcError {
    fn from(_: CapabilityError) -> Self {
        Self::Unauthorized
    }
}

pub struct Message {
    payload: [u8; MAX_INLINE_PAYLOAD],
    payload_length: u16,
    capabilities: [Option<Capability>; MAX_CAPABILITIES],
    capability_count: u8,
}

impl Message {
    pub fn new(payload: &[u8], capabilities: &[Capability]) -> Result<Self, IpcError> {
        if payload.len() > MAX_INLINE_PAYLOAD {
            return Err(IpcError::PayloadTooLarge);
        }
        if capabilities.len() > MAX_CAPABILITIES {
            return Err(IpcError::TooManyCapabilities);
        }
        let mut message = Self {
            payload: [0; MAX_INLINE_PAYLOAD],
            payload_length: payload.len() as u16,
            capabilities: [None; MAX_CAPABILITIES],
            capability_count: capabilities.len() as u8,
        };
        message.payload[..payload.len()].copy_from_slice(payload);
        for (slot, capability) in message.capabilities.iter_mut().zip(capabilities) {
            *slot = Some(*capability);
        }
        Ok(message)
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload[..self.payload_length as usize]
    }

    pub fn capabilities(&self) -> impl Iterator<Item = Capability> + '_ {
        self.capabilities[..self.capability_count as usize]
            .iter()
            .flatten()
            .copied()
    }

    /// Encodes without allocation. Tokens only have meaning in a
    /// kernel-created capability space and are never authority by themselves.
    pub fn encode(&self, sequence: u64, output: &mut [u8]) -> Result<usize, IpcError> {
        if sequence == 0 {
            return Err(IpcError::Malformed);
        }
        let capability_bytes = self.capability_count as usize * 8;
        let required = WIRE_HEADER_LENGTH + capability_bytes + self.payload_length as usize;
        if output.len() < required {
            return Err(IpcError::BufferTooSmall);
        }
        output[..4].copy_from_slice(&WIRE_MAGIC);
        output[4] = WIRE_VERSION;
        output[5] = 0;
        output[6] = self.capability_count;
        output[7] = 0;
        output[8..16].copy_from_slice(&sequence.to_le_bytes());
        output[16..18].copy_from_slice(&self.payload_length.to_le_bytes());
        output[18..24].fill(0);
        let mut offset = WIRE_HEADER_LENGTH;
        for capability in self.capabilities() {
            output[offset..offset + 8].copy_from_slice(&capability.token().to_le_bytes());
            offset += 8;
        }
        output[offset..required].copy_from_slice(self.payload());
        Ok(required)
    }

    pub fn decode(input: &[u8]) -> Result<(u64, Self), IpcError> {
        if input.len() < WIRE_HEADER_LENGTH
            || input[..4] != WIRE_MAGIC
            || input[4] != WIRE_VERSION
            || input[5] != 0
            || input[7] != 0
            || input[18..24].iter().any(|byte| *byte != 0)
        {
            return Err(IpcError::Malformed);
        }
        let capability_count = input[6] as usize;
        if capability_count > MAX_CAPABILITIES {
            return Err(IpcError::TooManyCapabilities);
        }
        let sequence = read_u64(&input[8..16]);
        if sequence == 0 {
            return Err(IpcError::Malformed);
        }
        let payload_length = u16::from_le_bytes([input[16], input[17]]) as usize;
        if payload_length > MAX_INLINE_PAYLOAD {
            return Err(IpcError::PayloadTooLarge);
        }
        let required = WIRE_HEADER_LENGTH
            .checked_add(capability_count.checked_mul(8).ok_or(IpcError::Malformed)?)
            .and_then(|length| length.checked_add(payload_length))
            .ok_or(IpcError::Malformed)?;
        if input.len() != required {
            return Err(IpcError::Malformed);
        }
        let mut capabilities = [None; MAX_CAPABILITIES];
        let mut offset = WIRE_HEADER_LENGTH;
        for slot in &mut capabilities[..capability_count] {
            let token = read_u64(&input[offset..offset + 8]);
            if token == 0 || token >> 32 == 0 {
                return Err(IpcError::Malformed);
            }
            *slot = Some(Capability::from_token(token));
            offset += 8;
        }
        let base = Self::new(&input[offset..], &[])?;
        Ok((
            sequence,
            Self {
                capabilities,
                capability_count: capability_count as u8,
                ..base
            },
        ))
    }
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("fixed-size wire field"))
}

pub struct Endpoint {
    object: ObjectId,
    next_sequence: u64,
}

impl Endpoint {
    pub const fn new(object: ObjectId) -> Self {
        Self {
            object,
            next_sequence: 1,
        }
    }

    pub fn authorize_send<const N: usize>(
        &mut self,
        sender_space: &CapabilitySpace<N>,
        endpoint: Capability,
        _message: &Message,
    ) -> Result<u64, IpcError> {
        if sender_space.authorize(endpoint, Rights::SIGNAL)? != self.object {
            return Err(IpcError::Unauthorized);
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(IpcError::SequenceExhausted)?;
        Ok(sequence)
    }
}

pub struct Receiver {
    next_sequence: u64,
}

impl Receiver {
    pub const fn new() -> Self {
        Self { next_sequence: 1 }
    }

    pub fn decode(&mut self, input: &[u8]) -> Result<Message, IpcError> {
        let (sequence, message) = Message::decode(input)?;
        if sequence != self.next_sequence {
            return Err(IpcError::Replay);
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(IpcError::SequenceExhausted)?;
        Ok(message)
    }
}

impl Default for Receiver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_requires_authority_and_sequences_messages() {
        let mut space = CapabilitySpace::<2>::new();
        let allowed = space.insert(ObjectId(9), Rights::SIGNAL).unwrap();
        let wrong = space.insert(ObjectId(10), Rights::SIGNAL).unwrap();
        let message = Message::new(b"run", &[]).unwrap();
        let mut endpoint = Endpoint::new(ObjectId(9));
        assert_eq!(endpoint.authorize_send(&space, allowed, &message), Ok(1));
        assert_eq!(endpoint.authorize_send(&space, allowed, &message), Ok(2));
        assert_eq!(
            endpoint.authorize_send(&space, wrong, &message),
            Err(IpcError::Unauthorized)
        );
    }

    #[test]
    fn message_size_and_capability_count_are_bounded() {
        let payload = [0u8; MAX_INLINE_PAYLOAD + 1];
        assert!(matches!(
            Message::new(&payload, &[]),
            Err(IpcError::PayloadTooLarge)
        ));
        let mut space = CapabilitySpace::<9>::new();
        let capabilities: [Capability; 9] = core::array::from_fn(|index| {
            space.insert(ObjectId(index as u64), Rights::READ).unwrap()
        });
        assert!(matches!(
            Message::new(&[], &capabilities),
            Err(IpcError::TooManyCapabilities)
        ));
    }

    #[test]
    fn wire_encoding_is_exact_bounded_and_replay_checked() {
        let mut space = CapabilitySpace::<1>::new();
        let capability = space.insert(ObjectId(3), Rights::READ).unwrap();
        let message = Message::new(b"abc", &[capability]).unwrap();
        let mut wire = [0u8; MAX_WIRE_MESSAGE];
        let length = message.encode(1, &mut wire).unwrap();
        assert_eq!(length, WIRE_HEADER_LENGTH + 8 + 3);
        let mut receiver = Receiver::new();
        let decoded = receiver.decode(&wire[..length]).unwrap();
        assert_eq!(decoded.payload(), b"abc");
        assert_eq!(decoded.capabilities().next(), Some(capability));
        assert_eq!(
            receiver.decode(&wire[..length]).err(),
            Some(IpcError::Replay)
        );
        assert_eq!(
            Message::decode(&wire[..length - 1]).err(),
            Some(IpcError::Malformed)
        );
        wire[18] = 1;
        assert_eq!(
            Message::decode(&wire[..length]).err(),
            Some(IpcError::Malformed)
        );
    }
}
