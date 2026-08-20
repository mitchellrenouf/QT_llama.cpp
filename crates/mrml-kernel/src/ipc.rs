use crate::{Capability, CapabilityError, CapabilitySpace, ObjectId, Rights};

pub const MAX_INLINE_PAYLOAD: usize = 4096;
pub const MAX_CAPABILITIES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IpcError {
    PayloadTooLarge,
    TooManyCapabilities,
    Unauthorized,
    SequenceExhausted,
}

impl From<CapabilityError> for IpcError {
    fn from(_: CapabilityError) -> Self {
        Self::Unauthorized
    }
}

/// A bounded control-plane message. Bulk data uses separately mapped shared
/// memory; keeping it out of IPC prevents attacker-selected kernel allocation.
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
}

/// Kernel-owned endpoint state. The caller must present SIGNAL authority for
/// this endpoint on every send; knowing an endpoint number conveys no rights.
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
}
