use crate::{Capability, TaskId};

pub const X86_USER_CALL_VECTOR: u8 = 0x80;
pub const MAX_SYSCALL_INLINE_PAYLOAD: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyscallError {
    UnknownOperation,
    ReservedArgument,
    InvalidCapability,
    InvalidTask,
    PayloadTooLarge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyscallRequest {
    Yield,
    SendInline {
        endpoint: Capability,
        receiver: TaskId,
        payload: [u8; MAX_SYSCALL_INLINE_PAYLOAD],
        length: u8,
    },
}

impl SyscallRequest {
    /// Decodes the pointer-free MRML register ABI. No user address is ever
    /// dereferenced: up to 24 payload bytes travel by value in r10/r8/r9.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        operation: u64,
        rdi: u64,
        rsi: u64,
        rdx: u64,
        r10: u64,
        r8: u64,
        r9: u64,
    ) -> Result<Self, SyscallError> {
        match operation {
            0 => {
                if rdi | rsi | rdx | r10 | r8 | r9 != 0 {
                    return Err(SyscallError::ReservedArgument);
                }
                Ok(Self::Yield)
            }
            1 => {
                if rdi == 0 || rdi >> 32 == 0 {
                    return Err(SyscallError::InvalidCapability);
                }
                if rsi == 0 || rsi >> 32 == 0 {
                    return Err(SyscallError::InvalidTask);
                }
                let length = usize::try_from(rdx).map_err(|_| SyscallError::PayloadTooLarge)?;
                if length > MAX_SYSCALL_INLINE_PAYLOAD {
                    return Err(SyscallError::PayloadTooLarge);
                }
                let mut payload = [0u8; MAX_SYSCALL_INLINE_PAYLOAD];
                payload[..8].copy_from_slice(&r10.to_le_bytes());
                payload[8..16].copy_from_slice(&r8.to_le_bytes());
                payload[16..24].copy_from_slice(&r9.to_le_bytes());
                payload[length..].fill(0);
                Ok(Self::SendInline {
                    endpoint: Capability::from_token(rdi),
                    receiver: TaskId::from_token(rsi),
                    payload,
                    length: length as u8,
                })
            }
            _ => Err(SyscallError::UnknownOperation),
        }
    }

    pub fn payload(&self) -> &[u8] {
        match self {
            Self::Yield => &[],
            Self::SendInline {
                payload, length, ..
            } => &payload[..usize::from(*length)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yield_requires_every_reserved_register_to_be_zero() {
        assert_eq!(
            SyscallRequest::decode(0, 0, 0, 0, 0, 0, 0),
            Ok(SyscallRequest::Yield)
        );
        assert_eq!(
            SyscallRequest::decode(0, 0, 0, 0, 1, 0, 0),
            Err(SyscallError::ReservedArgument)
        );
    }

    #[test]
    fn inline_send_is_pointer_free_canonical_and_bounded() {
        let endpoint = (7u64 << 32) | 3;
        let receiver = (9u64 << 32) | 2;
        let request = SyscallRequest::decode(
            1,
            endpoint,
            receiver,
            11,
            u64::from_le_bytes(*b"hello wo"),
            u64::from_le_bytes(*b"rld\0\0\0\0\0"),
            u64::MAX,
        )
        .unwrap();
        assert_eq!(request.payload(), b"hello world");
        assert_eq!(
            SyscallRequest::decode(1, endpoint, receiver, 25, 0, 0, 0),
            Err(SyscallError::PayloadTooLarge)
        );
        assert_eq!(
            SyscallRequest::decode(1, 3, receiver, 0, 0, 0, 0),
            Err(SyscallError::InvalidCapability)
        );
        assert_eq!(
            SyscallRequest::decode(1, endpoint, 2, 0, 0, 0, 0),
            Err(SyscallError::InvalidTask)
        );
    }

    #[test]
    fn unknown_operations_fail_without_interpreting_arguments() {
        assert_eq!(
            SyscallRequest::decode(u64::MAX, 1, 2, 3, 4, 5, 6),
            Err(SyscallError::UnknownOperation)
        );
    }
}
