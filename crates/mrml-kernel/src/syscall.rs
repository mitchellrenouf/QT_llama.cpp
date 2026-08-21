use crate::arch::x86_64::{USER_CODE_SELECTOR, USER_DATA_SELECTOR};
use crate::{Capability, TaskId};

pub const X86_USER_CALL_VECTOR: u8 = 0x80;
pub const MAX_SYSCALL_INLINE_PAYLOAD: usize = 24;
const USER_RETURN_FLAGS: u64 = 2
    | (1 << 0)
    | (1 << 2)
    | (1 << 4)
    | (1 << 6)
    | (1 << 7)
    | (1 << 8)
    | (1 << 9)
    | (1 << 10)
    | (1 << 11)
    | (1 << 18)
    | (1 << 21);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyscallError {
    UnknownOperation,
    ReservedArgument,
    InvalidCapability,
    InvalidTask,
    PayloadTooLarge,
    InvalidPrivilege,
    InvalidInstruction,
    InvalidStack,
    InvalidFlags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct UserCallFrame {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rax: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

impl UserCallFrame {
    pub fn request(&self) -> Result<SyscallRequest, SyscallError> {
        if self.cs != u64::from(USER_CODE_SELECTOR) || self.ss != u64::from(USER_DATA_SELECTOR) {
            return Err(SyscallError::InvalidPrivilege);
        }
        if !user_address(self.rip) {
            return Err(SyscallError::InvalidInstruction);
        }
        if !user_address(self.rsp) {
            return Err(SyscallError::InvalidStack);
        }
        if self.rflags & 2 == 0 || self.rflags & !USER_RETURN_FLAGS != 0 {
            return Err(SyscallError::InvalidFlags);
        }
        SyscallRequest::decode(
            self.rax, self.rdi, self.rsi, self.rdx, self.r10, self.r8, self.r9,
        )
    }

    pub fn complete(&mut self, sequence: u64) {
        self.rax = 0;
        self.rdx = sequence;
    }
}

const fn user_address(address: u64) -> bool {
    address != 0 && address < 1 << 47
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

    #[test]
    fn hardware_call_frame_is_exact_and_rejects_forged_return_state() {
        assert_eq!(core::mem::size_of::<UserCallFrame>(), 20 * 8);
        assert_eq!(core::mem::offset_of!(UserCallFrame, r15), 0);
        assert_eq!(core::mem::offset_of!(UserCallFrame, rax), 14 * 8);
        assert_eq!(core::mem::offset_of!(UserCallFrame, rip), 15 * 8);
        assert_eq!(core::mem::offset_of!(UserCallFrame, rsp), 18 * 8);
        let mut frame = UserCallFrame {
            r15: 0,
            r14: 0,
            r13: 0,
            r12: 0,
            r11: 0,
            r10: 0,
            r9: 0,
            r8: 0,
            rdi: 0,
            rsi: 0,
            rbp: 0,
            rbx: 0,
            rdx: 0,
            rcx: 0,
            rax: 0,
            rip: 0x40_1000,
            cs: u64::from(USER_CODE_SELECTOR),
            rflags: 0x202,
            rsp: 0x7000_0000,
            ss: u64::from(USER_DATA_SELECTOR),
        };
        assert_eq!(frame.request(), Ok(SyscallRequest::Yield));
        frame.cs = 0x38;
        assert_eq!(frame.request(), Err(SyscallError::InvalidPrivilege));
        frame.cs = u64::from(USER_CODE_SELECTOR);
        frame.rip = 0xffff_8000_0000_0000;
        assert_eq!(frame.request(), Err(SyscallError::InvalidInstruction));
        frame.rip = 0x40_1000;
        frame.rflags = 0;
        assert_eq!(frame.request(), Err(SyscallError::InvalidFlags));
        frame.rflags = 0x202 | (3 << 12);
        assert_eq!(frame.request(), Err(SyscallError::InvalidFlags));
    }
}
