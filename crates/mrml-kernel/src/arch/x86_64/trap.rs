#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TrapFrame {
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
    pub vector: u64,
    pub error: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrapError {
    InvalidVector,
    InvalidPrivilege,
    NonCanonicalInstruction,
    NonCanonicalStack,
    InvalidFlags,
    InvalidErrorCode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrapDisposition {
    /// Destroy the current user task and revoke its capabilities. Execution may
    /// continue only after the scheduler installs a different context.
    TerminateUser { vector: u8, address: Option<u64> },
    /// Kernel and integrity-critical CPU faults are never resumed.
    HaltKernel { vector: u8 },
}

const ERROR_CODE_VECTORS: u32 = (1 << 8)
    | (1 << 10)
    | (1 << 11)
    | (1 << 12)
    | (1 << 13)
    | (1 << 14)
    | (1 << 17)
    | (1 << 21)
    | (1 << 29)
    | (1 << 30);
const NEVER_RECOVER_VECTORS: u32 = (1 << 2) | (1 << 8) | (1 << 18);

impl TrapFrame {
    pub fn validate(&self) -> Result<(), TrapError> {
        if self.vector >= 32 {
            return Err(TrapError::InvalidVector);
        }
        let privilege = self.cs & 3;
        if privilege != 0 && privilege != 3 {
            return Err(TrapError::InvalidPrivilege);
        }
        if !canonical(self.rip) {
            return Err(TrapError::NonCanonicalInstruction);
        }
        if privilege == 3 && !canonical(self.rsp) {
            return Err(TrapError::NonCanonicalStack);
        }
        // Architectural bit 1 is fixed. NT and VM are forbidden in the
        // long-mode contexts admitted by this kernel.
        if self.rflags & 2 == 0 || self.rflags & ((1 << 14) | (1 << 17)) != 0 {
            return Err(TrapError::InvalidFlags);
        }
        let has_error = ERROR_CODE_VECTORS & (1 << self.vector) != 0;
        if !has_error && self.error != 0 {
            return Err(TrapError::InvalidErrorCode);
        }
        Ok(())
    }

    pub fn disposition(&self, fault_address: Option<u64>) -> Result<TrapDisposition, TrapError> {
        self.validate()?;
        let vector = self.vector as u8;
        if self.cs & 3 == 3 && NEVER_RECOVER_VECTORS & (1 << self.vector) == 0 {
            let address = if vector == 14 {
                let address = fault_address.ok_or(TrapError::InvalidErrorCode)?;
                if !canonical(address) {
                    return Err(TrapError::NonCanonicalInstruction);
                }
                Some(address)
            } else {
                if fault_address.is_some() {
                    return Err(TrapError::InvalidErrorCode);
                }
                None
            };
            Ok(TrapDisposition::TerminateUser { vector, address })
        } else {
            Ok(TrapDisposition::HaltKernel { vector })
        }
    }
}

const fn canonical(address: u64) -> bool {
    ((address << 16) as i64 >> 16) as u64 == address
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(vector: u64, error: u64, privilege: u64) -> TrapFrame {
        TrapFrame {
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
            vector,
            error,
            rip: 0x0000_7fff_ffff_f000,
            cs: privilege,
            rflags: 2,
            rsp: 0x0000_7000_0000_0000,
            ss: privilege,
        }
    }

    #[test]
    fn user_faults_terminate_only_the_current_task() {
        assert_eq!(
            frame(0, 0, 3).disposition(None),
            Ok(TrapDisposition::TerminateUser {
                vector: 0,
                address: None
            })
        );
        assert_eq!(
            frame(14, 7, 3).disposition(Some(0x0000_1234_5678_9000)),
            Ok(TrapDisposition::TerminateUser {
                vector: 14,
                address: Some(0x0000_1234_5678_9000)
            })
        );
    }

    #[test]
    fn kernel_and_integrity_faults_fail_stop() {
        assert_eq!(
            frame(13, 0, 0).disposition(None),
            Ok(TrapDisposition::HaltKernel { vector: 13 })
        );
        assert_eq!(
            frame(8, 0, 3).disposition(None),
            Ok(TrapDisposition::HaltKernel { vector: 8 })
        );
        assert_eq!(
            frame(18, 0, 3).disposition(None),
            Ok(TrapDisposition::HaltKernel { vector: 18 })
        );
    }

    #[test]
    fn malformed_frames_never_reach_policy() {
        assert_eq!(frame(32, 0, 3).validate(), Err(TrapError::InvalidVector));
        assert_eq!(frame(6, 1, 3).validate(), Err(TrapError::InvalidErrorCode));
        assert_eq!(
            frame(14, 0, 3).disposition(None),
            Err(TrapError::InvalidErrorCode)
        );
        let mut forged = frame(14, 0, 3);
        forged.rip = 0x0000_8000_0000_0000;
        assert_eq!(forged.validate(), Err(TrapError::NonCanonicalInstruction));
        forged = frame(14, 0, 3);
        forged.rflags = 1 << 14;
        assert_eq!(forged.validate(), Err(TrapError::InvalidFlags));
    }

    #[test]
    fn trap_frame_layout_matches_assembly_contract() {
        assert_eq!(core::mem::size_of::<TrapFrame>(), 22 * 8);
        assert_eq!(core::mem::offset_of!(TrapFrame, vector), 15 * 8);
        assert_eq!(core::mem::offset_of!(TrapFrame, rip), 17 * 8);
    }
}
