use mrml_kernel::{GuestAccess, HandoffError, PAGE_SIZE, PeError, VmExit};

pub const WHP_EXIT_CONTEXT_BYTES: usize = 224;
const EXIT_MEMORY_ACCESS: u32 = 1;
const EXIT_IO_PORT_ACCESS: u32 = 2;
const EXIT_UNRECOVERABLE_EXCEPTION: u32 = 4;
const EXIT_INVALID_VP_REGISTER: u32 = 5;
const EXIT_UNSUPPORTED_FEATURE: u32 = 6;
const EXIT_INTERRUPT_WINDOW: u32 = 7;
const EXIT_HALT: u32 = 8;
const EXIT_APIC_EOI: u32 = 9;
const EXIT_MSR_ACCESS: u32 = 0x1000;
const EXIT_CPUID: u32 = 0x1001;
const EXIT_EXCEPTION: u32 = 0x1002;
const EXIT_CANCELED: u32 = 0x2001;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhpError {
    EmptyMemory,
    UnalignedMemory,
    MemoryOverflow,
    MemoryTableFull,
    MemoryOverlap,
    InvalidPermissions,
    InvalidRegisterState,
    InvalidVcpu,
    InvalidInterrupt,
    InvalidMapping,
    UnmappedMemory,
    ReadOnlyMemory,
    PageTable,
    Pe(PeError),
    Handoff(HandoffError),
    TruncatedExit,
    MalformedExit,
    UnsupportedExit,
    PlatformUnavailable,
    MissingPlatformFunction,
    SystemCall(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MapPermissions(u32);

impl MapPermissions {
    pub const READ: Self = Self(1);
    pub const WRITE: Self = Self(2);
    pub const EXECUTE: Self = Self(4);

    pub const fn read_only() -> Self {
        Self::READ
    }
    pub const fn read_write() -> Self {
        Self(Self::READ.0 | Self::WRITE.0)
    }
    pub const fn read_execute() -> Self {
        Self(Self::READ.0 | Self::EXECUTE.0)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }

    fn validate(self) -> Result<(), WhpError> {
        if self.0 == 0
            || self.0 & !7 != 0
            || self.0 & Self::WRITE.0 != 0 && self.0 & Self::EXECUTE.0 != 0
        {
            return Err(WhpError::InvalidPermissions);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestRange {
    guest_address: u64,
    size: u64,
    permissions: MapPermissions,
}

impl GuestRange {
    pub fn new(
        guest_address: u64,
        size: u64,
        permissions: MapPermissions,
    ) -> Result<Self, WhpError> {
        permissions.validate()?;
        if size == 0 {
            return Err(WhpError::EmptyMemory);
        }
        if !guest_address.is_multiple_of(PAGE_SIZE) || !size.is_multiple_of(PAGE_SIZE) {
            return Err(WhpError::UnalignedMemory);
        }
        guest_address
            .checked_add(size)
            .ok_or(WhpError::MemoryOverflow)?;
        Ok(Self {
            guest_address,
            size,
            permissions,
        })
    }

    pub const fn guest_address(self) -> u64 {
        self.guest_address
    }
    pub const fn size(self) -> u64 {
        self.size
    }
    pub const fn permissions(self) -> MapPermissions {
        self.permissions
    }
}

/// Decodes only the stable prefix and context variants whose complete byte layout is used.
/// Unknown and architecture-specific exits fail closed instead of exposing an unchecked union.
pub fn decode_exit_context(input: &[u8]) -> Result<VmExit, WhpError> {
    if input.len() < 8 {
        return Err(WhpError::TruncatedExit);
    }
    match u32_at(input, 0)? {
        EXIT_HALT => Ok(VmExit::Halted),
        EXIT_CANCELED | EXIT_INTERRUPT_WINDOW => Ok(VmExit::Interrupted),
        EXIT_MEMORY_ACCESS => decode_memory(input),
        EXIT_IO_PORT_ACCESS => decode_io(input),
        EXIT_EXCEPTION => decode_exception(input),
        EXIT_UNRECOVERABLE_EXCEPTION
        | EXIT_INVALID_VP_REGISTER
        | EXIT_UNSUPPORTED_FEATURE
        | EXIT_APIC_EOI
        | EXIT_MSR_ACCESS
        | EXIT_CPUID => Err(WhpError::UnsupportedExit),
        _ => Err(WhpError::UnsupportedExit),
    }
}

fn decode_exception(input: &[u8]) -> Result<VmExit, WhpError> {
    let info = u32_at(input, 68)?;
    if info & !3 != 0 {
        return Err(WhpError::MalformedExit);
    }
    let exception = *input.get(72).ok_or(WhpError::TruncatedExit)?;
    Ok(VmExit::Unknown {
        reason: (EXIT_EXCEPTION as u64) << 32 | exception as u64,
    })
}

fn decode_memory(input: &[u8]) -> Result<VmExit, WhpError> {
    // 8-byte exit prefix + 40-byte VP context. MemoryAccessInfo is a 32-bit bitfield,
    // followed by instruction bytes, GPA, and GVA.
    let info = u32_at(input, 68)?;
    let access = info & 3;
    let access = match access {
        0 => GuestAccess::Read,
        1 => GuestAccess::Write,
        2 => GuestAccess::Execute,
        _ => return Err(WhpError::MalformedExit),
    };
    if info & !0x0f != 0 {
        return Err(WhpError::MalformedExit);
    }
    let guest_address = u64_at(input, 72)?;
    Ok(VmExit::GuestMemoryFault {
        guest_address,
        access,
    })
}

fn decode_io(input: &[u8]) -> Result<VmExit, WhpError> {
    // The I/O context uses a 32-bit access-info bitfield, port, and RAX value.
    let info = u32_at(input, 68)?;
    if info & !0x3f != 0 {
        return Err(WhpError::MalformedExit);
    }
    let write = info & 1 != 0;
    let size = ((info >> 1) & 7) as u8;
    if !matches!(size, 1 | 2 | 4) {
        return Err(WhpError::MalformedExit);
    }
    let port = u16_at(input, 72)?;
    let value = u64_at(input, 80)? as u32;
    Ok(VmExit::Io {
        port,
        size,
        write,
        value,
    })
}

fn u16_at(input: &[u8], at: usize) -> Result<u16, WhpError> {
    Ok(u16::from_ne_bytes(
        input
            .get(at..at + 2)
            .ok_or(WhpError::TruncatedExit)?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(input: &[u8], at: usize) -> Result<u32, WhpError> {
    Ok(u32::from_ne_bytes(
        input
            .get(at..at + 4)
            .ok_or(WhpError::TruncatedExit)?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(input: &[u8], at: usize) -> Result<u64, WhpError> {
    Ok(u64::from_ne_bytes(
        input
            .get(at..at + 8)
            .ok_or(WhpError::TruncatedExit)?
            .try_into()
            .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_require_pages_and_w_x() {
        assert!(GuestRange::new(0x2000, 0x3000, MapPermissions::read_write()).is_ok());
        assert_eq!(
            GuestRange::new(1, 0x1000, MapPermissions::READ),
            Err(WhpError::UnalignedMemory)
        );
        assert_eq!(
            GuestRange::new(0, 0x1000, MapPermissions(7)),
            Err(WhpError::InvalidPermissions)
        );
        assert_eq!(
            GuestRange::new(u64::MAX - 0xfff, 0x2000, MapPermissions::READ),
            Err(WhpError::MemoryOverflow)
        );
    }

    #[test]
    fn halt_and_cancel_have_no_union_dependency() {
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&EXIT_HALT.to_ne_bytes());
        assert_eq!(decode_exit_context(&bytes), Ok(VmExit::Halted));
        bytes[..4].copy_from_slice(&EXIT_CANCELED.to_ne_bytes());
        assert_eq!(decode_exit_context(&bytes), Ok(VmExit::Interrupted));
    }

    #[test]
    fn malformed_and_truncated_exits_fail_closed() {
        assert_eq!(decode_exit_context(&[0; 7]), Err(WhpError::TruncatedExit));
        let mut bytes = [0u8; 80];
        bytes[..4].copy_from_slice(&EXIT_IO_PORT_ACCESS.to_ne_bytes());
        bytes[68..72].copy_from_slice(&(3u32 << 1).to_ne_bytes());
        assert_eq!(decode_exit_context(&bytes), Err(WhpError::MalformedExit));
    }

    #[test]
    fn breakpoint_exception_preserves_architectural_vector() {
        let mut bytes = [0u8; 80];
        bytes[..4].copy_from_slice(&EXIT_EXCEPTION.to_ne_bytes());
        bytes[72] = 3;
        assert_eq!(
            decode_exit_context(&bytes),
            Ok(VmExit::Unknown {
                reason: (EXIT_EXCEPTION as u64) << 32 | 3,
            })
        );
        bytes[68..72].copy_from_slice(&4u32.to_ne_bytes());
        assert_eq!(decode_exit_context(&bytes), Err(WhpError::MalformedExit));
    }
}
