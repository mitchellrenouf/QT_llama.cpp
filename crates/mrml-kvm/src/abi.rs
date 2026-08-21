use mrml_kernel::{VmExit, PAGE_SIZE};

pub const KVM_API_VERSION: i32 = 12;
pub const MRML_KVM_HYPERCALL: u64 = 0x4d52_4d4c;
pub const KVM_MEMORY_REGION_BYTES: usize = 32;
pub const MAX_KVM_MEMORY_SLOTS: u32 = 32;

const UNION: usize = 32;
const EXIT_UNKNOWN: u32 = 0;
const EXIT_EXCEPTION: u32 = 1;
const EXIT_IO: u32 = 2;
const EXIT_HYPERCALL: u32 = 3;
const EXIT_HLT: u32 = 5;
const EXIT_MMIO: u32 = 6;
const EXIT_SHUTDOWN: u32 = 8;
const EXIT_INTR: u32 = 10;
const MEMORY_READONLY: u32 = 1 << 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvmError {
    TruncatedRun,
    MalformedExit,
    UnsupportedExit,
    UnalignedMemory,
    EmptyMemory,
    MemoryOverflow,
    InvalidSlot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmMemoryRegion {
    slot: u32,
    flags: u32,
    guest_address: u64,
    size: u64,
    userspace_address: u64,
}

impl KvmMemoryRegion {
    pub fn new(
        slot: u32,
        guest_address: u64,
        size: u64,
        userspace_address: u64,
        readonly: bool,
    ) -> Result<Self, KvmError> {
        if slot >= MAX_KVM_MEMORY_SLOTS {
            return Err(KvmError::InvalidSlot);
        }
        if size == 0 {
            return Err(KvmError::EmptyMemory);
        }
        if guest_address % PAGE_SIZE != 0
            || size % PAGE_SIZE != 0
            || userspace_address % PAGE_SIZE != 0
        {
            return Err(KvmError::UnalignedMemory);
        }
        guest_address
            .checked_add(size)
            .ok_or(KvmError::MemoryOverflow)?;
        userspace_address
            .checked_add(size)
            .ok_or(KvmError::MemoryOverflow)?;
        Ok(Self {
            slot,
            flags: if readonly { MEMORY_READONLY } else { 0 },
            guest_address,
            size,
            userspace_address,
        })
    }

    pub fn encode(self) -> [u8; KVM_MEMORY_REGION_BYTES] {
        let mut output = [0u8; KVM_MEMORY_REGION_BYTES];
        output[0..4].copy_from_slice(&self.slot.to_ne_bytes());
        output[4..8].copy_from_slice(&self.flags.to_ne_bytes());
        output[8..16].copy_from_slice(&self.guest_address.to_ne_bytes());
        output[16..24].copy_from_slice(&self.size.to_ne_bytes());
        output[24..32].copy_from_slice(&self.userspace_address.to_ne_bytes());
        output
    }
}

/// Decodes a kernel-owned `kvm_run` page as bytes, avoiding references to its C union.
pub fn decode_run_page(run: &[u8]) -> Result<VmExit, KvmError> {
    if run.len() < UNION {
        return Err(KvmError::TruncatedRun);
    }
    let reason = u32_at(run, 8)?;
    match reason {
        EXIT_UNKNOWN => Ok(VmExit::Unknown {
            reason: u64_at(run, UNION)?,
        }),
        EXIT_EXCEPTION => Ok(VmExit::Unknown {
            reason: (reason as u64) << 32 | u32_at(run, UNION)? as u64,
        }),
        EXIT_IO => decode_io(run),
        EXIT_HYPERCALL => decode_hypercall(run),
        EXIT_HLT => Ok(VmExit::Halted),
        EXIT_MMIO => decode_mmio(run),
        EXIT_SHUTDOWN => Ok(VmExit::Unknown {
            reason: EXIT_SHUTDOWN as u64,
        }),
        EXIT_INTR => Ok(VmExit::Interrupted),
        _ => Err(KvmError::UnsupportedExit),
    }
}

fn decode_io(run: &[u8]) -> Result<VmExit, KvmError> {
    let direction = byte(run, 32)?;
    let size = byte(run, 33)?;
    let port = u16_at(run, 34)?;
    let count = u32_at(run, 36)?;
    let offset = usize::try_from(u64_at(run, 40)?).map_err(|_| KvmError::MalformedExit)?;
    if !matches!(direction, 0 | 1) || !matches!(size, 1 | 2 | 4) || count != 1 {
        return Err(KvmError::MalformedExit);
    }
    let end = offset
        .checked_add(size as usize)
        .ok_or(KvmError::MalformedExit)?;
    let data = run.get(offset..end).ok_or(KvmError::TruncatedRun)?;
    let mut value = [0u8; 4];
    value[..size as usize].copy_from_slice(data);
    Ok(VmExit::Io {
        port,
        size,
        write: direction == 1,
        value: u32::from_ne_bytes(value),
    })
}

fn decode_hypercall(run: &[u8]) -> Result<VmExit, KvmError> {
    if u64_at(run, 32)? != MRML_KVM_HYPERCALL {
        return Err(KvmError::UnsupportedExit);
    }
    let descriptor_address = u64_at(run, 40)?;
    for at in [48, 56, 64, 72, 80] {
        if u64_at(run, at)? != 0 {
            return Err(KvmError::MalformedExit);
        }
    }
    Ok(VmExit::Hypercall { descriptor_address })
}

fn decode_mmio(run: &[u8]) -> Result<VmExit, KvmError> {
    let guest_address = u64_at(run, 32)?;
    let size = u32_at(run, 48)?;
    let write = byte(run, 52)?;
    if !matches!(size, 1 | 2 | 4 | 8) || !matches!(write, 0 | 1) {
        return Err(KvmError::MalformedExit);
    }
    let data = run.get(40..48).ok_or(KvmError::TruncatedRun)?;
    let mut value = [0u8; 8];
    value[..size as usize].copy_from_slice(&data[..size as usize]);
    Ok(VmExit::Mmio {
        guest_address,
        size: size as u8,
        write: write == 1,
        value: u64::from_ne_bytes(value),
    })
}

fn byte(input: &[u8], at: usize) -> Result<u8, KvmError> {
    input.get(at).copied().ok_or(KvmError::TruncatedRun)
}
fn u16_at(input: &[u8], at: usize) -> Result<u16, KvmError> {
    Ok(u16::from_ne_bytes(
        input
            .get(at..at + 2)
            .ok_or(KvmError::TruncatedRun)?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(input: &[u8], at: usize) -> Result<u32, KvmError> {
    Ok(u32::from_ne_bytes(
        input
            .get(at..at + 4)
            .ok_or(KvmError::TruncatedRun)?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(input: &[u8], at: usize) -> Result<u64, KvmError> {
    Ok(u64::from_ne_bytes(
        input
            .get(at..at + 8)
            .ok_or(KvmError::TruncatedRun)?
            .try_into()
            .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_regions_are_aligned_bounded_and_canonical() {
        let encoded = KvmMemoryRegion::new(3, 0x2000, 0x4000, 0x8000, true)
            .unwrap()
            .encode();
        assert_eq!(u32::from_ne_bytes(encoded[0..4].try_into().unwrap()), 3);
        assert_eq!(
            u32::from_ne_bytes(encoded[4..8].try_into().unwrap()),
            MEMORY_READONLY
        );
        assert_eq!(
            KvmMemoryRegion::new(32, 0, 0x1000, 0x1000, false),
            Err(KvmError::InvalidSlot)
        );
        assert_eq!(
            KvmMemoryRegion::new(0, 1, 0x1000, 0x1000, false),
            Err(KvmError::UnalignedMemory)
        );
    }

    #[test]
    fn io_exit_requires_one_bounded_scalar_transfer() {
        let mut run = [0u8; 128];
        run[8..12].copy_from_slice(&EXIT_IO.to_ne_bytes());
        run[32] = 1;
        run[33] = 2;
        run[34..36].copy_from_slice(&0x3f8u16.to_ne_bytes());
        run[36..40].copy_from_slice(&1u32.to_ne_bytes());
        run[40..48].copy_from_slice(&96u64.to_ne_bytes());
        run[96..98].copy_from_slice(&0x1234u16.to_ne_bytes());
        assert_eq!(
            decode_run_page(&run),
            Ok(VmExit::Io {
                port: 0x3f8,
                size: 2,
                write: true,
                value: 0x1234
            })
        );
        run[36..40].copy_from_slice(&2u32.to_ne_bytes());
        assert_eq!(decode_run_page(&run), Err(KvmError::MalformedExit));
    }

    #[test]
    fn hypercall_requires_number_and_zero_unused_arguments() {
        let mut run = [0u8; 96];
        run[8..12].copy_from_slice(&EXIT_HYPERCALL.to_ne_bytes());
        run[32..40].copy_from_slice(&MRML_KVM_HYPERCALL.to_ne_bytes());
        run[40..48].copy_from_slice(&0x4000u64.to_ne_bytes());
        assert_eq!(
            decode_run_page(&run),
            Ok(VmExit::Hypercall {
                descriptor_address: 0x4000
            })
        );
        run[80] = 1;
        assert_eq!(decode_run_page(&run), Err(KvmError::MalformedExit));
    }

    #[test]
    fn mmio_and_truncation_are_strictly_decoded() {
        let mut run = [0u8; 64];
        run[8..12].copy_from_slice(&EXIT_MMIO.to_ne_bytes());
        run[32..40].copy_from_slice(&0xfee0_0000u64.to_ne_bytes());
        run[40..44].copy_from_slice(&7u32.to_ne_bytes());
        run[48..52].copy_from_slice(&4u32.to_ne_bytes());
        run[52] = 1;
        assert_eq!(
            decode_run_page(&run),
            Ok(VmExit::Mmio {
                guest_address: 0xfee0_0000,
                size: 4,
                write: true,
                value: 7
            })
        );
        assert_eq!(decode_run_page(&run[..35]), Err(KvmError::TruncatedRun));
    }
}
