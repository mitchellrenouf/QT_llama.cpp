use crate::{
    BootEvidence, BootValidationError, FramebufferError, FramebufferInfo, MemoryError, MemoryKind,
    MemoryMap, MemoryRegion, PhysAddr, PixelFormat,
};

pub const HANDOFF_HEADER_BYTES: usize = 168;
pub const HANDOFF_REGION_BYTES: usize = 24;
pub const MAX_HANDOFF_REGIONS: usize = 128;
pub const MAX_HANDOFF_MADT_BYTES: usize = 8 * 1024;

const FLAG_SECURE_BOOT: u16 = 1;
const FLAG_MEASURED_BOOT: u16 = 1 << 1;
const FLAG_ROLLBACK_PROTECTED: u16 = 1 << 2;
const FLAG_MADT_SNAPSHOT: u16 = 1 << 3;
const FLAG_AP_TRAMPOLINE: u16 = 1 << 4;
const KNOWN_FLAGS: u16 = FLAG_SECURE_BOOT
    | FLAG_MEASURED_BOOT
    | FLAG_ROLLBACK_PROTECTED
    | FLAG_MADT_SNAPSHOT
    | FLAG_AP_TRAMPOLINE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffError {
    Truncated,
    BadMagic,
    NonCanonical,
    UnknownFlags,
    TooManyRegions,
    MissingAcpiRoot,
    InvalidFramebuffer(FramebufferError),
    FramebufferOutsideMmio,
    InvalidMemoryMap(MemoryError),
    InvalidBootEvidence(BootValidationError),
}

/// Validated architecture-neutral data passed after UEFI boot services exit.
/// Firmware-owned structures must be normalized into this format first.
pub struct BootHandoff {
    evidence: BootEvidence,
    acpi_root: u64,
    framebuffer: FramebufferInfo,
    region_count: usize,
    madt_offset: usize,
    madt_length: usize,
    ap_trampoline: Option<u64>,
}

impl BootHandoff {
    pub const fn evidence(&self) -> &BootEvidence {
        &self.evidence
    }

    pub const fn acpi_root(&self) -> u64 {
        self.acpi_root
    }

    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    pub const fn framebuffer(&self) -> FramebufferInfo {
        self.framebuffer
    }

    pub fn madt<'a>(&self, encoded: &'a [u8]) -> Option<&'a [u8]> {
        (self.madt_length != 0)
            .then(|| &encoded[self.madt_offset..self.madt_offset + self.madt_length])
    }

    pub const fn ap_trampoline(&self) -> Option<u64> {
        self.ap_trampoline
    }

    /// Parses a bounded canonical handoff and emits validated regions in
    /// ascending address order. Callers can place them directly into fixed
    /// early-boot storage; the parser never allocates or dereferences firmware
    /// pointers.
    pub fn decode<F>(input: &[u8], mut emit: F) -> Result<Self, HandoffError>
    where
        F: FnMut(MemoryRegion),
    {
        if input.len() < HANDOFF_HEADER_BYTES {
            return Err(HandoffError::Truncated);
        }
        if &input[..16] != b"MRML-HANDOFF-v1\0" {
            return Err(HandoffError::BadMagic);
        }
        let encoded_length = read_u32(input, 16) as usize;
        let region_count = read_u16(input, 20) as usize;
        let flags = read_u16(input, 22);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(HandoffError::UnknownFlags);
        }
        if region_count == 0 || region_count > MAX_HANDOFF_REGIONS {
            return Err(HandoffError::TooManyRegions);
        }
        let regions_end = HANDOFF_HEADER_BYTES
            .checked_add(
                region_count
                    .checked_mul(HANDOFF_REGION_BYTES)
                    .ok_or(HandoffError::NonCanonical)?,
            )
            .ok_or(HandoffError::NonCanonical)?;
        let madt_length = read_u16(input, 165) as usize;
        if (flags & FLAG_MADT_SNAPSHOT != 0) != (madt_length != 0)
            || madt_length > MAX_HANDOFF_MADT_BYTES
            || (madt_length != 0 && madt_length < 44)
            || input[167] != 0
        {
            return Err(HandoffError::NonCanonical);
        }
        let trampoline_bytes = usize::from(flags & FLAG_AP_TRAMPOLINE != 0) * 8;
        let madt_offset = regions_end
            .checked_add(trampoline_bytes)
            .ok_or(HandoffError::NonCanonical)?;
        let expected_length = madt_offset
            .checked_add(madt_length)
            .ok_or(HandoffError::NonCanonical)?;
        if encoded_length != expected_length || input.len() != expected_length {
            return Err(HandoffError::NonCanonical);
        }
        if madt_length != 0 {
            let madt = &input[madt_offset..expected_length];
            if &madt[..4] != b"APIC"
                || read_u32(madt, 4) as usize != madt_length
                || madt.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) != 0
            {
                return Err(HandoffError::NonCanonical);
            }
        }

        let image_version = read_u64(input, 24);
        let entropy = input[32..64].try_into().unwrap();
        let image_measurement = input[64..128].try_into().unwrap();
        let acpi_address = read_u64(input, 128);
        if acpi_address == 0 {
            return Err(HandoffError::MissingAcpiRoot);
        }
        let acpi_root = acpi_address;
        let ap_trampoline = if trampoline_bytes != 0 {
            let physical = read_u64(input, regions_end);
            if !(crate::PAGE_SIZE..0x10_0000).contains(&physical)
                || !physical.is_multiple_of(crate::PAGE_SIZE)
            {
                return Err(HandoffError::NonCanonical);
            }
            Some(physical)
        } else {
            None
        };
        let pixel_format = match input[164] {
            0 => PixelFormat::RedGreenBlueReserved,
            1 => PixelFormat::BlueGreenRedReserved,
            _ => {
                return Err(HandoffError::InvalidFramebuffer(
                    FramebufferError::UnsupportedPixelFormat,
                ));
            }
        };
        let framebuffer = FramebufferInfo::new(
            read_u64(input, 136),
            read_u64(input, 144),
            read_u32(input, 152),
            read_u32(input, 156),
            read_u32(input, 160),
            pixel_format,
        )
        .map_err(HandoffError::InvalidFramebuffer)?;

        let mut previous_start = 0u64;
        let mut previous_end = 0u64;
        let mut framebuffer_is_mmio = false;
        for index in 0..region_count {
            let region = decode_region(input, index)?;
            let start = region.start().get();
            if index != 0 && start < previous_start {
                return Err(HandoffError::InvalidMemoryMap(MemoryError::Unsorted));
            }
            if index != 0 && start < previous_end {
                return Err(HandoffError::InvalidMemoryMap(MemoryError::Overlap));
            }
            previous_start = start;
            previous_end = region.end();
            if region.kind() == MemoryKind::Mmio
                && region.start().get() <= framebuffer.base().get()
                && region.end() >= framebuffer.end()
            {
                framebuffer_is_mmio = true;
            }
        }
        if !framebuffer_is_mmio {
            return Err(HandoffError::FramebufferOutsideMmio);
        }

        let evidence = BootEvidence::new(
            entropy,
            image_measurement,
            image_version,
            flags & FLAG_SECURE_BOOT != 0,
            flags & FLAG_MEASURED_BOOT != 0,
            flags & FLAG_ROLLBACK_PROTECTED != 0,
        )
        .map_err(HandoffError::InvalidBootEvidence)?;
        for index in 0..region_count {
            emit(decode_region(input, index)?);
        }
        Ok(Self {
            evidence,
            acpi_root,
            framebuffer,
            region_count,
            madt_offset,
            madt_length,
            ap_trampoline,
        })
    }
}

// Each argument is an independently validated field of the fixed wire format;
// grouping them would hide rather than reduce this trust-boundary surface.
#[allow(clippy::too_many_arguments)]
pub fn encode_handoff(
    image_version: u64,
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    secure_boot: bool,
    measured_boot: bool,
    rollback_protected: bool,
    acpi_root: u64,
    framebuffer: FramebufferInfo,
    regions: &[MemoryRegion],
    output: &mut [u8],
) -> Result<usize, HandoffError> {
    encode_handoff_inner(
        image_version,
        entropy,
        image_measurement,
        secure_boot,
        measured_boot,
        rollback_protected,
        acpi_root,
        framebuffer,
        regions,
        None,
        None,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_handoff_with_madt(
    image_version: u64,
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    secure_boot: bool,
    measured_boot: bool,
    rollback_protected: bool,
    acpi_root: u64,
    framebuffer: FramebufferInfo,
    regions: &[MemoryRegion],
    madt: &[u8],
    output: &mut [u8],
) -> Result<usize, HandoffError> {
    if !(44..=MAX_HANDOFF_MADT_BYTES).contains(&madt.len())
        || &madt[..4] != b"APIC"
        || read_u32(madt, 4) as usize != madt.len()
        || madt.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) != 0
    {
        return Err(HandoffError::NonCanonical);
    }
    encode_handoff_inner(
        image_version,
        entropy,
        image_measurement,
        secure_boot,
        measured_boot,
        rollback_protected,
        acpi_root,
        framebuffer,
        regions,
        Some(madt),
        None,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_handoff_with_smp(
    image_version: u64,
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    secure_boot: bool,
    measured_boot: bool,
    rollback_protected: bool,
    acpi_root: u64,
    framebuffer: FramebufferInfo,
    regions: &[MemoryRegion],
    ap_trampoline: u64,
    madt: &[u8],
    output: &mut [u8],
) -> Result<usize, HandoffError> {
    if !(crate::PAGE_SIZE..0x10_0000).contains(&ap_trampoline)
        || !ap_trampoline.is_multiple_of(crate::PAGE_SIZE)
        || !(44..=MAX_HANDOFF_MADT_BYTES).contains(&madt.len())
        || &madt[..4] != b"APIC"
        || read_u32(madt, 4) as usize != madt.len()
        || madt.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) != 0
    {
        return Err(HandoffError::NonCanonical);
    }
    encode_handoff_inner(
        image_version,
        entropy,
        image_measurement,
        secure_boot,
        measured_boot,
        rollback_protected,
        acpi_root,
        framebuffer,
        regions,
        Some(madt),
        Some(ap_trampoline),
        output,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_handoff_inner(
    image_version: u64,
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    secure_boot: bool,
    measured_boot: bool,
    rollback_protected: bool,
    acpi_root: u64,
    framebuffer: FramebufferInfo,
    regions: &[MemoryRegion],
    madt: Option<&[u8]>,
    ap_trampoline: Option<u64>,
    output: &mut [u8],
) -> Result<usize, HandoffError> {
    if regions.is_empty() || regions.len() > MAX_HANDOFF_REGIONS {
        return Err(HandoffError::TooManyRegions);
    }
    let regions_end = HANDOFF_HEADER_BYTES
        .checked_add(
            regions
                .len()
                .checked_mul(HANDOFF_REGION_BYTES)
                .ok_or(HandoffError::NonCanonical)?,
        )
        .ok_or(HandoffError::NonCanonical)?;
    let madt_length = madt.map_or(0, <[u8]>::len);
    let madt_offset = regions_end
        .checked_add(usize::from(ap_trampoline.is_some()) * 8)
        .ok_or(HandoffError::NonCanonical)?;
    let length = madt_offset
        .checked_add(madt_length)
        .ok_or(HandoffError::NonCanonical)?;
    if output.len() < length || image_version == 0 || acpi_root == 0 {
        return Err(HandoffError::NonCanonical);
    }
    MemoryMap::new(regions).map_err(HandoffError::InvalidMemoryMap)?;
    if !regions.iter().any(|region| {
        region.kind() == MemoryKind::Mmio
            && region.start().get() <= framebuffer.base().get()
            && region.end() >= framebuffer.end()
    }) {
        return Err(HandoffError::FramebufferOutsideMmio);
    }
    BootEvidence::new(
        entropy,
        image_measurement,
        image_version,
        secure_boot,
        measured_boot,
        rollback_protected,
    )
    .map_err(HandoffError::InvalidBootEvidence)?;
    output[..length].fill(0);
    output[..16].copy_from_slice(b"MRML-HANDOFF-v1\0");
    output[16..20].copy_from_slice(&(length as u32).to_le_bytes());
    output[20..22].copy_from_slice(&(regions.len() as u16).to_le_bytes());
    let flags = ((secure_boot as u16) * FLAG_SECURE_BOOT)
        | ((measured_boot as u16) * FLAG_MEASURED_BOOT)
        | ((rollback_protected as u16) * FLAG_ROLLBACK_PROTECTED)
        | ((madt.is_some() as u16) * FLAG_MADT_SNAPSHOT)
        | ((ap_trampoline.is_some() as u16) * FLAG_AP_TRAMPOLINE);
    output[22..24].copy_from_slice(&flags.to_le_bytes());
    output[24..32].copy_from_slice(&image_version.to_le_bytes());
    output[32..64].copy_from_slice(&entropy);
    output[64..128].copy_from_slice(&image_measurement);
    output[128..136].copy_from_slice(&acpi_root.to_le_bytes());
    output[136..144].copy_from_slice(&framebuffer.base().get().to_le_bytes());
    output[144..152].copy_from_slice(&framebuffer.byte_length().to_le_bytes());
    output[152..156].copy_from_slice(&framebuffer.width().to_le_bytes());
    output[156..160].copy_from_slice(&framebuffer.height().to_le_bytes());
    output[160..164].copy_from_slice(&framebuffer.stride().to_le_bytes());
    output[164] = match framebuffer.pixel_format() {
        PixelFormat::RedGreenBlueReserved => 0,
        PixelFormat::BlueGreenRedReserved => 1,
    };
    output[165..167].copy_from_slice(&(madt_length as u16).to_le_bytes());
    for (index, region) in regions.iter().copied().enumerate() {
        let offset = HANDOFF_HEADER_BYTES + index * HANDOFF_REGION_BYTES;
        output[offset..offset + 8].copy_from_slice(&region.start().get().to_le_bytes());
        output[offset + 8..offset + 16].copy_from_slice(&region.pages().to_le_bytes());
        output[offset + 16] = region.kind() as u8;
    }
    if let Some(physical) = ap_trampoline {
        output[regions_end..regions_end + 8].copy_from_slice(&physical.to_le_bytes());
    }
    if let Some(madt) = madt {
        output[madt_offset..length].copy_from_slice(madt);
    }
    Ok(length)
}

fn decode_region(input: &[u8], index: usize) -> Result<MemoryRegion, HandoffError> {
    let offset = HANDOFF_HEADER_BYTES + index * HANDOFF_REGION_BYTES;
    let start = read_u64(input, offset);
    let pages = read_u64(input, offset + 8);
    let kind = decode_kind(input[offset + 16])?;
    if input[offset + 17..offset + HANDOFF_REGION_BYTES]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(HandoffError::NonCanonical);
    }
    MemoryRegion::new(
        PhysAddr::new(start).map_err(HandoffError::InvalidMemoryMap)?,
        pages,
        kind,
    )
    .map_err(HandoffError::InvalidMemoryMap)
}

fn decode_kind(value: u8) -> Result<MemoryKind, HandoffError> {
    match value {
        0 => Ok(MemoryKind::Free),
        1 => Ok(MemoryKind::Kernel),
        2 => Ok(MemoryKind::Firmware),
        3 => Ok(MemoryKind::Mmio),
        4 => Ok(MemoryKind::Acpi),
        5 => Ok(MemoryKind::Reserved),
        _ => Err(HandoffError::NonCanonical),
    }
}

fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(input[offset..offset + 2].try_into().unwrap())
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

    const THREE_REGION_BYTES: usize = HANDOFF_HEADER_BYTES + 3 * HANDOFF_REGION_BYTES;

    fn valid_handoff() -> [u8; THREE_REGION_BYTES] {
        let mut encoded = [0u8; THREE_REGION_BYTES];
        encoded[..16].copy_from_slice(b"MRML-HANDOFF-v1\0");
        encoded[16..20].copy_from_slice(&(THREE_REGION_BYTES as u32).to_le_bytes());
        encoded[20..22].copy_from_slice(&3u16.to_le_bytes());
        encoded[22..24].copy_from_slice(
            &(KNOWN_FLAGS & !(FLAG_MADT_SNAPSHOT | FLAG_AP_TRAMPOLINE)).to_le_bytes(),
        );
        encoded[24..32].copy_from_slice(&7u64.to_le_bytes());
        encoded[32..64].fill(1);
        encoded[64..128].fill(2);
        encoded[128..136].copy_from_slice(&0x9000u64.to_le_bytes());
        encoded[136..144].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[144..152].copy_from_slice(&0x1000u64.to_le_bytes());
        encoded[152..156].copy_from_slice(&16u32.to_le_bytes());
        encoded[156..160].copy_from_slice(&16u32.to_le_bytes());
        encoded[160..164].copy_from_slice(&16u32.to_le_bytes());
        encoded[164] = 1;
        encoded[168..176].copy_from_slice(&0x1000u64.to_le_bytes());
        encoded[176..184].copy_from_slice(&2u64.to_le_bytes());
        encoded[184] = 0;
        encoded[192..200].copy_from_slice(&0x3000u64.to_le_bytes());
        encoded[200..208].copy_from_slice(&1u64.to_le_bytes());
        encoded[208] = 1;
        encoded[216..224].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[224..232].copy_from_slice(&1u64.to_le_bytes());
        encoded[232] = 3;
        encoded
    }

    #[test]
    fn decodes_canonical_bounded_handoff() {
        let encoded = valid_handoff();
        let mut starts = [0u64; 3];
        let mut count = 0;
        let handoff = BootHandoff::decode(&encoded, |region| {
            starts[count] = region.start().get();
            count += 1;
        })
        .unwrap();
        assert_eq!(handoff.region_count(), 3);
        assert_eq!(handoff.acpi_root(), 0x9000);
        assert_eq!(starts, [0x1000, 0x3000, 0xa0000]);
        assert_eq!(handoff.framebuffer().width(), 16);
        assert_eq!(handoff.evidence().image_measurement(), &[2; 64]);
        let mut round_trip = [0u8; THREE_REGION_BYTES];
        let mut regions = [decode_region(&encoded, 0).unwrap(); 3];
        for (index, region) in regions.iter_mut().enumerate() {
            *region = decode_region(&encoded, index).unwrap();
        }
        assert_eq!(
            encode_handoff(
                7,
                [1; 32],
                [2; 64],
                true,
                true,
                true,
                0x9008,
                handoff.framebuffer(),
                &regions,
                &mut round_trip
            ),
            Ok(THREE_REGION_BYTES)
        );
        assert_eq!(
            BootHandoff::decode(&round_trip, |_| {})
                .unwrap()
                .acpi_root(),
            0x9008
        );
    }

    #[test]
    fn rejects_lengths_flags_reserved_bytes_and_overlaps() {
        let mut encoded = valid_handoff();
        encoded[22..24].copy_from_slice(&0x8000u16.to_le_bytes());
        assert_eq!(
            BootHandoff::decode(&encoded, |_| {}).err(),
            Some(HandoffError::UnknownFlags)
        );

        let mut encoded = valid_handoff();
        encoded[185] = 1;
        assert_eq!(
            BootHandoff::decode(&encoded, |_| {}).err(),
            Some(HandoffError::NonCanonical)
        );

        let mut encoded = valid_handoff();
        encoded[192..200].copy_from_slice(&0x2000u64.to_le_bytes());
        let mut emitted = 0;
        assert_eq!(
            BootHandoff::decode(&encoded, |_| emitted += 1).err(),
            Some(HandoffError::InvalidMemoryMap(MemoryError::Overlap))
        );
        assert_eq!(emitted, 0);

        let encoded = valid_handoff();
        assert_eq!(
            BootHandoff::decode(&encoded[..encoded.len() - 1], |_| {}).err(),
            Some(HandoffError::NonCanonical)
        );
    }

    #[test]
    fn appended_madt_is_canonical_and_bound_to_the_length() {
        let encoded = valid_handoff();
        let handoff = BootHandoff::decode(&encoded, |_| {}).unwrap();
        let mut regions = [decode_region(&encoded, 0).unwrap(); 3];
        for (index, region) in regions.iter_mut().enumerate() {
            *region = decode_region(&encoded, index).unwrap();
        }
        let mut madt = [0u8; 52];
        madt[..4].copy_from_slice(b"APIC");
        madt[4..8].copy_from_slice(&52u32.to_le_bytes());
        madt[36..40].copy_from_slice(&0xfee0_0000u32.to_le_bytes());
        madt[44..].copy_from_slice(&[0, 8, 0, 1, 1, 0, 0, 0]);
        madt[9] = 0u8.wrapping_sub(madt.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)));
        let mut with_madt = [0u8; THREE_REGION_BYTES + 52];
        assert_eq!(
            encode_handoff_with_madt(
                7,
                [1; 32],
                [2; 64],
                true,
                true,
                true,
                0x9000,
                handoff.framebuffer(),
                &regions,
                &madt,
                &mut with_madt,
            ),
            Ok(with_madt.len())
        );
        let decoded = BootHandoff::decode(&with_madt, |_| {}).unwrap();
        assert_eq!(decoded.madt(&with_madt), Some(madt.as_slice()));
        let mut with_smp = [0u8; THREE_REGION_BYTES + 8 + 52];
        assert_eq!(
            encode_handoff_with_smp(
                7,
                [1; 32],
                [2; 64],
                true,
                true,
                true,
                0x9000,
                handoff.framebuffer(),
                &regions,
                0x8000,
                &madt,
                &mut with_smp,
            ),
            Ok(with_smp.len())
        );
        let smp = BootHandoff::decode(&with_smp, |_| {}).unwrap();
        assert_eq!(smp.ap_trampoline(), Some(0x8000));
        assert_eq!(smp.madt(&with_smp), Some(madt.as_slice()));
        with_madt[165] = 51;
        assert_eq!(
            BootHandoff::decode(&with_madt, |_| {}).err(),
            Some(HandoffError::NonCanonical)
        );
    }
}
