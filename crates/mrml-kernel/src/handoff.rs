use crate::{
    BootEvidence, BootValidationError, FramebufferError, FramebufferInfo, MemoryError, MemoryKind,
    MemoryMap, MemoryRegion, PhysAddr, PixelFormat,
};

pub const HANDOFF_HEADER_BYTES: usize = 168;
pub const HANDOFF_REGION_BYTES: usize = 24;
pub const MAX_HANDOFF_REGIONS: usize = 128;

const FLAG_SECURE_BOOT: u16 = 1;
const FLAG_MEASURED_BOOT: u16 = 1 << 1;
const FLAG_ROLLBACK_PROTECTED: u16 = 1 << 2;
const KNOWN_FLAGS: u16 = FLAG_SECURE_BOOT | FLAG_MEASURED_BOOT | FLAG_ROLLBACK_PROTECTED;

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
        let expected_length = HANDOFF_HEADER_BYTES
            .checked_add(
                region_count
                    .checked_mul(HANDOFF_REGION_BYTES)
                    .ok_or(HandoffError::NonCanonical)?,
            )
            .ok_or(HandoffError::NonCanonical)?;
        if encoded_length != expected_length || input.len() != expected_length {
            return Err(HandoffError::NonCanonical);
        }

        let image_version = read_u64(input, 24);
        let entropy = input[32..64].try_into().unwrap();
        let image_measurement = input[64..128].try_into().unwrap();
        let acpi_address = read_u64(input, 128);
        if acpi_address == 0 {
            return Err(HandoffError::MissingAcpiRoot);
        }
        let acpi_root = acpi_address;
        if input[165..168].iter().any(|byte| *byte != 0) {
            return Err(HandoffError::NonCanonical);
        }
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
        })
    }
}

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
    if regions.is_empty() || regions.len() > MAX_HANDOFF_REGIONS {
        return Err(HandoffError::TooManyRegions);
    }
    let length = HANDOFF_HEADER_BYTES
        .checked_add(
            regions
                .len()
                .checked_mul(HANDOFF_REGION_BYTES)
                .ok_or(HandoffError::NonCanonical)?,
        )
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
    let flags = (secure_boot as u16) * FLAG_SECURE_BOOT
        | (measured_boot as u16) * FLAG_MEASURED_BOOT
        | (rollback_protected as u16) * FLAG_ROLLBACK_PROTECTED;
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
    for (index, region) in regions.iter().copied().enumerate() {
        let offset = HANDOFF_HEADER_BYTES + index * HANDOFF_REGION_BYTES;
        output[offset..offset + 8].copy_from_slice(&region.start().get().to_le_bytes());
        output[offset + 8..offset + 16].copy_from_slice(&region.pages().to_le_bytes());
        output[offset + 16] = region.kind() as u8;
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
        encoded[22..24].copy_from_slice(&KNOWN_FLAGS.to_le_bytes());
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
        for index in 0..3 {
            regions[index] = decode_region(&encoded, index).unwrap();
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
}
