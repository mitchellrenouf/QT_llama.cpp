use mrml_crypto::Sha256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthenticodeError {
    Truncated,
    InvalidPe,
    TooManySections,
    InvalidCertificateTable,
    InvalidSection,
    Overflow,
}

pub fn sha256(image: &[u8]) -> Result<[u8; 32], AuthenticodeError> {
    if read_u16(image, 0)? != 0x5a4d {
        return Err(AuthenticodeError::InvalidPe);
    }
    let pe = read_u32(image, 0x3c)? as usize;
    if read_u32(image, pe)? != 0x0000_4550 || read_u16(image, pe + 4)? != 0x8664 {
        return Err(AuthenticodeError::InvalidPe);
    }
    let section_count = read_u16(image, pe + 6)? as usize;
    if section_count == 0 || section_count > 32 {
        return Err(AuthenticodeError::TooManySections);
    }
    let optional_size = read_u16(image, pe + 20)? as usize;
    let optional = pe.checked_add(24).ok_or(AuthenticodeError::Overflow)?;
    if optional_size < 152 || read_u16(image, optional)? != 0x020b {
        return Err(AuthenticodeError::InvalidPe);
    }
    let checksum = optional + 64;
    let security_directory = optional + 144;
    let headers_size = read_u32(image, optional + 60)? as usize;
    if headers_size > image.len() || security_directory + 8 > headers_size {
        return Err(AuthenticodeError::InvalidPe);
    }
    let certificate_offset = read_u32(image, security_directory)? as usize;
    let certificate_size = read_u32(image, security_directory + 4)? as usize;
    let content_end = match (certificate_offset, certificate_size) {
        (0, 0) => image.len(),
        (offset, size)
            if offset % 8 == 0 && size >= 8 && offset.checked_add(size) == Some(image.len()) =>
        {
            offset
        }
        _ => return Err(AuthenticodeError::InvalidCertificateTable),
    };

    let section_table = optional
        .checked_add(optional_size)
        .ok_or(AuthenticodeError::Overflow)?;
    let mut sections = [(0usize, 0usize); 32];
    for index in 0..section_count {
        let at = section_table
            .checked_add(index.checked_mul(40).ok_or(AuthenticodeError::Overflow)?)
            .ok_or(AuthenticodeError::Overflow)?;
        let size = read_u32(image, at + 16)? as usize;
        let start = read_u32(image, at + 20)? as usize;
        let end = start.checked_add(size).ok_or(AuthenticodeError::Overflow)?;
        if size != 0 && (start < headers_size || end > content_end) {
            return Err(AuthenticodeError::InvalidSection);
        }
        sections[index] = (start, size);
    }
    sections[..section_count].sort_unstable_by_key(|section| section.0);
    let mut previous_end = headers_size;
    for &(start, size) in &sections[..section_count] {
        if size != 0 && start < previous_end {
            return Err(AuthenticodeError::InvalidSection);
        }
        if size != 0 {
            previous_end = start + size;
        }
    }

    let mut hash = Sha256::new();
    hash.update(&image[..checksum]);
    hash.update(&image[checksum + 4..security_directory]);
    hash.update(&image[security_directory + 8..headers_size]);
    let mut hashed = headers_size;
    for &(start, size) in &sections[..section_count] {
        if size != 0 {
            hash.update(&image[start..start + size]);
            hashed = hashed
                .checked_add(size)
                .ok_or(AuthenticodeError::Overflow)?;
        }
    }
    if hashed > content_end {
        return Err(AuthenticodeError::InvalidSection);
    }
    hash.update(&image[hashed..content_end]);
    Ok(hash.finalize())
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, AuthenticodeError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .ok_or(AuthenticodeError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, AuthenticodeError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .ok_or(AuthenticodeError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> [u8; 1024] {
        let mut image = [0u8; 1024];
        image[..2].copy_from_slice(&0x5a4d_u16.to_le_bytes());
        image[0x3c..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        image[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        image[0x84..0x86].copy_from_slice(&0x8664_u16.to_le_bytes());
        image[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        image[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        image[0x98..0x9a].copy_from_slice(&0x020b_u16.to_le_bytes());
        image[0xd4..0xd8].copy_from_slice(&512u32.to_le_bytes());
        image[0x198..0x19c].copy_from_slice(&512u32.to_le_bytes());
        image[0x19c..0x1a0].copy_from_slice(&512u32.to_le_bytes());
        image[512..].fill(0x5a);
        image
    }

    #[test]
    fn excludes_checksum_and_certificate_directory_fields() {
        let mut original = image().to_vec();
        original.extend_from_slice(&[8, 0, 0, 0, 0, 2, 2, 0]);
        original[0x128..0x12c].copy_from_slice(&1024u32.to_le_bytes());
        original[0x12c..0x130].copy_from_slice(&8u32.to_le_bytes());
        let expected = sha256(&original).unwrap();
        let mut changed = original.clone();
        changed[0xd8..0xdc].copy_from_slice(&0xfeed_beef_u32.to_le_bytes());
        changed[1028..].fill(0xa5);
        assert_eq!(sha256(&changed), Ok(expected));
        changed[600] ^= 1;
        assert_ne!(sha256(&changed), Ok(expected));
    }
}
