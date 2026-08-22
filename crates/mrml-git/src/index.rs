use core::fmt;
use mrml_runtime::{Text, Vector};

use crate::{ObjectId, Sha1};

const HEADER_LENGTH: usize = 12;
const CHECKSUM_LENGTH: usize = 20;
const FIXED_ENTRY_LENGTH: usize = 62;
const MAX_ENTRIES: usize = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexEntry {
    pub path: Text,
    pub id: ObjectId,
    pub mode: u32,
    pub size: u32,
    pub stage: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Index {
    pub version: u32,
    pub entries: Vector<IndexEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexError {
    Truncated,
    InvalidSignature,
    UnsupportedVersion,
    TooManyEntries,
    InvalidChecksum,
    InvalidPath,
    InvalidPadding,
    InvalidExtension,
    RequiredExtension,
}

impl Index {
    pub fn empty() -> Self {
        Self { version: 2, entries: Vector::new() }
    }

    pub fn parse(source: &[u8]) -> Result<Self, IndexError> {
        if source.len() < HEADER_LENGTH + CHECKSUM_LENGTH {
            return Err(IndexError::Truncated);
        }
        let body_length = source.len() - CHECKSUM_LENGTH;
        if Sha1::digest(&source[..body_length]) != source[body_length..] {
            return Err(IndexError::InvalidChecksum);
        }
        if &source[..4] != b"DIRC" {
            return Err(IndexError::InvalidSignature);
        }
        let version = be_u32(source, 4)?;
        if !matches!(version, 2 | 3) {
            return Err(IndexError::UnsupportedVersion);
        }
        let count = be_u32(source, 8)? as usize;
        if count > MAX_ENTRIES {
            return Err(IndexError::TooManyEntries);
        }

        let mut cursor = HEADER_LENGTH;
        let mut entries = Vector::new();
        for _ in 0..count {
            let start = cursor;
            if cursor.checked_add(FIXED_ENTRY_LENGTH).is_none_or(|end| end > body_length) {
                return Err(IndexError::Truncated);
            }
            let mode = be_u32(source, cursor + 24)?;
            let size = be_u32(source, cursor + 36)?;
            let id = ObjectId(source[cursor + 40..cursor + 60].try_into().map_err(|_| IndexError::Truncated)?);
            let flags = be_u16(source, cursor + 60)?;
            cursor += FIXED_ENTRY_LENGTH;
            if flags & 0x4000 != 0 {
                if version != 3 || cursor.checked_add(2).is_none_or(|end| end > body_length) {
                    return Err(IndexError::Truncated);
                }
                cursor += 2;
            }
            let nul = source[cursor..body_length]
                .iter()
                .position(|byte| *byte == 0)
                .map(|offset| cursor + offset)
                .ok_or(IndexError::Truncated)?;
            let path = core::str::from_utf8(&source[cursor..nul]).map_err(|_| IndexError::InvalidPath)?;
            validate_path(path)?;
            entries.push(IndexEntry {
                path: path.into(), id, mode, size, stage: ((flags >> 12) & 3) as u8,
            });
            let unpadded = nul.checked_add(1).ok_or(IndexError::Truncated)? - start;
            let padded = unpadded.checked_add(7).ok_or(IndexError::Truncated)? & !7;
            cursor = start.checked_add(padded).ok_or(IndexError::Truncated)?;
            if cursor > body_length || source[nul + 1..cursor].iter().any(|byte| *byte != 0) {
                return Err(IndexError::InvalidPadding);
            }
        }

        while cursor < body_length {
            if cursor.checked_add(8).is_none_or(|end| end > body_length) {
                return Err(IndexError::InvalidExtension);
            }
            let signature = &source[cursor..cursor + 4];
            if signature[0].is_ascii_lowercase() {
                return Err(IndexError::RequiredExtension);
            }
            let size = be_u32(source, cursor + 4)? as usize;
            cursor = cursor.checked_add(8).and_then(|value| value.checked_add(size)).ok_or(IndexError::InvalidExtension)?;
            if cursor > body_length {
                return Err(IndexError::InvalidExtension);
            }
        }
        Ok(Self { version, entries })
    }

    pub fn entry(&self, path: &str) -> Option<&IndexEntry> {
        self.entries.iter().find(|entry| entry.stage == 0 && entry.path == path)
    }

    pub fn upsert(&mut self, entry: IndexEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|item| item.path == entry.path && item.stage == entry.stage) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
        self.entries.sort_unstable_by(|left, right| left.path.as_bytes().cmp(right.path.as_bytes()).then(left.stage.cmp(&right.stage)));
    }

    pub fn remove(&mut self, path: &str) -> bool {
        let before = self.entries.len();
        let mut index = 0;
        while index < self.entries.len() {
            if self.entries[index].path == path { self.entries.remove(index); } else { index += 1; }
        }
        self.entries.len() != before
    }

    pub fn encode(&self) -> Result<Vector<u8>, IndexError> {
        if self.entries.len() > MAX_ENTRIES { return Err(IndexError::TooManyEntries); }
        let mut output = Vector::new();
        output.extend(*b"DIRC"); output.extend(2u32.to_be_bytes()); output.extend((self.entries.len() as u32).to_be_bytes());
        for entry in &self.entries {
            validate_path(&entry.path)?;
            let start = output.len();
            output.extend([0u8; 24]);
            output.extend(entry.mode.to_be_bytes());
            output.extend([0u8; 8]);
            output.extend(entry.size.to_be_bytes());
            output.extend(entry.id.0);
            let path_length = entry.path.len().min(0x0fff) as u16;
            output.extend((path_length | ((entry.stage as u16 & 3) << 12)).to_be_bytes());
            output.extend(entry.path.as_bytes().iter().copied()); output.push(0);
            while (output.len() - start) % 8 != 0 { output.push(0); }
        }
        let checksum = Sha1::digest(&output); output.extend(checksum); Ok(output)
    }
}

fn validate_path(path: &str) -> Result<(), IndexError> {
    if path.is_empty() || path.starts_with('/') || path.starts_with('\\') || path.contains('\\')
        || path.split('/').any(|part| part.is_empty() || matches!(part, "." | ".."))
        || path.chars().any(char::is_control)
    {
        Err(IndexError::InvalidPath)
    } else {
        Ok(())
    }
}

fn be_u32(source: &[u8], offset: usize) -> Result<u32, IndexError> {
    Ok(u32::from_be_bytes(source.get(offset..offset + 4).ok_or(IndexError::Truncated)?.try_into().map_err(|_| IndexError::Truncated)?))
}

fn be_u16(source: &[u8], offset: usize) -> Result<u16, IndexError> {
    Ok(u16::from_be_bytes(source.get(offset..offset + 2).ok_or(IndexError::Truncated)?.try_into().map_err(|_| IndexError::Truncated)?))
}

impl fmt::Display for IndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Truncated => "truncated Git index", Self::InvalidSignature => "invalid Git index signature",
            Self::UnsupportedVersion => "unsupported Git index version", Self::TooManyEntries => "Git index has too many entries",
            Self::InvalidChecksum => "Git index checksum mismatch", Self::InvalidPath => "unsafe or invalid Git index path",
            Self::InvalidPadding => "invalid Git index entry padding", Self::InvalidExtension => "invalid Git index extension",
            Self::RequiredExtension => "unsupported required Git index extension",
        })
    }
}

impl core::error::Error for IndexError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_entry_index() -> Vector<u8> {
        let path = b"src/lib.rs";
        let mut bytes = Vector::from(*b"DIRC");
        bytes.extend(2u32.to_be_bytes());
        bytes.extend(1u32.to_be_bytes());
        let mut fixed = [0u8; FIXED_ENTRY_LENGTH];
        fixed[24..28].copy_from_slice(&0o100644u32.to_be_bytes());
        fixed[36..40].copy_from_slice(&3u32.to_be_bytes());
        fixed[40..60].copy_from_slice(&ObjectId::blob(b"abc").0);
        fixed[60..62].copy_from_slice(&(path.len() as u16).to_be_bytes());
        bytes.extend(fixed);
        bytes.extend(*path);
        bytes.push(0);
        while (bytes.len() - HEADER_LENGTH) % 8 != 0 { bytes.push(0); }
        let checksum = Sha1::digest(&bytes);
        bytes.extend(checksum);
        bytes
    }

    #[test]
    fn parses_and_authenticates_v2_entry() {
        let index = Index::parse(&one_entry_index()).unwrap();
        let entry = index.entry("src/lib.rs").unwrap();
        assert_eq!(entry.id, ObjectId::blob(b"abc"));
        assert_eq!(entry.mode, 0o100644);
    }

    #[test]
    fn rejects_checksum_tampering() {
        let mut bytes = one_entry_index();
        bytes[20] ^= 1;
        assert_eq!(Index::parse(&bytes), Err(IndexError::InvalidChecksum));
    }

    #[test]
    fn encoded_index_round_trips() {
        let parsed = Index::parse(&one_entry_index()).unwrap();
        assert_eq!(Index::parse(&parsed.encode().unwrap()).unwrap(), parsed);
    }
}
