use core::fmt;
use mrml_runtime::{Text, Vector};

use crate::inflate::inflate_zlib;
use crate::{ObjectId, Sha1};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectKind {
    Blob,
    Tree,
    Commit,
    Tag,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Object {
    pub kind: ObjectKind,
    pub contents: Vector<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Commit {
    pub tree: ObjectId,
    pub parents: Vector<ObjectId>,
    pub author: Text,
    pub message: Text,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectError {
    Truncated,
    UnsupportedCompression,
    InvalidBlock,
    InvalidChecksum,
    InvalidHeader,
    SizeMismatch,
}

impl ObjectKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Blob => "blob",
            Self::Tree => "tree",
            Self::Commit => "commit",
            Self::Tag => "tag",
        }
    }
}

/// Produces a Git loose object using a standards-compatible zlib stream whose
/// DEFLATE blocks are deliberately uncompressed. This keeps the implementation
/// small and auditable while remaining readable by other Git implementations.
pub fn encode_loose_object(kind: ObjectKind, contents: &[u8]) -> (ObjectId, Vector<u8>) {
    let header = mrml_runtime::mrml_format!("{} {}\0", kind.name(), contents.len());
    let mut hash = Sha1::new();
    hash.update(header.as_bytes());
    hash.update(contents);
    let id = ObjectId(hash.finalize());
    let total = header.len() + contents.len();
    let mut plain = Vector::new();
    plain.extend(header.as_bytes().iter().copied());
    plain.extend(contents.iter().copied());
    let mut output = Vector::new();
    output.extend([0x78, 0x01]);
    let mut offset = 0;
    while offset < total || (total == 0 && offset == 0) {
        let length = (total - offset).min(65_535);
        let final_block = offset + length == total;
        output.push(if final_block { 1 } else { 0 });
        output.extend((length as u16).to_le_bytes());
        output.extend((!(length as u16)).to_le_bytes());
        output.extend(plain[offset..offset + length].iter().copied());
        offset += length;
        if final_block {
            break;
        }
    }
    output.extend(adler32(&plain).to_be_bytes());
    (id, output)
}

pub fn decode_loose_object(source: &[u8]) -> Result<Object, ObjectError> {
    let plain = inflate_zlib(source).map_err(|_| ObjectError::InvalidBlock)?;
    let nul = plain
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(ObjectError::InvalidHeader)?;
    let header = core::str::from_utf8(&plain[..nul]).map_err(|_| ObjectError::InvalidHeader)?;
    let (kind, size) = header.split_once(' ').ok_or(ObjectError::InvalidHeader)?;
    let kind = match kind {
        "blob" => ObjectKind::Blob,
        "tree" => ObjectKind::Tree,
        "commit" => ObjectKind::Commit,
        "tag" => ObjectKind::Tag,
        _ => return Err(ObjectError::InvalidHeader),
    };
    let size: usize = size.parse().map_err(|_| ObjectError::InvalidHeader)?;
    if size != plain.len() - nul - 1 {
        return Err(ObjectError::SizeMismatch);
    }
    let mut contents = Vector::new();
    contents.extend(plain[nul + 1..].iter().copied());
    Ok(Object { kind, contents })
}

impl fmt::Display for ObjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Truncated => "truncated loose object",
            Self::UnsupportedCompression => "unsupported loose-object compression",
            Self::InvalidBlock => "invalid DEFLATE stored block",
            Self::InvalidChecksum => "loose-object checksum mismatch",
            Self::InvalidHeader => "invalid Git object header",
            Self::SizeMismatch => "Git object size mismatch",
        })
    }
}
impl core::error::Error for ObjectError {}

impl Commit {
    pub fn parse(contents: &[u8]) -> Result<Self, ObjectError> {
        let text = core::str::from_utf8(contents).map_err(|_| ObjectError::InvalidHeader)?;
        let (headers, message) = text.split_once("\n\n").ok_or(ObjectError::InvalidHeader)?;
        let mut tree = None;
        let mut parents = Vector::new();
        let mut author = None;
        for line in headers.lines() {
            if let Some(value) = line.strip_prefix("tree ") { tree = ObjectId::parse(value); }
            else if let Some(value) = line.strip_prefix("parent ") { parents.push(ObjectId::parse(value).ok_or(ObjectError::InvalidHeader)?); }
            else if let Some(value) = line.strip_prefix("author ") { author = Some(Text::from(value)); }
        }
        Ok(Self { tree: tree.ok_or(ObjectError::InvalidHeader)?, parents, author: author.ok_or(ObjectError::InvalidHeader)?, message: message.into() })
    }
}

fn adler32(bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in bytes.chunks(5_552) {
        for byte in chunk {
            a += *byte as u32;
            b += a;
        }
        a %= MODULUS;
        b %= MODULUS;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loose_blob_has_git_id_and_valid_stored_stream() {
        let (id, bytes) = encode_loose_object(ObjectKind::Blob, b"");
        assert_eq!(id.to_hex(), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
        assert_eq!(&bytes[..7], &[0x78, 0x01, 1, 7, 0, 0xf8, 0xff]);
        assert_eq!(&bytes[7..14], b"blob 0\0");
    }

    #[test]
    fn splits_large_objects_into_deflate_blocks() {
        let (_, bytes) = encode_loose_object(ObjectKind::Blob, &[42; 70_000]);
        assert_eq!(bytes[2], 0);
        let second = 2 + 5 + 65_535;
        assert_eq!(bytes[second], 1);
    }

    #[test]
    fn native_loose_objects_round_trip() {
        let (_, encoded) = encode_loose_object(ObjectKind::Commit, b"tree deadbeef\n\nmessage\n");
        assert_eq!(
            decode_loose_object(&encoded).unwrap(),
            Object {
                kind: ObjectKind::Commit,
                contents: Vector::from(*b"tree deadbeef\n\nmessage\n")
            }
        );
    }

    #[test]
    fn parses_commit_graph_metadata() {
        let tree = "1111111111111111111111111111111111111111";
        let parent = "2222222222222222222222222222222222222222";
        let text = mrml_runtime::mrml_format!("tree {tree}\nparent {parent}\nauthor MRML <mrml@example.invalid> 1 +0000\ncommitter MRML <mrml@example.invalid> 1 +0000\n\nmessage\n");
        let commit = Commit::parse(text.as_bytes()).unwrap();
        assert_eq!(commit.tree, ObjectId::parse(tree).unwrap());
        assert_eq!(commit.parents[0], ObjectId::parse(parent).unwrap());
        assert_eq!(commit.message, "message\n");
    }
}
