use mrml_runtime::Vector;

use crate::{ObjectId, Sha1};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectKind { Blob, Tree, Commit, Tag }

impl ObjectKind {
    pub const fn name(self) -> &'static str {
        match self { Self::Blob => "blob", Self::Tree => "tree", Self::Commit => "commit", Self::Tag => "tag" }
    }
}

/// Produces a Git loose object using a standards-compatible zlib stream whose
/// DEFLATE blocks are deliberately uncompressed. This keeps the implementation
/// small and auditable while remaining readable by other Git implementations.
pub fn encode_loose_object(kind: ObjectKind, contents: &[u8]) -> (ObjectId, Vector<u8>) {
    let header = mrml_runtime::mrml_format!("{} {}\0", kind.name(), contents.len());
    let mut hash = Sha1::new(); hash.update(header.as_bytes()); hash.update(contents);
    let id = ObjectId(hash.finalize());
    let total = header.len() + contents.len();
    let mut plain = Vector::new(); plain.extend(header.as_bytes().iter().copied()); plain.extend(contents.iter().copied());
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
        if final_block { break; }
    }
    output.extend(adler32(&plain).to_be_bytes());
    (id, output)
}

fn adler32(bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in bytes.chunks(5_552) {
        for byte in chunk { a += *byte as u32; b += a; }
        a %= MODULUS; b %= MODULUS;
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
}
