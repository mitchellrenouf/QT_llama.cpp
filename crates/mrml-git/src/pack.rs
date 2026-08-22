use crate::inflate::inflate_zlib_prefix;
use crate::{ObjectId, ObjectKind, Sha1, encode_loose_object};
use core::fmt;
use mrml_runtime::Vector;

const MAX_OBJECTS: usize = 1_000_000;
const MAX_OBJECT: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackObject {
    pub id: ObjectId,
    pub kind: ObjectKind,
    pub contents: Vector<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackError {
    Truncated,
    Signature,
    Version,
    Checksum,
    TooManyObjects,
    TooLarge,
    ObjectType,
    Compression,
    Delta,
    MissingBase,
    TrailingData,
}

struct Decoded {
    offset: usize,
    object: PackObject,
}

enum Base {
    None,
    Offset(usize),
    Id(ObjectId),
}

struct Packed {
    offset: usize,
    object_type: u8,
    base: Base,
    inflated: Vector<u8>,
}

pub fn parse_pack(source: &[u8]) -> Result<Vector<PackObject>, PackError> {
    if source.len() < 12 + 20 {
        return Err(PackError::Truncated);
    }
    let data_end = source.len() - 20;
    if &source[..4] != b"PACK" {
        return Err(PackError::Signature);
    }
    let version = be_u32(source, 4)?;
    if !matches!(version, 2 | 3) {
        return Err(PackError::Version);
    }
    let count = be_u32(source, 8)? as usize;
    if count > MAX_OBJECTS {
        return Err(PackError::TooManyObjects);
    }
    if Sha1::digest(&source[..data_end]) != source[data_end..] {
        return Err(PackError::Checksum);
    }
    let mut cursor = 12usize;
    let mut packed = Vector::new();
    for _ in 0..count {
        let offset = cursor;
        let first = take(source, &mut cursor, data_end)?;
        let object_type = (first >> 4) & 7;
        let mut size = (first & 15) as usize;
        let mut shift = 4;
        let mut header = first;
        while header & 0x80 != 0 {
            header = take(source, &mut cursor, data_end)?;
            if shift >= usize::BITS as usize {
                return Err(PackError::TooLarge);
            }
            size |= ((header & 0x7f) as usize)
                .checked_shl(shift as u32)
                .ok_or(PackError::TooLarge)?;
            shift += 7;
        }
        if size > MAX_OBJECT {
            return Err(PackError::TooLarge);
        }
        let base = match object_type {
            6 => {
                let mut byte = take(source, &mut cursor, data_end)?;
                let mut distance = (byte & 0x7f) as usize;
                while byte & 0x80 != 0 {
                    byte = take(source, &mut cursor, data_end)?;
                    distance = distance
                        .checked_add(1)
                        .and_then(|value| value.checked_shl(7))
                        .and_then(|value| value.checked_add((byte & 0x7f) as usize))
                        .ok_or(PackError::Delta)?;
                }
                let base_offset = offset.checked_sub(distance).ok_or(PackError::Delta)?;
                Base::Offset(base_offset)
            }
            7 => {
                let id = ObjectId(
                    source
                        .get(cursor..cursor + 20)
                        .ok_or(PackError::Truncated)?
                        .try_into()
                        .map_err(|_| PackError::Truncated)?,
                );
                cursor += 20;
                Base::Id(id)
            }
            _ => Base::None,
        };
        let (inflated, consumed) =
            inflate_zlib_prefix(source.get(cursor..data_end).ok_or(PackError::Truncated)?)
                .map_err(|_| PackError::Compression)?;
        cursor = cursor.checked_add(consumed).ok_or(PackError::TooLarge)?;
        if inflated.len() != size {
            return Err(PackError::TooLarge);
        }
        if !matches!(object_type, 1 | 2 | 3 | 4 | 6 | 7) {
            return Err(PackError::ObjectType);
        }
        packed.push(Packed {
            offset,
            object_type,
            base,
            inflated,
        });
    }
    if cursor != data_end {
        return Err(PackError::TrailingData);
    }
    let mut decoded: Vector<Decoded> = Vector::new();
    let mut remaining = packed;
    while !remaining.is_empty() {
        let before = remaining.len();
        let mut deferred = Vector::new();
        for entry in remaining {
            let base = match &entry.base {
                Base::None => None,
                Base::Offset(offset) => decoded.iter().find(|decoded| decoded.offset == *offset),
                Base::Id(id) => decoded.iter().find(|decoded| decoded.object.id == *id),
            };
            if !matches!(&entry.base, Base::None) && base.is_none() {
                deferred.push(entry);
                continue;
            }
            let (kind, contents) = if let Some(base) = base {
                (
                    base.object.kind,
                    apply_delta(&base.object.contents, &entry.inflated)?,
                )
            } else {
                let kind = match entry.object_type {
                    1 => ObjectKind::Commit,
                    2 => ObjectKind::Tree,
                    3 => ObjectKind::Blob,
                    4 => ObjectKind::Tag,
                    _ => return Err(PackError::ObjectType),
                };
                (kind, entry.inflated)
            };
            let (id, _) = encode_loose_object(kind, &contents);
            decoded.push(Decoded {
                offset: entry.offset,
                object: PackObject { id, kind, contents },
            });
        }
        if deferred.len() == before {
            return Err(PackError::MissingBase);
        }
        remaining = deferred;
    }
    Ok(decoded.into_iter().map(|entry| entry.object).collect())
}

pub fn encode_pack(objects: &[PackObject]) -> Result<Vector<u8>, PackError> {
    if objects.len() > MAX_OBJECTS {
        return Err(PackError::TooManyObjects);
    }
    let mut pack = Vector::from(*b"PACK");
    pack.extend(2u32.to_be_bytes());
    pack.extend((objects.len() as u32).to_be_bytes());
    for object in objects {
        if object.contents.len() > MAX_OBJECT {
            return Err(PackError::TooLarge);
        }
        let kind = match object.kind {
            ObjectKind::Commit => 1,
            ObjectKind::Tree => 2,
            ObjectKind::Blob => 3,
            ObjectKind::Tag => 4,
        };
        encode_object_header(kind, object.contents.len(), &mut pack);
        deflate_stored(&object.contents, &mut pack);
    }
    let checksum = Sha1::digest(&pack);
    pack.extend(checksum);
    Ok(pack)
}

fn encode_object_header(kind: u8, mut size: usize, output: &mut Vector<u8>) {
    let mut first = (kind << 4) | (size as u8 & 15);
    size >>= 4;
    if size != 0 {
        first |= 0x80;
    }
    output.push(first);
    while size != 0 {
        let mut byte = (size & 0x7f) as u8;
        size >>= 7;
        if size != 0 {
            byte |= 0x80;
        }
        output.push(byte);
    }
}
fn deflate_stored(source: &[u8], output: &mut Vector<u8>) {
    output.extend([0x78, 0x01]);
    let mut offset = 0usize;
    loop {
        let length = (source.len() - offset).min(65_535);
        let final_block = offset + length == source.len();
        output.push(final_block as u8);
        output.extend((length as u16).to_le_bytes());
        output.extend((!(length as u16)).to_le_bytes());
        output.extend(source[offset..offset + length].iter().copied());
        offset += length;
        if final_block {
            break;
        }
    }
    output.extend(adler32(source).to_be_bytes());
}

fn adler32(source: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in source.chunks(5552) {
        for byte in chunk {
            a += *byte as u32;
            b += a;
        }
        a %= 65_521;
        b %= 65_521;
    }
    b << 16 | a
}

fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vector<u8>, PackError> {
    let mut cursor = 0;
    let base_size = delta_varint(delta, &mut cursor)?;
    if base_size != base.len() {
        return Err(PackError::Delta);
    }
    let result_size = delta_varint(delta, &mut cursor)?;
    if result_size > MAX_OBJECT {
        return Err(PackError::TooLarge);
    }
    let mut output = Vector::new();
    while cursor < delta.len() {
        let opcode = delta[cursor];
        cursor += 1;
        if opcode & 0x80 != 0 {
            let mut offset = 0usize;
            let mut size = 0usize;
            for bit in 0..4 {
                if opcode & (1 << bit) != 0 {
                    offset |= (take_delta(delta, &mut cursor)? as usize) << (bit * 8);
                }
            }
            for bit in 0..3 {
                if opcode & (0x10 << bit) != 0 {
                    size |= (take_delta(delta, &mut cursor)? as usize) << (bit * 8);
                }
            }
            if size == 0 {
                size = 65_536;
            }
            let end = offset.checked_add(size).ok_or(PackError::Delta)?;
            let slice = base.get(offset..end).ok_or(PackError::Delta)?;
            if output
                .len()
                .checked_add(size)
                .is_none_or(|length| length > result_size)
            {
                return Err(PackError::Delta);
            }
            output.extend(slice.iter().copied());
        } else if opcode != 0 {
            let size = opcode as usize;
            let end = cursor.checked_add(size).ok_or(PackError::Delta)?;
            let slice = delta.get(cursor..end).ok_or(PackError::Truncated)?;
            if output
                .len()
                .checked_add(size)
                .is_none_or(|length| length > result_size)
            {
                return Err(PackError::Delta);
            }
            output.extend(slice.iter().copied());
            cursor = end;
        } else {
            return Err(PackError::Delta);
        }
    }
    if output.len() != result_size {
        return Err(PackError::Delta);
    }
    Ok(output)
}

fn delta_varint(source: &[u8], cursor: &mut usize) -> Result<usize, PackError> {
    let mut value = 0usize;
    let mut shift = 0;
    loop {
        let byte = take_delta(source, cursor)?;
        if shift >= usize::BITS as usize {
            return Err(PackError::TooLarge);
        }
        value |= ((byte & 0x7f) as usize)
            .checked_shl(shift as u32)
            .ok_or(PackError::TooLarge)?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}
fn take_delta(source: &[u8], cursor: &mut usize) -> Result<u8, PackError> {
    let value = *source.get(*cursor).ok_or(PackError::Truncated)?;
    *cursor += 1;
    Ok(value)
}
fn take(source: &[u8], cursor: &mut usize, end: usize) -> Result<u8, PackError> {
    if *cursor >= end {
        return Err(PackError::Truncated);
    }
    let value = source[*cursor];
    *cursor += 1;
    Ok(value)
}
fn be_u32(source: &[u8], offset: usize) -> Result<u32, PackError> {
    Ok(u32::from_be_bytes(
        source
            .get(offset..offset + 4)
            .ok_or(PackError::Truncated)?
            .try_into()
            .map_err(|_| PackError::Truncated)?,
    ))
}
impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Truncated => "truncated pack",
            Self::Signature => "invalid pack signature",
            Self::Version => "unsupported pack version",
            Self::Checksum => "pack checksum mismatch",
            Self::TooManyObjects => "too many packed objects",
            Self::TooLarge => "packed object exceeds limit",
            Self::ObjectType => "unsupported packed object type",
            Self::Compression => "invalid packed DEFLATE stream",
            Self::Delta => "invalid pack delta",
            Self::MissingBase => "thin pack or missing delta base",
            Self::TrailingData => "unexpected trailing pack data",
        })
    }
}
impl core::error::Error for PackError {}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw_stored(data: &[u8]) -> Vector<u8> {
        let mut out = Vector::new();
        deflate_stored(data, &mut out);
        out
    }
    fn pack_blob(data: &[u8]) -> Vector<u8> {
        let mut pack = Vector::from(*b"PACK");
        pack.extend(2u32.to_be_bytes());
        pack.extend(1u32.to_be_bytes());
        pack.push(0x30 | data.len() as u8);
        pack.extend(raw_stored(data));
        let sum = Sha1::digest(&pack);
        pack.extend(sum);
        pack
    }
    #[test]
    fn parses_and_authenticates_blob_pack() {
        let objects = parse_pack(&pack_blob(b"native")).unwrap();
        assert_eq!(objects[0].kind, ObjectKind::Blob);
        assert_eq!(&*objects[0].contents, b"native");
        assert_eq!(objects[0].id, ObjectId::blob(b"native"));
    }
    #[test]
    fn rejects_pack_checksum_tampering() {
        let mut pack = pack_blob(b"native");
        pack[12] ^= 1;
        assert_eq!(parse_pack(&pack), Err(PackError::Checksum));
    }
    #[test]
    fn applies_copy_and_insert_delta() {
        let delta = [6, 7, 0x90, 3, 4, b'X', b'Y', b'Z', b'!'];
        assert_eq!(&*apply_delta(b"native", &delta).unwrap(), b"natXYZ!");
    }
    #[test]
    fn resolves_ofs_delta_in_pack() {
        let base = b"native";
        let delta = [6, 7, 0x90, 3, 4, b'X', b'Y', b'Z', b'!'];
        let mut pack = Vector::from(*b"PACK");
        pack.extend(2u32.to_be_bytes());
        pack.extend(2u32.to_be_bytes());
        let base_offset = pack.len();
        pack.push(0x36);
        pack.extend(raw_stored(base));
        let delta_offset = pack.len();
        pack.push(0x69);
        pack.push((delta_offset - base_offset) as u8);
        pack.extend(raw_stored(&delta));
        let sum = Sha1::digest(&pack);
        pack.extend(sum);
        let objects = parse_pack(&pack).unwrap();
        assert_eq!(&*objects[1].contents, b"natXYZ!");
        assert_eq!(objects[1].id, ObjectId::blob(b"natXYZ!"));
    }
    #[test]
    fn resolves_ref_delta_whose_base_appears_later() {
        let base = b"native";
        let delta = [6, 7, 0x90, 3, 4, b'X', b'Y', b'Z', b'!'];
        let mut pack = Vector::from(*b"PACK");
        pack.extend(2u32.to_be_bytes());
        pack.extend(2u32.to_be_bytes());
        encode_object_header(7, delta.len(), &mut pack);
        pack.extend(ObjectId::blob(base).0);
        pack.extend(raw_stored(&delta));
        encode_object_header(3, base.len(), &mut pack);
        pack.extend(raw_stored(base));
        let sum = Sha1::digest(&pack);
        pack.extend(sum);
        let objects = parse_pack(&pack).unwrap();
        assert!(objects.iter().any(|object| &*object.contents == base));
        assert!(objects.iter().any(
            |object| &*object.contents == b"natXYZ!" && object.id == ObjectId::blob(b"natXYZ!")
        ));
    }
    #[test]
    fn generated_pack_round_trips_large_and_empty_objects() {
        let inputs = Vector::from([
            PackObject {
                id: ObjectId::blob(b""),
                kind: ObjectKind::Blob,
                contents: Vector::new(),
            },
            PackObject {
                id: ObjectId::blob(&[7; 70_000]),
                kind: ObjectKind::Blob,
                contents: Vector::from([7; 70_000]),
            },
        ]);
        let encoded = encode_pack(&inputs).unwrap();
        let decoded = parse_pack(&encoded).unwrap();
        assert_eq!(decoded, inputs);
    }
}
