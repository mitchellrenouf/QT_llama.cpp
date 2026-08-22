use core::fmt;
use mrml_runtime::Vector;
use crate::{ObjectId, ObjectKind, Sha1, encode_loose_object};
use crate::inflate::inflate_raw;

const MAX_OBJECTS: usize = 1_000_000;
const MAX_OBJECT: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackObject { pub id: ObjectId, pub kind: ObjectKind, pub contents: Vector<u8> }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackError { Truncated, Signature, Version, Checksum, TooManyObjects, TooLarge, ObjectType, Compression, Delta, MissingBase, TrailingData }

struct Decoded { offset: usize, object: PackObject }

pub fn parse_pack(source: &[u8]) -> Result<Vector<PackObject>, PackError> {
    if source.len() < 12 + 20 { return Err(PackError::Truncated); }
    let data_end = source.len() - 20;
    if &source[..4] != b"PACK" { return Err(PackError::Signature); }
    let version = be_u32(source, 4)?; if !matches!(version, 2 | 3) { return Err(PackError::Version); }
    let count = be_u32(source, 8)? as usize; if count > MAX_OBJECTS { return Err(PackError::TooManyObjects); }
    if Sha1::digest(&source[..data_end]) != source[data_end..] { return Err(PackError::Checksum); }
    let mut cursor = 12usize; let mut decoded: Vector<Decoded> = Vector::new();
    for _ in 0..count {
        let offset = cursor;
        let first = take(source, &mut cursor, data_end)?;
        let object_type = (first >> 4) & 7;
        let mut size = (first & 15) as usize; let mut shift = 4; let mut header = first;
        while header & 0x80 != 0 { header = take(source, &mut cursor, data_end)?; if shift >= usize::BITS as usize { return Err(PackError::TooLarge); } size |= ((header & 0x7f) as usize).checked_shl(shift as u32).ok_or(PackError::TooLarge)?; shift += 7; }
        if size > MAX_OBJECT { return Err(PackError::TooLarge); }
        let base = match object_type {
            6 => {
                let mut byte = take(source, &mut cursor, data_end)?; let mut distance = (byte & 0x7f) as usize;
                while byte & 0x80 != 0 { byte = take(source, &mut cursor, data_end)?; distance = distance.checked_add(1).and_then(|value| value.checked_shl(7)).and_then(|value| value.checked_add((byte & 0x7f) as usize)).ok_or(PackError::Delta)?; }
                let base_offset = offset.checked_sub(distance).ok_or(PackError::Delta)?;
                Some(decoded.iter().find(|entry| entry.offset == base_offset).map(|entry| entry.object.clone()).ok_or(PackError::MissingBase)?)
            }
            7 => { let id = ObjectId(source.get(cursor..cursor + 20).ok_or(PackError::Truncated)?.try_into().map_err(|_| PackError::Truncated)?); cursor += 20; Some(decoded.iter().find(|entry| entry.object.id == id).map(|entry| entry.object.clone()).ok_or(PackError::MissingBase)?) }
            _ => None,
        };
        let (inflated, consumed) = inflate_raw(source.get(cursor..data_end).ok_or(PackError::Truncated)?).map_err(|_| PackError::Compression)?;
        cursor = cursor.checked_add(consumed).ok_or(PackError::TooLarge)?;
        if inflated.len() != size { return Err(PackError::TooLarge); }
        let (kind, contents) = match object_type {
            1 => (ObjectKind::Commit, inflated), 2 => (ObjectKind::Tree, inflated), 3 => (ObjectKind::Blob, inflated), 4 => (ObjectKind::Tag, inflated),
            6 | 7 => { let base = base.ok_or(PackError::MissingBase)?; (base.kind, apply_delta(&base.contents, &inflated)?) }
            _ => return Err(PackError::ObjectType),
        };
        let (id, _) = encode_loose_object(kind, &contents);
        decoded.push(Decoded { offset, object: PackObject { id, kind, contents } });
    }
    if cursor != data_end { return Err(PackError::TrailingData); }
    Ok(decoded.into_iter().map(|entry| entry.object).collect())
}

fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vector<u8>, PackError> {
    let mut cursor = 0; let base_size = delta_varint(delta, &mut cursor)?; if base_size != base.len() { return Err(PackError::Delta); }
    let result_size = delta_varint(delta, &mut cursor)?; if result_size > MAX_OBJECT { return Err(PackError::TooLarge); }
    let mut output = Vector::new();
    while cursor < delta.len() {
        let opcode = delta[cursor]; cursor += 1;
        if opcode & 0x80 != 0 {
            let mut offset = 0usize; let mut size = 0usize;
            for bit in 0..4 { if opcode & (1 << bit) != 0 { offset |= (take_delta(delta, &mut cursor)? as usize) << (bit * 8); } }
            for bit in 0..3 { if opcode & (0x10 << bit) != 0 { size |= (take_delta(delta, &mut cursor)? as usize) << (bit * 8); } }
            if size == 0 { size = 65_536; }
            let end = offset.checked_add(size).ok_or(PackError::Delta)?; let slice = base.get(offset..end).ok_or(PackError::Delta)?;
            if output.len().checked_add(size).is_none_or(|length| length > result_size) { return Err(PackError::Delta); } output.extend(slice.iter().copied());
        } else if opcode != 0 {
            let size = opcode as usize; let end = cursor.checked_add(size).ok_or(PackError::Delta)?; let slice = delta.get(cursor..end).ok_or(PackError::Truncated)?;
            if output.len().checked_add(size).is_none_or(|length| length > result_size) { return Err(PackError::Delta); } output.extend(slice.iter().copied()); cursor = end;
        } else { return Err(PackError::Delta); }
    }
    if output.len() != result_size { return Err(PackError::Delta); } Ok(output)
}

fn delta_varint(source: &[u8], cursor: &mut usize) -> Result<usize, PackError> { let mut value=0usize;let mut shift=0;loop{let byte=take_delta(source,cursor)?;if shift>=usize::BITS as usize{return Err(PackError::TooLarge);}value|=((byte&0x7f)as usize).checked_shl(shift as u32).ok_or(PackError::TooLarge)?;if byte&0x80==0{return Ok(value);}shift+=7;} }
fn take_delta(source:&[u8],cursor:&mut usize)->Result<u8,PackError>{let value=*source.get(*cursor).ok_or(PackError::Truncated)?;*cursor+=1;Ok(value)}
fn take(source:&[u8],cursor:&mut usize,end:usize)->Result<u8,PackError>{if *cursor>=end{return Err(PackError::Truncated);}let value=source[*cursor];*cursor+=1;Ok(value)}
fn be_u32(source:&[u8],offset:usize)->Result<u32,PackError>{Ok(u32::from_be_bytes(source.get(offset..offset+4).ok_or(PackError::Truncated)?.try_into().map_err(|_|PackError::Truncated)?))}
impl fmt::Display for PackError { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str(match self{Self::Truncated=>"truncated pack",Self::Signature=>"invalid pack signature",Self::Version=>"unsupported pack version",Self::Checksum=>"pack checksum mismatch",Self::TooManyObjects=>"too many packed objects",Self::TooLarge=>"packed object exceeds limit",Self::ObjectType=>"unsupported packed object type",Self::Compression=>"invalid packed DEFLATE stream",Self::Delta=>"invalid pack delta",Self::MissingBase=>"thin pack or missing delta base",Self::TrailingData=>"unexpected trailing pack data"})}}
impl core::error::Error for PackError{}

#[cfg(test)] mod tests { use super::*;
 fn raw_stored(data:&[u8])->Vector<u8>{let mut out=Vector::new();out.push(1);out.extend((data.len()as u16).to_le_bytes());out.extend((!(data.len()as u16)).to_le_bytes());out.extend(data.iter().copied());out}
 fn pack_blob(data:&[u8])->Vector<u8>{let mut pack=Vector::from(*b"PACK");pack.extend(2u32.to_be_bytes());pack.extend(1u32.to_be_bytes());pack.push(0x30|data.len()as u8);pack.extend(raw_stored(data));let sum=Sha1::digest(&pack);pack.extend(sum);pack}
 #[test] fn parses_and_authenticates_blob_pack(){let objects=parse_pack(&pack_blob(b"native")).unwrap();assert_eq!(objects[0].kind,ObjectKind::Blob);assert_eq!(&*objects[0].contents,b"native");assert_eq!(objects[0].id,ObjectId::blob(b"native"));}
 #[test] fn rejects_pack_checksum_tampering(){let mut pack=pack_blob(b"native");pack[12]^=1;assert_eq!(parse_pack(&pack),Err(PackError::Checksum));}
 #[test] fn applies_copy_and_insert_delta(){let delta=[6,7,0x90,3,4,b'X',b'Y',b'Z',b'!'];assert_eq!(&*apply_delta(b"native",&delta).unwrap(),b"natXYZ!");}
 #[test] fn resolves_ofs_delta_in_pack(){let base=b"native";let delta=[6,7,0x90,3,4,b'X',b'Y',b'Z',b'!'];let mut pack=Vector::from(*b"PACK");pack.extend(2u32.to_be_bytes());pack.extend(2u32.to_be_bytes());let base_offset=pack.len();pack.push(0x36);pack.extend(raw_stored(base));let delta_offset=pack.len();pack.push(0x69);pack.push((delta_offset-base_offset)as u8);pack.extend(raw_stored(&delta));let sum=Sha1::digest(&pack);pack.extend(sum);let objects=parse_pack(&pack).unwrap();assert_eq!(&*objects[1].contents,b"natXYZ!");assert_eq!(objects[1].id,ObjectId::blob(b"natXYZ!"));}
}
