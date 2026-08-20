#![no_std]

use core::fmt;
use mrml_runtime::Vector;

pub const FRAME_MAGIC: u32 = 0xfd2f_b528;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Error {
    Truncated,
    InvalidMagic(u32),
    ReservedHeaderBit,
    DictionaryRequired(u32),
    ReservedBlock,
    UnsupportedCompressedBlock,
    ContentSizeMismatch,
    TrailingData,
    Allocation,
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => output.write_str("truncated Zstandard frame"),
            Self::InvalidMagic(magic) => write!(output, "invalid Zstandard magic 0x{magic:08x}"),
            Self::ReservedHeaderBit => output.write_str("reserved Zstandard frame bit is set"),
            Self::DictionaryRequired(id) => write!(output, "Zstandard dictionary {id} is required"),
            Self::ReservedBlock => output.write_str("reserved Zstandard block type"),
            Self::UnsupportedCompressedBlock => {
                output.write_str("compressed Zstandard blocks are not implemented yet")
            }
            Self::ContentSizeMismatch => {
                output.write_str("Zstandard content size does not match decoded data")
            }
            Self::TrailingData => output.write_str("trailing data after Zstandard frame"),
            Self::Allocation => output.write_str("not enough memory for Zstandard output"),
        }
    }
}

impl core::error::Error for Error {}
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct FrameInfo {
    pub header_bytes: usize,
    pub window_size: u64,
    pub content_size: Option<u64>,
    pub checksum: bool,
}

pub fn inspect_frame(input: &[u8]) -> Result<FrameInfo> {
    let magic = read_u32(input, 0)?;
    if magic != FRAME_MAGIC {
        return Err(Error::InvalidMagic(magic));
    }
    let descriptor = *input.get(4).ok_or(Error::Truncated)?;
    if descriptor & 0x08 != 0 {
        return Err(Error::ReservedHeaderBit);
    }
    let single_segment = descriptor & 0x20 != 0;
    let checksum = descriptor & 0x04 != 0;
    let dictionary_size = [0usize, 1, 2, 4][(descriptor & 3) as usize];
    let content_flag = descriptor >> 6;
    let content_size_bytes = match content_flag {
        0 if single_segment => 1,
        0 => 0,
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let mut cursor = 5;
    let window_size = if single_segment {
        0
    } else {
        let value = *input.get(cursor).ok_or(Error::Truncated)?;
        cursor += 1;
        let exponent = (value >> 3) as u32 + 10;
        let base = 1u64.checked_shl(exponent).ok_or(Error::Truncated)?;
        base + (base >> 3) * (value & 7) as u64
    };
    let dictionary_id = read_variable(input, cursor, dictionary_size)? as u32;
    cursor += dictionary_size;
    if dictionary_id != 0 {
        return Err(Error::DictionaryRequired(dictionary_id));
    }
    let mut content_size = if content_size_bytes == 0 {
        None
    } else {
        Some(read_variable(input, cursor, content_size_bytes)?)
    };
    if content_size_bytes == 2 {
        content_size = content_size.map(|size| size + 256);
    }
    cursor += content_size_bytes;
    Ok(FrameInfo {
        header_bytes: cursor,
        window_size: if single_segment {
            content_size.unwrap_or(0)
        } else {
            window_size
        },
        content_size,
        checksum,
    })
}

/// Decodes complete frames containing raw and run-length blocks. Entropy-coded
/// blocks deliberately return a distinct error while the Huffman/FSE stages
/// are implemented, rather than accepting partially decoded data.
pub fn decode(input: &[u8]) -> Result<Vector<u8>> {
    let info = inspect_frame(input)?;
    let capacity = info
        .content_size
        .and_then(|size| usize::try_from(size).ok())
        .unwrap_or(0);
    let mut output = Vector::with_capacity(capacity).map_err(|_| Error::Allocation)?;
    let mut cursor = info.header_bytes;
    loop {
        let header = read_u24(input, cursor)?;
        cursor += 3;
        let last = header & 1 != 0;
        let block_type = (header >> 1) & 3;
        let block_size = (header >> 3) as usize;
        match block_type {
            0 => {
                let bytes = input
                    .get(cursor..cursor.checked_add(block_size).ok_or(Error::Truncated)?)
                    .ok_or(Error::Truncated)?;
                output
                    .try_extend_from_slice(bytes)
                    .map_err(|_| Error::Allocation)?;
                cursor += block_size;
            }
            1 => {
                let byte = *input.get(cursor).ok_or(Error::Truncated)?;
                cursor += 1;
                for _ in 0..block_size {
                    output.push(byte);
                }
            }
            2 => return Err(Error::UnsupportedCompressedBlock),
            _ => return Err(Error::ReservedBlock),
        }
        if last {
            break;
        }
    }
    if info.checksum {
        cursor = cursor.checked_add(4).ok_or(Error::Truncated)?;
    }
    if cursor > input.len() {
        return Err(Error::Truncated);
    }
    if cursor != input.len() {
        return Err(Error::TrailingData);
    }
    if info
        .content_size
        .is_some_and(|size| size != output.len() as u64)
    {
        return Err(Error::ContentSizeMismatch);
    }
    Ok(output)
}

fn read_variable(input: &[u8], offset: usize, width: usize) -> Result<u64> {
    let bytes = input
        .get(offset..offset.checked_add(width).ok_or(Error::Truncated)?)
        .ok_or(Error::Truncated)?;
    let mut value = 0u64;
    for (shift, byte) in bytes.iter().enumerate() {
        value |= (*byte as u64) << (shift * 8);
    }
    Ok(value)
}

fn read_u24(input: &[u8], offset: usize) -> Result<u32> {
    let bytes = input.get(offset..offset + 3).ok_or(Error::Truncated)?;
    Ok(bytes[0] as u32 | (bytes[1] as u32) << 8 | (bytes[2] as u32) << 16)
}

fn read_u32(input: &[u8], offset: usize) -> Result<u32> {
    input
        .get(offset..offset + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .ok_or(Error::Truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_single_segment_raw_block() {
        let frame = [
            0x28, 0xb5, 0x2f, 0xfd, 0x20, 3, 0x19, 0, 0, b'a', b'b', b'c',
        ];
        assert_eq!(&*decode(&frame).unwrap(), b"abc");
    }

    #[test]
    fn decodes_single_segment_rle_block() {
        let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 5, 0x2b, 0, 0, b'x'];
        assert_eq!(&*decode(&frame).unwrap(), b"xxxxx");
    }

    #[test]
    fn parses_kiwix_style_streaming_header() {
        let prefix = [0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x68, 0x4c, 0xb0, 0x00];
        let info = inspect_frame(&prefix).unwrap();
        assert_eq!(info.header_bytes, 6);
        assert_eq!(info.window_size, 1 << 23);
        assert_eq!(info.content_size, None);
        assert_eq!(read_u24(&prefix, info.header_bytes).unwrap() >> 3, 5641);
    }
}
