#![no_std]

mod bits;
mod fse;
mod huffman;
mod sequences;

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
    InvalidLiteralsSection,
    InvalidSequencesSection,
    ContentSizeMismatch,
    TrailingData,
    Allocation,
    InvalidBitstream,
    InvalidFseTable,
    InvalidFseState,
    InvalidHuffmanTree,
    InvalidHuffmanStream,
    HuffmanSymbolTruncated(u8, usize),
    InvalidOffset,
    BlockTooLarge,
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
            Self::InvalidLiteralsSection => output.write_str("invalid Zstandard literals section"),
            Self::InvalidSequencesSection => {
                output.write_str("invalid Zstandard sequences section")
            }
            Self::ContentSizeMismatch => {
                output.write_str("Zstandard content size does not match decoded data")
            }
            Self::TrailingData => output.write_str("trailing data after Zstandard frame"),
            Self::Allocation => output.write_str("not enough memory for Zstandard output"),
            Self::InvalidBitstream => output.write_str("invalid Zstandard reverse bitstream"),
            Self::InvalidFseTable => output.write_str("invalid Zstandard FSE table"),
            Self::InvalidFseState => output.write_str("invalid Zstandard FSE state"),
            Self::InvalidHuffmanTree => output.write_str("invalid Zstandard Huffman tree"),
            Self::InvalidHuffmanStream => output.write_str("invalid Zstandard Huffman stream"),
            Self::HuffmanSymbolTruncated(bits, remaining) => write!(
                output,
                "Zstandard Huffman symbol needs {bits} bits with {remaining} remaining"
            ),
            Self::InvalidOffset => output.write_str("invalid Zstandard match offset"),
            Self::BlockTooLarge => output.write_str("Zstandard block exceeds 128 KiB"),
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
    let mut huffman_table = None;
    let mut sequence_state = sequences::State::new();
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
            2 => {
                let block = input
                    .get(cursor..cursor.checked_add(block_size).ok_or(Error::Truncated)?)
                    .ok_or(Error::Truncated)?;
                decode_compressed_block(
                    block,
                    &mut output,
                    &mut huffman_table,
                    &mut sequence_state,
                )?;
                cursor += block_size;
            }
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

enum Literals<'a> {
    Raw(&'a [u8]),
    Rle { byte: u8, count: usize },
    Decoded(Vector<u8>),
}

fn decode_compressed_block(
    block: &[u8],
    output: &mut Vector<u8>,
    huffman_table: &mut Option<huffman::Table>,
    sequence_state: &mut sequences::State,
) -> Result<()> {
    let (literals, consumed) = parse_literals(block, huffman_table)?;
    let commands = sequences::decode(
        block
            .get(consumed..)
            .ok_or(Error::InvalidSequencesSection)?,
        sequence_state,
    )?;
    match literals {
        Literals::Raw(bytes) => sequences::execute(&commands, bytes, output)?,
        Literals::Rle { byte, count } => {
            let mut bytes = Vector::with_capacity(count).map_err(|_| Error::Allocation)?;
            bytes.resize(count, byte);
            sequences::execute(&commands, &bytes, output)?;
        }
        Literals::Decoded(bytes) => sequences::execute(&commands, &bytes, output)?,
    }
    Ok(())
}

fn parse_literals<'a>(
    block: &'a [u8],
    previous_table: &mut Option<huffman::Table>,
) -> Result<(Literals<'a>, usize)> {
    let first = *block.first().ok_or(Error::InvalidLiteralsSection)?;
    let kind = first & 3;
    let size_format = (first >> 2) & 3;
    if kind >= 2 {
        return parse_huffman_literals(block, kind == 3, size_format, previous_table);
    }
    let (header_bytes, regenerated_size): (usize, usize) = match size_format {
        0 | 2 => (1, (first >> 3) as usize),
        1 => {
            let second = *block.get(1).ok_or(Error::InvalidLiteralsSection)?;
            (2, (first as usize >> 4) | (second as usize) << 4)
        }
        _ => {
            let second = *block.get(1).ok_or(Error::InvalidLiteralsSection)?;
            let third = *block.get(2).ok_or(Error::InvalidLiteralsSection)?;
            (
                3,
                (first as usize >> 4) | (second as usize) << 4 | (third as usize) << 12,
            )
        }
    };
    if kind == 0 {
        let end = header_bytes
            .checked_add(regenerated_size)
            .ok_or(Error::InvalidLiteralsSection)?;
        let bytes = block
            .get(header_bytes..end)
            .ok_or(Error::InvalidLiteralsSection)?;
        Ok((Literals::Raw(bytes), end))
    } else {
        let byte = *block
            .get(header_bytes)
            .ok_or(Error::InvalidLiteralsSection)?;
        Ok((
            Literals::Rle {
                byte,
                count: regenerated_size,
            },
            header_bytes + 1,
        ))
    }
}

fn parse_huffman_literals<'a>(
    block: &'a [u8],
    reuse: bool,
    size_format: u8,
    previous_table: &mut Option<huffman::Table>,
) -> Result<(Literals<'a>, usize)> {
    let (regenerated, compressed, header_bytes, four_streams) = match size_format {
        0 | 1 => {
            let bytes = block.get(..3).ok_or(Error::InvalidLiteralsSection)?;
            let regenerated = (bytes[0] as usize >> 4) | ((bytes[1] as usize & 0x3f) << 4);
            let compressed = (bytes[1] as usize >> 6) | (bytes[2] as usize) << 2;
            (regenerated, compressed, 3, size_format == 1)
        }
        2 => {
            let bytes = block.get(..4).ok_or(Error::InvalidLiteralsSection)?;
            let regenerated =
                (bytes[0] as usize >> 4) | (bytes[1] as usize) << 4 | (bytes[2] as usize & 3) << 12;
            let compressed = (bytes[2] as usize >> 2) | (bytes[3] as usize) << 6;
            (regenerated, compressed, 4, true)
        }
        _ => {
            let bytes = block.get(..5).ok_or(Error::InvalidLiteralsSection)?;
            let packed = read_variable(bytes, 0, 5)?;
            (
                ((packed >> 4) & 0x3ffff) as usize,
                ((packed >> 22) & 0x3ffff) as usize,
                5,
                true,
            )
        }
    };
    if regenerated > 128 * 1024 || compressed == 0 {
        return Err(Error::InvalidLiteralsSection);
    }
    let payload = block
        .get(header_bytes..header_bytes + compressed)
        .ok_or(Error::InvalidLiteralsSection)?;
    let (table, tree_bytes) = if reuse {
        (previous_table.clone().ok_or(Error::InvalidHuffmanTree)?, 0)
    } else {
        huffman::parse_tree(payload)?
    };
    let streams = payload
        .get(tree_bytes..)
        .ok_or(Error::InvalidLiteralsSection)?;
    let mut decoded = Vector::with_capacity(regenerated).map_err(|_| Error::Allocation)?;
    if four_streams {
        let jump = streams.get(..6).ok_or(Error::InvalidHuffmanStream)?;
        let first = u16::from_le_bytes(jump[0..2].try_into().unwrap()) as usize;
        let second = u16::from_le_bytes(jump[2..4].try_into().unwrap()) as usize;
        let third = u16::from_le_bytes(jump[4..6].try_into().unwrap()) as usize;
        let total = 6usize
            .checked_add(first)
            .and_then(|value| value.checked_add(second))
            .and_then(|value| value.checked_add(third))
            .ok_or(Error::InvalidHuffmanStream)?;
        if total >= streams.len() || regenerated < 4 {
            return Err(Error::InvalidHuffmanStream);
        }
        let per = regenerated.div_ceil(4);
        let last = regenerated
            .checked_sub(per * 3)
            .ok_or(Error::InvalidHuffmanStream)?;
        let starts = [6, 6 + first, 6 + first + second, total];
        let ends = [6 + first, 6 + first + second, total, streams.len()];
        let counts = [per, per, per, last];
        for index in 0..4 {
            huffman::decode_stream(
                &streams[starts[index]..ends[index]],
                &table,
                counts[index],
                &mut decoded,
            )?;
        }
    } else {
        huffman::decode_stream(streams, &table, regenerated, &mut decoded)?;
    }
    *previous_table = Some(table);
    Ok((Literals::Decoded(decoded), header_bytes + compressed))
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

    #[test]
    fn decodes_compressed_block_with_raw_literals_and_no_sequences() {
        let frame = [
            0x28, 0xb5, 0x2f, 0xfd, 0x20, 3, 0x2d, 0, 0, 0x18, b'a', b'b', b'c', 0,
        ];
        assert_eq!(&*decode(&frame).unwrap(), b"abc");
    }

    #[test]
    fn decodes_compressed_block_with_rle_literals_and_no_sequences() {
        let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x20, 5, 0x1d, 0, 0, 0x29, b'x', 0];
        assert_eq!(&*decode(&frame).unwrap(), b"xxxxx");
    }
}
