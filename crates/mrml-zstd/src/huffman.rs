use crate::bits::ReverseBits;
use crate::fse::{State as FseState, parse_table};
use crate::{Error, Result};
use mrml_runtime::Vector;

const MAX_BITS: u8 = 11;

#[derive(Clone, Copy, Default)]
struct Slot {
    symbol: u8,
    bits: u8,
}

#[derive(Clone)]
pub(crate) struct Table {
    maximum_bits: u8,
    lookup: Vector<Slot>,
}

impl Table {
    fn from_weights(weights: &[u8]) -> Result<Self> {
        let mut total = 0u32;
        for weight in weights {
            if *weight > MAX_BITS {
                return Err(Error::InvalidHuffmanTree);
            }
            if *weight != 0 {
                total = total
                    .checked_add(1u32 << (*weight - 1))
                    .ok_or(Error::InvalidHuffmanTree)?;
            }
        }
        if total == 0 {
            return Err(Error::InvalidHuffmanTree);
        }
        // Zstandard reserves one implicit final symbol. Consequently the
        // table log is floor(log2(explicit weight total)) + 1 even when the
        // explicit total is already a power of two.
        let maximum_bits = (32 - total.leading_zeros()) as u8;
        if maximum_bits > MAX_BITS {
            return Err(Error::InvalidHuffmanTree);
        }
        let missing = (1u32 << maximum_bits)
            .checked_sub(total)
            .ok_or(Error::InvalidHuffmanTree)?;
        if !missing.is_power_of_two() {
            return Err(Error::InvalidHuffmanTree);
        }
        let final_weight = missing.trailing_zeros() as u8 + 1;
        let count = weights
            .len()
            .checked_add(1)
            .ok_or(Error::InvalidHuffmanTree)?;
        if count > 256 {
            return Err(Error::InvalidHuffmanTree);
        }
        let mut lengths = Vector::with_capacity(count).map_err(|_| Error::Allocation)?;
        for weight in weights
            .iter()
            .copied()
            .chain(core::iter::once(final_weight))
        {
            lengths.push(if weight == 0 {
                0
            } else {
                maximum_bits + 1 - weight
            });
        }
        Self::from_lengths(&lengths)
    }

    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let maximum_bits = *lengths.iter().max().ok_or(Error::InvalidHuffmanTree)?;
        if maximum_bits == 0 || maximum_bits > MAX_BITS {
            return Err(Error::InvalidHuffmanTree);
        }
        let mut counts = [0u16; MAX_BITS as usize + 1];
        for length in lengths {
            if *length != 0 {
                counts[*length as usize] += 1;
            }
        }
        let kraft: u32 = (1..=maximum_bits)
            .map(|bits| (counts[bits as usize] as u32) << (maximum_bits - bits))
            .sum();
        if kraft != 1u32 << maximum_bits {
            return Err(Error::InvalidHuffmanTree);
        }
        let mut next = [0u16; MAX_BITS as usize + 1];
        for bits in (1..maximum_bits).rev() {
            next[bits as usize] = (next[bits as usize + 1] + counts[bits as usize + 1]) >> 1;
        }
        let size = 1usize << maximum_bits;
        let mut lookup = Vector::with_capacity(size).map_err(|_| Error::Allocation)?;
        lookup.resize(size, Slot::default());
        for bits in (1..=maximum_bits).rev() {
            for (symbol, length) in lengths.iter().enumerate() {
                if *length != bits {
                    continue;
                }
                let code = next[bits as usize] as usize;
                next[bits as usize] += 1;
                let shift = maximum_bits - bits;
                let start = code << shift;
                let end = start + (1usize << shift);
                if end > lookup.len() {
                    return Err(Error::InvalidHuffmanTree);
                }
                for slot in &mut lookup[start..end] {
                    *slot = Slot {
                        symbol: symbol as u8,
                        bits,
                    };
                }
            }
        }
        Ok(Self {
            maximum_bits,
            lookup,
        })
    }

    fn decode_symbol(&self, bits: &mut ReverseBits<'_>) -> Result<u8> {
        let slot = self.lookup[bits.peek_padded(self.maximum_bits) as usize];
        if slot.bits == 0 {
            return Err(Error::InvalidHuffmanStream);
        }
        if slot.bits as usize > bits.remaining() {
            return Err(Error::HuffmanSymbolTruncated(slot.bits, bits.remaining()));
        }
        bits.consume(slot.bits)?;
        Ok(slot.symbol)
    }
}

pub(crate) fn parse_tree(bytes: &[u8]) -> Result<(Table, usize)> {
    let header = *bytes.first().ok_or(Error::InvalidHuffmanTree)?;
    let (weights, consumed) = if header >= 128 {
        let count = header as usize - 127;
        let packed = count.div_ceil(2);
        let data = bytes.get(1..1 + packed).ok_or(Error::InvalidHuffmanTree)?;
        let mut weights = Vector::with_capacity(count).map_err(|_| Error::Allocation)?;
        for index in 0..count {
            weights.push(if index & 1 == 0 {
                data[index / 2] >> 4
            } else {
                data[index / 2] & 15
            });
        }
        (weights, 1 + packed)
    } else {
        let length = header as usize;
        let payload = bytes.get(1..1 + length).ok_or(Error::InvalidHuffmanTree)?;
        (decode_fse_weights(payload)?, 1 + length)
    };
    let table = Table::from_weights(&weights)?;
    Ok((table, consumed))
}

fn decode_fse_weights(payload: &[u8]) -> Result<Vector<u8>> {
    let (table, header_bytes) = parse_table(payload, 6, MAX_BITS as u16)?;
    let stream = payload
        .get(header_bytes..)
        .ok_or(Error::InvalidHuffmanTree)?;
    let mut bits = ReverseBits::new(stream)?;
    let mut first = FseState::new(&table, &mut bits)?;
    let mut second = FseState::new(&table, &mut bits)?;
    let mut output = Vector::with_capacity(32).map_err(|_| Error::Allocation)?;
    loop {
        if output.len() >= 255 {
            return Err(Error::InvalidHuffmanTree);
        }
        output.push(first.symbol(&table) as u8);
        let entry = first.entry(&table);
        if bits.remaining() < entry.bits as usize {
            output.push(second.symbol(&table) as u8);
            break;
        }
        first.advance(&table, &mut bits)?;
        if output.len() >= 255 {
            return Err(Error::InvalidHuffmanTree);
        }
        output.push(second.symbol(&table) as u8);
        let entry = second.entry(&table);
        if bits.remaining() < entry.bits as usize {
            output.push(first.symbol(&table) as u8);
            break;
        }
        second.advance(&table, &mut bits)?;
    }
    Ok(output)
}

pub(crate) fn decode_stream(
    bytes: &[u8],
    table: &Table,
    count: usize,
    output: &mut Vector<u8>,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let mut bits = ReverseBits::new(bytes)?;
    for _ in 0..count {
        output.push(table.decode_symbol(&mut bits)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_direct_tree_from_spec_example() {
        let (table, consumed) = parse_tree(&[132, 0x43, 0x20, 0x10]).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(table.maximum_bits, 4);
    }
}
