use crate::bits::{ForwardBits, ReverseBits};
use crate::{Error, Result};
use mrml_runtime::Vector;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Entry {
    pub(crate) symbol: u16,
    pub(crate) bits: u8,
    pub(crate) baseline: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct Table {
    pub(crate) accuracy: u8,
    pub(crate) entries: Vector<Entry>,
}

impl Table {
    pub(crate) fn from_probabilities(probabilities: &[i16], accuracy: u8) -> Result<Self> {
        if !(5..=9).contains(&accuracy) {
            return Err(Error::InvalidFseTable);
        }
        let size = 1usize << accuracy;
        let mut symbols = Vector::with_capacity(size).map_err(|_| Error::Allocation)?;
        symbols.resize(size, u16::MAX);
        let mut high = size;
        for (symbol, probability) in probabilities.iter().enumerate() {
            if *probability == -1 {
                high = high.checked_sub(1).ok_or(Error::InvalidFseTable)?;
                symbols[high] = symbol as u16;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mut position = 0usize;
        for (symbol, probability) in probabilities.iter().enumerate() {
            if *probability <= 0 {
                continue;
            }
            for _ in 0..*probability as usize {
                while symbols[position] != u16::MAX {
                    position = (position + step) & (size - 1);
                }
                symbols[position] = symbol as u16;
                position = (position + step) & (size - 1);
            }
        }
        if symbols.iter().any(|symbol| *symbol == u16::MAX) {
            return Err(Error::InvalidFseTable);
        }
        let mut next = Vector::with_capacity(probabilities.len()).map_err(|_| Error::Allocation)?;
        next.resize(probabilities.len(), 0u32);
        for (symbol, probability) in probabilities.iter().enumerate() {
            next[symbol] = if *probability == -1 {
                1
            } else {
                (*probability).max(0) as u32
            };
        }
        let mut entries = Vector::with_capacity(size).map_err(|_| Error::Allocation)?;
        entries.resize(size, Entry::default());
        for state in 0..size {
            let symbol = symbols[state] as usize;
            if probabilities[symbol] == -1 {
                entries[state] = Entry {
                    symbol: symbol as u16,
                    bits: accuracy,
                    baseline: 0,
                };
                continue;
            }
            let value = next[symbol];
            next[symbol] += 1;
            if value == 0 {
                return Err(Error::InvalidFseTable);
            }
            let bits = accuracy - (31 - value.leading_zeros()) as u8;
            let baseline = (value << bits) as i32 - size as i32;
            if !(0..size as i32).contains(&baseline) {
                return Err(Error::InvalidFseTable);
            }
            entries[state] = Entry {
                symbol: symbol as u16,
                bits,
                baseline: baseline as u16,
            };
        }
        Ok(Self { accuracy, entries })
    }
}

pub(crate) fn parse_table(
    bytes: &[u8],
    maximum_accuracy: u8,
    maximum_symbol: u16,
) -> Result<(Table, usize)> {
    let mut bits = ForwardBits::new(bytes);
    let accuracy = bits.read(4)? as u8 + 5;
    if accuracy > maximum_accuracy {
        return Err(Error::InvalidFseTable);
    }
    let mut remaining = (1i32 << accuracy) + 1;
    let mut probabilities =
        Vector::with_capacity(maximum_symbol as usize + 1).map_err(|_| Error::Allocation)?;
    let mut symbol = 0usize;
    let mut repeat_zero = false;
    while remaining > 1 {
        if symbol > maximum_symbol as usize {
            return Err(Error::InvalidFseTable);
        }
        if repeat_zero {
            let mut zeros = 0usize;
            loop {
                let count = bits.read(2)? as usize;
                zeros += count;
                if count != 3 {
                    break;
                }
            }
            for _ in 0..zeros {
                if symbol > maximum_symbol as usize {
                    return Err(Error::InvalidFseTable);
                }
                probabilities.push(0);
                symbol += 1;
            }
            repeat_zero = false;
            continue;
        }
        let rem = remaining as u32;
        let width = (32 - rem.leading_zeros()) as u8;
        let threshold = (1u32 << width) - 1 - rem;
        let peek = bits.peek(width)?;
        let low = peek & ((1u32 << (width - 1)) - 1);
        let (value, consumed) = if low < threshold {
            (low, width - 1)
        } else {
            (
                if peek >= 1u32 << (width - 1) {
                    peek - threshold
                } else {
                    peek
                },
                width,
            )
        };
        bits.consume(consumed)?;
        let probability = value as i16 - 1;
        probabilities.push(probability);
        symbol += 1;
        repeat_zero = probability == 0;
        if probability != 0 {
            remaining -= if probability < 0 {
                1
            } else {
                probability as i32
            };
            if remaining < 1 {
                return Err(Error::InvalidFseTable);
            }
        }
    }
    let table = Table::from_probabilities(&probabilities, accuracy)?;
    Ok((table, bits.consumed_bytes()))
}

pub(crate) struct State {
    value: u16,
}

impl State {
    pub(crate) const fn from_value(value: u16) -> Self {
        Self { value }
    }
    pub(crate) fn new(table: &Table, bits: &mut ReverseBits<'_>) -> Result<Self> {
        let value = bits.read(table.accuracy)? as u16;
        if value as usize >= table.entries.len() {
            return Err(Error::InvalidFseState);
        }
        Ok(Self { value })
    }
    pub(crate) fn symbol(&self, table: &Table) -> u16 {
        table.entries[self.value as usize].symbol
    }
    pub(crate) fn entry(&self, table: &Table) -> Entry {
        table.entries[self.value as usize]
    }
    pub(crate) fn advance(&mut self, table: &Table, bits: &mut ReverseBits<'_>) -> Result<()> {
        let entry = table.entries[self.value as usize];
        let value = entry.baseline as u32 + bits.read(entry.bits)?;
        if value as usize >= table.entries.len() {
            return Err(Error::InvalidFseState);
        }
        self.value = value as u16;
        Ok(())
    }
}
