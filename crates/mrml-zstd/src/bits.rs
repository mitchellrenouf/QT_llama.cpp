use crate::{Error, Result};

pub(crate) struct ForwardBits<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ForwardBits<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    pub(crate) fn consumed_bytes(&self) -> usize {
        self.position.div_ceil(8)
    }
    pub(crate) fn peek(&self, count: u8) -> Result<u32> {
        if count > 24 || self.position + count as usize > self.bytes.len() * 8 {
            return Err(Error::Truncated);
        }
        let mut value = 0u32;
        for bit in 0..count as usize {
            let source = self.position + bit;
            value |= (((self.bytes[source / 8] >> (source & 7)) & 1) as u32) << bit;
        }
        Ok(value)
    }
    pub(crate) fn read(&mut self, count: u8) -> Result<u32> {
        let value = self.peek(count)?;
        self.position += count as usize;
        Ok(value)
    }
    pub(crate) fn consume(&mut self, count: u8) -> Result<()> {
        if self.position + count as usize > self.bytes.len() * 8 {
            return Err(Error::Truncated);
        }
        self.position += count as usize;
        Ok(())
    }
}

pub(crate) struct ReverseBits<'a> {
    bytes: &'a [u8],
    remaining: usize,
}

impl<'a> ReverseBits<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self> {
        let last = *bytes.last().ok_or(Error::Truncated)?;
        if last == 0 {
            return Err(Error::InvalidBitstream);
        }
        let marker = 7usize - last.leading_zeros() as usize;
        Ok(Self {
            bytes,
            remaining: (bytes.len() - 1) * 8 + marker,
        })
    }
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
    pub(crate) fn peek_padded(&self, count: u8) -> u32 {
        let available = core::cmp::min(count as usize, self.remaining);
        let mut value = 0u32;
        for bit in 0..available {
            let source = self.remaining - 1 - bit;
            value = (value << 1) | ((self.bytes[source / 8] >> (source & 7)) & 1) as u32;
        }
        value << (count as usize - available)
    }
    pub(crate) fn consume(&mut self, count: u8) -> Result<()> {
        if count as usize > self.remaining {
            return Err(Error::Truncated);
        }
        self.remaining -= count as usize;
        Ok(())
    }
    pub(crate) fn read(&mut self, count: u8) -> Result<u32> {
        let count = count as usize;
        if count > 24 || count > self.remaining {
            return Err(Error::Truncated);
        }
        let value = self.peek_padded(count as u8);
        self.remaining -= count;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_reader_skips_end_marker() {
        let mut bits = ReverseBits::new(&[0xab, 0x01]).unwrap();
        assert_eq!(bits.remaining(), 8);
        assert_eq!(bits.read(4).unwrap(), 0x0a);
        assert_eq!(bits.read(4).unwrap(), 0x0b);
    }

    #[test]
    fn forward_reader_is_little_endian() {
        let mut bits = ForwardBits::new(&[0b1011_0100, 0b10]);
        assert_eq!(bits.read(4).unwrap(), 4);
        assert_eq!(bits.read(6).unwrap(), 0b10_1011);
    }
}
