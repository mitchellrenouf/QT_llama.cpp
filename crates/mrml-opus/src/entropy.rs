use crate::Error;

const TOP: u32 = 1 << 23;

fn fractional_tell(nbits_total: u32, range: u32) -> u32 {
    let mut log = u32::BITS - range.leading_zeros();
    let mut normalized = range >> (log - 16);
    for _ in 0..3 {
        normalized = ((u64::from(normalized) * u64::from(normalized)) >> 15) as u32;
        let bit = normalized >> 16;
        log = log * 2 + bit;
        normalized >>= bit;
    }
    nbits_total * 8 - log
}

/// Bit-exact RFC 6716 range encoder writing into a caller-owned frame buffer.
pub struct RangeEncoder<'a> {
    data: &'a mut [u8],
    front: usize,
    back: usize,
    rng: u32,
    val: u32,
    rem: i16,
    ext: u32,
    nbits_total: u32,
    raw_window: u64,
    raw_bits: u8,
    finalizing: bool,
    overlapped: bool,
}

impl<'a> RangeEncoder<'a> {
    pub fn new(data: &'a mut [u8]) -> Self {
        data.fill(0);
        let back = data.len();
        Self {
            data,
            front: 0,
            back,
            rng: 1 << 31,
            val: 0,
            rem: -1,
            ext: 0,
            nbits_total: 33,
            raw_window: 0,
            raw_bits: 0,
            finalizing: false,
            overlapped: false,
        }
    }

    pub const fn range(&self) -> u32 {
        self.rng
    }

    pub fn encode(&mut self, low: u32, high: u32, total: u32) -> Result<(), Error> {
        if low >= high || high > total || total == 0 || total > 65_535 {
            return Err(Error::InvalidPacket);
        }
        let unit = self.rng / total;
        if low > 0 {
            self.val = self.val.wrapping_add(self.rng - unit * (total - low));
            self.rng = unit * (high - low);
        } else {
            self.rng -= unit * (total - high);
        }
        self.normalize()
    }

    pub fn encode_icdf(&mut self, symbol: usize, icdf: &[u8], bits: u8) -> Result<(), Error> {
        if symbol >= icdf.len()
            || icdf.is_empty()
            || bits == 0
            || bits > 8
            || *icdf.last().unwrap_or(&1) != 0
        {
            return Err(Error::InvalidPacket);
        }
        let total = 1u32 << bits;
        let mut previous = total;
        for &inverse in icdf {
            if u32::from(inverse) >= previous {
                return Err(Error::InvalidPacket);
            }
            previous = u32::from(inverse);
        }
        let low = if symbol == 0 {
            0
        } else {
            total - u32::from(icdf[symbol - 1])
        };
        let high = total - u32::from(icdf[symbol]);
        self.encode(low, high, total)
    }

    pub fn encode_bit_logp(&mut self, value: bool, logp: u8) -> Result<(), Error> {
        if logp == 0 || logp > 31 {
            return Err(Error::InvalidPacket);
        }
        let split = self.rng >> logp;
        if split == 0 {
            return Err(Error::InvalidPacket);
        }
        if value {
            self.val = self.val.wrapping_add(self.rng - split);
            self.rng = split;
        } else {
            self.rng -= split;
        }
        self.normalize()
    }

    pub fn raw_bits(&mut self, value: u32, bits: u8) -> Result<(), Error> {
        if bits > 32 || (bits < 32 && value >= 1u32 << bits) {
            return Err(Error::InvalidPacket);
        }
        self.raw_window |= u64::from(value) << self.raw_bits;
        self.raw_bits += bits;
        while self.raw_bits >= 8 {
            if self.back <= self.front {
                return Err(Error::BufferTooSmall);
            }
            self.back -= 1;
            self.data[self.back] = self.raw_window as u8;
            self.raw_window >>= 8;
            self.raw_bits -= 8;
        }
        self.nbits_total = self
            .nbits_total
            .checked_add(u32::from(bits))
            .ok_or(Error::InvalidPacket)?;
        Ok(())
    }

    pub fn encode_uint(&mut self, value: u32, total: u32) -> Result<(), Error> {
        if total == 0 || value >= total {
            return Err(Error::InvalidPacket);
        }
        let bits = (u32::BITS - (total - 1).leading_zeros()) as u8;
        if bits <= 8 {
            self.encode(value, value + 1, total)
        } else {
            let low_bits = bits - 8;
            let high = value >> low_bits;
            let high_total = ((total - 1) >> low_bits) + 1;
            self.encode(high, high + 1, high_total)?;
            self.raw_bits(value & ((1u32 << low_bits) - 1), low_bits)
        }
    }

    pub fn encode_pdf(&mut self, symbol: usize, frequencies: &[u8]) -> Result<(), Error> {
        if symbol >= frequencies.len() {
            return Err(Error::InvalidPacket);
        }
        let total = frequencies
            .iter()
            .try_fold(0u32, |sum, &frequency| {
                sum.checked_add(u32::from(frequency))
            })
            .ok_or(Error::InvalidPacket)?;
        let low = frequencies[..symbol]
            .iter()
            .fold(0u32, |sum, &frequency| sum + u32::from(frequency));
        let high = low + u32::from(frequencies[symbol]);
        if low == high {
            return Err(Error::InvalidPacket);
        }
        self.encode(low, high, total)
    }

    pub fn tell(&self) -> u32 {
        self.nbits_total - (u32::BITS - self.rng.leading_zeros())
    }
    /// Conservative bit usage in units of one eighth bit.
    pub fn tell_frac(&self) -> u32 {
        fractional_tell(self.nbits_total, self.rng)
    }

    pub fn finish(mut self) -> Result<usize, Error> {
        self.finalize()?;
        Ok(self.data.len())
    }

    /// Finalizes a range-only stream and returns its actual byte count.
    /// SILK has no tail-packed raw bits, so unused capacity must not become
    /// part of the Opus frame (where it would signal implicit redundancy).
    pub fn finish_compact(mut self) -> Result<usize, Error> {
        if self.raw_bits != 0 || self.raw_window != 0 || self.back != self.data.len() {
            return Err(Error::InvalidPacket);
        }
        self.finalize()?;
        let used = self.front;
        self.data[used..].fill(0);
        Ok(used)
    }

    fn finalize(&mut self) -> Result<(), Error> {
        self.finalizing = true;
        let mut bits = self.rng.leading_zeros();
        let mut mask = 0x7fff_ffffu32 >> bits;
        let mut end = self.val.wrapping_add(mask) & !mask;
        if end | mask >= self.val.wrapping_add(self.rng) {
            bits += 1;
            mask >>= 1;
            end = self.val.wrapping_add(mask) & !mask;
        }
        let termination_bits = bits;
        while bits > 0 {
            self.carry_out(end >> 23)?;
            end = (end << 8) & 0x7fff_ffff;
            bits = bits.saturating_sub(8);
        }
        if self.rem > 0 || self.ext > 0 {
            self.carry_out(0)?;
        }
        if self.raw_bits > 0 {
            if self.back > self.front {
                self.back -= 1;
                self.data[self.back] = self.raw_window as u8;
            } else {
                let range_bits_in_last_byte = ((termination_bits - 1) & 7) + 1;
                if self.back != self.front
                    || self.front == 0
                    || u32::from(self.raw_bits) + range_bits_in_last_byte > 8
                {
                    return Err(Error::BufferTooSmall);
                }
                self.data[self.front - 1] |= self.raw_window as u8;
            }
        }
        if self.front > self.back && !self.overlapped {
            return Err(Error::BufferTooSmall);
        }
        if self.front <= self.back {
            self.data[self.front..self.back].fill(0);
        }
        Ok(())
    }

    fn normalize(&mut self) -> Result<(), Error> {
        while self.rng <= TOP {
            self.carry_out(self.val >> 23)?;
            self.val = (self.val << 8) & 0x7fff_ffff;
            self.rng <<= 8;
            self.nbits_total += 8;
        }
        Ok(())
    }

    fn carry_out(&mut self, carry: u32) -> Result<(), Error> {
        if carry == 255 {
            self.ext = self.ext.checked_add(1).ok_or(Error::BufferTooSmall)?;
            return Ok(());
        }
        let bit = (carry >> 8) as u8;
        if self.rem >= 0 {
            self.write_front((self.rem as u8).wrapping_add(bit))?;
        }
        let fill = if bit != 0 { 0 } else { 255 };
        while self.ext > 0 {
            self.write_front(fill)?;
            self.ext -= 1;
        }
        self.rem = (carry & 255) as i16;
        Ok(())
    }

    fn write_front(&mut self, byte: u8) -> Result<(), Error> {
        if self.front >= self.back {
            if self.finalizing
                && !self.overlapped
                && self.front == self.back
                && self.front < self.data.len()
            {
                self.data[self.front] |= byte;
                self.front += 1;
                self.overlapped = true;
                return Ok(());
            }
            return Err(Error::BufferTooSmall);
        }
        self.data[self.front] = byte;
        self.front += 1;
        Ok(())
    }
}

/// Bit-exact RFC 6716 range and raw-bit decoder.
pub struct RangeDecoder<'a> {
    data: &'a [u8],
    storage_end: usize,
    front: usize,
    back: usize,
    rng: u32,
    val: u32,
    rem: u8,
    nbits_total: u32,
    raw_window: u64,
    raw_bits: u8,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let first = data.first().copied().unwrap_or(0);
        let mut decoder = Self {
            data,
            storage_end: data.len(),
            front: usize::from(!data.is_empty()),
            back: data.len(),
            rng: 128,
            val: 127 - u32::from(first >> 1),
            rem: first & 1,
            nbits_total: 9,
            raw_window: 0,
            raw_bits: 0,
        };
        decoder.normalize();
        decoder
    }

    pub const fn range(&self) -> u32 {
        self.rng
    }
    pub const fn value(&self) -> u32 {
        self.val
    }

    /// Current byte length available to the range and raw-bit decoders.
    pub const fn storage_len(&self) -> usize {
        self.storage_end
    }

    /// Reserves a byte-aligned payload at the end of the frame and returns
    /// its start offset. This is used by RFC 6716 transition redundancy
    /// before the CELT layer begins reading raw bits from the frame tail.
    pub fn reserve_tail(&mut self, bytes: usize) -> Result<usize, Error> {
        if bytes == 0
            || self.raw_bits != 0
            || self.raw_window != 0
            || self.back != self.storage_end
            || bytes >= self.storage_end
        {
            return Err(Error::InvalidPacket);
        }
        let start = self.storage_end - bytes;
        // Range decoding deliberately reads several bytes ahead. RFC 6716
        // section 4.1.2.1 requires data consumed by another tail reader to
        // be allowed to overlap that lookahead. The already-read bytes are
        // represented in `val`; shrinking the logical storage boundary does
        // not invalidate the current range state.
        self.storage_end = start;
        self.back = start;
        Ok(start)
    }

    /// Returns the range-coded frequency value in `[0, total)`.
    pub fn decode(&self, total: u32) -> Result<u32, Error> {
        if total == 0 || total > 65_535 {
            return Err(Error::InvalidPacket);
        }
        let unit = self.rng / total;
        if unit == 0 {
            return Err(Error::InvalidPacket);
        }
        Ok(total - (self.val / unit + 1).min(total))
    }

    /// Commits the cumulative interval selected by [`decode`](Self::decode).
    pub fn update(&mut self, low: u32, high: u32, total: u32) -> Result<(), Error> {
        if low >= high || high > total || total == 0 || total > 65_535 {
            return Err(Error::InvalidPacket);
        }
        let unit = self.rng / total;
        let scaled_high = unit.checked_mul(total - high).ok_or(Error::InvalidPacket)?;
        if self.val < scaled_high {
            return Err(Error::InvalidPacket);
        }
        self.val -= scaled_high;
        self.rng = if low > 0 {
            unit.checked_mul(high - low).ok_or(Error::InvalidPacket)?
        } else {
            self.rng
                .checked_sub(scaled_high)
                .ok_or(Error::InvalidPacket)?
        };
        if self.rng == 0 {
            return Err(Error::InvalidPacket);
        }
        self.normalize();
        Ok(())
    }

    pub fn decode_icdf(&mut self, icdf: &[u8], bits: u8) -> Result<usize, Error> {
        if icdf.is_empty() || bits == 0 || bits > 8 || *icdf.last().unwrap_or(&1) != 0 {
            return Err(Error::InvalidPacket);
        }
        let total = 1u32 << bits;
        let mut boundary = total;
        for &inverse in icdf {
            if u32::from(inverse) >= boundary {
                return Err(Error::InvalidPacket);
            }
            boundary = u32::from(inverse);
        }
        let value = self.decode(total)?;
        let mut previous = total;
        for (symbol, &inverse) in icdf.iter().enumerate() {
            let boundary = u32::from(inverse);
            if boundary >= previous || boundary >= total {
                return Err(Error::InvalidPacket);
            }
            let high = total - boundary;
            let low = total - previous;
            if value < high {
                self.update(low, high, total)?;
                return Ok(symbol);
            }
            previous = boundary;
        }
        Err(Error::InvalidPacket)
    }

    pub fn decode_bit_logp(&mut self, logp: u8) -> Result<bool, Error> {
        if logp == 0 || logp > 31 {
            return Err(Error::InvalidPacket);
        }
        let split = self.rng >> logp;
        if split == 0 {
            return Err(Error::InvalidPacket);
        }
        let zero_range = self.rng - split;
        let one = self.val < split;
        if one {
            self.rng = split;
        } else {
            self.val -= split;
            self.rng = zero_range;
        }
        self.normalize();
        Ok(one)
    }

    /// Reads up to 32 raw CELT bits from the back of the frame.
    pub fn raw_bits(&mut self, bits: u8) -> Result<u32, Error> {
        if bits > 32 {
            return Err(Error::InvalidPacket);
        }
        while self.raw_bits < bits {
            if self.back == 0 {
                return Err(Error::InvalidPacket);
            }
            self.back -= 1;
            self.raw_window |= u64::from(self.data[self.back]) << self.raw_bits;
            self.raw_bits = self.raw_bits.checked_add(8).ok_or(Error::InvalidPacket)?;
        }
        let mask = if bits == 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << bits) - 1
        };
        let value = (self.raw_window & mask) as u32;
        self.raw_window >>= bits;
        self.raw_bits -= bits;
        self.nbits_total = self
            .nbits_total
            .checked_add(u32::from(bits))
            .ok_or(Error::InvalidPacket)?;
        Ok(value)
    }

    pub fn decode_uint(&mut self, total: u32) -> Result<u32, Error> {
        if total == 0 {
            return Err(Error::InvalidPacket);
        }
        let max = total - 1;
        let bits = (u32::BITS - max.leading_zeros()) as u8;
        let value = if bits <= 8 {
            let value = self.decode(total)?;
            self.update(value, value + 1, total)?;
            value
        } else {
            let low_bits = bits - 8;
            let high_total = (max >> low_bits) + 1;
            let high = self.decode(high_total)?;
            self.update(high, high + 1, high_total)?;
            (high << low_bits) | self.raw_bits(low_bits)?
        };
        // The split representation has unused codes when `total` is not a
        // power of two. RFC 6716 section 4.1.5 permits concealment by
        // saturating such a reconstructed value to the last valid symbol.
        Ok(value.min(max))
    }

    pub fn decode_pdf(&mut self, frequencies: &[u8]) -> Result<usize, Error> {
        if frequencies.is_empty() {
            return Err(Error::InvalidPacket);
        }
        let total = frequencies
            .iter()
            .try_fold(0u32, |sum, &frequency| {
                sum.checked_add(u32::from(frequency))
            })
            .ok_or(Error::InvalidPacket)?;
        if total == 0 || total > 65_535 {
            return Err(Error::InvalidPacket);
        }
        let value = self.decode(total)?;
        let mut low = 0u32;
        for (symbol, &frequency) in frequencies.iter().enumerate() {
            let high = low + u32::from(frequency);
            if value < high {
                self.update(low, high, total)?;
                return Ok(symbol);
            }
            low = high;
        }
        Err(Error::InvalidPacket)
    }

    pub fn tell(&self) -> u32 {
        self.nbits_total - (u32::BITS - self.rng.leading_zeros())
    }
    /// Conservative bit usage in units of one eighth bit.
    pub fn tell_frac(&self) -> u32 {
        fractional_tell(self.nbits_total, self.rng)
    }

    fn normalize(&mut self) {
        while self.rng <= TOP {
            self.rng <<= 8;
            let next = if self.front < self.storage_end {
                let byte = self.data[self.front];
                self.front += 1;
                byte
            } else {
                0
            };
            let symbol = (self.rem << 7) | (next >> 1);
            self.rem = next & 1;
            self.val = ((self.val << 8).wrapping_add(u32::from(255 - symbol))) & 0x7fff_ffff;
            self.nbits_total += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_and_tell_match_rfc_invariant() {
        for first in [0, 1, 127, 128, 255] {
            let data = [first];
            let decoder = RangeDecoder::new(&data);
            assert!(decoder.range() > TOP);
            assert_eq!(decoder.tell(), 1);
            assert!((1..=8).contains(&decoder.tell_frac()));
        }
    }

    #[test]
    fn raw_bits_are_lsb_first_from_frame_end() {
        let mut decoder = RangeDecoder::new(&[0, 0b1010_0110, 0b0011_0101]);
        assert_eq!(decoder.raw_bits(4), Ok(0b0101));
        assert_eq!(decoder.raw_bits(6), Ok(0b100011));
    }

    #[test]
    fn final_range_byte_merges_with_packed_raw_byte() {
        let mut byte = [0u8; 1];
        let mut encoder = RangeEncoder::new(&mut byte);
        encoder.encode_bit_logp(false, 15).unwrap();
        encoder.raw_bits(0xa5, 8).unwrap();
        assert_eq!(encoder.finish(), Ok(1));

        let mut decoder = RangeDecoder::new(&byte);
        assert_eq!(decoder.decode_bit_logp(15), Ok(false));
        assert_eq!(decoder.raw_bits(8), Ok(0xa5));
    }

    #[test]
    fn malformed_contexts_are_rejected() {
        let mut decoder = RangeDecoder::new(&[0; 4]);
        assert_eq!(decoder.decode(0), Err(Error::InvalidPacket));
        assert_eq!(decoder.decode_icdf(&[], 8), Err(Error::InvalidPacket));
        assert_eq!(
            decoder.decode_icdf(&[2, 3, 0], 8),
            Err(Error::InvalidPacket)
        );
        assert_eq!(decoder.raw_bits(33), Err(Error::InvalidPacket));
    }

    #[test]
    fn split_uniform_decoder_saturates_unused_top_codes() {
        let mut saturated = 0usize;
        for first in 0..=u8::MAX {
            for last in [0u8, 1, 127, 255] {
                let data = [first, last];
                let mut decoder = RangeDecoder::new(&data);
                let value = decoder.decode_uint(257).unwrap();
                assert!(value < 257);
                saturated += usize::from(value == 256);
            }
        }
        assert!(saturated > 0);
    }

    #[test]
    fn reserving_tail_excludes_redundancy_bytes_from_both_coder_ends() {
        let data = [0x7a, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        let mut decoder = RangeDecoder::new(&data);
        assert_eq!(decoder.reserve_tail(3), Ok(5));
        assert_eq!(decoder.storage_len(), 5);
        assert_eq!(decoder.raw_bits(8), Ok(0x44));
        assert_eq!(decoder.raw_bits(8), Ok(0x33));
        assert_eq!(&data[5..], &[0x55, 0x66, 0x77]);
    }

    #[test]
    fn reserving_tail_allows_normative_range_lookahead_overlap() {
        let data = [0x7a, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        let mut decoder = RangeDecoder::new(&data);
        assert!(decoder.front > 3);
        assert_eq!(decoder.reserve_tail(5), Ok(3));
        assert_eq!(decoder.storage_len(), 3);
        assert_eq!(&data[3..], &[0x33, 0x44, 0x55, 0x66, 0x77]);
    }

    #[test]
    fn reserving_tail_rejects_the_whole_frame_and_late_raw_bit_use() {
        let data = [0u8; 4];
        let mut decoder = RangeDecoder::new(&data);
        assert_eq!(decoder.reserve_tail(4), Err(Error::InvalidPacket));
        assert_eq!(decoder.raw_bits(1), Ok(0));
        assert_eq!(decoder.reserve_tail(1), Err(Error::InvalidPacket));
    }

    #[test]
    fn encoder_decoder_round_trip_and_ranges_match() {
        let mut frame = [0u8; 64];
        let mut encoder = RangeEncoder::new(&mut frame);
        let icdf = [200, 100, 20, 0];
        encoder.encode_icdf(2, &icdf, 8).unwrap();
        let range_1 = encoder.range();
        let frac_1 = encoder.tell_frac();
        encoder.encode_bit_logp(true, 5).unwrap();
        let range_2 = encoder.range();
        let frac_2 = encoder.tell_frac();
        encoder.encode_uint(17, 31).unwrap();
        let range_3 = encoder.range();
        let frac_3 = encoder.tell_frac();
        encoder.encode_uint(65_537, 100_000).unwrap();
        let range_4 = encoder.range();
        let bytes = encoder.finish().unwrap();

        let mut decoder = RangeDecoder::new(&frame[..bytes]);
        assert_eq!(decoder.decode_icdf(&icdf, 8), Ok(2));
        assert_eq!(decoder.range(), range_1);
        assert_eq!(decoder.tell_frac(), frac_1);
        assert_eq!(decoder.decode_bit_logp(5), Ok(true));
        assert_eq!(decoder.range(), range_2);
        assert_eq!(decoder.tell_frac(), frac_2);
        assert_eq!(decoder.decode_uint(31), Ok(17));
        assert_eq!(decoder.range(), range_3);
        assert_eq!(decoder.tell_frac(), frac_3);
        assert_eq!(decoder.decode_uint(100_000), Ok(65_537));
        assert_eq!(decoder.range(), range_4);
    }

    #[test]
    fn compact_range_only_finalization_excludes_unused_capacity() {
        let mut frame = [0xa5u8; 64];
        let mut encoder = RangeEncoder::new(&mut frame);
        encoder.encode_bit_logp(true, 5).unwrap();
        let expected_range = encoder.range();
        let used = encoder.finish_compact().unwrap();
        assert!(used < frame.len());
        assert!(frame[used..].iter().all(|&byte| byte == 0));
        let mut decoder = RangeDecoder::new(&frame[..used]);
        assert_eq!(decoder.decode_bit_logp(5), Ok(true));
        assert_eq!(decoder.range(), expected_range);
    }

    #[test]
    fn encoder_rejects_overflow_without_writing_outside_buffer() {
        let mut frame = [0xa5u8; 1];
        let mut encoder = RangeEncoder::new(&mut frame);
        for _ in 0..64 {
            if encoder.encode(1, 2, 2).is_err() {
                return;
            }
        }
        assert!(encoder.finish().is_err());
    }
}
