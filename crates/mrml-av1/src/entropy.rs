//! AV1 arithmetic symbol decoder (specification section 8.2).

use crate::Error;

const CDF_PROB_TOP: u32 = 1 << 15;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

/// Arithmetic decoder for one AV1 tile data partition.
pub struct SymbolDecoder<'a> {
    data: &'a [u8],
    bit_position: usize,
    range: u32,
    value: u32,
    max_bits: i64,
    disable_cdf_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    pub fn new(data: &'a [u8], disable_cdf_update: bool) -> Result<Self, Error> {
        let num_bits = (data.len().saturating_mul(8)).min(15);
        let mut decoder = Self {
            data,
            bit_position: 0,
            range: CDF_PROB_TOP,
            value: 0,
            max_bits: (data.len() as i64).saturating_mul(8) - 15,
            disable_cdf_update,
        };
        let buffer = decoder.read_raw(num_bits as u8)?;
        let padded = buffer << (15 - num_bits);
        decoder.value = (CDF_PROB_TOP - 1) ^ padded;
        Ok(decoder)
    }

    pub fn read_bool(&mut self) -> Result<bool, Error> {
        let mut cdf = [1 << 14, 1 << 15, 0];
        Ok(self.read_symbol_internal(&mut cdf, false)? != 0)
    }

    pub fn read_literal(&mut self, bits: u8) -> Result<u64, Error> {
        if bits > 64 {
            return Err(Error::InvalidObu);
        }
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u64::from(self.read_bool()?);
        }
        Ok(value)
    }

    /// Decode and adapt a forward CDF. The final entry stores the adaptation
    /// count, so an N-symbol alphabet has N+1 entries.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<usize, Error> {
        self.read_symbol_internal(cdf, !self.disable_cdf_update)
    }

    pub fn read_ns(&mut self, n: u32) -> Result<u32, Error> {
        if n <= 1 {
            return if n == 1 {
                Ok(0)
            } else {
                Err(Error::InvalidObu)
            };
        }
        let width = 32 - (n - 1).leading_zeros();
        let split = (1u32 << width) - n;
        let value = self.read_literal((width - 1) as u8)? as u32;
        if value < split {
            Ok(value)
        } else {
            Ok((value << 1) - split + u32::from(self.read_bool()?))
        }
    }

    fn read_symbol_internal(&mut self, cdf: &mut [u16], update: bool) -> Result<usize, Error> {
        if cdf.len() < 3 {
            return Err(Error::InvalidObu);
        }
        let symbols = cdf.len() - 1;
        if cdf[symbols - 1] != CDF_PROB_TOP as u16 || cdf[symbols] > 32 {
            return Err(Error::InvalidObu);
        }
        if cdf[..symbols].windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(Error::InvalidObu);
        }

        let mut current = self.range;
        let mut previous;
        let mut symbol = 0usize;
        loop {
            previous = current;
            let frequency = CDF_PROB_TOP - u32::from(cdf[symbol]);
            current = ((self.range >> 8) * (frequency >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            current += EC_MIN_PROB * (symbols - symbol - 1) as u32;
            if self.value >= current {
                break;
            }
            symbol += 1;
            if symbol >= symbols {
                return Err(Error::InvalidObu);
            }
        }

        self.range = previous - current;
        self.value -= current;
        if self.range == 0 {
            return Err(Error::InvalidObu);
        }
        let bits = 15 - (31 - self.range.leading_zeros());
        self.range <<= bits;
        let num_bits = i64::from(bits).min(self.max_bits.max(0)) as u8;
        let new_data = self.read_raw(num_bits)?;
        let padded_data = new_data << (bits - u32::from(num_bits));
        self.value = padded_data ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= i64::from(bits);
        if self.max_bits < -14 {
            return Err(Error::InvalidObu);
        }

        if update {
            update_cdf(cdf, symbol)?;
        }
        Ok(symbol)
    }

    /// Validate and consume the arithmetic partition's trailing-one bit and
    /// zero padding. This must be called after the tile syntax is complete.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.max_bits < -14 {
            return Err(self.termination_error());
        }
        let lookback = 15i64.min(self.max_bits + 15);
        let trailing = (self.bit_position as i64)
            .checked_sub(lookback)
            .ok_or_else(|| self.termination_error())? as usize;
        let skip = self.max_bits.max(0) as usize;
        self.bit_position = self
            .bit_position
            .checked_add(skip)
            .ok_or(Error::LimitExceeded)?;
        if self.bit_position > self.data.len() * 8 || !bit_at(self.data, trailing)? {
            return Err(self.termination_error());
        }
        for position in trailing + 1..self.bit_position {
            if bit_at(self.data, position)? {
                return Err(self.termination_error());
            }
        }
        if !self.bit_position.is_multiple_of(8) {
            return Err(self.termination_error());
        }
        Ok(())
    }

    const fn termination_error(&self) -> Error {
        Error::InvalidTileTermination {
            bit_position: self.bit_position,
            max_bits: self.max_bits,
        }
    }

    fn read_raw(&mut self, count: u8) -> Result<u32, Error> {
        let end = self
            .bit_position
            .checked_add(count as usize)
            .ok_or(Error::LimitExceeded)?;
        if end > self.data.len() * 8 {
            return Err(Error::Truncated);
        }
        let mut value = 0u32;
        while self.bit_position < end {
            value = (value << 1) | u32::from(bit_at(self.data, self.bit_position)?);
            self.bit_position += 1;
        }
        Ok(value)
    }
}

pub fn update_cdf(cdf: &mut [u16], symbol: usize) -> Result<(), Error> {
    if cdf.len() < 3 || symbol >= cdf.len() - 1 {
        return Err(Error::InvalidObu);
    }
    let symbols = cdf.len() - 1;
    let count = cdf[symbols];
    if count > 32 {
        return Err(Error::InvalidObu);
    }
    let rate = 3
        + u32::from(count > 15)
        + u32::from(count > 31)
        + (31 - (symbols as u32).leading_zeros()).min(2);
    let mut target = 0u32;
    for (index, probability) in cdf[..symbols - 1].iter_mut().enumerate() {
        if index == symbol {
            target = CDF_PROB_TOP;
        }
        let old = u32::from(*probability);
        let adjusted = if target < old {
            old - ((old - target) >> rate)
        } else {
            old + ((target - old) >> rate)
        };
        *probability = adjusted as u16;
    }
    cdf[symbols] += u16::from(count < 32);
    Ok(())
}

fn bit_at(data: &[u8], position: usize) -> Result<bool, Error> {
    let byte = *data.get(position / 8).ok_or(Error::Truncated)?;
    Ok(byte & (1 << (7 - position % 8)) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_update_matches_normative_equations() {
        let mut cdf = [8192, 16384, 24576, 32768, 0];
        update_cdf(&mut cdf, 2).unwrap();
        assert_eq!(cdf, [7936, 15872, 24832, 32768, 1]);
    }

    #[test]
    fn rejects_invalid_cdfs() {
        let mut decoder = SymbolDecoder::new(&[0x80, 0], false).unwrap();
        assert_eq!(
            decoder.read_symbol(&mut [20, 10, 0]),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            decoder.read_symbol(&mut [10, 20, 0]),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn ns_one_consumes_no_bits() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        assert_eq!(decoder.read_ns(1), Ok(0));
    }
}
