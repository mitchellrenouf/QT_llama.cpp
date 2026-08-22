//! AV1 arithmetic symbol encoder foundation (specification section 8.2).
//!
//! Encoding records the normative rounded intervals in forward order, then
//! solves the decoder normalization equations backward at `finish`. This
//! keeps the implementation small and makes every emitted bit auditable.

use crate::{Error, entropy::update_cdf};
use mrml_runtime::Vector;

const CDF_PROB_TOP: u32 = 1 << 15;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Interval {
    current: u32,
    previous: u32,
    normalization_bits: u8,
    emitted: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SymbolEncoder {
    intervals: Vector<Interval>,
    disable_cdf_update: bool,
}

impl SymbolEncoder {
    pub fn new(disable_cdf_update: bool) -> Self {
        Self {
            intervals: Vector::new(),
            disable_cdf_update,
        }
    }

    pub fn write_bool(&mut self, value: bool) -> Result<(), Error> {
        let cdf = [1 << 14, 1 << 15, 0];
        self.record_symbol(&cdf, usize::from(value))
    }

    pub fn write_literal(&mut self, value: u64, bits: u8) -> Result<(), Error> {
        if bits > 64 || (bits < 64 && value >= 1u64 << bits) {
            return Err(Error::InvalidObu);
        }
        for shift in (0..bits).rev() {
            self.write_bool(value & (1u64 << shift) != 0)?;
        }
        Ok(())
    }

    pub fn write_symbol(&mut self, cdf: &mut [u16], symbol: usize) -> Result<(), Error> {
        self.record_symbol(cdf, symbol)?;
        if !self.disable_cdf_update {
            update_cdf(cdf, symbol)?;
        }
        Ok(())
    }

    pub fn write_ns(&mut self, value: u32, n: u32) -> Result<(), Error> {
        if n == 0 || value >= n {
            return Err(Error::InvalidObu);
        }
        if n == 1 {
            return Ok(());
        }
        let width = 32 - (n - 1).leading_zeros();
        let split = (1u32 << width) - n;
        if value < split {
            self.write_literal(u64::from(value), (width - 1) as u8)
        } else {
            let encoded = value + split;
            self.write_literal(u64::from(encoded >> 1), (width - 1) as u8)?;
            self.write_bool(encoded & 1 != 0)
        }
    }

    fn record_symbol(&mut self, cdf: &[u16], symbol: usize) -> Result<(), Error> {
        if cdf.len() < 3 || symbol >= cdf.len() - 1 {
            return Err(Error::InvalidObu);
        }
        let symbols = cdf.len() - 1;
        if cdf[symbols - 1] != CDF_PROB_TOP as u16
            || cdf[symbols] > 32
            || cdf[..symbols].windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(Error::InvalidObu);
        }
        let range = self.intervals.last().map_or(CDF_PROB_TOP, |interval| {
            (interval.previous - interval.current) << interval.normalization_bits
        });
        let threshold = |index: usize| {
            let frequency = CDF_PROB_TOP - u32::from(cdf[index]);
            (((range >> 8) * (frequency >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (symbols - index - 1) as u32
        };
        let previous = if symbol == 0 {
            range
        } else {
            threshold(symbol - 1)
        };
        let current = threshold(symbol);
        if current >= previous {
            return Err(Error::InvalidObu);
        }
        let selected_range = previous - current;
        let normalization_bits = (15 - (31 - selected_range.leading_zeros())) as u8;
        self.intervals
            .try_push(Interval {
                current,
                previous,
                normalization_bits,
                emitted: 0,
            })
            .map_err(|_| Error::LimitExceeded)
    }

    /// Emit enough bytes for a decoder to recover every recorded symbol.
    /// Arithmetic trailing-bit termination is a separate tile-packet stage.
    pub fn finish(mut self) -> Result<Vector<u8>, Error> {
        let value = self.solve(0)?;
        self.emit(value)
    }

    /// Emit a byte-aligned arithmetic partition with the normative trailing
    /// one bit and zero padding accepted by `SymbolDecoder::finish`.
    pub fn finish_tile(mut self) -> Result<Vector<u8>, Error> {
        let final_range = self.intervals.last().map_or(CDF_PROB_TOP, |interval| {
            (interval.previous - interval.current) << interval.normalization_bits
        });
        let normalization_bits = self
            .intervals
            .iter()
            .try_fold(0usize, |sum, interval| {
                sum.checked_add(usize::from(interval.normalization_bits))
            })
            .ok_or(Error::LimitExceeded)?;
        for final_value in 0..final_range {
            let Ok(initial_value) = self.solve(final_value) else {
                continue;
            };
            let bytes = self.emit(initial_value)?;
            if !bit_from(&bytes, normalization_bits)? {
                continue;
            }
            let mut valid = true;
            for position in normalization_bits + 1..bytes.len() * 8 {
                if bit_from(&bytes, position)? {
                    valid = false;
                    break;
                }
            }
            if valid {
                return Ok(bytes);
            }
        }
        Err(Error::InvalidObu)
    }

    fn solve(&mut self, final_value: u32) -> Result<u32, Error> {
        let mut value = final_value;
        for interval in self.intervals.iter_mut().rev() {
            let scale = 1u32 << interval.normalization_bits;
            let mask = scale - 1;
            interval.emitted = scale.wrapping_sub((value + 1) & mask) & mask;
            let quotient = (value + 1 + interval.emitted) >> interval.normalization_bits;
            value = interval
                .current
                .checked_add(quotient)
                .and_then(|result| result.checked_sub(1))
                .ok_or(Error::InvalidObu)?;
            if value < interval.current || value >= interval.previous {
                return Err(Error::InvalidObu);
            }
        }
        if value >= CDF_PROB_TOP {
            return Err(Error::InvalidObu);
        }
        Ok(value)
    }

    fn emit(&self, value: u32) -> Result<Vector<u8>, Error> {
        let total_bits = 15usize
            .checked_add(
                self.intervals
                    .iter()
                    .try_fold(0usize, |sum, interval| {
                        sum.checked_add(usize::from(interval.normalization_bits))
                    })
                    .ok_or(Error::LimitExceeded)?,
            )
            .ok_or(Error::LimitExceeded)?;
        let mut writer = OutputBits::new(total_bits.div_ceil(8))?;
        writer.write(CDF_PROB_TOP - 1 - value, 15)?;
        for interval in &self.intervals {
            writer.write(interval.emitted, interval.normalization_bits)?;
        }
        writer.finish()
    }
}

fn bit_from(bytes: &[u8], position: usize) -> Result<bool, Error> {
    let byte = *bytes.get(position / 8).ok_or(Error::Truncated)?;
    Ok(byte & (1 << (7 - position % 8)) != 0)
}

struct OutputBits {
    bytes: Vector<u8>,
    position: usize,
}

impl OutputBits {
    fn new(capacity: usize) -> Result<Self, Error> {
        Ok(Self {
            bytes: Vector::with_capacity(capacity).map_err(|_| Error::LimitExceeded)?,
            position: 0,
        })
    }

    fn write(&mut self, value: u32, bits: u8) -> Result<(), Error> {
        if bits > 32 || (bits < 32 && value >= 1u32 << bits) {
            return Err(Error::InvalidObu);
        }
        for shift in (0..bits).rev() {
            if self.position.is_multiple_of(8) {
                self.bytes.try_push(0).map_err(|_| Error::LimitExceeded)?;
            }
            let byte = self.position / 8;
            self.bytes[byte] |= (((value >> shift) & 1) as u8) << (7 - self.position % 8);
            self.position += 1;
        }
        Ok(())
    }

    fn finish(self) -> Result<Vector<u8>, Error> {
        Ok(self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::SymbolDecoder;

    #[test]
    fn adaptive_symbols_literals_and_ns_round_trip() {
        let mut encoder = SymbolEncoder::new(false);
        let mut encoder_cdf = [8192, 16384, 24576, 32768, 0];
        for symbol in [2, 0, 3, 1, 1, 2, 3, 0] {
            encoder.write_symbol(&mut encoder_cdf, symbol).unwrap();
        }
        encoder.write_literal(0b101101, 6).unwrap();
        for (value, n) in [(0, 1), (0, 3), (1, 3), (2, 3), (4, 5)] {
            encoder.write_ns(value, n).unwrap();
        }
        let bytes = encoder.finish().unwrap();
        let mut decoder = SymbolDecoder::new(&bytes, false).unwrap();
        let mut decoder_cdf = [8192, 16384, 24576, 32768, 0];
        for symbol in [2, 0, 3, 1, 1, 2, 3, 0] {
            assert_eq!(decoder.read_symbol(&mut decoder_cdf), Ok(symbol));
        }
        assert_eq!(decoder.read_literal(6), Ok(0b101101));
        for (value, n) in [(0, 1), (0, 3), (1, 3), (2, 3), (4, 5)] {
            assert_eq!(decoder.read_ns(n), Ok(value));
        }
        assert_eq!(encoder_cdf, decoder_cdf);
    }

    #[test]
    fn tile_termination_passes_decoder_finish_validation() {
        let mut encoder = SymbolEncoder::new(false);
        let mut encoder_cdf = [8192, 16384, 24576, 32768, 0];
        let symbols = [2, 0, 3, 1, 1, 2, 3, 0];
        for symbol in symbols {
            encoder.write_symbol(&mut encoder_cdf, symbol).unwrap();
        }
        let bytes = encoder.finish_tile().unwrap();
        let mut decoder = SymbolDecoder::new(&bytes, false).unwrap();
        let mut decoder_cdf = [8192, 16384, 24576, 32768, 0];
        for symbol in symbols {
            assert_eq!(decoder.read_symbol(&mut decoder_cdf), Ok(symbol));
        }
        assert_eq!(decoder.finish(), Ok(()));

        let empty = SymbolEncoder::new(false).finish_tile().unwrap();
        assert_eq!(SymbolDecoder::new(&empty, false).unwrap().finish(), Ok(()));
    }
}
