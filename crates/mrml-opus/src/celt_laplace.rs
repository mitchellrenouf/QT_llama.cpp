//! CELT's bounded signed Laplace entropy model.

use crate::{Error, RangeDecoder, RangeEncoder};

const TOTAL: u32 = 32_768;
const GUARANTEED_TAIL: u32 = 16;
const MIN_FREQUENCY: u32 = 1;

fn validate(zero_frequency: u32, decay_q15: u32) -> Result<(), Error> {
    if zero_frequency == 0
        || zero_frequency > TOTAL - 2 * GUARANTEED_TAIL * MIN_FREQUENCY
        || decay_q15 == 0
        || decay_q15 > 11_456
    {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

fn first_tail_frequency(zero_frequency: u32, decay_q15: u32) -> u32 {
    let available = TOTAL - 2 * GUARANTEED_TAIL * MIN_FREQUENCY - zero_frequency;
    (available * (16_384 - decay_q15)) >> 15
}

/// Encodes a signed value and returns the representable value actually coded.
/// Very large magnitudes are clamped into the model's unit-frequency tail.
pub fn encode(
    encoder: &mut RangeEncoder<'_>,
    value: i32,
    zero_frequency: u32,
    decay_q15: u32,
) -> Result<i32, Error> {
    validate(zero_frequency, decay_q15)?;
    let mut low = 0u32;
    let mut frequency = zero_frequency;
    let mut coded = value;
    if value != 0 {
        let negative = value < 0;
        let magnitude = value.unsigned_abs();
        low = zero_frequency;
        frequency = first_tail_frequency(zero_frequency, decay_q15);
        let mut represented = 1u32;
        while frequency > 0 && represented < magnitude {
            frequency *= 2;
            low = low
                .checked_add(frequency + 2 * MIN_FREQUENCY)
                .ok_or(Error::InvalidPacket)?;
            frequency = (frequency * decay_q15) >> 15;
            represented += 1;
        }
        if frequency == 0 {
            let available = TOTAL.checked_sub(low).ok_or(Error::InvalidPacket)?;
            let maximum_delta = ((available + u32::from(negative)) / 2).saturating_sub(1);
            let delta = magnitude.saturating_sub(represented).min(maximum_delta);
            low += (2 * delta + 1 - u32::from(negative)) * MIN_FREQUENCY;
            frequency = MIN_FREQUENCY.min(TOTAL - low);
            let actual_magnitude = represented + delta;
            coded = if negative {
                -(actual_magnitude as i32)
            } else {
                actual_magnitude as i32
            };
        } else {
            frequency += MIN_FREQUENCY;
            if !negative {
                low += frequency;
            }
        }
    }
    let high = low
        .checked_add(frequency)
        .ok_or(Error::InvalidPacket)?
        .min(TOTAL);
    if low >= high {
        return Err(Error::InvalidPacket);
    }
    encoder.encode(low, high, TOTAL)?;
    Ok(coded)
}

pub fn decode(
    decoder: &mut RangeDecoder<'_>,
    zero_frequency: u32,
    decay_q15: u32,
) -> Result<i32, Error> {
    validate(zero_frequency, decay_q15)?;
    let value = decoder.decode(TOTAL)?;
    let mut low = 0u32;
    let mut frequency = zero_frequency;
    let mut result = 0i32;
    if value >= zero_frequency {
        let mut magnitude = 1i32;
        low = zero_frequency;
        frequency = first_tail_frequency(zero_frequency, decay_q15) + MIN_FREQUENCY;
        while frequency > MIN_FREQUENCY && value >= low + 2 * frequency {
            frequency *= 2;
            low += frequency;
            frequency = ((frequency - 2 * MIN_FREQUENCY) * decay_q15) >> 15;
            frequency += MIN_FREQUENCY;
            magnitude += 1;
        }
        if frequency <= MIN_FREQUENCY {
            let delta = (value - low) / (2 * MIN_FREQUENCY);
            magnitude = magnitude
                .checked_add(delta as i32)
                .ok_or(Error::InvalidPacket)?;
            low += 2 * delta * MIN_FREQUENCY;
        }
        if value < low + frequency {
            result = -magnitude;
        } else {
            low += frequency;
            result = magnitude;
        }
    }
    let high = (low + frequency).min(TOTAL);
    decoder.update(low, high, TOTAL)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_values_and_unit_frequency_tails_round_trip() {
        for (zero, decay) in [(3_072, 11_456), (9_216, 8_192), (22_656, 640)] {
            for value in -128..=128 {
                let mut bytes = [0; 16];
                let mut encoder = RangeEncoder::new(&mut bytes);
                let coded = encode(&mut encoder, value, zero, decay).unwrap();
                let range = encoder.range();
                let tell = encoder.tell_frac();
                encoder.finish().unwrap();
                let mut decoder = RangeDecoder::new(&bytes);
                assert_eq!(decode(&mut decoder, zero, decay), Ok(coded));
                assert_eq!(decoder.range(), range);
                assert_eq!(decoder.tell_frac(), tell);
            }
        }
    }

    #[test]
    fn zero_uses_exact_requested_frequency() {
        let mut bytes = [0; 8];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode(&mut encoder, 0, 9_216, 8_128).unwrap();
        assert_eq!(encoder.range(), (1u32 << 31) / TOTAL * 9_216);
        encoder.finish().unwrap();
        assert_eq!(decode(&mut RangeDecoder::new(&bytes), 9_216, 8_128), Ok(0));
    }

    #[test]
    fn invalid_models_are_rejected() {
        let mut bytes = [0; 8];
        let mut encoder = RangeEncoder::new(&mut bytes);
        assert_eq!(encode(&mut encoder, 0, 0, 1), Err(Error::InvalidPacket));
        assert_eq!(encode(&mut encoder, 0, TOTAL, 1), Err(Error::InvalidPacket));
        assert_eq!(
            encode(&mut encoder, 0, 1, 11_457),
            Err(Error::InvalidPacket)
        );
    }
}
