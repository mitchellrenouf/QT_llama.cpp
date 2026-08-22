//! CELT split-angle resolution and entropy coding.
//!
//! A split angle describes the relative energy of two recursively coded
//! partitions.  Angles are represented on `[0, 16384]`, corresponding to
//! `[0, pi/2]`, so the entropy syntax remains entirely integer and bit exact.

use crate::{Error, RangeDecoder, RangeEncoder};

const EXP2_EIGHTHS_Q14: [i32; 8] = [
    16_384, 17_866, 19_483, 21_247, 23_170, 25_267, 27_554, 30_048,
];

/// Context needed to choose and code a CELT split angle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThetaConfig {
    /// Number of coefficients in each side of the split.
    pub dimensions: usize,
    /// Shape budget remaining at this split, in eighth-bit units.
    pub bits: i32,
    /// Base-2 logarithm of the partition dimensions, in eighth-bit units.
    pub pulse_cap: i32,
    /// True for a channel (mid/side) split, false for a time split.
    pub stereo: bool,
    /// Number of short blocks before this split.
    pub original_blocks: usize,
    /// Forces intensity stereo, for which no angle symbol is present.
    pub intensity: bool,
}

impl ThetaConfig {
    fn validate(self) -> Result<(), Error> {
        if self.dimensions == 0 || self.original_blocks == 0 || self.bits < 0 {
            return Err(Error::InvalidFrameSize);
        }
        Ok(())
    }
}

/// Decoded split-angle information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theta {
    /// Number of equal angle intervals selected by the budget.
    pub resolution: u16,
    /// Quantized interval index in `[0, resolution]`.
    pub index: u16,
    /// Angle on CELT's Q14 `[0, pi/2]` scale.
    pub angle_q14: u16,
    /// Cosine and sine gains for the first and second partitions.
    pub first_gain: f32,
    pub second_gain: f32,
    /// Bit-exact Q15 gains used by CELT's downstream bit allocation.
    pub first_gain_q15: i16,
    pub second_gain_q15: i16,
    /// Bit-exact Q11 `log2(second_gain/first_gain)` estimate.
    pub log2_tangent_q11: i32,
    /// Actual entropy cost of the angle symbol, in eighth-bit units.
    pub allocation: u32,
}

/// Computes CELT's angle resolution from the current split budget.
pub fn resolution(config: ThetaConfig) -> Result<u16, Error> {
    config.validate()?;
    if config.intensity {
        return Ok(1);
    }
    let mut degrees = config
        .dimensions
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::InvalidFrameSize)?;
    if config.stereo && config.dimensions == 2 {
        degrees -= 1;
    }
    let degrees = i32::try_from(degrees).map_err(|_| Error::InvalidFrameSize)?;
    let offset = config.pulse_cap / 2
        - if config.stereo && config.dimensions == 2 {
            16
        } else {
            4
        };
    let fair = (config.bits + degrees * offset) / degrees;
    let qb = fair.min(config.bits - config.pulse_cap - 32).min(64);
    if qb < 4 {
        return Ok(1);
    }
    let exponent = qb >> 3;
    let shift = u32::try_from(14 - exponent).map_err(|_| Error::InvalidPacket)?;
    let estimate = EXP2_EIGHTHS_Q14[usize::try_from(qb & 7).unwrap_or(0)] >> shift;
    u16::try_from((estimate + 1) & !1).map_err(|_| Error::InvalidPacket)
}

/// Encodes a pre-quantized split angle. `index` is on `[0, resolution]`.
pub fn encode(
    encoder: &mut RangeEncoder<'_>,
    config: ThetaConfig,
    index: u16,
) -> Result<Theta, Error> {
    let qn = resolution(config)?;
    if index > qn || (qn == 1 && index != 0) {
        return Err(Error::InvalidPacket);
    }
    let tell = encoder.tell_frac();
    if qn != 1 {
        encode_index(encoder, config, qn, index)?;
    }
    make_theta(qn, index, encoder.tell_frac() - tell)
}

/// Decodes one split angle using the same budget-derived resolution.
pub fn decode(decoder: &mut RangeDecoder<'_>, config: ThetaConfig) -> Result<Theta, Error> {
    let qn = resolution(config)?;
    let tell = decoder.tell_frac();
    let index = if qn == 1 {
        0
    } else {
        decode_index(decoder, config, qn)?
    };
    make_theta(qn, index, decoder.tell_frac() - tell)
}

fn encode_index(
    encoder: &mut RangeEncoder<'_>,
    config: ThetaConfig,
    qn: u16,
    index: u16,
) -> Result<(), Error> {
    let qn = u32::from(qn);
    let index = u32::from(index);
    if config.stereo && config.dimensions > 2 {
        let midpoint = qn / 2;
        let total = 3 * (midpoint + 1) + midpoint;
        let (low, high) = stepped_interval(index, midpoint);
        encoder.encode(low, high, total)
    } else if config.original_blocks > 1 || config.stereo {
        encoder.encode_uint(index, qn + 1)
    } else {
        let (low, high, total) = triangular_interval(index, qn);
        encoder.encode(low, high, total)
    }
}

fn decode_index(
    decoder: &mut RangeDecoder<'_>,
    config: ThetaConfig,
    qn: u16,
) -> Result<u16, Error> {
    let qn = u32::from(qn);
    let index = if config.stereo && config.dimensions > 2 {
        let midpoint = qn / 2;
        let total = 3 * (midpoint + 1) + midpoint;
        let frequency = decoder.decode(total)?;
        let index = if frequency < 3 * (midpoint + 1) {
            frequency / 3
        } else {
            midpoint + 1 + frequency - 3 * (midpoint + 1)
        };
        let (low, high) = stepped_interval(index, midpoint);
        decoder.update(low, high, total)?;
        index
    } else if config.original_blocks > 1 || config.stereo {
        decoder.decode_uint(qn + 1)?
    } else {
        let total = (qn / 2 + 1).pow(2);
        let frequency = decoder.decode(total)?;
        let lower_mass = (qn / 2) * (qn / 2 + 1) / 2;
        let index = if frequency < lower_mass {
            (integer_sqrt(8 * frequency + 1) - 1) / 2
        } else {
            (2 * (qn + 1) - integer_sqrt(8 * (total - frequency - 1) + 1)) / 2
        };
        let (low, high, _) = triangular_interval(index, qn);
        decoder.update(low, high, total)?;
        index
    };
    u16::try_from(index).map_err(|_| Error::InvalidPacket)
}

fn stepped_interval(index: u32, midpoint: u32) -> (u32, u32) {
    if index <= midpoint {
        (3 * index, 3 * (index + 1))
    } else {
        (
            index - 1 - midpoint + 3 * (midpoint + 1),
            index - midpoint + 3 * (midpoint + 1),
        )
    }
}

fn triangular_interval(index: u32, qn: u32) -> (u32, u32, u32) {
    let total = (qn / 2 + 1).pow(2);
    let frequency = if index <= qn / 2 {
        index + 1
    } else {
        qn + 1 - index
    };
    let low = if index <= qn / 2 {
        index * (index + 1) / 2
    } else {
        total - (qn + 1 - index) * (qn + 2 - index) / 2
    };
    (low, low + frequency, total)
}

fn integer_sqrt(value: u32) -> u32 {
    let mut bit = 1u32 << 30;
    let mut remainder = value;
    let mut root = 0;
    while bit > remainder {
        bit >>= 2;
    }
    while bit != 0 {
        if remainder >= root + bit {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn make_theta(qn: u16, index: u16, allocation: u32) -> Result<Theta, Error> {
    let angle = u32::from(index) * 16_384 / u32::from(qn);
    let angle_q14 = u16::try_from(angle).map_err(|_| Error::InvalidPacket)?;
    let radians = angle as f32 * core::f32::consts::FRAC_PI_2 / 16_384.0;
    let (first_gain_q15, second_gain_q15, log2_tangent_q11) = if angle == 0 {
        (32_767, 0, -16_384)
    } else if angle == 16_384 {
        (0, 32_767, 16_384)
    } else {
        let first = bitexact_cos(angle_q14);
        let second = bitexact_cos(16_384 - angle_q14);
        (first, second, bitexact_log2_tangent(second, first)?)
    };
    Ok(Theta {
        resolution: qn,
        index,
        angle_q14,
        first_gain: mrml_math::cos(radians),
        second_gain: mrml_math::sin(radians),
        first_gain_q15,
        second_gain_q15,
        log2_tangent_q11,
        allocation,
    })
}

/// CELT's platform-independent cosine approximation on its Q14 angle scale.
pub fn bitexact_cos(angle_q14: u16) -> i16 {
    let angle = i32::from(angle_q14.min(16_384));
    let mut squared = (4096 + angle * angle) >> 13;
    squared = squared.min(32_767);
    let inner = 8_277 + frac_mul(-626, squared);
    let polynomial = -7_651 + frac_mul(squared, inner);
    i16::try_from(1 + (32_767 - squared) + frac_mul(squared, polynomial)).unwrap_or(0)
}

/// Bit-exact Q11 approximation of `log2(sine/cosine)`.
pub fn bitexact_log2_tangent(sine: i16, cosine: i16) -> Result<i32, Error> {
    if sine <= 0 || cosine <= 0 {
        return Err(Error::InvalidPacket);
    }
    let mut sine = i32::from(sine);
    let mut cosine = i32::from(cosine);
    let sine_log = 32 - sine.leading_zeros() as i32;
    let cosine_log = 32 - cosine.leading_zeros() as i32;
    sine <<= 15 - sine_log;
    cosine <<= 15 - cosine_log;
    Ok(
        (sine_log - cosine_log) * 2048 + frac_mul(sine, frac_mul(sine, -2597) + 7932)
            - frac_mul(cosine, frac_mul(cosine, -2597) + 7932),
    )
}

/// Converts the Q11 tangent estimate into CELT's split-bit delta.
pub fn allocation_delta(dimensions: usize, theta: Theta) -> Result<i32, Error> {
    if dimensions == 0 {
        return Err(Error::InvalidFrameSize);
    }
    if theta.angle_q14 == 0 {
        Ok(-16_384)
    } else if theta.angle_q14 == 16_384 {
        Ok(16_384)
    } else {
        let tangent = bitexact_log2_tangent(theta.second_gain_q15, theta.first_gain_q15)?;
        let factor = i32::try_from(dimensions - 1).map_err(|_| Error::InvalidFrameSize)? << 7;
        Ok(frac_mul(factor, tangent))
    }
}

const fn frac_mul(left: i32, right: i32) -> i32 {
    ((left as i64 * right as i64 + 16_384) >> 15) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(config: ThetaConfig) {
        let qn = resolution(config).unwrap();
        for index in 0..=qn {
            if qn == 1 && index != 0 {
                continue;
            }
            let mut bytes = [0u8; 32];
            let mut encoder = RangeEncoder::new(&mut bytes);
            let encoded = encode(&mut encoder, config, index).unwrap();
            encoder.finish().unwrap();
            let decoded = decode(&mut RangeDecoder::new(&bytes), config).unwrap();
            assert_eq!(decoded.resolution, qn);
            assert_eq!(decoded.index, index);
            assert_eq!(decoded.angle_q14, encoded.angle_q14);
            assert_eq!(decoded.allocation, encoded.allocation);
        }
    }

    #[test]
    fn budget_resolution_is_even_bounded_and_monotonic() {
        let mut previous = 1;
        for bits in 0..=512 {
            let qn = resolution(ThetaConfig {
                dimensions: 8,
                bits,
                pulse_cap: 32,
                stereo: false,
                original_blocks: 1,
                intensity: false,
            })
            .unwrap();
            assert!(qn == 1 || qn.is_multiple_of(2));
            assert!(qn <= 256);
            assert!(qn >= previous);
            previous = qn;
        }
    }

    #[test]
    fn all_three_angle_models_round_trip() {
        for (stereo, dimensions, blocks) in [(true, 8, 1), (false, 8, 4), (false, 8, 1)] {
            round_trip(ThetaConfig {
                dimensions,
                bits: 320,
                pulse_cap: 32,
                stereo,
                original_blocks: blocks,
                intensity: false,
            });
        }
    }

    #[test]
    fn intensity_and_small_budgets_omit_the_symbol() {
        for intensity in [false, true] {
            let config = ThetaConfig {
                dimensions: 2,
                bits: 0,
                pulse_cap: 24,
                stereo: true,
                original_blocks: 1,
                intensity,
            };
            assert_eq!(resolution(config), Ok(1));
            round_trip(config);
        }
    }

    #[test]
    fn endpoints_produce_exact_partition_gains() {
        let config = ThetaConfig {
            dimensions: 8,
            bits: 320,
            pulse_cap: 32,
            stereo: false,
            original_blocks: 1,
            intensity: false,
        };
        let qn = resolution(config).unwrap();
        let mut bytes = [0u8; 32];
        let mut encoder = RangeEncoder::new(&mut bytes);
        let first = encode(&mut encoder, config, 0).unwrap();
        assert_eq!(first.angle_q14, 0);
        assert_eq!(first.first_gain, 1.0);
        assert_eq!(first.second_gain, 0.0);
        assert_eq!(first.first_gain_q15, 32_767);
        assert_eq!(allocation_delta(8, first), Ok(-16_384));

        let mut bytes = [0u8; 32];
        let mut encoder = RangeEncoder::new(&mut bytes);
        let second = encode(&mut encoder, config, qn).unwrap();
        assert_eq!(second.angle_q14, 16_384);
        assert!(second.first_gain.abs() < 0.000_01);
        assert!((second.second_gain - 1.0).abs() < 0.000_01);
        assert_eq!(second.second_gain_q15, 32_767);
        assert_eq!(allocation_delta(8, second), Ok(16_384));
    }

    #[test]
    fn fixed_angle_math_matches_normative_checkpoints() {
        assert_eq!(bitexact_cos(64), 32_767);
        assert_eq!(bitexact_cos(8_192), 23_171);
        assert_eq!(bitexact_cos(16_320), 200);
        for angle in 64..=8_192 {
            let first = bitexact_cos(angle);
            let second = bitexact_cos(16_384 - angle);
            assert_eq!(
                bitexact_log2_tangent(first, second),
                bitexact_log2_tangent(second, first).map(|value| -value)
            );
        }
    }
}
