//! CELT log-energy refinement from RFC 6716 section 4.3.2.2.

use crate::{Error, RangeDecoder, RangeEncoder, bands::BAND_COUNT, celt_laplace};

pub const MAX_CHANNELS: usize = 2;
const PREDICTION_Q15: [i16; 4] = [29_440, 26_112, 21_248, 16_384];
const BETA_Q15: [i16; 4] = [30_147, 22_282, 12_124, 6_554];
const BETA_INTRA_Q15: i16 = 4_915;
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Fixed Q8 zero-probability and decay pairs by LM, prediction mode, and band.
pub const COARSE_PROBABILITY: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

/// Fixed-size base-2 log-energy state, indexed by channel then critical band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogEnergies {
    values: [[f32; BAND_COUNT]; MAX_CHANNELS],
}

impl LogEnergies {
    pub const fn new() -> Self {
        Self {
            values: [[0.0; BAND_COUNT]; MAX_CHANNELS],
        }
    }
    pub const fn from_values(values: [[f32; BAND_COUNT]; MAX_CHANNELS]) -> Self {
        Self { values }
    }
    pub const fn values(&self) -> &[[f32; BAND_COUNT]; MAX_CHANNELS] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [[f32; BAND_COUNT]; MAX_CHANNELS] {
        &mut self.values
    }
}
impl Default for LogEnergies {
    fn default() -> Self {
        Self::new()
    }
}

/// Chooses integer coarse residuals nearest to a target envelope while
/// simulating the decoder's predictor exactly.
#[allow(clippy::needless_range_loop)] // Predictor state is band-major, then channel-major.
pub fn residuals_for_target(
    config: CoarseConfig,
    initial: &LogEnergies,
    target: &LogEnergies,
    output: &mut [[i16; BAND_COUNT]; 2],
) -> Result<(), Error> {
    config.validate()?;
    let mut state = *initial;
    let mut previous_frequency = [0.0f32; 2];
    output.fill([0; BAND_COUNT]);
    for band in config.start..config.end {
        for channel in 0..usize::from(config.channels) {
            let coefficient = if config.intra {
                0.0
            } else {
                f32::from(PREDICTION_Q15[usize::from(config.lm)]) / 32_768.0
            };
            let predicted =
                coefficient * state.values[channel][band].max(-9.0) + previous_frequency[channel];
            let delta = mrml_math::round(target.values[channel][band] - predicted)
                .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            output[channel][band] = delta;
            update_coarse_state(
                &mut state,
                &mut previous_frequency,
                config,
                band,
                channel,
                i32::from(delta),
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoarseConfig {
    pub channels: u8,
    pub lm: u8,
    pub intra: bool,
    pub start: usize,
    pub end: usize,
    pub frame_bytes: usize,
}

impl CoarseConfig {
    fn validate(self) -> Result<(), Error> {
        if !(1..=2).contains(&self.channels)
            || self.lm > 3
            || self.start >= self.end
            || self.end > BAND_COUNT
            || self.frame_bytes == 0
            || self.frame_bytes > crate::MAX_FRAME_BYTES
        {
            return Err(Error::InvalidPacket);
        }
        Ok(())
    }
}

fn update_coarse_state(
    state: &mut LogEnergies,
    previous_frequency: &mut [f32; 2],
    config: CoarseConfig,
    band: usize,
    channel: usize,
    delta: i32,
) {
    let coefficient = if config.intra {
        0.0
    } else {
        f32::from(PREDICTION_Q15[usize::from(config.lm)]) / 32_768.0
    };
    let beta = if config.intra {
        f32::from(BETA_INTRA_Q15) / 32_768.0
    } else {
        f32::from(BETA_Q15[usize::from(config.lm)]) / 32_768.0
    };
    let old = state.values[channel][band].max(-9.0);
    state.values[channel][band] = coefficient * old + previous_frequency[channel] + delta as f32;
    previous_frequency[channel] += (1.0 - beta) * delta as f32;
}

/// Decodes coarse energies and records the integer prediction residuals.
#[allow(clippy::needless_range_loop)] // Bitstream order is band-major, then channel-major.
pub fn decode_coarse(
    decoder: &mut RangeDecoder<'_>,
    config: CoarseConfig,
    state: &mut LogEnergies,
    residuals: &mut [[i16; BAND_COUNT]; 2],
) -> Result<(), Error> {
    config.validate()?;
    let budget = i32::try_from(config.frame_bytes * 8).map_err(|_| Error::InvalidPacket)?;
    let model = &COARSE_PROBABILITY[usize::from(config.lm)][usize::from(config.intra)];
    let mut previous_frequency = [0.0; 2];
    residuals.fill([0; BAND_COUNT]);
    for band in config.start..config.end {
        for channel in 0..usize::from(config.channels) {
            let remaining = budget - decoder.tell() as i32;
            let delta = if remaining >= 15 {
                let parameter = band.min(20) * 2;
                celt_laplace::decode(
                    decoder,
                    u32::from(model[parameter]) << 7,
                    u32::from(model[parameter + 1]) << 6,
                )?
            } else if remaining >= 2 {
                let symbol = decoder.decode_icdf(&SMALL_ENERGY_ICDF, 2)? as i32;
                (symbol >> 1) ^ -(symbol & 1)
            } else if remaining >= 1 {
                -i32::from(decoder.decode_bit_logp(1)?)
            } else {
                -1
            };
            residuals[channel][band] = i16::try_from(delta).map_err(|_| Error::InvalidPacket)?;
            update_coarse_state(state, &mut previous_frequency, config, band, channel, delta);
        }
    }
    Ok(())
}

/// Encoder mirror of [`decode_coarse`]. `coded_residuals` receives any tail
/// values clamped by the finite Laplace alphabet.
#[allow(clippy::needless_range_loop)] // Must mirror the decoder's normative nesting.
pub fn encode_coarse(
    encoder: &mut RangeEncoder<'_>,
    config: CoarseConfig,
    requested_residuals: &[[i16; BAND_COUNT]; 2],
    state: &mut LogEnergies,
    coded_residuals: &mut [[i16; BAND_COUNT]; 2],
) -> Result<(), Error> {
    config.validate()?;
    let budget = i32::try_from(config.frame_bytes * 8).map_err(|_| Error::InvalidPacket)?;
    let model = &COARSE_PROBABILITY[usize::from(config.lm)][usize::from(config.intra)];
    let mut previous_frequency = [0.0; 2];
    coded_residuals.fill([0; BAND_COUNT]);
    for band in config.start..config.end {
        for channel in 0..usize::from(config.channels) {
            let requested = i32::from(requested_residuals[channel][band]);
            let remaining = budget - encoder.tell() as i32;
            let delta = if remaining >= 15 {
                let parameter = band.min(20) * 2;
                celt_laplace::encode(
                    encoder,
                    requested,
                    u32::from(model[parameter]) << 7,
                    u32::from(model[parameter + 1]) << 6,
                )?
            } else if remaining >= 2 {
                let symbol = match requested {
                    0 => 0,
                    -1 => 1,
                    1 => 2,
                    _ => return Err(Error::InvalidPacket),
                };
                encoder.encode_icdf(symbol, &SMALL_ENERGY_ICDF, 2)?;
                requested
            } else if remaining >= 1 {
                if !matches!(requested, -1 | 0) {
                    return Err(Error::InvalidPacket);
                }
                encoder.encode_bit_logp(requested == -1, 1)?;
                requested
            } else {
                if requested != -1 {
                    return Err(Error::InvalidPacket);
                }
                -1
            };
            coded_residuals[channel][band] = delta as i16;
            update_coarse_state(state, &mut previous_frequency, config, band, channel, delta);
        }
    }
    Ok(())
}

fn validate(
    channels: u8,
    start: usize,
    end: usize,
    fine_bits: &[u8; BAND_COUNT],
) -> Result<(), Error> {
    if !(1..=2).contains(&channels)
        || start >= end
        || end > BAND_COUNT
        || fine_bits[start..end].iter().any(|&bits| bits > 8)
    {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

/// Decodes the fixed-rate fine energy symbols and applies their centered
/// correction to an already coarsely quantized energy envelope.
pub fn decode_fine(
    decoder: &mut RangeDecoder<'_>,
    channels: u8,
    start: usize,
    end: usize,
    fine_bits: &[u8; BAND_COUNT],
    energies: &mut LogEnergies,
) -> Result<(), Error> {
    validate(channels, start, end, fine_bits)?;
    for (band, &bits) in fine_bits.iter().enumerate().take(end).skip(start) {
        if bits == 0 {
            continue;
        }
        let denominator = (1u32 << bits) as f32;
        for channel in 0..usize::from(channels) {
            let symbol = decoder.raw_bits(bits)?;
            energies.values[channel][band] += (symbol as f32 + 0.5) / denominator - 0.5;
        }
    }
    Ok(())
}

/// Quantizes target log energies to the nearest fine-energy cell, emits the
/// symbols, and updates `energies` exactly as the decoder will.
pub fn encode_fine(
    encoder: &mut RangeEncoder<'_>,
    channels: u8,
    start: usize,
    end: usize,
    fine_bits: &[u8; BAND_COUNT],
    target: &LogEnergies,
    energies: &mut LogEnergies,
) -> Result<(), Error> {
    validate(channels, start, end, fine_bits)?;
    for (band, &bits) in fine_bits.iter().enumerate().take(end).skip(start) {
        if bits == 0 {
            continue;
        }
        let levels = 1u32 << bits;
        let scale = levels as f32;
        for channel in 0..usize::from(channels) {
            let residual = target.values[channel][band] - energies.values[channel][band];
            let symbol = ((residual + 0.5) * scale) as i32;
            let symbol = symbol.clamp(0, levels as i32 - 1) as u32;
            encoder.raw_bits(symbol, bits)?;
            energies.values[channel][band] += (symbol as f32 + 0.5) / scale - 0.5;
        }
    }
    Ok(())
}

/// Applies the optional extra fine bit to priority-0 bands first and then
/// priority-1 bands. Returns the number of whole bits consumed.
#[allow(clippy::too_many_arguments)] // Mirrors the normative state machine without hidden storage.
pub fn decode_final(
    decoder: &mut RangeDecoder<'_>,
    channels: u8,
    start: usize,
    end: usize,
    fine_bits: &[u8; BAND_COUNT],
    priority: &[u8; BAND_COUNT],
    available_bits: usize,
    energies: &mut LogEnergies,
) -> Result<usize, Error> {
    validate(channels, start, end, fine_bits)?;
    if priority[start..end].iter().any(|&value| value > 1) {
        return Err(Error::InvalidPacket);
    }
    let channels = usize::from(channels);
    let mut used = 0;
    for wanted in 0..=1 {
        for band in start..end {
            if priority[band] != wanted || available_bits - used < channels {
                continue;
            }
            let magnitude = 0.25 / (1u32 << fine_bits[band]) as f32;
            for channel in 0..channels {
                let high = decoder.raw_bits(1)? != 0;
                energies.values[channel][band] += if high { magnitude } else { -magnitude };
            }
            used += channels;
        }
    }
    Ok(used)
}

/// Encoder mirror of [`decode_final`].
#[allow(clippy::too_many_arguments)] // Mirrors the decoder inputs plus its target envelope.
pub fn encode_final(
    encoder: &mut RangeEncoder<'_>,
    channels: u8,
    start: usize,
    end: usize,
    fine_bits: &[u8; BAND_COUNT],
    priority: &[u8; BAND_COUNT],
    available_bits: usize,
    target: &LogEnergies,
    energies: &mut LogEnergies,
) -> Result<usize, Error> {
    validate(channels, start, end, fine_bits)?;
    if priority[start..end].iter().any(|&value| value > 1) {
        return Err(Error::InvalidPacket);
    }
    let channels = usize::from(channels);
    let mut used = 0;
    for wanted in 0..=1 {
        for band in start..end {
            if priority[band] != wanted || available_bits - used < channels {
                continue;
            }
            let magnitude = 0.25 / (1u32 << fine_bits[band]) as f32;
            for channel in 0..channels {
                let high = target.values[channel][band] >= energies.values[channel][band];
                encoder.raw_bits(u32::from(high), 1)?;
                energies.values[channel][band] += if high { magnitude } else { -magnitude };
            }
            used += channels;
        }
    }
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fine_depth_and_channel_count_round_trips() {
        for channels in 1..=2 {
            for bits in 0..=8 {
                let mut fine_bits = [0; BAND_COUNT];
                fine_bits[..5].fill(bits);
                let mut target_values = [[0.0; BAND_COUNT]; 2];
                for (channel, values) in target_values.iter_mut().enumerate().take(channels) {
                    for (band, value) in values.iter_mut().enumerate().take(5) {
                        *value = (band as f32 - 2.0) * 0.17 + channel as f32 * 0.09;
                    }
                }
                let target = LogEnergies::from_values(target_values);
                let mut encoded = LogEnergies::new();
                let mut bytes = [0; 64];
                let mut range = RangeEncoder::new(&mut bytes);
                encode_fine(
                    &mut range,
                    channels as u8,
                    0,
                    5,
                    &fine_bits,
                    &target,
                    &mut encoded,
                )
                .unwrap();
                range.finish().unwrap();
                let mut decoded = LogEnergies::new();
                decode_fine(
                    &mut RangeDecoder::new(&bytes),
                    channels as u8,
                    0,
                    5,
                    &fine_bits,
                    &mut decoded,
                )
                .unwrap();
                assert_eq!(decoded, encoded);
            }
        }
    }

    #[test]
    fn final_bits_follow_priority_then_band_order() {
        let fine_bits = [2; BAND_COUNT];
        let mut priority = [1; BAND_COUNT];
        priority[1] = 0;
        priority[3] = 0;
        let target = LogEnergies::from_values([[1.0; BAND_COUNT], [-1.0; BAND_COUNT]]);
        let mut encoded = LogEnergies::new();
        let mut bytes = [0; 16];
        let mut range = RangeEncoder::new(&mut bytes);
        assert_eq!(
            encode_final(
                &mut range,
                2,
                0,
                5,
                &fine_bits,
                &priority,
                6,
                &target,
                &mut encoded
            ),
            Ok(6)
        );
        range.finish().unwrap();
        let mut decoded = LogEnergies::new();
        assert_eq!(
            decode_final(
                &mut RangeDecoder::new(&bytes),
                2,
                0,
                5,
                &fine_bits,
                &priority,
                6,
                &mut decoded
            ),
            Ok(6)
        );
        assert_eq!(decoded, encoded);
        assert_ne!(decoded.values()[0][1], 0.0);
        assert_ne!(decoded.values()[0][3], 0.0);
        assert_ne!(decoded.values()[0][0], 0.0);
        assert_eq!(decoded.values()[0][2], 0.0);
    }

    #[test]
    fn malformed_ranges_and_allocations_are_rejected() {
        let mut bits = [0; BAND_COUNT];
        bits[0] = 9;
        let mut energies = LogEnergies::new();
        let bytes = [0; 8];
        assert_eq!(
            decode_fine(
                &mut RangeDecoder::new(&bytes),
                1,
                0,
                1,
                &bits,
                &mut energies
            ),
            Err(Error::InvalidPacket)
        );
        let mut priority = [0; BAND_COUNT];
        priority[0] = 2;
        bits[0] = 0;
        assert_eq!(
            decode_final(
                &mut RangeDecoder::new(&bytes),
                1,
                0,
                1,
                &bits,
                &priority,
                1,
                &mut energies
            ),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn coarse_models_round_trip_every_frame_size_and_prediction_mode() {
        for lm in 0..=3 {
            for intra in [false, true] {
                let config = CoarseConfig {
                    channels: 2,
                    lm,
                    intra,
                    start: 0,
                    end: BAND_COUNT,
                    frame_bytes: 256,
                };
                let mut requested = [[0; BAND_COUNT]; 2];
                for (channel, bands) in requested.iter_mut().enumerate() {
                    for (band, value) in bands.iter_mut().enumerate() {
                        *value = ((band + channel * 2) % 7) as i16 - 3;
                    }
                }
                let initial = LogEnergies::from_values([[1.25; BAND_COUNT], [-2.5; BAND_COUNT]]);
                let mut encoded_state = initial;
                let mut coded = [[0; BAND_COUNT]; 2];
                let mut bytes = [0; 256];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_coarse(
                    &mut encoder,
                    config,
                    &requested,
                    &mut encoded_state,
                    &mut coded,
                )
                .unwrap();
                let tell = encoder.tell_frac();
                encoder.finish().unwrap();
                let mut decoded_state = initial;
                let mut decoded = [[0; BAND_COUNT]; 2];
                let mut decoder = RangeDecoder::new(&bytes);
                decode_coarse(&mut decoder, config, &mut decoded_state, &mut decoded).unwrap();
                assert_eq!(decoded, coded);
                assert_eq!(decoded_state, encoded_state);
                assert_eq!(decoder.tell_frac(), tell);
            }
        }
    }

    #[test]
    fn coarse_low_budget_fallbacks_remain_symmetric() {
        let config = CoarseConfig {
            channels: 1,
            lm: 0,
            intra: true,
            start: 0,
            end: 4,
            frame_bytes: 1,
        };
        let requested = [[-1; BAND_COUNT], [0; BAND_COUNT]];
        let mut encoded_state = LogEnergies::new();
        let mut coded = [[0; BAND_COUNT]; 2];
        let mut bytes = [0; 8];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_coarse(
            &mut encoder,
            config,
            &requested,
            &mut encoded_state,
            &mut coded,
        )
        .unwrap();
        encoder.finish().unwrap();
        let mut decoded_state = LogEnergies::new();
        let mut decoded = [[0; BAND_COUNT]; 2];
        decode_coarse(
            &mut RangeDecoder::new(&bytes),
            config,
            &mut decoded_state,
            &mut decoded,
        )
        .unwrap();
        assert_eq!(decoded, coded);
        assert_eq!(decoded_state, encoded_state);
    }
}
