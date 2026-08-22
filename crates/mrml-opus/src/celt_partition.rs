//! Recursive CELT mono-partition shape decoding.

use crate::{
    Error, RangeDecoder, RangeEncoder,
    celt_theta::{
        ThetaConfig, allocation_delta, decode as decode_theta, encode as encode_theta,
        resolution as theta_resolution,
    },
    pvq,
};

/// Mutable budget and deterministic-noise state shared by all band partitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionState {
    /// Frame shape budget still available, in eighth-bit units.
    pub remaining_bits: i32,
    pub seed: u32,
}

/// Parameters for one recursively decodable mono band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartitionConfig {
    /// CELT critical-band index used by the normative pulse-cost cache.
    pub band: usize,
    /// Budget assigned to this partition, in eighth-bit units.
    pub bits: i32,
    /// Maximum remaining split depth. CELT starts this at the frame LM.
    pub lm: i8,
    /// Number of time blocks represented by this vector.
    pub blocks: usize,
    pub spread: u8,
    pub gain: f32,
}

/// Joint-stereo controls for one CELT band.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StereoConfig {
    /// Intensity stereo omits the theta symbol and codes only a combined shape.
    pub intensity: bool,
    /// Prevents phase inversion when the output may subsequently be downmixed.
    pub disable_inversion: bool,
}

fn theta_pulse_cap(band: usize, lm: i8) -> Result<i32, Error> {
    pvq::band_log_n(band, lm)
}

const fn inversion_is_coded(
    resolution: u16,
    bits: i32,
    remaining_bits: i32,
    disabled: bool,
) -> bool {
    resolution == 1 && bits > 16 && remaining_bits > 16 && !disabled
}

/// Decodes one CELT band, recursively splitting codebooks that exceed the
/// codec's 32-bit PVQ limit. Returns one collapse bit per time block.
pub fn decode(
    decoder: &mut RangeDecoder<'_>,
    config: PartitionConfig,
    state: &mut PartitionState,
    output: &mut [f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    validate(config, state, output, pulse_workspace, recurrence_scratch)?;
    decode_inner(
        decoder,
        config,
        state,
        output,
        pulse_workspace,
        recurrence_scratch,
    )
}

/// Encodes one normalized mono band using the decoder's recursive allocation
/// and rebalancing rules. `pulse_workspace` receives the quantized PVQ vector.
pub fn encode(
    encoder: &mut RangeEncoder<'_>,
    config: PartitionConfig,
    state: &mut PartitionState,
    input: &[f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    validate(config, state, input, pulse_workspace, recurrence_scratch)?;
    encode_inner(
        encoder,
        config,
        state,
        input,
        pulse_workspace,
        recurrence_scratch,
    )
}

fn encode_inner(
    encoder: &mut RangeEncoder<'_>,
    mut config: PartitionConfig,
    state: &mut PartitionState,
    input: &[f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let maximum = maximum_leaf_cost(config.band, config.lm, recurrence_scratch)?;
    if config.lm >= 0 && input.len() > 2 && config.bits > i32::from(maximum) + 12 {
        let half = input.len() / 2;
        if half * 2 != input.len() {
            return Err(Error::InvalidFrameSize);
        }
        let original_blocks = config.blocks;
        config.lm -= 1;
        if config.blocks != 1 {
            config.blocks = config.blocks.div_ceil(2);
        }
        let pulse_cap = theta_pulse_cap(config.band, config.lm)?;
        let theta_config = ThetaConfig {
            dimensions: half,
            bits: config.bits,
            pulse_cap,
            stereo: false,
            original_blocks,
            intensity: false,
        };
        let resolution = theta_resolution(theta_config)?;
        let index = choose_theta_index(&input[..half], &input[half..], resolution);
        let theta = encode_theta(encoder, theta_config, index)?;
        let allocation = i32::try_from(theta.allocation).map_err(|_| Error::InvalidPacket)?;
        config.bits = config
            .bits
            .checked_sub(allocation)
            .ok_or(Error::InvalidPacket)?;
        state.remaining_bits = state
            .remaining_bits
            .checked_sub(allocation)
            .ok_or(Error::InvalidPacket)?;
        let mut delta = allocation_delta(half, theta)?;
        if original_blocks > 1 && theta.angle_q14 != 0 && theta.angle_q14 != 16_384 {
            if theta.angle_q14 > 8_192 {
                let shift = u32::try_from(4 - config.lm).map_err(|_| Error::InvalidPacket)?;
                delta = delta
                    .checked_sub(delta >> shift)
                    .ok_or(Error::InvalidPacket)?;
            } else {
                let shift = u32::try_from(5 - config.lm).map_err(|_| Error::InvalidPacket)?;
                let masking = i32::try_from(half)
                    .map_err(|_| Error::InvalidFrameSize)?
                    .checked_shl(3)
                    .ok_or(Error::InvalidPacket)?
                    >> shift;
                delta = delta
                    .checked_add(masking)
                    .ok_or(Error::InvalidPacket)?
                    .min(0);
            }
        }
        let first_bits = ((config.bits - delta) / 2).clamp(0, config.bits);
        let second_bits = config.bits - first_bits;
        let (first_input, second_input) = input.split_at(half);
        let (first_pulses, second_pulses) = pulse_workspace.split_at_mut(half);
        let starting_budget = state.remaining_bits;
        let (first_mask, second_mask) = if first_bits >= second_bits {
            let first_mask = encode_inner(
                encoder,
                PartitionConfig {
                    bits: first_bits,
                    gain: config.gain * f32::from(theta.first_gain_q15) / 32_768.0,
                    ..config
                },
                state,
                first_input,
                first_pulses,
                recurrence_scratch,
            )?;
            let rebalance = first_bits - (starting_budget - state.remaining_bits);
            let adjusted = if rebalance > 24 && theta.angle_q14 != 0 {
                second_bits + rebalance - 24
            } else {
                second_bits
            };
            let second_mask = encode_inner(
                encoder,
                PartitionConfig {
                    bits: adjusted,
                    gain: config.gain * f32::from(theta.second_gain_q15) / 32_768.0,
                    ..config
                },
                state,
                second_input,
                second_pulses,
                recurrence_scratch,
            )?;
            (first_mask, second_mask)
        } else {
            let second_mask = encode_inner(
                encoder,
                PartitionConfig {
                    bits: second_bits,
                    gain: config.gain * f32::from(theta.second_gain_q15) / 32_768.0,
                    ..config
                },
                state,
                second_input,
                second_pulses,
                recurrence_scratch,
            )?;
            let rebalance = second_bits - (starting_budget - state.remaining_bits);
            let adjusted = if rebalance > 24 && theta.angle_q14 != 16_384 {
                first_bits + rebalance - 24
            } else {
                first_bits
            };
            let first_mask = encode_inner(
                encoder,
                PartitionConfig {
                    bits: adjusted,
                    gain: config.gain * f32::from(theta.first_gain_q15) / 32_768.0,
                    ..config
                },
                state,
                first_input,
                first_pulses,
                recurrence_scratch,
            )?;
            (first_mask, second_mask)
        };
        return Ok(first_mask
            | second_mask
                .checked_shl((original_blocks / 2) as u32)
                .unwrap_or(0));
    }
    encode_leaf(
        encoder,
        config,
        state,
        input,
        pulse_workspace,
        recurrence_scratch,
    )
}

fn encode_leaf(
    encoder: &mut RangeEncoder<'_>,
    config: PartitionConfig,
    state: &mut PartitionState,
    input: &[f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let target = config.bits.max(0);
    let available = state.remaining_bits.max(0);
    let pulses = pvq::pulses_for_band_target(
        config.band,
        config.lm,
        u16::try_from(target.min(i32::from(u16::MAX))).unwrap_or(u16::MAX),
        u16::try_from(available.min(i32::from(u16::MAX))).unwrap_or(u16::MAX),
        recurrence_scratch,
    )?;
    quantize_pulses(input, pulses, pulse_workspace)?;
    if pulses == 0 {
        return Ok(0);
    }
    let used = i32::from(pvq::band_pulse_cost(
        config.band,
        config.lm,
        pulses,
        recurrence_scratch,
    )?);
    state.remaining_bits = state
        .remaining_bits
        .checked_sub(used)
        .ok_or(Error::InvalidPacket)?;
    pvq::encode_range(encoder, pulses, pulse_workspace, recurrence_scratch)?;
    collapse_mask(pulse_workspace, config.blocks)
}

fn choose_theta_index(first: &[f32], second: &[f32], resolution: u16) -> u16 {
    if resolution <= 1 {
        return 0;
    }
    let first_energy = first.iter().fold(0.0, |sum, value| sum + value * value);
    let second_energy = second.iter().fold(0.0, |sum, value| sum + value * value);
    if first_energy + second_energy == 0.0 {
        return resolution / 2;
    }
    let target = mrml_math::sqrt(first_energy / (first_energy + second_energy));
    let mut best = 0;
    let mut best_error = f32::MAX;
    for index in 0..=resolution {
        let angle = core::f32::consts::FRAC_PI_2 * index as f32 / resolution as f32;
        let error = (mrml_math::cos(angle) - target).abs();
        if error < best_error {
            best = index;
            best_error = error;
        }
    }
    best
}

fn quantize_pulses(input: &[f32], pulses: usize, output: &mut [i32]) -> Result<(), Error> {
    if input.len() != output.len() {
        return Err(Error::InvalidFrameSize);
    }
    output.fill(0);
    if pulses == 0 {
        return Ok(());
    }
    let total = input.iter().fold(0.0, |sum, value| sum + value.abs());
    if total == 0.0 {
        output[0] = pulses as i32;
        return Ok(());
    }
    for _ in 0..pulses {
        let mut best = 0;
        let mut best_score = f32::MIN;
        for (index, &value) in input.iter().enumerate() {
            let score = value.abs() * pulses as f32 / total - output[index].unsigned_abs() as f32;
            if score > best_score {
                best = index;
                best_score = score;
            }
        }
        output[best] += if input[best] < 0.0 { -1 } else { 1 };
    }
    Ok(())
}

/// Decodes one jointly coded stereo band and converts its mid/side shapes back
/// to independently normalized left and right shapes.
#[allow(clippy::too_many_arguments)]
pub fn decode_stereo(
    decoder: &mut RangeDecoder<'_>,
    config: PartitionConfig,
    stereo: StereoConfig,
    state: &mut PartitionState,
    left: &mut [f32],
    right: &mut [f32],
    left_pulses: &mut [i32],
    right_pulses: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    validate(config, state, left, left_pulses, recurrence_scratch)?;
    validate(config, state, right, right_pulses, recurrence_scratch)?;
    if left.len() != right.len() {
        return Err(Error::InvalidFrameSize);
    }
    let dimensions = left.len();
    let pulse_cap = theta_pulse_cap(config.band, config.lm)?;
    let tell = decoder.tell_frac();
    let theta = decode_theta(
        decoder,
        ThetaConfig {
            dimensions,
            bits: config.bits,
            pulse_cap,
            stereo: true,
            original_blocks: config.blocks,
            intensity: stereo.intensity,
        },
    )?;
    let mut inverted = false;
    if inversion_is_coded(
        theta.resolution,
        config.bits,
        state.remaining_bits,
        stereo.disable_inversion,
    ) {
        inverted = decoder.decode_bit_logp(2)?;
    }
    let allocation = i32::try_from(decoder.tell_frac() - tell).map_err(|_| Error::InvalidPacket)?;
    let bits = config
        .bits
        .checked_sub(allocation)
        .ok_or(Error::InvalidPacket)?;
    state.remaining_bits = state
        .remaining_bits
        .checked_sub(allocation)
        .ok_or(Error::InvalidPacket)?;

    if dimensions == 2 {
        return decode_stereo_two_bin(
            decoder,
            config,
            theta,
            bits,
            inverted,
            state,
            left,
            right,
            left_pulses,
            right_pulses,
            recurrence_scratch,
        );
    }

    let delta = allocation_delta(dimensions, theta)?;
    let mid_bits = ((bits - delta) / 2).clamp(0, bits);
    let side_bits = bits - mid_bits;
    let starting_budget = state.remaining_bits;
    let (mid_mask, side_mask) = if mid_bits >= side_bits {
        let mid_mask = decode(
            decoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )?;
        let rebalance = mid_bits - (starting_budget - state.remaining_bits);
        let side_bits = if rebalance > 24 && theta.angle_q14 != 0 {
            side_bits + rebalance - 24
        } else {
            side_bits
        };
        let side_mask = decode(
            decoder,
            PartitionConfig {
                bits: side_bits,
                gain: f32::from(theta.second_gain_q15) / 32_768.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )?;
        (mid_mask, side_mask)
    } else {
        let side_mask = decode(
            decoder,
            PartitionConfig {
                bits: side_bits,
                gain: f32::from(theta.second_gain_q15) / 32_768.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )?;
        let rebalance = side_bits - (starting_budget - state.remaining_bits);
        let mid_bits = if rebalance > 24 && theta.angle_q14 != 16_384 {
            mid_bits + rebalance - 24
        } else {
            mid_bits
        };
        let mid_mask = decode(
            decoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )?;
        (mid_mask, side_mask)
    };
    stereo_merge(left, right, f32::from(theta.first_gain_q15) / 32_768.0)?;
    if inverted {
        for value in right {
            *value = -*value;
        }
    }
    Ok(mid_mask | side_mask)
}

/// Encodes one jointly coded stereo band. The input vectors are converted in
/// place from normalized left/right shapes to the quantized mid/side domain.
#[allow(clippy::too_many_arguments)]
pub fn encode_stereo(
    encoder: &mut RangeEncoder<'_>,
    config: PartitionConfig,
    stereo: StereoConfig,
    state: &mut PartitionState,
    left: &mut [f32],
    right: &mut [f32],
    left_pulses: &mut [i32],
    right_pulses: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    validate(config, state, left, left_pulses, recurrence_scratch)?;
    validate(config, state, right, right_pulses, recurrence_scratch)?;
    if left.len() != right.len() {
        return Err(Error::InvalidFrameSize);
    }
    let dimensions = left.len();
    for index in 0..dimensions {
        let l = left[index];
        let r = right[index];
        left[index] = 0.5 * (l + r);
        right[index] = 0.5 * (r - l);
    }
    let mid_energy = vector_energy(left);
    let side_energy = vector_energy(right);
    normalize_or_zero(left, mid_energy);
    normalize_or_zero(right, side_energy);
    let pulse_cap = theta_pulse_cap(config.band, config.lm)?;
    let theta_config = ThetaConfig {
        dimensions,
        bits: config.bits,
        pulse_cap,
        stereo: true,
        original_blocks: config.blocks,
        intensity: stereo.intensity,
    };
    let resolution = theta_resolution(theta_config)?;
    let theta_index = if stereo.intensity {
        0
    } else {
        choose_theta_from_energy(mid_energy, side_energy, resolution)
    };
    let tell = encoder.tell_frac();
    let theta = encode_theta(encoder, theta_config, theta_index)?;
    let inverted = false;
    if inversion_is_coded(
        theta.resolution,
        config.bits,
        state.remaining_bits,
        stereo.disable_inversion,
    ) {
        encoder.encode_bit_logp(inverted, 2)?;
    }
    let allocation = i32::try_from(encoder.tell_frac() - tell).map_err(|_| Error::InvalidPacket)?;
    let bits = config
        .bits
        .checked_sub(allocation)
        .ok_or(Error::InvalidPacket)?;
    state.remaining_bits = state
        .remaining_bits
        .checked_sub(allocation)
        .ok_or(Error::InvalidPacket)?;
    if dimensions == 2 {
        return encode_stereo_two_bin(
            encoder,
            config,
            theta,
            bits,
            state,
            left,
            right,
            left_pulses,
            right_pulses,
            recurrence_scratch,
        );
    }
    let delta = allocation_delta(dimensions, theta)?;
    let mid_bits = ((bits - delta) / 2).clamp(0, bits);
    let side_bits = bits - mid_bits;
    let starting_budget = state.remaining_bits;
    let (mid_mask, side_mask) = if mid_bits >= side_bits {
        let mid_mask = encode(
            encoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )?;
        let rebalance = mid_bits - (starting_budget - state.remaining_bits);
        let adjusted = if rebalance > 24 && theta.angle_q14 != 0 {
            side_bits + rebalance - 24
        } else {
            side_bits
        };
        let side_mask = encode(
            encoder,
            PartitionConfig {
                bits: adjusted,
                gain: f32::from(theta.second_gain_q15) / 32_768.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )?;
        (mid_mask, side_mask)
    } else {
        let side_mask = encode(
            encoder,
            PartitionConfig {
                bits: side_bits,
                gain: f32::from(theta.second_gain_q15) / 32_768.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )?;
        let rebalance = side_bits - (starting_budget - state.remaining_bits);
        let adjusted = if rebalance > 24 && theta.angle_q14 != 16_384 {
            mid_bits + rebalance - 24
        } else {
            mid_bits
        };
        let mid_mask = encode(
            encoder,
            PartitionConfig {
                bits: adjusted,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )?;
        (mid_mask, side_mask)
    };
    Ok(mid_mask | side_mask)
}

#[allow(clippy::too_many_arguments)]
fn encode_stereo_two_bin(
    encoder: &mut RangeEncoder<'_>,
    config: PartitionConfig,
    theta: crate::celt_theta::Theta,
    bits: i32,
    state: &mut PartitionState,
    left: &mut [f32],
    right: &mut [f32],
    left_pulses: &mut [i32],
    right_pulses: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let side_bits = if theta.angle_q14 != 0 && theta.angle_q14 != 16_384 {
        8
    } else {
        0
    };
    let mid_bits = bits.checked_sub(side_bits).ok_or(Error::InvalidPacket)?;
    if side_bits != 0 {
        state.remaining_bits = state
            .remaining_bits
            .checked_sub(8)
            .ok_or(Error::InvalidPacket)?;
        encoder.raw_bits(0, 1)?;
    }
    if theta.angle_q14 > 8_192 {
        encode(
            encoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )
    } else {
        encode(
            encoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )
    }
}

fn vector_energy(vector: &[f32]) -> f32 {
    vector.iter().fold(0.0, |sum, value| sum + value * value)
}

fn normalize_or_zero(vector: &mut [f32], energy: f32) {
    if energy > 0.0 {
        let scale = 1.0 / mrml_math::sqrt(energy);
        for value in vector {
            *value *= scale;
        }
    } else {
        vector.fill(0.0);
    }
}

fn choose_theta_from_energy(first: f32, second: f32, resolution: u16) -> u16 {
    if resolution <= 1 {
        return 0;
    }
    if first + second == 0.0 {
        return resolution / 2;
    }
    let target = mrml_math::sqrt(first / (first + second));
    let mut best = 0;
    let mut best_error = f32::MAX;
    for index in 0..=resolution {
        let angle = core::f32::consts::FRAC_PI_2 * index as f32 / resolution as f32;
        let error = (mrml_math::cos(angle) - target).abs();
        if error < best_error {
            best = index;
            best_error = error;
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn decode_stereo_two_bin(
    decoder: &mut RangeDecoder<'_>,
    config: PartitionConfig,
    theta: crate::celt_theta::Theta,
    bits: i32,
    inverted: bool,
    state: &mut PartitionState,
    left: &mut [f32],
    right: &mut [f32],
    left_pulses: &mut [i32],
    right_pulses: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let side_bits = if theta.angle_q14 != 0 && theta.angle_q14 != 16_384 {
        8
    } else {
        0
    };
    let mid_bits = bits.checked_sub(side_bits).ok_or(Error::InvalidPacket)?;
    let sign = if side_bits != 0 {
        state.remaining_bits = state
            .remaining_bits
            .checked_sub(8)
            .ok_or(Error::InvalidPacket)?;
        if decoder.raw_bits(1)? == 0 { 1.0 } else { -1.0 }
    } else {
        1.0
    };
    let use_right = theta.angle_q14 > 8_192;
    let mask = if use_right {
        decode(
            decoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            right,
            right_pulses,
            recurrence_scratch,
        )?
    } else {
        decode(
            decoder,
            PartitionConfig {
                bits: mid_bits,
                gain: 1.0,
                ..config
            },
            state,
            left,
            left_pulses,
            recurrence_scratch,
        )?
    };
    if use_right {
        left[0] = -sign * right[1];
        left[1] = sign * right[0];
    } else {
        right[0] = -sign * left[1];
        right[1] = sign * left[0];
    }
    let mid = f32::from(theta.first_gain_q15) / 32_768.0;
    let side = f32::from(theta.second_gain_q15) / 32_768.0;
    for index in 0..2 {
        let mid_value = mid * left[index];
        let side_value = side * right[index];
        left[index] = mid_value - side_value;
        right[index] = mid_value + side_value;
        if inverted {
            right[index] = -right[index];
        }
    }
    Ok(mask)
}

fn stereo_merge(left: &mut [f32], right: &mut [f32], mid_gain: f32) -> Result<(), Error> {
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for index in 0..left.len() {
        let mid = mid_gain * left[index];
        let side = right[index];
        left[index] = mid - side;
        right[index] = mid + side;
        left_energy += left[index] * left[index];
        right_energy += right[index] * right[index];
    }
    if left_energy < 0.000_6 || right_energy < 0.000_6 {
        right.copy_from_slice(left);
        return Ok(());
    }
    let left_scale = 1.0 / mrml_math::sqrt(left_energy);
    let right_scale = 1.0 / mrml_math::sqrt(right_energy);
    for index in 0..left.len() {
        left[index] *= left_scale;
        right[index] *= right_scale;
    }
    Ok(())
}

fn decode_inner(
    decoder: &mut RangeDecoder<'_>,
    mut config: PartitionConfig,
    state: &mut PartitionState,
    output: &mut [f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let maximum = maximum_leaf_cost(config.band, config.lm, recurrence_scratch)?;
    if config.lm >= 0 && output.len() > 2 && config.bits > i32::from(maximum) + 12 {
        let half = output.len() / 2;
        if half * 2 != output.len() {
            return Err(Error::InvalidFrameSize);
        }
        let original_blocks = config.blocks;
        config.lm -= 1;
        if config.blocks == 1 {
            config.blocks = 1;
        } else {
            config.blocks = config.blocks.div_ceil(2);
        }
        let pulse_cap = theta_pulse_cap(config.band, config.lm)?;
        let theta = decode_theta(
            decoder,
            ThetaConfig {
                dimensions: half,
                bits: config.bits,
                pulse_cap,
                stereo: false,
                original_blocks,
                intensity: false,
            },
        )?;
        config.bits = config
            .bits
            .checked_sub(i32::try_from(theta.allocation).map_err(|_| Error::InvalidPacket)?)
            .ok_or(Error::InvalidPacket)?;
        state.remaining_bits = state
            .remaining_bits
            .checked_sub(i32::try_from(theta.allocation).map_err(|_| Error::InvalidPacket)?)
            .ok_or(Error::InvalidPacket)?;
        let mut delta = allocation_delta(half, theta)?;
        if original_blocks > 1 && theta.angle_q14 != 0 && theta.angle_q14 != 16_384 {
            if theta.angle_q14 > 8_192 {
                let shift = u32::try_from(4 - config.lm).map_err(|_| Error::InvalidPacket)?;
                delta = delta
                    .checked_sub(delta >> shift)
                    .ok_or(Error::InvalidPacket)?;
            } else {
                let shift = u32::try_from(5 - config.lm).map_err(|_| Error::InvalidPacket)?;
                let masking = i32::try_from(half)
                    .map_err(|_| Error::InvalidFrameSize)?
                    .checked_shl(3)
                    .ok_or(Error::InvalidPacket)?
                    >> shift;
                delta = delta
                    .checked_add(masking)
                    .ok_or(Error::InvalidPacket)?
                    .min(0);
            }
        }
        let first_bits = ((config.bits - delta) / 2).clamp(0, config.bits);
        let second_bits = config.bits - first_bits;
        let (first_output, second_output) = output.split_at_mut(half);
        let (first_pulses, second_pulses) = pulse_workspace.split_at_mut(half);
        let starting_budget = state.remaining_bits;
        let first_gain = config.gain * (f32::from(theta.first_gain_q15) / 32_768.0);
        let second_gain = config.gain * (f32::from(theta.second_gain_q15) / 32_768.0);
        let (first_mask, second_mask) = if first_bits >= second_bits {
            let first_mask = decode_inner(
                decoder,
                PartitionConfig {
                    bits: first_bits,
                    gain: first_gain,
                    ..config
                },
                state,
                first_output,
                first_pulses,
                recurrence_scratch,
            )?;
            let used = starting_budget - state.remaining_bits;
            let rebalance = first_bits - used;
            let second_bits = if rebalance > 24 && theta.angle_q14 != 0 {
                second_bits + rebalance - 24
            } else {
                second_bits
            };
            let second_mask = decode_inner(
                decoder,
                PartitionConfig {
                    bits: second_bits,
                    gain: second_gain,
                    ..config
                },
                state,
                second_output,
                second_pulses,
                recurrence_scratch,
            )?;
            (first_mask, second_mask)
        } else {
            let second_mask = decode_inner(
                decoder,
                PartitionConfig {
                    bits: second_bits,
                    gain: second_gain,
                    ..config
                },
                state,
                second_output,
                second_pulses,
                recurrence_scratch,
            )?;
            let used = starting_budget - state.remaining_bits;
            let rebalance = second_bits - used;
            let first_bits = if rebalance > 24 && theta.angle_q14 != 16_384 {
                first_bits + rebalance - 24
            } else {
                first_bits
            };
            let first_mask = decode_inner(
                decoder,
                PartitionConfig {
                    bits: first_bits,
                    gain: first_gain,
                    ..config
                },
                state,
                first_output,
                first_pulses,
                recurrence_scratch,
            )?;
            (first_mask, second_mask)
        };
        let shift = original_blocks / 2;
        return Ok(first_mask | second_mask.checked_shl(shift as u32).unwrap_or(0));
    }

    decode_leaf(
        decoder,
        config,
        state,
        output,
        pulse_workspace,
        recurrence_scratch,
    )
}

fn decode_leaf(
    decoder: &mut RangeDecoder<'_>,
    config: PartitionConfig,
    state: &mut PartitionState,
    output: &mut [f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
) -> Result<u16, Error> {
    let target = config.bits.max(0);
    let available = state.remaining_bits.max(0);
    let pulses = pvq::pulses_for_band_target(
        config.band,
        config.lm,
        u16::try_from(target.min(i32::from(u16::MAX))).unwrap_or(u16::MAX),
        u16::try_from(available.min(i32::from(u16::MAX))).unwrap_or(u16::MAX),
        recurrence_scratch,
    )?;
    if pulses == 0 {
        pulse_workspace.fill(0);
        output.fill(0.0);
        return Ok(0);
    }
    let used = i32::from(pvq::band_pulse_cost(
        config.band,
        config.lm,
        pulses,
        recurrence_scratch,
    )?);
    state.remaining_bits = state
        .remaining_bits
        .checked_sub(used)
        .ok_or(Error::InvalidPacket)?;
    pvq::decode_range(
        decoder,
        pulses,
        config.spread,
        pulse_workspace,
        output,
        recurrence_scratch,
    )?;
    for value in output.iter_mut() {
        *value *= config.gain;
    }
    collapse_mask(pulse_workspace, config.blocks)
}

pub(crate) fn maximum_leaf_cost(band: usize, lm: i8, scratch: &mut [u32]) -> Result<u16, Error> {
    pvq::maximum_band_cost(band, lm, scratch)
}

fn collapse_mask(pulses: &[i32], blocks: usize) -> Result<u16, Error> {
    if blocks == 0 || blocks > 16 || !pulses.len().is_multiple_of(blocks) {
        return Err(Error::InvalidFrameSize);
    }
    if blocks == 1 {
        return Ok(1);
    }
    let width = pulses.len() / blocks;
    let mut mask = 0;
    for (block, values) in pulses.chunks_exact(width).enumerate() {
        if values.iter().any(|&value| value != 0) {
            mask |= 1 << block;
        }
    }
    Ok(mask)
}

fn validate(
    config: PartitionConfig,
    state: &PartitionState,
    output: &[f32],
    pulse_workspace: &[i32],
    recurrence_scratch: &[u32],
) -> Result<(), Error> {
    if output.len() < 2
        || pulse_workspace.len() != output.len()
        || recurrence_scratch.len() < 2
        || config.bits < 0
        || config.band >= 21
        || state.remaining_bits < 0
        || !(-1..=3).contains(&config.lm)
        || config.blocks == 0
        || config.blocks > 16
        || !output.len().is_multiple_of(config.blocks)
        || config.spread > 3
        || !config.gain.is_finite()
        || config.gain < 0.0
    {
        return Err(Error::InvalidFrameSize);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_capacity_rounds_fractional_dimension_cost_up() {
        assert_eq!(theta_pulse_cap(15, -1), Ok(13));
        assert_eq!(theta_pulse_cap(17, 0), Ok(24));
        assert_eq!(theta_pulse_cap(16, 1), Ok(29));
        assert_eq!(theta_pulse_cap(21, 0), Err(Error::InvalidFrameSize));
    }

    #[test]
    fn inversion_symbol_is_present_only_when_theta_is_omitted() {
        assert!(inversion_is_coded(1, 17, 17, false));
        assert!(!inversion_is_coded(2, 320, 320, false));
        assert!(!inversion_is_coded(1, 16, 320, false));
        assert!(!inversion_is_coded(1, 320, 320, true));
    }
    use crate::RangeEncoder;

    #[test]
    fn recursive_mono_encoder_matches_decoder_budget_and_pulses() {
        let mut input = [0.0f32; 32];
        for (index, value) in input.iter_mut().enumerate() {
            *value = (index as f32 - 12.0) / 20.0;
        }
        let config = PartitionConfig {
            band: 16,
            bits: 520,
            lm: 2,
            blocks: 4,
            spread: 0,
            gain: 1.0,
        };
        let mut encoded_state = PartitionState {
            remaining_bits: 520,
            seed: 7,
        };
        let mut encoded_pulses = [0i32; 32];
        let mut recurrence = [0u32; pvq::MAX_PULSES + 1];
        let mut bytes = [0u8; 128];
        let encoded_mask = {
            let mut encoder = RangeEncoder::new(&mut bytes);
            let mask = encode(
                &mut encoder,
                config,
                &mut encoded_state,
                &input,
                &mut encoded_pulses,
                &mut recurrence,
            )
            .unwrap();
            encoder.finish().unwrap();
            mask
        };
        let mut decoded_state = PartitionState {
            remaining_bits: 520,
            seed: 7,
        };
        let mut decoded_pulses = [0i32; 32];
        let mut normalized = [0.0f32; 32];
        let decoded_mask = decode(
            &mut RangeDecoder::new(&bytes),
            config,
            &mut decoded_state,
            &mut normalized,
            &mut decoded_pulses,
            &mut recurrence,
        )
        .unwrap();
        assert_eq!(decoded_state, encoded_state);
        assert_eq!(decoded_pulses, encoded_pulses);
        assert_eq!(decoded_mask, encoded_mask);
    }

    #[test]
    fn joint_stereo_encoder_matches_decoder_entropy_state() {
        let mut left = [0.0f32; 16];
        let mut right = [0.0f32; 16];
        for index in 0..16 {
            left[index] = (index as f32 - 5.0) * 0.1;
            right[index] = (9.0 - index as f32) * 0.08;
        }
        let left_energy = vector_energy(&left);
        let right_energy = vector_energy(&right);
        normalize_or_zero(&mut left, left_energy);
        normalize_or_zero(&mut right, right_energy);
        let config = PartitionConfig {
            band: 12,
            bits: 360,
            lm: 2,
            blocks: 4,
            spread: 0,
            gain: 1.0,
        };
        let stereo = StereoConfig {
            intensity: false,
            disable_inversion: false,
        };
        let mut encoded_state = PartitionState {
            remaining_bits: 360,
            seed: 4,
        };
        let mut left_pulses = [0i32; 16];
        let mut right_pulses = [0i32; 16];
        let mut recurrence = [0u32; pvq::MAX_PULSES + 1];
        let mut bytes = [0u8; 96];
        let encoded_mask = {
            let mut encoder = RangeEncoder::new(&mut bytes);
            let mask = encode_stereo(
                &mut encoder,
                config,
                stereo,
                &mut encoded_state,
                &mut left,
                &mut right,
                &mut left_pulses,
                &mut right_pulses,
                &mut recurrence,
            )
            .unwrap();
            encoder.finish().unwrap();
            mask
        };
        let mut decoded_state = PartitionState {
            remaining_bits: 360,
            seed: 4,
        };
        let mut decoded_left = [0.0f32; 16];
        let mut decoded_right = [0.0f32; 16];
        let mut decoded_left_pulses = [0i32; 16];
        let mut decoded_right_pulses = [0i32; 16];
        let decoded_mask = decode_stereo(
            &mut RangeDecoder::new(&bytes),
            config,
            stereo,
            &mut decoded_state,
            &mut decoded_left,
            &mut decoded_right,
            &mut decoded_left_pulses,
            &mut decoded_right_pulses,
            &mut recurrence,
        )
        .unwrap();
        assert_eq!(decoded_state, encoded_state);
        assert_eq!(decoded_mask, encoded_mask);
        assert_eq!(decoded_left_pulses, left_pulses);
        assert_eq!(decoded_right_pulses, right_pulses);
        assert!(decoded_left.iter().all(|value| value.is_finite()));
        assert!(decoded_right.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn leaf_pvq_decodes_and_reports_each_live_block() {
        let pulses = [1, 0, 0, 0, 0, 0, -1, 0];
        let mut bytes = [0u8; 64];
        let mut recurrence = [0u32; pvq::MAX_PULSES + 1];
        let mut encoder = RangeEncoder::new(&mut bytes);
        pvq::encode_range(&mut encoder, 2, &pulses, &mut recurrence).unwrap();
        encoder.finish().unwrap();
        let mut output = [0.0; 8];
        let mut decoded_pulses = [0; 8];
        let mut state = PartitionState {
            remaining_bits: 128,
            seed: 1,
        };
        let mask = decode(
            &mut RangeDecoder::new(&bytes),
            PartitionConfig {
                band: 16,
                bits: i32::from(pvq::band_pulse_cost(16, 0, 2, &mut recurrence).unwrap()),
                lm: 0,
                blocks: 2,
                spread: 0,
                gain: 0.5,
            },
            &mut state,
            &mut output,
            &mut decoded_pulses,
            &mut recurrence,
        )
        .unwrap();
        assert_eq!(decoded_pulses, pulses);
        assert_eq!(mask, 0b11);
        let energy = output.iter().fold(0.0, |sum, value| sum + value * value);
        assert!((energy - 0.25).abs() < 0.000_01);
    }

    #[test]
    fn collapse_mask_groups_time_ordered_pulses() {
        assert_eq!(collapse_mask(&[1, 0, 0, 0, 0, 0, 2, 0], 4), Ok(0b1001));
        assert_eq!(collapse_mask(&[0; 8], 1), Ok(1));
        let mut divided = [0; 16];
        divided[0] = 1;
        divided[15] = -1;
        assert_eq!(collapse_mask(&divided, 16), Ok(0x8001));
        assert_eq!(collapse_mask(&[0; 7], 4), Err(Error::InvalidFrameSize));
    }

    #[test]
    fn oversized_codebook_recursively_splits_before_pvq() {
        let mut recurrence = [0u32; 256];
        let bits = i32::from(maximum_leaf_cost(12, 2, &mut recurrence).unwrap()) + 13;
        let pulse_cap = i32::from(pvq::fractional_log2(8).unwrap());
        let theta_config = ThetaConfig {
            dimensions: 8,
            bits,
            pulse_cap,
            stereo: false,
            original_blocks: 1,
            intensity: false,
        };
        let mut bytes = [0u8; 256];
        let mut encoder = RangeEncoder::new(&mut bytes);
        let theta = crate::celt_theta::encode(&mut encoder, theta_config, 0).unwrap();
        let child_bits = bits - i32::try_from(theta.allocation).unwrap();
        let pulses = pvq::pulses_for_allocation(
            8,
            u16::try_from(child_bits).unwrap_or(u16::MAX),
            &mut recurrence,
        )
        .unwrap();
        let mut expected_pulses = [0i32; 8];
        expected_pulses[0] = i32::try_from(pulses).unwrap();
        pvq::encode_range(&mut encoder, pulses, &expected_pulses, &mut recurrence).unwrap();
        encoder.finish().unwrap();

        let mut output = [0.0f32; 16];
        let mut decoded_pulses = [0i32; 16];
        let mut state = PartitionState {
            remaining_bits: bits,
            seed: 7,
        };
        let mask = decode(
            &mut RangeDecoder::new(&bytes),
            PartitionConfig {
                band: 12,
                bits,
                lm: 2,
                blocks: 1,
                spread: 0,
                gain: 1.0,
            },
            &mut state,
            &mut output,
            &mut decoded_pulses,
            &mut recurrence,
        )
        .unwrap();
        assert_eq!(&decoded_pulses[..8], &expected_pulses);
        assert_eq!(&decoded_pulses[8..], &[0; 8]);
        assert_eq!(mask, 1);
        assert!(output[..8].iter().any(|value| *value != 0.0));
        assert!(output[8..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn intensity_stereo_decodes_one_shape_into_both_channels() {
        let mut recurrence = [0u32; pvq::MAX_PULSES + 1];
        let pulses = 2;
        let vector = [1, 0, 0, -1, 0, 0, 0, 0];
        let bits = i32::from(pvq::band_pulse_cost(16, 0, pulses, &mut recurrence).unwrap());
        let mut bytes = [0u8; 64];
        let mut encoder = RangeEncoder::new(&mut bytes);
        pvq::encode_range(&mut encoder, pulses, &vector, &mut recurrence).unwrap();
        encoder.finish().unwrap();
        let mut left = [0.0; 8];
        let mut right = [0.0; 8];
        let mut left_pulses = [0; 8];
        let mut right_pulses = [0; 8];
        let mut state = PartitionState {
            remaining_bits: bits,
            seed: 1,
        };
        let mask = decode_stereo(
            &mut RangeDecoder::new(&bytes),
            PartitionConfig {
                band: 16,
                bits,
                lm: 0,
                blocks: 1,
                spread: 0,
                gain: 1.0,
            },
            StereoConfig {
                intensity: true,
                disable_inversion: true,
            },
            &mut state,
            &mut left,
            &mut right,
            &mut left_pulses,
            &mut right_pulses,
            &mut recurrence,
        )
        .unwrap();
        assert_eq!(mask, 1);
        assert_eq!(left_pulses, vector);
        assert_eq!(right_pulses, [0; 8]);
        assert_eq!(left, right);
    }

    #[test]
    fn stereo_merge_normalizes_both_channels() {
        let mut mid = [0.5, -0.5, 0.5, -0.5];
        let mut side = [0.25, 0.25, -0.25, -0.25];
        stereo_merge(&mut mid, &mut side, 0.75).unwrap();
        let left_energy = mid.iter().fold(0.0, |sum, value| sum + value * value);
        let right_energy = side.iter().fold(0.0, |sum, value| sum + value * value);
        assert!((left_energy - 1.0).abs() < 0.000_01);
        assert!((right_energy - 1.0).abs() < 0.000_01);
        assert_ne!(mid, side);
    }
}
