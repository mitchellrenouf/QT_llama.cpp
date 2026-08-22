//! CELT frame syntax, energy, and allocation orchestration.

use crate::{
    Error, RangeDecoder, RangeEncoder,
    bands::{self, BAND_COUNT},
    celt::{PostFilterParameters, decode_spread},
    celt_allocation::{
        self, FinalAllocation, FinalizeConfig, Reservations, band_caps, base_allocation,
        decode_boosts, decode_final_allocation, decode_trim, reserve_flags,
    },
    celt_energy::{self, CoarseConfig, LogEnergies},
    celt_partition::{self, PartitionConfig, PartitionState, StereoConfig},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodeRequest {
    pub silence: bool,
    pub post_filter: Option<PostFilterParameters>,
    pub transient: bool,
    pub intra_energy: bool,
    pub tf_flags: [bool; BAND_COUNT],
    pub tf_select: bool,
    pub spread: u8,
    pub residuals: [[i16; BAND_COUNT]; 2],
    pub boosts: [i32; BAND_COUNT],
    pub trim: u8,
    pub coded_bands: usize,
    pub intensity: usize,
    pub dual_stereo: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameConfig {
    pub frame_bytes: usize,
    pub channels: u8,
    pub lm: u8,
    pub start: usize,
    pub end: usize,
}

impl FrameConfig {
    fn validate(self) -> Result<(), Error> {
        if self.frame_bytes == 0
            || self.frame_bytes > crate::MAX_FRAME_BYTES
            || !(1..=2).contains(&self.channels)
            || self.lm > 3
            || self.start >= self.end
            || self.end > BAND_COUNT
        {
            return Err(Error::InvalidPacket);
        }
        Ok(())
    }
}

/// All syntax and allocation information required before CELT shape decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePlan {
    pub silence: bool,
    pub post_filter: Option<PostFilterParameters>,
    pub transient: bool,
    pub intra_energy: bool,
    pub tf_adjustments: [i8; BAND_COUNT],
    pub tf_select: bool,
    pub spread: u8,
    pub caps: [i32; BAND_COUNT],
    pub boosts: [i32; BAND_COUNT],
    pub total_boost: i32,
    pub trim: u8,
    pub reservations: Reservations,
    pub allocation: Option<FinalAllocation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShapeResult {
    pub collapse_masks: [u8; BAND_COUNT * 2],
    pub anti_collapse: bool,
    pub final_energy_bits: usize,
    pub seed: u32,
}

fn update_shape_balance(
    balance: i32,
    allocated: i32,
    tell_before: i32,
    tell_after: i32,
) -> Result<i32, Error> {
    if tell_after < tell_before {
        return Err(Error::InvalidPacket);
    }
    balance
        .checked_add(allocated)
        .and_then(|value| value.checked_sub(tell_after - tell_before))
        .ok_or(Error::InvalidPacket)
}

const fn dual_channel_bits(bits: i32) -> i32 {
    bits / 2
}

/// Decodes CELT's prefix through fixed-rate fine energy, in normative order.
/// Shape PVQ, anti-collapse, final fine energy, and synthesis follow this plan.
pub fn decode_plan(
    decoder: &mut RangeDecoder<'_>,
    config: FrameConfig,
    energies: &mut LogEnergies,
    residuals: &mut [[i16; BAND_COUNT]; 2],
) -> Result<FramePlan, Error> {
    config.validate()?;
    let total_whole = u32::try_from(config.frame_bytes * 8).map_err(|_| Error::InvalidPacket)?;
    let mut tell = decoder.tell();
    let silence = if tell >= total_whole {
        true
    } else if tell == 1 {
        decoder.decode_bit_logp(15)?
    } else {
        false
    };
    if silence {
        return silent_plan(config);
    }
    tell = decoder.tell();
    let post_filter = if config.start == 0 && tell + 16 <= total_whole {
        PostFilterParameters::decode(decoder)?
    } else {
        None
    };
    tell = decoder.tell();
    let transient = config.lm > 0 && tell + 3 <= total_whole && decoder.decode_bit_logp(3)?;
    tell = decoder.tell();
    let intra_energy = tell + 3 <= total_whole && decoder.decode_bit_logp(3)?;
    celt_energy::decode_coarse(
        decoder,
        CoarseConfig {
            channels: config.channels,
            lm: config.lm,
            intra: intra_energy,
            start: config.start,
            end: config.end,
            frame_bytes: config.frame_bytes,
        },
        energies,
        residuals,
    )?;
    let mut tf_adjustments = [0; BAND_COUNT];
    let tf_select = bands::decode_tf_resolution_bounded(
        decoder,
        config.lm,
        transient,
        &mut tf_adjustments[config.start..config.end],
        total_whole,
    )?;
    let spread = if decoder.tell() + 4 <= total_whole {
        decode_spread(decoder)?
    } else {
        2
    };
    let mut caps = [0; BAND_COUNT];
    band_caps(config.channels, config.lm, &mut caps)?;
    let total_fractional =
        i32::try_from(config.frame_bytes * 64).map_err(|_| Error::InvalidPacket)?;
    let mut boosts = [0; BAND_COUNT];
    let total_boost = decode_boosts(
        decoder,
        config.lm,
        config.start,
        config.end,
        total_fractional,
        &caps,
        &mut boosts,
    )?;
    let trim = decode_trim(decoder, total_fractional, total_boost)?;
    let adjusted_tell = decoder.tell_frac();
    let reservations = reserve_flags(
        config.frame_bytes,
        adjusted_tell,
        config.channels,
        config.lm,
        transient,
        config.start,
        config.end,
    )?;
    let base = base_allocation(
        config.channels,
        config.lm,
        config.start,
        config.end,
        reservations.available,
        trim,
        &boosts,
        &caps,
    )?;
    let allocation = decode_final_allocation(
        decoder,
        FinalizeConfig {
            channels: config.channels,
            lm: config.lm,
            start: config.start,
            end: config.end,
            reservations,
        },
        &base,
        &caps,
    )?;
    celt_energy::decode_fine(
        decoder,
        config.channels,
        config.start,
        config.end,
        &allocation.bands.fine,
        energies,
    )?;
    Ok(FramePlan {
        silence,
        post_filter,
        transient,
        intra_energy,
        tf_adjustments,
        tf_select,
        spread,
        caps,
        boosts,
        total_boost,
        trim,
        reservations,
        allocation: Some(allocation),
    })
}

/// Encoder mirror of [`decode_plan`], through fixed-rate fine energy.
#[allow(clippy::needless_range_loop)] // Mirrors normative band indexing.
pub fn encode_plan(
    encoder: &mut RangeEncoder<'_>,
    config: FrameConfig,
    request: &EncodeRequest,
    target_energies: &LogEnergies,
    energies: &mut LogEnergies,
    coded_residuals: &mut [[i16; BAND_COUNT]; 2],
) -> Result<FramePlan, Error> {
    config.validate()?;
    let total_whole = u32::try_from(config.frame_bytes * 8).map_err(|_| Error::InvalidPacket)?;
    if encoder.tell() == 1 {
        encoder.encode_bit_logp(request.silence, 15)?;
    } else if request.silence {
        return Err(Error::InvalidPacket);
    }
    if request.silence {
        return silent_plan(config);
    }
    let post_filter = if config.start == 0 && encoder.tell() + 16 <= total_whole {
        PostFilterParameters::encode(request.post_filter, encoder)?;
        request.post_filter
    } else {
        if request.post_filter.is_some() {
            return Err(Error::InvalidPacket);
        }
        None
    };
    let transient = if config.lm > 0 && encoder.tell() + 3 <= total_whole {
        encoder.encode_bit_logp(request.transient, 3)?;
        request.transient
    } else {
        if request.transient {
            return Err(Error::InvalidPacket);
        }
        false
    };
    let intra_energy = if encoder.tell() + 3 <= total_whole {
        encoder.encode_bit_logp(request.intra_energy, 3)?;
        request.intra_energy
    } else {
        if request.intra_energy {
            return Err(Error::InvalidPacket);
        }
        false
    };
    celt_energy::encode_coarse(
        encoder,
        CoarseConfig {
            channels: config.channels,
            lm: config.lm,
            intra: intra_energy,
            start: config.start,
            end: config.end,
            frame_bytes: config.frame_bytes,
        },
        &request.residuals,
        energies,
        coded_residuals,
    )?;
    bands::encode_tf_resolution_bounded(
        encoder,
        config.lm,
        transient,
        &request.tf_flags[config.start..config.end],
        request.tf_select,
        total_whole,
    )?;
    let mut tf_adjustments = [0i8; BAND_COUNT];
    for band in config.start..config.end {
        tf_adjustments[band] = bands::tf_adjustment(
            config.lm,
            transient,
            request.tf_select,
            request.tf_flags[band],
        )
        .ok_or(Error::InvalidFrameSize)?;
    }
    let spread = if encoder.tell() + 4 <= total_whole {
        crate::celt::encode_spread(encoder, request.spread)?;
        request.spread
    } else if request.spread == 2 {
        2
    } else {
        return Err(Error::InvalidPacket);
    };
    let mut caps = [0; BAND_COUNT];
    band_caps(config.channels, config.lm, &mut caps)?;
    let total_fractional =
        i32::try_from(config.frame_bytes * 64).map_err(|_| Error::InvalidPacket)?;
    let total_boost = celt_allocation::encode_boosts(
        encoder,
        config.lm,
        config.start,
        config.end,
        total_fractional,
        &caps,
        &request.boosts,
    )?;
    celt_allocation::encode_trim(encoder, total_fractional, total_boost, request.trim)?;
    let adjusted_tell = encoder.tell_frac();
    let reservations = reserve_flags(
        config.frame_bytes,
        adjusted_tell,
        config.channels,
        config.lm,
        transient,
        config.start,
        config.end,
    )?;
    let base = base_allocation(
        config.channels,
        config.lm,
        config.start,
        config.end,
        reservations.available,
        request.trim,
        &request.boosts,
        &caps,
    )?;
    let finalize = FinalizeConfig {
        channels: config.channels,
        lm: config.lm,
        start: config.start,
        end: config.end,
        reservations,
    };
    let requested_coded_bands = if request.coded_bands == 0 {
        celt_allocation::maximum_coded_bands(finalize, &base)?
    } else {
        request.coded_bands
    };
    let allocation = celt_allocation::encode_final_allocation(
        encoder,
        finalize,
        &base,
        &caps,
        requested_coded_bands,
        request.intensity,
        request.dual_stereo,
    )?;
    celt_energy::encode_fine(
        encoder,
        config.channels,
        config.start,
        config.end,
        &allocation.bands.fine,
        target_energies,
        energies,
    )?;
    Ok(FramePlan {
        silence: false,
        post_filter,
        transient,
        intra_energy,
        tf_adjustments,
        tf_select: request.tf_select,
        spread,
        caps,
        boosts: request.boosts,
        total_boost,
        trim: request.trim,
        reservations,
        allocation: Some(allocation),
    })
}

/// Decodes all normalized spectral shapes described by a [`FramePlan`], then
/// consumes anti-collapse and final fine-energy syntax.
#[allow(clippy::too_many_arguments)]
pub fn decode_shapes(
    decoder: &mut RangeDecoder<'_>,
    config: FrameConfig,
    plan: &FramePlan,
    energies: &mut LogEnergies,
    previous_energies: &LogEnergies,
    older_energies: &LogEnergies,
    spectra: &mut [f32],
    pulse_workspace: &mut [i32],
    tf_scratch: &mut [f32],
    recurrence_scratch: &mut [u32],
    seed: u32,
) -> Result<ShapeResult, Error> {
    config.validate()?;
    let bins = 120usize << config.lm;
    let channels = usize::from(config.channels);
    if spectra.len() < bins * channels
        || pulse_workspace.len() < bins * channels
        || tf_scratch.len() < 176
        || recurrence_scratch.len() < crate::pvq::MAX_PULSES + 1
    {
        return Err(Error::BufferTooSmall);
    }
    spectra[..bins * channels].fill(0.0);
    pulse_workspace[..bins * channels].fill(0);
    let Some(allocation) = plan.allocation else {
        return Ok(ShapeResult {
            collapse_masks: [0; BAND_COUNT * 2],
            anti_collapse: false,
            final_energy_bits: 0,
            seed,
        });
    };
    let total_shape_bits = i32::try_from(config.frame_bytes * 64)
        .map_err(|_| Error::InvalidPacket)?
        - plan.reservations.anti_collapse;
    let mut balance = 0i32;
    let mut collapse_masks = [0u8; BAND_COUNT * 2];
    let frame_blocks = if plan.transient {
        1usize << config.lm
    } else {
        1
    };
    let mut dual_stereo = allocation.dual_stereo;
    let mut state = PartitionState {
        remaining_bits: total_shape_bits,
        seed,
    };
    for band in config.start..config.end {
        let tell = i32::try_from(decoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        let remaining = total_shape_bits
            .checked_sub(tell + 1)
            .ok_or(Error::InvalidPacket)?;
        state.remaining_bits = remaining.max(0);
        let bits = if band < allocation.coded_bands {
            let divisor = (allocation.coded_bands - band).min(3) as i32;
            let current_balance = balance / divisor;
            (allocation.bands.shape[band] + current_balance)
                .min(remaining + 1)
                .clamp(0, 16_383)
        } else {
            0
        };
        let range = bands::band_range(band, config.lm)?;
        let width = range.len();
        let partition_blocks = if width == 1 {
            1
        } else {
            bands::tf_layout(width, frame_blocks, plan.tf_adjustments[band])?.partition_blocks
        };
        if dual_stereo && band == allocation.intensity {
            dual_stereo = false;
        }
        let masks = if width == 1 {
            decode_one_bin(decoder, channels, bins, range.start, &mut state, spectra)?
                .map(u16::from)
        } else if channels == 1 {
            let mask = celt_partition::decode(
                decoder,
                PartitionConfig {
                    bits,
                    // TF changes reshape B/N_B but do not change the cache LM.
                    lm: config.lm as i8,
                    blocks: partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                &mut state,
                &mut spectra[range.clone()],
                &mut pulse_workspace[range.clone()],
                recurrence_scratch,
            )?;
            [mask, mask]
        } else if dual_stereo {
            let (left_spectrum, right_spectrum) = spectra.split_at_mut(bins);
            let (left_pulses, right_pulses) = pulse_workspace.split_at_mut(bins);
            let left_mask = celt_partition::decode(
                decoder,
                PartitionConfig {
                    bits: dual_channel_bits(bits),
                    lm: config.lm as i8,
                    blocks: partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                &mut state,
                &mut left_spectrum[range.clone()],
                &mut left_pulses[range.clone()],
                recurrence_scratch,
            )?;
            let right_mask = celt_partition::decode(
                decoder,
                PartitionConfig {
                    bits: dual_channel_bits(bits),
                    lm: config.lm as i8,
                    blocks: partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                &mut state,
                &mut right_spectrum[range.clone()],
                &mut right_pulses[range.clone()],
                recurrence_scratch,
            )?;
            [left_mask, right_mask]
        } else {
            let (left_spectrum, right_spectrum) = spectra.split_at_mut(bins);
            let (left_pulses, right_pulses) = pulse_workspace.split_at_mut(bins);
            let mask = celt_partition::decode_stereo(
                decoder,
                PartitionConfig {
                    bits,
                    lm: config.lm as i8,
                    blocks: partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                StereoConfig {
                    intensity: band >= allocation.intensity,
                    disable_inversion: false,
                },
                &mut state,
                &mut left_spectrum[range.clone()],
                &mut right_spectrum[range.clone()],
                &mut left_pulses[range.clone()],
                &mut right_pulses[range.clone()],
                recurrence_scratch,
            )?;
            [mask, mask]
        };
        let mut restored_masks = [0u8; 2];
        if width > 1 {
            for channel in 0..channels {
                let channel_range = channel * bins + range.start..channel * bins + range.end;
                restored_masks[channel] = bands::restore_tf_resolution(
                    &mut spectra[channel_range],
                    tf_scratch,
                    frame_blocks,
                    plan.tf_adjustments[band],
                    masks[channel],
                )?;
            }
        } else {
            restored_masks = [
                u8::try_from(masks[0]).map_err(|_| Error::InvalidFrameSize)?,
                u8::try_from(masks[1]).map_err(|_| Error::InvalidFrameSize)?,
            ];
        }
        collapse_masks[band * channels] = restored_masks[0];
        collapse_masks[band * channels + channels - 1] = restored_masks[channels - 1];
        let final_tell = i32::try_from(decoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        balance = update_shape_balance(balance, allocation.bands.shape[band], tell, final_tell)?;
    }
    let anti_collapse = plan.reservations.anti_collapse > 0 && decoder.raw_bits(1)? != 0;
    let available_final = (config.frame_bytes * 8).saturating_sub(decoder.tell() as usize);
    let final_energy_bits = celt_energy::decode_final(
        decoder,
        config.channels,
        config.start,
        config.end,
        &allocation.bands.fine,
        &allocation.bands.priority,
        available_final,
        energies,
    )?;
    if anti_collapse {
        let mut current = [0.0f32; BAND_COUNT * 2];
        let mut previous = [0.0f32; BAND_COUNT * 2];
        let mut older = [0.0f32; BAND_COUNT * 2];
        for channel in 0..2 {
            current[channel * BAND_COUNT..(channel + 1) * BAND_COUNT]
                .copy_from_slice(&energies.values()[channel]);
            previous[channel * BAND_COUNT..(channel + 1) * BAND_COUNT]
                .copy_from_slice(&previous_energies.values()[channel]);
            older[channel * BAND_COUNT..(channel + 1) * BAND_COUNT]
                .copy_from_slice(&older_energies.values()[channel]);
        }
        state.seed = crate::celt_anticollapse::apply(
            spectra,
            &collapse_masks,
            config.lm,
            channels,
            config.start,
            config.end,
            &current[..channels * BAND_COUNT],
            &previous,
            &older,
            &allocation.bands.shape,
            state.seed,
        )?;
    }
    Ok(ShapeResult {
        collapse_masks,
        anti_collapse,
        final_energy_bits,
        seed: state.seed,
    })
}

/// Encodes normalized mono CELT band shapes, anti-collapse, and final energy.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn encode_shapes_mono(
    encoder: &mut RangeEncoder<'_>,
    config: FrameConfig,
    plan: &FramePlan,
    spectra: &[f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
    target_energies: &LogEnergies,
    energies: &mut LogEnergies,
    seed: u32,
) -> Result<ShapeResult, Error> {
    config.validate()?;
    let bins = 120usize << config.lm;
    if config.channels != 1
        || spectra.len() < bins
        || pulse_workspace.len() < bins
        || recurrence_scratch.len() < crate::pvq::MAX_PULSES + 1
    {
        return Err(Error::InvalidFrameSize);
    }
    pulse_workspace[..bins].fill(0);
    let Some(allocation) = plan.allocation else {
        return Ok(ShapeResult {
            collapse_masks: [0; BAND_COUNT * 2],
            anti_collapse: false,
            final_energy_bits: 0,
            seed,
        });
    };
    let total_shape_bits = i32::try_from(config.frame_bytes * 64)
        .map_err(|_| Error::InvalidPacket)?
        - plan.reservations.anti_collapse;
    let mut balance = 0i32;
    let mut state = PartitionState {
        remaining_bits: total_shape_bits,
        seed,
    };
    let mut collapse_masks = [0u8; BAND_COUNT * 2];
    let frame_blocks = if plan.transient {
        1usize << config.lm
    } else {
        1
    };
    let mut band_vector = [0.0f32; 176];
    let mut tf_scratch = [0.0f32; 176];
    for band in config.start..config.end {
        let tell = i32::try_from(encoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        let remaining = total_shape_bits
            .checked_sub(tell + 1)
            .ok_or(Error::InvalidPacket)?;
        state.remaining_bits = remaining.max(0);
        let bits = if band < allocation.coded_bands {
            let divisor = (allocation.coded_bands - band).min(3) as i32;
            (allocation.bands.shape[band] + balance / divisor)
                .min(remaining + 1)
                .clamp(0, 16_383)
        } else {
            0
        };
        let range = bands::band_range(band, config.lm)?;
        let mask = if range.len() == 1 {
            let negative = spectra[range.start] < 0.0;
            if state.remaining_bits >= 8 {
                state.remaining_bits -= 8;
                encoder.raw_bits(u32::from(negative), 1)?;
            } else if negative {
                return Err(Error::InvalidPacket);
            }
            pulse_workspace[range.start] = if negative { -1 } else { 1 };
            1
        } else {
            let width = range.len();
            band_vector[..width].copy_from_slice(&spectra[range.clone()]);
            let layout = bands::prepare_tf_resolution(
                &mut band_vector[..width],
                &mut tf_scratch,
                frame_blocks,
                plan.tf_adjustments[band],
            )?;
            let partition_mask = celt_partition::encode(
                encoder,
                PartitionConfig {
                    bits,
                    lm: config.lm as i8,
                    blocks: layout.partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                &mut state,
                &band_vector[..width],
                &mut pulse_workspace[range],
                recurrence_scratch,
            )?;
            bands::restore_tf_resolution(
                &mut band_vector[..width],
                &mut tf_scratch,
                frame_blocks,
                plan.tf_adjustments[band],
                partition_mask,
            )?
        };
        collapse_masks[band] = mask;
        let final_tell = i32::try_from(encoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        balance = update_shape_balance(balance, allocation.bands.shape[band], tell, final_tell)?;
    }
    if plan.reservations.anti_collapse > 0 {
        encoder.raw_bits(0, 1)?;
    }
    let available_final = (config.frame_bytes * 8).saturating_sub(encoder.tell() as usize);
    let final_energy_bits = celt_energy::encode_final(
        encoder,
        config.channels,
        config.start,
        config.end,
        &allocation.bands.fine,
        &allocation.bands.priority,
        available_final,
        target_energies,
        energies,
    )?;
    Ok(ShapeResult {
        collapse_masks,
        anti_collapse: false,
        final_energy_bits,
        seed: state.seed,
    })
}

/// Encodes dual or theta-coupled stereo CELT shapes, switching to intensity
/// stereo at the allocation boundary.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn encode_shapes_stereo(
    encoder: &mut RangeEncoder<'_>,
    config: FrameConfig,
    plan: &FramePlan,
    spectra: &[f32],
    pulse_workspace: &mut [i32],
    recurrence_scratch: &mut [u32],
    target_energies: &LogEnergies,
    energies: &mut LogEnergies,
    seed: u32,
) -> Result<ShapeResult, Error> {
    config.validate()?;
    let bins = 120usize << config.lm;
    if config.channels != 2
        || spectra.len() < bins * 2
        || pulse_workspace.len() < bins * 2
        || recurrence_scratch.len() < crate::pvq::MAX_PULSES + 1
    {
        return Err(Error::InvalidFrameSize);
    }
    pulse_workspace[..bins * 2].fill(0);
    let Some(allocation) = plan.allocation else {
        return Ok(ShapeResult {
            collapse_masks: [0; BAND_COUNT * 2],
            anti_collapse: false,
            final_energy_bits: 0,
            seed,
        });
    };
    let total_shape_bits = i32::try_from(config.frame_bytes * 64)
        .map_err(|_| Error::InvalidPacket)?
        - plan.reservations.anti_collapse;
    let frame_blocks = if plan.transient {
        1usize << config.lm
    } else {
        1
    };
    let mut balance = 0i32;
    let mut state = PartitionState {
        remaining_bits: total_shape_bits,
        seed,
    };
    let mut collapse_masks = [0u8; BAND_COUNT * 2];
    let mut band_vector = [0.0f32; 176];
    let mut right_band_vector = [0.0f32; 176];
    let mut tf_scratch = [0.0f32; 176];
    let mut dual_stereo = allocation.dual_stereo;
    for band in config.start..config.end {
        let tell = i32::try_from(encoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        let remaining = total_shape_bits
            .checked_sub(tell + 1)
            .ok_or(Error::InvalidPacket)?;
        state.remaining_bits = remaining.max(0);
        let bits = if band < allocation.coded_bands {
            let divisor = (allocation.coded_bands - band).min(3) as i32;
            (allocation.bands.shape[band] + balance / divisor)
                .min(remaining + 1)
                .clamp(0, 16_383)
        } else {
            0
        };
        let range = bands::band_range(band, config.lm)?;
        if dual_stereo && band == allocation.intensity {
            dual_stereo = false;
        }
        if range.len() > 1 && !dual_stereo {
            let width = range.len();
            band_vector[..width].copy_from_slice(&spectra[range.clone()]);
            right_band_vector[..width]
                .copy_from_slice(&spectra[bins + range.start..bins + range.end]);
            let layout = bands::prepare_tf_resolution(
                &mut band_vector[..width],
                &mut tf_scratch,
                frame_blocks,
                plan.tf_adjustments[band],
            )?;
            bands::prepare_tf_resolution(
                &mut right_band_vector[..width],
                &mut tf_scratch,
                frame_blocks,
                plan.tf_adjustments[band],
            )?;
            let (left_pulses, right_pulses) = pulse_workspace.split_at_mut(bins);
            let mask = celt_partition::encode_stereo(
                encoder,
                PartitionConfig {
                    bits,
                    lm: config.lm as i8,
                    blocks: layout.partition_blocks,
                    spread: plan.spread,
                    gain: 1.0,
                },
                celt_partition::StereoConfig {
                    intensity: band >= allocation.intensity,
                    disable_inversion: false,
                },
                &mut state,
                &mut band_vector[..width],
                &mut right_band_vector[..width],
                &mut left_pulses[range.clone()],
                &mut right_pulses[range.clone()],
                recurrence_scratch,
            )?;
            let restored = bands::restore_tf_resolution(
                &mut band_vector[..width],
                &mut tf_scratch,
                frame_blocks,
                plan.tf_adjustments[band],
                mask,
            )?;
            collapse_masks[band * 2] = restored;
            collapse_masks[band * 2 + 1] = restored;
            let final_tell =
                i32::try_from(encoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
            balance =
                update_shape_balance(balance, allocation.bands.shape[band], tell, final_tell)?;
            continue;
        }
        for channel in 0..2 {
            let channel_range = channel * bins + range.start..channel * bins + range.end;
            let mask = if range.len() == 1 {
                let negative = spectra[channel_range.start] < 0.0;
                if state.remaining_bits >= 8 {
                    state.remaining_bits -= 8;
                    encoder.raw_bits(u32::from(negative), 1)?;
                } else if negative {
                    return Err(Error::InvalidPacket);
                }
                pulse_workspace[channel_range.start] = if negative { -1 } else { 1 };
                1
            } else {
                let width = range.len();
                band_vector[..width].copy_from_slice(&spectra[channel_range.clone()]);
                let layout = bands::prepare_tf_resolution(
                    &mut band_vector[..width],
                    &mut tf_scratch,
                    frame_blocks,
                    plan.tf_adjustments[band],
                )?;
                let partition_mask = celt_partition::encode(
                    encoder,
                    PartitionConfig {
                        bits: dual_channel_bits(bits),
                        lm: config.lm as i8,
                        blocks: layout.partition_blocks,
                        spread: plan.spread,
                        gain: 1.0,
                    },
                    &mut state,
                    &band_vector[..width],
                    &mut pulse_workspace[channel_range],
                    recurrence_scratch,
                )?;
                bands::restore_tf_resolution(
                    &mut band_vector[..width],
                    &mut tf_scratch,
                    frame_blocks,
                    plan.tf_adjustments[band],
                    partition_mask,
                )?
            };
            collapse_masks[band * 2 + channel] = mask;
        }
        let final_tell = i32::try_from(encoder.tell_frac()).map_err(|_| Error::InvalidPacket)?;
        balance = update_shape_balance(balance, allocation.bands.shape[band], tell, final_tell)?;
    }
    if plan.reservations.anti_collapse > 0 {
        encoder.raw_bits(0, 1)?;
    }
    let available_final = (config.frame_bytes * 8).saturating_sub(encoder.tell() as usize);
    let final_energy_bits = celt_energy::encode_final(
        encoder,
        config.channels,
        config.start,
        config.end,
        &allocation.bands.fine,
        &allocation.bands.priority,
        available_final,
        target_energies,
        energies,
    )?;
    Ok(ShapeResult {
        collapse_masks,
        anti_collapse: false,
        final_energy_bits,
        seed: state.seed,
    })
}

fn decode_one_bin(
    decoder: &mut RangeDecoder<'_>,
    channels: usize,
    bins: usize,
    index: usize,
    state: &mut PartitionState,
    spectra: &mut [f32],
) -> Result<[u8; 2], Error> {
    for channel in 0..channels {
        let negative = if state.remaining_bits >= 8 {
            state.remaining_bits -= 8;
            decoder.raw_bits(1)? != 0
        } else {
            false
        };
        spectra[channel * bins + index] = if negative { -1.0 } else { 1.0 };
    }
    Ok([1, 1])
}

fn silent_plan(config: FrameConfig) -> Result<FramePlan, Error> {
    let mut caps = [0; BAND_COUNT];
    band_caps(config.channels, config.lm, &mut caps)?;
    Ok(FramePlan {
        silence: true,
        post_filter: None,
        transient: false,
        intra_energy: false,
        tf_adjustments: [0; BAND_COUNT],
        tf_select: false,
        spread: 2,
        caps,
        boosts: [0; BAND_COUNT],
        total_boost: 0,
        trim: 5,
        reservations: Reservations {
            available: 0,
            anti_collapse: 0,
            skip: 0,
            intensity: 0,
            dual_stereo: 0,
        },
        allocation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_stereo_keeps_odd_fractional_bit_in_band_balance() {
        assert_eq!(dual_channel_bits(32), 16);
        assert_eq!(dual_channel_bits(33), 16);
        assert_eq!(dual_channel_bits(41), 20);
    }

    #[test]
    fn shape_balance_counts_only_each_bands_entropy_delta() {
        assert_eq!(update_shape_balance(0, 80, 400, 456), Ok(24));
        assert_eq!(update_shape_balance(24, 64, 456, 504), Ok(40));
        assert_eq!(update_shape_balance(0, 80, 4_000, 4_056), Ok(24));
        assert_eq!(update_shape_balance(0, 8, 10, 9), Err(Error::InvalidPacket));
    }

    #[test]
    fn exhausted_range_decodes_as_silence_without_following_symbols() {
        let mut bytes = [0u8; 8];
        let mut encoder = crate::RangeEncoder::new(&mut bytes);
        encoder.encode_bit_logp(true, 15).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let mut energies = LogEnergies::new();
        let mut residuals = [[0; BAND_COUNT]; 2];
        let plan = decode_plan(
            &mut decoder,
            FrameConfig {
                frame_bytes: bytes.len(),
                channels: 1,
                lm: 0,
                start: 0,
                end: 13,
            },
            &mut energies,
            &mut residuals,
        )
        .unwrap();
        assert!(plan.silence);
        assert!(plan.allocation.is_none());
        assert_eq!(plan.trim, 5);
    }

    #[test]
    fn invalid_frame_geometry_is_rejected_before_state_changes() {
        let bytes = [0; 8];
        let mut energies = LogEnergies::new();
        let before = energies;
        let mut residuals = [[0; BAND_COUNT]; 2];
        assert_eq!(
            decode_plan(
                &mut RangeDecoder::new(&bytes),
                FrameConfig {
                    frame_bytes: 8,
                    channels: 3,
                    lm: 0,
                    start: 0,
                    end: 13,
                },
                &mut energies,
                &mut residuals,
            ),
            Err(Error::InvalidPacket)
        );
        assert_eq!(energies, before);
    }

    #[test]
    fn zero_shape_plan_traverses_bands_without_allocating() {
        let config = FrameConfig {
            frame_bytes: 8,
            channels: 1,
            lm: 0,
            start: 8,
            end: 9,
        };
        let mut caps = [0; BAND_COUNT];
        band_caps(1, 0, &mut caps).unwrap();
        let plan = FramePlan {
            silence: false,
            post_filter: None,
            transient: false,
            intra_energy: false,
            tf_adjustments: [0; BAND_COUNT],
            tf_select: false,
            spread: 2,
            caps,
            boosts: [0; BAND_COUNT],
            total_boost: 0,
            trim: 5,
            reservations: Reservations {
                available: 0,
                anti_collapse: 0,
                skip: 0,
                intensity: 0,
                dual_stereo: 0,
            },
            allocation: Some(FinalAllocation {
                coded_bands: 9,
                intensity: 8,
                dual_stereo: false,
                bands: crate::celt_allocation::AllocationResult {
                    shape: [0; BAND_COUNT],
                    fine: [0; BAND_COUNT],
                    priority: [0; BAND_COUNT],
                    balance: 0,
                },
            }),
        };
        let bytes = [0u8; 8];
        let mut energies = LogEnergies::new();
        let history = LogEnergies::new();
        let mut spectra = [1.0f32; 120];
        let mut pulses = [1i32; 120];
        let mut tf_scratch = [0.0f32; 176];
        let mut recurrence = [0u32; crate::pvq::MAX_PULSES + 1];
        let result = decode_shapes(
            &mut RangeDecoder::new(&bytes),
            config,
            &plan,
            &mut energies,
            &history,
            &history,
            &mut spectra,
            &mut pulses,
            &mut tf_scratch,
            &mut recurrence,
            5,
        )
        .unwrap();
        assert!(!result.anti_collapse);
        assert_eq!(result.seed, 5);
        assert!(spectra.iter().all(|value| *value == 0.0));
        assert!(pulses.iter().all(|value| *value == 0));
    }

    #[test]
    fn frame_prefix_encoder_and_decoder_are_symmetric() {
        let config = FrameConfig {
            frame_bytes: 512,
            channels: 1,
            lm: 2,
            start: 0,
            end: BAND_COUNT,
        };
        let request = EncodeRequest {
            silence: false,
            post_filter: None,
            transient: false,
            intra_energy: true,
            tf_flags: [false; BAND_COUNT],
            tf_select: false,
            spread: 2,
            residuals: [[0; BAND_COUNT]; 2],
            boosts: [0; BAND_COUNT],
            trim: 5,
            coded_bands: BAND_COUNT,
            intensity: BAND_COUNT,
            dual_stereo: false,
        };
        let target = LogEnergies::new();
        let mut encoded_energies = LogEnergies::new();
        let mut coded = [[0; BAND_COUNT]; 2];
        let mut bytes = [0u8; 512];
        let encoded_plan = {
            let mut encoder = RangeEncoder::new(&mut bytes);
            let plan = encode_plan(
                &mut encoder,
                config,
                &request,
                &target,
                &mut encoded_energies,
                &mut coded,
            )
            .unwrap();
            encoder.finish().unwrap();
            plan
        };
        let mut decoded_energies = LogEnergies::new();
        let mut residuals = [[0; BAND_COUNT]; 2];
        let decoded_plan = decode_plan(
            &mut RangeDecoder::new(&bytes),
            config,
            &mut decoded_energies,
            &mut residuals,
        )
        .unwrap();
        assert_eq!(decoded_plan, encoded_plan);
        assert_eq!(decoded_energies, encoded_energies);
        assert_eq!(residuals, coded);
    }

    #[test]
    fn complete_transient_mono_frame_writer_matches_shape_decoder() {
        let config = FrameConfig {
            frame_bytes: 512,
            channels: 1,
            lm: 2,
            start: 0,
            end: BAND_COUNT,
        };
        let request = EncodeRequest {
            silence: false,
            post_filter: None,
            transient: true,
            intra_energy: true,
            tf_flags: [false; BAND_COUNT],
            tf_select: false,
            spread: 0,
            residuals: [[0; BAND_COUNT]; 2],
            boosts: [0; BAND_COUNT],
            trim: 5,
            coded_bands: BAND_COUNT,
            intensity: BAND_COUNT,
            dual_stereo: false,
        };
        let target = LogEnergies::new();
        let mut encoded_energies = LogEnergies::new();
        let mut coded = [[0; BAND_COUNT]; 2];
        let mut spectra = [0.0f32; 480];
        for band in 0..BAND_COUNT {
            let range = bands::band_range(band, 2).unwrap();
            for (offset, value) in spectra[range.clone()].iter_mut().enumerate() {
                *value = offset as f32 + 1.0;
            }
            let norm = mrml_math::sqrt(
                spectra[range.clone()]
                    .iter()
                    .fold(0.0, |sum, value| sum + value * value),
            );
            for value in &mut spectra[range] {
                *value /= norm;
            }
        }
        let mut encoded_pulses = [0i32; 480];
        let mut recurrence = [0u32; crate::pvq::MAX_PULSES + 1];
        let mut bytes = [0u8; 512];
        let (encoded_plan, encoded_result) = {
            let mut encoder = RangeEncoder::new(&mut bytes);
            let plan = encode_plan(
                &mut encoder,
                config,
                &request,
                &target,
                &mut encoded_energies,
                &mut coded,
            )
            .unwrap();
            let result = encode_shapes_mono(
                &mut encoder,
                config,
                &plan,
                &spectra,
                &mut encoded_pulses,
                &mut recurrence,
                &target,
                &mut encoded_energies,
                9,
            )
            .unwrap();
            encoder.finish().unwrap();
            (plan, result)
        };
        let mut decoder = RangeDecoder::new(&bytes);
        let mut decoded_energies = LogEnergies::new();
        let mut residuals = [[0; BAND_COUNT]; 2];
        let decoded_plan =
            decode_plan(&mut decoder, config, &mut decoded_energies, &mut residuals).unwrap();
        let history = LogEnergies::new();
        let mut decoded_spectra = [0.0f32; 480];
        let mut decoded_pulses = [0i32; 480];
        let mut tf_scratch = [0.0f32; 176];
        let decoded_result = decode_shapes(
            &mut decoder,
            config,
            &decoded_plan,
            &mut decoded_energies,
            &history,
            &history,
            &mut decoded_spectra,
            &mut decoded_pulses,
            &mut tf_scratch,
            &mut recurrence,
            9,
        )
        .unwrap();
        assert_eq!(decoded_plan, encoded_plan);
        assert_eq!(decoded_result, encoded_result);
        assert_eq!(decoded_energies, encoded_energies);
        assert_eq!(decoded_pulses, encoded_pulses);
    }
}
