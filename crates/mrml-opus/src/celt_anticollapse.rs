//! CELT transient anti-collapse reconstruction.

use crate::{
    Error,
    bands::{BAND_COUNT, BAND_EDGES_2_5_MS, band_range},
};

/// Restores energy in short MDCT blocks whose PVQ shape collapsed to zero.
///
/// `spectra` is channel-major and contains 100 bins shifted by `lm` per
/// channel. `collapse_masks` is band-major (`band * channels + channel`).
/// Log energies use base two. The two history arrays always retain two
/// channels, because CELT consults both histories after stereo-to-mono
/// transitions.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    spectra: &mut [f32],
    collapse_masks: &[u8],
    lm: u8,
    channels: usize,
    start: usize,
    end: usize,
    log_energy: &[f32],
    previous_log_energy: &[f32],
    older_log_energy: &[f32],
    shape_bits: &[i32],
    mut seed: u32,
) -> Result<u32, Error> {
    if lm > 3 || !(1..=2).contains(&channels) || start > end || end > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    let channel_size = usize::from(BAND_EDGES_2_5_MS[BAND_COUNT]) << lm;
    if spectra.len() < channel_size * channels
        || collapse_masks.len() < BAND_COUNT * channels
        || log_energy.len() < BAND_COUNT * channels
        || previous_log_energy.len() < BAND_COUNT * 2
        || older_log_energy.len() < BAND_COUNT * 2
        || shape_bits.len() < BAND_COUNT
    {
        return Err(Error::BufferTooSmall);
    }
    let blocks = 1usize << lm;
    let valid_mask = ((1u16 << blocks) - 1) as u8;
    for band in start..end {
        if shape_bits[band] < 0 {
            return Err(Error::InvalidPacket);
        }
        let base_width = usize::from(
            BAND_EDGES_2_5_MS[band + 1]
                .checked_sub(BAND_EDGES_2_5_MS[band])
                .ok_or(Error::InvalidPacket)?,
        );
        let depth = ((shape_bits[band] + 1) / i32::try_from(base_width).unwrap_or(1)) >> lm;
        let threshold = 0.5 * mrml_math::pow(2.0, -(depth as f32) * 0.125);
        let inverse_sqrt = 1.0 / mrml_math::sqrt((base_width * blocks) as f32);
        let range = band_range(band, lm)?;
        for channel in 0..channels {
            let mask = collapse_masks[band * channels + channel];
            if mask & !valid_mask != 0 {
                return Err(Error::InvalidPacket);
            }
            if mask == valid_mask {
                continue;
            }
            let history = if channels == 1 {
                previous_log_energy[band].max(previous_log_energy[BAND_COUNT + band])
            } else {
                previous_log_energy[channel * BAND_COUNT + band]
            };
            let older = if channels == 1 {
                older_log_energy[band].max(older_log_energy[BAND_COUNT + band])
            } else {
                older_log_energy[channel * BAND_COUNT + band]
            };
            let energy_drop =
                (log_energy[channel * BAND_COUNT + band] - history.min(older)).max(0.0);
            let mut noise = 2.0 * mrml_math::pow(2.0, -energy_drop);
            if lm == 3 {
                noise *= core::f32::consts::SQRT_2;
            }
            noise = noise.min(threshold) * inverse_sqrt;
            let channel_offset = channel * channel_size;
            for block in 0..blocks {
                if mask & (1 << block) != 0 {
                    continue;
                }
                for coefficient in 0..base_width {
                    seed = lcg(seed);
                    let index = channel_offset + range.start + coefficient * blocks + block;
                    spectra[index] = if seed & 0x8000 != 0 { noise } else { -noise };
                }
            }
            normalize(&mut spectra[channel_offset + range.start..channel_offset + range.end])?;
        }
    }
    Ok(seed)
}

const fn lcg(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn normalize(vector: &mut [f32]) -> Result<(), Error> {
    let energy = vector.iter().fold(0.0, |sum, &value| sum + value * value);
    if !energy.is_finite() || energy <= 0.0 {
        return Err(Error::InvalidPacket);
    }
    let scale = 1.0 / mrml_math::sqrt(energy);
    for value in vector {
        *value *= scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_only_collapsed_short_blocks_and_renormalizes() {
        let lm = 2;
        let mut spectra = [0.0f32; 400];
        let band = 17;
        let range = band_range(band, lm).unwrap();
        for coefficient in 0..range.len() / 4 {
            spectra[range.start + coefficient * 4] = 0.25;
            spectra[range.start + coefficient * 4 + 2] = -0.25;
        }
        let mut masks = [0u8; BAND_COUNT];
        masks[band] = 0b0101;
        let energies = [0.0f32; BAND_COUNT];
        let history = [0.0f32; BAND_COUNT * 2];
        let mut bits = [0i32; BAND_COUNT];
        bits[band] = 64;
        let seed = apply(
            &mut spectra,
            &masks,
            lm,
            1,
            band,
            band + 1,
            &energies,
            &history,
            &history,
            &bits,
            1,
        )
        .unwrap();
        assert_ne!(seed, 1);
        let band_vector = &spectra[range];
        for coefficient in 0..band_vector.len() / 4 {
            assert_eq!(band_vector[coefficient * 4].signum(), 1.0);
            assert_eq!(band_vector[coefficient * 4 + 2].signum(), -1.0);
            assert_ne!(band_vector[coefficient * 4 + 1], 0.0);
            assert_ne!(band_vector[coefficient * 4 + 3], 0.0);
        }
        let energy = band_vector
            .iter()
            .fold(0.0, |sum, value| sum + value * value);
        assert!((energy - 1.0).abs() < 0.000_01);
    }

    #[test]
    fn complete_masks_leave_spectrum_and_seed_unchanged() {
        let mut spectra = [0.0f32; 800];
        spectra[0] = 1.0;
        let masks = [0xffu8; BAND_COUNT * 2];
        let energies = [0.0f32; BAND_COUNT * 2];
        let history = [0.0f32; BAND_COUNT * 2];
        let bits = [0i32; BAND_COUNT];
        assert_eq!(
            apply(
                &mut spectra,
                &masks,
                3,
                1,
                0,
                BAND_COUNT,
                &energies[..BAND_COUNT],
                &history,
                &history,
                &bits,
                99,
            ),
            Ok(99)
        );
        assert_eq!(spectra[0], 1.0);
    }

    #[test]
    fn malformed_masks_and_lengths_are_rejected() {
        let mut spectra = [0.0f32; 400];
        let mut masks = [0u8; BAND_COUNT];
        masks[0] = 0x80;
        let energies = [0.0f32; BAND_COUNT];
        let history = [0.0f32; BAND_COUNT * 2];
        let bits = [0i32; BAND_COUNT];
        assert_eq!(
            apply(
                &mut spectra,
                &masks,
                2,
                1,
                0,
                1,
                &energies,
                &history,
                &history,
                &bits,
                0,
            ),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            apply(
                &mut spectra[..3],
                &masks,
                2,
                1,
                0,
                1,
                &energies,
                &history,
                &history,
                &bits,
                0,
            ),
            Err(Error::BufferTooSmall)
        );
    }
}
