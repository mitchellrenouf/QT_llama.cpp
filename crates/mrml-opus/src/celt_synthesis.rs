//! Allocation-free CELT spectral denormalization and overlap synthesis.

use crate::{Error, bands, celt::Deemphasis};

pub const OVERLAP: usize = 120;
pub const MAX_FRAME_SAMPLES: usize = 960;

#[derive(Clone, Copy, Debug)]
pub struct SynthesisState {
    overlap: [[f32; OVERLAP]; 2],
    deemphasis: [Deemphasis; 2],
}

impl SynthesisState {
    pub const fn new() -> Self {
        Self {
            overlap: [[0.0; OVERLAP]; 2],
            deemphasis: [Deemphasis::new(); 2],
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Synthesizes one channel from CELT-normalized spectral coefficients.
    /// `transform_scratch` holds `frame_samples + OVERLAP` temporary samples.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_channel(
        &mut self,
        channel: usize,
        normalized: &[f32],
        lm: u8,
        transient: bool,
        start: usize,
        end: usize,
        log_energies: &[f32],
        spectral_scratch: &mut [f32],
        transform_scratch: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        let n = 120usize
            .checked_shl(u32::from(lm))
            .ok_or(Error::InvalidFrameSize)?;
        if channel >= 2
            || n > MAX_FRAME_SAMPLES
            || spectral_scratch.len() < n
            || transform_scratch.len() < n + OVERLAP
            || output.len() < n
        {
            return Err(Error::BufferTooSmall);
        }
        spectral_scratch[..n].fill(0.0);
        bands::denormalize_log_bands(
            normalized,
            lm,
            start,
            end,
            log_energies,
            &mut spectral_scratch[..n],
        )?;
        if transient {
            inverse_short_blocks(
                &spectral_scratch[..n],
                1usize << lm,
                &mut transform_scratch[..n + OVERLAP],
            )?;
        } else {
            inverse_low_overlap(
                &spectral_scratch[..n],
                &mut transform_scratch[..n + OVERLAP],
            )?;
        }
        for (index, sample) in output[..n].iter_mut().enumerate() {
            *sample = transform_scratch[index]
                + if index < OVERLAP {
                    self.overlap[channel][index]
                } else {
                    0.0
                };
        }
        self.overlap[channel].copy_from_slice(&transform_scratch[n..n + OVERLAP]);
        self.deemphasis[channel].apply(&mut output[..n]);
        Ok(())
    }
}

/// Synthesizes interleaved 2.5 ms transient blocks and overlap-adds adjacent
/// short transforms into one frame-sized temporary.
pub fn inverse_short_blocks(
    coefficients: &[f32],
    blocks: usize,
    output: &mut [f32],
) -> Result<(), Error> {
    if !matches!(blocks, 1 | 2 | 4 | 8)
        || coefficients.len() != OVERLAP * blocks
        || output.len() != coefficients.len() + OVERLAP
    {
        return Err(Error::InvalidFrameSize);
    }
    output.fill(0.0);
    let mut block_coefficients = [0.0f32; OVERLAP];
    let mut block_output = [0.0f32; OVERLAP * 2];
    for block in 0..blocks {
        for frequency in 0..OVERLAP {
            block_coefficients[frequency] = coefficients[frequency * blocks + block];
        }
        inverse_low_overlap(&block_coefficients, &mut block_output)?;
        let offset = block * OVERLAP;
        for (target, &sample) in output[offset..offset + OVERLAP * 2]
            .iter_mut()
            .zip(&block_output)
        {
            *target += sample;
        }
    }
    Ok(())
}

impl Default for SynthesisState {
    fn default() -> Self {
        Self::new()
    }
}

/// Direct O(N²) inverse low-overlap MDCT conformance baseline.
pub fn inverse_low_overlap(coefficients: &[f32], output: &mut [f32]) -> Result<(), Error> {
    let n = coefficients.len();
    if !(OVERLAP..=MAX_FRAME_SAMPLES).contains(&n) || output.len() != n + OVERLAP {
        return Err(Error::InvalidFrameSize);
    }
    let crop = (n - OVERLAP) / 2;
    let phase_scale = core::f32::consts::PI / n as f32;
    let amplitude_scale = 1.0 / n as f32;
    for (index, sample) in output.iter_mut().enumerate() {
        let full_index = index + crop;
        let time = full_index as f32 + 0.5 + n as f32 * 0.5;
        let mut sum = 0.0;
        for (frequency, &coefficient) in coefficients.iter().enumerate() {
            sum += coefficient * mrml_math::cos(phase_scale * time * (frequency as f32 + 0.5));
        }
        let mut value = sum * amplitude_scale;
        if index < OVERLAP {
            value *= crate::celt::window_sample(index, OVERLAP)?;
        } else if index >= n {
            value *= crate::celt::window_sample(n + OVERLAP - 1 - index, OVERLAP)?;
        }
        *sample = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_spectrum_preserves_zero_state() {
        let mut state = SynthesisState::new();
        let normalized = [0.0f32; 120];
        let energies = [0.0f32; bands::BAND_COUNT];
        let mut spectral = [1.0f32; 120];
        let mut transform = [1.0f32; 240];
        let mut output = [1.0f32; 120];
        state
            .synthesize_channel(
                0,
                &normalized,
                0,
                false,
                0,
                bands::BAND_COUNT,
                &energies,
                &mut spectral,
                &mut transform,
                &mut output,
            )
            .unwrap();
        assert!(output.iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn inverse_is_finite_for_every_frame_size() {
        let coefficients = [0.25f32; MAX_FRAME_SAMPLES];
        let mut output = [0.0f32; MAX_FRAME_SAMPLES + OVERLAP];
        for n in [120, 240, 480, 960] {
            inverse_low_overlap(&coefficients[..n], &mut output[..n + OVERLAP]).unwrap();
            assert!(
                output[..n + OVERLAP]
                    .iter()
                    .all(|sample| sample.is_finite())
            );
        }
    }

    #[test]
    fn transient_blocks_deinterleave_and_overlap_to_frame_length() {
        let mut coefficients = [0.0f32; 480];
        coefficients[2] = 1.0;
        let mut output = [0.0f32; 600];
        inverse_short_blocks(&coefficients, 4, &mut output).unwrap();
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().any(|&sample| sample != 0.0));
    }
}
