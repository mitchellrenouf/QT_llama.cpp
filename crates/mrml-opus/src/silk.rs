//! Allocation-free SILK excitation and synthesis primitives.

use crate::Error;

const LTP_HISTORY: usize = 512;
const LPC_HISTORY: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalType {
    Inactive,
    Unvoiced,
    Voiced,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationOffset {
    Low,
    High,
}

pub const fn quantization_offset(signal: SignalType, quantization: QuantizationOffset) -> i32 {
    match (signal, quantization) {
        (SignalType::Inactive | SignalType::Unvoiced, QuantizationOffset::Low) => 25,
        (SignalType::Inactive | SignalType::Unvoiced, QuantizationOffset::High) => 60,
        (SignalType::Voiced, QuantizationOffset::Low) => 8,
        (SignalType::Voiced, QuantizationOffset::High) => 25,
    }
}

/// Reconstructs RFC 6716 Section 4.2.7.8.6 Q23 excitation exactly.
pub fn reconstruct_excitation(
    raw: &[i32],
    mut seed: u32,
    signal: SignalType,
    quantization: QuantizationOffset,
    output_q23: &mut [i32],
) -> Result<u32, Error> {
    if output_q23.len() < raw.len() {
        return Err(Error::BufferTooSmall);
    }
    let offset = quantization_offset(signal, quantization);
    for (&raw, output) in raw.iter().zip(output_q23.iter_mut()) {
        let sign = raw.signum();
        let mut excitation = raw
            .checked_mul(256)
            .and_then(|value| value.checked_sub(sign * 20))
            .and_then(|value| value.checked_add(offset))
            .ok_or(Error::InvalidPacket)?;
        seed = 196_314_165u32.wrapping_mul(seed).wrapping_add(907_633_515);
        if seed & 0x8000_0000 != 0 {
            excitation = excitation.checked_neg().ok_or(Error::InvalidPacket)?;
        }
        *output = excitation;
        seed = seed.wrapping_add(raw as u32);
    }
    Ok(seed)
}

#[derive(Clone, Copy, Debug)]
pub struct VoicedParameters {
    pub pitch_lag: u16,
    pub coefficients_q7: [i8; 5],
}

pub struct Synthesis {
    residual: [f32; LTP_HISTORY],
    residual_position: usize,
    lpc: [f32; LPC_HISTORY],
    output: [f32; LPC_HISTORY],
    lpc_position: usize,
    last_voiced: Option<VoicedParameters>,
}

impl Synthesis {
    pub const fn new() -> Self {
        Self {
            residual: [0.0; LTP_HISTORY],
            residual_position: 0,
            lpc: [0.0; LPC_HISTORY],
            output: [0.0; LPC_HISTORY],
            lpc_position: 0,
            last_voiced: None,
        }
    }

    pub fn reset(&mut self) {
        self.residual.fill(0.0);
        self.lpc.fill(0.0);
        self.output.fill(0.0);
        self.residual_position = 0;
        self.lpc_position = 0;
        self.last_voiced = None;
    }

    pub fn scale_ltp_history(&mut self, scale_q14: u16) -> Result<(), Error> {
        if scale_q14 > 16_384 {
            return Err(Error::InvalidPacket);
        }
        let scale = f32::from(scale_q14) / 16_384.0;
        for value in &mut self.residual {
            *value *= scale;
        }
        Ok(())
    }

    /// Runs one SILK subframe through optional LTP and mandatory LPC synthesis.
    pub fn subframe(
        &mut self,
        excitation_q23: &[i32],
        gain_q16: u32,
        lpc_q12: &[i16],
        voiced: Option<VoicedParameters>,
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.subframe_rfc(excitation_q23, gain_q16, lpc_q12, voiced, 16_384, 0, output)
    }

    /// Runs one subframe with RFC 6716 Section 4.2.7.9.1 rewhitening.
    #[allow(clippy::too_many_arguments)]
    pub fn subframe_rfc(
        &mut self,
        excitation_q23: &[i32],
        gain_q16: u32,
        lpc_q12: &[i16],
        voiced: Option<VoicedParameters>,
        ltp_scale_q14: u16,
        rewhiten_distance: usize,
        output: &mut [f32],
    ) -> Result<(), Error> {
        if output.len() < excitation_q23.len() || !matches!(lpc_q12.len(), 10 | 16) || gain_q16 == 0
        {
            return Err(Error::InvalidFrameSize);
        }
        if let Some(parameters) = voiced
            && !(16..=288).contains(&parameters.pitch_lag)
        {
            return Err(Error::InvalidPacket);
        }
        if ltp_scale_q14 > 16_384 || rewhiten_distance > 3 * excitation_q23.len() {
            return Err(Error::InvalidPacket);
        }
        if let Some(parameters) = voiced {
            self.rewhiten(
                parameters.pitch_lag.into(),
                gain_q16,
                lpc_q12,
                ltp_scale_q14,
                rewhiten_distance,
            )?;
        }
        let gain = gain_q16 as f32 / 65_536.0;
        for (&excitation, output) in excitation_q23.iter().zip(output.iter_mut()) {
            let mut residual = excitation as f32 / 8_388_608.0;
            if let Some(parameters) = voiced {
                let pitch = usize::from(parameters.pitch_lag);
                for (tap, coefficient) in parameters.coefficients_q7.iter().enumerate() {
                    residual +=
                        self.past_residual(pitch + tap - 2) * f32::from(*coefficient) / 128.0;
                }
            }
            self.residual[self.residual_position] = residual;
            self.residual_position = (self.residual_position + 1) % LTP_HISTORY;
            let mut reconstructed = gain * residual;
            for (tap, coefficient) in lpc_q12.iter().enumerate() {
                reconstructed += self.past_lpc(tap + 1) * f32::from(*coefficient) / 4096.0;
            }
            self.lpc[self.lpc_position] = reconstructed;
            let clamped = reconstructed.clamp(-1.0, 1.0);
            self.output[self.lpc_position] = clamped;
            self.lpc_position = (self.lpc_position + 1) % LPC_HISTORY;
            *output = clamped;
        }
        self.last_voiced = voiced;
        Ok(())
    }

    fn rewhiten(
        &mut self,
        pitch: usize,
        gain_q16: u32,
        lpc_q12: &[i16],
        scale_q14: u16,
        recent_distance: usize,
    ) -> Result<(), Error> {
        if pitch + lpc_q12.len() + 2 >= LPC_HISTORY
            || recent_distance + lpc_q12.len() >= LPC_HISTORY
        {
            return Err(Error::InvalidPacket);
        }
        let gain = gain_q16 as f32;
        for distance in recent_distance + 1..=pitch + 2 {
            let mut whitened = self.past_output(distance);
            for (tap, &coefficient) in lpc_q12.iter().enumerate() {
                whitened -= self.past_output(distance + tap + 1) * f32::from(coefficient) / 4096.0;
            }
            self.set_past_residual(
                distance,
                4.0 * f32::from(scale_q14) / gain * whitened.clamp(-1.0, 1.0),
            );
        }
        for distance in 1..=recent_distance {
            let mut whitened = self.past_lpc(distance);
            for (tap, &coefficient) in lpc_q12.iter().enumerate() {
                whitened -= self.past_lpc(distance + tap + 1) * f32::from(coefficient) / 4096.0;
            }
            self.set_past_residual(distance, 65_536.0 / gain * whitened);
        }
        Ok(())
    }

    /// SILK PLC using the retained pitch and LPC state with progressive decay.
    pub fn conceal(&mut self, lpc_q12: &[i16], output: &mut [f32]) -> Result<(), Error> {
        if !matches!(lpc_q12.len(), 10 | 16) {
            return Err(Error::InvalidFrameSize);
        }
        let output_len = output.len().max(1) as f32;
        for (index, sample) in output.iter_mut().enumerate() {
            let decay = 1.0 - 0.35 * index as f32 / output_len;
            let mut residual = 0.0;
            if let Some(parameters) = self.last_voiced {
                for (tap, coefficient) in parameters.coefficients_q7.iter().enumerate() {
                    residual += self.past_residual(usize::from(parameters.pitch_lag) + tap - 2)
                        * f32::from(*coefficient)
                        / 128.0;
                }
            }
            residual *= decay;
            self.residual[self.residual_position] = residual;
            self.residual_position = (self.residual_position + 1) % LTP_HISTORY;
            let mut reconstructed = residual;
            for (tap, coefficient) in lpc_q12.iter().enumerate() {
                reconstructed += self.past_lpc(tap + 1) * f32::from(*coefficient) / 4096.0;
            }
            self.lpc[self.lpc_position] = reconstructed;
            let clamped = reconstructed.clamp(-1.0, 1.0);
            self.output[self.lpc_position] = clamped;
            self.lpc_position = (self.lpc_position + 1) % LPC_HISTORY;
            *sample = clamped;
        }
        Ok(())
    }

    fn past_residual(&self, distance: usize) -> f32 {
        self.residual[(self.residual_position + LTP_HISTORY - distance) % LTP_HISTORY]
    }
    fn past_lpc(&self, distance: usize) -> f32 {
        self.lpc[(self.lpc_position + LPC_HISTORY - distance) % LPC_HISTORY]
    }
    fn past_output(&self, distance: usize) -> f32 {
        self.output[(self.lpc_position + LPC_HISTORY - distance) % LPC_HISTORY]
    }
    fn set_past_residual(&mut self, distance: usize, value: f32) {
        let index = (self.residual_position + LTP_HISTORY - distance) % LTP_HISTORY;
        self.residual[index] = value;
    }
}
impl Default for Synthesis {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StereoUnmixer {
    previous_w0_q13: i16,
    previous_w1_q13: i16,
    mid_2: f32,
    mid_1: f32,
    side_1: f32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StereoWeights {
    pub w0_q13: i16,
    pub w1_q13: i16,
}
impl StereoUnmixer {
    pub const fn new() -> Self {
        Self {
            previous_w0_q13: 0,
            previous_w1_q13: 0,
            mid_2: 0.0,
            mid_1: 0.0,
            side_1: 0.0,
        }
    }
    pub fn unmix(
        &mut self,
        mid: &[f32],
        side: Option<&[f32]>,
        sample_rate: u32,
        weights: StereoWeights,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        if left.len() < mid.len()
            || right.len() < mid.len()
            || side.is_some_and(|values| values.len() < mid.len())
            || !matches!(sample_rate, 8_000 | 12_000 | 16_000)
        {
            return Err(Error::InvalidFrameSize);
        }
        let interpolation = (sample_rate as usize * 8 / 1000).max(1);
        for index in 0..mid.len() {
            let step = index.min(interpolation) as f32 / interpolation as f32;
            let w0 = (f32::from(self.previous_w0_q13)
                + step * f32::from(weights.w0_q13 - self.previous_w0_q13))
                / 8192.0;
            let w1 = (f32::from(self.previous_w1_q13)
                + step * f32::from(weights.w1_q13 - self.previous_w1_q13))
                / 8192.0;
            let low = (self.mid_2 + 2.0 * self.mid_1 + mid[index]) * 0.25;
            left[index] = ((1.0 + w1) * self.mid_1 + self.side_1 + w0 * low).clamp(-1.0, 1.0);
            right[index] = ((1.0 - w1) * self.mid_1 - self.side_1 - w0 * low).clamp(-1.0, 1.0);
            self.mid_2 = self.mid_1;
            self.mid_1 = mid[index];
            self.side_1 = side.map_or(0.0, |values| values[index]);
        }
        self.previous_w0_q13 = weights.w0_q13;
        self.previous_w1_q13 = weights.w1_q13;
        Ok(())
    }
    pub const fn current_weights(&self) -> StereoWeights {
        StereoWeights {
            w0_q13: self.previous_w0_q13,
            w1_q13: self.previous_w1_q13,
        }
    }
}
impl Default for StereoUnmixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Non-normative linear SILK resampler with caller-selected output length.
pub fn resample_linear(input: &[f32], output: &mut [f32]) -> Result<(), Error> {
    if input.is_empty() || output.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    if output.len() == 1 {
        output[0] = input[0];
        return Ok(());
    }
    let scale = (input.len() - 1) as f32 / (output.len() - 1) as f32;
    for (index, sample) in output.iter_mut().enumerate() {
        let position = index as f32 * scale;
        let left = position as usize;
        let right = (left + 1).min(input.len() - 1);
        let fraction = position - left as f32;
        *sample = input[left] + fraction * (input[right] - input[left]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn excitation_is_deterministic_and_updates_seed() {
        let raw = [0, 1, -2, 3];
        let mut output = [0; 4];
        let seed = reconstruct_excitation(
            &raw,
            2,
            SignalType::Voiced,
            QuantizationOffset::Low,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [8, 244, -484, -756]);
        assert_eq!(
            seed,
            2u32.wrapping_mul(196_314_165)
                .wrapping_add(907_633_515)
                .wrapping_mul(196_314_165)
                .wrapping_add(907_633_515)
                .wrapping_add(1)
                .wrapping_mul(196_314_165)
                .wrapping_add(907_633_515)
                .wrapping_add((-2i32) as u32)
                .wrapping_mul(196_314_165)
                .wrapping_add(907_633_515)
                .wrapping_add(3)
        );
    }
    #[test]
    fn unvoiced_lpc_synthesis_and_clamping() {
        let mut synthesis = Synthesis::new();
        let excitation = [1 << 22; 80];
        let mut output = [0.0; 80];
        synthesis
            .subframe(&excitation, 65_536, &[0; 10], None, &mut output)
            .unwrap();
        assert!(
            output
                .iter()
                .all(|value| (*value - 0.5).abs() < f32::EPSILON)
        );
        let loud = [i32::MAX; 4];
        synthesis
            .subframe(&loud, 65_536, &[0; 10], None, &mut output[..4])
            .unwrap();
        assert_eq!(&output[..4], &[1.0; 4]);
    }
    #[test]
    fn voiced_synthesis_and_plc_remain_finite() {
        let mut synthesis = Synthesis::new();
        let excitation = [1 << 20; 320];
        let voiced = VoicedParameters {
            pitch_lag: 32,
            coefficients_q7: [0, 0, 96, 0, 0],
        };
        let mut output = [0.0; 320];
        synthesis
            .subframe(&excitation, 65_536, &[0; 16], Some(voiced), &mut output)
            .unwrap();
        let mut lost = [0.0; 160];
        synthesis.conceal(&[0; 16], &mut lost).unwrap();
        assert!(
            lost.iter()
                .all(|value| value.is_finite() && value.abs() <= 1.0)
        );
        assert!(lost.iter().any(|value| value.abs() > 0.0));
    }
    #[test]
    fn rfc_rewhitening_uses_ltp_scale_and_recent_lpc_regions() {
        fn render(scale: u16) -> [f32; 80] {
            let mut synthesis = Synthesis::new();
            let history = [1 << 21; 160];
            let mut discarded = [0.0; 160];
            synthesis
                .subframe(&history, 65_536, &[0; 16], None, &mut discarded)
                .unwrap();
            let mut output = [0.0; 80];
            synthesis
                .subframe_rfc(
                    &[0; 80],
                    65_536,
                    &[0; 16],
                    Some(VoicedParameters {
                        pitch_lag: 80,
                        coefficients_q7: [0, 0, 96, 0, 0],
                    }),
                    scale,
                    40,
                    &mut output,
                )
                .unwrap();
            output
        }
        let half = render(8_192);
        let full = render(16_384);
        assert!(half.iter().zip(full).any(|(a, b)| (*a - b).abs() > 1e-6));
        assert!(half.iter().all(|value| value.is_finite()));
    }
    #[test]
    fn stereo_unmix_delays_and_resampler_preserves_endpoints() {
        let mid = [0.25; 128];
        let side = [0.1; 128];
        let mut left = [0.0; 128];
        let mut right = [0.0; 128];
        StereoUnmixer::new()
            .unmix(
                &mid,
                Some(&side),
                16_000,
                StereoWeights {
                    w0_q13: 0,
                    w1_q13: 0,
                },
                &mut left,
                &mut right,
            )
            .unwrap();
        assert_eq!(left[0], 0.0);
        assert!((left[2] - 0.35).abs() < 1e-6);
        assert!((right[2] - 0.15).abs() < 1e-6);
        let mut up = [0.0; 7];
        resample_linear(&[0.0, 1.0, 0.0], &mut up).unwrap();
        assert_eq!(up[0], 0.0);
        assert_eq!(up[6], 0.0);
        assert_eq!(up[3], 1.0);
    }
}
