//! Stateful integration of one mono SILK frame parameter stream.

use crate::{
    Bandwidth, Error, RangeDecoder, RangeEncoder,
    silk::{
        QuantizationOffset, SignalType, StereoUnmixer, Synthesis, VoicedParameters,
        reconstruct_excitation,
    },
    silk_entropy::{MAX_EXCITATION_SAMPLES, decode_excitation_blocks},
    silk_frame::{decode_frame_type, decode_gains, encode_frame_type, encode_gains},
    silk_lsf::{LsfIndices, decode_lsf, encode_lsf},
    silk_pitch::{
        LtpFilters, decode_contour, decode_ltp_filters, decode_primary_lag, decode_seed,
        encode_contour, encode_ltp_filters, encode_primary_lag, encode_seed,
    },
    silk_stereo::{
        StereoPrediction, decode_mid_only, decode_prediction, encode_mid_only, encode_prediction,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameContext {
    pub active: bool,
    pub independent_gain: bool,
    pub absolute_pitch: bool,
    pub ltp_scale_present: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonoFrameParameters {
    pub signal: SignalType,
    pub quantization: QuantizationOffset,
    pub gain_symbols: [u8; 4],
    pub lsf: LsfIndices,
    pub primary_pitch: Option<i16>,
    pub contour_index: u8,
    pub ltp: Option<LtpFilters>,
    pub seed: u8,
    pub rate_level: u8,
    pub excitation: [i32; MAX_EXCITATION_SAMPLES],
}

pub struct MonoEncoder {
    previous_gain: Option<u8>,
    previous_pitch: Option<i16>,
}

impl MonoEncoder {
    pub const fn new() -> Self {
        Self {
            previous_gain: None,
            previous_pitch: None,
        }
    }
    pub fn reset(&mut self) {
        self.previous_gain = None;
        self.previous_pitch = None;
    }
    pub const fn has_previous_pitch(&self) -> bool {
        self.previous_pitch.is_some()
    }
    pub fn encode_frame(
        &mut self,
        encoder: &mut RangeEncoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        context: FrameContext,
        parameters: &MonoFrameParameters,
    ) -> Result<usize, Error> {
        let subframe_samples = match bandwidth {
            Bandwidth::Narrow => 40,
            Bandwidth::Medium => 60,
            Bandwidth::Wide => 80,
            _ => return Err(Error::InvalidPacket),
        };
        let subframes = if twenty_ms { 4 } else { 2 };
        let sample_count = subframe_samples * subframes;
        encode_frame_type(
            encoder,
            context.active,
            parameters.signal,
            parameters.quantization,
        )?;
        let gains = encode_gains(
            encoder,
            parameters.signal,
            subframes as u8,
            context.independent_gain,
            self.previous_gain,
            parameters.gain_symbols,
        )?;
        encode_lsf(
            encoder,
            bandwidth,
            parameters.signal,
            twenty_ms,
            &parameters.lsf,
        )?;
        if parameters.signal == SignalType::Voiced {
            let lag = parameters.primary_pitch.ok_or(Error::InvalidPacket)?;
            let previous = if context.absolute_pitch {
                None
            } else {
                self.previous_pitch
            };
            encode_primary_lag(encoder, bandwidth, previous, lag, context.absolute_pitch)?;
            encode_contour(encoder, bandwidth, twenty_ms, lag, parameters.contour_index)?;
            encode_ltp_filters(
                encoder,
                parameters.ltp.as_ref().ok_or(Error::InvalidPacket)?,
                context.ltp_scale_present,
            )?;
            self.previous_pitch = Some(lag);
        } else if parameters.primary_pitch.is_some() || parameters.ltp.is_some() {
            return Err(Error::InvalidPacket);
        } else {
            self.previous_pitch = None;
        }
        encode_seed(encoder, parameters.seed)?;
        crate::silk_entropy::encode_excitation_blocks(
            encoder,
            parameters.signal,
            parameters.quantization,
            parameters.rate_level,
            &parameters.excitation[..sample_count],
        )?;
        self.previous_gain = Some(gains.log[subframes - 1]);
        Ok(sample_count)
    }
}
impl Default for MonoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MonoDecoder {
    synthesis: Synthesis,
    previous_gain: Option<u8>,
    previous_lsf_q15: [i16; 16],
    has_previous_lsf: bool,
    previous_pitch: Option<i16>,
    last_lpc_q12: [i16; 16],
    lpc_order: u8,
}

impl MonoDecoder {
    pub const fn new() -> Self {
        Self {
            synthesis: Synthesis::new(),
            previous_gain: None,
            previous_lsf_q15: [0; 16],
            has_previous_lsf: false,
            previous_pitch: None,
            last_lpc_q12: [0; 16],
            lpc_order: 0,
        }
    }
    pub fn reset(&mut self) {
        self.synthesis.reset();
        self.previous_gain = None;
        self.has_previous_lsf = false;
        self.previous_pitch = None;
        self.previous_lsf_q15.fill(0);
        self.last_lpc_q12.fill(0);
        self.lpc_order = 0;
    }
    pub const fn has_previous_pitch(&self) -> bool {
        self.previous_pitch.is_some()
    }
    pub fn conceal(&mut self, sample_count: usize, output: &mut [f32]) -> Result<(), Error> {
        if output.len() < sample_count {
            return Err(Error::BufferTooSmall);
        }
        if !matches!(self.lpc_order, 10 | 16) {
            output[..sample_count].fill(0.0);
            return Ok(());
        }
        self.synthesis.conceal(
            &self.last_lpc_q12[..usize::from(self.lpc_order)],
            &mut output[..sample_count],
        )
    }

    pub fn decode_frame(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        context: FrameContext,
        output: &mut [f32],
    ) -> Result<usize, Error> {
        let subframe_samples = match bandwidth {
            Bandwidth::Narrow => 40,
            Bandwidth::Medium => 60,
            Bandwidth::Wide => 80,
            _ => return Err(Error::InvalidPacket),
        };
        let subframes = if twenty_ms { 4 } else { 2 };
        let sample_count = subframe_samples * subframes;
        if output.len() < sample_count {
            return Err(Error::BufferTooSmall);
        }
        let (signal, quantization) = decode_frame_type(decoder, context.active)?;
        let gains = decode_gains(
            decoder,
            signal,
            subframes as u8,
            context.independent_gain,
            self.previous_gain,
        )?;
        let had_previous_lsf = self.has_previous_lsf;
        let previous_lsf = self
            .has_previous_lsf
            .then_some(self.previous_lsf_q15.as_slice());
        let mut current_lsf = [0i16; 16];
        let mut first_lpc = [0i16; 16];
        let mut second_lpc = [0i16; 16];
        let lsf_indices = decode_lsf(
            decoder,
            bandwidth,
            signal,
            twenty_ms,
            previous_lsf,
            &mut current_lsf,
            &mut first_lpc,
            &mut second_lpc,
        )?;
        let mut contour = None;
        let mut ltp = None;
        let mut primary = None;
        if signal == SignalType::Voiced {
            let previous = if context.absolute_pitch {
                None
            } else {
                self.previous_pitch
            };
            let lag = decode_primary_lag(decoder, bandwidth, previous)?;
            primary = Some(lag);
            contour = Some(decode_contour(decoder, bandwidth, twenty_ms, lag)?);
            ltp = Some(decode_ltp_filters(
                decoder,
                subframes as u8,
                context.ltp_scale_present,
            )?);
        }
        let seed = decode_seed(decoder)?;
        let mut raw = [0i32; MAX_EXCITATION_SAMPLES];
        decode_excitation_blocks(decoder, signal, quantization, sample_count, &mut raw)?;
        let mut excitation = [0i32; MAX_EXCITATION_SAMPLES];
        reconstruct_excitation(
            &raw[..sample_count],
            u32::from(seed),
            signal,
            quantization,
            &mut excitation,
        )?;
        let interpolated = had_previous_lsf
            && twenty_ms
            && lsf_indices
                .interpolation_q2
                .is_some_and(|factor| factor < 4);
        let order = if bandwidth == Bandwidth::Wide { 16 } else { 10 };
        for subframe in 0..subframes {
            let range = subframe * subframe_samples..(subframe + 1) * subframe_samples;
            let lpc = if twenty_ms && subframe < 2 {
                &first_lpc[..order]
            } else {
                &second_lpc[..order]
            };
            let voiced = match (contour, ltp) {
                (Some(c), Some(f)) => Some(VoicedParameters {
                    pitch_lag: c.lags[subframe],
                    coefficients_q7: f.taps_q7[subframe],
                }),
                _ => None,
            };
            let (scale_q14, recent_distance) = match ltp {
                Some(_) if interpolated && subframe >= 2 => {
                    (16_384, (subframe - 2) * subframe_samples)
                }
                Some(filters) => (filters.scale_q14, subframe * subframe_samples),
                None => (16_384, 0),
            };
            self.synthesis.subframe_rfc(
                &excitation[range.clone()],
                gains.q16[subframe],
                lpc,
                voiced,
                scale_q14,
                recent_distance,
                &mut output[range],
            )?;
        }
        self.previous_gain = Some(gains.log[subframes - 1]);
        self.previous_lsf_q15 = current_lsf;
        self.has_previous_lsf = true;
        self.previous_pitch = primary;
        self.last_lpc_q12[..order].copy_from_slice(&second_lpc[..order]);
        self.lpc_order = order as u8;
        Ok(sample_count)
    }
}
impl Default for MonoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StereoDecoder {
    mid: MonoDecoder,
    side: MonoDecoder,
    unmixer: StereoUnmixer,
}

impl StereoDecoder {
    pub const fn new() -> Self {
        Self {
            mid: MonoDecoder::new(),
            side: MonoDecoder::new(),
            unmixer: StereoUnmixer::new(),
        }
    }
    pub fn reset(&mut self) {
        self.mid.reset();
        self.side.reset();
        self.unmixer = StereoUnmixer::new();
    }
    pub const fn mid_has_previous_pitch(&self) -> bool {
        self.mid.has_previous_pitch()
    }
    pub const fn side_has_previous_pitch(&self) -> bool {
        self.side.has_previous_pitch()
    }
    pub fn reset_mid(&mut self) {
        self.mid.reset();
    }
    pub fn reset_side(&mut self) {
        self.side.reset();
    }
    pub fn conceal(
        &mut self,
        sample_rate: u32,
        sample_count: usize,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        if left.len() < sample_count || right.len() < sample_count {
            return Err(Error::BufferTooSmall);
        }
        let mut mid = [0.0f32; MAX_EXCITATION_SAMPLES];
        let mut side = [0.0f32; MAX_EXCITATION_SAMPLES];
        self.mid.conceal(sample_count, &mut mid)?;
        self.side.conceal(sample_count, &mut side)?;
        self.unmixer.unmix(
            &mid[..sample_count],
            Some(&side[..sample_count]),
            sample_rate,
            self.unmixer.current_weights(),
            left,
            right,
        )
    }
    pub fn decode_side_only(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        context: FrameContext,
        output: &mut [f32],
    ) -> Result<usize, Error> {
        self.side
            .decode_frame(decoder, bandwidth, twenty_ms, context, output)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn decode_interval(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        mid_context: FrameContext,
        side_context: FrameContext,
        mid_only_flag_present: bool,
        side_coded: bool,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(usize, bool), Error> {
        let prediction = decode_prediction(decoder)?;
        let mid_only = if mid_only_flag_present {
            decode_mid_only(decoder)?
        } else {
            false
        };
        let sample_count = match bandwidth {
            Bandwidth::Narrow => 80,
            Bandwidth::Medium => 120,
            Bandwidth::Wide => 160,
            _ => return Err(Error::InvalidPacket),
        } * if twenty_ms { 2 } else { 1 };
        if left.len() < sample_count || right.len() < sample_count {
            return Err(Error::BufferTooSmall);
        }
        let mut mid = [0f32; MAX_EXCITATION_SAMPLES];
        let mut side = [0f32; MAX_EXCITATION_SAMPLES];
        self.mid
            .decode_frame(decoder, bandwidth, twenty_ms, mid_context, &mut mid)?;
        if mid_only {
            self.side.reset();
        } else if side_coded {
            self.side
                .decode_frame(decoder, bandwidth, twenty_ms, side_context, &mut side)?;
        } else {
            self.side.conceal(sample_count, &mut side)?;
        }
        let sample_rate = match bandwidth {
            Bandwidth::Narrow => 8_000,
            Bandwidth::Medium => 12_000,
            Bandwidth::Wide => 16_000,
            _ => return Err(Error::InvalidPacket),
        };
        self.unmixer.unmix(
            &mid[..sample_count],
            (!mid_only).then_some(&side[..sample_count]),
            sample_rate,
            prediction.weights,
            left,
            right,
        )?;
        Ok((sample_count, mid_only))
    }
}
impl Default for StereoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StereoEncoder {
    mid: MonoEncoder,
    side: MonoEncoder,
}
impl StereoEncoder {
    pub const fn new() -> Self {
        Self {
            mid: MonoEncoder::new(),
            side: MonoEncoder::new(),
        }
    }
    pub fn reset(&mut self) {
        self.mid.reset();
        self.side.reset();
    }
    pub const fn mid_has_previous_pitch(&self) -> bool {
        self.mid.has_previous_pitch()
    }
    pub const fn side_has_previous_pitch(&self) -> bool {
        self.side.has_previous_pitch()
    }
    pub fn reset_mid(&mut self) {
        self.mid.reset();
    }
    pub fn reset_side(&mut self) {
        self.side.reset();
    }
    pub fn encode_side_only(
        &mut self,
        encoder: &mut RangeEncoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        context: FrameContext,
        parameters: &MonoFrameParameters,
    ) -> Result<usize, Error> {
        self.side
            .encode_frame(encoder, bandwidth, twenty_ms, context, parameters)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode_interval(
        &mut self,
        encoder: &mut RangeEncoder<'_>,
        bandwidth: Bandwidth,
        twenty_ms: bool,
        prediction: &StereoPrediction,
        mid_only_flag: Option<bool>,
        mid_context: FrameContext,
        side_context: FrameContext,
        side_coded: bool,
        mid_parameters: &MonoFrameParameters,
        side_parameters: Option<&MonoFrameParameters>,
    ) -> Result<usize, Error> {
        encode_prediction(encoder, prediction)?;
        if let Some(value) = mid_only_flag {
            encode_mid_only(encoder, value)?;
        }
        let mid_only = mid_only_flag == Some(true);
        let count =
            self.mid
                .encode_frame(encoder, bandwidth, twenty_ms, mid_context, mid_parameters)?;
        if mid_only {
            if side_parameters.is_some() {
                return Err(Error::InvalidPacket);
            }
            self.side.reset();
        } else if side_coded {
            self.side.encode_frame(
                encoder,
                bandwidth,
                twenty_ms,
                side_context,
                side_parameters.ok_or(Error::InvalidPacket)?,
            )?;
        } else if side_parameters.is_some() {
            return Err(Error::InvalidPacket);
        }
        Ok(count)
    }
}
impl Default for StereoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RangeEncoder,
        silk::QuantizationOffset,
        silk_entropy::encode_excitation_blocks,
        silk_frame::{encode_frame_type, encode_gains},
        silk_lsf::{LsfIndices, Stage2, encode_lsf},
        silk_pitch::encode_seed,
        silk_stereo::prediction_from_indices,
    };

    #[test]
    fn complete_unvoiced_frame_decodes_to_audio_and_retains_state() {
        let mut decoder_state = MonoDecoder::new();
        let mut previous_gain = None;
        for first in [true, false] {
            let mut bytes = [0u8; 1024];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_frame_type(
                &mut encoder,
                true,
                SignalType::Unvoiced,
                QuantizationOffset::Low,
            )
            .unwrap();
            let symbols = if first { [20, 4, 0, 0] } else { [4, 4, 0, 0] };
            let gains = encode_gains(
                &mut encoder,
                SignalType::Unvoiced,
                2,
                first,
                previous_gain,
                symbols,
            )
            .unwrap();
            previous_gain = Some(gains.log[1]);
            let indices = LsfIndices {
                stage1: 3,
                stage2: Stage2 {
                    order: 10,
                    index: [0; 16],
                },
                interpolation_q2: None,
            };
            encode_lsf(
                &mut encoder,
                Bandwidth::Narrow,
                SignalType::Unvoiced,
                false,
                &indices,
            )
            .unwrap();
            encode_seed(&mut encoder, 2).unwrap();
            let excitation = [0i32; 80];
            encode_excitation_blocks(
                &mut encoder,
                SignalType::Unvoiced,
                QuantizationOffset::Low,
                0,
                &excitation,
            )
            .unwrap();
            encoder.finish().unwrap();
            let mut output = [0f32; 80];
            assert_eq!(
                decoder_state.decode_frame(
                    &mut RangeDecoder::new(&bytes),
                    Bandwidth::Narrow,
                    false,
                    FrameContext {
                        active: true,
                        independent_gain: first,
                        absolute_pitch: true,
                        ltp_scale_present: false
                    },
                    &mut output
                ),
                Ok(80)
            );
            assert!(
                output
                    .iter()
                    .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
            );
            assert!(output.iter().any(|sample| *sample != 0.0));
        }
    }

    #[test]
    fn stateful_encoder_and_decoder_integrate_a_voiced_wideband_frame() {
        let parameters = MonoFrameParameters {
            signal: SignalType::Voiced,
            quantization: QuantizationOffset::Low,
            gain_symbols: [25, 4, 4, 4],
            lsf: LsfIndices {
                stage1: 4,
                stage2: Stage2 {
                    order: 16,
                    index: [0; 16],
                },
                interpolation_q2: Some(4),
            },
            primary_pitch: Some(80),
            contour_index: 0,
            ltp: Some(LtpFilters {
                periodicity: 0,
                subframes: 4,
                indices: [0; 4],
                taps_q7: [[4, 6, 24, 7, 5]; 4],
                scale_q14: 15_565,
            }),
            seed: 1,
            rate_level: 2,
            excitation: [0; MAX_EXCITATION_SAMPLES],
        };
        let context = FrameContext {
            active: true,
            independent_gain: true,
            absolute_pitch: true,
            ltp_scale_present: true,
        };
        let mut bytes = [0u8; 2048];
        let mut range = RangeEncoder::new(&mut bytes);
        assert_eq!(
            MonoEncoder::new().encode_frame(
                &mut range,
                Bandwidth::Wide,
                true,
                context,
                &parameters
            ),
            Ok(320)
        );
        range.finish().unwrap();
        let mut output = [0f32; 320];
        assert_eq!(
            MonoDecoder::new().decode_frame(
                &mut RangeDecoder::new(&bytes),
                Bandwidth::Wide,
                true,
                context,
                &mut output
            ),
            Ok(320)
        );
        assert!(
            output
                .iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        );
    }

    #[test]
    fn stereo_wrapper_handles_coded_side_and_mid_only_intervals() {
        let parameters = MonoFrameParameters {
            signal: SignalType::Unvoiced,
            quantization: QuantizationOffset::Low,
            gain_symbols: [20, 4, 0, 0],
            lsf: LsfIndices {
                stage1: 3,
                stage2: Stage2 {
                    order: 10,
                    index: [0; 16],
                },
                interpolation_q2: None,
            },
            primary_pitch: None,
            contour_index: 0,
            ltp: None,
            seed: 0,
            rate_level: 0,
            excitation: [0; MAX_EXCITATION_SAMPLES],
        };
        let prediction = prediction_from_indices(12, 1, 2, 1, 2).unwrap();
        let context = FrameContext {
            active: true,
            independent_gain: true,
            absolute_pitch: true,
            ltp_scale_present: false,
        };
        for mid_only in [false, true] {
            let mut bytes = [0u8; 2048];
            let mut range = RangeEncoder::new(&mut bytes);
            StereoEncoder::new()
                .encode_interval(
                    &mut range,
                    Bandwidth::Narrow,
                    false,
                    &prediction,
                    Some(mid_only),
                    context,
                    context,
                    !mid_only,
                    &parameters,
                    (!mid_only).then_some(&parameters),
                )
                .unwrap();
            range.finish().unwrap();
            let mut left = [0f32; 80];
            let mut right = [0f32; 80];
            assert_eq!(
                StereoDecoder::new().decode_interval(
                    &mut RangeDecoder::new(&bytes),
                    Bandwidth::Narrow,
                    false,
                    context,
                    context,
                    true,
                    !mid_only,
                    &mut left,
                    &mut right
                ),
                Ok((80, mid_only))
            );
            assert!(
                left.iter()
                    .chain(&right)
                    .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
            );
        }
    }
}
