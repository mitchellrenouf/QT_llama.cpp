//! Allocation-free mono SILK payload ordering, including in-band LBRR/FEC.

use crate::{
    Bandwidth, Error, RangeDecoder, RangeEncoder,
    silk_codec::{
        FrameContext, MonoDecoder, MonoEncoder, MonoFrameParameters, StereoDecoder, StereoEncoder,
    },
    silk_frame::{LayerHeader, decode_layer_header, encode_layer_header},
    silk_stereo::StereoPrediction,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadResult {
    pub regular_samples: usize,
    pub fec_samples: usize,
    pub fec_mask: u8,
}

fn geometry(bandwidth: Bandwidth, duration_ms: u8) -> Result<(usize, u8, bool), Error> {
    let per_20 = match bandwidth {
        Bandwidth::Narrow => 160,
        Bandwidth::Medium => 240,
        Bandwidth::Wide => 320,
        _ => return Err(Error::InvalidPacket),
    };
    match duration_ms {
        10 => Ok((per_20 / 2, 1, false)),
        20 => Ok((per_20, 1, true)),
        40 => Ok((per_20, 2, true)),
        60 => Ok((per_20, 3, true)),
        _ => Err(Error::InvalidFrameSize),
    }
}

pub struct MonoPayloadDecoder {
    regular: MonoDecoder,
    fec: MonoDecoder,
}
impl MonoPayloadDecoder {
    pub const fn new() -> Self {
        Self {
            regular: MonoDecoder::new(),
            fec: MonoDecoder::new(),
        }
    }
    pub fn reset(&mut self) {
        self.regular.reset();
        self.fec.reset();
    }
    pub fn conceal(&mut self, sample_count: usize, output: &mut [f32]) -> Result<(), Error> {
        self.regular.conceal(sample_count, output)
    }
    pub fn decode(
        &mut self,
        data: &[u8],
        bandwidth: Bandwidth,
        duration_ms: u8,
        regular_output: &mut [f32],
        fec_output: &mut [f32],
    ) -> Result<PayloadResult, Error> {
        self.decode_range(
            &mut RangeDecoder::new(data),
            bandwidth,
            duration_ms,
            regular_output,
            fec_output,
        )
    }
    pub fn decode_range(
        &mut self,
        range: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        duration_ms: u8,
        regular_output: &mut [f32],
        fec_output: &mut [f32],
    ) -> Result<PayloadResult, Error> {
        let (interval_samples, frames, twenty_ms) = geometry(bandwidth, duration_ms)?;
        let total = interval_samples * usize::from(frames);
        if regular_output.len() < total || fec_output.len() < total {
            return Err(Error::BufferTooSmall);
        }
        regular_output[..total].fill(0.0);
        fec_output[..total].fill(0.0);
        let header = decode_layer_header(range, duration_ms, 1)?;
        let channel = header.channel[0];
        self.fec.reset();
        let mut previous_coded = false;
        let mut fec_samples = 0;
        for frame in 0..frames {
            if channel.lbrr & (1 << frame) != 0 {
                let context = FrameContext {
                    active: true,
                    independent_gain: !previous_coded,
                    absolute_pitch: !previous_coded || !self.fec.has_previous_pitch(),
                    ltp_scale_present: !previous_coded,
                };
                let start = usize::from(frame) * interval_samples;
                self.fec.decode_frame(
                    range,
                    bandwidth,
                    twenty_ms,
                    context,
                    &mut fec_output[start..start + interval_samples],
                )?;
                fec_samples += interval_samples;
                previous_coded = true;
            } else {
                previous_coded = false;
                self.fec.reset();
            }
        }
        for frame in 0..frames {
            let context = FrameContext {
                active: channel.vad & (1 << frame) != 0,
                independent_gain: frame == 0,
                absolute_pitch: frame == 0 || !self.regular.has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            let start = usize::from(frame) * interval_samples;
            self.regular.decode_frame(
                range,
                bandwidth,
                twenty_ms,
                context,
                &mut regular_output[start..start + interval_samples],
            )?;
        }
        Ok(PayloadResult {
            regular_samples: total,
            fec_samples,
            fec_mask: channel.lbrr,
        })
    }
}
impl Default for MonoPayloadDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MonoPayloadEncoder {
    regular: MonoEncoder,
    fec: MonoEncoder,
}
impl MonoPayloadEncoder {
    pub const fn new() -> Self {
        Self {
            regular: MonoEncoder::new(),
            fec: MonoEncoder::new(),
        }
    }
    pub fn reset(&mut self) {
        self.regular.reset();
        self.fec.reset();
    }
    pub fn encode(
        &mut self,
        output: &mut [u8],
        bandwidth: Bandwidth,
        duration_ms: u8,
        header: LayerHeader,
        regular: &[MonoFrameParameters],
        fec: &[Option<&MonoFrameParameters>],
    ) -> Result<usize, Error> {
        let mut range = RangeEncoder::new(output);
        self.encode_range(&mut range, bandwidth, duration_ms, header, regular, fec)?;
        range.finish_compact()
    }
    pub fn encode_range(
        &mut self,
        range: &mut RangeEncoder<'_>,
        bandwidth: Bandwidth,
        duration_ms: u8,
        header: LayerHeader,
        regular: &[MonoFrameParameters],
        fec: &[Option<&MonoFrameParameters>],
    ) -> Result<(), Error> {
        let (_, frames, twenty_ms) = geometry(bandwidth, duration_ms)?;
        let count = usize::from(frames);
        if header.channels != 1
            || header.frames != frames
            || regular.len() != count
            || fec.len() != count
        {
            return Err(Error::InvalidPacket);
        }
        let expected = fec.iter().enumerate().fold(0u8, |mask, (index, item)| {
            mask | (u8::from(item.is_some()) << index)
        });
        if expected != header.channel[0].lbrr {
            return Err(Error::InvalidPacket);
        }
        encode_layer_header(range, duration_ms, header)?;
        self.fec.reset();
        let mut previous_coded = false;
        for parameters in fec {
            if let Some(parameters) = parameters {
                let context = FrameContext {
                    active: true,
                    independent_gain: !previous_coded,
                    absolute_pitch: !previous_coded || !self.fec.has_previous_pitch(),
                    ltp_scale_present: !previous_coded,
                };
                self.fec
                    .encode_frame(range, bandwidth, twenty_ms, context, parameters)?;
                previous_coded = true;
            } else {
                previous_coded = false;
                self.fec.reset();
            }
        }
        for (frame, parameters) in regular.iter().enumerate() {
            let context = FrameContext {
                active: header.channel[0].vad & (1 << frame) != 0,
                independent_gain: frame == 0,
                absolute_pitch: frame == 0 || !self.regular.has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            self.regular
                .encode_frame(range, bandwidth, twenty_ms, context, parameters)?;
        }
        Ok(())
    }
}
impl Default for MonoPayloadEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct StereoFrameParameters<'a> {
    pub prediction: Option<&'a StereoPrediction>,
    pub mid_only: Option<bool>,
    pub mid: Option<&'a MonoFrameParameters>,
    pub side: Option<&'a MonoFrameParameters>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StereoPayloadResult {
    pub regular_samples: usize,
    pub fec_samples: usize,
    pub mid_fec_mask: u8,
    pub side_fec_mask: u8,
}

pub struct StereoPayloadDecoder {
    regular: StereoDecoder,
    fec: StereoDecoder,
}
impl StereoPayloadDecoder {
    pub const fn new() -> Self {
        Self {
            regular: StereoDecoder::new(),
            fec: StereoDecoder::new(),
        }
    }
    pub fn reset(&mut self) {
        self.regular.reset();
        self.fec.reset();
    }
    pub fn conceal(
        &mut self,
        sample_rate: u32,
        sample_count: usize,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), Error> {
        self.regular.conceal(sample_rate, sample_count, left, right)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &mut self,
        data: &[u8],
        bandwidth: Bandwidth,
        duration_ms: u8,
        regular_left: &mut [f32],
        regular_right: &mut [f32],
        fec_left: &mut [f32],
        fec_right: &mut [f32],
    ) -> Result<StereoPayloadResult, Error> {
        self.decode_range(
            &mut RangeDecoder::new(data),
            bandwidth,
            duration_ms,
            regular_left,
            regular_right,
            fec_left,
            fec_right,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn decode_range(
        &mut self,
        range: &mut RangeDecoder<'_>,
        bandwidth: Bandwidth,
        duration_ms: u8,
        regular_left: &mut [f32],
        regular_right: &mut [f32],
        fec_left: &mut [f32],
        fec_right: &mut [f32],
    ) -> Result<StereoPayloadResult, Error> {
        let (interval_samples, frames, twenty_ms) = geometry(bandwidth, duration_ms)?;
        let total = interval_samples * usize::from(frames);
        if regular_left.len() < total
            || regular_right.len() < total
            || fec_left.len() < total
            || fec_right.len() < total
        {
            return Err(Error::BufferTooSmall);
        }
        regular_left[..total].fill(0.0);
        regular_right[..total].fill(0.0);
        fec_left[..total].fill(0.0);
        fec_right[..total].fill(0.0);
        let header = decode_layer_header(range, duration_ms, 2)?;
        let mid = header.channel[0];
        let side = header.channel[1];
        self.fec.reset();
        let mut previous_mid = false;
        let mut previous_side = false;
        let mut fec_samples = 0;
        for frame in 0..frames {
            let bit = 1 << frame;
            let has_mid = mid.lbrr & bit != 0;
            let has_side = side.lbrr & bit != 0;
            let start = usize::from(frame) * interval_samples;
            let mid_context = FrameContext {
                active: true,
                independent_gain: !previous_mid,
                absolute_pitch: !previous_mid || !self.fec.mid_has_previous_pitch(),
                ltp_scale_present: !previous_mid,
            };
            let side_context = FrameContext {
                active: true,
                independent_gain: !previous_side,
                absolute_pitch: !previous_side || !self.fec.side_has_previous_pitch(),
                ltp_scale_present: !previous_side,
            };
            if has_mid {
                self.fec.decode_interval(
                    range,
                    bandwidth,
                    twenty_ms,
                    mid_context,
                    side_context,
                    !has_side,
                    has_side,
                    &mut fec_left[start..start + interval_samples],
                    &mut fec_right[start..start + interval_samples],
                )?;
            } else if has_side {
                let mut scratch = [0.0; crate::silk_entropy::MAX_EXCITATION_SAMPLES];
                self.fec.decode_side_only(
                    range,
                    bandwidth,
                    twenty_ms,
                    side_context,
                    &mut scratch[..interval_samples],
                )?;
                self.fec.reset_mid();
            }
            if has_mid || has_side {
                fec_samples += interval_samples;
            }
            if !has_mid {
                self.fec.reset_mid();
            }
            if !has_side {
                self.fec.reset_side();
            }
            previous_mid = has_mid;
            previous_side = has_side;
        }
        let mut previous_regular_side = false;
        for frame in 0..frames {
            let bit = 1 << frame;
            let side_header_active = side.vad & bit != 0;
            let mid_context = FrameContext {
                active: mid.vad & bit != 0,
                independent_gain: frame == 0,
                absolute_pitch: frame == 0 || !self.regular.mid_has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            let side_context = FrameContext {
                active: side_header_active,
                independent_gain: !previous_regular_side,
                absolute_pitch: !previous_regular_side || !self.regular.side_has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            let start = usize::from(frame) * interval_samples;
            let (_, mid_only) = self.regular.decode_interval(
                range,
                bandwidth,
                twenty_ms,
                mid_context,
                side_context,
                !side_header_active,
                true,
                &mut regular_left[start..start + interval_samples],
                &mut regular_right[start..start + interval_samples],
            )?;
            previous_regular_side = !mid_only;
        }
        Ok(StereoPayloadResult {
            regular_samples: total,
            fec_samples,
            mid_fec_mask: mid.lbrr,
            side_fec_mask: side.lbrr,
        })
    }
}
impl Default for StereoPayloadDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StereoPayloadEncoder {
    regular: StereoEncoder,
    fec: StereoEncoder,
}
impl StereoPayloadEncoder {
    pub const fn new() -> Self {
        Self {
            regular: StereoEncoder::new(),
            fec: StereoEncoder::new(),
        }
    }
    pub fn reset(&mut self) {
        self.regular.reset();
        self.fec.reset();
    }
    pub fn encode(
        &mut self,
        output: &mut [u8],
        bandwidth: Bandwidth,
        duration_ms: u8,
        header: LayerHeader,
        regular: &[StereoFrameParameters<'_>],
        fec: &[StereoFrameParameters<'_>],
    ) -> Result<usize, Error> {
        let mut range = RangeEncoder::new(output);
        self.encode_range(&mut range, bandwidth, duration_ms, header, regular, fec)?;
        range.finish_compact()
    }
    pub fn encode_range(
        &mut self,
        range: &mut RangeEncoder<'_>,
        bandwidth: Bandwidth,
        duration_ms: u8,
        header: LayerHeader,
        regular: &[StereoFrameParameters<'_>],
        fec: &[StereoFrameParameters<'_>],
    ) -> Result<(), Error> {
        let (_, frames, twenty_ms) = geometry(bandwidth, duration_ms)?;
        let count = usize::from(frames);
        if header.channels != 2
            || header.frames != frames
            || regular.len() != count
            || fec.len() != count
        {
            return Err(Error::InvalidPacket);
        }
        let mid_mask = fec
            .iter()
            .enumerate()
            .fold(0, |m, (i, p)| m | (u8::from(p.mid.is_some()) << i));
        let side_mask = fec
            .iter()
            .enumerate()
            .fold(0, |m, (i, p)| m | (u8::from(p.side.is_some()) << i));
        if mid_mask != header.channel[0].lbrr || side_mask != header.channel[1].lbrr {
            return Err(Error::InvalidPacket);
        }
        encode_layer_header(range, duration_ms, header)?;
        self.fec.reset();
        let mut previous_mid = false;
        let mut previous_side = false;
        for p in fec {
            let has_mid = p.mid.is_some();
            let has_side = p.side.is_some();
            let mid_context = FrameContext {
                active: true,
                independent_gain: !previous_mid,
                absolute_pitch: !previous_mid || !self.fec.mid_has_previous_pitch(),
                ltp_scale_present: !previous_mid,
            };
            let side_context = FrameContext {
                active: true,
                independent_gain: !previous_side,
                absolute_pitch: !previous_side || !self.fec.side_has_previous_pitch(),
                ltp_scale_present: !previous_side,
            };
            if has_mid {
                if p.prediction.is_none()
                    || (has_side && p.mid_only.is_some())
                    || (!has_side && p.mid_only.is_none())
                {
                    return Err(Error::InvalidPacket);
                }
                self.fec.encode_interval(
                    range,
                    bandwidth,
                    twenty_ms,
                    p.prediction.unwrap(),
                    p.mid_only,
                    mid_context,
                    side_context,
                    has_side,
                    p.mid.unwrap(),
                    p.side,
                )?;
            } else if has_side {
                if p.prediction.is_some() || p.mid_only.is_some() {
                    return Err(Error::InvalidPacket);
                }
                self.fec.encode_side_only(
                    range,
                    bandwidth,
                    twenty_ms,
                    side_context,
                    p.side.unwrap(),
                )?;
            } else if p.prediction.is_some() || p.mid_only.is_some() {
                return Err(Error::InvalidPacket);
            }
            if !has_mid {
                self.fec.reset_mid();
            }
            if !has_side {
                self.fec.reset_side();
            }
            previous_mid = has_mid;
            previous_side = has_side;
        }
        let mut previous_regular_side = false;
        for (frame, p) in regular.iter().enumerate() {
            let bit = 1 << frame;
            let side_header_active = header.channel[1].vad & bit != 0;
            let mid_parameters = p.mid.ok_or(Error::InvalidPacket)?;
            let prediction = p.prediction.ok_or(Error::InvalidPacket)?;
            if side_header_active {
                if p.mid_only.is_some() || p.side.is_none() {
                    return Err(Error::InvalidPacket);
                }
            } else if p.mid_only.is_none() || (p.mid_only == Some(true)) != p.side.is_none() {
                return Err(Error::InvalidPacket);
            }
            let mid_context = FrameContext {
                active: header.channel[0].vad & bit != 0,
                independent_gain: frame == 0,
                absolute_pitch: frame == 0 || !self.regular.mid_has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            let side_context = FrameContext {
                active: side_header_active,
                independent_gain: !previous_regular_side,
                absolute_pitch: !previous_regular_side || !self.regular.side_has_previous_pitch(),
                ltp_scale_present: frame == 0,
            };
            self.regular.encode_interval(
                range,
                bandwidth,
                twenty_ms,
                prediction,
                p.mid_only,
                mid_context,
                side_context,
                true,
                mid_parameters,
                p.side,
            )?;
            previous_regular_side = p.mid_only != Some(true);
        }
        Ok(())
    }
}
impl Default for StereoPayloadEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        silk::{QuantizationOffset, SignalType},
        silk_entropy::MAX_EXCITATION_SAMPLES,
        silk_frame::ChannelHeader,
        silk_lsf::{LsfIndices, Stage2},
        silk_stereo::prediction_from_indices,
    };
    fn parameters(twenty: bool) -> MonoFrameParameters {
        MonoFrameParameters {
            signal: SignalType::Unvoiced,
            quantization: QuantizationOffset::Low,
            gain_symbols: [20, 4, 4, 4],
            lsf: LsfIndices {
                stage1: 3,
                stage2: Stage2 {
                    order: 10,
                    index: [0; 16],
                },
                interpolation_q2: twenty.then_some(4),
            },
            primary_pitch: None,
            contour_index: 0,
            ltp: None,
            seed: 1,
            rate_level: 0,
            excitation: [0; MAX_EXCITATION_SAMPLES],
        }
    }
    #[test]
    fn every_duration_orders_regular_and_fec_frames() {
        for (duration, frames, mask) in [(10, 1, 1u8), (20, 1, 1), (40, 2, 3), (60, 3, 5)] {
            let twenty = duration != 10;
            let p = parameters(twenty);
            let regular = [p; 3];
            let fec_values = [
                Some(&p),
                if mask & 2 != 0 { Some(&p) } else { None },
                if mask & 4 != 0 { Some(&p) } else { None },
            ];
            let header = LayerHeader {
                channels: 1,
                frames,
                channel: [
                    ChannelHeader {
                        vad: (1 << frames) - 1,
                        lbrr: mask,
                    },
                    ChannelHeader { vad: 0, lbrr: 0 },
                ],
            };
            let mut bytes = [0u8; 8192];
            let size = MonoPayloadEncoder::new()
                .encode(
                    &mut bytes,
                    Bandwidth::Narrow,
                    duration,
                    header,
                    &regular[..usize::from(frames)],
                    &fec_values[..usize::from(frames)],
                )
                .unwrap();
            assert!(size > 0);
            let total = if duration == 10 {
                80
            } else {
                160 * usize::from(frames)
            };
            let mut decoded = [0f32; 480];
            let mut fec = [0f32; 480];
            assert_eq!(
                MonoPayloadDecoder::new().decode(
                    &bytes[..size],
                    Bandwidth::Narrow,
                    duration,
                    &mut decoded,
                    &mut fec
                ),
                Ok(PayloadResult {
                    regular_samples: total,
                    fec_samples: (if duration == 10 { 80 } else { 160 })
                        * mask.count_ones() as usize,
                    fec_mask: mask
                })
            );
        }
    }

    #[test]
    fn stereo_payload_orders_asymmetric_fec_and_mid_only_frames() {
        let p = parameters(true);
        let mut delta = p;
        delta.gain_symbols[0] = 4;
        let prediction = prediction_from_indices(12, 1, 2, 1, 2).unwrap();
        let regular = [
            StereoFrameParameters {
                prediction: Some(&prediction),
                mid_only: None,
                mid: Some(&p),
                side: Some(&p),
            },
            StereoFrameParameters {
                prediction: Some(&prediction),
                mid_only: Some(true),
                mid: Some(&delta),
                side: None,
            },
            StereoFrameParameters {
                prediction: Some(&prediction),
                mid_only: None,
                mid: Some(&delta),
                side: Some(&p),
            },
        ];
        let fec = [
            StereoFrameParameters {
                prediction: Some(&prediction),
                mid_only: Some(true),
                mid: Some(&p),
                side: None,
            },
            StereoFrameParameters {
                prediction: None,
                mid_only: None,
                mid: None,
                side: Some(&p),
            },
            StereoFrameParameters {
                prediction: Some(&prediction),
                mid_only: None,
                mid: Some(&p),
                side: Some(&delta),
            },
        ];
        let header = LayerHeader {
            channels: 2,
            frames: 3,
            channel: [
                ChannelHeader { vad: 7, lbrr: 5 },
                ChannelHeader { vad: 5, lbrr: 6 },
            ],
        };
        let mut bytes = [0u8; 16_384];
        let size = StereoPayloadEncoder::new()
            .encode(&mut bytes, Bandwidth::Narrow, 60, header, &regular, &fec)
            .unwrap();
        let mut left = [0.0; 480];
        let mut right = [0.0; 480];
        let mut fec_left = [0.0; 480];
        let mut fec_right = [0.0; 480];
        assert_eq!(
            StereoPayloadDecoder::new().decode(
                &bytes[..size],
                Bandwidth::Narrow,
                60,
                &mut left,
                &mut right,
                &mut fec_left,
                &mut fec_right,
            ),
            Ok(StereoPayloadResult {
                regular_samples: 480,
                fec_samples: 480,
                mid_fec_mask: 5,
                side_fec_mask: 6,
            })
        );
        assert!(
            left.iter()
                .chain(right.iter())
                .all(|sample| sample.is_finite())
        );
        assert!(
            fec_left
                .iter()
                .chain(fec_right.iter())
                .all(|sample| sample.is_finite())
        );
    }
}
