//! Allocation-free CELT transform and synthesis primitives.

use crate::{Error, RangeDecoder, RangeEncoder};

const PI: f32 = core::f32::consts::PI;
const PREEMPHASIS: f32 = 0.850_006_1;
const POSTFILTER_ICDF: [u8; 3] = [2, 1, 0];
const SPREAD_PDF: [u8; 4] = [7, 2, 21, 2];
const PLC_HISTORY: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStart {
    pub silence: bool,
    pub post_filter: Option<PostFilterParameters>,
    pub transient: bool,
    pub intra_energy: bool,
}

impl FrameStart {
    pub fn decode(decoder: &mut RangeDecoder<'_>, lm: u8) -> Result<Self, Error> {
        if lm > 3 {
            return Err(Error::InvalidFrameSize);
        }
        let silence = decoder.decode_bit_logp(15)?;
        let post_filter = PostFilterParameters::decode(decoder)?;
        let transient = lm != 0 && decoder.decode_bit_logp(3)?;
        let intra_energy = decoder.decode_bit_logp(3)?;
        Ok(Self {
            silence,
            post_filter,
            transient,
            intra_energy,
        })
    }

    pub fn encode(&self, encoder: &mut RangeEncoder<'_>, lm: u8) -> Result<(), Error> {
        if lm > 3 || (lm == 0 && self.transient) {
            return Err(Error::InvalidPacket);
        }
        encoder.encode_bit_logp(self.silence, 15)?;
        PostFilterParameters::encode(self.post_filter, encoder)?;
        if lm != 0 {
            encoder.encode_bit_logp(self.transient, 3)?;
        }
        encoder.encode_bit_logp(self.intra_energy, 3)?;
        Ok(())
    }
}

pub fn decode_spread(decoder: &mut RangeDecoder<'_>) -> Result<u8, Error> {
    Ok(decoder.decode_pdf(&SPREAD_PDF)? as u8)
}

pub fn encode_spread(encoder: &mut RangeEncoder<'_>, spread: u8) -> Result<(), Error> {
    if spread > 3 {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(spread), &SPREAD_PDF)
}

/// Computes one sample of the RFC 6716 low-overlap base window.
pub fn window_sample(index: usize, overlap: usize) -> Result<f32, Error> {
    if overlap == 0 || index >= overlap {
        return Err(Error::InvalidPacket);
    }
    let phase = PI * (index as f32 + 0.5) / (2.0 * overlap as f32);
    let inner = mrml_math::sin(phase);
    Ok(mrml_math::sin(0.5 * PI * inner * inner))
}

/// Fills a caller-owned CELT overlap window.
pub fn make_window(output: &mut [f32]) -> Result<(), Error> {
    let overlap = output.len();
    if overlap == 0 {
        return Err(Error::BufferTooSmall);
    }
    for (index, value) in output.iter_mut().enumerate() {
        *value = window_sample(index, overlap)?;
    }
    Ok(())
}

/// Direct forward MDCT. Input length must be twice the coefficient count.
///
/// This deliberately simple O(N²) implementation establishes the normative
/// transform and is suitable as the conformance baseline for a later FFT path.
pub fn forward_mdct(input: &[f32], coefficients: &mut [f32]) -> Result<(), Error> {
    let n = coefficients.len();
    if n == 0 || input.len() != n.checked_mul(2).ok_or(Error::InvalidFrameSize)? {
        return Err(Error::InvalidFrameSize);
    }
    let scale = PI / n as f32;
    for (k, coefficient) in coefficients.iter_mut().enumerate() {
        let frequency = k as f32 + 0.5;
        let mut sum = 0.0f32;
        for (sample, &value) in input.iter().enumerate() {
            let phase = scale * (sample as f32 + 0.5 + n as f32 * 0.5) * frequency;
            sum += value * mrml_math::cos(phase);
        }
        *coefficient = sum;
    }
    Ok(())
}

/// Direct inverse MDCT producing twice as many time-domain samples.
pub fn inverse_mdct(coefficients: &[f32], output: &mut [f32]) -> Result<(), Error> {
    let n = coefficients.len();
    if n == 0 || output.len() != n.checked_mul(2).ok_or(Error::InvalidFrameSize)? {
        return Err(Error::InvalidFrameSize);
    }
    let phase_scale = PI / n as f32;
    let amplitude_scale = 1.0 / n as f32;
    for (sample, value) in output.iter_mut().enumerate() {
        let time = sample as f32 + 0.5 + n as f32 * 0.5;
        let mut sum = 0.0f32;
        for (k, &coefficient) in coefficients.iter().enumerate() {
            sum += coefficient * mrml_math::cos(phase_scale * time * (k as f32 + 0.5));
        }
        *value = sum * amplitude_scale;
    }
    Ok(())
}

/// Direct short-block CELT analysis. Each 2.5 ms block is transformed
/// independently and written in frequency-major interleaved order.
pub fn forward_short_blocks(samples: &[f32], coefficients: &mut [f32]) -> Result<(), Error> {
    if samples.len() != coefficients.len()
        || samples.is_empty()
        || !samples.len().is_multiple_of(120)
        || samples.len() > 960
    {
        return Err(Error::InvalidFrameSize);
    }
    let blocks = samples.len() / 120;
    let mut input = [0.0f32; 240];
    let mut block_coefficients = [0.0f32; 120];
    for block in 0..blocks {
        let source = &samples[block * 120..(block + 1) * 120];
        input[..120].copy_from_slice(source);
        input[120..].copy_from_slice(source);
        forward_mdct(&input, &mut block_coefficients)?;
        for frequency in 0..120 {
            coefficients[frequency * blocks + block] = block_coefficients[frequency];
        }
    }
    Ok(())
}

/// Applies CELT's weighted overlap-add to the two overlap regions.
pub fn overlap_add(
    previous: &[f32],
    current: &[f32],
    window: &[f32],
    output: &mut [f32],
) -> Result<(), Error> {
    let n = window.len();
    if n == 0 || previous.len() != n || current.len() != n || output.len() != n {
        return Err(Error::InvalidFrameSize);
    }
    for index in 0..n {
        let left = window[index];
        let right = window[n - 1 - index];
        output[index] = previous[index] * right + current[index] * left;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Deemphasis {
    memory: f32,
}
impl Deemphasis {
    pub const fn new() -> Self {
        Self { memory: 0.0 }
    }
    pub fn apply(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let value = *sample + PREEMPHASIS * self.memory;
            self.memory = value;
            *sample = value;
        }
    }
    pub const fn memory(&self) -> f32 {
        self.memory
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostFilterParameters {
    pub period: u16,
    pub gain: f32,
    pub tapset: u8,
}

impl PostFilterParameters {
    pub fn decode(decoder: &mut RangeDecoder<'_>) -> Result<Option<Self>, Error> {
        if !decoder.decode_bit_logp(1)? {
            return Ok(None);
        }
        let octave = decoder.decode_uint(6)? as u8;
        let fine = decoder.raw_bits(4 + octave)?;
        let period =
            u16::try_from((16u32 << octave) + fine - 1).map_err(|_| Error::InvalidPacket)?;
        if !(15..=1022).contains(&period) {
            return Err(Error::InvalidPacket);
        }
        let gain = 3.0 * (decoder.raw_bits(3)? + 1) as f32 / 32.0;
        let tapset = decoder.decode_icdf(&POSTFILTER_ICDF, 2)? as u8;
        Ok(Some(Self {
            period,
            gain,
            tapset,
        }))
    }

    pub fn encode(parameters: Option<Self>, encoder: &mut RangeEncoder<'_>) -> Result<(), Error> {
        encoder.encode_bit_logp(parameters.is_some(), 1)?;
        let Some(parameters) = parameters else {
            return Ok(());
        };
        if !(15..=1022).contains(&parameters.period)
            || parameters.tapset > 2
            || !(3.0 / 32.0..=24.0 / 32.0).contains(&parameters.gain)
        {
            return Err(Error::InvalidPacket);
        }
        let base = u32::from(parameters.period) + 1;
        let octave = (u32::BITS - base.leading_zeros() - 5) as u8;
        if octave > 5 {
            return Err(Error::InvalidPacket);
        }
        let fine = base - (16u32 << octave);
        encoder.encode_uint(u32::from(octave), 6)?;
        encoder.raw_bits(fine, 4 + octave)?;
        let quantized_gain = ((parameters.gain * 32.0 / 3.0) as u32).clamp(1, 8) - 1;
        encoder.raw_bits(quantized_gain, 3)?;
        encoder.encode_icdf(usize::from(parameters.tapset), &POSTFILTER_ICDF, 2)
    }
}

pub struct PostFilter {
    history: [f32; 1024],
    position: usize,
    current_gain: f32,
    parameters: Option<PostFilterParameters>,
}

impl PostFilter {
    pub const fn new() -> Self {
        Self {
            history: [0.0; 1024],
            position: 0,
            current_gain: 0.0,
            parameters: None,
        }
    }
    pub fn set(&mut self, parameters: Option<PostFilterParameters>) {
        self.parameters = parameters;
    }
    pub fn apply(&mut self, samples: &mut [f32]) {
        let target = self.parameters.map_or(0.0, |parameters| parameters.gain);
        let increment = if samples.is_empty() {
            0.0
        } else {
            (target - self.current_gain) / samples.len() as f32
        };
        for sample in samples {
            self.current_gain += increment;
            if let Some(parameters) = self.parameters {
                let taps = match parameters.tapset {
                    0 => [0.306_640_63, 0.217_041_02, 0.129_638_67],
                    1 => [0.463_867_2, 0.268_066_4, 0.0],
                    _ => [0.799_804_7, 0.100_097_656, 0.0],
                };
                let period = usize::from(parameters.period);
                let delayed = self.past(period);
                let adjacent = self.past(period - 1) + self.past(period + 1);
                let outer = self.past(period - 2) + self.past(period + 2);
                *sample +=
                    self.current_gain * (taps[0] * delayed + taps[1] * adjacent + taps[2] * outer);
            }
            self.history[self.position] = *sample;
            self.position = (self.position + 1) & 1023;
        }
        self.current_gain = target;
    }
    fn past(&self, distance: usize) -> f32 {
        self.history[(self.position + 1024 - distance) & 1023]
    }
}
impl Default for PostFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation-free pitch repetition PLC for a single decoded channel.
pub struct CeltPlc {
    history: [f32; PLC_HISTORY],
    position: usize,
    filled: usize,
    last_period: usize,
}

impl CeltPlc {
    pub const fn new() -> Self {
        Self {
            history: [0.0; PLC_HISTORY],
            position: 0,
            filled: 0,
            last_period: 120,
        }
    }
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.history[self.position] = sample;
            self.position = (self.position + 1) % PLC_HISTORY;
            self.filled = (self.filled + 1).min(PLC_HISTORY);
        }
    }
    pub fn conceal(&mut self, output: &mut [f32]) {
        if self.filled < 32 {
            output.fill(0.0);
            self.push(output);
            return;
        }
        let period = self.find_period().min(self.filled);
        self.last_period = period;
        let output_len = output.len().max(1) as f32;
        for (index, sample) in output.iter_mut().enumerate() {
            let source = (self.position + PLC_HISTORY - period) % PLC_HISTORY;
            let attenuation = 1.0 - 0.2 * index as f32 / output_len;
            *sample = self.history[source] * attenuation;
            self.history[self.position] = *sample;
            self.position = (self.position + 1) % PLC_HISTORY;
            self.filled = (self.filled + 1).min(PLC_HISTORY);
        }
    }
    pub const fn period(&self) -> usize {
        self.last_period
    }
    fn find_period(&self) -> usize {
        let window = self.filled.min(480);
        let max_lag = self.filled.saturating_sub(window).min(1022);
        if max_lag < 15 {
            return self.filled.min(120);
        }
        let mut best_lag = 15;
        let mut best_score = f64::MIN;
        for lag in 15..=max_lag {
            let mut cross = 0.0f64;
            let mut delayed_energy = 1e-12f64;
            for offset in 0..window {
                let recent =
                    self.history[(self.position + PLC_HISTORY - 1 - offset) % PLC_HISTORY] as f64;
                let delayed = self.history
                    [(self.position + PLC_HISTORY - 1 - offset - lag) % PLC_HISTORY]
                    as f64;
                cross += recent * delayed;
                delayed_energy += delayed * delayed;
            }
            let score = cross * cross / delayed_energy;
            if cross > 0.0 && score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }
        best_lag
    }
}
impl Default for CeltPlc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_start_and_spread_round_trip_in_normative_order() {
        for lm in 0..=3 {
            for transient in [false, true] {
                if lm == 0 && transient {
                    continue;
                }
                for spread in 0..=3 {
                    let expected = FrameStart {
                        silence: spread == 0,
                        post_filter: (spread == 2).then_some(PostFilterParameters {
                            period: 80,
                            gain: 6.0 / 32.0,
                            tapset: 1,
                        }),
                        transient,
                        intra_energy: spread & 1 != 0,
                    };
                    let mut bytes = [0; 16];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    expected.encode(&mut encoder, lm).unwrap();
                    encode_spread(&mut encoder, spread).unwrap();
                    encoder.finish().unwrap();
                    let mut decoder = RangeDecoder::new(&bytes);
                    assert_eq!(FrameStart::decode(&mut decoder, lm), Ok(expected));
                    assert_eq!(decode_spread(&mut decoder), Ok(spread));
                }
            }
        }
        let invalid = FrameStart {
            silence: false,
            post_filter: None,
            transient: true,
            intra_energy: false,
        };
        assert_eq!(
            invalid.encode(&mut RangeEncoder::new(&mut [0; 8]), 0),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn window_is_power_complementary() {
        let mut window = [0.0f32; 120];
        make_window(&mut window).unwrap();
        for index in 0..window.len() {
            let sum = window[index] * window[index]
                + window[window.len() - 1 - index] * window[window.len() - 1 - index];
            assert!((sum - 1.0).abs() < 2e-6);
        }
    }

    #[test]
    fn mdct_zero_and_impulse_are_deterministic() {
        let mut coefficients = [7.0f32; 120];
        forward_mdct(&[0.0; 240], &mut coefficients).unwrap();
        assert_eq!(coefficients, [0.0; 120]);
        let mut impulse = [0.0f32; 240];
        impulse[0] = 1.0;
        forward_mdct(&impulse, &mut coefficients).unwrap();
        let mut output = [0.0f32; 240];
        inverse_mdct(&coefficients, &mut output).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| value.abs() > 0.01));
    }
    #[test]
    fn short_mdct_interleaves_all_blocks() {
        let mut samples = [0.0f32; 480];
        samples[240] = 1.0;
        let mut coefficients = [0.0f32; 480];
        forward_short_blocks(&samples, &mut coefficients).unwrap();
        assert!(coefficients.iter().all(|value| value.is_finite()));
        assert!(
            coefficients
                .iter()
                .skip(2)
                .step_by(4)
                .any(|&value| value != 0.0)
        );
        assert!(coefficients.iter().step_by(4).all(|&value| value == 0.0));
    }

    #[test]
    fn overlap_and_deemphasis_preserve_state() {
        let mut output = [0.0; 4];
        overlap_add(&[1.0; 4], &[1.0; 4], &[0.0, 0.25, 0.75, 1.0], &mut output).unwrap();
        assert_eq!(output, [1.0, 1.0, 1.0, 1.0]);
        let mut deemphasis = Deemphasis::new();
        let mut first = [1.0, 0.0];
        deemphasis.apply(&mut first);
        let memory = deemphasis.memory();
        let mut second = [0.0];
        deemphasis.apply(&mut second);
        assert_eq!(second[0], PREEMPHASIS * memory);
    }

    #[test]
    fn postfilter_parameters_round_trip_through_entropy_coder() {
        let expected = PostFilterParameters {
            period: 300,
            gain: 9.0 / 32.0,
            tapset: 2,
        };
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        PostFilterParameters::encode(Some(expected), &mut encoder).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let decoded = PostFilterParameters::decode(&mut decoder).unwrap().unwrap();
        assert_eq!(decoded.period, expected.period);
        assert_eq!(decoded.tapset, expected.tapset);
        assert!((decoded.gain - expected.gain).abs() < f32::EPSILON);
    }

    #[test]
    fn postfilter_is_stateful_and_bounded() {
        let mut filter = PostFilter::new();
        filter.set(Some(PostFilterParameters {
            period: 15,
            gain: 3.0 / 32.0,
            tapset: 0,
        }));
        let mut samples = [0.0f32; 64];
        samples[0] = 1.0;
        filter.apply(&mut samples);
        assert!(samples.iter().all(|sample| sample.is_finite()));
        assert!(samples[15].abs() > 0.0);
    }

    #[test]
    fn plc_detects_and_repeats_periodic_history() {
        let mut plc = CeltPlc::new();
        let mut history = [0.0f32; 1200];
        for (index, sample) in history.iter_mut().enumerate() {
            *sample = mrml_math::sin(2.0 * PI * index as f32 / 80.0);
        }
        plc.push(&history);
        let mut concealed = [0.0f32; 160];
        plc.conceal(&mut concealed);
        assert!(plc.period() % 80 <= 1 || 80 - plc.period() % 80 <= 1);
        assert!(concealed.iter().any(|sample| sample.abs() > 0.5));
        assert!(concealed.iter().all(|sample| sample.is_finite()));
    }
}
