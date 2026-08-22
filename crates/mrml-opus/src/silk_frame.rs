//! SILK layer framing and frame-type entropy syntax.

use crate::{
    Error, RangeDecoder, RangeEncoder,
    silk::{QuantizationOffset, SignalType},
};

const LBRR_40: [u8; 4] = [0, 53, 53, 150];
const LBRR_60: [u8; 8] = [0, 41, 20, 29, 41, 15, 28, 82];
const FRAME_INACTIVE: [u8; 6] = [26, 230, 0, 0, 0, 0];
const FRAME_ACTIVE: [u8; 6] = [0, 0, 24, 74, 148, 10];
const GAIN_MSB: [[u8; 8]; 3] = [
    [32, 112, 68, 29, 12, 1, 1, 1],
    [2, 17, 45, 60, 62, 47, 19, 4],
    [1, 3, 26, 71, 94, 50, 9, 2],
];
const GAIN_LSB: [u8; 8] = [32; 8];
const GAIN_DELTA: [u8; 41] = [
    6, 5, 11, 31, 132, 21, 8, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gains {
    pub subframes: u8,
    pub log: [u8; 4],
    pub q16: [u32; 4],
}

fn signal_index(signal: SignalType) -> usize {
    match signal {
        SignalType::Inactive => 0,
        SignalType::Unvoiced => 1,
        SignalType::Voiced => 2,
    }
}

/// Normative fixed-point `silk_log2lin()` specialization used by gain decoding.
pub fn gain_index_to_q16(log_gain: u8) -> Result<u32, Error> {
    if log_gain > 63 {
        return Err(Error::InvalidPacket);
    }
    let in_log_q7 = ((0x1D1C71u64 * u64::from(log_gain)) >> 16) + 2090;
    let integer = u32::try_from(in_log_q7 >> 7).map_err(|_| Error::InvalidPacket)?;
    let fraction = in_log_q7 & 127;
    let base = 1u64.checked_shl(integer).ok_or(Error::InvalidPacket)?;
    let curve = ((-174i64 * fraction as i64 * (128 - fraction) as i64) >> 16) + fraction as i64;
    let result = i128::from(base) + i128::from(curve) * i128::from(base >> 7);
    u32::try_from(result).map_err(|_| Error::InvalidPacket)
}

fn apply_delta(previous: u8, delta: u8) -> Result<u8, Error> {
    if delta > 40 {
        return Err(Error::InvalidPacket);
    }
    let floor = i16::from(delta) * 2 - 16;
    let relative = i16::from(previous) + i16::from(delta) - 4;
    Ok(floor.max(relative).clamp(0, 63) as u8)
}

/// Decodes two (10 ms) or four (20 ms) SILK subframe gains.
pub fn decode_gains(
    decoder: &mut RangeDecoder<'_>,
    signal: SignalType,
    subframes: u8,
    independent_first: bool,
    previous: Option<u8>,
) -> Result<Gains, Error> {
    if !matches!(subframes, 2 | 4) || (!independent_first && previous.is_none()) {
        return Err(Error::InvalidPacket);
    }
    let mut result = Gains {
        subframes,
        log: [0; 4],
        q16: [0; 4],
    };
    let mut prior = previous;
    for index in 0..usize::from(subframes) {
        let log = if index == 0 && independent_first {
            let msb = decoder.decode_pdf(&GAIN_MSB[signal_index(signal)])?;
            let lsb = decoder.decode_pdf(&GAIN_LSB)?;
            let coded = u8::try_from(msb * 8 + lsb).map_err(|_| Error::InvalidPacket)?;
            prior.map_or(coded, |old| coded.max(old.saturating_sub(16)))
        } else {
            let delta =
                u8::try_from(decoder.decode_pdf(&GAIN_DELTA)?).map_err(|_| Error::InvalidPacket)?;
            apply_delta(prior.ok_or(Error::InvalidPacket)?, delta)?
        };
        result.log[index] = log;
        result.q16[index] = gain_index_to_q16(log)?;
        prior = Some(log);
    }
    Ok(result)
}

/// Encodes gain symbols and returns their normative reconstructed values.
/// `symbols[0]` is an absolute 0..63 index when independent, otherwise all
/// symbols are delta indices in 0..40.
pub fn encode_gains(
    encoder: &mut RangeEncoder<'_>,
    signal: SignalType,
    subframes: u8,
    independent_first: bool,
    previous: Option<u8>,
    symbols: [u8; 4],
) -> Result<Gains, Error> {
    if !matches!(subframes, 2 | 4) || (!independent_first && previous.is_none()) {
        return Err(Error::InvalidPacket);
    }
    let mut result = Gains {
        subframes,
        log: [0; 4],
        q16: [0; 4],
    };
    let mut prior = previous;
    for (index, &symbol) in symbols[..usize::from(subframes)].iter().enumerate() {
        let log = if index == 0 && independent_first {
            let coded = symbol;
            if coded > 63 {
                return Err(Error::InvalidPacket);
            }
            encoder.encode_pdf(usize::from(coded >> 3), &GAIN_MSB[signal_index(signal)])?;
            encoder.encode_pdf(usize::from(coded & 7), &GAIN_LSB)?;
            prior.map_or(coded, |old| coded.max(old.saturating_sub(16)))
        } else {
            encoder.encode_pdf(usize::from(symbol), &GAIN_DELTA)?;
            apply_delta(prior.ok_or(Error::InvalidPacket)?, symbol)?
        };
        result.log[index] = log;
        result.q16[index] = gain_index_to_q16(log)?;
        prior = Some(log);
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelHeader {
    /// One bit per regular 20 ms SILK frame, least-significant frame first.
    pub vad: u8,
    /// One bit per protected frame, least-significant frame first.
    pub lbrr: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerHeader {
    pub channels: u8,
    pub frames: u8,
    pub channel: [ChannelHeader; 2],
}

fn frame_count(duration_ms: u8) -> Result<u8, Error> {
    match duration_ms {
        10 | 20 => Ok(1),
        40 => Ok(2),
        60 => Ok(3),
        _ => Err(Error::InvalidFrameSize),
    }
}

pub fn decode_layer_header(
    decoder: &mut RangeDecoder<'_>,
    duration_ms: u8,
    channels: u8,
) -> Result<LayerHeader, Error> {
    if !(1..=2).contains(&channels) {
        return Err(Error::InvalidPacket);
    }
    let frames = frame_count(duration_ms)?;
    let mut channel = [ChannelHeader { vad: 0, lbrr: 0 }; 2];
    let mut global = [false; 2];
    for index in 0..usize::from(channels) {
        for frame in 0..frames {
            channel[index].vad |= u8::from(decoder.decode_bit_logp(1)?) << frame;
        }
        global[index] = decoder.decode_bit_logp(1)?;
    }
    for index in 0..usize::from(channels) {
        channel[index].lbrr =
            if !global[index] {
                0
            } else {
                match frames {
                    1 => 1,
                    2 => u8::try_from(decoder.decode_pdf(&LBRR_40)?)
                        .map_err(|_| Error::InvalidPacket)?,
                    3 => u8::try_from(decoder.decode_pdf(&LBRR_60)?)
                        .map_err(|_| Error::InvalidPacket)?,
                    _ => return Err(Error::InvalidFrameSize),
                }
            };
        if global[index] && channel[index].lbrr == 0 {
            return Err(Error::InvalidPacket);
        }
    }
    Ok(LayerHeader {
        channels,
        frames,
        channel,
    })
}

pub fn encode_layer_header(
    encoder: &mut RangeEncoder<'_>,
    duration_ms: u8,
    header: LayerHeader,
) -> Result<(), Error> {
    let frames = frame_count(duration_ms)?;
    if header.channels == 0 || header.channels > 2 || header.frames != frames {
        return Err(Error::InvalidPacket);
    }
    let valid_mask = (1u8 << frames) - 1;
    for channel in &header.channel[..usize::from(header.channels)] {
        if channel.vad & !valid_mask != 0 || channel.lbrr & !valid_mask != 0 {
            return Err(Error::InvalidPacket);
        }
        for frame in 0..frames {
            encoder.encode_bit_logp(channel.vad & (1 << frame) != 0, 1)?;
        }
        encoder.encode_bit_logp(channel.lbrr != 0, 1)?;
    }
    if frames > 1 {
        let pdf: &[u8] = if frames == 2 { &LBRR_40 } else { &LBRR_60 };
        for channel in &header.channel[..usize::from(header.channels)] {
            if channel.lbrr != 0 {
                encoder.encode_pdf(usize::from(channel.lbrr), pdf)?;
            }
        }
    } else if header.channel[..usize::from(header.channels)]
        .iter()
        .any(|channel| channel.lbrr > 1)
    {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

pub fn decode_frame_type(
    decoder: &mut RangeDecoder<'_>,
    active: bool,
) -> Result<(SignalType, QuantizationOffset), Error> {
    let symbol = decoder.decode_pdf(if active {
        &FRAME_ACTIVE
    } else {
        &FRAME_INACTIVE
    })?;
    frame_type_from_symbol(symbol)
}

pub fn encode_frame_type(
    encoder: &mut RangeEncoder<'_>,
    active: bool,
    signal: SignalType,
    quantization: QuantizationOffset,
) -> Result<(), Error> {
    let symbol = match (signal, quantization) {
        (SignalType::Inactive, QuantizationOffset::Low) => 0,
        (SignalType::Inactive, QuantizationOffset::High) => 1,
        (SignalType::Unvoiced, QuantizationOffset::Low) => 2,
        (SignalType::Unvoiced, QuantizationOffset::High) => 3,
        (SignalType::Voiced, QuantizationOffset::Low) => 4,
        (SignalType::Voiced, QuantizationOffset::High) => 5,
    };
    if active != (symbol >= 2) {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(
        symbol,
        if active {
            &FRAME_ACTIVE
        } else {
            &FRAME_INACTIVE
        },
    )
}

fn frame_type_from_symbol(symbol: usize) -> Result<(SignalType, QuantizationOffset), Error> {
    Ok(match symbol {
        0 => (SignalType::Inactive, QuantizationOffset::Low),
        1 => (SignalType::Inactive, QuantizationOffset::High),
        2 => (SignalType::Unvoiced, QuantizationOffset::Low),
        3 => (SignalType::Unvoiced, QuantizationOffset::High),
        4 => (SignalType::Voiced, QuantizationOffset::Low),
        5 => (SignalType::Voiced, QuantizationOffset::High),
        _ => return Err(Error::InvalidPacket),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layer_header_round_trips() {
        for duration in [10, 20, 40, 60] {
            let frames = frame_count(duration).unwrap();
            let mask = (1 << frames) - 1;
            for channels in 1..=2 {
                for vad in 0..=mask {
                    for lbrr in 0..=mask {
                        let header = LayerHeader {
                            channels,
                            frames,
                            channel: [
                                ChannelHeader { vad, lbrr },
                                if channels == 2 {
                                    ChannelHeader {
                                        vad: mask ^ vad,
                                        lbrr: mask ^ lbrr,
                                    }
                                } else {
                                    ChannelHeader { vad: 0, lbrr: 0 }
                                },
                            ],
                        };
                        let mut bytes = [0u8; 32];
                        let mut encoder = RangeEncoder::new(&mut bytes);
                        encode_layer_header(&mut encoder, duration, header).unwrap();
                        encoder.finish().unwrap();
                        let mut decoder = RangeDecoder::new(&bytes);
                        assert_eq!(
                            decode_layer_header(&mut decoder, duration, channels),
                            Ok(header)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_valid_frame_type_round_trips() {
        for (active, signal) in [
            (false, SignalType::Inactive),
            (true, SignalType::Unvoiced),
            (true, SignalType::Voiced),
        ] {
            for quantization in [QuantizationOffset::Low, QuantizationOffset::High] {
                let mut bytes = [0u8; 8];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_frame_type(&mut encoder, active, signal, quantization).unwrap();
                encoder.finish().unwrap();
                let mut decoder = RangeDecoder::new(&bytes);
                assert_eq!(
                    decode_frame_type(&mut decoder, active),
                    Ok((signal, quantization))
                );
            }
        }
    }

    #[test]
    fn gain_conversion_matches_normative_endpoints() {
        assert_eq!(gain_index_to_q16(0), Ok(81_920));
        assert_eq!(gain_index_to_q16(63), Ok(1_686_110_208));
    }

    #[test]
    fn independent_and_delta_gains_round_trip() {
        for signal in [
            SignalType::Inactive,
            SignalType::Unvoiced,
            SignalType::Voiced,
        ] {
            for subframes in [2, 4] {
                for first in [0, 1, 31, 63] {
                    let symbols = [first, 0, 4, 40];
                    let mut bytes = [0u8; 32];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    let expected =
                        encode_gains(&mut encoder, signal, subframes, true, Some(37), symbols)
                            .unwrap();
                    encoder.finish().unwrap();
                    let mut decoder = RangeDecoder::new(&bytes);
                    assert_eq!(
                        decode_gains(&mut decoder, signal, subframes, true, Some(37)),
                        Ok(expected)
                    );
                }
                let symbols = [0, 3, 17, 40];
                let mut bytes = [0u8; 32];
                let mut encoder = RangeEncoder::new(&mut bytes);
                let expected =
                    encode_gains(&mut encoder, signal, subframes, false, Some(22), symbols)
                        .unwrap();
                encoder.finish().unwrap();
                let mut decoder = RangeDecoder::new(&bytes);
                assert_eq!(
                    decode_gains(&mut decoder, signal, subframes, false, Some(22)),
                    Ok(expected)
                );
            }
        }
    }
}
