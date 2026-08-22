//! SILK long-term-prediction pitch entropy syntax.

use crate::{Bandwidth, Error, RangeDecoder, RangeEncoder};

const LAG_HIGH: [u8; 32] = [
    3, 3, 6, 11, 21, 30, 32, 19, 11, 10, 12, 13, 13, 12, 11, 9, 8, 7, 6, 4, 2, 2, 2, 1, 1, 1, 1, 1,
    1, 1, 1, 1,
];
const LAG_LOW_NB: [u8; 4] = [64; 4];
const LAG_LOW_MB: [u8; 6] = [43, 42, 43, 43, 42, 43];
const LAG_LOW_WB: [u8; 8] = [32; 8];
const LAG_DELTA: [u8; 21] = [
    46, 2, 2, 3, 4, 6, 10, 15, 26, 38, 30, 22, 15, 10, 7, 6, 4, 4, 2, 2, 2,
];
const CONTOUR_NB_10_PDF: [u8; 3] = [143, 50, 63];
const CONTOUR_NB_20_PDF: [u8; 11] = [68, 12, 21, 17, 19, 22, 30, 24, 17, 16, 10];
const CONTOUR_WIDE_10_PDF: [u8; 12] = [91, 46, 39, 19, 14, 12, 8, 7, 6, 5, 5, 4];
const CONTOUR_WIDE_20_PDF: [u8; 34] = [
    33, 22, 18, 16, 15, 14, 14, 13, 13, 10, 9, 9, 8, 6, 6, 6, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2,
    2, 2, 1, 1, 1, 1,
];
const CONTOUR_NB_10: [[i8; 4]; 3] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]];
const CONTOUR_NB_20: [[i8; 4]; 11] = [
    [0, 0, 0, 0],
    [2, 1, 0, -1],
    [-1, 0, 1, 2],
    [-1, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [1, 0, 0, -1],
];
const CONTOUR_WIDE_10: [[i8; 4]; 12] = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [-1, 1, 0, 0],
    [1, -1, 0, 0],
    [-1, 2, 0, 0],
    [2, -1, 0, 0],
    [-2, 2, 0, 0],
    [2, -2, 0, 0],
    [-2, 3, 0, 0],
    [3, -2, 0, 0],
    [-3, 3, 0, 0],
];
const CONTOUR_WIDE_20: [[i8; 4]; 34] = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [-1, 0, 0, 1],
    [0, 0, 0, -1],
    [-1, 0, 1, 2],
    [1, 0, 0, -1],
    [-2, -1, 1, 2],
    [2, 1, 0, -1],
    [-2, 0, 0, 2],
    [-2, 0, 1, 3],
    [2, 1, -1, -2],
    [-3, -1, 1, 3],
    [2, 0, 0, -2],
    [3, 1, 0, -2],
    [-3, -1, 2, 4],
    [-4, -1, 1, 4],
    [3, 1, -1, -3],
    [-4, -1, 2, 5],
    [4, 2, -1, -3],
    [4, 1, -1, -4],
    [-5, -1, 2, 6],
    [5, 2, -1, -4],
    [-6, -2, 2, 6],
    [-5, -2, 2, 5],
    [6, 2, -1, -5],
    [-7, -2, 3, 8],
    [6, 2, -2, -6],
    [5, 2, -2, -5],
    [8, 3, -2, -7],
    [-9, -3, 3, 9],
];
const PERIODICITY_PDF: [u8; 3] = [77, 80, 99];
const LTP_PDF_0: [u8; 8] = [185, 15, 13, 13, 9, 9, 6, 6];
const LTP_PDF_1: [u8; 16] = [57, 34, 21, 20, 15, 13, 12, 13, 10, 10, 9, 10, 9, 8, 7, 8];
const LTP_PDF_2: [u8; 32] = [
    15, 16, 14, 12, 12, 12, 11, 11, 11, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 6, 6, 5, 4, 5, 4, 4, 4,
    3, 4, 3, 2,
];
const LTP_SCALE_PDF: [u8; 3] = [128, 64, 64];
const LTP_SCALE_Q14: [u16; 3] = [15_565, 12_288, 8_192];
const SEED_PDF: [u8; 4] = [64; 4];
const LTP_0: [[i8; 5]; 8] = [
    [4, 6, 24, 7, 5],
    [0, 0, 2, 0, 0],
    [12, 28, 41, 13, -4],
    [-9, 15, 42, 25, 14],
    [1, -2, 62, 41, -9],
    [-10, 37, 65, -4, 3],
    [-6, 4, 66, 7, -8],
    [16, 14, 38, -3, 33],
];
const LTP_1: [[i8; 5]; 16] = [
    [13, 22, 39, 23, 12],
    [-1, 36, 64, 27, -6],
    [-7, 10, 55, 43, 17],
    [1, 1, 8, 1, 1],
    [6, -11, 74, 53, -9],
    [-12, 55, 76, -12, 8],
    [-3, 3, 93, 27, -4],
    [26, 39, 59, 3, -8],
    [2, 0, 77, 11, 9],
    [-8, 22, 44, -6, 7],
    [40, 9, 26, 3, 9],
    [-7, 20, 101, -7, 4],
    [3, -8, 42, 26, 0],
    [-15, 33, 68, 2, 23],
    [-2, 55, 46, -2, 15],
    [3, -1, 21, 16, 41],
];
const LTP_2: [[i8; 5]; 32] = [
    [-6, 27, 61, 39, 5],
    [-11, 42, 88, 4, 1],
    [-2, 60, 65, 6, -4],
    [-1, -5, 73, 56, 1],
    [-9, 19, 94, 29, -9],
    [0, 12, 99, 6, 4],
    [8, -19, 102, 46, -13],
    [3, 2, 13, 3, 2],
    [9, -21, 84, 72, -18],
    [-11, 46, 104, -22, 8],
    [18, 38, 48, 23, 0],
    [-16, 70, 83, -21, 11],
    [5, -11, 117, 22, -8],
    [-6, 23, 117, -12, 3],
    [3, -8, 95, 28, 4],
    [-10, 15, 77, 60, -15],
    [-1, 4, 124, 2, -4],
    [3, 38, 84, 24, -25],
    [2, 13, 42, 13, 31],
    [21, -4, 56, 46, -1],
    [-1, 35, 79, -13, 19],
    [-7, 65, 88, -9, -14],
    [20, 4, 81, 49, -29],
    [20, 0, 75, 3, -17],
    [5, -9, 44, 92, -8],
    [1, -3, 22, 69, 31],
    [-6, 95, 41, -12, 5],
    [39, 67, 16, -4, 1],
    [0, -6, 120, 55, -36],
    [-13, 44, 122, 4, -24],
    [81, 5, 11, 3, 7],
    [2, 0, 9, 10, 88],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LtpFilters {
    pub periodicity: u8,
    pub subframes: u8,
    pub indices: [u8; 4],
    pub taps_q7: [[i8; 5]; 4],
    pub scale_q14: u16,
}

fn ltp_model(periodicity: u8) -> Result<(&'static [u8], &'static [[i8; 5]]), Error> {
    match periodicity {
        0 => Ok((&LTP_PDF_0, &LTP_0)),
        1 => Ok((&LTP_PDF_1, &LTP_1)),
        2 => Ok((&LTP_PDF_2, &LTP_2)),
        _ => Err(Error::InvalidPacket),
    }
}

pub fn decode_ltp_filters(
    decoder: &mut RangeDecoder<'_>,
    subframes: u8,
    scale_present: bool,
) -> Result<LtpFilters, Error> {
    if !matches!(subframes, 2 | 4) {
        return Err(Error::InvalidFrameSize);
    }
    let periodicity =
        u8::try_from(decoder.decode_pdf(&PERIODICITY_PDF)?).map_err(|_| Error::InvalidPacket)?;
    let (pdf, codebook) = ltp_model(periodicity)?;
    let mut result = LtpFilters {
        periodicity,
        subframes,
        indices: [0; 4],
        taps_q7: [[0; 5]; 4],
        scale_q14: 15_565,
    };
    for subframe in 0..usize::from(subframes) {
        let index = decoder.decode_pdf(pdf)?;
        result.indices[subframe] = index as u8;
        result.taps_q7[subframe] = *codebook.get(index).ok_or(Error::InvalidPacket)?;
    }
    if scale_present {
        let index = decoder.decode_pdf(&LTP_SCALE_PDF)?;
        result.scale_q14 = LTP_SCALE_Q14[index];
    }
    Ok(result)
}

pub fn encode_ltp_filters(
    encoder: &mut RangeEncoder<'_>,
    parameters: &LtpFilters,
    scale_present: bool,
) -> Result<(), Error> {
    if !matches!(parameters.subframes, 2 | 4) {
        return Err(Error::InvalidFrameSize);
    }
    let (pdf, codebook) = ltp_model(parameters.periodicity)?;
    encoder.encode_pdf(usize::from(parameters.periodicity), &PERIODICITY_PDF)?;
    for subframe in 0..usize::from(parameters.subframes) {
        let index = usize::from(parameters.indices[subframe]);
        if codebook.get(index) != Some(&parameters.taps_q7[subframe]) {
            return Err(Error::InvalidPacket);
        }
        encoder.encode_pdf(index, pdf)?;
    }
    if scale_present {
        let index = LTP_SCALE_Q14
            .iter()
            .position(|&value| value == parameters.scale_q14)
            .ok_or(Error::InvalidPacket)?;
        encoder.encode_pdf(index, &LTP_SCALE_PDF)?;
    } else if parameters.scale_q14 != 15_565 {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

pub fn decode_seed(decoder: &mut RangeDecoder<'_>) -> Result<u8, Error> {
    u8::try_from(decoder.decode_pdf(&SEED_PDF)?).map_err(|_| Error::InvalidPacket)
}

pub fn encode_seed(encoder: &mut RangeEncoder<'_>, seed: u8) -> Result<(), Error> {
    if seed > 3 {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(seed), &SEED_PDF)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PitchContour {
    pub index: u8,
    pub subframes: u8,
    pub lags: [u16; 4],
}

type ContourModel = (&'static [u8], &'static [[i8; 4]], u8);

fn contour_model(bandwidth: Bandwidth, twenty_ms: bool) -> Result<ContourModel, Error> {
    match (bandwidth, twenty_ms) {
        (Bandwidth::Narrow, false) => Ok((&CONTOUR_NB_10_PDF, &CONTOUR_NB_10, 2)),
        (Bandwidth::Narrow, true) => Ok((&CONTOUR_NB_20_PDF, &CONTOUR_NB_20, 4)),
        (Bandwidth::Medium | Bandwidth::Wide, false) => {
            Ok((&CONTOUR_WIDE_10_PDF, &CONTOUR_WIDE_10, 2))
        }
        (Bandwidth::Medium | Bandwidth::Wide, true) => {
            Ok((&CONTOUR_WIDE_20_PDF, &CONTOUR_WIDE_20, 4))
        }
        _ => Err(Error::InvalidPacket),
    }
}

fn contour_from_index(
    bandwidth: Bandwidth,
    twenty_ms: bool,
    primary: i16,
    index: usize,
) -> Result<PitchContour, Error> {
    let (_, vectors, subframes) = contour_model(bandwidth, twenty_ms)?;
    let vector = vectors.get(index).ok_or(Error::InvalidPacket)?;
    let (_, _, minimum) = low_model(bandwidth)?;
    let maximum = minimum * 9;
    let mut lags = [0u16; 4];
    for subframe in 0..usize::from(subframes) {
        lags[subframe] =
            u16::try_from((primary + i16::from(vector[subframe])).clamp(minimum, maximum))
                .map_err(|_| Error::InvalidPacket)?;
    }
    Ok(PitchContour {
        index: index as u8,
        subframes,
        lags,
    })
}

pub fn decode_contour(
    decoder: &mut RangeDecoder<'_>,
    bandwidth: Bandwidth,
    twenty_ms: bool,
    primary: i16,
) -> Result<PitchContour, Error> {
    let (pdf, _, _) = contour_model(bandwidth, twenty_ms)?;
    let index = decoder.decode_pdf(pdf)?;
    contour_from_index(bandwidth, twenty_ms, primary, index)
}

pub fn encode_contour(
    encoder: &mut RangeEncoder<'_>,
    bandwidth: Bandwidth,
    twenty_ms: bool,
    primary: i16,
    index: u8,
) -> Result<PitchContour, Error> {
    let (pdf, _, _) = contour_model(bandwidth, twenty_ms)?;
    let result = contour_from_index(bandwidth, twenty_ms, primary, usize::from(index))?;
    encoder.encode_pdf(usize::from(index), pdf)?;
    Ok(result)
}

fn low_model(bandwidth: Bandwidth) -> Result<(&'static [u8], i16, i16), Error> {
    match bandwidth {
        Bandwidth::Narrow => Ok((&LAG_LOW_NB, 4, 16)),
        Bandwidth::Medium => Ok((&LAG_LOW_MB, 6, 24)),
        Bandwidth::Wide => Ok((&LAG_LOW_WB, 8, 32)),
        _ => Err(Error::InvalidPacket),
    }
}

fn decode_absolute(decoder: &mut RangeDecoder<'_>, bandwidth: Bandwidth) -> Result<i16, Error> {
    let (low_pdf, scale, minimum) = low_model(bandwidth)?;
    let high = i16::try_from(decoder.decode_pdf(&LAG_HIGH)?).map_err(|_| Error::InvalidPacket)?;
    let low = i16::try_from(decoder.decode_pdf(low_pdf)?).map_err(|_| Error::InvalidPacket)?;
    Ok(high * scale + low + minimum)
}

pub fn decode_primary_lag(
    decoder: &mut RangeDecoder<'_>,
    bandwidth: Bandwidth,
    previous: Option<i16>,
) -> Result<i16, Error> {
    if let Some(previous) = previous {
        let delta =
            i16::try_from(decoder.decode_pdf(&LAG_DELTA)?).map_err(|_| Error::InvalidPacket)?;
        if delta != 0 {
            return previous.checked_add(delta - 9).ok_or(Error::InvalidPacket);
        }
    }
    decode_absolute(decoder, bandwidth)
}

pub fn encode_primary_lag(
    encoder: &mut RangeEncoder<'_>,
    bandwidth: Bandwidth,
    previous: Option<i16>,
    lag: i16,
    force_absolute: bool,
) -> Result<(), Error> {
    if let Some(previous) = previous {
        let delta = lag.checked_sub(previous).ok_or(Error::InvalidPacket)?;
        if !force_absolute && (-8..=11).contains(&delta) {
            return encoder.encode_pdf(
                usize::try_from(delta + 9).map_err(|_| Error::InvalidPacket)?,
                &LAG_DELTA,
            );
        }
        encoder.encode_pdf(0, &LAG_DELTA)?;
    }
    let (low_pdf, scale, minimum) = low_model(bandwidth)?;
    let offset = lag
        .checked_sub(minimum)
        .filter(|&value| value >= 0)
        .ok_or(Error::InvalidPacket)?;
    let high = offset / scale;
    let low = offset % scale;
    if high >= 32 {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(high as usize, &LAG_HIGH)?;
    encoder.encode_pdf(low as usize, low_pdf)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absolute_and_relative_lags_round_trip() {
        for (bandwidth, minimum, maximum) in [
            (Bandwidth::Narrow, 16, 143),
            (Bandwidth::Medium, 24, 215),
            (Bandwidth::Wide, 32, 287),
        ] {
            for lag in minimum..=maximum {
                let mut bytes = [0u8; 16];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_primary_lag(&mut encoder, bandwidth, None, lag, true).unwrap();
                encoder.finish().unwrap();
                assert_eq!(
                    decode_primary_lag(&mut RangeDecoder::new(&bytes), bandwidth, None),
                    Ok(lag)
                );
            }
            for delta in -8..=11 {
                let previous = 100;
                let mut bytes = [0u8; 16];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_primary_lag(
                    &mut encoder,
                    bandwidth,
                    Some(previous),
                    previous + delta,
                    false,
                )
                .unwrap();
                encoder.finish().unwrap();
                assert_eq!(
                    decode_primary_lag(&mut RangeDecoder::new(&bytes), bandwidth, Some(previous)),
                    Ok(previous + delta)
                );
            }
        }
    }

    #[test]
    fn every_pitch_contour_round_trips_and_clamps() {
        for bandwidth in [Bandwidth::Narrow, Bandwidth::Medium, Bandwidth::Wide] {
            for twenty_ms in [false, true] {
                let (pdf, _, _) = contour_model(bandwidth, twenty_ms).unwrap();
                for index in 0..pdf.len() {
                    let mut bytes = [0u8; 16];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    let expected =
                        encode_contour(&mut encoder, bandwidth, twenty_ms, 10_000, index as u8)
                            .unwrap();
                    encoder.finish().unwrap();
                    assert_eq!(
                        decode_contour(
                            &mut RangeDecoder::new(&bytes),
                            bandwidth,
                            twenty_ms,
                            10_000
                        ),
                        Ok(expected)
                    );
                    let (_, _, minimum) = low_model(bandwidth).unwrap();
                    assert!(
                        expected.lags[..usize::from(expected.subframes)]
                            .iter()
                            .all(|&lag| (minimum as u16..=(minimum * 9) as u16).contains(&lag))
                    );
                }
            }
        }
    }

    #[test]
    fn every_ltp_filter_vector_and_scale_round_trips() {
        for periodicity in 0..=2 {
            let (_, codebook) = ltp_model(periodicity).unwrap();
            for (index, &taps) in codebook.iter().enumerate() {
                for subframes in [2, 4] {
                    for scale_q14 in LTP_SCALE_Q14 {
                        let mut expected = LtpFilters {
                            periodicity,
                            subframes,
                            indices: [0; 4],
                            taps_q7: [[0; 5]; 4],
                            scale_q14,
                        };
                        expected.indices[..usize::from(subframes)].fill(index as u8);
                        expected.taps_q7[..usize::from(subframes)].fill(taps);
                        let mut bytes = [0u8; 64];
                        let mut encoder = RangeEncoder::new(&mut bytes);
                        encode_ltp_filters(&mut encoder, &expected, true).unwrap();
                        encoder.finish().unwrap();
                        assert_eq!(
                            decode_ltp_filters(&mut RangeDecoder::new(&bytes), subframes, true),
                            Ok(expected)
                        );
                    }
                }
            }
        }
        for seed in 0..=3 {
            let mut bytes = [0u8; 8];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_seed(&mut encoder, seed).unwrap();
            encoder.finish().unwrap();
            assert_eq!(decode_seed(&mut RangeDecoder::new(&bytes)), Ok(seed));
        }
    }
}
