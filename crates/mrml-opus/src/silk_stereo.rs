//! SILK stereo predictor and mid-only entropy syntax.

use crate::{Error, RangeDecoder, RangeEncoder, silk::StereoWeights};

const STAGE1: [u8; 25] = [
    7, 2, 1, 1, 1, 10, 24, 8, 1, 1, 3, 23, 92, 23, 3, 1, 1, 8, 24, 10, 1, 1, 1, 2, 7,
];
const STAGE2: [u8; 3] = [85, 86, 85];
const STAGE3: [u8; 5] = [51, 51, 52, 51, 51];
const WEIGHT_Q13: [i16; 16] = [
    -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820, 820, 2950, 5000, 6500, 7526, 8266,
    10050, 13732,
];
const MID_ONLY: [u8; 2] = [192, 64];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StereoPrediction {
    pub stage1: u8,
    pub low0: u8,
    pub interpolation0: u8,
    pub low1: u8,
    pub interpolation1: u8,
    pub weights: StereoWeights,
}

fn reconstruct(
    stage1: u8,
    low0: u8,
    interpolation0: u8,
    low1: u8,
    interpolation1: u8,
) -> Result<StereoWeights, Error> {
    if stage1 >= 25 || low0 >= 3 || low1 >= 3 || interpolation0 >= 5 || interpolation1 >= 5 {
        return Err(Error::InvalidPacket);
    }
    let index0 = usize::from(low0 + 3 * (stage1 / 5));
    let index1 = usize::from(low1 + 3 * (stage1 % 5));
    let interpolate = |index: usize, factor: u8| -> i32 {
        let low = i32::from(WEIGHT_Q13[index]);
        let step = ((i32::from(WEIGHT_Q13[index + 1]) - low) * 6554) >> 16;
        low + step * (2 * i32::from(factor) + 1)
    };
    let w1 = interpolate(index1, interpolation1);
    let w0 = interpolate(index0, interpolation0) - w1;
    Ok(StereoWeights {
        w0_q13: i16::try_from(w0).map_err(|_| Error::InvalidPacket)?,
        w1_q13: i16::try_from(w1).map_err(|_| Error::InvalidPacket)?,
    })
}

pub fn prediction_from_indices(
    stage1: u8,
    low0: u8,
    interpolation0: u8,
    low1: u8,
    interpolation1: u8,
) -> Result<StereoPrediction, Error> {
    Ok(StereoPrediction {
        stage1,
        low0,
        interpolation0,
        low1,
        interpolation1,
        weights: reconstruct(stage1, low0, interpolation0, low1, interpolation1)?,
    })
}

pub fn decode_prediction(decoder: &mut RangeDecoder<'_>) -> Result<StereoPrediction, Error> {
    let stage1 = decoder.decode_pdf(&STAGE1)? as u8;
    let low0 = decoder.decode_pdf(&STAGE2)? as u8;
    let interpolation0 = decoder.decode_pdf(&STAGE3)? as u8;
    let low1 = decoder.decode_pdf(&STAGE2)? as u8;
    let interpolation1 = decoder.decode_pdf(&STAGE3)? as u8;
    Ok(StereoPrediction {
        stage1,
        low0,
        interpolation0,
        low1,
        interpolation1,
        weights: reconstruct(stage1, low0, interpolation0, low1, interpolation1)?,
    })
}

pub fn encode_prediction(
    encoder: &mut RangeEncoder<'_>,
    prediction: &StereoPrediction,
) -> Result<(), Error> {
    if reconstruct(
        prediction.stage1,
        prediction.low0,
        prediction.interpolation0,
        prediction.low1,
        prediction.interpolation1,
    )? != prediction.weights
    {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(prediction.stage1), &STAGE1)?;
    encoder.encode_pdf(usize::from(prediction.low0), &STAGE2)?;
    encoder.encode_pdf(usize::from(prediction.interpolation0), &STAGE3)?;
    encoder.encode_pdf(usize::from(prediction.low1), &STAGE2)?;
    encoder.encode_pdf(usize::from(prediction.interpolation1), &STAGE3)
}

pub fn decode_mid_only(decoder: &mut RangeDecoder<'_>) -> Result<bool, Error> {
    Ok(decoder.decode_pdf(&MID_ONLY)? != 0)
}
pub fn encode_mid_only(encoder: &mut RangeEncoder<'_>, mid_only: bool) -> Result<(), Error> {
    encoder.encode_pdf(usize::from(mid_only), &MID_ONLY)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_stereo_predictor_and_mid_only_value_round_trips() {
        for n in 0..25 {
            for a in 0..3 {
                for b in 0..5 {
                    for c in 0..3 {
                        for d in 0..5 {
                            let expected = StereoPrediction {
                                stage1: n,
                                low0: a,
                                interpolation0: b,
                                low1: c,
                                interpolation1: d,
                                weights: reconstruct(n, a, b, c, d).unwrap(),
                            };
                            let mut bytes = [0u8; 32];
                            let mut encoder = RangeEncoder::new(&mut bytes);
                            encode_prediction(&mut encoder, &expected).unwrap();
                            encoder.finish().unwrap();
                            assert_eq!(
                                decode_prediction(&mut RangeDecoder::new(&bytes)),
                                Ok(expected)
                            );
                        }
                    }
                }
            }
        }
        for value in [false, true] {
            let mut bytes = [0u8; 8];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_mid_only(&mut encoder, value).unwrap();
            encoder.finish().unwrap();
            assert_eq!(decode_mid_only(&mut RangeDecoder::new(&bytes)), Ok(value));
        }
    }
}
