//! Normative SILK entropy models and shell-block pulse metadata.

use crate::{
    Error, RangeDecoder, RangeEncoder,
    silk::{QuantizationOffset, SignalType},
};

const RATE_INACTIVE_UNVOICED: [u8; 9] = [15, 51, 12, 46, 45, 13, 33, 27, 14];
const RATE_VOICED: [u8; 9] = [33, 30, 36, 17, 34, 49, 18, 21, 18];

// First probability in each binary PDF from RFC 6716 Table 52. The final
// axis is the shell pulse count, clamped to the "6 or more" context.
const SIGN_NEGATIVE: [[[u8; 7]; 2]; 3] = [
    [
        [2, 207, 189, 179, 174, 163, 157],
        [58, 245, 238, 232, 225, 220, 211],
    ],
    [
        [1, 210, 190, 178, 169, 162, 152],
        [48, 242, 235, 224, 214, 205, 190],
    ],
    [
        [1, 162, 152, 147, 144, 141, 138],
        [8, 203, 187, 176, 168, 161, 154],
    ],
];

fn sign_pdf(signal: SignalType, quantization: QuantizationOffset, pulses: u8) -> [u8; 2] {
    let signal = match signal {
        SignalType::Inactive => 0,
        SignalType::Unvoiced => 1,
        SignalType::Voiced => 2,
    };
    let quantization = match quantization {
        QuantizationOffset::Low => 0,
        QuantizationOffset::High => 1,
    };
    let negative = SIGN_NEGATIVE[signal][quantization][usize::from(pulses.min(6))];
    [negative, 0u8.wrapping_sub(negative)]
}

/// Applies Table 52 signs to all non-zero excitation magnitudes.
pub fn decode_signs(
    decoder: &mut RangeDecoder<'_>,
    values: &mut [i32],
    signal: SignalType,
    quantization: QuantizationOffset,
    pulses: u8,
) -> Result<(), Error> {
    let pdf = sign_pdf(signal, quantization, pulses);
    for value in values.iter_mut().filter(|value| **value != 0) {
        if decoder.decode_pdf(&pdf)? == 0 {
            *value = value.checked_neg().ok_or(Error::InvalidPacket)?;
        }
    }
    Ok(())
}

/// Encodes Table 52 signs. Input zeroes consume no symbols.
pub fn encode_signs(
    encoder: &mut RangeEncoder<'_>,
    values: &[i32],
    signal: SignalType,
    quantization: QuantizationOffset,
    pulses: u8,
) -> Result<(), Error> {
    let pdf = sign_pdf(signal, quantization, pulses);
    for &value in values.iter().filter(|value| **value != 0) {
        encoder.encode_pdf(usize::from(value > 0), &pdf)?;
    }
    Ok(())
}

const PULSE_COUNT: [[u8; 18]; 11] = [
    [131, 74, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [58, 93, 60, 23, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [43, 51, 46, 33, 24, 16, 11, 8, 6, 3, 3, 3, 2, 1, 1, 2, 1, 2],
    [17, 52, 71, 57, 31, 12, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [6, 21, 41, 53, 49, 35, 21, 11, 6, 3, 2, 2, 1, 1, 1, 1, 1, 1],
    [
        7, 14, 22, 28, 29, 28, 25, 20, 17, 13, 11, 9, 7, 5, 4, 4, 3, 10,
    ],
    [2, 5, 14, 29, 42, 46, 41, 31, 19, 11, 6, 3, 2, 1, 1, 1, 1, 1],
    [
        1, 2, 4, 10, 19, 29, 35, 37, 34, 28, 20, 14, 8, 5, 4, 2, 2, 2,
    ],
    [
        1, 2, 2, 5, 9, 14, 20, 24, 27, 28, 26, 23, 20, 15, 11, 8, 6, 15,
    ],
    [1, 1, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2],
    [2, 1, 6, 27, 58, 56, 39, 25, 14, 10, 6, 3, 3, 2, 1, 1, 2, 0],
];

// RFC 6716 Tables 47-50, concatenated by total pulse count (1 through 16).
const SHELL_16: [u8; 152] = [
    126, 130, 56, 142, 58, 25, 101, 104, 26, 12, 60, 108, 64, 12, 7, 35, 84, 87, 37, 6, 4, 20, 59,
    86, 63, 21, 3, 3, 12, 38, 72, 75, 42, 12, 2, 2, 8, 25, 54, 73, 59, 27, 7, 1, 2, 5, 17, 39, 63,
    65, 42, 18, 4, 1, 1, 4, 12, 28, 49, 63, 54, 30, 11, 3, 1, 1, 4, 8, 20, 37, 55, 57, 41, 22, 8,
    2, 1, 1, 3, 7, 15, 28, 44, 53, 48, 33, 16, 6, 1, 1, 1, 2, 6, 12, 21, 35, 47, 48, 40, 25, 12, 5,
    1, 1, 1, 1, 4, 10, 17, 27, 37, 47, 43, 33, 21, 9, 4, 1, 1, 1, 1, 1, 8, 14, 22, 33, 40, 43, 38,
    28, 16, 8, 1, 1, 1, 1, 1, 1, 1, 13, 18, 27, 36, 41, 41, 34, 24, 14, 1, 1, 1, 1,
];
const SHELL_8: [u8; 152] = [
    127, 129, 53, 149, 54, 22, 105, 106, 23, 11, 61, 111, 63, 10, 6, 35, 86, 88, 36, 5, 4, 20, 59,
    87, 62, 21, 3, 3, 13, 40, 71, 73, 41, 13, 2, 3, 9, 27, 53, 70, 56, 28, 9, 1, 3, 8, 19, 37, 57,
    61, 44, 20, 6, 1, 3, 7, 15, 28, 44, 54, 49, 33, 17, 5, 1, 1, 7, 13, 22, 34, 46, 48, 38, 28, 14,
    4, 1, 1, 1, 11, 22, 27, 35, 42, 47, 33, 25, 10, 1, 1, 1, 1, 6, 14, 26, 37, 43, 43, 37, 26, 14,
    6, 1, 1, 1, 1, 4, 10, 20, 31, 40, 42, 40, 31, 20, 10, 4, 1, 1, 1, 1, 3, 8, 16, 26, 35, 38, 38,
    35, 26, 16, 8, 3, 1, 1, 1, 1, 2, 6, 12, 21, 30, 36, 38, 36, 30, 21, 12, 6, 2, 1, 1,
];
const SHELL_4: [u8; 152] = [
    127, 129, 49, 157, 50, 20, 107, 109, 20, 11, 60, 113, 62, 10, 7, 36, 84, 87, 36, 6, 6, 24, 57,
    82, 60, 23, 4, 5, 18, 39, 64, 68, 42, 16, 4, 6, 14, 29, 47, 61, 52, 30, 14, 3, 1, 15, 23, 35,
    51, 50, 40, 30, 10, 1, 1, 1, 21, 32, 42, 52, 46, 41, 18, 1, 1, 1, 6, 16, 27, 36, 42, 42, 36,
    27, 16, 6, 1, 1, 5, 12, 21, 31, 38, 40, 38, 31, 21, 12, 5, 1, 1, 3, 9, 17, 26, 34, 38, 38, 34,
    26, 17, 9, 3, 1, 1, 3, 7, 14, 22, 29, 34, 36, 34, 29, 22, 14, 7, 3, 1, 1, 2, 5, 11, 18, 25, 31,
    35, 35, 31, 25, 18, 11, 5, 2, 1, 1, 1, 4, 9, 15, 21, 28, 32, 34, 32, 28, 21, 15, 9, 4, 1, 1,
];
const SHELL_2: [u8; 152] = [
    128, 128, 42, 172, 42, 21, 107, 107, 21, 12, 60, 112, 61, 11, 8, 34, 86, 86, 35, 7, 8, 23, 55,
    90, 55, 20, 5, 5, 15, 38, 72, 72, 36, 15, 3, 6, 12, 27, 52, 77, 47, 20, 10, 5, 6, 19, 28, 35,
    40, 40, 35, 28, 19, 6, 4, 14, 22, 31, 37, 40, 37, 31, 22, 14, 4, 3, 10, 18, 26, 33, 38, 38, 33,
    26, 18, 10, 3, 2, 8, 13, 21, 29, 36, 38, 36, 29, 21, 13, 8, 2, 1, 5, 10, 17, 25, 32, 38, 38,
    32, 25, 17, 10, 5, 1, 1, 4, 7, 13, 21, 29, 35, 36, 35, 29, 21, 13, 7, 4, 1, 1, 2, 5, 10, 17,
    25, 32, 36, 36, 32, 25, 17, 10, 5, 2, 1, 1, 2, 4, 7, 13, 21, 28, 34, 36, 34, 28, 21, 13, 7, 4,
    2, 1,
];

fn shell_pdf(partition: usize, pulses: u8) -> Result<&'static [u8], Error> {
    if !(1..=16).contains(&pulses) {
        return Err(Error::InvalidPacket);
    }
    let table = match partition {
        16 => &SHELL_16,
        8 => &SHELL_8,
        4 => &SHELL_4,
        2 => &SHELL_2,
        _ => return Err(Error::InvalidPacket),
    };
    let p = usize::from(pulses);
    let start = p * (p + 1) / 2 - 1;
    Ok(&table[start..start + p + 1])
}

fn decode_shell_node(
    decoder: &mut RangeDecoder<'_>,
    out: &mut [u32],
    pulses: u8,
) -> Result<(), Error> {
    if pulses == 0 {
        out.fill(0);
        return Ok(());
    }
    if out.len() == 1 {
        out[0] = u32::from(pulses);
        return Ok(());
    }
    let left = u8::try_from(decoder.decode_pdf(shell_pdf(out.len(), pulses)?)?)
        .map_err(|_| Error::InvalidPacket)?;
    let (a, b) = out.split_at_mut(out.len() / 2);
    decode_shell_node(decoder, a, left)?;
    decode_shell_node(decoder, b, pulses - left)
}

/// Decodes one normative 16-sample SILK shell block.
pub fn decode_shell(
    decoder: &mut RangeDecoder<'_>,
    pulses: u8,
    out: &mut [u32; 16],
) -> Result<(), Error> {
    if pulses > 16 {
        return Err(Error::InvalidPacket);
    }
    decode_shell_node(decoder, out, pulses)
}

fn encode_shell_node(
    encoder: &mut RangeEncoder<'_>,
    values: &[u32],
    pulses: u8,
) -> Result<(), Error> {
    if pulses == 0 || values.len() == 1 {
        return Ok(());
    }
    let mid = values.len() / 2;
    let left_sum: u32 = values[..mid].iter().try_fold(0u32, |sum, &v| {
        sum.checked_add(v).ok_or(Error::InvalidPacket)
    })?;
    let left = u8::try_from(left_sum).map_err(|_| Error::InvalidPacket)?;
    if left > pulses {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(left), shell_pdf(values.len(), pulses)?)?;
    encode_shell_node(encoder, &values[..mid], left)?;
    encode_shell_node(encoder, &values[mid..], pulses - left)
}

/// Encodes one 16-sample SILK shell block whose total may not exceed 16.
pub fn encode_shell(encoder: &mut RangeEncoder<'_>, values: &[u32; 16]) -> Result<(), Error> {
    let total = values.iter().try_fold(0u32, |sum, &v| {
        sum.checked_add(v).ok_or(Error::InvalidPacket)
    })?;
    let pulses = u8::try_from(total).map_err(|_| Error::InvalidPacket)?;
    if pulses > 16 {
        return Err(Error::InvalidPacket);
    }
    encode_shell_node(encoder, values, pulses)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PulseCount {
    pub base: u8,
    pub extra_lsb: u8,
}

pub const MAX_EXCITATION_SAMPLES: usize = 320;
const MAX_EXCITATION_BLOCKS: usize = MAX_EXCITATION_SAMPLES / 16;

/// Decodes the complete entropy-coded SILK excitation section in its normative
/// cross-block order: rate level, counts, shells, LSBs, then signs.
pub fn decode_excitation_blocks(
    decoder: &mut RangeDecoder<'_>,
    signal: SignalType,
    quantization: QuantizationOffset,
    sample_count: usize,
    output: &mut [i32],
) -> Result<u8, Error> {
    if sample_count == 0 || sample_count > MAX_EXCITATION_SAMPLES || output.len() < sample_count {
        return Err(Error::InvalidFrameSize);
    }
    let blocks = sample_count.div_ceil(16);
    let rate_level = decode_rate_level(decoder, signal)?;
    let mut counts = [PulseCount {
        base: 0,
        extra_lsb: 0,
    }; MAX_EXCITATION_BLOCKS];
    for count in &mut counts[..blocks] {
        *count = decode_pulse_count(decoder, rate_level)?;
    }
    let mut magnitudes = [0u32; MAX_EXCITATION_SAMPLES];
    for (block, count) in counts[..blocks].iter().enumerate() {
        let shell: &mut [u32; 16] = (&mut magnitudes[block * 16..block * 16 + 16])
            .try_into()
            .map_err(|_| Error::InvalidFrameSize)?;
        decode_shell(decoder, count.base, shell)?;
    }
    for (block, count) in counts[..blocks].iter().enumerate() {
        decode_lsbs(
            decoder,
            &mut magnitudes[block * 16..block * 16 + 16],
            count.extra_lsb,
        )?;
    }
    let mut signed = [0i32; MAX_EXCITATION_SAMPLES];
    for (dst, &src) in signed.iter_mut().zip(magnitudes.iter()) {
        *dst = i32::try_from(src).map_err(|_| Error::InvalidPacket)?;
    }
    for (block, count) in counts[..blocks].iter().enumerate() {
        decode_signs(
            decoder,
            &mut signed[block * 16..block * 16 + 16],
            signal,
            quantization,
            count.base,
        )?;
    }
    output[..sample_count].copy_from_slice(&signed[..sample_count]);
    Ok(rate_level)
}

/// Encodes a complete SILK excitation section, choosing the smallest escape
/// depth that reduces each shell block to at most sixteen pulses.
pub fn encode_excitation_blocks(
    encoder: &mut RangeEncoder<'_>,
    signal: SignalType,
    quantization: QuantizationOffset,
    rate_level: u8,
    input: &[i32],
) -> Result<(), Error> {
    if input.is_empty() || input.len() > MAX_EXCITATION_SAMPLES {
        return Err(Error::InvalidFrameSize);
    }
    let blocks = input.len().div_ceil(16);
    let mut full = [0u32; MAX_EXCITATION_SAMPLES];
    let mut signed = [0i32; MAX_EXCITATION_SAMPLES];
    for (index, &value) in input.iter().enumerate() {
        full[index] = value
            .checked_abs()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or(Error::InvalidPacket)?;
        signed[index] = value;
    }
    let mut counts = [PulseCount {
        base: 0,
        extra_lsb: 0,
    }; MAX_EXCITATION_BLOCKS];
    let mut bases = [0u32; MAX_EXCITATION_SAMPLES];
    for (block, count) in counts[..blocks].iter_mut().enumerate() {
        let range = block * 16..block * 16 + 16;
        let mut shift = 0u8;
        loop {
            let sum = full[range.clone()].iter().try_fold(0u32, |sum, &v| {
                sum.checked_add(v >> shift).ok_or(Error::InvalidPacket)
            })?;
            if sum <= 16 {
                *count = PulseCount {
                    base: sum as u8,
                    extra_lsb: shift,
                };
                for index in range.clone() {
                    bases[index] = full[index] >> shift;
                }
                break;
            }
            shift = shift
                .checked_add(1)
                .filter(|&v| v <= 10)
                .ok_or(Error::InvalidPacket)?;
        }
    }
    encode_rate_level(encoder, signal, rate_level)?;
    for &count in &counts[..blocks] {
        encode_pulse_count(encoder, rate_level, count)?;
    }
    for block in 0..blocks {
        let shell: &[u32; 16] = (&bases[block * 16..block * 16 + 16])
            .try_into()
            .map_err(|_| Error::InvalidFrameSize)?;
        encode_shell(encoder, shell)?;
    }
    for (block, count) in counts[..blocks].iter().enumerate() {
        encode_lsbs(
            encoder,
            &bases[block * 16..block * 16 + 16],
            &full[block * 16..block * 16 + 16],
            count.extra_lsb,
        )?;
    }
    for (block, count) in counts[..blocks].iter().enumerate() {
        encode_signs(
            encoder,
            &signed[block * 16..block * 16 + 16],
            signal,
            quantization,
            count.base,
        )?;
    }
    Ok(())
}

pub fn decode_rate_level(decoder: &mut RangeDecoder<'_>, signal: SignalType) -> Result<u8, Error> {
    let table = if signal == SignalType::Voiced {
        &RATE_VOICED
    } else {
        &RATE_INACTIVE_UNVOICED
    };
    u8::try_from(decoder.decode_pdf(table)?).map_err(|_| Error::InvalidPacket)
}

pub fn encode_rate_level(
    encoder: &mut RangeEncoder<'_>,
    signal: SignalType,
    level: u8,
) -> Result<(), Error> {
    let table = if signal == SignalType::Voiced {
        &RATE_VOICED
    } else {
        &RATE_INACTIVE_UNVOICED
    };
    encoder.encode_pdf(usize::from(level), table)
}

pub fn decode_pulse_count(
    decoder: &mut RangeDecoder<'_>,
    rate_level: u8,
) -> Result<PulseCount, Error> {
    if rate_level > 8 {
        return Err(Error::InvalidPacket);
    }
    let mut table = usize::from(rate_level);
    let mut extra = 0u8;
    loop {
        let count = decoder.decode_pdf(&PULSE_COUNT[table])?;
        if count < 17 {
            return Ok(PulseCount {
                base: count as u8,
                extra_lsb: extra,
            });
        }
        extra = extra.checked_add(1).ok_or(Error::InvalidPacket)?;
        if extra > 10 {
            return Err(Error::InvalidPacket);
        }
        table = if extra == 10 { 10 } else { 9 };
    }
}

pub fn encode_pulse_count(
    encoder: &mut RangeEncoder<'_>,
    rate_level: u8,
    count: PulseCount,
) -> Result<(), Error> {
    if rate_level > 8 || count.base > 16 || count.extra_lsb > 10 {
        return Err(Error::InvalidPacket);
    }
    if count.extra_lsb == 0 {
        return encoder.encode_pdf(
            usize::from(count.base),
            &PULSE_COUNT[usize::from(rate_level)],
        );
    }
    encoder.encode_pdf(17, &PULSE_COUNT[usize::from(rate_level)])?;
    for layer in 1..count.extra_lsb {
        encoder.encode_pdf(17, &PULSE_COUNT[if layer == 10 { 10 } else { 9 }])?;
    }
    encoder.encode_pdf(
        usize::from(count.base),
        &PULSE_COUNT[if count.extra_lsb == 10 { 10 } else { 9 }],
    )
}

/// Decodes the per-coefficient extra least-significant pulse bits.
pub fn decode_lsbs(
    decoder: &mut RangeDecoder<'_>,
    magnitudes: &mut [u32],
    extra_lsb: u8,
) -> Result<(), Error> {
    if extra_lsb > 10 {
        return Err(Error::InvalidPacket);
    }
    // RFC 6716 section 4.2.7.8.4 codes every bit of one coefficient,
    // most-significant first, before advancing to the next coefficient.
    for magnitude in magnitudes.iter_mut() {
        for _ in 0..extra_lsb {
            *magnitude = magnitude
                .checked_mul(2)
                .and_then(|value| {
                    value.checked_add(u32::try_from(decoder.decode_pdf(&[136, 120]).ok()?).ok()?)
                })
                .ok_or(Error::InvalidPacket)?;
        }
    }
    Ok(())
}

pub fn encode_lsbs(
    encoder: &mut RangeEncoder<'_>,
    base: &[u32],
    full: &[u32],
    extra_lsb: u8,
) -> Result<(), Error> {
    if base.len() != full.len() || extra_lsb > 10 {
        return Err(Error::InvalidPacket);
    }
    for (&base, &full) in base.iter().zip(full) {
        for bit in (0..extra_lsb).rev() {
            if full >> extra_lsb != base {
                return Err(Error::InvalidPacket);
            }
            encoder.encode_pdf(usize::from(((full >> bit) & 1) as u8), &[136, 120])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lsb_symbols_are_coefficient_major_and_most_significant_first() {
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        // Base magnitudes [1, 0] followed by two LSBs per coefficient:
        // coefficient 0 -> 1,0; coefficient 1 -> 0,1.
        for symbol in [1usize, 0, 0, 1] {
            encoder.encode_pdf(symbol, &[136, 120]).unwrap();
        }
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let mut magnitudes = [1, 0];
        decode_lsbs(&mut decoder, &mut magnitudes, 2).unwrap();
        assert_eq!(magnitudes, [6, 1]);
    }
    #[test]
    fn every_rate_level_round_trips_for_each_signal_type() {
        for signal in [
            SignalType::Inactive,
            SignalType::Unvoiced,
            SignalType::Voiced,
        ] {
            for level in 0..=8 {
                let mut bytes = [0u8; 16];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_rate_level(&mut encoder, signal, level).unwrap();
                encoder.finish().unwrap();
                let mut decoder = RangeDecoder::new(&bytes);
                assert_eq!(decode_rate_level(&mut decoder, signal), Ok(level));
            }
        }
    }
    #[test]
    fn pulse_escape_depths_round_trip() {
        for level in 0..=8 {
            for extra in 0..=10 {
                for base in [0, 1, 8, 16] {
                    let expected = PulseCount {
                        base,
                        extra_lsb: extra,
                    };
                    let mut bytes = [0u8; 32];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    encode_pulse_count(&mut encoder, level, expected).unwrap();
                    encoder.finish().unwrap();
                    let mut decoder = RangeDecoder::new(&bytes);
                    assert_eq!(decode_pulse_count(&mut decoder, level), Ok(expected));
                }
            }
        }
    }
    #[test]
    fn coefficient_lsbs_round_trip() {
        let base = [1, 0, 3, 2];
        let full = [13, 2, 25, 19];
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_lsbs(&mut encoder, &base, &full, 3).unwrap();
        encoder.finish().unwrap();
        let mut decoded = base;
        let mut decoder = RangeDecoder::new(&bytes);
        decode_lsbs(&mut decoder, &mut decoded, 3).unwrap();
        assert_eq!(decoded, full);
    }

    #[test]
    fn shell_blocks_round_trip_across_totals_and_positions() {
        for total in 0u32..=16 {
            for position in 0..16 {
                let mut expected = [0u32; 16];
                expected[position] = total;
                let mut bytes = [0u8; 64];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_shell(&mut encoder, &expected).unwrap();
                encoder.finish().unwrap();
                let mut actual = [99u32; 16];
                let mut decoder = RangeDecoder::new(&bytes);
                decode_shell(&mut decoder, total as u8, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn shell_distributed_vectors_round_trip() {
        for total in 1u32..=16 {
            let mut expected = [0u32; 16];
            for pulse in 0..total {
                expected[((pulse * 7 + total * 3) & 15) as usize] += 1;
            }
            let mut bytes = [0u8; 64];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_shell(&mut encoder, &expected).unwrap();
            encoder.finish().unwrap();
            let mut actual = [0u32; 16];
            let mut decoder = RangeDecoder::new(&bytes);
            decode_shell(&mut decoder, total as u8, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn shell_rejects_excess_pulses() {
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        let mut values = [0u32; 16];
        values[0] = 17;
        assert_eq!(
            encode_shell(&mut encoder, &values),
            Err(Error::InvalidPacket)
        );
        let mut decoder = RangeDecoder::new(&bytes);
        assert_eq!(
            decode_shell(&mut decoder, 17, &mut values),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn signs_round_trip_in_every_context() {
        for signal in [
            SignalType::Inactive,
            SignalType::Unvoiced,
            SignalType::Voiced,
        ] {
            for quantization in [QuantizationOffset::Low, QuantizationOffset::High] {
                for pulses in 0..=16 {
                    let expected = [-9, 0, 3, -1, 0, 7];
                    let mut bytes = [0u8; 32];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    encode_signs(&mut encoder, &expected, signal, quantization, pulses).unwrap();
                    encoder.finish().unwrap();
                    let mut actual = [9, 0, 3, 1, 0, 7];
                    let mut decoder = RangeDecoder::new(&bytes);
                    decode_signs(&mut decoder, &mut actual, signal, quantization, pulses).unwrap();
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn complete_excitation_sections_round_trip() {
        for sample_count in [1usize, 16, 17, 80, 160, 168, 320] {
            let mut expected = [0i32; MAX_EXCITATION_SAMPLES];
            for (index, value) in expected[..sample_count].iter_mut().enumerate() {
                let magnitude = ((index * 37 + sample_count * 11) % 513) as i32;
                *value = if index % 5 == 0 {
                    0
                } else if index & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
            let mut bytes = [0u8; 4096];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_excitation_blocks(
                &mut encoder,
                SignalType::Voiced,
                QuantizationOffset::High,
                4,
                &expected[..sample_count],
            )
            .unwrap();
            encoder.finish().unwrap();
            let mut actual = [0i32; MAX_EXCITATION_SAMPLES];
            let mut decoder = RangeDecoder::new(&bytes);
            assert_eq!(
                decode_excitation_blocks(
                    &mut decoder,
                    SignalType::Voiced,
                    QuantizationOffset::High,
                    sample_count,
                    &mut actual
                ),
                Ok(4)
            );
            assert_eq!(&actual[..sample_count], &expected[..sample_count]);
        }
    }

    #[test]
    fn excitation_rejects_invalid_sizes_and_unrepresentable_magnitudes() {
        let mut bytes = [0u8; 64];
        let mut encoder = RangeEncoder::new(&mut bytes);
        assert_eq!(
            encode_excitation_blocks(
                &mut encoder,
                SignalType::Unvoiced,
                QuantizationOffset::Low,
                0,
                &[]
            ),
            Err(Error::InvalidFrameSize)
        );
        let huge = [i32::MAX; 16];
        assert_eq!(
            encode_excitation_blocks(
                &mut encoder,
                SignalType::Unvoiced,
                QuantizationOffset::Low,
                0,
                &huge
            ),
            Err(Error::InvalidPacket)
        );
        let mut output = [0i32; 1];
        let mut decoder = RangeDecoder::new(&bytes);
        assert_eq!(
            decode_excitation_blocks(
                &mut decoder,
                SignalType::Unvoiced,
                QuantizationOffset::Low,
                0,
                &mut output
            ),
            Err(Error::InvalidFrameSize)
        );
    }
}
