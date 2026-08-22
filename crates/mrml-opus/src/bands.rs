//! CELT spectral bands and time-frequency resolution controls.

use crate::{Error, RangeDecoder, RangeEncoder};

pub const BAND_COUNT: usize = 21;
pub const BAND_EDGES_2_5_MS: [u16; BAND_COUNT + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// CELT's per-band mean log energies, in base-2 units.
pub const ENERGY_MEANS: [f32; BAND_COUNT] = [
    6.4375, 6.25, 5.75, 5.3125, 5.0625, 4.8125, 4.5, 4.375, 4.875, 4.6875, 4.5625, 4.4375, 4.875,
    4.625, 4.3125, 4.5, 4.375, 4.625, 4.75, 4.4375, 3.75,
];

pub const fn band_edge(band: usize, lm: u8) -> Option<usize> {
    if band > BAND_COUNT || lm > 3 {
        None
    } else {
        Some((BAND_EDGES_2_5_MS[band] as usize) << lm)
    }
}
pub fn band_range(band: usize, lm: u8) -> Result<core::ops::Range<usize>, Error> {
    let start = band_edge(band, lm).ok_or(Error::InvalidFrameSize)?;
    let end = band_edge(band + 1, lm).ok_or(Error::InvalidFrameSize)?;
    Ok(start..end)
}

pub fn normalize_bands(
    input: &[f32],
    lm: u8,
    amplitudes: &mut [f32],
    normalized: &mut [f32],
) -> Result<(), Error> {
    let bins = band_edge(BAND_COUNT, lm).ok_or(Error::InvalidFrameSize)?;
    if amplitudes.len() < BAND_COUNT || normalized.len() < bins || input.len() < bins {
        return Err(Error::BufferTooSmall);
    }
    for (band, amplitude) in amplitudes.iter_mut().enumerate().take(BAND_COUNT) {
        let range = band_range(band, lm)?;
        let energy = input[range.clone()]
            .iter()
            .fold(0.0f32, |sum, value| sum + value * value);
        *amplitude = mrml_math::sqrt(energy);
        if *amplitude > 0.0 {
            let inverse = 1.0 / *amplitude;
            for index in range {
                normalized[index] = input[index] * inverse;
            }
        } else {
            for index in range {
                normalized[index] = 0.0;
            }
        }
    }
    Ok(())
}

pub fn denormalize_bands(
    normalized: &[f32],
    lm: u8,
    amplitudes: &[f32],
    output: &mut [f32],
) -> Result<(), Error> {
    let bins = band_edge(BAND_COUNT, lm).ok_or(Error::InvalidFrameSize)?;
    if amplitudes.len() < BAND_COUNT || output.len() < bins || normalized.len() < bins {
        return Err(Error::BufferTooSmall);
    }
    for (band, &amplitude) in amplitudes.iter().enumerate().take(BAND_COUNT) {
        for index in band_range(band, lm)? {
            output[index] = normalized[index] * amplitude;
        }
    }
    Ok(())
}

/// Restores linear CELT coefficients from normalized shapes and decoded log
/// energies. Bands outside `start..end` are cleared.
pub fn denormalize_log_bands(
    normalized: &[f32],
    lm: u8,
    start: usize,
    end: usize,
    log_energies: &[f32],
    output: &mut [f32],
) -> Result<(), Error> {
    let bins = band_edge(BAND_COUNT, lm).ok_or(Error::InvalidFrameSize)?;
    if start >= end
        || end > BAND_COUNT
        || normalized.len() < bins
        || output.len() < bins
        || log_energies.len() < BAND_COUNT
    {
        return Err(Error::BufferTooSmall);
    }
    output[..bins].fill(0.0);
    for band in start..end {
        let log_energy = log_energies[band];
        if !log_energy.is_finite() {
            return Err(Error::InvalidPacket);
        }
        // Capping the exponent bounds hostile inputs while remaining far above
        // any useful decoded audio level.
        let amplitude = mrml_math::pow(2.0, (log_energy + ENERGY_MEANS[band]).min(32.0));
        if !amplitude.is_finite() {
            return Err(Error::InvalidPacket);
        }
        for index in band_range(band, lm)? {
            output[index] = normalized[index] * amplitude;
        }
    }
    Ok(())
}

pub const fn tf_adjustment(lm: u8, transient: bool, select: bool, flag: bool) -> Option<i8> {
    if lm > 3 {
        return None;
    }
    let i = lm as usize;
    Some(match (transient, select, flag) {
        (false, false, false) | (false, true, false) => 0,
        (false, false, true) => [-1, -1, -2, -2][i],
        (false, true, true) => [-1, -2, -3, -3][i],
        (true, false, false) => [0, 1, 2, 3][i],
        (true, false, true) => [-1, 0, 0, 0][i],
        (true, true, false) => [0, 1, 1, 1][i],
        (true, true, true) => -1,
    })
}

pub fn decode_tf_flags(
    decoder: &mut RangeDecoder<'_>,
    transient: bool,
    flags: &mut [bool],
) -> Result<(), Error> {
    if flags.is_empty() || flags.len() > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    let mut previous = false;
    for (index, flag) in flags.iter_mut().enumerate() {
        let logp = if index == 0 {
            if transient { 2 } else { 4 }
        } else if transient {
            4
        } else {
            5
        };
        previous ^= decoder.decode_bit_logp(logp)?;
        *flag = previous;
    }
    Ok(())
}

pub fn encode_tf_flags(
    encoder: &mut RangeEncoder<'_>,
    transient: bool,
    flags: &[bool],
) -> Result<(), Error> {
    if flags.is_empty() || flags.len() > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    let mut previous = false;
    for (index, &flag) in flags.iter().enumerate() {
        let logp = if index == 0 {
            if transient { 2 } else { 4 }
        } else if transient {
            4
        } else {
            5
        };
        encoder.encode_bit_logp(flag ^ previous, logp)?;
        previous = flag;
    }
    Ok(())
}

fn tf_select_needed(lm: u8, transient: bool, flags: &[bool]) -> Result<bool, Error> {
    if lm > 3 || flags.is_empty() || flags.len() > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    Ok(flags.iter().any(|&flag| {
        tf_adjustment(lm, transient, false, flag) != tf_adjustment(lm, transient, true, flag)
    }))
}

/// Decodes per-band resolution changes and the conditional `tf_select` bit.
pub fn decode_tf_resolution(
    decoder: &mut RangeDecoder<'_>,
    lm: u8,
    transient: bool,
    adjustments: &mut [i8],
) -> Result<bool, Error> {
    decode_tf_resolution_bounded(decoder, lm, transient, adjustments, u32::MAX)
}

pub(crate) fn decode_tf_resolution_bounded(
    decoder: &mut RangeDecoder<'_>,
    lm: u8,
    transient: bool,
    adjustments: &mut [i8],
    budget_bits: u32,
) -> Result<bool, Error> {
    if adjustments.is_empty() || adjustments.len() > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    let mut flags = [false; BAND_COUNT];
    let mut previous = false;
    for (index, flag) in flags[..adjustments.len()].iter_mut().enumerate() {
        let logp = if index == 0 {
            if transient { 2 } else { 4 }
        } else if transient {
            4
        } else {
            5
        };
        if decoder.tell().saturating_add(u32::from(logp)) <= budget_bits {
            previous ^= decoder.decode_bit_logp(logp)?;
        }
        *flag = previous;
    }
    let needed = tf_select_needed(lm, transient, &flags[..adjustments.len()])?;
    let select =
        needed && decoder.tell().saturating_add(1) <= budget_bits && decoder.decode_bit_logp(1)?;
    for (output, &flag) in adjustments.iter_mut().zip(&flags) {
        *output = tf_adjustment(lm, transient, select, flag).ok_or(Error::InvalidFrameSize)?;
    }
    Ok(select)
}

/// Encoder mirror of [`decode_tf_resolution`].
pub fn encode_tf_resolution(
    encoder: &mut RangeEncoder<'_>,
    lm: u8,
    transient: bool,
    flags: &[bool],
    select: bool,
) -> Result<(), Error> {
    encode_tf_resolution_bounded(encoder, lm, transient, flags, select, u32::MAX)
}

pub(crate) fn encode_tf_resolution_bounded(
    encoder: &mut RangeEncoder<'_>,
    lm: u8,
    transient: bool,
    flags: &[bool],
    select: bool,
    budget_bits: u32,
) -> Result<(), Error> {
    if flags.is_empty() || flags.len() > BAND_COUNT {
        return Err(Error::InvalidFrameSize);
    }
    let mut previous = false;
    for (index, &flag) in flags.iter().enumerate() {
        let logp = if index == 0 {
            if transient { 2 } else { 4 }
        } else if transient {
            4
        } else {
            5
        };
        if encoder.tell().saturating_add(u32::from(logp)) <= budget_bits {
            encoder.encode_bit_logp(flag ^ previous, logp)?;
            previous = flag;
        } else if flag != previous {
            return Err(Error::InvalidPacket);
        }
    }
    let needed = tf_select_needed(lm, transient, flags)?;
    if !needed && select {
        return Err(Error::InvalidPacket);
    }
    if needed && encoder.tell().saturating_add(1) <= budget_bits {
        encoder.encode_bit_logp(select, 1)?;
    } else if needed && select {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

pub fn hadamard(data: &mut [f32], levels: u8) -> Result<(), Error> {
    if levels as usize >= usize::BITS as usize {
        return Err(Error::InvalidFrameSize);
    }
    let size = 1usize << levels;
    if !data.len().is_multiple_of(size) {
        return Err(Error::InvalidFrameSize);
    }
    let scale = core::f32::consts::FRAC_1_SQRT_2;
    for chunk in data.chunks_exact_mut(size) {
        let mut step = 1usize;
        while step < size {
            for base in (0..size).step_by(step * 2) {
                for offset in 0..step {
                    let left = chunk[base + offset];
                    let right = chunk[base + offset + step];
                    chunk[base + offset] = (left + right) * scale;
                    chunk[base + offset + step] = (left - right) * scale;
                }
            }
            step *= 2;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TfLayout {
    pub partition_blocks: usize,
    recombine: usize,
    time_divide: usize,
    block_width: usize,
    long_blocks: bool,
}

/// Computes the block geometry used by CELT partition decoding after a
/// per-band time/frequency adjustment.
pub fn tf_layout(
    dimensions: usize,
    initial_blocks: usize,
    adjustment: i8,
) -> Result<TfLayout, Error> {
    if dimensions == 0
        || initial_blocks == 0
        || initial_blocks > 8
        || !initial_blocks.is_power_of_two()
        || !dimensions.is_multiple_of(initial_blocks)
    {
        return Err(Error::InvalidFrameSize);
    }
    let long_blocks = initial_blocks == 1;
    let recombine = usize::try_from(adjustment.max(0)).map_err(|_| Error::InvalidFrameSize)?;
    if recombine >= usize::BITS as usize || initial_blocks < 1usize << recombine {
        return Err(Error::InvalidFrameSize);
    }
    let mut blocks = initial_blocks >> recombine;
    let mut block_width = (dimensions / initial_blocks) << recombine;
    let mut change = adjustment;
    let mut time_divide = 0;
    while block_width.is_multiple_of(2) && change < 0 {
        blocks = blocks.checked_mul(2).ok_or(Error::InvalidFrameSize)?;
        block_width >>= 1;
        time_divide += 1;
        change += 1;
    }
    if blocks > 16 || !dimensions.is_multiple_of(blocks) {
        return Err(Error::InvalidFrameSize);
    }
    Ok(TfLayout {
        partition_blocks: blocks,
        recombine,
        time_divide,
        block_width,
        long_blocks,
    })
}

/// Restores frequency ordering after PVQ decoding with a TF adjustment and
/// maps its collapse bits back to the frame's original short blocks.
pub fn restore_tf_resolution(
    vector: &mut [f32],
    scratch: &mut [f32],
    initial_blocks: usize,
    adjustment: i8,
    mut collapse_mask: u16,
) -> Result<u8, Error> {
    if scratch.len() < vector.len() {
        return Err(Error::BufferTooSmall);
    }
    let layout = tf_layout(vector.len(), initial_blocks, adjustment)?;
    let mut blocks = layout.partition_blocks;
    let mut block_width = layout.block_width;
    if blocks > 1 {
        interleave_hadamard(
            vector,
            scratch,
            block_width >> layout.recombine,
            blocks << layout.recombine,
            layout.long_blocks,
        )?;
    }
    for _ in 0..layout.time_divide {
        blocks >>= 1;
        block_width <<= 1;
        collapse_mask |= collapse_mask >> blocks;
        haar1(vector, block_width, blocks)?;
    }
    const MASK_DEINTERLEAVE: [u16; 16] = [
        0x00, 0x03, 0x0c, 0x0f, 0x30, 0x33, 0x3c, 0x3f, 0xc0, 0xc3, 0xcc, 0xcf, 0xf0, 0xf3, 0xfc,
        0xff,
    ];
    for level in 0..layout.recombine {
        collapse_mask = MASK_DEINTERLEAVE[usize::from(collapse_mask & 0x0f)];
        haar1(vector, vector.len() >> level, 1 << level)?;
    }
    blocks <<= layout.recombine;
    let mask = (1u16 << blocks) - 1;
    Ok((collapse_mask & mask) as u8)
}

/// Converts frequency-ordered coefficients into the partition layout consumed
/// by CELT PVQ. This is the exact inverse of [`restore_tf_resolution`] for the
/// coefficient vector; collapse masks are produced after partition coding.
pub fn prepare_tf_resolution(
    vector: &mut [f32],
    scratch: &mut [f32],
    initial_blocks: usize,
    adjustment: i8,
) -> Result<TfLayout, Error> {
    if scratch.len() < vector.len() {
        return Err(Error::BufferTooSmall);
    }
    let layout = tf_layout(vector.len(), initial_blocks, adjustment)?;
    for level in (0..layout.recombine).rev() {
        haar1(vector, vector.len() >> level, 1 << level)?;
    }
    let mut blocks = layout.partition_blocks >> layout.time_divide;
    let mut block_width = layout.block_width << layout.time_divide;
    for _ in 0..layout.time_divide {
        haar1(vector, block_width, blocks)?;
        blocks <<= 1;
        block_width >>= 1;
    }
    if layout.partition_blocks > 1 {
        deinterleave_hadamard(
            vector,
            scratch,
            layout.block_width >> layout.recombine,
            layout.partition_blocks << layout.recombine,
            layout.long_blocks,
        )?;
    }
    Ok(layout)
}

fn haar1(vector: &mut [f32], length: usize, stride: usize) -> Result<(), Error> {
    if length == 0 || stride == 0 || length * stride > vector.len() || !length.is_multiple_of(2) {
        return Err(Error::InvalidFrameSize);
    }
    let pairs = length / 2;
    for offset in 0..stride {
        for pair in 0..pairs {
            let first = stride * 2 * pair + offset;
            let second = first + stride;
            let left = core::f32::consts::FRAC_1_SQRT_2 * vector[first];
            let right = core::f32::consts::FRAC_1_SQRT_2 * vector[second];
            vector[first] = left + right;
            vector[second] = left - right;
        }
    }
    Ok(())
}

fn interleave_hadamard(
    vector: &mut [f32],
    scratch: &mut [f32],
    block_width: usize,
    stride: usize,
    hadamard_order: bool,
) -> Result<(), Error> {
    const ORDER: [usize; 30] = [
        1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2,
        10, 5,
    ];
    let length = block_width
        .checked_mul(stride)
        .ok_or(Error::InvalidFrameSize)?;
    if length != vector.len() || scratch.len() < length || !stride.is_power_of_two() || stride > 16
    {
        return Err(Error::InvalidFrameSize);
    }
    let order = if hadamard_order {
        ORDER
            .get(stride - 2..stride - 2 + stride)
            .ok_or(Error::InvalidFrameSize)?
    } else {
        &[]
    };
    for block in 0..stride {
        let source_block = if hadamard_order { order[block] } else { block };
        for coefficient in 0..block_width {
            scratch[coefficient * stride + block] =
                vector[source_block * block_width + coefficient];
        }
    }
    vector.copy_from_slice(&scratch[..length]);
    Ok(())
}

fn deinterleave_hadamard(
    vector: &mut [f32],
    scratch: &mut [f32],
    block_width: usize,
    stride: usize,
    hadamard_order: bool,
) -> Result<(), Error> {
    const ORDER: [usize; 30] = [
        1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2,
        10, 5,
    ];
    let length = block_width
        .checked_mul(stride)
        .ok_or(Error::InvalidFrameSize)?;
    if length != vector.len() || scratch.len() < length || !stride.is_power_of_two() || stride > 16
    {
        return Err(Error::InvalidFrameSize);
    }
    let order = if hadamard_order {
        ORDER
            .get(stride - 2..stride - 2 + stride)
            .ok_or(Error::InvalidFrameSize)?
    } else {
        &[]
    };
    for block in 0..stride {
        let target_block = if hadamard_order { order[block] } else { block };
        for coefficient in 0..block_width {
            scratch[target_block * block_width + coefficient] =
                vector[coefficient * stride + block];
        }
    }
    vector.copy_from_slice(&scratch[..length]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn band_geometry_matches_rfc_table_55() {
        let expected = [
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6, 8, 12, 18, 22,
        ];
        for (band, width) in expected.into_iter().enumerate() {
            assert_eq!(band_range(band, 0).unwrap().len(), width);
            assert_eq!(band_range(band, 3).unwrap().len(), width * 8);
        }
    }
    #[test]
    fn normalization_round_trips() {
        let mut input = [0.0f32; 100];
        for (index, value) in input.iter_mut().enumerate() {
            *value = index as f32 - 20.0;
        }
        let mut amplitudes = [0.0; BAND_COUNT];
        let mut normalized = [0.0; 100];
        normalize_bands(&input, 0, &mut amplitudes, &mut normalized).unwrap();
        let mut output = [0.0; 100];
        denormalize_bands(&normalized, 0, &amplitudes, &mut output).unwrap();
        for (actual, expected) in output.iter().zip(input) {
            assert!((*actual - expected).abs() < 2e-5);
        }
    }
    #[test]
    fn log_denormalization_uses_band_means_and_clears_uncoded_bins() {
        let normalized = [1.0f32; 100];
        let mut log_energies = [0.0f32; BAND_COUNT];
        log_energies[1] = 1.0;
        let mut output = [9.0f32; 100];
        denormalize_log_bands(&normalized, 0, 1, 2, &log_energies, &mut output).unwrap();
        assert_eq!(output[0], 0.0);
        assert!((output[1] - mrml_math::pow(2.0, 7.25)).abs() < 1e-3);
        assert!(output[2..].iter().all(|&value| value == 0.0));
        log_energies[1] = f32::NAN;
        assert_eq!(
            denormalize_log_bands(&normalized, 0, 1, 2, &log_energies, &mut output),
            Err(Error::InvalidPacket)
        );
    }
    #[test]
    fn tf_tables_cover_rfc_combinations() {
        assert_eq!(tf_adjustment(3, false, false, true), Some(-2));
        assert_eq!(tf_adjustment(3, false, true, true), Some(-3));
        assert_eq!(tf_adjustment(3, true, false, false), Some(3));
        assert_eq!(tf_adjustment(3, true, true, true), Some(-1));
        assert_eq!(tf_adjustment(4, true, true, true), None);
    }
    #[test]
    fn time_division_folds_sixteen_partition_collapse_bits_to_eight_blocks() {
        let mut vector = [0.0; 16];
        let mut scratch = [0.0; 16];
        assert_eq!(
            restore_tf_resolution(&mut vector, &mut scratch, 8, -1, 0x8001),
            Ok(0x81)
        );
    }
    #[test]
    fn tf_flags_stop_when_the_symbol_would_exceed_the_frame_budget() {
        let bytes = [0x55; 8];
        let mut decoder = RangeDecoder::new(&bytes);
        let initial_range = decoder.range();
        let mut adjustments = [9; 3];
        let budget = decoder.tell() + 3;
        assert_eq!(
            decode_tf_resolution_bounded(&mut decoder, 1, false, &mut adjustments, budget),
            Ok(false)
        );
        assert_eq!(adjustments, [0; 3]);
        assert_eq!(decoder.range(), initial_range);

        let mut encoded = [0u8; 8];
        let mut encoder = RangeEncoder::new(&mut encoded);
        let initial_range = encoder.range();
        let budget = encoder.tell() + 3;
        assert_eq!(
            encode_tf_resolution_bounded(&mut encoder, 1, false, &[false; 3], false, budget,),
            Ok(())
        );
        assert_eq!(encoder.range(), initial_range);

        let mut encoded = [0u8; 8];
        let mut encoder = RangeEncoder::new(&mut encoded);
        let budget = encoder.tell() + 4;
        encode_tf_resolution_bounded(&mut encoder, 1, false, &[true; 3], false, budget).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&encoded);
        let mut carried = [0; 3];
        decode_tf_resolution_bounded(&mut decoder, 1, false, &mut carried, budget).unwrap();
        assert_eq!(carried, [-1; 3]);
    }
    #[test]
    fn hadamard_is_self_inverse() {
        let original = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let mut transformed = original;
        hadamard(&mut transformed, 3).unwrap();
        let before: f32 = original.iter().map(|v| v * v).sum();
        let after: f32 = transformed.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
        hadamard(&mut transformed, 3).unwrap();
        for (actual, expected) in transformed.iter().zip(original) {
            assert!((*actual - expected).abs() < 2e-6);
        }
    }
    #[test]
    fn tf_flags_round_trip() {
        let flags = [false, true, true, false, true];
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_tf_flags(&mut encoder, true, &flags).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let mut decoded = [false; 5];
        decode_tf_flags(&mut decoder, true, &mut decoded).unwrap();
        assert_eq!(decoded, flags);
    }
    #[test]
    fn conditional_tf_select_round_trips_all_modes() {
        for lm in 0..=3 {
            for transient in [false, true] {
                for select in [false, true] {
                    let flags = [false, true, true, false, true];
                    if select && !tf_select_needed(lm, transient, &flags).unwrap() {
                        continue;
                    }
                    let mut bytes = [0; 16];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    encode_tf_resolution(&mut encoder, lm, transient, &flags, select).unwrap();
                    let encoded_tell = encoder.tell_frac();
                    encoder.finish().unwrap();
                    let mut decoder = RangeDecoder::new(&bytes);
                    let mut adjustments = [0; 5];
                    assert_eq!(
                        decode_tf_resolution(&mut decoder, lm, transient, &mut adjustments),
                        Ok(select)
                    );
                    assert_eq!(decoder.tell_frac(), encoded_tell);
                    for (index, &flag) in flags.iter().enumerate() {
                        assert_eq!(
                            adjustments[index],
                            tf_adjustment(lm, transient, select, flag).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tf_layout_tracks_frequency_and_time_resolution_changes() {
        assert_eq!(tf_layout(16, 4, 1).unwrap().partition_blocks, 2);
        assert_eq!(tf_layout(16, 4, -1).unwrap().partition_blocks, 8);
        assert_eq!(tf_layout(12, 4, -3).unwrap().partition_blocks, 4);
        assert_eq!(tf_layout(16, 1, 1), Err(Error::InvalidFrameSize));
    }

    #[test]
    fn restoring_tf_resolution_preserves_energy_and_maps_masks() {
        for adjustment in [-1, 1] {
            let mut vector = [0.0f32; 16];
            for (index, value) in vector.iter_mut().enumerate() {
                *value = index as f32 - 4.0;
            }
            let before = vector.iter().fold(0.0, |sum, value| sum + value * value);
            let mut scratch = [0.0; 16];
            let layout = tf_layout(16, 4, adjustment).unwrap();
            let input_mask = ((1u16 << layout.partition_blocks) - 1) as u8;
            let output_mask = restore_tf_resolution(
                &mut vector,
                &mut scratch,
                4,
                adjustment,
                u16::from(input_mask),
            )
            .unwrap();
            let after = vector.iter().fold(0.0, |sum, value| sum + value * value);
            assert!((before - after).abs() < 0.001);
            assert_eq!(output_mask, 0b1111);
        }
    }
    #[test]
    fn tf_preparation_and_restoration_are_exact_inverses() {
        for initial_blocks in [1usize, 2, 4, 8] {
            for adjustment in -3i8..=3 {
                let mut vector = [0.0f32; 96];
                for (index, value) in vector.iter_mut().enumerate() {
                    *value = index as f32 * 0.125 - 3.0;
                }
                let original = vector;
                let mut scratch = [0.0f32; 96];
                if prepare_tf_resolution(&mut vector, &mut scratch, initial_blocks, adjustment)
                    .is_err()
                {
                    continue;
                }
                restore_tf_resolution(&mut vector, &mut scratch, initial_blocks, adjustment, 0xff)
                    .unwrap();
                for (actual, expected) in vector.iter().zip(original) {
                    assert!((actual - expected).abs() < 2e-5);
                }
            }
        }
    }
}
