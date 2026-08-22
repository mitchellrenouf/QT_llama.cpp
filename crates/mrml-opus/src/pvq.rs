//! RFC 6716 CELT pyramid vector quantization.

use crate::{Error, RangeDecoder, RangeEncoder};

/// CELT's restricted pulse-count sequence. The index, rather than K itself,
/// is what the allocation cache searches.
pub const fn pulses_for_index(index: u8) -> Option<usize> {
    if index < 8 {
        Some(index as usize)
    } else {
        let shift = (index >> 3) - 1;
        if shift >= usize::BITS as u8 {
            None
        } else {
            (8usize + (index as usize & 7)).checked_shl(shift as u32)
        }
    }
}

/// Recurrence workspace bound for dimensions above four.
pub const MAX_PULSES: usize = 128;
/// Highest pulse-sequence index present in any CELT allocation-cache run.
/// Individual runs can end earlier when their Q3 cost no longer fits in the
/// cache's one-byte entries.
pub const MAX_PULSE_INDEX: u8 = 40;

// RFC 6716's standard 48 kHz CELT mode uses an LM-major lookup from
// `(LM + 1, band)` to one of 23 packed pulse-cost profiles.  The offsets are
// standards data; costs themselves remain generated from V(N,K) below.
const CACHE_INDEX: [i16; 105] = [
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222, 0, 0, 0, 0,
    0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266, 283, 295, 41, 41, 41, 41, 41,
    41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305, 318, 328, 336, 123, 123, 123,
    123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318, 343, 351, 358, 364, 240,
    240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376, 382,
    387,
];

const CACHE_PROFILES: [(i16, usize, u8); 23] = [
    (0, 1, 40),
    (41, 2, 40),
    (82, 3, 40),
    (123, 4, 40),
    (164, 6, 35),
    (200, 9, 21),
    (222, 11, 17),
    (240, 8, 25),
    (266, 12, 16),
    (283, 18, 11),
    (295, 22, 9),
    (305, 16, 12),
    (318, 24, 9),
    (328, 36, 7),
    (336, 44, 6),
    (343, 32, 7),
    (351, 48, 6),
    (358, 72, 5),
    (364, 88, 5),
    (370, 64, 5),
    (376, 96, 5),
    (382, 144, 4),
    (387, 176, 4),
];

fn cache_profile(band: usize, lm: i8) -> Result<Option<(usize, u8)>, Error> {
    if band >= 21 || !(-1..=3).contains(&lm) {
        return Err(Error::InvalidFrameSize);
    }
    let row = usize::try_from(lm + 1).map_err(|_| Error::InvalidFrameSize)?;
    let offset = CACHE_INDEX[row * 21 + band];
    if offset < 0 {
        return Ok(None);
    }
    CACHE_PROFILES
        .iter()
        .find(|profile| profile.0 == offset)
        .map(|profile| Some((profile.1, profile.2)))
        .ok_or(Error::InvalidPacket)
}

/// Number of permitted pulse-sequence entries in each standard CELT cache
/// run. These limits follow directly from the largest restricted pulse count
/// whose `V(N, K)` codebook remains below the 32-bit split threshold.
pub const fn pulse_cache_run_len(dimensions: usize) -> Option<u8> {
    match dimensions {
        1..=4 => Some(40),
        6 => Some(35),
        8 => Some(25),
        9 => Some(21),
        11 => Some(17),
        12 => Some(16),
        16 => Some(12),
        18 => Some(11),
        22 | 24 => Some(9),
        32 | 36 => Some(7),
        44 | 48 => Some(6),
        64 | 72 | 88 | 96 => Some(5),
        144 | 176 => Some(4),
        _ => None,
    }
}

fn pulse_index_limit(dimensions: usize) -> u8 {
    pulse_cache_run_len(dimensions).unwrap_or(MAX_PULSE_INDEX)
}

fn needs_recurrence_workspace(dimensions: usize) -> bool {
    dimensions > 4
}

/// Integer approximation used by CELT for log2 values, in 1/8-bit units.
pub fn fractional_log2(value: u32) -> Result<u16, Error> {
    if value == 0 {
        return Err(Error::InvalidPacket);
    }
    let integer_log = u32::BITS - value.leading_zeros();
    let mut normalized = if integer_log >= 16 {
        value >> (integer_log - 16)
    } else {
        value << (16 - integer_log)
    };
    let mut log = integer_log;
    for _ in 0..3 {
        normalized = ((u64::from(normalized) * u64::from(normalized)) >> 15) as u32;
        let bit = normalized >> 16;
        log = log * 2 + bit;
        normalized >>= bit;
    }
    u16::try_from(log - 8).map_err(|_| Error::InvalidPacket)
}

/// Conservative Q3 storage cost for a uniformly coded codebook.
pub fn codebook_cost(value: u32) -> Result<u16, Error> {
    let floor = fractional_log2(value)?;
    if value.is_power_of_two() {
        Ok(floor)
    } else {
        floor.checked_add(1).ok_or(Error::InvalidPacket)
    }
}

/// Conservative Q3 logarithm used to generate CELT pulse-cache entries.
///
/// This is kept distinct from [`codebook_cost`]: recursive band splitting
/// needs the maximum from its applicable cache run, so substituting this
/// value into the unrestricted path changes the bitstream grammar.
pub fn pulse_cache_cost(value: u32) -> Result<u16, Error> {
    if value == 0 {
        return Err(Error::InvalidPacket);
    }
    let integer_log = u32::BITS - value.leading_zeros();
    if value.is_power_of_two() {
        return u16::try_from((integer_log - 1) << 3).map_err(|_| Error::InvalidPacket);
    }
    let mut normalized = if integer_log > 16 {
        let shift = integer_log - 16;
        (value >> shift) + u32::from(value & ((1 << shift) - 1) != 0)
    } else {
        value << (16 - integer_log)
    };
    let mut result = (integer_log - 1) << 3;
    for fractional_bit in (0..=3).rev() {
        let carry = normalized >> 16;
        result += carry << fractional_bit;
        normalized = (normalized + carry) >> carry;
        normalized = ((u64::from(normalized) * u64::from(normalized) + 0x7fff) >> 15) as u32;
    }
    result += u32::from(normalized > 0x8000);
    u16::try_from(result).map_err(|_| Error::InvalidPacket)
}

/// One-byte representation used by CELT's packed allocation cache.
pub fn packed_pulse_cache_cost(value: u32) -> Result<u8, Error> {
    Ok(pulse_cache_cost(value)?
        .saturating_sub(1)
        .min(u16::from(u8::MAX)) as u8)
}

/// Finds the largest allowed CELT pulse codebook whose fractional cost does
/// not exceed the supplied shape allocation.
pub fn pulses_for_allocation(
    dimensions: usize,
    allocation_eighth_bits: u16,
    scratch: &mut [u32],
) -> Result<usize, Error> {
    if dimensions == 0 || scratch.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    let mut selected = 0;
    for index in 1..=pulse_index_limit(dimensions) {
        let Some(pulses) = pulses_for_index(index) else {
            break;
        };
        if needs_recurrence_workspace(dimensions)
            && (pulses > MAX_PULSES || pulses >= scratch.len())
        {
            break;
        }
        let size = codebook_size(dimensions, pulses, scratch)?;
        if size == u32::MAX || u16::from(packed_pulse_cache_cost(size)?) > allocation_eighth_bits {
            break;
        }
        selected = pulses;
    }
    Ok(selected)
}

/// Selects the permitted pulse count whose codebook cost is nearest to the
/// requested allocation, rounding downward on a tie and never exceeding the
/// capacity still available in the frame.
pub fn pulses_for_target(
    dimensions: usize,
    target_eighth_bits: u16,
    available_eighth_bits: u16,
    scratch: &mut [u32],
) -> Result<usize, Error> {
    pulses_for_profile_target(
        dimensions,
        pulse_index_limit(dimensions),
        target_eighth_bits,
        available_eighth_bits,
        scratch,
    )
}

/// Selects pulses using the normative cache profile for one CELT band while
/// leaving the actual PVQ codebook dimension independent.
pub fn pulses_for_band_target(
    band: usize,
    lm: i8,
    target_eighth_bits: u16,
    available_eighth_bits: u16,
    scratch: &mut [u32],
) -> Result<usize, Error> {
    let (profile_dimensions, limit) = cache_profile(band, lm)?.ok_or(Error::InvalidFrameSize)?;
    pulses_for_profile_target(
        profile_dimensions,
        limit,
        target_eighth_bits,
        available_eighth_bits,
        scratch,
    )
}

fn pulses_for_profile_target(
    profile_dimensions: usize,
    limit: u8,
    target_eighth_bits: u16,
    available_eighth_bits: u16,
    scratch: &mut [u32],
) -> Result<usize, Error> {
    if profile_dimensions == 0 || scratch.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    // CELT compares `bits - 1` against the packed Q3 cache. Entry zero is an
    // implicit -1 rather than a zero-cost cache byte.
    let target = i32::from(target_eighth_bits) - 1;
    let mut lower_index = 0u8;
    let mut lower_cost = -1i32;
    let mut upper = None;
    for index in 1..=limit {
        let Some(pulses) = pulses_for_index(index) else {
            break;
        };
        if needs_recurrence_workspace(profile_dimensions)
            && (pulses > MAX_PULSES || pulses >= scratch.len())
        {
            break;
        }
        let size = codebook_size(profile_dimensions, pulses, scratch)?;
        if size == u32::MAX {
            break;
        }
        let packed_cost = u16::from(packed_pulse_cache_cost(size)?);
        let cost = i32::from(packed_cost);
        if cost >= target {
            upper = Some((index, cost));
            break;
        }
        lower_index = index;
        lower_cost = cost;
    }
    let mut selected_index = match upper {
        Some((upper_index, upper_cost))
            if (target - lower_cost).unsigned_abs() > (upper_cost - target).unsigned_abs() =>
        {
            upper_index
        }
        _ => lower_index,
    };
    while selected_index != 0 {
        let pulses = pulses_for_index(selected_index).ok_or(Error::InvalidPacket)?;
        let size = codebook_size(profile_dimensions, pulses, scratch)?;
        let cost = u16::from(packed_pulse_cache_cost(size)?) + 1;
        if cost <= available_eighth_bits {
            break;
        }
        selected_index -= 1;
    }
    Ok(pulses_for_index(selected_index).unwrap_or(0))
}

/// Returns the normative cached cost for a pulse sequence in one band.
pub fn band_pulse_cost(
    band: usize,
    lm: i8,
    pulses: usize,
    scratch: &mut [u32],
) -> Result<u16, Error> {
    let (profile_dimensions, limit) = cache_profile(band, lm)?.ok_or(Error::InvalidFrameSize)?;
    let index = (0..=limit)
        .find(|&index| pulses_for_index(index) == Some(pulses))
        .ok_or(Error::InvalidPacket)?;
    if index == 0 {
        return Ok(0);
    }
    let size = codebook_size(profile_dimensions, pulses, scratch)?;
    Ok(u16::from(packed_pulse_cache_cost(size)?) + 1)
}

/// Highest packed Q3 cost in one normative band/LM cache run.
pub fn maximum_band_cost(band: usize, lm: i8, scratch: &mut [u32]) -> Result<u16, Error> {
    let (profile_dimensions, limit) = cache_profile(band, lm)?.ok_or(Error::InvalidFrameSize)?;
    let pulses = pulses_for_index(limit).ok_or(Error::InvalidPacket)?;
    let size = codebook_size(profile_dimensions, pulses, scratch)?;
    packed_pulse_cache_cost(size).map(u16::from)
}

/// Decodes one CELT PVQ codebook using the shared range coder.
pub fn decode_range(
    decoder: &mut RangeDecoder<'_>,
    pulses: usize,
    spread_mode: u8,
    pulse_vector: &mut [i32],
    normalized: &mut [f32],
    scratch: &mut [u32],
) -> Result<(), Error> {
    if pulse_vector.len() != normalized.len() || pulse_vector.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    if pulses == 0 {
        pulse_vector.fill(0);
        normalized.fill(0.0);
        return Ok(());
    }
    let size = codebook_size(pulse_vector.len(), pulses, scratch)?;
    if size == u32::MAX {
        return Err(Error::InvalidPacket);
    }
    let index = decoder.decode_uint(size)?;
    decode(index, pulses, pulse_vector, scratch)?;
    normalize(pulse_vector, normalized)?;
    spread(normalized, pulses, spread_mode)
}

/// Encoder mirror of [`decode_range`].
pub fn encode_range(
    encoder: &mut RangeEncoder<'_>,
    pulses: usize,
    pulse_vector: &[i32],
    scratch: &mut [u32],
) -> Result<(), Error> {
    if pulse_vector.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    if pulses == 0 {
        return if pulse_vector.iter().all(|&value| value == 0) {
            Ok(())
        } else {
            Err(Error::InvalidPacket)
        };
    }
    let size = codebook_size(pulse_vector.len(), pulses, scratch)?;
    if size == u32::MAX {
        return Err(Error::InvalidPacket);
    }
    let index = encode(pulse_vector, pulses, scratch)?;
    encoder.encode_uint(index, size)
}

/// Computes V(N,K), saturating when the RFC's 32-bit codebook limit is exceeded.
/// `scratch` must contain at least `pulses + 1` entries.
pub fn codebook_size(dimensions: usize, pulses: usize, scratch: &mut [u32]) -> Result<u32, Error> {
    if dimensions <= 4 {
        let k = pulses as u128;
        let value = match (dimensions, pulses) {
            (_, 0) => 1,
            (0, _) => 0,
            (1, _) => 2,
            (2, _) => 4 * k,
            (3, _) => 4 * k * k + 2,
            (4, _) => (8 * k * k * k + 16 * k) / 3,
            _ => unreachable!(),
        };
        return Ok(u32::try_from(value).unwrap_or(u32::MAX));
    }
    if scratch.len() <= pulses {
        return Err(Error::BufferTooSmall);
    }
    scratch[..=pulses].fill(0);
    scratch[0] = 1;
    for _ in 0..dimensions {
        let mut diagonal = scratch[0];
        for pulse in 1..=pulses {
            let above = scratch[pulse];
            scratch[pulse] = u32::try_from(
                (u64::from(above) + u64::from(scratch[pulse - 1]) + u64::from(diagonal))
                    .min(u64::from(u32::MAX)),
            )
            .unwrap_or(u32::MAX);
            diagonal = above;
        }
    }
    Ok(scratch[pulses])
}

/// Decodes one RFC PVQ codeword index into a signed pulse vector.
pub fn decode(
    index: u32,
    pulses: usize,
    output: &mut [i32],
    scratch: &mut [u32],
) -> Result<(), Error> {
    if output.is_empty() || (needs_recurrence_workspace(output.len()) && scratch.len() <= pulses) {
        return Err(Error::BufferTooSmall);
    }
    let total = codebook_size(output.len(), pulses, scratch)?;
    if index >= total || total == u32::MAX {
        return Err(Error::InvalidPacket);
    }
    let mut index = u64::from(index);
    let mut remaining = pulses;
    let dimensions = output.len();
    for (position, value) in output.iter_mut().enumerate() {
        let n = dimensions - position;
        let left = u64::from(codebook_size(n - 1, remaining, scratch)?);
        let whole = u64::from(codebook_size(n, remaining, scratch)?);
        let midpoint = (left + whole) / 2;
        let sign = if index < midpoint {
            1
        } else {
            index -= midpoint;
            -1
        };
        let original = remaining;
        let mut boundary = midpoint.checked_sub(left).ok_or(Error::InvalidPacket)?;
        while boundary > index {
            remaining = remaining.checked_sub(1).ok_or(Error::InvalidPacket)?;
            boundary = boundary
                .checked_sub(u64::from(codebook_size(n - 1, remaining, scratch)?))
                .ok_or(Error::InvalidPacket)?;
        }
        *value = sign * i32::try_from(original - remaining).map_err(|_| Error::InvalidPacket)?;
        index -= boundary;
    }
    if remaining != 0 || index != 0 {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

/// Computes the RFC PVQ codeword index for a signed pulse vector.
pub fn encode(vector: &[i32], pulses: usize, scratch: &mut [u32]) -> Result<u32, Error> {
    if vector.is_empty() || (needs_recurrence_workspace(vector.len()) && scratch.len() <= pulses) {
        return Err(Error::BufferTooSmall);
    }
    let actual = vector
        .iter()
        .try_fold(0usize, |sum, value| {
            sum.checked_add(value.unsigned_abs() as usize)
        })
        .ok_or(Error::InvalidPacket)?;
    if actual != pulses {
        return Err(Error::InvalidPacket);
    }
    let mut index = 0u64;
    let mut remaining = pulses;
    let dimensions = vector.len();
    for (position, &value) in vector.iter().enumerate() {
        let n = dimensions - position;
        let left = u64::from(codebook_size(n - 1, remaining, scratch)?);
        let whole = u64::from(codebook_size(n, remaining, scratch)?);
        let midpoint = (left + whole) / 2;
        if value < 0 {
            index = index.checked_add(midpoint).ok_or(Error::InvalidPacket)?;
        }
        let magnitude = value.unsigned_abs() as usize;
        let next = remaining
            .checked_sub(magnitude)
            .ok_or(Error::InvalidPacket)?;
        let mut boundary = midpoint.checked_sub(left).ok_or(Error::InvalidPacket)?;
        for k in (next..remaining).rev() {
            boundary = boundary
                .checked_sub(u64::from(codebook_size(n - 1, k, scratch)?))
                .ok_or(Error::InvalidPacket)?;
        }
        index = index.checked_add(boundary).ok_or(Error::InvalidPacket)?;
        remaining = next;
    }
    let total = codebook_size(dimensions, pulses, scratch)?;
    if remaining != 0 || index >= u64::from(total) {
        return Err(Error::InvalidPacket);
    }
    u32::try_from(index).map_err(|_| Error::InvalidPacket)
}

/// Converts a signed pulse vector to unit L2 norm.
pub fn normalize(pulses: &[i32], output: &mut [f32]) -> Result<(), Error> {
    if pulses.len() != output.len() || pulses.is_empty() {
        return Err(Error::InvalidFrameSize);
    }
    let energy = pulses.iter().fold(0u64, |sum, &value| {
        sum + u64::from(value.unsigned_abs()).pow(2)
    });
    if energy == 0 {
        return Err(Error::InvalidPacket);
    }
    let scale = 1.0 / mrml_math::sqrt(energy as f32);
    for (output, &pulse) in output.iter_mut().zip(pulses) {
        *output = pulse as f32 * scale;
    }
    Ok(())
}

/// Applies the RFC spreading rotation in-place.
pub fn spread(vector: &mut [f32], pulses: usize, spread: u8) -> Result<(), Error> {
    spread_blocks(vector, pulses, spread, 1)
}

/// Applies CELT spreading for one or more short time blocks, including the
/// interleaved pre-rotation required for blocks of at least eight samples.
pub fn spread_blocks(
    vector: &mut [f32],
    pulses: usize,
    spread: u8,
    blocks: usize,
) -> Result<(), Error> {
    if vector.len() < 2 || spread > 3 {
        return Err(Error::InvalidFrameSize);
    }
    if blocks == 0 || !vector.len().is_multiple_of(blocks) {
        return Err(Error::InvalidFrameSize);
    }
    if spread == 0 {
        return Ok(());
    }
    let factor = [0.0, 15.0, 10.0, 5.0][usize::from(spread)];
    let gain = vector.len() as f32 / (vector.len() as f32 + factor * pulses as f32);
    let angle = core::f32::consts::PI * gain * gain * 0.25;
    let block_size = vector.len() / blocks;
    if blocks > 1 && block_size >= 8 {
        let stride = rounded_sqrt(block_size).max(1);
        let extra = core::f32::consts::FRAC_PI_2 - angle;
        let cosine = mrml_math::cos(extra);
        let sine = mrml_math::sin(extra);
        for block in 0..blocks {
            let base = block * block_size;
            for offset in 0..stride.min(block_size) {
                rotate_strided_sequence(
                    vector,
                    base + offset,
                    block_size.saturating_sub(offset).div_ceil(stride),
                    stride,
                    cosine,
                    sine,
                );
            }
        }
    }
    let cosine = mrml_math::cos(angle);
    let sine = mrml_math::sin(angle);
    for block in vector.chunks_exact_mut(block_size) {
        rotate_strided_sequence(block, 0, block.len(), 1, cosine, sine);
    }
    Ok(())
}

fn rounded_sqrt(value: usize) -> usize {
    let mut root = 0usize;
    while (root + 1).saturating_mul(root + 1) <= value {
        root += 1;
    }
    let lower = root * root;
    let upper = (root + 1).saturating_mul(root + 1);
    if value - lower >= upper.saturating_sub(value) {
        root + 1
    } else {
        root
    }
}

fn rotate_strided_sequence(
    vector: &mut [f32],
    start: usize,
    count: usize,
    stride: usize,
    cosine: f32,
    sine: f32,
) {
    if count < 2 {
        return;
    }
    for index in 0..count - 1 {
        let left = start + index * stride;
        rotate(vector, left, left + stride, cosine, sine);
    }
    for index in (0..count - 2).rev() {
        let left = start + index * stride;
        rotate(vector, left, left + stride, cosine, sine);
    }
}

fn rotate(vector: &mut [f32], left: usize, right: usize, cosine: f32, sine: f32) {
    let a = vector[left];
    let b = vector[right];
    vector[left] = cosine * a + sine * b;
    vector[right] = -sine * a + cosine * b;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrence_matches_small_known_codebooks() {
        let mut scratch = [0; 17];
        assert_eq!(codebook_size(1, 4, &mut scratch), Ok(2));
        assert_eq!(codebook_size(2, 1, &mut scratch), Ok(4));
        assert_eq!(codebook_size(2, 2, &mut scratch), Ok(8));
        assert_eq!(codebook_size(3, 2, &mut scratch), Ok(18));
        assert_eq!(codebook_size(4, 64, &mut scratch), Ok(699_392));
        assert_eq!(codebook_size(4, 160, &mut scratch), Ok(10_923_520));
    }

    #[test]
    fn every_small_codeword_round_trips() {
        let mut scratch = [0; 17];
        let mut vector = [0i32; 8];
        for dimensions in 1..=8 {
            for pulses in 0..=8 {
                let count = codebook_size(dimensions, pulses, &mut scratch).unwrap();
                for index in 0..count {
                    decode(index, pulses, &mut vector[..dimensions], &mut scratch).unwrap();
                    assert_eq!(
                        encode(&vector[..dimensions], pulses, &mut scratch),
                        Ok(index)
                    );
                }
            }
        }
    }

    #[test]
    fn sampled_large_codewords_round_trip() {
        let mut scratch = [0; MAX_PULSES + 1];
        let mut vector = [0i32; 32];
        for dimensions in 5..=32 {
            for pulses in [1, 2, 4, 8, 16, 32, 64, 128] {
                let count = codebook_size(dimensions, pulses, &mut scratch).unwrap();
                if count == u32::MAX {
                    continue;
                }
                for index in [0, count / 7, count / 3, count / 2, count - 1] {
                    decode(index, pulses, &mut vector[..dimensions], &mut scratch).unwrap();
                    assert_eq!(
                        encode(&vector[..dimensions], pulses, &mut scratch),
                        Ok(index),
                        "N={dimensions} K={pulses} index={index} count={count}",
                    );
                }
            }
        }
    }

    #[test]
    fn normalization_and_rotation_preserve_energy() {
        let mut normalized = [0.0; 4];
        normalize(&[1, -2, 0, 3], &mut normalized).unwrap();
        let before: f32 = normalized.iter().map(|value| value * value).sum();
        spread(&mut normalized, 6, 2).unwrap();
        let after: f32 = normalized.iter().map(|value| value * value).sum();
        assert!((before - 1.0).abs() < 1e-6);
        assert!((after - before).abs() < 2e-6);
    }

    #[test]
    fn multi_block_spreading_preserves_energy_and_blocks() {
        let mut vector = [0.0; 32];
        for (index, value) in vector.iter_mut().enumerate() {
            *value = (index as f32 - 15.0) / 32.0;
        }
        let before: f32 = vector.iter().map(|value| value * value).sum();
        spread_blocks(&mut vector, 12, 3, 4).unwrap();
        let after: f32 = vector.iter().map(|value| value * value).sum();
        assert!((after - before).abs() < 2e-5);
        assert!(vector.iter().all(|value| value.is_finite()));
        assert_eq!(rounded_sqrt(7), 3);
        assert_eq!(rounded_sqrt(12), 3);
    }

    #[test]
    fn restricted_pulse_sequence_and_fractional_logs_match_boundaries() {
        let expected = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20,
        ];
        for (index, pulses) in expected.into_iter().enumerate() {
            assert_eq!(pulses_for_index(index as u8), Some(pulses));
        }
        for power in 0..=30 {
            assert_eq!(fractional_log2(1 << power), Ok(power * 8));
            assert_eq!(codebook_cost(1 << power), Ok(power * 8));
        }
        assert_eq!(fractional_log2(3), Ok(12));
        assert_eq!(codebook_cost(3), Ok(13));
        assert_eq!(pulse_cache_cost(3), Ok(13));
        assert_eq!(pulse_cache_cost(699_392), Ok(156));
        assert_eq!(pulse_cache_cost(4_196_289_420), Ok(256));
        assert_eq!(packed_pulse_cache_cost(2), Ok(7));
        assert_eq!(packed_pulse_cache_cost(4), Ok(15));
        assert_eq!(packed_pulse_cache_cost(699_392), Ok(155));
        assert_eq!(packed_pulse_cache_cost(4_196_289_420), Ok(255));
    }

    #[test]
    fn nearest_pulse_codebook_respects_remaining_capacity_and_ties() {
        let mut scratch = [0; MAX_PULSES + 1];
        let low = pulses_for_index(8).unwrap();
        let high = pulses_for_index(9).unwrap();
        let low_cost = codebook_cost(codebook_size(8, low, &mut scratch).unwrap()).unwrap();
        let high_cost = codebook_cost(codebook_size(8, high, &mut scratch).unwrap()).unwrap();
        assert!(high_cost > low_cost);
        assert_eq!(
            pulses_for_target(8, high_cost - 1, high_cost, &mut scratch),
            Ok(high)
        );
        assert_eq!(
            pulses_for_target(8, high_cost, high_cost - 1, &mut scratch),
            Ok(low)
        );
        let midpoint = low_cost + (high_cost - low_cost) / 2;
        assert_eq!(
            pulses_for_target(8, midpoint, high_cost, &mut scratch),
            Ok(low)
        );
    }

    #[test]
    fn packed_cache_bias_matches_a_high_band_boundary() {
        let mut scratch = [0; MAX_PULSES + 1];
        assert_eq!(
            pulses_for_target(12, 156, u16::MAX, &mut scratch),
            Ok(pulses_for_index(7).unwrap()),
        );
    }

    #[test]
    fn allocation_selection_is_restricted_and_monotonic() {
        let mut scratch = [0; 129];
        let mut previous = 0;
        for bits in 0..=200 {
            let pulses = pulses_for_allocation(8, bits, &mut scratch).unwrap();
            assert!(pulses >= previous);
            assert!((0..=u8::MAX).any(|index| pulses_for_index(index) == Some(pulses)));
            previous = pulses;
        }
    }

    #[test]
    fn low_dimension_cache_stops_at_the_normative_index_limit() {
        let mut scratch = [0; 1];
        let selected = pulses_for_allocation(2, u16::MAX, &mut scratch).unwrap();
        assert_eq!(selected, pulses_for_index(MAX_PULSE_INDEX).unwrap());
        assert!(pulses_for_index(MAX_PULSE_INDEX + 1).unwrap() > selected);
    }

    #[test]
    fn standard_dimension_runs_end_before_the_32_bit_split_threshold() {
        let profiles = [
            (1, 40),
            (2, 40),
            (3, 40),
            (4, 40),
            (6, 35),
            (8, 25),
            (9, 21),
            (11, 17),
            (12, 16),
            (16, 12),
            (18, 11),
            (22, 9),
            (24, 9),
            (32, 7),
            (36, 7),
            (44, 6),
            (48, 6),
            (64, 5),
            (72, 5),
            (88, 5),
            (96, 5),
            (144, 4),
            (176, 4),
        ];
        let mut scratch = [0; MAX_PULSES + 1];
        for (dimensions, expected) in profiles {
            assert_eq!(pulse_cache_run_len(dimensions), Some(expected));
            let last = pulses_for_index(expected).unwrap();
            assert_ne!(codebook_size(dimensions, last, &mut scratch), Ok(u32::MAX));
            if expected < MAX_PULSE_INDEX {
                let next = pulses_for_index(expected + 1).unwrap();
                assert_eq!(codebook_size(dimensions, next, &mut scratch), Ok(u32::MAX));
            }
        }
        assert_eq!(pulse_cache_run_len(5), None);
    }

    #[test]
    fn generated_runs_match_the_complete_normative_cache_fingerprint() {
        // These are the 23 run profiles, in their normative packed order.
        // Hashing the leading run length and every generated Q3 cost keeps
        // the full 392-byte cache covered without storing a second copy.
        let profiles = [
            (1usize, 40u8),
            (2, 40),
            (3, 40),
            (4, 40),
            (6, 35),
            (9, 21),
            (11, 17),
            (8, 25),
            (12, 16),
            (18, 11),
            (22, 9),
            (16, 12),
            (24, 9),
            (36, 7),
            (44, 6),
            (32, 7),
            (48, 6),
            (72, 5),
            (88, 5),
            (64, 5),
            (96, 5),
            (144, 4),
            (176, 4),
        ];
        let mut scratch = [0; MAX_PULSES + 1];
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut length = 0usize;
        for (dimensions, count) in profiles {
            hash = (hash ^ u64::from(count)).wrapping_mul(0x0000_0100_0000_01b3);
            length += 1;
            for index in 1..=count {
                let pulses = pulses_for_index(index).unwrap();
                let size = codebook_size(dimensions, pulses, &mut scratch).unwrap();
                let cost = packed_pulse_cache_cost(size).unwrap();
                hash = (hash ^ u64::from(cost)).wrapping_mul(0x0000_0100_0000_01b3);
                length += 1;
            }
        }
        assert_eq!(length, 392);
        assert_eq!(hash, 0x302c_ab98_a404_f495);
    }

    #[test]
    fn band_lm_lookup_uses_normative_proxy_profiles() {
        let mut scratch = [0; MAX_PULSES + 1];
        // Band 16 has eight actual coefficients at LM=0, but the normative
        // cache prices it with the six-dimensional run.
        assert_eq!(band_pulse_cost(16, 0, 1, &mut scratch), Ok(29));
        assert_eq!(maximum_band_cost(16, 0, &mut scratch), Ok(251));
        assert_eq!(
            u16::from(packed_pulse_cache_cost(codebook_size(8, 1, &mut scratch).unwrap()).unwrap())
                + 1,
            32
        );

        // Recursive LM=-1 profiles and the upper LM boundary are present;
        // tuples that cannot contain a partition remain sentinels.
        assert_eq!(band_pulse_cost(16, -1, 1, &mut scratch), Ok(21));
        assert!(maximum_band_cost(20, 3, &mut scratch).is_ok());
        assert_eq!(
            maximum_band_cost(0, -1, &mut scratch),
            Err(Error::InvalidFrameSize)
        );
    }

    #[test]
    fn range_coded_pvq_vectors_round_trip() {
        let vectors: [(&[i32], usize); 4] = [
            (&[1, 0, 0, 0], 1),
            (&[-1, 1, 0, 0], 2),
            (&[2, -1, 1, 0], 4),
            (&[0, 0, 0, 0], 0),
        ];
        for (expected, pulses) in vectors {
            let mut bytes = [0; 32];
            let mut scratch = [0; 33];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_range(&mut encoder, pulses, expected, &mut scratch).unwrap();
            let tell = encoder.tell_frac();
            encoder.finish().unwrap();
            let mut decoded_pulses = [7; 4];
            let mut normalized = [7.0; 4];
            let mut decoder = RangeDecoder::new(&bytes);
            decode_range(
                &mut decoder,
                pulses,
                0,
                &mut decoded_pulses,
                &mut normalized,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(&decoded_pulses, expected);
            assert_eq!(decoder.tell_frac(), tell);
            if pulses == 0 {
                assert_eq!(normalized, [0.0; 4]);
            } else {
                let energy: f32 = normalized.iter().map(|value| value * value).sum();
                assert!((energy - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn low_dimension_direct_large_codebook_needs_no_large_workspace() {
        let expected = [160, 0, 0, 0];
        let mut bytes = [0u8; 16];
        let mut scratch = [0u32; 1];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_range(&mut encoder, 160, &expected, &mut scratch).unwrap();
        encoder.finish().unwrap();
        let mut decoded = [0; 4];
        let mut normalized = [0.0; 4];
        decode_range(
            &mut RangeDecoder::new(&bytes),
            160,
            0,
            &mut decoded,
            &mut normalized,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(decoded, expected);
    }
}
