//! Exact CELT allocation arithmetic specified by RFC 6716 section 4.3.3.

use crate::{
    Error, RangeDecoder, RangeEncoder,
    bands::{BAND_COUNT, BAND_EDGES_2_5_MS},
};

const TRIM_PDF: [u8; 11] = [2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2];
const LOG2_FRAC: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];
const MAX_FINE_BITS: i32 = 8;

fn allocation_log_n(width: usize) -> Result<i32, Error> {
    let width = u32::try_from(width).map_err(|_| Error::InvalidFrameSize)?;
    // The allocation table stores a conservative Q3 logarithm. Truncating
    // non-power-of-two band widths changes both fine-energy depth and the
    // residual shape budget.
    Ok(i32::from(crate::pvq::codebook_cost(width)?))
}
const FINE_OFFSET: i32 = 21;

/// RFC 6716 section 4.3.3 maximum-allocation cache. Rows are indexed by
/// `2 * LM + stereo` and columns by CELT band.
const CACHE_CAPS: [[u8; BAND_COUNT]; 8] = [
    [
        224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168,
        134, 61, 37,
    ],
    [
        224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198, 183,
        144, 66, 40,
    ],
    [
        160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193, 183, 183, 172,
        138, 64, 38,
    ],
    [
        240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204, 204, 204, 193, 193, 180,
        143, 66, 40,
    ],
    [
        185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193, 193, 193, 193, 193, 183, 183, 172,
        138, 65, 39,
    ],
    [
        207, 207, 207, 207, 207, 207, 207, 207, 204, 204, 204, 204, 201, 201, 201, 188, 188, 176,
        141, 66, 40,
    ],
    [
        193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173,
        139, 65, 39,
    ],
    [
        204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175,
        140, 66, 40,
    ],
];

pub const QUALITY_COUNT: usize = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reservations {
    pub available: i32,
    pub anti_collapse: i32,
    pub skip: i32,
    pub intensity: i32,
    pub dual_stereo: i32,
}

/// Reserves CELT's conditional allocation flags, all in 1/8-bit units.
pub fn reserve_flags(
    frame_bytes: usize,
    tell_fractional: u32,
    channels: u8,
    lm: u8,
    transient: bool,
    start: usize,
    end: usize,
) -> Result<Reservations, Error> {
    if frame_bytes == 0
        || frame_bytes > crate::MAX_FRAME_BYTES
        || !(1..=2).contains(&channels)
        || lm > 3
        || start >= end
        || end > BAND_COUNT
        || end - start >= LOG2_FRAC.len()
    {
        return Err(Error::InvalidPacket);
    }
    let capacity = i32::try_from(frame_bytes * 64).map_err(|_| Error::InvalidPacket)?;
    let mut available = (capacity - tell_fractional as i32 - 1).max(0);
    let anti_collapse = if transient && lm > 1 && available >= (i32::from(lm) + 2) * 8 {
        8
    } else {
        0
    };
    available = (available - anti_collapse).max(0);
    // RFC 6716 section 4.3.3 uses a strict capacity check here: with
    // exactly one bit left there is no room to reserve the skip flag and
    // still leave shape capacity.
    let skip = if available > 8 { 8 } else { 0 };
    available -= skip;
    let mut intensity = 0;
    let mut dual_stereo = 0;
    if channels == 2 {
        let candidate = i32::from(LOG2_FRAC[end - start]);
        if candidate <= available {
            intensity = candidate;
            available -= candidate;
            if available > 8 {
                dual_stereo = 8;
                available -= 8;
            }
        }
    }
    Ok(Reservations {
        available,
        anti_collapse,
        skip,
        intensity,
        dual_stereo,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationResult {
    pub shape: [i32; BAND_COUNT],
    pub fine: [u8; BAND_COUNT],
    pub priority: [u8; BAND_COUNT],
    pub balance: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FinalizeConfig {
    pub channels: u8,
    pub lm: u8,
    pub start: usize,
    pub end: usize,
    pub reservations: Reservations,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FinalAllocation {
    pub coded_bands: usize,
    pub intensity: usize,
    pub dual_stereo: bool,
    pub bands: AllocationResult,
}

#[allow(clippy::too_many_arguments)]
fn skip_bands<F>(
    config: FinalizeConfig,
    base: &BaseAllocation,
    mut decide_stop: F,
) -> Result<(usize, i32, i32, i32, [i32; BAND_COUNT]), Error>
where
    F: FnMut(usize) -> Result<bool, Error>,
{
    if !(1..=2).contains(&config.channels)
        || config.lm > 3
        || config.start >= config.end
        || config.end > BAND_COUNT
        || base.skip_start < config.start
        || base.skip_start >= config.end
    {
        return Err(Error::InvalidPacket);
    }
    let allocation_floor = i32::from(config.channels) * 8;
    let mut total = config.reservations.available;
    let mut used = base.used;
    let mut intensity_reserve = config.reservations.intensity;
    let mut bits = base.bits;
    let mut coded_bands = config.end;
    loop {
        let band = coded_bands - 1;
        if band <= base.skip_start {
            total += config.reservations.skip;
            break;
        }
        let denominator =
            i32::from(BAND_EDGES_2_5_MS[coded_bands] - BAND_EDGES_2_5_MS[config.start]);
        let mut left = total - used;
        if left < 0 || denominator <= 0 {
            return Err(Error::InvalidPacket);
        }
        let per_coefficient = left / denominator;
        left -= denominator * per_coefficient;
        let band_offset = i32::from(BAND_EDGES_2_5_MS[band] - BAND_EDGES_2_5_MS[config.start]);
        let remainder = (left - band_offset).max(0);
        let width = i32::from(BAND_EDGES_2_5_MS[coded_bands] - BAND_EDGES_2_5_MS[band]);
        let mut band_bits = bits[band] + per_coefficient * width + remainder;
        if band_bits >= base.thresholds[band].max(allocation_floor + 8) {
            if decide_stop(coded_bands)? {
                break;
            }
            used += 8;
            band_bits -= 8;
        }
        used -= bits[band] + intensity_reserve;
        if intensity_reserve > 0 {
            intensity_reserve = i32::from(LOG2_FRAC[band - config.start]);
        }
        used += intensity_reserve;
        if band_bits >= allocation_floor {
            used += allocation_floor;
            bits[band] = allocation_floor;
        } else {
            bits[band] = 0;
        }
        coded_bands -= 1;
    }
    if coded_bands <= config.start {
        return Err(Error::InvalidPacket);
    }
    Ok((coded_bands, total, intensity_reserve, used, bits))
}

/// Returns the highest coded-band count reachable at the current allocation
/// budget without consuming any entropy symbols.
pub fn maximum_coded_bands(config: FinalizeConfig, base: &BaseAllocation) -> Result<usize, Error> {
    let (coded_bands, _, _, _, _) = skip_bands(config, base, |_| Ok(true))?;
    Ok(coded_bands)
}

fn redistribute(
    start: usize,
    coded_bands: usize,
    total: i32,
    mut used: i32,
    bits: &mut [i32; BAND_COUNT],
) -> Result<(), Error> {
    let coefficients = i32::from(BAND_EDGES_2_5_MS[coded_bands] - BAND_EDGES_2_5_MS[start]);
    let mut left = total - used;
    if left < 0 || coefficients <= 0 {
        return Err(Error::InvalidPacket);
    }
    let per_coefficient = left / coefficients;
    left -= coefficients * per_coefficient;
    for band in start..coded_bands {
        let width = i32::from(BAND_EDGES_2_5_MS[band + 1] - BAND_EDGES_2_5_MS[band]);
        let addition = per_coefficient * width;
        bits[band] += addition;
        used += addition;
    }
    for band in start..coded_bands {
        let width = i32::from(BAND_EDGES_2_5_MS[band + 1] - BAND_EDGES_2_5_MS[band]);
        let addition = left.min(width);
        bits[band] += addition;
        used += addition;
        left -= addition;
    }
    if used != total || left != 0 {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

pub fn decode_final_allocation(
    decoder: &mut RangeDecoder<'_>,
    config: FinalizeConfig,
    base: &BaseAllocation,
    caps: &[i32; BAND_COUNT],
) -> Result<FinalAllocation, Error> {
    let (coded_bands, mut total, intensity_reserve, used, mut bits) =
        skip_bands(config, base, |_| decoder.decode_bit_logp(1))?;
    let intensity = if intensity_reserve > 0 {
        config.start + decoder.decode_uint((coded_bands + 1 - config.start) as u32)? as usize
    } else {
        coded_bands
    };
    let mut dual_reserve = config.reservations.dual_stereo;
    if intensity <= config.start {
        total += dual_reserve;
        dual_reserve = 0;
    }
    let dual_stereo = dual_reserve > 0 && decoder.decode_bit_logp(1)?;
    redistribute(config.start, coded_bands, total, used, &mut bits)?;
    let bands = split_fine_shape(
        config.channels,
        config.lm,
        config.start,
        config.end,
        coded_bands,
        intensity,
        dual_stereo,
        &bits,
        caps,
    )?;
    Ok(FinalAllocation {
        coded_bands,
        intensity,
        dual_stereo,
        bands,
    })
}

pub fn encode_final_allocation(
    encoder: &mut RangeEncoder<'_>,
    config: FinalizeConfig,
    base: &BaseAllocation,
    caps: &[i32; BAND_COUNT],
    requested_coded_bands: usize,
    requested_intensity: usize,
    requested_dual_stereo: bool,
) -> Result<FinalAllocation, Error> {
    if requested_coded_bands <= config.start || requested_coded_bands > config.end {
        return Err(Error::InvalidPacket);
    }
    let (coded_bands, mut total, intensity_reserve, used, mut bits) =
        skip_bands(config, base, |current| {
            let stop = current == requested_coded_bands;
            encoder.encode_bit_logp(stop, 1)?;
            Ok(stop)
        })?;
    if coded_bands != requested_coded_bands {
        return Err(Error::InvalidPacket);
    }
    let intensity = if intensity_reserve > 0 {
        let intensity = requested_intensity.min(coded_bands);
        if intensity < config.start {
            return Err(Error::InvalidPacket);
        }
        encoder.encode_uint(
            (intensity - config.start) as u32,
            (coded_bands + 1 - config.start) as u32,
        )?;
        intensity
    } else {
        coded_bands
    };
    let mut dual_reserve = config.reservations.dual_stereo;
    if intensity <= config.start {
        total += dual_reserve;
        dual_reserve = 0;
    }
    let dual_stereo = if dual_reserve > 0 {
        encoder.encode_bit_logp(requested_dual_stereo, 1)?;
        requested_dual_stereo
    } else {
        false
    };
    redistribute(config.start, coded_bands, total, used, &mut bits)?;
    let bands = split_fine_shape(
        config.channels,
        config.lm,
        config.start,
        config.end,
        coded_bands,
        intensity,
        dual_stereo,
        &bits,
        caps,
    )?;
    Ok(FinalAllocation {
        coded_bands,
        intensity,
        dual_stereo,
        bands,
    })
}

/// Splits already-distributed per-band capacity between fine energy and PVQ
/// shape, including cap excess rebalancing.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn split_fine_shape(
    channels: u8,
    lm: u8,
    start: usize,
    end: usize,
    coded_bands: usize,
    intensity: usize,
    dual_stereo: bool,
    bits: &[i32; BAND_COUNT],
    caps: &[i32; BAND_COUNT],
) -> Result<AllocationResult, Error> {
    if !(1..=2).contains(&channels)
        || lm > 3
        || start >= coded_bands
        || coded_bands > end
        || end > BAND_COUNT
        || intensity < start
        || intensity > coded_bands
        || bits[start..end].iter().any(|&value| value < 0)
        || caps[start..coded_bands].iter().any(|&value| value < 0)
    {
        return Err(Error::InvalidPacket);
    }
    let stereo = i32::from(channels == 2);
    let channels_i32 = i32::from(channels);
    let mut result = AllocationResult {
        shape: [0; BAND_COUNT],
        fine: [0; BAND_COUNT],
        priority: [0; BAND_COUNT],
        balance: 0,
    };
    for band in start..coded_bands {
        let base_width = shortest_width(band).ok_or(Error::InvalidPacket)?;
        let dimensions = base_width << lm;
        let bit = bits[band] + result.balance;
        let mut excess;
        if dimensions > 1 {
            excess = (bit - caps[band]).max(0);
            result.shape[band] = bit - excess;
            let coupled = channels == 2 && dimensions > 2 && !dual_stereo && band < intensity;
            let denominator = channels_i32 * dimensions as i32 + i32::from(coupled);
            let log_n = allocation_log_n(base_width)?;
            let n_c_log_n = denominator * (log_n + i32::from(lm) * 8);
            let mut offset = n_c_log_n / 2 - denominator * FINE_OFFSET;
            if dimensions == 2 {
                offset += denominator * 2;
            }
            if result.shape[band] + offset < denominator * 16 {
                offset += n_c_log_n / 4;
            } else if result.shape[band] + offset < denominator * 24 {
                offset += n_c_log_n / 8;
            }
            let numerator = (result.shape[band] + offset + denominator * 4).max(0);
            let mut fine = numerator / denominator / 8;
            if channels_i32 * fine > result.shape[band] / 8 {
                fine = result.shape[band] >> stereo >> 3;
            }
            fine = fine.min(MAX_FINE_BITS);
            result.fine[band] = fine as u8;
            result.priority[band] = u8::from(fine * denominator * 8 >= result.shape[band] + offset);
            result.shape[band] -= channels_i32 * fine * 8;
        } else {
            excess = (bit - channels_i32 * 8).max(0);
            result.shape[band] = bit - excess;
            result.priority[band] = 1;
        }
        if excess > 0 {
            let fine = i32::from(result.fine[band]);
            let extra_fine = (excess >> (stereo + 3)).min(MAX_FINE_BITS - fine);
            result.fine[band] = (fine + extra_fine) as u8;
            let extra_bits = extra_fine * channels_i32 * 8;
            result.priority[band] = u8::from(extra_bits >= excess - result.balance);
            excess -= extra_bits;
        }
        result.balance = excess;
    }
    for band in coded_bands..end {
        result.fine[band] = (bits[band] >> stereo >> 3) as u8;
        if result.fine[band] > MAX_FINE_BITS as u8 {
            return Err(Error::InvalidPacket);
        }
        if channels_i32 * i32::from(result.fine[band]) * 8 != bits[band] {
            return Err(Error::InvalidPacket);
        }
        result.priority[band] = u8::from(result.fine[band] < 1);
    }
    Ok(result)
}

/// RFC 6716 Table 57, in 1/32 bit per shortest-frame MDCT bin.
pub const STATIC_ALLOCATION: [[u8; QUALITY_COUNT]; BAND_COUNT] = [
    [0, 90, 110, 118, 126, 134, 144, 152, 162, 172, 200],
    [0, 80, 100, 110, 119, 127, 137, 145, 155, 165, 200],
    [0, 75, 90, 103, 112, 120, 130, 138, 148, 158, 200],
    [0, 69, 84, 93, 104, 114, 124, 132, 142, 152, 200],
    [0, 63, 78, 86, 95, 103, 113, 123, 133, 143, 200],
    [0, 56, 71, 80, 89, 97, 107, 117, 127, 137, 200],
    [0, 49, 65, 75, 83, 91, 101, 111, 121, 131, 200],
    [0, 40, 58, 70, 78, 85, 95, 105, 115, 125, 200],
    [0, 34, 51, 65, 72, 78, 88, 98, 108, 118, 198],
    [0, 29, 45, 59, 66, 72, 82, 92, 102, 112, 193],
    [0, 20, 39, 53, 60, 66, 76, 86, 96, 106, 188],
    [0, 18, 32, 47, 54, 60, 70, 80, 90, 100, 183],
    [0, 10, 26, 40, 47, 54, 64, 74, 84, 94, 178],
    [0, 0, 20, 31, 39, 47, 57, 67, 77, 87, 173],
    [0, 0, 12, 23, 32, 41, 51, 61, 71, 81, 168],
    [0, 0, 0, 15, 25, 35, 45, 55, 65, 75, 163],
    [0, 0, 0, 4, 17, 29, 39, 49, 59, 69, 158],
    [0, 0, 0, 0, 12, 23, 33, 43, 53, 63, 153],
    [0, 0, 0, 0, 1, 16, 26, 36, 46, 56, 148],
    [0, 0, 0, 0, 0, 10, 15, 20, 30, 45, 129],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 20, 104],
];

pub const fn shortest_width(band: usize) -> Option<usize> {
    if band >= BAND_COUNT {
        None
    } else {
        Some(BAND_EDGES_2_5_MS[band + 1] as usize - BAND_EDGES_2_5_MS[band] as usize)
    }
}

/// Generates CELT's maximum useful allocation for every standard band.
/// Values are returned in eighth-bit units and include all coded channels.
pub fn band_caps(channels: u8, lm: u8, output: &mut [i32; BAND_COUNT]) -> Result<(), Error> {
    if !(1..=2).contains(&channels) || lm > 3 {
        return Err(Error::InvalidPacket);
    }
    let row = usize::from(lm) * 2 + usize::from(channels == 2);
    for (band, output) in output.iter_mut().enumerate() {
        let dimensions = i32::from(channels)
            * i32::try_from(shortest_width(band).ok_or(Error::InvalidPacket)? << lm)
                .map_err(|_| Error::InvalidPacket)?;
        *output = (i32::from(CACHE_CAPS[row][band]) + 64) * dimensions / 4;
    }
    Ok(())
}

/// Computes one interpolated static allocation vector in 1/8-bit units.
/// `quality_q6` spans 0 through 640, with each table column 64 units apart.
pub fn static_vector(
    channels: u8,
    lm: u8,
    quality_q6: u16,
    output: &mut [i32; BAND_COUNT],
) -> Result<(), Error> {
    if !(1..=2).contains(&channels) || lm > 3 || quality_q6 > 640 {
        return Err(Error::InvalidPacket);
    }
    let lower = usize::from(quality_q6 / 64).min(QUALITY_COUNT - 1);
    let upper = (lower + 1).min(QUALITY_COUNT - 1);
    let fraction = i32::from(quality_q6 % 64);
    for band in 0..BAND_COUNT {
        let low = i32::from(STATIC_ALLOCATION[band][lower]);
        let high = i32::from(STATIC_ALLOCATION[band][upper]);
        let interpolated = low * 64 + (high - low) * fraction;
        let bins = shortest_width(band).ok_or(Error::InvalidPacket)? as i32;
        output[band] = (i32::from(channels) * bins * interpolated * (1 << lm)) >> 8;
    }
    Ok(())
}

/// Per-band trim adjustment in 1/8 bits.
pub fn trim_offsets(
    channels: u8,
    lm: u8,
    trim: u8,
    start: usize,
    end: usize,
    output: &mut [i32; BAND_COUNT],
) -> Result<(), Error> {
    if !(1..=2).contains(&channels) || lm > 3 || trim > 10 || start >= end || end > BAND_COUNT {
        return Err(Error::InvalidPacket);
    }
    output.fill(0);
    for (band, value) in output.iter_mut().enumerate().take(end).skip(start) {
        let width = shortest_width(band).ok_or(Error::InvalidPacket)? as i32;
        let remaining = (end - band - 1) as i32;
        *value = ((i32::from(trim) - 5 - i32::from(lm))
            * i32::from(channels)
            * width
            * remaining
            * (1 << lm)
            * 8)
            >> 6;
        if (width as usize) << lm == 1 {
            *value -= 8 * i32::from(channels);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaseAllocation {
    pub bits: [i32; BAND_COUNT],
    pub thresholds: [i32; BAND_COUNT],
    pub used: i32,
    pub lower_quality: u8,
    pub upper_quality: u8,
    pub interpolation_q6: u8,
    pub skip_start: usize,
}

#[allow(clippy::too_many_arguments)]
fn candidate_sum(
    start: usize,
    end: usize,
    channels: u8,
    lower: &[i32; BAND_COUNT],
    difference: &[i32; BAND_COUNT],
    interpolation_q6: i32,
    thresholds: &[i32; BAND_COUNT],
    caps: &[i32; BAND_COUNT],
) -> i32 {
    let allocation_floor = i32::from(channels) * 8;
    let mut sum = 0;
    let mut reached_threshold = false;
    for band in (start..end).rev() {
        let value = lower[band] + ((interpolation_q6 * difference[band]) >> 6);
        if value >= thresholds[band] || reached_threshold {
            reached_threshold = true;
            sum += value.min(caps[band]);
        } else if value >= allocation_floor {
            sum += allocation_floor;
        }
    }
    sum
}

/// Searches Table 57 and performs CELT's six-step interpolation before band
/// skipping and leftover redistribution.
#[allow(clippy::too_many_arguments)]
pub fn base_allocation(
    channels: u8,
    lm: u8,
    start: usize,
    end: usize,
    total: i32,
    trim: u8,
    boosts: &[i32; BAND_COUNT],
    caps: &[i32; BAND_COUNT],
) -> Result<BaseAllocation, Error> {
    if !(1..=2).contains(&channels)
        || lm > 3
        || start >= end
        || end > BAND_COUNT
        || total < 0
        || trim > 10
        || boosts[start..end].iter().any(|&value| value < 0)
        || caps[start..end].iter().any(|&value| value < 0)
    {
        return Err(Error::InvalidPacket);
    }
    let mut thresholds = [0; BAND_COUNT];
    let mut trim_values = [0; BAND_COUNT];
    trim_offsets(channels, lm, trim, start, end, &mut trim_values)?;
    for (band, threshold) in thresholds.iter_mut().enumerate().take(end).skip(start) {
        *threshold = shape_threshold(band, lm, channels)?;
    }
    let mut low_quality = 1usize;
    let mut high_quality = QUALITY_COUNT - 1;
    let mut vector = [0; BAND_COUNT];
    while low_quality <= high_quality {
        let midpoint = (low_quality + high_quality) / 2;
        static_vector(channels, lm, (midpoint * 64) as u16, &mut vector)?;
        for band in start..end {
            if vector[band] > 0 {
                vector[band] = (vector[band] + trim_values[band]).max(0);
            }
            vector[band] += boosts[band];
        }
        let difference = [0; BAND_COUNT];
        let sum = candidate_sum(
            start,
            end,
            channels,
            &vector,
            &difference,
            0,
            &thresholds,
            caps,
        );
        if sum > total {
            if midpoint == 0 {
                break;
            }
            high_quality = midpoint - 1;
        } else {
            low_quality = midpoint + 1;
        }
    }
    let lower_quality = low_quality.saturating_sub(1).min(QUALITY_COUNT - 1);
    let upper_quality = low_quality.min(QUALITY_COUNT);
    let mut lower = [0; BAND_COUNT];
    let mut upper = [0; BAND_COUNT];
    static_vector(channels, lm, (lower_quality * 64) as u16, &mut lower)?;
    if upper_quality < QUALITY_COUNT {
        static_vector(channels, lm, (upper_quality * 64) as u16, &mut upper)?;
    } else {
        upper.copy_from_slice(caps);
    }
    let mut skip_start = start;
    for band in start..end {
        if lower[band] > 0 {
            lower[band] = (lower[band] + trim_values[band]).max(0);
        }
        if upper[band] > 0 {
            upper[band] = (upper[band] + trim_values[band]).max(0);
        }
        if lower_quality > 0 {
            lower[band] += boosts[band];
        }
        upper[band] += boosts[band];
        if boosts[band] > 0 {
            skip_start = band;
        }
        upper[band] = (upper[band] - lower[band]).max(0);
    }
    let mut interpolation_low = 0i32;
    let mut interpolation_high = 64i32;
    for _ in 0..6 {
        let midpoint = (interpolation_low + interpolation_high) / 2;
        let sum = candidate_sum(
            start,
            end,
            channels,
            &lower,
            &upper,
            midpoint,
            &thresholds,
            caps,
        );
        if sum > total {
            interpolation_high = midpoint;
        } else {
            interpolation_low = midpoint;
        }
    }
    let allocation_floor = i32::from(channels) * 8;
    let mut bits = [0; BAND_COUNT];
    let mut used = 0;
    let mut reached_threshold = false;
    for band in (start..end).rev() {
        let mut value = lower[band] + ((interpolation_low * upper[band]) >> 6);
        if value < thresholds[band] && !reached_threshold {
            value = if value >= allocation_floor {
                allocation_floor
            } else {
                0
            };
        } else {
            reached_threshold = true;
        }
        value = value.min(caps[band]);
        bits[band] = value;
        used += value;
    }
    Ok(BaseAllocation {
        bits,
        thresholds,
        used,
        lower_quality: lower_quality as u8,
        upper_quality: upper_quality.min(QUALITY_COUNT - 1) as u8,
        interpolation_q6: interpolation_low as u8,
        skip_start,
    })
}

/// RFC boost quantum for a band, in 1/8-bit units.
pub fn boost_quantum(band: usize, lm: u8) -> Result<i32, Error> {
    if lm > 3 {
        return Err(Error::InvalidPacket);
    }
    let bins = shortest_width(band).ok_or(Error::InvalidPacket)? << lm;
    Ok((8 * bins).min(48usize.max(bins)) as i32)
}

/// Hard minimum shape allocation from section 4.3.3, in 1/8 bits.
pub fn shape_threshold(band: usize, lm: u8, channels: u8) -> Result<i32, Error> {
    if !(1..=2).contains(&channels) || lm > 3 {
        return Err(Error::InvalidPacket);
    }
    let bins = (shortest_width(band).ok_or(Error::InvalidPacket)? << lm) as i32;
    Ok((24 * bins / 16).max(8 * i32::from(channels)))
}

/// Decodes the adaptive per-band boost sequence. All bit counts use CELT's
/// fractional 1/8-bit units.
pub fn decode_boosts(
    decoder: &mut RangeDecoder<'_>,
    lm: u8,
    start: usize,
    end: usize,
    mut total_bits: i32,
    caps: &[i32; BAND_COUNT],
    boosts: &mut [i32; BAND_COUNT],
) -> Result<i32, Error> {
    if lm > 3
        || start >= end
        || end > BAND_COUNT
        || total_bits < 0
        || caps[start..end].iter().any(|&cap| cap < 0)
    {
        return Err(Error::InvalidPacket);
    }
    boosts.fill(0);
    let mut dynalloc_logp = 6u8;
    let mut total_boost = 0i32;
    for band in start..end {
        let quantum = boost_quantum(band, lm)?;
        let mut loop_logp = dynalloc_logp;
        while i32::from(loop_logp) * 8 + (decoder.tell_frac() as i32) < total_bits + total_boost
            && boosts[band] < caps[band]
        {
            let enabled = decoder.decode_bit_logp(loop_logp)?;
            if !enabled {
                break;
            }
            boosts[band] += quantum;
            total_boost += quantum;
            total_bits -= quantum;
            loop_logp = 1;
        }
        if boosts[band] != 0 && dynalloc_logp > 2 {
            dynalloc_logp -= 1;
        }
    }
    Ok(total_boost)
}

/// Encoder mirror of [`decode_boosts`]. Requested boosts must be whole boost
/// quanta and must be reachable under the same capacity gates as the decoder.
pub fn encode_boosts(
    encoder: &mut RangeEncoder<'_>,
    lm: u8,
    start: usize,
    end: usize,
    mut total_bits: i32,
    caps: &[i32; BAND_COUNT],
    requested: &[i32; BAND_COUNT],
) -> Result<i32, Error> {
    if lm > 3
        || start >= end
        || end > BAND_COUNT
        || total_bits < 0
        || caps[start..end].iter().any(|&cap| cap < 0)
        || requested[start..end].iter().any(|&boost| boost < 0)
    {
        return Err(Error::InvalidPacket);
    }
    let mut dynalloc_logp = 6u8;
    let mut total_boost = 0i32;
    for band in start..end {
        let quantum = boost_quantum(band, lm)?;
        if requested[band] % quantum != 0 {
            return Err(Error::InvalidPacket);
        }
        let mut emitted = 0;
        let mut loop_logp = dynalloc_logp;
        while i32::from(loop_logp) * 8 + (encoder.tell_frac() as i32) < total_bits + total_boost
            && emitted < caps[band]
        {
            let enabled = emitted < requested[band];
            encoder.encode_bit_logp(enabled, loop_logp)?;
            if !enabled {
                break;
            }
            emitted += quantum;
            total_boost += quantum;
            total_bits -= quantum;
            loop_logp = 1;
        }
        if emitted != requested[band] {
            return Err(Error::InvalidPacket);
        }
        if emitted != 0 && dynalloc_logp > 2 {
            dynalloc_logp -= 1;
        }
    }
    Ok(total_boost)
}

pub fn decode_trim(
    decoder: &mut RangeDecoder<'_>,
    total_bits: i32,
    total_boost: i32,
) -> Result<u8, Error> {
    if total_bits < 0 || total_boost < 0 {
        return Err(Error::InvalidPacket);
    }
    if decoder.tell_frac() as i32 + 48 <= total_bits - total_boost {
        Ok(decoder.decode_pdf(&TRIM_PDF)? as u8)
    } else {
        Ok(5)
    }
}

pub fn encode_trim(
    encoder: &mut RangeEncoder<'_>,
    total_bits: i32,
    total_boost: i32,
    trim: u8,
) -> Result<(), Error> {
    if total_bits < 0 || total_boost < 0 || trim > 10 {
        return Err(Error::InvalidPacket);
    }
    if encoder.tell_frac() as i32 + 48 <= total_bits - total_boost {
        encoder.encode_pdf(usize::from(trim), &TRIM_PDF)
    } else if trim == 5 {
        Ok(())
    } else {
        Err(Error::InvalidPacket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_log_rounds_fractional_band_widths_up() {
        assert_eq!(allocation_log_n(1), Ok(0));
        assert_eq!(allocation_log_n(3), Ok(13));
        assert_eq!(allocation_log_n(5), Ok(19));
        assert_eq!(allocation_log_n(6), Ok(21));
        assert_eq!(allocation_log_n(8), Ok(24));
    }

    #[test]
    fn generated_band_caps_cover_every_standard_mode() {
        assert_eq!(CACHE_CAPS[0][8], 160);
        assert_eq!(CACHE_CAPS[0][12], 185);
        assert_eq!(CACHE_CAPS[0][20], 37);
        assert_eq!(CACHE_CAPS[7][20], 40);
        for channels in 1..=2 {
            let mut previous = [0; BAND_COUNT];
            for lm in 0..=3 {
                let mut caps = [0; BAND_COUNT];
                band_caps(channels, lm, &mut caps).unwrap();
                for band in 0..BAND_COUNT {
                    assert!(caps[band] > 0);
                    if lm > 0 {
                        assert!(caps[band] >= previous[band]);
                    }
                }
                previous = caps;
            }
        }
    }

    #[test]
    fn cap_generation_rejects_invalid_stream_geometry() {
        let mut caps = [0; BAND_COUNT];
        assert_eq!(band_caps(0, 0, &mut caps), Err(Error::InvalidPacket));
        assert_eq!(band_caps(1, 4, &mut caps), Err(Error::InvalidPacket));
    }

    #[test]
    fn table_57_endpoints_and_interpolation_are_exact() {
        let mut low = [0; BAND_COUNT];
        let mut high = [0; BAND_COUNT];
        let mut midpoint = [0; BAND_COUNT];
        static_vector(1, 0, 64, &mut low).unwrap();
        static_vector(1, 0, 128, &mut high).unwrap();
        static_vector(1, 0, 96, &mut midpoint).unwrap();
        for band in 0..BAND_COUNT {
            let width = shortest_width(band).unwrap() as i32;
            let expected = width
                * (i32::from(STATIC_ALLOCATION[band][1]) + i32::from(STATIC_ALLOCATION[band][2]))
                / 8;
            assert_eq!(midpoint[band], expected);
        }
        assert_eq!(low[0], 22);
        assert_eq!(high[0], 27);
        assert_eq!(high[20], 0);
    }

    #[test]
    fn channel_and_frame_size_scaling_is_linear() {
        let mut mono = [0; BAND_COUNT];
        let mut stereo_long = [0; BAND_COUNT];
        static_vector(1, 0, 320, &mut mono).unwrap();
        static_vector(2, 3, 320, &mut stereo_long).unwrap();
        for band in 0..BAND_COUNT {
            assert!(stereo_long[band] >= mono[band] * 16);
            assert!(stereo_long[band] < mono[band] * 16 + 16);
        }
    }

    #[test]
    fn trim_boost_and_threshold_follow_rfc_bounds() {
        let mut neutral = [0; BAND_COUNT];
        trim_offsets(1, 0, 5, 0, BAND_COUNT, &mut neutral).unwrap();
        assert_eq!(neutral[0], -8);
        assert_eq!(neutral[8], 0);
        assert_eq!(boost_quantum(0, 0), Ok(8));
        assert_eq!(boost_quantum(20, 3), Ok(176));
        assert_eq!(shape_threshold(0, 0, 2), Ok(16));
        assert_eq!(shape_threshold(20, 3, 1), Ok(264));

        let mut negative = [0; BAND_COUNT];
        trim_offsets(1, 0, 4, 0, BAND_COUNT, &mut negative).unwrap();
        assert_eq!(negative[0], -11);
    }

    #[test]
    fn invalid_allocation_inputs_are_rejected() {
        let mut output = [0; BAND_COUNT];
        assert_eq!(
            static_vector(0, 0, 0, &mut output),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            static_vector(1, 4, 0, &mut output),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            trim_offsets(1, 0, 11, 0, 1, &mut output),
            Err(Error::InvalidPacket)
        );
        assert_eq!(boost_quantum(BAND_COUNT, 0), Err(Error::InvalidPacket));
    }

    #[test]
    fn adaptive_boosts_and_trim_round_trip_with_identical_tell() {
        let caps = [256; BAND_COUNT];
        let mut requested = [0; BAND_COUNT];
        requested[0] = 16;
        requested[1] = 8;
        requested[8] = 48;
        let total_bits = 64 * 64;
        let mut bytes = [0; 64];
        let mut encoder = RangeEncoder::new(&mut bytes);
        let boost = encode_boosts(&mut encoder, 0, 0, 10, total_bits, &caps, &requested).unwrap();
        encode_trim(&mut encoder, total_bits, boost, 7).unwrap();
        let final_tell = encoder.tell_frac();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let mut decoded = [0; BAND_COUNT];
        assert_eq!(
            decode_boosts(&mut decoder, 0, 0, 10, total_bits, &caps, &mut decoded),
            Ok(boost)
        );
        assert_eq!(decoded, requested);
        assert_eq!(decode_trim(&mut decoder, total_bits, boost), Ok(7));
        assert_eq!(decoder.tell_frac(), final_tell);
    }

    #[test]
    fn trim_defaults_when_six_bits_are_unavailable() {
        let mut bytes = [0; 8];
        let mut encoder = RangeEncoder::new(&mut bytes);
        assert_eq!(encode_trim(&mut encoder, 48, 0, 5), Ok(()));
        assert_eq!(
            encode_trim(&mut encoder, 48, 0, 4),
            Err(Error::InvalidPacket)
        );
        encoder.finish().unwrap();
        assert_eq!(decode_trim(&mut RangeDecoder::new(&bytes), 48, 0), Ok(5));
    }

    #[test]
    fn reservations_follow_transient_and_stereo_capacity_gates() {
        assert_eq!(
            reserve_flags(20, 8, 2, 3, true, 0, BAND_COUNT),
            Ok(Reservations {
                available: 1_280 - 8 - 1 - 8 - 8 - 36 - 8,
                anti_collapse: 8,
                skip: 8,
                intensity: 36,
                dual_stereo: 8,
            })
        );
        assert_eq!(
            reserve_flags(1, 55, 1, 0, false, 0, 1),
            Ok(Reservations {
                available: 8,
                anti_collapse: 0,
                skip: 0,
                intensity: 0,
                dual_stereo: 0,
            })
        );
        assert_eq!(
            reserve_flags(20, 8, 2, 3, false, 17, 21),
            Ok(Reservations {
                available: 1_280 - 8 - 1 - 8 - 19 - 8,
                anti_collapse: 0,
                skip: 8,
                intensity: 19,
                dual_stereo: 8,
            })
        );
    }

    #[test]
    fn fine_shape_split_preserves_capacity_and_caps() {
        let mut bits = [0; BAND_COUNT];
        let caps = [240; BAND_COUNT];
        for (band, value) in bits.iter_mut().enumerate() {
            *value = if band < 18 { 40 + band as i32 * 13 } else { 0 };
        }
        let result = split_fine_shape(2, 2, 0, BAND_COUNT, 18, 12, false, &bits, &caps).unwrap();
        for (band, &cap) in caps.iter().enumerate().take(18) {
            assert!(result.shape[band] >= 0);
            assert!(result.shape[band] <= cap);
            assert!(result.fine[band] <= 8);
            assert!(result.priority[band] <= 1);
        }
        assert!(result.balance >= 0);
    }

    #[test]
    fn skipped_bands_must_contain_only_whole_fine_bits() {
        let caps = [100; BAND_COUNT];
        let mut bits = [0; BAND_COUNT];
        bits[2] = 16;
        assert!(split_fine_shape(2, 0, 0, 3, 2, 2, false, &bits, &caps).is_ok());
        bits[2] = 15;
        assert_eq!(
            split_fine_shape(2, 0, 0, 3, 2, 2, false, &bits, &caps),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn base_quality_search_is_capacity_monotonic() {
        let caps = [2_000; BAND_COUNT];
        let boosts = [0; BAND_COUNT];
        let mut previous_quality = 0u16;
        let mut previous_used = 0;
        for total in [64, 256, 512, 1_024, 2_048, 4_096] {
            let allocation =
                base_allocation(2, 2, 0, BAND_COUNT, total, 5, &boosts, &caps).unwrap();
            let quality =
                u16::from(allocation.lower_quality) * 64 + u16::from(allocation.interpolation_q6);
            assert!(quality >= previous_quality);
            assert!(allocation.used >= previous_used);
            assert!(allocation.used <= total);
            assert!(allocation.bits.iter().all(|&value| value >= 0));
            previous_quality = quality;
            previous_used = allocation.used;
        }
    }

    #[test]
    fn boosts_set_the_last_unskippable_band() {
        let caps = [1_000; BAND_COUNT];
        let mut boosts = [0; BAND_COUNT];
        boosts[3] = 8;
        boosts[11] = 32;
        let allocation = base_allocation(1, 0, 0, BAND_COUNT, 1_000, 5, &boosts, &caps).unwrap();
        assert_eq!(allocation.skip_start, 11);
    }

    #[test]
    fn trim_has_no_slope_on_the_highest_coded_band() {
        let mut offsets = [0; BAND_COUNT];
        trim_offsets(1, 3, 10, 8, 16, &mut offsets).unwrap();
        assert_eq!(offsets[15], 0);
        assert!(offsets[8] > offsets[14]);
    }

    #[test]
    fn backward_skip_intensity_and_dual_stereo_round_trip() {
        let caps = [1_500; BAND_COUNT];
        let boosts = [0; BAND_COUNT];
        let mut bytes = [0; 64];
        let frame_bytes = bytes.len();
        let mut encoder = RangeEncoder::new(&mut bytes);
        let reservations =
            reserve_flags(frame_bytes, encoder.tell_frac(), 2, 2, true, 0, BAND_COUNT).unwrap();
        let base = base_allocation(
            2,
            2,
            0,
            BAND_COUNT,
            reservations.available,
            5,
            &boosts,
            &caps,
        )
        .unwrap();
        let config = FinalizeConfig {
            channels: 2,
            lm: 2,
            start: 0,
            end: BAND_COUNT,
            reservations,
        };
        let encoded =
            encode_final_allocation(&mut encoder, config, &base, &caps, 15, 10, true).unwrap();
        let tell = encoder.tell_frac();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        let decoded = decode_final_allocation(&mut decoder, config, &base, &caps).unwrap();
        assert_eq!(decoded, encoded);
        assert_eq!(decoded.coded_bands, 15);
        assert_eq!(decoded.intensity, 10);
        assert!(decoded.dual_stereo);
        assert_eq!(decoder.tell_frac(), tell);
    }
}
