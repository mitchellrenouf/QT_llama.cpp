//! Palette color cache and literal/delta reconstruction.

use crate::{Error, cdf::TileCdfs, entropy::SymbolDecoder};
use mrml_runtime::Vector;

pub const MAX_PALETTE_SIZE: usize = 8;
pub const MAX_PALETTE_CACHE: usize = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PaletteColors {
    pub sizes: [u8; 2],
    pub y: [u16; MAX_PALETTE_SIZE],
    pub u: [u16; MAX_PALETTE_SIZE],
    pub v: [u16; MAX_PALETTE_SIZE],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaletteColorMap {
    pub width: u16,
    pub height: u16,
    pub indices: Vector<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaletteMapConfig {
    pub palette_size: u8,
    pub chroma: bool,
    pub block_width: u16,
    pub block_height: u16,
    pub onscreen_width: u16,
    pub onscreen_height: u16,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
}

pub fn palette_color_context(
    map: &[u8],
    width: usize,
    row: usize,
    column: usize,
    palette_size: u8,
) -> Result<(u8, [u8; MAX_PALETTE_SIZE]), Error> {
    if !(2..=8).contains(&palette_size)
        || width == 0
        || column >= width
        || row
            .checked_mul(width)
            .and_then(|v| v.checked_add(column))
            .is_none_or(|i| i >= map.len())
    {
        return Err(Error::InvalidObu);
    }
    let mut scores = [0u8; MAX_PALETTE_SIZE];
    let mut order = core::array::from_fn(|index| index as u8);
    let mut add = |neighbor_row: usize, neighbor_column: usize, weight: u8| -> Result<(), Error> {
        let index = neighbor_row
            .checked_mul(width)
            .and_then(|v| v.checked_add(neighbor_column))
            .ok_or(Error::LimitExceeded)?;
        let color = usize::from(*map.get(index).ok_or(Error::InvalidObu)?);
        if color >= usize::from(palette_size) {
            return Err(Error::InvalidObu);
        }
        scores[color] = scores[color]
            .checked_add(weight)
            .ok_or(Error::LimitExceeded)?;
        Ok(())
    };
    if column > 0 {
        add(row, column - 1, 2)?;
    }
    if row > 0 && column > 0 {
        add(row - 1, column - 1, 1)?;
    }
    if row > 0 {
        add(row - 1, column, 2)?;
    }
    for index in 0..3 {
        let mut maximum = index;
        for candidate in index + 1..usize::from(palette_size) {
            if scores[candidate] > scores[maximum] {
                maximum = candidate;
            }
        }
        if maximum != index {
            let score = scores[maximum];
            let color = order[maximum];
            for position in (index + 1..=maximum).rev() {
                scores[position] = scores[position - 1];
                order[position] = order[position - 1];
            }
            scores[index] = score;
            order[index] = color;
        }
    }
    let hash = scores[0] + 2 * scores[1] + 2 * scores[2];
    let context = match hash {
        2 => 0,
        5 => 4,
        6 => 3,
        7 => 2,
        8 => 1,
        _ => return Err(Error::InvalidObu),
    };
    Ok((context, order))
}

pub fn read_palette_color_map(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    config: PaletteMapConfig,
) -> Result<PaletteColorMap, Error> {
    let palette_size = config.palette_size;
    let chroma = config.chroma;
    let mut block_width = config.block_width;
    let mut block_height = config.block_height;
    let mut onscreen_width = config.onscreen_width;
    let mut onscreen_height = config.onscreen_height;
    if !(2..=8).contains(&palette_size)
        || block_width == 0
        || block_height == 0
        || onscreen_width == 0
        || onscreen_height == 0
        || onscreen_width > block_width
        || onscreen_height > block_height
    {
        return Err(Error::InvalidObu);
    }
    if chroma {
        block_width >>= u8::from(config.subsampling_x);
        block_height >>= u8::from(config.subsampling_y);
        onscreen_width >>= u8::from(config.subsampling_x);
        onscreen_height >>= u8::from(config.subsampling_y);
        if block_width < 4 {
            block_width = block_width.checked_add(2).ok_or(Error::LimitExceeded)?;
            onscreen_width = onscreen_width.checked_add(2).ok_or(Error::LimitExceeded)?;
        }
        if block_height < 4 {
            block_height = block_height.checked_add(2).ok_or(Error::LimitExceeded)?;
            onscreen_height = onscreen_height.checked_add(2).ok_or(Error::LimitExceeded)?;
        }
    }
    let length = usize::from(block_width)
        .checked_mul(usize::from(block_height))
        .ok_or(Error::LimitExceeded)?;
    let mut indices = Vector::with_capacity(length).map_err(|_| Error::LimitExceeded)?;
    indices
        .try_resize(length, 0)
        .map_err(|_| Error::LimitExceeded)?;
    indices[0] =
        u8::try_from(decoder.read_ns(u32::from(palette_size))?).map_err(|_| Error::InvalidObu)?;
    let width = usize::from(block_width);
    let onscreen_width_usize = usize::from(onscreen_width);
    let onscreen_height_usize = usize::from(onscreen_height);
    for diagonal in 1..onscreen_height_usize + onscreen_width_usize - 1 {
        let mut column = diagonal.min(onscreen_width_usize - 1);
        let minimum = diagonal.saturating_sub(onscreen_height_usize - 1);
        loop {
            let row = diagonal - column;
            let (context, order) =
                palette_color_context(&indices, width, row, column, palette_size)?;
            let coded = usize::from(cdfs.read_palette_color_index(
                decoder,
                palette_size,
                context,
                chroma,
            )?);
            indices[row * width + column] = *order.get(coded).ok_or(Error::InvalidObu)?;
            if column == minimum {
                break;
            }
            column -= 1;
        }
    }
    for row in 0..onscreen_height_usize {
        let edge = indices[row * width + onscreen_width_usize - 1];
        indices[row * width + onscreen_width_usize..row * width + width].fill(edge);
    }
    for row in onscreen_height_usize..usize::from(block_height) {
        let source = (onscreen_height_usize - 1) * width;
        let destination = row * width;
        for column in 0..width {
            indices[destination + column] = indices[source + column];
        }
    }
    Ok(PaletteColorMap {
        width: block_width,
        height: block_height,
        indices,
    })
}

/// Merges the sorted above and left palettes and removes duplicates.
pub fn merge_palette_cache(
    above: &[u16],
    left: &[u16],
    output: &mut [u16; MAX_PALETTE_CACHE],
) -> Result<usize, Error> {
    if above.len() > MAX_PALETTE_SIZE || left.len() > MAX_PALETTE_SIZE {
        return Err(Error::InvalidObu);
    }
    let mut above_index = 0;
    let mut left_index = 0;
    let mut count = 0;
    while above_index < above.len() || left_index < left.len() {
        let value = if left_index >= left.len()
            || (above_index < above.len() && above[above_index] <= left[left_index])
        {
            let value = above[above_index];
            above_index += 1;
            if left_index < left.len() && left[left_index] == value {
                left_index += 1;
            }
            value
        } else {
            let value = left[left_index];
            left_index += 1;
            value
        };
        if count == 0 || output[count - 1] != value {
            output[count] = value;
            count += 1;
        }
    }
    Ok(count)
}

pub fn read_palette_colors(
    decoder: &mut SymbolDecoder<'_>,
    bit_depth: u8,
    sizes: [u8; 2],
    y_cache: &[u16],
    u_cache: &[u16],
) -> Result<PaletteColors, Error> {
    if !(8..=12).contains(&bit_depth)
        || sizes
            .iter()
            .any(|&size| size != 0 && !(2..=8).contains(&size))
        || y_cache.len() > MAX_PALETTE_CACHE
        || u_cache.len() > MAX_PALETTE_CACHE
    {
        return Err(Error::InvalidObu);
    }
    let mut colors = PaletteColors {
        sizes,
        ..PaletteColors::default()
    };
    if sizes[0] > 0 {
        read_sorted_plane(
            decoder,
            bit_depth,
            usize::from(sizes[0]),
            y_cache,
            &mut colors.y,
            true,
        )?;
    }
    if sizes[1] > 0 {
        read_sorted_plane(
            decoder,
            bit_depth,
            usize::from(sizes[1]),
            u_cache,
            &mut colors.u,
            false,
        )?;
        read_v_plane(decoder, bit_depth, usize::from(sizes[1]), &mut colors.v)?;
    }
    Ok(colors)
}

fn read_sorted_plane(
    decoder: &mut SymbolDecoder<'_>,
    bit_depth: u8,
    size: usize,
    cache: &[u16],
    output: &mut [u16; MAX_PALETTE_SIZE],
    luma: bool,
) -> Result<(), Error> {
    let maximum = 1u32 << bit_depth;
    let mut index = 0;
    for &cached in cache {
        if index == size {
            break;
        }
        if u32::from(cached) >= maximum {
            return Err(Error::InvalidObu);
        }
        if decoder.read_bool()? {
            output[index] = cached;
            index += 1;
        }
    }
    if index < size {
        output[index] =
            u16::try_from(decoder.read_literal(bit_depth)?).map_err(|_| Error::InvalidObu)?;
        index += 1;
    }
    let mut palette_bits = 0;
    if index < size {
        palette_bits = bit_depth - 3
            + u8::try_from(decoder.read_literal(2)?).map_err(|_| Error::InvalidObu)?;
    }
    while index < size {
        let mut delta =
            u32::try_from(decoder.read_literal(palette_bits)?).map_err(|_| Error::InvalidObu)?;
        if luma {
            delta = delta.checked_add(1).ok_or(Error::LimitExceeded)?;
        }
        let value = u32::from(output[index - 1])
            .saturating_add(delta)
            .min(maximum - 1);
        output[index] = u16::try_from(value).map_err(|_| Error::LimitExceeded)?;
        let range = maximum - value - u32::from(luma);
        palette_bits = palette_bits.min(ceil_log2(range));
        index += 1;
    }
    output[..size].sort_unstable();
    Ok(())
}

fn read_v_plane(
    decoder: &mut SymbolDecoder<'_>,
    bit_depth: u8,
    size: usize,
    output: &mut [u16; MAX_PALETTE_SIZE],
) -> Result<(), Error> {
    let maximum = 1i32 << bit_depth;
    if decoder.read_bool()? {
        let palette_bits = bit_depth - 4
            + u8::try_from(decoder.read_literal(2)?).map_err(|_| Error::InvalidObu)?;
        output[0] =
            u16::try_from(decoder.read_literal(bit_depth)?).map_err(|_| Error::InvalidObu)?;
        for index in 1..size {
            let delta = i32::try_from(decoder.read_literal(palette_bits)?)
                .map_err(|_| Error::InvalidObu)?;
            let signed_delta = if delta != 0 && decoder.read_bool()? {
                -delta
            } else {
                delta
            };
            let value = (i32::from(output[index - 1]) + signed_delta).rem_euclid(maximum);
            output[index] = u16::try_from(value).map_err(|_| Error::LimitExceeded)?;
        }
    } else {
        for value in &mut output[..size] {
            *value =
                u16::try_from(decoder.read_literal(bit_depth)?).map_err(|_| Error::InvalidObu)?;
        }
    }
    Ok(())
}

const fn ceil_log2(value: u32) -> u8 {
    if value <= 1 {
        0
    } else {
        (u32::BITS - (value - 1).leading_zeros()) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_cache_is_a_sorted_unique_merge() {
        let mut cache = [0; MAX_PALETTE_CACHE];
        let count = merge_palette_cache(&[1, 4, 4, 9], &[2, 4, 7, 9], &mut cache).unwrap();
        assert_eq!(&cache[..count], &[1, 2, 4, 7, 9]);
        assert_eq!(
            merge_palette_cache(&[0; 9], &[], &mut cache),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn palette_bit_width_rounding_covers_zero_and_non_powers() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(256), 8);
    }

    #[test]
    fn palette_color_context_orders_neighbors_by_normative_scores() {
        let map = [2, 0, 1, 0, 1, 0, 0, 0, 0];
        let (context, order) = palette_color_context(&map, 3, 0, 1, 3).unwrap();
        assert_eq!(context, 0);
        assert_eq!(&order[..3], &[2, 0, 1]);

        let (context, order) = palette_color_context(&map, 3, 1, 1, 3).unwrap();
        assert_eq!(context, 3);
        assert_eq!(&order[..3], &[0, 2, 1]);
        assert_eq!(
            palette_color_context(&[8], 1, 0, 0, 8),
            Err(Error::InvalidObu)
        );
    }
}
