//! Constrained directional enhancement filter primitives.

use crate::{
    ChromaSampling, Error,
    block_state::MiGrid,
    params::Cdef,
    partition::TileBounds,
    reconstruction::{FrameBuffer, Plane},
};

const DIV_TABLE: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
const DIRECTIONS: [[(i32, i32); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];
const PRIMARY_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
const SECONDARY_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];
const UV_DIRECTIONS: [[[u8; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

pub fn apply_frame_region(
    frame: &mut FrameBuffer,
    grid: &MiGrid,
    tile: TileBounds,
    parameters: &Cdef,
) -> Result<(), Error> {
    let source = frame.clone();
    let bit_depth = frame.bit_depth();
    let (subsampling_x, subsampling_y, has_chroma) = match frame.sampling() {
        ChromaSampling::Cs400 => (false, false, false),
        ChromaSampling::Cs420 => (true, true, true),
        ChromaSampling::Cs422 => (true, false, true),
        ChromaSampling::Cs444 => (false, false, true),
    };
    let coefficient_shift = bit_depth - 8;
    let mut row = tile.row_start;
    while row < tile.row_end {
        let mut column = tile.column_start;
        while column < tile.column_end {
            let base_row = row & !15;
            let base_column = column & !15;
            let index = cdef_index(grid, base_row, base_column);
            let skipped = (0..2).all(|y| {
                (0..2).all(|x| grid.get(row + y, column + x).is_none_or(|state| state.skip))
            });
            if let Some(index) = index.filter(|_| !skipped) {
                let index = usize::from(index);
                let mut luma = [0u16; 64];
                let start_x = usize::try_from(column * 4).map_err(|_| Error::LimitExceeded)?;
                let start_y = usize::try_from(row * 4).map_err(|_| Error::LimitExceeded)?;
                for y in 0..8 {
                    for x in 0..8 {
                        luma[y * 8 + x] = source.y.sample(
                            (start_x + x).min(source.y.width() - 1),
                            (start_y + y).min(source.y.height() - 1),
                        )?;
                    }
                }
                let (direction, variance) = find_direction(&luma, bit_depth)?;
                let primary = u16::from(
                    *parameters
                        .y_pri_strength
                        .get(index)
                        .ok_or(Error::InvalidObu)?,
                ) << coefficient_shift;
                let secondary = u16::from(
                    *parameters
                        .y_sec_strength
                        .get(index)
                        .ok_or(Error::InvalidObu)?,
                ) << coefficient_shift;
                let variance_strength = if variance >> 6 == 0 {
                    0
                } else {
                    (31 - (variance >> 6).leading_zeros()).min(12)
                };
                let primary = if variance == 0 {
                    0
                } else {
                    u16::try_from((u32::from(primary) * (4 + variance_strength) + 8) >> 4)
                        .map_err(|_| Error::LimitExceeded)?
                };
                filter_cdef_plane(
                    &source.y,
                    &mut frame.y,
                    start_x,
                    start_y,
                    8,
                    8,
                    tile,
                    false,
                    false,
                    primary,
                    secondary,
                    parameters.damping + coefficient_shift,
                    if primary == 0 { 0 } else { direction },
                    bit_depth,
                )?;
                if has_chroma {
                    let uv_direction = UV_DIRECTIONS[usize::from(subsampling_x)]
                        [usize::from(subsampling_y)][usize::from(direction)];
                    let uv_primary =
                        u16::from(parameters.uv_pri_strength[index]) << coefficient_shift;
                    let uv_secondary =
                        u16::from(parameters.uv_sec_strength[index]) << coefficient_shift;
                    for plane in 1..=2 {
                        let (source_plane, destination) = if plane == 1 {
                            (
                                source.u.as_ref().ok_or(Error::InvalidObu)?,
                                frame.u.as_mut().ok_or(Error::InvalidObu)?,
                            )
                        } else {
                            (
                                source.v.as_ref().ok_or(Error::InvalidObu)?,
                                frame.v.as_mut().ok_or(Error::InvalidObu)?,
                            )
                        };
                        filter_cdef_plane(
                            source_plane,
                            destination,
                            start_x >> usize::from(subsampling_x),
                            start_y >> usize::from(subsampling_y),
                            8 >> usize::from(subsampling_x),
                            8 >> usize::from(subsampling_y),
                            tile,
                            subsampling_x,
                            subsampling_y,
                            uv_primary,
                            uv_secondary,
                            parameters.damping + coefficient_shift - 1,
                            if uv_primary == 0 { 0 } else { uv_direction },
                            bit_depth,
                        )?;
                    }
                }
            }
            column += 2;
        }
        row += 2;
    }
    Ok(())
}

fn cdef_index(grid: &MiGrid, base_row: u32, base_column: u32) -> Option<u8> {
    for row in base_row..base_row.saturating_add(16).min(grid.rows()) {
        for column in base_column..base_column.saturating_add(16).min(grid.columns()) {
            if let Some(index) = grid.get(row, column).and_then(|state| state.cdef_index) {
                return Some(index);
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn filter_cdef_plane(
    source: &Plane,
    destination: &mut Plane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    tile: TileBounds,
    subsampling_x: bool,
    subsampling_y: bool,
    primary_strength: u16,
    secondary_strength: u16,
    damping: u8,
    direction: u8,
    bit_depth: u8,
) -> Result<(), Error> {
    let region_x = usize::try_from((tile.column_start * 4) >> u32::from(subsampling_x))
        .map_err(|_| Error::LimitExceeded)?;
    let region_y = usize::try_from((tile.row_start * 4) >> u32::from(subsampling_y))
        .map_err(|_| Error::LimitExceeded)?;
    let region_end_x = usize::try_from((tile.column_end * 4) >> u32::from(subsampling_x))
        .map_err(|_| Error::LimitExceeded)?
        .min(source.width());
    let region_end_y = usize::try_from((tile.row_end * 4) >> u32::from(subsampling_y))
        .map_err(|_| Error::LimitExceeded)?
        .min(source.height());
    filter_block(
        source,
        destination,
        FilterConfig {
            x,
            y,
            width: width.min(source.width().saturating_sub(x)),
            height: height.min(source.height().saturating_sub(y)),
            region_x,
            region_y,
            region_width: region_end_x.saturating_sub(region_x),
            region_height: region_end_y.saturating_sub(region_y),
            primary_strength,
            secondary_strength,
            damping,
            direction,
            bit_depth,
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterConfig {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub region_x: usize,
    pub region_y: usize,
    pub region_width: usize,
    pub region_height: usize,
    pub primary_strength: u16,
    pub secondary_strength: u16,
    pub damping: u8,
    pub direction: u8,
    pub bit_depth: u8,
}

/// Detects the dominant direction and variance of one 8 by 8 luma block.
///
/// The returned direction uses the AV1 `Cdef_Directions` numbering (0..=7).
pub fn find_direction(block: &[u16; 64], bit_depth: u8) -> Result<(u8, u32), Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidObu);
    }
    let mut partial = [[0i32; 15]; 8];
    for row in 0..8usize {
        for column in 0..8usize {
            let value = i32::from(block[row * 8 + column] >> (bit_depth - 8)) - 128;
            partial[0][row + column] += value;
            partial[1][row + column / 2] += value;
            partial[2][row] += value;
            partial[3][3 + row - column / 2] += value;
            partial[4][7 + row - column] += value;
            partial[5][3 + column - row / 2] += value;
            partial[6][column] += value;
            partial[7][row / 2 + column] += value;
        }
    }

    let square = |value: i32| i64::from(value) * i64::from(value);
    let mut cost = [0i64; 8];
    for (&horizontal, &vertical) in partial[2].iter().zip(&partial[6]).take(8) {
        cost[2] += square(horizontal);
        cost[6] += square(vertical);
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for index in 0..7 {
        cost[0] +=
            (square(partial[0][index]) + square(partial[0][14 - index])) * DIV_TABLE[index + 1];
        cost[4] +=
            (square(partial[4][index]) + square(partial[4][14 - index])) * DIV_TABLE[index + 1];
    }
    cost[0] += square(partial[0][7]) * DIV_TABLE[8];
    cost[4] += square(partial[4][7]) * DIV_TABLE[8];
    for direction in [1usize, 3, 5, 7] {
        for index in 0..5 {
            cost[direction] += square(partial[direction][3 + index]);
        }
        cost[direction] *= DIV_TABLE[8];
        for index in 0..3 {
            cost[direction] += (square(partial[direction][index])
                + square(partial[direction][10 - index]))
                * DIV_TABLE[2 * index + 2];
        }
    }

    let mut direction = 0usize;
    let mut best_cost = cost[0];
    for (index, candidate) in cost.iter().enumerate().skip(1) {
        if *candidate > best_cost {
            best_cost = *candidate;
            direction = index;
        }
    }
    let variance = (best_cost - cost[(direction + 4) & 7]) >> 10;
    Ok((
        u8::try_from(direction).map_err(|_| Error::InvalidObu)?,
        u32::try_from(variance).map_err(|_| Error::InvalidObu)?,
    ))
}

/// Applies AV1's CDEF difference constraint.
pub fn constrain(diff: i32, threshold: u16, damping: u8) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let threshold = i32::from(threshold);
    let damping_adjustment = i32::from(damping)
        .saturating_sub(31 - threshold.leading_zeros() as i32)
        .max(0) as u32;
    let magnitude = diff.abs();
    diff.signum() * magnitude.min((threshold - (magnitude >> damping_adjustment)).max(0))
}

/// Filters a reconstructed block from `source` into `destination`.
///
/// The source and destination are deliberately separate because all CDEF taps
/// for a frame read the pre-CDEF reconstruction. `region_*` describes the
/// normative filter region; samples outside it are unavailable even when they
/// exist in the plane.
pub fn filter_block(
    source: &Plane,
    destination: &mut Plane,
    config: FilterConfig,
) -> Result<(), Error> {
    if !matches!(config.bit_depth, 8 | 10 | 12)
        || config.direction >= 8
        || config.width == 0
        || config.height == 0
        || config.width > 8
        || config.height > 8
        || source.width() != destination.width()
        || source.height() != destination.height()
        || !rectangle_fits(source, config.x, config.y, config.width, config.height)
        || !rectangle_fits(
            source,
            config.region_x,
            config.region_y,
            config.region_width,
            config.region_height,
        )
    {
        return Err(Error::InvalidObu);
    }
    let coefficient_shift = config.bit_depth - 8;
    let tap_set = usize::from((config.primary_strength >> coefficient_shift) & 1);
    let maximum_sample = (1u16 << config.bit_depth) - 1;
    if config.primary_strength > maximum_sample || config.secondary_strength > maximum_sample {
        return Err(Error::InvalidObu);
    }
    for row in 0..config.height {
        for column in 0..config.width {
            let x = config.x + column;
            let y = config.y + row;
            let center = source.sample(x, y)?;
            let mut sum = 0i32;
            let mut minimum = center;
            let mut maximum = center;
            for distance in 0..2usize {
                for sign in [-1i32, 1] {
                    if let Some(sample) = directional_sample(
                        source,
                        x,
                        y,
                        DIRECTIONS[usize::from(config.direction)][distance],
                        sign,
                        config,
                    )? {
                        sum += PRIMARY_TAPS[tap_set][distance]
                            * constrain(
                                i32::from(sample) - i32::from(center),
                                config.primary_strength,
                                config.damping,
                            );
                        minimum = minimum.min(sample);
                        maximum = maximum.max(sample);
                    }
                    for direction_offset in [6u8, 2] {
                        let direction = (config.direction + direction_offset) & 7;
                        if let Some(sample) = directional_sample(
                            source,
                            x,
                            y,
                            DIRECTIONS[usize::from(direction)][distance],
                            sign,
                            config,
                        )? {
                            sum += SECONDARY_TAPS[tap_set][distance]
                                * constrain(
                                    i32::from(sample) - i32::from(center),
                                    config.secondary_strength,
                                    config.damping,
                                );
                            minimum = minimum.min(sample);
                            maximum = maximum.max(sample);
                        }
                    }
                }
            }
            let adjustment = (8 + sum - i32::from(sum < 0)) >> 4;
            let filtered =
                (i32::from(center) + adjustment).clamp(i32::from(minimum), i32::from(maximum));
            destination.set_sample(
                x,
                y,
                u16::try_from(filtered).map_err(|_| Error::InvalidObu)?,
            )?;
        }
    }
    Ok(())
}

fn rectangle_fits(plane: &Plane, x: usize, y: usize, width: usize, height: usize) -> bool {
    width != 0
        && height != 0
        && x.checked_add(width).is_some_and(|end| end <= plane.width())
        && y.checked_add(height)
            .is_some_and(|end| end <= plane.height())
}

fn directional_sample(
    source: &Plane,
    x: usize,
    y: usize,
    direction: (i32, i32),
    sign: i32,
    config: FilterConfig,
) -> Result<Option<u16>, Error> {
    let candidate_x =
        i64::try_from(x).map_err(|_| Error::LimitExceeded)? + i64::from(sign * direction.1);
    let candidate_y =
        i64::try_from(y).map_err(|_| Error::LimitExceeded)? + i64::from(sign * direction.0);
    let Ok(candidate_x) = usize::try_from(candidate_x) else {
        return Ok(None);
    };
    let Ok(candidate_y) = usize::try_from(candidate_y) else {
        return Ok(None);
    };
    let region_end_x = config
        .region_x
        .checked_add(config.region_width)
        .ok_or(Error::LimitExceeded)?;
    let region_end_y = config
        .region_y
        .checked_add(config.region_height)
        .ok_or(Error::LimitExceeded)?;
    if candidate_x < config.region_x
        || candidate_x >= region_end_x
        || candidate_y < config.region_y
        || candidate_y >= region_end_y
    {
        return Ok(None);
    }
    source.sample(candidate_x, candidate_y).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        block_state::BlockState,
        partition::{BlockRect, BlockSize},
    };

    #[test]
    fn frame_region_orchestration_preserves_zero_strength_blocks() {
        let mut frame = FrameBuffer::new(8, 8, 8, ChromaSampling::Cs400).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                frame.y.set_sample(x, y, (x * 17 + y) as u16).unwrap();
            }
        }
        let original = frame.clone();
        let mut grid = MiGrid::new(2, 2).unwrap();
        grid.fill(
            BlockRect::new(0, 0, BlockSize::Block8x8),
            BlockState {
                cdef_index: Some(0),
                ..BlockState::default()
            },
        )
        .unwrap();
        apply_frame_region(
            &mut frame,
            &grid,
            TileBounds {
                column_start: 0,
                column_end: 2,
                row_start: 0,
                row_end: 2,
            },
            &Cdef::default(),
        )
        .unwrap();
        assert_eq!(frame, original);
    }

    #[test]
    fn flat_block_has_zero_variance_and_first_tie_direction() {
        assert_eq!(find_direction(&[512; 64], 10), Ok((0, 0)));
    }

    #[test]
    fn vertical_bands_select_vertical_direction() {
        let mut block = [0u16; 64];
        for row in 0..8 {
            for column in 0..8 {
                block[row * 8 + column] = if column < 4 { 32 } else { 224 };
            }
        }
        let (direction, variance) = find_direction(&block, 8).unwrap();
        assert_eq!(direction, 6);
        assert!(variance > 0);
    }

    #[test]
    fn constraint_reduces_large_differences_and_preserves_sign() {
        assert_eq!(constrain(3, 8, 3), 3);
        assert_eq!(constrain(20, 8, 3), 0);
        assert_eq!(constrain(-6, 8, 3), -2);
        assert_eq!(constrain(4, 0, 3), 0);
    }

    #[test]
    fn filtering_preserves_constant_blocks_at_high_bit_depth() {
        let source = Plane::new(8, 8, 777).unwrap();
        let mut destination = Plane::new(8, 8, 0).unwrap();
        filter_block(
            &source,
            &mut destination,
            FilterConfig {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                region_x: 0,
                region_y: 0,
                region_width: 8,
                region_height: 8,
                primary_strength: 12,
                secondary_strength: 4,
                damping: 5,
                direction: 3,
                bit_depth: 10,
            },
        )
        .unwrap();
        assert_eq!(destination.samples(), &[777; 64]);
    }

    #[test]
    fn unavailable_samples_do_not_cross_filter_region() {
        let mut source = Plane::new(4, 1, 0).unwrap();
        source.set_sample(2, 0, 255).unwrap();
        source.set_sample(3, 0, 255).unwrap();
        let mut destination = source.clone();
        filter_block(
            &source,
            &mut destination,
            FilterConfig {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
                region_x: 0,
                region_y: 0,
                region_width: 2,
                region_height: 1,
                primary_strength: 15,
                secondary_strength: 4,
                damping: 3,
                direction: 2,
                bit_depth: 8,
            },
        )
        .unwrap();
        assert_eq!(destination.samples(), source.samples());
    }
}
