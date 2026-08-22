//! Normative intra prediction primitives.

use crate::Error;
use crate::reconstruction::Plane;
use mrml_runtime::Vector;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasicIntraMode {
    Dc,
    Vertical,
    Horizontal,
    Paeth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmoothMode {
    Both,
    Vertical,
    Horizontal,
}

const SMOOTH_4: [u16; 4] = [255, 149, 85, 64];
const SMOOTH_8: [u16; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
const SMOOTH_16: [u16; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
const SMOOTH_32: [u16; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
const SMOOTH_64: [u16; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133,
    127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38,
    35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

const DIRECTIONAL_DERIVATIVE: [u16; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

const FILTER_INTRA_TAPS: [[[i16; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

pub struct IntraEdges<'a> {
    pub above: Option<&'a [u16]>,
    pub left: Option<&'a [u16]>,
    pub top_left: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictionRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChromaFromLumaConfig {
    pub region: PredictionRegion,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub alpha: i8,
    pub bit_depth: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntraPredictionConfig {
    pub region: PredictionRegion,
    pub bit_depth: u8,
    /// Numeric AV1 intra prediction mode (`DC_PRED` through `PAETH_PRED`).
    pub mode: u8,
    pub angle_delta: i8,
    pub filter_intra_mode: Option<u8>,
    pub have_left: bool,
    pub have_above: bool,
    pub have_above_right: bool,
    pub have_below_left: bool,
}

/// Prepares normative intra edges and dispatches section 7.11.2 for one
/// transform-sized prediction region.
pub fn predict_intra_block(plane: &mut Plane, config: IntraPredictionConfig) -> Result<(), Error> {
    let region = config.region;
    if region.width == 0
        || region.height == 0
        || !matches!(config.bit_depth, 8 | 10 | 12)
        || region
            .x
            .checked_add(region.width)
            .is_none_or(|end| end > plane.width())
        || region
            .y
            .checked_add(region.height)
            .is_none_or(|end| end > plane.height())
        || (config.have_left && region.x == 0)
        || (config.have_above && region.y == 0)
        || config.angle_delta.unsigned_abs() > 3
    {
        return Err(Error::InvalidObu);
    }
    let edge_length = region
        .width
        .checked_add(region.height)
        .ok_or(Error::LimitExceeded)?;
    let midpoint = 1u16 << (config.bit_depth - 1);
    let mut above = Vector::with_capacity(edge_length).map_err(|_| Error::LimitExceeded)?;
    let mut left = Vector::with_capacity(edge_length).map_err(|_| Error::LimitExceeded)?;
    for index in 0..edge_length {
        let above_sample = if config.have_above {
            let limit = region
                .x
                .checked_add(if config.have_above_right {
                    region.width.saturating_mul(2)
                } else {
                    region.width
                })
                .and_then(|value| value.checked_sub(1))
                .ok_or(Error::LimitExceeded)?
                .min(plane.width() - 1);
            plane.sample(region.x.saturating_add(index).min(limit), region.y - 1)?
        } else if config.have_left {
            plane.sample(region.x - 1, region.y)?
        } else {
            midpoint - 1
        };
        above
            .try_push(above_sample)
            .map_err(|_| Error::LimitExceeded)?;
        let left_sample = if config.have_left {
            let limit = region
                .y
                .checked_add(if config.have_below_left {
                    region.height.saturating_mul(2)
                } else {
                    region.height
                })
                .and_then(|value| value.checked_sub(1))
                .ok_or(Error::LimitExceeded)?
                .min(plane.height() - 1);
            plane.sample(region.x - 1, region.y.saturating_add(index).min(limit))?
        } else if config.have_above {
            plane.sample(region.x, region.y - 1)?
        } else {
            midpoint + 1
        };
        left.try_push(left_sample)
            .map_err(|_| Error::LimitExceeded)?;
    }
    let top_left = if config.have_above && config.have_left {
        plane.sample(region.x - 1, region.y - 1)?
    } else if config.have_above {
        plane.sample(region.x, region.y - 1)?
    } else if config.have_left {
        plane.sample(region.x - 1, region.y)?
    } else {
        midpoint
    };
    let edges = || IntraEdges {
        above: Some(&above),
        left: Some(&left),
        top_left,
    };
    if let Some(filter_mode) = config.filter_intra_mode {
        return predict_filter_intra(plane, region, filter_mode, config.bit_depth, edges());
    }
    match config.mode {
        0 => predict_basic(plane, region, config.bit_depth, BasicIntraMode::Dc, edges()),
        1..=8 => {
            const ANGLES: [u16; 9] = [0, 90, 180, 45, 135, 113, 157, 203, 67];
            let angle =
                i32::from(ANGLES[usize::from(config.mode)]) + i32::from(config.angle_delta) * 3;
            predict_directional(
                plane,
                region,
                u16::try_from(angle).map_err(|_| Error::InvalidObu)?,
                edges(),
            )
        }
        9 => predict_smooth(plane, region, SmoothMode::Both, edges()),
        10 => predict_smooth(plane, region, SmoothMode::Vertical, edges()),
        11 => predict_smooth(plane, region, SmoothMode::Horizontal, edges()),
        12 => predict_basic(
            plane,
            region,
            config.bit_depth,
            BasicIntraMode::Paeth,
            edges(),
        ),
        _ => Err(Error::InvalidObu),
    }
}

pub fn predict_basic(
    plane: &mut Plane,
    region: PredictionRegion,
    bit_depth: u8,
    mode: BasicIntraMode,
    edges: IntraEdges<'_>,
) -> Result<(), Error> {
    let PredictionRegion {
        x,
        y,
        width,
        height,
    } = region;
    if width == 0
        || height == 0
        || x.checked_add(width).is_none_or(|end| end > plane.width())
        || y.checked_add(height).is_none_or(|end| end > plane.height())
        || !matches!(bit_depth, 8 | 10 | 12)
    {
        return Err(Error::InvalidObu);
    }
    if edges.above.is_some_and(|above| above.len() < width)
        || edges.left.is_some_and(|left| left.len() < height)
    {
        return Err(Error::InvalidObu);
    }
    let dc = if mode == BasicIntraMode::Dc {
        Some(dc_value(edges.above, edges.left, width, height, bit_depth)?)
    } else {
        None
    };
    for row in 0..height {
        for column in 0..width {
            let value = match mode {
                BasicIntraMode::Dc => dc.unwrap(),
                BasicIntraMode::Vertical => *edges
                    .above
                    .ok_or(Error::InvalidObu)?
                    .get(column)
                    .ok_or(Error::InvalidObu)?,
                BasicIntraMode::Horizontal => *edges
                    .left
                    .ok_or(Error::InvalidObu)?
                    .get(row)
                    .ok_or(Error::InvalidObu)?,
                BasicIntraMode::Paeth => paeth(
                    *edges
                        .left
                        .ok_or(Error::InvalidObu)?
                        .get(row)
                        .ok_or(Error::InvalidObu)?,
                    *edges
                        .above
                        .ok_or(Error::InvalidObu)?
                        .get(column)
                        .ok_or(Error::InvalidObu)?,
                    edges.top_left,
                ),
            };
            plane.set_sample(x + column, y + row, value)?;
        }
    }
    Ok(())
}

pub fn predict_smooth(
    plane: &mut Plane,
    region: PredictionRegion,
    mode: SmoothMode,
    edges: IntraEdges<'_>,
) -> Result<(), Error> {
    let PredictionRegion {
        x,
        y,
        width,
        height,
    } = region;
    let above = edges.above.ok_or(Error::InvalidObu)?;
    let left = edges.left.ok_or(Error::InvalidObu)?;
    if above.len() < width
        || left.len() < height
        || x.checked_add(width).is_none_or(|end| end > plane.width())
        || y.checked_add(height).is_none_or(|end| end > plane.height())
    {
        return Err(Error::InvalidObu);
    }
    let horizontal_weights = smooth_weights(width)?;
    let vertical_weights = smooth_weights(height)?;
    let right = u32::from(above[width - 1]);
    let bottom = u32::from(left[height - 1]);
    for row in 0..height {
        for column in 0..width {
            let above_sample = u32::from(above[column]);
            let left_sample = u32::from(left[row]);
            let horizontal_weight = u32::from(horizontal_weights[column]);
            let vertical_weight = u32::from(vertical_weights[row]);
            let value = match mode {
                SmoothMode::Both => {
                    (vertical_weight * above_sample
                        + (256 - vertical_weight) * bottom
                        + horizontal_weight * left_sample
                        + (256 - horizontal_weight) * right
                        + 256)
                        >> 9
                }
                SmoothMode::Vertical => {
                    (vertical_weight * above_sample + (256 - vertical_weight) * bottom + 128) >> 8
                }
                SmoothMode::Horizontal => {
                    (horizontal_weight * left_sample + (256 - horizontal_weight) * right + 128) >> 8
                }
            };
            plane.set_sample(
                x + column,
                y + row,
                u16::try_from(value).map_err(|_| Error::InvalidObu)?,
            )?;
        }
    }
    Ok(())
}

pub fn predict_directional(
    plane: &mut Plane,
    region: PredictionRegion,
    angle: u16,
    edges: IntraEdges<'_>,
) -> Result<(), Error> {
    let PredictionRegion {
        x,
        y,
        width,
        height,
    } = region;
    if x.checked_add(width).is_none_or(|end| end > plane.width())
        || y.checked_add(height).is_none_or(|end| end > plane.height())
        || angle == 0
        || angle >= 270
    {
        return Err(Error::InvalidObu);
    }
    let above = edges.above.ok_or(Error::InvalidObu)?;
    let left = edges.left.ok_or(Error::InvalidObu)?;
    if angle == 90 {
        return predict_basic(plane, region, 8, BasicIntraMode::Vertical, edges);
    }
    if angle == 180 {
        return predict_basic(plane, region, 8, BasicIntraMode::Horizontal, edges);
    }
    let dx = if angle < 90 {
        derivative(angle)?
    } else if angle < 180 {
        derivative(180 - angle)?
    } else {
        0
    };
    let dy = if angle > 90 && angle < 180 {
        derivative(angle - 90)?
    } else if angle > 180 {
        derivative(270 - angle)?
    } else {
        0
    };
    for row in 0..height {
        for column in 0..width {
            let value = if angle < 90 {
                let index = i32::try_from(row + 1).map_err(|_| Error::LimitExceeded)? * dx;
                let base = index / 64 + i32::try_from(column).map_err(|_| Error::LimitExceeded)?;
                let maximum =
                    i32::try_from(width + height - 1).map_err(|_| Error::LimitExceeded)?;
                if base >= maximum {
                    directional_edge(above, edges.top_left, maximum)?
                } else {
                    interpolate(above, edges.top_left, base, (index >> 1) & 31)?
                }
            } else if angle < 180 {
                let index = i32::try_from(column).map_err(|_| Error::LimitExceeded)? * 64
                    - i32::try_from(row + 1).map_err(|_| Error::LimitExceeded)? * dx;
                let base = index >> 6;
                if base >= -1 {
                    interpolate(above, edges.top_left, base, (index >> 1) & 31)?
                } else {
                    let index = i32::try_from(row).map_err(|_| Error::LimitExceeded)? * 64
                        - i32::try_from(column + 1).map_err(|_| Error::LimitExceeded)? * dy;
                    interpolate(left, edges.top_left, index >> 6, (index >> 1) & 31)?
                }
            } else {
                let index = i32::try_from(column + 1).map_err(|_| Error::LimitExceeded)? * dy;
                let base = index / 64 + i32::try_from(row).map_err(|_| Error::LimitExceeded)?;
                interpolate(left, edges.top_left, base, (index >> 1) & 31)?
            };
            plane.set_sample(x + column, y + row, value)?;
        }
    }
    Ok(())
}

pub fn predict_filter_intra(
    plane: &mut Plane,
    region: PredictionRegion,
    filter_mode: u8,
    bit_depth: u8,
    edges: IntraEdges<'_>,
) -> Result<(), Error> {
    let PredictionRegion {
        x,
        y,
        width,
        height,
    } = region;
    let taps = FILTER_INTRA_TAPS
        .get(usize::from(filter_mode))
        .ok_or(Error::InvalidObu)?;
    let above = edges.above.ok_or(Error::InvalidObu)?;
    let left = edges.left.ok_or(Error::InvalidObu)?;
    if width == 0
        || height == 0
        || !width.is_multiple_of(4)
        || !height.is_multiple_of(2)
        || above.len() < width
        || left.len() < height
        || !matches!(bit_depth, 8 | 10 | 12)
        || x.checked_add(width).is_none_or(|end| end > plane.width())
        || y.checked_add(height).is_none_or(|end| end > plane.height())
    {
        return Err(Error::InvalidObu);
    }
    let maximum = (1i32 << bit_depth) - 1;
    for row_pair in 0..height / 2 {
        for column_group in 0..width / 4 {
            let mut neighbors = [0u16; 7];
            for (index, neighbor) in neighbors.iter_mut().enumerate().take(5) {
                *neighbor = if row_pair == 0 {
                    let column = column_group * 4 + index;
                    if column == 0 {
                        edges.top_left
                    } else {
                        above[column - 1]
                    }
                } else if column_group == 0 && index == 0 {
                    left[row_pair * 2 - 1]
                } else {
                    plane.sample(x + column_group * 4 + index - 1, y + row_pair * 2 - 1)?
                };
            }
            for (index, neighbor) in neighbors.iter_mut().enumerate().skip(5) {
                *neighbor = if column_group == 0 {
                    left[row_pair * 2 + index - 5]
                } else {
                    plane.sample(x + column_group * 4 - 1, y + row_pair * 2 + index - 5)?
                };
            }
            for inner_row in 0..2 {
                for inner_column in 0..4 {
                    let coefficients = taps[inner_row * 4 + inner_column];
                    let sum = coefficients
                        .iter()
                        .zip(neighbors)
                        .map(|(coefficient, sample)| i32::from(*coefficient) * i32::from(sample))
                        .sum::<i32>();
                    let predicted = round2_signed(sum, 4).clamp(0, maximum) as u16;
                    plane.set_sample(
                        x + column_group * 4 + inner_column,
                        y + row_pair * 2 + inner_row,
                        predicted,
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub fn predict_chroma_from_luma(
    luma: &Plane,
    chroma: &mut Plane,
    config: ChromaFromLumaConfig,
) -> Result<(), Error> {
    let ChromaFromLumaConfig {
        region,
        subsampling_x,
        subsampling_y,
        alpha,
        bit_depth,
    } = config;
    let PredictionRegion {
        x,
        y,
        width,
        height,
    } = region;
    if width == 0
        || height == 0
        || !width.is_power_of_two()
        || !height.is_power_of_two()
        || alpha.unsigned_abs() > 16
        || !matches!(bit_depth, 8 | 10 | 12)
        || x.checked_add(width).is_none_or(|end| end > chroma.width())
        || y.checked_add(height)
            .is_none_or(|end| end > chroma.height())
    {
        return Err(Error::InvalidObu);
    }
    let sub_x = usize::from(subsampling_x);
    let sub_y = usize::from(subsampling_y);
    let mut luma_sum = 0u64;
    for row in 0..height {
        for column in 0..width {
            luma_sum = luma_sum
                .checked_add(u64::from(subsampled_luma(
                    luma, x, y, column, row, sub_x, sub_y,
                )?))
                .ok_or(Error::LimitExceeded)?;
        }
    }
    let area_log2 = width.trailing_zeros() + height.trailing_zeros();
    let luma_average = if area_log2 == 0 {
        luma_sum
    } else {
        (luma_sum + (1u64 << (area_log2 - 1))) >> area_log2
    } as i32;
    let maximum = (1i32 << bit_depth) - 1;
    for row in 0..height {
        for column in 0..width {
            let luma_value = i32::from(subsampled_luma(luma, x, y, column, row, sub_x, sub_y)?);
            let scaled = round2_signed(i32::from(alpha) * (luma_value - luma_average), 6);
            let predicted = i32::from(chroma.sample(x + column, y + row)?);
            chroma.set_sample(
                x + column,
                y + row,
                predicted.saturating_add(scaled).clamp(0, maximum) as u16,
            )?;
        }
    }
    Ok(())
}

fn subsampled_luma(
    luma: &Plane,
    start_x: usize,
    start_y: usize,
    column: usize,
    row: usize,
    sub_x: usize,
    sub_y: usize,
) -> Result<u16, Error> {
    let sample_width = 1usize << sub_x;
    let sample_height = 1usize << sub_y;
    if luma.width() < sample_width || luma.height() < sample_height {
        return Err(Error::InvalidObu);
    }
    let luma_x = ((start_x + column) << sub_x).min(luma.width() - sample_width);
    let luma_y = ((start_y + row) << sub_y).min(luma.height() - sample_height);
    let mut sum = 0u32;
    for offset_y in 0..sample_height {
        for offset_x in 0..sample_width {
            sum += u32::from(luma.sample(luma_x + offset_x, luma_y + offset_y)?);
        }
    }
    let scaled = sum << (3 - sub_x - sub_y);
    u16::try_from(scaled).map_err(|_| Error::InvalidObu)
}

fn round2_signed(value: i32, shift: u8) -> i32 {
    if value < 0 {
        -(((value.unsigned_abs() + (1 << (shift - 1))) >> shift) as i32)
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

fn derivative(angle: u16) -> Result<i32, Error> {
    let value = *DIRECTIONAL_DERIVATIVE
        .get(usize::from(angle))
        .ok_or(Error::InvalidObu)?;
    if value == 0 {
        return Err(Error::InvalidObu);
    }
    Ok(i32::from(value))
}

fn interpolate(edge: &[u16], top_left: u16, base: i32, shift: i32) -> Result<u16, Error> {
    if !(-1..=31).contains(&shift) {
        return Err(Error::InvalidObu);
    }
    let first = directional_edge(edge, top_left, base)?;
    let second = directional_edge(edge, top_left, base + 1)?;
    let value = (i32::from(first) * (32 - shift) + i32::from(second) * shift + 16) >> 5;
    u16::try_from(value).map_err(|_| Error::InvalidObu)
}

fn directional_edge(edge: &[u16], top_left: u16, index: i32) -> Result<u16, Error> {
    if index == -1 {
        Ok(top_left)
    } else {
        let index = usize::try_from(index).map_err(|_| Error::InvalidObu)?;
        edge.get(index).copied().ok_or(Error::InvalidObu)
    }
}

pub fn filter_corner(left: u16, corner: u16, above: u16) -> u16 {
    ((u32::from(left) * 5 + u32::from(corner) * 6 + u32::from(above) * 5 + 8) >> 4) as u16
}

pub fn edge_filter_strength(width: usize, height: usize, filter_type: bool, delta: i16) -> u8 {
    let distance = delta.unsigned_abs();
    let extent = width.saturating_add(height);
    if !filter_type {
        if extent <= 8 {
            u8::from(distance >= 56)
        } else if extent <= 16 {
            u8::from(distance >= 40)
        } else if extent <= 24 {
            if distance >= 32 {
                3
            } else if distance >= 16 {
                2
            } else {
                u8::from(distance >= 8)
            }
        } else if extent <= 32 {
            if distance >= 32 {
                3
            } else if distance >= 4 {
                2
            } else {
                1
            }
        } else {
            3
        }
    } else if extent <= 8 {
        if distance >= 64 {
            2
        } else {
            u8::from(distance >= 40)
        }
    } else if extent <= 16 {
        if distance >= 48 {
            2
        } else {
            u8::from(distance >= 20)
        }
    } else if extent <= 24 {
        3 * u8::from(distance >= 4)
    } else {
        3
    }
}

pub fn should_upsample_edge(width: usize, height: usize, filter_type: bool, delta: i16) -> bool {
    let distance = delta.unsigned_abs();
    distance > 0
        && distance < 40
        && if filter_type {
            width.saturating_add(height) <= 8
        } else {
            width.saturating_add(height) <= 16
        }
}

pub fn filter_edge(corner: u16, samples: &mut [u16], strength: u8) -> Result<(), Error> {
    const KERNELS: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    if strength == 0 {
        return Ok(());
    }
    if strength > 3 || samples.len() + 1 > 129 {
        return Err(Error::InvalidObu);
    }
    let size = samples.len() + 1;
    let mut edge = [0u16; 129];
    edge[0] = corner;
    edge[1..size].copy_from_slice(samples);
    let kernel = KERNELS[usize::from(strength - 1)];
    for index in 1..size {
        let mut sum = 0i32;
        for (tap, coefficient) in kernel.iter().enumerate() {
            let source = (index + tap).saturating_sub(2).min(size - 1);
            sum += coefficient * i32::from(edge[source]);
        }
        samples[index - 1] = ((sum + 8) >> 4) as u16;
    }
    Ok(())
}

/// Upsample an edge into `output`, whose indices map from logical `-2` at
/// offset zero through logical `2*num_px-2` at its final offset.
pub fn upsample_edge(
    corner: u16,
    samples: &[u16],
    output: &mut [u16],
    bit_depth: u8,
) -> Result<(), Error> {
    if samples.is_empty()
        || samples.len() > 129
        || output.len() != samples.len() * 2 + 1
        || !matches!(bit_depth, 8 | 10 | 12)
    {
        return Err(Error::InvalidObu);
    }
    let mut duplicate = [0u16; 132];
    duplicate[0] = corner;
    duplicate[1] = corner;
    duplicate[2..samples.len() + 2].copy_from_slice(samples);
    duplicate[samples.len() + 2] = samples[samples.len() - 1];
    output[0] = corner;
    let maximum = (1i32 << bit_depth) - 1;
    for index in 0..samples.len() {
        let sum = -i32::from(duplicate[index])
            + 9 * i32::from(duplicate[index + 1])
            + 9 * i32::from(duplicate[index + 2])
            - i32::from(duplicate[index + 3]);
        output[2 * index + 1] = ((sum + 8) >> 4).clamp(0, maximum) as u16;
        output[2 * index + 2] = duplicate[index + 2];
    }
    Ok(())
}

fn smooth_weights(size: usize) -> Result<&'static [u16], Error> {
    match size {
        4 => Ok(&SMOOTH_4),
        8 => Ok(&SMOOTH_8),
        16 => Ok(&SMOOTH_16),
        32 => Ok(&SMOOTH_32),
        64 => Ok(&SMOOTH_64),
        _ => Err(Error::InvalidObu),
    }
}

fn dc_value(
    above: Option<&[u16]>,
    left: Option<&[u16]>,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Result<u16, Error> {
    let (sum, count) = match (above, left) {
        (Some(above), Some(left)) => {
            let above_sum = above[..width]
                .iter()
                .map(|value| u64::from(*value))
                .sum::<u64>();
            let left_sum = left[..height]
                .iter()
                .map(|value| u64::from(*value))
                .sum::<u64>();
            (above_sum + left_sum, width + height)
        }
        (Some(above), None) => (
            above[..width].iter().map(|value| u64::from(*value)).sum(),
            width,
        ),
        (None, Some(left)) => (
            left[..height].iter().map(|value| u64::from(*value)).sum(),
            height,
        ),
        (None, None) => return Ok(1u16 << (bit_depth - 1)),
    };
    let rounded = sum
        .checked_add((count / 2) as u64)
        .ok_or(Error::LimitExceeded)?
        / count as u64;
    u16::try_from(rounded).map_err(|_| Error::InvalidObu)
}

fn paeth(left: u16, above: u16, top_left: u16) -> u16 {
    let base = i32::from(above) + i32::from(left) - i32::from(top_left);
    let left_distance = (base - i32::from(left)).unsigned_abs();
    let above_distance = (base - i32::from(above)).unsigned_abs();
    let corner_distance = (base - i32::from(top_left)).unsigned_abs();
    if left_distance <= above_distance && left_distance <= corner_distance {
        left
    } else if above_distance <= corner_distance {
        above
    } else {
        top_left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_intra_dispatch_prepares_normative_missing_edges() {
        let mut plane = Plane::new(4, 4, 0).unwrap();
        predict_intra_block(
            &mut plane,
            IntraPredictionConfig {
                region: PredictionRegion {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                },
                bit_depth: 8,
                mode: 0,
                angle_delta: 0,
                filter_intra_mode: None,
                have_left: false,
                have_above: false,
                have_above_right: false,
                have_below_left: false,
            },
        )
        .unwrap();
        assert_eq!(plane.sample(0, 0), Ok(128));
        assert_eq!(plane.sample(1, 1), Ok(128));

        plane.set_sample(0, 2, 55).unwrap();
        predict_intra_block(
            &mut plane,
            IntraPredictionConfig {
                region: PredictionRegion {
                    x: 1,
                    y: 2,
                    width: 2,
                    height: 2,
                },
                bit_depth: 8,
                mode: 1,
                angle_delta: 0,
                filter_intra_mode: None,
                have_left: true,
                have_above: false,
                have_above_right: false,
                have_below_left: false,
            },
        )
        .unwrap();
        assert_eq!(plane.sample(1, 2), Ok(55));
    }

    #[test]
    fn dc_averages_both_edges_with_rounding() {
        let mut plane = Plane::new(2, 2, 0).unwrap();
        predict_basic(
            &mut plane,
            PredictionRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            8,
            BasicIntraMode::Dc,
            IntraEdges {
                above: Some(&[10, 20]),
                left: Some(&[30, 41]),
                top_left: 0,
            },
        )
        .unwrap();
        assert_eq!(plane.samples(), &[25, 25, 25, 25]);
    }

    #[test]
    fn paeth_uses_normative_tie_order() {
        assert_eq!(paeth(20, 20, 10), 20);
        assert_eq!(paeth(30, 10, 20), 20);
    }

    #[test]
    fn absent_dc_edges_use_bit_depth_midpoint() {
        assert_eq!(dc_value(None, None, 4, 4, 12), Ok(2048));
    }

    #[test]
    fn smooth_modes_use_normative_endpoint_weights() {
        let above = [100; 4];
        let left = [200; 4];
        for (mode, expected) in [
            (SmoothMode::Both, 150),
            (SmoothMode::Vertical, 100),
            (SmoothMode::Horizontal, 200),
        ] {
            let mut plane = Plane::new(4, 4, 0).unwrap();
            predict_smooth(
                &mut plane,
                PredictionRegion {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                mode,
                IntraEdges {
                    above: Some(&above),
                    left: Some(&left),
                    top_left: 0,
                },
            )
            .unwrap();
            assert_eq!(plane.sample(0, 0), Ok(expected));
        }
    }

    #[test]
    fn forty_five_degree_prediction_advances_above_edge_each_row() {
        let above = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [0; 8];
        let mut plane = Plane::new(4, 4, 0).unwrap();
        predict_directional(
            &mut plane,
            PredictionRegion {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            45,
            IntraEdges {
                above: Some(&above),
                left: Some(&left),
                top_left: 0,
            },
        )
        .unwrap();
        assert_eq!(plane.sample(0, 0), Ok(20));
        assert_eq!(plane.sample(3, 3), Ok(80));
    }

    #[test]
    fn edge_filters_preserve_constant_inputs() {
        let mut edge = [77; 8];
        filter_edge(77, &mut edge, 3).unwrap();
        assert_eq!(edge, [77; 8]);
        let mut upsampled = [0; 17];
        upsample_edge(77, &edge, &mut upsampled, 8).unwrap();
        assert_eq!(upsampled, [77; 17]);
    }

    #[test]
    fn strength_and_upsample_thresholds_match_boundaries() {
        assert_eq!(edge_filter_strength(8, 8, false, 39), 0);
        assert_eq!(edge_filter_strength(8, 8, false, 40), 1);
        assert!(should_upsample_edge(8, 8, false, 39));
        assert!(!should_upsample_edge(8, 8, false, 40));
    }

    #[test]
    fn filter_intra_preserves_constant_edges() {
        let above = [91; 8];
        let left = [91; 4];
        for mode in 0..5 {
            let mut plane = Plane::new(8, 4, 0).unwrap();
            predict_filter_intra(
                &mut plane,
                PredictionRegion {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 4,
                },
                mode,
                8,
                IntraEdges {
                    above: Some(&above),
                    left: Some(&left),
                    top_left: 91,
                },
            )
            .unwrap();
            assert!(plane.samples().iter().all(|sample| *sample == 91));
        }
    }

    #[test]
    fn chroma_from_luma_applies_signed_ac_correlation() {
        let mut luma = Plane::new(4, 2, 0).unwrap();
        for row in 0..2 {
            for column in 0..4 {
                luma.set_sample(column, row, if column < 2 { 10 } else { 30 })
                    .unwrap();
            }
        }
        let mut chroma = Plane::new(2, 1, 100).unwrap();
        predict_chroma_from_luma(
            &luma,
            &mut chroma,
            ChromaFromLumaConfig {
                region: PredictionRegion {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 1,
                },
                subsampling_x: true,
                subsampling_y: true,
                alpha: 8,
                bit_depth: 8,
            },
        )
        .unwrap();
        assert_eq!(chroma.samples(), &[90, 110]);
    }
}
