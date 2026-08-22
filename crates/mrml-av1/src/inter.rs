//! Inter-prediction rounding and compound blending.

use crate::Error;
use crate::{
    mode::MotionMode,
    motion::{
        GlobalMotion, GlobalMotionType, MotionScaleInput, MotionVector, ScaledMotion, ShearParams,
        scale_motion_vector, setup_shear,
    },
};
use mrml_runtime::Vector;

#[rustfmt::skip]
const INTER_INTRA_WEIGHTS: [u8; 128] = [
    60,58,56,54,52,50,48,47,45,44,42,41,39,38,37,35,34,33,32,31,30,29,28,27,26,25,24,23,22,22,21,20,
    19,19,18,18,17,16,16,15,15,14,14,13,13,12,12,12,11,11,10,10,10,9,9,9,8,8,8,8,7,7,7,7,
    6,6,6,6,6,5,5,5,5,5,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
];

pub fn blend_inter_intra(
    plane: &mut Plane,
    region: PredictionRegion,
    mode: u8,
    inter_prediction: &[u16],
) -> Result<(), Error> {
    let count = region
        .width
        .checked_mul(region.height)
        .ok_or(Error::LimitExceeded)?;
    if region.width == 0 || region.height == 0 || inter_prediction.len() != count || mode > 3 {
        return Err(Error::InvalidObu);
    }
    let scale = 128 / region.width.max(region.height);
    if scale == 0 {
        return Err(Error::InvalidObu);
    }
    for row in 0..region.height {
        for column in 0..region.width {
            let weight_index = match mode {
                1 => row * scale,
                2 => column * scale,
                3 => row.min(column) * scale,
                _ => 18,
            };
            let weight = if mode == 0 {
                32
            } else {
                *INTER_INTRA_WEIGHTS
                    .get(weight_index)
                    .ok_or(Error::InvalidObu)?
            };
            let intra = u32::from(plane.sample(region.x + column, region.y + row)?);
            let inter = u32::from(inter_prediction[row * region.width + column]);
            let value = (u32::from(weight) * intra + u32::from(64 - weight) * inter + 32) >> 6;
            plane.set_sample(
                region.x + column,
                region.y + row,
                u16::try_from(value).map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

pub fn blend_wedge_inter_intra(
    plane: &mut Plane,
    region: PredictionRegion,
    luma_width: u16,
    luma_height: u16,
    index: u8,
    sign: bool,
    inter_prediction: &[u16],
) -> Result<(), Error> {
    let count = region
        .width
        .checked_mul(region.height)
        .ok_or(Error::LimitExceeded)?;
    if inter_prediction.len() != count {
        return Err(Error::InvalidObu);
    }
    for row in 0..region.height {
        for column in 0..region.width {
            let weight = u32::from(wedge_luma_weight(
                luma_width,
                luma_height,
                index,
                sign,
                column,
                row,
            )?);
            let intra = u32::from(plane.sample(region.x + column, region.y + row)?);
            let inter = u32::from(inter_prediction[row * region.width + column]);
            let value = (weight * intra + (64 - weight) * inter + 32) >> 6;
            plane.set_sample(
                region.x + column,
                region.y + row,
                u16::try_from(value).map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

pub fn subsample_mask(
    luma: &[u8],
    luma_width: usize,
    luma_height: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> Result<Vector<u8>, Error> {
    if luma.len()
        != luma_width
            .checked_mul(luma_height)
            .ok_or(Error::LimitExceeded)?
    {
        return Err(Error::InvalidObu);
    }
    let sx = usize::from(subsampling_x);
    let sy = usize::from(subsampling_y);
    let width = luma_width >> sx;
    let height = luma_height >> sy;
    let mut output = Vector::with_capacity(width.checked_mul(height).ok_or(Error::LimitExceeded)?)
        .map_err(|_| Error::LimitExceeded)?;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u16;
            for dy in 0..1usize << sy {
                for dx in 0..1usize << sx {
                    sum += u16::from(luma[((y << sy) + dy) * luma_width + (x << sx) + dx]);
                }
            }
            let shift = sx + sy;
            let rounding = if shift == 0 { 0 } else { 1u16 << (shift - 1) };
            output
                .try_push(((sum + rounding) >> shift) as u8)
                .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(output)
}

const WARPED_FILTER_PHASES: usize = 193;
const WARPED_FILTER_B64: &[u8] = b"AAB/AQAAAAAA/38CAAAAAAH9fwT/AAAAAfx+Bv4BAAAB+34I/QEAAAH6fQH8AQAAAfl8DfwBAAAC+HsP+wEAAAL3ehL6AQAAAvZ5FPoBAAAC/3gC+QIAAAL0dxn4AgAAA/N1G/gCAAAD83Qd9wIAAAPyciD2AwAAA/FxI/YCAAAD8W8l/wMAAAPwbSj/AwAAA/BsKvQDAAAE72ot8wMAAATvaC/zAwAABO9mMvIDAAAE72Q08gMAAATuYgXxBAAABO5gOvEDAAAE7l488AQAAATuWz/wBAAABO5ZQfAEAAAE7ldE7wQAAATuVUbvBAAABO5SSe8EAAAE7lBL7wQAAATuTk7uBAAABO9LUO4EAAAE70lS7gQAAATvRlXuBAAABO9EV+4EAAAE8EFZ7gQAAATwP1vuBAAABPA8Xu4EAAAD8Tpg7gQAAATxBWLuBAAAA/I0ZO8EAAAD8jJm7wQAAAPzL2jvBAAAA/Mtau8EAAAD9Cps8AMAAAP/KG3wAwAAA/8lb/EDAAAC9iNx8QMAAAP2IHLyAwAAAvcddPMDAAAC+Bt18wMAAAL4GXf0AgAAAvkCeP8CAAAB+hR59gIAAAH6Enr3AgAAAfsPe/gCAAAB/A18+QEAAAH8AX36AQAAAf0IfvsBAAAB/gZ+/AEAAAD/BH/9AQAAAAACf/8AAAAAAAB/AQAAAAAA/38CAAAAAAH9fwT+AQAAAft/Bv4BAAAC+n4I/QEA/wL5fgH8Av//A/h9DfsC//8D9nwQ+gP//wT/exL5A///BPR6FPkD//8E83kX+AP//gXyeBn3BP//BfF3G/YE//8F8HYe/wT//gbvdAP0Bf/+Bu9yI/QF//4G7nEm8wX//gftbynyBv7+B+1uK/EG/v4H7Gwu8Qb+/gfsajHwBv7+B+toM/AH/v4H62Y27wf+/gjrZDjuB/7+CP5iO+4H/v4I/mA+7Qf+/gj+XkDtB/7+CP5bQ+wI/v4I/llF7Aj+/gj+V0jrCP7+COtUSusI/v4I/lIH6wj+AQ2d/gjrT0/rCP7+COsHUv4I/v4I60pU6wj+/gjrSFf+CP7+COxFWf4I/v4I7ENb/gj+/gftQF7+CP7+B+0+YP4I/v4H7jti/gj+/gfuOGTrCP7+B+82ZusH/v4H8DNo6wf+/gbwMWrsB/7+BvEubOwH/v4G8Stu7Qf+/gbyKW/tB/7/BfMmce4G/v8F9CNy7wb+/wX0A3TvBv7/BP8edvAF//8E9ht38QX//wT3GXjyBf7/A/gXefME//8D+RR69AT//wP5Env/BP//A/oQfPYD//8C+w19+AP//wL8AX75Av8AAf0IfvoCAAAB/gZ/+wEAAAH+BH/9AQAAAAACf/8AAAAAAAF/AAAAAAAA/38CAAAAAAH9fwT/AAAAAfx+Bv4BAAAB+34I/QEAAAH6fQH8AQAAAfl8DfwBAAAC+HsP+wEAAAL3ehL6AQAAAvZ5FPoBAAAC/3gC+QIAAAL0dxn4AgAAA/N1G/gCAAAD83Qd9wIAAAPyciD2AwAAA/FxI/YCAAAD8W8l/wMAAAPwbSj/AwAAA/BsKvQDAAAE72ot8wMAAATvaC/zAwAABO9mMvIDAAAE72Q08gMAAATuYgXxBAAABO5gOvEDAAAE7l488AQAAATuWz/wBAAABO5ZQfAEAAAE7ldE7wQAAATuVUbvBAAABO5SSe8EAAAE7lBL7wQAAATuTk7uBAAABO9LUO4EAAAE70lS7gQAAATvRlXuBAAABO9EV+4EAAAE8EFZ7gQAAATwP1vuBAAABPA8Xu4EAAAD8Tpg7gQAAATxBWLuBAAAA/I0ZO8EAAAD8jJm7wQAAAPzL2jvBAAAA/Mtau8EAAAD9Cps8AMAAAP/KG3wAwAAA/8lb/EDAAAC9iNx8QMAAAP2IHLyAwAAAvcddPMDAAAC+Bt18wMAAAL4GXf0AgAAAvkCeP8CAAAB+hR59gIAAAH6Enr3AgAAAfsPe/gCAAAB/A18+QEAAAH8AX36AQAAAf0IfvsBAAAB/gZ+/AEAAAD/BH/9AQAAAAACf/8AAQ6dAAA=";

pub fn warped_filter(phase: i16) -> Result<[i16; 8], Error> {
    let phase = phase.checked_add(64).ok_or(Error::InvalidObu)?;
    let phase = usize::try_from(phase).map_err(|_| Error::InvalidObu)?;
    if phase >= WARPED_FILTER_PHASES {
        return Err(Error::InvalidObu);
    }
    let start = phase * 8;
    let mut filter = [0i16; 8];
    for (offset, coefficient) in filter.iter_mut().enumerate() {
        *coefficient = i16::from(warped_filter_byte(start + offset)? as i8);
    }
    Ok(filter)
}

fn warped_filter_byte(index: usize) -> Result<u8, Error> {
    if index >= WARPED_FILTER_PHASES * 8 {
        return Err(Error::InvalidObu);
    }
    let quartet = (index / 3) * 4;
    let a = u32::from(base64_value(WARPED_FILTER_B64[quartet])?);
    let b = u32::from(base64_value(WARPED_FILTER_B64[quartet + 1])?);
    let c = u32::from(base64_value(WARPED_FILTER_B64[quartet + 2])?);
    let d = u32::from(base64_value(WARPED_FILTER_B64[quartet + 3])?);
    let word = (a << 18) | (b << 12) | (c << 6) | d;
    Ok(match index % 3 {
        0 => (word >> 16) as u8,
        1 => (word >> 8) as u8,
        _ => word as u8,
    })
}

const fn base64_value(byte: u8) -> Result<u8, Error> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        b'=' => Ok(0),
        _ => Err(Error::InvalidObu),
    }
}

const OBMC_MASK_2: [u8; 2] = [45, 64];
const OBMC_MASK_4: [u8; 4] = [39, 50, 59, 64];
const OBMC_MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
const OBMC_MASK_16: [u8; 16] = [
    34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
];
const OBMC_MASK_32: [u8; 32] = [
    33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
];

pub fn obmc_mask(length: usize) -> Result<&'static [u8], Error> {
    match length {
        2 => Ok(&OBMC_MASK_2),
        4 => Ok(&OBMC_MASK_4),
        8 => Ok(&OBMC_MASK_8),
        16 => Ok(&OBMC_MASK_16),
        32 => Ok(&OBMC_MASK_32),
        _ => Err(Error::InvalidObu),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObmcPass {
    Above,
    Left,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObmcTraversalConfig {
    pub block: crate::partition::BlockRect,
    pub tile: crate::partition::TileBounds,
    pub prediction_width: u16,
    pub prediction_height: u16,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub residual_at_least_8x8: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObmcNeighbor {
    pub pass: ObmcPass,
    pub candidate_row: u32,
    pub candidate_column: u32,
    pub region: PredictionRegion,
    pub reference_frame: i8,
    pub motion_vector: crate::motion::MotionVector,
}

/// Enumerates section 7.11.3.9's above and left OBMC candidates in normative
/// order. Prediction and blending stay in the callback so reference lookup and
/// scaling remain owned by the frame decoder.
pub fn walk_obmc_neighbors<F>(
    grid: &crate::block_state::MiGrid,
    config: ObmcTraversalConfig,
    mut visit: F,
) -> Result<(), Error>
where
    F: FnMut(ObmcNeighbor) -> Result<(), Error>,
{
    if config.prediction_width == 0 || config.prediction_height == 0 {
        return Err(Error::InvalidObu);
    }
    let block = config.block;
    let sub_x = u32::from(config.subsampling_x);
    let sub_y = u32::from(config.subsampling_y);
    if config.residual_at_least_8x8 && block.row > config.tile.row_start {
        let mut column = block.column;
        let end = block
            .column
            .checked_add(u32::from(block.width_mi))
            .ok_or(Error::LimitExceeded)?
            .min(grid.columns())
            .min(config.tile.column_end);
        let limit = 4usize.min(usize::from(block.width_mi.ilog2() as u8));
        let mut count = 0usize;
        while count < limit && column < end {
            let candidate_row = block.row - 1;
            let candidate_column = (column | 1).min(end - 1);
            let state = grid
                .get(candidate_row, candidate_column)
                .ok_or(Error::InvalidObu)?;
            let size = state.size.ok_or(Error::InvalidObu)?;
            let step = u32::from(size.dimensions().0 / 4).clamp(2, 16);
            if state.reference_frames[0] > 0 {
                count += 1;
                let width = u32::from(config.prediction_width).min((step * 4) >> sub_x);
                let height = (u32::from(config.prediction_height) >> 1).min(32 >> sub_y);
                visit(ObmcNeighbor {
                    pass: ObmcPass::Above,
                    candidate_row,
                    candidate_column,
                    region: PredictionRegion {
                        x: usize::try_from((column * 4) >> sub_x)
                            .map_err(|_| Error::LimitExceeded)?,
                        y: usize::try_from((block.row * 4) >> sub_y)
                            .map_err(|_| Error::LimitExceeded)?,
                        width: usize::try_from(width).map_err(|_| Error::LimitExceeded)?,
                        height: usize::try_from(height).map_err(|_| Error::LimitExceeded)?,
                    },
                    reference_frame: state.reference_frames[0],
                    motion_vector: state.motion_vectors[0],
                })?;
            }
            column = column.checked_add(step).ok_or(Error::LimitExceeded)?;
        }
    }
    if block.column > config.tile.column_start {
        let mut row = block.row;
        let end = block
            .row
            .checked_add(u32::from(block.height_mi))
            .ok_or(Error::LimitExceeded)?
            .min(grid.rows())
            .min(config.tile.row_end);
        let limit = 4usize.min(usize::from(block.height_mi.ilog2() as u8));
        let mut count = 0usize;
        while count < limit && row < end {
            let candidate_row = (row | 1).min(end - 1);
            let candidate_column = block.column - 1;
            let state = grid
                .get(candidate_row, candidate_column)
                .ok_or(Error::InvalidObu)?;
            let size = state.size.ok_or(Error::InvalidObu)?;
            let step = u32::from(size.dimensions().1 / 4).clamp(2, 16);
            if state.reference_frames[0] > 0 {
                count += 1;
                let width = (u32::from(config.prediction_width) >> 1).min(32 >> sub_x);
                let height = u32::from(config.prediction_height).min((step * 4) >> sub_y);
                visit(ObmcNeighbor {
                    pass: ObmcPass::Left,
                    candidate_row,
                    candidate_column,
                    region: PredictionRegion {
                        x: usize::try_from((block.column * 4) >> sub_x)
                            .map_err(|_| Error::LimitExceeded)?,
                        y: usize::try_from((row * 4) >> sub_y).map_err(|_| Error::LimitExceeded)?,
                        width: usize::try_from(width).map_err(|_| Error::LimitExceeded)?,
                        height: usize::try_from(height).map_err(|_| Error::LimitExceeded)?,
                    },
                    reference_frame: state.reference_frames[0],
                    motion_vector: state.motion_vectors[0],
                })?;
            }
            row = row.checked_add(step).ok_or(Error::LimitExceeded)?;
        }
    }
    Ok(())
}

pub fn apply_obmc_neighbors<F>(
    grid: &crate::block_state::MiGrid,
    plane: &mut Plane,
    config: ObmcTraversalConfig,
    mut predict: F,
) -> Result<(), Error>
where
    F: FnMut(ObmcNeighbor) -> Result<Vector<u16>, Error>,
{
    walk_obmc_neighbors(grid, config, |neighbor| {
        let prediction = predict(neighbor)?;
        blend_obmc(plane, neighbor.region, neighbor.pass, &prediction)
    })
}

pub fn blend_obmc(
    plane: &mut Plane,
    region: PredictionRegion,
    pass: ObmcPass,
    neighbor_prediction: &[u16],
) -> Result<(), Error> {
    if region.width == 0
        || region.height == 0
        || neighbor_prediction.len()
            != region
                .width
                .checked_mul(region.height)
                .ok_or(Error::LimitExceeded)?
    {
        return Err(Error::InvalidObu);
    }
    let mask = obmc_mask(match pass {
        ObmcPass::Above => region.height,
        ObmcPass::Left => region.width,
    })?;
    for row in 0..region.height {
        for column in 0..region.width {
            let weight = u32::from(match pass {
                ObmcPass::Above => mask[row],
                ObmcPass::Left => mask[column],
            });
            let current = u32::from(plane.sample(region.x + column, region.y + row)?);
            let neighbor = u32::from(neighbor_prediction[row * region.width + column]);
            let blended = (weight * current + (64 - weight) * neighbor + 32) >> 6;
            plane.set_sample(
                region.x + column,
                region.y + row,
                u16::try_from(blended).map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WarpBlockConfig {
    pub x: u32,
    pub y: u32,
    pub width: u16,
    pub height: u16,
    pub unit_column: u16,
    pub unit_row: u16,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub warp: [i32; 6],
    pub shear: ShearParams,
    pub rounding: InterRounding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WarpPredictionConfig {
    pub x: u32,
    pub y: u32,
    pub width: u16,
    pub height: u16,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub warp: [i32; 6],
    pub shear: ShearParams,
    pub rounding: InterRounding,
}

pub fn warp_reference(
    reference: &Plane,
    config: WarpPredictionConfig,
) -> Result<Vector<i32>, Error> {
    if config.width == 0 || config.height == 0 || config.width > 128 || config.height > 128 {
        return Err(Error::InvalidObu);
    }
    let length = usize::from(config.width)
        .checked_mul(usize::from(config.height))
        .ok_or(Error::LimitExceeded)?;
    let mut prediction = filled_vector(length, 0i32)?;
    let rows = config.height.div_ceil(8);
    let columns = config.width.div_ceil(8);
    for unit_row in 0..rows {
        for unit_column in 0..columns {
            warp_prediction_unit(
                reference,
                WarpBlockConfig {
                    x: config.x,
                    y: config.y,
                    width: config.width,
                    height: config.height,
                    unit_column,
                    unit_row,
                    subsampling_x: config.subsampling_x,
                    subsampling_y: config.subsampling_y,
                    warp: config.warp,
                    shear: config.shear,
                    rounding: config.rounding,
                },
                &mut prediction,
            )?;
        }
    }
    Ok(prediction)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterReferenceConfig {
    pub x: u32,
    pub y: u32,
    pub width: u16,
    pub height: u16,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub bit_depth: u8,
    pub compound: bool,
    pub force_integer_mv: bool,
    pub motion_mode: MotionMode,
    pub local_warp: Option<[i32; 6]>,
    pub global_mode: bool,
    pub global_motion: GlobalMotion,
    pub reference_scaled: bool,
    pub scaled_motion: ScaledMotion,
    pub horizontal_filter: InterpolationFilter,
    pub vertical_filter: InterpolationFilter,
}

/// Selects SIMPLE, LOCALWARP, or eligible affine global prediction and runs
/// the corresponding complete reference predictor.
pub fn predict_inter_reference(
    reference: &Plane,
    config: InterReferenceConfig,
) -> Result<Vector<i32>, Error> {
    let rounding = InterRounding::derive(config.bit_depth, config.compound)?;
    let large_enough = config.width >= 8 && config.height >= 8;
    let warp = if large_enough && !config.force_integer_mv {
        if config.motion_mode == MotionMode::LocalWarp {
            config.local_warp
        } else if config.global_mode
            && config.global_motion.kind > GlobalMotionType::Translation
            && !config.reference_scaled
        {
            Some(config.global_motion.params)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(warp) = warp {
        let shear = setup_shear(warp)?;
        if shear.valid {
            return warp_reference(
                reference,
                WarpPredictionConfig {
                    x: config.x,
                    y: config.y,
                    width: config.width,
                    height: config.height,
                    subsampling_x: config.subsampling_x,
                    subsampling_y: config.subsampling_y,
                    warp,
                    shear,
                    rounding,
                },
            );
        }
    }
    convolve_reference(
        reference,
        ConvolutionConfig {
            width: usize::from(config.width),
            height: usize::from(config.height),
            start_x: config.scaled_motion.start_x,
            start_y: config.scaled_motion.start_y,
            step_x: config.scaled_motion.step_x,
            step_y: config.scaled_motion.step_y,
            rounding,
        },
        interpolation_filter_bank(config.horizontal_filter, usize::from(config.width)),
        interpolation_filter_bank(config.vertical_filter, usize::from(config.height)),
    )
}

/// Runs section 7.11.3.5 for one clipped 8x8 output unit. `prediction`
/// stores the complete block in raster order and may be filled unit by unit.
pub fn warp_prediction_unit(
    reference: &Plane,
    config: WarpBlockConfig,
    prediction: &mut [i32],
) -> Result<(), Error> {
    if !config.shear.valid
        || config.width == 0
        || config.height == 0
        || reference.width() == 0
        || reference.height() == 0
    {
        return Err(Error::InvalidObu);
    }
    let width = usize::from(config.width);
    let height = usize::from(config.height);
    if prediction.len() != width.checked_mul(height).ok_or(Error::LimitExceeded)? {
        return Err(Error::InvalidObu);
    }
    let unit_x = usize::from(config.unit_column) * 8;
    let unit_y = usize::from(config.unit_row) * 8;
    if unit_x >= width || unit_y >= height {
        return Err(Error::InvalidObu);
    }
    let sub_x = u32::from(config.subsampling_x);
    let sub_y = u32::from(config.subsampling_y);
    let source_x = i64::from(config.x)
        .checked_add(i64::try_from(unit_x).map_err(|_| Error::LimitExceeded)?)
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_shl(sub_x))
        .ok_or(Error::LimitExceeded)?;
    let source_y = i64::from(config.y)
        .checked_add(i64::try_from(unit_y).map_err(|_| Error::LimitExceeded)?)
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_shl(sub_y))
        .ok_or(Error::LimitExceeded)?;
    let destination_x = i64::from(config.warp[2])
        .checked_mul(source_x)
        .and_then(|value| value.checked_add(i64::from(config.warp[3]) * source_y))
        .and_then(|value| value.checked_add(i64::from(config.warp[0])))
        .ok_or(Error::LimitExceeded)?;
    let destination_y = i64::from(config.warp[4])
        .checked_mul(source_x)
        .and_then(|value| value.checked_add(i64::from(config.warp[5]) * source_y))
        .and_then(|value| value.checked_add(i64::from(config.warp[1])))
        .ok_or(Error::LimitExceeded)?;
    let x4 = destination_x >> sub_x;
    let y4 = destination_y >> sub_y;
    let integer_x = x4 >> 16;
    let integer_y = y4 >> 16;
    let fractional_x = x4 & 0xffff;
    let fractional_y = y4 & 0xffff;
    let reduced_fractional_x =
        (fractional_x - 4 * i64::from(config.shear.alpha) - 4 * i64::from(config.shear.beta)) & !63;
    let reduced_fractional_y =
        (fractional_y - 4 * i64::from(config.shear.gamma) - 4 * i64::from(config.shear.delta))
            & !63;
    let last_x =
        i64::try_from(reference.width().saturating_sub(1)).map_err(|_| Error::LimitExceeded)?;
    let last_y =
        i64::try_from(reference.height().saturating_sub(1)).map_err(|_| Error::LimitExceeded)?;
    let mut intermediate = [0i32; 15 * 8];
    for intermediate_row in 0..15usize {
        let i1 = i64::try_from(intermediate_row).map_err(|_| Error::LimitExceeded)? - 7;
        for intermediate_column in 0..8usize {
            let i2 = i64::try_from(intermediate_column).map_err(|_| Error::LimitExceeded)? - 4;
            let phase = round2(
                reduced_fractional_x
                    + i64::from(config.shear.alpha) * (i2 + 4)
                    + i64::from(config.shear.beta) * (i1 + 4),
                10,
            )?
            .checked_add(64)
            .ok_or(Error::LimitExceeded)?;
            let filter = warped_filter(i16::try_from(phase - 64).map_err(|_| Error::InvalidObu)?)?;
            let sample_y = (integer_y + i1).clamp(0, last_y) as usize;
            let mut sum = 0i64;
            for (tap, coefficient) in filter.into_iter().enumerate() {
                let tap = i64::try_from(tap).map_err(|_| Error::LimitExceeded)?;
                let sample_x = (integer_x + i2 - 3 + tap).clamp(0, last_x) as usize;
                sum = sum
                    .checked_add(
                        i64::from(coefficient) * i64::from(reference.sample(sample_x, sample_y)?),
                    )
                    .ok_or(Error::LimitExceeded)?;
            }
            intermediate[intermediate_row * 8 + intermediate_column] =
                i32::try_from(round2(sum, config.rounding.horizontal)?)
                    .map_err(|_| Error::LimitExceeded)?;
        }
    }
    let output_height = 8.min(height - unit_y);
    let output_width = 8.min(width - unit_x);
    for local_y in 0..output_height {
        let i1 = i64::try_from(local_y).map_err(|_| Error::LimitExceeded)? - 4;
        for local_x in 0..output_width {
            let i2 = i64::try_from(local_x).map_err(|_| Error::LimitExceeded)? - 4;
            let phase = round2(
                reduced_fractional_y
                    + i64::from(config.shear.gamma) * (i2 + 4)
                    + i64::from(config.shear.delta) * (i1 + 4),
                10,
            )?
            .checked_add(64)
            .ok_or(Error::LimitExceeded)?;
            let filter = warped_filter(i16::try_from(phase - 64).map_err(|_| Error::InvalidObu)?)?;
            let mut sum = 0i64;
            for (tap, coefficient) in filter.into_iter().enumerate() {
                let row = local_y + tap;
                sum = sum
                    .checked_add(
                        i64::from(coefficient) * i64::from(intermediate[row * 8 + local_x]),
                    )
                    .ok_or(Error::LimitExceeded)?;
            }
            prediction[(unit_y + local_y) * width + unit_x + local_x] =
                i32::try_from(round2(sum, config.rounding.vertical)?)
                    .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(())
}

pub type SubpelFilterBank = [[i16; 8]; 16];

pub const BILINEAR_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

pub const SMALL_REGULAR_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -4, 126, 8, -2, 0, 0],
    [0, 0, -8, 122, 18, -4, 0, 0],
    [0, 0, -10, 116, 28, -6, 0, 0],
    [0, 0, -12, 110, 38, -8, 0, 0],
    [0, 0, -12, 102, 48, -10, 0, 0],
    [0, 0, -14, 94, 58, -10, 0, 0],
    [0, 0, -12, 84, 66, -10, 0, 0],
    [0, 0, -12, 76, 76, -12, 0, 0],
    [0, 0, -10, 66, 84, -12, 0, 0],
    [0, 0, -10, 58, 94, -14, 0, 0],
    [0, 0, -10, 48, 102, -12, 0, 0],
    [0, 0, -8, 38, 110, -12, 0, 0],
    [0, 0, -6, 28, 116, -10, 0, 0],
    [0, 0, -4, 18, 122, -8, 0, 0],
    [0, 0, -2, 8, 126, -4, 0, 0],
];

pub const SMALL_SMOOTH_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 30, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, 0, 14, 54, 48, 12, 0, 0],
    [0, 0, 12, 52, 52, 12, 0, 0],
    [0, 0, 12, 48, 54, 14, 0, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 30, 0, 0],
];

pub const REGULAR_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

pub const SMOOTH_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],
    [0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 28, 2, 0],
];

pub const SHARP_FILTER: SubpelFilterBank = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],
    [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],
    [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],
    [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],
    [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],
    [0, 2, -2, 8, 126, -6, 2, -2],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpolationFilter {
    Regular,
    Smooth,
    Sharp,
    Bilinear,
}

impl InterpolationFilter {
    pub const fn from_av1(value: u8) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::Regular),
            1 => Ok(Self::Smooth),
            2 => Ok(Self::Sharp),
            3 => Ok(Self::Bilinear),
            _ => Err(Error::InvalidObu),
        }
    }
}

pub fn interpolation_filter_bank(
    filter: InterpolationFilter,
    block_extent: usize,
) -> &'static SubpelFilterBank {
    match (filter, block_extent <= 4) {
        (InterpolationFilter::Regular, true) | (InterpolationFilter::Sharp, true) => {
            &SMALL_REGULAR_FILTER
        }
        (InterpolationFilter::Smooth, true) => &SMALL_SMOOTH_FILTER,
        (InterpolationFilter::Regular, false) => &REGULAR_FILTER,
        (InterpolationFilter::Smooth, false) => &SMOOTH_FILTER,
        (InterpolationFilter::Sharp, false) => &SHARP_FILTER,
        (InterpolationFilter::Bilinear, _) => &BILINEAR_FILTER,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvolutionConfig {
    pub width: usize,
    pub height: usize,
    pub start_x: i64,
    pub start_y: i64,
    pub step_x: i64,
    pub step_y: i64,
    pub rounding: InterRounding,
}

pub fn convolve_reference(
    reference: &Plane,
    config: ConvolutionConfig,
    horizontal: &SubpelFilterBank,
    vertical: &SubpelFilterBank,
) -> Result<Vector<i32>, Error> {
    const SCALE_SUBPEL_BITS: u8 = 10;
    if config.width == 0
        || config.height == 0
        || config.width > 128
        || config.height > 128
        || config.step_x <= 0
        || config.step_y <= 0
        || reference.width() == 0
        || reference.height() == 0
    {
        return Err(Error::InvalidObu);
    }
    let height_span = i64::try_from(config.height - 1)
        .map_err(|_| Error::LimitExceeded)?
        .checked_mul(config.step_y)
        .ok_or(Error::LimitExceeded)?;
    let intermediate_height = usize::try_from(
        ((height_span + ((1i64 << SCALE_SUBPEL_BITS) - 1)) >> SCALE_SUBPEL_BITS) + 8,
    )
    .map_err(|_| Error::LimitExceeded)?;
    let intermediate_len = intermediate_height
        .checked_mul(config.width)
        .ok_or(Error::LimitExceeded)?;
    let mut intermediate = filled_vector(intermediate_len, 0i32)?;
    let last_x = i64::try_from(reference.width() - 1).map_err(|_| Error::LimitExceeded)?;
    let last_y = i64::try_from(reference.height() - 1).map_err(|_| Error::LimitExceeded)?;
    for row in 0..intermediate_height {
        let source_y = ((config.start_y >> SCALE_SUBPEL_BITS)
            + i64::try_from(row).map_err(|_| Error::LimitExceeded)?
            - 3)
        .clamp(0, last_y) as usize;
        for column in 0..config.width {
            let position = config
                .start_x
                .checked_add(
                    config
                        .step_x
                        .checked_mul(i64::try_from(column).map_err(|_| Error::LimitExceeded)?)
                        .ok_or(Error::LimitExceeded)?,
                )
                .ok_or(Error::LimitExceeded)?;
            let phase = ((position >> 6) & 15) as usize;
            let mut sum = 0i64;
            for (tap, coefficient) in horizontal[phase].iter().enumerate() {
                let source_x = ((position >> SCALE_SUBPEL_BITS)
                    + i64::try_from(tap).map_err(|_| Error::LimitExceeded)?
                    - 3)
                .clamp(0, last_x) as usize;
                sum += i64::from(*coefficient) * i64::from(reference.sample(source_x, source_y)?);
            }
            intermediate[row * config.width + column] =
                i32::try_from(round2(sum, config.rounding.horizontal)?)
                    .map_err(|_| Error::LimitExceeded)?;
        }
    }
    let output_len = config
        .width
        .checked_mul(config.height)
        .ok_or(Error::LimitExceeded)?;
    let mut output = filled_vector(output_len, 0i32)?;
    for row in 0..config.height {
        for column in 0..config.width {
            let position = (config.start_y & 1023)
                .checked_add(
                    config
                        .step_y
                        .checked_mul(i64::try_from(row).map_err(|_| Error::LimitExceeded)?)
                        .ok_or(Error::LimitExceeded)?,
                )
                .ok_or(Error::LimitExceeded)?;
            let phase = ((position >> 6) & 15) as usize;
            let base =
                usize::try_from(position >> SCALE_SUBPEL_BITS).map_err(|_| Error::InvalidObu)?;
            let mut sum = 0i64;
            for (tap, coefficient) in vertical[phase].iter().enumerate() {
                let source_row = base.checked_add(tap).ok_or(Error::LimitExceeded)?;
                let sample = *intermediate
                    .get(source_row * config.width + column)
                    .ok_or(Error::InvalidObu)?;
                sum += i64::from(*coefficient) * i64::from(sample);
            }
            output[row * config.width + column] =
                i32::try_from(round2(sum, config.rounding.vertical)?)
                    .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(output)
}

fn filled_vector<T: Clone>(length: usize, value: T) -> Result<Vector<T>, Error> {
    let mut output = Vector::with_capacity(length).map_err(|_| Error::LimitExceeded)?;
    for _ in 0..length {
        output
            .try_push(value.clone())
            .map_err(|_| Error::LimitExceeded)?;
    }
    Ok(output)
}
use crate::prediction::PredictionRegion;
use crate::reconstruction::Plane;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterRounding {
    pub horizontal: u8,
    pub vertical: u8,
    pub post: u8,
}

impl InterRounding {
    pub fn derive(bit_depth: u8, compound: bool) -> Result<Self, Error> {
        if !matches!(bit_depth, 8 | 10 | 12) {
            return Err(Error::InvalidObu);
        }
        let horizontal = if bit_depth == 12 { 5 } else { 3 };
        let vertical = if compound {
            7
        } else if bit_depth == 12 {
            9
        } else {
            11
        };
        Ok(Self {
            horizontal,
            vertical,
            post: 14 - horizontal - vertical,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompoundBlend<'a> {
    Average,
    Distance {
        forward: u8,
        backward: u8,
    },
    Mask(&'a [u8]),
    DifferenceWeighted {
        inverse: bool,
    },
    Wedge {
        luma_width: u16,
        luma_height: u16,
        index: u8,
        sign: bool,
        subsampling_x: bool,
        subsampling_y: bool,
    },
}

pub fn distance_weights(reference_distances: [u8; 2]) -> Result<[u8; 2], Error> {
    const WEIGHT: [[u8; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, 31]];
    const LOOKUP: [[u8; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];
    if reference_distances.iter().any(|&distance| distance > 31) {
        return Err(Error::InvalidObu);
    }
    let d0 = reference_distances[1];
    let d1 = reference_distances[0];
    let order = usize::from(d0 <= d1);
    let index = if d0 == 0 || d1 == 0 {
        3
    } else {
        (0..3)
            .find(|&index| {
                let c0 = WEIGHT[index][order];
                let c1 = WEIGHT[index][1 - order];
                if order != 0 {
                    u16::from(d0) * u16::from(c0) > u16::from(d1) * u16::from(c1)
                } else {
                    u16::from(d0) * u16::from(c0) < u16::from(d1) * u16::from(c1)
                }
            })
            .unwrap_or(3)
    };
    Ok([LOOKUP[index][order], LOOKUP[index][1 - order]])
}

pub struct InterBlockPredictionConfig<'a> {
    pub region: PredictionRegion,
    pub bit_depth: u8,
    pub first_reference: &'a Plane,
    pub first: InterReferenceConfig,
    pub second_reference: Option<&'a Plane>,
    pub second: Option<InterReferenceConfig>,
    pub blend: CompoundBlend<'a>,
    pub mask_output: Option<&'a mut Vector<u8>>,
}

#[derive(Clone, Copy)]
pub struct InterPredictionSource<'a> {
    pub reference: &'a Plane,
    pub reference_upscaled_width: u32,
    pub reference_height: u32,
    pub motion_vector: MotionVector,
    pub global_motion: GlobalMotion,
    pub global_mode: bool,
    pub reference_scaled: bool,
}

pub struct ScaledInterBlockConfig<'a> {
    pub region: PredictionRegion,
    pub frame_width: u32,
    pub frame_height: u32,
    pub bit_depth: u8,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub force_integer_mv: bool,
    pub motion_mode: MotionMode,
    pub local_warp: Option<[i32; 6]>,
    pub horizontal_filter: InterpolationFilter,
    pub vertical_filter: InterpolationFilter,
    pub first: InterPredictionSource<'a>,
    pub second: Option<InterPredictionSource<'a>>,
    pub blend: CompoundBlend<'a>,
    pub mask_output: Option<&'a mut Vector<u8>>,
}

/// Derives section 7.11 scaled coordinates for each source and predicts a
/// complete inter block into the destination plane.
pub fn predict_scaled_inter_block(
    plane: &mut Plane,
    config: ScaledInterBlockConfig<'_>,
) -> Result<(), Error> {
    let compound = config.second.is_some();
    let reference_config =
        |source: InterPredictionSource<'_>| -> Result<InterReferenceConfig, Error> {
            let x = u32::try_from(config.region.x).map_err(|_| Error::LimitExceeded)?;
            let y = u32::try_from(config.region.y).map_err(|_| Error::LimitExceeded)?;
            Ok(InterReferenceConfig {
                x,
                y,
                width: u16::try_from(config.region.width).map_err(|_| Error::LimitExceeded)?,
                height: u16::try_from(config.region.height).map_err(|_| Error::LimitExceeded)?,
                subsampling_x: config.subsampling_x,
                subsampling_y: config.subsampling_y,
                bit_depth: config.bit_depth,
                compound,
                force_integer_mv: config.force_integer_mv,
                motion_mode: config.motion_mode,
                local_warp: config.local_warp,
                global_mode: source.global_mode,
                global_motion: source.global_motion,
                reference_scaled: source.reference_scaled,
                scaled_motion: scale_motion_vector(MotionScaleInput {
                    frame_width: config.frame_width,
                    frame_height: config.frame_height,
                    reference_upscaled_width: source.reference_upscaled_width,
                    reference_height: source.reference_height,
                    x,
                    y,
                    motion_vector: [source.motion_vector.row, source.motion_vector.column],
                    subsampling_x: config.subsampling_x,
                    subsampling_y: config.subsampling_y,
                })?,
                horizontal_filter: config.horizontal_filter,
                vertical_filter: config.vertical_filter,
            })
        };
    let first = reference_config(config.first)?;
    let (second_reference, second) = if let Some(source) = config.second {
        (Some(source.reference), Some(reference_config(source)?))
    } else {
        (None, None)
    };
    predict_inter_block(
        plane,
        InterBlockPredictionConfig {
            region: config.region,
            bit_depth: config.bit_depth,
            first_reference: config.first.reference,
            first,
            second_reference,
            second,
            blend: config.blend,
            mask_output: config.mask_output,
        },
    )
}

/// Produces and writes one complete single- or compound-reference prediction.
pub fn predict_inter_block(
    plane: &mut Plane,
    config: InterBlockPredictionConfig<'_>,
) -> Result<(), Error> {
    let first = predict_inter_reference(config.first_reference, config.first)?;
    match (config.second_reference, config.second) {
        (None, None) => write_single_prediction(plane, config.region, &first, config.bit_depth),
        (Some(reference), Some(second_config)) => {
            let second = predict_inter_reference(reference, second_config)?;
            blend_compound(
                plane,
                config.region,
                &first,
                &second,
                config.blend,
                InterRounding::derive(config.bit_depth, true)?,
                config.bit_depth,
                config.mask_output,
            )
        }
        _ => Err(Error::InvalidObu),
    }
}

pub fn write_single_prediction(
    plane: &mut Plane,
    region: PredictionRegion,
    prediction: &[i32],
    bit_depth: u8,
) -> Result<(), Error> {
    validate_region(plane, region, prediction.len(), bit_depth)?;
    let maximum = (1i32 << bit_depth) - 1;
    for row in 0..region.height {
        for column in 0..region.width {
            let value = prediction[row * region.width + column].clamp(0, maximum) as u16;
            plane.set_sample(region.x + column, region.y + row, value)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn blend_compound(
    plane: &mut Plane,
    region: PredictionRegion,
    first: &[i32],
    second: &[i32],
    blend: CompoundBlend<'_>,
    rounding: InterRounding,
    bit_depth: u8,
    mut mask_output: Option<&mut Vector<u8>>,
) -> Result<(), Error> {
    validate_region(plane, region, first.len(), bit_depth)?;
    let area = region
        .width
        .checked_mul(region.height)
        .ok_or(Error::LimitExceeded)?;
    if second.len() != area {
        return Err(Error::InvalidObu);
    }
    match blend {
        CompoundBlend::Distance { forward, backward }
            if forward.checked_add(backward) != Some(16) =>
        {
            return Err(Error::InvalidObu);
        }
        CompoundBlend::Mask(mask) if mask.len() != area || mask.iter().any(|value| *value > 64) => {
            return Err(Error::InvalidObu);
        }
        _ => {}
    }
    if matches!(blend, CompoundBlend::DifferenceWeighted { .. }) {
        let output = mask_output.as_mut().ok_or(Error::InvalidObu)?;
        output.clear();
        output.try_reserve(area).map_err(|_| Error::LimitExceeded)?;
        let rounding = InterRounding::derive(bit_depth, true)?;
        for index in 0..area {
            let difference = (first[index] - second[index]).unsigned_abs();
            let shift = u32::from(bit_depth - 8) + u32::from(rounding.post);
            let scaled = if shift == 0 {
                difference
            } else {
                (difference + (1 << (shift - 1))) >> shift
            };
            let base = (38 + scaled / 16).min(64) as u8;
            let weight = if matches!(blend, CompoundBlend::DifferenceWeighted { inverse: true }) {
                64 - base
            } else {
                base
            };
            output.try_push(weight).map_err(|_| Error::LimitExceeded)?;
        }
    }
    let maximum = (1i64 << bit_depth) - 1;
    for row in 0..region.height {
        for column in 0..region.width {
            let index = row * region.width + column;
            let a = i64::from(first[index]);
            let b = i64::from(second[index]);
            let (sum, shift) = match blend {
                CompoundBlend::Average => (a + b, 1 + rounding.post),
                CompoundBlend::Distance { forward, backward } => (
                    i64::from(forward) * a + i64::from(backward) * b,
                    4 + rounding.post,
                ),
                CompoundBlend::Mask(mask) => {
                    let weight = i64::from(mask[index]);
                    (weight * a + (64 - weight) * b, 6 + rounding.post)
                }
                CompoundBlend::DifferenceWeighted { .. } => {
                    let mask = mask_output.as_ref().ok_or(Error::InvalidObu)?;
                    let weight = i64::from(*mask.get(index).ok_or(Error::InvalidObu)?);
                    (weight * a + (64 - weight) * b, 6 + rounding.post)
                }
                CompoundBlend::Wedge {
                    luma_width,
                    luma_height,
                    index,
                    sign,
                    subsampling_x,
                    subsampling_y,
                } => {
                    let weight = i64::from(wedge_plane_weight(
                        luma_width,
                        luma_height,
                        index,
                        sign,
                        column,
                        row,
                        subsampling_x,
                        subsampling_y,
                    )?);
                    (weight * a + (64 - weight) * b, 6 + rounding.post)
                }
            };
            let value = round2(sum, shift)?.clamp(0, maximum) as u16;
            plane.set_sample(region.x + column, region.y + row, value)?;
        }
    }
    Ok(())
}

#[rustfmt::skip]
const WEDGE_OBLIQUE_ODD: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,6,18,
    37,53,60,63,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];
#[rustfmt::skip]
const WEDGE_OBLIQUE_EVEN: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,4,11,27,
    46,58,62,63,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];
#[rustfmt::skip]
const WEDGE_VERTICAL: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,7,21,
    43,57,62,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];

const WEDGE_CODEBOOK: [[[u8; 3]; 16]; 3] = [
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [0, 4, 2],
        [0, 4, 4],
        [0, 4, 6],
        [1, 4, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [1, 2, 4],
        [1, 4, 4],
        [1, 6, 4],
        [0, 4, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [0, 4, 2],
        [0, 4, 6],
        [1, 2, 4],
        [1, 6, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
];

#[allow(clippy::too_many_arguments)]
fn wedge_plane_weight(
    luma_width: u16,
    luma_height: u16,
    index: u8,
    sign: bool,
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
) -> Result<u8, Error> {
    let sx = usize::from(sub_x);
    let sy = usize::from(sub_y);
    let mut sum = 0u16;
    for dy in 0..1usize << sy {
        for dx in 0..1usize << sx {
            sum += u16::from(wedge_luma_weight(
                luma_width,
                luma_height,
                index,
                sign,
                (x << sx) + dx,
                (y << sy) + dy,
            )?);
        }
    }
    let shift = sx + sy;
    let rounding = if shift == 0 { 0 } else { 1u16 << (shift - 1) };
    Ok(((sum + rounding) >> shift) as u8)
}

fn wedge_luma_weight(
    width: u16,
    height: u16,
    index: u8,
    sign: bool,
    x: usize,
    y: usize,
) -> Result<u8, Error> {
    if !(8..=32).contains(&width)
        || !(8..=32).contains(&height)
        || index >= 16
        || x >= usize::from(width)
        || y >= usize::from(height)
    {
        return Err(Error::InvalidObu);
    }
    let shape = if height > width {
        0
    } else if height < width {
        1
    } else {
        2
    };
    let [direction, x_offset, y_offset] = WEDGE_CODEBOOK[shape][usize::from(index)];
    let origin_x = 32 - ((usize::from(x_offset) * usize::from(width)) >> 3);
    let origin_y = 32 - ((usize::from(y_offset) * usize::from(height)) >> 3);
    let master = master_wedge(direction, origin_y + y, origin_x + x)?;
    let mut edge_sum = 0u32;
    for column in 0..usize::from(width) {
        edge_sum += u32::from(master_wedge(direction, origin_y, origin_x + column)?);
    }
    for row in 1..usize::from(height) {
        edge_sum += u32::from(master_wedge(direction, origin_y + row, origin_x)?);
    }
    let count = u32::from(width) + u32::from(height) - 1;
    let flip_sign = (edge_sum + count / 2) / count < 32;
    Ok(if sign == flip_sign {
        master
    } else {
        64 - master
    })
}

fn master_wedge(direction: u8, y: usize, x: usize) -> Result<u8, Error> {
    if x >= 64 || y >= 64 {
        return Err(Error::InvalidObu);
    }
    let oblique63 = |row: usize, column: usize| {
        let shift = if row & 1 == 0 {
            16i32 - (row / 2) as i32
        } else {
            15i32 - (row / 2) as i32
        };
        let index = (column as i32 - shift).clamp(0, 63) as usize;
        if row & 1 == 0 {
            WEDGE_OBLIQUE_EVEN[index]
        } else {
            WEDGE_OBLIQUE_ODD[index]
        }
    };
    Ok(match direction {
        0 => WEDGE_VERTICAL[y],
        1 => WEDGE_VERTICAL[x],
        2 => oblique63(x, y),
        3 => oblique63(y, x),
        4 => 64 - oblique63(y, 63 - x),
        5 => 64 - oblique63(x, 63 - y),
        _ => return Err(Error::InvalidObu),
    })
}

fn validate_region(
    plane: &Plane,
    region: PredictionRegion,
    prediction_length: usize,
    bit_depth: u8,
) -> Result<(), Error> {
    if region.width == 0
        || region.height == 0
        || !matches!(bit_depth, 8 | 10 | 12)
        || prediction_length
            != region
                .width
                .checked_mul(region.height)
                .ok_or(Error::LimitExceeded)?
        || region
            .x
            .checked_add(region.width)
            .is_none_or(|end| end > plane.width())
        || region
            .y
            .checked_add(region.height)
            .is_none_or(|end| end > plane.height())
    {
        return Err(Error::InvalidObu);
    }
    Ok(())
}

fn round2(value: i64, shift: u8) -> Result<i64, Error> {
    if shift == 0 {
        return Ok(value);
    }
    value
        .checked_add(1i64 << (shift - 1))
        .map(|rounded| rounded >> shift)
        .ok_or(Error::LimitExceeded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warped_filter_bank_matches_normative_hash_and_endpoints() {
        assert_eq!(warped_filter(-64), Ok([0, 0, 127, 1, 0, 0, 0, 0]));
        assert_eq!(warped_filter(0), Ok([0, 0, 0, 127, 1, 0, 0, 0]));
        assert_eq!(warped_filter(129), Err(Error::InvalidObu));
        let mut hash = 0xcbf29ce484222325u64;
        for phase in -64..=128 {
            for coefficient in warped_filter(phase).unwrap() {
                hash ^= u64::from(coefficient as i8 as u8);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(hash, 0xb72c6b783fe2d984);
    }

    #[test]
    fn identity_warp_preserves_constant_reference_samples() {
        let reference = Plane::new(16, 16, 77).unwrap();
        let warp = crate::motion::GlobalMotion::default().params;
        let shear = crate::motion::setup_shear(warp).unwrap();
        let mut prediction = [0i32; 64];
        warp_prediction_unit(
            &reference,
            WarpBlockConfig {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                unit_column: 0,
                unit_row: 0,
                subsampling_x: false,
                subsampling_y: false,
                warp,
                shear,
                rounding: InterRounding::derive(8, false).unwrap(),
            },
            &mut prediction,
        )
        .unwrap();
        assert_eq!(prediction, [77; 64]);
    }

    #[test]
    fn fractional_warp_uses_ten_bit_filter_phase() {
        let reference = Plane::new(16, 16, 83).unwrap();
        let mut warp = crate::motion::GlobalMotion::default().params;
        warp[0] = 40_000;
        let shear = crate::motion::setup_shear(warp).unwrap();
        let mut prediction = [0i32; 64];
        warp_prediction_unit(
            &reference,
            WarpBlockConfig {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                unit_column: 0,
                unit_row: 0,
                subsampling_x: false,
                subsampling_y: false,
                warp,
                shear,
                rounding: InterRounding::derive(8, false).unwrap(),
            },
            &mut prediction,
        )
        .unwrap();
        assert!(prediction.iter().all(|&sample| sample == prediction[0]));
    }

    #[test]
    fn whole_warp_prediction_covers_clipped_edge_units() {
        let reference = Plane::new(20, 20, 51).unwrap();
        let warp = crate::motion::GlobalMotion::default().params;
        let prediction = warp_reference(
            &reference,
            WarpPredictionConfig {
                x: 0,
                y: 0,
                width: 13,
                height: 9,
                subsampling_x: false,
                subsampling_y: false,
                warp,
                shear: crate::motion::setup_shear(warp).unwrap(),
                rounding: InterRounding::derive(8, false).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(prediction.len(), 117);
        assert!(prediction.iter().all(|&sample| sample == 51));
    }

    #[test]
    fn interpolation_filter_values_match_av1_enumeration() {
        assert_eq!(
            InterpolationFilter::from_av1(0),
            Ok(InterpolationFilter::Regular)
        );
        assert_eq!(
            InterpolationFilter::from_av1(1),
            Ok(InterpolationFilter::Smooth)
        );
        assert_eq!(
            InterpolationFilter::from_av1(2),
            Ok(InterpolationFilter::Sharp)
        );
        assert_eq!(
            InterpolationFilter::from_av1(3),
            Ok(InterpolationFilter::Bilinear)
        );
        assert_eq!(InterpolationFilter::from_av1(4), Err(Error::InvalidObu));
    }

    #[test]
    fn inter_dispatcher_selects_eligible_global_affine_warp() {
        let reference = Plane::new(16, 16, 29).unwrap();
        let prediction = predict_inter_reference(
            &reference,
            InterReferenceConfig {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                subsampling_x: false,
                subsampling_y: false,
                bit_depth: 8,
                compound: false,
                force_integer_mv: false,
                motion_mode: MotionMode::Simple,
                local_warp: None,
                global_mode: true,
                global_motion: GlobalMotion {
                    kind: GlobalMotionType::Affine,
                    ..GlobalMotion::default()
                },
                reference_scaled: false,
                scaled_motion: ScaledMotion {
                    start_x: 0,
                    start_y: 0,
                    step_x: 1024,
                    step_y: 1024,
                },
                horizontal_filter: InterpolationFilter::Regular,
                vertical_filter: InterpolationFilter::Regular,
            },
        )
        .unwrap();
        assert_eq!(prediction.len(), 64);
        assert!(prediction.iter().all(|&sample| sample == 29));
    }

    #[test]
    fn invalid_local_warp_falls_back_to_translation_prediction() {
        let reference = Plane::new(16, 16, 29).unwrap();
        let prediction = predict_inter_reference(
            &reference,
            InterReferenceConfig {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
                subsampling_x: false,
                subsampling_y: false,
                bit_depth: 8,
                compound: false,
                force_integer_mv: false,
                motion_mode: MotionMode::LocalWarp,
                local_warp: None,
                global_mode: false,
                global_motion: GlobalMotion::default(),
                reference_scaled: false,
                scaled_motion: ScaledMotion {
                    start_x: 0,
                    start_y: 0,
                    step_x: 1024,
                    step_y: 1024,
                },
                horizontal_filter: InterpolationFilter::Regular,
                vertical_filter: InterpolationFilter::Regular,
            },
        )
        .unwrap();
        assert!(prediction.iter().all(|&sample| sample == 29));
    }

    #[test]
    fn scaled_inter_block_dispatches_motion_and_compound_blending() {
        let first = Plane::new(16, 16, 20).unwrap();
        let second = Plane::new(16, 16, 100).unwrap();
        let mut output = Plane::new(8, 8, 0).unwrap();
        let source = |reference| InterPredictionSource {
            reference,
            reference_upscaled_width: 16,
            reference_height: 16,
            motion_vector: MotionVector::default(),
            global_motion: GlobalMotion::default(),
            global_mode: false,
            reference_scaled: false,
        };
        predict_scaled_inter_block(
            &mut output,
            ScaledInterBlockConfig {
                region: PredictionRegion {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                frame_width: 16,
                frame_height: 16,
                bit_depth: 8,
                subsampling_x: false,
                subsampling_y: false,
                force_integer_mv: false,
                motion_mode: MotionMode::Simple,
                local_warp: None,
                horizontal_filter: InterpolationFilter::Regular,
                vertical_filter: InterpolationFilter::Regular,
                first: source(&first),
                second: Some(source(&second)),
                blend: CompoundBlend::Average,
                mask_output: None,
            },
        )
        .unwrap();
        assert!(output.samples().iter().all(|&sample| sample == 60));
    }

    #[test]
    fn obmc_blending_uses_normative_directional_masks() {
        let mut plane = Plane::new(2, 2, 64).unwrap();
        blend_obmc(
            &mut plane,
            PredictionRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            ObmcPass::Above,
            &[0; 4],
        )
        .unwrap();
        assert_eq!(plane.samples(), &[45, 45, 64, 64]);
        assert_eq!(obmc_mask(3), Err(Error::InvalidObu));
    }

    #[test]
    fn obmc_traversal_visits_above_then_left_in_scan_order() {
        let mut grid = crate::block_state::MiGrid::new(8, 8).unwrap();
        let neighbor = crate::block_state::BlockState {
            size: Some(crate::partition::BlockSize::Block8x8),
            is_inter: true,
            reference_frames: [1, -1],
            ..crate::block_state::BlockState::default()
        };
        for (column, row) in [(2, 0), (4, 0), (0, 2), (0, 4)] {
            grid.fill(
                crate::partition::BlockRect::new(
                    column,
                    row,
                    crate::partition::BlockSize::Block8x8,
                ),
                neighbor,
            )
            .unwrap();
        }
        let mut visits: Vector<ObmcNeighbor> = Vector::with_capacity(4).unwrap();
        walk_obmc_neighbors(
            &grid,
            ObmcTraversalConfig {
                block: crate::partition::BlockRect::new(
                    2,
                    2,
                    crate::partition::BlockSize::Block16x16,
                ),
                tile: crate::partition::TileBounds {
                    column_start: 0,
                    column_end: 8,
                    row_start: 0,
                    row_end: 8,
                },
                prediction_width: 16,
                prediction_height: 16,
                subsampling_x: false,
                subsampling_y: false,
                residual_at_least_8x8: true,
            },
            |neighbor| visits.try_push(neighbor).map_err(|_| Error::LimitExceeded),
        )
        .unwrap();
        assert_eq!(visits.len(), 4);
        assert_eq!(visits[0].pass, ObmcPass::Above);
        assert_eq!(visits[1].candidate_column, 5);
        assert_eq!(visits[2].pass, ObmcPass::Left);
        assert_eq!(visits[3].candidate_row, 5);
    }

    const REGION: PredictionRegion = PredictionRegion {
        x: 0,
        y: 0,
        width: 2,
        height: 1,
    };

    #[test]
    fn rounding_changes_for_twelve_bit_and_compound() {
        assert_eq!(
            InterRounding::derive(8, false),
            Ok(InterRounding {
                horizontal: 3,
                vertical: 11,
                post: 0
            })
        );
        assert_eq!(
            InterRounding::derive(12, true),
            Ok(InterRounding {
                horizontal: 5,
                vertical: 7,
                post: 2
            })
        );
    }

    #[test]
    fn average_distance_and_mask_blends_agree_at_half_weight() {
        let first = [40, 100];
        let second = [80, 200];
        let rounding = InterRounding::derive(8, true).unwrap();
        for blend in [
            CompoundBlend::Average,
            CompoundBlend::Distance {
                forward: 8,
                backward: 8,
            },
            CompoundBlend::Mask(&[32, 32]),
        ] {
            let mut plane = Plane::new(2, 1, 0).unwrap();
            // Compound predictors carry `post` fractional bits.
            let scaled_first = first.map(|value| value << rounding.post);
            let scaled_second = second.map(|value| value << rounding.post);
            blend_compound(
                &mut plane,
                REGION,
                &scaled_first,
                &scaled_second,
                blend,
                rounding,
                8,
                None,
            )
            .unwrap();
            assert_eq!(plane.samples(), &[60, 150]);
        }
    }

    #[test]
    fn inter_intra_dc_blends_both_predictions_equally() {
        let mut plane = Plane::new(8, 8, 60).unwrap();
        blend_inter_intra(
            &mut plane,
            PredictionRegion {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            0,
            &[20; 64],
        )
        .unwrap();
        assert_eq!(plane.sample(0, 0), Ok(40));
        assert_eq!(plane.sample(7, 7), Ok(40));
    }

    #[test]
    fn wedge_signs_are_exact_complements_for_every_codebook_entry() {
        for &(width, height) in &[(8, 16), (16, 8), (8, 8), (32, 32)] {
            for index in 0..16 {
                for y in 0..height {
                    for x in 0..width {
                        let first = wedge_luma_weight(
                            width,
                            height,
                            index,
                            false,
                            usize::from(x),
                            usize::from(y),
                        )
                        .unwrap();
                        let second = wedge_luma_weight(
                            width,
                            height,
                            index,
                            true,
                            usize::from(x),
                            usize::from(y),
                        )
                        .unwrap();
                        assert_eq!(first + second, 64);
                    }
                }
            }
        }
    }

    #[test]
    fn difference_mask_is_generated_and_chroma_subsampled() {
        let rounding = InterRounding::derive(8, true).unwrap();
        let first = [
            0i32 << rounding.post,
            160i32 << rounding.post,
            0,
            160i32 << rounding.post,
        ];
        let second = [0i32, 0, 0, 0];
        let mut plane = Plane::new(2, 2, 0).unwrap();
        let mut mask = Vector::new();
        blend_compound(
            &mut plane,
            PredictionRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            &first,
            &second,
            CompoundBlend::DifferenceWeighted { inverse: false },
            rounding,
            8,
            Some(&mut mask),
        )
        .unwrap();
        assert_eq!(&mask[..], &[38, 48, 38, 48]);
        assert_eq!(&subsample_mask(&mask, 2, 2, true, true).unwrap()[..], &[43]);
    }

    #[test]
    fn complete_inter_block_dispatch_writes_single_and_compound_predictions() {
        let first_reference = Plane::new(8, 8, 20).unwrap();
        let second_reference = Plane::new(8, 8, 40).unwrap();
        let reference_config = |compound| InterReferenceConfig {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
            subsampling_x: false,
            subsampling_y: false,
            bit_depth: 8,
            compound,
            force_integer_mv: false,
            motion_mode: MotionMode::Simple,
            local_warp: None,
            global_mode: false,
            global_motion: GlobalMotion::default(),
            reference_scaled: false,
            scaled_motion: ScaledMotion {
                start_x: 0,
                start_y: 0,
                step_x: 1024,
                step_y: 1024,
            },
            horizontal_filter: InterpolationFilter::Regular,
            vertical_filter: InterpolationFilter::Regular,
        };
        let region = PredictionRegion {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        let mut single = Plane::new(4, 4, 0).unwrap();
        predict_inter_block(
            &mut single,
            InterBlockPredictionConfig {
                region,
                bit_depth: 8,
                first_reference: &first_reference,
                first: reference_config(false),
                second_reference: None,
                second: None,
                blend: CompoundBlend::Average,
                mask_output: None,
            },
        )
        .unwrap();
        assert_eq!(single.samples(), &[20; 16]);

        let mut compound = Plane::new(4, 4, 0).unwrap();
        predict_inter_block(
            &mut compound,
            InterBlockPredictionConfig {
                region,
                bit_depth: 8,
                first_reference: &first_reference,
                first: reference_config(true),
                second_reference: Some(&second_reference),
                second: Some(reference_config(true)),
                blend: CompoundBlend::Average,
                mask_output: None,
            },
        )
        .unwrap();
        assert_eq!(compound.samples(), &[30; 16]);
    }

    #[test]
    fn distance_compound_weights_follow_order_hint_ratios() {
        assert_eq!(distance_weights([4, 4]), Ok([7, 9]));
        assert_eq!(distance_weights([1, 4]), Ok([13, 3]));
        assert_eq!(distance_weights([4, 1]), Ok([3, 13]));
        assert_eq!(distance_weights([0, 2]), Ok([13, 3]));
        assert_eq!(distance_weights([32, 1]), Err(Error::InvalidObu));
    }

    #[test]
    fn integer_phase_convolution_copies_reference_samples() {
        let mut reference = Plane::new(2, 2, 0).unwrap();
        for (y, row) in [[10, 20], [30, 40]].into_iter().enumerate() {
            for (x, sample) in row.into_iter().enumerate() {
                reference.set_sample(x, y, sample).unwrap();
            }
        }
        let identity = [[0, 0, 0, 128, 0, 0, 0, 0]; 16];
        let prediction = convolve_reference(
            &reference,
            ConvolutionConfig {
                width: 2,
                height: 2,
                start_x: 32,
                start_y: 32,
                step_x: 1024,
                step_y: 1024,
                rounding: InterRounding::derive(8, false).unwrap(),
            },
            &identity,
            &identity,
        )
        .unwrap();
        assert_eq!(&prediction[..], &[10, 20, 30, 40]);
    }

    #[test]
    fn every_small_and_bilinear_phase_has_unit_gain() {
        for bank in [
            &BILINEAR_FILTER,
            &SMALL_REGULAR_FILTER,
            &SMALL_SMOOTH_FILTER,
            &REGULAR_FILTER,
            &SMOOTH_FILTER,
            &SHARP_FILTER,
        ] {
            for phase in bank {
                assert_eq!(
                    phase.iter().map(|value| i32::from(*value)).sum::<i32>(),
                    128
                );
            }
        }
        assert!(core::ptr::eq(
            interpolation_filter_bank(InterpolationFilter::Sharp, 4),
            &SMALL_REGULAR_FILTER
        ));
    }
}
