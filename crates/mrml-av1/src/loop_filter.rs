//! AV1 deblocking masks and narrow/wide edge filters.

use crate::{
    ChromaSampling, Error,
    block_state::{BlockState, MiGrid},
    params::{LoopFilter, Segmentation},
    reconstruction::{FrameBuffer, Plane},
};

pub fn apply_frame(
    frame: &mut FrameBuffer,
    grid: &MiGrid,
    parameters: &LoopFilter,
    segmentation: &Segmentation,
    delta_lf_multi: bool,
    frame_width: u32,
    frame_height: u32,
) -> Result<(), Error> {
    let (subsampling_x, subsampling_y, planes) = match frame.sampling() {
        ChromaSampling::Cs400 => (false, false, 1),
        ChromaSampling::Cs420 => (true, true, 3),
        ChromaSampling::Cs422 => (true, false, 3),
        ChromaSampling::Cs444 => (false, false, 3),
    };
    let bit_depth = frame.bit_depth();
    for plane_index in 0..planes {
        if plane_index != 0 && parameters.level[plane_index + 1] == 0 {
            continue;
        }
        let chroma = plane_index != 0;
        let sub_x = u32::from(chroma && subsampling_x);
        let sub_y = u32::from(chroma && subsampling_y);
        let row_step = 1u32 << sub_y;
        let column_step = 1u32 << sub_x;
        for pass in 0..2usize {
            let mut row = 0;
            while row < grid.rows() {
                let mut column = 0;
                while column < grid.columns() {
                    filter_grid_edge(
                        plane_mut(frame, plane_index)?,
                        grid,
                        parameters,
                        segmentation,
                        EdgeConfig {
                            plane: plane_index,
                            pass,
                            row,
                            column,
                            sub_x,
                            sub_y,
                            delta_lf_multi,
                            frame_width,
                            frame_height,
                            bit_depth,
                        },
                    )?;
                    column += column_step;
                }
                row += row_step;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct EdgeConfig {
    plane: usize,
    pass: usize,
    row: u32,
    column: u32,
    sub_x: u32,
    sub_y: u32,
    delta_lf_multi: bool,
    frame_width: u32,
    frame_height: u32,
    bit_depth: u8,
}

fn filter_grid_edge(
    plane: &mut Plane,
    grid: &MiGrid,
    parameters: &LoopFilter,
    segmentation: &Segmentation,
    config: EdgeConfig,
) -> Result<(), Error> {
    let x = config.column * 4;
    let y = config.row * 4;
    if x >= config.frame_width
        || y >= config.frame_height
        || (config.pass == 0 && x == 0)
        || (config.pass == 1 && y == 0)
    {
        return Ok(());
    }
    let row = config.row | config.sub_y;
    let column = config.column | config.sub_x;
    let previous_row = row.saturating_sub(u32::from(config.pass == 1) << config.sub_y);
    let previous_column = column.saturating_sub(u32::from(config.pass == 0) << config.sub_x);
    let current = grid.get(row, column).ok_or(Error::InvalidObu)?;
    let previous = grid
        .get(previous_row, previous_column)
        .ok_or(Error::InvalidObu)?;
    let tx_size = current.loop_filter_tx_sizes[config.plane];
    let previous_tx_size = previous.loop_filter_tx_sizes[config.plane];
    let block_size = current.size.ok_or(Error::InvalidObu)?.plane_residual_size(
        config.plane != 0 && config.sub_x != 0,
        config.plane != 0 && config.sub_y != 0,
    )?;
    let x_plane = x >> config.sub_x;
    let y_plane = y >> config.sub_y;
    let (block_width, block_height) = block_size.dimensions();
    let (tx_width, tx_height) = tx_size.dimensions();
    let block_edge = if config.pass == 0 {
        x_plane.is_multiple_of(u32::from(block_width))
    } else {
        y_plane.is_multiple_of(u32::from(block_height))
    };
    let tx_edge = if config.pass == 0 {
        x_plane.is_multiple_of(u32::from(tx_width))
    } else {
        y_plane.is_multiple_of(u32::from(tx_height))
    };
    if !tx_edge || (!block_edge && current.skip && current.reference_frames[0] > 0) {
        return Ok(());
    }
    let filter_size = {
        let (previous_width, previous_height) = previous_tx_size.dimensions();
        let base = if config.pass == 0 {
            tx_width.min(previous_width)
        } else {
            tx_height.min(previous_height)
        };
        if config.plane == 0 {
            base.min(16)
        } else {
            base.min(8)
        }
    };
    let mut strength = adaptive_strength(current, parameters, segmentation, config)?;
    if strength.level == 0 {
        strength = adaptive_strength(previous, parameters, segmentation, config)?;
    }
    if strength.level == 0 {
        return Ok(());
    }
    for offset in 0..4usize {
        let edge_x = usize::try_from(x_plane).map_err(|_| Error::LimitExceeded)?
            + usize::from(config.pass == 1) * offset;
        let edge_y = usize::try_from(y_plane).map_err(|_| Error::LimitExceeded)?
            + usize::from(config.pass == 0) * offset;
        filter_plane_samples(
            plane,
            edge_x,
            edge_y,
            config.pass,
            FilterMaskConfig {
                limit: strength.limit,
                blimit: strength.blimit,
                threshold: strength.threshold,
                filter_size,
                luma: config.plane == 0,
                bit_depth: config.bit_depth,
            },
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AdaptiveStrength {
    level: u8,
    limit: u8,
    blimit: u8,
    threshold: u8,
}

fn adaptive_strength(
    state: &BlockState,
    parameters: &LoopFilter,
    segmentation: &Segmentation,
    config: EdgeConfig,
) -> Result<AdaptiveStrength, Error> {
    let component = if config.plane == 0 {
        config.pass
    } else {
        config.plane + 1
    };
    let delta_index = if config.delta_lf_multi { component } else { 0 };
    let mut level = (i16::from(parameters.level[component])
        + i16::from(state.delta_lf[delta_index]))
    .clamp(0, 63);
    let segment = usize::from(state.segment_id);
    let feature = 1 + component;
    if segmentation.enabled
        && *segmentation
            .feature_enabled
            .get(segment)
            .and_then(|features| features.get(feature))
            .ok_or(Error::InvalidObu)?
    {
        level = (level + segmentation.feature_data[segment][feature]).clamp(0, 63);
    }
    if parameters.delta_enabled {
        let reference =
            usize::try_from(state.reference_frames[0]).map_err(|_| Error::InvalidObu)?;
        let shift = level >> 5;
        level += i16::from(
            *parameters
                .ref_deltas
                .get(reference)
                .ok_or(Error::InvalidObu)?,
        ) << shift;
        if reference != 0 {
            let mode_type = usize::from(
                state.prediction_mode >= 14 && !matches!(state.prediction_mode, 16 | 24),
            );
            level += i16::from(parameters.mode_deltas[mode_type]) << shift;
        }
        level = level.clamp(0, 63);
    }
    let level = u8::try_from(level).map_err(|_| Error::InvalidObu)?;
    let shift = if parameters.sharpness > 4 {
        2
    } else {
        u8::from(parameters.sharpness > 0)
    };
    let limit = if parameters.sharpness > 0 {
        (level >> shift).clamp(1, 9 - parameters.sharpness)
    } else {
        (level >> shift).max(1)
    };
    Ok(AdaptiveStrength {
        level,
        limit,
        blimit: 2 * (level + 2) + limit,
        threshold: level >> 4,
    })
}

fn filter_plane_samples(
    plane: &mut Plane,
    x: usize,
    y: usize,
    pass: usize,
    config: FilterMaskConfig,
) -> Result<(), Error> {
    let mut p = [0u16; 7];
    let mut q = [0u16; 7];
    for distance in 0..7usize {
        let (px, py, qx, qy) = if pass == 0 {
            (
                x.saturating_sub(distance + 1),
                y,
                (x + distance).min(plane.width() - 1),
                y,
            )
        } else {
            (
                x,
                y.saturating_sub(distance + 1),
                x,
                (y + distance).min(plane.height() - 1),
            )
        };
        p[distance] = plane.sample(px, py)?;
        q[distance] = plane.sample(qx, qy)?;
    }
    filter_edge(&mut p, &mut q, config)?;
    let count = if config.filter_size == 16 {
        6
    } else if config.filter_size >= 8 {
        if config.luma { 3 } else { 2 }
    } else {
        2
    };
    for distance in 0..count {
        if pass == 0 {
            plane.set_sample(x - distance - 1, y, p[distance])?;
            plane.set_sample(x + distance, y, q[distance])?;
        } else {
            plane.set_sample(x, y - distance - 1, p[distance])?;
            plane.set_sample(x, y + distance, q[distance])?;
        }
    }
    Ok(())
}

fn plane_mut(frame: &mut FrameBuffer, plane: usize) -> Result<&mut Plane, Error> {
    match plane {
        0 => Ok(&mut frame.y),
        1 => frame.u.as_mut().ok_or(Error::InvalidObu),
        2 => frame.v.as_mut().ok_or(Error::InvalidObu),
        _ => Err(Error::InvalidObu),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterMasks {
    pub high_edge_variance: bool,
    pub filter: bool,
    pub flat: bool,
    pub flat_wide: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterMaskConfig {
    pub limit: u8,
    pub blimit: u8,
    pub threshold: u8,
    pub filter_size: u8,
    pub luma: bool,
    pub bit_depth: u8,
}

pub fn derive_masks(
    p: &[u16; 7],
    q: &[u16; 7],
    config: FilterMaskConfig,
) -> Result<FilterMasks, Error> {
    let FilterMaskConfig {
        limit,
        blimit,
        threshold,
        filter_size,
        luma,
        bit_depth,
    } = config;
    if !matches!(filter_size, 4 | 8 | 16) || !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidObu);
    }
    let scale = 1i32 << (bit_depth - 8);
    let threshold = i32::from(threshold) * scale;
    let limit = i32::from(limit) * scale;
    let blimit = i32::from(blimit) * scale;
    let difference = |a: u16, b: u16| (i32::from(a) - i32::from(b)).abs();
    let high_edge_variance =
        difference(p[1], p[0]) > threshold || difference(q[1], q[0]) > threshold;
    let filter_len = if filter_size == 4 {
        4
    } else if !luma {
        6
    } else {
        filter_size
    };
    let mut blocked = difference(p[1], p[0]) > limit
        || difference(q[1], q[0]) > limit
        || difference(p[0], q[0]) * 2 + difference(p[1], q[1]) / 2 > blimit;
    if filter_len >= 6 {
        blocked |= difference(p[2], p[1]) > limit || difference(q[2], q[1]) > limit;
    }
    if filter_len >= 8 {
        blocked |= difference(p[3], p[2]) > limit || difference(q[3], q[2]) > limit;
    }
    let flat_threshold = scale;
    let flat = filter_size >= 8
        && difference(p[1], p[0]) <= flat_threshold
        && difference(q[1], q[0]) <= flat_threshold
        && difference(p[2], p[0]) <= flat_threshold
        && difference(q[2], q[0]) <= flat_threshold
        && (filter_len < 8
            || (difference(p[3], p[0]) <= flat_threshold
                && difference(q[3], q[0]) <= flat_threshold));
    let flat_wide = filter_size >= 16
        && difference(p[6], p[0]) <= flat_threshold
        && difference(q[6], q[0]) <= flat_threshold
        && difference(p[5], p[0]) <= flat_threshold
        && difference(q[5], q[0]) <= flat_threshold
        && difference(p[4], p[0]) <= flat_threshold
        && difference(q[4], q[0]) <= flat_threshold;
    Ok(FilterMasks {
        high_edge_variance,
        filter: !blocked,
        flat,
        flat_wide,
    })
}

pub fn narrow_filter(
    p: &mut [u16; 2],
    q: &mut [u16; 2],
    high_edge_variance: bool,
    bit_depth: u8,
) -> Result<(), Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidObu);
    }
    let offset = 0x80i32 << (bit_depth - 8);
    let minimum = -(1i32 << (bit_depth - 1));
    let maximum = (1i32 << (bit_depth - 1)) - 1;
    let clamp = |value: i32| value.clamp(minimum, maximum);
    let p1 = i32::from(p[1]) - offset;
    let p0 = i32::from(p[0]) - offset;
    let q0 = i32::from(q[0]) - offset;
    let q1 = i32::from(q[1]) - offset;
    let mut filter = if high_edge_variance {
        clamp(p1 - q1)
    } else {
        0
    };
    filter = clamp(filter + 3 * (q0 - p0));
    let filter1 = clamp(filter + 4) >> 3;
    let filter2 = clamp(filter + 3) >> 3;
    q[0] = u16::try_from(clamp(q0 - filter1) + offset).map_err(|_| Error::InvalidObu)?;
    p[0] = u16::try_from(clamp(p0 + filter2) + offset).map_err(|_| Error::InvalidObu)?;
    if !high_edge_variance {
        let outer = (filter1 + 1) >> 1;
        q[1] = u16::try_from(clamp(q1 - outer) + offset).map_err(|_| Error::InvalidObu)?;
        p[1] = u16::try_from(clamp(p1 + outer) + offset).map_err(|_| Error::InvalidObu)?;
    }
    Ok(())
}

/// Applies the normative flat-region low-pass filter from section 7.14.6.4.
///
/// Samples are ordered away from the edge: `p[0]` and `q[0]` are adjacent.
/// For an 8-tap luma filter the first three samples on each side are replaced;
/// for 8-tap chroma the first two are replaced; a 16-tap filter replaces six.
pub fn wide_filter(
    p: &mut [u16; 7],
    q: &mut [u16; 7],
    filter_size: u8,
    luma: bool,
) -> Result<(), Error> {
    if !matches!(filter_size, 8 | 16) {
        return Err(Error::InvalidObu);
    }
    let (n, n2) = if filter_size == 16 {
        (6i32, 1i32)
    } else if luma {
        (3, 0)
    } else {
        (2, 1)
    };
    let log2_size = if filter_size == 16 { 4 } else { 3 };
    let sample = |position: i32| -> u16 {
        if position < 0 {
            p[usize::try_from(-position - 1).unwrap()]
        } else {
            q[usize::try_from(position).unwrap()]
        }
    };
    let mut filtered_p = [0u16; 6];
    let mut filtered_q = [0u16; 6];
    for i in -n..n {
        let mut total = 0u32;
        for j in -n..=n {
            let position = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            total += u32::from(sample(position)) * tap;
        }
        let value = u16::try_from((total + (1 << (log2_size - 1))) >> log2_size)
            .map_err(|_| Error::InvalidObu)?;
        if i < 0 {
            filtered_p[usize::try_from(-i - 1).unwrap()] = value;
        } else {
            filtered_q[usize::try_from(i).unwrap()] = value;
        }
    }
    let count = usize::try_from(n).unwrap();
    p[..count].copy_from_slice(&filtered_p[..count]);
    q[..count].copy_from_slice(&filtered_q[..count]);
    Ok(())
}

/// Selects the normative narrow, 8-tap, or 16-tap operation for one edge.
pub fn filter_edge(
    p: &mut [u16; 7],
    q: &mut [u16; 7],
    config: FilterMaskConfig,
) -> Result<FilterMasks, Error> {
    let masks = derive_masks(p, q, config)?;
    if !masks.filter {
        return Ok(masks);
    }
    if config.filter_size == 16 && masks.flat && masks.flat_wide {
        wide_filter(p, q, 16, config.luma)?;
    } else if config.filter_size >= 8 && masks.flat {
        wide_filter(p, q, 8, config.luma)?;
    } else {
        let mut inner_p = [p[0], p[1]];
        let mut inner_q = [q[0], q[1]];
        narrow_filter(
            &mut inner_p,
            &mut inner_q,
            masks.high_edge_variance,
            config.bit_depth,
        )?;
        p[..2].copy_from_slice(&inner_p);
        q[..2].copy_from_slice(&inner_q);
    }
    Ok(masks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        block_state::BlockState,
        partition::{BlockRect, BlockSize},
        transform::TxSize,
    };

    #[test]
    fn frame_pass_filters_internal_transform_boundaries() {
        let mut frame = FrameBuffer::new(16, 8, 8, ChromaSampling::Cs400).unwrap();
        for y in 0..8 {
            for x in 0..16 {
                frame
                    .y
                    .set_sample(x, y, if x < 8 { 100 } else { 104 })
                    .unwrap();
            }
        }
        let mut grid = MiGrid::new(4, 2).unwrap();
        for column in [0, 2] {
            grid.fill(
                BlockRect::new(column, 0, BlockSize::Block8x8),
                BlockState {
                    size: Some(BlockSize::Block8x8),
                    loop_filter_tx_sizes: [TxSize::Tx8x8; 3],
                    ..BlockState::default()
                },
            )
            .unwrap();
        }
        let mut parameters = LoopFilter::default();
        parameters.level[0] = 20;
        apply_frame(
            &mut frame,
            &grid,
            &parameters,
            &Segmentation::default(),
            false,
            16,
            8,
        )
        .unwrap();
        assert_ne!(frame.y.sample(7, 0), Ok(100));
        assert_ne!(frame.y.sample(8, 0), Ok(104));
    }

    #[test]
    fn constant_region_is_flat_and_filterable() {
        let masks = derive_masks(
            &[80; 7],
            &[80; 7],
            FilterMaskConfig {
                limit: 4,
                blimit: 8,
                threshold: 2,
                filter_size: 16,
                luma: true,
                bit_depth: 8,
            },
        )
        .unwrap();
        assert_eq!(
            masks,
            FilterMasks {
                high_edge_variance: false,
                filter: true,
                flat: true,
                flat_wide: true,
            }
        );
    }

    #[test]
    fn narrow_filter_updates_both_sides_symmetrically() {
        let mut p = [100, 100];
        let mut q = [104, 104];
        narrow_filter(&mut p, &mut q, false, 8).unwrap();
        assert_eq!(p, [101, 101]);
        assert_eq!(q, [102, 103]);
    }

    #[test]
    fn wide_filter_preserves_a_constant_region() {
        for (size, luma) in [(8, true), (8, false), (16, true)] {
            let mut p = [777; 7];
            let mut q = [777; 7];
            wide_filter(&mut p, &mut q, size, luma).unwrap();
            assert_eq!(p, [777; 7]);
            assert_eq!(q, [777; 7]);
        }
    }

    #[test]
    fn wide_filter_only_replaces_its_normative_span() {
        let mut p = [0; 7];
        let mut q = [64; 7];
        wide_filter(&mut p, &mut q, 8, false).unwrap();
        assert_eq!(p, [24, 8, 0, 0, 0, 0, 0]);
        assert_eq!(q, [40, 56, 64, 64, 64, 64, 64]);
    }

    #[test]
    fn blocked_edge_is_unchanged() {
        let original_p = [0; 7];
        let original_q = [255; 7];
        let mut p = original_p;
        let mut q = original_q;
        let masks = filter_edge(
            &mut p,
            &mut q,
            FilterMaskConfig {
                limit: 1,
                blimit: 1,
                threshold: 1,
                filter_size: 16,
                luma: true,
                bit_depth: 8,
            },
        )
        .unwrap();
        assert!(!masks.filter);
        assert_eq!(p, original_p);
        assert_eq!(q, original_q);
    }
}
