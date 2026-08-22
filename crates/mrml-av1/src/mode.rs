//! Common entropy-coded mode-info decisions.

use crate::{
    Error,
    block_state::{CdefIndexGrid, MiGrid, SegmentPredictionContexts},
    cdf::TileCdfs,
    entropy::SymbolDecoder,
    motion::{
        GlobalMotionType, MotionContexts, MotionStack, MotionVector, MotionVectorSyntax,
        collect_warp_samples, has_overlappable_candidates, read_motion_vector,
    },
    params::{MAX_SEGMENTS, Segmentation},
    partition::{BlockRect, BlockSize, TileBounds, negative_deinterleave},
    transform::TxSize,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ActiveSegmentFeatures {
    pub skip: bool,
    pub reference_frame: Option<u8>,
    pub global_mv: bool,
}

pub fn active_segment_features(
    segmentation: &Segmentation,
    segment_id: u8,
) -> Result<ActiveSegmentFeatures, Error> {
    let segment = usize::from(segment_id);
    if segment >= MAX_SEGMENTS {
        return Err(Error::InvalidObu);
    }
    if !segmentation.enabled {
        return Ok(ActiveSegmentFeatures::default());
    }
    let reference_frame = if segmentation.feature_enabled[segment][5] {
        let reference = segmentation.feature_data[segment][5];
        if !(0..=7).contains(&reference) {
            return Err(Error::InvalidObu);
        }
        Some(reference as u8)
    } else {
        None
    };
    Ok(ActiveSegmentFeatures {
        skip: segmentation.feature_enabled[segment][6],
        reference_frame,
        global_mv: segmentation.feature_enabled[segment][7],
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum IntraMode {
    Dc,
    Vertical,
    Horizontal,
    D45,
    D135,
    D113,
    D157,
    D203,
    D67,
    Smooth,
    SmoothVertical,
    SmoothHorizontal,
    Paeth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ChromaMode {
    Dc,
    Vertical,
    Horizontal,
    D45,
    D135,
    D113,
    D157,
    D203,
    D67,
    Smooth,
    SmoothVertical,
    SmoothHorizontal,
    Paeth,
    Cfl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterMode {
    Nearest,
    Near,
    Global,
    New,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MotionMode {
    #[default]
    Simple,
    Obmc,
    LocalWarp,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReferenceContexts {
    pub compound_mode: u8,
    pub compound_type: u8,
    pub unidirectional: [u8; 3],
    pub forward: [u8; 3],
    pub backward: [u8; 2],
    pub single: [u8; 6],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceFrameConfig {
    pub size: BlockSize,
    pub skip_mode: bool,
    pub skip_mode_frames: [i8; 2],
    pub segment: ActiveSegmentFeatures,
    pub reference_select: bool,
    pub contexts: ReferenceContexts,
}

pub fn reference_contexts(grid: &MiGrid, block: BlockRect, tile: TileBounds) -> ReferenceContexts {
    let above = (block.row > tile.row_start)
        .then(|| grid.get(block.row - 1, block.column))
        .flatten();
    let left = (block.column > tile.column_start)
        .then(|| grid.get(block.row, block.column - 1))
        .flatten();
    let refs = |state: Option<&crate::block_state::BlockState>| {
        state.map_or([0, -1], |state| state.reference_frames)
    };
    let above_refs = refs(above);
    let left_refs = refs(left);
    let single = |frames: [i8; 2]| frames[1] <= 0;
    let intra = |frames: [i8; 2]| frames[0] <= 0;
    let backward = |reference: i8| (5..=7).contains(&reference);
    let same_direction = |frames: [i8; 2]| backward(frames[0]) == backward(frames[1]);
    let compound_mode = match (above, left) {
        (Some(_), Some(_)) if single(above_refs) && single(left_refs) => {
            u8::from(backward(above_refs[0]) ^ backward(left_refs[0]))
        }
        (Some(_), Some(_)) if single(above_refs) => {
            2 + u8::from(backward(above_refs[0]) || intra(above_refs))
        }
        (Some(_), Some(_)) if single(left_refs) => {
            2 + u8::from(backward(left_refs[0]) || intra(left_refs))
        }
        (Some(_), Some(_)) => 4,
        (Some(_), None) => {
            if single(above_refs) {
                u8::from(backward(above_refs[0]))
            } else {
                3
            }
        }
        (None, Some(_)) => {
            if single(left_refs) {
                u8::from(backward(left_refs[0]))
            } else {
                3
            }
        }
        (None, None) => 1,
    };
    let above_comp = above.is_some() && !intra(above_refs) && !single(above_refs);
    let left_comp = left.is_some() && !intra(left_refs) && !single(left_refs);
    let above_uni = above_comp && same_direction(above_refs);
    let left_uni = left_comp && same_direction(left_refs);
    let compound_type =
        if above.is_some() && left.is_some() && !intra(above_refs) && !intra(left_refs) {
            let same = same_direction([above_refs[0], left_refs[0]]);
            if !above_comp && !left_comp {
                1 + 2 * u8::from(same)
            } else if !above_comp {
                if !left_uni { 1 } else { 3 + u8::from(same) }
            } else if !left_comp {
                if !above_uni { 1 } else { 3 + u8::from(same) }
            } else if !above_uni && !left_uni {
                0
            } else if !above_uni || !left_uni {
                2
            } else {
                3 + u8::from((above_refs[0] == 5) == (left_refs[0] == 5))
            }
        } else if above.is_some() && left.is_some() {
            if above_comp {
                1 + 2 * u8::from(above_uni)
            } else if left_comp {
                1 + 2 * u8::from(left_uni)
            } else {
                2
            }
        } else if above_comp {
            4 * u8::from(above_uni)
        } else if left_comp {
            4 * u8::from(left_uni)
        } else {
            2
        };
    let count = |reference: i8| -> u8 {
        [above_refs, left_refs]
            .into_iter()
            .flatten()
            .filter(|&candidate| candidate == reference)
            .count() as u8
    };
    let count_context = |first: u8, second: u8| {
        if first < second {
            0
        } else if first == second {
            1
        } else {
            2
        }
    };
    let forward_group = count_context(count(1) + count(2), count(3) + count(4));
    let forward_near = count_context(count(1), count(2));
    let forward_far = count_context(count(3), count(4));
    let backward_group = count_context(count(5) + count(6), count(7));
    let backward_near = count_context(count(5), count(6));
    ReferenceContexts {
        compound_mode,
        compound_type,
        unidirectional: [
            count_context(
                count(1) + count(2) + count(3) + count(4),
                count(5) + count(6) + count(7),
            ),
            count_context(count(2), count(3) + count(4)),
            forward_far,
        ],
        forward: [forward_group, forward_near, forward_far],
        backward: [backward_group, backward_near],
        single: [
            count_context(
                count(1) + count(2) + count(3) + count(4),
                count(5) + count(6) + count(7),
            ),
            backward_group,
            forward_group,
            forward_near,
            forward_far,
            backward_near,
        ],
    }
}

/// Reads section 5.11.25's complete reference decision tree. AV1 reference
/// enum values are returned directly (`LAST=1` through `ALTREF=7`).
pub fn read_reference_frames(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    config: ReferenceFrameConfig,
) -> Result<[i8; 2], Error> {
    if config.skip_mode {
        if config.skip_mode_frames[0] <= 0 || config.skip_mode_frames[1] <= 0 {
            return Err(Error::InvalidObu);
        }
        return Ok(config.skip_mode_frames);
    }
    if let Some(reference) = config.segment.reference_frame {
        return Ok([i8::try_from(reference).map_err(|_| Error::InvalidObu)?, -1]);
    }
    if config.segment.skip || config.segment.global_mv {
        return Ok([1, -1]);
    }
    let (width, height) = config.size.dimensions();
    let compound = config.reference_select
        && width.min(height) >= 8
        && cdfs.read_comp_mode(decoder, config.contexts.compound_mode)?;
    if compound {
        let unidirectional = cdfs.read_comp_ref_type(decoder, config.contexts.compound_type)?;
        if unidirectional {
            if cdfs.read_uni_comp_ref(decoder, 0, config.contexts.unidirectional[0])? {
                return Ok([5, 7]);
            }
            if cdfs.read_uni_comp_ref(decoder, 1, config.contexts.unidirectional[1])? {
                return Ok(
                    if cdfs.read_uni_comp_ref(decoder, 2, config.contexts.unidirectional[2])? {
                        [1, 4]
                    } else {
                        [1, 3]
                    },
                );
            }
            return Ok([1, 2]);
        }
        let forward = if !cdfs.read_comp_ref(decoder, 0, config.contexts.forward[0])? {
            if cdfs.read_comp_ref(decoder, 1, config.contexts.forward[1])? {
                2
            } else {
                1
            }
        } else if cdfs.read_comp_ref(decoder, 2, config.contexts.forward[2])? {
            4
        } else {
            3
        };
        let backward = if !cdfs.read_comp_bwd_ref(decoder, 0, config.contexts.backward[0])? {
            if cdfs.read_comp_bwd_ref(decoder, 1, config.contexts.backward[1])? {
                6
            } else {
                5
            }
        } else {
            7
        };
        return Ok([forward, backward]);
    }
    let reference = if cdfs.read_single_ref(decoder, 0, config.contexts.single[0])? {
        if !cdfs.read_single_ref(decoder, 1, config.contexts.single[1])? {
            if cdfs.read_single_ref(decoder, 5, config.contexts.single[5])? {
                6
            } else {
                5
            }
        } else {
            7
        }
    } else if cdfs.read_single_ref(decoder, 2, config.contexts.single[2])? {
        if cdfs.read_single_ref(decoder, 4, config.contexts.single[4])? {
            4
        } else {
            3
        }
    } else if cdfs.read_single_ref(decoder, 3, config.contexts.single[3])? {
        2
    } else {
        1
    };
    Ok([reference, -1])
}

pub fn read_reference_frames_from_grid(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    mut config: ReferenceFrameConfig,
) -> Result<[i8; 2], Error> {
    config.contexts = reference_contexts(grid, block, tile);
    read_reference_frames(decoder, cdfs, config)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionModeConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub skip_mode: bool,
    pub switchable: bool,
    pub force_integer_mv: bool,
    pub global_mode: bool,
    pub global_type: GlobalMotionType,
    pub compound: bool,
    pub interintra: bool,
    pub reference_frame: i8,
    pub motion_vector: MotionVector,
    pub allow_warped_motion: bool,
    pub reference_scaled: bool,
}

pub fn read_motion_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: MotionModeConfig,
) -> Result<MotionMode, Error> {
    let (width, height) = config.size.dimensions();
    if config.skip_mode
        || !config.switchable
        || width.min(height) < 8
        || (!config.force_integer_mv
            && config.global_mode
            && config.global_type > GlobalMotionType::Translation)
        || config.compound
        || config.interintra
        || !has_overlappable_candidates(grid, config.block, config.tile)
    {
        return Ok(MotionMode::Simple);
    }
    let samples = collect_warp_samples(
        grid,
        config.block,
        config.tile,
        config.reference_frame,
        config.motion_vector,
    )?;
    if config.force_integer_mv
        || samples.is_empty()
        || !config.allow_warped_motion
        || config.reference_scaled
    {
        return Ok(if cdfs.read_use_obmc(decoder, config.size)? {
            MotionMode::Obmc
        } else {
            MotionMode::Simple
        });
    }
    match cdfs.read_motion_mode(decoder, config.size)? {
        0 => Ok(MotionMode::Simple),
        1 => Ok(MotionMode::Obmc),
        2 => Ok(MotionMode::LocalWarp),
        _ => Err(Error::InvalidObu),
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InterIntraMode {
    pub enabled: bool,
    pub mode: u8,
    pub wedge: bool,
    pub wedge_index: u8,
}

pub fn read_inter_intra_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    size: BlockSize,
    skip_mode: bool,
    enabled: bool,
    compound: bool,
) -> Result<InterIntraMode, Error> {
    if skip_mode || !enabled || compound || !(3..=9).contains(&(size as u8)) {
        return Ok(InterIntraMode::default());
    }
    let context = size.size_group().checked_sub(1).ok_or(Error::InvalidObu)?;
    if !cdfs.read_inter_intra(decoder, context)? {
        return Ok(InterIntraMode::default());
    }
    let mode = cdfs.read_inter_intra_mode(decoder, context)?;
    if mode >= 4 {
        return Err(Error::InvalidObu);
    }
    let wedge = cdfs.read_wedge_inter_intra(decoder, size)?;
    let wedge_index = if wedge {
        cdfs.read_wedge_index(decoder, size)?
    } else {
        0
    };
    Ok(InterIntraMode {
        enabled: true,
        mode,
        wedge,
        wedge_index,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CompoundType {
    #[default]
    Average,
    Distance,
    Wedge,
    DifferenceWeighted,
    Intra,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompoundTypeResult {
    pub kind: CompoundType,
    pub group_index: u8,
    pub compound_index: u8,
    pub wedge_index: u8,
    pub wedge_sign: bool,
    pub mask_type: bool,
}

pub fn compound_contexts(
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    equal_reference_distance: bool,
) -> (u8, u8) {
    let above = (block.row > tile.row_start)
        .then(|| grid.get(block.row - 1, block.column))
        .flatten();
    let left = (block.column > tile.column_start)
        .then(|| grid.get(block.row, block.column - 1))
        .flatten();
    let contribution = |state: Option<&crate::block_state::BlockState>, group: bool| {
        state.map_or(0, |state| {
            if state.reference_frames[1] > 0 {
                if group {
                    state.compound_group_index
                } else {
                    state.compound_index
                }
            } else {
                3 * u8::from(state.reference_frames[0] == 7 && group)
                    + u8::from(state.reference_frames[0] == 7 && !group)
            }
        })
    };
    let group = (contribution(above, true) + contribution(left, true)).min(5);
    let index = 3 * u8::from(equal_reference_distance)
        + contribution(above, false)
        + contribution(left, false);
    (group, index.min(5))
}

#[derive(Clone, Copy, Debug)]
pub struct CompoundTypeConfig {
    pub size: BlockSize,
    pub skip_mode: bool,
    pub compound: bool,
    pub inter_intra: InterIntraMode,
    pub enable_masked_compound: bool,
    pub enable_joint_compound: bool,
    pub contexts: (u8, u8),
}

pub fn read_compound_type(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    config: CompoundTypeConfig,
) -> Result<CompoundTypeResult, Error> {
    let mut result = CompoundTypeResult {
        compound_index: 1,
        ..CompoundTypeResult::default()
    };
    if config.skip_mode {
        return Ok(result);
    }
    if !config.compound {
        result.kind = if config.inter_intra.enabled {
            if config.inter_intra.wedge {
                result.wedge_index = config.inter_intra.wedge_index;
                CompoundType::Wedge
            } else {
                CompoundType::Intra
            }
        } else {
            CompoundType::Average
        };
        return Ok(result);
    }
    result.group_index = u8::from(
        config.enable_masked_compound && cdfs.read_comp_group_idx(decoder, config.contexts.0)?,
    );
    if result.group_index == 0 {
        result.compound_index = u8::from(
            !config.enable_joint_compound || cdfs.read_compound_idx(decoder, config.contexts.1)?,
        );
        result.kind = if result.compound_index != 0 {
            CompoundType::Average
        } else {
            CompoundType::Distance
        };
        return Ok(result);
    }
    let wedge_bits = u8::from(matches!(config.size as u8, 3..=9 | 18..=19)) * 4;
    result.kind = if wedge_bits != 0 && !cdfs.read_compound_type(decoder, config.size)? {
        CompoundType::Wedge
    } else {
        CompoundType::DifferenceWeighted
    };
    if result.kind == CompoundType::Wedge {
        result.wedge_index = cdfs.read_wedge_index(decoder, config.size)?;
        result.wedge_sign = decoder.read_bool()?;
    } else {
        result.mask_type = decoder.read_bool()?;
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
pub struct InterpolationFilterConfig {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub size: BlockSize,
    pub references: [i8; 2],
    pub skip_mode: bool,
    pub motion_mode: MotionMode,
    pub y_mode: u8,
    pub global_types: [GlobalMotionType; 2],
    pub frame_filter: u8,
    pub dual_filter: bool,
}

pub fn read_interpolation_filters(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: InterpolationFilterConfig,
) -> Result<[u8; 2], Error> {
    if config.frame_filter > 4 {
        return Err(Error::InvalidObu);
    }
    if config.frame_filter != 4 {
        return Ok([config.frame_filter; 2]);
    }
    let (width, height) = config.size.dimensions();
    let large = width.min(height) >= 8;
    let needs_filter = !config.skip_mode
        && config.motion_mode != MotionMode::LocalWarp
        && if large && config.y_mode == 16 {
            config.global_types[0] == GlobalMotionType::Translation
        } else if large && config.y_mode == 24 {
            config.global_types[0] == GlobalMotionType::Translation
                || config.global_types[1] == GlobalMotionType::Translation
        } else {
            true
        };
    let mut filters = [0u8; 2];
    let directions = if config.dual_filter { 2 } else { 1 };
    for (direction, filter) in filters.iter_mut().enumerate().take(directions) {
        if !needs_filter {
            *filter = 0;
            continue;
        }
        let left = (config.block.column > config.tile.column_start)
            .then(|| grid.get(config.block.row, config.block.column - 1))
            .flatten();
        let above = (config.block.row > config.tile.row_start)
            .then(|| grid.get(config.block.row - 1, config.block.column))
            .flatten();
        let neighbor_filter = |state: Option<&crate::block_state::BlockState>| {
            state.and_then(|state| {
                state
                    .reference_frames
                    .contains(&config.references[0])
                    .then_some(state.interpolation_filters[direction])
            })
        };
        let left = neighbor_filter(left);
        let above = neighbor_filter(above);
        let neighbor_type = match (left, above) {
            (Some(left), Some(above)) if left == above => left,
            (Some(left), None) => left,
            (None, Some(above)) => above,
            _ => 3,
        };
        let context = ((direction as u8 * 2 + u8::from(config.references[1] > 0)) * 4)
            .checked_add(neighbor_type)
            .ok_or(Error::LimitExceeded)?;
        *filter = cdfs.read_interp_filter(decoder, context)?;
    }
    if !config.dual_filter {
        filters[1] = filters[0];
    }
    Ok(filters)
}

#[derive(Clone, Copy, Debug)]
pub struct InterPostMotionConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub skip_mode: bool,
    pub references: [i8; 2],
    pub motion: InterMotionResult,
    pub global_types: [GlobalMotionType; 2],
    pub enable_inter_intra: bool,
    pub motion_mode_switchable: bool,
    pub force_integer_mv: bool,
    pub allow_warped_motion: bool,
    pub reference_scaled: bool,
    pub enable_masked_compound: bool,
    pub enable_joint_compound: bool,
    pub equal_reference_distance: bool,
    pub frame_filter: u8,
    pub dual_filter: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterPostMotionResult {
    pub references: [i8; 2],
    pub inter_intra: InterIntraMode,
    pub motion_mode: MotionMode,
    pub compound: CompoundTypeResult,
    pub interpolation_filters: [u8; 2],
}

pub fn read_inter_post_motion(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: InterPostMotionConfig,
) -> Result<InterPostMotionResult, Error> {
    let is_compound = config.references[1] > 0;
    let inter_intra = read_inter_intra_mode(
        decoder,
        cdfs,
        config.size,
        config.skip_mode,
        config.enable_inter_intra,
        is_compound,
    )?;
    let mut references = config.references;
    if inter_intra.enabled {
        references[1] = 0;
    }
    let motion_mode = read_motion_mode(
        decoder,
        cdfs,
        grid,
        MotionModeConfig {
            block: config.block,
            size: config.size,
            tile: config.tile,
            skip_mode: config.skip_mode,
            switchable: config.motion_mode_switchable,
            force_integer_mv: config.force_integer_mv,
            global_mode: matches!(config.motion.y_mode, 16 | 24),
            global_type: config.global_types[0],
            compound: is_compound,
            interintra: inter_intra.enabled,
            reference_frame: references[0],
            motion_vector: config.motion.motion_vectors[0],
            allow_warped_motion: config.allow_warped_motion,
            reference_scaled: config.reference_scaled,
        },
    )?;
    let contexts = compound_contexts(
        grid,
        config.block,
        config.tile,
        config.equal_reference_distance,
    );
    let compound = read_compound_type(
        decoder,
        cdfs,
        CompoundTypeConfig {
            size: config.size,
            skip_mode: config.skip_mode,
            compound: is_compound,
            inter_intra,
            enable_masked_compound: config.enable_masked_compound,
            enable_joint_compound: config.enable_joint_compound,
            contexts,
        },
    )?;
    let interpolation_filters = read_interpolation_filters(
        decoder,
        cdfs,
        grid,
        InterpolationFilterConfig {
            block: config.block,
            tile: config.tile,
            size: config.size,
            references,
            skip_mode: config.skip_mode,
            motion_mode,
            y_mode: config.motion.y_mode,
            global_types: config.global_types,
            frame_filter: config.frame_filter,
            dual_filter: config.dual_filter,
        },
    )?;
    Ok(InterPostMotionResult {
        references,
        inter_intra,
        motion_mode,
        compound,
        interpolation_filters,
    })
}

impl IntraMode {
    fn from_symbol(symbol: u8) -> Result<Self, Error> {
        match symbol {
            0 => Ok(Self::Dc),
            1 => Ok(Self::Vertical),
            2 => Ok(Self::Horizontal),
            3 => Ok(Self::D45),
            4 => Ok(Self::D135),
            5 => Ok(Self::D113),
            6 => Ok(Self::D157),
            7 => Ok(Self::D203),
            8 => Ok(Self::D67),
            9 => Ok(Self::Smooth),
            10 => Ok(Self::SmoothVertical),
            11 => Ok(Self::SmoothHorizontal),
            12 => Ok(Self::Paeth),
            _ => Err(Error::InvalidObu),
        }
    }

    pub const fn is_directional(self) -> bool {
        matches!(
            self,
            Self::Vertical
                | Self::Horizontal
                | Self::D45
                | Self::D135
                | Self::D113
                | Self::D157
                | Self::D203
                | Self::D67
        )
    }
}

impl ChromaMode {
    fn from_symbol(symbol: u8) -> Result<Self, Error> {
        match symbol {
            0 => Ok(Self::Dc),
            1 => Ok(Self::Vertical),
            2 => Ok(Self::Horizontal),
            3 => Ok(Self::D45),
            4 => Ok(Self::D135),
            5 => Ok(Self::D113),
            6 => Ok(Self::D157),
            7 => Ok(Self::D203),
            8 => Ok(Self::D67),
            9 => Ok(Self::Smooth),
            10 => Ok(Self::SmoothVertical),
            11 => Ok(Self::SmoothHorizontal),
            12 => Ok(Self::Paeth),
            13 => Ok(Self::Cfl),
            _ => Err(Error::InvalidObu),
        }
    }

    pub const fn is_directional(self) -> bool {
        matches!(
            self,
            Self::Vertical
                | Self::Horizontal
                | Self::D45
                | Self::D135
                | Self::D113
                | Self::D157
                | Self::D203
                | Self::D67
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChromaIntraResult {
    pub mode: ChromaMode,
    pub angle_delta: i8,
    pub cfl_alphas: [i8; 2],
}

pub const fn palette_block_size_context(size: BlockSize) -> Result<u8, Error> {
    let (width, height) = size.dimensions();
    if width < 8 || height < 8 || width > 64 || height > 64 {
        return Err(Error::InvalidObu);
    }
    let context = width.ilog2() + height.ilog2() - 6;
    if context > 6 {
        return Err(Error::InvalidObu);
    }
    Ok(context as u8)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaletteModeConfig {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub size: BlockSize,
    pub enabled: bool,
    pub y_mode: IntraMode,
    pub uv_mode: ChromaMode,
    pub has_chroma: bool,
}

pub fn read_palette_sizes(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: PaletteModeConfig,
) -> Result<[u8; 2], Error> {
    if !config.enabled {
        return Ok([0; 2]);
    }
    let size_context = palette_block_size_context(config.size)?;
    let mut sizes = [0; 2];
    if config.y_mode == IntraMode::Dc {
        let above = config.block.row > config.tile.row_start
            && grid
                .get(config.block.row - 1, config.block.column)
                .is_some_and(|state| state.palette_sizes[0] > 0);
        let left = config.block.column > config.tile.column_start
            && grid
                .get(config.block.row, config.block.column - 1)
                .is_some_and(|state| state.palette_sizes[0] > 0);
        let context = u8::from(above) + u8::from(left);
        if cdfs.read_palette_y_mode(decoder, size_context, context)? {
            sizes[0] = cdfs.read_palette_size(decoder, size_context, false)?;
        }
    }
    if config.has_chroma
        && config.uv_mode == ChromaMode::Dc
        && cdfs.read_palette_uv_mode(decoder, sizes[0] > 0)?
    {
        sizes[1] = cdfs.read_palette_size(decoder, size_context, true)?;
    }
    Ok(sizes)
}

pub const fn cfl_allowed(
    lossless: bool,
    block_size: BlockSize,
    plane_residual_size: BlockSize,
) -> bool {
    if lossless {
        plane_residual_size as u8 == BlockSize::Block4x4 as u8
    } else {
        let (width, height) = block_size.dimensions();
        width <= 32 && height <= 32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SkipModeConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub skip_mode_present: bool,
    pub segment: ActiveSegmentFeatures,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TxSizeConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub lossless: bool,
    pub allow_select: bool,
    pub tx_mode_select: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VarTxSizeConfig {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub size: BlockSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VarTxNode {
    pub row: u32,
    pub column: u32,
    pub size: TxSize,
    pub depth: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockTxSizeConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub lossless: bool,
    pub skip: bool,
    pub is_inter: bool,
    pub tx_mode_select: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformTreeNode {
    pub start_x: u32,
    pub start_y: u32,
    pub width: u8,
    pub height: u8,
    pub frame_width: u32,
    pub frame_height: u32,
}

pub fn read_spatial_segment_id(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    last_active_segment_id: u8,
    skip: bool,
) -> Result<u8, Error> {
    if last_active_segment_id >= 8 {
        return Err(Error::InvalidObu);
    }
    let (prediction, context) = grid.segment_prediction(block, tile);
    if prediction > last_active_segment_id {
        return Err(Error::InvalidObu);
    }
    if skip {
        return Ok(prediction);
    }
    let coded = cdfs.read_segment_id(decoder, context)?;
    let maximum = last_active_segment_id + 1;
    if coded >= maximum {
        return Err(Error::InvalidObu);
    }
    negative_deinterleave(coded, prediction, maximum)
}

#[derive(Debug)]
pub struct InterSegmentConfig<'a> {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub segmentation: &'a Segmentation,
    pub previous_segments: Option<&'a MiGrid>,
    pub pre_skip: bool,
    pub skip: bool,
}

pub fn read_inter_segment_id(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    prediction_contexts: &mut SegmentPredictionContexts,
    config: InterSegmentConfig<'_>,
) -> Result<u8, Error> {
    if !config.segmentation.enabled {
        return Ok(0);
    }
    let predicted = match config.previous_segments {
        Some(previous) => previous.minimum_segment_id(config.block)?,
        None => 0,
    };
    if !config.segmentation.update_map {
        return Ok(predicted);
    }
    let pre_skip_enabled = config.segmentation.segment_id_pre_skip();
    if config.pre_skip && !pre_skip_enabled {
        return Ok(0);
    }
    let last_active = config.segmentation.last_active_segment_id();
    if !config.pre_skip && config.skip {
        prediction_contexts.update(config.block, false)?;
        return read_spatial_segment_id(
            decoder,
            cdfs,
            grid,
            config.block,
            config.tile,
            last_active,
            false,
        );
    }
    if config.segmentation.temporal_update {
        let context = prediction_contexts.context(config.block)?;
        let use_prediction = cdfs.read_segment_id_predicted(decoder, context)?;
        let segment = if use_prediction {
            predicted
        } else {
            read_spatial_segment_id(
                decoder,
                cdfs,
                grid,
                config.block,
                config.tile,
                last_active,
                false,
            )?
        };
        prediction_contexts.update(config.block, use_prediction)?;
        Ok(segment)
    } else {
        read_spatial_segment_id(
            decoder,
            cdfs,
            grid,
            config.block,
            config.tile,
            last_active,
            false,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockModePrefixConfig<'a> {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub segmentation: &'a Segmentation,
    pub previous_segments: Option<&'a MiGrid>,
    pub skip_mode_present: bool,
    pub frame_is_intra: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockModePrefix {
    pub segment_id: u8,
    pub segment: ActiveSegmentFeatures,
    pub skip_mode: bool,
    pub skip: bool,
    pub is_inter: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockModePreInter {
    pub segment_id: u8,
    pub segment: ActiveSegmentFeatures,
    pub skip_mode: bool,
    pub skip: bool,
}

pub fn read_block_mode_prefix(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    prediction_contexts: &mut SegmentPredictionContexts,
    config: BlockModePrefixConfig<'_>,
) -> Result<BlockModePrefix, Error> {
    let pre_inter = read_block_mode_pre_inter(decoder, cdfs, grid, prediction_contexts, config)?;
    finish_block_mode_prefix(
        decoder,
        cdfs,
        grid,
        config.block,
        config.tile,
        pre_inter,
        config.frame_is_intra,
    )
}

/// Reads mode-info through segmentation and skip. Delta-Q, delta-LF and CDEF
/// syntax occur after this stage and before `finish_block_mode_prefix`.
pub fn read_block_mode_pre_inter(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    prediction_contexts: &mut SegmentPredictionContexts,
    config: BlockModePrefixConfig<'_>,
) -> Result<BlockModePreInter, Error> {
    let pre_skip = config.segmentation.segment_id_pre_skip();
    if config.frame_is_intra {
        let mut segment_id = if config.segmentation.enabled && pre_skip {
            read_spatial_segment_id(
                decoder,
                cdfs,
                grid,
                config.block,
                config.tile,
                config.segmentation.last_active_segment_id(),
                false,
            )?
        } else {
            0
        };
        let mut segment = active_segment_features(config.segmentation, segment_id)?;
        let skip = read_skip(
            decoder,
            cdfs,
            grid,
            config.block,
            config.tile,
            pre_skip,
            segment,
        )?;
        if config.segmentation.enabled && !pre_skip {
            segment_id = read_spatial_segment_id(
                decoder,
                cdfs,
                grid,
                config.block,
                config.tile,
                config.segmentation.last_active_segment_id(),
                skip,
            )?;
            segment = active_segment_features(config.segmentation, segment_id)?;
        }
        return Ok(BlockModePreInter {
            segment_id,
            segment,
            skip_mode: false,
            skip,
        });
    }
    let mut segment_id = read_inter_segment_id(
        decoder,
        cdfs,
        grid,
        prediction_contexts,
        InterSegmentConfig {
            block: config.block,
            tile: config.tile,
            segmentation: config.segmentation,
            previous_segments: config.previous_segments,
            pre_skip: true,
            skip: false,
        },
    )?;
    let mut segment = active_segment_features(config.segmentation, segment_id)?;
    let skip_mode = read_skip_mode(
        decoder,
        cdfs,
        grid,
        SkipModeConfig {
            block: config.block,
            size: config.size,
            tile: config.tile,
            skip_mode_present: config.skip_mode_present,
            segment,
        },
    )?;
    let skip = if skip_mode {
        true
    } else {
        read_skip(
            decoder,
            cdfs,
            grid,
            config.block,
            config.tile,
            pre_skip,
            segment,
        )?
    };
    if !pre_skip {
        segment_id = read_inter_segment_id(
            decoder,
            cdfs,
            grid,
            prediction_contexts,
            InterSegmentConfig {
                block: config.block,
                tile: config.tile,
                segmentation: config.segmentation,
                previous_segments: config.previous_segments,
                pre_skip: false,
                skip,
            },
        )?;
        segment = active_segment_features(config.segmentation, segment_id)?;
    }
    Ok(BlockModePreInter {
        segment_id,
        segment,
        skip_mode,
        skip,
    })
}

pub fn finish_block_mode_prefix(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    pre_inter: BlockModePreInter,
    frame_is_intra: bool,
) -> Result<BlockModePrefix, Error> {
    let is_inter = if frame_is_intra {
        false
    } else {
        read_is_inter(
            decoder,
            cdfs,
            grid,
            block,
            tile,
            pre_inter.skip_mode,
            pre_inter.segment,
        )?
    };
    Ok(BlockModePrefix {
        segment_id: pre_inter.segment_id,
        segment: pre_inter.segment,
        skip_mode: pre_inter.skip_mode,
        skip: pre_inter.skip,
        is_inter,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CdefIndexConfig {
    pub block: BlockRect,
    pub skip: bool,
    pub coded_lossless: bool,
    pub enabled: bool,
    pub allow_intrabc: bool,
    pub bits: u8,
}

/// Reads section 5.11.56 once per covered 64x64 CDEF region.
pub fn read_cdef_index(
    decoder: &mut SymbolDecoder<'_>,
    indices: &mut CdefIndexGrid,
    config: CdefIndexConfig,
) -> Result<Option<u8>, Error> {
    if config.bits > 3 {
        return Err(Error::InvalidObu);
    }
    if config.skip || config.coded_lossless || !config.enabled || config.allow_intrabc {
        return Ok(None);
    }
    let existing = indices
        .get(config.block.row, config.block.column)
        .ok_or(Error::InvalidObu)?;
    if let Some(index) = existing {
        return Ok(Some(index));
    }
    let index = u8::try_from(decoder.read_literal(config.bits)?).map_err(|_| Error::InvalidObu)?;
    indices.fill_block(config.block, index)?;
    Ok(Some(index))
}

pub fn read_skip_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: SkipModeConfig,
) -> Result<bool, Error> {
    let (width, height) = config.size.dimensions();
    if config.segment.skip
        || config.segment.reference_frame.is_some()
        || config.segment.global_mv
        || !config.skip_mode_present
        || width < 8
        || height < 8
    {
        return Ok(false);
    }
    cdfs.read_skip_mode(decoder, grid.skip_mode_context(config.block, config.tile))
}

pub fn read_skip(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    segment_id_pre_skip: bool,
    segment: ActiveSegmentFeatures,
) -> Result<bool, Error> {
    if segment_id_pre_skip && segment.skip {
        Ok(true)
    } else {
        cdfs.read_skip(decoder, grid.skip_context(block, tile))
    }
}

pub fn read_is_inter(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    skip_mode: bool,
    segment: ActiveSegmentFeatures,
) -> Result<bool, Error> {
    if skip_mode {
        Ok(true)
    } else if let Some(reference_frame) = segment.reference_frame {
        Ok(reference_frame != 0)
    } else if segment.global_mv {
        Ok(true)
    } else {
        cdfs.read_is_inter(decoder, grid.is_inter_context(block, tile))
    }
}

pub fn read_tx_size(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: TxSizeConfig,
) -> Result<TxSize, Error> {
    if config.lossless {
        return Ok(TxSize::Tx4x4);
    }
    let maximum = config.size.maximum_transform_size();
    if config.size == BlockSize::Block4x4 || !config.allow_select || !config.tx_mode_select {
        return Ok(maximum);
    }
    let maximum_depth = config.size.max_transform_depth();
    let context = grid.tx_depth_context(config.block, config.tile, maximum);
    let depth = cdfs
        .read_tx_depth(decoder, maximum_depth, context)
        .map_err(|error| {
            if error == Error::InvalidObu {
                Error::InvalidTransformDepth {
                    row: config.block.row,
                    column: config.block.column,
                    block_size: config.size,
                    maximum,
                    context,
                }
            } else {
                error
            }
        })?;
    if depth > maximum_depth.min(2) {
        return Err(Error::InvalidObu);
    }
    let mut size = maximum;
    for _ in 0..depth {
        size = size.split();
    }
    Ok(size)
}

/// Reads section 5.11.16 and records the selected size in every covered MI.
pub fn read_block_tx_size(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &mut MiGrid,
    config: BlockTxSizeConfig,
) -> Result<TxSize, Error> {
    if config.tx_mode_select
        && config.size != BlockSize::Block4x4
        && config.is_inter
        && !config.skip
        && !config.lossless
    {
        let maximum = config.size.maximum_transform_size();
        let (width, height) = config.size.dimensions();
        let (step_width, step_height) = maximum.dimensions();
        let mut row = 0u32;
        while row < u32::from(height / 4) {
            let mut column = 0u32;
            while column < u32::from(width / 4) {
                read_var_tx_size(
                    decoder,
                    cdfs,
                    grid,
                    VarTxSizeConfig {
                        block: config.block,
                        tile: config.tile,
                        size: config.size,
                    },
                    VarTxNode {
                        row: config.block.row + row,
                        column: config.block.column + column,
                        size: maximum,
                        depth: 0,
                    },
                )?;
                column += u32::from(step_width / 4);
            }
            row += u32::from(step_height / 4);
        }
        let last_row = config
            .block
            .row
            .checked_add(u32::from(config.block.height_mi) - 1)
            .ok_or(Error::LimitExceeded)?
            .min(grid.rows().checked_sub(1).ok_or(Error::InvalidObu)?);
        let last_column = config
            .block
            .column
            .checked_add(u32::from(config.block.width_mi) - 1)
            .ok_or(Error::LimitExceeded)?
            .min(grid.columns().checked_sub(1).ok_or(Error::InvalidObu)?);
        return grid
            .get(last_row, last_column)
            .and_then(|state| state.tx_size)
            .ok_or(Error::InvalidObu);
    }
    let size = read_tx_size(
        decoder,
        cdfs,
        grid,
        TxSizeConfig {
            block: config.block,
            size: config.size,
            tile: config.tile,
            lossless: config.lossless,
            // Section 5.11.16 reads tx_depth only when the block is not
            // skipped. This applies equally to intra and inter blocks.
            allow_select: !config.skip,
            tx_mode_select: config.tx_mode_select,
        },
    )?;
    let (tx_width, tx_height) = size.dimensions();
    let mut row = 0u32;
    while row < u32::from(config.block.height_mi)
        && config.block.row.saturating_add(row) < grid.rows()
    {
        let mut column = 0u32;
        while column < u32::from(config.block.width_mi)
            && config.block.column.saturating_add(column) < grid.columns()
        {
            grid.fill_tx_size(config.block.row + row, config.block.column + column, size)?;
            column += u32::from(tx_width / 4);
        }
        row += u32::from(tx_height / 4);
    }
    Ok(size)
}

/// Reads section 5.11.17's variable transform-size tree for an inter block.
/// The resulting leaf size is recorded in every covered `MiGrid` cell.
pub fn read_var_tx_size(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &mut MiGrid,
    config: VarTxSizeConfig,
    node: VarTxNode,
) -> Result<(), Error> {
    let VarTxNode {
        row,
        column,
        size,
        depth,
    } = node;
    if row >= grid.rows() || column >= grid.columns() {
        return Ok(());
    }
    let split = if size == TxSize::Tx4x4 || depth == 2 {
        false
    } else {
        let (above_width, left_height) =
            grid.inter_tx_neighbor_dimensions(row, column, config.block, config.tile);
        let (tx_width, tx_height) = size.dimensions();
        let (block_width, block_height) = config.size.dimensions();
        let maximum = block_width.max(block_height).min(64);
        let maximum_square = match maximum {
            0..=4 => TxSize::Tx4x4,
            5..=8 => TxSize::Tx8x8,
            9..=16 => TxSize::Tx16x16,
            17..=32 => TxSize::Tx32x32,
            _ => TxSize::Tx64x64,
        };
        let square_up = size.square_up();
        let context = u8::from(square_up != maximum_square) * 3
            + (4 - maximum_square as u8) * 6
            + u8::from(above_width < u16::from(tx_width))
            + u8::from(left_height < u16::from(tx_height));
        cdfs.read_txfm_split(decoder, context).map_err(|error| {
            if error == Error::InvalidObu {
                Error::InvalidTransformPosition {
                    row,
                    column,
                    size,
                    depth,
                    context,
                }
            } else {
                error
            }
        })?
    };
    if !split {
        return grid.fill_tx_size(row, column, size);
    }

    let sub_size = size.split();
    let (width, height) = size.dimensions();
    let (step_width, step_height) = sub_size.dimensions();
    let height_mi = u32::from(height / 4);
    let width_mi = u32::from(width / 4);
    let step_height_mi = u32::from(step_height / 4);
    let step_width_mi = u32::from(step_width / 4);
    let mut y = 0;
    while y < height_mi {
        let mut x = 0;
        while x < width_mi {
            read_var_tx_size(
                decoder,
                cdfs,
                grid,
                config,
                VarTxNode {
                    row: row.checked_add(y).ok_or(Error::LimitExceeded)?,
                    column: column.checked_add(x).ok_or(Error::LimitExceeded)?,
                    size: sub_size,
                    depth: depth + 1,
                },
            )?;
            x += step_width_mi;
        }
        y += step_height_mi;
    }
    Ok(())
}

/// Walks section 5.11.36's transform tree using the previously decoded
/// `InterTxSizes` values stored in `grid`.
pub fn walk_inter_transform_tree<F>(
    grid: &MiGrid,
    node: TransformTreeNode,
    visit: &mut F,
) -> Result<(), Error>
where
    F: FnMut(u32, u32, TxSize) -> Result<(), Error>,
{
    if node.start_x >= node.frame_width || node.start_y >= node.frame_height {
        return Ok(());
    }
    if node.width < 4 || node.height < 4 {
        return Err(Error::InvalidObu);
    }
    let row = node.start_y / 4;
    let column = node.start_x / 4;
    let leaf = grid
        .get(row, column)
        .and_then(|state| state.tx_size)
        .ok_or(Error::InvalidObu)?;
    let (leaf_width, leaf_height) = leaf.dimensions();
    if node.width <= leaf_width && node.height <= leaf_height {
        return visit(
            node.start_x,
            node.start_y,
            TxSize::from_dimensions(node.width, node.height)?,
        );
    }

    let (child_width, child_height, children): (u8, u8, u8) = if node.width > node.height {
        (node.width / 2, node.height, 2)
    } else if node.width < node.height {
        (node.width, node.height / 2, 2)
    } else {
        (node.width / 2, node.height / 2, 4)
    };
    for child in 0..children {
        let x_offset = if children == 2 && node.width > node.height {
            child * child_width
        } else if children == 4 {
            (child & 1) * child_width
        } else {
            0
        };
        let y_offset = if children == 2 && node.width < node.height {
            child * child_height
        } else if children == 4 {
            (child >> 1) * child_height
        } else {
            0
        };
        walk_inter_transform_tree(
            grid,
            TransformTreeNode {
                start_x: node.start_x + u32::from(x_offset),
                start_y: node.start_y + u32::from(y_offset),
                width: child_width,
                height: child_height,
                ..node
            },
            visit,
        )?;
    }
    Ok(())
}

pub fn read_luma_intra_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    size: BlockSize,
) -> Result<(IntraMode, i8), Error> {
    let mode = IntraMode::from_symbol(cdfs.read_y_mode(decoder, size.size_group())?)?;
    let angle_delta = if (size as u8) >= BlockSize::Block8x8 as u8 && mode.is_directional() {
        cdfs.read_angle_delta(decoder, mode as u8 - IntraMode::Vertical as u8)?
    } else {
        0
    };
    Ok((mode, angle_delta))
}

pub fn read_chroma_intra_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    y_mode: IntraMode,
    block_size: BlockSize,
    plane_residual_size: BlockSize,
    lossless: bool,
) -> Result<ChromaIntraResult, Error> {
    let allow_cfl = cfl_allowed(lossless, block_size, plane_residual_size);
    let mode = ChromaMode::from_symbol(cdfs.read_uv_mode(decoder, y_mode as u8, allow_cfl)?)?;
    if mode == ChromaMode::Cfl && !allow_cfl {
        return Err(Error::InvalidObu);
    }
    let angle_delta = if (block_size as u8) >= BlockSize::Block8x8 as u8 && mode.is_directional() {
        cdfs.read_angle_delta(decoder, mode as u8 - ChromaMode::Vertical as u8)?
    } else {
        0
    };
    let cfl_alphas = if mode == ChromaMode::Cfl {
        read_cfl_alphas(decoder, cdfs)?
    } else {
        [0, 0]
    };
    Ok(ChromaIntraResult {
        mode,
        angle_delta,
        cfl_alphas,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntraBlockModeConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub tile: TileBounds,
    pub plane_residual_size: BlockSize,
    pub lossless: bool,
    pub has_chroma: bool,
    pub palette_enabled: bool,
    pub filter_intra_enabled: bool,
    pub frame_is_intra: bool,
    pub allow_intrabc: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntraBlockMode {
    pub use_intrabc: bool,
    pub y_mode: IntraMode,
    pub y_angle_delta: i8,
    pub chroma: ChromaIntraResult,
    pub palette_sizes: [u8; 2],
    pub filter_intra_mode: Option<u8>,
}

/// Returns the intra direction used to select the transform-type CDF.
///
/// Filter-intra predictions use their normative directional equivalents for
/// transform signaling instead of the block's base luma mode.
pub const fn intra_tx_type_direction(
    mode: IntraMode,
    filter_intra_mode: Option<u8>,
) -> Result<u8, Error> {
    match filter_intra_mode {
        None => Ok(mode as u8),
        Some(0 | 4) => Ok(IntraMode::Dc as u8),
        Some(1) => Ok(IntraMode::Vertical as u8),
        Some(2) => Ok(IntraMode::Horizontal as u8),
        Some(3) => Ok(IntraMode::D157 as u8),
        Some(_) => Err(Error::InvalidObu),
    }
}

pub fn read_intra_block_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    grid: &MiGrid,
    config: IntraBlockModeConfig,
) -> Result<IntraBlockMode, Error> {
    let use_intrabc =
        config.frame_is_intra && config.allow_intrabc && cdfs.read_intrabc(decoder)?;
    if use_intrabc {
        return Ok(IntraBlockMode {
            use_intrabc: true,
            y_mode: IntraMode::Dc,
            y_angle_delta: 0,
            chroma: ChromaIntraResult {
                mode: ChromaMode::Dc,
                angle_delta: 0,
                cfl_alphas: [0; 2],
            },
            palette_sizes: [0; 2],
            filter_intra_mode: None,
        });
    }
    let (y_mode, y_angle_delta) = if config.frame_is_intra {
        let above = if config.block.row > config.tile.row_start {
            grid.get(config.block.row - 1, config.block.column)
                .map_or(0, |state| state.prediction_mode)
        } else {
            0
        };
        let left = if config.block.column > config.tile.column_start {
            grid.get(config.block.row, config.block.column - 1)
                .map_or(0, |state| state.prediction_mode)
        } else {
            0
        };
        let mode = IntraMode::from_symbol(cdfs.read_intra_frame_y_mode(decoder, above, left)?)?;
        let delta = if config.size as u8 >= BlockSize::Block8x8 as u8 && mode.is_directional() {
            cdfs.read_angle_delta(decoder, mode as u8 - IntraMode::Vertical as u8)?
        } else {
            0
        };
        (mode, delta)
    } else {
        read_luma_intra_mode(decoder, cdfs, config.size)?
    };
    let chroma = if config.has_chroma {
        read_chroma_intra_mode(
            decoder,
            cdfs,
            y_mode,
            config.size,
            config.plane_residual_size,
            config.lossless,
        )?
    } else {
        ChromaIntraResult {
            mode: ChromaMode::Dc,
            angle_delta: 0,
            cfl_alphas: [0; 2],
        }
    };
    let palette_sizes = read_palette_sizes(
        decoder,
        cdfs,
        grid,
        PaletteModeConfig {
            block: config.block,
            tile: config.tile,
            size: config.size,
            enabled: config.palette_enabled,
            y_mode,
            uv_mode: chroma.mode,
            has_chroma: config.has_chroma,
        },
    )?;
    let filter_intra_mode = read_filter_intra(
        decoder,
        cdfs,
        config.size,
        config.filter_intra_enabled,
        y_mode,
        palette_sizes[0],
    )?;
    Ok(IntraBlockMode {
        use_intrabc,
        y_mode,
        y_angle_delta,
        chroma,
        palette_sizes,
        filter_intra_mode,
    })
}

pub fn read_delta_qindex(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    current_qindex: u8,
    resolution: u8,
) -> Result<u8, Error> {
    if resolution > 3 {
        return Err(Error::InvalidObu);
    }
    let coded = cdfs.read_delta_q_abs(decoder)?;
    let absolute = read_delta_absolute(decoder, coded)?;
    if absolute == 0 {
        return Ok(current_qindex);
    }
    let magnitude = i32::try_from(absolute).map_err(|_| Error::LimitExceeded)?;
    let signed = if decoder.read_bool()? {
        -magnitude
    } else {
        magnitude
    };
    let delta = signed
        .checked_shl(u32::from(resolution))
        .ok_or(Error::LimitExceeded)?;
    Ok((i32::from(current_qindex) + delta).clamp(1, 255) as u8)
}

pub fn read_single_inter_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    new_mv_context: u8,
    zero_mv_context: u8,
    ref_mv_context: u8,
) -> Result<InterMode, Error> {
    if !cdfs.read_new_mv(decoder, new_mv_context)? {
        Ok(InterMode::New)
    } else if !cdfs.read_zero_mv(decoder, zero_mv_context)? {
        Ok(InterMode::Global)
    } else if !cdfs.read_ref_mv(decoder, ref_mv_context)? {
        Ok(InterMode::Nearest)
    } else {
        Ok(InterMode::Near)
    }
}

pub fn compound_inter_mode_context(
    new_mv_context: u8,
    reference_mv_context: u8,
) -> Result<u8, Error> {
    const MAP: [[u8; 5]; 3] = [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];
    MAP.get(usize::from(reference_mv_context >> 1))
        .and_then(|row| row.get(usize::from(new_mv_context.min(4))))
        .copied()
        .ok_or(Error::InvalidObu)
}

pub fn compound_inter_modes(symbol: u8) -> Result<[InterMode; 2], Error> {
    const MODES: [[InterMode; 2]; 8] = [
        [InterMode::Nearest, InterMode::Nearest],
        [InterMode::Near, InterMode::Near],
        [InterMode::Nearest, InterMode::New],
        [InterMode::New, InterMode::Nearest],
        [InterMode::Near, InterMode::New],
        [InterMode::New, InterMode::Near],
        [InterMode::Global, InterMode::Global],
        [InterMode::New, InterMode::New],
    ];
    MODES
        .get(usize::from(symbol))
        .copied()
        .ok_or(Error::InvalidObu)
}

pub fn read_compound_inter_mode(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    motion_contexts: MotionContexts,
) -> Result<[InterMode; 2], Error> {
    let context =
        compound_inter_mode_context(motion_contexts.new_mv, motion_contexts.reference_mv)?;
    compound_inter_modes(cdfs.read_compound_inter_mode(decoder, context)?)
}

#[derive(Debug)]
pub struct InterMotionConfig<'a> {
    pub skip_mode: bool,
    pub forced_global: bool,
    pub compound: bool,
    pub stack: &'a crate::motion::CompleteMotionStack,
    pub global_vectors: [MotionVector; 2],
    pub syntax: MotionVectorSyntax,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterMotionResult {
    pub modes: [InterMode; 2],
    pub y_mode: u8,
    pub reference_mv_index: u8,
    pub motion_vectors: [MotionVector; 2],
}

/// Completes section 5.11.23 from an already constructed reference-MV stack.
pub fn read_inter_motion(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    config: InterMotionConfig<'_>,
) -> Result<InterMotionResult, Error> {
    let modes = if config.skip_mode {
        if !config.compound {
            return Err(Error::InvalidObu);
        }
        [InterMode::Nearest, InterMode::Nearest]
    } else if config.forced_global {
        [InterMode::Global, InterMode::Global]
    } else if config.compound {
        read_compound_inter_mode(decoder, cdfs, config.stack.contexts)?
    } else {
        [
            read_single_inter_mode(
                decoder,
                cdfs,
                config.stack.contexts.new_mv,
                config.stack.zero_mv_context,
                config.stack.contexts.reference_mv,
            )?,
            InterMode::Nearest,
        ]
    };
    let lists = if config.compound { 2 } else { 1 };
    let reference_mv_index = read_reference_mv_index(
        decoder,
        cdfs,
        &modes[..lists],
        &config.stack.contexts,
        config.stack.candidates_found,
    )?;
    let motion_vectors = assign_motion_vectors(
        decoder,
        cdfs,
        &config.stack.stack,
        AssignMotionConfig {
            modes,
            compound: config.compound,
            reference_mv_index,
            global_vectors: config.global_vectors,
            syntax: config.syntax,
        },
    )?;
    let y_mode = inter_modes_to_y_mode(modes, config.compound)?;
    Ok(InterMotionResult {
        modes,
        y_mode,
        reference_mv_index,
        motion_vectors,
    })
}

pub fn inter_modes_to_y_mode(modes: [InterMode; 2], compound: bool) -> Result<u8, Error> {
    if !compound {
        return Ok(match modes[0] {
            InterMode::Nearest => 14,
            InterMode::Near => 15,
            InterMode::Global => 16,
            InterMode::New => 17,
        });
    }
    const COMPOUND: [[InterMode; 2]; 8] = [
        [InterMode::Nearest, InterMode::Nearest],
        [InterMode::Near, InterMode::Near],
        [InterMode::Nearest, InterMode::New],
        [InterMode::New, InterMode::Nearest],
        [InterMode::Near, InterMode::New],
        [InterMode::New, InterMode::Near],
        [InterMode::Global, InterMode::Global],
        [InterMode::New, InterMode::New],
    ];
    COMPOUND
        .iter()
        .position(|&candidate| candidate == modes)
        .and_then(|index| u8::try_from(index).ok())
        .and_then(|index| index.checked_add(18))
        .ok_or(Error::InvalidObu)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssignMotionConfig {
    pub modes: [InterMode; 2],
    pub compound: bool,
    pub reference_mv_index: u8,
    pub global_vectors: [MotionVector; 2],
    pub syntax: MotionVectorSyntax,
}

pub fn read_reference_mv_index(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    modes: &[InterMode],
    motion_contexts: &MotionContexts,
    candidates_found: usize,
) -> Result<u8, Error> {
    if modes.is_empty() || modes.len() > 2 || candidates_found > 8 {
        return Err(Error::InvalidObu);
    }
    let has_new = modes.contains(&InterMode::New);
    let has_near = modes.contains(&InterMode::Near);
    let (start, end) = if has_new {
        (0usize, 2usize)
    } else if has_near {
        (1usize, 3usize)
    } else {
        return Ok(0);
    };
    let mut selected = start;
    for index in start..end {
        if candidates_found > index + 1 {
            if !cdfs.read_drl_mode(decoder, motion_contexts.drl[index])? {
                selected = index;
                break;
            }
            selected = index + 1;
        }
    }
    u8::try_from(selected).map_err(|_| Error::LimitExceeded)
}

pub fn assign_motion_vectors(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    stack: &MotionStack,
    config: AssignMotionConfig,
) -> Result<[MotionVector; 2], Error> {
    let lists = if config.compound { 2 } else { 1 };
    let mut output = [MotionVector::default(); 2];
    for (list, slot) in output.iter_mut().enumerate().take(lists) {
        let mode = config.modes[list];
        if mode == InterMode::Global {
            *slot = config.global_vectors[list];
            continue;
        }
        let position = if mode == InterMode::Nearest
            || (mode == InterMode::New && stack.entries().len() <= 1)
        {
            0
        } else {
            usize::from(config.reference_mv_index)
        };
        let predictor = stack
            .entries()
            .get(position)
            .ok_or(Error::InvalidObu)?
            .vectors[list];
        *slot = if mode == InterMode::New {
            read_motion_vector(decoder, cdfs.motion_vectors(), predictor, config.syntax)?
        } else {
            predictor
        };
    }
    Ok(output)
}

pub fn cfl_signs(symbol: u8) -> Result<(u8, u8), Error> {
    if symbol >= 8 {
        return Err(Error::InvalidObu);
    }
    Ok(((symbol + 1) / 3, (symbol + 1) % 3))
}

pub fn read_cfl_alphas(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
) -> Result<[i8; 2], Error> {
    let (sign_u, sign_v) = cfl_signs(cdfs.read_cfl_signs(decoder)?)?;
    let read = |decoder: &mut SymbolDecoder<'_>,
                cdfs: &mut TileCdfs,
                sign: u8,
                other: u8|
     -> Result<i8, Error> {
        if sign == 0 {
            return Ok(0);
        }
        let context = (sign - 1) * 3 + other;
        let magnitude = i8::try_from(cdfs.read_cfl_alpha(decoder, context)?)
            .map_err(|_| Error::InvalidObu)?
            + 1;
        Ok(if sign == 1 { -magnitude } else { magnitude })
    };
    Ok([
        read(decoder, cdfs, sign_u, sign_v)?,
        read(decoder, cdfs, sign_v, sign_u)?,
    ])
}

pub fn read_filter_intra(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    block_size: BlockSize,
    enabled: bool,
    luma_mode: IntraMode,
    palette_size_y: u8,
) -> Result<Option<u8>, Error> {
    let (width, height) = block_size.dimensions();
    if !enabled || luma_mode != IntraMode::Dc || palette_size_y != 0 || width.max(height) > 32 {
        return Ok(None);
    }
    if !cdfs.read_use_filter_intra(decoder, block_size)? {
        return Ok(None);
    }
    let mode = cdfs.read_filter_intra_mode(decoder)?;
    if mode >= 5 {
        return Err(Error::InvalidObu);
    }
    Ok(Some(mode))
}

pub fn read_delta_lf(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    current_level: i8,
    resolution: u8,
    component: u8,
) -> Result<i8, Error> {
    if resolution > 3 || component >= 4 {
        return Err(Error::InvalidObu);
    }
    let coded = cdfs.read_delta_lf_abs(decoder, component)?;
    let absolute = read_delta_absolute(decoder, coded)?;
    if absolute == 0 {
        return Ok(current_level);
    }
    let magnitude = i32::try_from(absolute).map_err(|_| Error::LimitExceeded)?;
    let signed = if decoder.read_bool()? {
        -magnitude
    } else {
        magnitude
    };
    let delta = signed
        .checked_shl(u32::from(resolution))
        .ok_or(Error::LimitExceeded)?;
    Ok((i32::from(current_level) + delta).clamp(-63, 63) as i8)
}

fn read_delta_absolute(decoder: &mut SymbolDecoder<'_>, coded: u8) -> Result<u32, Error> {
    if coded < 3 {
        return Ok(u32::from(coded));
    }
    if coded != 3 {
        return Err(Error::InvalidObu);
    }
    let remainder_bits = u8::try_from(decoder.read_literal(3)?).map_err(|_| Error::InvalidObu)? + 1;
    let remainder =
        u32::try_from(decoder.read_literal(remainder_bits)?).map_err(|_| Error::InvalidObu)?;
    Ok(remainder + (1u32 << remainder_bits) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdef_index_is_read_once_and_propagated_across_large_blocks() {
        let mut indices = CdefIndexGrid::new(32, 32).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let block = BlockRect::new(0, 0, BlockSize::Block128x128);
        let config = CdefIndexConfig {
            block,
            skip: false,
            coded_lossless: false,
            enabled: true,
            allow_intrabc: false,
            bits: 0,
        };
        assert_eq!(
            read_cdef_index(&mut decoder, &mut indices, config),
            Ok(Some(0))
        );
        assert_eq!(indices.get(20, 20), Some(Some(0)));
        assert_eq!(
            read_cdef_index(&mut decoder, &mut indices, config),
            Ok(Some(0))
        );
    }

    #[test]
    fn block_tx_size_fills_every_mi_for_fixed_transform_mode() {
        let mut grid = MiGrid::new(8, 8).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let block = BlockRect::new(0, 0, BlockSize::Block16x8);
        let size = read_block_tx_size(
            &mut decoder,
            &mut cdfs,
            &mut grid,
            BlockTxSizeConfig {
                block,
                size: BlockSize::Block16x8,
                tile: bounds(),
                lossless: false,
                skip: true,
                is_inter: true,
                tx_mode_select: false,
            },
        )
        .unwrap();
        assert_eq!(size, TxSize::Tx16x8);
        for row in 0..2 {
            for column in 0..4 {
                assert_eq!(grid.get(row, column).unwrap().tx_size, Some(size));
            }
        }
    }

    #[test]
    fn variable_tx_size_returns_last_visible_cell_for_cropped_edge_block() {
        let mut grid = MiGrid::new(88, 72).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 64], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let block = BlockRect::new(64, 0, BlockSize::Block128x128);
        let size = read_block_tx_size(
            &mut decoder,
            &mut cdfs,
            &mut grid,
            BlockTxSizeConfig {
                block,
                size: BlockSize::Block128x128,
                tile: TileBounds {
                    column_start: 0,
                    column_end: 88,
                    row_start: 0,
                    row_end: 72,
                },
                lossless: false,
                skip: false,
                is_inter: true,
                tx_mode_select: true,
            },
        )
        .unwrap();
        assert_eq!(grid.get(31, 87).unwrap().tx_size, Some(size));
    }

    #[test]
    fn skipped_intra_block_does_not_read_transform_depth() {
        let mut grid = MiGrid::new(8, 8).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let before = cdfs.clone();
        let block = BlockRect::new(0, 0, BlockSize::Block16x16);
        assert_eq!(
            read_block_tx_size(
                &mut decoder,
                &mut cdfs,
                &mut grid,
                BlockTxSizeConfig {
                    block,
                    size: BlockSize::Block16x16,
                    tile: bounds(),
                    lossless: false,
                    skip: true,
                    is_inter: false,
                    tx_mode_select: true,
                },
            ),
            Ok(TxSize::Tx16x16)
        );
        assert_eq!(cdfs, before);
    }

    #[test]
    fn compound_inter_mode_context_and_symbols_cover_normative_tables() {
        assert_eq!(compound_inter_mode_context(0, 0), Ok(0));
        assert_eq!(compound_inter_mode_context(2, 3), Ok(3));
        assert_eq!(compound_inter_mode_context(5, 5), Ok(7));
        assert_eq!(compound_inter_mode_context(0, 6), Err(Error::InvalidObu));
        assert_eq!(
            compound_inter_modes(4),
            Ok([InterMode::Near, InterMode::New])
        );
        assert_eq!(
            compound_inter_modes(6),
            Ok([InterMode::Global, InterMode::Global])
        );
        assert_eq!(compound_inter_modes(8), Err(Error::InvalidObu));
        assert_eq!(
            inter_modes_to_y_mode([InterMode::New, InterMode::Near], true),
            Ok(23)
        );
        assert_eq!(
            inter_modes_to_y_mode([InterMode::Nearest, InterMode::Global], true),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn compound_neighbor_contexts_and_skip_post_motion_are_normative() {
        let mut grid = MiGrid::new(8, 8).unwrap();
        grid.fill(
            BlockRect::new(2, 0, BlockSize::Block8x8),
            crate::block_state::BlockState {
                size: Some(BlockSize::Block8x8),
                is_inter: true,
                reference_frames: [1, 5],
                compound_group_index: 1,
                compound_index: 0,
                ..crate::block_state::BlockState::default()
            },
        )
        .unwrap();
        grid.fill(
            BlockRect::new(0, 2, BlockSize::Block8x8),
            crate::block_state::BlockState {
                size: Some(BlockSize::Block8x8),
                is_inter: true,
                reference_frames: [7, -1],
                ..crate::block_state::BlockState::default()
            },
        )
        .unwrap();
        let block = BlockRect::new(2, 2, BlockSize::Block8x8);
        let tile = TileBounds {
            column_start: 0,
            column_end: 8,
            row_start: 0,
            row_end: 8,
        };
        assert_eq!(compound_contexts(&grid, block, tile, true), (4, 4));

        let mut decoder = SymbolDecoder::new(&[0x80, 0], true).unwrap();
        let mut cdfs = TileCdfs::default();
        let result = read_inter_post_motion(
            &mut decoder,
            &mut cdfs,
            &grid,
            InterPostMotionConfig {
                block,
                size: BlockSize::Block8x8,
                tile,
                skip_mode: true,
                references: [1, 5],
                motion: InterMotionResult {
                    modes: [InterMode::Nearest; 2],
                    y_mode: 18,
                    reference_mv_index: 0,
                    motion_vectors: [MotionVector::default(); 2],
                },
                global_types: [GlobalMotionType::Identity; 2],
                enable_inter_intra: true,
                motion_mode_switchable: true,
                force_integer_mv: false,
                allow_warped_motion: true,
                reference_scaled: false,
                enable_masked_compound: true,
                enable_joint_compound: true,
                equal_reference_distance: true,
                frame_filter: 2,
                dual_filter: true,
            },
        )
        .unwrap();
        assert_eq!(result.motion_mode, MotionMode::Simple);
        assert_eq!(result.compound.kind, CompoundType::Average);
        assert_eq!(result.interpolation_filters, [2; 2]);
    }

    #[test]
    fn forced_global_inter_motion_assigns_global_without_entropy_symbols() {
        let grid = MiGrid::new(8, 8).unwrap();
        let global = MotionVector {
            row: 17,
            column: -9,
        };
        let complete = crate::motion::build_complete_motion_stack(
            &grid,
            crate::motion::CompleteMotionStackConfig {
                spatial: crate::motion::SpatialScan {
                    block: BlockRect::new(0, 0, BlockSize::Block8x8),
                    tile: TileBounds {
                        column_start: 0,
                        column_end: 8,
                        row_start: 0,
                        row_end: 8,
                    },
                    references: [1, -1],
                    compound: false,
                    global_types: [GlobalMotionType::Identity; 2],
                    global_vectors: [MotionVector::default(); 2],
                },
                temporal_field: None,
                temporal: None,
                global_vectors: [global, MotionVector::default()],
                compound_candidates: [[MotionVector::default(); 2]; 2],
            },
        )
        .unwrap();
        let mut decoder = SymbolDecoder::new(&[0x80, 0], true).unwrap();
        let mut cdfs = TileCdfs::default();
        assert_eq!(
            read_inter_motion(
                &mut decoder,
                &mut cdfs,
                InterMotionConfig {
                    skip_mode: false,
                    forced_global: true,
                    compound: false,
                    stack: &complete,
                    global_vectors: [global, MotionVector::default()],
                    syntax: MotionVectorSyntax {
                        force_integer: false,
                        allow_high_precision: true,
                        intrabc: false,
                    },
                },
            ),
            Ok(InterMotionResult {
                modes: [InterMode::Global, InterMode::Global],
                y_mode: 16,
                reference_mv_index: 0,
                motion_vectors: [global, MotionVector::default()],
            })
        );
    }

    fn bounds() -> TileBounds {
        TileBounds {
            column_start: 0,
            column_end: 4,
            row_start: 0,
            row_end: 4,
        }
    }

    #[test]
    fn segment_features_force_skip_mode_and_inter_without_symbols() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let grid = MiGrid::new(4, 4).unwrap();
        let block = BlockRect::new(0, 0, BlockSize::Block16x16);
        let segment = ActiveSegmentFeatures {
            global_mv: true,
            ..ActiveSegmentFeatures::default()
        };
        assert_eq!(
            read_skip_mode(
                &mut decoder,
                &mut cdfs,
                &grid,
                SkipModeConfig {
                    block,
                    size: BlockSize::Block16x16,
                    tile: bounds(),
                    skip_mode_present: true,
                    segment,
                },
            ),
            Ok(false)
        );
        assert_eq!(
            read_is_inter(
                &mut decoder,
                &mut cdfs,
                &grid,
                block,
                bounds(),
                false,
                segment,
            ),
            Ok(true)
        );
    }

    #[test]
    fn segmentation_features_map_to_block_mode_constraints() {
        let mut segmentation = Segmentation {
            enabled: true,
            ..Segmentation::default()
        };
        segmentation.feature_enabled[3][5] = true;
        segmentation.feature_data[3][5] = 4;
        segmentation.feature_enabled[3][6] = true;
        segmentation.feature_enabled[3][7] = true;
        assert_eq!(
            active_segment_features(&segmentation, 3),
            Ok(ActiveSegmentFeatures {
                skip: true,
                reference_frame: Some(4),
                global_mv: true,
            })
        );
        assert_eq!(
            active_segment_features(&segmentation, 8),
            Err(Error::InvalidObu)
        );
        segmentation.feature_data[3][5] = 8;
        assert_eq!(
            active_segment_features(&segmentation, 3),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn inter_segment_id_reuses_previous_map_when_updates_are_disabled() {
        let mut previous = MiGrid::new(4, 4).unwrap();
        previous
            .fill(
                BlockRect::new(0, 0, BlockSize::Block8x8),
                crate::block_state::BlockState {
                    segment_id: 4,
                    ..crate::block_state::BlockState::default()
                },
            )
            .unwrap();
        let grid = MiGrid::new(4, 4).unwrap();
        let mut contexts = SegmentPredictionContexts::new(4, 4).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let segmentation = Segmentation {
            enabled: true,
            update_map: false,
            ..Segmentation::default()
        };
        assert_eq!(
            read_inter_segment_id(
                &mut decoder,
                &mut cdfs,
                &grid,
                &mut contexts,
                InterSegmentConfig {
                    block: BlockRect::new(0, 0, BlockSize::Block8x8),
                    tile: bounds(),
                    segmentation: &segmentation,
                    previous_segments: Some(&previous),
                    pre_skip: true,
                    skip: false,
                },
            ),
            Ok(4)
        );
    }

    #[test]
    fn common_block_prefix_honors_forced_segment_skip_and_inter() {
        let mut previous = MiGrid::new(4, 4).unwrap();
        previous
            .fill(
                BlockRect::new(0, 0, BlockSize::Block8x8),
                crate::block_state::BlockState {
                    segment_id: 3,
                    ..crate::block_state::BlockState::default()
                },
            )
            .unwrap();
        let grid = MiGrid::new(4, 4).unwrap();
        let mut contexts = SegmentPredictionContexts::new(4, 4).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let mut segmentation = Segmentation {
            enabled: true,
            update_map: false,
            ..Segmentation::default()
        };
        segmentation.feature_enabled[3][6] = true;
        segmentation.feature_enabled[3][7] = true;
        assert_eq!(
            read_block_mode_prefix(
                &mut decoder,
                &mut cdfs,
                &grid,
                &mut contexts,
                BlockModePrefixConfig {
                    block: BlockRect::new(0, 0, BlockSize::Block8x8),
                    size: BlockSize::Block8x8,
                    tile: bounds(),
                    segmentation: &segmentation,
                    previous_segments: Some(&previous),
                    skip_mode_present: true,
                    frame_is_intra: false,
                },
            ),
            Ok(BlockModePrefix {
                segment_id: 3,
                segment: ActiveSegmentFeatures {
                    skip: true,
                    reference_frame: None,
                    global_mv: true,
                },
                skip_mode: false,
                skip: true,
                is_inter: true,
            })
        );
    }

    #[test]
    fn variable_tx_tree_forced_leaf_records_its_complete_footprint() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let mut grid = MiGrid::new(8, 8).unwrap();
        let block = BlockRect::new(0, 0, BlockSize::Block32x32);
        read_var_tx_size(
            &mut decoder,
            &mut cdfs,
            &mut grid,
            VarTxSizeConfig {
                block,
                tile: bounds(),
                size: BlockSize::Block32x32,
            },
            VarTxNode {
                row: 0,
                column: 0,
                size: TxSize::Tx8x8,
                depth: 2,
            },
        )
        .unwrap();
        assert_eq!(grid.get(0, 0).unwrap().tx_size, Some(TxSize::Tx8x8));
        assert_eq!(grid.get(1, 1).unwrap().tx_size, Some(TxSize::Tx8x8));
        assert_eq!(grid.get(2, 2).unwrap().tx_size, None);
    }

    #[test]
    fn forced_motion_modes_do_not_consume_entropy_symbols() {
        let mut decoder = SymbolDecoder::new(&[0x80, 0], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let grid = MiGrid::new(4, 4).unwrap();
        assert_eq!(
            read_motion_mode(
                &mut decoder,
                &mut cdfs,
                &grid,
                MotionModeConfig {
                    block: BlockRect::new(0, 0, BlockSize::Block8x8),
                    size: BlockSize::Block8x8,
                    tile: bounds(),
                    skip_mode: true,
                    switchable: true,
                    force_integer_mv: false,
                    global_mode: false,
                    global_type: GlobalMotionType::Identity,
                    compound: false,
                    interintra: false,
                    reference_frame: 1,
                    motion_vector: MotionVector::default(),
                    allow_warped_motion: true,
                    reference_scaled: false,
                },
            ),
            Ok(MotionMode::Simple)
        );
    }

    #[test]
    fn forced_reference_paths_do_not_consume_entropy_symbols() {
        let mut decoder = SymbolDecoder::new(&[0x80, 0], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let base = ReferenceFrameConfig {
            size: BlockSize::Block16x16,
            skip_mode: true,
            skip_mode_frames: [2, 6],
            segment: ActiveSegmentFeatures::default(),
            reference_select: true,
            contexts: ReferenceContexts::default(),
        };
        assert_eq!(
            read_reference_frames(&mut decoder, &mut cdfs, base),
            Ok([2, 6])
        );
        assert_eq!(
            read_reference_frames(
                &mut decoder,
                &mut cdfs,
                ReferenceFrameConfig {
                    skip_mode: false,
                    segment: ActiveSegmentFeatures {
                        global_mv: true,
                        ..ActiveSegmentFeatures::default()
                    },
                    ..base
                },
            ),
            Ok([1, -1])
        );
    }

    #[test]
    fn reference_contexts_derive_from_above_and_left_references() {
        let mut grid = MiGrid::new(4, 4).unwrap();
        grid.fill(
            BlockRect::new(2, 0, BlockSize::Block8x8),
            crate::block_state::BlockState {
                size: Some(BlockSize::Block8x8),
                is_inter: true,
                reference_frames: [1, -1],
                ..crate::block_state::BlockState::default()
            },
        )
        .unwrap();
        grid.fill(
            BlockRect::new(0, 2, BlockSize::Block8x8),
            crate::block_state::BlockState {
                size: Some(BlockSize::Block8x8),
                is_inter: true,
                reference_frames: [5, -1],
                ..crate::block_state::BlockState::default()
            },
        )
        .unwrap();
        let contexts =
            reference_contexts(&grid, BlockRect::new(2, 2, BlockSize::Block8x8), bounds());
        assert_eq!(contexts.compound_mode, 1);
        assert_eq!(contexts.compound_type, 1);
        assert_eq!(contexts.single[0], 1);
        assert_eq!(contexts.forward[1], 2);
        assert_eq!(contexts.backward[1], 2);
    }

    #[test]
    fn motion_assignment_uses_global_and_stack_predictors_without_symbols() {
        let mut decoder = SymbolDecoder::new(&[0x80, 0], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let mut stack = MotionStack::new().unwrap();
        stack
            .add(
                [
                    MotionVector { row: 8, column: -4 },
                    MotionVector { row: 3, column: 9 },
                ],
                2,
            )
            .unwrap();
        assert_eq!(
            assign_motion_vectors(
                &mut decoder,
                &mut cdfs,
                &stack,
                AssignMotionConfig {
                    modes: [InterMode::Nearest, InterMode::Global],
                    compound: true,
                    reference_mv_index: 0,
                    global_vectors: [
                        MotionVector::default(),
                        MotionVector {
                            row: -16,
                            column: 24
                        },
                    ],
                    syntax: MotionVectorSyntax {
                        force_integer: false,
                        allow_high_precision: true,
                        intrabc: false,
                    },
                },
            ),
            Ok([
                MotionVector { row: 8, column: -4 },
                MotionVector {
                    row: -16,
                    column: 24
                },
            ])
        );
    }

    #[test]
    fn reference_mv_index_defaults_match_new_and_near_modes() {
        let mut decoder = SymbolDecoder::new(&[0x80, 0], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let contexts = MotionContexts {
            drl: [0; 8],
            new_mv: 0,
            reference_mv: 0,
        };
        assert_eq!(
            read_reference_mv_index(&mut decoder, &mut cdfs, &[InterMode::New], &contexts, 1,),
            Ok(0)
        );
        assert_eq!(
            read_reference_mv_index(&mut decoder, &mut cdfs, &[InterMode::Near], &contexts, 1,),
            Ok(1)
        );
    }

    #[test]
    fn inter_transform_tree_visits_normative_quadrant_order() {
        let mut grid = MiGrid::new(8, 8).unwrap();
        for row in [0, 4] {
            for column in [0, 4] {
                grid.fill_tx_size(row, column, TxSize::Tx16x16).unwrap();
            }
        }
        let mut visits: mrml_runtime::Vector<(u32, u32, TxSize)> =
            mrml_runtime::Vector::with_capacity(4).unwrap();
        walk_inter_transform_tree(
            &grid,
            TransformTreeNode {
                start_x: 0,
                start_y: 0,
                width: 32,
                height: 32,
                frame_width: 32,
                frame_height: 32,
            },
            &mut |x, y, size| {
                visits
                    .try_push((x, y, size))
                    .map_err(|_| Error::LimitExceeded)
            },
        )
        .unwrap();
        assert_eq!(
            &visits[..],
            &[
                (0, 0, TxSize::Tx16x16),
                (16, 0, TxSize::Tx16x16),
                (0, 16, TxSize::Tx16x16),
                (16, 16, TxSize::Tx16x16),
            ]
        );
    }

    #[test]
    fn skipped_segment_id_uses_spatial_prediction() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let grid = MiGrid::new(4, 4).unwrap();
        assert_eq!(
            read_spatial_segment_id(
                &mut decoder,
                &mut cdfs,
                &grid,
                BlockRect::new(0, 0, BlockSize::Block8x8),
                bounds(),
                7,
                true,
            ),
            Ok(0)
        );
    }

    #[test]
    fn forced_transform_sizes_consume_no_symbols() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let grid = MiGrid::new(4, 4).unwrap();
        let base = TxSizeConfig {
            block: BlockRect::new(0, 0, BlockSize::Block16x32),
            size: BlockSize::Block16x32,
            tile: bounds(),
            lossless: false,
            allow_select: false,
            tx_mode_select: true,
        };
        assert_eq!(
            read_tx_size(&mut decoder, &mut cdfs, &grid, base),
            Ok(TxSize::Tx16x32)
        );
        assert_eq!(
            read_tx_size(
                &mut decoder,
                &mut cdfs,
                &grid,
                TxSizeConfig {
                    lossless: true,
                    ..base
                },
            ),
            Ok(TxSize::Tx4x4)
        );
    }

    #[test]
    fn intra_mode_classification_matches_directional_range() {
        assert!(!IntraMode::Dc.is_directional());
        for mode in [IntraMode::Vertical, IntraMode::D45, IntraMode::D67] {
            assert!(mode.is_directional());
        }
        assert!(!IntraMode::Smooth.is_directional());
        assert!(!IntraMode::Paeth.is_directional());
        assert_eq!(BlockSize::Block4x16.size_group(), 0);
        assert_eq!(BlockSize::Block64x16.size_group(), 2);
    }

    #[test]
    fn small_delta_magnitudes_require_no_literal_extension() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        assert_eq!(read_delta_absolute(&mut decoder, 0), Ok(0));
        assert_eq!(read_delta_absolute(&mut decoder, 2), Ok(2));
        assert_eq!(read_delta_absolute(&mut decoder, 4), Err(Error::InvalidObu));
    }

    #[test]
    fn cfl_joint_sign_mapping_excludes_the_all_zero_pair() {
        let mut seen = [[false; 3]; 3];
        for symbol in 0..8 {
            let (u, v) = cfl_signs(symbol).unwrap();
            assert_ne!((u, v), (0, 0));
            seen[usize::from(u)][usize::from(v)] = true;
        }
        assert_eq!(seen.iter().flatten().filter(|value| **value).count(), 8);
    }

    #[test]
    fn chroma_modes_and_cfl_eligibility_follow_normative_rules() {
        for mode in [ChromaMode::Vertical, ChromaMode::D45, ChromaMode::D67] {
            assert!(mode.is_directional());
        }
        assert!(!ChromaMode::Cfl.is_directional());
        assert_eq!(ChromaMode::from_symbol(13), Ok(ChromaMode::Cfl));
        assert_eq!(ChromaMode::from_symbol(14), Err(Error::InvalidObu));

        assert!(cfl_allowed(
            false,
            BlockSize::Block32x32,
            BlockSize::Block8x8
        ));
        assert!(!cfl_allowed(
            false,
            BlockSize::Block32x64,
            BlockSize::Block8x8
        ));
        assert!(cfl_allowed(
            true,
            BlockSize::Block64x64,
            BlockSize::Block4x4
        ));
        assert!(!cfl_allowed(true, BlockSize::Block8x8, BlockSize::Block8x8));
    }

    #[test]
    fn palette_block_context_covers_normative_eight_to_sixty_four_range() {
        assert_eq!(palette_block_size_context(BlockSize::Block8x8), Ok(0));
        assert_eq!(palette_block_size_context(BlockSize::Block8x16), Ok(1));
        assert_eq!(palette_block_size_context(BlockSize::Block32x32), Ok(4));
        assert_eq!(palette_block_size_context(BlockSize::Block64x64), Ok(6));
        assert_eq!(
            palette_block_size_context(BlockSize::Block4x4),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            palette_block_size_context(BlockSize::Block128x128),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn intra_block_orchestrator_handles_monochrome_without_chroma_symbols() {
        let mut decoder = SymbolDecoder::new(&[0; 32], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let grid = MiGrid::new(4, 4).unwrap();
        let mode = read_intra_block_mode(
            &mut decoder,
            &mut cdfs,
            &grid,
            IntraBlockModeConfig {
                block: BlockRect::new(0, 0, BlockSize::Block16x16),
                size: BlockSize::Block16x16,
                tile: bounds(),
                plane_residual_size: BlockSize::Block16x16,
                lossless: false,
                has_chroma: false,
                palette_enabled: false,
                filter_intra_enabled: false,
                frame_is_intra: false,
                allow_intrabc: false,
            },
        )
        .unwrap();
        assert_eq!(mode.chroma.mode, ChromaMode::Dc);
        assert_eq!(mode.chroma.cfl_alphas, [0; 2]);
        assert_eq!(mode.palette_sizes, [0; 2]);
        assert_eq!(mode.filter_intra_mode, None);
    }

    #[test]
    fn filter_intra_is_forced_off_outside_eligible_modes() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        assert_eq!(
            read_filter_intra(
                &mut decoder,
                &mut cdfs,
                BlockSize::Block64x64,
                true,
                IntraMode::Dc,
                0,
            ),
            Ok(None)
        );
        assert_eq!(
            read_filter_intra(
                &mut decoder,
                &mut cdfs,
                BlockSize::Block16x16,
                true,
                IntraMode::Paeth,
                0,
            ),
            Ok(None)
        );
    }

    #[test]
    fn filter_intra_remaps_transform_type_direction() {
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, None), Ok(0));
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, Some(0)), Ok(0));
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, Some(1)), Ok(1));
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, Some(2)), Ok(2));
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, Some(3)), Ok(6));
        assert_eq!(intra_tx_type_direction(IntraMode::Dc, Some(4)), Ok(0));
        assert_eq!(
            intra_tx_type_direction(IntraMode::Dc, Some(5)),
            Err(Error::InvalidObu)
        );
    }
}
