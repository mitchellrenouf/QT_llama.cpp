//! Uncompressed frame-header state and prefix parsing (section 5.9.2).

use crate::film_grain::{self, FilmGrain};
use crate::motion::{self, GlobalMotion};
use crate::params::{
    self, Cdef, DeltaParams, LoopFilter, Quantization, Restoration, Segmentation, TxMode,
};
use crate::tile::TileLayout;
use crate::{Bits, Error, Sequence};
use mrml_runtime::Vector;

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const PRIMARY_REF_NONE: u8 = 7;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FrameType {
    #[default]
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    pub fn is_intra(self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReferenceInfo {
    pub valid: bool,
    pub frame_id: u32,
    pub order_hint: u32,
    pub frame_type: FrameType,
    pub showable_frame: bool,
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub segmentation: Segmentation,
    pub loop_filter: LoopFilter,
    pub global_motion: [GlobalMotion; REFS_PER_FRAME],
    pub film_grain: FilmGrain,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub allow_high_precision_mv: bool,
    /// AV1 interpolation filter enum; 4 is `SWITCHABLE`.
    pub interpolation_filter: u8,
    pub motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub frame_size_override: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub current_frame_id: Option<u32>,
    /// Reference slots invalidated by frame-id or error-resilient checks.
    pub invalidated_reference_slots: u8,
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    pub delta_frame_id: [u32; REFS_PER_FRAME],
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    /// Normative `SuperresDenom`; 8 when super-resolution is disabled.
    pub superres_denom: u8,
    pub allow_intrabc: bool,
    pub disable_frame_end_update_cdf: bool,
    pub tile_layout: Option<TileLayout>,
    pub quantization: Quantization,
    pub segmentation: Segmentation,
    pub delta_params: DeltaParams,
    pub lossless_segments: [bool; params::MAX_SEGMENTS],
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub restoration: Restoration,
    pub tx_mode: TxMode,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    /// Indices in `ref_frame_idx`, not reference-slot numbers.
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub global_motion: [GlobalMotion; REFS_PER_FRAME],
    pub film_grain: FilmGrain,
    pub buffer_removal_times: Vector<Option<u32>>,
    pub frame_presentation_time: Option<u32>,
    /// Exact bit position of the next uncompressed-header syntax element.
    pub bits_consumed: usize,
}

pub fn parse(
    data: &[u8],
    sequence: &Sequence,
    references: &[ReferenceInfo; NUM_REF_FRAMES],
    previous_frame_id: Option<u32>,
    temporal_id: u8,
    spatial_id: u8,
) -> Result<FrameHeader, Error> {
    let mut b = Bits::new(data);
    // Error-resilient headers may invalidate slots for this frame. Work on a
    // copy so a truncated header cannot partially mutate decoder state.
    let mut references = *references;
    let mut header = FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: 0,
        frame_type: FrameType::Key,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: true,
        disable_cdf_update: false,
        allow_screen_content_tools: false,
        force_integer_mv: true,
        allow_high_precision_mv: false,
        interpolation_filter: 0,
        motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        frame_size_override: false,
        order_hint: 0,
        primary_ref_frame: PRIMARY_REF_NONE,
        refresh_frame_flags: 0xff,
        current_frame_id: None,
        invalidated_reference_slots: 0,
        ref_frame_idx: [0; REFS_PER_FRAME],
        delta_frame_id: [0; REFS_PER_FRAME],
        frame_width: sequence.max_width,
        frame_height: sequence.max_height,
        upscaled_width: sequence.max_width,
        render_width: sequence.max_width,
        render_height: sequence.max_height,
        use_superres: false,
        superres_denom: 8,
        allow_intrabc: false,
        disable_frame_end_update_cdf: false,
        tile_layout: None,
        quantization: Quantization::default(),
        segmentation: Segmentation::default(),
        delta_params: DeltaParams::default(),
        lossless_segments: [false; params::MAX_SEGMENTS],
        coded_lossless: false,
        all_lossless: false,
        loop_filter: LoopFilter::default(),
        cdef: Cdef::default(),
        restoration: Restoration::default(),
        tx_mode: TxMode::default(),
        reference_select: false,
        skip_mode_present: false,
        skip_mode_frame: [0; 2],
        allow_warped_motion: false,
        reduced_tx_set: false,
        global_motion: [GlobalMotion::default(); REFS_PER_FRAME],
        film_grain: FilmGrain::default(),
        buffer_removal_times: Vector::new(),
        frame_presentation_time: None,
        bits_consumed: 0,
    };

    if !sequence.reduced_still_picture_header {
        header.show_existing_frame = b.bit()?;
        if header.show_existing_frame {
            header.frame_to_show_map_idx = b.read(3)? as u8;
            let reference = references[header.frame_to_show_map_idx as usize];
            if !reference.valid {
                return Err(Error::InvalidObu);
            }
            if !reference.showable_frame {
                return Err(Error::InvalidObu);
            }
            if needs_presentation_time(sequence) {
                let width = sequence
                    .decoder_model
                    .as_ref()
                    .unwrap()
                    .frame_presentation_time_length;
                header.frame_presentation_time = Some(b.read(width)? as u32);
            }
            if sequence.frame_id_numbers_present {
                let display_id = b.read(sequence.frame_id_length)? as u32;
                if display_id != reference.frame_id {
                    return Err(Error::InvalidObu);
                }
            }
            header.frame_width = reference.frame_width;
            header.frame_height = reference.frame_height;
            header.upscaled_width = reference.upscaled_width;
            header.render_width = reference.render_width;
            header.render_height = reference.render_height;
            header.film_grain = reference.film_grain;
            header.frame_type = reference.frame_type;
            header.refresh_frame_flags = if reference.frame_type == FrameType::Key {
                0xff
            } else {
                0
            };
            header.bits_consumed = b.position();
            return Ok(header);
        }
        header.frame_type = match b.read(2)? {
            0 => FrameType::Key,
            1 => FrameType::Inter,
            2 => FrameType::IntraOnly,
            3 => FrameType::Switch,
            _ => unreachable!(),
        };
        header.show_frame = b.bit()?;
        if header.show_frame && needs_presentation_time(sequence) {
            let width = sequence
                .decoder_model
                .as_ref()
                .unwrap()
                .frame_presentation_time_length;
            header.frame_presentation_time = Some(b.read(width)? as u32);
        }
        header.showable_frame = if header.show_frame {
            header.frame_type != FrameType::Key
        } else {
            b.bit()?
        };
        header.error_resilient_mode = if header.frame_type == FrameType::Switch
            || (header.frame_type == FrameType::Key && header.show_frame)
        {
            true
        } else {
            b.bit()?
        };
    }
    if header.frame_type == FrameType::Key && header.show_frame {
        for reference in &mut references {
            reference.valid = false;
            reference.order_hint = 0;
        }
        header.invalidated_reference_slots = 0xff;
    }
    header.disable_cdf_update = b.bit()?;
    header.allow_screen_content_tools =
        select_feature(&mut b, sequence.seq_force_screen_content_tools)?;
    header.force_integer_mv = if header.allow_screen_content_tools {
        select_feature(&mut b, sequence.seq_force_integer_mv)?
    } else {
        false
    };
    if header.frame_type.is_intra() {
        header.force_integer_mv = true;
    }
    if sequence.frame_id_numbers_present {
        let current = b.read(sequence.frame_id_length)? as u32;
        if (header.frame_type != FrameType::Key || !header.show_frame)
            && let Some(previous) = previous_frame_id
        {
            let modulus = 1u32 << sequence.frame_id_length;
            let difference = current.wrapping_add(modulus).wrapping_sub(previous) % modulus;
            if difference == 0 || difference >= modulus / 2 {
                return Err(Error::InvalidObu);
            }
        }
        header.current_frame_id = Some(current);
        mark_reference_frames(
            current,
            sequence.frame_id_length,
            sequence.delta_frame_id_length,
            &mut references,
            &mut header.invalidated_reference_slots,
        );
    }
    header.frame_size_override = if header.frame_type == FrameType::Switch {
        true
    } else if sequence.reduced_still_picture_header {
        false
    } else {
        b.bit()?
    };
    if sequence.enable_order_hint {
        header.order_hint = b.read(sequence.order_hint_bits)? as u32;
    }
    header.primary_ref_frame = if header.frame_type.is_intra() || header.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        b.read(3)? as u8
    };

    if let Some(decoder_model) = &sequence.decoder_model {
        let present = b.bit()?;
        for op in &sequence.operating_points {
            let applies = op.idc == 0
                || ((op.idc >> temporal_id) & 1 != 0 && (op.idc >> (spatial_id + 8)) & 1 != 0);
            let value = if present && applies && op.decoder_buffer_delay.is_some() {
                Some(b.read(decoder_model.buffer_removal_time_length)? as u32)
            } else {
                None
            };
            header
                .buffer_removal_times
                .try_push(value)
                .map_err(|_| Error::LimitExceeded)?;
        }
    }
    header.refresh_frame_flags = if header.frame_type == FrameType::Switch
        || (header.frame_type == FrameType::Key && header.show_frame)
    {
        0xff
    } else {
        b.read(8)? as u8
    };
    if (!header.frame_type.is_intra() || header.refresh_frame_flags != 0xff)
        && header.error_resilient_mode
        && sequence.enable_order_hint
    {
        for (index, reference) in references.iter_mut().enumerate() {
            let signaled = b.read(sequence.order_hint_bits)? as u32;
            if signaled != reference.order_hint {
                reference.valid = false;
                header.invalidated_reference_slots |= 1 << index;
            }
        }
    }

    if header.frame_type.is_intra() {
        read_frame_size(&mut b, sequence, header.frame_size_override, &mut header)?;
        read_render_size(&mut b, &mut header)?;
        if header.allow_screen_content_tools && header.upscaled_width == header.frame_width {
            header.allow_intrabc = b.bit()?;
        }
    } else {
        let short_signaling = sequence.enable_order_hint && b.bit()?;
        if short_signaling {
            let last_frame_idx = b.read(3)? as u8;
            let gold_frame_idx = b.read(3)? as u8;
            header.ref_frame_idx = set_frame_refs(
                last_frame_idx,
                gold_frame_idx,
                header.order_hint,
                sequence.order_hint_bits,
                &references,
            )?;
        }
        for index in 0..REFS_PER_FRAME {
            if !short_signaling {
                header.ref_frame_idx[index] = b.read(3)? as u8;
            }
            let reference = references[header.ref_frame_idx[index] as usize];
            if !reference.valid {
                return Err(Error::InvalidObu);
            }
            if sequence.frame_id_numbers_present {
                let delta = b.read(sequence.delta_frame_id_length)? as u32 + 1;
                header.delta_frame_id[index] = delta;
                let current = header.current_frame_id.ok_or(Error::InvalidObu)?;
                let mask = (1u32 << sequence.frame_id_length) - 1;
                if reference.frame_id != current.wrapping_sub(delta) & mask {
                    return Err(Error::InvalidObu);
                }
            }
        }
        let mut from_reference = false;
        if header.frame_size_override && !header.error_resilient_mode {
            for index in 0..REFS_PER_FRAME {
                if b.bit()? {
                    copy_size(
                        references[header.ref_frame_idx[index] as usize],
                        &mut header,
                    );
                    read_superres(&mut b, sequence, &mut header)?;
                    from_reference = true;
                    break;
                }
            }
        }
        if !from_reference {
            read_frame_size(&mut b, sequence, header.frame_size_override, &mut header)?;
            read_render_size(&mut b, &mut header)?;
        }
        if !header.force_integer_mv {
            header.allow_high_precision_mv = b.bit()?;
        }
        header.interpolation_filter = if b.bit()? { 4 } else { b.read(2)? as u8 };
        header.motion_mode_switchable = b.bit()?;
        if sequence.enable_ref_frame_mvs && !header.error_resilient_mode {
            header.use_ref_frame_mvs = b.bit()?;
        }
    }

    header.disable_frame_end_update_cdf =
        if sequence.reduced_still_picture_header || header.disable_cdf_update {
            true
        } else {
            b.bit()?
        };
    let mi_cols = 2 * ((header.frame_width + 7) >> 3);
    let mi_rows = 2 * ((header.frame_height + 7) >> 3);
    header.tile_layout = Some(TileLayout::parse(
        &mut b,
        mi_cols,
        mi_rows,
        sequence.use_128x128_superblock,
    )?);
    header.quantization = Quantization::parse(&mut b, sequence)?;
    let previous = previous_parameters(&header, &references);
    header.segmentation = Segmentation::parse(
        &mut b,
        header.primary_ref_frame == PRIMARY_REF_NONE,
        previous.map(|reference| reference.segmentation),
    )?;
    header.delta_params =
        DeltaParams::parse(&mut b, header.quantization.base_q_idx, header.allow_intrabc)?;
    let (segments, coded_lossless, all_lossless) = params::derive_lossless(
        &header.quantization,
        &header.segmentation,
        header.frame_width,
        header.upscaled_width,
    )?;
    header.lossless_segments = segments;
    header.coded_lossless = coded_lossless;
    header.all_lossless = all_lossless;
    header.loop_filter = LoopFilter::parse(
        &mut b,
        sequence,
        header.coded_lossless,
        header.allow_intrabc,
        previous.map(|reference| reference.loop_filter),
    )?;
    header.cdef = Cdef::parse(
        &mut b,
        sequence,
        header.coded_lossless,
        header.allow_intrabc,
    )?;
    header.restoration =
        Restoration::parse(&mut b, sequence, header.all_lossless, header.allow_intrabc)?;
    header.tx_mode = TxMode::parse(&mut b, header.coded_lossless)?;
    header.reference_select = !header.frame_type.is_intra() && b.bit()?;
    if let Some(skip_frames) = derive_skip_mode(&header, sequence, &references) {
        header.skip_mode_frame = skip_frames;
        header.skip_mode_present = b.bit()?;
    }
    header.allow_warped_motion = !header.frame_type.is_intra()
        && !header.error_resilient_mode
        && sequence.enable_warped_motion
        && b.bit()?;
    header.reduced_tx_set = b.bit()?;
    let previous_global = previous
        .map(|reference| reference.global_motion)
        .unwrap_or([GlobalMotion::default(); REFS_PER_FRAME]);
    header.global_motion =
        motion::parse_global_motion(&mut b, header.frame_type.is_intra(), &previous_global)?;
    let reference_grain = core::array::from_fn(|index| {
        references[index]
            .valid
            .then_some(references[index].film_grain)
    });
    header.film_grain = film_grain::parse(
        &mut b,
        sequence,
        header.show_frame,
        header.showable_frame,
        header.frame_type == FrameType::Inter,
        &header.ref_frame_idx,
        &reference_grain,
    )?;
    header.bits_consumed = b.position();
    Ok(header)
}

fn mark_reference_frames(
    current_frame_id: u32,
    id_length: u8,
    delta_length: u8,
    references: &mut [ReferenceInfo; NUM_REF_FRAMES],
    invalidated: &mut u8,
) {
    let modulus = 1u32 << id_length;
    let window = 1u32 << delta_length;
    for (index, reference) in references.iter_mut().enumerate() {
        let outside = if current_frame_id > window {
            reference.frame_id > current_frame_id || reference.frame_id < current_frame_id - window
        } else {
            reference.frame_id > current_frame_id
                && reference.frame_id < modulus + current_frame_id - window
        };
        if outside {
            reference.valid = false;
            *invalidated |= 1 << index;
        }
    }
}

fn set_frame_refs(
    last_frame_idx: u8,
    gold_frame_idx: u8,
    order_hint: u32,
    order_hint_bits: u8,
    references: &[ReferenceInfo; NUM_REF_FRAMES],
) -> Result<[u8; REFS_PER_FRAME], Error> {
    const LAST: usize = 0;
    const LAST2: usize = 1;
    const LAST3: usize = 2;
    const GOLDEN: usize = 3;
    const BWDREF: usize = 4;
    const ALTREF2: usize = 5;
    const ALTREF: usize = 6;

    if order_hint_bits == 0
        || usize::from(last_frame_idx) >= NUM_REF_FRAMES
        || usize::from(gold_frame_idx) >= NUM_REF_FRAMES
    {
        return Err(Error::InvalidObu);
    }
    let mut selected = [-1i8; REFS_PER_FRAME];
    selected[LAST] = last_frame_idx as i8;
    selected[GOLDEN] = gold_frame_idx as i8;
    let mut used = [false; NUM_REF_FRAMES];
    used[usize::from(last_frame_idx)] = true;
    used[usize::from(gold_frame_idx)] = true;

    let current = 1i32 << (order_hint_bits - 1);
    let shifted = core::array::from_fn(|index| {
        current + relative_distance(references[index].order_hint, order_hint, order_hint_bits)
    });
    if shifted[usize::from(last_frame_idx)] >= current
        || shifted[usize::from(gold_frame_idx)] >= current
    {
        return Err(Error::InvalidObu);
    }

    if let Some(index) = select_unused(&used, &shifted, current, Selection::LatestBackward) {
        selected[ALTREF] = index as i8;
        used[index] = true;
    }
    if let Some(index) = select_unused(&used, &shifted, current, Selection::EarliestBackward) {
        selected[BWDREF] = index as i8;
        used[index] = true;
    }
    if let Some(index) = select_unused(&used, &shifted, current, Selection::EarliestBackward) {
        selected[ALTREF2] = index as i8;
        used[index] = true;
    }
    for reference_type in [LAST2, LAST3, BWDREF, ALTREF2, ALTREF] {
        if selected[reference_type] < 0
            && let Some(index) = select_unused(&used, &shifted, current, Selection::LatestForward)
        {
            selected[reference_type] = index as i8;
            used[index] = true;
        }
    }
    let earliest = shifted
        .iter()
        .enumerate()
        .min_by_key(|(_, hint)| *hint)
        .map(|(index, _)| index as i8)
        .ok_or(Error::InvalidObu)?;
    Ok(selected.map(|index| {
        if index < 0 {
            earliest as u8
        } else {
            index as u8
        }
    }))
}

#[derive(Clone, Copy)]
enum Selection {
    LatestBackward,
    EarliestBackward,
    LatestForward,
}

fn select_unused(
    used: &[bool; NUM_REF_FRAMES],
    hints: &[i32; NUM_REF_FRAMES],
    current: i32,
    selection: Selection,
) -> Option<usize> {
    let mut result = None;
    for index in 0..NUM_REF_FRAMES {
        if used[index] {
            continue;
        }
        let hint = hints[index];
        let eligible = match selection {
            Selection::LatestBackward | Selection::EarliestBackward => hint >= current,
            Selection::LatestForward => hint < current,
        };
        if !eligible {
            continue;
        }
        let replace = result.is_none_or(|previous| match selection {
            Selection::LatestBackward | Selection::LatestForward => hint >= hints[previous],
            Selection::EarliestBackward => hint < hints[previous],
        });
        if replace {
            result = Some(index);
        }
    }
    result
}

fn derive_skip_mode(
    header: &FrameHeader,
    sequence: &Sequence,
    references: &[ReferenceInfo; NUM_REF_FRAMES],
) -> Option<[u8; 2]> {
    if header.frame_type.is_intra() || !header.reference_select || !sequence.enable_order_hint {
        return None;
    }
    let mut forward: Option<(usize, i32)> = None;
    let mut backward: Option<(usize, i32)> = None;
    for (index, slot) in header.ref_frame_idx.iter().enumerate() {
        let reference = references[usize::from(*slot)];
        if !reference.valid {
            continue;
        }
        let distance = relative_distance(
            reference.order_hint,
            header.order_hint,
            sequence.order_hint_bits,
        );
        if distance < 0 && forward.is_none_or(|(_, best)| distance > best) {
            forward = Some((index, distance));
        } else if distance > 0 && backward.is_none_or(|(_, best)| distance < best) {
            backward = Some((index, distance));
        }
    }
    let (forward_index, forward_distance) = forward?;
    if let Some((backward_index, _)) = backward {
        return Some(order_skip_indices(forward_index, backward_index));
    }
    let mut second: Option<(usize, i32)> = None;
    for (index, slot) in header.ref_frame_idx.iter().enumerate() {
        if index == forward_index {
            continue;
        }
        let reference = references[usize::from(*slot)];
        if !reference.valid {
            continue;
        }
        let distance = relative_distance(
            reference.order_hint,
            header.order_hint,
            sequence.order_hint_bits,
        );
        if distance < forward_distance && second.is_none_or(|(_, best)| distance > best) {
            second = Some((index, distance));
        }
    }
    second.map(|(second_index, _)| order_skip_indices(forward_index, second_index))
}

fn relative_distance(a: u32, b: u32, bits: u8) -> i32 {
    if bits == 0 {
        return 0;
    }
    let modulus = 1u32 << bits;
    let mask = modulus - 1;
    let difference = a.wrapping_sub(b) & mask;
    if difference & (modulus >> 1) != 0 {
        difference as i32 - modulus as i32
    } else {
        difference as i32
    }
}

fn order_skip_indices(first: usize, second: usize) -> [u8; 2] {
    [first.min(second) as u8, first.max(second) as u8]
}

fn previous_parameters<'a>(
    header: &FrameHeader,
    references: &'a [ReferenceInfo; NUM_REF_FRAMES],
) -> Option<&'a ReferenceInfo> {
    if header.primary_ref_frame == PRIMARY_REF_NONE {
        return None;
    }
    let list_index = usize::from(header.primary_ref_frame);
    let slot = *header.ref_frame_idx.get(list_index)? as usize;
    references.get(slot).filter(|reference| reference.valid)
}

fn select_feature(bits: &mut Bits<'_>, selector: u8) -> Result<bool, Error> {
    match selector {
        0 => Ok(false),
        1 => Ok(true),
        2 => bits.bit(),
        _ => Err(Error::InvalidSequence),
    }
}

fn needs_presentation_time(sequence: &Sequence) -> bool {
    sequence.decoder_model.is_some()
        && sequence
            .timing
            .as_ref()
            .is_some_and(|timing| timing.num_ticks_per_picture.is_none())
}

fn read_frame_size(
    bits: &mut Bits<'_>,
    sequence: &Sequence,
    override_size: bool,
    header: &mut FrameHeader,
) -> Result<(), Error> {
    header.frame_width = if override_size {
        bits.read(sequence.frame_width_bits)? as u32 + 1
    } else {
        sequence.max_width
    };
    header.frame_height = if override_size {
        bits.read(sequence.frame_height_bits)? as u32 + 1
    } else {
        sequence.max_height
    };
    header.upscaled_width = header.frame_width;
    read_superres(bits, sequence, header)
}

fn read_superres(
    bits: &mut Bits<'_>,
    sequence: &Sequence,
    header: &mut FrameHeader,
) -> Result<(), Error> {
    header.use_superres = sequence.enable_superres && bits.bit()?;
    if header.use_superres {
        let denominator = bits.read(3)? as u32 + 9;
        header.superres_denom = u8::try_from(denominator).map_err(|_| Error::InvalidObu)?;
        header.frame_width = (header.upscaled_width * 8 + denominator / 2) / denominator;
        header.frame_width = header.frame_width.max(16);
    } else {
        header.superres_denom = 8;
        header.frame_width = header.upscaled_width;
    }
    Ok(())
}

fn read_render_size(bits: &mut Bits<'_>, header: &mut FrameHeader) -> Result<(), Error> {
    if bits.bit()? {
        header.render_width = bits.read(16)? as u32 + 1;
        header.render_height = bits.read(16)? as u32 + 1;
    } else {
        header.render_width = header.upscaled_width;
        header.render_height = header.frame_height;
    }
    Ok(())
}

fn copy_size(reference: ReferenceInfo, header: &mut FrameHeader) {
    header.upscaled_width = reference.upscaled_width;
    header.frame_width = reference.frame_width;
    header.frame_height = reference.frame_height;
    header.render_width = reference.render_width;
    header.render_height = reference.render_height;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_types_classify_intra() {
        assert!(FrameType::Key.is_intra());
        assert!(FrameType::IntraOnly.is_intra());
        assert!(!FrameType::Inter.is_intra());
    }

    #[test]
    fn relative_order_hint_wraps_at_sign_bit() {
        assert_eq!(relative_distance(1, 15, 4), 2);
        assert_eq!(relative_distance(15, 1, 4), -2);
        assert_eq!(relative_distance(7, 1, 4), 6);
    }

    #[test]
    fn short_signaling_assigns_forward_and_backward_references() {
        let mut references = [ReferenceInfo::default(); NUM_REF_FRAMES];
        let hints = [7, 6, 9, 10, 12, 5, 4, 11];
        for (reference, hint) in references.iter_mut().zip(hints) {
            reference.valid = true;
            reference.order_hint = hint;
        }
        assert_eq!(
            set_frame_refs(0, 1, 8, 4, &references),
            Ok([0, 5, 6, 1, 2, 3, 4])
        );
    }

    #[test]
    fn short_signaling_requires_past_last_and_golden() {
        let mut references = [ReferenceInfo::default(); NUM_REF_FRAMES];
        references[0].order_hint = 8;
        references[1].order_hint = 7;
        assert_eq!(
            set_frame_refs(0, 1, 8, 4, &references),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn frame_id_marking_preserves_wrapped_window_only() {
        let mut references = [ReferenceInfo::default(); NUM_REF_FRAMES];
        for (reference, frame_id) in references.iter_mut().zip([15, 0, 1, 3, 4, 8, 14, 2]) {
            reference.valid = true;
            reference.frame_id = frame_id;
        }
        let mut invalidated = 0;
        mark_reference_frames(3, 4, 2, &mut references, &mut invalidated);
        assert_eq!(invalidated, 0b0111_0000);
        assert!(references[0].valid);
        assert!(references[3].valid);
        assert!(!references[4].valid);
        assert!(!references[6].valid);
    }
}
