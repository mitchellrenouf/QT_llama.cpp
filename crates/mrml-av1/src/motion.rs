//! Global-motion syntax and subexponential integer decoding.

use crate::{
    Bits, Error,
    block_state::{BlockState, MiGrid},
    cdf::MotionVectorCdfs,
    entropy::SymbolDecoder,
    partition::{BlockRect, TileBounds},
};
use mrml_runtime::Vector;

const MAX_REF_MV_STACK_SIZE: usize = 8;
const MAX_FRAME_DISTANCE: i32 = 31;
const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];
const INVALID_MOTION_VECTOR: MotionVector = MotionVector {
    row: -1 << 15,
    column: -1 << 15,
};
const INTER_REFERENCE_COUNT: usize = 7;
const MFMV_STACK_SIZE: i8 = 3;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MotionVector {
    /// Vertical displacement in 1/8-luma-sample units.
    pub row: i32,
    /// Horizontal displacement in 1/8-luma-sample units.
    pub column: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MotionField {
    width8: u32,
    height8: u32,
    vectors: Vector<MotionVector>,
}

impl MotionField {
    pub fn new(mi_columns: u32, mi_rows: u32) -> Result<Self, Error> {
        let width8 = mi_columns >> 1;
        let height8 = mi_rows >> 1;
        if width8 == 0 || height8 == 0 {
            return Err(Error::InvalidObu);
        }
        let length = usize::try_from(width8)
            .map_err(|_| Error::LimitExceeded)?
            .checked_mul(usize::try_from(height8).map_err(|_| Error::LimitExceeded)?)
            .and_then(|value| value.checked_mul(INTER_REFERENCE_COUNT))
            .ok_or(Error::LimitExceeded)?;
        let mut vectors = Vector::with_capacity(length).map_err(|_| Error::LimitExceeded)?;
        vectors
            .try_resize(length, INVALID_MOTION_VECTOR)
            .map_err(|_| Error::LimitExceeded)?;
        Ok(Self {
            width8,
            height8,
            vectors,
        })
    }

    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width8, self.height8)
    }

    fn index(&self, reference: i8, x8: u32, y8: u32) -> Result<usize, Error> {
        if !(1..=7).contains(&reference) || x8 >= self.width8 || y8 >= self.height8 {
            return Err(Error::InvalidObu);
        }
        let plane = usize::try_from(reference - 1).map_err(|_| Error::InvalidObu)?;
        let width = usize::try_from(self.width8).map_err(|_| Error::LimitExceeded)?;
        let height = usize::try_from(self.height8).map_err(|_| Error::LimitExceeded)?;
        plane
            .checked_mul(width.checked_mul(height).ok_or(Error::LimitExceeded)?)
            .and_then(|value| value.checked_add(usize::try_from(y8).ok()?.checked_mul(width)?))
            .and_then(|value| value.checked_add(usize::try_from(x8).ok()?))
            .ok_or(Error::LimitExceeded)
    }

    pub fn get(&self, reference: i8, x8: u32, y8: u32) -> Result<Option<MotionVector>, Error> {
        let vector = *self
            .vectors
            .get(self.index(reference, x8, y8)?)
            .ok_or(Error::InvalidObu)?;
        Ok((vector != INVALID_MOTION_VECTOR).then_some(vector))
    }

    pub fn set(
        &mut self,
        reference: i8,
        x8: u32,
        y8: u32,
        vector: MotionVector,
    ) -> Result<(), Error> {
        if vector == INVALID_MOTION_VECTOR {
            return Err(Error::InvalidObu);
        }
        let index = self.index(reference, x8, y8)?;
        *self.vectors.get_mut(index).ok_or(Error::InvalidObu)? = vector;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TemporalProjection<'a> {
    pub source: &'a MiGrid,
    pub source_is_inter: bool,
    pub source_order_hint: u32,
    pub current_order_hint: u32,
    pub source_reference_order_hints: [u32; 8],
    pub destination_order_hints: [u32; 8],
    pub order_hint_bits: u8,
    pub destination_sign: i32,
}

/// Saved per-slot state needed by the frame-level motion-field estimation
/// process in AV1 section 7.9.1.
#[derive(Clone, Copy, Debug)]
pub struct SavedMotionFieldReference<'a> {
    pub grid: &'a MiGrid,
    pub is_inter: bool,
    pub order_hints: [u32; 8],
}

/// Inputs to the frame-level motion-field estimation process. Reference-frame
/// numbers use the AV1 values `LAST_FRAME` (1) through `ALTREF_FRAME` (7).
#[derive(Clone, Copy, Debug)]
pub struct MotionFieldEstimation<'a> {
    pub references: [Option<SavedMotionFieldReference<'a>>; 8],
    pub ref_frame_idx: [u8; 7],
    pub order_hints: [u32; 8],
    pub current_order_hint: u32,
    pub order_hint_bits: u8,
}

/// Records which normative source projections were attempted and valid.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MotionFieldEstimationResult {
    pub used_last: bool,
    pub projected_last: bool,
    pub projected_bwdref: bool,
    pub projected_altref2: bool,
    pub projected_altref: bool,
    pub projected_last2: bool,
}

/// Runs the complete source-selection order from AV1 section 7.9.1.
pub fn estimate_motion_field(
    mi_columns: u32,
    mi_rows: u32,
    input: MotionFieldEstimation<'_>,
) -> Result<(MotionField, MotionFieldEstimationResult), Error> {
    let mut field = MotionField::new(mi_columns, mi_rows)?;
    if input.order_hint_bits == 0 || input.order_hint_bits > 31 {
        return Err(Error::InvalidObu);
    }
    for slot in input.ref_frame_idx {
        if slot >= 8 {
            return Err(Error::InvalidObu);
        }
    }

    let project = |field: &mut MotionField, reference: i8, sign: i32| {
        let list_index = usize::try_from(reference - 1).map_err(|_| Error::InvalidObu)?;
        let slot = usize::from(input.ref_frame_idx[list_index]);
        let Some(saved) = input.references[slot] else {
            return Ok(false);
        };
        project_reference_motion_field(
            field,
            TemporalProjection {
                source: saved.grid,
                source_is_inter: saved.is_inter,
                source_order_hint: input.order_hints
                    [usize::try_from(reference).map_err(|_| Error::InvalidObu)?],
                current_order_hint: input.current_order_hint,
                source_reference_order_hints: saved.order_hints,
                destination_order_hints: input.order_hints,
                order_hint_bits: input.order_hint_bits,
                destination_sign: sign,
            },
        )
    };

    let last_slot = usize::from(input.ref_frame_idx[0]);
    let last_alt_order_hint = input.references[last_slot]
        .map(|saved| saved.order_hints[7])
        .unwrap_or(input.order_hints[4]);
    let mut result = MotionFieldEstimationResult {
        used_last: last_alt_order_hint != input.order_hints[4],
        ..MotionFieldEstimationResult::default()
    };
    if result.used_last {
        result.projected_last = project(&mut field, 1, -1)?;
    }

    let mut reference_stamp = MFMV_STACK_SIZE - 2;
    if relative_order_hint_distance(
        input.order_hints[5],
        input.current_order_hint,
        input.order_hint_bits,
    )? > 0
    {
        result.projected_bwdref = project(&mut field, 5, 1)?;
        reference_stamp -= i8::from(result.projected_bwdref);
    }
    if relative_order_hint_distance(
        input.order_hints[6],
        input.current_order_hint,
        input.order_hint_bits,
    )? > 0
    {
        result.projected_altref2 = project(&mut field, 6, 1)?;
        reference_stamp -= i8::from(result.projected_altref2);
    }
    if relative_order_hint_distance(
        input.order_hints[7],
        input.current_order_hint,
        input.order_hint_bits,
    )? > 0
        && reference_stamp >= 0
    {
        result.projected_altref = project(&mut field, 7, 1)?;
        reference_stamp -= i8::from(result.projected_altref);
    }
    if reference_stamp >= 0 {
        result.projected_last2 = project(&mut field, 2, -1)?;
    }
    Ok((field, result))
}

pub fn relative_order_hint_distance(a: u32, b: u32, bits: u8) -> Result<i32, Error> {
    if bits == 0 {
        return Ok(0);
    }
    if bits > 31 {
        return Err(Error::InvalidObu);
    }
    let mask = (1u32 << bits) - 1;
    let sign = 1u32 << (bits - 1);
    let difference = a.wrapping_sub(b) & mask;
    Ok(
        i32::try_from(difference & (sign - 1)).map_err(|_| Error::LimitExceeded)?
            - i32::try_from(difference & sign).map_err(|_| Error::LimitExceeded)?,
    )
}

/// Projects one saved reference frame into all seven current reference planes.
/// Returns `false` without modifying `field` when the source is ineligible.
pub fn project_reference_motion_field(
    field: &mut MotionField,
    input: TemporalProjection<'_>,
) -> Result<bool, Error> {
    let (width8, height8) = field.dimensions();
    if !input.source_is_inter
        || input.source.columns() != width8.saturating_mul(2)
        || input.source.rows() != height8.saturating_mul(2)
    {
        return Ok(false);
    }
    if !matches!(input.destination_sign, -1 | 1) || input.order_hint_bits == 0 {
        return Err(Error::InvalidObu);
    }
    let reference_to_current = relative_order_hint_distance(
        input.source_order_hint,
        input.current_order_hint,
        input.order_hint_bits,
    )?;
    for y8 in 0..height8 {
        for x8 in 0..width8 {
            let row = y8
                .checked_mul(2)
                .and_then(|v| v.checked_add(1))
                .ok_or(Error::LimitExceeded)?;
            let column = x8
                .checked_mul(2)
                .and_then(|v| v.checked_add(1))
                .ok_or(Error::LimitExceeded)?;
            let state = input.source.get(row, column).ok_or(Error::InvalidObu)?;
            let source_reference = state.reference_frames[0];
            if !(1..=7).contains(&source_reference) {
                continue;
            }
            let reference_offset = relative_order_hint_distance(
                input.source_order_hint,
                input.source_reference_order_hints
                    [usize::try_from(source_reference).map_err(|_| Error::InvalidObu)?],
                input.order_hint_bits,
            )?;
            if reference_to_current.abs() > MAX_FRAME_DISTANCE
                || reference_offset.abs() > MAX_FRAME_DISTANCE
                || reference_offset <= 0
            {
                continue;
            }
            let motion = state.motion_vectors[0];
            let location_vector = project_motion_vector(
                motion,
                reference_to_current
                    .checked_mul(input.destination_sign)
                    .ok_or(Error::LimitExceeded)?,
                reference_offset,
            )?;
            let Some((destination_x8, destination_y8)) = project_motion_field_position(
                x8,
                y8,
                input.destination_sign,
                location_vector,
                input.source.columns(),
                input.source.rows(),
            )?
            else {
                continue;
            };
            for destination_reference in 1i8..=7 {
                let reference_to_destination = relative_order_hint_distance(
                    input.current_order_hint,
                    input.destination_order_hints
                        [usize::try_from(destination_reference).map_err(|_| Error::InvalidObu)?],
                    input.order_hint_bits,
                )?;
                let projected =
                    project_motion_vector(motion, reference_to_destination, reference_offset)?;
                field.set(
                    destination_reference,
                    destination_x8,
                    destination_y8,
                    projected,
                )?;
            }
        }
    }
    Ok(true)
}

/// Projects a saved motion vector across a different order-hint distance as
/// specified by AV1 section 7.9.3.
pub fn project_motion_vector(
    motion: MotionVector,
    numerator: i32,
    denominator: i32,
) -> Result<MotionVector, Error> {
    if denominator <= 0 {
        return Err(Error::InvalidObu);
    }
    let denominator = denominator.min(MAX_FRAME_DISTANCE);
    let numerator = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let multiplier =
        i64::from(DIV_MULT[usize::try_from(denominator).map_err(|_| Error::LimitExceeded)?]);
    let project = |component: i32| -> Result<i32, Error> {
        let scaled = i64::from(component)
            .checked_mul(i64::from(numerator))
            .and_then(|value| value.checked_mul(multiplier))
            .ok_or(Error::LimitExceeded)?;
        let rounded = if scaled < 0 {
            -((-scaled + (1 << 13)) >> 14)
        } else {
            (scaled + (1 << 13)) >> 14
        };
        i32::try_from(rounded.clamp(-((1 << 14) - 1), (1 << 14) - 1))
            .map_err(|_| Error::LimitExceeded)
    };
    Ok(MotionVector {
        row: project(motion.row)?,
        column: project(motion.column)?,
    })
}

/// Projects an 8x8 motion-field location as specified by AV1 section 7.9.4.
/// `mi_columns` and `mi_rows` are the current frame dimensions in 4x4 units.
pub fn project_motion_field_position(
    x8: u32,
    y8: u32,
    destination_sign: i32,
    projected: MotionVector,
    mi_columns: u32,
    mi_rows: u32,
) -> Result<Option<(u32, u32)>, Error> {
    if !matches!(destination_sign, -1 | 1) || mi_columns < 2 || mi_rows < 2 {
        return Err(Error::InvalidObu);
    }
    let project = |position: u32,
                   delta: i32,
                   maximum: u32,
                   maximum_offset: i64|
     -> Result<Option<u32>, Error> {
        let position = i64::from(position);
        let base = (position >> 3) << 3;
        let magnitude = i64::from(delta).abs() >> 6;
        let offset = if delta < 0 { -magnitude } else { magnitude };
        let projected_position = position
            .checked_add(i64::from(destination_sign) * offset)
            .ok_or(Error::LimitExceeded)?;
        if projected_position < 0
            || projected_position >= i64::from(maximum)
            || projected_position < base - maximum_offset
            || projected_position >= base + 8 + maximum_offset
        {
            return Ok(None);
        }
        Ok(Some(
            u32::try_from(projected_position).map_err(|_| Error::LimitExceeded)?,
        ))
    };
    let Some(y) = project(y8, projected.row, mi_rows >> 1, 0)? else {
        return Ok(None);
    };
    let Some(x) = project(x8, projected.column, mi_columns >> 1, 8)? else {
        return Ok(None);
    };
    Ok(Some((x, y)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionVectorSyntax {
    pub force_integer: bool,
    pub allow_high_precision: bool,
    pub intrabc: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntrabcValidation {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub block_width: u16,
    pub block_height: u16,
    pub has_chroma: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub use_128x128_superblock: bool,
}

pub fn validate_intrabc_motion(
    vector: MotionVector,
    config: IntrabcValidation,
) -> Result<(), Error> {
    if vector.row.unsigned_abs() >= 1 << 14
        || vector.column.unsigned_abs() >= 1 << 14
        || vector.row & 7 != 0
        || vector.column & 7 != 0
    {
        return Err(Error::InvalidObu);
    }
    let delta_row = vector.row >> 3;
    let delta_column = vector.column >> 3;
    let mut top = i64::from(config.block.row * 4) + i64::from(delta_row);
    let mut left = i64::from(config.block.column * 4) + i64::from(delta_column);
    let bottom = top + i64::from(config.block_height);
    let right = left + i64::from(config.block_width);
    if config.has_chroma && config.block_width < 8 && config.subsampling_x {
        left -= 4;
    }
    if config.has_chroma && config.block_height < 8 && config.subsampling_y {
        top -= 4;
    }
    if top < i64::from(config.tile.row_start * 4)
        || left < i64::from(config.tile.column_start * 4)
        || bottom > i64::from(config.tile.row_end * 4)
        || right > i64::from(config.tile.column_end * 4)
    {
        return Err(Error::InvalidObu);
    }
    let superblock_height = if config.use_128x128_superblock {
        128
    } else {
        64
    };
    let active_row = i64::from(config.block.row * 4) / superblock_height;
    let active_column = i64::from(config.block.column * 4) >> 6;
    let source_row = (bottom - 1) / superblock_height;
    let source_column = (right - 1) >> 6;
    let per_row = i64::from(((config.tile.column_end - config.tile.column_start - 1) >> 4) + 1);
    let active = active_row * per_row + active_column;
    let source = source_row * per_row + source_column;
    if source >= active - 1 {
        return Err(Error::InvalidObu);
    }
    let gradient = 2 + i64::from(config.use_128x128_superblock);
    let wavefront_offset = gradient * (active_row - source_row);
    if source_row > active_row || source_column >= active_column - 1 + wavefront_offset {
        return Err(Error::InvalidObu);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MotionCandidate {
    pub vectors: [MotionVector; 2],
    pub weight: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MotionStack {
    entries: Vector<MotionCandidate>,
}

impl MotionStack {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            entries: Vector::with_capacity(MAX_REF_MV_STACK_SIZE)
                .map_err(|_| Error::LimitExceeded)?,
        })
    }

    pub fn entries(&self) -> &[MotionCandidate] {
        &self.entries
    }

    /// Adds a candidate or increases the existing candidate's weight.
    pub fn add(&mut self, vectors: [MotionVector; 2], weight: u16) -> Result<(), Error> {
        if let Some(candidate) = self.entries.iter_mut().find(|item| item.vectors == vectors) {
            candidate.weight = candidate
                .weight
                .checked_add(weight)
                .ok_or(Error::LimitExceeded)?;
        } else if self.entries.len() < MAX_REF_MV_STACK_SIZE {
            self.entries
                .try_push(MotionCandidate { vectors, weight })
                .map_err(|_| Error::LimitExceeded)?;
        }
        Ok(())
    }

    /// Stable descending sort used after the nearest spatial searches.
    pub fn stable_sort_by_weight(&mut self, start: usize, end: usize) -> Result<(), Error> {
        if start > end || end > self.entries.len() {
            return Err(Error::InvalidObu);
        }
        for index in start + 1..end {
            let mut position = index;
            while position > start
                && self.entries[position].weight > self.entries[position - 1].weight
            {
                self.entries.swap(position, position - 1);
                position -= 1;
            }
        }
        Ok(())
    }

    fn add_category_weight(&mut self, end: usize) -> Result<(), Error> {
        if end > self.entries.len() {
            return Err(Error::InvalidObu);
        }
        for candidate in &mut self.entries[..end] {
            candidate.weight = candidate
                .weight
                .checked_add(640)
                .ok_or(Error::LimitExceeded)?;
        }
        Ok(())
    }

    /// Applies section 7.10.2.12's final two-entry fallback construction.
    pub fn ensure_fallbacks(
        &mut self,
        compound: bool,
        global_vectors: [MotionVector; 2],
        compound_candidates: [[MotionVector; 2]; 2],
    ) -> Result<(), Error> {
        if compound {
            if self.entries.len() == 1 {
                let selected = if self.entries[0].vectors == compound_candidates[0] {
                    compound_candidates[1]
                } else {
                    compound_candidates[0]
                };
                self.push_fallback(selected)?;
            } else {
                for candidate in compound_candidates {
                    if self.entries.len() >= 2 {
                        break;
                    }
                    self.push_fallback(candidate)?;
                }
            }
        } else {
            while self.entries.len() < 2 {
                self.push_fallback([global_vectors[0], MotionVector::default()])?;
            }
        }
        Ok(())
    }

    fn push_fallback(&mut self, vectors: [MotionVector; 2]) -> Result<(), Error> {
        self.entries
            .try_push(MotionCandidate { vectors, weight: 2 })
            .map_err(|_| Error::LimitExceeded)
    }
}

/// Adds a neighboring inter block when its references match the current block.
pub fn add_reference_candidate(
    stack: &mut MotionStack,
    candidate: BlockState,
    references: [i8; 2],
    compound: bool,
    weight: u16,
) -> Result<bool, Error> {
    if !candidate.is_inter {
        return Ok(false);
    }
    let mut matched = false;
    if compound {
        if candidate.reference_frames == references {
            stack.add(candidate.motion_vectors, weight)?;
            matched = true;
        }
    } else {
        for list in 0..2 {
            if candidate.reference_frames[list] == references[0] {
                stack.add(
                    [candidate.motion_vectors[list], MotionVector::default()],
                    weight,
                )?;
                matched = true;
            }
        }
    }
    Ok(matched)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpatialScan {
    pub block: BlockRect,
    pub tile: TileBounds,
    pub references: [i8; 2],
    pub compound: bool,
    pub global_types: [GlobalMotionType; 2],
    pub global_vectors: [MotionVector; 2],
}

fn add_spatial_reference_candidate(
    stack: &mut MotionStack,
    mut candidate: BlockState,
    scan: SpatialScan,
    weight: u16,
) -> Result<bool, Error> {
    let dimensions = candidate.size.ok_or(Error::InvalidObu)?.dimensions();
    if dimensions.0.min(dimensions.1) >= 8 {
        for list in 0..(1 + usize::from(scan.compound)) {
            if matches!(candidate.prediction_mode, 16 | 24)
                && scan.global_types[list] > GlobalMotionType::Translation
            {
                candidate.motion_vectors[list] = scan.global_vectors[list];
            }
        }
    }
    add_reference_candidate(stack, candidate, scan.references, scan.compound, weight)
}

pub fn scan_row(
    grid: &MiGrid,
    stack: &mut MotionStack,
    scan: SpatialScan,
    mut delta_row: i32,
) -> Result<bool, Error> {
    if scan.block.column >= grid.columns() || scan.block.row >= grid.rows() {
        return Err(Error::InvalidObu);
    }
    let width = u32::from(scan.block.width_mi);
    let end = width.min(grid.columns() - scan.block.column).min(16);
    let mut delta_column = 0i32;
    let distant = delta_row.unsigned_abs() > 1;
    if distant {
        delta_row += i32::try_from(scan.block.row & 1).map_err(|_| Error::LimitExceeded)?;
        delta_column =
            1 - i32::try_from(scan.block.column & 1).map_err(|_| Error::LimitExceeded)?;
    }
    let mut offset = 0u32;
    let mut found = false;
    while offset < end {
        let row = i64::from(scan.block.row) + i64::from(delta_row);
        let column = i64::from(scan.block.column) + i64::from(delta_column) + i64::from(offset);
        let Some(candidate) = candidate_at(grid, scan.tile, row, column) else {
            break;
        };
        let candidate_width = candidate.size.ok_or(Error::InvalidObu)?.dimensions().0 / 4;
        let mut length = width.min(u32::from(candidate_width));
        if distant {
            length = length.max(2);
        }
        if width >= 16 {
            length = length.max(4);
        }
        found |= add_spatial_reference_candidate(
            stack,
            *candidate,
            scan,
            u16::try_from(length * 2).map_err(|_| Error::LimitExceeded)?,
        )?;
        offset = offset.checked_add(length).ok_or(Error::LimitExceeded)?;
    }
    Ok(found)
}

pub fn scan_column(
    grid: &MiGrid,
    stack: &mut MotionStack,
    scan: SpatialScan,
    mut delta_column: i32,
) -> Result<bool, Error> {
    if scan.block.column >= grid.columns() || scan.block.row >= grid.rows() {
        return Err(Error::InvalidObu);
    }
    let height = u32::from(scan.block.height_mi);
    let end = height.min(grid.rows() - scan.block.row).min(16);
    let mut delta_row = 0i32;
    let distant = delta_column.unsigned_abs() > 1;
    if distant {
        delta_row = 1 - i32::try_from(scan.block.row & 1).map_err(|_| Error::LimitExceeded)?;
        delta_column += i32::try_from(scan.block.column & 1).map_err(|_| Error::LimitExceeded)?;
    }
    let mut offset = 0u32;
    let mut found = false;
    while offset < end {
        let row = i64::from(scan.block.row) + i64::from(delta_row) + i64::from(offset);
        let column = i64::from(scan.block.column) + i64::from(delta_column);
        let Some(candidate) = candidate_at(grid, scan.tile, row, column) else {
            break;
        };
        let candidate_height = candidate.size.ok_or(Error::InvalidObu)?.dimensions().1 / 4;
        let mut length = height.min(u32::from(candidate_height));
        if distant {
            length = length.max(2);
        }
        if height >= 16 {
            length = length.max(4);
        }
        found |= add_spatial_reference_candidate(
            stack,
            *candidate,
            scan,
            u16::try_from(length * 2).map_err(|_| Error::LimitExceeded)?,
        )?;
        offset = offset.checked_add(length).ok_or(Error::LimitExceeded)?;
    }
    Ok(found)
}

pub fn scan_point(
    grid: &MiGrid,
    stack: &mut MotionStack,
    scan: SpatialScan,
    delta_row: i32,
    delta_column: i32,
) -> Result<bool, Error> {
    let row = i64::from(scan.block.row) + i64::from(delta_row);
    let column = i64::from(scan.block.column) + i64::from(delta_column);
    if let Some(candidate) = candidate_at(grid, scan.tile, row, column) {
        return add_spatial_reference_candidate(stack, *candidate, scan, 4);
    }
    Ok(false)
}

fn candidate_at(grid: &MiGrid, tile: TileBounds, row: i64, column: i64) -> Option<&BlockState> {
    if !tile.contains(row, column) {
        return None;
    }
    let row = u32::try_from(row).ok()?;
    let column = u32::try_from(column).ok()?;
    grid.get(row, column).filter(|state| state.size.is_some())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpatialStackResult {
    pub stack: MotionStack,
    pub close_matches: u8,
    pub total_matches: u8,
    pub nearest_count: usize,
    pub any_new_nearest: bool,
}

pub struct CompleteMotionStackConfig<'a> {
    pub spatial: SpatialScan,
    pub temporal_field: Option<&'a MotionField>,
    pub temporal: Option<TemporalScanConfig>,
    pub global_vectors: [MotionVector; 2],
    pub compound_candidates: [[MotionVector; 2]; 2],
}

pub struct NormativeMotionStackConfig<'a> {
    pub spatial: SpatialScan,
    pub temporal_field: Option<&'a MotionField>,
    pub temporal: Option<TemporalScanConfig>,
    pub global_vectors: [MotionVector; 2],
    pub reference_sign_bias: [bool; 8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteMotionStack {
    pub stack: MotionStack,
    pub contexts: MotionContexts,
    pub zero_mv_context: u8,
    pub candidates_found: usize,
}

pub fn build_complete_motion_stack(
    grid: &MiGrid,
    config: CompleteMotionStackConfig<'_>,
) -> Result<CompleteMotionStack, Error> {
    let spatial = build_spatial_motion_stack(grid, config.spatial)?;
    let mut stack = spatial.stack;
    let zero_mv_context = match (config.temporal_field, config.temporal) {
        (Some(field), Some(temporal)) => {
            scan_temporal_motion_field(field, &mut stack, temporal)?.zero_mv_context
        }
        (None, None) => 0,
        _ => return Err(Error::InvalidObu),
    };
    let syntax_candidates = stack.entries.len();
    stack.ensure_fallbacks(
        config.spatial.compound,
        config.global_vectors,
        config.compound_candidates,
    )?;
    let candidates_found = if config.spatial.compound {
        stack.entries.len()
    } else {
        syntax_candidates
    };
    let contexts = derive_motion_contexts_and_clamp(
        &mut stack,
        MotionContextConfig {
            block: config.spatial.block,
            mi_columns: grid.columns(),
            mi_rows: grid.rows(),
            compound: config.spatial.compound,
            close_matches: spatial.close_matches,
            total_matches: spatial.total_matches,
            any_new_nearest: spatial.any_new_nearest,
        },
    )?;
    Ok(CompleteMotionStack {
        stack,
        contexts,
        zero_mv_context,
        candidates_found,
    })
}

/// Builds the complete stack including section 7.10.2.12's partial-reference
/// neighbor search. This is the frame decoder entry point; the lower-level
/// builder remains available for callers that have already collected those
/// two compound candidates.
pub fn build_normative_motion_stack(
    grid: &MiGrid,
    config: NormativeMotionStackConfig<'_>,
) -> Result<CompleteMotionStack, Error> {
    let spatial = build_spatial_motion_stack(grid, config.spatial)?;
    let mut stack = spatial.stack;
    let zero_mv_context = match (config.temporal_field, config.temporal) {
        (Some(field), Some(temporal)) => {
            scan_temporal_motion_field(field, &mut stack, temporal)?.zero_mv_context
        }
        (None, None) => 0,
        _ => return Err(Error::InvalidObu),
    };
    let compound_candidates = collect_extra_motion_candidates(
        grid,
        config.spatial,
        config.reference_sign_bias,
        config.global_vectors,
        (!config.spatial.compound).then_some(&mut stack),
    )?;
    let syntax_candidates = stack.entries.len();
    stack.ensure_fallbacks(
        config.spatial.compound,
        config.global_vectors,
        compound_candidates,
    )?;
    let candidates_found = if config.spatial.compound {
        stack.entries.len()
    } else {
        syntax_candidates
    };
    let contexts = derive_motion_contexts_and_clamp(
        &mut stack,
        MotionContextConfig {
            block: config.spatial.block,
            mi_columns: grid.columns(),
            mi_rows: grid.rows(),
            compound: config.spatial.compound,
            close_matches: spatial.close_matches,
            total_matches: spatial.total_matches,
            any_new_nearest: spatial.any_new_nearest,
        },
    )?;
    Ok(CompleteMotionStack {
        stack,
        contexts,
        zero_mv_context,
        candidates_found,
    })
}

fn collect_extra_motion_candidates(
    grid: &MiGrid,
    scan: SpatialScan,
    sign_bias: [bool; 8],
    globals: [MotionVector; 2],
    mut single_stack: Option<&mut MotionStack>,
) -> Result<[[MotionVector; 2]; 2], Error> {
    let mut identical = [[MotionVector::default(); 2]; 2];
    let mut different = [[MotionVector::default(); 2]; 2];
    let mut identical_count = [0usize; 2];
    let mut different_count = [0usize; 2];
    let width = u32::from(scan.block.width_mi)
        .min(16)
        .min(grid.columns().saturating_sub(scan.block.column));
    let height = u32::from(scan.block.height_mi)
        .min(16)
        .min(grid.rows().saturating_sub(scan.block.row));
    let count = width.min(height);
    for pass in 0..2 {
        let mut offset = 0u32;
        while offset < count {
            let (row, column) = if pass == 0 {
                (
                    i64::from(scan.block.row) - 1,
                    i64::from(scan.block.column + offset),
                )
            } else {
                (
                    i64::from(scan.block.row + offset),
                    i64::from(scan.block.column) - 1,
                )
            };
            let Some(candidate) = candidate_at(grid, scan.tile, row, column) else {
                break;
            };
            for candidate_list in 0..2 {
                let candidate_reference = candidate.reference_frames[candidate_list];
                if !(1..=7).contains(&candidate_reference) {
                    continue;
                }
                let candidate_mv = candidate.motion_vectors[candidate_list];
                if scan.compound {
                    for list in 0..2 {
                        let mut candidate_mv = candidate_mv;
                        if candidate_reference == scan.references[list] && identical_count[list] < 2
                        {
                            identical[list][identical_count[list]] = candidate_mv;
                            identical_count[list] += 1;
                        } else if different_count[list] < 2 {
                            if sign_bias[usize::try_from(candidate_reference)
                                .map_err(|_| Error::InvalidObu)?]
                                != sign_bias[usize::try_from(scan.references[list])
                                    .map_err(|_| Error::InvalidObu)?]
                            {
                                candidate_mv.row =
                                    candidate_mv.row.checked_neg().ok_or(Error::LimitExceeded)?;
                                candidate_mv.column = candidate_mv
                                    .column
                                    .checked_neg()
                                    .ok_or(Error::LimitExceeded)?;
                            }
                            different[list][different_count[list]] = candidate_mv;
                            different_count[list] += 1;
                        }
                    }
                } else {
                    let mut candidate_mv = candidate_mv;
                    if sign_bias
                        [usize::try_from(candidate_reference).map_err(|_| Error::InvalidObu)?]
                        != sign_bias
                            [usize::try_from(scan.references[0]).map_err(|_| Error::InvalidObu)?]
                    {
                        candidate_mv.row =
                            candidate_mv.row.checked_neg().ok_or(Error::LimitExceeded)?;
                        candidate_mv.column = candidate_mv
                            .column
                            .checked_neg()
                            .ok_or(Error::LimitExceeded)?;
                    }
                    if let Some(stack) = single_stack.as_deref_mut() {
                        stack.add([candidate_mv, MotionVector::default()], 2)?;
                    }
                }
            }
            let step = candidate.size.ok_or(Error::InvalidObu)?.dimensions();
            offset = offset
                .checked_add(u32::from(if pass == 0 { step.0 } else { step.1 }) / 4)
                .ok_or(Error::LimitExceeded)?;
        }
    }
    let mut output = [[MotionVector::default(); 2]; 2];
    for list in 0..2 {
        let mut position = 0usize;
        for &candidate in &identical[list][..identical_count[list]] {
            output[position][list] = candidate;
            position += 1;
        }
        for &candidate in &different[list][..different_count[list]] {
            if position >= 2 {
                break;
            }
            output[position][list] = candidate;
            position += 1;
        }
        while position < 2 {
            output[position][list] = globals[list];
            position += 1;
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionContexts {
    pub drl: [u8; MAX_REF_MV_STACK_SIZE],
    pub new_mv: u8,
    pub reference_mv: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionContextConfig {
    pub block: BlockRect,
    pub mi_columns: u32,
    pub mi_rows: u32,
    pub compound: bool,
    pub close_matches: u8,
    pub total_matches: u8,
    pub any_new_nearest: bool,
}

pub fn derive_motion_contexts_and_clamp(
    stack: &mut MotionStack,
    config: MotionContextConfig,
) -> Result<MotionContexts, Error> {
    let block = config.block;
    if block.column >= config.mi_columns
        || block.row >= config.mi_rows
        || config.close_matches > 2
        || config.total_matches > 2
    {
        return Err(Error::InvalidObu);
    }
    let mut drl = [0u8; MAX_REF_MV_STACK_SIZE];
    for (index, pair) in stack.entries.windows(2).enumerate() {
        drl[index] = if pair[0].weight >= 640 {
            u8::from(pair[1].weight < 640)
        } else {
            2
        };
    }
    let row_border = 128i64 + i64::from(block.height_mi) * 32;
    let column_border = 128i64 + i64::from(block.width_mi) * 32;
    let top = -i64::from(block.row) * 32;
    let left = -i64::from(block.column) * 32;
    let bottom =
        (i64::from(config.mi_rows) - i64::from(block.height_mi) - i64::from(block.row)) * 32;
    let right =
        (i64::from(config.mi_columns) - i64::from(block.width_mi) - i64::from(block.column)) * 32;
    let lists = if config.compound { 2 } else { 1 };
    for candidate in &mut stack.entries {
        for vector in &mut candidate.vectors[..lists] {
            vector.row =
                i32::try_from(i64::from(vector.row).clamp(top - row_border, bottom + row_border))
                    .map_err(|_| Error::LimitExceeded)?;
            vector.column = i32::try_from(
                i64::from(vector.column).clamp(left - column_border, right + column_border),
            )
            .map_err(|_| Error::LimitExceeded)?;
        }
    }
    let new = u8::from(config.any_new_nearest);
    let (new_mv, reference_mv) = match config.close_matches {
        0 => (config.total_matches.min(1), config.total_matches),
        1 => (3 - new, 2 + config.total_matches),
        2 => (5 - new, 5),
        _ => return Err(Error::InvalidObu),
    };
    Ok(MotionContexts {
        drl,
        new_mv,
        reference_mv,
    })
}

/// Performs the spatial portion of the normative Find MV Stack process.
/// Temporal-field and extra-reference searches are applied separately.
pub fn build_spatial_motion_stack(
    grid: &MiGrid,
    scan: SpatialScan,
) -> Result<SpatialStackResult, Error> {
    let mut stack = MotionStack::new()?;
    let mut found_above = scan_row(grid, &mut stack, scan, -1)?;
    let mut found_left = scan_column(grid, &mut stack, scan, -1)?;
    if scan.block.width_mi.max(scan.block.height_mi) <= 16 {
        found_above |= scan_point(grid, &mut stack, scan, -1, i32::from(scan.block.width_mi))?;
    }
    let close_matches = u8::from(found_above) + u8::from(found_left);
    let any_new_nearest = immediate_new_mv_match(grid, scan)?;
    let nearest_count = stack.entries.len();
    if nearest_count > 0 {
        stack.add_category_weight(nearest_count)?;
    }

    found_above |= scan_point(grid, &mut stack, scan, -1, -1)?;
    found_above |= scan_row(grid, &mut stack, scan, -3)?;
    found_left |= scan_column(grid, &mut stack, scan, -3)?;
    if scan.block.height_mi > 1 {
        found_above |= scan_row(grid, &mut stack, scan, -5)?;
    }
    if scan.block.width_mi > 1 {
        found_left |= scan_column(grid, &mut stack, scan, -5)?;
    }
    let total_matches = u8::from(found_above) + u8::from(found_left);
    let end = stack.entries.len();
    stack.stable_sort_by_weight(0, nearest_count)?;
    stack.stable_sort_by_weight(nearest_count, end)?;
    Ok(SpatialStackResult {
        stack,
        close_matches,
        total_matches,
        nearest_count,
        any_new_nearest,
    })
}

fn immediate_new_mv_match(grid: &MiGrid, scan: SpatialScan) -> Result<bool, Error> {
    let matches = |candidate: &BlockState| {
        if !candidate.is_inter || !has_new_mv(candidate.prediction_mode) {
            return false;
        }
        if scan.compound {
            candidate.reference_frames == scan.references
        } else {
            candidate.reference_frames.contains(&scan.references[0])
        }
    };
    let scan_line = |row_step: i32, column_step: i32, extent: u32, horizontal: bool| {
        let mut offset = 0u32;
        while offset < extent {
            let row = i64::from(scan.block.row)
                + i64::from(row_step)
                + i64::from(if horizontal { 0 } else { offset });
            let column = i64::from(scan.block.column)
                + i64::from(column_step)
                + i64::from(if horizontal { offset } else { 0 });
            let Some(candidate) = candidate_at(grid, scan.tile, row, column) else {
                break;
            };
            if matches(candidate) {
                return Ok(true);
            }
            let dimensions = candidate.size.ok_or(Error::InvalidObu)?.dimensions();
            offset = offset
                .checked_add(
                    u32::from(if horizontal {
                        dimensions.0
                    } else {
                        dimensions.1
                    }) / 4,
                )
                .ok_or(Error::LimitExceeded)?;
        }
        Ok(false)
    };
    let width = u32::from(scan.block.width_mi)
        .min(grid.columns().saturating_sub(scan.block.column))
        .min(16);
    let height = u32::from(scan.block.height_mi)
        .min(grid.rows().saturating_sub(scan.block.row))
        .min(16);
    if scan_line(-1, 0, width, true)? || scan_line(0, -1, height, false)? {
        return Ok(true);
    }
    if scan.block.width_mi.max(scan.block.height_mi) <= 16 {
        let row = i64::from(scan.block.row) - 1;
        let column = i64::from(scan.block.column) + i64::from(scan.block.width_mi);
        return Ok(candidate_at(grid, scan.tile, row, column).is_some_and(matches));
    }
    Ok(false)
}

const fn has_new_mv(y_mode: u8) -> bool {
    matches!(y_mode, 17 | 20 | 21 | 22 | 23 | 25)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalScanConfig {
    pub block: BlockRect,
    pub references: [i8; 2],
    pub compound: bool,
    pub force_integer: bool,
    pub allow_high_precision: bool,
    pub global_motion: [MotionVector; 2],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TemporalScanResult {
    pub zero_mv_context: u8,
    pub samples_added: u16,
}

/// Performs the normative temporal scan from sections 7.10.2.5-7.10.2.6.
pub fn scan_temporal_motion_field(
    field: &MotionField,
    stack: &mut MotionStack,
    config: TemporalScanConfig,
) -> Result<TemporalScanResult, Error> {
    let mut result = TemporalScanResult::default();
    let sample = |delta_row: i32,
                  delta_column: i32,
                  stack: &mut MotionStack,
                  result: &mut TemporalScanResult|
     -> Result<(), Error> {
        // Section 7.10.2.6 pessimistically sets this context for the origin
        // sample before attempting the motion-field lookup. An unavailable
        // origin therefore leaves the context at one; only a valid candidate
        // close to the global motion resets it to zero.
        if delta_row == 0 && delta_column == 0 {
            result.zero_mv_context = 1;
        }
        let row = (i64::from(config.block.row) + i64::from(delta_row)) | 1;
        let column = (i64::from(config.block.column) + i64::from(delta_column)) | 1;
        let (width8, height8) = field.dimensions();
        if row < 0
            || column < 0
            || row >= i64::from(height8.saturating_mul(2))
            || column >= i64::from(width8.saturating_mul(2))
        {
            return Ok(());
        }
        let x8 = u32::try_from(column >> 1).map_err(|_| Error::LimitExceeded)?;
        let y8 = u32::try_from(row >> 1).map_err(|_| Error::LimitExceeded)?;
        let lists = if config.compound { 2 } else { 1 };
        let mut vectors = [MotionVector::default(); 2];
        for (list, vector) in vectors.iter_mut().enumerate().take(lists) {
            *vector = match field.get(config.references[list], x8, y8)? {
                Some(vector) => lower_motion_vector_precision(
                    vector,
                    config.force_integer,
                    config.allow_high_precision,
                )?,
                None => return Ok(()),
            };
        }
        if delta_row == 0 && delta_column == 0 {
            result.zero_mv_context =
                u8::from(vectors[..lists].iter().enumerate().any(|(list, vector)| {
                    vector.row.abs_diff(config.global_motion[list].row) >= 16
                        || vector.column.abs_diff(config.global_motion[list].column) >= 16
                }));
        }
        stack.add(vectors, 2)?;
        result.samples_added = result
            .samples_added
            .checked_add(1)
            .ok_or(Error::LimitExceeded)?;
        Ok(())
    };

    let width = i32::from(config.block.width_mi);
    let height = i32::from(config.block.height_mi);
    let step_width = if width >= 16 { 4 } else { 2 };
    let step_height = if height >= 16 { 4 } else { 2 };
    let mut row = 0;
    while row < height.min(16) {
        let mut column = 0;
        while column < width.min(16) {
            sample(row, column, stack, &mut result)?;
            column += step_width;
        }
        row += step_height;
    }
    if (2..16).contains(&height) && (2..16).contains(&width) {
        for (delta_row, delta_column) in [(height, -2), (height, width), (height - 2, width)] {
            let superblock_row =
                i32::try_from(config.block.row & 15).map_err(|_| Error::LimitExceeded)? + delta_row;
            let superblock_column = i32::try_from(config.block.column & 15)
                .map_err(|_| Error::LimitExceeded)?
                + delta_column;
            if (0..16).contains(&superblock_row) && (0..16).contains(&superblock_column) {
                sample(delta_row, delta_column, stack, &mut result)?;
            }
        }
    }
    Ok(result)
}

pub fn has_overlappable_candidates(grid: &MiGrid, block: BlockRect, tile: TileBounds) -> bool {
    let upper_row = i64::from(block.row) - 1;
    if tile.contains(upper_row, i64::from(block.column)) {
        let end = block
            .column
            .saturating_add(u32::from(block.width_mi))
            .min(grid.columns());
        let mut column = block.column;
        while column < end {
            let sample_column = column | 1;
            if tile.contains(upper_row, i64::from(sample_column))
                && grid
                    .get(block.row - 1, sample_column)
                    .is_some_and(|state| state.size.is_some() && state.reference_frames[0] > 0)
            {
                return true;
            }
            column = column.saturating_add(2);
        }
    }
    let left_column = i64::from(block.column) - 1;
    if tile.contains(i64::from(block.row), left_column) {
        let end = block
            .row
            .saturating_add(u32::from(block.height_mi))
            .min(grid.rows());
        let mut row = block.row;
        while row < end {
            let sample_row = row | 1;
            if tile.contains(i64::from(sample_row), left_column)
                && grid
                    .get(sample_row, block.column - 1)
                    .is_some_and(|state| state.size.is_some() && state.reference_frames[0] > 0)
            {
                return true;
            }
            row = row.saturating_add(2);
        }
    }
    false
}

pub fn read_motion_vector(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut MotionVectorCdfs,
    predictor: MotionVector,
    syntax: MotionVectorSyntax,
) -> Result<MotionVector, Error> {
    let context = u8::from(syntax.intrabc);
    let joint = cdfs.read_joint(decoder, context)?;
    if joint >= 4 {
        return Err(Error::InvalidObu);
    }
    let row = if matches!(joint, 2 | 3) {
        read_component(decoder, cdfs, context, 0, syntax)?
    } else {
        0
    };
    let column = if matches!(joint, 1 | 3) {
        read_component(decoder, cdfs, context, 1, syntax)?
    } else {
        0
    };
    Ok(MotionVector {
        row: predictor.row.checked_add(row).ok_or(Error::LimitExceeded)?,
        column: predictor
            .column
            .checked_add(column)
            .ok_or(Error::LimitExceeded)?,
    })
}

fn read_component(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut MotionVectorCdfs,
    context: u8,
    component: u8,
    syntax: MotionVectorSyntax,
) -> Result<i32, Error> {
    let negative = cdfs.read_sign(decoder, context, component)?;
    let class = cdfs.read_class(decoder, context, component)?;
    if class > 10 {
        return Err(Error::InvalidObu);
    }
    let magnitude = if class == 0 {
        let class0_bit = cdfs.read_class0_bit(decoder, context, component)?;
        let fraction = if syntax.force_integer {
            3
        } else {
            cdfs.read_fraction(decoder, context, component, Some(class0_bit))?
        };
        let high_precision = if syntax.allow_high_precision {
            cdfs.read_high_precision(decoder, context, component, true)?
        } else {
            1
        };
        compose_motion_magnitude(class, class0_bit, 0, fraction, high_precision)?
    } else {
        let mut offset = 0u32;
        for index in 0..class {
            offset |= u32::from(cdfs.read_offset_bit(decoder, context, component, index)?) << index;
        }
        let fraction = if syntax.force_integer {
            3
        } else {
            cdfs.read_fraction(decoder, context, component, None)?
        };
        let high_precision = if syntax.allow_high_precision {
            cdfs.read_high_precision(decoder, context, component, false)?
        } else {
            1
        };
        compose_motion_magnitude(class, 0, offset, fraction, high_precision)?
    };
    let magnitude = i32::try_from(magnitude).map_err(|_| Error::LimitExceeded)?;
    Ok(if negative { -magnitude } else { magnitude })
}

pub fn compose_motion_magnitude(
    class: u8,
    class0_bit: u8,
    offset: u32,
    fraction: u8,
    high_precision: u8,
) -> Result<u32, Error> {
    if class > 10 || class0_bit > 1 || fraction > 3 || high_precision > 1 {
        return Err(Error::InvalidObu);
    }
    if class == 0 {
        if offset != 0 {
            return Err(Error::InvalidObu);
        }
        Ok(
            ((u32::from(class0_bit) << 3) | (u32::from(fraction) << 1) | u32::from(high_precision))
                + 1,
        )
    } else {
        if offset >= (1u32 << class) {
            return Err(Error::InvalidObu);
        }
        Ok((2u32 << (class + 2))
            + ((offset << 3) | (u32::from(fraction) << 1) | u32::from(high_precision))
            + 1)
    }
}

pub fn lower_motion_vector_precision(
    motion: MotionVector,
    force_integer: bool,
    allow_high_precision: bool,
) -> Result<MotionVector, Error> {
    if allow_high_precision {
        return Ok(motion);
    }
    let lower = |value: i32| -> Result<i32, Error> {
        if force_integer {
            let magnitude = value.checked_abs().ok_or(Error::LimitExceeded)?;
            let integer = magnitude.checked_add(3).ok_or(Error::LimitExceeded)? >> 3;
            let rounded = integer.checked_shl(3).ok_or(Error::LimitExceeded)?;
            Ok(if value > 0 { rounded } else { -rounded })
        } else if value & 1 != 0 {
            Ok(if value > 0 { value - 1 } else { value + 1 })
        } else {
            Ok(value)
        }
    };
    Ok(MotionVector {
        row: lower(motion.row)?,
        column: lower(motion.column)?,
    })
}

/// Projects the center of a block through its reference's global-motion model
/// to produce the predictor used by AV1 section 7.10.2.1.
pub fn setup_global_motion_vector(
    global: GlobalMotion,
    block: BlockRect,
    allow_high_precision: bool,
    force_integer: bool,
) -> Result<MotionVector, Error> {
    const WARPEDMODEL_PREC_BITS: u32 = 16;
    let motion = match global.kind {
        GlobalMotionType::Identity => MotionVector::default(),
        GlobalMotionType::Translation => MotionVector {
            row: global.params[0] >> (WARPEDMODEL_PREC_BITS - 3),
            column: global.params[1] >> (WARPEDMODEL_PREC_BITS - 3),
        },
        GlobalMotionType::RotZoom | GlobalMotionType::Affine => {
            let width = u16::from(block.width_mi) * 4;
            let height = u16::from(block.height_mi) * 4;
            let x = i64::from(block.column)
                .checked_mul(4)
                .and_then(|value| value.checked_add(i64::from(width / 2) - 1))
                .ok_or(Error::LimitExceeded)?;
            let y = i64::from(block.row)
                .checked_mul(4)
                .and_then(|value| value.checked_add(i64::from(height / 2) - 1))
                .ok_or(Error::LimitExceeded)?;
            let one = 1i64 << WARPEDMODEL_PREC_BITS;
            let xc = (i64::from(global.params[2]) - one)
                .checked_mul(x)
                .and_then(|value| value.checked_add(i64::from(global.params[3]) * y))
                .and_then(|value| value.checked_add(i64::from(global.params[0])))
                .ok_or(Error::LimitExceeded)?;
            let yc = i64::from(global.params[4])
                .checked_mul(x)
                .and_then(|value| value.checked_add((i64::from(global.params[5]) - one) * y))
                .and_then(|value| value.checked_add(i64::from(global.params[1])))
                .ok_or(Error::LimitExceeded)?;
            let shift = WARPEDMODEL_PREC_BITS - if allow_high_precision { 3 } else { 2 };
            let round = |value: i64| -> Result<i32, Error> {
                let magnitude = value.unsigned_abs();
                let rounded = (magnitude + (1u64 << (shift - 1))) >> shift;
                let signed = if value < 0 {
                    -i64::try_from(rounded).map_err(|_| Error::LimitExceeded)?
                } else {
                    i64::try_from(rounded).map_err(|_| Error::LimitExceeded)?
                };
                let scaled = if allow_high_precision {
                    signed
                } else {
                    signed.checked_mul(2).ok_or(Error::LimitExceeded)?
                };
                i32::try_from(scaled).map_err(|_| Error::LimitExceeded)
            };
            MotionVector {
                row: round(yc)?,
                column: round(xc)?,
            }
        }
    };
    lower_motion_vector_precision(motion, force_integer, allow_high_precision)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotionScaleInput {
    pub frame_width: u32,
    pub frame_height: u32,
    pub reference_upscaled_width: u32,
    pub reference_height: u32,
    pub x: u32,
    pub y: u32,
    /// Row and column components in eighth-luma-sample units.
    pub motion_vector: [i32; 2],
    pub subsampling_x: bool,
    pub subsampling_y: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaledMotion {
    /// Reference coordinate in 1/1024 sample units.
    pub start_x: i64,
    /// Reference coordinate in 1/1024 sample units.
    pub start_y: i64,
    pub step_x: i64,
    pub step_y: i64,
}

pub fn scale_motion_vector(input: MotionScaleInput) -> Result<ScaledMotion, Error> {
    const REFERENCE_SCALE_SHIFT: u8 = 14;
    const SUBPEL_BITS: u8 = 4;
    const SCALE_SUBPEL_BITS: u8 = 10;
    if input.frame_width == 0
        || input.frame_height == 0
        || input.reference_upscaled_width == 0
        || input.reference_height == 0
        || u64::from(input.frame_width) * 2 < u64::from(input.reference_upscaled_width)
        || u64::from(input.frame_height) * 2 < u64::from(input.reference_height)
        || u64::from(input.frame_width) > u64::from(input.reference_upscaled_width) * 16
        || u64::from(input.frame_height) > u64::from(input.reference_height) * 16
    {
        return Err(Error::InvalidObu);
    }
    let x_scale = ((u64::from(input.reference_upscaled_width) << REFERENCE_SCALE_SHIFT)
        + u64::from(input.frame_width / 2))
        / u64::from(input.frame_width);
    let y_scale = ((u64::from(input.reference_height) << REFERENCE_SCALE_SHIFT)
        + u64::from(input.frame_height / 2))
        / u64::from(input.frame_height);
    let half_sample = 1i64 << (SUBPEL_BITS - 1);
    let original_x = (i64::from(input.x) << SUBPEL_BITS)
        + ((2 * i64::from(input.motion_vector[1])) >> usize::from(input.subsampling_x))
        + half_sample;
    let original_y = (i64::from(input.y) << SUBPEL_BITS)
        + ((2 * i64::from(input.motion_vector[0])) >> usize::from(input.subsampling_y))
        + half_sample;
    let base_x = original_x
        .checked_mul(i64::try_from(x_scale).map_err(|_| Error::LimitExceeded)?)
        .and_then(|value| value.checked_sub(half_sample << REFERENCE_SCALE_SHIFT))
        .ok_or(Error::LimitExceeded)?;
    let base_y = original_y
        .checked_mul(i64::try_from(y_scale).map_err(|_| Error::LimitExceeded)?)
        .and_then(|value| value.checked_sub(half_sample << REFERENCE_SCALE_SHIFT))
        .ok_or(Error::LimitExceeded)?;
    let offset = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
    Ok(ScaledMotion {
        start_x: round2_signed(
            base_x,
            REFERENCE_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS,
        )? + offset,
        start_y: round2_signed(
            base_y,
            REFERENCE_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS,
        )? + offset,
        step_x: round2_signed(
            i64::try_from(x_scale).map_err(|_| Error::LimitExceeded)?,
            REFERENCE_SCALE_SHIFT - SCALE_SUBPEL_BITS,
        )?,
        step_y: round2_signed(
            i64::try_from(y_scale).map_err(|_| Error::LimitExceeded)?,
            REFERENCE_SCALE_SHIFT - SCALE_SUBPEL_BITS,
        )?,
    })
}

fn round2_signed(value: i64, shift: u8) -> Result<i64, Error> {
    if shift == 0 {
        return Ok(value);
    }
    let magnitude = value.unsigned_abs();
    let rounded = magnitude
        .checked_add(1u64 << (shift - 1))
        .ok_or(Error::LimitExceeded)?
        >> shift;
    let rounded = i64::try_from(rounded).map_err(|_| Error::LimitExceeded)?;
    Ok(if value < 0 { -rounded } else { rounded })
}

const WARPEDMODEL_PREC_BITS: u8 = 16;
const GM_ABS_TRANS_ONLY_BITS: u8 = 9;
const GM_ABS_TRANS_BITS: u8 = 12;
const GM_ABS_ALPHA_BITS: u8 = 12;
const GM_TRANS_ONLY_PREC_BITS: u8 = 3;
const GM_TRANS_PREC_BITS: u8 = 6;
const GM_ALPHA_PREC_BITS: u8 = 15;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
pub enum GlobalMotionType {
    #[default]
    Identity,
    Translation,
    RotZoom,
    Affine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalMotion {
    pub kind: GlobalMotionType,
    pub params: [i32; 6],
}

impl Default for GlobalMotion {
    fn default() -> Self {
        Self {
            kind: GlobalMotionType::Identity,
            params: [
                0,
                0,
                1 << WARPEDMODEL_PREC_BITS,
                0,
                0,
                1 << WARPEDMODEL_PREC_BITS,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShearParams {
    pub valid: bool,
    pub alpha: i32,
    pub beta: i32,
    pub gamma: i32,
    pub delta: i32,
}

/// Section 7.11.3.7's reciprocal approximation without storing the normative
/// 256-entry table: every entry is the nearest integer to 2^22 / (256 + f).
pub fn resolve_divisor(divisor: i64) -> Result<(u8, i32), Error> {
    const DIV_LUT_BITS: u32 = 8;
    const DIV_LUT_PREC_BITS: u32 = 14;
    let magnitude = divisor.unsigned_abs();
    if magnitude == 0 {
        return Err(Error::InvalidObu);
    }
    let n = 63 - magnitude.leading_zeros();
    let excess = magnitude - (1 << n);
    let f = if n > DIV_LUT_BITS {
        excess
            .checked_add(1 << (n - DIV_LUT_BITS - 1))
            .ok_or(Error::LimitExceeded)?
            >> (n - DIV_LUT_BITS)
    } else {
        excess
            .checked_shl(DIV_LUT_BITS - n)
            .ok_or(Error::LimitExceeded)?
    };
    if f > 256 {
        return Err(Error::InvalidObu);
    }
    let denominator = 256u64.checked_add(f).ok_or(Error::LimitExceeded)?;
    let factor = ((1u64 << 22) + denominator / 2) / denominator;
    let factor = i32::try_from(factor).map_err(|_| Error::LimitExceeded)?;
    Ok((
        u8::try_from(n + DIV_LUT_PREC_BITS).map_err(|_| Error::LimitExceeded)?,
        if divisor < 0 { -factor } else { factor },
    ))
}

/// Derives and validates the two shear operations for an affine warp as
/// specified by section 7.11.3.6.
pub fn setup_shear(warp: [i32; 6]) -> Result<ShearParams, Error> {
    const REDUCE_BITS: u8 = 6;
    let unit = 1i32 << WARPEDMODEL_PREC_BITS;
    let alpha0 = warp[2].saturating_sub(unit).clamp(-32768, 32767);
    let beta0 = warp[3].clamp(-32768, 32767);
    let (div_shift, div_factor) = resolve_divisor(i64::from(warp[2]))?;
    let gamma_product = i64::from(warp[4])
        .checked_shl(u32::from(WARPEDMODEL_PREC_BITS))
        .and_then(|value| value.checked_mul(i64::from(div_factor)))
        .ok_or(Error::LimitExceeded)?;
    let gamma0 = i32::try_from(round2_signed(gamma_product, div_shift)?)
        .map_err(|_| Error::LimitExceeded)?
        .clamp(-32768, 32767);
    let cross = i64::from(warp[3])
        .checked_mul(i64::from(warp[4]))
        .and_then(|value| value.checked_mul(i64::from(div_factor)))
        .ok_or(Error::LimitExceeded)?;
    let correction =
        i32::try_from(round2_signed(cross, div_shift)?).map_err(|_| Error::LimitExceeded)?;
    let delta0 = warp[5]
        .saturating_sub(correction)
        .saturating_sub(unit)
        .clamp(-32768, 32767);
    let reduce = |value: i32| -> Result<i32, Error> {
        i32::try_from(round2_signed(i64::from(value), REDUCE_BITS)?)
            .map_err(|_| Error::LimitExceeded)?
            .checked_shl(u32::from(REDUCE_BITS))
            .ok_or(Error::LimitExceeded)
    };
    let alpha = reduce(alpha0)?;
    let beta = reduce(beta0)?;
    let gamma = reduce(gamma0)?;
    let delta = reduce(delta0)?;
    let valid = 4i64 * i64::from(alpha.abs()) + 7i64 * i64::from(beta.abs()) < i64::from(unit)
        && 4i64 * i64::from(gamma.abs()) + 4i64 * i64::from(delta.abs()) < i64::from(unit);
    Ok(ShearParams {
        valid,
        alpha,
        beta,
        gamma,
        delta,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WarpSample {
    pub source_y: i32,
    pub source_x: i32,
    pub destination_y: i32,
    pub destination_x: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalWarpConfig {
    pub mi_row: u32,
    pub mi_column: u32,
    pub width_mi: u8,
    pub height_mi: u8,
    pub motion_vector: MotionVector,
}

/// Section 7.11.3.8's regularized two-variable least-squares affine fit.
/// Returns `Ok(None)` when the determinant is zero and local warp is invalid.
pub fn estimate_local_warp(
    samples: &[WarpSample],
    config: LocalWarpConfig,
) -> Result<Option<[i32; 6]>, Error> {
    if samples.len() > 8 || config.width_mi == 0 || config.height_mi == 0 {
        return Err(Error::InvalidObu);
    }
    let mid_y = i64::from(config.mi_row)
        .checked_mul(4)
        .and_then(|value| value.checked_add(i64::from(config.height_mi) * 2))
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::LimitExceeded)?;
    let mid_x = i64::from(config.mi_column)
        .checked_mul(4)
        .and_then(|value| value.checked_add(i64::from(config.width_mi) * 2))
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::LimitExceeded)?;
    let source_center_y = mid_y.checked_mul(8).ok_or(Error::LimitExceeded)?;
    let source_center_x = mid_x.checked_mul(8).ok_or(Error::LimitExceeded)?;
    let destination_center_y = source_center_y
        .checked_add(i64::from(config.motion_vector.row))
        .ok_or(Error::LimitExceeded)?;
    let destination_center_x = source_center_x
        .checked_add(i64::from(config.motion_vector.column))
        .ok_or(Error::LimitExceeded)?;
    let mut a00 = 0i64;
    let mut a01 = 0i64;
    let mut a11 = 0i64;
    let mut bx0 = 0i64;
    let mut bx1 = 0i64;
    let mut by0 = 0i64;
    let mut by1 = 0i64;
    for sample in samples {
        let sy = i64::from(sample.source_y) - source_center_y;
        let sx = i64::from(sample.source_x) - source_center_x;
        let dy = i64::from(sample.destination_y) - destination_center_y;
        let dx = i64::from(sample.destination_x) - destination_center_x;
        if (sx - dx).unsigned_abs() >= 256 || (sy - dy).unsigned_abs() >= 256 {
            continue;
        }
        a00 = checked_add(
            a00,
            ls_product(sx, sx)?
                .checked_add(8)
                .ok_or(Error::LimitExceeded)?,
        )?;
        a01 = checked_add(
            a01,
            ls_product(sx, sy)?
                .checked_add(4)
                .ok_or(Error::LimitExceeded)?,
        )?;
        a11 = checked_add(
            a11,
            ls_product(sy, sy)?
                .checked_add(8)
                .ok_or(Error::LimitExceeded)?,
        )?;
        bx0 = checked_add(
            bx0,
            ls_product(sx, dx)?
                .checked_add(8)
                .ok_or(Error::LimitExceeded)?,
        )?;
        bx1 = checked_add(
            bx1,
            ls_product(sy, dx)?
                .checked_add(4)
                .ok_or(Error::LimitExceeded)?,
        )?;
        by0 = checked_add(
            by0,
            ls_product(sx, dy)?
                .checked_add(4)
                .ok_or(Error::LimitExceeded)?,
        )?;
        by1 = checked_add(
            by1,
            ls_product(sy, dy)?
                .checked_add(8)
                .ok_or(Error::LimitExceeded)?,
        )?;
    }
    let determinant = i128::from(a00) * i128::from(a11) - i128::from(a01) * i128::from(a01);
    if determinant == 0 {
        return Ok(None);
    }
    let determinant_i64 = i64::try_from(determinant).map_err(|_| Error::LimitExceeded)?;
    let (raw_shift, mut factor) = resolve_divisor(determinant_i64)?;
    let mut shift = i32::from(raw_shift) - i32::from(WARPEDMODEL_PREC_BITS);
    if shift < 0 {
        factor = factor
            .checked_shl(u32::try_from(-shift).map_err(|_| Error::LimitExceeded)?)
            .ok_or(Error::LimitExceeded)?;
        shift = 0;
    }
    let shift = u8::try_from(shift).map_err(|_| Error::LimitExceeded)?;
    let divide = |value: i128, diagonal: bool| -> Result<i32, Error> {
        let scaled = value
            .checked_mul(i128::from(factor))
            .ok_or(Error::LimitExceeded)?;
        let value = round2_signed_i128(scaled, shift)?;
        let clamp = 1i128 << 13;
        let value = if diagonal {
            value.clamp((1i128 << 16) - clamp + 1, (1i128 << 16) + clamp - 1)
        } else {
            value.clamp(-clamp + 1, clamp - 1)
        };
        i32::try_from(value).map_err(|_| Error::LimitExceeded)
    };
    let p2 = divide(
        i128::from(a11) * i128::from(bx0) - i128::from(a01) * i128::from(bx1),
        true,
    )?;
    let p3 = divide(
        -i128::from(a01) * i128::from(bx0) + i128::from(a00) * i128::from(bx1),
        false,
    )?;
    let p4 = divide(
        i128::from(a11) * i128::from(by0) - i128::from(a01) * i128::from(by1),
        false,
    )?;
    let p5 = divide(
        -i128::from(a01) * i128::from(by0) + i128::from(a00) * i128::from(by1),
        true,
    )?;
    let mv_scale = 1i128 << (WARPEDMODEL_PREC_BITS - 3);
    let vx = i128::from(config.motion_vector.column) * mv_scale
        - (i128::from(mid_x) * i128::from(p2 - (1 << 16)) + i128::from(mid_y) * i128::from(p3));
    let vy = i128::from(config.motion_vector.row) * mv_scale
        - (i128::from(mid_x) * i128::from(p4) + i128::from(mid_y) * i128::from(p5 - (1 << 16)));
    let translation = 1i128 << 23;
    Ok(Some([
        i32::try_from(vx.clamp(-translation, translation - 1)).map_err(|_| Error::LimitExceeded)?,
        i32::try_from(vy.clamp(-translation, translation - 1)).map_err(|_| Error::LimitExceeded)?,
        p2,
        p3,
        p4,
        p5,
    ]))
}

pub fn collect_warp_samples(
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    reference_frame: i8,
    current_mv: MotionVector,
) -> Result<Vector<WarpSample>, Error> {
    if reference_frame <= 0 || block.width_mi == 0 || block.height_mi == 0 {
        return Err(Error::InvalidObu);
    }
    let mut samples = Vector::with_capacity(8).map_err(|_| Error::LimitExceeded)?;
    let mut seen = [None; 8];
    let mut scanned = 0usize;
    let mut valid_count = 0usize;
    let threshold = (i32::from(block.width_mi.max(block.height_mi)) * 4).clamp(16, 112);
    let mut add = |delta_row: i32, delta_column: i32| -> Result<(), Error> {
        if scanned >= 8 {
            return Ok(());
        }
        let row = i64::from(block.row) + i64::from(delta_row);
        let column = i64::from(block.column) + i64::from(delta_column);
        if !tile.contains(row, column) {
            return Ok(());
        }
        let row = u32::try_from(row).map_err(|_| Error::InvalidObu)?;
        let column = u32::try_from(column).map_err(|_| Error::InvalidObu)?;
        let state = grid.get(row, column).ok_or(Error::InvalidObu)?;
        if state.size.is_none() || state.reference_frames != [reference_frame, -1] {
            return Ok(());
        }
        let candidate_size = state.size.ok_or(Error::InvalidObu)?;
        let (candidate_width, candidate_height) = candidate_size.dimensions();
        let width_mi = u32::from(candidate_width / 4);
        let height_mi = u32::from(candidate_height / 4);
        let candidate_row = row & !(height_mi - 1);
        let candidate_column = column & !(width_mi - 1);
        if seen[..scanned].contains(&Some((candidate_row, candidate_column))) {
            return Ok(());
        }
        seen[scanned] = Some((candidate_row, candidate_column));
        scanned += 1;
        let candidate = grid
            .get(candidate_row, candidate_column)
            .ok_or(Error::InvalidObu)?;
        let vector = candidate.motion_vectors[0];
        let difference = i32::try_from(vector.row.abs_diff(current_mv.row))
            .map_err(|_| Error::LimitExceeded)?
            .checked_add(
                i32::try_from(vector.column.abs_diff(current_mv.column))
                    .map_err(|_| Error::LimitExceeded)?,
            )
            .ok_or(Error::LimitExceeded)?;
        let valid = difference <= threshold;
        if !valid && scanned > 1 {
            return Ok(());
        }
        let mid_y = candidate_row
            .checked_mul(4)
            .and_then(|value| value.checked_add(height_mi * 2))
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::LimitExceeded)?;
        let mid_x = candidate_column
            .checked_mul(4)
            .and_then(|value| value.checked_add(width_mi * 2))
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::LimitExceeded)?;
        let sample = WarpSample {
            source_y: i32::try_from(mid_y.checked_mul(8).ok_or(Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?,
            source_x: i32::try_from(mid_x.checked_mul(8).ok_or(Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?,
            destination_y: i32::try_from(mid_y.checked_mul(8).ok_or(Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?
                .checked_add(vector.row)
                .ok_or(Error::LimitExceeded)?,
            destination_x: i32::try_from(mid_x.checked_mul(8).ok_or(Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?
                .checked_add(vector.column)
                .ok_or(Error::LimitExceeded)?,
        };
        if valid {
            if valid_count < samples.len() {
                samples[valid_count] = sample;
            } else {
                samples.try_push(sample).map_err(|_| Error::LimitExceeded)?;
            }
            valid_count += 1;
        } else if scanned == 1 {
            samples.try_push(sample).map_err(|_| Error::LimitExceeded)?;
        }
        Ok(())
    };

    let width_mi = u32::from(block.width_mi);
    let height_mi = u32::from(block.height_mi);
    let mut do_top_left = true;
    let mut do_top_right = true;
    if block.row > tile.row_start {
        let above = grid
            .get(block.row - 1, block.column)
            .ok_or(Error::InvalidObu)?;
        if let Some(size) = above.size {
            let source_width = u32::from(size.dimensions().0 / 4);
            if width_mi <= source_width {
                let column_offset = -i64::from(block.column & (source_width - 1));
                if column_offset < 0 {
                    do_top_left = false;
                }
                if column_offset + i64::from(source_width) > i64::from(width_mi) {
                    do_top_right = false;
                }
                add(-1, 0)?;
            } else {
                let limit = width_mi.min(grid.columns() - block.column);
                let mut offset = 0u32;
                while offset < limit {
                    let size = grid
                        .get(block.row - 1, block.column + offset)
                        .and_then(|state| state.size)
                        .ok_or(Error::InvalidObu)?;
                    let source_width = u32::from(size.dimensions().0 / 4);
                    add(-1, i32::try_from(offset).map_err(|_| Error::LimitExceeded)?)?;
                    offset = offset
                        .checked_add(width_mi.min(source_width))
                        .ok_or(Error::LimitExceeded)?;
                }
            }
        }
    }
    if block.column > tile.column_start {
        let left = grid
            .get(block.row, block.column - 1)
            .ok_or(Error::InvalidObu)?;
        if let Some(size) = left.size {
            let source_height = u32::from(size.dimensions().1 / 4);
            if height_mi <= source_height {
                let row_offset = -i64::from(block.row & (source_height - 1));
                if row_offset < 0 {
                    do_top_left = false;
                }
                add(0, -1)?;
            } else {
                let limit = height_mi.min(grid.rows() - block.row);
                let mut offset = 0u32;
                while offset < limit {
                    let size = grid
                        .get(block.row + offset, block.column - 1)
                        .and_then(|state| state.size)
                        .ok_or(Error::InvalidObu)?;
                    let source_height = u32::from(size.dimensions().1 / 4);
                    add(i32::try_from(offset).map_err(|_| Error::LimitExceeded)?, -1)?;
                    offset = offset
                        .checked_add(height_mi.min(source_height))
                        .ok_or(Error::LimitExceeded)?;
                }
            }
        }
    }
    if do_top_left {
        add(-1, -1)?;
    }
    if do_top_right && width_mi.max(height_mi) <= 16 {
        add(
            -1,
            i32::try_from(width_mi).map_err(|_| Error::LimitExceeded)?,
        )?;
    }
    if valid_count > 0 {
        samples.truncate(valid_count);
    } else if scanned == 0 {
        samples.clear();
    } else {
        samples.truncate(1);
    }
    Ok(samples)
}

pub fn derive_local_warp(
    grid: &MiGrid,
    block: BlockRect,
    tile: TileBounds,
    reference_frame: i8,
    motion_vector: MotionVector,
) -> Result<Option<[i32; 6]>, Error> {
    let samples = collect_warp_samples(grid, block, tile, reference_frame, motion_vector)?;
    estimate_local_warp(
        &samples,
        LocalWarpConfig {
            mi_row: block.row,
            mi_column: block.column,
            width_mi: block.width_mi,
            height_mi: block.height_mi,
            motion_vector,
        },
    )
}

fn ls_product(a: i64, b: i64) -> Result<i64, Error> {
    a.checked_mul(b)
        .map(|product| product >> 2)
        .and_then(|product| product.checked_add(a))
        .and_then(|product| product.checked_add(b))
        .ok_or(Error::LimitExceeded)
}

fn checked_add(a: i64, b: i64) -> Result<i64, Error> {
    a.checked_add(b).ok_or(Error::LimitExceeded)
}

fn round2_signed_i128(value: i128, shift: u8) -> Result<i128, Error> {
    if shift == 0 {
        return Ok(value);
    }
    let magnitude = value.unsigned_abs();
    let rounded = magnitude
        .checked_add(1u128 << (shift - 1))
        .ok_or(Error::LimitExceeded)?
        >> shift;
    let rounded = i128::try_from(rounded).map_err(|_| Error::LimitExceeded)?;
    Ok(if value < 0 { -rounded } else { rounded })
}

pub(crate) fn parse_global_motion(
    bits: &mut Bits<'_>,
    frame_is_intra: bool,
    previous: &[GlobalMotion; 7],
) -> Result<[GlobalMotion; 7], Error> {
    let mut result = [GlobalMotion::default(); 7];
    if frame_is_intra {
        return Ok(result);
    }
    for reference in 0..7 {
        let kind = if !bits.bit()? {
            GlobalMotionType::Identity
        } else if bits.bit()? {
            GlobalMotionType::RotZoom
        } else if bits.bit()? {
            GlobalMotionType::Translation
        } else {
            GlobalMotionType::Affine
        };
        result[reference].kind = kind;
        if kind >= GlobalMotionType::RotZoom {
            result[reference].params[2] =
                read_global_param(bits, kind, 2, previous[reference].params[2])?;
            result[reference].params[3] =
                read_global_param(bits, kind, 3, previous[reference].params[3])?;
            if kind == GlobalMotionType::Affine {
                result[reference].params[4] =
                    read_global_param(bits, kind, 4, previous[reference].params[4])?;
                result[reference].params[5] =
                    read_global_param(bits, kind, 5, previous[reference].params[5])?;
            } else {
                result[reference].params[4] = -result[reference].params[3];
                result[reference].params[5] = result[reference].params[2];
            }
        }
        if kind >= GlobalMotionType::Translation {
            result[reference].params[0] =
                read_global_param(bits, kind, 0, previous[reference].params[0])?;
            result[reference].params[1] =
                read_global_param(bits, kind, 1, previous[reference].params[1])?;
        }
    }
    Ok(result)
}

fn read_global_param(
    bits: &mut Bits<'_>,
    kind: GlobalMotionType,
    index: usize,
    previous: i32,
) -> Result<i32, Error> {
    let (absolute_bits, precision_bits) = if index < 2 {
        if kind == GlobalMotionType::Translation {
            (GM_ABS_TRANS_ONLY_BITS, GM_TRANS_ONLY_PREC_BITS)
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let precision_difference = WARPEDMODEL_PREC_BITS - precision_bits;
    let sub = if index % 3 == 2 {
        1 << precision_bits
    } else {
        0
    };
    let previous_reference = (previous >> precision_difference) - sub;
    let maximum = 1i32 << absolute_bits;
    let decoded = decode_signed_subexp_with_ref(bits, -maximum, maximum + 1, previous_reference)?;
    decoded
        .checked_add(sub)
        .and_then(|value| value.checked_shl(u32::from(precision_difference)))
        .ok_or(Error::LimitExceeded)
}

fn decode_signed_subexp_with_ref(
    bits: &mut Bits<'_>,
    low: i32,
    high: i32,
    reference: i32,
) -> Result<i32, Error> {
    let symbols = high.checked_sub(low).ok_or(Error::InvalidObu)? as u32;
    let shifted_reference = reference.checked_sub(low).ok_or(Error::InvalidObu)?;
    if shifted_reference < 0 || shifted_reference as u32 >= symbols {
        return Err(Error::InvalidObu);
    }
    let value = decode_unsigned_subexp_with_ref(bits, symbols, shifted_reference as u32)?;
    low.checked_add(value as i32).ok_or(Error::LimitExceeded)
}

fn decode_unsigned_subexp_with_ref(
    bits: &mut Bits<'_>,
    symbols: u32,
    reference: u32,
) -> Result<u32, Error> {
    if symbols == 0 || reference >= symbols {
        return Err(Error::InvalidObu);
    }
    let value = decode_subexp(bits, symbols)?;
    Ok(if (reference << 1) <= symbols {
        inverse_recenter(reference, value)
    } else {
        symbols - 1 - inverse_recenter(symbols - 1 - reference, value)
    })
}

fn decode_subexp(bits: &mut Bits<'_>, symbols: u32) -> Result<u32, Error> {
    if symbols == 0 {
        return Err(Error::InvalidObu);
    }
    let mut index = 0u32;
    let mut consumed = 0u32;
    loop {
        let width = if index == 0 { 3 } else { 3 + index - 1 };
        let bucket = 1u32.checked_shl(width).ok_or(Error::InvalidObu)?;
        if symbols <= consumed.saturating_add(bucket.saturating_mul(3)) {
            return bits
                .read_ns(symbols - consumed)?
                .checked_add(consumed)
                .ok_or(Error::LimitExceeded);
        }
        if bits.bit()? {
            index += 1;
            consumed = consumed.checked_add(bucket).ok_or(Error::LimitExceeded)?;
        } else {
            return (bits.read(width as u8)? as u32)
                .checked_add(consumed)
                .ok_or(Error::LimitExceeded);
        }
    }
}

fn inverse_recenter(reference: u32, value: u32) -> u32 {
    if value > (reference << 1) {
        value
    } else if value & 1 != 0 {
        reference - ((value + 1) >> 1)
    } else {
        reference + (value >> 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrabc_motion_enforces_integer_tile_and_delay_constraints() {
        let config = IntrabcValidation {
            block: BlockRect {
                row: 0,
                column: 96,
                width_mi: 4,
                height_mi: 4,
            },
            tile: TileBounds {
                row_start: 0,
                row_end: 16,
                column_start: 0,
                column_end: 160,
            },
            block_width: 16,
            block_height: 16,
            has_chroma: true,
            subsampling_x: true,
            subsampling_y: true,
            use_128x128_superblock: false,
        };
        assert_eq!(
            validate_intrabc_motion(
                MotionVector {
                    row: 0,
                    column: -256 * 8,
                },
                config,
            ),
            Ok(())
        );
        assert_eq!(
            validate_intrabc_motion(MotionVector { row: 0, column: -1 }, config,),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            validate_intrabc_motion(
                MotionVector {
                    row: 0,
                    column: -32 * 8,
                },
                config,
            ),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn inverse_recentering_is_bijective_near_reference() {
        let values = [10, 9, 11, 8, 12, 7, 13];
        for (encoded, expected) in values.into_iter().enumerate() {
            assert_eq!(inverse_recenter(10, encoded as u32), expected);
        }
    }

    #[test]
    fn identity_defaults_use_fixed_point_diagonal() {
        assert_eq!(GlobalMotion::default().params, [0, 0, 65536, 0, 0, 65536]);
    }

    #[test]
    fn divisor_resolution_matches_normative_lookup_entries() {
        assert_eq!(resolve_divisor(256), Ok((22, 16384)));
        assert_eq!(resolve_divisor(257), Ok((22, 16320)));
        assert_eq!(resolve_divisor(-260), Ok((22, -16132)));
        assert_eq!(resolve_divisor(65_528), Ok((29, 8192)));
        assert_eq!(resolve_divisor(0), Err(Error::InvalidObu));
    }

    #[test]
    fn shear_setup_accepts_identity_and_rejects_excessive_skew() {
        assert_eq!(
            setup_shear(GlobalMotion::default().params),
            Ok(ShearParams {
                valid: true,
                alpha: 0,
                beta: 0,
                gamma: 0,
                delta: 0,
            })
        );
        let mut skewed = GlobalMotion::default().params;
        skewed[3] = 32767;
        assert!(!setup_shear(skewed).unwrap().valid);
    }

    #[test]
    fn local_warp_fit_recovers_identity_from_identity_samples() {
        let samples = [
            WarpSample {
                source_y: 16,
                source_x: 16,
                destination_y: 16,
                destination_x: 16,
            },
            WarpSample {
                source_y: 16,
                source_x: 32,
                destination_y: 16,
                destination_x: 32,
            },
            WarpSample {
                source_y: 32,
                source_x: 16,
                destination_y: 32,
                destination_x: 16,
            },
        ];
        assert_eq!(
            estimate_local_warp(
                &samples,
                LocalWarpConfig {
                    mi_row: 0,
                    mi_column: 0,
                    width_mi: 2,
                    height_mi: 2,
                    motion_vector: MotionVector::default(),
                },
            ),
            Ok(Some(GlobalMotion::default().params))
        );
        assert_eq!(
            estimate_local_warp(
                &[],
                LocalWarpConfig {
                    mi_row: 0,
                    mi_column: 0,
                    width_mi: 2,
                    height_mi: 2,
                    motion_vector: MotionVector::default(),
                },
            ),
            Ok(None)
        );
    }

    #[test]
    fn warp_sample_search_collects_above_then_left_candidates() {
        let mut grid = MiGrid::new(8, 8).unwrap();
        let neighbor = crate::block_state::BlockState {
            size: Some(crate::partition::BlockSize::Block8x8),
            is_inter: true,
            reference_frames: [1, -1],
            motion_vectors: [MotionVector::default(); 2],
            ..crate::block_state::BlockState::default()
        };
        grid.fill(
            BlockRect::new(2, 0, crate::partition::BlockSize::Block8x8),
            neighbor,
        )
        .unwrap();
        grid.fill(
            BlockRect::new(0, 2, crate::partition::BlockSize::Block8x8),
            neighbor,
        )
        .unwrap();
        let samples = collect_warp_samples(
            &grid,
            BlockRect::new(2, 2, crate::partition::BlockSize::Block8x8),
            TileBounds {
                column_start: 0,
                column_end: 8,
                row_start: 0,
                row_end: 8,
            },
            1,
            MotionVector::default(),
        )
        .unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!((samples[0].source_y, samples[0].source_x), (24, 88));
        assert_eq!((samples[1].source_y, samples[1].source_x), (88, 24));
    }

    #[test]
    fn unscaled_motion_uses_1024_units_per_sample() {
        let scaled = scale_motion_vector(MotionScaleInput {
            frame_width: 1920,
            frame_height: 1080,
            reference_upscaled_width: 1920,
            reference_height: 1080,
            x: 10,
            y: 20,
            motion_vector: [0, 8],
            subsampling_x: false,
            subsampling_y: false,
        })
        .unwrap();
        assert_eq!(scaled.start_x, 11 * 1024 + 32);
        assert_eq!(scaled.start_y, 20 * 1024 + 32);
        assert_eq!((scaled.step_x, scaled.step_y), (1024, 1024));
    }

    #[test]
    fn invalid_reference_scale_is_rejected() {
        assert_eq!(
            scale_motion_vector(MotionScaleInput {
                frame_width: 100,
                frame_height: 100,
                reference_upscaled_width: 201,
                reference_height: 100,
                x: 0,
                y: 0,
                motion_vector: [0, 0],
                subsampling_x: false,
                subsampling_y: false,
            }),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn motion_magnitude_matches_class_zero_and_offset_classes() {
        assert_eq!(compose_motion_magnitude(0, 0, 0, 0, 0), Ok(1));
        assert_eq!(compose_motion_magnitude(0, 1, 0, 3, 1), Ok(16));
        assert_eq!(compose_motion_magnitude(1, 0, 0, 0, 0), Ok(17));
        assert_eq!(compose_motion_magnitude(1, 0, 1, 3, 1), Ok(32));
        assert_eq!(
            compose_motion_magnitude(1, 0, 2, 0, 0),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn lower_precision_rounds_with_normative_sign_rules() {
        assert_eq!(
            lower_motion_vector_precision(
                MotionVector {
                    row: 13,
                    column: -13
                },
                false,
                false,
            ),
            Ok(MotionVector {
                row: 12,
                column: -12
            })
        );
        assert_eq!(
            lower_motion_vector_precision(
                MotionVector {
                    row: 13,
                    column: -13
                },
                true,
                false,
            ),
            Ok(MotionVector {
                row: 16,
                column: -16
            })
        );
    }

    #[test]
    fn global_motion_vector_projects_block_center_and_lowers_precision() {
        use crate::partition::BlockSize;

        let block = BlockRect::new(3, 5, BlockSize::Block16x8);
        assert_eq!(
            setup_global_motion_vector(GlobalMotion::default(), block, true, false),
            Ok(MotionVector::default())
        );
        let translation = GlobalMotion {
            kind: GlobalMotionType::Translation,
            params: [3 << 13, -5 << 13, 1 << 16, 0, 0, 1 << 16],
        };
        assert_eq!(
            setup_global_motion_vector(translation, block, true, false),
            Ok(MotionVector { row: 3, column: -5 })
        );
        let affine = GlobalMotion {
            kind: GlobalMotionType::Affine,
            params: [0, 0, (1 << 16) + (1 << 13), 0, 0, 1 << 16],
        };
        assert_eq!(
            setup_global_motion_vector(affine, block, false, false),
            Ok(MotionVector { row: 0, column: 20 })
        );
    }

    #[test]
    fn motion_stack_deduplicates_and_sorts_stably() {
        let a = [MotionVector { row: 1, column: 2 }, MotionVector::default()];
        let b = [MotionVector { row: 3, column: 4 }, MotionVector::default()];
        let c = [MotionVector { row: 5, column: 6 }, MotionVector::default()];
        let mut stack = MotionStack::new().unwrap();
        stack.add(a, 2).unwrap();
        stack.add(b, 4).unwrap();
        stack.add(c, 4).unwrap();
        stack.add(a, 3).unwrap();
        assert_eq!(stack.entries().len(), 3);
        stack.stable_sort_by_weight(0, 3).unwrap();
        assert_eq!(stack.entries()[0].vectors, a);
        assert_eq!(stack.entries()[1].vectors, b);
        assert_eq!(stack.entries()[2].vectors, c);
    }

    #[test]
    fn motion_stack_fallbacks_preserve_required_duplicate_globals() {
        let global = MotionVector {
            row: 8,
            column: -16,
        };
        let mut single = MotionStack::new().unwrap();
        single
            .ensure_fallbacks(
                false,
                [global, MotionVector::default()],
                [[MotionVector::default(); 2]; 2],
            )
            .unwrap();
        assert_eq!(single.entries().len(), 2);
        assert_eq!(single.entries()[0].vectors[0], global);
        assert_eq!(single.entries()[1].vectors[0], global);

        let mut compound = MotionStack::new().unwrap();
        let candidates = [
            [global, MotionVector::default()],
            [MotionVector::default(), global],
        ];
        compound.add(candidates[0], 4).unwrap();
        compound
            .ensure_fallbacks(true, [MotionVector::default(); 2], candidates)
            .unwrap();
        assert_eq!(compound.entries()[1].vectors, candidates[1]);
    }

    #[test]
    fn normative_motion_stack_combines_partial_compound_neighbors() {
        use crate::partition::BlockSize;

        let first = MotionVector { row: 8, column: 16 };
        let second = MotionVector {
            row: -24,
            column: 32,
        };
        let mut grid = MiGrid::new(8, 8).unwrap();
        grid.fill(
            BlockRect::new(0, 0, BlockSize::Block8x8),
            BlockState {
                size: Some(BlockSize::Block8x8),
                is_inter: true,
                reference_frames: [1, 7],
                motion_vectors: [first, second],
                ..BlockState::default()
            },
        )
        .unwrap();
        let complete = build_normative_motion_stack(
            &grid,
            NormativeMotionStackConfig {
                spatial: SpatialScan {
                    block: BlockRect::new(0, 2, BlockSize::Block8x8),
                    tile: TileBounds {
                        column_start: 0,
                        column_end: 8,
                        row_start: 0,
                        row_end: 8,
                    },
                    references: [1, 5],
                    compound: true,
                    global_types: [GlobalMotionType::Identity; 2],
                    global_vectors: [MotionVector::default(); 2],
                },
                temporal_field: None,
                temporal: None,
                global_vectors: [MotionVector::default(); 2],
                reference_sign_bias: [false; 8],
            },
        )
        .unwrap();
        assert_eq!(complete.candidates_found, 2);
        assert_eq!(complete.stack.entries()[0].vectors, [first, first]);
        assert_eq!(complete.stack.entries()[1].vectors, [second, second]);
    }

    #[test]
    fn complete_motion_stack_orchestrates_empty_spatial_fallback() {
        let grid = MiGrid::new(8, 8).unwrap();
        let global = MotionVector { row: 8, column: 16 };
        let complete = build_complete_motion_stack(
            &grid,
            CompleteMotionStackConfig {
                spatial: SpatialScan {
                    block: BlockRect::new(0, 0, crate::partition::BlockSize::Block8x8),
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
        assert_eq!(complete.candidates_found, 0);
        assert_eq!(complete.stack.entries().len(), 2);
        assert_eq!(complete.stack.entries()[0].vectors[0], global);
        assert_eq!(complete.zero_mv_context, 0);
    }

    #[test]
    fn reference_candidate_requires_matching_inter_reference() {
        let mut stack = MotionStack::new().unwrap();
        let candidate = BlockState {
            is_inter: true,
            reference_frames: [2, -1],
            motion_vectors: [MotionVector { row: 8, column: -4 }, MotionVector::default()],
            ..BlockState::default()
        };
        add_reference_candidate(&mut stack, candidate, [1, -1], false, 4).unwrap();
        assert!(stack.entries().is_empty());
        add_reference_candidate(&mut stack, candidate, [2, -1], false, 4).unwrap();
        assert_eq!(stack.entries()[0].vectors[0], candidate.motion_vectors[0]);
    }

    #[test]
    fn spatial_row_scan_uses_candidate_width_as_weight() {
        use crate::partition::BlockSize;

        let mut grid = MiGrid::new(8, 4).unwrap();
        let candidate = BlockState {
            size: Some(BlockSize::Block16x8),
            is_inter: true,
            reference_frames: [1, -1],
            motion_vectors: [MotionVector { row: 4, column: 8 }, MotionVector::default()],
            ..BlockState::default()
        };
        grid.fill(BlockRect::new(0, 0, BlockSize::Block16x8), candidate)
            .unwrap();
        grid.fill(BlockRect::new(4, 0, BlockSize::Block16x8), candidate)
            .unwrap();
        let scan = SpatialScan {
            block: BlockRect::new(0, 2, BlockSize::Block32x8),
            tile: TileBounds {
                column_start: 0,
                column_end: 8,
                row_start: 0,
                row_end: 4,
            },
            references: [1, -1],
            compound: false,
            global_types: [GlobalMotionType::Identity; 2],
            global_vectors: [MotionVector::default(); 2],
        };
        let mut stack = MotionStack::new().unwrap();
        scan_row(&grid, &mut stack, scan, -2).unwrap();
        assert_eq!(stack.entries().len(), 1);
        assert_eq!(stack.entries()[0].weight, 16);
    }

    #[test]
    fn point_scan_does_not_cross_tile_boundary() {
        use crate::partition::BlockSize;

        let mut grid = MiGrid::new(4, 4).unwrap();
        grid.fill(
            BlockRect::new(1, 1, BlockSize::Block4x4),
            BlockState {
                size: Some(BlockSize::Block4x4),
                is_inter: true,
                reference_frames: [1, -1],
                ..BlockState::default()
            },
        )
        .unwrap();
        let scan = SpatialScan {
            block: BlockRect::new(2, 2, BlockSize::Block8x8),
            tile: TileBounds {
                column_start: 2,
                column_end: 4,
                row_start: 2,
                row_end: 4,
            },
            references: [1, -1],
            compound: false,
            global_types: [GlobalMotionType::Identity; 2],
            global_vectors: [MotionVector::default(); 2],
        };
        let mut stack = MotionStack::new().unwrap();
        scan_point(&grid, &mut stack, scan, -1, -1).unwrap();
        assert!(stack.entries().is_empty());
    }

    #[test]
    fn spatial_stack_marks_immediate_candidates_as_nearest_category() {
        use crate::partition::BlockSize;

        let mut grid = MiGrid::new(8, 8).unwrap();
        let candidate = BlockState {
            size: Some(BlockSize::Block16x16),
            is_inter: true,
            reference_frames: [1, -1],
            motion_vectors: [MotionVector { row: 2, column: 6 }, MotionVector::default()],
            prediction_mode: 17,
            ..BlockState::default()
        };
        grid.fill(BlockRect::new(2, 0, BlockSize::Block16x16), candidate)
            .unwrap();
        grid.fill(BlockRect::new(0, 2, BlockSize::Block16x16), candidate)
            .unwrap();
        let result = build_spatial_motion_stack(
            &grid,
            SpatialScan {
                block: BlockRect::new(2, 2, BlockSize::Block16x16),
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
        )
        .unwrap();
        assert_eq!(result.close_matches, 2);
        assert_eq!(result.nearest_count, 1);
        assert!(result.any_new_nearest);
        assert!(result.stack.entries()[0].weight >= 640);
    }

    #[test]
    fn spatial_stack_uses_global_vector_for_warped_global_neighbor() {
        use crate::partition::BlockSize;

        let mut grid = MiGrid::new(8, 8).unwrap();
        let stored = MotionVector { row: 0, column: -7 };
        for (rect, size, prediction_mode) in [
            (
                BlockRect::new(2, 0, BlockSize::Block8x8),
                BlockSize::Block8x8,
                14,
            ),
            (
                BlockRect::new(0, 2, BlockSize::Block8x8),
                BlockSize::Block8x8,
                16,
            ),
        ] {
            grid.fill(
                rect,
                BlockState {
                    size: Some(size),
                    is_inter: true,
                    reference_frames: [1, -1],
                    motion_vectors: [stored, MotionVector::default()],
                    prediction_mode,
                    ..BlockState::default()
                },
            )
            .unwrap();
        }
        let global = MotionVector {
            row: -1,
            column: -7,
        };
        let result = build_spatial_motion_stack(
            &grid,
            SpatialScan {
                block: BlockRect::new(2, 2, BlockSize::Block8x8),
                tile: TileBounds {
                    column_start: 0,
                    column_end: 8,
                    row_start: 0,
                    row_end: 8,
                },
                references: [1, -1],
                compound: false,
                global_types: [GlobalMotionType::Affine, GlobalMotionType::Identity],
                global_vectors: [global, MotionVector::default()],
            },
        )
        .unwrap();
        assert_eq!(result.nearest_count, 2);
        assert_eq!(result.stack.entries()[0].vectors[0], stored);
        assert_eq!(result.stack.entries()[1].vectors[0], global);
        assert!(
            result.stack.entries()[..2]
                .iter()
                .all(|candidate| candidate.weight >= 640)
        );
    }

    #[test]
    fn motion_contexts_classify_weights_and_clamp_vectors() {
        let mut stack = MotionStack::new().unwrap();
        stack
            .add(
                [
                    MotionVector {
                        row: -10_000,
                        column: 10_000,
                    },
                    MotionVector::default(),
                ],
                650,
            )
            .unwrap();
        stack
            .add([MotionVector::default(), MotionVector::default()], 10)
            .unwrap();
        let contexts = derive_motion_contexts_and_clamp(
            &mut stack,
            MotionContextConfig {
                block: BlockRect {
                    column: 0,
                    row: 0,
                    width_mi: 4,
                    height_mi: 4,
                },
                mi_columns: 16,
                mi_rows: 16,
                compound: false,
                close_matches: 1,
                total_matches: 2,
                any_new_nearest: true,
            },
        )
        .unwrap();
        assert_eq!(contexts.drl[0], 1);
        assert_eq!((contexts.new_mv, contexts.reference_mv), (2, 4));
        assert_eq!(stack.entries()[0].vectors[0].row, -256);
        assert_eq!(stack.entries()[0].vectors[0].column, 640);
    }

    #[test]
    fn overlap_candidate_search_samples_at_eight_by_eight_granularity() {
        use crate::partition::BlockSize;

        let mut grid = MiGrid::new(8, 8).unwrap();
        grid.fill(
            BlockRect::new(2, 0, BlockSize::Block16x8),
            BlockState {
                size: Some(BlockSize::Block16x8),
                is_inter: true,
                reference_frames: [1, -1],
                ..BlockState::default()
            },
        )
        .unwrap();
        let tile = TileBounds {
            column_start: 0,
            column_end: 8,
            row_start: 0,
            row_end: 8,
        };
        assert!(has_overlappable_candidates(
            &grid,
            BlockRect::new(2, 2, BlockSize::Block16x16),
            tile,
        ));
    }

    #[test]
    fn temporal_motion_projection_uses_normative_reciprocal_table() {
        let motion = MotionVector {
            row: 100,
            column: -200,
        };
        assert_eq!(
            project_motion_vector(motion, 2, 4),
            Ok(MotionVector {
                row: 50,
                column: -100,
            })
        );
        assert_eq!(
            project_motion_vector(motion, -2, 4),
            Ok(MotionVector {
                row: -50,
                column: 100,
            })
        );
        assert_eq!(
            project_motion_vector(MotionVector { row: 1, column: -1 }, 1, 3),
            Ok(MotionVector::default())
        );
        assert_eq!(project_motion_vector(motion, 1, 0), Err(Error::InvalidObu));
        assert_eq!(
            project_motion_vector(
                MotionVector {
                    row: i32::MAX,
                    column: i32::MIN,
                },
                31,
                1,
            ),
            Ok(MotionVector {
                row: 16383,
                column: -16383,
            })
        );
    }

    #[test]
    fn temporal_motion_field_position_enforces_projection_windows() {
        assert_eq!(
            project_motion_field_position(
                8,
                8,
                1,
                MotionVector {
                    row: 64,
                    column: -128,
                },
                64,
                64,
            ),
            Ok(Some((6, 9)))
        );
        assert_eq!(
            project_motion_field_position(
                8,
                8,
                -1,
                MotionVector {
                    row: 0,
                    column: -128,
                },
                64,
                64,
            ),
            Ok(Some((10, 8)))
        );
        assert_eq!(
            project_motion_field_position(
                8,
                8,
                1,
                MotionVector {
                    row: 512,
                    column: 0,
                },
                64,
                64,
            ),
            Ok(None)
        );
        assert_eq!(
            project_motion_field_position(0, 0, 0, MotionVector::default(), 64, 64),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn temporal_motion_field_is_reference_partitioned_and_bounds_checked() {
        let mut field = MotionField::new(16, 12).unwrap();
        assert_eq!(field.dimensions(), (8, 6));
        assert_eq!(field.get(1, 3, 4), Ok(None));
        let vector = MotionVector {
            row: -24,
            column: 41,
        };
        field.set(1, 3, 4, vector).unwrap();
        assert_eq!(field.get(1, 3, 4), Ok(Some(vector)));
        assert_eq!(field.get(2, 3, 4), Ok(None));
        assert_eq!(field.get(0, 3, 4), Err(Error::InvalidObu));
        assert_eq!(field.set(1, 8, 4, vector), Err(Error::InvalidObu));
        assert_eq!(MotionField::new(1, 12), Err(Error::InvalidObu));
    }

    #[test]
    fn saved_reference_motion_is_projected_into_each_destination_plane() {
        use crate::partition::BlockSize;

        assert_eq!(relative_order_hint_distance(1, 15, 4), Ok(2));
        assert_eq!(relative_order_hint_distance(15, 1, 4), Ok(-2));

        let mut source = MiGrid::new(8, 8).unwrap();
        source
            .fill(
                BlockRect::new(0, 0, BlockSize::Block32x32),
                BlockState {
                    size: Some(BlockSize::Block32x32),
                    is_inter: true,
                    reference_frames: [1, -1],
                    motion_vectors: [
                        MotionVector {
                            row: 100,
                            column: -200,
                        },
                        MotionVector::default(),
                    ],
                    ..BlockState::default()
                },
            )
            .unwrap();
        let mut field = MotionField::new(8, 8).unwrap();
        let mut destination_hints = [0; 8];
        destination_hints[1..].fill(6);
        assert_eq!(
            project_reference_motion_field(
                &mut field,
                TemporalProjection {
                    source: &source,
                    source_is_inter: true,
                    source_order_hint: 10,
                    current_order_hint: 8,
                    source_reference_order_hints: [6; 8],
                    destination_order_hints: destination_hints,
                    order_hint_bits: 5,
                    destination_sign: 1,
                },
            ),
            Ok(true)
        );
        assert_eq!(
            field.get(1, 0, 0),
            Ok(Some(MotionVector {
                row: 50,
                column: -100,
            }))
        );
        assert!(field.get(7, 2, 3).unwrap().is_some());

        let mut ineligible = MotionField::new(8, 8).unwrap();
        assert_eq!(
            project_reference_motion_field(
                &mut ineligible,
                TemporalProjection {
                    source: &source,
                    source_is_inter: false,
                    source_order_hint: 10,
                    current_order_hint: 8,
                    source_reference_order_hints: [6; 8],
                    destination_order_hints: destination_hints,
                    order_hint_bits: 5,
                    destination_sign: 1,
                },
            ),
            Ok(false)
        );
        assert_eq!(ineligible.get(1, 0, 0), Ok(None));
    }

    #[test]
    fn frame_motion_field_estimation_obeys_normative_source_stamp_order() {
        use crate::partition::BlockSize;

        let mut source = MiGrid::new(8, 8).unwrap();
        source
            .fill(
                BlockRect::new(0, 0, BlockSize::Block32x32),
                BlockState {
                    size: Some(BlockSize::Block32x32),
                    is_inter: true,
                    reference_frames: [1, -1],
                    motion_vectors: [
                        MotionVector {
                            row: 32,
                            column: -64,
                        },
                        MotionVector::default(),
                    ],
                    ..BlockState::default()
                },
            )
            .unwrap();
        let saved = SavedMotionFieldReference {
            grid: &source,
            is_inter: true,
            order_hints: [6; 8],
        };
        let references = [Some(saved); 8];
        let order_hints = [0, 7, 6, 5, 4, 10, 11, 12];
        let (field, result) = estimate_motion_field(
            8,
            8,
            MotionFieldEstimation {
                references,
                ref_frame_idx: [0, 1, 2, 3, 4, 5, 6],
                order_hints,
                current_order_hint: 8,
                order_hint_bits: 5,
            },
        )
        .unwrap();
        assert_eq!(
            result,
            MotionFieldEstimationResult {
                used_last: true,
                projected_last: true,
                projected_bwdref: true,
                projected_altref2: true,
                projected_altref: false,
                projected_last2: false,
            }
        );
        assert!(field.get(1, 0, 0).unwrap().is_some());
    }

    #[test]
    fn invalid_future_source_does_not_consume_motion_field_stamp() {
        let valid = MiGrid::new(8, 8).unwrap();
        let invalid = MiGrid::new(4, 4).unwrap();
        let saved_valid = SavedMotionFieldReference {
            grid: &valid,
            is_inter: true,
            order_hints: [6; 8],
        };
        let saved_invalid = SavedMotionFieldReference {
            grid: &invalid,
            is_inter: true,
            order_hints: [6; 8],
        };
        let mut references = [Some(saved_valid); 8];
        references[4] = Some(saved_invalid);
        let (_, result) = estimate_motion_field(
            8,
            8,
            MotionFieldEstimation {
                references,
                ref_frame_idx: [0, 1, 2, 3, 4, 5, 6],
                order_hints: [0, 7, 6, 5, 4, 10, 7, 12],
                current_order_hint: 8,
                order_hint_bits: 5,
            },
        )
        .unwrap();
        assert!(result.used_last);
        assert!(!result.projected_bwdref);
        assert!(!result.projected_altref2);
        assert!(result.projected_altref);
        assert!(result.projected_last2);
    }

    #[test]
    fn temporal_scan_adds_lowered_candidates_and_updates_zero_context() {
        use crate::partition::BlockSize;

        let mut field = MotionField::new(16, 16).unwrap();
        let vector = MotionVector {
            row: 17,
            column: -19,
        };
        for y in 0..2 {
            for x in 0..2 {
                field.set(1, x, y, vector).unwrap();
            }
        }
        let mut stack = MotionStack::new().unwrap();
        let result = scan_temporal_motion_field(
            &field,
            &mut stack,
            TemporalScanConfig {
                block: BlockRect::new(0, 0, BlockSize::Block16x16),
                references: [1, -1],
                compound: false,
                force_integer: false,
                allow_high_precision: false,
                global_motion: [MotionVector::default(); 2],
            },
        )
        .unwrap();
        assert_eq!(result.zero_mv_context, 1);
        assert_eq!(result.samples_added, 4);
        assert_eq!(stack.entries().len(), 1);
        assert_eq!(
            stack.entries()[0].vectors[0],
            MotionVector {
                row: 16,
                column: -18
            }
        );
        assert_eq!(stack.entries()[0].weight, 8);
    }

    #[test]
    fn unavailable_temporal_origin_preserves_nonzero_zero_mv_context() {
        use crate::partition::BlockSize;

        let field = MotionField::new(16, 16).unwrap();
        let mut stack = MotionStack::new().unwrap();
        let result = scan_temporal_motion_field(
            &field,
            &mut stack,
            TemporalScanConfig {
                block: BlockRect::new(0, 0, BlockSize::Block16x16),
                references: [1, -1],
                compound: false,
                force_integer: false,
                allow_high_precision: false,
                global_motion: [MotionVector::default(); 2],
            },
        )
        .unwrap();

        assert_eq!(result.zero_mv_context, 1);
        assert_eq!(result.samples_added, 0);
        assert!(stack.entries().is_empty());
    }
}
