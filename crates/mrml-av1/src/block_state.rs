//! Per-4x4 decoded state shared by partition, mode, and prediction syntax.

use crate::{
    Error,
    motion::MotionVector,
    partition::{BlockRect, BlockSize, TileBounds, partition_context},
    transform::{TxSize, TxType},
};
use mrml_runtime::Vector;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockState {
    pub size: Option<BlockSize>,
    pub segment_id: u8,
    pub skip: bool,
    pub skip_mode: bool,
    pub is_inter: bool,
    pub tx_size: Option<TxSize>,
    pub tx_type: TxType,
    pub loop_filter_tx_sizes: [TxSize; 3],
    pub qindex: u8,
    pub delta_lf: [i8; 4],
    pub cdef_index: Option<u8>,
    /// AV1 reference-frame enums; the second entry is `-1` for `NONE`.
    pub reference_frames: [i8; 2],
    pub motion_vectors: [MotionVector; 2],
    /// Numeric AV1 prediction-mode enum retained for neighbor contexts.
    pub prediction_mode: u8,
    pub motion_mode: u8,
    pub compound_type: u8,
    pub compound_group_index: u8,
    pub compound_index: u8,
    pub interpolation_filters: [u8; 2],
    pub inter_intra_mode: u8,
    pub wedge_index: u8,
    pub wedge_sign: bool,
    pub mask_type: bool,
    /// Decoded luma and chroma palette sizes retained for neighbor contexts.
    pub palette_sizes: [u8; 2],
    /// Sorted luma and U palette colors used by the above/left cache process.
    pub palette_colors: [[u16; 8]; 2],
}

impl Default for BlockState {
    fn default() -> Self {
        Self {
            size: None,
            segment_id: 0,
            skip: false,
            skip_mode: false,
            is_inter: false,
            tx_size: None,
            tx_type: TxType::DctDct,
            loop_filter_tx_sizes: [TxSize::Tx4x4; 3],
            qindex: 0,
            delta_lf: [0; 4],
            cdef_index: None,
            reference_frames: [0, -1],
            motion_vectors: [MotionVector::default(); 2],
            prediction_mode: 0,
            motion_mode: 0,
            compound_type: 0,
            compound_group_index: 0,
            compound_index: 1,
            interpolation_filters: [0; 2],
            inter_intra_mode: 0,
            wedge_index: 0,
            wedge_sign: false,
            mask_type: false,
            palette_sizes: [0; 2],
            palette_colors: [[0; 8]; 2],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiGrid {
    columns: u32,
    rows: u32,
    cells: Vector<BlockState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentPredictionContexts {
    above: Vector<u8>,
    left: Vector<u8>,
}

/// One CDEF strength index per 64x64 region. `None` represents the spec's
/// cleared `-1` sentinel at superblock entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CdefIndexGrid {
    columns: u32,
    rows: u32,
    cells: Vector<Option<u8>>,
}

impl CdefIndexGrid {
    pub fn new(mi_columns: u32, mi_rows: u32) -> Result<Self, Error> {
        let columns = mi_columns.div_ceil(16);
        let rows = mi_rows.div_ceil(16);
        if columns == 0 || rows == 0 {
            return Err(Error::InvalidObu);
        }
        let length = usize::try_from(columns)
            .map_err(|_| Error::LimitExceeded)?
            .checked_mul(usize::try_from(rows).map_err(|_| Error::LimitExceeded)?)
            .ok_or(Error::LimitExceeded)?;
        let mut cells = Vector::new();
        cells
            .try_resize(length, None)
            .map_err(|_| Error::LimitExceeded)?;
        Ok(Self {
            columns,
            rows,
            cells,
        })
    }

    pub fn get(&self, mi_row: u32, mi_column: u32) -> Option<Option<u8>> {
        self.index(mi_row, mi_column)
            .and_then(|index| self.cells.get(index).copied())
    }

    pub fn fill_block(&mut self, block: BlockRect, value: u8) -> Result<(), Error> {
        let first_row = block.row / 16;
        let first_column = block.column / 16;
        let last_row = block
            .row
            .checked_add(u32::from(block.height_mi).saturating_sub(1))
            .ok_or(Error::LimitExceeded)?
            / 16;
        let last_column = block
            .column
            .checked_add(u32::from(block.width_mi).saturating_sub(1))
            .ok_or(Error::LimitExceeded)?
            / 16;
        for row in first_row..=last_row.min(self.rows - 1) {
            for column in first_column..=last_column.min(self.columns - 1) {
                let index = usize::try_from(row.saturating_mul(self.columns) + column)
                    .map_err(|_| Error::LimitExceeded)?;
                self.cells[index] = Some(value);
            }
        }
        Ok(())
    }

    fn index(&self, mi_row: u32, mi_column: u32) -> Option<usize> {
        let row = mi_row / 16;
        let column = mi_column / 16;
        if row >= self.rows || column >= self.columns {
            return None;
        }
        usize::try_from(row.saturating_mul(self.columns) + column).ok()
    }
}

impl SegmentPredictionContexts {
    pub fn new(columns: u32, rows: u32) -> Result<Self, Error> {
        if columns == 0 || rows == 0 {
            return Err(Error::InvalidObu);
        }
        let mut above =
            Vector::with_capacity(usize::try_from(columns).map_err(|_| Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?;
        let mut left =
            Vector::with_capacity(usize::try_from(rows).map_err(|_| Error::LimitExceeded)?)
                .map_err(|_| Error::LimitExceeded)?;
        above
            .try_resize(
                usize::try_from(columns).map_err(|_| Error::LimitExceeded)?,
                0,
            )
            .map_err(|_| Error::LimitExceeded)?;
        left.try_resize(usize::try_from(rows).map_err(|_| Error::LimitExceeded)?, 0)
            .map_err(|_| Error::LimitExceeded)?;
        Ok(Self { above, left })
    }

    pub fn context(&self, block: BlockRect) -> Result<u8, Error> {
        let above = *self
            .above
            .get(usize::try_from(block.column).map_err(|_| Error::LimitExceeded)?)
            .ok_or(Error::InvalidObu)?;
        let left = *self
            .left
            .get(usize::try_from(block.row).map_err(|_| Error::LimitExceeded)?)
            .ok_or(Error::InvalidObu)?;
        above.checked_add(left).ok_or(Error::LimitExceeded)
    }

    pub fn update(&mut self, block: BlockRect, predicted: bool) -> Result<(), Error> {
        let value = u8::from(predicted);
        let start_column = usize::try_from(block.column).map_err(|_| Error::LimitExceeded)?;
        let end_column = start_column
            .checked_add(usize::from(block.width_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.above.len());
        let start_row = usize::try_from(block.row).map_err(|_| Error::LimitExceeded)?;
        let end_row = start_row
            .checked_add(usize::from(block.height_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.left.len());
        if start_column >= end_column || start_row >= end_row {
            return Err(Error::InvalidObu);
        }
        self.above[start_column..end_column].fill(value);
        self.left[start_row..end_row].fill(value);
        Ok(())
    }
}

impl MiGrid {
    pub fn new(columns: u32, rows: u32) -> Result<Self, Error> {
        if columns == 0 || rows == 0 {
            return Err(Error::InvalidObu);
        }
        let count = usize::try_from(columns.checked_mul(rows).ok_or(Error::LimitExceeded)?)
            .map_err(|_| Error::LimitExceeded)?;
        let mut cells = Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
        for _ in 0..count {
            cells
                .try_push(BlockState::default())
                .map_err(|_| Error::LimitExceeded)?;
        }
        Ok(Self {
            columns,
            rows,
            cells,
        })
    }

    pub fn columns(&self) -> u32 {
        self.columns
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn get(&self, row: u32, column: u32) -> Option<&BlockState> {
        self.index(row, column)
            .and_then(|index| self.cells.get(index))
    }

    pub fn minimum_segment_id(&self, block: BlockRect) -> Result<u8, Error> {
        if block.column >= self.columns || block.row >= self.rows {
            return Err(Error::InvalidObu);
        }
        let end_column = block
            .column
            .checked_add(u32::from(block.width_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.columns);
        let end_row = block
            .row
            .checked_add(u32::from(block.height_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.rows);
        let mut segment = 7;
        for row in block.row..end_row {
            for column in block.column..end_column {
                segment = segment.min(self.get(row, column).ok_or(Error::InvalidObu)?.segment_id);
            }
        }
        Ok(segment)
    }

    /// Writes state to every in-frame 4x4 covered by a decoded block.
    pub fn fill(&mut self, block: BlockRect, state: BlockState) -> Result<(), Error> {
        if block.width_mi == 0
            || block.height_mi == 0
            || block.column >= self.columns
            || block.row >= self.rows
        {
            return Err(Error::InvalidObu);
        }
        let end_column = block
            .column
            .checked_add(u32::from(block.width_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.columns);
        let end_row = block
            .row
            .checked_add(u32::from(block.height_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.rows);
        for row in block.row..end_row {
            for column in block.column..end_column {
                let index = self.index(row, column).ok_or(Error::InvalidObu)?;
                self.cells[index] = state;
            }
        }
        Ok(())
    }

    pub fn fill_preserving_tx_size(
        &mut self,
        block: BlockRect,
        state: BlockState,
    ) -> Result<(), Error> {
        let end_column = block
            .column
            .checked_add(u32::from(block.width_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.columns);
        let end_row = block
            .row
            .checked_add(u32::from(block.height_mi))
            .ok_or(Error::LimitExceeded)?
            .min(self.rows);
        for row in block.row..end_row {
            for column in block.column..end_column {
                let index = self.index(row, column).ok_or(Error::InvalidObu)?;
                let tx_size = self.cells[index].tx_size;
                self.cells[index] = BlockState { tx_size, ..state };
            }
        }
        Ok(())
    }

    /// Records an inter transform size over its clipped 4x4 footprint.
    pub fn fill_tx_size(&mut self, row: u32, column: u32, size: TxSize) -> Result<(), Error> {
        let (width, height) = size.dimensions();
        if row >= self.rows || column >= self.columns {
            return Err(Error::InvalidObu);
        }
        let end_row = row.saturating_add(u32::from(height / 4)).min(self.rows);
        let end_column = column
            .saturating_add(u32::from(width / 4))
            .min(self.columns);
        for y in row..end_row {
            for x in column..end_column {
                let index = self.index(y, x).ok_or(Error::InvalidObu)?;
                self.cells[index].tx_size = Some(size);
            }
        }
        Ok(())
    }

    pub fn fill_tx_type(
        &mut self,
        row: u32,
        column: u32,
        size: TxSize,
        tx_type: TxType,
    ) -> Result<(), Error> {
        let (width, height) = size.dimensions();
        let end_row = row.saturating_add(u32::from(height / 4)).min(self.rows);
        let end_column = column
            .saturating_add(u32::from(width / 4))
            .min(self.columns);
        if row >= end_row || column >= end_column {
            return Err(Error::InvalidObu);
        }
        for y in row..end_row {
            for x in column..end_column {
                let index = self.index(y, x).ok_or(Error::InvalidObu)?;
                self.cells[index].tx_type = tx_type;
            }
        }
        Ok(())
    }

    pub fn fill_loop_filter_tx_size(
        &mut self,
        plane: usize,
        plane_row: u32,
        plane_column: u32,
        size: TxSize,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Result<(), Error> {
        if plane >= 3 {
            return Err(Error::InvalidObu);
        }
        let sub_x = u32::from(plane != 0 && subsampling_x);
        let sub_y = u32::from(plane != 0 && subsampling_y);
        let (width, height) = size.dimensions();
        let row = plane_row << sub_y;
        let column = plane_column << sub_x;
        let end_row = row
            .saturating_add(u32::from(height / 4) << sub_y)
            .min(self.rows);
        let end_column = column
            .saturating_add(u32::from(width / 4) << sub_x)
            .min(self.columns);
        if row >= end_row || column >= end_column {
            return Err(Error::InvalidObu);
        }
        for y in row..end_row {
            for x in column..end_column {
                let index = self.index(y, x).ok_or(Error::InvalidObu)?;
                self.cells[index].loop_filter_tx_sizes[plane] = size;
            }
        }
        Ok(())
    }

    pub fn inter_tx_neighbor_dimensions(
        &self,
        row: u32,
        column: u32,
        block: BlockRect,
        tile: TileBounds,
    ) -> (u16, u16) {
        let above = if row == block.row {
            tile.contains(i64::from(row) - 1, i64::from(column))
                .then(|| self.get(row - 1, column))
                .flatten()
                .map_or(0, |state| {
                    if state.is_inter {
                        state.size.map_or(0, |size| size.dimensions().0)
                    } else {
                        state
                            .tx_size
                            .map_or(0, |size| u16::from(size.dimensions().0))
                    }
                })
        } else {
            row.checked_sub(1)
                .and_then(|y| self.get(y, column))
                .and_then(|state| state.tx_size)
                .map_or(0, |size| u16::from(size.dimensions().0))
        };
        let left = if column == block.column {
            tile.contains(i64::from(row), i64::from(column) - 1)
                .then(|| self.get(row, column - 1))
                .flatten()
                .map_or(0, |state| {
                    if state.is_inter {
                        state.size.map_or(0, |size| size.dimensions().1)
                    } else {
                        state
                            .tx_size
                            .map_or(0, |size| u16::from(size.dimensions().1))
                    }
                })
        } else {
            column
                .checked_sub(1)
                .and_then(|x| self.get(row, x))
                .and_then(|state| state.tx_size)
                .map_or(0, |size| u16::from(size.dimensions().1))
        };
        (above, left)
    }

    pub fn partition_context(&self, block: BlockRect, size: BlockSize) -> Result<u8, Error> {
        self.partition_context_with_availability(block, size, true, true)
    }

    pub fn partition_context_with_availability(
        &self,
        block: BlockRect,
        size: BlockSize,
        upper_available: bool,
        left_available: bool,
    ) -> Result<u8, Error> {
        let above = upper_available
            .then(|| block.row.checked_sub(1))
            .flatten()
            .and_then(|row| self.get(row, block.column))
            .and_then(|state| state.size);
        let left = left_available
            .then(|| block.column.checked_sub(1))
            .flatten()
            .and_then(|column| self.get(block.row, column))
            .and_then(|state| state.size);
        partition_context(size, above, left)
    }

    pub fn skip_context(&self, block: BlockRect, tile: TileBounds) -> u8 {
        self.binary_neighbor_context(block, tile, |state| state.skip)
    }

    pub fn skip_mode_context(&self, block: BlockRect, tile: TileBounds) -> u8 {
        self.binary_neighbor_context(block, tile, |state| state.skip_mode)
    }

    pub fn is_inter_context(&self, block: BlockRect, tile: TileBounds) -> u8 {
        let above = tile
            .contains(i64::from(block.row) - 1, i64::from(block.column))
            .then(|| block.row.checked_sub(1))
            .flatten()
            .and_then(|row| self.get(row, block.column));
        let left = tile
            .contains(i64::from(block.row), i64::from(block.column) - 1)
            .then(|| block.column.checked_sub(1))
            .flatten()
            .and_then(|column| self.get(block.row, column));
        match (above, left) {
            (Some(above), Some(left)) => {
                let above_intra = !above.is_inter;
                let left_intra = !left.is_inter;
                if above_intra && left_intra {
                    3
                } else {
                    u8::from(above_intra || left_intra)
                }
            }
            (Some(state), None) | (None, Some(state)) => 2 * u8::from(!state.is_inter),
            (None, None) => 0,
        }
    }

    /// Returns `(predicted_segment_id, segment_id_cdf_context)`.
    pub fn segment_prediction(&self, block: BlockRect, tile: TileBounds) -> (u8, u8) {
        let upper_available = tile.contains(i64::from(block.row) - 1, i64::from(block.column));
        let left_available = tile.contains(i64::from(block.row), i64::from(block.column) - 1);
        let upper = upper_available
            .then(|| block.row.checked_sub(1))
            .flatten()
            .and_then(|row| self.get(row, block.column))
            .map(|state| state.segment_id);
        let left = left_available
            .then(|| block.column.checked_sub(1))
            .flatten()
            .and_then(|column| self.get(block.row, column))
            .map(|state| state.segment_id);
        let upper_left = (upper_available && left_available)
            .then(|| Some((block.row.checked_sub(1)?, block.column.checked_sub(1)?)))
            .flatten()
            .and_then(|(row, column)| self.get(row, column))
            .map(|state| state.segment_id);
        let prediction = match (upper, left) {
            (None, None) => 0,
            (Some(value), None) | (None, Some(value)) => value,
            (Some(upper), Some(left)) => {
                if upper_left == Some(upper) {
                    upper
                } else {
                    left
                }
            }
        };
        let context = match (upper_left, upper, left) {
            (Some(upper_left), Some(upper), Some(left)) if upper_left == upper && upper == left => {
                2
            }
            (Some(upper_left), Some(upper), Some(left))
                if upper_left == upper || upper_left == left || upper == left =>
            {
                1
            }
            _ => 0,
        };
        (prediction, context)
    }

    pub fn tx_depth_context(&self, block: BlockRect, tile: TileBounds, maximum: TxSize) -> u8 {
        let above = tile
            .contains(i64::from(block.row) - 1, i64::from(block.column))
            .then(|| block.row.checked_sub(1))
            .flatten()
            .and_then(|row| self.get(row, block.column));
        let left = tile
            .contains(i64::from(block.row), i64::from(block.column) - 1)
            .then(|| block.column.checked_sub(1))
            .flatten()
            .and_then(|column| self.get(block.row, column));
        let above_width = above.map_or(0, |state| {
            if state.is_inter {
                state.size.map_or(0, |size| size.dimensions().0)
            } else {
                state
                    .tx_size
                    .map_or(0, |size| u16::from(size.dimensions().0))
            }
        });
        let left_height = left.map_or(0, |state| {
            if state.is_inter {
                state.size.map_or(0, |size| size.dimensions().1)
            } else {
                state
                    .tx_size
                    .map_or(0, |size| u16::from(size.dimensions().1))
            }
        });
        let (maximum_width, maximum_height) = maximum.dimensions();
        u8::from(above_width >= u16::from(maximum_width))
            + u8::from(left_height >= u16::from(maximum_height))
    }

    fn binary_neighbor_context(
        &self,
        block: BlockRect,
        tile: TileBounds,
        value: impl Fn(&BlockState) -> bool,
    ) -> u8 {
        let above = tile
            .contains(i64::from(block.row) - 1, i64::from(block.column))
            .then(|| block.row.checked_sub(1))
            .flatten()
            .and_then(|row| self.get(row, block.column))
            .is_some_and(&value);
        let left = tile
            .contains(i64::from(block.row), i64::from(block.column) - 1)
            .then(|| block.column.checked_sub(1))
            .flatten()
            .and_then(|column| self.get(block.row, column))
            .is_some_and(value);
        u8::from(above) + u8::from(left)
    }

    fn index(&self, row: u32, column: u32) -> Option<usize> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        usize::try_from(row.checked_mul(self.columns)?.checked_add(column)?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_at_frame_edges_are_clipped_to_allocated_mi_cells() {
        let mut grid = MiGrid::new(5, 3).unwrap();
        let state = BlockState {
            size: Some(BlockSize::Block16x16),
            segment_id: 3,
            skip: true,
            skip_mode: false,
            is_inter: true,
            ..BlockState::default()
        };
        grid.fill(BlockRect::new(4, 2, BlockSize::Block16x16), state)
            .unwrap();
        assert_eq!(grid.get(2, 4), Some(&state));
        assert_eq!(grid.get(2, 3), Some(&BlockState::default()));
    }

    #[test]
    fn stored_neighbor_sizes_drive_partition_context() {
        let mut grid = MiGrid::new(32, 32).unwrap();
        grid.fill(
            BlockRect::new(8, 7, BlockSize::Block32x16),
            BlockState {
                size: Some(BlockSize::Block32x16),
                ..BlockState::default()
            },
        )
        .unwrap();
        grid.fill(
            BlockRect::new(7, 8, BlockSize::Block16x32),
            BlockState {
                size: Some(BlockSize::Block16x32),
                ..BlockState::default()
            },
        )
        .unwrap();
        assert_eq!(
            grid.partition_context(
                BlockRect::new(8, 8, BlockSize::Block64x64),
                BlockSize::Block64x64,
            ),
            Ok(3)
        );
    }

    #[test]
    fn variable_tx_context_uses_inter_block_size_even_when_not_skipped() {
        let mut grid = MiGrid::new(16, 16).unwrap();
        grid.fill(
            BlockRect::new(0, 0, BlockSize::Block16x16),
            BlockState {
                size: Some(BlockSize::Block16x16),
                tx_size: Some(TxSize::Tx4x4),
                is_inter: true,
                skip: false,
                ..BlockState::default()
            },
        )
        .unwrap();
        let tile = TileBounds {
            column_start: 0,
            column_end: 16,
            row_start: 0,
            row_end: 16,
        };
        assert_eq!(
            grid.inter_tx_neighbor_dimensions(
                4,
                0,
                BlockRect::new(0, 4, BlockSize::Block16x16),
                tile,
            ),
            (16, 0)
        );
    }

    #[test]
    fn mode_contexts_ignore_neighbors_across_tile_boundaries() {
        let mut grid = MiGrid::new(4, 4).unwrap();
        let marked = BlockState {
            skip: true,
            skip_mode: true,
            is_inter: false,
            ..BlockState::default()
        };
        grid.fill(BlockRect::new(0, 0, BlockSize::Block8x8), marked)
            .unwrap();
        let block = BlockRect::new(2, 2, BlockSize::Block8x8);
        let isolated = TileBounds {
            column_start: 2,
            column_end: 4,
            row_start: 2,
            row_end: 4,
        };
        assert_eq!(grid.skip_context(block, isolated), 0);
        assert_eq!(grid.skip_mode_context(block, isolated), 0);
        assert_eq!(grid.is_inter_context(block, isolated), 0);
    }

    #[test]
    fn segment_prediction_uses_upper_left_tie_rule() {
        let mut grid = MiGrid::new(4, 4).unwrap();
        for (column, row, segment_id) in [(0, 0, 2), (1, 0, 2), (0, 1, 5)] {
            grid.fill(
                BlockRect::new(column, row, BlockSize::Block4x4),
                BlockState {
                    segment_id,
                    ..BlockState::default()
                },
            )
            .unwrap();
        }
        let tile = TileBounds {
            column_start: 0,
            column_end: 4,
            row_start: 0,
            row_end: 4,
        };
        assert_eq!(
            grid.segment_prediction(BlockRect::new(1, 1, BlockSize::Block4x4), tile),
            (2, 1)
        );
    }

    #[test]
    fn previous_segment_minimum_and_prediction_contexts_cover_block_extent() {
        let mut grid = MiGrid::new(4, 4).unwrap();
        grid.fill(
            BlockRect::new(0, 0, BlockSize::Block8x8),
            BlockState {
                segment_id: 5,
                ..BlockState::default()
            },
        )
        .unwrap();
        grid.fill(
            BlockRect::new(1, 1, BlockSize::Block4x4),
            BlockState {
                segment_id: 2,
                ..BlockState::default()
            },
        )
        .unwrap();
        assert_eq!(
            grid.minimum_segment_id(BlockRect::new(0, 0, BlockSize::Block8x8)),
            Ok(2)
        );

        let mut contexts = SegmentPredictionContexts::new(4, 4).unwrap();
        let block = BlockRect::new(1, 1, BlockSize::Block8x8);
        assert_eq!(contexts.context(block), Ok(0));
        contexts.update(block, true).unwrap();
        assert_eq!(contexts.context(block), Ok(2));
        assert_eq!(
            contexts.context(BlockRect::new(4, 0, BlockSize::Block4x4)),
            Err(Error::InvalidObu)
        );
    }
}
