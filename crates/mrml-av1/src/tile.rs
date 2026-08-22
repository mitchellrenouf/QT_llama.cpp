//! Tile layout and tile-group parsing (AV1 sections 5.9.15 and 5.9.16).

use crate::{
    Bits, Error,
    block_state::MiGrid,
    cdf::TileCdfs,
    entropy::SymbolDecoder,
    partition::{BlockRect, BlockSize, PartitionCdfProvider, TileBounds, decode_partition_tree},
};
use mrml_runtime::Vector;

const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const MAX_TILE_COLS: u32 = 64;
const MAX_TILE_ROWS: u32 = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileLayout {
    pub column_starts_sb: Vector<u32>,
    pub row_starts_sb: Vector<u32>,
    pub context_update_tile_id: u16,
    pub tile_size_bytes: u8,
}

/// Decodes every accumulated tile from the same initial CDF state and returns
/// the final state of `context_update_tile_id`, as required by section 7.4.
pub struct AccumulatedTileDecodeConfig<'a> {
    pub layout: &'a TileLayout,
    pub mi_columns: u32,
    pub mi_rows: u32,
    pub use_128x128: bool,
    pub disable_cdf_update: bool,
    pub initial_cdfs: &'a TileCdfs,
}

pub fn decode_accumulated_partition_trees<S, F>(
    accumulator: &TileAccumulator,
    config: AccumulatedTileDecodeConfig<'_>,
    grid: &mut MiGrid,
    mut decode_superblock_prefix: S,
    mut decode_block: F,
) -> Result<TileCdfs, Error>
where
    S: FnMut(
        usize,
        TileBounds,
        &mut SymbolDecoder<'_>,
        &mut TileCdfs,
        BlockRect,
    ) -> Result<(), Error>,
    F: FnMut(
        usize,
        TileBounds,
        &mut SymbolDecoder<'_>,
        &mut TileCdfs,
        &mut MiGrid,
        BlockRect,
        BlockSize,
    ) -> Result<(), Error>,
{
    let tiles = accumulator.tiles()?;
    if tiles.len() != config.layout.tile_count() {
        return Err(Error::InvalidObu);
    }
    let update_tile = usize::from(config.layout.context_update_tile_id);
    let mut update_cdfs = config.initial_cdfs.clone();
    for (tile_number, tile) in tiles.iter().enumerate() {
        let data = tile.as_ref().ok_or(Error::InvalidObu)?;
        let bounds = config.layout.bounds(
            tile_number,
            config.mi_columns,
            config.mi_rows,
            config.use_128x128,
        )?;
        let mut tile_cdfs = config.initial_cdfs.clone();
        decode_tile_partition_trees(
            data,
            bounds,
            config.use_128x128,
            config.disable_cdf_update,
            &mut tile_cdfs,
            grid,
            |decoder, cdfs, root| {
                decode_superblock_prefix(tile_number, bounds, decoder, cdfs, root)
            },
            |decoder, cdfs, grid, block, size| {
                decode_block(tile_number, bounds, decoder, cdfs, grid, block, size)
            },
        )?;
        if tile_number == update_tile {
            update_cdfs = tile_cdfs;
        }
    }
    Ok(update_cdfs)
}

impl TileLayout {
    pub fn columns(&self) -> usize {
        self.column_starts_sb.len() - 1
    }
    pub fn rows(&self) -> usize {
        self.row_starts_sb.len() - 1
    }
    pub fn tile_count(&self) -> usize {
        self.columns() * self.rows()
    }

    /// Returns this tile's decoding bounds in 4x4 units.
    pub fn bounds(
        &self,
        tile_number: usize,
        mi_cols: u32,
        mi_rows: u32,
        use_128x128: bool,
    ) -> Result<TileBounds, Error> {
        if tile_number >= self.tile_count() || mi_cols == 0 || mi_rows == 0 {
            return Err(Error::InvalidObu);
        }
        let columns = self.columns();
        let row = tile_number / columns;
        let column = tile_number % columns;
        let sb_mi = if use_128x128 { 32 } else { 16 };
        Ok(TileBounds {
            column_start: self.column_starts_sb[column]
                .checked_mul(sb_mi)
                .ok_or(Error::LimitExceeded)?
                .min(mi_cols),
            column_end: self.column_starts_sb[column + 1]
                .checked_mul(sb_mi)
                .ok_or(Error::LimitExceeded)?
                .min(mi_cols),
            row_start: self.row_starts_sb[row]
                .checked_mul(sb_mi)
                .ok_or(Error::LimitExceeded)?
                .min(mi_rows),
            row_end: self.row_starts_sb[row + 1]
                .checked_mul(sb_mi)
                .ok_or(Error::LimitExceeded)?
                .min(mi_rows),
        })
    }

    /// Parse a tile-info bit string. `mi_cols` and `mi_rows` are frame dimensions
    /// in 4x4 units. The returned bit count allows a frame-header parser to
    /// continue at the exact following syntax element.
    pub fn parse_header(
        data: &[u8],
        mi_cols: u32,
        mi_rows: u32,
        use_128x128: bool,
    ) -> Result<(Self, usize), Error> {
        let mut bits = Bits::new(data);
        let layout = Self::parse(&mut bits, mi_cols, mi_rows, use_128x128)?;
        Ok((layout, bits.position()))
    }

    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        mi_cols: u32,
        mi_rows: u32,
        use_128x128: bool,
    ) -> Result<Self, Error> {
        if mi_cols == 0 || mi_rows == 0 {
            return Err(Error::InvalidObu);
        }
        let sb_shift = if use_128x128 { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
        let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
        let max_width_sb = MAX_TILE_WIDTH >> sb_size;
        let max_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_cols = tile_log2(max_width_sb, sb_cols);
        let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles =
            min_log2_cols.max(tile_log2(max_area_sb, sb_rows.saturating_mul(sb_cols)));

        let (columns, rows, log2_cols, log2_rows) = if bits.bit()? {
            let mut log2_cols = min_log2_cols;
            while log2_cols < max_log2_cols && bits.bit()? {
                log2_cols += 1;
            }
            let width = (sb_cols + (1 << log2_cols) - 1) >> log2_cols;
            let columns = uniform_starts(sb_cols, width)?;
            let mut log2_rows = min_log2_tiles.saturating_sub(log2_cols);
            while log2_rows < max_log2_rows && bits.bit()? {
                log2_rows += 1;
            }
            let height = (sb_rows + (1 << log2_rows) - 1) >> log2_rows;
            let rows = uniform_starts(sb_rows, height)?;
            (columns, rows, log2_cols, log2_rows)
        } else {
            let mut columns = Vector::with_capacity(MAX_TILE_COLS as usize + 1)
                .map_err(|_| Error::LimitExceeded)?;
            columns.try_push(0).map_err(|_| Error::LimitExceeded)?;
            let mut widest = 0;
            while *columns.last().unwrap() < sb_cols {
                let start = *columns.last().unwrap();
                let maximum = (sb_cols - start).min(max_width_sb);
                let width = bits.read_ns(maximum)? + 1;
                widest = widest.max(width);
                columns
                    .try_push(start + width)
                    .map_err(|_| Error::LimitExceeded)?;
            }
            let log2_cols = tile_log2(1, columns.len() as u32 - 1);
            let adjusted_area =
                sb_rows.checked_mul(sb_cols).ok_or(Error::LimitExceeded)? >> (min_log2_tiles + 1);
            let max_height = (adjusted_area / widest).max(1);
            let mut rows = Vector::with_capacity(MAX_TILE_ROWS as usize + 1)
                .map_err(|_| Error::LimitExceeded)?;
            rows.try_push(0).map_err(|_| Error::LimitExceeded)?;
            while *rows.last().unwrap() < sb_rows {
                let start = *rows.last().unwrap();
                let maximum = (sb_rows - start).min(max_height);
                let height = bits.read_ns(maximum)? + 1;
                rows.try_push(start + height)
                    .map_err(|_| Error::LimitExceeded)?;
            }
            let log2_rows = tile_log2(1, rows.len() as u32 - 1);
            (columns, rows, log2_cols, log2_rows)
        };
        if columns.len() - 1 > MAX_TILE_COLS as usize || rows.len() - 1 > MAX_TILE_ROWS as usize {
            return Err(Error::InvalidObu);
        }
        let tile_bits = log2_cols + log2_rows;
        let (context_update_tile_id, tile_size_bytes) = if tile_bits > 0 {
            (bits.read(tile_bits as u8)? as u16, bits.read(2)? as u8 + 1)
        } else {
            (0, 0)
        };
        if usize::from(context_update_tile_id) >= (columns.len() - 1) * (rows.len() - 1) {
            return Err(Error::InvalidObu);
        }
        Ok(Self {
            column_starts_sb: columns,
            row_starts_sb: rows,
            context_update_tile_id,
            tile_size_bytes,
        })
    }

    pub fn parse_group<'a>(&self, payload: &'a [u8]) -> Result<TileGroup<'a>, Error> {
        let count = self.tile_count();
        if count == 0 {
            return Err(Error::InvalidObu);
        }
        let tile_bits = ceil_log2(count as u32) as u8;
        let mut bits = Bits::new(payload);
        let (start, end) = if count > 1 && bits.bit()? {
            (
                bits.read(tile_bits)? as usize,
                bits.read(tile_bits)? as usize,
            )
        } else {
            (0, count - 1)
        };
        if start > end || end >= count {
            return Err(Error::InvalidObu);
        }
        bits.align_zero()?;
        let mut position = bits.position() / 8;
        let mut tiles = Vector::with_capacity(end - start + 1).map_err(|_| Error::LimitExceeded)?;
        for number in start..=end {
            let size = if number == end {
                payload
                    .len()
                    .checked_sub(position)
                    .ok_or(Error::Truncated)?
            } else {
                if self.tile_size_bytes == 0 {
                    return Err(Error::InvalidObu);
                }
                let size_end = position
                    .checked_add(self.tile_size_bytes as usize)
                    .ok_or(Error::LimitExceeded)?;
                let encoded = payload.get(position..size_end).ok_or(Error::Truncated)?;
                position = size_end;
                let minus_one = encoded
                    .iter()
                    .enumerate()
                    .fold(0usize, |value, (shift, byte)| {
                        value | (usize::from(*byte) << (shift * 8))
                    });
                minus_one.checked_add(1).ok_or(Error::LimitExceeded)?
            };
            let tile_end = position.checked_add(size).ok_or(Error::LimitExceeded)?;
            let data = payload.get(position..tile_end).ok_or(Error::Truncated)?;
            tiles
                .try_push(Tile {
                    number: number as u16,
                    data,
                })
                .map_err(|_| Error::LimitExceeded)?;
            position = tile_end;
        }
        if position != payload.len() {
            return Err(Error::InvalidObu);
        }
        Ok(TileGroup {
            start: start as u16,
            end: end as u16,
            tiles,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tile<'a> {
    pub number: u16,
    pub data: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileGroup<'a> {
    pub start: u16,
    pub end: u16,
    pub tiles: Vector<Tile<'a>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileAccumulator {
    tiles: Vector<Option<Vector<u8>>>,
    next_tile: usize,
}

impl TileAccumulator {
    pub fn new(tile_count: usize) -> Result<Self, Error> {
        if tile_count == 0 || tile_count > (MAX_TILE_COLS * MAX_TILE_ROWS) as usize {
            return Err(Error::InvalidObu);
        }
        let mut tiles = Vector::with_capacity(tile_count).map_err(|_| Error::LimitExceeded)?;
        tiles
            .try_resize(tile_count, None)
            .map_err(|_| Error::LimitExceeded)?;
        Ok(Self {
            tiles,
            next_tile: 0,
        })
    }

    pub fn push(&mut self, group: &TileGroup<'_>) -> Result<bool, Error> {
        if usize::from(group.start) != self.next_tile
            || usize::from(group.end) < usize::from(group.start)
            || group.tiles.len() != usize::from(group.end - group.start) + 1
        {
            return Err(Error::InvalidObu);
        }
        for (offset, tile) in group.tiles.iter().enumerate() {
            let number = self
                .next_tile
                .checked_add(offset)
                .ok_or(Error::LimitExceeded)?;
            if usize::from(tile.number) != number || number >= self.tiles.len() {
                return Err(Error::InvalidObu);
            }
            let mut owned =
                Vector::with_capacity(tile.data.len()).map_err(|_| Error::LimitExceeded)?;
            owned
                .try_extend_from_slice(tile.data)
                .map_err(|_| Error::LimitExceeded)?;
            if self.tiles[number].replace(owned).is_some() {
                return Err(Error::InvalidObu);
            }
        }
        self.next_tile = usize::from(group.end)
            .checked_add(1)
            .ok_or(Error::LimitExceeded)?;
        Ok(self.next_tile == self.tiles.len())
    }

    pub fn is_complete(&self) -> bool {
        self.next_tile == self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.next_tile == 0
    }

    pub fn tiles(&self) -> Result<&[Option<Vector<u8>>], Error> {
        if !self.is_complete() {
            return Err(Error::InvalidObu);
        }
        Ok(&self.tiles)
    }
}

/// Decodes every superblock partition tree in one tile. The block callback is
/// responsible for consuming mode, palette, transform, and residual syntax and
/// recording the resulting state in `grid`.
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_partition_trees<P, S, F>(
    data: &[u8],
    bounds: TileBounds,
    use_128x128: bool,
    disable_cdf_update: bool,
    cdfs: &mut P,
    grid: &mut MiGrid,
    mut decode_superblock_prefix: S,
    mut decode_block: F,
) -> Result<(), Error>
where
    P: PartitionCdfProvider,
    S: FnMut(&mut SymbolDecoder<'_>, &mut P, BlockRect) -> Result<(), Error>,
    F: FnMut(
        &mut SymbolDecoder<'_>,
        &mut P,
        &mut MiGrid,
        BlockRect,
        BlockSize,
    ) -> Result<(), Error>,
{
    if bounds.column_start >= bounds.column_end
        || bounds.row_start >= bounds.row_end
        || bounds.column_end > grid.columns()
        || bounds.row_end > grid.rows()
    {
        return Err(Error::InvalidObu);
    }
    let root_size = if use_128x128 {
        BlockSize::Block128x128
    } else {
        BlockSize::Block64x64
    };
    let root_mi = if use_128x128 { 32u32 } else { 16u32 };
    if !bounds.column_start.is_multiple_of(root_mi) || !bounds.row_start.is_multiple_of(root_mi) {
        return Err(Error::InvalidObu);
    }
    let mut decoder = SymbolDecoder::new(data, disable_cdf_update)?;
    let mut row = bounds.row_start;
    while row < bounds.row_end {
        let mut column = bounds.column_start;
        while column < bounds.column_end {
            let root = BlockRect::new(column, row, root_size);
            decode_superblock_prefix(&mut decoder, cdfs, root)?;
            decode_partition_tree(
                &mut decoder,
                cdfs,
                grid,
                bounds,
                root,
                root_size,
                &mut decode_block,
            )?;
            column = column.checked_add(root_mi).ok_or(Error::LimitExceeded)?;
        }
        row = row.checked_add(root_mi).ok_or(Error::LimitExceeded)?;
    }
    decoder.finish()
}

fn uniform_starts(total: u32, size: u32) -> Result<Vector<u32>, Error> {
    let capacity = total.div_ceil(size) as usize + 1;
    let mut starts = Vector::with_capacity(capacity).map_err(|_| Error::LimitExceeded)?;
    let mut position = 0;
    while position < total {
        starts
            .try_push(position)
            .map_err(|_| Error::LimitExceeded)?;
        position = (position + size).min(total);
    }
    starts.try_push(total).map_err(|_| Error::LimitExceeded)?;
    Ok(starts)
}

fn tile_log2(block_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (u64::from(block_size) << k) < u64::from(target) {
        k += 1;
    }
    k
}

fn ceil_log2(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        32 - (value - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_logarithms_cover_target() {
        assert_eq!(tile_log2(64, 65), 1);
        assert_eq!(tile_log2(1, 64), 6);
        assert_eq!(ceil_log2(5), 3);
    }

    #[test]
    fn parses_single_tile_group() {
        let layout = TileLayout {
            column_starts_sb: [0, 1].into_iter().collect(),
            row_starts_sb: [0, 1].into_iter().collect(),
            context_update_tile_id: 0,
            tile_size_bytes: 0,
        };
        let group = layout.parse_group(&[1, 2, 3]).unwrap();
        assert_eq!(group.start, 0);
        assert_eq!(group.tiles[0].data, &[1, 2, 3]);
    }

    #[test]
    fn tile_group_alignment_rejects_nonzero_padding() {
        let layout = TileLayout {
            column_starts_sb: [0, 1, 2].into_iter().collect(),
            row_starts_sb: [0, 1].into_iter().collect(),
            context_update_tile_id: 0,
            tile_size_bytes: 1,
        };
        assert_eq!(layout.parse_group(&[0b0100_0000]), Err(Error::InvalidObu));
    }

    #[test]
    fn tile_bounds_clip_the_last_superblock_to_frame() {
        let layout = TileLayout {
            column_starts_sb: [0, 1, 2].into_iter().collect(),
            row_starts_sb: [0, 1].into_iter().collect(),
            context_update_tile_id: 0,
            tile_size_bytes: 1,
        };
        assert_eq!(
            layout.bounds(1, 20, 13, false),
            Ok(TileBounds {
                column_start: 16,
                column_end: 20,
                row_start: 0,
                row_end: 13,
            })
        );
    }

    #[test]
    fn tile_accumulator_requires_ordered_complete_nonoverlapping_groups() {
        let first = TileGroup {
            start: 0,
            end: 0,
            tiles: [Tile {
                number: 0,
                data: &[1, 2],
            }]
            .into_iter()
            .collect(),
        };
        let second = TileGroup {
            start: 1,
            end: 1,
            tiles: [Tile {
                number: 1,
                data: &[3, 4],
            }]
            .into_iter()
            .collect(),
        };
        let mut accumulator = TileAccumulator::new(2).unwrap();
        assert_eq!(accumulator.push(&first), Ok(false));
        assert_eq!(accumulator.tiles(), Err(Error::InvalidObu));
        assert_eq!(accumulator.push(&second), Ok(true));
        let tiles = accumulator.tiles().unwrap();
        assert_eq!(tiles[0].as_deref(), Some(&[1, 2][..]));
        assert_eq!(tiles[1].as_deref(), Some(&[3, 4][..]));
        assert_eq!(accumulator.push(&second), Err(Error::InvalidObu));
    }

    #[test]
    fn accumulated_tiles_decode_with_independent_cdfs_and_tile_bounds() {
        let layout = TileLayout {
            column_starts_sb: [0, 1].into_iter().collect(),
            row_starts_sb: [0, 1].into_iter().collect(),
            context_update_tile_id: 0,
            tile_size_bytes: 0,
        };
        let group = TileGroup {
            start: 0,
            end: 0,
            tiles: [Tile {
                number: 0,
                data: &[0x80, 0],
            }]
            .into_iter()
            .collect(),
        };
        let mut accumulator = TileAccumulator::new(1).unwrap();
        assert_eq!(accumulator.push(&group), Ok(true));
        let initial = TileCdfs::default();
        let mut grid = MiGrid::new(1, 1).unwrap();
        let mut tile_mask = 0u8;
        let final_cdfs = decode_accumulated_partition_trees(
            &accumulator,
            AccumulatedTileDecodeConfig {
                layout: &layout,
                mi_columns: 1,
                mi_rows: 1,
                use_128x128: false,
                disable_cdf_update: true,
                initial_cdfs: &initial,
            },
            &mut grid,
            |_tile, _bounds, _decoder, _cdfs, _root| Ok(()),
            |tile, bounds, _decoder, _cdfs, grid, block, size| {
                tile_mask |= 1 << tile;
                assert!(block.column >= bounds.column_start);
                assert!(block.column < bounds.column_end);
                grid.fill(
                    block,
                    crate::block_state::BlockState {
                        size: Some(size),
                        ..crate::block_state::BlockState::default()
                    },
                )
            },
        )
        .unwrap();
        assert_eq!(tile_mask, 0b1);
        assert_eq!(final_cdfs, initial);
    }
}
