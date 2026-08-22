//! Block-size and partition geometry used by coded tile traversal.

use crate::transform::TxSize;
use crate::{Error, block_state::MiGrid, entropy::SymbolDecoder};

const CDF_TOP: u16 = 1 << 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BlockSize {
    Block4x4,
    Block4x8,
    Block8x4,
    Block8x8,
    Block8x16,
    Block16x8,
    Block16x16,
    Block16x32,
    Block32x16,
    Block32x32,
    Block32x64,
    Block64x32,
    Block64x64,
    Block64x128,
    Block128x64,
    Block128x128,
    Block4x16,
    Block16x4,
    Block8x32,
    Block32x8,
    Block16x64,
    Block64x16,
}

impl BlockSize {
    pub const ALL: [Self; 22] = [
        Self::Block4x4,
        Self::Block4x8,
        Self::Block8x4,
        Self::Block8x8,
        Self::Block8x16,
        Self::Block16x8,
        Self::Block16x16,
        Self::Block16x32,
        Self::Block32x16,
        Self::Block32x32,
        Self::Block32x64,
        Self::Block64x32,
        Self::Block64x64,
        Self::Block64x128,
        Self::Block128x64,
        Self::Block128x128,
        Self::Block4x16,
        Self::Block16x4,
        Self::Block8x32,
        Self::Block32x8,
        Self::Block16x64,
        Self::Block64x16,
    ];

    pub const fn dimensions(self) -> (u16, u16) {
        match self {
            Self::Block4x4 => (4, 4),
            Self::Block4x8 => (4, 8),
            Self::Block8x4 => (8, 4),
            Self::Block8x8 => (8, 8),
            Self::Block8x16 => (8, 16),
            Self::Block16x8 => (16, 8),
            Self::Block16x16 => (16, 16),
            Self::Block16x32 => (16, 32),
            Self::Block32x16 => (32, 16),
            Self::Block32x32 => (32, 32),
            Self::Block32x64 => (32, 64),
            Self::Block64x32 => (64, 32),
            Self::Block64x64 => (64, 64),
            Self::Block64x128 => (64, 128),
            Self::Block128x64 => (128, 64),
            Self::Block128x128 => (128, 128),
            Self::Block4x16 => (4, 16),
            Self::Block16x4 => (16, 4),
            Self::Block8x32 => (8, 32),
            Self::Block32x8 => (32, 8),
            Self::Block16x64 => (16, 64),
            Self::Block64x16 => (64, 16),
        }
    }

    pub fn from_dimensions(width: u16, height: u16) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|size| size.dimensions() == (width, height))
            .ok_or(Error::InvalidObu)
    }

    /// Returns `get_plane_residual_size` for the requested chroma sampling.
    pub fn plane_residual_size(
        self,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Result<Self, Error> {
        let (width, height) = self.dimensions();
        Self::from_dimensions(
            (width >> u8::from(subsampling_x)).max(4),
            (height >> u8::from(subsampling_y)).max(4),
        )
    }

    /// Maximum number of transform splits from section 5.11.15.
    pub const fn max_transform_depth(self) -> u8 {
        const DEPTH: [u8; 22] = [
            0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
        ];
        DEPTH[self as usize]
    }

    pub const fn width_log2_mi(self) -> u8 {
        self.dimensions().0.ilog2() as u8 - 2
    }

    pub const fn height_log2_mi(self) -> u8 {
        self.dimensions().1.ilog2() as u8 - 2
    }

    pub const fn maximum_transform_size(self) -> TxSize {
        match self {
            Self::Block4x4 => TxSize::Tx4x4,
            Self::Block4x8 => TxSize::Tx4x8,
            Self::Block8x4 => TxSize::Tx8x4,
            Self::Block8x8 => TxSize::Tx8x8,
            Self::Block8x16 => TxSize::Tx8x16,
            Self::Block16x8 => TxSize::Tx16x8,
            Self::Block16x16 => TxSize::Tx16x16,
            Self::Block16x32 => TxSize::Tx16x32,
            Self::Block32x16 => TxSize::Tx32x16,
            Self::Block32x32 => TxSize::Tx32x32,
            Self::Block32x64 => TxSize::Tx32x64,
            Self::Block64x32 => TxSize::Tx64x32,
            Self::Block64x64 | Self::Block64x128 | Self::Block128x64 | Self::Block128x128 => {
                TxSize::Tx64x64
            }
            Self::Block4x16 => TxSize::Tx4x16,
            Self::Block16x4 => TxSize::Tx16x4,
            Self::Block8x32 => TxSize::Tx8x32,
            Self::Block32x8 => TxSize::Tx32x8,
            Self::Block16x64 => TxSize::Tx16x64,
            Self::Block64x16 => TxSize::Tx64x16,
        }
    }

    pub const fn size_group(self) -> u8 {
        const GROUP: [u8; 22] = [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
        ];
        GROUP[self as usize]
    }
}

/// Selects one of the four partition CDF contexts from the upper and left
/// decoded block sizes (section 9.3).
pub fn partition_context(
    block_size: BlockSize,
    above: Option<BlockSize>,
    left: Option<BlockSize>,
) -> Result<u8, Error> {
    let (width, height) = block_size.dimensions();
    if width != height || width < 8 {
        return Err(Error::InvalidObu);
    }
    let level = block_size.width_log2_mi();
    let above_smaller = above.is_some_and(|size| size.width_log2_mi() < level);
    let left_smaller = left.is_some_and(|size| size.height_log2_mi() < level);
    Ok(u8::from(left_smaller) * 2 + u8::from(above_smaller))
}

/// Reconstructs a spatially predicted segment id (section 5.11.9).
pub fn negative_deinterleave(diff: u8, reference: u8, maximum: u8) -> Result<u8, Error> {
    if maximum == 0 || diff >= maximum || reference >= maximum {
        return Err(Error::InvalidObu);
    }
    let diff = u16::from(diff);
    let reference = u16::from(reference);
    let maximum = u16::from(maximum);
    let value = if reference == 0 {
        diff
    } else if reference >= maximum - 1 {
        maximum - diff - 1
    } else if 2 * reference < maximum {
        if diff <= 2 * reference {
            if diff & 1 != 0 {
                reference + ((diff + 1) >> 1)
            } else {
                reference - (diff >> 1)
            }
        } else {
            diff
        }
    } else if diff <= 2 * (maximum - reference - 1) {
        if diff & 1 != 0 {
            reference + ((diff + 1) >> 1)
        } else {
            reference - (diff >> 1)
        }
    } else {
        maximum - (diff + 1)
    };
    u8::try_from(value).map_err(|_| Error::InvalidObu)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Partition {
    None,
    Horizontal,
    Vertical,
    Split,
    HorizontalA,
    HorizontalB,
    VerticalA,
    VerticalB,
    Horizontal4,
    Vertical4,
}

impl Partition {
    fn from_symbol(symbol: usize) -> Result<Self, Error> {
        match symbol {
            0 => Ok(Self::None),
            1 => Ok(Self::Horizontal),
            2 => Ok(Self::Vertical),
            3 => Ok(Self::Split),
            4 => Ok(Self::HorizontalA),
            5 => Ok(Self::HorizontalB),
            6 => Ok(Self::VerticalA),
            7 => Ok(Self::VerticalB),
            8 => Ok(Self::Horizontal4),
            9 => Ok(Self::Vertical4),
            _ => Err(Error::InvalidObu),
        }
    }
}

/// Decodes the partition syntax including the reduced alphabets at frame edges.
pub fn decode_partition(
    decoder: &mut SymbolDecoder<'_>,
    partition_cdf: &mut [u16],
    block_size: BlockSize,
    has_rows: bool,
    has_columns: bool,
) -> Result<Partition, Error> {
    let (width, height) = block_size.dimensions();
    if width != height {
        return Err(Error::InvalidObu);
    }
    if width < 8 {
        return Ok(Partition::None);
    }
    if has_rows && has_columns {
        let partition = Partition::from_symbol(decoder.read_symbol(partition_cdf)?)?;
        if width == 128 && matches!(partition, Partition::Horizontal4 | Partition::Vertical4) {
            return Err(Error::InvalidObu);
        }
        return Ok(partition);
    }
    if !has_rows && !has_columns {
        return Ok(Partition::Split);
    }
    let mut restricted = restricted_partition_cdf(partition_cdf, block_size, !has_rows)?;
    let split = decoder.read_symbol(&mut restricted)? != 0;
    if split {
        Ok(Partition::Split)
    } else if has_columns {
        Ok(Partition::Horizontal)
    } else {
        Ok(Partition::Vertical)
    }
}

/// Constructs the two-symbol CDF used when a square overlaps one frame edge.
/// `bottom_edge` selects `split_or_horz`; otherwise `split_or_vert` is used.
pub fn restricted_partition_cdf(
    partition_cdf: &[u16],
    block_size: BlockSize,
    bottom_edge: bool,
) -> Result<[u16; 3], Error> {
    let (width, height) = block_size.dimensions();
    if width != height || width < 16 || partition_cdf.len() < 9 {
        return Err(Error::InvalidObu);
    }
    let symbols = partition_cdf.len() - 1;
    if partition_cdf[symbols - 1] != CDF_TOP || partition_cdf[symbols] > 32 {
        return Err(Error::InvalidObu);
    }
    let mass = |symbol: usize| -> Result<u32, Error> {
        if symbol >= symbols {
            return Err(Error::InvalidObu);
        }
        let lower = if symbol == 0 {
            0
        } else {
            partition_cdf[symbol - 1]
        };
        partition_cdf[symbol]
            .checked_sub(lower)
            .map(u32::from)
            .ok_or(Error::InvalidObu)
    };
    let selected = if bottom_edge {
        [2usize, 3, 4, 6, 7]
    } else {
        [1usize, 3, 4, 5, 6]
    };
    let mut split_mass = 0u32;
    for symbol in selected {
        split_mass = split_mass
            .checked_add(mass(symbol)?)
            .ok_or(Error::LimitExceeded)?;
    }
    if width != 128 {
        split_mass = split_mass
            .checked_add(mass(if bottom_edge { 9 } else { 8 })?)
            .ok_or(Error::LimitExceeded)?;
    }
    if split_mass > u32::from(CDF_TOP) {
        return Err(Error::InvalidObu);
    }
    Ok([
        CDF_TOP - u16::try_from(split_mass).map_err(|_| Error::InvalidObu)?,
        CDF_TOP,
        0,
    ])
}

pub trait PartitionCdfProvider {
    fn cdf(&mut self, block_size: BlockSize, context: u8) -> Result<&mut [u16], Error>;
}

/// Recursively decodes one square partition tree and invokes `decode_block`
/// for each in-frame leaf in normative raster order.
pub fn decode_partition_tree<P, F>(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut P,
    grid: &mut MiGrid,
    tile_bounds: TileBounds,
    root: BlockRect,
    root_size: BlockSize,
    mut decode_block: F,
) -> Result<(), Error>
where
    P: PartitionCdfProvider,
    F: FnMut(
        &mut SymbolDecoder<'_>,
        &mut P,
        &mut MiGrid,
        BlockRect,
        BlockSize,
    ) -> Result<(), Error>,
{
    fn recurse<P, F>(
        decoder: &mut SymbolDecoder<'_>,
        cdfs: &mut P,
        grid: &mut MiGrid,
        tile_bounds: TileBounds,
        block: BlockRect,
        size: BlockSize,
        decode_block: &mut F,
    ) -> Result<(), Error>
    where
        P: PartitionCdfProvider,
        F: FnMut(
            &mut SymbolDecoder<'_>,
            &mut P,
            &mut MiGrid,
            BlockRect,
            BlockSize,
        ) -> Result<(), Error>,
    {
        if block.row >= grid.rows() || block.column >= grid.columns() {
            return Ok(());
        }
        let (width, height) = size.dimensions();
        if width != height
            || block.width_mi != (width / 4) as u8
            || block.height_mi != (height / 4) as u8
        {
            return Err(Error::InvalidObu);
        }
        if width < 8 {
            return decode_block(decoder, cdfs, grid, block, size);
        }
        let half = u32::from(block.width_mi / 2);
        let has_rows = block
            .row
            .checked_add(half)
            .is_some_and(|row| row < grid.rows());
        let has_columns = block
            .column
            .checked_add(half)
            .is_some_and(|column| column < grid.columns());
        let context = grid.partition_context_with_availability(
            block,
            size,
            tile_bounds.contains(i64::from(block.row) - 1, i64::from(block.column)),
            tile_bounds.contains(i64::from(block.row), i64::from(block.column) - 1),
        )?;
        let partition = decode_partition(
            decoder,
            cdfs.cdf(size, context)?,
            size,
            has_rows,
            has_columns,
        )
        .map_err(|error| match error {
            Error::InvalidObu => Error::InvalidPartitionSyntax,
            other => other,
        })?;
        let children = block.partition(partition)?;
        for child in children.iter() {
            if child.row >= grid.rows() || child.column >= grid.columns() {
                continue;
            }
            let child_size = BlockSize::from_dimensions(
                u16::from(child.width_mi) * 4,
                u16::from(child.height_mi) * 4,
            )?;
            if partition == Partition::Split {
                recurse(
                    decoder,
                    cdfs,
                    grid,
                    tile_bounds,
                    child,
                    child_size,
                    decode_block,
                )?;
            } else {
                decode_block(decoder, cdfs, grid, child, child_size)?;
            }
        }
        Ok(())
    }

    recurse(
        decoder,
        cdfs,
        grid,
        tile_bounds,
        root,
        root_size,
        &mut decode_block,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockRect {
    /// Horizontal position in 4x4 units.
    pub column: u32,
    /// Vertical position in 4x4 units.
    pub row: u32,
    pub width_mi: u8,
    pub height_mi: u8,
}

impl BlockRect {
    pub fn new(column: u32, row: u32, size: BlockSize) -> Self {
        let (width, height) = size.dimensions();
        Self {
            column,
            row,
            width_mi: (width / 4) as u8,
            height_mi: (height / 4) as u8,
        }
    }

    pub fn partition(self, kind: Partition) -> Result<PartitionChildren, Error> {
        let w = self.width_mi;
        let h = self.height_mi;
        let half_w = w / 2;
        let half_h = h / 2;
        let rect = |column, row, width_mi, height_mi| BlockRect {
            column,
            row,
            width_mi,
            height_mi,
        };
        let children = match kind {
            Partition::None => [Some(self), None, None, None],
            Partition::Horizontal if h >= 2 && h.is_multiple_of(2) => [
                Some(rect(self.column, self.row, w, half_h)),
                Some(rect(self.column, self.row + u32::from(half_h), w, half_h)),
                None,
                None,
            ],
            Partition::Vertical if w >= 2 && w.is_multiple_of(2) => [
                Some(rect(self.column, self.row, half_w, h)),
                Some(rect(self.column + u32::from(half_w), self.row, half_w, h)),
                None,
                None,
            ],
            Partition::Split if w >= 2 && h >= 2 && w.is_multiple_of(2) && h.is_multiple_of(2) => [
                Some(rect(self.column, self.row, half_w, half_h)),
                Some(rect(
                    self.column + u32::from(half_w),
                    self.row,
                    half_w,
                    half_h,
                )),
                Some(rect(
                    self.column,
                    self.row + u32::from(half_h),
                    half_w,
                    half_h,
                )),
                Some(rect(
                    self.column + u32::from(half_w),
                    self.row + u32::from(half_h),
                    half_w,
                    half_h,
                )),
            ],
            Partition::HorizontalA
                if w >= 2 && h >= 2 && w.is_multiple_of(2) && h.is_multiple_of(2) =>
            {
                [
                    Some(rect(self.column, self.row, half_w, half_h)),
                    Some(rect(
                        self.column + u32::from(half_w),
                        self.row,
                        half_w,
                        half_h,
                    )),
                    Some(rect(self.column, self.row + u32::from(half_h), w, half_h)),
                    None,
                ]
            }
            Partition::HorizontalB
                if w >= 2 && h >= 2 && w.is_multiple_of(2) && h.is_multiple_of(2) =>
            {
                [
                    Some(rect(self.column, self.row, w, half_h)),
                    Some(rect(
                        self.column,
                        self.row + u32::from(half_h),
                        half_w,
                        half_h,
                    )),
                    Some(rect(
                        self.column + u32::from(half_w),
                        self.row + u32::from(half_h),
                        half_w,
                        half_h,
                    )),
                    None,
                ]
            }
            Partition::VerticalA
                if w >= 2 && h >= 2 && w.is_multiple_of(2) && h.is_multiple_of(2) =>
            {
                [
                    Some(rect(self.column, self.row, half_w, half_h)),
                    Some(rect(
                        self.column,
                        self.row + u32::from(half_h),
                        half_w,
                        half_h,
                    )),
                    Some(rect(self.column + u32::from(half_w), self.row, half_w, h)),
                    None,
                ]
            }
            Partition::VerticalB
                if w >= 2 && h >= 2 && w.is_multiple_of(2) && h.is_multiple_of(2) =>
            {
                [
                    Some(rect(self.column, self.row, half_w, h)),
                    Some(rect(
                        self.column + u32::from(half_w),
                        self.row,
                        half_w,
                        half_h,
                    )),
                    Some(rect(
                        self.column + u32::from(half_w),
                        self.row + u32::from(half_h),
                        half_w,
                        half_h,
                    )),
                    None,
                ]
            }
            Partition::Horizontal4 if h >= 4 && h.is_multiple_of(4) => {
                let quarter = h / 4;
                core::array::from_fn(|index| {
                    Some(rect(
                        self.column,
                        self.row + index as u32 * u32::from(quarter),
                        w,
                        quarter,
                    ))
                })
            }
            Partition::Vertical4 if w >= 4 && w.is_multiple_of(4) => {
                let quarter = w / 4;
                core::array::from_fn(|index| {
                    Some(rect(
                        self.column + index as u32 * u32::from(quarter),
                        self.row,
                        quarter,
                        h,
                    ))
                })
            }
            _ => return Err(Error::InvalidObu),
        };
        Ok(PartitionChildren(children))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionChildren([Option<BlockRect>; 4]);

impl PartitionChildren {
    pub fn iter(&self) -> impl Iterator<Item = BlockRect> + '_ {
        self.0.iter().flatten().copied()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileBounds {
    pub column_start: u32,
    pub column_end: u32,
    pub row_start: u32,
    pub row_end: u32,
}

impl TileBounds {
    pub fn contains(self, row: i64, column: i64) -> bool {
        row >= i64::from(self.row_start)
            && row < i64::from(self.row_end)
            && column >= i64::from(self.column_start)
            && column < i64::from(self.column_end)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockAvailability {
    pub has_chroma: bool,
    pub upper: bool,
    pub left: bool,
    pub upper_chroma: bool,
    pub left_chroma: bool,
}

/// Derives the luma/chroma neighbor state at the start of `decode_block`.
pub fn block_availability(
    block: BlockRect,
    tile: TileBounds,
    subsampling_x: bool,
    subsampling_y: bool,
    monochrome: bool,
) -> BlockAvailability {
    let row = i64::from(block.row);
    let column = i64::from(block.column);
    let has_chroma = !monochrome
        && !(block.height_mi == 1 && subsampling_y && block.row.is_multiple_of(2))
        && !(block.width_mi == 1 && subsampling_x && block.column.is_multiple_of(2));
    let upper = tile.contains(row - 1, column);
    let left = tile.contains(row, column - 1);
    let upper_chroma = has_chroma
        && if subsampling_y && block.height_mi == 1 {
            tile.contains(row - 2, column)
        } else {
            upper
        };
    let left_chroma = has_chroma
        && if subsampling_x && block.width_mi == 1 {
            tile.contains(row, column - 2)
        } else {
            left
        };
    BlockAvailability {
        has_chroma,
        upper,
        left,
        upper_chroma,
        left_chroma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_normative_block_sizes_round_trip_dimensions() {
        for size in BlockSize::ALL {
            let (width, height) = size.dimensions();
            assert_eq!(BlockSize::from_dimensions(width, height), Ok(size));
        }
        assert_eq!(
            BlockSize::Block4x8.plane_residual_size(true, true),
            Ok(BlockSize::Block4x4)
        );
        assert_eq!(
            BlockSize::Block32x16.plane_residual_size(true, false),
            Ok(BlockSize::Block16x16)
        );
    }

    #[test]
    fn transform_depth_table_covers_extended_rectangles() {
        assert_eq!(BlockSize::Block4x4.max_transform_depth(), 0);
        assert_eq!(BlockSize::Block16x32.max_transform_depth(), 3);
        assert_eq!(BlockSize::Block16x64.max_transform_depth(), 4);
        assert_eq!(
            BlockSize::Block128x128.maximum_transform_size(),
            TxSize::Tx64x64
        );
        assert_eq!(
            BlockSize::Block8x32.maximum_transform_size(),
            TxSize::Tx8x32
        );
    }

    #[test]
    fn partition_context_tracks_smaller_upper_and_left_neighbors() {
        assert_eq!(
            partition_context(
                BlockSize::Block64x64,
                Some(BlockSize::Block32x64),
                Some(BlockSize::Block64x32),
            ),
            Ok(3)
        );
        assert_eq!(
            partition_context(
                BlockSize::Block64x64,
                Some(BlockSize::Block64x16),
                Some(BlockSize::Block16x64),
            ),
            Ok(0)
        );
    }

    #[test]
    fn segment_deinterleave_is_a_permutation_for_every_predictor() {
        for maximum in 1..=8 {
            for reference in 0..maximum {
                let mut seen = [false; 8];
                for diff in 0..maximum {
                    let value = negative_deinterleave(diff, reference, maximum).unwrap();
                    assert!(value < maximum);
                    assert!(!seen[usize::from(value)]);
                    seen[usize::from(value)] = true;
                }
            }
        }
    }

    #[test]
    fn edge_partition_cdfs_collect_disallowed_probability_mass() {
        let cdf = [
            3_000, 6_000, 9_000, 12_000, 15_000, 18_000, 21_000, 24_000, 27_000, 32_768, 0,
        ];
        assert_eq!(
            restricted_partition_cdf(&cdf, BlockSize::Block64x64, true),
            Ok([12_000, 32_768, 0])
        );
        assert_eq!(
            restricted_partition_cdf(&cdf, BlockSize::Block64x64, false),
            Ok([14_768, 32_768, 0])
        );
    }

    #[test]
    fn corner_overlap_forces_split_without_consuming_a_symbol() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdf = [8_000, 16_000, 24_000, 32_768, 0];
        assert_eq!(
            decode_partition(&mut decoder, &mut cdf, BlockSize::Block32x32, false, false,),
            Ok(Partition::Split)
        );
        assert_eq!(cdf, [8_000, 16_000, 24_000, 32_768, 0]);
    }

    struct TestCdfs([u16; 11]);

    impl PartitionCdfProvider for TestCdfs {
        fn cdf(&mut self, _block_size: BlockSize, _context: u8) -> Result<&mut [u16], Error> {
            Ok(&mut self.0)
        }
    }

    #[test]
    fn partition_tree_clips_forced_splits_at_bottom_right_corner() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TestCdfs([
            3_000, 6_000, 9_000, 12_000, 15_000, 18_000, 21_000, 24_000, 27_000, 32_768, 0,
        ]);
        let mut grid = MiGrid::new(1, 1).unwrap();
        let mut visits = 0;
        decode_partition_tree(
            &mut decoder,
            &mut cdfs,
            &mut grid,
            TileBounds {
                column_start: 0,
                column_end: 1,
                row_start: 0,
                row_end: 1,
            },
            BlockRect::new(0, 0, BlockSize::Block16x16),
            BlockSize::Block16x16,
            |_decoder, _cdfs, grid, block, size| {
                visits += 1;
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
        assert_eq!(visits, 1);
        assert_eq!(grid.get(0, 0).unwrap().size, Some(BlockSize::Block4x4));
    }

    #[test]
    fn every_partition_covers_the_parent_area_once() {
        let parent = BlockRect::new(11, 13, BlockSize::Block64x64);
        for kind in [
            Partition::None,
            Partition::Horizontal,
            Partition::Vertical,
            Partition::Split,
            Partition::HorizontalA,
            Partition::HorizontalB,
            Partition::VerticalA,
            Partition::VerticalB,
            Partition::Horizontal4,
            Partition::Vertical4,
        ] {
            let area: u32 = parent
                .partition(kind)
                .unwrap()
                .iter()
                .map(|child| u32::from(child.width_mi) * u32::from(child.height_mi))
                .sum();
            assert_eq!(
                area,
                u32::from(parent.width_mi) * u32::from(parent.height_mi)
            );
        }
    }

    #[test]
    fn four_way_partition_rejects_too_small_dimension() {
        let block = BlockRect::new(0, 0, BlockSize::Block8x8);
        assert_eq!(
            block.partition(Partition::Horizontal4),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            block.partition(Partition::Vertical4),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn subsampled_chroma_is_owned_by_only_one_small_block() {
        let tile = TileBounds {
            column_start: 0,
            column_end: 8,
            row_start: 0,
            row_end: 8,
        };
        let even = BlockRect::new(0, 0, BlockSize::Block4x4);
        let odd = BlockRect::new(1, 1, BlockSize::Block4x4);
        assert!(!block_availability(even, tile, true, true, false).has_chroma);
        assert!(block_availability(odd, tile, true, true, false).has_chroma);
    }

    #[test]
    fn tile_boundary_hides_cross_tile_neighbors() {
        let tile = TileBounds {
            column_start: 4,
            column_end: 12,
            row_start: 8,
            row_end: 16,
        };
        let first = BlockRect::new(4, 8, BlockSize::Block8x8);
        let inside = BlockRect::new(6, 10, BlockSize::Block8x8);
        assert_eq!(
            (
                block_availability(first, tile, true, true, false).upper,
                block_availability(first, tile, true, true, false).left
            ),
            (false, false)
        );
        assert_eq!(
            (
                block_availability(inside, tile, true, true, false).upper,
                block_availability(inside, tile, true, true, false).left
            ),
            (true, true)
        );
    }
}
