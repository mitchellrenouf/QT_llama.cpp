//! Normative inverse-transform primitives.

use crate::Error;

const COS128: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transform1d {
    Dct,
    Adst,
    Identity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxType {
    DctDct,
    AdstDct,
    DctAdst,
    AdstAdst,
    FlipAdstDct,
    DctFlipAdst,
    FlipAdstFlipAdst,
    AdstFlipAdst,
    FlipAdstAdst,
    Identity,
    VerticalDct,
    HorizontalDct,
    VerticalAdst,
    HorizontalAdst,
    VerticalFlipAdst,
    HorizontalFlipAdst,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxSet {
    DctOnly,
    Inter1,
    Inter2,
    Inter3,
    Intra1,
    Intra2,
}

impl TxSet {
    pub const fn symbol_count(self) -> usize {
        match self {
            Self::DctOnly => 1,
            Self::Inter1 => 16,
            Self::Inter2 => 12,
            Self::Inter3 => 2,
            Self::Intra1 => 7,
            Self::Intra2 => 5,
        }
    }

    pub fn type_from_symbol(self, symbol: usize) -> Result<TxType, Error> {
        const INTRA_1: [TxType; 7] = [
            TxType::Identity,
            TxType::DctDct,
            TxType::VerticalDct,
            TxType::HorizontalDct,
            TxType::AdstAdst,
            TxType::AdstDct,
            TxType::DctAdst,
        ];
        const INTRA_2: [TxType; 5] = [
            TxType::Identity,
            TxType::DctDct,
            TxType::AdstAdst,
            TxType::AdstDct,
            TxType::DctAdst,
        ];
        const INTER_1: [TxType; 16] = [
            TxType::Identity,
            TxType::VerticalDct,
            TxType::HorizontalDct,
            TxType::VerticalAdst,
            TxType::HorizontalAdst,
            TxType::VerticalFlipAdst,
            TxType::HorizontalFlipAdst,
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::AdstAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
        ];
        const INTER_2: [TxType; 12] = [
            TxType::Identity,
            TxType::VerticalDct,
            TxType::HorizontalDct,
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::AdstAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
        ];
        const INTER_3: [TxType; 2] = [TxType::Identity, TxType::DctDct];
        let value = match self {
            Self::DctOnly => {
                return (symbol == 0)
                    .then_some(TxType::DctDct)
                    .ok_or(Error::InvalidObu);
            }
            Self::Inter1 => INTER_1.get(symbol),
            Self::Inter2 => INTER_2.get(symbol),
            Self::Inter3 => INTER_3.get(symbol),
            Self::Intra1 => INTRA_1.get(symbol),
            Self::Intra2 => INTRA_2.get(symbol),
        };
        value.copied().ok_or(Error::InvalidObu)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TxTypeComponents {
    pub row: Transform1d,
    pub column: Transform1d,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
}

impl TxType {
    pub const fn components(self) -> TxTypeComponents {
        use Transform1d::{Adst, Dct, Identity};
        match self {
            Self::DctDct => TxTypeComponents {
                row: Dct,
                column: Dct,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::AdstDct => TxTypeComponents {
                row: Dct,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::DctAdst => TxTypeComponents {
                row: Adst,
                column: Dct,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::AdstAdst => TxTypeComponents {
                row: Adst,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::FlipAdstDct => TxTypeComponents {
                row: Dct,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: true,
            },
            Self::DctFlipAdst => TxTypeComponents {
                row: Adst,
                column: Dct,
                flip_horizontal: true,
                flip_vertical: false,
            },
            Self::FlipAdstFlipAdst => TxTypeComponents {
                row: Adst,
                column: Adst,
                flip_horizontal: true,
                flip_vertical: true,
            },
            Self::AdstFlipAdst => TxTypeComponents {
                row: Adst,
                column: Adst,
                flip_horizontal: true,
                flip_vertical: false,
            },
            Self::FlipAdstAdst => TxTypeComponents {
                row: Adst,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: true,
            },
            Self::Identity => TxTypeComponents {
                row: Identity,
                column: Identity,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::VerticalDct => TxTypeComponents {
                row: Identity,
                column: Dct,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::HorizontalDct => TxTypeComponents {
                row: Dct,
                column: Identity,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::VerticalAdst => TxTypeComponents {
                row: Identity,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::HorizontalAdst => TxTypeComponents {
                row: Adst,
                column: Identity,
                flip_horizontal: false,
                flip_vertical: false,
            },
            Self::VerticalFlipAdst => TxTypeComponents {
                row: Identity,
                column: Adst,
                flip_horizontal: false,
                flip_vertical: true,
            },
            Self::HorizontalFlipAdst => TxTypeComponents {
                row: Adst,
                column: Identity,
                flip_horizontal: true,
                flip_vertical: false,
            },
        }
    }

    pub fn allowed_in_set(self, set: u8, inter: bool) -> Result<bool, Error> {
        let index = self as usize;
        if inter {
            match set {
                0 => Ok(index == 0),
                1 => Ok(true),
                2 => Ok(index <= 11),
                3 => Ok(matches!(self, Self::DctDct | Self::Identity)),
                _ => Err(Error::InvalidObu),
            }
        } else {
            match set {
                0 => Ok(index == 0),
                1 => Ok(index <= 3
                    || matches!(
                        self,
                        Self::Identity | Self::VerticalDct | Self::HorizontalDct
                    )),
                2 => Ok(index <= 3 || self == Self::Identity),
                _ => Err(Error::InvalidObu),
            }
        }
    }
}

pub fn chroma_intra_tx_type(
    mode: u8,
    size: TxSize,
    reduced_tx_set: bool,
    lossless: bool,
) -> Result<TxType, Error> {
    const MODE_TO_TXFM: [TxType; 14] = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::DctDct,
        TxType::AdstAdst,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::DctAdst,
        TxType::AdstDct,
        TxType::AdstAdst,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
        TxType::DctDct,
    ];
    if lossless || matches!(size.square_up(), TxSize::Tx64x64) {
        return Ok(TxType::DctDct);
    }
    let proposed = *MODE_TO_TXFM
        .get(usize::from(mode))
        .ok_or(Error::InvalidObu)?;
    let set = match size.set(false, reduced_tx_set) {
        TxSet::DctOnly => 0,
        TxSet::Intra1 => 1,
        TxSet::Intra2 => 2,
        TxSet::Inter1 | TxSet::Inter2 | TxSet::Inter3 => return Err(Error::InvalidObu),
    };
    Ok(if proposed.allowed_in_set(set, false)? {
        proposed
    } else {
        TxType::DctDct
    })
}

pub fn chroma_inter_tx_type(
    luma: TxType,
    size: TxSize,
    reduced_tx_set: bool,
    lossless: bool,
) -> Result<TxType, Error> {
    if lossless || matches!(size.square_up(), TxSize::Tx64x64) {
        return Ok(TxType::DctDct);
    }
    let set = match size.set(true, reduced_tx_set) {
        TxSet::DctOnly => 0,
        TxSet::Inter1 => 1,
        TxSet::Inter2 => 2,
        TxSet::Inter3 => 3,
        TxSet::Intra1 | TxSet::Intra2 => return Err(Error::InvalidObu),
    };
    Ok(if luma.allowed_in_set(set, true)? {
        luma
    } else {
        TxType::DctDct
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxSize {
    Tx4x4,
    Tx8x8,
    Tx16x16,
    Tx32x32,
    Tx64x64,
    Tx4x8,
    Tx8x4,
    Tx8x16,
    Tx16x8,
    Tx16x32,
    Tx32x16,
    Tx32x64,
    Tx64x32,
    Tx4x16,
    Tx16x4,
    Tx8x32,
    Tx32x8,
    Tx16x64,
    Tx64x16,
}

impl TxSize {
    pub fn from_dimensions(width: u8, height: u8) -> Result<Self, Error> {
        const ALL: [TxSize; 19] = [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx32x32,
            TxSize::Tx64x64,
            TxSize::Tx4x8,
            TxSize::Tx8x4,
            TxSize::Tx8x16,
            TxSize::Tx16x8,
            TxSize::Tx16x32,
            TxSize::Tx32x16,
            TxSize::Tx32x64,
            TxSize::Tx64x32,
            TxSize::Tx4x16,
            TxSize::Tx16x4,
            TxSize::Tx8x32,
            TxSize::Tx32x8,
            TxSize::Tx16x64,
            TxSize::Tx64x16,
        ];
        ALL.into_iter()
            .find(|size| size.dimensions() == (width, height))
            .ok_or(Error::InvalidObu)
    }

    pub const fn row_shift(self) -> u8 {
        const SHIFTS: [u8; 19] = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
        SHIFTS[self as usize]
    }

    pub const fn dimensions(self) -> (u8, u8) {
        match self {
            Self::Tx4x4 => (4, 4),
            Self::Tx8x8 => (8, 8),
            Self::Tx16x16 => (16, 16),
            Self::Tx32x32 => (32, 32),
            Self::Tx64x64 => (64, 64),
            Self::Tx4x8 => (4, 8),
            Self::Tx8x4 => (8, 4),
            Self::Tx8x16 => (8, 16),
            Self::Tx16x8 => (16, 8),
            Self::Tx16x32 => (16, 32),
            Self::Tx32x16 => (32, 16),
            Self::Tx32x64 => (32, 64),
            Self::Tx64x32 => (64, 32),
            Self::Tx4x16 => (4, 16),
            Self::Tx16x4 => (16, 4),
            Self::Tx8x32 => (8, 32),
            Self::Tx32x8 => (32, 8),
            Self::Tx16x64 => (16, 64),
            Self::Tx64x16 => (64, 16),
        }
    }

    pub const fn split(self) -> Self {
        match self {
            Self::Tx4x4 | Self::Tx4x8 | Self::Tx8x4 => Self::Tx4x4,
            Self::Tx8x8 => Self::Tx4x4,
            Self::Tx16x16 => Self::Tx8x8,
            Self::Tx32x32 => Self::Tx16x16,
            Self::Tx64x64 => Self::Tx32x32,
            Self::Tx8x16 => Self::Tx8x8,
            Self::Tx16x8 => Self::Tx8x8,
            Self::Tx16x32 => Self::Tx16x16,
            Self::Tx32x16 => Self::Tx16x16,
            Self::Tx32x64 => Self::Tx32x32,
            Self::Tx64x32 => Self::Tx32x32,
            Self::Tx4x16 => Self::Tx4x8,
            Self::Tx16x4 => Self::Tx8x4,
            Self::Tx8x32 => Self::Tx8x16,
            Self::Tx32x8 => Self::Tx16x8,
            Self::Tx16x64 => Self::Tx16x32,
            Self::Tx64x16 => Self::Tx32x16,
        }
    }

    pub const fn square_up(self) -> Self {
        let (width, height) = self.dimensions();
        let largest = if width > height { width } else { height };
        match largest {
            4 => Self::Tx4x4,
            8 => Self::Tx8x8,
            16 => Self::Tx16x16,
            32 => Self::Tx32x32,
            _ => Self::Tx64x64,
        }
    }

    pub const fn square(self) -> Self {
        let (width, height) = self.dimensions();
        let smallest = if width < height { width } else { height };
        match smallest {
            4 => Self::Tx4x4,
            8 => Self::Tx8x8,
            16 => Self::Tx16x16,
            32 => Self::Tx32x32,
            _ => Self::Tx64x64,
        }
    }

    pub const fn set(self, inter: bool, reduced: bool) -> TxSet {
        if matches!(self.square_up(), Self::Tx64x64) {
            return TxSet::DctOnly;
        }
        if inter {
            if reduced || matches!(self.square_up(), Self::Tx32x32) {
                TxSet::Inter3
            } else if matches!(self.square(), Self::Tx16x16) {
                TxSet::Inter2
            } else {
                TxSet::Inter1
            }
        } else if matches!(self.square_up(), Self::Tx32x32) {
            TxSet::DctOnly
        } else if reduced || matches!(self.square(), Self::Tx16x16) {
            TxSet::Intra2
        } else {
            TxSet::Intra1
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InverseTransformConfig {
    pub width: usize,
    pub height: usize,
    pub row: Transform1d,
    pub column: Transform1d,
    pub row_shift: u8,
    pub bit_depth: u8,
    pub lossless: bool,
}

pub fn inverse_2d(coefficients: &mut [i32], config: InverseTransformConfig) -> Result<(), Error> {
    let InverseTransformConfig {
        width,
        height,
        row,
        column,
        row_shift,
        bit_depth,
        lossless,
    } = config;
    if !valid_transform_size(width, height)
        || coefficients.len() != width.checked_mul(height).ok_or(Error::LimitExceeded)?
        || !matches!(bit_depth, 8 | 10 | 12)
        || (lossless && (width != 4 || height != 4))
    {
        return Err(Error::InvalidObu);
    }
    let row_clamp = bit_depth + 8;
    let column_clamp = (bit_depth + 6).max(16);
    let rectangular = width.abs_diff(height) == width.min(height);
    let mut temporary = [0i32; 64];
    for y in 0..height {
        for x in 0..width {
            temporary[x] = if x < 32 && y < 32 {
                coefficients[y * width + x]
            } else {
                0
            };
            if rectangular {
                temporary[x] = i32::try_from(round2(i64::from(temporary[x]) * 2896, 12)?)
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
        if lossless {
            inverse_walsh_hadamard(
                (&mut temporary[..4])
                    .try_into()
                    .map_err(|_| Error::InvalidObu)?,
                2,
            )?;
        } else {
            apply_1d(&mut temporary[..width], row, row_clamp)?;
        }
        for x in 0..width {
            let shifted = round2(
                i64::from(temporary[x]),
                if lossless { 0 } else { row_shift },
            )?;
            coefficients[y * width + x] = clip_precision(shifted, column_clamp)?;
        }
    }
    for x in 0..width {
        for y in 0..height {
            temporary[y] = coefficients[y * width + x];
        }
        if lossless {
            inverse_walsh_hadamard(
                (&mut temporary[..4])
                    .try_into()
                    .map_err(|_| Error::InvalidObu)?,
                0,
            )?;
        } else {
            apply_1d(&mut temporary[..height], column, column_clamp)?;
        }
        for y in 0..height {
            coefficients[y * width + x] = i32::try_from(round2(
                i64::from(temporary[y]),
                if lossless { 0 } else { 4 },
            )?)
            .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(())
}

fn valid_transform_size(width: usize, height: usize) -> bool {
    matches!(
        (width, height),
        (4, 4)
            | (8, 8)
            | (16, 16)
            | (32, 32)
            | (64, 64)
            | (4, 8)
            | (8, 4)
            | (8, 16)
            | (16, 8)
            | (16, 32)
            | (32, 16)
            | (32, 64)
            | (64, 32)
            | (4, 16)
            | (16, 4)
            | (8, 32)
            | (32, 8)
            | (16, 64)
            | (64, 16)
    )
}

fn apply_1d(values: &mut [i32], transform: Transform1d, clamp_bits: u8) -> Result<(), Error> {
    match transform {
        Transform1d::Dct => inverse_dct(values, clamp_bits),
        Transform1d::Adst => inverse_adst(values, clamp_bits),
        Transform1d::Identity => inverse_identity(values),
    }
}

pub fn inverse_dct(values: &mut [i32], clamp_bits: u8) -> Result<(), Error> {
    let n = match values.len() {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => return Err(Error::InvalidObu),
    };
    if clamp_bits == 0 || clamp_bits > 31 {
        return Err(Error::InvalidObu);
    }
    bit_reverse_permute(values, n);
    if n == 6 {
        for i in 0..16 {
            butterfly(
                values,
                32 + i,
                63 - i,
                63 - 4 * brev(4, i),
                false,
                clamp_bits,
            )?;
        }
    }
    if n >= 5 {
        for i in 0..8 {
            butterfly(
                values,
                16 + i,
                31 - i,
                6 + (brev(3, 7 - i) << 3),
                false,
                clamp_bits,
            )?;
        }
    }
    if n == 6 {
        for i in 0..16 {
            hadamard(values, 32 + i * 2, 33 + i * 2, i & 1 != 0, clamp_bits)?;
        }
    }
    if n >= 4 {
        for i in 0..4 {
            butterfly(
                values,
                8 + i,
                15 - i,
                12 + (brev(2, 3 - i) << 4),
                false,
                clamp_bits,
            )?;
        }
    }
    if n >= 5 {
        for i in 0..8 {
            hadamard(values, 16 + 2 * i, 17 + 2 * i, i & 1 != 0, clamp_bits)?;
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                butterfly(
                    values,
                    62 - i * 4 - j,
                    33 + i * 4 + j,
                    60 - 16 * brev(2, i) + 64 * j,
                    true,
                    clamp_bits,
                )?;
            }
        }
    }
    if n >= 3 {
        for i in 0..2 {
            butterfly(values, 4 + i, 7 - i, 56 - 32 * i, false, clamp_bits)?;
        }
    }
    if n >= 4 {
        for i in 0..4 {
            hadamard(values, 8 + 2 * i, 9 + 2 * i, i & 1 != 0, clamp_bits)?;
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                butterfly(
                    values,
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + (j << 6) + ((1 - i) << 5),
                    true,
                    clamp_bits,
                )?;
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                hadamard(
                    values,
                    32 + i * 4 + j,
                    35 + i * 4 - j,
                    i & 1 != 0,
                    clamp_bits,
                )?;
            }
        }
    }
    for i in 0..2 {
        butterfly(values, 2 * i, 2 * i + 1, 32 + 16 * i, i == 0, clamp_bits)?;
    }
    if n >= 3 {
        for i in 0..2 {
            hadamard(values, 4 + 2 * i, 5 + 2 * i, i != 0, clamp_bits)?;
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(values, 14 - i, 9 + i, 48 + 64 * i, true, clamp_bits)?;
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                hadamard(
                    values,
                    16 + 4 * i + j,
                    19 + 4 * i - j,
                    i & 1 != 0,
                    clamp_bits,
                )?;
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                butterfly(
                    values,
                    61 - i * 8 - j,
                    34 + i * 8 + j,
                    56 - i * 32 + (j >> 1) * 64,
                    true,
                    clamp_bits,
                )?;
            }
        }
    }
    for i in 0..2 {
        hadamard(values, i, 3 - i, false, clamp_bits)?;
    }
    if n >= 3 {
        butterfly(values, 6, 5, 32, true, clamp_bits)?;
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                hadamard(values, 8 + 4 * i + j, 11 + 4 * i - j, i != 0, clamp_bits)?;
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(values, 29 - i, 18 + i, 48 + (i >> 1) * 64, true, clamp_bits)?;
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                hadamard(
                    values,
                    32 + 8 * i + j,
                    39 + 8 * i - j,
                    i & 1 != 0,
                    clamp_bits,
                )?;
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            hadamard(values, i, 7 - i, false, clamp_bits)?;
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(values, 13 - i, 10 + i, 32, true, clamp_bits)?;
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                hadamard(values, 16 + i * 8 + j, 23 + i * 8 - j, i != 0, clamp_bits)?;
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(
                values,
                59 - i,
                36 + i,
                if i < 4 { 48 } else { 112 },
                true,
                clamp_bits,
            )?;
        }
    }
    if n >= 4 {
        for i in 0..8 {
            hadamard(values, i, 15 - i, false, clamp_bits)?;
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(values, 27 - i, 20 + i, 32, true, clamp_bits)?;
        }
    }
    if n == 6 {
        for i in 0..8 {
            hadamard(values, 32 + i, 47 - i, false, clamp_bits)?;
            hadamard(values, 48 + i, 63 - i, true, clamp_bits)?;
        }
    }
    if n >= 5 {
        for i in 0..16 {
            hadamard(values, i, 31 - i, false, clamp_bits)?;
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(values, 55 - i, 40 + i, 32, true, clamp_bits)?;
        }
        for i in 0..32 {
            hadamard(values, i, 63 - i, false, clamp_bits)?;
        }
    }
    Ok(())
}

pub fn inverse_adst(values: &mut [i32], clamp_bits: u8) -> Result<(), Error> {
    if clamp_bits == 0 || clamp_bits > 31 {
        return Err(Error::InvalidObu);
    }
    match values.len() {
        4 => inverse_adst4(values, clamp_bits),
        8 => {
            adst_input_permute(values, 3);
            for i in 0..4 {
                butterfly(values, 2 * i, 2 * i + 1, 60 - 16 * i, true, clamp_bits)?;
            }
            for i in 0..4 {
                hadamard(values, i, 4 + i, false, clamp_bits)?;
            }
            for i in 0..2 {
                butterfly(values, 4 + 3 * i, 5 + i, 48 - 32 * i, true, clamp_bits)?;
            }
            for j in 0..2 {
                for i in 0..2 {
                    hadamard(values, 4 * j + i, 2 + 4 * j + i, false, clamp_bits)?;
                }
            }
            for i in 0..2 {
                butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true, clamp_bits)?;
            }
            adst_output_permute(values, 3);
            Ok(())
        }
        16 => {
            adst_input_permute(values, 4);
            for i in 0..8 {
                butterfly(values, 2 * i, 2 * i + 1, 62 - 8 * i, true, clamp_bits)?;
            }
            for i in 0..8 {
                hadamard(values, i, 8 + i, false, clamp_bits)?;
            }
            for i in 0..2 {
                butterfly(values, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true, clamp_bits)?;
                butterfly(values, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true, clamp_bits)?;
            }
            for j in 0..2 {
                for i in 0..4 {
                    hadamard(values, 8 * j + i, 4 + 8 * j + i, false, clamp_bits)?;
                }
            }
            for j in 0..2 {
                for i in 0..2 {
                    butterfly(
                        values,
                        4 + 8 * j + 3 * i,
                        5 + 8 * j + i,
                        48 - 32 * i,
                        true,
                        clamp_bits,
                    )?;
                }
            }
            for j in 0..4 {
                for i in 0..2 {
                    hadamard(values, 4 * j + i, 2 + 4 * j + i, false, clamp_bits)?;
                }
            }
            for i in 0..4 {
                butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true, clamp_bits)?;
            }
            adst_output_permute(values, 4);
            Ok(())
        }
        _ => Err(Error::InvalidObu),
    }
}

fn inverse_adst4(values: &mut [i32], clamp_bits: u8) -> Result<(), Error> {
    let t0 = i64::from(values[0]);
    let t1 = i64::from(values[1]);
    let t2 = i64::from(values[2]);
    let t3 = i64::from(values[3]);
    let mut s0 = 1321 * t0;
    let mut s1 = 2482 * t0;
    let mut s3 = 3803 * t2;
    let s4 = 1321 * t2;
    let s5 = 2482 * t3;
    let s6 = 3803 * t3;
    let a7 = t0.checked_sub(t2).ok_or(Error::LimitExceeded)?;
    let b7 = a7.checked_add(t3).ok_or(Error::LimitExceeded)?;
    checked_precision(b7, clamp_bits)?;
    s0 = s0.checked_add(s3).ok_or(Error::LimitExceeded)?;
    s1 = s1.checked_sub(s4).ok_or(Error::LimitExceeded)?;
    s3 = 3344 * t1;
    let s2 = 3344 * b7;
    s0 = s0.checked_add(s5).ok_or(Error::LimitExceeded)?;
    s1 = s1.checked_sub(s6).ok_or(Error::LimitExceeded)?;
    let output = [
        s0.checked_add(s3).ok_or(Error::LimitExceeded)?,
        s1.checked_add(s3).ok_or(Error::LimitExceeded)?,
        s2,
        s0.checked_add(s1)
            .and_then(|value| value.checked_sub(s3))
            .ok_or(Error::LimitExceeded)?,
    ];
    for (value, transformed) in values.iter_mut().zip(output) {
        *value = checked_precision(round2(transformed, 12)?, clamp_bits)?;
    }
    Ok(())
}

fn adst_input_permute(values: &mut [i32], bits: usize) {
    let mut copy = [0i32; 16];
    copy[..values.len()].copy_from_slice(values);
    let length = values.len();
    for (index, value) in values.iter_mut().enumerate() {
        let source = if index & 1 != 0 {
            index - 1
        } else {
            length - index - 1
        };
        *value = copy[source];
    }
    debug_assert_eq!(length, 1 << bits);
}

fn adst_output_permute(values: &mut [i32], bits: usize) {
    let mut copy = [0i32; 16];
    copy[..values.len()].copy_from_slice(values);
    for (index, value) in values.iter_mut().enumerate() {
        let a = (index >> 3) & 1;
        let b = ((index >> 2) & 1) ^ ((index >> 3) & 1);
        let c = ((index >> 1) & 1) ^ ((index >> 2) & 1);
        let d = (index & 1) ^ ((index >> 1) & 1);
        let source = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - bits);
        *value = if index & 1 != 0 {
            -copy[source]
        } else {
            copy[source]
        };
    }
}

fn bit_reverse_permute(values: &mut [i32], bits: usize) {
    let mut copy = [0i32; 64];
    copy[..values.len()].copy_from_slice(values);
    for (index, value) in values.iter_mut().enumerate() {
        *value = copy[brev(bits, index)];
    }
}

fn brev(bits: usize, value: usize) -> usize {
    let mut result = 0;
    for bit in 0..bits {
        result |= ((value >> bit) & 1) << (bits - 1 - bit);
    }
    result
}

fn butterfly(
    values: &mut [i32],
    a: usize,
    b: usize,
    angle: usize,
    flip: bool,
    clamp_bits: u8,
) -> Result<(), Error> {
    let left = i64::from(*values.get(a).ok_or(Error::InvalidObu)?);
    let right = i64::from(*values.get(b).ok_or(Error::InvalidObu)?);
    let cosine = i64::from(cos128(angle as i32));
    let sine = i64::from(cos128(angle as i32 - 64));
    let x = left * cosine - right * sine;
    let y = left * sine + right * cosine;
    let mut first = checked_precision(round2(x, 12)?, clamp_bits)?;
    let mut second = checked_precision(round2(y, 12)?, clamp_bits)?;
    if flip {
        core::mem::swap(&mut first, &mut second);
    }
    values[a] = first;
    values[b] = second;
    Ok(())
}

fn hadamard(
    values: &mut [i32],
    a: usize,
    b: usize,
    flip: bool,
    clamp_bits: u8,
) -> Result<(), Error> {
    let (first_index, second_index) = if flip { (b, a) } else { (a, b) };
    let first = i64::from(*values.get(first_index).ok_or(Error::InvalidObu)?);
    let second = i64::from(*values.get(second_index).ok_or(Error::InvalidObu)?);
    values[first_index] = clip_precision(first + second, clamp_bits)?;
    values[second_index] = clip_precision(first - second, clamp_bits)?;
    Ok(())
}

fn cos128(angle: i32) -> i32 {
    let angle = angle.rem_euclid(256) as usize;
    match angle {
        0..=64 => COS128[angle],
        65..=128 => -COS128[128 - angle],
        129..=192 => -COS128[angle - 128],
        _ => COS128[256 - angle],
    }
}

fn checked_precision(value: i64, bits: u8) -> Result<i32, Error> {
    let minimum = -(1i64 << (bits - 1));
    let maximum = (1i64 << (bits - 1)) - 1;
    if value < minimum || value > maximum {
        return Err(Error::InvalidObu);
    }
    Ok(value as i32)
}

fn clip_precision(value: i64, bits: u8) -> Result<i32, Error> {
    let minimum = -(1i64 << (bits - 1));
    let maximum = (1i64 << (bits - 1)) - 1;
    Ok(value.clamp(minimum, maximum) as i32)
}

pub fn inverse_identity(values: &mut [i32]) -> Result<(), Error> {
    let (factor, shift) = match values.len() {
        4 => (5793i64, 12),
        8 => (2, 0),
        16 => (11586, 12),
        32 => (4, 0),
        _ => return Err(Error::InvalidObu),
    };
    for value in values {
        let scaled = i64::from(*value)
            .checked_mul(factor)
            .ok_or(Error::LimitExceeded)?;
        let rounded = round2(scaled, shift)?;
        *value = i32::try_from(rounded).map_err(|_| Error::LimitExceeded)?;
    }
    Ok(())
}

pub fn inverse_walsh_hadamard(values: &mut [i32; 4], shift: u8) -> Result<(), Error> {
    if shift > 31 {
        return Err(Error::InvalidObu);
    }
    let mut a = i64::from(values[0]) >> shift;
    let mut c = i64::from(values[1]) >> shift;
    let mut d = i64::from(values[2]) >> shift;
    let mut b = i64::from(values[3]) >> shift;
    a = a.checked_add(c).ok_or(Error::LimitExceeded)?;
    d = d.checked_sub(b).ok_or(Error::LimitExceeded)?;
    let e = a.checked_sub(d).ok_or(Error::LimitExceeded)? >> 1;
    b = e.checked_sub(b).ok_or(Error::LimitExceeded)?;
    c = e.checked_sub(c).ok_or(Error::LimitExceeded)?;
    a = a.checked_sub(b).ok_or(Error::LimitExceeded)?;
    d = d.checked_add(c).ok_or(Error::LimitExceeded)?;
    for (output, value) in values.iter_mut().zip([a, b, c, d]) {
        *output = i32::try_from(value).map_err(|_| Error::LimitExceeded)?;
    }
    Ok(())
}

fn round2(value: i64, shift: u8) -> Result<i64, Error> {
    if shift == 0 {
        return Ok(value);
    }
    if shift >= 63 {
        return Err(Error::InvalidObu);
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
    fn transform_size_splits_follow_normative_rectangular_table() {
        assert_eq!(TxSize::Tx64x64.split(), TxSize::Tx32x32);
        assert_eq!(TxSize::Tx4x16.split(), TxSize::Tx4x8);
        assert_eq!(TxSize::Tx16x64.split(), TxSize::Tx16x32);
        assert_eq!(TxSize::Tx64x16.split(), TxSize::Tx32x16);
    }

    #[test]
    fn flipped_transform_types_map_to_axis_and_output_flip() {
        assert_eq!(
            TxType::FlipAdstDct.components(),
            TxTypeComponents {
                row: Transform1d::Dct,
                column: Transform1d::Adst,
                flip_horizontal: false,
                flip_vertical: true,
            }
        );
        assert_eq!(
            TxType::HorizontalFlipAdst.components(),
            TxTypeComponents {
                row: Transform1d::Adst,
                column: Transform1d::Identity,
                flip_horizontal: true,
                flip_vertical: false,
            }
        );
    }

    #[test]
    fn transform_type_sets_enforce_intra_and_inter_membership() {
        assert_eq!(TxType::FlipAdstDct.allowed_in_set(1, true), Ok(true));
        assert_eq!(TxType::FlipAdstDct.allowed_in_set(2, false), Ok(false));
        assert_eq!(TxType::Identity.allowed_in_set(3, true), Ok(true));
        assert_eq!(TxType::VerticalDct.allowed_in_set(2, true), Ok(true));
        assert_eq!(TxSize::Tx16x64.square_up(), TxSize::Tx64x64);
    }

    #[test]
    fn chroma_intra_transform_type_follows_mode_and_set() {
        assert_eq!(
            chroma_intra_tx_type(1, TxSize::Tx8x8, false, false),
            Ok(TxType::AdstDct)
        );
        assert_eq!(
            chroma_intra_tx_type(1, TxSize::Tx32x32, false, false),
            Ok(TxType::DctDct)
        );
        assert_eq!(
            chroma_intra_tx_type(13, TxSize::Tx8x8, false, false),
            Ok(TxType::DctDct)
        );
    }

    #[test]
    fn transform_set_selection_and_symbol_inversion_are_normative() {
        assert_eq!(TxSize::Tx8x16.set(false, false), TxSet::Intra1);
        assert_eq!(TxSize::Tx16x16.set(false, false), TxSet::Intra2);
        assert_eq!(TxSize::Tx16x16.set(true, false), TxSet::Inter2);
        assert_eq!(TxSize::Tx32x32.set(true, false), TxSet::Inter3);
        assert_eq!(TxSize::Tx16x64.set(true, false), TxSet::DctOnly);
        assert_eq!(
            TxSet::Inter1.type_from_symbol(13),
            Ok(TxType::FlipAdstFlipAdst)
        );
        assert_eq!(TxSet::Intra1.type_from_symbol(2), Ok(TxType::VerticalDct));
        assert_eq!(TxSet::Inter3.type_from_symbol(2), Err(Error::InvalidObu));
    }

    #[test]
    fn identity_scaling_depends_on_transform_length() {
        let mut four = [4096, -4096, 0, 1];
        inverse_identity(&mut four).unwrap();
        assert_eq!(four[..3], [5793, -5793, 0]);
        let mut sixteen = [0; 16];
        sixteen[0] = 4096;
        inverse_identity(&mut sixteen).unwrap();
        assert_eq!(sixteen[0], 11586);
    }

    #[test]
    fn lossless_wht_prescaling_is_applied() {
        let mut values = [4, 0, 0, 0];
        inverse_walsh_hadamard(&mut values, 2).unwrap();
        assert_eq!(values, [1, 0, 0, 0]);
    }

    #[test]
    fn unsupported_identity_size_is_rejected() {
        assert_eq!(inverse_identity(&mut [0; 2]), Err(Error::InvalidObu));
    }

    #[test]
    fn dct4_dc_coefficient_spreads_evenly() {
        let mut values = [4096, 0, 0, 0];
        inverse_dct(&mut values, 16).unwrap();
        assert_eq!(values, [2896; 4]);
    }

    #[test]
    fn dct_dc_basis_is_constant_at_every_length() {
        let mut eight = [0; 8];
        eight[0] = 4096;
        inverse_dct(&mut eight, 20).unwrap();
        assert_eq!(eight, [2896; 8]);
        let mut sixteen = [0; 16];
        sixteen[0] = 4096;
        inverse_dct(&mut sixteen, 20).unwrap();
        assert_eq!(sixteen, [2896; 16]);
        let mut thirty_two = [0; 32];
        thirty_two[0] = 4096;
        inverse_dct(&mut thirty_two, 20).unwrap();
        assert_eq!(thirty_two, [2896; 32]);
        let mut sixty_four = [0; 64];
        sixty_four[0] = 4096;
        inverse_dct(&mut sixty_four, 20).unwrap();
        assert_eq!(sixty_four, [2896; 64]);
    }

    #[test]
    fn cosine_extension_covers_all_quadrants() {
        assert_eq!(cos128(0), 4096);
        assert_eq!(cos128(64), 0);
        assert_eq!(cos128(128), -4096);
        assert_eq!(cos128(192), 0);
        assert_eq!(cos128(256), 4096);
        assert_eq!(cos128(-64), 0);
    }

    #[test]
    fn adst4_first_basis_matches_sine_constants() {
        let mut values = [4096, 0, 0, 0];
        inverse_adst(&mut values, 20).unwrap();
        assert_eq!(values, [1321, 2482, 3344, 3803]);
    }

    #[test]
    fn two_dimensional_dct_applies_column_shift() {
        let mut coefficients = [0; 16];
        coefficients[0] = 4096;
        inverse_2d(
            &mut coefficients,
            InverseTransformConfig {
                width: 4,
                height: 4,
                row: Transform1d::Dct,
                column: Transform1d::Dct,
                row_shift: 0,
                bit_depth: 8,
                lossless: false,
            },
        )
        .unwrap();
        assert_eq!(coefficients, [128; 16]);
    }

    #[test]
    fn q7_four_by_eight_dct_matches_normative_residual() {
        let mut coefficients = [
            -1736, -70, 0, 35, -35, 0, 0, 0, -70, 0, 0, 0, 70, -35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        inverse_2d(
            &mut coefficients,
            InverseTransformConfig {
                width: 4,
                height: 8,
                row: Transform1d::Dct,
                column: Transform1d::Dct,
                row_shift: TxSize::Tx4x8.row_shift(),
                bit_depth: 8,
                lossless: false,
            },
        )
        .unwrap();
        assert_eq!(
            coefficients,
            [
                -42, -42, -37, -37, -42, -42, -39, -39, -40, -42, -39, -40, -39, -39, -36, -37,
                -37, -37, -33, -32, -38, -37, -32, -32, -40, -40, -36, -36, -41, -42, -40, -41,
            ]
        );
    }

    #[test]
    fn invalid_transform_geometry_is_rejected() {
        let mut coefficients = [0; 64];
        assert_eq!(
            inverse_2d(
                &mut coefficients,
                InverseTransformConfig {
                    width: 4,
                    height: 64,
                    row: Transform1d::Dct,
                    column: Transform1d::Dct,
                    row_shift: 0,
                    bit_depth: 8,
                    lossless: false,
                }
            ),
            Err(Error::InvalidObu)
        );
    }
}
