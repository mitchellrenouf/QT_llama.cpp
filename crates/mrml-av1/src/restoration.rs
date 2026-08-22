//! Loop-restoration unit syntax and filtering primitives (sections 5.11.57 and 7.17).

use crate::{
    ChromaSampling, Error,
    cdf::TileCdfs,
    entropy::SymbolDecoder,
    params::{Restoration, RestorationType},
    partition::BlockRect,
    reconstruction::{FrameBuffer, Plane},
};
use mrml_runtime::Vector;

pub const WIENER_TAPS_MIN: [i16; 3] = [-5, -23, -17];
pub const WIENER_TAPS_MAX: [i16; 3] = [10, 8, 46];
pub const WIENER_TAPS_K: [u8; 3] = [1, 2, 3];
pub const SGRPROJ_XQD_MIN: [i16; 2] = [-96, -32];
pub const SGRPROJ_XQD_MAX: [i16; 2] = [31, 95];
const SGR_PARAMS: [[u8; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RestorationUnit {
    pub kind: RestorationType,
    pub wiener: [[i16; 3]; 2],
    pub sgr_set: u8,
    pub sgr_xqd: [i16; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestorationUnits {
    units: [Vector<RestorationUnit>; 3],
    rows: [u32; 3],
    columns: [u32; 3],
    reference_wiener: [[[i16; 3]; 2]; 3],
    reference_sgr: [[i16; 2]; 3],
}

impl RestorationUnits {
    pub fn new(
        parameters: &Restoration,
        upscaled_width: u32,
        frame_height: u32,
        sampling: ChromaSampling,
    ) -> Result<Self, Error> {
        let (sub_x, sub_y) = subsampling(sampling);
        let rows = core::array::from_fn(|plane| {
            let shift = if plane == 0 { 0 } else { sub_y };
            count_units(
                parameters.unit_size[plane],
                frame_height.div_ceil(1 << shift),
            )
        });
        let columns = core::array::from_fn(|plane| {
            let shift = if plane == 0 { 0 } else { sub_x };
            count_units(
                parameters.unit_size[plane],
                upscaled_width.div_ceil(1 << shift),
            )
        });
        let mut units: [Vector<RestorationUnit>; 3] = core::array::from_fn(|_| Vector::new());
        for plane in 0..3 {
            if parameters.frame_type[plane] == RestorationType::None {
                continue;
            }
            let count = usize::try_from(
                rows[plane]
                    .checked_mul(columns[plane])
                    .ok_or(Error::LimitExceeded)?,
            )
            .map_err(|_| Error::LimitExceeded)?;
            units[plane] = Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
            units[plane]
                .try_resize(count, RestorationUnit::default())
                .map_err(|_| Error::LimitExceeded)?;
        }
        Ok(Self {
            units,
            rows,
            columns,
            reference_wiener: [[[3, -7, 15]; 2]; 3],
            reference_sgr: [[-32, 31]; 3],
        })
    }

    pub fn reset_tile_references(&mut self) {
        self.reference_wiener = [[[3, -7, 15]; 2]; 3];
        self.reference_sgr = [[-32, 31]; 3];
    }

    #[allow(clippy::too_many_arguments)]
    pub fn read_superblock(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfs,
        root: BlockRect,
        parameters: &Restoration,
        sampling: ChromaSampling,
        use_superres: bool,
        superres_denom: u8,
    ) -> Result<(), Error> {
        let (sub_x, sub_y) = subsampling(sampling);
        let width4 = u32::from(root.width_mi);
        let height4 = u32::from(root.height_mi);
        for plane in 0..3 {
            if parameters.frame_type[plane] == RestorationType::None {
                continue;
            }
            let sx = if plane == 0 { 0 } else { sub_x };
            let sy = if plane == 0 { 0 } else { sub_y };
            let unit_size = u32::from(parameters.unit_size[plane]);
            if unit_size == 0 {
                return Err(Error::InvalidObu);
            }
            let row_scale = 4 >> sy;
            let row_start = ceil_div(root.row * row_scale, unit_size);
            let row_end =
                ceil_div((root.row + height4) * row_scale, unit_size).min(self.rows[plane]);
            let (numerator, denominator) = if use_superres {
                ((4 >> sx) * u32::from(superres_denom), unit_size * 8)
            } else {
                (4 >> sx, unit_size)
            };
            let column_start = ceil_div(root.column * numerator, denominator);
            let column_end =
                ceil_div((root.column + width4) * numerator, denominator).min(self.columns[plane]);
            for row in row_start..row_end {
                for column in column_start..column_end {
                    self.read_unit(
                        decoder,
                        cdfs,
                        parameters.frame_type[plane],
                        plane,
                        row,
                        column,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn unit(&self, plane: usize, row: u32, column: u32) -> Result<RestorationUnit, Error> {
        let index = usize::try_from(
            row.checked_mul(*self.columns.get(plane).ok_or(Error::InvalidObu)?)
                .and_then(|value| value.checked_add(column))
                .ok_or(Error::LimitExceeded)?,
        )
        .map_err(|_| Error::LimitExceeded)?;
        self.units
            .get(plane)
            .and_then(|units| units.get(index))
            .copied()
            .ok_or(Error::InvalidObu)
    }

    fn read_unit(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfs,
        frame_kind: RestorationType,
        plane: usize,
        row: u32,
        column: u32,
    ) -> Result<(), Error> {
        let kind = match frame_kind {
            RestorationType::None => RestorationType::None,
            RestorationType::Wiener => {
                if cdfs.read_restoration_wiener(decoder)? {
                    RestorationType::Wiener
                } else {
                    RestorationType::None
                }
            }
            RestorationType::Sgrproj => {
                if cdfs.read_restoration_sgrproj(decoder)? {
                    RestorationType::Sgrproj
                } else {
                    RestorationType::None
                }
            }
            RestorationType::Switchable => match cdfs.read_restoration_switchable(decoder)? {
                0 => RestorationType::None,
                1 => RestorationType::Wiener,
                2 => RestorationType::Sgrproj,
                _ => return Err(Error::InvalidObu),
            },
        };
        let mut unit = RestorationUnit {
            kind,
            ..RestorationUnit::default()
        };
        if kind == RestorationType::Wiener {
            for pass in 0..2 {
                let first = usize::from(plane != 0);
                for coefficient in first..3 {
                    let value = decode_signed_subexp_with_ref(
                        decoder,
                        WIENER_TAPS_MIN[coefficient],
                        WIENER_TAPS_MAX[coefficient] + 1,
                        WIENER_TAPS_K[coefficient],
                        self.reference_wiener[plane][pass][coefficient],
                    )?;
                    unit.wiener[pass][coefficient] = value;
                    self.reference_wiener[plane][pass][coefficient] = value;
                }
            }
        } else if kind == RestorationType::Sgrproj {
            unit.sgr_set = u8::try_from(decoder.read_literal(4)?).map_err(|_| Error::InvalidObu)?;
            for pass in 0..2 {
                let value = if SGR_PARAMS[usize::from(unit.sgr_set)][pass * 2] != 0 {
                    decode_signed_subexp_with_ref(
                        decoder,
                        SGRPROJ_XQD_MIN[pass],
                        SGRPROJ_XQD_MAX[pass] + 1,
                        4,
                        self.reference_sgr[plane][pass],
                    )?
                } else if pass == 1 {
                    (128 - self.reference_sgr[plane][0])
                        .clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1])
                } else {
                    0
                };
                unit.sgr_xqd[pass] = value;
                self.reference_sgr[plane][pass] = value;
            }
        }
        let index = usize::try_from(
            row.checked_mul(self.columns[plane])
                .and_then(|value| value.checked_add(column))
                .ok_or(Error::LimitExceeded)?,
        )
        .map_err(|_| Error::LimitExceeded)?;
        *self.units[plane].get_mut(index).ok_or(Error::InvalidObu)? = unit;
        Ok(())
    }
}

pub fn apply_frame(
    deblocked: &FrameBuffer,
    cdef: &FrameBuffer,
    units: &RestorationUnits,
    parameters: &Restoration,
    upscaled_width: u32,
    frame_height: u32,
) -> Result<FrameBuffer, Error> {
    if deblocked.bit_depth() != cdef.bit_depth() || deblocked.sampling() != cdef.sampling() {
        return Err(Error::InvalidObu);
    }
    let mut output = cdef.clone();
    apply_plane(
        &deblocked.y,
        &cdef.y,
        &mut output.y,
        units,
        parameters,
        0,
        upscaled_width,
        frame_height,
        0,
        cdef.bit_depth(),
    )?;
    let (sub_x, sub_y) = subsampling(cdef.sampling());
    if cdef.sampling() != ChromaSampling::Cs400 {
        apply_plane(
            deblocked.u.as_ref().ok_or(Error::InvalidObu)?,
            cdef.u.as_ref().ok_or(Error::InvalidObu)?,
            output.u.as_mut().ok_or(Error::InvalidObu)?,
            units,
            parameters,
            1,
            upscaled_width.div_ceil(1 << sub_x),
            frame_height.div_ceil(1 << sub_y),
            sub_y,
            cdef.bit_depth(),
        )?;
        apply_plane(
            deblocked.v.as_ref().ok_or(Error::InvalidObu)?,
            cdef.v.as_ref().ok_or(Error::InvalidObu)?,
            output.v.as_mut().ok_or(Error::InvalidObu)?,
            units,
            parameters,
            2,
            upscaled_width.div_ceil(1 << sub_x),
            frame_height.div_ceil(1 << sub_y),
            sub_y,
            cdef.bit_depth(),
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_plane(
    deblocked: &Plane,
    cdef: &Plane,
    output: &mut Plane,
    units: &RestorationUnits,
    parameters: &Restoration,
    plane: usize,
    width: u32,
    height: u32,
    sub_y: u32,
    bit_depth: u8,
) -> Result<(), Error> {
    if parameters.frame_type[plane] == RestorationType::None {
        return Ok(());
    }
    let unit_size = u32::from(parameters.unit_size[plane]);
    let width = usize::try_from(width).map_err(|_| Error::LimitExceeded)?;
    let height = usize::try_from(height).map_err(|_| Error::LimitExceeded)?;
    let round0 = if bit_depth == 12 { 5u32 } else { 3u32 };
    let round1 = 14 - round0;
    let offset = 1i64 << (u32::from(bit_depth) + 7 - round0 - 1);
    let limit = (1i64 << (u32::from(bit_depth) + 1 + 7 - round0)) - 1;
    let maximum = (1i64 << bit_depth) - 1;
    for y in 0..height {
        let luma_y = u32::try_from(y).map_err(|_| Error::LimitExceeded)? << sub_y;
        let unit_row = (((luma_y + 8) >> sub_y) / unit_size).min(units.rows[plane] - 1);
        let stripe = (luma_y + 8) / 64;
        let stripe_start = (-8i64 + i64::from(stripe) * 64) >> sub_y;
        let stripe_end = stripe_start + i64::from(64 >> sub_y) - 1;
        for x in 0..width {
            let unit_column = (u32::try_from(x).map_err(|_| Error::LimitExceeded)? / unit_size)
                .min(units.columns[plane] - 1);
            let unit = units.unit(plane, unit_row, unit_column)?;
            match unit.kind {
                RestorationType::None => continue,
                RestorationType::Sgrproj => {
                    let source = i64::from(cdef.sample(x, y)?) << 4;
                    let filtered0 = sgr_filtered_sample(
                        deblocked,
                        cdef,
                        x,
                        y,
                        width,
                        height,
                        stripe_start,
                        stripe_end,
                        unit.sgr_set,
                        0,
                        bit_depth,
                    )?;
                    let filtered1 = sgr_filtered_sample(
                        deblocked,
                        cdef,
                        x,
                        y,
                        width,
                        height,
                        stripe_start,
                        stripe_end,
                        unit.sgr_set,
                        1,
                        bit_depth,
                    )?;
                    let w0 = i64::from(unit.sgr_xqd[0]);
                    let w1 = i64::from(unit.sgr_xqd[1]);
                    let w2 = 128 - w0 - w1;
                    let params = SGR_PARAMS[usize::from(unit.sgr_set)];
                    let mut value = w1 * source;
                    value += w0 * if params[0] == 0 { source } else { filtered0 };
                    value += w2 * if params[2] == 0 { source } else { filtered1 };
                    output.set_sample(
                        x,
                        y,
                        u16::try_from(round2(value, 11).clamp(0, maximum))
                            .map_err(|_| Error::LimitExceeded)?,
                    )?;
                    continue;
                }
                RestorationType::Switchable => return Err(Error::InvalidObu),
                RestorationType::Wiener => {}
            }
            let horizontal = wiener_coefficients(unit.wiener[1]);
            let vertical = wiener_coefficients(unit.wiener[0]);
            let mut vertical_sum = 0i64;
            for (vertical_tap, &vertical_coefficient) in vertical.iter().enumerate() {
                let source_y = i64::try_from(y).map_err(|_| Error::LimitExceeded)?
                    + i64::try_from(vertical_tap).map_err(|_| Error::LimitExceeded)?
                    - 3;
                let mut horizontal_sum = 0i64;
                for (horizontal_tap, &horizontal_coefficient) in horizontal.iter().enumerate() {
                    let source_x = i64::try_from(x).map_err(|_| Error::LimitExceeded)?
                        + i64::try_from(horizontal_tap).map_err(|_| Error::LimitExceeded)?
                        - 3;
                    horizontal_sum += i64::from(horizontal_coefficient)
                        * i64::from(source_sample(
                            deblocked,
                            cdef,
                            source_x,
                            source_y,
                            width,
                            height,
                            stripe_start,
                            stripe_end,
                        )?);
                }
                let intermediate = round2(horizontal_sum, round0).clamp(-offset, limit - offset);
                vertical_sum += i64::from(vertical_coefficient) * intermediate;
            }
            output.set_sample(
                x,
                y,
                u16::try_from(round2(vertical_sum, round1).clamp(0, maximum))
                    .map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sgr_filtered_sample(
    deblocked: &Plane,
    cdef: &Plane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stripe_start: i64,
    stripe_end: i64,
    set: u8,
    pass: usize,
    bit_depth: u8,
) -> Result<i64, Error> {
    let parameters = *SGR_PARAMS.get(usize::from(set)).ok_or(Error::InvalidObu)?;
    let radius = parameters[pass * 2];
    if radius == 0 {
        return Ok(i64::from(cdef.sample(x, y)?) << 4);
    }
    let mut a = 0i64;
    let mut b = 0i64;
    let y_i64 = i64::try_from(y).map_err(|_| Error::LimitExceeded)?;
    let x_i64 = i64::try_from(x).map_err(|_| Error::LimitExceeded)?;
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let weight = if pass == 0 {
                if (y_i64 + dy) & 1 != 0 {
                    if dx == 0 { 6 } else { 5 }
                } else {
                    0
                }
            } else if dx == 0 || dy == 0 {
                4
            } else {
                3
            };
            if weight == 0 {
                continue;
            }
            let (sample_a, sample_b) = sgr_ab(
                deblocked,
                cdef,
                x_i64 + dx,
                y_i64 + dy,
                width,
                height,
                stripe_start,
                stripe_end,
                radius,
                parameters[pass * 2 + 1],
                bit_depth,
            )?;
            a += weight * sample_a;
            b += weight * sample_b;
        }
    }
    let shift = if pass == 0 && y & 1 != 0 { 4 } else { 5 };
    let source = i64::from(cdef.sample(x, y)?);
    Ok(round2(a * source + b, 8 + shift - 4))
}

#[allow(clippy::too_many_arguments)]
fn sgr_ab(
    deblocked: &Plane,
    cdef: &Plane,
    x: i64,
    y: i64,
    width: usize,
    height: usize,
    stripe_start: i64,
    stripe_end: i64,
    radius: u8,
    epsilon: u8,
    bit_depth: u8,
) -> Result<(i64, i64), Error> {
    let radius = i64::from(radius);
    let mut squares = 0i64;
    let mut sum = 0i64;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sample = i64::from(source_sample(
                deblocked,
                cdef,
                x + dx,
                y + dy,
                width,
                height,
                stripe_start,
                stripe_end,
            )?);
            squares = squares
                .checked_add(sample * sample)
                .ok_or(Error::LimitExceeded)?;
            sum = sum.checked_add(sample).ok_or(Error::LimitExceeded)?;
        }
    }
    let side = 2 * radius + 1;
    let n = side * side;
    let n2e = n * n * i64::from(epsilon);
    if n2e == 0 {
        return Err(Error::InvalidObu);
    }
    let scale = ((1i64 << 20) + n2e / 2) / n2e;
    let depth_shift = u32::from(bit_depth - 8);
    let scaled_squares = round2(squares, 2 * depth_shift);
    let scaled_sum = round2(sum, depth_shift);
    let variance = (scaled_squares * n - scaled_sum * scaled_sum).max(0);
    let z = round2(variance.checked_mul(scale).ok_or(Error::LimitExceeded)?, 20);
    let a = if z >= 255 {
        256
    } else if z == 0 {
        1
    } else {
        ((z << 8) + z / 2) / (z + 1)
    };
    let reciprocal = ((1i64 << 12) + n / 2) / n;
    let b = round2(
        (256 - a)
            .checked_mul(sum)
            .and_then(|value| value.checked_mul(reciprocal))
            .ok_or(Error::LimitExceeded)?,
        12,
    );
    Ok((a, b))
}

#[allow(clippy::too_many_arguments)]
fn source_sample(
    deblocked: &Plane,
    cdef: &Plane,
    x: i64,
    y: i64,
    width: usize,
    height: usize,
    stripe_start: i64,
    stripe_end: i64,
) -> Result<u16, Error> {
    let x = x.clamp(
        0,
        i64::try_from(width - 1).map_err(|_| Error::LimitExceeded)?,
    );
    let y = y.clamp(
        0,
        i64::try_from(height - 1).map_err(|_| Error::LimitExceeded)?,
    );
    let (source, y) = if y < stripe_start {
        (deblocked, y.max(stripe_start - 2))
    } else if y > stripe_end {
        (deblocked, y.min(stripe_end + 2))
    } else {
        (cdef, y)
    };
    source.sample(
        usize::try_from(x).map_err(|_| Error::LimitExceeded)?,
        usize::try_from(y).map_err(|_| Error::LimitExceeded)?,
    )
}

const fn wiener_coefficients(coefficients: [i16; 3]) -> [i16; 7] {
    [
        coefficients[0],
        coefficients[1],
        coefficients[2],
        128 - 2 * (coefficients[0] + coefficients[1] + coefficients[2]),
        coefficients[2],
        coefficients[1],
        coefficients[0],
    ]
}

const fn round2(value: i64, shift: u32) -> i64 {
    (value + (1 << (shift - 1))) >> shift
}

fn count_units(unit_size: u16, frame_size: u32) -> u32 {
    if unit_size == 0 {
        0
    } else {
        ((frame_size + (u32::from(unit_size) >> 1)) / u32::from(unit_size)).max(1)
    }
}

const fn ceil_div(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

const fn subsampling(sampling: ChromaSampling) -> (u32, u32) {
    match sampling {
        ChromaSampling::Cs400 | ChromaSampling::Cs444 => (0, 0),
        ChromaSampling::Cs420 => (1, 1),
        ChromaSampling::Cs422 => (1, 0),
    }
}

pub fn decode_signed_subexp_with_ref(
    decoder: &mut SymbolDecoder<'_>,
    low: i16,
    high: i16,
    k: u8,
    reference: i16,
) -> Result<i16, Error> {
    if low >= high || reference < low || reference >= high {
        return Err(Error::InvalidObu);
    }
    let symbols =
        u32::try_from(i32::from(high) - i32::from(low)).map_err(|_| Error::LimitExceeded)?;
    let centered_reference =
        u32::try_from(i32::from(reference) - i32::from(low)).map_err(|_| Error::LimitExceeded)?;
    let value = decode_unsigned_subexp_with_ref(decoder, symbols, k, centered_reference)?;
    i16::try_from(i32::from(low) + i32::try_from(value).map_err(|_| Error::LimitExceeded)?)
        .map_err(|_| Error::LimitExceeded)
}

fn decode_unsigned_subexp_with_ref(
    decoder: &mut SymbolDecoder<'_>,
    symbols: u32,
    k: u8,
    reference: u32,
) -> Result<u32, Error> {
    if symbols == 0 || reference >= symbols {
        return Err(Error::InvalidObu);
    }
    let value = decode_subexp(decoder, symbols, k)?;
    if reference.saturating_mul(2) <= symbols {
        inverse_recenter(reference, value)
    } else {
        let mirrored = symbols - 1 - reference;
        Ok(symbols - 1 - inverse_recenter(mirrored, value)?)
    }
}

fn decode_subexp(decoder: &mut SymbolDecoder<'_>, symbols: u32, k: u8) -> Result<u32, Error> {
    if symbols == 0 || k > 30 {
        return Err(Error::InvalidObu);
    }
    let mut index = 0u32;
    let mut base = 0u32;
    loop {
        let bits = if index == 0 {
            u32::from(k)
        } else {
            u32::from(k)
                .checked_add(index - 1)
                .ok_or(Error::LimitExceeded)?
        };
        if bits >= 31 {
            return Err(Error::LimitExceeded);
        }
        let alphabet = 1u32 << bits;
        if symbols <= base.saturating_add(3 * alphabet) {
            return decoder
                .read_ns(symbols - base)?
                .checked_add(base)
                .ok_or(Error::LimitExceeded);
        }
        if decoder.read_bool()? {
            index = index.checked_add(1).ok_or(Error::LimitExceeded)?;
            base = base.checked_add(alphabet).ok_or(Error::LimitExceeded)?;
        } else {
            return u32::try_from(
                decoder.read_literal(u8::try_from(bits).map_err(|_| Error::LimitExceeded)?)?,
            )
            .map_err(|_| Error::LimitExceeded)?
            .checked_add(base)
            .ok_or(Error::LimitExceeded);
        }
    }
}

fn inverse_recenter(reference: u32, value: u32) -> Result<u32, Error> {
    if value > reference.saturating_mul(2) {
        Ok(value)
    } else if value & 1 != 0 {
        reference
            .checked_sub(value.div_ceil(2))
            .ok_or(Error::InvalidObu)
    } else {
        reference.checked_add(value / 2).ok_or(Error::LimitExceeded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_recenter_orders_values_around_reference() {
        let values = (0..8)
            .map(|value| inverse_recenter(3, value).unwrap())
            .collect::<mrml_runtime::Vector<_>>();
        assert_eq!(&values[..], &[3, 2, 4, 1, 5, 0, 6, 7]);
    }

    #[test]
    fn restoration_parameter_ranges_match_the_spec() {
        assert_eq!(WIENER_TAPS_MIN, [-5, -23, -17]);
        assert_eq!(WIENER_TAPS_MAX, [10, 8, 46]);
        assert_eq!(SGRPROJ_XQD_MIN, [-96, -32]);
        assert_eq!(SGRPROJ_XQD_MAX, [31, 95]);
    }

    #[test]
    fn wiener_coefficients_are_symmetric_with_unit_gain() {
        let filter = wiener_coefficients([3, -7, 15]);
        assert_eq!(filter, [3, -7, 15, 106, 15, -7, 3]);
        assert_eq!(filter.iter().sum::<i16>(), 128);
    }

    #[test]
    fn identity_wiener_unit_preserves_a_frame() {
        let parameters = Restoration {
            frame_type: [
                RestorationType::Wiener,
                RestorationType::None,
                RestorationType::None,
            ],
            unit_size: [256, 0, 0],
            ..Restoration::default()
        };
        let input = FrameBuffer::new(16, 9, 8, ChromaSampling::Cs400).unwrap();
        let mut units = RestorationUnits::new(&parameters, 16, 9, ChromaSampling::Cs400).unwrap();
        units.units[0][0].kind = RestorationType::Wiener;
        let output = apply_frame(&input, &input, &units, &parameters, 16, 9).unwrap();
        assert_eq!(output.into_frame(), input.into_frame());
    }

    #[test]
    fn zero_weight_active_sgr_pass_preserves_a_frame() {
        let parameters = Restoration {
            frame_type: [
                RestorationType::Sgrproj,
                RestorationType::None,
                RestorationType::None,
            ],
            unit_size: [256, 0, 0],
            ..Restoration::default()
        };
        let input = FrameBuffer::new(16, 9, 8, ChromaSampling::Cs400).unwrap();
        let mut units = RestorationUnits::new(&parameters, 16, 9, ChromaSampling::Cs400).unwrap();
        units.units[0][0] = RestorationUnit {
            kind: RestorationType::Sgrproj,
            sgr_set: 14,
            sgr_xqd: [0, 95],
            ..RestorationUnit::default()
        };
        let output = apply_frame(&input, &input, &units, &parameters, 16, 9).unwrap();
        assert_eq!(output.into_frame(), input.into_frame());
    }
}
