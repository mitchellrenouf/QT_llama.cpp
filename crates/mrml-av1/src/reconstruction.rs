//! Checked planar storage used by prediction, transforms, and in-loop filters.

use crate::{
    ChromaSampling, Error, Frame,
    block_state::MiGrid,
    cdf::TileCdfs,
    coeff::{
        CoefficientBlockConfig, CoefficientBlockResult, TileCoefficientConfig, TxTypeSelection,
        decode_tile_coefficient_block, effective_scan_size,
    },
    entropy::SymbolDecoder,
    mode::{self, TransformTreeNode},
    palette::PaletteColorMap,
    params::Quantization,
    partition::{BlockRect, BlockSize},
    quant::{
        DequantConfig, ac_quantizer, dc_quantizer, dequantize, write_selected_quantizer_matrix,
    },
    transform::{InverseTransformConfig, TxSize, TxType, inverse_2d},
};
use mrml_runtime::Vector;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedResidualWalkConfig {
    pub block: BlockRect,
    pub size: BlockSize,
    pub luma_tx_size: TxSize,
    pub lossless: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub has_chroma: bool,
    pub mi_columns: u32,
    pub mi_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualBlock {
    pub plane: u8,
    pub x: u32,
    pub y: u32,
    pub size: TxSize,
}

pub fn collect_residual_blocks(
    grid: &MiGrid,
    config: FixedResidualWalkConfig,
    inter: bool,
) -> Result<Vector<ResidualBlock>, Error> {
    let (block_width, block_height) = config.size.dimensions();
    let width_chunks = u32::from((block_width >> 6).max(1));
    let height_chunks = u32::from((block_height >> 6).max(1));
    let chunk_size = if width_chunks > 1 || height_chunks > 1 {
        BlockSize::Block64x64
    } else {
        config.size
    };
    let planes = if config.has_chroma { 3 } else { 1 };
    let mut blocks = Vector::new();
    for chunk_y in 0..height_chunks {
        for chunk_x in 0..width_chunks {
            for plane in 0..planes {
                let chroma = plane != 0;
                let sub_x = u32::from(chroma && config.subsampling_x);
                let sub_y = u32::from(chroma && config.subsampling_y);
                let plane_size = chunk_size.plane_residual_size(
                    chroma && config.subsampling_x,
                    chroma && config.subsampling_y,
                )?;
                let base_x = (config.block.column >> sub_x) * 4;
                let base_y = (config.block.row >> sub_y) * 4;
                let chunk_x4 = (chunk_x << 4) >> sub_x;
                let chunk_y4 = (chunk_y << 4) >> sub_y;
                let start_x = base_x + 4 * chunk_x4;
                let start_y = base_y + 4 * chunk_y4;
                if inter && !config.lossless && plane == 0 {
                    let (width, height) = plane_size.dimensions();
                    mode::walk_inter_transform_tree(
                        grid,
                        TransformTreeNode {
                            start_x,
                            start_y,
                            width: u8::try_from(width).map_err(|_| Error::LimitExceeded)?,
                            height: u8::try_from(height).map_err(|_| Error::LimitExceeded)?,
                            frame_width: config.mi_columns * 4,
                            frame_height: config.mi_rows * 4,
                        },
                        &mut |x, y, size| {
                            blocks
                                .try_push(ResidualBlock {
                                    plane: 0,
                                    x,
                                    y,
                                    size,
                                })
                                .map_err(|_| Error::LimitExceeded)
                        },
                    )?;
                    continue;
                }
                let tx_size = if config.lossless {
                    TxSize::Tx4x4
                } else if plane == 0 {
                    config.luma_tx_size
                } else {
                    chroma_transform_size(plane_size)
                };
                let (plane_width, plane_height) = plane_size.dimensions();
                let (tx_width, tx_height) = tx_size.dimensions();
                let max_x = (config.mi_columns * 4) >> sub_x;
                let max_y = (config.mi_rows * 4) >> sub_y;
                let mut y4 = 0u32;
                while y4 < u32::from(plane_height / 4) {
                    let mut x4 = 0u32;
                    while x4 < u32::from(plane_width / 4) {
                        let x = start_x + 4 * x4;
                        let y = start_y + 4 * y4;
                        if x < max_x && y < max_y {
                            blocks
                                .try_push(ResidualBlock {
                                    plane: plane as u8,
                                    x,
                                    y,
                                    size: tx_size,
                                })
                                .map_err(|_| Error::LimitExceeded)?;
                        }
                        x4 += u32::from(tx_width / 4);
                    }
                    y4 += u32::from(tx_height / 4);
                }
            }
        }
    }
    Ok(blocks)
}

/// Walks the fixed-size branch of section 5.11.34 in normative
/// chunk-Y/chunk-X/plane/Y/X order. Inter luma variable trees are handled by
/// `mode::walk_inter_transform_tree` instead.
pub fn walk_fixed_residual_blocks<F>(
    config: FixedResidualWalkConfig,
    mut visit: F,
) -> Result<(), Error>
where
    F: FnMut(u8, u32, u32, TxSize) -> Result<(), Error>,
{
    let (block_width, block_height) = config.size.dimensions();
    let width_chunks = u32::from((block_width >> 6).max(1));
    let height_chunks = u32::from((block_height >> 6).max(1));
    let chunk_size = if width_chunks > 1 || height_chunks > 1 {
        BlockSize::Block64x64
    } else {
        config.size
    };
    let planes = if config.has_chroma { 3 } else { 1 };
    for chunk_y in 0..height_chunks {
        for chunk_x in 0..width_chunks {
            for plane in 0..planes {
                let chroma = plane != 0;
                let sub_x = u32::from(chroma && config.subsampling_x);
                let sub_y = u32::from(chroma && config.subsampling_y);
                let plane_size = chunk_size.plane_residual_size(
                    chroma && config.subsampling_x,
                    chroma && config.subsampling_y,
                )?;
                let tx_size = if config.lossless {
                    TxSize::Tx4x4
                } else if plane == 0 {
                    config.luma_tx_size
                } else {
                    chroma_transform_size(plane_size)
                };
                let (plane_width, plane_height) = plane_size.dimensions();
                let (tx_width, tx_height) = tx_size.dimensions();
                let base_x = (config.block.column >> sub_x) * 4;
                let base_y = (config.block.row >> sub_y) * 4;
                let chunk_x4 = (chunk_x << 4) >> sub_x;
                let chunk_y4 = (chunk_y << 4) >> sub_y;
                let max_x = (config.mi_columns * 4) >> sub_x;
                let max_y = (config.mi_rows * 4) >> sub_y;
                let mut y4 = 0u32;
                while y4 < u32::from(plane_height / 4) {
                    let mut x4 = 0u32;
                    while x4 < u32::from(plane_width / 4) {
                        let x = base_x + 4 * (chunk_x4 + x4);
                        let y = base_y + 4 * (chunk_y4 + y4);
                        if x < max_x && y < max_y {
                            visit(plane as u8, x, y, tx_size)?;
                        }
                        x4 += u32::from(tx_width / 4);
                    }
                    y4 += u32::from(tx_height / 4);
                }
            }
        }
    }
    Ok(())
}

fn chroma_transform_size(size: BlockSize) -> TxSize {
    let maximum = size.maximum_transform_size();
    match maximum.dimensions() {
        (16, 64) => TxSize::Tx16x32,
        (64, 16) => TxSize::Tx32x16,
        (64, _) | (_, 64) => TxSize::Tx32x32,
        _ => maximum,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plane {
    width: usize,
    height: usize,
    stride: usize,
    samples: Vector<u16>,
}

impl Plane {
    pub fn new(width: usize, height: usize, initial: u16) -> Result<Self, Error> {
        let count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
        let mut samples = Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
        for _ in 0..count {
            samples
                .try_push(initial)
                .map_err(|_| Error::LimitExceeded)?;
        }
        Ok(Self {
            width,
            height,
            stride: width,
            samples,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn sample(&self, x: usize, y: usize) -> Result<u16, Error> {
        let index = self.index(x, y)?;
        self.samples.get(index).copied().ok_or(Error::InvalidObu)
    }

    pub fn set_sample(&mut self, x: usize, y: usize, value: u16) -> Result<(), Error> {
        let index = self.index(x, y)?;
        *self.samples.get_mut(index).ok_or(Error::InvalidObu)? = value;
        Ok(())
    }

    pub fn samples(&self) -> &[u16] {
        &self.samples
    }

    fn index(&self, x: usize, y: usize) -> Result<usize, Error> {
        if x >= self.width || y >= self.height {
            return Err(Error::InvalidObu);
        }
        y.checked_mul(self.stride)
            .and_then(|row| row.checked_add(x))
            .ok_or(Error::LimitExceeded)
    }
}

/// Materializes a decoded palette color map into a reconstruction plane.
pub fn paint_palette_map(
    plane: &mut Plane,
    start_x: usize,
    start_y: usize,
    map: &PaletteColorMap,
    colors: &[u16],
    bit_depth: u8,
) -> Result<(), Error> {
    if colors.len() < 2 || colors.len() > 8 || !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidObu);
    }
    let width = usize::from(map.width);
    let height = usize::from(map.height);
    let expected = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
    if width == 0 || height == 0 || map.indices.len() != expected {
        return Err(Error::InvalidObu);
    }
    let maximum = (1u16 << bit_depth) - 1;
    if colors.iter().any(|&color| color > maximum) {
        return Err(Error::InvalidObu);
    }
    for y in 0..height {
        let output_y = start_y.checked_add(y).ok_or(Error::LimitExceeded)?;
        if output_y >= plane.height() {
            break;
        }
        for x in 0..width {
            let output_x = start_x.checked_add(x).ok_or(Error::LimitExceeded)?;
            if output_x >= plane.width() {
                break;
            }
            let color = *colors
                .get(usize::from(map.indices[y * width + x]))
                .ok_or(Error::InvalidObu)?;
            plane.set_sample(output_x, output_y, color)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub bit_depth: u8,
}

pub fn add_residual(
    plane: &mut Plane,
    region: ResidualRegion,
    residual: &[i32],
) -> Result<(), Error> {
    let ResidualRegion {
        x,
        y,
        width,
        height,
        flip_horizontal,
        flip_vertical,
        bit_depth,
    } = region;
    if !matches!(bit_depth, 8 | 10 | 12)
        || residual.len() != width.checked_mul(height).ok_or(Error::LimitExceeded)?
        || x >= plane.width()
        || y >= plane.height()
    {
        return Err(Error::InvalidObu);
    }
    let visible_width = width.min(plane.width() - x);
    let visible_height = height.min(plane.height() - y);
    let maximum = (1i64 << bit_depth) - 1;
    for output_row in 0..visible_height {
        for output_column in 0..visible_width {
            let source_column = if flip_horizontal {
                width - output_column - 1
            } else {
                output_column
            };
            let source_row = if flip_vertical {
                height - output_row - 1
            } else {
                output_row
            };
            let prediction = i64::from(plane.sample(x + output_column, y + output_row)?);
            let reconstructed = prediction
                .checked_add(i64::from(residual[source_row * width + source_column]))
                .ok_or(Error::LimitExceeded)?
                .clamp(0, maximum) as u16;
            plane.set_sample(x + output_column, y + output_row, reconstructed)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformBlockConfig<'a> {
    pub x: usize,
    pub y: usize,
    pub size: TxSize,
    pub tx_type: TxType,
    pub bit_depth: u8,
    pub lossless: bool,
    pub dc_quantizer: u16,
    pub ac_quantizer: u16,
    pub quantizer_matrix: Option<&'a [u8]>,
}

pub fn reconstruct_transform_block(
    plane: &mut Plane,
    coefficients: &mut [i32],
    config: TransformBlockConfig<'_>,
) -> Result<(), Error> {
    let (width, height) = config.size.dimensions();
    let width = usize::from(width);
    let height = usize::from(height);
    if coefficients.len() != width.checked_mul(height).ok_or(Error::LimitExceeded)? {
        return Err(Error::InvalidObu);
    }
    dequantize(
        coefficients,
        DequantConfig {
            size: config.size,
            bit_depth: config.bit_depth,
            dc_quantizer: config.dc_quantizer,
            ac_quantizer: config.ac_quantizer,
            matrix: config.quantizer_matrix,
        },
    )?;
    let components = config.tx_type.components();
    inverse_2d(
        coefficients,
        InverseTransformConfig {
            width,
            height,
            row: components.row,
            column: components.column,
            row_shift: config.size.row_shift(),
            bit_depth: config.bit_depth,
            lossless: config.lossless,
        },
    )?;
    add_residual(
        plane,
        ResidualRegion {
            x: config.x,
            y: config.y,
            width,
            height,
            flip_horizontal: components.flip_horizontal,
            flip_vertical: components.flip_vertical,
            bit_depth: config.bit_depth,
        },
        coefficients,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizedTransformBlockConfig<'a> {
    pub x: usize,
    pub y: usize,
    pub size: TxSize,
    pub tx_type: TxType,
    pub bit_depth: u8,
    pub lossless: bool,
    pub plane: u8,
    pub qindex: u8,
    pub quantization: &'a Quantization,
}

pub fn reconstruct_quantized_transform_block(
    plane: &mut Plane,
    coefficients: &mut [i32],
    config: QuantizedTransformBlockConfig<'_>,
) -> Result<(), Error> {
    if config.plane >= 3 {
        return Err(Error::InvalidObu);
    }
    let (dc_delta, ac_delta) = match config.plane {
        0 => (config.quantization.delta_q_y_dc, 0),
        1 => (
            config.quantization.delta_q_u_dc,
            config.quantization.delta_q_u_ac,
        ),
        _ => (
            config.quantization.delta_q_v_dc,
            config.quantization.delta_q_v_ac,
        ),
    };
    let dc = dc_quantizer(
        config.bit_depth,
        i16::from(config.qindex) + i16::from(dc_delta),
    )?;
    let ac = ac_quantizer(
        config.bit_depth,
        i16::from(config.qindex) + i16::from(ac_delta),
    )?;
    let mut matrix = [0u8; 1024];
    let matrix_length = write_selected_quantizer_matrix(
        config.quantization,
        config.lossless,
        config.plane,
        config.tx_type,
        config.size,
        &mut matrix,
    )?;
    reconstruct_transform_block(
        plane,
        coefficients,
        TransformBlockConfig {
            x: config.x,
            y: config.y,
            size: config.size,
            tx_type: config.tx_type,
            bit_depth: config.bit_depth,
            lossless: config.lossless,
            dc_quantizer: dc,
            ac_quantizer: ac,
            quantizer_matrix: matrix_length.map(|length| &matrix[..length]),
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedTransformBlockConfig<'a> {
    pub x: usize,
    pub y: usize,
    pub size: TxSize,
    pub tx_type: TxType,
    pub bit_depth: u8,
    pub lossless: bool,
    pub plane: u8,
    pub qindex: u8,
    pub base_q_index: u8,
    pub dc_sign_context: u8,
    pub txb_skip_context: u8,
    pub tx_type_selection: Option<TxTypeSelection>,
    pub quantization: &'a Quantization,
}

pub fn decode_and_reconstruct_transform_block(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    plane: &mut Plane,
    config: DecodedTransformBlockConfig<'_>,
) -> Result<CoefficientBlockResult, Error> {
    let effective = effective_scan_size(config.size);
    let (effective_width, effective_height) = effective.dimensions();
    let effective_width = usize::from(effective_width);
    let effective_count = effective_width
        .checked_mul(usize::from(effective_height))
        .ok_or(Error::LimitExceeded)?;
    let mut compact = [0i32; 1024];
    let result = decode_tile_coefficient_block(
        decoder,
        cdfs,
        TileCoefficientConfig {
            block: CoefficientBlockConfig {
                size: config.size,
                tx_type: config.tx_type,
                dc_sign_context: config.dc_sign_context,
            },
            base_q_index: config.base_q_index,
            chroma: config.plane > 0,
            txb_skip_context: config.txb_skip_context,
            tx_type_selection: config.tx_type_selection,
        },
        &mut compact[..effective_count],
    )?;
    let tx_type = result.tx_type;
    let (width, height) = config.size.dimensions();
    let width = usize::from(width);
    let height = usize::from(height);
    let mut coefficients = [0i32; 4096];
    for row in 0..usize::from(effective_height) {
        for column in 0..effective_width {
            coefficients[row * width + column] = compact[row * effective_width + column];
        }
    }
    reconstruct_quantized_transform_block(
        plane,
        &mut coefficients[..width.checked_mul(height).ok_or(Error::LimitExceeded)?],
        QuantizedTransformBlockConfig {
            x: config.x,
            y: config.y,
            size: config.size,
            tx_type,
            bit_depth: config.bit_depth,
            lossless: config.lossless,
            plane: config.plane,
            qindex: config.qindex,
            quantization: config.quantization,
        },
    )?;
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameBuffer {
    pub y: Plane,
    pub u: Option<Plane>,
    pub v: Option<Plane>,
    width: u32,
    height: u32,
    bit_depth: u8,
    sampling: ChromaSampling,
}

impl FrameBuffer {
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    pub const fn sampling(&self) -> ChromaSampling {
        self.sampling
    }

    pub fn new(
        width: u32,
        height: u32,
        bit_depth: u8,
        sampling: ChromaSampling,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 || !matches!(bit_depth, 8 | 10 | 12) {
            return Err(Error::InvalidObu);
        }
        let width_usize = usize::try_from(width.div_ceil(8).saturating_mul(8))
            .map_err(|_| Error::LimitExceeded)?;
        let height_usize = usize::try_from(height.div_ceil(8).saturating_mul(8))
            .map_err(|_| Error::LimitExceeded)?;
        let midpoint = 1u16 << (bit_depth - 1);
        let y = Plane::new(width_usize, height_usize, midpoint)?;
        let chroma_dimensions = match sampling {
            ChromaSampling::Cs400 => None,
            ChromaSampling::Cs420 => Some((width_usize.div_ceil(2), height_usize.div_ceil(2))),
            ChromaSampling::Cs422 => Some((width_usize.div_ceil(2), height_usize)),
            ChromaSampling::Cs444 => Some((width_usize, height_usize)),
        };
        let (u, v) = if let Some((chroma_width, chroma_height)) = chroma_dimensions {
            (
                Some(Plane::new(chroma_width, chroma_height, midpoint)?),
                Some(Plane::new(chroma_width, chroma_height, midpoint)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            y,
            u,
            v,
            width,
            height,
            bit_depth,
            sampling,
        })
    }

    pub fn from_frame(frame: &Frame) -> Result<Self, Error> {
        let mut buffer = Self::new(
            frame.width,
            frame.height,
            frame.bit_depth,
            frame.chroma_sampling,
        )?;
        decode_plane_region(
            &mut buffer.y,
            &frame.y,
            usize::try_from(frame.width).map_err(|_| Error::LimitExceeded)?,
            usize::try_from(frame.height).map_err(|_| Error::LimitExceeded)?,
            frame.bit_depth,
        )?;
        let logical_width = usize::try_from(frame.width).map_err(|_| Error::LimitExceeded)?;
        let logical_height = usize::try_from(frame.height).map_err(|_| Error::LimitExceeded)?;
        let chroma_dimensions = match frame.chroma_sampling {
            ChromaSampling::Cs400 => None,
            ChromaSampling::Cs420 => Some((logical_width.div_ceil(2), logical_height.div_ceil(2))),
            ChromaSampling::Cs422 => Some((logical_width.div_ceil(2), logical_height)),
            ChromaSampling::Cs444 => Some((logical_width, logical_height)),
        };
        match (
            buffer.u.as_mut(),
            buffer.v.as_mut(),
            chroma_dimensions,
            frame.u.is_empty(),
            frame.v.is_empty(),
        ) {
            (None, None, None, true, true) => {}
            (Some(u), Some(v), Some((width, height)), false, false) => {
                decode_plane_region(u, &frame.u, width, height, frame.bit_depth)?;
                decode_plane_region(v, &frame.v, width, height, frame.bit_depth)?;
            }
            _ => return Err(Error::InvalidObu),
        }
        Ok(buffer)
    }

    pub fn into_frame(self) -> Result<Frame, Error> {
        let logical_width = usize::try_from(self.width).map_err(|_| Error::LimitExceeded)?;
        let logical_height = usize::try_from(self.height).map_err(|_| Error::LimitExceeded)?;
        let chroma_dimensions = match self.sampling {
            ChromaSampling::Cs400 => None,
            ChromaSampling::Cs420 => Some((logical_width.div_ceil(2), logical_height.div_ceil(2))),
            ChromaSampling::Cs422 => Some((logical_width.div_ceil(2), logical_height)),
            ChromaSampling::Cs444 => Some((logical_width, logical_height)),
        };
        Ok(Frame {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            chroma_sampling: self.sampling,
            y: encode_plane_region(&self.y, logical_width, logical_height, self.bit_depth)?,
            u: encode_optional_plane_region(self.u.as_ref(), chroma_dimensions, self.bit_depth)?,
            v: encode_optional_plane_region(self.v.as_ref(), chroma_dimensions, self.bit_depth)?,
            presentation_time: None,
            buffer_removal_times: Vector::new(),
        })
    }
}

fn decode_plane_region(
    plane: &mut Plane,
    encoded: &[u8],
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Result<(), Error> {
    if width == 0 || height == 0 || width > plane.width() || height > plane.height() {
        return Err(Error::InvalidObu);
    }
    let bytes_per_sample = if bit_depth == 8 { 1 } else { 2 };
    let expected = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(bytes_per_sample))
        .ok_or(Error::LimitExceeded)?;
    if encoded.len() != expected {
        return Err(Error::InvalidObu);
    }
    let maximum = (1u16 << bit_depth) - 1;
    for y in 0..plane.height() {
        let source_y = y.min(height - 1);
        for x in 0..plane.width() {
            let source_x = x.min(width - 1);
            let offset = (source_y * width + source_x) * bytes_per_sample;
            let sample = if bytes_per_sample == 1 {
                u16::from(encoded[offset])
            } else {
                u16::from_le_bytes([encoded[offset], encoded[offset + 1]])
            };
            if sample > maximum {
                return Err(Error::InvalidObu);
            }
            plane.set_sample(x, y, sample)?;
        }
    }
    Ok(())
}

fn encode_optional_plane_region(
    plane: Option<&Plane>,
    dimensions: Option<(usize, usize)>,
    bit_depth: u8,
) -> Result<Vector<u8>, Error> {
    match (plane, dimensions) {
        (Some(plane), Some((width, height))) => {
            encode_plane_region(plane, width, height, bit_depth)
        }
        (None, None) => Ok(Vector::new()),
        _ => Err(Error::InvalidObu),
    }
}

fn encode_plane_region(
    plane: &Plane,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Result<Vector<u8>, Error> {
    if width == 0 || height == 0 || width > plane.width() || height > plane.height() {
        return Err(Error::InvalidObu);
    }
    let bytes_per_sample = if bit_depth == 8 { 1 } else { 2 };
    let capacity = width
        .checked_mul(height)
        .ok_or(Error::LimitExceeded)?
        .checked_mul(bytes_per_sample)
        .ok_or(Error::LimitExceeded)?;
    let mut output = Vector::with_capacity(capacity).map_err(|_| Error::LimitExceeded)?;
    let maximum = (1u16 << bit_depth) - 1;
    for row in 0..height {
        for column in 0..width {
            let sample = plane.sample(column, row)?;
            if sample > maximum {
                return Err(Error::InvalidObu);
            }
            output
                .try_push(sample as u8)
                .map_err(|_| Error::LimitExceeded)?;
            if bytes_per_sample == 2 {
                output
                    .try_push((sample >> 8) as u8)
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_residual_walk_keeps_chunk_then_plane_order() {
        let mut visits = Vector::new();
        walk_fixed_residual_blocks(
            FixedResidualWalkConfig {
                block: BlockRect::new(0, 0, BlockSize::Block128x128),
                size: BlockSize::Block128x128,
                luma_tx_size: TxSize::Tx64x64,
                lossless: false,
                subsampling_x: true,
                subsampling_y: true,
                has_chroma: true,
                mi_columns: 32,
                mi_rows: 32,
            },
            |plane, x, y, size| {
                visits
                    .try_push((plane, x, y, size))
                    .map_err(|_| Error::LimitExceeded)
            },
        )
        .unwrap();
        assert_eq!(visits.len(), 12);
        assert_eq!(visits[0], (0, 0, 0, TxSize::Tx64x64));
        assert_eq!(visits[1], (1, 0, 0, TxSize::Tx32x32));
        assert_eq!(visits[2], (2, 0, 0, TxSize::Tx32x32));
        assert_eq!(visits[3], (0, 64, 0, TxSize::Tx64x64));
        assert_eq!(visits[11], (2, 32, 32, TxSize::Tx32x32));
    }

    #[test]
    fn inter_residual_collection_expands_luma_transform_trees_before_chroma() {
        let mut grid = MiGrid::new(32, 32).unwrap();
        grid.fill(
            BlockRect::new(0, 0, BlockSize::Block128x128),
            crate::block_state::BlockState {
                size: Some(BlockSize::Block128x128),
                is_inter: true,
                ..crate::block_state::BlockState::default()
            },
        )
        .unwrap();
        for row in (0..32).step_by(8) {
            for column in (0..32).step_by(8) {
                grid.fill_tx_size(row, column, TxSize::Tx32x32).unwrap();
            }
        }
        let blocks = collect_residual_blocks(
            &grid,
            FixedResidualWalkConfig {
                block: BlockRect::new(0, 0, BlockSize::Block128x128),
                size: BlockSize::Block128x128,
                luma_tx_size: TxSize::Tx32x32,
                lossless: false,
                subsampling_x: true,
                subsampling_y: true,
                has_chroma: true,
                mi_columns: 32,
                mi_rows: 32,
            },
            true,
        )
        .unwrap();
        assert_eq!(blocks.len(), 24);
        assert!(blocks[..4].iter().all(|block| block.plane == 0));
        assert_eq!(blocks[4].plane, 1);
        assert_eq!(blocks[5].plane, 2);
        assert_eq!((blocks[6].x, blocks[6].y), (64, 0));
    }

    #[test]
    fn odd_420_dimensions_round_up_chroma() {
        let buffer = FrameBuffer::new(5, 3, 8, ChromaSampling::Cs420).unwrap();
        assert_eq!((buffer.y.width(), buffer.y.height()), (8, 8));
        assert_eq!(
            (
                buffer.u.as_ref().unwrap().width(),
                buffer.u.as_ref().unwrap().height()
            ),
            (4, 4)
        );
        let frame = buffer.into_frame().unwrap();
        assert_eq!(frame.y.len(), 5 * 3);
        assert_eq!(frame.u.len(), 3 * 2);
    }

    #[test]
    fn frame_import_decodes_samples_and_extends_visible_edges() {
        let frame = Frame {
            width: 2,
            height: 2,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs400,
            y: [1, 2, 3, 4].into_iter().collect(),
            u: Vector::new(),
            v: Vector::new(),
            presentation_time: None,
            buffer_removal_times: Vector::new(),
        };
        let buffer = FrameBuffer::from_frame(&frame).unwrap();
        assert_eq!(buffer.y.sample(0, 0), Ok(1));
        assert_eq!(buffer.y.sample(1, 1), Ok(4));
        assert_eq!(buffer.y.sample(7, 7), Ok(4));
        assert_eq!(buffer.into_frame(), Ok(frame));
    }

    #[test]
    fn high_bit_depth_output_is_little_endian() {
        let mut buffer = FrameBuffer::new(1, 1, 10, ChromaSampling::Cs400).unwrap();
        buffer.y.set_sample(0, 0, 0x02aa).unwrap();
        let frame = buffer.into_frame().unwrap();
        assert_eq!(&frame.y[..], &[0xaa, 0x02]);
        assert!(frame.u.is_empty());
        assert!(frame.v.is_empty());
    }

    #[test]
    fn plane_access_rejects_out_of_bounds_coordinates() {
        let mut plane = Plane::new(2, 2, 0).unwrap();
        assert_eq!(plane.sample(2, 0), Err(Error::InvalidObu));
        assert_eq!(plane.set_sample(0, 2, 1), Err(Error::InvalidObu));
    }

    #[test]
    fn decoded_palette_map_is_clipped_and_materialized() {
        let mut plane = Plane::new(3, 2, 0).unwrap();
        let map = PaletteColorMap {
            width: 3,
            height: 2,
            indices: [0, 1, 0, 1, 0, 1].into_iter().collect(),
        };
        paint_palette_map(&mut plane, 1, 0, &map, &[17, 203], 8).unwrap();
        assert_eq!(plane.samples(), &[0, 17, 203, 0, 203, 17]);
        assert_eq!(
            paint_palette_map(&mut plane, 0, 0, &map, &[17, 300], 8),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn residual_addition_clips_and_flips() {
        let mut plane = Plane::new(2, 1, 100).unwrap();
        add_residual(
            &mut plane,
            ResidualRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
                flip_horizontal: true,
                flip_vertical: false,
                bit_depth: 8,
            },
            &[-200, 300],
        )
        .unwrap();
        assert_eq!(plane.samples(), &[255, 0]);
    }

    #[test]
    fn residual_addition_clips_transform_footprint_at_frame_edges() {
        let mut plane = Plane::new(3, 2, 10).unwrap();
        let residual: [i32; 16] = core::array::from_fn(|index| index as i32 + 1);
        add_residual(
            &mut plane,
            ResidualRegion {
                x: 1,
                y: 1,
                width: 4,
                height: 4,
                flip_horizontal: false,
                flip_vertical: false,
                bit_depth: 8,
            },
            &residual,
        )
        .unwrap();
        assert_eq!(plane.sample(1, 1), Ok(11));
        assert_eq!(plane.sample(2, 1), Ok(12));
    }

    #[test]
    fn transform_block_bridge_dequantizes_inverts_and_adds() {
        let mut plane = Plane::new(4, 4, 128).unwrap();
        let mut coefficients = [0i32; 16];
        coefficients[0] = 16;
        reconstruct_transform_block(
            &mut plane,
            &mut coefficients,
            TransformBlockConfig {
                x: 0,
                y: 0,
                size: TxSize::Tx4x4,
                tx_type: TxType::DctDct,
                bit_depth: 8,
                lossless: false,
                dc_quantizer: 4,
                ac_quantizer: 4,
                quantizer_matrix: None,
            },
        )
        .unwrap();
        assert!(plane.samples().iter().all(|&sample| sample <= 255));
        assert!(plane.samples().iter().any(|&sample| sample != 128));
    }

    #[test]
    fn quantized_transform_bridge_selects_plane_quantizers_and_matrix() {
        let mut plane = Plane::new(4, 4, 128).unwrap();
        let mut coefficients = [0i32; 16];
        coefficients[0] = 8;
        let quantization = Quantization {
            base_q_idx: 20,
            using_qmatrix: true,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
            ..Quantization::default()
        };
        reconstruct_quantized_transform_block(
            &mut plane,
            &mut coefficients,
            QuantizedTransformBlockConfig {
                x: 0,
                y: 0,
                size: TxSize::Tx4x4,
                tx_type: TxType::DctDct,
                bit_depth: 8,
                lossless: false,
                plane: 0,
                qindex: quantization.base_q_idx,
                quantization: &quantization,
            },
        )
        .unwrap();
        assert!(plane.samples().iter().any(|&sample| sample != 128));
    }

    #[test]
    fn entropy_transform_bridge_decodes_directly_into_plane_storage() {
        let quantization = Quantization {
            base_q_idx: 20,
            ..Quantization::default()
        };
        let mut plane = Plane::new(4, 4, 128).unwrap();
        let mut decoder = SymbolDecoder::new(&[0; 64], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let result = decode_and_reconstruct_transform_block(
            &mut decoder,
            &mut cdfs,
            &mut plane,
            DecodedTransformBlockConfig {
                x: 0,
                y: 0,
                size: TxSize::Tx4x4,
                tx_type: TxType::DctDct,
                bit_depth: 8,
                lossless: false,
                plane: 0,
                qindex: 20,
                base_q_index: 20,
                dc_sign_context: 0,
                txb_skip_context: 0,
                tx_type_selection: None,
                quantization: &quantization,
            },
        )
        .unwrap();
        assert!(result.eob <= 16);
        assert!(plane.samples().iter().all(|&sample| sample <= 255));
    }
}
