//! CPU encoder bitstream foundations.
//!
//! This module writes canonical AV1 syntax; mode decision, transforms,
//! quantization, entropy coding, and loop filtering are intentionally separate
//! future stages.  Keeping syntax generation independent lets those stages be
//! tested without a container or native codec library.

use crate::{
    ChromaSampling, Error, ObuType, ScalabilityMetadata, TimecodeMetadata, vector_extend,
    vector_push, write_obu,
};
use mrml_runtime::Vector;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceConfig {
    pub profile: u8,
    pub width: u16,
    pub height: u16,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub chroma_sampling: ChromaSampling,
    pub level: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoSequenceConfig {
    pub sequence: SequenceConfig,
    pub tier: bool,
    pub use_128x128_superblock: bool,
    pub enable_superres: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReducedStillFrameConfig {
    pub sequence: SequenceConfig,
    pub reduced_tx_set: bool,
}

impl Default for VideoSequenceConfig {
    fn default() -> Self {
        Self {
            sequence: SequenceConfig::default(),
            tier: false,
            use_128x128_superblock: false,
            enable_superres: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObuStream {
    bytes: Vector<u8>,
}

impl ObuStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(
        &mut self,
        kind: ObuType,
        temporal_id: u8,
        spatial_id: u8,
        payload: &[u8],
    ) -> Result<(), Error> {
        let obu = write_obu(kind, temporal_id, spatial_id, payload)?;
        vector_extend(&mut self.bytes, &obu)
    }

    pub fn push_encoded(&mut self, obu: &[u8]) -> Result<(), Error> {
        vector_extend(&mut self.bytes, obu)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn finish(self) -> Vector<u8> {
        self.bytes
    }
}

pub fn temporal_delimiter() -> Result<Vector<u8>, Error> {
    write_obu(ObuType::TemporalDelimiter, 0, 0, &[])
}

pub fn hdr_content_light_level(max_cll: u16, max_fall: u16) -> Result<Vector<u8>, Error> {
    let mut payload = Vector::with_capacity(6).map_err(|_| Error::LimitExceeded)?;
    vector_push(&mut payload, 1)?;
    vector_extend(&mut payload, &max_cll.to_be_bytes())?;
    vector_extend(&mut payload, &max_fall.to_be_bytes())?;
    vector_push(&mut payload, 0x80)?;
    write_obu(ObuType::Metadata, 0, 0, &payload)
}

pub fn hdr_mastering_display_color_volume(
    primaries_x: [u16; 3],
    primaries_y: [u16; 3],
    white_point_x: u16,
    white_point_y: u16,
    luminance_max: u32,
    luminance_min: u32,
) -> Result<Vector<u8>, Error> {
    let mut payload = Vector::with_capacity(26).map_err(|_| Error::LimitExceeded)?;
    vector_push(&mut payload, 2)?;
    for index in 0..3 {
        vector_extend(&mut payload, &primaries_x[index].to_be_bytes())?;
        vector_extend(&mut payload, &primaries_y[index].to_be_bytes())?;
    }
    vector_extend(&mut payload, &white_point_x.to_be_bytes())?;
    vector_extend(&mut payload, &white_point_y.to_be_bytes())?;
    vector_extend(&mut payload, &luminance_max.to_be_bytes())?;
    vector_extend(&mut payload, &luminance_min.to_be_bytes())?;
    vector_push(&mut payload, 0x80)?;
    write_obu(ObuType::Metadata, 0, 0, &payload)
}

pub fn itu_t35(
    country_code: u8,
    country_code_extension: Option<u8>,
    bytes: &[u8],
) -> Result<Vector<u8>, Error> {
    if (country_code == 0xff) != country_code_extension.is_some() {
        return Err(Error::InvalidObu);
    }
    let mut payload =
        Vector::with_capacity(bytes.len().saturating_add(4)).map_err(|_| Error::LimitExceeded)?;
    vector_push(&mut payload, 4)?;
    vector_push(&mut payload, country_code)?;
    if let Some(extension) = country_code_extension {
        vector_push(&mut payload, extension)?;
    }
    vector_extend(&mut payload, bytes)?;
    vector_push(&mut payload, 0x80)?;
    write_obu(ObuType::Metadata, 0, 0, &payload)
}

pub fn timecode(value: TimecodeMetadata) -> Result<Vector<u8>, Error> {
    if value.counting_type > 31
        || value.frames > 511
        || value.seconds.is_some_and(|part| part > 59)
        || value.minutes.is_some_and(|part| part > 59)
        || value.hours.is_some_and(|part| part > 23)
        || value.time_offset_length > 31
        || (value.full_timestamp
            && !(value.seconds.is_some() && value.minutes.is_some() && value.hours.is_some()))
        || (value.time_offset_length == 0 && value.time_offset != 0)
        || (!value.full_timestamp
            && (value.hours.is_some() && value.minutes.is_none()
                || value.minutes.is_some() && value.seconds.is_none()))
    {
        return Err(Error::InvalidObu);
    }
    let mut bits = BitWriter::new()?;
    bits.write(u64::from(value.counting_type), 5);
    bits.bit(value.full_timestamp);
    bits.bit(value.discontinuity);
    bits.bit(value.count_dropped);
    bits.write(u64::from(value.frames), 9);
    if value.full_timestamp {
        bits.write(u64::from(value.seconds.ok_or(Error::InvalidObu)?), 6);
        bits.write(u64::from(value.minutes.ok_or(Error::InvalidObu)?), 6);
        bits.write(u64::from(value.hours.ok_or(Error::InvalidObu)?), 5);
    } else {
        bits.bit(value.seconds.is_some());
        if let Some(seconds) = value.seconds {
            bits.write(u64::from(seconds), 6);
            bits.bit(value.minutes.is_some());
            if let Some(minutes) = value.minutes {
                bits.write(u64::from(minutes), 6);
                bits.bit(value.hours.is_some());
                if let Some(hours) = value.hours {
                    bits.write(u64::from(hours), 5);
                }
            }
        }
    }
    bits.write(u64::from(value.time_offset_length), 5);
    if value.time_offset_length > 0 {
        let minimum = -(1i64 << (value.time_offset_length - 1));
        let maximum = (1i64 << (value.time_offset_length - 1)) - 1;
        let offset = i64::from(value.time_offset);
        if !(minimum..=maximum).contains(&offset) {
            return Err(Error::InvalidObu);
        }
        let mask = (1u64 << value.time_offset_length) - 1;
        bits.write((offset as u64) & mask, value.time_offset_length);
    }
    bits.trailing_bits();
    let mut payload =
        Vector::with_capacity(1 + bits.bytes.len()).map_err(|_| Error::LimitExceeded)?;
    vector_push(&mut payload, 5)?;
    vector_extend(&mut payload, &bits.finish()?)?;
    write_obu(ObuType::Metadata, 0, 0, &payload)
}

pub fn scalability(value: &ScalabilityMetadata) -> Result<Vector<u8>, Error> {
    const SCALABILITY_SS: u8 = 14;
    if (value.mode_idc == SCALABILITY_SS) != value.structure.is_some() {
        return Err(Error::InvalidObu);
    }
    let mut bits = BitWriter::with_capacity(4096)?;
    bits.write(u64::from(value.mode_idc), 8);
    if let Some(structure) = &value.structure {
        let layer_count = structure.spatial_layers.len();
        if !(1..=4).contains(&layer_count) || structure.temporal_groups.len() > 255 {
            return Err(Error::InvalidObu);
        }
        let dimensions_present = structure
            .spatial_layers
            .iter()
            .all(|layer| layer.maximum_width.is_some() && layer.maximum_height.is_some());
        let dimensions_absent = structure
            .spatial_layers
            .iter()
            .all(|layer| layer.maximum_width.is_none() && layer.maximum_height.is_none());
        let descriptions_present = structure
            .spatial_layers
            .iter()
            .all(|layer| layer.reference_id.is_some());
        let descriptions_absent = structure
            .spatial_layers
            .iter()
            .all(|layer| layer.reference_id.is_none());
        if !dimensions_present && !dimensions_absent
            || !descriptions_present && !descriptions_absent
        {
            return Err(Error::InvalidObu);
        }
        bits.write(
            u64::try_from(layer_count - 1).map_err(|_| Error::LimitExceeded)?,
            2,
        );
        bits.bit(dimensions_present);
        bits.bit(descriptions_present);
        bits.bit(!structure.temporal_groups.is_empty());
        bits.write(0, 3);
        if dimensions_present {
            for layer in &structure.spatial_layers {
                bits.write(u64::from(layer.maximum_width.ok_or(Error::InvalidObu)?), 16);
                bits.write(
                    u64::from(layer.maximum_height.ok_or(Error::InvalidObu)?),
                    16,
                );
            }
        }
        if descriptions_present {
            for layer in &structure.spatial_layers {
                bits.write(u64::from(layer.reference_id.ok_or(Error::InvalidObu)?), 8);
            }
        }
        if !structure.temporal_groups.is_empty() {
            bits.write(
                u64::try_from(structure.temporal_groups.len()).map_err(|_| Error::LimitExceeded)?,
                8,
            );
            for group in &structure.temporal_groups {
                if group.temporal_id > 7 || group.reference_picture_differences.len() > 7 {
                    return Err(Error::InvalidObu);
                }
                bits.write(u64::from(group.temporal_id), 3);
                bits.bit(group.temporal_switching_up_point);
                bits.bit(group.spatial_switching_up_point);
                bits.write(
                    u64::try_from(group.reference_picture_differences.len())
                        .map_err(|_| Error::LimitExceeded)?,
                    3,
                );
                for difference in &group.reference_picture_differences {
                    bits.write(u64::from(*difference), 8);
                }
            }
        }
    }
    bits.trailing_bits();
    let encoded = bits.finish()?;
    let mut payload =
        Vector::with_capacity(encoded.len().saturating_add(1)).map_err(|_| Error::LimitExceeded)?;
    vector_push(&mut payload, 3)?;
    vector_extend(&mut payload, &encoded)?;
    write_obu(ObuType::Metadata, 0, 0, &payload)
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            profile: 0,
            width: 640,
            height: 480,
            bit_depth: 8,
            monochrome: false,
            chroma_sampling: ChromaSampling::Cs420,
            level: 0,
        }
    }
}

/// Emit a reduced-still-picture sequence header OBU. This is the first encoder
/// primitive: it is also useful for independently exercising decoder headers.
pub fn sequence_header(config: SequenceConfig) -> Result<Vector<u8>, Error> {
    validate(config)?;
    let mut bits = BitWriter::new()?;
    bits.write(config.profile as u64, 3);
    bits.bit(true); // still_picture
    bits.bit(true); // reduced_still_picture_header
    bits.write(config.level as u64, 5);
    let width_bits = bits_required(u32::from(config.width) - 1);
    let height_bits = bits_required(u32::from(config.height) - 1);
    bits.write(u64::from(width_bits - 1), 4);
    bits.write(u64::from(height_bits - 1), 4);
    bits.write(u64::from(config.width - 1), width_bits);
    bits.write(u64::from(config.height - 1), height_bits);
    bits.bit(false); // use_128x128_superblock
    bits.bit(true); // enable_filter_intra
    bits.bit(true); // enable_intra_edge_filter
    bits.bit(false); // enable_superres
    bits.bit(true); // enable_cdef
    bits.bit(true); // enable_restoration
    bits.bit(config.bit_depth > 8);
    if config.profile == 2 && config.bit_depth > 8 {
        bits.bit(config.bit_depth == 12);
    }
    if config.profile != 1 {
        bits.bit(config.monochrome);
    }
    bits.bit(false); // color_description_present_flag
    if config.monochrome {
        bits.bit(false); // color_range
    } else {
        bits.bit(false); // color_range
        if config.profile == 2 && config.bit_depth == 12 {
            let (sx, sy) = match config.chroma_sampling {
                ChromaSampling::Cs420 => (true, true),
                ChromaSampling::Cs422 => (true, false),
                ChromaSampling::Cs444 => (false, false),
                ChromaSampling::Cs400 => unreachable!(),
            };
            bits.bit(sx);
            if sx {
                bits.bit(sy);
            }
        }
        if config.chroma_sampling == ChromaSampling::Cs420 {
            bits.write(0, 2);
        }
        bits.bit(false); // separate_uv_delta_q
    }
    bits.bit(false); // film_grain_params_present
    bits.trailing_bits();
    write_obu(ObuType::SequenceHeader, 0, 0, &bits.finish()?)
}

/// Emit a canonical single-operating-point video sequence header. All
/// normative inter tools are advertised so later encoder stages can select
/// them per frame without rewriting sequence state.
pub fn video_sequence_header(config: VideoSequenceConfig) -> Result<Vector<u8>, Error> {
    validate(config.sequence)?;
    if config.tier && config.sequence.level <= 7 {
        return Err(Error::InvalidSequence);
    }
    let sequence = config.sequence;
    let mut bits = BitWriter::new()?;
    bits.write(u64::from(sequence.profile), 3);
    bits.bit(false); // still_picture
    bits.bit(false); // reduced_still_picture_header
    bits.bit(false); // timing_info_present_flag
    bits.bit(false); // initial_display_delay_present_flag
    bits.write(0, 5); // operating_points_cnt_minus_1
    bits.write(0, 12); // operating_point_idc[0]
    bits.write(u64::from(sequence.level), 5);
    if sequence.level > 7 {
        bits.bit(config.tier);
    }
    let width_bits = bits_required(u32::from(sequence.width) - 1);
    let height_bits = bits_required(u32::from(sequence.height) - 1);
    bits.write(u64::from(width_bits - 1), 4);
    bits.write(u64::from(height_bits - 1), 4);
    bits.write(u64::from(sequence.width - 1), width_bits);
    bits.write(u64::from(sequence.height - 1), height_bits);
    bits.bit(false); // frame_id_numbers_present_flag
    bits.bit(config.use_128x128_superblock);
    bits.bit(true); // enable_filter_intra
    bits.bit(true); // enable_intra_edge_filter
    bits.bit(true); // enable_interintra_compound
    bits.bit(true); // enable_masked_compound
    bits.bit(true); // enable_warped_motion
    bits.bit(true); // enable_dual_filter
    bits.bit(true); // enable_order_hint
    bits.bit(true); // enable_jnt_comp
    bits.bit(true); // enable_ref_frame_mvs
    bits.bit(false); // seq_choose_screen_content_tools
    bits.bit(false); // seq_force_screen_content_tools = 0
    bits.write(6, 3); // order_hint_bits_minus_1 (7-bit hints)
    bits.bit(config.enable_superres);
    bits.bit(true); // enable_cdef
    bits.bit(true); // enable_restoration
    write_color_config(&mut bits, sequence);
    bits.bit(false); // film_grain_params_present
    bits.trailing_bits();
    write_obu(ObuType::SequenceHeader, 0, 0, &bits.finish()?)
}

/// Emit a lossless reduced-still key-frame header with a uniform tile layout.
/// The returned OBU intentionally does not contain coded tile data.
pub fn reduced_still_frame_header(config: ReducedStillFrameConfig) -> Result<Vector<u8>, Error> {
    validate(config.sequence)?;
    let mut bits = BitWriter::new()?;
    bits.bit(true); // disable_cdf_update
    bits.bit(false); // allow_screen_content_tools
    bits.bit(false); // render_and_frame_size_different
    write_uniform_tile_info(
        &mut bits,
        config.sequence.width,
        config.sequence.height,
        false,
    );
    bits.write(0, 8); // base_q_idx: lossless
    bits.bit(false); // delta_q_y_dc coded
    if !config.sequence.monochrome {
        bits.bit(false); // delta_q_u_dc coded
        bits.bit(false); // delta_q_u_ac coded
    }
    bits.bit(false); // using_qmatrix
    bits.bit(false); // segmentation_enabled
    // Delta-Q, loop filter, CDEF, restoration, and tx_mode carry no syntax in
    // the derived coded-lossless state.
    bits.bit(config.reduced_tx_set);
    bits.trailing_bits();
    write_obu(ObuType::FrameHeader, 0, 0, &bits.finish()?)
}

fn write_uniform_tile_info(bits: &mut BitWriter, width: u16, height: u16, use_128x128: bool) {
    const MAX_TILE_WIDTH: u32 = 4096;
    const MAX_TILE_AREA: u32 = 4096 * 2304;
    let mi_columns = 2 * ((u32::from(width) + 7) >> 3);
    let mi_rows = 2 * ((u32::from(height) + 7) >> 3);
    let superblock_shift = if use_128x128 { 5 } else { 4 };
    let superblock_size_log2 = superblock_shift + 2;
    let superblock_columns = (mi_columns + (1 << superblock_shift) - 1) >> superblock_shift;
    let superblock_rows = (mi_rows + (1 << superblock_shift) - 1) >> superblock_shift;
    let maximum_width_superblocks = MAX_TILE_WIDTH >> superblock_size_log2;
    let maximum_area_superblocks = MAX_TILE_AREA >> (2 * superblock_size_log2);
    let minimum_log2_columns = tile_log2(maximum_width_superblocks, superblock_columns);
    let maximum_log2_columns = tile_log2(1, superblock_columns.min(64));
    let maximum_log2_rows = tile_log2(1, superblock_rows.min(64));
    let minimum_log2_tiles = minimum_log2_columns.max(tile_log2(
        maximum_area_superblocks,
        superblock_rows.saturating_mul(superblock_columns),
    ));
    bits.bit(true); // uniform_tile_spacing_flag
    if minimum_log2_columns < maximum_log2_columns {
        bits.bit(false);
    }
    let log2_rows = minimum_log2_tiles.saturating_sub(minimum_log2_columns);
    if log2_rows < maximum_log2_rows {
        bits.bit(false);
    }
    if minimum_log2_columns + log2_rows > 0 {
        bits.write(0, (minimum_log2_columns + log2_rows) as u8);
        bits.write(0, 2); // tile_size_bytes_minus_1
    }
}

fn tile_log2(block_size: u32, target: u32) -> u32 {
    let mut exponent = 0;
    while block_size << exponent < target {
        exponent += 1;
    }
    exponent
}

fn write_color_config(bits: &mut BitWriter, config: SequenceConfig) {
    bits.bit(config.bit_depth > 8);
    if config.profile == 2 && config.bit_depth > 8 {
        bits.bit(config.bit_depth == 12);
    }
    if config.profile != 1 {
        bits.bit(config.monochrome);
    }
    bits.bit(false); // color_description_present_flag
    if config.monochrome {
        bits.bit(false); // color_range
    } else {
        bits.bit(false); // color_range
        if config.profile == 2 && config.bit_depth == 12 {
            let (subsampling_x, subsampling_y) = match config.chroma_sampling {
                ChromaSampling::Cs420 => (true, true),
                ChromaSampling::Cs422 => (true, false),
                ChromaSampling::Cs444 => (false, false),
                ChromaSampling::Cs400 => unreachable!(),
            };
            bits.bit(subsampling_x);
            if subsampling_x {
                bits.bit(subsampling_y);
            }
        }
        if config.chroma_sampling == ChromaSampling::Cs420 {
            bits.write(0, 2);
        }
        bits.bit(false); // separate_uv_delta_q
    }
}

fn validate(c: SequenceConfig) -> Result<(), Error> {
    if c.profile > 2 || c.width == 0 || c.height == 0 || c.level > 31 {
        return Err(Error::InvalidSequence);
    }
    let valid_depth = matches!((c.profile, c.bit_depth), (0 | 1, 8 | 10) | (2, 8 | 10 | 12));
    let valid_chroma = if c.monochrome {
        c.profile != 1 && c.chroma_sampling == ChromaSampling::Cs400
    } else {
        match c.profile {
            0 => c.chroma_sampling == ChromaSampling::Cs420,
            1 => c.chroma_sampling == ChromaSampling::Cs444,
            2 if c.bit_depth == 12 => matches!(
                c.chroma_sampling,
                ChromaSampling::Cs420 | ChromaSampling::Cs422 | ChromaSampling::Cs444
            ),
            2 => c.chroma_sampling == ChromaSampling::Cs422,
            _ => false,
        }
    };
    if !valid_depth || !valid_chroma {
        return Err(Error::InvalidSequence);
    }
    Ok(())
}

fn bits_required(value: u32) -> u8 {
    (32 - value.leading_zeros()).max(1) as u8
}

struct BitWriter {
    bytes: Vector<u8>,
    used: u8,
}
impl BitWriter {
    fn new() -> Result<Self, Error> {
        Self::with_capacity(128)
    }

    fn with_capacity(capacity: usize) -> Result<Self, Error> {
        Ok(Self {
            // A sequence header is bounded well below this capacity. Reserving
            // once makes every subsequent bit write allocation-free.
            bytes: Vector::with_capacity(capacity).map_err(|_| Error::LimitExceeded)?,
            used: 0,
        })
    }
    fn bit(&mut self, value: bool) {
        self.write(u64::from(value), 1);
    }
    fn write(&mut self, value: u64, width: u8) {
        debug_assert!(width <= 64 && (width == 64 || value < (1u64 << width)));
        for shift in (0..width).rev() {
            if self.used == 0 {
                self.bytes.try_push(0).expect("MRML allocation failed");
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= (((value >> shift) & 1) as u8) << (7 - self.used);
            self.used = (self.used + 1) & 7;
        }
    }
    fn trailing_bits(&mut self) {
        self.bit(true);
        while self.used != 0 {
            self.bit(false);
        }
    }
    fn finish(self) -> Result<Vector<u8>, Error> {
        Ok(self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Decoder, Metadata, ScalabilitySpatialLayer, ScalabilityStructure, ScalabilityTemporalGroup,
        vector_from_slice,
    };

    #[test]
    fn generated_sequence_round_trips_through_decoder() {
        for config in [
            SequenceConfig::default(),
            SequenceConfig {
                profile: 1,
                bit_depth: 10,
                chroma_sampling: ChromaSampling::Cs444,
                ..SequenceConfig::default()
            },
            SequenceConfig {
                profile: 2,
                bit_depth: 12,
                chroma_sampling: ChromaSampling::Cs422,
                ..SequenceConfig::default()
            },
            SequenceConfig {
                monochrome: true,
                chroma_sampling: ChromaSampling::Cs400,
                ..SequenceConfig::default()
            },
        ] {
            let obu = sequence_header(config).unwrap();
            let mut decoder = Decoder::new();
            assert!(decoder.decode_obus(&obu).unwrap().is_empty());
            let parsed = decoder.sequence().unwrap();
            assert_eq!(
                (parsed.max_width, parsed.max_height),
                (u32::from(config.width), u32::from(config.height))
            );
            assert_eq!(parsed.bit_depth, config.bit_depth);
            assert_eq!(parsed.chroma_sampling, config.chroma_sampling);
        }
    }

    #[test]
    fn generated_video_sequence_advertises_inter_tool_foundation() {
        let config = VideoSequenceConfig {
            sequence: SequenceConfig {
                width: 1920,
                height: 1080,
                level: 8,
                ..SequenceConfig::default()
            },
            tier: true,
            use_128x128_superblock: true,
            enable_superres: true,
        };
        let obu = video_sequence_header(config).unwrap();
        let mut decoder = Decoder::new();
        decoder.decode_obus(&obu).unwrap();
        let parsed = decoder.sequence().unwrap();
        assert!(!parsed.still_picture);
        assert!(!parsed.reduced_still_picture_header);
        assert_eq!((parsed.max_width, parsed.max_height), (1920, 1080));
        assert!(parsed.operating_points[0].tier);
        assert!(parsed.use_128x128_superblock);
        assert!(parsed.enable_interintra_compound);
        assert!(parsed.enable_masked_compound);
        assert!(parsed.enable_warped_motion);
        assert!(parsed.enable_order_hint);
        assert!(parsed.enable_jnt_comp);
        assert!(parsed.enable_ref_frame_mvs);
        assert_eq!(parsed.order_hint_bits, 7);
        assert!(parsed.enable_superres);
        assert!(parsed.enable_cdef);
        assert!(parsed.enable_restoration);
    }

    #[test]
    fn reduced_still_frame_header_enters_tile_decode_state() {
        for sequence in [
            SequenceConfig::default(),
            SequenceConfig {
                width: 8192,
                height: 4096,
                ..SequenceConfig::default()
            },
            SequenceConfig {
                monochrome: true,
                chroma_sampling: ChromaSampling::Cs400,
                ..SequenceConfig::default()
            },
        ] {
            let sequence_obu = sequence_header(sequence).unwrap();
            let frame_obu = reduced_still_frame_header(ReducedStillFrameConfig {
                sequence,
                reduced_tx_set: true,
            })
            .unwrap();
            let mut stream = ObuStream::new();
            stream.push_encoded(&sequence_obu).unwrap();
            stream.push_encoded(&frame_obu).unwrap();
            // An identical ordinary frame header is a legal frame_header_copy
            // while tile groups are pending.
            stream.push_encoded(&frame_obu).unwrap();
            let mut decoder = Decoder::new();
            assert!(decoder.decode_obus(stream.as_bytes()).unwrap().is_empty());
        }
    }

    #[test]
    fn typed_metadata_encoders_round_trip() {
        let mut stream = ObuStream::new();
        stream.push_encoded(&temporal_delimiter().unwrap()).unwrap();
        stream
            .push_encoded(&hdr_content_light_level(1000, 400).unwrap())
            .unwrap();
        stream
            .push_encoded(
                &hdr_mastering_display_color_volume([1, 2, 3], [4, 5, 6], 7, 8, 9, 10).unwrap(),
            )
            .unwrap();
        stream
            .push_encoded(&itu_t35(0xff, Some(0x42), &[9, 8]).unwrap())
            .unwrap();
        let timestamp = TimecodeMetadata {
            counting_type: 3,
            full_timestamp: false,
            discontinuity: true,
            count_dropped: false,
            frames: 17,
            seconds: Some(12),
            minutes: Some(34),
            hours: Some(5),
            time_offset_length: 6,
            time_offset: -7,
        };
        stream.push_encoded(&timecode(timestamp).unwrap()).unwrap();
        let mut decoder = Decoder::new();
        assert!(decoder.decode_obus(stream.as_bytes()).unwrap().is_empty());
        assert_eq!(decoder.metadata().len(), 4);
        assert_eq!(
            decoder.metadata()[0],
            Metadata::HdrContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }
        );
        assert_eq!(decoder.metadata()[3], Metadata::Timecode(timestamp));
    }

    #[test]
    fn metadata_encoder_rejects_inconsistent_optional_fields() {
        assert_eq!(itu_t35(1, Some(2), &[]), Err(Error::InvalidObu));
        assert_eq!(
            timecode(TimecodeMetadata {
                counting_type: 0,
                full_timestamp: false,
                discontinuity: false,
                count_dropped: false,
                frames: 0,
                seconds: None,
                minutes: Some(1),
                hours: None,
                time_offset_length: 0,
                time_offset: 0,
            }),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn structured_scalability_encoder_round_trips() {
        let mut layers = Vector::new();
        layers
            .try_push(ScalabilitySpatialLayer {
                maximum_width: Some(640),
                maximum_height: Some(360),
                reference_id: Some(2),
            })
            .unwrap();
        layers
            .try_push(ScalabilitySpatialLayer {
                maximum_width: Some(1280),
                maximum_height: Some(720),
                reference_id: Some(5),
            })
            .unwrap();
        let mut groups = Vector::new();
        groups
            .try_push(ScalabilityTemporalGroup {
                temporal_id: 1,
                temporal_switching_up_point: true,
                spatial_switching_up_point: false,
                reference_picture_differences: vector_from_slice(&[1, 3]).unwrap(),
            })
            .unwrap();
        let metadata = ScalabilityMetadata {
            mode_idc: 14,
            structure: Some(ScalabilityStructure {
                spatial_layers: layers,
                temporal_groups: groups,
            }),
        };
        let obu = scalability(&metadata).unwrap();
        let mut decoder = Decoder::new();
        decoder.decode_obus(&obu).unwrap();
        assert_eq!(decoder.metadata(), &[Metadata::Scalability(metadata)]);
    }
}
