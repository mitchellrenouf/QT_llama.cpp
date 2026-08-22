//! A dependency-free, CPU-only AV1 elementary stream decoder.
//!
//! The decoder is deliberately incremental: callers may feed a complete IVF
//! file with [`decode_ivf`] or push low-overhead AV1 OBUs into [`Decoder`].
//! It currently implements container, OBU, sequence/header, entropy, and a
//! growing normative coded-tile pipeline. Invalid syntax and unavailable
//! caller-supplied reference state are reported separately.

#![no_std]
#![forbid(unsafe_code)]

use core::fmt;
use mrml_runtime::Vector;

pub mod block_state;
pub mod cdef;
pub mod cdf;
pub mod coeff;
pub mod encoder;
pub mod entropy;
pub mod entropy_encoder;
pub mod film_grain;
pub mod frame_header;
pub mod inter;
pub mod loop_filter;
pub mod mode;
pub mod motion;
#[cfg(feature = "nvidia")]
pub mod nvidia;
pub mod palette;
pub mod params;
pub mod partition;
pub mod prediction;
pub mod quant;
pub mod reconstruction;
pub mod restoration;
pub mod superres;
pub mod tile;
pub mod tile_list;
pub mod transform;

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_TILE_GROUP: u8 = 4;
const OBU_METADATA: u8 = 5;
const OBU_FRAME: u8 = 6;
const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
const OBU_TILE_LIST: u8 = 8;
const OBU_PADDING: u8 = 15;

const fn mi_dimension(pixels: u32) -> u32 {
    2 * ((pixels + 7) >> 3)
}

const fn delta_syntax_present(
    read_deltas: bool,
    skip: bool,
    size: partition::BlockSize,
    use_128x128: bool,
) -> bool {
    let superblock = if use_128x128 {
        partition::BlockSize::Block128x128
    } else {
        partition::BlockSize::Block64x64
    };
    read_deltas && !(skip && size as u8 == superblock as u8)
}

#[derive(Clone, Copy)]
struct TileCopyRegion {
    source_x: usize,
    source_y: usize,
    destination_x: usize,
    destination_y: usize,
    width: usize,
    height: usize,
    sampling: ChromaSampling,
}

fn copy_tile_region(
    source: &reconstruction::FrameBuffer,
    destination: &mut reconstruction::FrameBuffer,
    region: TileCopyRegion,
) -> Result<(), Error> {
    copy_plane_region(&source.y, &mut destination.y, region, false)?;
    if region.sampling != ChromaSampling::Cs400 {
        copy_plane_region(
            source.u.as_ref().ok_or(Error::InvalidObu)?,
            destination.u.as_mut().ok_or(Error::InvalidObu)?,
            region,
            true,
        )?;
        copy_plane_region(
            source.v.as_ref().ok_or(Error::InvalidObu)?,
            destination.v.as_mut().ok_or(Error::InvalidObu)?,
            region,
            true,
        )?;
    }
    Ok(())
}

fn copy_plane_region(
    source: &reconstruction::Plane,
    destination: &mut reconstruction::Plane,
    region: TileCopyRegion,
    chroma: bool,
) -> Result<(), Error> {
    let sub_x = usize::from(
        chroma
            && matches!(
                region.sampling,
                ChromaSampling::Cs420 | ChromaSampling::Cs422
            ),
    );
    let sub_y = usize::from(chroma && region.sampling == ChromaSampling::Cs420);
    let width = region.width >> sub_x;
    let height = region.height >> sub_y;
    let source_x = region.source_x >> sub_x;
    let source_y = region.source_y >> sub_y;
    let destination_x = region.destination_x >> sub_x;
    let destination_y = region.destination_y >> sub_y;
    for y in 0..height {
        for x in 0..width {
            destination.set_sample(
                destination_x + x,
                destination_y + y,
                source.sample(source_x + x, source_y + y)?,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChromaSampling {
    Cs400,
    Cs420,
    Cs422,
    Cs444,
}

/// The AV1 OBU type values assigned by section 6.2.2 of the specification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObuType {
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
    RedundantFrameHeader = 7,
    TileList = 8,
    Padding = 15,
}

/// Wrap a payload in the AV1 low-overhead OBU format.
pub fn write_obu(
    kind: ObuType,
    temporal_id: u8,
    spatial_id: u8,
    payload: &[u8],
) -> Result<Vector<u8>, Error> {
    if temporal_id > 7 || spatial_id > 3 {
        return Err(Error::InvalidObu);
    }
    let extension = temporal_id != 0 || spatial_id != 0;
    let mut out = vector_with_capacity(payload.len().saturating_add(11))?;
    vector_push(
        &mut out,
        (kind as u8) << 3 | u8::from(extension) << 2 | 0b10,
    )?;
    if extension {
        vector_push(&mut out, temporal_id << 5 | spatial_id << 3)?;
    }
    write_leb128(payload.len(), &mut out)?;
    vector_extend(&mut out, payload)?;
    Ok(out)
}

/// Add AV1 OBU packets to an IVF container. IVF timestamps are monotonically
/// increasing packet indices; callers choose the time base.
pub fn write_ivf(
    packets: &[Vector<u8>],
    width: u16,
    height: u16,
    rate: u32,
    scale: u32,
) -> Result<Vector<u8>, Error> {
    if width == 0 || height == 0 || rate == 0 || scale == 0 || packets.len() > u32::MAX as usize {
        return Err(Error::InvalidIvf);
    }
    let payload_bytes = packets
        .iter()
        .try_fold(0usize, |n, p| n.checked_add(12)?.checked_add(p.len()));
    let mut out = vector_with_capacity(
        32usize
            .checked_add(payload_bytes.ok_or(Error::LimitExceeded)?)
            .ok_or(Error::LimitExceeded)?,
    )?;
    vector_extend(&mut out, b"DKIF")?;
    vector_extend(&mut out, &0u16.to_le_bytes())?;
    vector_extend(&mut out, &32u16.to_le_bytes())?;
    vector_extend(&mut out, b"AV01")?;
    vector_extend(&mut out, &width.to_le_bytes())?;
    vector_extend(&mut out, &height.to_le_bytes())?;
    vector_extend(&mut out, &rate.to_le_bytes())?;
    vector_extend(&mut out, &scale.to_le_bytes())?;
    vector_extend(&mut out, &(packets.len() as u32).to_le_bytes())?;
    vector_extend(&mut out, &0u32.to_le_bytes())?;
    for (timestamp, packet) in packets.iter().enumerate() {
        if packet.len() > u32::MAX as usize {
            return Err(Error::LimitExceeded);
        }
        vector_extend(&mut out, &(packet.len() as u32).to_le_bytes())?;
        vector_extend(&mut out, &(timestamp as u64).to_le_bytes())?;
        vector_extend(&mut out, packet)?;
    }
    Ok(out)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sequence {
    pub profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub max_width: u32,
    pub max_height: u32,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub chroma_sampling: ChromaSampling,
    pub timing: Option<TimingInfo>,
    pub decoder_model: Option<DecoderModelInfo>,
    pub initial_display_delay_present: bool,
    pub operating_points: Vector<OperatingPoint>,
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length: u8,
    pub frame_id_length: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// 0/1 is forced off/on; 2 means selected per frame.
    pub seq_force_screen_content_tools: u8,
    /// 0/1 is forced off/on; 2 means selected per frame.
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub num_ticks_per_picture: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length: u8,
    pub frame_presentation_time_length: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub level: u8,
    pub tier: bool,
    pub decoder_buffer_delay: Option<u32>,
    pub encoder_buffer_delay: Option<u32>,
    pub low_delay_mode: bool,
    pub initial_display_delay: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma_sampling: ChromaSampling,
    /// Planar samples. Eight-bit streams use one byte per sample; high-bit-depth
    /// streams use little-endian u16 samples.
    pub y: Vector<u8>,
    pub u: Vector<u8>,
    pub v: Vector<u8>,
    pub presentation_time: Option<u32>,
    pub buffer_removal_times: Vector<Option<u32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimecodeMetadata {
    pub counting_type: u8,
    pub full_timestamp: bool,
    pub discontinuity: bool,
    pub count_dropped: bool,
    pub frames: u16,
    pub seconds: Option<u8>,
    pub minutes: Option<u8>,
    pub hours: Option<u8>,
    pub time_offset_length: u8,
    pub time_offset: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalabilityMetadata {
    pub mode_idc: u8,
    pub structure: Option<ScalabilityStructure>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalabilityStructure {
    pub spatial_layers: Vector<ScalabilitySpatialLayer>,
    pub temporal_groups: Vector<ScalabilityTemporalGroup>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalabilitySpatialLayer {
    pub maximum_width: Option<u16>,
    pub maximum_height: Option<u16>,
    pub reference_id: Option<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalabilityTemporalGroup {
    pub temporal_id: u8,
    pub temporal_switching_up_point: bool,
    pub spatial_switching_up_point: bool,
    pub reference_picture_differences: Vector<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ItuT35Metadata {
    pub country_code: u8,
    pub country_code_extension: Option<u8>,
    pub payload: Vector<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Metadata {
    HdrContentLightLevel {
        max_cll: u16,
        max_fall: u16,
    },
    HdrMasteringDisplayColorVolume {
        primaries_x: [u16; 3],
        primaries_y: [u16; 3],
        white_point_x: u16,
        white_point_y: u16,
        luminance_max: u32,
        luminance_min: u32,
    },
    ItuTT35(ItuT35Metadata),
    Scalability(ScalabilityMetadata),
    Timecode(TimecodeMetadata),
    Reserved {
        kind: u64,
        payload: Vector<u8>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    Truncated,
    InvalidIvf,
    InvalidObu,
    InvalidFrameHeader,
    InvalidTileData,
    InvalidBlockSyntax,
    InvalidBlockPosition {
        row: u32,
        column: u32,
        size: partition::BlockSize,
    },
    InvalidRestorationSyntax,
    InvalidModeSyntax,
    InvalidModeStage(ModeStage),
    InvalidTransformPosition {
        row: u32,
        column: u32,
        size: transform::TxSize,
        depth: u8,
        context: u8,
    },
    InvalidTransformDepth {
        row: u32,
        column: u32,
        block_size: partition::BlockSize,
        maximum: transform::TxSize,
        context: u8,
    },
    InvalidPredictionSyntax,
    InvalidPredictionStage(PredictionStage),
    InvalidCoefficientSyntax,
    InvalidCoefficientStage(CoefficientStage),
    InvalidCoefficientPosition {
        eob: u16,
        coefficient: u16,
    },
    InvalidPartitionSyntax,
    InvalidTileTermination {
        bit_position: usize,
        max_bits: i64,
    },
    InvalidTileDecode {
        bit_position: usize,
        max_bits: i64,
        blocks: u32,
        skipped_blocks: u32,
        transform_blocks: u32,
        nonzero_transform_blocks: u32,
        coefficient_count: u32,
        block_flags: [u8; 4],
        reference_frames: [[i8; 2]; 4],
        inter_modes: [u8; 4],
        motion_vectors: [[i32; 2]; 4],
        luma_modes: [u8; 4],
        chroma_modes: [u8; 4],
        transform_types: [u8; 8],
    },
    InvalidSequence,
    MissingSequence,
    MissingReference,
    MissingAnchorFrame,
    LimitExceeded,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated AV1 bitstream"),
            Self::InvalidIvf => f.write_str("invalid IVF container"),
            Self::InvalidObu => f.write_str("invalid AV1 OBU"),
            Self::InvalidFrameHeader => f.write_str("invalid AV1 frame header"),
            Self::InvalidTileData => f.write_str("invalid AV1 tile data"),
            Self::InvalidBlockSyntax => f.write_str("invalid AV1 coded-block syntax"),
            Self::InvalidBlockPosition { .. } => {
                f.write_str("invalid AV1 coded-block syntax at block position")
            }
            Self::InvalidRestorationSyntax => f.write_str("invalid AV1 loop-restoration syntax"),
            Self::InvalidModeSyntax => f.write_str("invalid AV1 mode syntax"),
            Self::InvalidModeStage(stage) => write!(f, "invalid AV1 mode syntax at {stage:?}"),
            Self::InvalidTransformPosition { .. } => {
                f.write_str("invalid AV1 variable-transform position")
            }
            Self::InvalidTransformDepth { .. } => {
                f.write_str("invalid AV1 transform-depth position")
            }
            Self::InvalidPredictionSyntax => f.write_str("invalid AV1 prediction syntax"),
            Self::InvalidPredictionStage(stage) => {
                write!(f, "invalid AV1 prediction at {stage:?}")
            }
            Self::InvalidCoefficientSyntax => f.write_str("invalid AV1 coefficient syntax"),
            Self::InvalidCoefficientStage(stage) => {
                write!(f, "invalid AV1 coefficient syntax at {stage:?}")
            }
            Self::InvalidCoefficientPosition { .. } => {
                f.write_str("invalid AV1 coefficient base-level position")
            }
            Self::InvalidPartitionSyntax => f.write_str("invalid AV1 partition syntax"),
            Self::InvalidTileTermination { .. } => f.write_str("invalid AV1 tile termination"),
            Self::InvalidTileDecode { .. } => {
                f.write_str("invalid AV1 tile termination after coded-block decoding")
            }
            Self::InvalidSequence => f.write_str("invalid AV1 sequence header"),
            Self::MissingSequence => f.write_str("frame encountered before sequence header"),
            Self::MissingReference => f.write_str("required AV1 reference frame is unavailable"),
            Self::MissingAnchorFrame => {
                f.write_str("required external AV1 anchor frame is unavailable")
            }
            Self::LimitExceeded => f.write_str("configured decoder limit exceeded"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoefficientStage {
    Skip,
    TransformType,
    EndOfBlock,
    BaseLevel,
    BaseRange,
    Sign,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModeStage {
    Prefix,
    Cdef,
    Delta,
    References,
    InterMotion,
    InterPostMotion,
    TransformSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictionStage {
    Mode,
    IntraPixels,
    ChromaFromLuma,
}

impl core::error::Error for Error {}

const fn frame_header_error(error: Error) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidFrameHeader,
        other => other,
    }
}

const fn tile_data_error(error: Error) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidTileData,
        other => other,
    }
}

const fn restoration_syntax_error(error: Error) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidRestorationSyntax,
        other => other,
    }
}

const fn mode_stage_error(error: Error, stage: ModeStage) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidModeStage(stage),
        other => other,
    }
}

const fn prediction_stage_error(error: Error, stage: PredictionStage) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidPredictionStage(stage),
        other => other,
    }
}

const fn coefficient_syntax_error(error: Error) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidCoefficientSyntax,
        other => other,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_obu_size: usize,
    pub max_frames: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_width: 16_384,
            max_height: 16_384,
            max_obu_size: 256 << 20,
            max_frames: 1_000_000,
        }
    }
}

#[derive(Default)]
pub struct Decoder {
    sequence: Option<Sequence>,
    references: [Option<Frame>; 8],
    reference_info: [frame_header::ReferenceInfo; frame_header::NUM_REF_FRAMES],
    reference_cdfs: [Option<cdf::TileCdfs>; frame_header::NUM_REF_FRAMES],
    reference_grids: [Option<block_state::MiGrid>; frame_header::NUM_REF_FRAMES],
    reference_order_hints: [[u32; 8]; frame_header::NUM_REF_FRAMES],
    previous_frame_id: Option<u32>,
    pending_frame_header: Option<frame_header::FrameHeader>,
    pending_frame_header_bytes: Vector<u8>,
    pending_decode: Option<PendingDecode>,
    limits: Limits,
    decoded_frames: usize,
    metadata: Vector<Metadata>,
    anchor_frames: Vector<Option<Frame>>,
    camera_tile_decode: bool,
    operating_point: usize,
    previous_timecode: Option<TimecodeMetadata>,
}

struct PendingDecode {
    tiles: tile::TileAccumulator,
    grid: block_state::MiGrid,
    buffer: reconstruction::FrameBuffer,
    initial_cdfs: cdf::TileCdfs,
    temporal_field: Option<motion::MotionField>,
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }
    pub fn sequence(&self) -> Option<&Sequence> {
        self.sequence.as_ref()
    }

    /// Select the sequence operating point used for temporal/spatial OBU
    /// filtering. Selection zero is the default.
    pub fn select_operating_point(&mut self, index: usize) -> Result<(), Error> {
        if self.pending_frame_header.is_some() {
            return Err(Error::InvalidObu);
        }
        if self
            .sequence
            .as_ref()
            .is_some_and(|sequence| index >= sequence.operating_points.len())
        {
            return Err(Error::InvalidSequence);
        }
        self.operating_point = index;
        Ok(())
    }

    pub fn metadata(&self) -> &[Metadata] {
        &self.metadata
    }
    pub fn take_metadata(&mut self) -> Vector<Metadata> {
        core::mem::take(&mut self.metadata)
    }

    fn accept_metadata(&mut self, metadata: Metadata) -> Result<(), Error> {
        let metadata = if let Metadata::Timecode(mut timecode) = metadata {
            if !timecode.full_timestamp
                && (timecode.seconds.is_none()
                    || timecode.minutes.is_none()
                    || timecode.hours.is_none())
            {
                let previous = self.previous_timecode.ok_or(Error::InvalidObu)?;
                if timecode.seconds.is_none() {
                    timecode.seconds = previous.seconds;
                }
                if timecode.minutes.is_none() {
                    timecode.minutes = previous.minutes;
                }
                if timecode.hours.is_none() {
                    timecode.hours = previous.hours;
                }
            }
            if let Some(timing) = self
                .sequence
                .as_ref()
                .and_then(|sequence| sequence.timing.as_ref())
            {
                let denominator = u64::from(timing.num_units_in_display_tick)
                    .checked_mul(2)
                    .ok_or(Error::LimitExceeded)?;
                let maximum_fps = u64::from(timing.time_scale).div_ceil(denominator);
                if u64::from(timecode.frames) >= maximum_fps {
                    return Err(Error::InvalidObu);
                }
            }
            self.previous_timecode = Some(timecode);
            Metadata::Timecode(timecode)
        } else {
            metadata
        };
        vector_push(&mut self.metadata, metadata)
    }

    /// Supply an externally managed large-scale-tile anchor frame (section
    /// 6.11.2). Anchor frames are deliberately separate from AV1 reference
    /// slots because the specification assigns their lifetime to the caller.
    pub fn set_anchor_frame(&mut self, index: u8, frame: Frame) -> Result<(), Error> {
        if self.anchor_frames.is_empty() {
            self.anchor_frames = Vector::with_capacity(tile_list::MAX_ANCHOR_FRAMES)
                .map_err(|_| Error::LimitExceeded)?;
            for _ in 0..tile_list::MAX_ANCHOR_FRAMES {
                self.anchor_frames
                    .try_push(None)
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
        let slot = self
            .anchor_frames
            .get_mut(usize::from(index))
            .ok_or(Error::InvalidObu)?;
        if frame.width > self.limits.max_width || frame.height > self.limits.max_height {
            return Err(Error::LimitExceeded);
        }
        reconstruction::FrameBuffer::from_frame(&frame)?;
        *slot = Some(frame);
        Ok(())
    }

    pub fn clear_anchor_frames(&mut self) {
        self.anchor_frames.clear();
    }

    /// Decode all low-overhead OBUs in `data` and return displayed frames.
    pub fn decode_obus(&mut self, data: &[u8]) -> Result<Vector<Frame>, Error> {
        let mut frames = Vector::new();
        let mut pos = 0;
        while pos < data.len() {
            let (obu, used) = parse_obu(&data[pos..], self.limits.max_obu_size)?;
            pos = pos.checked_add(used).ok_or(Error::LimitExceeded)?;
            if let Some(sequence) = &self.sequence {
                let operating_point = sequence
                    .operating_points
                    .get(self.operating_point)
                    .ok_or(Error::InvalidSequence)?;
                validate_operating_point_extension(operating_point.idc, obu.kind, obu.extension)?;
                if !obu_in_operating_point(
                    operating_point.idc,
                    obu.kind,
                    obu.extension,
                    obu.temporal_id,
                    obu.spatial_id,
                ) {
                    continue;
                }
            }
            match obu.kind {
                OBU_SEQUENCE_HEADER => {
                    if self.pending_frame_header.is_some() {
                        return Err(Error::InvalidObu);
                    }
                    let seq = parse_sequence_header(obu.payload)?;
                    if self.operating_point >= seq.operating_points.len() {
                        return Err(Error::InvalidSequence);
                    }
                    if seq.max_width > self.limits.max_width
                        || seq.max_height > self.limits.max_height
                    {
                        return Err(Error::LimitExceeded);
                    }
                    self.sequence = Some(seq);
                    self.references = Default::default();
                    self.reference_info = Default::default();
                    self.reference_cdfs = Default::default();
                    self.reference_grids = Default::default();
                    self.reference_order_hints = Default::default();
                    self.previous_frame_id = None;
                    self.previous_timecode = None;
                    self.pending_frame_header_bytes.clear();
                    self.pending_decode = None;
                }
                OBU_TEMPORAL_DELIMITER => {
                    if self.pending_frame_header.is_some() || !obu.payload.is_empty() {
                        return Err(Error::InvalidObu);
                    }
                }
                OBU_PADDING => {}
                OBU_METADATA => self.accept_metadata(parse_metadata(obu.payload)?)?,
                OBU_FRAME_HEADER => {
                    if self.pending_frame_header.is_some() {
                        validate_frame_header_copy(
                            true,
                            &self.pending_frame_header_bytes,
                            obu.payload,
                        )?;
                    } else if let Some(frame) =
                        self.decode_frame_header(obu.payload, obu.temporal_id, obu.spatial_id)?
                    {
                        vector_push(&mut frames, frame)?;
                    }
                }
                OBU_REDUNDANT_FRAME_HEADER => {
                    validate_frame_header_copy(
                        self.pending_frame_header.is_some(),
                        &self.pending_frame_header_bytes,
                        obu.payload,
                    )?;
                }
                OBU_TILE_GROUP => {
                    if let Some(frame) = self.decode_tile_group(obu.payload)? {
                        vector_push(&mut frames, frame)?;
                    }
                }
                OBU_FRAME => {
                    if let Some(frame) =
                        self.decode_frame_obu(obu.payload, obu.temporal_id, obu.spatial_id)?
                    {
                        vector_push(&mut frames, frame)?;
                    }
                }
                OBU_TILE_LIST => {
                    let list = tile_list::parse(obu.payload, self.limits.max_obu_size)?;
                    self.validate_tile_list(&list)?;
                    vector_push(&mut frames, self.decode_tile_list(&list)?)?;
                }
                0 | 9..=14 => {
                    if !obu.payload.is_empty() && !obu.payload.iter().any(|byte| *byte != 0) {
                        return Err(Error::InvalidObu);
                    }
                }
                _ => return Err(Error::InvalidObu),
            }
        }
        Ok(frames)
    }

    fn validate_tile_list(&self, list: &tile_list::TileList<'_>) -> Result<(), Error> {
        let sequence = self.sequence.as_ref().ok_or(Error::MissingSequence)?;
        let header = self
            .pending_frame_header
            .as_ref()
            .ok_or(Error::InvalidObu)?;
        let layout = header.tile_layout.as_ref().ok_or(Error::InvalidObu)?;
        if !self
            .pending_decode
            .as_ref()
            .ok_or(Error::InvalidObu)?
            .tiles
            .is_empty()
        {
            return Err(Error::InvalidObu);
        }
        if sequence.enable_superres
            || sequence.enable_order_hint
            || sequence.still_picture
            || sequence.film_grain_params_present
            || sequence.timing.is_some()
            || sequence.decoder_model.is_some()
            || sequence.initial_display_delay_present
            || sequence.enable_restoration
            || sequence.enable_cdef
            || sequence.monochrome
            || header.show_existing_frame
            || header.frame_type != frame_header::FrameType::Inter
            || !header.show_frame
            || header.error_resilient_mode
            || !header.disable_cdf_update
            || !header.disable_frame_end_update_cdf
            || header.delta_params.delta_lf_present
            || header.delta_params.delta_q_present
            || header.frame_size_override
            || header.refresh_frame_flags != 0
            || header.use_ref_frame_mvs
            || header.segmentation.temporal_update
            || header.reference_select
            || header.loop_filter.level[0] != 0
            || header.loop_filter.level[1] != 0
        {
            return Err(Error::InvalidObu);
        }
        let superblock_mi = if sequence.use_128x128_superblock {
            32
        } else {
            16
        };
        if header.frame_width % 4 != 0
            || header.frame_height % 4 != 0
            || layout
                .row_starts_sb
                .windows(2)
                .any(|window| window[1] - window[0] != 1)
            || layout.column_starts_sb.len() < 2
        {
            return Err(Error::InvalidObu);
        }
        let tile_width_sb = layout.column_starts_sb[1] - layout.column_starts_sb[0];
        if tile_width_sb == 0
            || layout
                .column_starts_sb
                .windows(2)
                .any(|window| window[1] - window[0] != tile_width_sb)
            || layout.row_starts_sb[layout.row_starts_sb.len() - 1] * superblock_mi
                != mi_dimension(header.frame_height)
            || layout.column_starts_sb[layout.column_starts_sb.len() - 1] * superblock_mi
                != mi_dimension(header.frame_width)
        {
            return Err(Error::InvalidObu);
        }
        for entry in &list.entries {
            if usize::from(entry.anchor_tile_row) >= layout.rows()
                || usize::from(entry.anchor_tile_column) >= layout.columns()
            {
                return Err(Error::InvalidObu);
            }
            let anchor = self
                .anchor_frames
                .get(usize::from(entry.anchor_frame_index))
                .and_then(Option::as_ref)
                .ok_or(Error::MissingAnchorFrame)?;
            if anchor.width != header.upscaled_width
                || anchor.height != header.frame_height
                || anchor.bit_depth != sequence.bit_depth
                || anchor.chroma_sampling != sequence.chroma_sampling
            {
                return Err(Error::InvalidObu);
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn decode_tile_list(&mut self, list: &tile_list::TileList<'_>) -> Result<Frame, Error> {
        let sequence = self
            .sequence
            .as_ref()
            .ok_or(Error::MissingSequence)?
            .clone();
        let original_header = self
            .pending_frame_header
            .as_ref()
            .ok_or(Error::InvalidObu)?
            .clone();
        let original_layout = original_header
            .tile_layout
            .as_ref()
            .ok_or(Error::InvalidObu)?;
        let superblock_pixels = if sequence.use_128x128_superblock {
            128u32
        } else {
            64u32
        };
        let tile_width_sb =
            original_layout.column_starts_sb[1] - original_layout.column_starts_sb[0];
        let tile_width = tile_width_sb
            .checked_mul(superblock_pixels)
            .ok_or(Error::LimitExceeded)?;
        let tile_height = superblock_pixels;
        let output_width = u32::from(list.header.output_width_in_tiles)
            .checked_mul(tile_width)
            .ok_or(Error::LimitExceeded)?;
        let output_height = u32::from(list.header.output_height_in_tiles)
            .checked_mul(tile_height)
            .ok_or(Error::LimitExceeded)?;
        if output_width > self.limits.max_width || output_height > self.limits.max_height {
            return Err(Error::LimitExceeded);
        }
        let initial_cdfs = self
            .pending_decode
            .as_ref()
            .ok_or(Error::InvalidObu)?
            .initial_cdfs
            .clone();
        let saved_references = self.references.clone();
        let saved_reference_info = self.reference_info;
        let saved_reference_cdfs = self.reference_cdfs.clone();
        let saved_reference_grids = self.reference_grids.clone();
        let saved_order_hints = self.reference_order_hints;
        let saved_decoded_frames = self.decoded_frames;
        let saved_header_bytes = self.pending_frame_header_bytes.clone();
        let result = (|| {
            let mut output = reconstruction::FrameBuffer::new(
                output_width,
                output_height,
                sequence.bit_depth,
                sequence.chroma_sampling,
            )?;
            for (tile_number, entry) in list.entries.iter().enumerate() {
                let row = usize::from(entry.anchor_tile_row);
                let column = usize::from(entry.anchor_tile_column);
                let mut column_starts_sb =
                    Vector::with_capacity(2).map_err(|_| Error::LimitExceeded)?;
                vector_push(
                    &mut column_starts_sb,
                    original_layout.column_starts_sb[column],
                )?;
                vector_push(
                    &mut column_starts_sb,
                    original_layout.column_starts_sb[column + 1],
                )?;
                let mut row_starts_sb =
                    Vector::with_capacity(2).map_err(|_| Error::LimitExceeded)?;
                vector_push(&mut row_starts_sb, original_layout.row_starts_sb[row])?;
                vector_push(&mut row_starts_sb, original_layout.row_starts_sb[row + 1])?;
                let mut camera_header = original_header.clone();
                camera_header.tile_layout = Some(tile::TileLayout {
                    column_starts_sb,
                    row_starts_sb,
                    context_update_tile_id: 0,
                    tile_size_bytes: 1,
                });
                let last_slot = usize::from(camera_header.ref_frame_idx[0]);
                let anchor = self
                    .anchor_frames
                    .get(usize::from(entry.anchor_frame_index))
                    .and_then(Option::as_ref)
                    .ok_or(Error::MissingAnchorFrame)?
                    .clone();
                self.references[last_slot] = Some(anchor);
                self.reference_info[last_slot].valid = true;
                self.reference_info[last_slot].upscaled_width = camera_header.upscaled_width;
                self.reference_info[last_slot].frame_width = camera_header.frame_width;
                self.reference_info[last_slot].frame_height = camera_header.frame_height;
                self.pending_frame_header = Some(camera_header);
                self.pending_decode = Some(PendingDecode {
                    tiles: tile::TileAccumulator::new(1)?,
                    grid: block_state::MiGrid::new(
                        mi_dimension(original_header.frame_width),
                        mi_dimension(original_header.frame_height),
                    )?,
                    buffer: reconstruction::FrameBuffer::new(
                        original_header.frame_width,
                        original_header.frame_height,
                        sequence.bit_depth,
                        sequence.chroma_sampling,
                    )?,
                    initial_cdfs: initial_cdfs.clone(),
                    temporal_field: None,
                });
                self.camera_tile_decode = true;
                let decoded_result = self.decode_tile_group(entry.coded_tile_data);
                self.camera_tile_decode = false;
                let decoded = decoded_result?.ok_or(Error::InvalidObu)?;
                let decoded = reconstruction::FrameBuffer::from_frame(&decoded)?;
                let source_x =
                    usize::try_from(original_layout.column_starts_sb[column] * superblock_pixels)
                        .map_err(|_| Error::LimitExceeded)?;
                let source_y =
                    usize::try_from(original_layout.row_starts_sb[row] * superblock_pixels)
                        .map_err(|_| Error::LimitExceeded)?;
                let destination_x = tile_number % usize::from(list.header.output_width_in_tiles)
                    * usize::try_from(tile_width).map_err(|_| Error::LimitExceeded)?;
                let destination_y = tile_number / usize::from(list.header.output_width_in_tiles)
                    * usize::try_from(tile_height).map_err(|_| Error::LimitExceeded)?;
                copy_tile_region(
                    &decoded,
                    &mut output,
                    TileCopyRegion {
                        source_x,
                        source_y,
                        destination_x,
                        destination_y,
                        width: usize::try_from(tile_width).map_err(|_| Error::LimitExceeded)?,
                        height: usize::try_from(tile_height).map_err(|_| Error::LimitExceeded)?,
                        sampling: sequence.chroma_sampling,
                    },
                )?;
            }
            output.into_frame()
        })();
        self.references = saved_references;
        self.reference_info = saved_reference_info;
        self.reference_cdfs = saved_reference_cdfs;
        self.reference_grids = saved_reference_grids;
        self.reference_order_hints = saved_order_hints;
        self.pending_frame_header = None;
        self.pending_decode = None;
        self.pending_frame_header_bytes.clear();
        self.decoded_frames = saved_decoded_frames;
        self.camera_tile_decode = false;
        let frame = match result {
            Ok(frame) => frame,
            Err(error) => {
                self.pending_frame_header = Some(original_header.clone());
                self.pending_frame_header_bytes = saved_header_bytes;
                self.pending_decode = Some(PendingDecode {
                    tiles: tile::TileAccumulator::new(original_layout.tile_count())?,
                    grid: block_state::MiGrid::new(
                        mi_dimension(original_header.frame_width),
                        mi_dimension(original_header.frame_height),
                    )?,
                    buffer: reconstruction::FrameBuffer::new(
                        original_header.frame_width,
                        original_header.frame_height,
                        sequence.bit_depth,
                        sequence.chroma_sampling,
                    )?,
                    initial_cdfs,
                    temporal_field: None,
                });
                return Err(error);
            }
        };
        self.note_frame()?;
        Ok(frame)
    }

    /// Decode Annex-B temporal units. Annex B supplies an outer length for
    /// every OBU, so both sized and unsized OBU headers are accepted here.
    pub fn decode_annex_b(&mut self, data: &[u8]) -> Result<Vector<Frame>, Error> {
        let mut displayed = Vector::new();
        let mut stream = LengthReader::new(data);
        while !stream.is_empty() {
            let temporal = stream.item(self.limits.max_obu_size)?;
            let mut temporal = LengthReader::new(temporal);
            while !temporal.is_empty() {
                let frame_unit = temporal.item(self.limits.max_obu_size)?;
                let mut frame_unit = LengthReader::new(frame_unit);
                while !frame_unit.is_empty() {
                    let obu_bytes = frame_unit.item(self.limits.max_obu_size)?;
                    for frame in self.decode_external_obu(obu_bytes)? {
                        vector_push(&mut displayed, frame)?;
                    }
                }
            }
        }
        Ok(displayed)
    }

    fn decode_external_obu(&mut self, data: &[u8]) -> Result<Vector<Frame>, Error> {
        if data.is_empty() {
            return Err(Error::InvalidObu);
        }
        if data[0] & 0b10 != 0 {
            return self.decode_obus(data);
        }
        let extension = data[0] & 4 != 0;
        let header_len = 1 + usize::from(extension);
        if data.len() < header_len {
            return Err(Error::Truncated);
        }
        let mut normalized = vector_with_capacity(data.len().saturating_add(9))?;
        vector_push(&mut normalized, data[0] | 0b10)?;
        if extension {
            vector_push(&mut normalized, data[1])?;
        }
        write_leb128(data.len() - header_len, &mut normalized)?;
        vector_extend(&mut normalized, &data[header_len..])?;
        self.decode_obus(&normalized)
    }

    fn decode_frame_header(
        &mut self,
        payload: &[u8],
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<Option<Frame>, Error> {
        let seq = self.sequence.as_ref().ok_or(Error::MissingSequence)?;
        let header = frame_header::parse(
            payload,
            seq,
            &self.reference_info,
            self.previous_frame_id,
            temporal_id,
            spatial_id,
        )
        .map_err(frame_header_error)?;
        validate_trailing_from(payload, header.bits_consumed)?;
        self.accept_frame_header(header, payload)
    }

    fn decode_frame_obu(
        &mut self,
        payload: &[u8],
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<Option<Frame>, Error> {
        if self.pending_frame_header.is_some() {
            return Err(Error::InvalidObu);
        }
        let seq = self.sequence.as_ref().ok_or(Error::MissingSequence)?;
        let header = frame_header::parse(
            payload,
            seq,
            &self.reference_info,
            self.previous_frame_id,
            temporal_id,
            spatial_id,
        )
        .map_err(frame_header_error)?;
        if header.show_existing_frame {
            return Err(Error::InvalidObu);
        }
        let header_bytes =
            align_zero_from(payload, header.bits_consumed).map_err(frame_header_error)?;
        self.accept_frame_header(header, &payload[..header_bytes])?;
        self.decode_tile_group(&payload[header_bytes..])
            .map_err(tile_data_error)
    }

    fn accept_frame_header(
        &mut self,
        header: frame_header::FrameHeader,
        payload: &[u8],
    ) -> Result<Option<Frame>, Error> {
        if header.show_existing_frame {
            let slot = header.frame_to_show_map_idx as usize;
            let reference = self.references[slot]
                .clone()
                .ok_or(Error::MissingReference)?;
            let mut output = reconstruction::FrameBuffer::from_frame(&reference)?;
            let sequence = self.sequence.as_ref().ok_or(Error::MissingSequence)?;
            film_grain::apply(
                &mut output,
                &header.film_grain,
                sequence.matrix_coefficients,
                header.upscaled_width,
                header.frame_height,
            )?;
            let mut frame = output.into_frame()?;
            frame.presentation_time = header.frame_presentation_time;
            frame.buffer_removal_times = header.buffer_removal_times.clone();
            if header.frame_type == frame_header::FrameType::Key {
                let info = self.reference_info[slot];
                let cdfs = self.reference_cdfs[slot].clone();
                let grid = self.reference_grids[slot].clone();
                let order_hints = self.reference_order_hints[slot];
                for destination in 0..frame_header::NUM_REF_FRAMES {
                    self.references[destination] = Some(reference.clone());
                    self.reference_info[destination] = info;
                    self.reference_cdfs[destination] = cdfs.clone();
                    self.reference_grids[destination] = grid.clone();
                    self.reference_order_hints[destination] = order_hints;
                }
            }
            self.note_frame()?;
            return Ok(Some(frame));
        }
        if self.pending_frame_header.is_some() {
            return Err(Error::InvalidObu);
        }
        let mut header_bytes = vector_with_capacity(payload.len())?;
        vector_extend(&mut header_bytes, payload)?;
        for index in 0..frame_header::NUM_REF_FRAMES {
            if header.invalidated_reference_slots & (1 << index) != 0 {
                self.reference_info[index].valid = false;
                self.references[index] = None;
                self.reference_cdfs[index] = None;
                self.reference_grids[index] = None;
                self.reference_order_hints[index] = [0; 8];
            }
        }
        self.previous_frame_id = header.current_frame_id.or(self.previous_frame_id);
        let tile_count = header
            .tile_layout
            .as_ref()
            .ok_or(Error::InvalidObu)?
            .tile_count();
        let sequence = self
            .sequence
            .as_ref()
            .ok_or(Error::MissingSequence)?
            .clone();
        let initial_cdfs = cdf::initial_frame_cdfs(
            header.primary_ref_frame,
            &header.ref_frame_idx,
            &self.reference_cdfs,
            header.quantization.base_q_idx,
        )?;
        let mi_columns = mi_dimension(header.frame_width);
        let mi_rows = mi_dimension(header.frame_height);
        let temporal_field = if header.use_ref_frame_mvs {
            let order_hints = core::array::from_fn(|reference| {
                if reference == 0 {
                    0
                } else {
                    self.reference_info[usize::from(header.ref_frame_idx[reference - 1])].order_hint
                }
            });
            let references = core::array::from_fn(|slot| {
                self.reference_grids[slot]
                    .as_ref()
                    .map(|grid| motion::SavedMotionFieldReference {
                        grid,
                        is_inter: !self.reference_info[slot].frame_type.is_intra(),
                        order_hints: self.reference_order_hints[slot],
                    })
            });
            Some(
                motion::estimate_motion_field(
                    mi_columns,
                    mi_rows,
                    motion::MotionFieldEstimation {
                        references,
                        ref_frame_idx: header.ref_frame_idx,
                        order_hints,
                        current_order_hint: header.order_hint,
                        order_hint_bits: sequence.order_hint_bits,
                    },
                )?
                .0,
            )
        } else {
            None
        };
        self.pending_decode = Some(PendingDecode {
            tiles: tile::TileAccumulator::new(tile_count)?,
            grid: block_state::MiGrid::new(mi_columns, mi_rows)?,
            buffer: reconstruction::FrameBuffer::new(
                header.frame_width,
                header.frame_height,
                sequence.bit_depth,
                sequence.chroma_sampling,
            )?,
            initial_cdfs,
            temporal_field,
        });
        self.pending_frame_header = Some(header);
        self.pending_frame_header_bytes = header_bytes;
        Ok(None)
    }

    fn decode_tile_group(&mut self, payload: &[u8]) -> Result<Option<Frame>, Error> {
        let header = self
            .pending_frame_header
            .as_ref()
            .ok_or(Error::InvalidObu)?
            .clone();
        let layout = header.tile_layout.as_ref().ok_or(Error::InvalidObu)?;
        let group = layout.parse_group(payload)?;
        let complete = self
            .pending_decode
            .as_mut()
            .ok_or(Error::InvalidObu)?
            .tiles
            .push(&group)?;
        if !complete {
            return Ok(None);
        }
        let reference_info = self.reference_info;
        let mut reference_buffers: [Option<reconstruction::FrameBuffer>; 8] =
            core::array::from_fn(|_| None);
        for (slot, frame) in self.references.iter().enumerate() {
            if let Some(frame) = frame {
                reference_buffers[slot] = Some(reconstruction::FrameBuffer::from_frame(frame)?);
            }
        }
        let pending = self.pending_decode.as_mut().ok_or(Error::InvalidObu)?;
        if pending.grid.columns() != mi_dimension(header.frame_width)
            || pending.grid.rows() != mi_dimension(header.frame_height)
            || pending.buffer.y.width()
                != usize::try_from(mi_dimension(header.frame_width).saturating_mul(4))
                    .map_err(|_| Error::LimitExceeded)?
            || pending.buffer.y.height()
                != usize::try_from(mi_dimension(header.frame_height).saturating_mul(4))
                    .map_err(|_| Error::LimitExceeded)?
        {
            return Err(Error::InvalidObu);
        }
        let sequence = self
            .sequence
            .as_ref()
            .ok_or(Error::MissingSequence)?
            .clone();
        let PendingDecode {
            tiles,
            grid,
            initial_cdfs,
            temporal_field,
            buffer,
        } = pending;
        let temporal_field = temporal_field.as_ref();
        let mut prediction_contexts = None;
        let mut coefficient_contexts = None;
        let mut cdef_indices = None;
        let mut context_tile = None;
        let mut restoration_tile = None;
        let mut context_superblock = None;
        let mut current_qindex = header.quantization.base_q_idx;
        let mut delta_lf = [0i8; 4];
        let mut read_deltas = false;
        let mut current_block = None;
        // Leaf, skipped-leaf, transform, and nonzero-transform counts retained
        // for strict conformance diagnostics when tile termination fails.
        let mut decode_counts = [0u32; 5];
        let mut block_flags = [u8::MAX; 4];
        let mut reference_frames = [[-1i8; 2]; 4];
        let mut inter_modes = [u8::MAX; 4];
        let mut motion_vectors = [[i32::MIN; 2]; 4];
        let mut luma_modes = [u8::MAX; 4];
        let mut chroma_modes = [u8::MAX; 4];
        let mut transform_types = [u8::MAX; 8];
        let mut traced_transform_count = 0usize;
        let (subsampling_x, subsampling_y) = match sequence.chroma_sampling {
            ChromaSampling::Cs400 | ChromaSampling::Cs444 => (false, false),
            ChromaSampling::Cs420 => (true, true),
            ChromaSampling::Cs422 => (true, false),
        };
        let mut restoration_units = restoration::RestorationUnits::new(
            &header.restoration,
            header.upscaled_width,
            header.frame_height,
            sequence.chroma_sampling,
        )?;
        let final_cdfs = tile::decode_accumulated_partition_trees(
            tiles,
            tile::AccumulatedTileDecodeConfig {
                layout,
                mi_columns: mi_dimension(header.frame_width),
                mi_rows: mi_dimension(header.frame_height),
                use_128x128: sequence.use_128x128_superblock,
                disable_cdf_update: header.disable_cdf_update,
                initial_cdfs,
            },
            grid,
            |tile_number, _bounds, decoder, cdfs, root| {
                if restoration_tile != Some(tile_number) {
                    restoration_units.reset_tile_references();
                    restoration_tile = Some(tile_number);
                }
                restoration_units
                    .read_superblock(
                        decoder,
                        cdfs,
                        root,
                        &header.restoration,
                        sequence.chroma_sampling,
                        header.use_superres,
                        header.superres_denom,
                    )
                    .map_err(restoration_syntax_error)
            },
            |tile_number, bounds, decoder, cdfs, grid, block, size| {
                current_block = Some((block.row, block.column, size));
                decode_counts[0] = decode_counts[0].saturating_add(1);
                if context_tile != Some(tile_number) {
                    prediction_contexts = Some(block_state::SegmentPredictionContexts::new(
                        mi_dimension(header.frame_width),
                        mi_dimension(header.frame_height),
                    )?);
                    cdef_indices = Some(block_state::CdefIndexGrid::new(
                        mi_dimension(header.frame_width),
                        mi_dimension(header.frame_height),
                    )?);
                    coefficient_contexts = Some(coeff::CoefficientContexts::new(
                        mi_dimension(header.frame_width),
                        mi_dimension(header.frame_height),
                        subsampling_x,
                        subsampling_y,
                        sequence.monochrome,
                    )?);
                    current_qindex = header.quantization.base_q_idx;
                    delta_lf = [0; 4];
                    context_superblock = None;
                    context_tile = Some(tile_number);
                }
                let superblock_mi = if sequence.use_128x128_superblock {
                    32
                } else {
                    16
                };
                let superblock = (block.row / superblock_mi, block.column / superblock_mi);
                if context_superblock != Some(superblock) {
                    read_deltas = header.delta_params.delta_q_present;
                    context_superblock = Some(superblock);
                }
                let prefix_config = mode::BlockModePrefixConfig {
                    block,
                    size,
                    tile: bounds,
                    segmentation: &header.segmentation,
                    previous_segments: None,
                    skip_mode_present: header.skip_mode_present,
                    frame_is_intra: header.frame_type.is_intra(),
                };
                let pre_inter = mode::read_block_mode_pre_inter(
                    decoder,
                    cdfs,
                    grid,
                    prediction_contexts.as_mut().ok_or(Error::InvalidObu)?,
                    prefix_config,
                )
                .map_err(|error| mode_stage_error(error, ModeStage::Prefix))?;
                let cdef_index = mode::read_cdef_index(
                    decoder,
                    cdef_indices.as_mut().ok_or(Error::InvalidObu)?,
                    mode::CdefIndexConfig {
                        block,
                        skip: pre_inter.skip,
                        coded_lossless: header.coded_lossless,
                        enabled: sequence.enable_cdef,
                        allow_intrabc: header.allow_intrabc,
                        bits: header.cdef.bits,
                    },
                )
                .map_err(|error| mode_stage_error(error, ModeStage::Cdef))?;
                // Sections 5.11.7 and 5.11.18 clear ReadDeltas after the first
                // leaf in a superblock. Sections 5.11.12 and 5.11.13 suppress
                // the symbols only when that leaf is itself a skipped full
                // superblock; a skipped child of a partitioned superblock
                // still consumes the delta syntax.
                if read_deltas {
                    if delta_syntax_present(
                        read_deltas,
                        pre_inter.skip,
                        size,
                        sequence.use_128x128_superblock,
                    ) {
                        current_qindex = mode::read_delta_qindex(
                            decoder,
                            cdfs,
                            current_qindex,
                            header.delta_params.delta_q_res,
                        )
                        .map_err(|error| mode_stage_error(error, ModeStage::Delta))?;
                        if header.delta_params.delta_lf_present {
                            let count = if header.delta_params.delta_lf_multi {
                                if sequence.monochrome { 2 } else { 4 }
                            } else {
                                1
                            };
                            for (component, level) in delta_lf.iter_mut().enumerate().take(count) {
                                *level = mode::read_delta_lf(
                                    decoder,
                                    cdfs,
                                    *level,
                                    header.delta_params.delta_lf_res,
                                    u8::try_from(component).map_err(|_| Error::LimitExceeded)?,
                                )
                                .map_err(|error| mode_stage_error(error, ModeStage::Delta))?;
                            }
                        }
                    }
                    read_deltas = false;
                }
                let prefix = mode::finish_block_mode_prefix(
                    decoder,
                    cdfs,
                    grid,
                    block,
                    bounds,
                    pre_inter,
                    header.frame_type.is_intra(),
                )
                .map_err(|error| mode_stage_error(error, ModeStage::Prefix))?;
                if prefix.skip {
                    decode_counts[1] = decode_counts[1].saturating_add(1);
                }
                let block_trace_index =
                    usize::try_from(decode_counts[0] - 1).map_err(|_| Error::LimitExceeded)?;
                if block_trace_index < block_flags.len() {
                    block_flags[block_trace_index] = u8::from(prefix.is_inter)
                        | (u8::from(prefix.skip) << 1)
                        | (u8::from(prefix.skip_mode) << 2);
                }
                let availability = partition::block_availability(
                    block,
                    bounds,
                    subsampling_x,
                    subsampling_y,
                    sequence.monochrome,
                );
                let references = if prefix.is_inter {
                    mode::read_reference_frames_from_grid(
                        decoder,
                        cdfs,
                        grid,
                        block,
                        bounds,
                        mode::ReferenceFrameConfig {
                            size,
                            skip_mode: prefix.skip_mode,
                            skip_mode_frames: [
                                i8::try_from(header.skip_mode_frame[0] + 1)
                                    .map_err(|_| Error::InvalidObu)?,
                                i8::try_from(header.skip_mode_frame[1] + 1)
                                    .map_err(|_| Error::InvalidObu)?,
                            ],
                            segment: prefix.segment,
                            reference_select: header.reference_select,
                            contexts: mode::ReferenceContexts::default(),
                        },
                    )
                    .map_err(|error| mode_stage_error(error, ModeStage::References))?
                } else {
                    [0, -1]
                };
                if block_trace_index < reference_frames.len() {
                    reference_frames[block_trace_index] = references;
                }
                let intra = if prefix.is_inter {
                    None
                } else {
                    Some(
                        mode::read_intra_block_mode(
                            decoder,
                            cdfs,
                            grid,
                            mode::IntraBlockModeConfig {
                                block,
                                size,
                                tile: bounds,
                                plane_residual_size: size
                                    .plane_residual_size(subsampling_x, subsampling_y)?,
                                lossless: header.lossless_segments[usize::from(prefix.segment_id)],
                                has_chroma: availability.has_chroma,
                                palette_enabled: header.allow_screen_content_tools,
                                filter_intra_enabled: sequence.enable_filter_intra,
                                frame_is_intra: header.frame_type.is_intra(),
                                allow_intrabc: header.allow_intrabc,
                            },
                        )
                        .map_err(|error| prediction_stage_error(error, PredictionStage::Mode))?,
                    )
                };
                if let Some(intra) = intra {
                    let trace_index =
                        usize::try_from(decode_counts[0] - 1).map_err(|_| Error::LimitExceeded)?;
                    if trace_index < luma_modes.len() {
                        luma_modes[trace_index] = intra.y_mode as u8;
                        chroma_modes[trace_index] = intra.chroma.mode as u8;
                    }
                }
                let intrabc_motion = if intra.is_some_and(|mode| mode.use_intrabc) {
                    let mut predictor = grid
                        .get(block.row.saturating_sub(1), block.column)
                        .filter(|state| state.is_inter && state.reference_frames[0] == 0)
                        .map_or(motion::MotionVector::default(), |state| {
                            state.motion_vectors[0]
                        });
                    if predictor == motion::MotionVector::default() {
                        predictor = grid
                            .get(block.row, block.column.saturating_sub(1))
                            .filter(|state| state.is_inter && state.reference_frames[0] == 0)
                            .map_or(motion::MotionVector::default(), |state| {
                                state.motion_vectors[0]
                            });
                    }
                    if predictor == motion::MotionVector::default() {
                        let superblock_pixels: i32 = if sequence.use_128x128_superblock {
                            128
                        } else {
                            64
                        };
                        let superblock_mi = u32::try_from(superblock_pixels / 4)
                            .map_err(|_| Error::LimitExceeded)?;
                        predictor = if block.row < bounds.row_start + superblock_mi {
                            motion::MotionVector {
                                row: 0,
                                column: -(superblock_pixels + 256) * 8,
                            }
                        } else {
                            motion::MotionVector {
                                row: -superblock_pixels * 8,
                                column: 0,
                            }
                        };
                    }
                    let vector = motion::read_motion_vector(
                        decoder,
                        cdfs.motion_vectors(),
                        predictor,
                        motion::MotionVectorSyntax {
                            force_integer: true,
                            allow_high_precision: false,
                            intrabc: true,
                        },
                    )?;
                    let (block_width, block_height) = size.dimensions();
                    motion::validate_intrabc_motion(
                        vector,
                        motion::IntrabcValidation {
                            block,
                            tile: bounds,
                            block_width,
                            block_height,
                            has_chroma: availability.has_chroma,
                            subsampling_x,
                            subsampling_y,
                            use_128x128_superblock: sequence.use_128x128_superblock,
                        },
                    )?;
                    Some(vector)
                } else {
                    None
                };
                let (inter_motion, global_types) = if prefix.is_inter {
                    let compound = references[1] > 0;
                    let mut global_vectors = [motion::MotionVector::default(); 2];
                    let mut global_types = [motion::GlobalMotionType::Identity; 2];
                    for list in 0..(1 + usize::from(compound)) {
                        let reference =
                            usize::try_from(references[list] - 1).map_err(|_| Error::InvalidObu)?;
                        let global = *header
                            .global_motion
                            .get(reference)
                            .ok_or(Error::InvalidObu)?;
                        global_types[list] = global.kind;
                        global_vectors[list] = motion::setup_global_motion_vector(
                            global,
                            block,
                            header.allow_high_precision_mv,
                            header.force_integer_mv,
                        )?;
                    }
                    let mut sign_bias = [false; 8];
                    for (reference, bias) in sign_bias.iter_mut().enumerate().skip(1) {
                        let slot = usize::from(header.ref_frame_idx[reference - 1]);
                        *bias = motion::relative_order_hint_distance(
                            reference_info[slot].order_hint,
                            header.order_hint,
                            sequence.order_hint_bits,
                        )? > 0;
                    }
                    let stack = motion::build_normative_motion_stack(
                        grid,
                        motion::NormativeMotionStackConfig {
                            spatial: motion::SpatialScan {
                                block,
                                tile: bounds,
                                references,
                                compound,
                            },
                            temporal_field,
                            temporal: temporal_field.map(|_| motion::TemporalScanConfig {
                                block,
                                references,
                                compound,
                                force_integer: header.force_integer_mv,
                                allow_high_precision: header.allow_high_precision_mv,
                                global_motion: global_vectors,
                            }),
                            global_vectors,
                            reference_sign_bias: sign_bias,
                        },
                    )?;
                    (
                        Some(
                            mode::read_inter_motion(
                                decoder,
                                cdfs,
                                mode::InterMotionConfig {
                                    skip_mode: prefix.skip_mode,
                                    forced_global: prefix.segment.skip || prefix.segment.global_mv,
                                    compound,
                                    stack: &stack,
                                    global_vectors,
                                    syntax: motion::MotionVectorSyntax {
                                        force_integer: header.force_integer_mv,
                                        allow_high_precision: header.allow_high_precision_mv,
                                        intrabc: false,
                                    },
                                },
                            )
                            .map_err(|error| mode_stage_error(error, ModeStage::InterMotion))?,
                        ),
                        global_types,
                    )
                } else {
                    (None, [motion::GlobalMotionType::Identity; 2])
                };
                let inter_post = if let Some(inter_motion) = inter_motion {
                    if block_trace_index < inter_modes.len() {
                        inter_modes[block_trace_index] = inter_motion.y_mode;
                        motion_vectors[block_trace_index] = [
                            inter_motion.motion_vectors[0].row,
                            inter_motion.motion_vectors[0].column,
                        ];
                    }
                    let first_slot = usize::from(
                        header.ref_frame_idx
                            [usize::try_from(references[0] - 1).map_err(|_| Error::InvalidObu)?],
                    );
                    let reference_scaled = {
                        const SCALE: u64 = 1 << 14;
                        let reference = reference_info[first_slot];
                        ((u64::from(reference.upscaled_width) << 14)
                            + u64::from(header.frame_width / 2))
                            / u64::from(header.frame_width)
                            != SCALE
                            || ((u64::from(reference.frame_height) << 14)
                                + u64::from(header.frame_height / 2))
                                / u64::from(header.frame_height)
                                != SCALE
                    };
                    let first_distance = motion::relative_order_hint_distance(
                        reference_info[first_slot].order_hint,
                        header.order_hint,
                        sequence.order_hint_bits,
                    )?
                    .abs();
                    let second_distance = if references[1] > 0 {
                        let slot = usize::from(
                            header.ref_frame_idx[usize::try_from(references[1] - 1)
                                .map_err(|_| Error::InvalidObu)?],
                        );
                        motion::relative_order_hint_distance(
                            reference_info[slot].order_hint,
                            header.order_hint,
                            sequence.order_hint_bits,
                        )?
                        .abs()
                    } else {
                        -1
                    };
                    Some(
                        mode::read_inter_post_motion(
                            decoder,
                            cdfs,
                            grid,
                            mode::InterPostMotionConfig {
                                block,
                                size,
                                tile: bounds,
                                skip_mode: prefix.skip_mode,
                                references,
                                motion: inter_motion,
                                global_types,
                                enable_inter_intra: sequence.enable_interintra_compound,
                                motion_mode_switchable: header.motion_mode_switchable,
                                force_integer_mv: header.force_integer_mv,
                                allow_warped_motion: header.allow_warped_motion,
                                reference_scaled,
                                enable_masked_compound: sequence.enable_masked_compound,
                                enable_joint_compound: sequence.enable_jnt_comp,
                                equal_reference_distance: first_distance == second_distance,
                                frame_filter: header.interpolation_filter,
                                dual_filter: sequence.enable_dual_filter,
                            },
                        )
                        .map_err(|error| mode_stage_error(error, ModeStage::InterPostMotion))?,
                    )
                } else {
                    None
                };
                let palette_data = if let Some(intra) = intra {
                    if intra.palette_sizes == [0; 2] {
                        None
                    } else {
                        let mut y_cache = [0u16; palette::MAX_PALETTE_CACHE];
                        let mut u_cache = [0u16; palette::MAX_PALETTE_CACHE];
                        let above = ((block.row * 4) % 64 != 0)
                            .then(|| grid.get(block.row.saturating_sub(1), block.column))
                            .flatten();
                        let left = availability
                            .left
                            .then(|| grid.get(block.row, block.column.saturating_sub(1)))
                            .flatten();
                        let y_cache_count = palette::merge_palette_cache(
                            above.map_or(&[][..], |state| {
                                &state.palette_colors[0][..usize::from(state.palette_sizes[0])]
                            }),
                            left.map_or(&[][..], |state| {
                                &state.palette_colors[0][..usize::from(state.palette_sizes[0])]
                            }),
                            &mut y_cache,
                        )?;
                        let u_cache_count = palette::merge_palette_cache(
                            above.map_or(&[][..], |state| {
                                &state.palette_colors[1][..usize::from(state.palette_sizes[1])]
                            }),
                            left.map_or(&[][..], |state| {
                                &state.palette_colors[1][..usize::from(state.palette_sizes[1])]
                            }),
                            &mut u_cache,
                        )?;
                        let colors = palette::read_palette_colors(
                            decoder,
                            sequence.bit_depth,
                            intra.palette_sizes,
                            &y_cache[..y_cache_count],
                            &u_cache[..u_cache_count],
                        )?;
                        let (block_width, block_height) = size.dimensions();
                        let onscreen_width = block_width.min(
                            u16::try_from((mi_dimension(header.frame_width) - block.column) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                        );
                        let onscreen_height = block_height.min(
                            u16::try_from((mi_dimension(header.frame_height) - block.row) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                        );
                        let y_map = if intra.palette_sizes[0] > 0 {
                            Some(palette::read_palette_color_map(
                                decoder,
                                cdfs,
                                palette::PaletteMapConfig {
                                    palette_size: intra.palette_sizes[0],
                                    chroma: false,
                                    block_width,
                                    block_height,
                                    onscreen_width,
                                    onscreen_height,
                                    subsampling_x,
                                    subsampling_y,
                                },
                            )?)
                        } else {
                            None
                        };
                        let uv_map = if intra.palette_sizes[1] > 0 {
                            Some(palette::read_palette_color_map(
                                decoder,
                                cdfs,
                                palette::PaletteMapConfig {
                                    palette_size: intra.palette_sizes[1],
                                    chroma: true,
                                    block_width,
                                    block_height,
                                    onscreen_width,
                                    onscreen_height,
                                    subsampling_x,
                                    subsampling_y,
                                },
                            )?)
                        } else {
                            None
                        };
                        Some((colors, y_map, uv_map))
                    }
                } else {
                    None
                };
                let tx_size = mode::read_block_tx_size(
                    decoder,
                    cdfs,
                    grid,
                    mode::BlockTxSizeConfig {
                        block,
                        size,
                        tile: bounds,
                        lossless: header.lossless_segments[usize::from(prefix.segment_id)],
                        skip: prefix.skip,
                        is_inter: prefix.is_inter || intrabc_motion.is_some(),
                        tx_mode_select: header.tx_mode == params::TxMode::Select,
                    },
                )
                .map_err(|error| mode_stage_error(error, ModeStage::TransformSize))?;
                if let Some((colors, y_map, uv_map)) = &palette_data {
                    if let Some(map) = y_map {
                        reconstruction::paint_palette_map(
                            &mut buffer.y,
                            usize::try_from(block.column * 4).map_err(|_| Error::LimitExceeded)?,
                            usize::try_from(block.row * 4).map_err(|_| Error::LimitExceeded)?,
                            map,
                            &colors.y[..usize::from(colors.sizes[0])],
                            sequence.bit_depth,
                        )?;
                    }
                    if let Some(map) = uv_map {
                        let x = usize::try_from((block.column >> u32::from(subsampling_x)) * 4)
                            .map_err(|_| Error::LimitExceeded)?;
                        let y = usize::try_from((block.row >> u32::from(subsampling_y)) * 4)
                            .map_err(|_| Error::LimitExceeded)?;
                        reconstruction::paint_palette_map(
                            buffer.u.as_mut().ok_or(Error::InvalidObu)?,
                            x,
                            y,
                            map,
                            &colors.u[..usize::from(colors.sizes[1])],
                            sequence.bit_depth,
                        )?;
                        reconstruction::paint_palette_map(
                            buffer.v.as_mut().ok_or(Error::InvalidObu)?,
                            x,
                            y,
                            map,
                            &colors.v[..usize::from(colors.sizes[1])],
                            sequence.bit_depth,
                        )?;
                    }
                }
                if let Some(intrabc_mv) = intrabc_motion {
                    let reference = buffer.clone();
                    for plane_index in 0..(if availability.has_chroma { 3 } else { 1 }) {
                        let chroma = plane_index != 0;
                        let plane_size = size.plane_residual_size(
                            chroma && subsampling_x,
                            chroma && subsampling_y,
                        )?;
                        let (width, height) = plane_size.dimensions();
                        let sub_x = u32::from(chroma && subsampling_x);
                        let sub_y = u32::from(chroma && subsampling_y);
                        let region = prediction::PredictionRegion {
                            x: usize::try_from((block.column >> sub_x) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                            y: usize::try_from((block.row >> sub_y) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                            width: usize::from(width),
                            height: usize::from(height),
                        };
                        let reference_plane = match plane_index {
                            0 => &reference.y,
                            1 => reference.u.as_ref().ok_or(Error::InvalidObu)?,
                            2 => reference.v.as_ref().ok_or(Error::InvalidObu)?,
                            _ => return Err(Error::InvalidObu),
                        };
                        let destination = match plane_index {
                            0 => &mut buffer.y,
                            1 => buffer.u.as_mut().ok_or(Error::InvalidObu)?,
                            2 => buffer.v.as_mut().ok_or(Error::InvalidObu)?,
                            _ => return Err(Error::InvalidObu),
                        };
                        inter::predict_scaled_inter_block(
                            destination,
                            inter::ScaledInterBlockConfig {
                                region,
                                frame_width: header.frame_width,
                                frame_height: header.frame_height,
                                bit_depth: sequence.bit_depth,
                                subsampling_x: chroma && subsampling_x,
                                subsampling_y: chroma && subsampling_y,
                                force_integer_mv: true,
                                motion_mode: mode::MotionMode::Simple,
                                local_warp: None,
                                horizontal_filter: inter::InterpolationFilter::Bilinear,
                                vertical_filter: inter::InterpolationFilter::Bilinear,
                                first: inter::InterPredictionSource {
                                    reference: reference_plane,
                                    reference_upscaled_width: header.frame_width,
                                    reference_height: header.frame_height,
                                    motion_vector: intrabc_mv,
                                    global_motion: motion::GlobalMotion::default(),
                                    global_mode: false,
                                    reference_scaled: false,
                                },
                                second: None,
                                blend: inter::CompoundBlend::Average,
                                mask_output: None,
                            },
                        )?;
                    }
                } else if prefix.is_inter {
                    let motion = inter_motion.ok_or(Error::InvalidObu)?;
                    let post = inter_post.ok_or(Error::InvalidObu)?;
                    let local_warp = if post.motion_mode == mode::MotionMode::LocalWarp {
                        Some(
                            motion::derive_local_warp(
                                grid,
                                block,
                                bounds,
                                post.references[0],
                                motion.motion_vectors[0],
                            )?
                            .ok_or(Error::InvalidObu)?,
                        )
                    } else {
                        None
                    };
                    let compound = post.references[1] > 0;
                    let blend = match post.compound.kind {
                        mode::CompoundType::Average => inter::CompoundBlend::Average,
                        mode::CompoundType::Distance => {
                            let mut distances = [0u8; 2];
                            for (list, distance) in distances.iter_mut().enumerate() {
                                let reference = usize::try_from(post.references[list] - 1)
                                    .map_err(|_| Error::InvalidObu)?;
                                let slot = usize::from(header.ref_frame_idx[reference]);
                                *distance = u8::try_from(
                                    motion::relative_order_hint_distance(
                                        reference_info[slot].order_hint,
                                        header.order_hint,
                                        sequence.order_hint_bits,
                                    )?
                                    .unsigned_abs()
                                    .min(31),
                                )
                                .map_err(|_| Error::LimitExceeded)?;
                            }
                            let weights = inter::distance_weights(distances)?;
                            inter::CompoundBlend::Distance {
                                forward: weights[0],
                                backward: weights[1],
                            }
                        }
                        mode::CompoundType::Wedge => {
                            let (luma_width, luma_height) = size.dimensions();
                            inter::CompoundBlend::Wedge {
                                luma_width,
                                luma_height,
                                index: post.compound.wedge_index,
                                sign: post.compound.wedge_sign,
                                subsampling_x: false,
                                subsampling_y: false,
                            }
                        }
                        mode::CompoundType::DifferenceWeighted => {
                            inter::CompoundBlend::DifferenceWeighted {
                                inverse: post.compound.mask_type,
                            }
                        }
                        mode::CompoundType::Intra => inter::CompoundBlend::Average,
                    };
                    let mut difference_mask = Vector::new();
                    for plane_index in 0..(if availability.has_chroma { 3 } else { 1 }) {
                        let chroma = plane_index != 0;
                        let plane_size = size.plane_residual_size(
                            chroma && subsampling_x,
                            chroma && subsampling_y,
                        )?;
                        let (plane_width, plane_height) = plane_size.dimensions();
                        let sub_x = u32::from(chroma && subsampling_x);
                        let sub_y = u32::from(chroma && subsampling_y);
                        let region = prediction::PredictionRegion {
                            x: usize::try_from((block.column >> sub_x) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                            y: usize::try_from((block.row >> sub_y) * 4)
                                .map_err(|_| Error::LimitExceeded)?,
                            width: usize::from(plane_width),
                            height: usize::from(plane_height),
                        };
                        let source =
                            |list: usize| -> Result<inter::InterPredictionSource<'_>, Error> {
                                let reference = usize::try_from(post.references[list] - 1)
                                    .map_err(|_| Error::InvalidObu)?;
                                let slot = usize::from(header.ref_frame_idx[reference]);
                                let reference_buffer = reference_buffers[slot]
                                    .as_ref()
                                    .ok_or(Error::MissingReference)?;
                                let reference_plane = match plane_index {
                                    0 => &reference_buffer.y,
                                    1 => reference_buffer.u.as_ref().ok_or(Error::InvalidObu)?,
                                    2 => reference_buffer.v.as_ref().ok_or(Error::InvalidObu)?,
                                    _ => return Err(Error::InvalidObu),
                                };
                                let info = reference_info[slot];
                                let reference_scaled = info.upscaled_width != header.frame_width
                                    || info.frame_height != header.frame_height;
                                Ok(inter::InterPredictionSource {
                                    reference: reference_plane,
                                    reference_upscaled_width: info.upscaled_width,
                                    reference_height: info.frame_height,
                                    motion_vector: motion.motion_vectors[list],
                                    global_motion: header.global_motion[reference],
                                    global_mode: matches!(motion.y_mode, 16 | 24),
                                    reference_scaled,
                                })
                            };
                        let destination = match plane_index {
                            0 => &mut buffer.y,
                            1 => buffer.u.as_mut().ok_or(Error::InvalidObu)?,
                            2 => buffer.v.as_mut().ok_or(Error::InvalidObu)?,
                            _ => return Err(Error::InvalidObu),
                        };
                        let chroma_difference_mask;
                        let blend = match blend {
                            inter::CompoundBlend::Wedge {
                                luma_width,
                                luma_height,
                                index,
                                sign,
                                ..
                            } => inter::CompoundBlend::Wedge {
                                luma_width,
                                luma_height,
                                index,
                                sign,
                                subsampling_x: chroma && subsampling_x,
                                subsampling_y: chroma && subsampling_y,
                            },
                            inter::CompoundBlend::DifferenceWeighted { .. } if chroma => {
                                let (luma_width, luma_height) = size.dimensions();
                                chroma_difference_mask = inter::subsample_mask(
                                    &difference_mask,
                                    usize::from(luma_width),
                                    usize::from(luma_height),
                                    subsampling_x,
                                    subsampling_y,
                                )?;
                                inter::CompoundBlend::Mask(&chroma_difference_mask)
                            }
                            other => other,
                        };
                        let mask_output =
                            matches!(blend, inter::CompoundBlend::DifferenceWeighted { .. })
                                .then_some(&mut difference_mask);
                        inter::predict_scaled_inter_block(
                            destination,
                            inter::ScaledInterBlockConfig {
                                region,
                                frame_width: header.frame_width,
                                frame_height: header.frame_height,
                                bit_depth: sequence.bit_depth,
                                subsampling_x: chroma && subsampling_x,
                                subsampling_y: chroma && subsampling_y,
                                force_integer_mv: header.force_integer_mv,
                                motion_mode: post.motion_mode,
                                local_warp,
                                horizontal_filter: inter::InterpolationFilter::from_av1(
                                    post.interpolation_filters[0],
                                )?,
                                vertical_filter: inter::InterpolationFilter::from_av1(
                                    post.interpolation_filters[1],
                                )?,
                                first: source(0)?,
                                second: if compound { Some(source(1)?) } else { None },
                                blend,
                                mask_output,
                            },
                        )?;
                        if post.inter_intra.enabled {
                            let count = region
                                .width
                                .checked_mul(region.height)
                                .ok_or(Error::LimitExceeded)?;
                            let mut inter_prediction =
                                Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
                            for row in 0..region.height {
                                for column in 0..region.width {
                                    inter_prediction
                                        .try_push(
                                            destination
                                                .sample(region.x + column, region.y + row)?,
                                        )
                                        .map_err(|_| Error::LimitExceeded)?;
                                }
                            }
                            let intra_mode = match post.inter_intra.mode {
                                0 => 0,
                                1 => 1,
                                2 => 2,
                                3 => 9,
                                _ => return Err(Error::InvalidObu),
                            };
                            prediction::predict_intra_block(
                                destination,
                                prediction::IntraPredictionConfig {
                                    region,
                                    bit_depth: sequence.bit_depth,
                                    mode: intra_mode,
                                    angle_delta: 0,
                                    filter_intra_mode: None,
                                    have_left: if chroma {
                                        availability.left_chroma
                                    } else {
                                        availability.left
                                    },
                                    have_above: if chroma {
                                        availability.upper_chroma
                                    } else {
                                        availability.upper
                                    },
                                    have_above_right: if chroma {
                                        availability.upper_chroma
                                    } else {
                                        availability.upper
                                    },
                                    have_below_left: if chroma {
                                        availability.left_chroma
                                    } else {
                                        availability.left
                                    },
                                },
                            )
                            .map_err(|error| {
                                prediction_stage_error(error, PredictionStage::IntraPixels)
                            })?;
                            if post.inter_intra.wedge {
                                let (luma_width, luma_height) = size.dimensions();
                                inter::blend_wedge_inter_intra(
                                    destination,
                                    region,
                                    luma_width,
                                    luma_height,
                                    post.inter_intra.wedge_index,
                                    false,
                                    &inter_prediction,
                                )?;
                            } else {
                                inter::blend_inter_intra(
                                    destination,
                                    region,
                                    post.inter_intra.mode,
                                    &inter_prediction,
                                )?;
                            }
                        }
                        if post.motion_mode == mode::MotionMode::Obmc {
                            let scratch_width = destination.width();
                            let scratch_height = destination.height();
                            inter::apply_obmc_neighbors(
                                grid,
                                destination,
                                inter::ObmcTraversalConfig {
                                    block,
                                    tile: bounds,
                                    prediction_width: u16::try_from(region.width)
                                        .map_err(|_| Error::LimitExceeded)?,
                                    prediction_height: u16::try_from(region.height)
                                        .map_err(|_| Error::LimitExceeded)?,
                                    subsampling_x: chroma && subsampling_x,
                                    subsampling_y: chroma && subsampling_y,
                                    residual_at_least_8x8: plane_size.dimensions().0 >= 8
                                        && plane_size.dimensions().1 >= 8,
                                },
                                |neighbor| {
                                    let reference = usize::try_from(neighbor.reference_frame - 1)
                                        .map_err(|_| Error::InvalidObu)?;
                                    let slot = usize::from(header.ref_frame_idx[reference]);
                                    let reference_buffer = reference_buffers[slot]
                                        .as_ref()
                                        .ok_or(Error::MissingReference)?;
                                    let reference_plane = match plane_index {
                                        0 => &reference_buffer.y,
                                        1 => {
                                            reference_buffer.u.as_ref().ok_or(Error::InvalidObu)?
                                        }
                                        2 => {
                                            reference_buffer.v.as_ref().ok_or(Error::InvalidObu)?
                                        }
                                        _ => return Err(Error::InvalidObu),
                                    };
                                    let info = reference_info[slot];
                                    let mut scratch = reconstruction::Plane::new(
                                        scratch_width,
                                        scratch_height,
                                        0,
                                    )?;
                                    inter::predict_scaled_inter_block(
                                        &mut scratch,
                                        inter::ScaledInterBlockConfig {
                                            region: neighbor.region,
                                            frame_width: header.frame_width,
                                            frame_height: header.frame_height,
                                            bit_depth: sequence.bit_depth,
                                            subsampling_x: chroma && subsampling_x,
                                            subsampling_y: chroma && subsampling_y,
                                            force_integer_mv: header.force_integer_mv,
                                            motion_mode: mode::MotionMode::Simple,
                                            local_warp: None,
                                            horizontal_filter:
                                                inter::InterpolationFilter::from_av1(
                                                    post.interpolation_filters[0],
                                                )?,
                                            vertical_filter: inter::InterpolationFilter::from_av1(
                                                post.interpolation_filters[1],
                                            )?,
                                            first: inter::InterPredictionSource {
                                                reference: reference_plane,
                                                reference_upscaled_width: info.upscaled_width,
                                                reference_height: info.frame_height,
                                                motion_vector: neighbor.motion_vector,
                                                global_motion: header.global_motion[reference],
                                                global_mode: false,
                                                reference_scaled: info.upscaled_width
                                                    != header.frame_width
                                                    || info.frame_height != header.frame_height,
                                            },
                                            second: None,
                                            blend: inter::CompoundBlend::Average,
                                            mask_output: None,
                                        },
                                    )?;
                                    let count = neighbor
                                        .region
                                        .width
                                        .checked_mul(neighbor.region.height)
                                        .ok_or(Error::LimitExceeded)?;
                                    let mut samples = Vector::with_capacity(count)
                                        .map_err(|_| Error::LimitExceeded)?;
                                    for row in 0..neighbor.region.height {
                                        for column in 0..neighbor.region.width {
                                            samples
                                                .try_push(scratch.sample(
                                                    neighbor.region.x + column,
                                                    neighbor.region.y + row,
                                                )?)
                                                .map_err(|_| Error::LimitExceeded)?;
                                        }
                                    }
                                    Ok(samples)
                                },
                            )?;
                        }
                    }
                }
                grid.fill_preserving_tx_size(
                    block,
                    block_state::BlockState {
                        size: Some(size),
                        segment_id: prefix.segment_id,
                        skip_mode: prefix.skip_mode,
                        skip: prefix.skip,
                        is_inter: prefix.is_inter || intrabc_motion.is_some(),
                        tx_size: Some(tx_size),
                        qindex: current_qindex,
                        delta_lf,
                        cdef_index,
                        reference_frames: if intrabc_motion.is_some() {
                            [0, -1]
                        } else {
                            inter_post.map_or(references, |post| post.references)
                        },
                        motion_vectors: if let Some(vector) = intrabc_motion {
                            [vector, motion::MotionVector::default()]
                        } else {
                            inter_motion.map_or([motion::MotionVector::default(); 2], |mode| {
                                mode.motion_vectors
                            })
                        },
                        prediction_mode: inter_motion.map_or_else(
                            || intra.map_or(0, |mode| mode.y_mode as u8),
                            |mode| mode.y_mode,
                        ),
                        motion_mode: inter_post.map_or(0, |post| post.motion_mode as u8),
                        compound_type: inter_post.map_or(0, |post| post.compound.kind as u8),
                        compound_group_index: inter_post
                            .map_or(0, |post| post.compound.group_index),
                        compound_index: inter_post.map_or(1, |post| post.compound.compound_index),
                        interpolation_filters: inter_post
                            .map_or([0; 2], |post| post.interpolation_filters),
                        inter_intra_mode: inter_post.map_or(0, |post| post.inter_intra.mode),
                        wedge_index: inter_post.map_or(0, |post| post.compound.wedge_index),
                        wedge_sign: inter_post.is_some_and(|post| post.compound.wedge_sign),
                        mask_type: inter_post.is_some_and(|post| post.compound.mask_type),
                        palette_sizes: intra.map_or([0; 2], |mode| mode.palette_sizes),
                        palette_colors: palette_data
                            .as_ref()
                            .map_or([[0; 8]; 2], |data| [data.0.y, data.0.u]),
                        ..block_state::BlockState::default()
                    },
                )?;
                if let Some(intra) = intra.filter(|mode| !mode.use_intrabc) {
                    let lossless = header.lossless_segments[usize::from(prefix.segment_id)];
                    let qindex = header
                        .segmentation
                        .qindex(usize::from(prefix.segment_id), current_qindex)?;
                    let residual_size = if size.dimensions().0 > 64 || size.dimensions().1 > 64 {
                        partition::BlockSize::Block64x64
                    } else {
                        size
                    };
                    reconstruction::walk_fixed_residual_blocks(
                        reconstruction::FixedResidualWalkConfig {
                            block,
                            size,
                            luma_tx_size: tx_size,
                            lossless,
                            subsampling_x,
                            subsampling_y,
                            has_chroma: availability.has_chroma,
                            mi_columns: mi_dimension(header.frame_width),
                            mi_rows: mi_dimension(header.frame_height),
                        },
                        |plane_index, x, y, transform_size| {
                            let plane = usize::from(plane_index);
                            let chroma = plane != 0;
                            let sub_x = u32::from(chroma && subsampling_x);
                            let sub_y = u32::from(chroma && subsampling_y);
                            let base_x = (block.column >> sub_x) * 4;
                            let base_y = (block.row >> sub_y) * 4;
                            let (prediction_width, prediction_height) = transform_size.dimensions();
                            let prediction_x =
                                usize::try_from(x).map_err(|_| Error::LimitExceeded)?;
                            let prediction_y =
                                usize::try_from(y).map_err(|_| Error::LimitExceeded)?;
                            let (plane_width, plane_height) = match plane {
                                0 => (buffer.y.width(), buffer.y.height()),
                                1 => {
                                    let plane = buffer.u.as_ref().ok_or(Error::InvalidObu)?;
                                    (plane.width(), plane.height())
                                }
                                2 => {
                                    let plane = buffer.v.as_ref().ok_or(Error::InvalidObu)?;
                                    (plane.width(), plane.height())
                                }
                                _ => return Err(Error::InvalidObu),
                            };
                            let visible_width = plane_width
                                .checked_sub(prediction_x)
                                .ok_or(Error::InvalidObu)?
                                .min(usize::from(prediction_width));
                            let visible_height = plane_height
                                .checked_sub(prediction_y)
                                .ok_or(Error::InvalidObu)?
                                .min(usize::from(prediction_height));
                            let cfl = chroma && intra.chroma.mode == mode::ChromaMode::Cfl;
                            let (prediction_mode, angle_delta, filter_intra_mode) = if plane == 0 {
                                (
                                    intra.y_mode as u8,
                                    intra.y_angle_delta,
                                    intra.filter_intra_mode,
                                )
                            } else if cfl {
                                (mode::ChromaMode::Dc as u8, 0, None)
                            } else {
                                (intra.chroma.mode as u8, intra.chroma.angle_delta, None)
                            };
                            let prediction_config = prediction::IntraPredictionConfig {
                                region: prediction::PredictionRegion {
                                    x: prediction_x,
                                    y: prediction_y,
                                    width: visible_width,
                                    height: visible_height,
                                },
                                bit_depth: sequence.bit_depth,
                                mode: prediction_mode,
                                angle_delta,
                                filter_intra_mode,
                                have_left: if chroma {
                                    availability.left_chroma || x > base_x
                                } else {
                                    availability.left || x > base_x
                                },
                                have_above: if chroma {
                                    availability.upper_chroma || y > base_y
                                } else {
                                    availability.upper || y > base_y
                                },
                                have_above_right: if chroma {
                                    availability.upper_chroma || y > base_y
                                } else {
                                    availability.upper || y > base_y
                                },
                                have_below_left: if chroma {
                                    availability.left_chroma || x > base_x
                                } else {
                                    availability.left || x > base_x
                                },
                            };
                            let palette_prediction = palette_data
                                .as_ref()
                                .is_some_and(|data| data.0.sizes[usize::from(chroma)] > 0);
                            match plane {
                                0 if !palette_prediction => prediction::predict_intra_block(
                                    &mut buffer.y,
                                    prediction_config,
                                )
                                .map_err(|error| {
                                    prediction_stage_error(error, PredictionStage::IntraPixels)
                                })?,
                                0 => {}
                                1 => {
                                    let chroma = buffer.u.as_mut().ok_or(Error::InvalidObu)?;
                                    if !palette_prediction {
                                        prediction::predict_intra_block(chroma, prediction_config)
                                            .map_err(|error| {
                                                prediction_stage_error(
                                                    error,
                                                    PredictionStage::IntraPixels,
                                                )
                                            })?;
                                    }
                                    if cfl {
                                        prediction::predict_chroma_from_luma(
                                            &buffer.y,
                                            chroma,
                                            prediction::ChromaFromLumaConfig {
                                                region: prediction_config.region,
                                                subsampling_x,
                                                subsampling_y,
                                                alpha: intra.chroma.cfl_alphas[0],
                                                bit_depth: sequence.bit_depth,
                                            },
                                        )
                                        .map_err(
                                            |error| {
                                                prediction_stage_error(
                                                    error,
                                                    PredictionStage::ChromaFromLuma,
                                                )
                                            },
                                        )?;
                                    }
                                }
                                2 => {
                                    let chroma = buffer.v.as_mut().ok_or(Error::InvalidObu)?;
                                    if !palette_prediction {
                                        prediction::predict_intra_block(chroma, prediction_config)
                                            .map_err(|error| {
                                                prediction_stage_error(
                                                    error,
                                                    PredictionStage::IntraPixels,
                                                )
                                            })?;
                                    }
                                    if cfl {
                                        prediction::predict_chroma_from_luma(
                                            &buffer.y,
                                            chroma,
                                            prediction::ChromaFromLumaConfig {
                                                region: prediction_config.region,
                                                subsampling_x,
                                                subsampling_y,
                                                alpha: intra.chroma.cfl_alphas[1],
                                                bit_depth: sequence.bit_depth,
                                            },
                                        )
                                        .map_err(
                                            |error| {
                                                prediction_stage_error(
                                                    error,
                                                    PredictionStage::ChromaFromLuma,
                                                )
                                            },
                                        )?;
                                    }
                                }
                                _ => return Err(Error::InvalidObu),
                            }
                            let plane_residual = residual_size.plane_residual_size(
                                chroma && subsampling_x,
                                chroma && subsampling_y,
                            )?;
                            let x4 = x / 4;
                            let y4 = y / 4;
                            grid.fill_loop_filter_tx_size(
                                plane,
                                y4,
                                x4,
                                transform_size,
                                subsampling_x,
                                subsampling_y,
                            )?;
                            let tile_start = (
                                bounds.column_start >> u32::from(chroma && subsampling_x),
                                bounds.row_start >> u32::from(chroma && subsampling_y),
                            );
                            let contexts =
                                coefficient_contexts.as_mut().ok_or(Error::InvalidObu)?;
                            if prefix.skip {
                                return contexts.update(
                                    plane,
                                    x4,
                                    y4,
                                    transform_size,
                                    coeff::CoefficientBlockResult::default(),
                                );
                            }
                            let derived = contexts.derive(coeff::CoefficientContextConfig {
                                plane,
                                x4,
                                y4,
                                size: transform_size,
                                residual_dimensions: {
                                    let (width, height) = plane_residual.dimensions();
                                    (
                                        u8::try_from(width).map_err(|_| Error::LimitExceeded)?,
                                        u8::try_from(height).map_err(|_| Error::LimitExceeded)?,
                                    )
                                },
                                tile_start,
                            })?;
                            let configured_tx_type = if plane == 0 {
                                transform::TxType::DctDct
                            } else {
                                transform::chroma_intra_tx_type(
                                    intra.chroma.mode as u8,
                                    transform_size,
                                    header.reduced_tx_set,
                                    lossless,
                                )?
                            };
                            let tx_type_selection = (plane == 0 && !lossless).then_some(
                                coeff::TxTypeSelection::Intra {
                                    reduced_tx_set: header.reduced_tx_set,
                                    direction: intra.y_mode as u8,
                                },
                            );
                            let destination = match plane {
                                0 => &mut buffer.y,
                                1 => buffer.u.as_mut().ok_or(Error::InvalidObu)?,
                                2 => buffer.v.as_mut().ok_or(Error::InvalidObu)?,
                                _ => return Err(Error::InvalidObu),
                            };
                            decode_counts[2] = decode_counts[2].saturating_add(1);
                            let result = reconstruction::decode_and_reconstruct_transform_block(
                                decoder,
                                cdfs,
                                destination,
                                reconstruction::DecodedTransformBlockConfig {
                                    x: usize::try_from(x).map_err(|_| Error::LimitExceeded)?,
                                    y: usize::try_from(y).map_err(|_| Error::LimitExceeded)?,
                                    size: transform_size,
                                    tx_type: configured_tx_type,
                                    bit_depth: sequence.bit_depth,
                                    lossless,
                                    plane: plane_index,
                                    qindex,
                                    base_q_index: header.quantization.base_q_idx,
                                    dc_sign_context: derived.dc_sign,
                                    txb_skip_context: derived.txb_skip,
                                    tx_type_selection,
                                    quantization: &header.quantization,
                                },
                            )
                            .map_err(coefficient_syntax_error)?;
                            if result.eob != 0 {
                                decode_counts[3] = decode_counts[3].saturating_add(1);
                            }
                            if traced_transform_count < transform_types.len() {
                                transform_types[traced_transform_count] = result.tx_type as u8;
                            }
                            traced_transform_count = traced_transform_count.saturating_add(1);
                            decode_counts[4] =
                                decode_counts[4].saturating_add(u32::from(result.eob));
                            if plane == 0 {
                                grid.fill_tx_type(y4, x4, transform_size, result.tx_type)?;
                            }
                            contexts.update(plane, x4, y4, transform_size, result)
                        },
                    )?;
                    return Ok(());
                }
                let lossless = header.lossless_segments[usize::from(prefix.segment_id)];
                let qindex = header
                    .segmentation
                    .qindex(usize::from(prefix.segment_id), current_qindex)?;
                let residual_size = if size.dimensions().0 > 64 || size.dimensions().1 > 64 {
                    partition::BlockSize::Block64x64
                } else {
                    size
                };
                let residual_blocks = reconstruction::collect_residual_blocks(
                    grid,
                    reconstruction::FixedResidualWalkConfig {
                        block,
                        size,
                        luma_tx_size: tx_size,
                        lossless,
                        subsampling_x,
                        subsampling_y,
                        has_chroma: availability.has_chroma,
                        mi_columns: mi_dimension(header.frame_width),
                        mi_rows: mi_dimension(header.frame_height),
                    },
                    true,
                )?;
                for residual in residual_blocks {
                    let plane = usize::from(residual.plane);
                    let chroma = plane != 0;
                    let x4 = residual.x / 4;
                    let y4 = residual.y / 4;
                    grid.fill_loop_filter_tx_size(
                        plane,
                        y4,
                        x4,
                        residual.size,
                        subsampling_x,
                        subsampling_y,
                    )?;
                    let plane_residual = residual_size
                        .plane_residual_size(chroma && subsampling_x, chroma && subsampling_y)?;
                    let contexts = coefficient_contexts.as_mut().ok_or(Error::InvalidObu)?;
                    if prefix.skip {
                        contexts.update(
                            plane,
                            x4,
                            y4,
                            residual.size,
                            coeff::CoefficientBlockResult::default(),
                        )?;
                        continue;
                    }
                    let derived = contexts.derive(coeff::CoefficientContextConfig {
                        plane,
                        x4,
                        y4,
                        size: residual.size,
                        residual_dimensions: {
                            let (width, height) = plane_residual.dimensions();
                            (
                                u8::try_from(width).map_err(|_| Error::LimitExceeded)?,
                                u8::try_from(height).map_err(|_| Error::LimitExceeded)?,
                            )
                        },
                        tile_start: (
                            bounds.column_start >> u32::from(chroma && subsampling_x),
                            bounds.row_start >> u32::from(chroma && subsampling_y),
                        ),
                    })?;
                    let configured_tx_type = if plane == 0 {
                        transform::TxType::DctDct
                    } else {
                        let luma_column = block.column.max(x4 << u32::from(subsampling_x));
                        let luma_row = block.row.max(y4 << u32::from(subsampling_y));
                        let luma_type = grid
                            .get(luma_row, luma_column)
                            .ok_or(Error::InvalidObu)?
                            .tx_type;
                        transform::chroma_inter_tx_type(
                            luma_type,
                            residual.size,
                            header.reduced_tx_set,
                            lossless,
                        )?
                    };
                    let destination = match plane {
                        0 => &mut buffer.y,
                        1 => buffer.u.as_mut().ok_or(Error::InvalidObu)?,
                        2 => buffer.v.as_mut().ok_or(Error::InvalidObu)?,
                        _ => return Err(Error::InvalidObu),
                    };
                    decode_counts[2] = decode_counts[2].saturating_add(1);
                    let result = reconstruction::decode_and_reconstruct_transform_block(
                        decoder,
                        cdfs,
                        destination,
                        reconstruction::DecodedTransformBlockConfig {
                            x: usize::try_from(residual.x).map_err(|_| Error::LimitExceeded)?,
                            y: usize::try_from(residual.y).map_err(|_| Error::LimitExceeded)?,
                            size: residual.size,
                            tx_type: configured_tx_type,
                            bit_depth: sequence.bit_depth,
                            lossless,
                            plane: residual.plane,
                            qindex,
                            base_q_index: header.quantization.base_q_idx,
                            dc_sign_context: derived.dc_sign,
                            txb_skip_context: derived.txb_skip,
                            tx_type_selection: (plane == 0 && !lossless).then_some(
                                coeff::TxTypeSelection::Inter {
                                    reduced_tx_set: header.reduced_tx_set,
                                },
                            ),
                            quantization: &header.quantization,
                        },
                    )
                    .map_err(coefficient_syntax_error)?;
                    if result.eob != 0 {
                        decode_counts[3] = decode_counts[3].saturating_add(1);
                    }
                    if traced_transform_count < transform_types.len() {
                        transform_types[traced_transform_count] = result.tx_type as u8;
                    }
                    traced_transform_count = traced_transform_count.saturating_add(1);
                    decode_counts[4] = decode_counts[4].saturating_add(u32::from(result.eob));
                    if plane == 0 {
                        grid.fill_tx_type(y4, x4, residual.size, result.tx_type)?;
                    }
                    contexts.update(plane, x4, y4, residual.size, result)?;
                }
                Ok(())
            },
        )
        .map_err(|error| match (error, current_block) {
            (
                Error::InvalidTileTermination {
                    bit_position,
                    max_bits,
                },
                _,
            ) => Error::InvalidTileDecode {
                bit_position,
                max_bits,
                blocks: decode_counts[0],
                skipped_blocks: decode_counts[1],
                transform_blocks: decode_counts[2],
                nonzero_transform_blocks: decode_counts[3],
                coefficient_count: decode_counts[4],
                block_flags,
                reference_frames,
                inter_modes,
                motion_vectors,
                luma_modes,
                chroma_modes,
                transform_types,
            },
            (Error::InvalidObu, Some((row, column, size))) => {
                Error::InvalidBlockPosition { row, column, size }
            }
            (other, _) => other,
        })?;
        if self.camera_tile_decode {
            let camera_bounds = layout.bounds(
                0,
                mi_dimension(header.frame_width),
                mi_dimension(header.frame_height),
                sequence.use_128x128_superblock,
            )?;
            for row in camera_bounds.row_start..camera_bounds.row_end {
                for column in camera_bounds.column_start..camera_bounds.column_end {
                    let state = grid.get(row, column).ok_or(Error::InvalidObu)?;
                    if state.is_inter && state.reference_frames[0] != 1 {
                        return Err(Error::InvalidObu);
                    }
                }
            }
        }
        let mut decoded_buffer = buffer.clone();
        loop_filter::apply_frame(
            &mut decoded_buffer,
            grid,
            &header.loop_filter,
            &header.segmentation,
            header.delta_params.delta_lf_multi,
            header.frame_width,
            header.frame_height,
        )?;
        let mut deblocked_buffer = decoded_buffer.clone();
        for tile_number in 0..layout.tile_count() {
            cdef::apply_frame_region(
                &mut decoded_buffer,
                grid,
                layout.bounds(
                    tile_number,
                    mi_dimension(header.frame_width),
                    mi_dimension(header.frame_height),
                    sequence.use_128x128_superblock,
                )?,
                &header.cdef,
            )?;
        }
        if header.use_superres {
            deblocked_buffer = superres::upscale(
                &deblocked_buffer,
                header.frame_width,
                header.upscaled_width,
                header.frame_height,
            )?;
            decoded_buffer = superres::upscale(
                &decoded_buffer,
                header.frame_width,
                header.upscaled_width,
                header.frame_height,
            )?;
        }
        decoded_buffer = restoration::apply_frame(
            &deblocked_buffer,
            &decoded_buffer,
            &restoration_units,
            &header.restoration,
            header.upscaled_width,
            header.frame_height,
        )?;
        let decoded_grid = grid.clone();
        let saved_cdfs = if header.disable_frame_end_update_cdf {
            initial_cdfs.clone()
        } else {
            final_cdfs
        };
        let reference_frame = decoded_buffer.clone().into_frame()?;
        film_grain::apply(
            &mut decoded_buffer,
            &header.film_grain,
            sequence.matrix_coefficients,
            header.upscaled_width,
            header.frame_height,
        )?;
        let mut frame = decoded_buffer.into_frame()?;
        frame.presentation_time = header.frame_presentation_time;
        frame.buffer_removal_times = header.buffer_removal_times.clone();
        let saved_order_hints = core::array::from_fn(|reference| {
            if reference == 0 {
                header.order_hint
            } else {
                reference_info[usize::from(header.ref_frame_idx[reference - 1])].order_hint
            }
        });
        let info = frame_header::ReferenceInfo {
            valid: true,
            frame_id: header.current_frame_id.unwrap_or(0),
            order_hint: header.order_hint,
            frame_type: header.frame_type,
            showable_frame: header.showable_frame,
            upscaled_width: header.upscaled_width,
            frame_width: header.frame_width,
            frame_height: header.frame_height,
            render_width: header.render_width,
            render_height: header.render_height,
            segmentation: header.segmentation,
            loop_filter: header.loop_filter,
            global_motion: header.global_motion,
            film_grain: header.film_grain,
        };
        for slot in 0..frame_header::NUM_REF_FRAMES {
            if header.refresh_frame_flags & (1 << slot) != 0 {
                self.references[slot] = Some(reference_frame.clone());
                self.reference_info[slot] = info;
                self.reference_cdfs[slot] = Some(saved_cdfs.clone());
                self.reference_grids[slot] = Some(decoded_grid.clone());
                self.reference_order_hints[slot] = saved_order_hints;
            }
        }
        self.pending_decode = None;
        self.pending_frame_header = None;
        self.pending_frame_header_bytes.clear();
        self.note_frame()?;
        Ok(header.show_frame.then_some(frame))
    }

    fn note_frame(&mut self) -> Result<(), Error> {
        self.decoded_frames = self
            .decoded_frames
            .checked_add(1)
            .ok_or(Error::LimitExceeded)?;
        if self.decoded_frames > self.limits.max_frames {
            return Err(Error::LimitExceeded);
        }
        Ok(())
    }
}

/// Decode an IVF file containing AV1 frame packets.
pub fn decode_ivf(data: &[u8]) -> Result<Vector<Frame>, Error> {
    if data.len() < 32
        || &data[..4] != b"DKIF"
        || le16(data, 4)? != 0
        || le16(data, 6)? < 32
        || &data[8..12] != b"AV01"
    {
        return Err(Error::InvalidIvf);
    }
    let header_len = le16(data, 6)? as usize;
    if header_len > data.len() {
        return Err(Error::Truncated);
    }
    let declared = le32(data, 24)? as usize;
    let mut decoder = Decoder::new();
    let mut out = Vector::new();
    let mut pos = header_len;
    let mut packets = 0usize;
    while pos < data.len() {
        if data.len() - pos < 12 {
            return Err(Error::Truncated);
        }
        let len = le32(data, pos)? as usize;
        pos += 12;
        let end = pos.checked_add(len).ok_or(Error::LimitExceeded)?;
        if end > data.len() {
            return Err(Error::Truncated);
        }
        for frame in decoder.decode_obus(&data[pos..end])? {
            vector_push(&mut out, frame)?;
        }
        pos = end;
        packets += 1;
    }
    if declared != 0 && packets != declared {
        return Err(Error::InvalidIvf);
    }
    Ok(out)
}

struct Obu<'a> {
    kind: u8,
    extension: bool,
    temporal_id: u8,
    spatial_id: u8,
    payload: &'a [u8],
}

const fn obu_in_operating_point(
    operating_point_idc: u16,
    kind: u8,
    extension: bool,
    temporal_id: u8,
    spatial_id: u8,
) -> bool {
    if kind == OBU_SEQUENCE_HEADER
        || kind == OBU_TEMPORAL_DELIMITER
        || operating_point_idc == 0
        || !extension
    {
        return true;
    }
    operating_point_idc & (1u16 << temporal_id) != 0
        && operating_point_idc & (1u16 << (spatial_id + 8)) != 0
}

fn validate_frame_header_copy(
    seen_frame_header: bool,
    original: &[u8],
    copy: &[u8],
) -> Result<(), Error> {
    if !seen_frame_header || original != copy {
        return Err(Error::InvalidObu);
    }
    Ok(())
}

fn validate_operating_point_extension(
    operating_point_idc: u16,
    kind: u8,
    extension: bool,
) -> Result<(), Error> {
    if kind != OBU_SEQUENCE_HEADER && extension != (operating_point_idc != 0) {
        return Err(Error::InvalidObu);
    }
    Ok(())
}

fn parse_obu(data: &[u8], max: usize) -> Result<(Obu<'_>, usize), Error> {
    let header = *data.first().ok_or(Error::Truncated)?;
    if header & 0x80 != 0 || header & 1 != 0 {
        return Err(Error::InvalidObu);
    }
    let kind = (header >> 3) & 15;
    let extension = header & 4 != 0;
    let has_size = header & 2 != 0;
    if !has_size {
        // Section 5.2 requires an internal size in the low-overhead format.
        // Annex B is normalized by `decode_external_obu`, where the enclosing
        // length supplies the otherwise absent `obu_size`.
        return Err(Error::InvalidObu);
    }
    let mut pos = 1;
    let mut temporal_id = 0;
    let mut spatial_id = 0;
    if extension {
        let ext = *data.get(pos).ok_or(Error::Truncated)?;
        if ext & 7 != 0 {
            return Err(Error::InvalidObu);
        }
        temporal_id = ext >> 5;
        spatial_id = (ext >> 3) & 3;
        pos += 1;
    }
    let (size, leb_len) = leb128(&data[pos..])?;
    pos += leb_len;
    if size > max {
        return Err(Error::LimitExceeded);
    }
    let end = pos.checked_add(size).ok_or(Error::LimitExceeded)?;
    let payload = data.get(pos..end).ok_or(Error::Truncated)?;
    Ok((
        Obu {
            kind,
            extension,
            temporal_id,
            spatial_id,
            payload,
        },
        end,
    ))
}

fn parse_sequence_header(data: &[u8]) -> Result<Sequence, Error> {
    let mut b = Bits::new(data);
    let profile = b.read(3)? as u8;
    if profile > 2 {
        return Err(Error::InvalidSequence);
    }
    let still_picture = b.bit()?;
    let reduced = b.bit()?;
    if reduced && !still_picture {
        return Err(Error::InvalidSequence);
    }
    let mut timing = None;
    let mut decoder_model = None;
    let mut initial_display_delay_present = false;
    let mut operating_points = Vector::new();
    if reduced {
        vector_push(
            &mut operating_points,
            OperatingPoint {
                idc: 0,
                level: b.read(5)? as u8,
                tier: false,
                decoder_buffer_delay: None,
                encoder_buffer_delay: None,
                low_delay_mode: false,
                initial_display_delay: None,
            },
        )?;
    } else {
        if b.bit()? {
            let num_units_in_display_tick = b.read(32)? as u32;
            let time_scale = b.read(32)? as u32;
            if num_units_in_display_tick == 0 || time_scale == 0 {
                return Err(Error::InvalidSequence);
            }
            let num_ticks_per_picture = if b.bit()? {
                Some(
                    read_uvlc(&mut b)?
                        .checked_add(1)
                        .ok_or(Error::InvalidSequence)?,
                )
            } else {
                None
            };
            timing = Some(TimingInfo {
                num_units_in_display_tick,
                time_scale,
                num_ticks_per_picture,
            });
            if b.bit()? {
                let buffer_delay_length = b.read(5)? as u8 + 1;
                let num_units_in_decoding_tick = b.read(32)? as u32;
                if num_units_in_decoding_tick == 0 {
                    return Err(Error::InvalidSequence);
                }
                decoder_model = Some(DecoderModelInfo {
                    buffer_delay_length,
                    num_units_in_decoding_tick,
                    buffer_removal_time_length: b.read(5)? as u8 + 1,
                    frame_presentation_time_length: b.read(5)? as u8 + 1,
                });
            }
        }
        initial_display_delay_present = b.bit()?;
        let operating_point_count = b.read(5)? as usize + 1;
        for _ in 0..operating_point_count {
            let idc = b.read(12)? as u16;
            let level = b.read(5)? as u8;
            let tier = level > 7 && b.bit()?;
            let (decoder_buffer_delay, encoder_buffer_delay, low_delay_mode) =
                if let Some(model) = &decoder_model {
                    if b.bit()? {
                        (
                            Some(b.read(model.buffer_delay_length)? as u32),
                            Some(b.read(model.buffer_delay_length)? as u32),
                            b.bit()?,
                        )
                    } else {
                        (None, None, false)
                    }
                } else {
                    (None, None, false)
                };
            let initial_display_delay = if initial_display_delay_present && b.bit()? {
                Some(b.read(4)? as u8 + 1)
            } else {
                None
            };
            vector_push(
                &mut operating_points,
                OperatingPoint {
                    idc,
                    level,
                    tier,
                    decoder_buffer_delay,
                    encoder_buffer_delay,
                    low_delay_mode,
                    initial_display_delay,
                },
            )?;
        }
    }
    let width_bits = b.read(4)? as u8 + 1;
    let height_bits = b.read(4)? as u8 + 1;
    let max_width = b.read(width_bits)? as u32 + 1;
    let max_height = b.read(height_bits)? as u32 + 1;
    if max_width == 0 || max_height == 0 {
        return Err(Error::InvalidSequence);
    }
    let frame_id_numbers_present = !reduced && b.bit()?;
    let (delta_frame_id_length, frame_id_length) = if frame_id_numbers_present {
        let delta = b.read(4)? as u8 + 2;
        let additional = b.read(3)? as u8 + 1;
        (delta, delta + additional)
    } else {
        (0, 0)
    };
    let use_128x128_superblock = b.bit()?;
    let enable_filter_intra = b.bit()?;
    let enable_intra_edge_filter = b.bit()?;
    let mut enable_interintra_compound = false;
    let mut enable_masked_compound = false;
    let mut enable_warped_motion = false;
    let mut enable_dual_filter = false;
    let mut enable_order_hint = false;
    let mut enable_jnt_comp = false;
    let mut enable_ref_frame_mvs = false;
    let mut seq_force_screen_content_tools = 2;
    let mut seq_force_integer_mv = 2;
    let mut order_hint_bits = 0;
    if !reduced {
        enable_interintra_compound = b.bit()?;
        enable_masked_compound = b.bit()?;
        enable_warped_motion = b.bit()?;
        enable_dual_filter = b.bit()?;
        enable_order_hint = b.bit()?;
        if enable_order_hint {
            enable_jnt_comp = b.bit()?;
            enable_ref_frame_mvs = b.bit()?;
        }
        seq_force_screen_content_tools = if b.bit()? { 2 } else { b.read(1)? as u8 };
        seq_force_integer_mv = if seq_force_screen_content_tools > 0 {
            if b.bit()? { 2 } else { b.read(1)? as u8 }
        } else {
            2
        };
        if enable_order_hint {
            order_hint_bits = b.read(3)? as u8 + 1;
        }
    }
    let enable_superres = b.bit()?;
    let enable_cdef = b.bit()?;
    let enable_restoration = b.bit()?;
    let high_bitdepth = b.bit()?;
    let twelve_bit = profile == 2 && high_bitdepth && b.bit()?;
    let bit_depth = if twelve_bit {
        12
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let monochrome = if profile == 1 { false } else { b.bit()? };
    let color_description = b.bit()?;
    let (cp, tc, mc) = if color_description {
        (b.read(8)? as u8, b.read(8)? as u8, b.read(8)? as u8)
    } else {
        (2, 2, 2)
    };
    let sampling;
    let color_range;
    let mut chroma_sample_position = 0;
    let mut separate_uv_delta_q = false;
    if monochrome {
        color_range = b.bit()?;
        sampling = ChromaSampling::Cs400;
    } else if cp == 1 && tc == 13 && mc == 0 {
        color_range = true;
        sampling = ChromaSampling::Cs444;
    } else {
        color_range = b.bit()?;
        sampling = match profile {
            0 => ChromaSampling::Cs420,
            1 => ChromaSampling::Cs444,
            _ if bit_depth == 12 => {
                let sub_x = b.bit()?;
                let sub_y = if sub_x { b.bit()? } else { false };
                match (sub_x, sub_y) {
                    (true, true) => ChromaSampling::Cs420,
                    (true, false) => ChromaSampling::Cs422,
                    _ => ChromaSampling::Cs444,
                }
            }
            _ => ChromaSampling::Cs422,
        };
        if sampling == ChromaSampling::Cs420 {
            chroma_sample_position = b.read(2)? as u8;
        }
        separate_uv_delta_q = b.bit()?;
    }
    let film_grain_params_present = b.bit()?;
    b.finish_trailing()?;
    Ok(Sequence {
        profile,
        still_picture,
        reduced_still_picture_header: reduced,
        max_width,
        max_height,
        bit_depth,
        monochrome,
        chroma_sampling: sampling,
        timing,
        decoder_model,
        initial_display_delay_present,
        operating_points,
        frame_width_bits: width_bits,
        frame_height_bits: height_bits,
        frame_id_numbers_present,
        delta_frame_id_length,
        frame_id_length,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color_primaries: cp,
        transfer_characteristics: tc,
        matrix_coefficients: mc,
        color_range,
        chroma_sample_position,
        separate_uv_delta_q,
        film_grain_params_present,
    })
}

fn parse_metadata(data: &[u8]) -> Result<Metadata, Error> {
    let (kind, prefix) = leb128(data)?;
    if kind == 5 {
        return parse_timecode_metadata(&data[prefix..]).map(Metadata::Timecode);
    }
    if kind == 3 {
        return parse_scalability_metadata(&data[prefix..]).map(Metadata::Scalability);
    }
    let body_with_trailing = &data[prefix..];
    let body = body_with_trailing
        .strip_suffix(&[0x80])
        .ok_or(Error::InvalidObu)?;
    let be16 = |at| -> Result<u16, Error> {
        Ok(u16::from_be_bytes(
            body.get(at..at + 2)
                .ok_or(Error::Truncated)?
                .try_into()
                .unwrap(),
        ))
    };
    let be32 = |at| -> Result<u32, Error> {
        Ok(u32::from_be_bytes(
            body.get(at..at + 4)
                .ok_or(Error::Truncated)?
                .try_into()
                .unwrap(),
        ))
    };
    match kind {
        1 if body.len() == 4 => Ok(Metadata::HdrContentLightLevel {
            max_cll: be16(0)?,
            max_fall: be16(2)?,
        }),
        2 if body.len() == 24 => Ok(Metadata::HdrMasteringDisplayColorVolume {
            primaries_x: [be16(0)?, be16(4)?, be16(8)?],
            primaries_y: [be16(2)?, be16(6)?, be16(10)?],
            white_point_x: be16(12)?,
            white_point_y: be16(14)?,
            luminance_max: be32(16)?,
            luminance_min: be32(20)?,
        }),
        1 | 2 => Err(Error::InvalidObu),
        4 if !body.is_empty() => {
            let country_code = body[0];
            let (country_code_extension, payload_start) = if country_code == 0xff {
                (Some(*body.get(1).ok_or(Error::Truncated)?), 2)
            } else {
                (None, 1)
            };
            Ok(Metadata::ItuTT35(ItuT35Metadata {
                country_code,
                country_code_extension,
                payload: vector_from_slice(&body[payload_start..])?,
            }))
        }
        4 => Err(Error::InvalidObu),
        _ => Ok(Metadata::Reserved {
            kind: kind as u64,
            payload: vector_from_slice(body)?,
        }),
    }
}

fn parse_scalability_metadata(data: &[u8]) -> Result<ScalabilityMetadata, Error> {
    const SCALABILITY_SS: u8 = 14;
    let mut bits = Bits::new(data);
    let mode_idc = bits.read(8)? as u8;
    let structure = if mode_idc == SCALABILITY_SS {
        let layer_count = bits.read(2)? as usize + 1;
        let dimensions_present = bits.bit()?;
        let descriptions_present = bits.bit()?;
        let temporal_groups_present = bits.bit()?;
        if bits.read(3)? != 0 {
            return Err(Error::InvalidObu);
        }
        let mut spatial_layers =
            Vector::with_capacity(layer_count).map_err(|_| Error::LimitExceeded)?;
        for _ in 0..layer_count {
            spatial_layers
                .try_push(ScalabilitySpatialLayer {
                    maximum_width: None,
                    maximum_height: None,
                    reference_id: None,
                })
                .map_err(|_| Error::LimitExceeded)?;
        }
        if dimensions_present {
            for layer in &mut spatial_layers {
                layer.maximum_width = Some(bits.read(16)? as u16);
                layer.maximum_height = Some(bits.read(16)? as u16);
            }
        }
        if descriptions_present {
            for layer in &mut spatial_layers {
                layer.reference_id = Some(bits.read(8)? as u8);
            }
        }
        let mut temporal_groups = Vector::new();
        if temporal_groups_present {
            let group_count = bits.read(8)? as usize;
            temporal_groups =
                Vector::with_capacity(group_count).map_err(|_| Error::LimitExceeded)?;
            for _ in 0..group_count {
                let temporal_id = bits.read(3)? as u8;
                let temporal_switching_up_point = bits.bit()?;
                let spatial_switching_up_point = bits.bit()?;
                let reference_count = bits.read(3)? as usize;
                let mut reference_picture_differences =
                    Vector::with_capacity(reference_count).map_err(|_| Error::LimitExceeded)?;
                for _ in 0..reference_count {
                    reference_picture_differences
                        .try_push(bits.read(8)? as u8)
                        .map_err(|_| Error::LimitExceeded)?;
                }
                temporal_groups
                    .try_push(ScalabilityTemporalGroup {
                        temporal_id,
                        temporal_switching_up_point,
                        spatial_switching_up_point,
                        reference_picture_differences,
                    })
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
        Some(ScalabilityStructure {
            spatial_layers,
            temporal_groups,
        })
    } else {
        None
    };
    bits.finish_trailing()?;
    Ok(ScalabilityMetadata {
        mode_idc,
        structure,
    })
}

fn parse_timecode_metadata(data: &[u8]) -> Result<TimecodeMetadata, Error> {
    let mut bits = Bits::new(data);
    let counting_type = bits.read(5)? as u8;
    let full_timestamp = bits.bit()?;
    let discontinuity = bits.bit()?;
    let count_dropped = bits.bit()?;
    let frames = bits.read(9)? as u16;
    let (seconds, minutes, hours) = if full_timestamp {
        (
            Some(bits.read(6)? as u8),
            Some(bits.read(6)? as u8),
            Some(bits.read(5)? as u8),
        )
    } else if bits.bit()? {
        let seconds = Some(bits.read(6)? as u8);
        if bits.bit()? {
            let minutes = Some(bits.read(6)? as u8);
            if bits.bit()? {
                (seconds, minutes, Some(bits.read(5)? as u8))
            } else {
                (seconds, minutes, None)
            }
        } else {
            (seconds, None, None)
        }
    } else {
        (None, None, None)
    };
    if seconds.is_some_and(|value| value > 59)
        || minutes.is_some_and(|value| value > 59)
        || hours.is_some_and(|value| value > 23)
    {
        return Err(Error::InvalidObu);
    }
    let time_offset_length = bits.read(5)? as u8;
    let time_offset = if time_offset_length == 0 {
        0
    } else {
        i32::try_from(bits.read_signed(time_offset_length)?).map_err(|_| Error::InvalidObu)?
    };
    bits.finish_trailing()?;
    Ok(TimecodeMetadata {
        counting_type,
        full_timestamp,
        discontinuity,
        count_dropped,
        frames,
        seconds,
        minutes,
        hours,
        time_offset_length,
        time_offset,
    })
}

pub(crate) struct Bits<'a> {
    data: &'a [u8],
    bit: usize,
}
impl<'a> Bits<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }
    pub(crate) fn bit(&mut self) -> Result<bool, Error> {
        Ok(self.read(1)? != 0)
    }
    pub(crate) fn read(&mut self, n: u8) -> Result<u64, Error> {
        if n > 64 || self.bit.checked_add(n as usize).ok_or(Error::Truncated)? > self.data.len() * 8
        {
            return Err(Error::Truncated);
        }
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | ((self.data[self.bit / 8] >> (7 - self.bit % 8)) & 1) as u64;
            self.bit += 1;
        }
        Ok(v)
    }
    pub(crate) fn read_signed(&mut self, n: u8) -> Result<i64, Error> {
        if n == 0 || n > 63 {
            return Err(Error::InvalidObu);
        }
        let value = self.read(n)?;
        let sign = 1u64 << (n - 1);
        if value & sign == 0 {
            Ok(value as i64)
        } else {
            Ok((value as i64) - (1i64 << n))
        }
    }
    pub(crate) fn position(&self) -> usize {
        self.bit
    }

    pub(crate) fn align_one(&mut self) -> Result<(), Error> {
        if !self.bit()? {
            return Err(Error::InvalidObu);
        }
        while !self.bit.is_multiple_of(8) {
            if self.bit()? {
                return Err(Error::InvalidObu);
            }
        }
        Ok(())
    }

    pub(crate) fn align_zero(&mut self) -> Result<(), Error> {
        while !self.bit.is_multiple_of(8) {
            if self.bit()? {
                return Err(Error::InvalidObu);
            }
        }
        Ok(())
    }

    pub(crate) fn finish_trailing(&mut self) -> Result<(), Error> {
        self.align_one()?;
        if self.bit != self.data.len() * 8 {
            return Err(Error::InvalidObu);
        }
        Ok(())
    }

    pub(crate) fn read_ns(&mut self, n: u32) -> Result<u32, Error> {
        if n <= 1 {
            return if n == 1 {
                Ok(0)
            } else {
                Err(Error::InvalidObu)
            };
        }
        let width = (32 - n.leading_zeros()) as u8;
        let split = (1u32 << width) - n;
        let value = self.read(width - 1)? as u32;
        if value < split {
            Ok(value)
        } else {
            Ok((value << 1) - split + self.read(1)? as u32)
        }
    }
}

fn read_uvlc(b: &mut Bits<'_>) -> Result<u32, Error> {
    let mut leading = 0u8;
    while !b.bit()? {
        leading = leading.checked_add(1).ok_or(Error::InvalidSequence)?;
        if leading >= 32 {
            return Ok(u32::MAX);
        }
    }
    Ok(((1u64 << leading) - 1 + b.read(leading)?) as u32)
}

fn leb128(data: &[u8]) -> Result<(usize, usize), Error> {
    let mut value = 0u64;
    for i in 0..8 {
        let byte = *data.get(i).ok_or(Error::Truncated)?;
        value |= u64::from(byte & 0x7f) << (i * 7);
        if byte & 0x80 == 0 {
            return Ok((
                usize::try_from(value).map_err(|_| Error::LimitExceeded)?,
                i + 1,
            ));
        }
    }
    Err(Error::InvalidObu)
}

fn write_leb128(mut value: usize, out: &mut Vector<u8>) -> Result<(), Error> {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        vector_push(out, byte)?;
        if value == 0 {
            return Ok(());
        }
    }
}

struct LengthReader<'a> {
    remaining: &'a [u8],
}
impl<'a> LengthReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { remaining: data }
    }
    fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
    fn item(&mut self, max: usize) -> Result<&'a [u8], Error> {
        let (size, prefix) = leb128(self.remaining)?;
        if size > max {
            return Err(Error::LimitExceeded);
        }
        let end = prefix.checked_add(size).ok_or(Error::LimitExceeded)?;
        let item = self.remaining.get(prefix..end).ok_or(Error::Truncated)?;
        self.remaining = &self.remaining[end..];
        Ok(item)
    }
}

/// Write Annex-B framing for temporal units containing frame units containing
/// complete OBUs. The input nesting mirrors the normative size hierarchy.
pub fn write_annex_b(temporal_units: &[Vector<Vector<Vector<u8>>>]) -> Result<Vector<u8>, Error> {
    let mut stream = Vector::new();
    for temporal in temporal_units {
        let mut temporal_bytes = Vector::new();
        for frame in temporal {
            let mut frame_bytes = Vector::new();
            for obu in frame {
                write_leb128(obu.len(), &mut frame_bytes)?;
                vector_extend(&mut frame_bytes, obu)?;
            }
            write_leb128(frame_bytes.len(), &mut temporal_bytes)?;
            vector_extend(&mut temporal_bytes, &frame_bytes)?;
        }
        write_leb128(temporal_bytes.len(), &mut stream)?;
        vector_extend(&mut stream, &temporal_bytes)?;
    }
    Ok(stream)
}

fn le16(data: &[u8], at: usize) -> Result<u16, Error> {
    Ok(u16::from_le_bytes(
        data.get(at..at + 2)
            .ok_or(Error::Truncated)?
            .try_into()
            .unwrap(),
    ))
}
fn le32(data: &[u8], at: usize) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(
        data.get(at..at + 4)
            .ok_or(Error::Truncated)?
            .try_into()
            .unwrap(),
    ))
}

fn bit_at(data: &[u8], position: usize) -> Result<bool, Error> {
    let byte = *data.get(position / 8).ok_or(Error::Truncated)?;
    Ok(byte & (1 << (7 - position % 8)) != 0)
}

fn align_zero_from(data: &[u8], position: usize) -> Result<usize, Error> {
    let total = data.len().checked_mul(8).ok_or(Error::LimitExceeded)?;
    if position > total {
        return Err(Error::Truncated);
    }
    let aligned = position.checked_add(7).ok_or(Error::LimitExceeded)? / 8;
    for bit in position..aligned * 8 {
        if bit_at(data, bit)? {
            return Err(Error::InvalidObu);
        }
    }
    Ok(aligned)
}

fn validate_trailing_from(data: &[u8], position: usize) -> Result<(), Error> {
    let total = data.len().checked_mul(8).ok_or(Error::LimitExceeded)?;
    if position >= total || !bit_at(data, position)? {
        return Err(Error::InvalidObu);
    }
    for bit in position + 1..total {
        if bit_at(data, bit)? {
            return Err(Error::InvalidObu);
        }
    }
    Ok(())
}

fn vector_with_capacity<T>(capacity: usize) -> Result<Vector<T>, Error> {
    Vector::with_capacity(capacity).map_err(|_| Error::LimitExceeded)
}

fn vector_push<T>(vector: &mut Vector<T>, value: T) -> Result<(), Error> {
    vector.try_push(value).map_err(|_| Error::LimitExceeded)
}

fn vector_extend<T: Clone>(vector: &mut Vector<T>, values: &[T]) -> Result<(), Error> {
    vector
        .try_extend_from_slice(values)
        .map_err(|_| Error::LimitExceeded)
}

fn vector_from_slice<T: Clone>(values: &[T]) -> Result<Vector<T>, Error> {
    let mut vector = vector_with_capacity(values.len())?;
    vector_extend(&mut vector, values)?;
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_syntax_is_suppressed_only_for_a_skipped_full_superblock() {
        assert!(!delta_syntax_present(
            true,
            true,
            partition::BlockSize::Block128x128,
            true,
        ));
        assert!(!delta_syntax_present(
            true,
            true,
            partition::BlockSize::Block64x64,
            false,
        ));
        assert!(delta_syntax_present(
            true,
            true,
            partition::BlockSize::Block64x64,
            true,
        ));
        assert!(delta_syntax_present(
            true,
            false,
            partition::BlockSize::Block128x128,
            true,
        ));
        assert!(!delta_syntax_present(
            false,
            false,
            partition::BlockSize::Block8x8,
            true,
        ));
    }

    #[test]
    fn camera_tile_copy_respects_chroma_subsampling() {
        let mut source =
            reconstruction::FrameBuffer::new(32, 32, 8, ChromaSampling::Cs420).unwrap();
        let mut destination =
            reconstruction::FrameBuffer::new(16, 16, 8, ChromaSampling::Cs420).unwrap();
        for y in 8..16 {
            for x in 8..16 {
                source.y.set_sample(x, y, 200).unwrap();
            }
        }
        for y in 4..8 {
            for x in 4..8 {
                source.u.as_mut().unwrap().set_sample(x, y, 90).unwrap();
                source.v.as_mut().unwrap().set_sample(x, y, 170).unwrap();
            }
        }
        copy_tile_region(
            &source,
            &mut destination,
            TileCopyRegion {
                source_x: 8,
                source_y: 8,
                destination_x: 0,
                destination_y: 0,
                width: 8,
                height: 8,
                sampling: ChromaSampling::Cs420,
            },
        )
        .unwrap();
        assert_eq!(destination.y.sample(7, 7), Ok(200));
        assert_eq!(destination.u.as_ref().unwrap().sample(3, 3), Ok(90));
        assert_eq!(destination.v.as_ref().unwrap().sample(3, 3), Ok(170));
        assert_eq!(destination.y.sample(8, 8), Ok(128));
    }

    #[test]
    fn frame_header_padding_modes_are_distinct() {
        assert_eq!(align_zero_from(&[0b1010_0000], 3), Ok(1));
        assert_eq!(align_zero_from(&[0b1011_0000], 3), Err(Error::InvalidObu));
        assert_eq!(validate_trailing_from(&[0b1010_0000], 2), Ok(()));
        assert_eq!(
            validate_trailing_from(&[0b1010_0001], 2),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn parses_leb128_and_rejects_overlong() {
        assert_eq!(leb128(&[0xe5, 0x8e, 0x26]), Ok((624485, 3)));
        assert_eq!(leb128(&[0x80; 8]), Err(Error::InvalidObu));
    }

    #[test]
    fn strict_obu_boundaries() {
        let (obu, used) = parse_obu(&[0x2a, 3, 1, 2, 3], 9).unwrap();
        assert_eq!(obu.kind, OBU_METADATA);
        assert_eq!(obu.payload, &[1, 2, 3]);
        assert_eq!(used, 5);
        assert_eq!(parse_obu(&[0x2a, 4, 1], 9).err(), Some(Error::Truncated));
        assert_eq!(parse_obu(&[0xaa, 0], 9).err(), Some(Error::InvalidObu));
    }

    #[test]
    fn accepts_empty_ivf() {
        let mut ivf = [0u8; 32];
        ivf[..4].copy_from_slice(b"DKIF");
        ivf[6..8].copy_from_slice(&32u16.to_le_bytes());
        ivf[8..12].copy_from_slice(b"AV01");
        assert_eq!(decode_ivf(&ivf), Ok(Vector::new()));
    }

    #[test]
    fn rejects_mismatched_ivf_packet_count() {
        let mut ivf = [0u8; 32];
        ivf[..4].copy_from_slice(b"DKIF");
        ivf[6..8].copy_from_slice(&32u16.to_le_bytes());
        ivf[8..12].copy_from_slice(b"AV01");
        ivf[24..28].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(decode_ivf(&ivf), Err(Error::InvalidIvf));
    }

    #[test]
    fn obu_writer_round_trips_header_and_payload() {
        let encoded = write_obu(ObuType::Metadata, 3, 2, &[9; 130]).unwrap();
        let (obu, used) = parse_obu(&encoded, 1024).unwrap();
        assert_eq!(used, encoded.len());
        assert_eq!(obu.kind, OBU_METADATA);
        assert_eq!(obu.payload, &[9; 130]);
        assert_eq!(&encoded[1..2], &[0b0111_0000]);
    }

    #[test]
    fn ivf_writer_produces_strict_container() {
        let packet = write_obu(ObuType::TemporalDelimiter, 0, 0, &[]).unwrap();
        let ivf = write_ivf(&[packet], 64, 48, 30, 1).unwrap();
        assert_eq!(decode_ivf(&ivf), Ok(Vector::new()));
    }

    #[test]
    fn annex_b_accepts_obus_without_internal_size() {
        let unsized_delimiter = [(OBU_TEMPORAL_DELIMITER << 3)].into_iter().collect();
        let frame: Vector<Vector<u8>> = [unsized_delimiter].into_iter().collect();
        let temporal: Vector<Vector<Vector<u8>>> = [frame].into_iter().collect();
        let stream = write_annex_b(&[temporal]).unwrap();
        assert_eq!(Decoder::new().decode_annex_b(&stream), Ok(Vector::new()));
    }

    #[test]
    fn low_overhead_format_rejects_obus_without_internal_size() {
        let mut decoder = Decoder::new();
        assert_eq!(
            decoder.decode_obus(&[(OBU_TEMPORAL_DELIMITER << 3), 0x80]),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn temporal_delimiter_is_empty_and_reserved_obus_are_skipped() {
        let mut decoder = Decoder::new();
        let reserved = [(9 << 3) | 2, 3, 0, 7, 0];
        assert!(decoder.decode_obus(&reserved).unwrap().is_empty());
        assert_eq!(
            decoder.decode_obus(&[(9 << 3) | 2, 2, 0, 0]),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            decoder.decode_obus(&[(OBU_TEMPORAL_DELIMITER << 3) | 2, 1, 0x80]),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn operating_point_filter_requires_both_temporal_and_spatial_membership() {
        let idc = (1 << 2) | (1 << (1 + 8));
        assert!(obu_in_operating_point(idc, OBU_FRAME, true, 2, 1));
        assert!(!obu_in_operating_point(idc, OBU_FRAME, true, 1, 1));
        assert!(!obu_in_operating_point(idc, OBU_FRAME, true, 2, 0));
        assert!(obu_in_operating_point(idc, OBU_FRAME, false, 7, 3));
        assert!(obu_in_operating_point(
            idc,
            OBU_TEMPORAL_DELIMITER,
            true,
            7,
            3
        ));
        assert!(obu_in_operating_point(0, OBU_FRAME, true, 7, 3));
    }

    #[test]
    fn operating_point_controls_required_obu_extension_flag() {
        assert_eq!(
            validate_operating_point_extension(0, OBU_FRAME, false),
            Ok(())
        );
        assert_eq!(
            validate_operating_point_extension(0, OBU_FRAME, true),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            validate_operating_point_extension(0x101, OBU_FRAME, true),
            Ok(())
        );
        assert_eq!(
            validate_operating_point_extension(0x101, OBU_FRAME, false),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            validate_operating_point_extension(0x101, OBU_SEQUENCE_HEADER, false),
            Ok(())
        );
    }

    #[test]
    fn frame_header_copy_requires_seen_identical_header() {
        assert_eq!(validate_frame_header_copy(true, &[1, 2], &[1, 2]), Ok(()));
        assert_eq!(
            validate_frame_header_copy(true, &[1, 2], &[1, 3]),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            validate_frame_header_copy(false, &[1, 2], &[1, 2]),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn annex_b_rejects_length_crossing_parent_boundary() {
        assert_eq!(
            Decoder::new().decode_annex_b(&[2, 3, 0]),
            Err(Error::Truncated)
        );
    }

    #[test]
    fn parses_hdr_content_light_metadata() {
        let obu = write_obu(ObuType::Metadata, 0, 0, &[1, 0x03, 0xe8, 0x01, 0x90, 0x80]).unwrap();
        let mut decoder = Decoder::new();
        decoder.decode_obus(&obu).unwrap();
        assert_eq!(
            decoder.metadata(),
            &[Metadata::HdrContentLightLevel {
                max_cll: 1000,
                max_fall: 400
            }]
        );
    }

    #[test]
    fn parses_non_byte_aligned_timecode_metadata() {
        // Zero-valued optional timestamp: 8 flag bits, 9 frame bits, one
        // seconds-present bit, five offset-length bits, then trailing_one_bit.
        assert_eq!(
            parse_metadata(&[5, 0, 0, 1]),
            Ok(Metadata::Timecode(TimecodeMetadata {
                counting_type: 0,
                full_timestamp: false,
                discontinuity: false,
                count_dropped: false,
                frames: 0,
                seconds: None,
                minutes: None,
                hours: None,
                time_offset_length: 0,
                time_offset: 0,
            }))
        );
        assert_eq!(parse_metadata(&[5, 0, 0, 0]), Err(Error::InvalidObu));
    }

    #[test]
    fn decoder_infers_omitted_timecode_components_from_previous_metadata() {
        let first = TimecodeMetadata {
            counting_type: 0,
            full_timestamp: true,
            discontinuity: false,
            count_dropped: false,
            frames: 1,
            seconds: Some(2),
            minutes: Some(3),
            hours: Some(4),
            time_offset_length: 0,
            time_offset: 0,
        };
        let second = TimecodeMetadata {
            full_timestamp: false,
            frames: 2,
            seconds: None,
            minutes: None,
            hours: None,
            ..first
        };
        let mut stream = encoder::timecode(first).unwrap();
        vector_extend(&mut stream, &encoder::timecode(second).unwrap()).unwrap();
        let mut decoder = Decoder::new();
        decoder.decode_obus(&stream).unwrap();
        let Metadata::Timecode(resolved) = decoder.metadata()[1] else {
            panic!("wrong metadata type")
        };
        assert_eq!(
            (resolved.seconds, resolved.minutes, resolved.hours),
            (Some(2), Some(3), Some(4))
        );

        let mut decoder = Decoder::new();
        assert_eq!(
            decoder.decode_obus(&encoder::timecode(second).unwrap()),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn parses_scalability_modes_and_structured_scalability() {
        assert_eq!(
            parse_metadata(&[3, 0, 0x80]),
            Ok(Metadata::Scalability(ScalabilityMetadata {
                mode_idc: 0,
                structure: None,
            }))
        );
        let parsed = parse_metadata(&[3, 14, 0, 0x80]).unwrap();
        let Metadata::Scalability(metadata) = parsed else {
            panic!("wrong metadata type")
        };
        let structure = metadata.structure.unwrap();
        assert_eq!(structure.spatial_layers.len(), 1);
        assert_eq!(structure.temporal_groups.len(), 0);
        assert_eq!(parse_metadata(&[3, 14, 1, 0x80]), Err(Error::InvalidObu));
    }

    #[test]
    fn parses_itu_t35_country_extension_separately_from_payload() {
        assert_eq!(
            parse_metadata(&[4, 0xff, 0x42, 9, 8, 0x80]),
            Ok(Metadata::ItuTT35(ItuT35Metadata {
                country_code: 0xff,
                country_code_extension: Some(0x42),
                payload: vector_from_slice(&[9, 8]).unwrap(),
            }))
        );
        assert_eq!(parse_metadata(&[4, 0xff, 0x80]), Err(Error::Truncated));
    }
}
