//! Frame-level quantization, segmentation, delta, loop-filter and CDEF syntax.

use crate::{Bits, Error, Sequence};

pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Quantization {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

impl Quantization {
    pub(crate) fn parse(bits: &mut Bits<'_>, sequence: &Sequence) -> Result<Self, Error> {
        let base_q_idx = bits.read(8)? as u8;
        let delta_q_y_dc = read_delta_q(bits)?;
        let (delta_q_u_dc, delta_q_u_ac, delta_q_v_dc, delta_q_v_ac) = if sequence.monochrome {
            (0, 0, 0, 0)
        } else {
            let different_uv = sequence.separate_uv_delta_q && bits.bit()?;
            let u_dc = read_delta_q(bits)?;
            let u_ac = read_delta_q(bits)?;
            if different_uv {
                (u_dc, u_ac, read_delta_q(bits)?, read_delta_q(bits)?)
            } else {
                (u_dc, u_ac, u_dc, u_ac)
            }
        };
        let using_qmatrix = bits.bit()?;
        let (qm_y, qm_u, qm_v) = if using_qmatrix {
            let y = bits.read(4)? as u8;
            let u = bits.read(4)? as u8;
            let v = if sequence.separate_uv_delta_q {
                bits.read(4)? as u8
            } else {
                u
            };
            (y, u, v)
        } else {
            (15, 15, 15)
        };
        Ok(Self {
            base_q_idx,
            delta_q_y_dc,
            delta_q_u_dc,
            delta_q_u_ac,
            delta_q_v_dc,
            delta_q_v_ac,
            using_qmatrix,
            qm_y,
            qm_u,
            qm_v,
        })
    }
}

fn read_delta_q(bits: &mut Bits<'_>) -> Result<i8, Error> {
    if bits.bit()? {
        Ok(bits.read_signed(7)? as i8)
    } else {
        Ok(0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub abs_or_delta_update: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }
}

impl Segmentation {
    pub fn last_active_segment_id(&self) -> u8 {
        if !self.enabled {
            return 0;
        }
        for segment in (0..MAX_SEGMENTS).rev() {
            if self.feature_enabled[segment].iter().any(|&enabled| enabled) {
                return segment as u8;
            }
        }
        0
    }

    pub fn segment_id_pre_skip(&self) -> bool {
        self.enabled
            && self
                .feature_enabled
                .iter()
                .any(|features| features[5..].iter().any(|&enabled| enabled))
    }

    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        primary_ref_none: bool,
        previous: Option<Self>,
    ) -> Result<Self, Error> {
        if !bits.bit()? {
            return Ok(Self::default());
        }
        let (update_map, temporal_update, update_data) = if primary_ref_none {
            (true, false, true)
        } else {
            let update_map = bits.bit()?;
            let temporal = update_map && bits.bit()?;
            (update_map, temporal, bits.bit()?)
        };
        let mut result = previous.unwrap_or_default();
        result.enabled = true;
        result.update_map = update_map;
        result.temporal_update = temporal_update;
        result.update_data = update_data;
        if update_data {
            result.abs_or_delta_update = bits.bit()?;
            result.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
            result.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
            for segment in 0..MAX_SEGMENTS {
                for feature in 0..SEG_LVL_MAX {
                    let enabled = bits.bit()?;
                    result.feature_enabled[segment][feature] = enabled;
                    if enabled {
                        let width = FEATURE_BITS[feature];
                        let magnitude = if width == 0 {
                            0
                        } else {
                            bits.read(width)? as i16
                        };
                        let sign = FEATURE_SIGNED[feature] && bits.bit()?;
                        let signed = if sign { -magnitude } else { magnitude };
                        result.feature_data[segment][feature] =
                            signed.clamp(-FEATURE_MAX[feature], FEATURE_MAX[feature]);
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn qindex(&self, segment: usize, base: u8) -> Result<u8, Error> {
        if segment >= MAX_SEGMENTS {
            return Err(Error::InvalidObu);
        }
        if !self.enabled || !self.feature_enabled[segment][0] {
            return Ok(base);
        }
        let feature = self.feature_data[segment][0];
        let value = if self.abs_or_delta_update {
            feature
        } else {
            i16::from(base) + feature
        };
        Ok(value.clamp(0, 255) as u8)
    }
}

const FEATURE_BITS: [u8; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
const FEATURE_MAX: [i16; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeltaParams {
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
}

impl DeltaParams {
    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        base_q_idx: u8,
        allow_intrabc: bool,
    ) -> Result<Self, Error> {
        let delta_q_present = base_q_idx > 0 && bits.bit()?;
        let delta_q_res = if delta_q_present {
            bits.read(2)? as u8
        } else {
            0
        };
        let delta_lf_present = delta_q_present && !allow_intrabc && bits.bit()?;
        let (delta_lf_res, delta_lf_multi) = if delta_lf_present {
            (bits.read(2)? as u8, bits.bit()?)
        } else {
            (0, false)
        };
        Ok(Self {
            delta_q_present,
            delta_q_res,
            delta_lf_present,
            delta_lf_res,
            delta_lf_multi,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopFilter {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> Self {
        Self {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0; 2],
        }
    }
}

impl LoopFilter {
    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        sequence: &Sequence,
        coded_lossless: bool,
        allow_intrabc: bool,
        previous: Option<Self>,
    ) -> Result<Self, Error> {
        let mut result = previous.unwrap_or_default();
        if coded_lossless || allow_intrabc {
            result.level = [0; 4];
            return Ok(result);
        }
        result.level[0] = bits.read(6)? as u8;
        result.level[1] = bits.read(6)? as u8;
        if !sequence.monochrome && (result.level[0] != 0 || result.level[1] != 0) {
            result.level[2] = bits.read(6)? as u8;
            result.level[3] = bits.read(6)? as u8;
        } else {
            result.level[2] = 0;
            result.level[3] = 0;
        }
        result.sharpness = bits.read(3)? as u8;
        result.delta_enabled = bits.bit()?;
        if result.delta_enabled && bits.bit()? {
            for value in &mut result.ref_deltas {
                if bits.bit()? {
                    *value = bits.read_signed(7)? as i8;
                }
            }
            for value in &mut result.mode_deltas {
                if bits.bit()? {
                    *value = bits.read_signed(7)? as i8;
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cdef {
    pub damping: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

impl Default for Cdef {
    fn default() -> Self {
        Self {
            damping: 3,
            bits: 0,
            y_pri_strength: [0; 8],
            y_sec_strength: [0; 8],
            uv_pri_strength: [0; 8],
            uv_sec_strength: [0; 8],
        }
    }
}

impl Cdef {
    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        sequence: &Sequence,
        coded_lossless: bool,
        allow_intrabc: bool,
    ) -> Result<Self, Error> {
        if coded_lossless || allow_intrabc || !sequence.enable_cdef {
            return Ok(Self::default());
        }
        let mut result = Self {
            damping: bits.read(2)? as u8 + 3,
            bits: bits.read(2)? as u8,
            ..Self::default()
        };
        for index in 0..1usize << result.bits {
            result.y_pri_strength[index] = bits.read(4)? as u8;
            result.y_sec_strength[index] = remap_secondary(bits.read(2)? as u8);
            if !sequence.monochrome {
                result.uv_pri_strength[index] = bits.read(4)? as u8;
                result.uv_sec_strength[index] = remap_secondary(bits.read(2)? as u8);
            }
        }
        Ok(result)
    }
}

fn remap_secondary(value: u8) -> u8 {
    if value == 3 { 4 } else { value }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RestorationType {
    #[default]
    None,
    Switchable,
    Wiener,
    Sgrproj,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Restoration {
    pub frame_type: [RestorationType; 3],
    pub unit_shift: u8,
    pub uv_shift: bool,
    pub unit_size: [u16; 3],
}

impl Restoration {
    pub(crate) fn parse(
        bits: &mut Bits<'_>,
        sequence: &Sequence,
        all_lossless: bool,
        allow_intrabc: bool,
    ) -> Result<Self, Error> {
        if all_lossless || allow_intrabc || !sequence.enable_restoration {
            return Ok(Self::default());
        }
        let planes = if sequence.monochrome { 1 } else { 3 };
        let mut result = Self::default();
        let mut uses_restoration = false;
        let mut uses_chroma = false;
        for plane in 0..planes {
            result.frame_type[plane] = match bits.read(2)? {
                0 => RestorationType::None,
                1 => RestorationType::Switchable,
                2 => RestorationType::Wiener,
                3 => RestorationType::Sgrproj,
                _ => unreachable!(),
            };
            if result.frame_type[plane] != RestorationType::None {
                uses_restoration = true;
                uses_chroma |= plane > 0;
            }
        }
        if uses_restoration {
            result.unit_shift = u8::from(sequence.use_128x128_superblock);
            if bits.bit()? {
                result.unit_shift += 1;
                if result.unit_shift == 1 && bits.bit()? {
                    result.unit_shift += 1;
                }
            }
            result.uv_shift = sequence.chroma_sampling == crate::ChromaSampling::Cs420
                && uses_chroma
                && bits.bit()?;
            result.unit_size[0] = 256 >> result.unit_shift;
            result.unit_size[1] = result.unit_size[0] >> u8::from(result.uv_shift);
            result.unit_size[2] = result.unit_size[1];
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TxMode {
    #[default]
    Only4x4,
    Largest,
    Select,
}

impl TxMode {
    pub(crate) fn parse(bits: &mut Bits<'_>, coded_lossless: bool) -> Result<Self, Error> {
        if coded_lossless {
            Ok(Self::Only4x4)
        } else if bits.bit()? {
            Ok(Self::Select)
        } else {
            Ok(Self::Largest)
        }
    }
}

pub fn derive_lossless(
    quantization: &Quantization,
    segmentation: &Segmentation,
    frame_width: u32,
    upscaled_width: u32,
) -> Result<([bool; MAX_SEGMENTS], bool, bool), Error> {
    let mut segments = [false; MAX_SEGMENTS];
    let mut coded_lossless = true;
    for (segment, lossless) in segments.iter_mut().enumerate() {
        *lossless = segmentation.qindex(segment, quantization.base_q_idx)? == 0
            && quantization.delta_q_y_dc == 0
            && quantization.delta_q_u_dc == 0
            && quantization.delta_q_u_ac == 0
            && quantization.delta_q_v_dc == 0
            && quantization.delta_q_v_ac == 0;
        coded_lossless &= *lossless;
    }
    Ok((
        segments,
        coded_lossless,
        coded_lossless && frame_width == upscaled_width,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_qindex_clamps() {
        let mut segmentation = Segmentation {
            enabled: true,
            ..Segmentation::default()
        };
        segmentation.feature_enabled[0][0] = true;
        segmentation.feature_data[0][0] = -200;
        assert_eq!(segmentation.qindex(0, 100), Ok(0));
        segmentation.abs_or_delta_update = true;
        segmentation.feature_data[0][0] = 300;
        assert_eq!(segmentation.qindex(0, 0), Ok(255));
    }

    #[test]
    fn segmentation_derives_active_range_and_pre_skip_ordering() {
        let mut segmentation = Segmentation {
            enabled: true,
            ..Segmentation::default()
        };
        assert_eq!(segmentation.last_active_segment_id(), 0);
        assert!(!segmentation.segment_id_pre_skip());
        segmentation.feature_enabled[5][0] = true;
        assert_eq!(segmentation.last_active_segment_id(), 5);
        assert!(!segmentation.segment_id_pre_skip());
        segmentation.feature_enabled[3][6] = true;
        assert!(segmentation.segment_id_pre_skip());
    }

    #[test]
    fn secondary_strength_three_maps_to_four() {
        assert_eq!(remap_secondary(3), 4);
    }
}
