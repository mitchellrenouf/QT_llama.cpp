//! Film-grain parameter syntax (section 5.9.30).

use crate::{
    Bits, ChromaSampling, Error, Sequence,
    reconstruction::{FrameBuffer, Plane},
};
use mrml_runtime::Vector;

pub const MAX_Y_POINTS: usize = 14;
pub const MAX_UV_POINTS: usize = 10;
pub const MAX_AR_COEFFS_Y: usize = 24;
pub const MAX_AR_COEFFS_UV: usize = 25;

// Normative AV1 Gaussian sequence (specification table 6.17). Keeping the
// fixed distribution local makes synthesis deterministic without an external
// codec or runtime data file.
const GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388,
    -732, -104, -188, 800, 112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500,
    -228, 12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192, 1000, -332, 652, -136,
    -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124, -24, 396, 540, -12, -104, 640,
    464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40, -156, 252,
    -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60,
    -264, -96, -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792,
    304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548,
    488, -344, -136, 488, -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168, -108,
    464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116, 436, 504, -232, 328,
    844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4,
    196, 44, -36, -648, 932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448,
    208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540, 28, 136, -88, -436, 256, 296,
    -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768,
    388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648, -136, 372,
    -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112,
    -220, -100, 996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412,
    -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240,
    -208, -40, 100, -592, 344, -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752, -100, 560, -1020, 180,
    -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164,
    -376, -684, 592, -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864,
    424, -728, 988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776,
    -824, -884, 524, -548, -564, -68, -164, -96, 692, 364, -692, -1012, -68, 260, -480, 876, -1116,
    452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112, -440,
    -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276,
    -508, 940, 24, -400, -120, 756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892, -888, 60, 164, 68, -760, 216,
    -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776,
    1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272,
    460, -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560,
    1040, -472, 116, -848, -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972,
    -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828, 524, 212, -212, 52, 12, 200, 268,
    -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436, 316, 372,
    -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4,
    -1168, 368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272,
    -256, -420, 320, 208, -144, -156, 156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360,
    32, -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840, 784, -460, 656, -544,
    -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308, 308, -112, -636, -760,
    280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636, -76,
    392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24,
    -1044, -32, 40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424,
    -1008, -324, 588, -1496, 576, 460, -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52,
    -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164, -304, -540,
    120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148, 716, -604, 404,
    -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508,
    132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384,
    -188, -612, -576, 736, 508, 892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508,
    -144, -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472, -204, 908, -188, -1000,
    -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760, -176, 376, -524, -452,
    -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832,
    148, 248, 652, 616, 1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64,
    156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524, -268, -720, 540,
    -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468,
    -28, -120, -32, -44, 1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156,
    36, 492, -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808, -416, 592, 316, -480,
    56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324, -416, -600, 768, 268, -248, -88, -132,
    -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480, -216, -92, 1688,
    -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332, 204,
    760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728,
    -520, 276, -188, 80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272,
    -404, 180, 816, -796, 48, 152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316, 476,
    12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576, 260, -352, 120, -16, -136, -260,
    -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84,
    192, 852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848,
    -344, 224, 208, -200, 484, 128, -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112,
    -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84, 200, -152, 0, -144, 584,
    280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300,
    24, -676, -596, 132, 1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488,
    424, 24, 756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424,
    888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628, -348, -228, -380, 1012, -452,
    -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344, 428, 548,
    -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284,
    -12, 824, 164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648,
    -620, -684, -24, -376, -384, -108, -920, -1032, 768, 180, -264, -508, -1268, -260, -60, 300,
    -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4, -216, -32,
    836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228,
    132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376,
    -108, -92, 788, 368, -56, 544, 400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200, -56, 276, -88, 484, -324,
    948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116,
    260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676,
    -396, -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212,
    252, -660, -708, 484, -444, -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196, 76, 388, 232, 412, -1168,
    -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168, -608, 448, 160, -532, -272,
    28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524, -572, -556,
    -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488,
    -192, -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12,
    -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716,
    592, -80, 436, 360, 4, -248, 160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300, -212, -1144, 32, -724,
    800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552,
    -52, -100, -656, 208, -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272,
    -876, -664, -648, -332, -136, 16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92, 356, 404, -408, 252, 196, -176,
    -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508,
    -476, 324, -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348,
    -288, -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12,
    168, 1020, 512, -552, 924, -148, 716, 188, 164, -340, -520, -184, 880, -152, -680, -208, -1156,
    -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280, -224, 96,
    568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428, -484,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilmGrain {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub reference_index: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; MAX_Y_POINTS],
    pub point_y_scaling: [u8; MAX_Y_POINTS],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; MAX_UV_POINTS],
    pub point_cb_scaling: [u8; MAX_UV_POINTS],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; MAX_UV_POINTS],
    pub point_cr_scaling: [u8; MAX_UV_POINTS],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; MAX_AR_COEFFS_Y],
    pub ar_coeffs_cb_plus_128: [u8; MAX_AR_COEFFS_UV],
    pub ar_coeffs_cr_plus_128: [u8; MAX_AR_COEFFS_UV],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap: bool,
    pub clip_to_restricted_range: bool,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            reference_index: 0,
            num_y_points: 0,
            point_y_value: [0; MAX_Y_POINTS],
            point_y_scaling: [0; MAX_Y_POINTS],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            point_cb_value: [0; MAX_UV_POINTS],
            point_cb_scaling: [0; MAX_UV_POINTS],
            num_cr_points: 0,
            point_cr_value: [0; MAX_UV_POINTS],
            point_cr_scaling: [0; MAX_UV_POINTS],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y_plus_128: [0; MAX_AR_COEFFS_Y],
            ar_coeffs_cb_plus_128: [0; MAX_AR_COEFFS_UV],
            ar_coeffs_cr_plus_128: [0; MAX_AR_COEFFS_UV],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap: false,
            clip_to_restricted_range: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrainRandom {
    register: u16,
}

impl GrainRandom {
    pub const fn new(seed: u16) -> Self {
        Self { register: seed }
    }

    pub fn reset(&mut self, seed: u16) {
        self.register = seed;
    }

    pub fn number(&mut self, bits: u8) -> Result<u16, Error> {
        if bits > 16 {
            return Err(Error::InvalidObu);
        }
        let r = self.register;
        let feedback = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.register = (r >> 1) | (feedback << 15);
        Ok(if bits == 0 {
            0
        } else {
            self.register >> (16 - bits)
        })
    }

    pub fn gaussian(&mut self) -> i16 {
        let index = usize::from(self.number(11).expect("11 is a valid bit count"));
        GAUSSIAN_SEQUENCE[index]
    }
}

/// Construct one normative 256-entry film-grain scaling lookup table.
pub fn scaling_lookup(values: &[u8], scales: &[u8], count: u8) -> Result<[u8; 256], Error> {
    let count = usize::from(count);
    if count > values.len() || count > scales.len() {
        return Err(Error::InvalidObu);
    }
    let mut lookup = [0u8; 256];
    if count == 0 {
        return Ok(lookup);
    }
    for index in 1..count {
        if values[index] <= values[index - 1] {
            return Err(Error::InvalidObu);
        }
    }
    lookup[..usize::from(values[0])].fill(scales[0]);
    for index in 0..count - 1 {
        let start = usize::from(values[index]);
        let delta_x = i32::from(values[index + 1] - values[index]);
        let delta_y = i32::from(scales[index + 1]) - i32::from(scales[index]);
        let delta = delta_y * ((65_536 + (delta_x >> 1)) / delta_x);
        for x in 0..usize::try_from(delta_x).map_err(|_| Error::InvalidObu)? {
            let interpolated = i32::from(scales[index])
                + ((i32::try_from(x).map_err(|_| Error::LimitExceeded)? * delta + 32_768) >> 16);
            lookup[start + x] = u8::try_from(interpolated).map_err(|_| Error::InvalidObu)?;
        }
    }
    lookup[usize::from(values[count - 1])..].fill(scales[count - 1]);
    Ok(lookup)
}

const LUMA_GRAIN_WIDTH: usize = 82;
const LUMA_GRAIN_HEIGHT: usize = 73;
const MAX_GRAIN_SAMPLES: usize = LUMA_GRAIN_WIDTH * LUMA_GRAIN_HEIGHT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrainPatterns {
    luma: [i16; MAX_GRAIN_SAMPLES],
    cb: [i16; MAX_GRAIN_SAMPLES],
    cr: [i16; MAX_GRAIN_SAMPLES],
    chroma_width: u8,
    chroma_height: u8,
}

impl GrainPatterns {
    pub fn luma(&self, x: usize, y: usize) -> Option<i16> {
        (x < LUMA_GRAIN_WIDTH && y < LUMA_GRAIN_HEIGHT).then(|| self.luma[y * LUMA_GRAIN_WIDTH + x])
    }

    pub fn chroma(&self, plane: u8, x: usize, y: usize) -> Option<i16> {
        let width = usize::from(self.chroma_width);
        if x >= width || y >= usize::from(self.chroma_height) {
            return None;
        }
        match plane {
            1 => Some(self.cb[y * width + x]),
            2 => Some(self.cr[y * width + x]),
            _ => None,
        }
    }
}

pub fn generate_grain(
    grain: &FilmGrain,
    bit_depth: u8,
    monochrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
) -> Result<GrainPatterns, Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidSequence);
    }
    let chroma_width = if subsampling_x { 44 } else { 82 };
    let chroma_height = if subsampling_y { 38 } else { 73 };
    let mut result = GrainPatterns {
        luma: [0; MAX_GRAIN_SAMPLES],
        cb: [0; MAX_GRAIN_SAMPLES],
        cr: [0; MAX_GRAIN_SAMPLES],
        chroma_width,
        chroma_height,
    };
    let white_shift = 12 - bit_depth + grain.grain_scale_shift;
    let grain_center = 128i32 << (bit_depth - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bit_depth - 8)) - 1 - grain_center;

    let mut random = GrainRandom::new(grain.grain_seed);
    if grain.num_y_points > 0 {
        for sample in &mut result.luma {
            *sample = i16::try_from(round2(i32::from(random.gaussian()), white_shift))
                .map_err(|_| Error::LimitExceeded)?;
        }
    }
    autoregressive_luma(&mut result.luma, grain, grain_min, grain_max)?;
    if !monochrome {
        let chroma_samples = usize::from(chroma_width) * usize::from(chroma_height);
        if grain.num_cb_points > 0 || grain.chroma_scaling_from_luma {
            random.reset(grain.grain_seed ^ 0xb524);
            for sample in &mut result.cb[..chroma_samples] {
                *sample = i16::try_from(round2(i32::from(random.gaussian()), white_shift))
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
        if grain.num_cr_points > 0 || grain.chroma_scaling_from_luma {
            random.reset(grain.grain_seed ^ 0x49d8);
            for sample in &mut result.cr[..chroma_samples] {
                *sample = i16::try_from(round2(i32::from(random.gaussian()), white_shift))
                    .map_err(|_| Error::LimitExceeded)?;
            }
        }
        autoregressive_chroma(
            &mut result,
            grain,
            grain_min,
            grain_max,
            subsampling_x,
            subsampling_y,
        )?;
    }
    Ok(result)
}

fn autoregressive_luma(
    samples: &mut [i16; MAX_GRAIN_SAMPLES],
    grain: &FilmGrain,
    minimum: i32,
    maximum: i32,
) -> Result<(), Error> {
    let lag = i32::from(grain.ar_coeff_lag);
    let shift = grain.ar_coeff_shift_minus_6 + 6;
    for y in 3..LUMA_GRAIN_HEIGHT {
        for x in 3..LUMA_GRAIN_WIDTH - 3 {
            let mut sum = 0i32;
            let mut position = 0usize;
            'rows: for delta_row in -lag..=0 {
                for delta_column in -lag..=lag {
                    if delta_row == 0 && delta_column == 0 {
                        break 'rows;
                    }
                    let source_y = usize::try_from(
                        i32::try_from(y).map_err(|_| Error::LimitExceeded)? + delta_row,
                    )
                    .map_err(|_| Error::InvalidObu)?;
                    let source_x = usize::try_from(
                        i32::try_from(x).map_err(|_| Error::LimitExceeded)? + delta_column,
                    )
                    .map_err(|_| Error::InvalidObu)?;
                    let coefficient = i32::from(grain.ar_coeffs_y_plus_128[position]) - 128;
                    sum += i32::from(samples[source_y * LUMA_GRAIN_WIDTH + source_x]) * coefficient;
                    position += 1;
                }
            }
            let index = y * LUMA_GRAIN_WIDTH + x;
            let filtered = i32::from(samples[index]) + round2(sum, shift);
            samples[index] = i16::try_from(filtered.clamp(minimum, maximum))
                .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(())
}

fn autoregressive_chroma(
    patterns: &mut GrainPatterns,
    grain: &FilmGrain,
    minimum: i32,
    maximum: i32,
    subsampling_x: bool,
    subsampling_y: bool,
) -> Result<(), Error> {
    let width = usize::from(patterns.chroma_width);
    let height = usize::from(patterns.chroma_height);
    let lag = i32::from(grain.ar_coeff_lag);
    let shift = grain.ar_coeff_shift_minus_6 + 6;
    for y in 3..height {
        for x in 3..width - 3 {
            let mut sums = [0i32; 2];
            let mut position = 0usize;
            'rows: for delta_row in -lag..=0 {
                for delta_column in -lag..=lag {
                    let cb_coefficient = i32::from(grain.ar_coeffs_cb_plus_128[position]) - 128;
                    let cr_coefficient = i32::from(grain.ar_coeffs_cr_plus_128[position]) - 128;
                    if delta_row == 0 && delta_column == 0 {
                        if grain.num_y_points > 0 {
                            let luma_x = ((x - 3) << usize::from(subsampling_x)) + 3;
                            let luma_y = ((y - 3) << usize::from(subsampling_y)) + 3;
                            let mut luma = 0i32;
                            for row in 0..=usize::from(subsampling_y) {
                                for column in 0..=usize::from(subsampling_x) {
                                    luma += i32::from(
                                        patterns.luma
                                            [(luma_y + row) * LUMA_GRAIN_WIDTH + luma_x + column],
                                    );
                                }
                            }
                            luma = round2(luma, u8::from(subsampling_x) + u8::from(subsampling_y));
                            sums[0] += luma * cb_coefficient;
                            sums[1] += luma * cr_coefficient;
                        }
                        break 'rows;
                    }
                    let source_y = usize::try_from(
                        i32::try_from(y).map_err(|_| Error::LimitExceeded)? + delta_row,
                    )
                    .map_err(|_| Error::InvalidObu)?;
                    let source_x = usize::try_from(
                        i32::try_from(x).map_err(|_| Error::LimitExceeded)? + delta_column,
                    )
                    .map_err(|_| Error::InvalidObu)?;
                    let source = source_y * width + source_x;
                    sums[0] += i32::from(patterns.cb[source]) * cb_coefficient;
                    sums[1] += i32::from(patterns.cr[source]) * cr_coefficient;
                    position += 1;
                }
            }
            let index = y * width + x;
            patterns.cb[index] = i16::try_from(
                (i32::from(patterns.cb[index]) + round2(sums[0], shift)).clamp(minimum, maximum),
            )
            .map_err(|_| Error::LimitExceeded)?;
            patterns.cr[index] = i16::try_from(
                (i32::from(patterns.cr[index]) + round2(sums[1], shift)).clamp(minimum, maximum),
            )
            .map_err(|_| Error::LimitExceeded)?;
        }
    }
    Ok(())
}

const fn round2(value: i32, shift: u8) -> i32 {
    if shift == 0 {
        value
    } else {
        (value + (1i32 << (shift - 1))) >> shift
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NoisePlane {
    width: usize,
    height: usize,
    samples: Vector<i16>,
}

impl NoisePlane {
    fn new(width: usize, height: usize) -> Result<Self, Error> {
        let count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
        let mut samples = Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
        for _ in 0..count {
            samples.try_push(0).map_err(|_| Error::LimitExceeded)?;
        }
        Ok(Self {
            width,
            height,
            samples,
        })
    }

    fn get(&self, x: usize, y: usize) -> Result<i16, Error> {
        if x >= self.width || y >= self.height {
            return Err(Error::InvalidObu);
        }
        self.samples
            .get(y * self.width + x)
            .copied()
            .ok_or(Error::InvalidObu)
    }

    fn set(&mut self, x: usize, y: usize, value: i16) -> Result<(), Error> {
        if x >= self.width || y >= self.height {
            return Err(Error::InvalidObu);
        }
        *self
            .samples
            .get_mut(y * self.width + x)
            .ok_or(Error::InvalidObu)? = value;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct NoiseImageConfig {
    width: usize,
    height: usize,
    planes: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    minimum: i32,
    maximum: i32,
}

fn build_noise_image(
    grain: &FilmGrain,
    patterns: &GrainPatterns,
    config: NoiseImageConfig,
) -> Result<[NoisePlane; 3], Error> {
    let NoiseImageConfig {
        width,
        height,
        planes,
        subsampling_x,
        subsampling_y,
        minimum,
        maximum,
    } = config;
    let chroma_width = (width + usize::from(subsampling_x)) >> usize::from(subsampling_x);
    let chroma_height = (height + usize::from(subsampling_y)) >> usize::from(subsampling_y);
    let mut image = [
        NoisePlane::new(width, height)?,
        NoisePlane::new(chroma_width, chroma_height)?,
        NoisePlane::new(chroma_width, chroma_height)?,
    ];
    let stripe_count = height.div_ceil(32);
    for (plane, output_plane) in image.iter_mut().enumerate().take(planes) {
        let plane_sub_x = plane > 0 && subsampling_x;
        let plane_sub_y = plane > 0 && subsampling_y;
        let stripe_height = 34 >> usize::from(plane_sub_y);
        let logical_width = if plane == 0 { width } else { chroma_width };
        let stripe_stride = logical_width.checked_add(34).ok_or(Error::LimitExceeded)?;
        let mut previous = NoisePlane::new(stripe_stride, stripe_height)?;
        let mut current = NoisePlane::new(stripe_stride, stripe_height)?;
        for luma_number in 0..stripe_count {
            current.samples.fill(0);
            let high =
                u16::try_from((luma_number * 37 + 178) & 255).map_err(|_| Error::LimitExceeded)?;
            let low =
                u16::try_from((luma_number * 173 + 105) & 255).map_err(|_| Error::LimitExceeded)?;
            let mut random = GrainRandom::new(grain.grain_seed ^ (high << 8) ^ low);
            for block_x in (0..width.div_ceil(2)).step_by(16) {
                let random_offset = random.number(8)?;
                let offset_x = usize::from(random_offset >> 4);
                let offset_y = usize::from(random_offset & 15);
                let source_x = if plane_sub_x {
                    6 + offset_x
                } else {
                    9 + 2 * offset_x
                };
                let source_y = if plane_sub_y {
                    6 + offset_y
                } else {
                    9 + 2 * offset_y
                };
                let block_width = 34 >> usize::from(plane_sub_x);
                for row in 0..stripe_height {
                    for column in 0..block_width {
                        let mut value = if plane == 0 {
                            patterns.luma(source_x + column, source_y + row)
                        } else {
                            patterns.chroma(plane as u8, source_x + column, source_y + row)
                        }
                        .ok_or(Error::InvalidObu)?;
                        let destination_x = if plane_sub_x {
                            block_x + column
                        } else {
                            block_x * 2 + column
                        };
                        if destination_x >= stripe_stride {
                            continue;
                        }
                        if grain.overlap && block_x > 0 && column < 2 >> usize::from(plane_sub_x) {
                            let old = i32::from(current.get(destination_x, row)?);
                            let mixed = if plane_sub_x {
                                old * 23 + i32::from(value) * 22
                            } else if column == 0 {
                                old * 27 + i32::from(value) * 17
                            } else {
                                old * 17 + i32::from(value) * 27
                            };
                            value = i16::try_from(round2(mixed, 5).clamp(minimum, maximum))
                                .map_err(|_| Error::LimitExceeded)?;
                        }
                        current.set(destination_x, row, value)?;
                    }
                }
            }
            let output_start = luma_number * (32 >> usize::from(plane_sub_y));
            let output_height = if plane == 0 { height } else { chroma_height };
            for row in 0..(32 >> usize::from(plane_sub_y)) {
                let output_y = output_start + row;
                if output_y >= output_height {
                    break;
                }
                for x in 0..logical_width {
                    let mut value = current.get(x, row)?;
                    let overlap_rows = 2 >> usize::from(plane_sub_y);
                    if grain.overlap && luma_number > 0 && row < overlap_rows {
                        let old =
                            i32::from(previous.get(x, row + (32 >> usize::from(plane_sub_y)))?);
                        let mixed = if plane_sub_y {
                            old * 23 + i32::from(value) * 22
                        } else if row == 0 {
                            old * 27 + i32::from(value) * 17
                        } else {
                            old * 17 + i32::from(value) * 27
                        };
                        value = i16::try_from(round2(mixed, 5).clamp(minimum, maximum))
                            .map_err(|_| Error::LimitExceeded)?;
                    }
                    output_plane.set(x, output_y, value)?;
                }
            }
            core::mem::swap(&mut previous, &mut current);
        }
    }
    Ok(image)
}

fn scale_lut(lookup: &[u8; 256], index: u16, bit_depth: u8) -> i32 {
    let shift = bit_depth - 8;
    let x = usize::from(index >> shift).min(255);
    if shift == 0 || x == 255 {
        return i32::from(lookup[x]);
    }
    let remainder = i32::from(index) - (i32::try_from(x).unwrap_or(255) << shift);
    let start = i32::from(lookup[x]);
    let end = i32::from(lookup[x + 1]);
    start + round2((end - start) * remainder, shift)
}

fn add_noise_to_plane(
    plane: &mut Plane,
    noise: &NoisePlane,
    lookup: &[u8; 256],
    bit_depth: u8,
    scaling_shift: u8,
    minimum: i32,
    maximum: i32,
) -> Result<(), Error> {
    for y in 0..noise.height {
        for x in 0..noise.width {
            let original = plane.sample(x, y)?;
            let scaled = scale_lut(lookup, original, bit_depth) * i32::from(noise.get(x, y)?);
            let value =
                (i32::from(original) + round2(scaled, scaling_shift)).clamp(minimum, maximum);
            plane.set_sample(
                x,
                y,
                u16::try_from(value).map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

/// Apply film grain to an output buffer. The caller must retain its unmodified
/// restoration output for reference-frame storage.
pub fn apply(
    output: &mut FrameBuffer,
    grain: &FilmGrain,
    matrix_coefficients: u8,
    width: u32,
    height: u32,
) -> Result<(), Error> {
    if !grain.apply_grain {
        return Ok(());
    }
    let bit_depth = output.bit_depth();
    let (subsampling_x, subsampling_y, planes) = match output.sampling() {
        ChromaSampling::Cs400 => (false, false, 1),
        ChromaSampling::Cs420 => (true, true, 3),
        ChromaSampling::Cs422 => (true, false, 3),
        ChromaSampling::Cs444 => (false, false, 3),
    };
    let width = usize::try_from(width).map_err(|_| Error::LimitExceeded)?;
    let height = usize::try_from(height).map_err(|_| Error::LimitExceeded)?;
    let grain_center = 128i32 << (bit_depth - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bit_depth - 8)) - 1 - grain_center;
    let patterns = generate_grain(grain, bit_depth, planes == 1, subsampling_x, subsampling_y)?;
    let noise = build_noise_image(
        grain,
        &patterns,
        NoiseImageConfig {
            width,
            height,
            planes,
            subsampling_x,
            subsampling_y,
            minimum: grain_min,
            maximum: grain_max,
        },
    )?;
    let y_lookup = scaling_lookup(
        &grain.point_y_value,
        &grain.point_y_scaling,
        grain.num_y_points,
    )?;
    let cb_lookup = if grain.chroma_scaling_from_luma {
        y_lookup
    } else {
        scaling_lookup(
            &grain.point_cb_value,
            &grain.point_cb_scaling,
            grain.num_cb_points,
        )?
    };
    let cr_lookup = if grain.chroma_scaling_from_luma {
        y_lookup
    } else {
        scaling_lookup(
            &grain.point_cr_value,
            &grain.point_cr_scaling,
            grain.num_cr_points,
        )?
    };
    let (minimum, maximum_luma, maximum_chroma) = if grain.clip_to_restricted_range {
        let minimum = 16i32 << (bit_depth - 8);
        let maximum_luma = 235i32 << (bit_depth - 8);
        let maximum_chroma = if matrix_coefficients == 0 {
            maximum_luma
        } else {
            240i32 << (bit_depth - 8)
        };
        (minimum, maximum_luma, maximum_chroma)
    } else {
        let maximum = (256i32 << (bit_depth - 8)) - 1;
        (0, maximum, maximum)
    };
    let scaling_shift = grain.grain_scaling_minus_8 + 8;

    if planes > 1 {
        let chroma_width = (width + usize::from(subsampling_x)) >> usize::from(subsampling_x);
        let chroma_height = (height + usize::from(subsampling_y)) >> usize::from(subsampling_y);
        let u = output.u.as_mut().ok_or(Error::InvalidObu)?;
        let v = output.v.as_mut().ok_or(Error::InvalidObu)?;
        for y in 0..chroma_height {
            for x in 0..chroma_width {
                let luma_x = x << usize::from(subsampling_x);
                let luma_y = y << usize::from(subsampling_y);
                let next_luma_x = (luma_x + 1).min(width - 1);
                let average_luma = if subsampling_x {
                    round2(
                        i32::from(output.y.sample(luma_x, luma_y)?)
                            + i32::from(output.y.sample(next_luma_x, luma_y)?),
                        1,
                    )
                } else {
                    i32::from(output.y.sample(luma_x, luma_y)?)
                };
                for (plane, destination, lookup, multiplier, luma_multiplier, offset) in [
                    (
                        1usize,
                        &mut *u,
                        &cb_lookup,
                        grain.cb_mult,
                        grain.cb_luma_mult,
                        grain.cb_offset,
                    ),
                    (
                        2usize,
                        &mut *v,
                        &cr_lookup,
                        grain.cr_mult,
                        grain.cr_luma_mult,
                        grain.cr_offset,
                    ),
                ] {
                    let enabled = if plane == 1 {
                        grain.num_cb_points > 0
                    } else {
                        grain.num_cr_points > 0
                    } || grain.chroma_scaling_from_luma;
                    if !enabled {
                        continue;
                    }
                    let original = destination.sample(x, y)?;
                    let merged = if grain.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined = average_luma * (i32::from(luma_multiplier) - 128)
                            + i32::from(original) * (i32::from(multiplier) - 128);
                        ((combined >> 6) + ((i32::from(offset) - 256) << (bit_depth - 8)))
                            .clamp(0, (1i32 << bit_depth) - 1)
                    };
                    let scaled = scale_lut(
                        lookup,
                        u16::try_from(merged).map_err(|_| Error::LimitExceeded)?,
                        bit_depth,
                    ) * i32::from(noise[plane].get(x, y)?);
                    let value = (i32::from(original) + round2(scaled, scaling_shift))
                        .clamp(minimum, maximum_chroma);
                    destination.set_sample(
                        x,
                        y,
                        u16::try_from(value).map_err(|_| Error::LimitExceeded)?,
                    )?;
                }
            }
        }
    }
    if grain.num_y_points > 0 {
        add_noise_to_plane(
            &mut output.y,
            &noise[0],
            &y_lookup,
            bit_depth,
            scaling_shift,
            minimum,
            maximum_luma,
        )?;
    }
    Ok(())
}

pub(crate) fn parse(
    bits: &mut Bits<'_>,
    sequence: &Sequence,
    show_frame: bool,
    showable_frame: bool,
    is_inter_frame: bool,
    ref_frame_idx: &[u8; 7],
    references: &[Option<FilmGrain>; 8],
) -> Result<FilmGrain, Error> {
    if !sequence.film_grain_params_present || (!show_frame && !showable_frame) {
        return Ok(FilmGrain::default());
    }
    if !bits.bit()? {
        return Ok(FilmGrain::default());
    }
    let seed = bits.read(16)? as u16;
    let update_grain = !is_inter_frame || bits.bit()?;
    if !update_grain {
        let index = bits.read(3)? as u8;
        if !ref_frame_idx.contains(&index) {
            return Err(Error::InvalidObu);
        }
        let mut inherited = references[usize::from(index)].ok_or(Error::InvalidObu)?;
        inherited.apply_grain = true;
        inherited.grain_seed = seed;
        inherited.update_grain = false;
        inherited.reference_index = index;
        return Ok(inherited);
    }

    let mut result = FilmGrain {
        apply_grain: true,
        grain_seed: seed,
        update_grain: true,
        ..FilmGrain::default()
    };
    result.num_y_points = bits.read(4)? as u8;
    if usize::from(result.num_y_points) > MAX_Y_POINTS {
        return Err(Error::InvalidObu);
    }
    read_points(
        bits,
        result.num_y_points,
        &mut result.point_y_value,
        &mut result.point_y_scaling,
    )?;
    result.chroma_scaling_from_luma = !sequence.monochrome && bits.bit()?;
    let suppress_uv_points = sequence.monochrome
        || result.chroma_scaling_from_luma
        || (sequence.chroma_sampling == ChromaSampling::Cs420 && result.num_y_points == 0);
    if !suppress_uv_points {
        result.num_cb_points = bits.read(4)? as u8;
        if usize::from(result.num_cb_points) > MAX_UV_POINTS {
            return Err(Error::InvalidObu);
        }
        read_points(
            bits,
            result.num_cb_points,
            &mut result.point_cb_value,
            &mut result.point_cb_scaling,
        )?;
        result.num_cr_points = bits.read(4)? as u8;
        if usize::from(result.num_cr_points) > MAX_UV_POINTS
            || (sequence.chroma_sampling == ChromaSampling::Cs420
                && result.num_cb_points == 0
                && result.num_cr_points != 0)
        {
            return Err(Error::InvalidObu);
        }
        read_points(
            bits,
            result.num_cr_points,
            &mut result.point_cr_value,
            &mut result.point_cr_scaling,
        )?;
    }
    result.grain_scaling_minus_8 = bits.read(2)? as u8;
    result.ar_coeff_lag = bits.read(2)? as u8;
    let luma_coefficients =
        2 * usize::from(result.ar_coeff_lag) * (usize::from(result.ar_coeff_lag) + 1);
    let chroma_coefficients = luma_coefficients + usize::from(result.num_y_points > 0);
    if result.num_y_points > 0 {
        read_values(bits, &mut result.ar_coeffs_y_plus_128[..luma_coefficients])?;
    }
    if result.chroma_scaling_from_luma || result.num_cb_points > 0 {
        read_values(
            bits,
            &mut result.ar_coeffs_cb_plus_128[..chroma_coefficients],
        )?;
    }
    if result.chroma_scaling_from_luma || result.num_cr_points > 0 {
        read_values(
            bits,
            &mut result.ar_coeffs_cr_plus_128[..chroma_coefficients],
        )?;
    }
    result.ar_coeff_shift_minus_6 = bits.read(2)? as u8;
    result.grain_scale_shift = bits.read(2)? as u8;
    if result.num_cb_points > 0 {
        result.cb_mult = bits.read(8)? as u8;
        result.cb_luma_mult = bits.read(8)? as u8;
        result.cb_offset = bits.read(9)? as u16;
    }
    if result.num_cr_points > 0 {
        result.cr_mult = bits.read(8)? as u8;
        result.cr_luma_mult = bits.read(8)? as u8;
        result.cr_offset = bits.read(9)? as u16;
    }
    result.overlap = bits.bit()?;
    result.clip_to_restricted_range = bits.bit()?;
    Ok(result)
}

fn read_points<const N: usize>(
    bits: &mut Bits<'_>,
    count: u8,
    values: &mut [u8; N],
    scaling: &mut [u8; N],
) -> Result<(), Error> {
    let mut previous = None;
    for index in 0..usize::from(count) {
        values[index] = bits.read(8)? as u8;
        scaling[index] = bits.read(8)? as u8;
        if previous.is_some_and(|value| values[index] <= value) {
            return Err(Error::InvalidObu);
        }
        previous = Some(values[index]);
    }
    Ok(())
}

fn read_values(bits: &mut Bits<'_>, output: &mut [u8]) -> Result<(), Error> {
    for value in output {
        *value = bits.read(8)? as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_ar_lag_fits_fixed_arrays() {
        let lag = 3usize;
        assert_eq!(2 * lag * (lag + 1), MAX_AR_COEFFS_Y);
        assert_eq!(2 * lag * (lag + 1) + 1, MAX_AR_COEFFS_UV);
    }

    #[test]
    fn normative_gaussian_sequence_is_complete_and_bounded() {
        assert_eq!(GAUSSIAN_SEQUENCE.len(), 2048);
        assert_eq!(&GAUSSIAN_SEQUENCE[..5], &[56, 568, -180, 172, 124]);
        assert_eq!(&GAUSSIAN_SEQUENCE[2043..], &[280, 288, 944, 428, -484]);
        assert!(
            GAUSSIAN_SEQUENCE
                .iter()
                .all(|value| (-2048..=2047).contains(value))
        );
        assert!(GAUSSIAN_SEQUENCE.iter().all(|value| value % 4 == 0));
    }

    #[test]
    fn random_process_updates_before_extracting_bits() {
        let mut random = GrainRandom::new(0x1234);
        let expected_register =
            (0x1234 >> 1) | (((0x1234 ^ (0x1234 >> 1) ^ (0x1234 >> 3) ^ (0x1234 >> 12)) & 1) << 15);
        assert_eq!(random.number(8), Ok(expected_register >> 8));
        assert_eq!(random.number(17), Err(Error::InvalidObu));
    }

    #[test]
    fn scaling_lookup_interpolates_and_extends_endpoints() {
        let lookup = scaling_lookup(&[16, 32], &[10, 30], 2).unwrap();
        assert_eq!(lookup[0], 10);
        assert_eq!(lookup[16], 10);
        assert_eq!(lookup[24], 20);
        assert_eq!(lookup[31], 29);
        assert_eq!(lookup[32], 30);
        assert_eq!(lookup[255], 30);
    }

    #[test]
    fn grain_generation_is_deterministic_and_bounded() {
        let mut grain = FilmGrain {
            grain_seed: 0x1234,
            num_y_points: 1,
            num_cb_points: 1,
            num_cr_points: 1,
            ar_coeff_lag: 3,
            ar_coeff_shift_minus_6: 2,
            grain_scale_shift: 1,
            ..FilmGrain::default()
        };
        grain.ar_coeffs_y_plus_128.fill(129);
        grain.ar_coeffs_cb_plus_128.fill(127);
        grain.ar_coeffs_cr_plus_128.fill(130);
        let first = generate_grain(&grain, 10, false, true, true).unwrap();
        let second = generate_grain(&grain, 10, false, true, true).unwrap();
        assert_eq!(first, second);
        assert!(
            first
                .luma
                .iter()
                .all(|sample| (-512..=511).contains(sample))
        );
        assert!(
            first.cb[..44 * 38]
                .iter()
                .all(|sample| (-512..=511).contains(sample))
        );
        assert!(
            first.cr[..44 * 38]
                .iter()
                .all(|sample| (-512..=511).contains(sample))
        );
        assert_eq!(first.chroma(1, 44, 0), None);
    }

    #[test]
    fn absent_scaling_points_produce_zero_grain() {
        let patterns = generate_grain(&FilmGrain::default(), 8, false, true, true).unwrap();
        assert!(patterns.luma.iter().all(|sample| *sample == 0));
        assert!(patterns.cb.iter().all(|sample| *sample == 0));
        assert!(patterns.cr.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn synthesis_changes_only_the_output_copy() {
        let reference = FrameBuffer::new(65, 35, 8, ChromaSampling::Cs420).unwrap();
        let mut output = reference.clone();
        let mut grain = FilmGrain {
            apply_grain: true,
            grain_seed: 0xace1,
            num_y_points: 2,
            num_cb_points: 2,
            num_cr_points: 2,
            overlap: true,
            ..FilmGrain::default()
        };
        grain.point_y_value[..2].copy_from_slice(&[0, 255]);
        grain.point_y_scaling[..2].copy_from_slice(&[64, 64]);
        grain.point_cb_value[..2].copy_from_slice(&[0, 255]);
        grain.point_cb_scaling[..2].copy_from_slice(&[64, 64]);
        grain.point_cr_value[..2].copy_from_slice(&[0, 255]);
        grain.point_cr_scaling[..2].copy_from_slice(&[64, 64]);
        apply(&mut output, &grain, 1, 65, 35).unwrap();
        assert_ne!(output, reference);
        assert!(reference.y.samples().iter().all(|sample| *sample == 128));
        assert!(output.y.samples().iter().all(|sample| *sample <= 255));
    }

    #[test]
    fn restricted_range_clips_synthesized_samples() {
        let mut output = FrameBuffer::new(33, 33, 10, ChromaSampling::Cs444).unwrap();
        let mut grain = FilmGrain {
            apply_grain: true,
            grain_seed: 7,
            num_y_points: 1,
            point_y_value: [0; MAX_Y_POINTS],
            point_y_scaling: [255; MAX_Y_POINTS],
            overlap: true,
            clip_to_restricted_range: true,
            ..FilmGrain::default()
        };
        grain.point_y_value[0] = 128;
        apply(&mut output, &grain, 1, 33, 33).unwrap();
        assert!(
            output
                .y
                .samples()
                .iter()
                .all(|sample| (64..=940).contains(sample))
        );
    }
}
