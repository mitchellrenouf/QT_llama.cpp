//! Coefficient end-of-block and Golomb extension primitives.

use crate::{
    CoefficientStage, Error,
    cdf::TileCdfs,
    entropy::SymbolDecoder,
    transform::{TxSize, TxType},
};
use mrml_runtime::Vector;

pub const NUM_BASE_LEVELS: u32 = 2;
pub const COEFF_BASE_RANGE: u32 = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanKind {
    Default,
    Row,
    Column,
}

#[cfg(test)]
const DEFAULT_SCAN_4X4: [u16; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
#[cfg(test)]
const DEFAULT_SCAN_4X8: [u16; 32] = [
    0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11, 14, 17, 20, 15, 18, 21, 24, 19, 22, 25, 28,
    23, 26, 29, 27, 30, 31,
];
#[cfg(test)]
const DEFAULT_SCAN_8X4: [u16; 32] = [
    0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 25, 18, 11, 4, 26, 19, 12, 5, 27, 20, 13, 6, 28, 21, 14, 7,
    29, 22, 15, 30, 23, 31,
];
#[cfg(test)]
const DEFAULT_SCAN_8X8: [u16; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
#[cfg(test)]
const DEFAULT_SCAN_8X16: [u16; 128] = [
    0, 1, 8, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34,
    41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 64, 23, 30, 37, 44, 51, 58,
    65, 72, 31, 38, 45, 52, 59, 66, 73, 80, 39, 46, 53, 60, 67, 74, 81, 88, 47, 54, 61, 68, 75, 82,
    89, 96, 55, 62, 69, 76, 83, 90, 97, 104, 63, 70, 77, 84, 91, 98, 105, 112, 71, 78, 85, 92, 99,
    106, 113, 120, 79, 86, 93, 100, 107, 114, 121, 87, 94, 101, 108, 115, 122, 95, 102, 109, 116,
    123, 103, 110, 117, 124, 111, 118, 125, 119, 126, 127,
];
#[cfg(test)]
const DEFAULT_SCAN_16X8: [u16; 128] = [
    0, 16, 1, 32, 17, 2, 48, 33, 18, 3, 64, 49, 34, 19, 4, 80, 65, 50, 35, 20, 5, 96, 81, 66, 51,
    36, 21, 6, 112, 97, 82, 67, 52, 37, 22, 7, 113, 98, 83, 68, 53, 38, 23, 8, 114, 99, 84, 69, 54,
    39, 24, 9, 115, 100, 85, 70, 55, 40, 25, 10, 116, 101, 86, 71, 56, 41, 26, 11, 117, 102, 87,
    72, 57, 42, 27, 12, 118, 103, 88, 73, 58, 43, 28, 13, 119, 104, 89, 74, 59, 44, 29, 14, 120,
    105, 90, 75, 60, 45, 30, 15, 121, 106, 91, 76, 61, 46, 31, 122, 107, 92, 77, 62, 47, 123, 108,
    93, 78, 63, 124, 109, 94, 79, 125, 110, 95, 126, 111, 127,
];
#[cfg(test)]
const DEFAULT_SCAN_4X16: [u16; 64] = [
    0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11, 14, 17, 20, 15, 18, 21, 24, 19, 22, 25, 28,
    23, 26, 29, 32, 27, 30, 33, 36, 31, 34, 37, 40, 35, 38, 41, 44, 39, 42, 45, 48, 43, 46, 49, 52,
    47, 50, 53, 56, 51, 54, 57, 60, 55, 58, 61, 59, 62, 63,
];
#[cfg(test)]
const DEFAULT_SCAN_16X4: [u16; 64] = [
    0, 16, 1, 32, 17, 2, 48, 33, 18, 3, 49, 34, 19, 4, 50, 35, 20, 5, 51, 36, 21, 6, 52, 37, 22, 7,
    53, 38, 23, 8, 54, 39, 24, 9, 55, 40, 25, 10, 56, 41, 26, 11, 57, 42, 27, 12, 58, 43, 28, 13,
    59, 44, 29, 14, 60, 45, 30, 15, 61, 46, 31, 62, 47, 63,
];
#[cfg(test)]
const DEFAULT_SCAN_16X16: [u16; 256] = [
    0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5, 20, 35, 50, 65, 80, 96, 81, 66, 51,
    36, 21, 6, 7, 22, 37, 52, 67, 82, 97, 112, 128, 113, 98, 83, 68, 53, 38, 23, 8, 9, 24, 39, 54,
    69, 84, 99, 114, 129, 144, 160, 145, 130, 115, 100, 85, 70, 55, 40, 25, 10, 11, 26, 41, 56, 71,
    86, 101, 116, 131, 146, 161, 176, 192, 177, 162, 147, 132, 117, 102, 87, 72, 57, 42, 27, 12,
    13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149,
    134, 119, 104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
    195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136, 121, 106, 91, 76, 61, 46, 31, 47,
    62, 77, 92, 107, 122, 137, 152, 167, 182, 197, 212, 227, 242, 243, 228, 213, 198, 183, 168,
    153, 138, 123, 108, 93, 78, 63, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245,
    230, 215, 200, 185, 170, 155, 140, 125, 110, 95, 111, 126, 141, 156, 171, 186, 201, 216, 231,
    246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 203, 218, 233, 248, 249,
    234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235, 250, 251, 236, 221, 206, 191, 207, 222,
    237, 252, 253, 238, 223, 239, 254, 255,
];
#[cfg(test)]
const DEFAULT_SCAN_8X32: [u16; 256] = [
    0, 1, 8, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34,
    41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 64, 23, 30, 37, 44, 51, 58,
    65, 72, 31, 38, 45, 52, 59, 66, 73, 80, 39, 46, 53, 60, 67, 74, 81, 88, 47, 54, 61, 68, 75, 82,
    89, 96, 55, 62, 69, 76, 83, 90, 97, 104, 63, 70, 77, 84, 91, 98, 105, 112, 71, 78, 85, 92, 99,
    106, 113, 120, 79, 86, 93, 100, 107, 114, 121, 128, 87, 94, 101, 108, 115, 122, 129, 136, 95,
    102, 109, 116, 123, 130, 137, 144, 103, 110, 117, 124, 131, 138, 145, 152, 111, 118, 125, 132,
    139, 146, 153, 160, 119, 126, 133, 140, 147, 154, 161, 168, 127, 134, 141, 148, 155, 162, 169,
    176, 135, 142, 149, 156, 163, 170, 177, 184, 143, 150, 157, 164, 171, 178, 185, 192, 151, 158,
    165, 172, 179, 186, 193, 200, 159, 166, 173, 180, 187, 194, 201, 208, 167, 174, 181, 188, 195,
    202, 209, 216, 175, 182, 189, 196, 203, 210, 217, 224, 183, 190, 197, 204, 211, 218, 225, 232,
    191, 198, 205, 212, 219, 226, 233, 240, 199, 206, 213, 220, 227, 234, 241, 248, 207, 214, 221,
    228, 235, 242, 249, 215, 222, 229, 236, 243, 250, 223, 230, 237, 244, 251, 231, 238, 245, 252,
    239, 246, 253, 247, 254, 255,
];
#[cfg(test)]
const DEFAULT_SCAN_32X8: [u16; 256] = [
    0, 32, 1, 64, 33, 2, 96, 65, 34, 3, 128, 97, 66, 35, 4, 160, 129, 98, 67, 36, 5, 192, 161, 130,
    99, 68, 37, 6, 224, 193, 162, 131, 100, 69, 38, 7, 225, 194, 163, 132, 101, 70, 39, 8, 226,
    195, 164, 133, 102, 71, 40, 9, 227, 196, 165, 134, 103, 72, 41, 10, 228, 197, 166, 135, 104,
    73, 42, 11, 229, 198, 167, 136, 105, 74, 43, 12, 230, 199, 168, 137, 106, 75, 44, 13, 231, 200,
    169, 138, 107, 76, 45, 14, 232, 201, 170, 139, 108, 77, 46, 15, 233, 202, 171, 140, 109, 78,
    47, 16, 234, 203, 172, 141, 110, 79, 48, 17, 235, 204, 173, 142, 111, 80, 49, 18, 236, 205,
    174, 143, 112, 81, 50, 19, 237, 206, 175, 144, 113, 82, 51, 20, 238, 207, 176, 145, 114, 83,
    52, 21, 239, 208, 177, 146, 115, 84, 53, 22, 240, 209, 178, 147, 116, 85, 54, 23, 241, 210,
    179, 148, 117, 86, 55, 24, 242, 211, 180, 149, 118, 87, 56, 25, 243, 212, 181, 150, 119, 88,
    57, 26, 244, 213, 182, 151, 120, 89, 58, 27, 245, 214, 183, 152, 121, 90, 59, 28, 246, 215,
    184, 153, 122, 91, 60, 29, 247, 216, 185, 154, 123, 92, 61, 30, 248, 217, 186, 155, 124, 93,
    62, 31, 249, 218, 187, 156, 125, 94, 63, 250, 219, 188, 157, 126, 95, 251, 220, 189, 158, 127,
    252, 221, 190, 159, 253, 222, 191, 254, 223, 255,
];
#[cfg(test)]
const DEFAULT_SCAN_16X32: [u16; 512] = [
    0, 1, 16, 2, 17, 32, 3, 18, 33, 48, 4, 19, 34, 49, 64, 5, 20, 35, 50, 65, 80, 6, 21, 36, 51,
    66, 81, 96, 7, 22, 37, 52, 67, 82, 97, 112, 8, 23, 38, 53, 68, 83, 98, 113, 128, 9, 24, 39, 54,
    69, 84, 99, 114, 129, 144, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 11, 26, 41, 56, 71,
    86, 101, 116, 131, 146, 161, 176, 12, 27, 42, 57, 72, 87, 102, 117, 132, 147, 162, 177, 192,
    13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 14, 29, 44, 59, 74, 89, 104,
    119, 134, 149, 164, 179, 194, 209, 224, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
    195, 210, 225, 240, 31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 181, 196, 211, 226, 241, 256,
    47, 62, 77, 92, 107, 122, 137, 152, 167, 182, 197, 212, 227, 242, 257, 272, 63, 78, 93, 108,
    123, 138, 153, 168, 183, 198, 213, 228, 243, 258, 273, 288, 79, 94, 109, 124, 139, 154, 169,
    184, 199, 214, 229, 244, 259, 274, 289, 304, 95, 110, 125, 140, 155, 170, 185, 200, 215, 230,
    245, 260, 275, 290, 305, 320, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 261, 276, 291,
    306, 321, 336, 127, 142, 157, 172, 187, 202, 217, 232, 247, 262, 277, 292, 307, 322, 337, 352,
    143, 158, 173, 188, 203, 218, 233, 248, 263, 278, 293, 308, 323, 338, 353, 368, 159, 174, 189,
    204, 219, 234, 249, 264, 279, 294, 309, 324, 339, 354, 369, 384, 175, 190, 205, 220, 235, 250,
    265, 280, 295, 310, 325, 340, 355, 370, 385, 400, 191, 206, 221, 236, 251, 266, 281, 296, 311,
    326, 341, 356, 371, 386, 401, 416, 207, 222, 237, 252, 267, 282, 297, 312, 327, 342, 357, 372,
    387, 402, 417, 432, 223, 238, 253, 268, 283, 298, 313, 328, 343, 358, 373, 388, 403, 418, 433,
    448, 239, 254, 269, 284, 299, 314, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464, 255, 270,
    285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 271, 286, 301, 316, 331,
    346, 361, 376, 391, 406, 421, 436, 451, 466, 481, 496, 287, 302, 317, 332, 347, 362, 377, 392,
    407, 422, 437, 452, 467, 482, 497, 303, 318, 333, 348, 363, 378, 393, 408, 423, 438, 453, 468,
    483, 498, 319, 334, 349, 364, 379, 394, 409, 424, 439, 454, 469, 484, 499, 335, 350, 365, 380,
    395, 410, 425, 440, 455, 470, 485, 500, 351, 366, 381, 396, 411, 426, 441, 456, 471, 486, 501,
    367, 382, 397, 412, 427, 442, 457, 472, 487, 502, 383, 398, 413, 428, 443, 458, 473, 488, 503,
    399, 414, 429, 444, 459, 474, 489, 504, 415, 430, 445, 460, 475, 490, 505, 431, 446, 461, 476,
    491, 506, 447, 462, 477, 492, 507, 463, 478, 493, 508, 479, 494, 509, 495, 510, 511,
];
#[cfg(test)]
const DEFAULT_SCAN_32X16: [u16; 512] = [
    0, 32, 1, 64, 33, 2, 96, 65, 34, 3, 128, 97, 66, 35, 4, 160, 129, 98, 67, 36, 5, 192, 161, 130,
    99, 68, 37, 6, 224, 193, 162, 131, 100, 69, 38, 7, 256, 225, 194, 163, 132, 101, 70, 39, 8,
    288, 257, 226, 195, 164, 133, 102, 71, 40, 9, 320, 289, 258, 227, 196, 165, 134, 103, 72, 41,
    10, 352, 321, 290, 259, 228, 197, 166, 135, 104, 73, 42, 11, 384, 353, 322, 291, 260, 229, 198,
    167, 136, 105, 74, 43, 12, 416, 385, 354, 323, 292, 261, 230, 199, 168, 137, 106, 75, 44, 13,
    448, 417, 386, 355, 324, 293, 262, 231, 200, 169, 138, 107, 76, 45, 14, 480, 449, 418, 387,
    356, 325, 294, 263, 232, 201, 170, 139, 108, 77, 46, 15, 481, 450, 419, 388, 357, 326, 295,
    264, 233, 202, 171, 140, 109, 78, 47, 16, 482, 451, 420, 389, 358, 327, 296, 265, 234, 203,
    172, 141, 110, 79, 48, 17, 483, 452, 421, 390, 359, 328, 297, 266, 235, 204, 173, 142, 111, 80,
    49, 18, 484, 453, 422, 391, 360, 329, 298, 267, 236, 205, 174, 143, 112, 81, 50, 19, 485, 454,
    423, 392, 361, 330, 299, 268, 237, 206, 175, 144, 113, 82, 51, 20, 486, 455, 424, 393, 362,
    331, 300, 269, 238, 207, 176, 145, 114, 83, 52, 21, 487, 456, 425, 394, 363, 332, 301, 270,
    239, 208, 177, 146, 115, 84, 53, 22, 488, 457, 426, 395, 364, 333, 302, 271, 240, 209, 178,
    147, 116, 85, 54, 23, 489, 458, 427, 396, 365, 334, 303, 272, 241, 210, 179, 148, 117, 86, 55,
    24, 490, 459, 428, 397, 366, 335, 304, 273, 242, 211, 180, 149, 118, 87, 56, 25, 491, 460, 429,
    398, 367, 336, 305, 274, 243, 212, 181, 150, 119, 88, 57, 26, 492, 461, 430, 399, 368, 337,
    306, 275, 244, 213, 182, 151, 120, 89, 58, 27, 493, 462, 431, 400, 369, 338, 307, 276, 245,
    214, 183, 152, 121, 90, 59, 28, 494, 463, 432, 401, 370, 339, 308, 277, 246, 215, 184, 153,
    122, 91, 60, 29, 495, 464, 433, 402, 371, 340, 309, 278, 247, 216, 185, 154, 123, 92, 61, 30,
    496, 465, 434, 403, 372, 341, 310, 279, 248, 217, 186, 155, 124, 93, 62, 31, 497, 466, 435,
    404, 373, 342, 311, 280, 249, 218, 187, 156, 125, 94, 63, 498, 467, 436, 405, 374, 343, 312,
    281, 250, 219, 188, 157, 126, 95, 499, 468, 437, 406, 375, 344, 313, 282, 251, 220, 189, 158,
    127, 500, 469, 438, 407, 376, 345, 314, 283, 252, 221, 190, 159, 501, 470, 439, 408, 377, 346,
    315, 284, 253, 222, 191, 502, 471, 440, 409, 378, 347, 316, 285, 254, 223, 503, 472, 441, 410,
    379, 348, 317, 286, 255, 504, 473, 442, 411, 380, 349, 318, 287, 505, 474, 443, 412, 381, 350,
    319, 506, 475, 444, 413, 382, 351, 507, 476, 445, 414, 383, 508, 477, 446, 415, 509, 478, 447,
    510, 479, 511,
];

pub const fn effective_scan_size(size: TxSize) -> TxSize {
    match size {
        TxSize::Tx16x64 => TxSize::Tx16x32,
        TxSize::Tx64x16 => TxSize::Tx32x16,
        TxSize::Tx64x64 | TxSize::Tx32x64 | TxSize::Tx64x32 => TxSize::Tx32x32,
        _ => size,
    }
}

pub const fn coefficient_scan_kind(size: TxSize, tx_type: TxType) -> ScanKind {
    if matches!(size.square_up(), TxSize::Tx64x64) || matches!(tx_type, TxType::Identity) {
        return ScanKind::Default;
    }
    match tx_type {
        TxType::VerticalDct | TxType::VerticalAdst | TxType::VerticalFlipAdst => ScanKind::Row,
        TxType::HorizontalDct | TxType::HorizontalAdst | TxType::HorizontalFlipAdst => {
            ScanKind::Column
        }
        _ => ScanKind::Default,
    }
}

pub fn write_directional_scan(
    size: TxSize,
    kind: ScanKind,
    output: &mut [u16],
) -> Result<usize, Error> {
    if kind == ScanKind::Default {
        return Err(Error::InvalidObu);
    }
    let effective = effective_scan_size(size);
    if !matches!(
        effective,
        TxSize::Tx4x4
            | TxSize::Tx4x8
            | TxSize::Tx8x4
            | TxSize::Tx8x8
            | TxSize::Tx8x16
            | TxSize::Tx16x8
            | TxSize::Tx16x16
            | TxSize::Tx4x16
            | TxSize::Tx16x4
    ) {
        return Err(Error::InvalidObu);
    }
    let (width, height) = effective.dimensions();
    let width = usize::from(width);
    let height = usize::from(height);
    let count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
    if output.len() < count {
        return Err(Error::InvalidObu);
    }
    let mut index = 0usize;
    match kind {
        ScanKind::Row => {
            for row in 0..height {
                for column in 0..width {
                    output[index] =
                        u16::try_from(row * width + column).map_err(|_| Error::LimitExceeded)?;
                    index += 1;
                }
            }
        }
        ScanKind::Column => {
            for column in 0..width {
                for row in 0..height {
                    output[index] =
                        u16::try_from(row * width + column).map_err(|_| Error::LimitExceeded)?;
                    index += 1;
                }
            }
        }
        ScanKind::Default => return Err(Error::InvalidObu),
    }
    Ok(count)
}

pub fn write_default_scan(size: TxSize, output: &mut [u16]) -> Result<usize, Error> {
    let effective = effective_scan_size(size);
    let (width, height) = effective.dimensions();
    write_rectangular_zigzag(usize::from(width), usize::from(height), output)
}

fn write_rectangular_zigzag(
    width: usize,
    height: usize,
    output: &mut [u16],
) -> Result<usize, Error> {
    let count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
    if width == 0 || height == 0 || width > 32 || height > 32 || output.len() < count {
        return Err(Error::InvalidObu);
    }
    let mut index = 0usize;
    for diagonal in 0..width + height - 1 {
        let first_row = diagonal.saturating_sub(width - 1);
        let last_row = diagonal.min(height - 1);
        if width > height || (width == height && diagonal % 2 == 0) {
            for row in (first_row..=last_row).rev() {
                let column = diagonal - row;
                output[index] =
                    u16::try_from(row * width + column).map_err(|_| Error::LimitExceeded)?;
                index += 1;
            }
        } else {
            for row in first_row..=last_row {
                let column = diagonal - row;
                output[index] =
                    u16::try_from(row * width + column).map_err(|_| Error::LimitExceeded)?;
                index += 1;
            }
        }
    }
    Ok(count)
}

pub fn write_coefficient_scan(
    size: TxSize,
    tx_type: TxType,
    output: &mut [u16],
) -> Result<usize, Error> {
    match coefficient_scan_kind(size, tx_type) {
        ScanKind::Default => write_default_scan(size, output),
        kind @ (ScanKind::Row | ScanKind::Column) => write_directional_scan(size, kind, output),
    }
}

pub fn luma_txb_skip_context(
    above_levels: &[u8],
    left_levels: &[u8],
    residual_dimensions: (u8, u8),
    transform_size: TxSize,
) -> u8 {
    if residual_dimensions == transform_size.dimensions() {
        return 0;
    }
    let above = above_levels.iter().copied().max().unwrap_or(0);
    let left = left_levels.iter().copied().max().unwrap_or(0);
    if above == 0 && left == 0 {
        1
    } else if above == 0 || left == 0 {
        2 + u8::from(above.max(left) > 3)
    } else if above.max(left) <= 3 {
        4
    } else if above.min(left) <= 3 {
        5
    } else {
        6
    }
}

pub fn chroma_txb_skip_context(
    above_levels: &[u8],
    above_dc: &[u8],
    left_levels: &[u8],
    left_dc: &[u8],
    residual_area_larger_than_transform: bool,
) -> u8 {
    let above = above_levels.iter().chain(above_dc).any(|&value| value != 0);
    let left = left_levels.iter().chain(left_dc).any(|&value| value != 0);
    7 + u8::from(above) + u8::from(left) + 3 * u8::from(residual_area_larger_than_transform)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TxClass {
    TwoDimensional,
    Vertical,
    Horizontal,
}

pub const fn tx_class(tx_type: TxType) -> TxClass {
    match tx_type {
        TxType::VerticalDct | TxType::VerticalAdst | TxType::VerticalFlipAdst => TxClass::Vertical,
        TxType::HorizontalDct | TxType::HorizontalAdst | TxType::HorizontalFlipAdst => {
            TxClass::Horizontal
        }
        _ => TxClass::TwoDimensional,
    }
}

const fn adjusted_dimensions(size: TxSize) -> (usize, usize) {
    let (width, height) = size.dimensions();
    (
        if width > 32 { 32 } else { width } as usize,
        if height > 32 { 32 } else { height } as usize,
    )
}

const COEFF_BASE_CTX_OFFSET: [[[u8; 5]; 5]; 19] = [
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
];

pub fn coefficient_base_eob_context(coefficient_index: usize, count: usize) -> Result<u8, Error> {
    if count == 0 || coefficient_index >= count {
        return Err(Error::InvalidObu);
    }
    Ok(if coefficient_index == 0 {
        0
    } else if coefficient_index <= count / 8 {
        1
    } else if coefficient_index <= count / 4 {
        2
    } else {
        3
    })
}

/// Derives the normative `coeff_base` context from the five forward scan
/// neighbors and the transform-size position table.
pub fn coefficient_base_context(
    quantized: &[i32],
    size: TxSize,
    tx_type: TxType,
    position: usize,
) -> Result<u8, Error> {
    const OFFSETS: [[(usize, usize); 5]; 3] = [
        [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
        [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
    ];
    let (width, height) = adjusted_dimensions(size);
    let count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
    if quantized.len() < count || position >= count {
        return Err(Error::InvalidObu);
    }
    let class = tx_class(tx_type);
    let class_index = match class {
        TxClass::TwoDimensional => 0,
        // Section 3 assigns TX_CLASS_HORIZ=1 and TX_CLASS_VERT=2; the
        // normative neighbor tables use that numeric order.
        TxClass::Horizontal => 1,
        TxClass::Vertical => 2,
    };
    let row = position / width;
    let column = position % width;
    let mut magnitude = 0u32;
    for (row_offset, column_offset) in OFFSETS[class_index] {
        let reference_row = row + row_offset;
        let reference_column = column + column_offset;
        if reference_row < height && reference_column < width {
            magnitude = magnitude.saturating_add(
                quantized[reference_row * width + reference_column]
                    .unsigned_abs()
                    .min(3),
            );
        }
    }
    let magnitude_context = ((magnitude + 1) >> 1).min(4) as u8;
    if class == TxClass::TwoDimensional {
        if row == 0 && column == 0 {
            return Ok(0);
        }
        let table = COEFF_BASE_CTX_OFFSET
            .get(size as usize)
            .ok_or(Error::InvalidObu)?;
        Ok(magnitude_context + table[row.min(4)][column.min(4)])
    } else {
        let index = if class == TxClass::Vertical {
            row
        } else {
            column
        };
        Ok(magnitude_context + [26, 31, 36][index.min(2)])
    }
}

/// Derives the `coeff_br` magnitude/position context from already-decoded
/// neighboring coefficients in the current transform block.
pub fn coefficient_br_context(
    quantized: &[i32],
    size: TxSize,
    tx_type: TxType,
    position: usize,
) -> Result<u8, Error> {
    const OFFSETS: [[(usize, usize); 3]; 3] = [
        [(0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (0, 2)],
        [(0, 1), (1, 0), (2, 0)],
    ];
    let (width, height) = adjusted_dimensions(size);
    let coefficient_count = width.checked_mul(height).ok_or(Error::LimitExceeded)?;
    if quantized.len() < coefficient_count || position >= coefficient_count {
        return Err(Error::InvalidObu);
    }
    let class = tx_class(tx_type);
    let class_index = match class {
        TxClass::TwoDimensional => 0,
        TxClass::Horizontal => 1,
        TxClass::Vertical => 2,
    };
    let row = position / width;
    let column = position % width;
    let mut magnitude = 0u32;
    for (row_offset, column_offset) in OFFSETS[class_index] {
        let reference_row = row + row_offset;
        let reference_column = column + column_offset;
        if reference_row < height && reference_column < width {
            magnitude = magnitude.saturating_add(
                quantized[reference_row * width + reference_column]
                    .unsigned_abs()
                    .min(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1),
            );
        }
    }
    let magnitude_context = ((magnitude + 1) >> 1).min(6) as u8;
    let position_offset = if position == 0 {
        0
    } else {
        match class {
            TxClass::TwoDimensional if row < 2 && column < 2 => 7,
            TxClass::Horizontal if column == 0 => 7,
            TxClass::Vertical if row == 0 => 7,
            _ => 14,
        }
    };
    Ok(magnitude_context + position_offset)
}

/// Reduces neighboring DC sign categories (0 none, 1 negative, 2 positive)
/// to the normative three-way DC sign context.
pub fn dc_sign_context(above: &[u8], left: &[u8]) -> Result<u8, Error> {
    let mut score = 0i32;
    for &sign in above.iter().chain(left) {
        score += match sign {
            0 => 0,
            1 => -1,
            2 => 1,
            _ => return Err(Error::InvalidObu),
        };
    }
    Ok(if score < 0 {
        1
    } else if score > 0 {
        2
    } else {
        0
    })
}

pub fn update_coefficient_contexts(
    above_levels: &mut [u8],
    left_levels: &mut [u8],
    above_dc: &mut [u8],
    left_dc: &mut [u8],
    cumulative_level: u32,
    dc_category: u8,
) -> Result<(), Error> {
    if dc_category > 2 {
        return Err(Error::InvalidObu);
    }
    let level = cumulative_level.min(63) as u8;
    above_levels.fill(level);
    left_levels.fill(level);
    above_dc.fill(dc_category);
    left_dc.fill(dc_category);
    Ok(())
}

pub fn segmented_eob_limit(size: TxSize) -> u16 {
    if matches!(size, TxSize::Tx16x64 | TxSize::Tx64x16) {
        512
    } else {
        let (width, height) = size.dimensions();
        u16::from(width).saturating_mul(u16::from(height)).min(1024)
    }
}

/// Converts the decoded EOB point and its extra-bit value into an EOB count.
pub fn eob_from_point(point: u8, extra: u16, limit: u16) -> Result<u16, Error> {
    if point == 0 || limit == 0 || point > 11 {
        return Err(Error::InvalidObu);
    }
    let value = if point < 2 {
        if extra != 0 {
            return Err(Error::InvalidObu);
        }
        u16::from(point)
    } else {
        let base = (1u16 << (point - 2)) + 1;
        let extra_values = if point < 3 { 1 } else { 1u16 << (point - 2) };
        if extra >= extra_values {
            return Err(Error::InvalidObu);
        }
        base.checked_add(extra).ok_or(Error::LimitExceeded)?
    };
    if value == 0 || value > limit {
        return Err(Error::InvalidObu);
    }
    Ok(value)
}

/// Number of EOB suffix bits carried after an EOB point. The most significant
/// suffix bit is arithmetic coded; any remaining bits are equiprobable.
pub const fn eob_extra_bit_count(point: u8) -> u8 {
    point.saturating_sub(2)
}

/// Reads the complete end-of-block value described by section 5.11.39.
pub fn decode_eob(
    decoder: &mut SymbolDecoder<'_>,
    point: u8,
    extra_cdf: &mut [u16; 3],
    limit: u16,
) -> Result<u16, Error> {
    let bit_count = eob_extra_bit_count(point);
    if bit_count == 0 {
        return eob_from_point(point, 0, limit);
    }
    let top_bit = decoder.read_symbol(extra_cdf)? != 0;
    decode_eob_suffix(decoder, point, top_bit, limit)
}

/// Completes EOB reconstruction after its adaptive most-significant suffix
/// bit has been read from the tile's EOB-extra CDF bank.
pub fn decode_eob_suffix(
    decoder: &mut SymbolDecoder<'_>,
    point: u8,
    top_bit: bool,
    limit: u16,
) -> Result<u16, Error> {
    let bit_count = eob_extra_bit_count(point);
    if bit_count == 0 {
        return eob_from_point(point, 0, limit);
    }
    let mut extra = u16::from(top_bit);
    for _ in 1..bit_count {
        let bit = u16::from(decoder.read_bool()?);
        extra = extra
            .checked_shl(1)
            .and_then(|value| value.checked_add(bit))
            .ok_or(Error::LimitExceeded)?;
    }
    eob_from_point(point, extra, limit)
}

/// Reads the exponential-Golomb suffix used beyond the base coefficient range.
pub fn read_golomb_value(decoder: &mut SymbolDecoder<'_>) -> Result<u32, Error> {
    let mut length = 0u8;
    while !decoder.read_bool()? {
        length = length.checked_add(1).ok_or(Error::LimitExceeded)?;
        if length > 20 {
            return Err(Error::InvalidObu);
        }
    }
    let mut value = 1u32;
    for _ in 0..length {
        let bit = u32::from(decoder.read_bool()?);
        value = value
            .checked_shl(1)
            .and_then(|value| value.checked_add(bit))
            .ok_or(Error::LimitExceeded)?;
    }
    Ok(value)
}

pub fn extended_coefficient_level(
    decoder: &mut SymbolDecoder<'_>,
    base_level: u32,
) -> Result<u32, Error> {
    let threshold = NUM_BASE_LEVELS + COEFF_BASE_RANGE;
    if base_level <= threshold {
        Ok(base_level)
    } else {
        read_golomb_value(decoder)?
            .checked_add(threshold)
            .ok_or(Error::LimitExceeded)
    }
}

const fn normalized_coefficient_level(level: u32) -> u32 {
    level & 0x000f_ffff
}

pub trait CoefficientSymbolReader {
    fn read_all_zero(&mut self, decoder: &mut SymbolDecoder<'_>) -> Result<bool, Error>;
    fn read_tx_type(
        &mut self,
        _decoder: &mut SymbolDecoder<'_>,
        configured: TxType,
    ) -> Result<TxType, Error> {
        Ok(configured)
    }
    fn read_eob_point(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        tx_type: TxType,
    ) -> Result<u8, Error>;
    fn read_eob_extra(&mut self, decoder: &mut SymbolDecoder<'_>, point: u8)
    -> Result<bool, Error>;
    fn read_base_eob(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error>;
    fn read_base(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error>;
    fn read_br(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error>;
    fn read_dc_sign(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8)
    -> Result<bool, Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientBlockConfig {
    pub size: TxSize,
    pub tx_type: TxType,
    pub dc_sign_context: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientBlockResult {
    pub eob: u16,
    pub cumulative_level: u8,
    pub dc_category: u8,
    pub tx_type: TxType,
}

impl Default for CoefficientBlockResult {
    fn default() -> Self {
        Self {
            eob: 0,
            cumulative_level: 0,
            dc_category: 0,
            tx_type: TxType::DctDct,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedCoefficientContexts {
    pub txb_skip: u8,
    pub dc_sign: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoefficientContextConfig {
    pub plane: usize,
    pub x4: u32,
    pub y4: u32,
    pub size: TxSize,
    pub residual_dimensions: (u8, u8),
    pub tile_start: (u32, u32),
}

struct CoefficientContextSlices<'a> {
    above_levels: &'a [u8],
    left_levels: &'a [u8],
    above_dc: &'a [u8],
    left_dc: &'a [u8],
}

/// Persistent above/left coefficient state for one tile. Coordinates and
/// extents are expressed in 4x4 units in the selected plane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientContexts {
    above_levels: [Vector<u8>; 3],
    left_levels: [Vector<u8>; 3],
    above_dc: [Vector<u8>; 3],
    left_dc: [Vector<u8>; 3],
}

impl CoefficientContexts {
    pub fn new(
        mi_columns: u32,
        mi_rows: u32,
        subsampling_x: bool,
        subsampling_y: bool,
        monochrome: bool,
    ) -> Result<Self, Error> {
        let chroma_columns = if subsampling_x {
            mi_columns.div_ceil(2)
        } else {
            mi_columns
        };
        let chroma_rows = if subsampling_y {
            mi_rows.div_ceil(2)
        } else {
            mi_rows
        };
        let columns = [mi_columns, chroma_columns, chroma_columns];
        let rows = [mi_rows, chroma_rows, chroma_rows];
        let mut result = Self {
            above_levels: core::array::from_fn(|_| Vector::new()),
            left_levels: core::array::from_fn(|_| Vector::new()),
            above_dc: core::array::from_fn(|_| Vector::new()),
            left_dc: core::array::from_fn(|_| Vector::new()),
        };
        for plane in 0..3 {
            if monochrome && plane != 0 {
                continue;
            }
            let column_count = usize::try_from(columns[plane]).map_err(|_| Error::LimitExceeded)?;
            let row_count = usize::try_from(rows[plane]).map_err(|_| Error::LimitExceeded)?;
            result.above_levels[plane]
                .try_resize(column_count, 0)
                .map_err(|_| Error::LimitExceeded)?;
            result.above_dc[plane]
                .try_resize(column_count, 0)
                .map_err(|_| Error::LimitExceeded)?;
            result.left_levels[plane]
                .try_resize(row_count, 0)
                .map_err(|_| Error::LimitExceeded)?;
            result.left_dc[plane]
                .try_resize(row_count, 0)
                .map_err(|_| Error::LimitExceeded)?;
        }
        Ok(result)
    }

    pub fn derive(
        &self,
        config: CoefficientContextConfig,
    ) -> Result<DerivedCoefficientContexts, Error> {
        let slices = self.context_slices(config.plane, config.x4, config.y4, config.size)?;
        let empty = &[][..];
        let available_above_levels = if config.y4 == config.tile_start.1 {
            empty
        } else {
            slices.above_levels
        };
        let available_above_dc = if config.y4 == config.tile_start.1 {
            empty
        } else {
            slices.above_dc
        };
        let available_left_levels = if config.x4 == config.tile_start.0 {
            empty
        } else {
            slices.left_levels
        };
        let available_left_dc = if config.x4 == config.tile_start.0 {
            empty
        } else {
            slices.left_dc
        };
        let residual_area_larger = u16::from(config.residual_dimensions.0)
            .saturating_mul(u16::from(config.residual_dimensions.1))
            > u16::from(config.size.dimensions().0)
                .saturating_mul(u16::from(config.size.dimensions().1));
        Ok(DerivedCoefficientContexts {
            txb_skip: if config.plane == 0 {
                luma_txb_skip_context(
                    available_above_levels,
                    available_left_levels,
                    config.residual_dimensions,
                    config.size,
                )
            } else {
                chroma_txb_skip_context(
                    available_above_levels,
                    available_above_dc,
                    available_left_levels,
                    available_left_dc,
                    residual_area_larger,
                )
            },
            dc_sign: dc_sign_context(available_above_dc, available_left_dc)?,
        })
    }

    pub fn update(
        &mut self,
        plane: usize,
        x4: u32,
        y4: u32,
        size: TxSize,
        result: CoefficientBlockResult,
    ) -> Result<(), Error> {
        let (width4, height4) = transform_context_dimensions(size);
        let x = usize::try_from(x4).map_err(|_| Error::LimitExceeded)?;
        let y = usize::try_from(y4).map_err(|_| Error::LimitExceeded)?;
        let above_range = clipped_range(
            x,
            width4,
            self.above_levels.get(plane).ok_or(Error::InvalidObu)?.len(),
        )?;
        let left_range = clipped_range(
            y,
            height4,
            self.left_levels.get(plane).ok_or(Error::InvalidObu)?.len(),
        )?;
        let level = result.cumulative_level.min(63);
        if result.dc_category > 2 {
            return Err(Error::InvalidObu);
        }
        self.above_levels[plane][above_range.clone()].fill(level);
        self.above_dc[plane][above_range].fill(result.dc_category);
        self.left_levels[plane][left_range.clone()].fill(level);
        self.left_dc[plane][left_range].fill(result.dc_category);
        Ok(())
    }

    fn context_slices(
        &self,
        plane: usize,
        x4: u32,
        y4: u32,
        size: TxSize,
    ) -> Result<CoefficientContextSlices<'_>, Error> {
        let above_levels = self.above_levels.get(plane).ok_or(Error::InvalidObu)?;
        let left_levels = self.left_levels.get(plane).ok_or(Error::InvalidObu)?;
        if above_levels.is_empty() || left_levels.is_empty() {
            return Err(Error::InvalidObu);
        }
        let (width4, height4) = transform_context_dimensions(size);
        let x = usize::try_from(x4).map_err(|_| Error::LimitExceeded)?;
        let y = usize::try_from(y4).map_err(|_| Error::LimitExceeded)?;
        let above_range = clipped_range(x, width4, above_levels.len())?;
        let left_range = clipped_range(y, height4, left_levels.len())?;
        Ok(CoefficientContextSlices {
            above_levels: &above_levels[above_range.clone()],
            left_levels: &left_levels[left_range.clone()],
            above_dc: &self.above_dc[plane][above_range],
            left_dc: &self.left_dc[plane][left_range],
        })
    }
}

fn transform_context_dimensions(size: TxSize) -> (usize, usize) {
    let (width, height) = size.dimensions();
    (
        usize::from(width).div_ceil(4),
        usize::from(height).div_ceil(4),
    )
}

fn clipped_range(
    start: usize,
    length: usize,
    limit: usize,
) -> Result<core::ops::Range<usize>, Error> {
    let end = start.checked_add(length).ok_or(Error::LimitExceeded)?;
    if start >= limit {
        return Err(Error::InvalidObu);
    }
    Ok(start..end.min(limit))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileCoefficientConfig {
    pub block: CoefficientBlockConfig,
    pub base_q_index: u8,
    pub chroma: bool,
    pub txb_skip_context: u8,
    pub tx_type_selection: Option<TxTypeSelection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TxTypeSelection {
    Intra { reduced_tx_set: bool, direction: u8 },
    Inter { reduced_tx_set: bool },
}

const fn coefficient_stage(error: Error, stage: CoefficientStage) -> Error {
    match error {
        Error::InvalidObu => Error::InvalidCoefficientStage(stage),
        other => other,
    }
}

struct TileCoefficientReader<'a> {
    cdfs: &'a mut TileCdfs,
    config: TileCoefficientConfig,
}

impl CoefficientSymbolReader for TileCoefficientReader<'_> {
    fn read_all_zero(&mut self, decoder: &mut SymbolDecoder<'_>) -> Result<bool, Error> {
        self.cdfs.read_transform_block_skip(
            decoder,
            self.config.base_q_index,
            self.config.block.size,
            self.config.txb_skip_context,
        )
    }

    fn read_tx_type(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        configured: TxType,
    ) -> Result<TxType, Error> {
        match self.config.tx_type_selection {
            None => Ok(configured),
            Some(TxTypeSelection::Intra {
                reduced_tx_set,
                direction,
            }) => self.cdfs.read_intra_tx_type(
                decoder,
                self.config.block.size,
                reduced_tx_set,
                direction,
            ),
            Some(TxTypeSelection::Inter { reduced_tx_set }) => {
                self.cdfs
                    .read_inter_tx_type(decoder, self.config.block.size, reduced_tx_set)
            }
        }
    }

    fn read_eob_point(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        tx_type: TxType,
    ) -> Result<u8, Error> {
        self.cdfs.read_eob_point(
            decoder,
            self.config.base_q_index,
            self.config.chroma,
            tx_type,
            self.config.block.size,
        )
    }

    fn read_eob_extra(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        point: u8,
    ) -> Result<bool, Error> {
        self.cdfs.read_eob_extra(
            decoder,
            self.config.base_q_index,
            self.config.block.size,
            self.config.chroma,
            point,
        )
    }

    fn read_base_eob(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error> {
        self.cdfs.read_coefficient_base_eob(
            decoder,
            self.config.base_q_index,
            self.config.block.size,
            self.config.chroma,
            context,
        )
    }

    fn read_base(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error> {
        self.cdfs.read_coefficient_base(
            decoder,
            self.config.base_q_index,
            self.config.block.size,
            self.config.chroma,
            context,
        )
    }

    fn read_br(&mut self, decoder: &mut SymbolDecoder<'_>, context: u8) -> Result<u8, Error> {
        self.cdfs.read_coefficient_br(
            decoder,
            self.config.base_q_index,
            self.config.block.size,
            self.config.chroma,
            context,
        )
    }

    fn read_dc_sign(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        context: u8,
    ) -> Result<bool, Error> {
        self.cdfs.read_dc_sign(
            decoder,
            self.config.base_q_index,
            self.config.chroma,
            context,
        )
    }
}

pub fn decode_tile_coefficient_block(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfs,
    config: TileCoefficientConfig,
    quantized: &mut [i32],
) -> Result<CoefficientBlockResult, Error> {
    let mut reader = TileCoefficientReader { cdfs, config };
    decode_coefficient_block(decoder, &mut reader, config.block, quantized)
}

pub fn decode_coefficient_block<R: CoefficientSymbolReader>(
    decoder: &mut SymbolDecoder<'_>,
    reader: &mut R,
    config: CoefficientBlockConfig,
    quantized: &mut [i32],
) -> Result<CoefficientBlockResult, Error> {
    let limit = segmented_eob_limit(config.size);
    let count = usize::from(limit);
    if quantized.len() < count {
        return Err(Error::InvalidObu);
    }
    quantized[..count].fill(0);
    if reader
        .read_all_zero(decoder)
        .map_err(|error| coefficient_stage(error, CoefficientStage::Skip))?
    {
        return Ok(CoefficientBlockResult::default());
    }
    let tx_type = reader
        .read_tx_type(decoder, config.tx_type)
        .map_err(|error| coefficient_stage(error, CoefficientStage::TransformType))?;
    let mut scan_storage = [0u16; 1024];
    let scan_count = write_coefficient_scan(config.size, tx_type, &mut scan_storage)?;
    if scan_count != count {
        return Err(Error::InvalidObu);
    }
    let scan = &scan_storage[..scan_count];
    let point = reader
        .read_eob_point(decoder, tx_type)
        .map_err(|error| coefficient_stage(error, CoefficientStage::EndOfBlock))?;
    let eob = if point < 3 {
        eob_from_point(point, 0, limit)?
    } else {
        let top_bit = reader
            .read_eob_extra(decoder, point)
            .map_err(|error| coefficient_stage(error, CoefficientStage::EndOfBlock))?;
        decode_eob_suffix(decoder, point, top_bit, limit)
            .map_err(|error| coefficient_stage(error, CoefficientStage::EndOfBlock))?
    };
    for coefficient in (0..usize::from(eob)).rev() {
        let position = usize::from(*scan.get(coefficient).ok_or(Error::InvalidObu)?);
        if position >= count {
            return Err(Error::InvalidObu);
        }
        let mut level = if coefficient + 1 == usize::from(eob) {
            let context = coefficient_base_eob_context(coefficient, count)?;
            u32::from(
                reader
                    .read_base_eob(decoder, context)
                    .map_err(|error| match error {
                        Error::InvalidObu => Error::InvalidCoefficientPosition {
                            eob,
                            coefficient: u16::try_from(coefficient).unwrap_or(u16::MAX),
                        },
                        other => other,
                    })?,
            )
            .checked_add(1)
            .ok_or(Error::LimitExceeded)?
        } else {
            let context =
                coefficient_base_context(&quantized[..count], config.size, tx_type, position)?;
            u32::from(
                reader
                    .read_base(decoder, context)
                    .map_err(|error| match error {
                        Error::InvalidObu => Error::InvalidCoefficientPosition {
                            eob,
                            coefficient: u16::try_from(coefficient).unwrap_or(u16::MAX),
                        },
                        other => other,
                    })?,
            )
        };
        if level > NUM_BASE_LEVELS {
            for _ in 0..COEFF_BASE_RANGE / 3 {
                let context =
                    coefficient_br_context(&quantized[..count], config.size, tx_type, position)?;
                let extension = u32::from(
                    reader
                        .read_br(decoder, context)
                        .map_err(|error| coefficient_stage(error, CoefficientStage::BaseRange))?,
                );
                if extension > 3 {
                    return Err(Error::InvalidObu);
                }
                level = level.checked_add(extension).ok_or(Error::LimitExceeded)?;
                if extension < 3 {
                    break;
                }
            }
        }
        quantized[position] = i32::try_from(level).map_err(|_| Error::LimitExceeded)?;
    }
    let mut cumulative_level = 0u32;
    let mut dc_category = 0u8;
    for (coefficient, &scan_position) in scan.iter().take(usize::from(eob)).enumerate() {
        let position = usize::from(scan_position);
        let mut level = quantized[position].unsigned_abs();
        if level == 0 {
            continue;
        }
        let negative = if coefficient == 0 {
            reader
                .read_dc_sign(decoder, config.dc_sign_context)
                .map_err(|error| coefficient_stage(error, CoefficientStage::Sign))?
        } else {
            decoder
                .read_bool()
                .map_err(|error| coefficient_stage(error, CoefficientStage::Sign))?
        };
        if level > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
            level = read_golomb_value(decoder)
                .map_err(|error| coefficient_stage(error, CoefficientStage::BaseRange))?
                .checked_add(NUM_BASE_LEVELS + COEFF_BASE_RANGE)
                .ok_or(Error::LimitExceeded)?;
        }
        level = normalized_coefficient_level(level);
        if position == 0 && level != 0 {
            dc_category = if negative { 1 } else { 2 };
        }
        cumulative_level = cumulative_level.saturating_add(level);
        let signed = i32::try_from(level).map_err(|_| Error::LimitExceeded)?;
        quantized[position] = if negative { -signed } else { signed };
    }
    Ok(CoefficientBlockResult {
        eob,
        cumulative_level: cumulative_level.min(63) as u8,
        dc_category,
        tx_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy_encoder::SymbolEncoder;

    struct FixedCoefficientSymbols {
        all_zero: bool,
        point: u8,
        base_eob: u8,
        negative_dc: bool,
    }

    #[test]
    fn golomb_suffix_uses_normative_one_based_code_values() {
        for (value, bits) in [
            (1, &[true][..]),
            (2, &[false, true, false][..]),
            (3, &[false, true, true][..]),
            (4, &[false, false, true, false, false][..]),
        ] {
            let mut encoder = SymbolEncoder::new(true);
            for &bit in bits {
                encoder.write_bool(bit).unwrap();
            }
            let bytes = encoder.finish().unwrap();
            let mut decoder = SymbolDecoder::new(&bytes, true).unwrap();
            assert_eq!(read_golomb_value(&mut decoder), Ok(value));
        }
    }

    #[test]
    fn escaped_coefficient_levels_are_reduced_to_twenty_bits() {
        assert_eq!(normalized_coefficient_level(0x000f_ffff), 0x000f_ffff);
        assert_eq!(normalized_coefficient_level(0x0010_0000), 0);
        assert_eq!(normalized_coefficient_level(0x0012_3456), 0x0002_3456);
    }

    impl CoefficientSymbolReader for FixedCoefficientSymbols {
        fn read_all_zero(&mut self, _: &mut SymbolDecoder<'_>) -> Result<bool, Error> {
            Ok(self.all_zero)
        }
        fn read_tx_type(&mut self, _: &mut SymbolDecoder<'_>, _: TxType) -> Result<TxType, Error> {
            Ok(TxType::VerticalDct)
        }
        fn read_eob_point(&mut self, _: &mut SymbolDecoder<'_>, _: TxType) -> Result<u8, Error> {
            Ok(self.point)
        }
        fn read_eob_extra(&mut self, _: &mut SymbolDecoder<'_>, _: u8) -> Result<bool, Error> {
            Ok(false)
        }
        fn read_base_eob(&mut self, _: &mut SymbolDecoder<'_>, _: u8) -> Result<u8, Error> {
            Ok(self.base_eob)
        }
        fn read_base(&mut self, _: &mut SymbolDecoder<'_>, _: u8) -> Result<u8, Error> {
            Ok(0)
        }
        fn read_br(&mut self, _: &mut SymbolDecoder<'_>, _: u8) -> Result<u8, Error> {
            Ok(0)
        }
        fn read_dc_sign(&mut self, _: &mut SymbolDecoder<'_>, _: u8) -> Result<bool, Error> {
            Ok(self.negative_dc)
        }
    }

    #[test]
    fn eob_points_cover_consecutive_ranges() {
        assert_eq!(eob_from_point(1, 0, 1024), Ok(1));
        assert_eq!(eob_from_point(2, 0, 1024), Ok(2));
        assert_eq!(eob_from_point(3, 0, 1024), Ok(3));
        assert_eq!(eob_from_point(3, 1, 1024), Ok(4));
        assert_eq!(eob_from_point(4, 0, 1024), Ok(5));
        assert_eq!(eob_from_point(4, 3, 1024), Ok(8));
    }

    #[test]
    fn largest_rectangular_transforms_have_512_coefficient_segments() {
        assert_eq!(segmented_eob_limit(TxSize::Tx16x64), 512);
        assert_eq!(segmented_eob_limit(TxSize::Tx64x16), 512);
        assert_eq!(segmented_eob_limit(TxSize::Tx64x64), 1024);
    }

    #[test]
    fn eob_suffix_width_follows_the_point() {
        assert_eq!(eob_extra_bit_count(1), 0);
        assert_eq!(eob_extra_bit_count(2), 0);
        assert_eq!(eob_extra_bit_count(3), 1);
        assert_eq!(eob_extra_bit_count(11), 9);
    }

    #[test]
    fn coefficient_br_context_uses_transform_class_neighbors() {
        let mut quantized = [0i32; 16];
        quantized[1] = 2;
        quantized[4] = -4;
        quantized[5] = 8;
        assert_eq!(
            coefficient_br_context(&quantized, TxSize::Tx4x4, TxType::DctDct, 0),
            Ok(6)
        );
        assert_eq!(
            coefficient_br_context(&quantized, TxSize::Tx4x4, TxType::VerticalDct, 0),
            Ok(3)
        );
        assert_eq!(
            coefficient_br_context(&quantized, TxSize::Tx4x4, TxType::DctDct, 1),
            Ok(11)
        );
    }

    #[test]
    fn directional_context_tables_follow_normative_class_numbering() {
        let mut quantized = [0i32; 64];
        // Position 55 is two columns to the right of position 53, but there
        // are no rows below it in a 16x4 transform. Horizontal class 1 sees
        // that sample; vertical class 2 does not.
        quantized[55] = 1;
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx16x4, TxType::VerticalFlipAdst, 53,),
            Ok(36)
        );
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx16x4, TxType::HorizontalFlipAdst, 53,),
            Ok(37)
        );
        assert_eq!(
            coefficient_br_context(&quantized, TxSize::Tx16x4, TxType::VerticalFlipAdst, 53,),
            Ok(14)
        );
        assert_eq!(
            coefficient_br_context(&quantized, TxSize::Tx16x4, TxType::HorizontalFlipAdst, 53,),
            Ok(15)
        );
        let zeros = [0i32; 64];
        assert_eq!(
            coefficient_br_context(&zeros, TxSize::Tx16x4, TxType::VerticalFlipAdst, 48,),
            Ok(14)
        );
        assert_eq!(
            coefficient_br_context(&zeros, TxSize::Tx16x4, TxType::HorizontalFlipAdst, 48,),
            Ok(7)
        );
    }

    #[test]
    fn coefficient_base_context_uses_normative_position_tables() {
        let mut quantized = [0i32; 16];
        quantized[1] = 2;
        quantized[4] = -4;
        quantized[5] = 8;
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx4x4, TxType::DctDct, 0),
            Ok(0)
        );
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx4x4, TxType::DctDct, 1),
            Ok(3)
        );
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx4x4, TxType::VerticalDct, 1),
            Ok(28)
        );
        assert_eq!(
            coefficient_base_context(&quantized, TxSize::Tx4x4, TxType::HorizontalDct, 1),
            Ok(33)
        );
    }

    #[test]
    fn coefficient_base_eob_context_partitions_scan_position() {
        assert_eq!(coefficient_base_eob_context(0, 64), Ok(0));
        assert_eq!(coefficient_base_eob_context(8, 64), Ok(1));
        assert_eq!(coefficient_base_eob_context(16, 64), Ok(2));
        assert_eq!(coefficient_base_eob_context(17, 64), Ok(3));
        assert_eq!(coefficient_base_eob_context(64, 64), Err(Error::InvalidObu));
    }

    #[test]
    fn dc_sign_context_accumulates_above_and_left_categories() {
        assert_eq!(dc_sign_context(&[1, 1], &[2]), Ok(1));
        assert_eq!(dc_sign_context(&[2], &[2, 1]), Ok(2));
        assert_eq!(dc_sign_context(&[1], &[2]), Ok(0));
        assert_eq!(dc_sign_context(&[3], &[]), Err(Error::InvalidObu));
    }

    #[test]
    fn transform_block_skip_contexts_follow_plane_rules() {
        assert_eq!(
            luma_txb_skip_context(&[8], &[0], (16, 16), TxSize::Tx8x8),
            3
        );
        assert_eq!(
            luma_txb_skip_context(&[63], &[0], (16, 16), TxSize::Tx8x8),
            3
        );
        assert_eq!(
            luma_txb_skip_context(&[2], &[3], (16, 16), TxSize::Tx8x8),
            4
        );
        assert_eq!(
            luma_txb_skip_context(&[8], &[2], (16, 16), TxSize::Tx8x8),
            5
        );
        assert_eq!(
            luma_txb_skip_context(&[8], &[9], (16, 16), TxSize::Tx8x8),
            6
        );
        assert_eq!(chroma_txb_skip_context(&[1], &[0], &[0], &[0], true), 11);
        assert_eq!(chroma_txb_skip_context(&[0], &[0], &[0], &[2], false), 8);
    }

    #[test]
    fn coefficient_context_updates_clip_levels_and_fill_spans() {
        let mut above_levels = [0; 3];
        let mut left_levels = [0; 2];
        let mut above_dc = [0; 3];
        let mut left_dc = [0; 2];
        update_coefficient_contexts(
            &mut above_levels,
            &mut left_levels,
            &mut above_dc,
            &mut left_dc,
            99,
            2,
        )
        .unwrap();
        assert_eq!(above_levels, [63; 3]);
        assert_eq!(left_levels, [63; 2]);
        assert_eq!(above_dc, [2; 3]);
        assert_eq!(left_dc, [2; 2]);
    }

    #[test]
    fn coefficient_token_loop_decodes_dc_and_all_zero_blocks() {
        let config = CoefficientBlockConfig {
            size: TxSize::Tx4x4,
            tx_type: TxType::DctDct,
            dc_sign_context: 0,
        };
        let mut quantized = [99; 16];
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let result = decode_coefficient_block(
            &mut decoder,
            &mut FixedCoefficientSymbols {
                all_zero: false,
                point: 1,
                base_eob: 0,
                negative_dc: true,
            },
            config,
            &mut quantized,
        )
        .unwrap();
        assert_eq!(result.eob, 1);
        assert_eq!(result.cumulative_level, 1);
        assert_eq!(result.dc_category, 1);
        assert_eq!(result.tx_type, TxType::VerticalDct);
        assert_eq!(quantized[0], -1);
        assert!(quantized[1..].iter().all(|&value| value == 0));

        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let result = decode_coefficient_block(
            &mut decoder,
            &mut FixedCoefficientSymbols {
                all_zero: true,
                point: 0,
                base_eob: 0,
                negative_dc: false,
            },
            config,
            &mut quantized,
        )
        .unwrap();
        assert_eq!(result, CoefficientBlockResult::default());
        assert_eq!(result.tx_type, TxType::DctDct);
        assert!(quantized.iter().all(|&value| value == 0));
    }

    #[test]
    fn scan_family_and_effective_size_follow_transform_rules() {
        assert_eq!(effective_scan_size(TxSize::Tx16x64), TxSize::Tx16x32);
        assert_eq!(effective_scan_size(TxSize::Tx64x32), TxSize::Tx32x32);
        assert_eq!(
            coefficient_scan_kind(TxSize::Tx8x8, TxType::VerticalAdst),
            ScanKind::Row
        );
        assert_eq!(
            coefficient_scan_kind(TxSize::Tx8x8, TxType::HorizontalDct),
            ScanKind::Column
        );
        assert_eq!(
            coefficient_scan_kind(TxSize::Tx8x8, TxType::Identity),
            ScanKind::Default
        );
        assert_eq!(
            coefficient_scan_kind(TxSize::Tx64x64, TxType::VerticalAdst),
            ScanKind::Default
        );
    }

    #[test]
    fn directional_scans_are_row_and_column_major() {
        let mut scan = [0; 32];
        assert_eq!(
            write_directional_scan(TxSize::Tx4x8, ScanKind::Row, &mut scan),
            Ok(32)
        );
        assert_eq!(&scan[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            write_directional_scan(TxSize::Tx4x8, ScanKind::Column, &mut scan),
            Ok(32)
        );
        assert_eq!(&scan[..8], &[0, 4, 8, 12, 16, 20, 24, 28]);
        assert_eq!(
            write_directional_scan(TxSize::Tx32x32, ScanKind::Row, &mut scan),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn fixed_default_scans_match_normative_small_transform_orders() {
        let mut scan = [0; 64];
        assert_eq!(write_default_scan(TxSize::Tx4x4, &mut scan), Ok(16));
        assert_eq!(&scan[..16], &DEFAULT_SCAN_4X4);
        assert_eq!(
            write_coefficient_scan(TxSize::Tx4x8, TxType::DctDct, &mut scan),
            Ok(32)
        );
        assert_eq!(&scan[..32], &DEFAULT_SCAN_4X8);
        assert_eq!(write_default_scan(TxSize::Tx8x4, &mut scan), Ok(32));
        assert_eq!(&scan[..32], &DEFAULT_SCAN_8X4);
        assert_eq!(write_default_scan(TxSize::Tx8x8, &mut scan), Ok(64));
        assert_eq!(&scan[..64], &DEFAULT_SCAN_8X8);
        assert_eq!(
            write_default_scan(TxSize::Tx32x32, &mut scan),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn fixed_default_scans_cover_eight_by_sixteen_rectangles() {
        let mut scan = [0; 128];
        assert_eq!(write_default_scan(TxSize::Tx8x16, &mut scan), Ok(128));
        assert_eq!(scan, DEFAULT_SCAN_8X16);
        assert_eq!(write_default_scan(TxSize::Tx16x8, &mut scan), Ok(128));
        assert_eq!(scan, DEFAULT_SCAN_16X8);
        let mut seen = [false; 128];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn fixed_default_scans_cover_four_by_sixteen_rectangles() {
        let mut scan = [0; 64];
        assert_eq!(write_default_scan(TxSize::Tx4x16, &mut scan), Ok(64));
        assert_eq!(scan, DEFAULT_SCAN_4X16);
        assert_eq!(write_default_scan(TxSize::Tx16x4, &mut scan), Ok(64));
        assert_eq!(scan, DEFAULT_SCAN_16X4);
        let mut seen = [false; 64];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn fixed_default_scan_covers_sixteen_by_sixteen() {
        let mut scan = [0; 256];
        assert_eq!(write_default_scan(TxSize::Tx16x16, &mut scan), Ok(256));
        assert_eq!(scan, DEFAULT_SCAN_16X16);
        let mut seen = [false; 256];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn fixed_default_scans_cover_eight_by_thirty_two_rectangles() {
        let mut scan = [0; 256];
        assert_eq!(write_default_scan(TxSize::Tx8x32, &mut scan), Ok(256));
        assert_eq!(scan, DEFAULT_SCAN_8X32);
        assert_eq!(write_default_scan(TxSize::Tx32x8, &mut scan), Ok(256));
        assert_eq!(scan, DEFAULT_SCAN_32X8);
        let mut seen = [false; 256];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn fixed_default_scan_covers_sixteen_by_thirty_two() {
        let mut scan = [0; 512];
        assert_eq!(write_default_scan(TxSize::Tx16x32, &mut scan), Ok(512));
        assert_eq!(scan, DEFAULT_SCAN_16X32);
        let mut seen = [false; 512];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn fixed_default_scan_covers_thirty_two_by_sixteen() {
        let mut scan = [0; 512];
        assert_eq!(write_default_scan(TxSize::Tx32x16, &mut scan), Ok(512));
        assert_eq!(scan, DEFAULT_SCAN_32X16);
        let mut seen = [false; 512];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
    }

    #[test]
    fn generated_thirty_two_square_scan_matches_normative_order() {
        let mut scan = [0; 1024];
        assert_eq!(write_default_scan(TxSize::Tx32x32, &mut scan), Ok(1024));
        assert_eq!(
            &scan[..16],
            &[0, 1, 32, 64, 33, 2, 3, 34, 65, 96, 128, 97, 66, 35, 4, 5]
        );
        assert_eq!(&scan[1016..], &[989, 1020, 1021, 990, 959, 991, 1022, 1023]);
        let mut seen = [false; 1024];
        for &position in &scan {
            seen[usize::from(position)] = true;
        }
        assert!(seen.iter().all(|&present| present));
        let hash = scan.iter().fold(0xcbf29ce484222325u64, |hash, &value| {
            value.to_le_bytes().into_iter().fold(hash, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            })
        });
        assert_eq!(hash, 0x1253e97474c55cc5);
        assert_eq!(write_default_scan(TxSize::Tx64x64, &mut scan), Ok(1024));
    }

    #[test]
    fn tile_coefficient_reader_checks_normative_context_bounds() {
        let mut decoder = SymbolDecoder::new(&[0; 2], false).unwrap();
        let mut cdfs = TileCdfs::default();
        let mut quantized = [0i32; 16];
        let config = TileCoefficientConfig {
            block: CoefficientBlockConfig {
                size: TxSize::Tx4x4,
                tx_type: TxType::DctDct,
                dc_sign_context: 0,
            },
            base_q_index: 0,
            chroma: false,
            txb_skip_context: 13,
            tx_type_selection: None,
        };
        assert_eq!(
            decode_tile_coefficient_block(&mut decoder, &mut cdfs, config, &mut quantized,),
            Err(Error::InvalidCoefficientStage(CoefficientStage::Skip))
        );
    }

    #[test]
    fn persistent_coefficient_contexts_follow_transform_neighbors() {
        let mut contexts = CoefficientContexts::new(8, 8, true, true, false).unwrap();
        assert_eq!(
            contexts.derive(CoefficientContextConfig {
                plane: 0,
                x4: 0,
                y4: 0,
                size: TxSize::Tx4x4,
                residual_dimensions: (4, 4),
                tile_start: (0, 0),
            }),
            Ok(DerivedCoefficientContexts {
                txb_skip: 0,
                dc_sign: 0,
            })
        );
        contexts
            .update(
                0,
                0,
                0,
                TxSize::Tx8x8,
                CoefficientBlockResult {
                    eob: 3,
                    cumulative_level: 9,
                    dc_category: 1,
                    tx_type: TxType::DctDct,
                },
            )
            .unwrap();
        assert_eq!(
            contexts.derive(CoefficientContextConfig {
                plane: 0,
                x4: 0,
                y4: 2,
                size: TxSize::Tx8x8,
                residual_dimensions: (16, 16),
                tile_start: (0, 0),
            }),
            Ok(DerivedCoefficientContexts {
                txb_skip: 3,
                dc_sign: 1,
            })
        );
    }

    #[test]
    fn monochrome_coefficient_contexts_reject_chroma_planes() {
        let contexts = CoefficientContexts::new(8, 8, true, true, true).unwrap();
        assert_eq!(
            contexts.derive(CoefficientContextConfig {
                plane: 1,
                x4: 0,
                y4: 0,
                size: TxSize::Tx4x4,
                residual_dimensions: (4, 4),
                tile_start: (0, 0),
            }),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn subsampled_coefficient_contexts_cover_odd_plane_edges() {
        let contexts = CoefficientContexts::new(5, 5, true, true, false).unwrap();
        assert_eq!(
            contexts.derive(CoefficientContextConfig {
                plane: 1,
                x4: 2,
                y4: 2,
                size: TxSize::Tx4x4,
                residual_dimensions: (4, 4),
                tile_start: (0, 0),
            }),
            Ok(DerivedCoefficientContexts {
                txb_skip: 7,
                dc_sign: 0,
            })
        );
    }

    #[test]
    fn coefficient_context_spans_clip_at_the_frame_edge() {
        let mut contexts = CoefficientContexts::new(5, 5, false, false, false).unwrap();
        contexts
            .update(
                0,
                4,
                4,
                TxSize::Tx16x16,
                CoefficientBlockResult {
                    eob: 1,
                    cumulative_level: 7,
                    dc_category: 2,
                    tx_type: TxType::DctDct,
                },
            )
            .unwrap();
        assert_eq!(
            contexts.derive(CoefficientContextConfig {
                plane: 0,
                x4: 4,
                y4: 4,
                size: TxSize::Tx16x16,
                residual_dimensions: (16, 16),
                tile_start: (0, 0),
            }),
            Ok(DerivedCoefficientContexts {
                txb_skip: 0,
                dc_sign: 2,
            })
        );
    }
}
