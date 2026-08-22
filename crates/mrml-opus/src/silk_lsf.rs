//! SILK normalized line-spectral-frequency entropy and stabilization.

use crate::{Bandwidth, Error, RangeDecoder, RangeEncoder, silk::SignalType};

const STAGE1: [[u8; 32]; 4] = [
    [
        44, 34, 30, 19, 21, 12, 11, 3, 3, 2, 16, 2, 2, 1, 5, 2, 1, 3, 3, 1, 1, 2, 2, 2, 3, 1, 9, 9,
        2, 7, 2, 1,
    ],
    [
        1, 10, 1, 8, 3, 8, 8, 14, 13, 14, 1, 14, 12, 13, 11, 11, 12, 11, 10, 10, 11, 8, 9, 8, 7, 8,
        1, 1, 6, 1, 6, 5,
    ],
    [
        31, 21, 3, 17, 1, 8, 17, 4, 1, 18, 16, 4, 2, 3, 1, 10, 1, 3, 16, 11, 16, 2, 2, 3, 2, 11, 1,
        4, 9, 8, 7, 3,
    ],
    [
        1, 4, 16, 5, 18, 11, 5, 14, 15, 1, 3, 12, 13, 14, 14, 6, 14, 12, 2, 6, 1, 12, 12, 11, 10,
        3, 10, 5, 1, 1, 1, 3,
    ],
];
const INTERPOLATION: [u8; 5] = [13, 22, 29, 11, 181];
const RESIDUAL_PDF: [[u8; 9]; 16] = [
    [1, 1, 1, 15, 224, 11, 1, 1, 1],
    [1, 1, 2, 34, 183, 32, 1, 1, 1],
    [1, 1, 4, 42, 149, 55, 2, 1, 1],
    [1, 1, 8, 52, 123, 61, 8, 1, 1],
    [1, 3, 16, 53, 101, 74, 6, 1, 1],
    [1, 3, 17, 55, 90, 73, 15, 1, 1],
    [1, 7, 24, 53, 74, 67, 26, 3, 1],
    [1, 1, 18, 63, 78, 58, 30, 6, 1],
    [1, 1, 1, 9, 232, 9, 1, 1, 1],
    [1, 1, 2, 28, 186, 35, 1, 1, 1],
    [1, 1, 3, 42, 152, 53, 2, 1, 1],
    [1, 1, 10, 49, 126, 65, 2, 1, 1],
    [1, 4, 19, 48, 100, 77, 5, 1, 1],
    [1, 1, 14, 54, 100, 72, 12, 1, 1],
    [1, 1, 15, 61, 87, 61, 25, 4, 1],
    [1, 7, 21, 50, 77, 81, 17, 1, 1],
];
const RESIDUAL_EXTENSION: [u8; 7] = [156, 60, 24, 9, 4, 2, 1];
const SELECT_NB_MB: [&[u8; 10]; 32] = [
    b"aaaaaaaaaa",
    b"bdbccbcbbb",
    b"cbbbbbbbbb",
    b"bccccbcbbb",
    b"cddddccccc",
    b"afddccccbb",
    b"accccccccb",
    b"cdgeeefeff",
    b"ceffefegee",
    b"ceehefeffe",
    b"edddcdcccc",
    b"bffgefefff",
    b"chegffffff",
    b"chfffffgfe",
    b"ddfeefefee",
    b"cddffeeeee",
    b"ceegefefff",
    b"cfegfffefe",
    b"chefefefff",
    b"cfeghgfgfe",
    b"dghegffgef",
    b"chgeeefeff",
    b"effeggfgfe",
    b"cffgfgegee",
    b"efffdheffe",
    b"cdeffgeffe",
    b"cdcddecddd",
    b"bbcccccdcc",
    b"effgggfgef",
    b"dffeeeeddc",
    b"cfdhffeefe",
    b"eefefgfgfe",
];
const SELECT_WB: [&[u8; 16]; 32] = [
    b"iiiiiiiiiiiiiiii",
    b"klllllkkkkkjjjil",
    b"knnlpmmnknmnnmll",
    b"ikjkkjjjjjiiiiij",
    b"ionmompnmmmnnmml",
    b"ilnnmllnllllllkm",
    b"iiiiiiiiiiiiiiii",
    b"ikolpknlmnnmllkl",
    b"iokoomnmonmmnlll",
    b"kjiiiiiiiiiiiiii",
    b"ijiiiiiiiiiiiiij",
    b"kklmnlllllllkkjl",
    b"kkllmllllllllkjl",
    b"lmmmommnlnmmnmlm",
    b"iomnmpnkonpmmlnl",
    b"ijijjjjjjjiiiiji",
    b"jonpnmnlmnmmmllm",
    b"jllmmllnkllnnnlm",
    b"kllkkklkjkjkjjjm",
    b"iklnllkkkjjiiiii",
    b"lmlnllkkjjjjjkkm",
    b"kolppmnmnlnllkll",
    b"klnoolnlmmllllkm",
    b"jllmmmmlnnnljjjj",
    b"knloompmmnlmmlll",
    b"iojjiiiiiiiiiiii",
    b"ioolnknnlmmppmmm",
    b"llplnmlllkklllkl",
    b"iijiiikjkjjkkkjj",
    b"ilknllklkjiijiij",
    b"lnnmpnllklkkjiji",
    b"klnlmlllkjkomiii",
];
const PREDICT_NB_MB: [[u8; 9]; 2] = [
    [179, 138, 140, 148, 151, 149, 153, 151, 163],
    [116, 67, 82, 59, 92, 72, 100, 89, 92],
];
const PREDICT_WB: [[u8; 15]; 2] = [
    [
        175, 148, 160, 176, 178, 173, 174, 164, 177, 174, 196, 182, 198, 192, 182,
    ],
    [
        68, 62, 66, 60, 72, 117, 85, 90, 118, 136, 151, 142, 160, 142, 155,
    ],
];
const PRED_SELECT_NB_MB: [&[u8; 9]; 32] = [
    b"ABAAAAAAA",
    b"BAAAAAAAA",
    b"AAAAAAAAA",
    b"BBBAAAABA",
    b"ABAAAAAAA",
    b"ABAAAAAAA",
    b"BABBAAABA",
    b"ABBAABBAA",
    b"AABBABABB",
    b"AABBAABBB",
    b"AAAAAAAAA",
    b"ABABBBBBA",
    b"ABABBBBBA",
    b"ABBBBBBBA",
    b"BABBABBBB",
    b"ABBBBBABA",
    b"AABBABABA",
    b"AABBBABBB",
    b"ABBAABBBA",
    b"AAABBBABA",
    b"ABBAABABA",
    b"ABBAAABBA",
    b"AAAAABBBB",
    b"AABBAAABB",
    b"AAABABBBB",
    b"ABBBBBBBA",
    b"AAAAAAAAA",
    b"AAAAAAAAA",
    b"AABABBABA",
    b"BAABAAAAA",
    b"AAABBABAB",
    b"BABBABBBB",
];
const PRED_SELECT_WB: [&[u8; 15]; 32] = [
    b"CCCCCCCCCCCCCCD",
    b"CCCCCCCCCCCCCCC",
    b"CCDCCDDDCDDDDCC",
    b"CCCCCCCCCCCCDCC",
    b"CDDCDCDDCDDDDDC",
    b"CCDCCCCCCCCCCCC",
    b"DCCCCCCCCCCDCDC",
    b"CDDCCCDCDDDCDCD",
    b"CDCDDCDCDCDDDDD",
    b"CCCCCCCCCCCCCCD",
    b"CDCCCCCCCCCCCCC",
    b"CCDCDDDDDDDCDCC",
    b"CCDCCDCDCDCCDCC",
    b"CCCCDDCDCDDDDCC",
    b"CDCCCDDCDDDCDDD",
    b"CCDDCCCCCCCCDDC",
    b"CDDCDCDDDDDCDCC",
    b"CCDCCCCDCCDDDCC",
    b"CCCCCCCCCCCCCCD",
    b"CCCCCCCCCCCCDCC",
    b"CCCCCCCCCCCCCCC",
    b"CDCDCDDCDCDCDDC",
    b"CCDDDDCDDCCDDCC",
    b"CDDCDCDCDCCCCDC",
    b"CCCDDCDCDDDDDDD",
    b"CCCCCCCCCCCCCCD",
    b"CDDCCCDDCCDDDDD",
    b"CCCCCDCDDDDCDDD",
    b"CCCCCCCCCCCCCCD",
    b"CCCCCCCCCCCCCCD",
    b"DCCCCCCCCCCDCCC",
    b"CCDCCDDDCCDCCDC",
];
const CODEBOOK_NB_MB: [[u8; 10]; 32] = [
    [12, 35, 60, 83, 108, 132, 157, 180, 206, 228],
    [15, 32, 55, 77, 101, 125, 151, 175, 201, 225],
    [19, 42, 66, 89, 114, 137, 162, 184, 209, 230],
    [12, 25, 50, 72, 97, 120, 147, 172, 200, 223],
    [26, 44, 69, 90, 114, 135, 159, 180, 205, 225],
    [13, 22, 53, 80, 106, 130, 156, 180, 205, 228],
    [15, 25, 44, 64, 90, 115, 142, 168, 196, 222],
    [19, 24, 62, 82, 100, 120, 145, 168, 190, 214],
    [22, 31, 50, 79, 103, 120, 151, 170, 203, 227],
    [21, 29, 45, 65, 106, 124, 150, 171, 196, 224],
    [30, 49, 75, 97, 121, 142, 165, 186, 209, 229],
    [19, 25, 52, 70, 93, 116, 143, 166, 192, 219],
    [26, 34, 62, 75, 97, 118, 145, 167, 194, 217],
    [25, 33, 56, 70, 91, 113, 143, 165, 196, 223],
    [21, 34, 51, 72, 97, 117, 145, 171, 196, 222],
    [20, 29, 50, 67, 90, 117, 144, 168, 197, 221],
    [22, 31, 48, 66, 95, 117, 146, 168, 196, 222],
    [24, 33, 51, 77, 116, 134, 158, 180, 200, 224],
    [21, 28, 70, 87, 106, 124, 149, 170, 194, 217],
    [26, 33, 53, 64, 83, 117, 152, 173, 204, 225],
    [27, 34, 65, 95, 108, 129, 155, 174, 210, 225],
    [20, 26, 72, 99, 113, 131, 154, 176, 200, 219],
    [34, 43, 61, 78, 93, 114, 155, 177, 205, 229],
    [23, 29, 54, 97, 124, 138, 163, 179, 209, 229],
    [30, 38, 56, 89, 118, 129, 158, 178, 200, 231],
    [21, 29, 49, 63, 85, 111, 142, 163, 193, 222],
    [27, 48, 77, 103, 133, 158, 179, 196, 215, 232],
    [29, 47, 74, 99, 124, 151, 176, 198, 220, 237],
    [33, 42, 61, 76, 93, 121, 155, 174, 207, 225],
    [29, 53, 87, 112, 136, 154, 170, 188, 208, 227],
    [24, 30, 52, 84, 131, 150, 166, 186, 203, 229],
    [37, 48, 64, 84, 104, 118, 156, 177, 201, 230],
];
const CODEBOOK_WB: [[u8; 16]; 32] = [
    [
        7, 23, 38, 54, 69, 85, 100, 116, 131, 147, 162, 178, 193, 208, 223, 239,
    ],
    [
        13, 25, 41, 55, 69, 83, 98, 112, 127, 142, 157, 171, 187, 203, 220, 236,
    ],
    [
        15, 21, 34, 51, 61, 78, 92, 106, 126, 136, 152, 167, 185, 205, 225, 240,
    ],
    [
        10, 21, 36, 50, 63, 79, 95, 110, 126, 141, 157, 173, 189, 205, 221, 237,
    ],
    [
        17, 20, 37, 51, 59, 78, 89, 107, 123, 134, 150, 164, 184, 205, 224, 240,
    ],
    [
        10, 15, 32, 51, 67, 81, 96, 112, 129, 142, 158, 173, 189, 204, 220, 236,
    ],
    [
        8, 21, 37, 51, 65, 79, 98, 113, 126, 138, 155, 168, 179, 192, 209, 218,
    ],
    [
        12, 15, 34, 55, 63, 78, 87, 108, 118, 131, 148, 167, 185, 203, 219, 236,
    ],
    [
        16, 19, 32, 36, 56, 79, 91, 108, 118, 136, 154, 171, 186, 204, 220, 237,
    ],
    [
        11, 28, 43, 58, 74, 89, 105, 120, 135, 150, 165, 180, 196, 211, 226, 241,
    ],
    [
        6, 16, 33, 46, 60, 75, 92, 107, 123, 137, 156, 169, 185, 199, 214, 225,
    ],
    [
        11, 19, 30, 44, 57, 74, 89, 105, 121, 135, 152, 169, 186, 202, 218, 234,
    ],
    [
        12, 19, 29, 46, 57, 71, 88, 100, 120, 132, 148, 165, 182, 199, 216, 233,
    ],
    [
        17, 23, 35, 46, 56, 77, 92, 106, 123, 134, 152, 167, 185, 204, 222, 237,
    ],
    [
        14, 17, 45, 53, 63, 75, 89, 107, 115, 132, 151, 171, 188, 206, 221, 240,
    ],
    [
        9, 16, 29, 40, 56, 71, 88, 103, 119, 137, 154, 171, 189, 205, 222, 237,
    ],
    [
        16, 19, 36, 48, 57, 76, 87, 105, 118, 132, 150, 167, 185, 202, 218, 236,
    ],
    [
        12, 17, 29, 54, 71, 81, 94, 104, 126, 136, 149, 164, 182, 201, 221, 237,
    ],
    [
        15, 28, 47, 62, 79, 97, 115, 129, 142, 155, 168, 180, 194, 208, 223, 238,
    ],
    [
        8, 14, 30, 45, 62, 78, 94, 111, 127, 143, 159, 175, 192, 207, 223, 239,
    ],
    [
        17, 30, 49, 62, 79, 92, 107, 119, 132, 145, 160, 174, 190, 204, 220, 235,
    ],
    [
        14, 19, 36, 45, 61, 76, 91, 108, 121, 138, 154, 172, 189, 205, 222, 238,
    ],
    [
        12, 18, 31, 45, 60, 76, 91, 107, 123, 138, 154, 171, 187, 204, 221, 236,
    ],
    [
        13, 17, 31, 43, 53, 70, 83, 103, 114, 131, 149, 167, 185, 203, 220, 237,
    ],
    [
        17, 22, 35, 42, 58, 78, 93, 110, 125, 139, 155, 170, 188, 206, 224, 240,
    ],
    [
        8, 15, 34, 50, 67, 83, 99, 115, 131, 146, 162, 178, 193, 209, 224, 239,
    ],
    [
        13, 16, 41, 66, 73, 86, 95, 111, 128, 137, 150, 163, 183, 206, 225, 241,
    ],
    [
        17, 25, 37, 52, 63, 75, 92, 102, 119, 132, 144, 160, 175, 191, 212, 231,
    ],
    [
        19, 31, 49, 65, 83, 100, 117, 133, 147, 161, 174, 187, 200, 213, 227, 242,
    ],
    [
        18, 31, 52, 68, 88, 103, 117, 126, 138, 149, 163, 177, 192, 207, 223, 239,
    ],
    [
        16, 29, 47, 61, 76, 90, 106, 119, 133, 147, 161, 176, 193, 209, 224, 240,
    ],
    [
        15, 21, 35, 50, 61, 73, 86, 97, 110, 119, 129, 141, 175, 198, 218, 237,
    ],
];
const SPACING_NB_MB: [i32; 11] = [250, 3, 6, 3, 3, 3, 4, 3, 3, 3, 461];
const SPACING_WB: [i32; 17] = [100, 3, 40, 3, 3, 3, 5, 14, 14, 10, 11, 3, 8, 9, 7, 3, 347];
const ORDER_NB_MB: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
const ORDER_WB: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
const COS_Q12: [i16; 129] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3997, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3613, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2676, 2599, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1568, 1474, 1380, 1285, 1189, 1093, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0, -101,
    -201, -301, -401, -501, -601, -700, -799, -897, -995, -1093, -1189, -1285, -1380, -1474, -1568,
    -1660, -1751, -1842, -1931, -2019, -2106, -2191, -2276, -2359, -2440, -2520, -2599, -2676,
    -2751, -2824, -2896, -2967, -3035, -3102, -3166, -3229, -3290, -3349, -3406, -3461, -3513,
    -3564, -3613, -3659, -3703, -3745, -3784, -3822, -3857, -3889, -3920, -3948, -3973, -3997,
    -4017, -4036, -4052, -4065, -4076, -4085, -4091, -4095, -4096,
];

pub fn nb_mb_stage1_codebook(index: u8) -> Result<&'static [u8; 10], Error> {
    CODEBOOK_NB_MB
        .get(usize::from(index))
        .ok_or(Error::InvalidPacket)
}

pub fn stage1_codebook(
    bandwidth: Bandwidth,
    index: u8,
    output_q8: &mut [u8],
) -> Result<usize, Error> {
    if wideband(bandwidth)? {
        let vector = CODEBOOK_WB
            .get(usize::from(index))
            .ok_or(Error::InvalidPacket)?;
        if output_q8.len() < 16 {
            return Err(Error::BufferTooSmall);
        }
        output_q8[..16].copy_from_slice(vector);
        Ok(16)
    } else {
        let vector = nb_mb_stage1_codebook(index)?;
        if output_q8.len() < 10 {
            return Err(Error::BufferTooSmall);
        }
        output_q8[..10].copy_from_slice(vector);
        Ok(10)
    }
}

fn inverse_harmonic_weights(codebook_q8: &[u8], output_q9: &mut [u16]) -> Result<(), Error> {
    if output_q9.len() < codebook_q8.len() || !matches!(codebook_q8.len(), 10 | 16) {
        return Err(Error::InvalidFrameSize);
    }
    for coefficient in 0..codebook_q8.len() {
        let previous = if coefficient == 0 {
            0
        } else {
            u32::from(codebook_q8[coefficient - 1])
        };
        let current = u32::from(codebook_q8[coefficient]);
        let next = codebook_q8
            .get(coefficient + 1)
            .map_or(256, |value| u32::from(*value));
        let left = current
            .checked_sub(previous)
            .filter(|&v| v != 0)
            .ok_or(Error::InvalidPacket)?;
        let right = next
            .checked_sub(current)
            .filter(|&v| v != 0)
            .ok_or(Error::InvalidPacket)?;
        let squared_q18 = (1024 / left + 1024 / right) << 16;
        let bits = 32 - squared_q18.leading_zeros();
        if !(8..=32).contains(&bits) {
            return Err(Error::InvalidPacket);
        }
        let fraction = (squared_q18 >> (bits - 8)) & 127;
        let root = (if bits & 1 != 0 { 32_768u32 } else { 46_214u32 }) >> ((32 - bits) >> 1);
        let weight = root + ((213 * fraction * root) >> 16);
        output_q9[coefficient] = u16::try_from(weight).map_err(|_| Error::InvalidPacket)?;
    }
    Ok(())
}

/// Reconstructs and stabilizes a complete stage-1/stage-2 LSF vector in Q15.
pub fn reconstruct(
    bandwidth: Bandwidth,
    stage1: u8,
    indices: &Stage2,
    output_q15: &mut [i16],
) -> Result<usize, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if output_q15.len() < order {
        return Err(Error::BufferTooSmall);
    }
    let mut codebook = [0u8; 16];
    stage1_codebook(bandwidth, stage1, &mut codebook)?;
    let mut residual = [0i32; 16];
    dequantize_stage2(bandwidth, stage1, indices, &mut residual)?;
    let mut weights = [0u16; 16];
    inverse_harmonic_weights(&codebook[..order], &mut weights)?;
    for coefficient in 0..order {
        let correction = (i64::from(residual[coefficient]) << 14) / i64::from(weights[coefficient]);
        let value = (i64::from(codebook[coefficient]) << 7) + correction;
        output_q15[coefficient] =
            i16::try_from(value.clamp(0, 32_767)).map_err(|_| Error::InvalidPacket)?;
    }
    stabilize(&mut output_q15[..order], bandwidth)?;
    Ok(order)
}

fn cosine_q17(lsf_q15: i16) -> Result<i64, Error> {
    if lsf_q15 < 0 {
        return Err(Error::InvalidPacket);
    }
    let value = i32::from(lsf_q15);
    let index = usize::try_from(value >> 8).map_err(|_| Error::InvalidPacket)?;
    let fraction = value & 255;
    let low = i32::from(*COS_Q12.get(index).ok_or(Error::InvalidPacket)?);
    let high = i32::from(*COS_Q12.get(index + 1).ok_or(Error::InvalidPacket)?);
    Ok(i64::from((low * 256 + (high - low) * fraction + 4) >> 3))
}

fn find_polynomial(cosines_q17: &[i64], parity: usize, half: usize) -> [i64; 10] {
    let mut previous = [0i64; 10];
    previous[0] = 1 << 16;
    previous[1] = -cosines_q17[parity];
    for root in 1..half {
        let mut current = [0i64; 10];
        for coefficient in 0..=root + 1 {
            let direct = if coefficient == root + 1 {
                previous[root - 1]
            } else {
                previous[coefficient]
            };
            let delayed = coefficient
                .checked_sub(2)
                .map_or(0, |index| previous[index]);
            let adjacent = if coefficient == 0 {
                0
            } else {
                previous[coefficient - 1]
            };
            current[coefficient] =
                direct + delayed - ((cosines_q17[2 * root + parity] * adjacent + 32_768) >> 16);
        }
        previous = current;
    }
    previous
}

fn bandwidth_expand(coefficients_q17: &mut [i64], chirp_q16: u32) {
    let mut chirp = u64::from(chirp_q16);
    for coefficient in coefficients_q17 {
        *coefficient = (*coefficient * chirp as i64) >> 16;
        chirp = (u64::from(chirp_q16) * chirp + 32_768) >> 16;
    }
}

fn lsf_to_lpc_q17_range_limited(
    lsf_q15: &[i16],
    bandwidth: Bandwidth,
    output_q17: &mut [i64; 16],
) -> Result<usize, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if lsf_q15.len() != order {
        return Err(Error::InvalidFrameSize);
    }
    let ordering: &[usize] = if order == 16 { &ORDER_WB } else { &ORDER_NB_MB };
    let mut cosines = [0i64; 16];
    for (source, &destination) in ordering.iter().enumerate() {
        cosines[destination] = cosine_q17(lsf_q15[source])?;
    }
    let half = order / 2;
    let p = find_polynomial(&cosines, 0, half);
    let q = find_polynomial(&cosines, 1, half);
    let mut coefficients = [0i64; 16];
    for k in 0..half {
        let difference = q[k + 1] - q[k];
        let sum = p[k + 1] + p[k];
        coefficients[k] = -difference - sum;
        coefficients[order - k - 1] = difference - sum;
    }
    for _ in 0..10 {
        let mut maximum_index = 0usize;
        let mut maximum = coefficients[0].unsigned_abs();
        for (index, &coefficient) in coefficients[1..order].iter().enumerate() {
            if coefficient.unsigned_abs() > maximum {
                maximum = coefficient.unsigned_abs();
                maximum_index = index + 1;
            }
        }
        let maximum_q12 = ((maximum + 16) >> 5).min(163_838);
        if maximum_q12 <= 32_767 {
            output_q17[..order].copy_from_slice(&coefficients[..order]);
            return Ok(order);
        }
        let numerator = (maximum_q12 - 32_767) << 14;
        let denominator = (maximum_q12 * (maximum_index as u64 + 1)) >> 2;
        let chirp = 65_470u64
            .checked_sub(numerator / denominator)
            .ok_or(Error::InvalidPacket)?;
        bandwidth_expand(
            &mut coefficients[..order],
            u32::try_from(chirp).map_err(|_| Error::InvalidPacket)?,
        );
    }
    for (source, target) in coefficients[..order].iter_mut().zip(output_q17.iter_mut()) {
        *source = ((source.saturating_add(16) >> 5).clamp(-32_768, 32_767)) << 5;
        *target = *source;
    }
    Ok(order)
}

/// Converts stabilized Q15 LSFs through polynomial construction and the
/// ten-round coefficient range limiter, without prediction-gain limiting.
pub fn lsf_to_lpc_range_limited(
    lsf_q15: &[i16],
    bandwidth: Bandwidth,
    output_q12: &mut [i16],
) -> Result<usize, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if output_q12.len() < order {
        return Err(Error::BufferTooSmall);
    }
    let mut coefficients = [0i64; 16];
    lsf_to_lpc_q17_range_limited(lsf_q15, bandwidth, &mut coefficients)?;
    for (&source, target) in coefficients[..order].iter().zip(output_q12.iter_mut()) {
        *target = i16::try_from((source + 16) >> 5).map_err(|_| Error::InvalidPacket)?;
    }
    Ok(order)
}

fn inverse_prediction_gain_is_stable(coefficients_q17: &[i64]) -> bool {
    let order = coefficients_q17.len();
    let mut current = [0i64; 16];
    let mut dc_response = 0i64;
    for (&source, target) in coefficients_q17.iter().zip(current.iter_mut()) {
        let q12 = (source + 16) >> 5;
        dc_response += q12;
        *target = q12 << 12;
    }
    if dc_response > 4096 {
        return false;
    }
    let mut inverse_gain_q30 = 1i64 << 30;
    for k in (0..order).rev() {
        if current[k].unsigned_abs() > 16_773_022 {
            return false;
        }
        let reflection_q31 = -current[k] << 7;
        let divisor_q30 = (1i64 << 30) - ((reflection_q31 * reflection_q31) >> 32);
        if divisor_q30 <= 0 {
            return false;
        }
        inverse_gain_q30 = ((inverse_gain_q30 * divisor_q30) >> 32) << 2;
        if inverse_gain_q30 < 107_374 {
            return false;
        }
        if k > 0 {
            let bits = 64 - (divisor_q30 as u64).leading_zeros();
            if !(20..=31).contains(&bits) {
                return false;
            }
            let fractional_bits = bits - 16;
            let denominator = divisor_q30 >> (fractional_bits + 1);
            if denominator == 0 {
                return false;
            }
            let reciprocal = ((1i64 << 29) - 1) / denominator;
            let error_q29 =
                (1i64 << 29) - (((divisor_q30 << (15 - fractional_bits)) * reciprocal) >> 16);
            let gain = (reciprocal << 16) + ((error_q29 * reciprocal) >> 13);
            let mut next = current;
            for n in 0..k {
                let numerator =
                    current[n] - ((current[k - n - 1] * reflection_q31 + (1i64 << 30)) >> 31);
                next[n] = (numerator * gain + (1i64 << (bits - 1))) >> bits;
            }
            current = next;
        }
    }
    true
}

/// Complete bit-exact LSF-to-LPC conversion, including the sixteen-round
/// prediction-gain limiter from Section 4.2.7.5.8.
pub fn lsf_to_lpc(
    lsf_q15: &[i16],
    bandwidth: Bandwidth,
    output_q12: &mut [i16],
) -> Result<usize, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if output_q12.len() < order {
        return Err(Error::BufferTooSmall);
    }
    let mut coefficients = [0i64; 16];
    lsf_to_lpc_q17_range_limited(lsf_q15, bandwidth, &mut coefficients)?;
    for round in 0..16 {
        if inverse_prediction_gain_is_stable(&coefficients[..order]) {
            for (&source, target) in coefficients[..order].iter().zip(output_q12.iter_mut()) {
                *target = i16::try_from((source + 16) >> 5).map_err(|_| Error::InvalidPacket)?;
            }
            return Ok(order);
        }
        bandwidth_expand(&mut coefficients[..order], 65_536 - (2u32 << round));
    }
    output_q12[..order].fill(0);
    Ok(order)
}

fn wideband(bandwidth: Bandwidth) -> Result<bool, Error> {
    match bandwidth {
        Bandwidth::Narrow | Bandwidth::Medium => Ok(false),
        Bandwidth::Wide => Ok(true),
        _ => Err(Error::InvalidPacket),
    }
}

fn model(bandwidth: Bandwidth, signal: SignalType) -> Result<&'static [u8; 32], Error> {
    let offset = usize::from(wideband(bandwidth)?) * 2;
    Ok(&STAGE1[offset + usize::from(signal == SignalType::Voiced)])
}

pub fn decode_stage1(
    decoder: &mut RangeDecoder<'_>,
    bandwidth: Bandwidth,
    signal: SignalType,
) -> Result<u8, Error> {
    u8::try_from(decoder.decode_pdf(model(bandwidth, signal)?)?).map_err(|_| Error::InvalidPacket)
}

pub fn encode_stage1(
    encoder: &mut RangeEncoder<'_>,
    bandwidth: Bandwidth,
    signal: SignalType,
    index: u8,
) -> Result<(), Error> {
    if index >= 32 {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(index), model(bandwidth, signal)?)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stage2 {
    pub order: u8,
    pub index: [i8; 16],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LsfIndices {
    pub stage1: u8,
    pub stage2: Stage2,
    /// Present only for 20 ms SILK frames.
    pub interpolation_q2: Option<u8>,
}

/// Decodes a complete SILK LSF parameter section and produces stable Q12 LPC
/// coefficients for both halves of the frame.
#[allow(clippy::too_many_arguments)] // Three independently stateful fixed-point outputs avoid hidden allocation.
pub fn decode_lsf(
    decoder: &mut RangeDecoder<'_>,
    bandwidth: Bandwidth,
    signal: SignalType,
    twenty_ms: bool,
    previous_q15: Option<&[i16]>,
    current_q15: &mut [i16],
    first_half_lpc_q12: &mut [i16],
    second_half_lpc_q12: &mut [i16],
) -> Result<LsfIndices, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if current_q15.len() < order
        || first_half_lpc_q12.len() < order
        || second_half_lpc_q12.len() < order
    {
        return Err(Error::BufferTooSmall);
    }
    let stage1 = decode_stage1(decoder, bandwidth, signal)?;
    let stage2 = decode_stage2(decoder, bandwidth, stage1)?;
    reconstruct(bandwidth, stage1, &stage2, current_q15)?;
    let interpolation_q2 = if twenty_ms {
        Some(decode_interpolation(decoder)?)
    } else {
        None
    };
    lsf_to_lpc(&current_q15[..order], bandwidth, second_half_lpc_q12)?;
    let effective = if previous_q15.is_some() {
        interpolation_q2.unwrap_or(4)
    } else {
        4
    };
    if twenty_ms && effective < 4 {
        let previous = previous_q15
            .filter(|value| value.len() >= order)
            .ok_or(Error::InvalidFrameSize)?;
        let mut interpolated = [0i16; 16];
        interpolate(
            &previous[..order],
            &current_q15[..order],
            effective,
            &mut interpolated[..order],
        )?;
        lsf_to_lpc(&interpolated[..order], bandwidth, first_half_lpc_q12)?;
    } else {
        first_half_lpc_q12[..order].copy_from_slice(&second_half_lpc_q12[..order]);
    }
    Ok(LsfIndices {
        stage1,
        stage2,
        interpolation_q2,
    })
}

pub fn encode_lsf(
    encoder: &mut RangeEncoder<'_>,
    bandwidth: Bandwidth,
    signal: SignalType,
    twenty_ms: bool,
    indices: &LsfIndices,
) -> Result<(), Error> {
    if twenty_ms != indices.interpolation_q2.is_some() {
        return Err(Error::InvalidPacket);
    }
    encode_stage1(encoder, bandwidth, signal, indices.stage1)?;
    encode_stage2(encoder, bandwidth, indices.stage1, &indices.stage2)?;
    if let Some(factor) = indices.interpolation_q2 {
        encode_interpolation(encoder, factor)?;
    }
    Ok(())
}

fn selector(bandwidth: Bandwidth, stage1: u8, coefficient: usize) -> Result<usize, Error> {
    if stage1 >= 32 {
        return Err(Error::InvalidPacket);
    }
    if wideband(bandwidth)? {
        SELECT_WB[usize::from(stage1)]
            .get(coefficient)
            .map(|value| 8 + usize::from(*value - b'i'))
            .ok_or(Error::InvalidFrameSize)
    } else {
        SELECT_NB_MB[usize::from(stage1)]
            .get(coefficient)
            .map(|value| usize::from(*value - b'a'))
            .ok_or(Error::InvalidFrameSize)
    }
}

pub fn decode_stage2(
    decoder: &mut RangeDecoder<'_>,
    bandwidth: Bandwidth,
    stage1: u8,
) -> Result<Stage2, Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    let mut result = Stage2 {
        order: order as u8,
        index: [0; 16],
    };
    for coefficient in 0..order {
        let symbol = i8::try_from(
            decoder.decode_pdf(&RESIDUAL_PDF[selector(bandwidth, stage1, coefficient)?])?,
        )
        .map_err(|_| Error::InvalidPacket)?;
        let mut value = symbol - 4;
        if value.abs() == 4 {
            let extension = i8::try_from(decoder.decode_pdf(&RESIDUAL_EXTENSION)?)
                .map_err(|_| Error::InvalidPacket)?;
            value += value.signum() * extension;
        }
        result.index[coefficient] = value;
    }
    Ok(result)
}

pub fn encode_stage2(
    encoder: &mut RangeEncoder<'_>,
    bandwidth: Bandwidth,
    stage1: u8,
    residual: &Stage2,
) -> Result<(), Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if usize::from(residual.order) != order
        || residual.index[order..].iter().any(|&value| value != 0)
    {
        return Err(Error::InvalidFrameSize);
    }
    for (coefficient, &value) in residual.index[..order].iter().enumerate() {
        if !(-10..=10).contains(&value) {
            return Err(Error::InvalidPacket);
        }
        let base = value.clamp(-4, 4);
        encoder.encode_pdf(
            usize::try_from(base + 4).map_err(|_| Error::InvalidPacket)?,
            &RESIDUAL_PDF[selector(bandwidth, stage1, coefficient)?],
        )?;
        if base.abs() == 4 {
            encoder.encode_pdf(usize::from(value.unsigned_abs() - 4), &RESIDUAL_EXTENSION)?;
        }
    }
    Ok(())
}

fn prediction_weight(bandwidth: Bandwidth, stage1: u8, coefficient: usize) -> Result<u8, Error> {
    if stage1 >= 32 {
        return Err(Error::InvalidPacket);
    }
    if wideband(bandwidth)? {
        let selection = PRED_SELECT_WB[usize::from(stage1)]
            .get(coefficient)
            .ok_or(Error::InvalidFrameSize)?;
        Ok(PREDICT_WB[usize::from(*selection == b'D')][coefficient])
    } else {
        let selection = PRED_SELECT_NB_MB[usize::from(stage1)]
            .get(coefficient)
            .ok_or(Error::InvalidFrameSize)?;
        Ok(PREDICT_NB_MB[usize::from(*selection == b'B')][coefficient])
    }
}

/// Reverses the stage-2 backwards predictor and produces Q10 residuals.
pub fn dequantize_stage2(
    bandwidth: Bandwidth,
    stage1: u8,
    indices: &Stage2,
    output_q10: &mut [i32],
) -> Result<(), Error> {
    let order = if wideband(bandwidth)? { 16 } else { 10 };
    if usize::from(indices.order) != order || output_q10.len() < order {
        return Err(Error::InvalidFrameSize);
    }
    let qstep = if order == 16 { 9_830i64 } else { 11_796i64 };
    for coefficient in (0..order).rev() {
        let index = i64::from(indices.index[coefficient]);
        let quantized = (((index << 10) - index.signum() * 102) * qstep) >> 16;
        let prediction = if coefficient + 1 < order {
            (i64::from(output_q10[coefficient + 1])
                * i64::from(prediction_weight(bandwidth, stage1, coefficient)?))
                >> 8
        } else {
            0
        };
        output_q10[coefficient] =
            i32::try_from(prediction + quantized).map_err(|_| Error::InvalidPacket)?;
    }
    Ok(())
}

pub fn decode_interpolation(decoder: &mut RangeDecoder<'_>) -> Result<u8, Error> {
    u8::try_from(decoder.decode_pdf(&INTERPOLATION)?).map_err(|_| Error::InvalidPacket)
}

pub fn encode_interpolation(encoder: &mut RangeEncoder<'_>, factor_q2: u8) -> Result<(), Error> {
    if factor_q2 > 4 {
        return Err(Error::InvalidPacket);
    }
    encoder.encode_pdf(usize::from(factor_q2), &INTERPOLATION)
}

pub fn interpolate(
    previous: &[i16],
    current: &[i16],
    factor_q2: u8,
    output: &mut [i16],
) -> Result<(), Error> {
    if previous.len() != current.len()
        || output.len() < current.len()
        || factor_q2 > 4
        || !matches!(current.len(), 10 | 16)
    {
        return Err(Error::InvalidFrameSize);
    }
    for ((&old, &new), out) in previous.iter().zip(current).zip(output.iter_mut()) {
        let value =
            i32::from(old) + ((i32::from(factor_q2) * (i32::from(new) - i32::from(old))) >> 2);
        *out = i16::try_from(value).map_err(|_| Error::InvalidPacket)?;
    }
    Ok(())
}

/// Stabilizes an order-10 or order-16 Q15 LSF vector exactly as Section
/// 4.2.7.5.4 specifies, including its bounded twenty-pass adjustment.
pub fn stabilize(values: &mut [i16], bandwidth: Bandwidth) -> Result<(), Error> {
    let spacing: &[i32] = if wideband(bandwidth)? {
        &SPACING_WB
    } else {
        &SPACING_NB_MB
    };
    let order = spacing.len() - 1;
    if values.len() != order {
        return Err(Error::InvalidFrameSize);
    }
    let mut work = [0i32; 16];
    for (&source, target) in values.iter().zip(work.iter_mut()) {
        *target = i32::from(source);
    }
    for _ in 0..20 {
        let mut index = 0usize;
        let mut minimum = work[0] - spacing[0];
        for i in 1..order {
            let violation = work[i] - work[i - 1] - spacing[i];
            if violation < minimum {
                minimum = violation;
                index = i;
            }
        }
        let last = 32_768 - work[order - 1] - spacing[order];
        if last < minimum {
            minimum = last;
            index = order;
        }
        if minimum >= 0 {
            return copy_back(&work[..order], values);
        }
        if index == 0 {
            work[0] = spacing[0];
        } else if index == order {
            work[order - 1] = 32_768 - spacing[order];
        } else {
            let minimum_center = (spacing[index] >> 1) + spacing[..index].iter().sum::<i32>();
            let maximum_center =
                32_768 - (spacing[index] >> 1) - spacing[index + 1..].iter().sum::<i32>();
            let center =
                ((work[index - 1] + work[index] + 1) >> 1).clamp(minimum_center, maximum_center);
            work[index - 1] = center - (spacing[index] >> 1);
            work[index] = work[index - 1] + spacing[index];
        }
    }
    for i in 1..order {
        let mut j = i;
        while j > 0 && work[j] < work[j - 1] {
            work.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut lower = 0;
    for i in 0..order {
        work[i] = work[i].max(lower + spacing[i]);
        lower = work[i];
    }
    let mut upper = 32_768;
    for i in (0..order).rev() {
        work[i] = work[i].min(upper - spacing[i + 1]);
        upper = work[i];
    }
    copy_back(&work[..order], values)
}

fn copy_back(work: &[i32], values: &mut [i16]) -> Result<(), Error> {
    for (&source, target) in work.iter().zip(values) {
        *target = i16::try_from(source).map_err(|_| Error::InvalidPacket)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage1_and_interpolation_symbol_round_trips() {
        for bandwidth in [Bandwidth::Narrow, Bandwidth::Medium, Bandwidth::Wide] {
            for signal in [
                SignalType::Inactive,
                SignalType::Unvoiced,
                SignalType::Voiced,
            ] {
                for index in 0..32 {
                    let mut bytes = [0u8; 16];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    encode_stage1(&mut encoder, bandwidth, signal, index).unwrap();
                    encoder.finish().unwrap();
                    assert_eq!(
                        decode_stage1(&mut RangeDecoder::new(&bytes), bandwidth, signal),
                        Ok(index)
                    );
                }
            }
        }
        for factor in 0..=4 {
            let mut bytes = [0u8; 8];
            let mut encoder = RangeEncoder::new(&mut bytes);
            encode_interpolation(&mut encoder, factor).unwrap();
            encoder.finish().unwrap();
            assert_eq!(
                decode_interpolation(&mut RangeDecoder::new(&bytes)),
                Ok(factor)
            );
        }
    }

    #[test]
    fn stabilization_enforces_every_spacing_constraint() {
        let mut nb = [30_000, -4, 20, 19, 18, 17, 16, 15, 14, 32_000];
        stabilize(&mut nb, Bandwidth::Narrow).unwrap();
        assert!(i32::from(nb[0]) >= SPACING_NB_MB[0]);
        for i in 1..10 {
            assert!(i32::from(nb[i]) - i32::from(nb[i - 1]) >= SPACING_NB_MB[i]);
        }
        assert!(32_768 - i32::from(nb[9]) >= SPACING_NB_MB[10]);
        let mut wb = [32_000i16; 16];
        stabilize(&mut wb, Bandwidth::Wide).unwrap();
        for i in 1..16 {
            assert!(i32::from(wb[i]) - i32::from(wb[i - 1]) >= SPACING_WB[i]);
        }
    }

    #[test]
    fn interpolation_has_exact_endpoints() {
        let old = [100i16; 10];
        let new = [500i16; 10];
        let mut out = [0i16; 10];
        interpolate(&old, &new, 0, &mut out).unwrap();
        assert_eq!(out, old);
        interpolate(&old, &new, 4, &mut out).unwrap();
        assert_eq!(out, new);
    }

    #[test]
    fn every_stage2_context_and_extension_round_trips() {
        for bandwidth in [Bandwidth::Narrow, Bandwidth::Wide] {
            let order = if bandwidth == Bandwidth::Wide { 16 } else { 10 };
            for stage1 in 0..32 {
                for value in -10..=10 {
                    let mut expected = Stage2 {
                        order,
                        index: [0; 16],
                    };
                    for (coefficient, item) in
                        expected.index[..usize::from(order)].iter_mut().enumerate()
                    {
                        *item = if coefficient & 1 == 0 { value } else { -value };
                    }
                    let mut bytes = [0u8; 128];
                    let mut encoder = RangeEncoder::new(&mut bytes);
                    encode_stage2(&mut encoder, bandwidth, stage1, &expected).unwrap();
                    encoder.finish().unwrap();
                    assert_eq!(
                        decode_stage2(&mut RangeDecoder::new(&bytes), bandwidth, stage1),
                        Ok(expected)
                    );
                }
            }
        }
    }

    #[test]
    fn predictor_tables_and_residual_dequantization_cover_all_contexts() {
        for (bandwidth, order) in [(Bandwidth::Narrow, 10u8), (Bandwidth::Wide, 16u8)] {
            for stage1 in 0..32 {
                let mut indices = Stage2 {
                    order,
                    index: [0; 16],
                };
                for (coefficient, value) in
                    indices.index[..usize::from(order)].iter_mut().enumerate()
                {
                    *value = (coefficient as i8 % 7) - 3;
                }
                let mut residual = [0i32; 16];
                dequantize_stage2(bandwidth, stage1, &indices, &mut residual).unwrap();
                assert!(
                    residual[..usize::from(order)]
                        .iter()
                        .all(|value| value.abs() < 4096)
                );
                let zero = Stage2 {
                    order,
                    index: [0; 16],
                };
                dequantize_stage2(bandwidth, stage1, &zero, &mut residual).unwrap();
                assert_eq!(
                    &residual[..usize::from(order)],
                    &[0; 16][..usize::from(order)]
                );
            }
        }
        assert!(
            SELECT_NB_MB
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| matches!(value, b'a'..=b'h'))
        );
        assert!(
            SELECT_WB
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| matches!(value, b'i'..=b'p'))
        );
        for stage1 in 0..32 {
            for coefficient in 0..16 {
                assert!((8..16).contains(&selector(Bandwidth::Wide, stage1, coefficient).unwrap()));
            }
        }
        assert!(
            PRED_SELECT_NB_MB
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| matches!(value, b'A' | b'B'))
        );
        assert!(
            PRED_SELECT_WB
                .iter()
                .flat_map(|row| row.iter())
                .all(|value| matches!(value, b'C' | b'D'))
        );
        for index in 0..32 {
            let vector = nb_mb_stage1_codebook(index).unwrap();
            assert!(vector.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(nb_mb_stage1_codebook(32), Err(Error::InvalidPacket));
    }

    #[test]
    fn codebooks_weights_and_reconstruction_cover_both_orders() {
        for (bandwidth, order) in [(Bandwidth::Narrow, 10u8), (Bandwidth::Wide, 16u8)] {
            for stage1 in 0..32 {
                let mut codebook = [0u8; 16];
                assert_eq!(
                    stage1_codebook(bandwidth, stage1, &mut codebook),
                    Ok(usize::from(order))
                );
                assert!(
                    codebook[..usize::from(order)]
                        .windows(2)
                        .all(|pair| pair[0] < pair[1])
                );
                let mut weights = [0u16; 16];
                inverse_harmonic_weights(&codebook[..usize::from(order)], &mut weights).unwrap();
                assert!(
                    weights[..usize::from(order)]
                        .iter()
                        .all(|&weight| (1819..=5227).contains(&weight))
                );
                for sign in [-1, 1] {
                    let mut indices = Stage2 {
                        order,
                        index: [0; 16],
                    };
                    indices.index[..usize::from(order)].fill(10 * sign);
                    let mut reconstructed = [0i16; 16];
                    assert_eq!(
                        reconstruct(bandwidth, stage1, &indices, &mut reconstructed),
                        Ok(usize::from(order))
                    );
                    let spacing: &[i32] = if order == 16 {
                        &SPACING_WB
                    } else {
                        &SPACING_NB_MB
                    };
                    assert!(i32::from(reconstructed[0]) >= spacing[0]);
                    for coefficient in 1..usize::from(order) {
                        assert!(
                            i32::from(reconstructed[coefficient])
                                - i32::from(reconstructed[coefficient - 1])
                                >= spacing[coefficient]
                        );
                    }
                    assert!(
                        32_768 - i32::from(reconstructed[usize::from(order) - 1])
                            >= spacing[usize::from(order)]
                    );
                }
            }
        }
    }

    #[test]
    fn cosine_interpolation_and_lpc_range_limiting_are_bounded() {
        assert_eq!(cosine_q17(0), Ok(131_072));
        assert_eq!(cosine_q17(16_384), Ok(0));
        assert_eq!(cosine_q17(32_767), Ok(-131_072));
        for (bandwidth, order) in [(Bandwidth::Narrow, 10u8), (Bandwidth::Wide, 16u8)] {
            for stage1 in 0..32 {
                let indices = Stage2 {
                    order,
                    index: [0; 16],
                };
                let mut lsf = [0i16; 16];
                reconstruct(bandwidth, stage1, &indices, &mut lsf).unwrap();
                let mut lpc = [0i16; 16];
                assert_eq!(
                    lsf_to_lpc_range_limited(&lsf[..usize::from(order)], bandwidth, &mut lpc),
                    Ok(usize::from(order))
                );
                assert_eq!(
                    lsf_to_lpc(&lsf[..usize::from(order)], bandwidth, &mut lpc),
                    Ok(usize::from(order))
                );
                let q17 = lpc.map(|value| i64::from(value) << 5);
                assert!(inverse_prediction_gain_is_stable(
                    &q17[..usize::from(order)]
                ));
            }
        }
        assert!(inverse_prediction_gain_is_stable(&[0; 10]));
        assert!(!inverse_prediction_gain_is_stable(&[4096i64 << 5; 10]));
    }

    #[test]
    fn complete_lsf_parameter_sections_round_trip_into_stable_lpc() {
        for (bandwidth, order) in [(Bandwidth::Narrow, 10u8), (Bandwidth::Wide, 16u8)] {
            for twenty_ms in [false, true] {
                let indices = LsfIndices {
                    stage1: 17,
                    stage2: Stage2 {
                        order,
                        index: [0, 1, -1, 4, -4, 10, -10, 2, -2, 3, 0, 0, 0, 0, 0, 0],
                    },
                    interpolation_q2: twenty_ms.then_some(2),
                };
                let mut bytes = [0u8; 128];
                let mut encoder = RangeEncoder::new(&mut bytes);
                encode_lsf(
                    &mut encoder,
                    bandwidth,
                    SignalType::Voiced,
                    twenty_ms,
                    &indices,
                )
                .unwrap();
                encoder.finish().unwrap();
                let previous = [
                    1000i16, 2500, 4000, 5500, 7000, 8500, 10000, 11500, 13000, 14500, 16000,
                    17500, 19000, 20500, 22000, 23500,
                ];
                let mut current = [0i16; 16];
                let mut first = [0i16; 16];
                let mut second = [0i16; 16];
                let decoded = decode_lsf(
                    &mut RangeDecoder::new(&bytes),
                    bandwidth,
                    SignalType::Voiced,
                    twenty_ms,
                    Some(&previous[..usize::from(order)]),
                    &mut current,
                    &mut first,
                    &mut second,
                )
                .unwrap();
                assert_eq!(decoded, indices);
                let first_q17 = first.map(|value| i64::from(value) << 5);
                let second_q17 = second.map(|value| i64::from(value) << 5);
                assert!(inverse_prediction_gain_is_stable(
                    &first_q17[..usize::from(order)]
                ));
                assert!(inverse_prediction_gain_is_stable(
                    &second_q17[..usize::from(order)]
                ));
            }
        }
    }
}
