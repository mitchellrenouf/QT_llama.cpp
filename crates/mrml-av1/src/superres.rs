//! Normative horizontal super-resolution upscaling (section 7.16).

use crate::{
    ChromaSampling, Error,
    reconstruction::{FrameBuffer, Plane},
};

const SCALE_BITS: u32 = 14;
const EXTRA_BITS: u32 = 8;
const SCALE_MASK: i64 = (1 << SCALE_BITS) - 1;
const FILTER_BITS: u32 = 7;
const FILTER_OFFSET: i64 = 3;

#[rustfmt::skip]
const UPSCALE_FILTER: [[i16; 8]; 64] = [
    [ 0, 0,  0,128, 0,  0, 0, 0], [ 0, 0, -1,128, 2, -1, 0, 0],
    [ 0, 1, -3,127, 4, -2, 1, 0], [ 0, 1, -4,127, 6, -3, 1, 0],
    [ 0, 2, -6,126, 8, -3, 1, 0], [ 0, 2, -7,125,11, -4, 1, 0],
    [-1, 2, -8,125,13, -5, 2, 0], [-1, 3, -9,124,15, -6, 2, 0],
    [-1, 3,-10,123,18, -6, 2,-1], [-1, 3,-11,122,20, -7, 3,-1],
    [-1, 4,-12,121,22, -8, 3,-1], [-1, 4,-13,120,25, -9, 3,-1],
    [-1, 4,-14,118,28, -9, 3,-1], [-1, 4,-15,117,30,-10, 4,-1],
    [-1, 5,-16,116,32,-11, 4,-1], [-1, 5,-16,114,35,-12, 4,-1],
    [-1, 5,-17,112,38,-12, 4,-1], [-1, 5,-18,111,40,-13, 5,-1],
    [-1, 5,-18,109,43,-14, 5,-1], [-1, 6,-19,107,45,-14, 5,-1],
    [-1, 6,-19,105,48,-15, 5,-1], [-1, 6,-19,103,51,-16, 5,-1],
    [-1, 6,-20,101,53,-16, 6,-1], [-1, 6,-20, 99,56,-17, 6,-1],
    [-1, 6,-20, 97,58,-17, 6,-1], [-1, 6,-20, 95,61,-18, 6,-1],
    [-2, 7,-20, 93,64,-18, 6,-2], [-2, 7,-20, 91,66,-19, 6,-1],
    [-2, 7,-20, 88,69,-19, 6,-1], [-2, 7,-20, 86,71,-19, 6,-1],
    [-2, 7,-20, 84,74,-20, 7,-2], [-2, 7,-20, 81,76,-20, 7,-1],
    [-2, 7,-20, 79,79,-20, 7,-2], [-1, 7,-20, 76,81,-20, 7,-2],
    [-2, 7,-20, 74,84,-20, 7,-2], [-1, 6,-19, 71,86,-20, 7,-2],
    [-1, 6,-19, 69,88,-20, 7,-2], [-1, 6,-19, 66,91,-20, 7,-2],
    [-2, 6,-18, 64,93,-20, 7,-2], [-1, 6,-18, 61,95,-20, 6,-1],
    [-1, 6,-17, 58,97,-20, 6,-1], [-1, 6,-17, 56,99,-20, 6,-1],
    [-1, 6,-16, 53,101,-20, 6,-1],[-1, 5,-16, 51,103,-19, 6,-1],
    [-1, 5,-15, 48,105,-19, 6,-1],[-1, 5,-14, 45,107,-19, 6,-1],
    [-1, 5,-14, 43,109,-18, 5,-1],[-1, 5,-13, 40,111,-18, 5,-1],
    [-1, 4,-12, 38,112,-17, 5,-1],[-1, 4,-12, 35,114,-16, 5,-1],
    [-1, 4,-11, 32,116,-16, 5,-1],[-1, 4,-10, 30,117,-15, 4,-1],
    [-1, 3, -9, 28,118,-14, 4,-1],[-1, 3, -9, 25,120,-13, 4,-1],
    [-1, 3, -8, 22,121,-12, 4,-1],[-1, 3, -7, 20,122,-11, 3,-1],
    [-1, 2, -6, 18,123,-10, 3,-1],[ 0, 2, -6, 15,124, -9, 3,-1],
    [ 0, 2, -5, 13,125, -8, 2,-1],[ 0, 1, -4, 11,125, -7, 2, 0],
    [ 0, 1, -3,  8,126, -6, 2, 0],[ 0, 1, -3,  6,127, -4, 1, 0],
    [ 0, 1, -2,  4,127, -3, 1, 0],[ 0, 0, -1,  2,128, -1, 0, 0],
];

pub fn upscale(
    input: &FrameBuffer,
    frame_width: u32,
    upscaled_width: u32,
    frame_height: u32,
) -> Result<FrameBuffer, Error> {
    if upscaled_width <= frame_width || frame_height == 0 {
        return Err(Error::InvalidObu);
    }
    let mut output = FrameBuffer::new(
        upscaled_width,
        frame_height,
        input.bit_depth(),
        input.sampling(),
    )?;
    upscale_plane(
        &input.y,
        &mut output.y,
        frame_width,
        upscaled_width,
        frame_height,
        false,
        false,
        input.bit_depth(),
    )?;
    let (sub_x, sub_y) = match input.sampling() {
        ChromaSampling::Cs400 => return Ok(output),
        ChromaSampling::Cs420 => (true, true),
        ChromaSampling::Cs422 => (true, false),
        ChromaSampling::Cs444 => (false, false),
    };
    upscale_plane(
        input.u.as_ref().ok_or(Error::InvalidObu)?,
        output.u.as_mut().ok_or(Error::InvalidObu)?,
        frame_width,
        upscaled_width,
        frame_height,
        sub_x,
        sub_y,
        input.bit_depth(),
    )?;
    upscale_plane(
        input.v.as_ref().ok_or(Error::InvalidObu)?,
        output.v.as_mut().ok_or(Error::InvalidObu)?,
        frame_width,
        upscaled_width,
        frame_height,
        sub_x,
        sub_y,
        input.bit_depth(),
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn upscale_plane(
    input: &Plane,
    output: &mut Plane,
    frame_width: u32,
    upscaled_width: u32,
    frame_height: u32,
    sub_x: bool,
    sub_y: bool,
    bit_depth: u8,
) -> Result<(), Error> {
    let sx = u32::from(sub_x);
    let sy = u32::from(sub_y);
    let down_w = i64::from(frame_width.div_ceil(1 << sx));
    let up_w = i64::from(upscaled_width.div_ceil(1 << sx));
    let plane_h =
        usize::try_from(frame_height.div_ceil(1 << sy)).map_err(|_| Error::LimitExceeded)?;
    let step = ((down_w << SCALE_BITS) + up_w / 2) / up_w;
    let err = up_w * step - (down_w << SCALE_BITS);
    let mut initial = (-((up_w - down_w) << (SCALE_BITS - 1)) + up_w / 2) / up_w
        + (1 << (EXTRA_BITS - 1))
        - err / 2;
    initial &= SCALE_MASK;
    let max_x = usize::try_from(frame_width.div_ceil(8).saturating_mul(8) >> sx)
        .map_err(|_| Error::LimitExceeded)?
        .checked_sub(1)
        .ok_or(Error::InvalidObu)?;
    let maximum = (1i64 << bit_depth) - 1;
    for y in 0..plane_h {
        for x in 0..usize::try_from(up_w).map_err(|_| Error::LimitExceeded)? {
            let src = -(1 << SCALE_BITS)
                + initial
                + i64::try_from(x).map_err(|_| Error::LimitExceeded)? * step;
            let src_px = src >> SCALE_BITS;
            let phase = usize::try_from((src & SCALE_MASK) >> EXTRA_BITS)
                .map_err(|_| Error::LimitExceeded)?;
            let mut sum = 0i64;
            for (k, &coefficient) in UPSCALE_FILTER[phase].iter().enumerate() {
                let sample_x = (src_px + i64::try_from(k).map_err(|_| Error::LimitExceeded)?
                    - FILTER_OFFSET)
                    .clamp(0, i64::try_from(max_x).map_err(|_| Error::LimitExceeded)?);
                sum += i64::from(input.sample(
                    usize::try_from(sample_x).map_err(|_| Error::LimitExceeded)?,
                    y,
                )?) * i64::from(coefficient);
            }
            let value = ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, maximum);
            output.set_sample(
                x,
                y,
                u16::try_from(value).map_err(|_| Error::LimitExceeded)?,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_filter_phase_preserves_a_constant() {
        assert!(
            UPSCALE_FILTER
                .iter()
                .all(|phase| phase.iter().sum::<i16>() == 128)
        );
    }

    #[test]
    fn upscaling_preserves_constant_planes() {
        let input = FrameBuffer::new(16, 9, 10, ChromaSampling::Cs420).unwrap();
        let output = upscale(&input, 16, 24, 9).unwrap();
        assert_eq!(output.y.sample(23, 8), Ok(512));
        assert_eq!(output.u.as_ref().unwrap().sample(11, 4), Ok(512));
        assert_eq!(output.into_frame().unwrap().width, 24);
    }
}
