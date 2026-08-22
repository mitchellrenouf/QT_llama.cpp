//! RFC 6716 mode-transition redundancy signaling and byte separation.

use crate::{Error, Mode, RangeDecoder, RangeEncoder};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RedundancyPosition {
    Beginning,
    End,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Redundancy {
    pub position: RedundancyPosition,
    pub offset: usize,
    pub len: usize,
}

fn remaining_bits(decoder: &RangeDecoder<'_>, frame_bytes: usize) -> Result<u32, Error> {
    let total = u32::try_from(frame_bytes)
        .map_err(|_| Error::InvalidPacket)?
        .checked_mul(8)
        .ok_or(Error::InvalidPacket)?;
    Ok(total.saturating_sub(decoder.tell()))
}

pub(crate) fn decode_header(
    decoder: &mut RangeDecoder<'_>,
    mode: Mode,
    frame_bytes: usize,
) -> Result<Option<Redundancy>, Error> {
    if decoder.storage_len() != frame_bytes {
        return Err(Error::InvalidPacket);
    }
    let present = match mode {
        Mode::Silk => remaining_bits(decoder, frame_bytes)? >= 17,
        Mode::Hybrid => {
            remaining_bits(decoder, frame_bytes)? >= 37 && decoder.decode_bit_logp(12)?
        }
        Mode::Celt => return Ok(None),
    };
    if !present {
        return Ok(None);
    }
    let position = if decoder.decode_bit_logp(1)? {
        RedundancyPosition::Beginning
    } else {
        RedundancyPosition::End
    };
    let available = remaining_bits(decoder, frame_bytes)? / 8;
    let len = match mode {
        Mode::Silk => usize::try_from(available).map_err(|_| Error::InvalidPacket)?,
        Mode::Hybrid => {
            usize::try_from(decoder.decode_uint(256)? + 2).map_err(|_| Error::InvalidPacket)?
        }
        Mode::Celt => unreachable!(),
    };
    if len < 2 || u32::try_from(len).map_err(|_| Error::InvalidPacket)? > available {
        return Err(Error::InvalidPacket);
    }
    let offset = decoder.reserve_tail(len)?;
    Ok(Some(Redundancy {
        position,
        offset,
        len,
    }))
}

pub(crate) fn encode_absence(
    encoder: &mut RangeEncoder<'_>,
    mode: Mode,
    frame_bytes: usize,
) -> Result<(), Error> {
    let total = u32::try_from(frame_bytes)
        .map_err(|_| Error::InvalidPacket)?
        .checked_mul(8)
        .ok_or(Error::InvalidPacket)?;
    let remaining = total
        .checked_sub(encoder.tell())
        .ok_or(Error::InvalidPacket)?;
    match mode {
        Mode::Hybrid if remaining >= 37 => encoder.encode_bit_logp(false, 12),
        Mode::Hybrid | Mode::Celt => Ok(()),
        // SILK-only presence is implicit whenever 17 bits remain, so an
        // encoder that wants no redundancy must terminate below that bound.
        Mode::Silk => Err(Error::InvalidPacket),
    }
}

pub(crate) fn encode_header(
    encoder: &mut RangeEncoder<'_>,
    mode: Mode,
    frame_bytes: usize,
    position: RedundancyPosition,
    len: usize,
) -> Result<(), Error> {
    if !(2..=257).contains(&len) {
        return Err(Error::InvalidPacket);
    }
    let total = u32::try_from(frame_bytes)
        .map_err(|_| Error::InvalidPacket)?
        .checked_mul(8)
        .ok_or(Error::InvalidPacket)?;
    let before = total.saturating_sub(encoder.tell());
    match mode {
        Mode::Hybrid => {
            if before < 37 {
                return Err(Error::InvalidPacket);
            }
            encoder.encode_bit_logp(true, 12)?;
        }
        Mode::Silk => {
            if before < 17 {
                return Err(Error::InvalidPacket);
            }
        }
        Mode::Celt => return Err(Error::InvalidPacket),
    }
    encoder.encode_bit_logp(position == RedundancyPosition::Beginning, 1)?;
    if mode == Mode::Hybrid {
        encoder.encode_uint(
            u32::try_from(len - 2).map_err(|_| Error::InvalidPacket)?,
            256,
        )?;
    }
    let available = total.saturating_sub(encoder.tell()) / 8;
    let len_u32 = u32::try_from(len).map_err(|_| Error::InvalidPacket)?;
    if (mode == Mode::Silk && available != len_u32) || (mode == Mode::Hybrid && available < len_u32)
    {
        return Err(Error::InvalidPacket);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hybrid_absence_does_not_shrink_the_entropy_storage() {
        let mut bytes = [0u8; 16];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_absence(&mut encoder, Mode::Hybrid, 16).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        assert_eq!(decode_header(&mut decoder, Mode::Hybrid, 16), Ok(None));
        assert_eq!(decoder.storage_len(), 16);
    }

    #[test]
    fn hybrid_header_reserves_the_explicit_byte_aligned_tail() {
        let mut bytes = [0u8; 24];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encode_header(
            &mut encoder,
            Mode::Hybrid,
            24,
            RedundancyPosition::Beginning,
            7,
        )
        .unwrap();
        encoder.finish().unwrap();
        bytes[17..].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
        let mut decoder = RangeDecoder::new(&bytes);
        assert_eq!(
            decode_header(&mut decoder, Mode::Hybrid, 24),
            Ok(Some(Redundancy {
                position: RedundancyPosition::Beginning,
                offset: 17,
                len: 7,
            }))
        );
        assert_eq!(decoder.storage_len(), 17);
    }

    #[test]
    fn hybrid_rejects_a_declared_tail_larger_than_whole_bytes_remaining() {
        let mut bytes = [0u8; 8];
        let mut encoder = RangeEncoder::new(&mut bytes);
        encoder.encode_bit_logp(true, 12).unwrap();
        encoder.encode_bit_logp(false, 1).unwrap();
        encoder.encode_uint(255, 256).unwrap();
        encoder.finish().unwrap();
        let mut decoder = RangeDecoder::new(&bytes);
        assert_eq!(
            decode_header(&mut decoder, Mode::Hybrid, 8),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn present_header_rejects_celt_mode_and_invalid_lengths() {
        let mut bytes = [0u8; 32];
        let mut encoder = RangeEncoder::new(&mut bytes);
        assert_eq!(
            encode_header(&mut encoder, Mode::Celt, 32, RedundancyPosition::End, 2,),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            encode_header(&mut encoder, Mode::Hybrid, 32, RedundancyPosition::End, 258,),
            Err(Error::InvalidPacket)
        );
    }
}
