//! Large-scale Tile List OBU syntax (sections 5.12 and 6.11).

use crate::Error;
use mrml_runtime::Vector;

pub const MAX_TILE_LIST_ENTRIES: usize = 512;
pub const MAX_ANCHOR_FRAMES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileListHeader {
    pub output_width_in_tiles: u16,
    pub output_height_in_tiles: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileListEntry<'a> {
    pub anchor_frame_index: u8,
    pub anchor_tile_row: u8,
    pub anchor_tile_column: u8,
    pub coded_tile_data: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileList<'a> {
    pub header: TileListHeader,
    pub entries: Vector<TileListEntry<'a>>,
}

pub fn parse(payload: &[u8], maximum_tile_data: usize) -> Result<TileList<'_>, Error> {
    let header_bytes = payload.get(..4).ok_or(Error::Truncated)?;
    let header = TileListHeader {
        output_width_in_tiles: u16::from(header_bytes[0]) + 1,
        output_height_in_tiles: u16::from(header_bytes[1]) + 1,
    };
    let count = usize::from(u16::from_be_bytes([header_bytes[2], header_bytes[3]])) + 1;
    if count > MAX_TILE_LIST_ENTRIES
        || count
            > usize::from(header.output_width_in_tiles)
                .checked_mul(usize::from(header.output_height_in_tiles))
                .ok_or(Error::LimitExceeded)?
    {
        return Err(Error::InvalidObu);
    }
    let mut entries = Vector::with_capacity(count).map_err(|_| Error::LimitExceeded)?;
    let mut position = 4usize;
    for _ in 0..count {
        let entry_header = payload
            .get(position..position.checked_add(5).ok_or(Error::LimitExceeded)?)
            .ok_or(Error::Truncated)?;
        let anchor_frame_index = entry_header[0];
        if usize::from(anchor_frame_index) >= MAX_ANCHOR_FRAMES {
            return Err(Error::InvalidObu);
        }
        let size = usize::from(u16::from_be_bytes([entry_header[3], entry_header[4]])) + 1;
        if size > maximum_tile_data {
            return Err(Error::LimitExceeded);
        }
        position = position.checked_add(5).ok_or(Error::LimitExceeded)?;
        let end = position.checked_add(size).ok_or(Error::LimitExceeded)?;
        let coded_tile_data = payload.get(position..end).ok_or(Error::Truncated)?;
        entries
            .try_push(TileListEntry {
                anchor_frame_index,
                anchor_tile_row: entry_header[1],
                anchor_tile_column: entry_header[2],
                coded_tile_data,
            })
            .map_err(|_| Error::LimitExceeded)?;
        position = end;
    }
    if position != payload.len() {
        return Err(Error::InvalidObu);
    }
    Ok(TileList { header, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_entries_and_big_endian_lengths() {
        let payload = [
            1, 0, 0, 1, // 2x1 output, two entries
            0, 2, 3, 0, 1, 0xaa, 0xbb, // two-byte tile
            127, 4, 5, 0, 0, 0xcc, // one-byte tile
        ];
        let list = parse(&payload, 2).unwrap();
        assert_eq!(list.header.output_width_in_tiles, 2);
        assert_eq!(list.header.output_height_in_tiles, 1);
        assert_eq!(list.entries.len(), 2);
        assert_eq!(list.entries[0].coded_tile_data, &[0xaa, 0xbb]);
        assert_eq!(list.entries[1].anchor_frame_index, 127);
    }

    #[test]
    fn rejects_excess_entries_anchor_indices_and_trailing_data() {
        assert_eq!(parse(&[0, 0, 0, 1], 64), Err(Error::InvalidObu));
        assert_eq!(
            parse(&[0, 0, 0, 0, 128, 0, 0, 0, 0, 1], 64),
            Err(Error::InvalidObu)
        );
        assert_eq!(
            parse(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2], 64),
            Err(Error::InvalidObu)
        );
    }

    #[test]
    fn distinguishes_truncation_from_configured_size_limit() {
        assert_eq!(
            parse(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 1),
            Err(Error::LimitExceeded)
        );
        assert_eq!(
            parse(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 2),
            Err(Error::Truncated)
        );
    }
}
