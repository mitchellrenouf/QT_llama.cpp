use crate::bits::ReverseBits;
use crate::fse::{Entry, State as FseState, Table, parse_table};
use crate::{Error, Result};
use mrml_runtime::Vector;

#[derive(Clone, Copy)]
pub(crate) struct Sequence {
    literals: u32,
    length: u32,
    offset: u32,
}

pub(crate) struct State {
    literal_table: Option<Table>,
    offset_table: Option<Table>,
    match_table: Option<Table>,
    offsets: [u32; 3],
}

impl State {
    pub(crate) const fn new() -> Self {
        Self {
            literal_table: None,
            offset_table: None,
            match_table: None,
            offsets: [1, 4, 8],
        }
    }
}

enum Kind {
    Literals,
    Offset,
    Match,
}

pub(crate) fn decode(bytes: &[u8], state: &mut State) -> Result<Vector<Sequence>> {
    let (count, count_bytes) = sequence_count(bytes)?;
    let mut output = Vector::with_capacity(count as usize).map_err(|_| Error::Allocation)?;
    if count == 0 {
        return Ok(output);
    }
    let modes = *bytes
        .get(count_bytes)
        .ok_or(Error::InvalidSequencesSection)?;
    if modes & 3 != 0 {
        return Err(Error::InvalidSequencesSection);
    }
    let mut cursor = count_bytes + 1;
    let literal_table = resolve(
        (modes >> 6) & 3,
        bytes,
        &mut cursor,
        &mut state.literal_table,
        Kind::Literals,
    )?;
    let offset_table = resolve(
        (modes >> 4) & 3,
        bytes,
        &mut cursor,
        &mut state.offset_table,
        Kind::Offset,
    )?;
    let match_table = resolve(
        (modes >> 2) & 3,
        bytes,
        &mut cursor,
        &mut state.match_table,
        Kind::Match,
    )?;
    let mut bits = ReverseBits::new(bytes.get(cursor..).ok_or(Error::InvalidSequencesSection)?)?;
    let mut literal_state = FseState::new(&literal_table, &mut bits)?;
    let mut offset_state = FseState::new(&offset_table, &mut bits)?;
    let mut match_state = FseState::new(&match_table, &mut bits)?;
    for index in 0..count {
        let literal_entry = literal_state.entry(&literal_table);
        let offset_entry = offset_state.entry(&offset_table);
        let match_entry = match_state.entry(&match_table);
        let (literal_base, literal_bits) = *LITERAL_LENGTHS
            .get(literal_entry.symbol as usize)
            .ok_or(Error::InvalidSequencesSection)?;
        let (match_base, match_bits) = *MATCH_LENGTHS
            .get(match_entry.symbol as usize)
            .ok_or(Error::InvalidSequencesSection)?;
        let offset_code = offset_entry.symbol as u8;
        if offset_code >= 32 {
            return Err(Error::InvalidSequencesSection);
        }
        let offset_value = (1u32 << offset_code) + bits.read(offset_code)?;
        let match_length = match_base + bits.read(match_bits)?;
        let literal_length = literal_base + bits.read(literal_bits)?;
        let offset = resolve_offset(offset_value, literal_length, &mut state.offsets)?;
        output.push(Sequence {
            literals: literal_length,
            length: match_length,
            offset,
        });
        if index + 1 != count {
            advance(&mut literal_state, literal_entry, &literal_table, &mut bits)?;
            advance(&mut match_state, match_entry, &match_table, &mut bits)?;
            advance(&mut offset_state, offset_entry, &offset_table, &mut bits)?;
        }
    }
    state.literal_table = Some(literal_table);
    state.offset_table = Some(offset_table);
    state.match_table = Some(match_table);
    Ok(output)
}

fn advance(
    state: &mut FseState,
    entry: Entry,
    table: &Table,
    bits: &mut ReverseBits<'_>,
) -> Result<()> {
    let value = entry.baseline as u32 + bits.read(entry.bits)?;
    if value as usize >= table.entries.len() {
        return Err(Error::InvalidFseState);
    }
    *state = FseState::from_value(value as u16);
    Ok(())
}

fn sequence_count(bytes: &[u8]) -> Result<(u32, usize)> {
    let first = *bytes.first().ok_or(Error::InvalidSequencesSection)?;
    if first < 128 {
        Ok((first as u32, 1))
    } else if first < 255 {
        Ok((
            ((first as u32 - 128) << 8)
                | *bytes.get(1).ok_or(Error::InvalidSequencesSection)? as u32,
            2,
        ))
    } else {
        Ok((
            0x7f00
                + *bytes.get(1).ok_or(Error::InvalidSequencesSection)? as u32
                + ((*bytes.get(2).ok_or(Error::InvalidSequencesSection)? as u32) << 8),
            3,
        ))
    }
}

fn resolve(
    mode: u8,
    bytes: &[u8],
    cursor: &mut usize,
    previous: &mut Option<Table>,
    kind: Kind,
) -> Result<Table> {
    match mode {
        0 => predefined(kind),
        1 => {
            let symbol = *bytes.get(*cursor).ok_or(Error::InvalidSequencesSection)? as u16;
            *cursor += 1;
            let mut entries = Vector::with_capacity(1).map_err(|_| Error::Allocation)?;
            entries.push(Entry {
                symbol,
                bits: 0,
                baseline: 0,
            });
            Ok(Table {
                accuracy: 0,
                entries,
            })
        }
        2 => {
            let (accuracy, maximum) = match kind {
                Kind::Literals => (9, 35),
                Kind::Offset => (8, 31),
                Kind::Match => (9, 52),
            };
            let (table, consumed) = parse_table(
                bytes.get(*cursor..).ok_or(Error::InvalidSequencesSection)?,
                accuracy,
                maximum,
            )?;
            *cursor += consumed;
            Ok(table)
        }
        3 => previous.clone().ok_or(Error::InvalidSequencesSection),
        _ => Err(Error::InvalidSequencesSection),
    }
}

fn predefined(kind: Kind) -> Result<Table> {
    match kind {
        Kind::Literals => Table::from_probabilities(
            &[
                4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1,
                1, 1, 1, 1, -1, -1, -1, -1,
            ],
            6,
        ),
        Kind::Offset => Table::from_probabilities(
            &[
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1,
                -1, -1,
            ],
            5,
        ),
        Kind::Match => Table::from_probabilities(
            &[
                1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
            ],
            6,
        ),
    }
}

fn resolve_offset(value: u32, literals: u32, previous: &mut [u32; 3]) -> Result<u32> {
    let offset;
    if value > 3 {
        offset = value - 3;
        previous[2] = previous[1];
        previous[1] = previous[0];
        previous[0] = offset;
    } else if literals == 0 {
        offset = match value {
            1 => previous[1],
            2 => previous[2],
            3 => previous[0].checked_sub(1).ok_or(Error::InvalidOffset)?,
            _ => return Err(Error::InvalidOffset),
        };
        match value {
            1 => previous.swap(0, 1),
            2 => {
                let old = previous[2];
                previous[2] = previous[1];
                previous[1] = previous[0];
                previous[0] = old;
            }
            3 => {
                previous[2] = previous[1];
                previous[1] = previous[0];
                previous[0] = offset;
            }
            _ => {}
        }
    } else {
        offset = previous[(value - 1) as usize];
        if value == 2 {
            previous.swap(0, 1);
        } else if value == 3 {
            let old = previous[2];
            previous[2] = previous[1];
            previous[1] = previous[0];
            previous[0] = old;
        }
    }
    if offset == 0 {
        return Err(Error::InvalidOffset);
    }
    Ok(offset)
}

pub(crate) fn execute(
    sequences: &[Sequence],
    literals: &[u8],
    output: &mut Vector<u8>,
) -> Result<()> {
    let before = output.len();
    let mut literal = 0usize;
    for sequence in sequences {
        let count = sequence.literals as usize;
        let end = literal
            .checked_add(count)
            .ok_or(Error::InvalidSequencesSection)?;
        output
            .try_extend_from_slice(
                literals
                    .get(literal..end)
                    .ok_or(Error::InvalidSequencesSection)?,
            )
            .map_err(|_| Error::Allocation)?;
        literal = end;
        let offset = sequence.offset as usize;
        if offset == 0 || offset > output.len() {
            return Err(Error::InvalidOffset);
        }
        for _ in 0..sequence.length {
            let byte = output[output.len() - offset];
            output.push(byte);
        }
        if output.len() - before > 128 * 1024 {
            return Err(Error::BlockTooLarge);
        }
    }
    output
        .try_extend_from_slice(
            literals
                .get(literal..)
                .ok_or(Error::InvalidSequencesSection)?,
        )
        .map_err(|_| Error::Allocation)?;
    if output.len() - before > 128 * 1024 {
        return Err(Error::BlockTooLarge);
    }
    Ok(())
}

const LITERAL_LENGTHS: [(u32, u8); 36] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 1),
    (18, 1),
    (20, 1),
    (22, 1),
    (24, 2),
    (28, 2),
    (32, 3),
    (40, 3),
    (48, 4),
    (64, 6),
    (128, 7),
    (256, 8),
    (512, 9),
    (1024, 10),
    (2048, 11),
    (4096, 12),
    (8192, 13),
    (16384, 14),
    (32768, 15),
    (65536, 16),
];
const MATCH_LENGTHS: [(u32, u8); 53] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 0),
    (17, 0),
    (18, 0),
    (19, 0),
    (20, 0),
    (21, 0),
    (22, 0),
    (23, 0),
    (24, 0),
    (25, 0),
    (26, 0),
    (27, 0),
    (28, 0),
    (29, 0),
    (30, 0),
    (31, 0),
    (32, 0),
    (33, 0),
    (34, 0),
    (35, 1),
    (37, 1),
    (39, 1),
    (41, 1),
    (43, 2),
    (47, 2),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 5),
    (131, 7),
    (259, 8),
    (515, 9),
    (1027, 10),
    (2051, 11),
    (4099, 12),
    (8195, 13),
    (16387, 14),
    (32771, 15),
    (65539, 16),
];
