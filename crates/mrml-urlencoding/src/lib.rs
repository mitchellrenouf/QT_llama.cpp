#![no_std]

use core::ops::Deref;
use mrml_runtime::{Text, Vector};

const HEX: &[u8; 16] = b"0123456789ABCDEF";

pub const fn encoded_len(input: &str) -> usize {
    let bytes = input.as_bytes();
    let mut index = 0;
    let mut length = 0;
    while index < bytes.len() {
        length += if is_unreserved(bytes[index]) { 1 } else { 3 };
        index += 1;
    }
    length
}

pub fn decoded_len(input: &str) -> Result<usize, DecodeError> {
    let bytes = input.as_bytes();
    let mut index = 0;
    let mut length = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            hex(*bytes.get(index + 1).ok_or(DecodeError)?)?;
            hex(*bytes.get(index + 2).ok_or(DecodeError)?)?;
            index += 3;
        } else {
            index += 1;
        }
        length += 1;
    }
    Ok(length)
}

pub fn encode_to<'a>(input: &str, output: &'a mut [u8]) -> Result<&'a str, BufferError> {
    let required = encoded_len(input);
    if output.len() < required {
        return Err(BufferError::new(required));
    }
    let mut written = 0;
    for byte in input.bytes() {
        if is_unreserved(byte) {
            output[written] = byte;
            written += 1;
        } else {
            output[written] = b'%';
            output[written + 1] = HEX[(byte >> 4) as usize];
            output[written + 2] = HEX[(byte & 15) as usize];
            written += 3;
        }
    }
    // Every written byte is either an ASCII input byte or an ASCII escape.
    Ok(unsafe { core::str::from_utf8_unchecked(&output[..written]) })
}

pub fn decode_to<'a>(input: &str, output: &'a mut [u8]) -> Result<&'a str, DecodeToError> {
    let required = decoded_len(input)?;
    if output.len() < required {
        return Err(DecodeToError::Buffer(BufferError::new(required)));
    }
    let bytes = input.as_bytes();
    let mut read = 0;
    let mut written = 0;
    while read < bytes.len() {
        if bytes[read] == b'%' {
            let high = hex(*bytes.get(read + 1).ok_or(DecodeError)?)?;
            let low = hex(*bytes.get(read + 2).ok_or(DecodeError)?)?;
            output[written] = high << 4 | low;
            read += 3;
        } else {
            output[written] = bytes[read];
            read += 1;
        }
        written += 1;
    }
    core::str::from_utf8(&output[..written]).map_err(|_| DecodeToError::InvalidEncoding)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextResult<'a> {
    Borrowed(&'a str),
    Owned(Text),
}

impl Deref for TextResult<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl core::fmt::Display for TextResult<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self)
    }
}

impl PartialEq<&str> for TextResult<'_> {
    fn eq(&self, other: &&str) -> bool {
        &**self == *other
    }
}

pub fn encode(input: &str) -> TextResult<'_> {
    if input.bytes().all(is_unreserved) {
        return TextResult::Borrowed(input);
    }
    let mut output = Text::with_capacity(encoded_len(input)).expect("MRML allocation failed");
    for byte in input.bytes() {
        if is_unreserved(byte) {
            output
                .try_push(byte as char)
                .expect("MRML allocation failed");
        } else {
            output.try_push('%').expect("MRML allocation failed");
            output
                .try_push(HEX[(byte >> 4) as usize] as char)
                .expect("MRML allocation failed");
            output
                .try_push(HEX[(byte & 15) as usize] as char)
                .expect("MRML allocation failed");
        }
    }
    TextResult::Owned(output)
}

pub fn decode(input: &str) -> Result<TextResult<'_>, DecodeError> {
    if !input.as_bytes().contains(&b'%') {
        return Ok(TextResult::Borrowed(input));
    }
    let bytes = input.as_bytes();
    let mut output = Vector::with_capacity(decoded_len(input)?).map_err(|_| DecodeError)?;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = hex(*bytes.get(index + 1).ok_or(DecodeError)?)?;
            let low = hex(*bytes.get(index + 2).ok_or(DecodeError)?)?;
            output.try_push(high << 4 | low).map_err(|_| DecodeError)?;
            index += 3;
        } else {
            output.try_push(bytes[index]).map_err(|_| DecodeError)?;
            index += 1;
        }
    }
    Text::try_from_utf8(output)
        .map(TextResult::Owned)
        .map_err(|_| DecodeError)
}

const fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

fn hex(byte: u8) -> Result<u8, DecodeError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(DecodeError),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError;

impl core::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("invalid percent encoding")
    }
}

impl core::error::Error for DecodeError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferError {
    pub required: usize,
}

impl BufferError {
    const fn new(required: usize) -> Self {
        Self { required }
    }
}

impl core::fmt::Display for BufferError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "output buffer requires {} bytes", self.required)
    }
}

impl core::error::Error for BufferError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeToError {
    Buffer(BufferError),
    InvalidEncoding,
}

impl From<DecodeError> for DecodeToError {
    fn from(_: DecodeError) -> Self {
        Self::InvalidEncoding
    }
}

impl core::fmt::Display for DecodeToError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Buffer(error) => error.fmt(formatter),
            Self::InvalidEncoding => formatter.write_str("invalid percent encoding"),
        }
    }
}

impl core::error::Error for DecodeToError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_buffers_need_no_allocation() {
        let mut encoded = [0; 64];
        let encoded = encode_to("solar panels/efficiency", &mut encoded).unwrap();
        assert_eq!(encoded, "solar%20panels%2Fefficiency");

        let mut decoded = [0; 64];
        assert_eq!(
            decode_to(encoded, &mut decoded).unwrap(),
            "solar panels/efficiency"
        );
        assert_eq!(
            encode_to("space", &mut [0; 4]),
            Err(BufferError { required: 5 })
        );
        assert_eq!(decode_to("%41%42%43", &mut [0; 3]).unwrap(), "ABC");
        assert_eq!(
            decode_to("%41%42%43", &mut [0; 2]),
            Err(DecodeToError::Buffer(BufferError { required: 3 }))
        );
        assert_eq!(decoded_len("broken%"), Err(DecodeError));
    }

    #[test]
    fn preserves_borrowing_and_round_trips_unicode() {
        assert!(matches!(encode("solar-panels"), TextResult::Borrowed(_)));
        assert!(matches!(
            decode("solar-panels"),
            Ok(TextResult::Borrowed(_))
        ));
        let encoded = encode("solar panels/效率 + cost");
        assert_eq!(decode(&encoded).unwrap(), "solar panels/效率 + cost");
    }

    #[test]
    fn rejects_malformed_and_non_utf8_sequences() {
        assert!(decode("%ZZ").is_err());
        assert!(decode("%F0%28%8C%28").is_err());
        assert!(decode("trailing%").is_err());
    }
}
