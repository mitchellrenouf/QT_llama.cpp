#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::borrow::Cow;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

#[cfg(feature = "alloc")]
pub fn encode(input: &str) -> Cow<'_, str> {
    if input.bytes().all(is_unreserved) {
        return Cow::Borrowed(input);
    }
    let mut output = String::with_capacity(input.len());
    for byte in input.bytes() {
        if is_unreserved(byte) {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 15) as usize] as char);
        }
    }
    Cow::Owned(output)
}

#[cfg(feature = "alloc")]
pub fn decode(input: &str) -> Result<Cow<'_, str>, DecodeError> {
    if !input.as_bytes().contains(&b'%') {
        return Ok(Cow::Borrowed(input));
    }
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = hex(*bytes.get(index + 1).ok_or(DecodeError)?)?;
            let low = hex(*bytes.get(index + 2).ok_or(DecodeError)?)?;
            output.push(high << 4 | low);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(output)
        .map(Cow::Owned)
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

    #[cfg(feature = "alloc")]
    #[test]
    fn preserves_borrowing_and_round_trips_unicode() {
        assert!(matches!(encode("solar-panels"), Cow::Borrowed(_)));
        assert!(matches!(decode("solar-panels"), Ok(Cow::Borrowed(_))));
        let encoded = encode("solar panels/效率 + cost");
        assert_eq!(decode(&encoded).unwrap(), "solar panels/效率 + cost");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn rejects_malformed_and_non_utf8_sequences() {
        assert!(decode("%ZZ").is_err());
        assert!(decode("%F0%28%8C%28").is_err());
        assert!(decode("trailing%").is_err());
    }
}
