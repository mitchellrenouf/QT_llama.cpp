#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

pub fn encode(input: &str) -> Cow<'_, str> {
    if input.bytes().all(is_unreserved) {
        return Cow::Borrowed(input);
    }
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
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
    String::from_utf8(output).map(Cow::Owned).map_err(|_| DecodeError)
}

fn is_unreserved(byte: u8) -> bool {
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

#[derive(Clone, Copy, Debug)]
pub struct DecodeError;

impl core::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("invalid percent encoding")
    }
}

impl core::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_borrowing_and_round_trips_unicode() {
        assert!(matches!(encode("solar-panels"), Cow::Borrowed(_)));
        assert!(matches!(decode("solar-panels"), Ok(Cow::Borrowed(_))));
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
