//! Encoding primitives used by media and browser tools.
use std::borrow::Cow;

pub fn percent_encode(input: &str) -> Cow<'_, str> {
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

pub fn percent_decode(input: &str) -> Result<Cow<'_, str>, &'static str> {
    if !input.as_bytes().contains(&b'%') {
        return Ok(Cow::Borrowed(input));
    }
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = hex(*bytes.get(index + 1).ok_or("incomplete percent escape")?)?;
            let low = hex(*bytes.get(index + 2).ok_or("incomplete percent escape")?)?;
            output.push(high << 4 | low);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(output)
        .map(Cow::Owned)
        .map_err(|_| "percent-decoded bytes are not UTF-8")
}

fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

fn hex(byte: u8) -> Result<u8, &'static str> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err("invalid percent escape"),
    }
}
#[cold]
#[inline(never)]
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let bits = (chunk[0] as u32) << 16
            | (chunk.get(1).copied().unwrap_or(0) as u32) << 8
            | chunk.get(2).copied().unwrap_or(0) as u32;
        output.push(ALPHABET[((bits >> 18) & 63) as usize] as char);
        output.push(ALPHABET[((bits >> 12) & 63) as usize] as char);
        output.push(if chunk.len() > 1 {
            ALPHABET[((bits >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(bits & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

pub fn base64_decode(input: &str) -> Result<Vec<u8>, &'static str> {
    fn value(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return Err("invalid base64 length");
    }
    let mut output = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let a = value(chunk[0]).ok_or("invalid base64 character")? as u32;
        let b = value(chunk[1]).ok_or("invalid base64 character")? as u32;
        let c = if chunk[2] == b'=' {
            0
        } else {
            value(chunk[2]).ok_or("invalid base64 character")? as u32
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            value(chunk[3]).ok_or("invalid base64 character")? as u32
        };
        let bits = a << 18 | b << 12 | c << 6 | d;
        output.push((bits >> 16) as u8);
        if chunk[2] != b'=' {
            output.push((bits >> 8) as u8)
        }
        if chunk[3] != b'=' {
            output.push(bits as u8)
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    #[test]
    fn base64_matches_standard_vectors() {
        for (raw, encoded) in [
            (b"".as_slice(), ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(super::base64_encode(raw), encoded);
            assert_eq!(super::base64_decode(encoded).unwrap(), raw);
        }
    }

    #[test]
    fn percent_encoding_borrows_and_round_trips_unicode() {
        assert!(matches!(super::percent_encode("solar-panels"), Cow::Borrowed(_)));
        assert!(matches!(super::percent_decode("solar-panels"), Ok(Cow::Borrowed(_))));
        let encoded = super::percent_encode("solar panels/效率 + cost");
        assert_eq!(super::percent_decode(&encoded).unwrap(), "solar panels/效率 + cost");
        assert!(super::percent_decode("%ZZ").is_err());
        assert!(super::percent_decode("%F0%28%8C%28").is_err());
        assert!(super::percent_decode("trailing%").is_err());
    }
}
