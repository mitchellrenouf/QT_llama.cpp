//! Fixed-buffer encoders available without `std` or a global allocator.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferTooSmall {
    pub required: usize,
    pub available: usize,
}

pub const fn base64_encoded_len(input_len: usize) -> Option<usize> {
    match input_len.checked_add(2) {
        Some(padded) => (padded / 3).checked_mul(4),
        None => None,
    }
}

pub fn base64_encode_into(input: &[u8], output: &mut [u8]) -> Result<usize, BufferTooSmall> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let required = base64_encoded_len(input.len()).unwrap_or(usize::MAX);
    if output.len() < required {
        return Err(BufferTooSmall {
            required,
            available: output.len(),
        });
    }

    let mut source = 0;
    let mut target = 0;
    while source < input.len() {
        let remaining = input.len() - source;
        let a = input[source] as u32;
        let b = if remaining > 1 {
            input[source + 1] as u32
        } else {
            0
        };
        let c = if remaining > 2 {
            input[source + 2] as u32
        } else {
            0
        };
        let bits = a << 16 | b << 8 | c;
        output[target] = ALPHABET[((bits >> 18) & 63) as usize];
        output[target + 1] = ALPHABET[((bits >> 12) & 63) as usize];
        output[target + 2] = if remaining > 1 {
            ALPHABET[((bits >> 6) & 63) as usize]
        } else {
            b'='
        };
        output[target + 3] = if remaining > 2 {
            ALPHABET[(bits & 63) as usize]
        } else {
            b'='
        };
        source += remaining.min(3);
        target += 4;
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_standard_vectors_without_allocation() {
        for (raw, expected) in [
            (b"".as_slice(), b"".as_slice()),
            (b"f", b"Zg=="),
            (b"fo", b"Zm8="),
            (b"foo", b"Zm9v"),
            (b"foobar", b"Zm9vYmFy"),
        ] {
            let mut output = [0u8; 8];
            let written = base64_encode_into(raw, &mut output).unwrap();
            assert_eq!(&output[..written], expected);
        }
    }

    #[test]
    fn reports_required_capacity() {
        assert_eq!(
            base64_encode_into(b"foo", &mut [0u8; 3]),
            Err(BufferTooSmall {
                required: 4,
                available: 3
            }),
        );
    }
}
