use crate::{ProtocolError, RsaPrivateKey, RsaPublicKey, parse_rsa_public_key};
use mrml_runtime::Vector;

const MAX_KEY_FILE: usize = 64 * 1024;

pub fn parse_rsa_private_pem(source: &str) -> Result<RsaPrivateKey, ProtocolError> {
    const BEGIN: &str = "-----BEGIN RSA PRIVATE KEY-----";
    const END: &str = "-----END RSA PRIVATE KEY-----";
    if source.len() > MAX_KEY_FILE {
        return Err(ProtocolError::Length);
    }
    let source = source.trim();
    let body = source
        .strip_prefix(BEGIN)
        .ok_or(ProtocolError::InvalidPublicKey)?
        .strip_suffix(END)
        .ok_or(ProtocolError::InvalidPublicKey)?;
    let der = base64_decode(body)?;
    let mut outer = Der::new(&der);
    let sequence = outer.element(0x30)?;
    if outer.remaining() != 0 {
        return Err(ProtocolError::InvalidPublicKey);
    }
    let mut key = Der::new(sequence);
    if key.integer()? != [0] {
        return Err(ProtocolError::InvalidPublicKey);
    }
    let modulus = positive(key.integer()?)?;
    let exponent = positive(key.integer()?)?;
    let private_exponent = positive(key.integer()?)?;
    for _ in 0..5 {
        positive(key.integer()?)?;
    }
    if key.remaining() != 0
        || modulus.len() < 128
        || modulus.len() > 512
        || private_exponent.len() > 512
    {
        return Err(ProtocolError::InvalidPublicKey);
    }
    Ok(RsaPrivateKey {
        public: RsaPublicKey {
            modulus: copy(modulus),
            exponent: copy(exponent),
        },
        private_exponent: copy(private_exponent),
    })
}

pub fn parse_rsa_public_line(source: &str) -> Result<RsaPublicKey, ProtocolError> {
    if source.len() > MAX_KEY_FILE
        || source
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
    {
        return Err(ProtocolError::InvalidPublicKey);
    }
    for line in source.lines() {
        let fields: Vector<&str> = line.split_whitespace().collect();
        for pair in fields.windows(2) {
            if pair[0] == "ssh-rsa" {
                let blob = base64_decode(pair[1])?;
                return parse_rsa_public_key(&blob);
            }
        }
    }
    Err(ProtocolError::InvalidPublicKey)
}

struct Der<'a> {
    source: &'a [u8],
    cursor: usize,
}
impl<'a> Der<'a> {
    fn new(source: &'a [u8]) -> Self {
        Self { source, cursor: 0 }
    }
    fn remaining(&self) -> usize {
        self.source.len() - self.cursor
    }
    fn integer(&mut self) -> Result<&'a [u8], ProtocolError> {
        self.element(2)
    }
    fn element(&mut self, tag: u8) -> Result<&'a [u8], ProtocolError> {
        if self.take(1)?[0] != tag {
            return Err(ProtocolError::InvalidPublicKey);
        }
        let first = self.take(1)?[0];
        let length = if first & 0x80 == 0 {
            first as usize
        } else {
            let count = (first & 0x7f) as usize;
            if count == 0 || count > 4 {
                return Err(ProtocolError::InvalidPublicKey);
            }
            let bytes = self.take(count)?;
            if bytes[0] == 0 {
                return Err(ProtocolError::InvalidPublicKey);
            }
            let mut value = 0usize;
            for byte in bytes {
                value = value
                    .checked_mul(256)
                    .and_then(|v| v.checked_add(*byte as usize))
                    .ok_or(ProtocolError::Length)?;
            }
            if value < 128 {
                return Err(ProtocolError::InvalidPublicKey);
            }
            value
        };
        self.take(length)
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], ProtocolError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(ProtocolError::Length)?;
        let value = self
            .source
            .get(self.cursor..end)
            .ok_or(ProtocolError::Truncated)?;
        self.cursor = end;
        Ok(value)
    }
}
fn positive(value: &[u8]) -> Result<&[u8], ProtocolError> {
    if value.is_empty() || value[0] & 0x80 != 0 {
        return Err(ProtocolError::InvalidPublicKey);
    }
    if value[0] == 0 {
        if value.len() == 1 || value[1] & 0x80 == 0 {
            return Err(ProtocolError::InvalidPublicKey);
        }
        Ok(&value[1..])
    } else {
        Ok(value)
    }
}
fn copy(value: &[u8]) -> Vector<u8> {
    let mut out = Vector::new();
    out.extend(value.iter().copied());
    out
}

pub(crate) fn base64_decode(source: &str) -> Result<Vector<u8>, ProtocolError> {
    let mut clean = Vector::new();
    for byte in source.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if clean.len() >= MAX_KEY_FILE {
            return Err(ProtocolError::Length);
        }
        clean.push(byte);
    }
    if clean.is_empty() || clean.len() % 4 != 0 {
        return Err(ProtocolError::InvalidPublicKey);
    }
    let mut output = Vector::new();
    for (index, chunk) in clean.chunks(4).enumerate() {
        let last = index + 1 == clean.len() / 4;
        let a = b64(chunk[0])?;
        let b = b64(chunk[1])?;
        let c = if chunk[2] == b'=' { 64 } else { b64(chunk[2])? };
        let d = if chunk[3] == b'=' { 64 } else { b64(chunk[3])? };
        if a == 64 || b == 64 || c == 64 && d != 64 || (!last && (c == 64 || d == 64)) {
            return Err(ProtocolError::InvalidPublicKey);
        }
        output.push(a << 2 | b >> 4);
        if c != 64 {
            output.push(b << 4 | c >> 2);
        }
        if d != 64 {
            output.push(c << 6 | d);
        }
        if c == 64 && b & 15 != 0 || d == 64 && c != 64 && c & 3 != 0 {
            return Err(ProtocolError::InvalidPublicKey);
        }
    }
    Ok(output)
}
pub(crate) fn base64_encode(bytes: &[u8]) -> mrml_runtime::Text {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = mrml_runtime::Text::new();
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        out.push(A[(a >> 2) as usize] as char);
        out.push(A[(((a & 3) << 4) | (b >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            A[(((b & 15) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            A[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
fn b64(byte: u8) -> Result<u8, ProtocolError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(ProtocolError::InvalidPublicKey),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn length(out: &mut Vector<u8>, value: usize) {
        if value < 128 {
            out.push(value as u8)
        } else if value <= 255 {
            out.push(0x81);
            out.push(value as u8)
        } else {
            out.push(0x82);
            out.extend((value as u16).to_be_bytes())
        }
    }
    fn integer(out: &mut Vector<u8>, value: &[u8]) {
        out.push(2);
        length(out, value.len());
        out.extend(value.iter().copied())
    }
    fn encode64(bytes: &[u8]) -> mrml_runtime::Text {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = mrml_runtime::Text::new();
        for chunk in bytes.chunks(3) {
            let a = chunk[0];
            let b = chunk.get(1).copied().unwrap_or(0);
            let c = chunk.get(2).copied().unwrap_or(0);
            out.push(A[(a >> 2) as usize] as char);
            out.push(A[(((a & 3) << 4) | (b >> 4)) as usize] as char);
            out.push(if chunk.len() > 1 {
                A[(((b & 15) << 2) | (c >> 6)) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                A[(c & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }
    #[test]
    fn strict_base64_vectors() {
        assert_eq!(&*base64_decode("TQ==").unwrap(), b"M");
        assert_eq!(&*base64_decode("TWE=").unwrap(), b"Ma");
        assert_eq!(&*base64_decode("TWFu").unwrap(), b"Man");
        assert!(base64_decode("TR==").is_err());
        assert!(base64_decode("TQ=A").is_err());
    }
    #[test]
    fn rejects_noncanonical_der_lengths() {
        let invalid = "-----BEGIN RSA PRIVATE KEY-----\nM4GAAQA=\n-----END RSA PRIVATE KEY-----";
        assert!(parse_rsa_private_pem(invalid).is_err());
    }
    #[test]
    fn parses_canonical_pkcs1_container() {
        let mut body = Vector::new();
        integer(&mut body, &[0]);
        let mut modulus = [0u8; 129];
        modulus[1] = 0x80;
        integer(&mut body, &modulus);
        integer(&mut body, &[1, 0, 1]);
        integer(&mut body, &[7]);
        for _ in 0..5 {
            integer(&mut body, &[1]);
        }
        let mut der = Vector::new();
        der.push(0x30);
        length(&mut der, body.len());
        der.extend(body);
        let pem = mrml_runtime::mrml_format!(
            "-----BEGIN RSA PRIVATE KEY-----\n{}\n-----END RSA PRIVATE KEY-----\n",
            encode64(&der)
        );
        let key = parse_rsa_private_pem(&pem).unwrap();
        assert_eq!(key.public.modulus.len(), 128);
        assert_eq!(&*key.public.exponent, &[1, 0, 1]);
        assert_eq!(&*key.private_exponent, &[7]);
    }

    #[test]
    fn parses_raw_known_hosts_and_allowed_signers_lines() {
        let public = RsaPublicKey {
            modulus: Vector::from([0x80; 128]),
            exponent: Vector::from([1, 0, 1]),
        };
        let blob = crate::encode_rsa_public_key(&public).unwrap();
        let encoded = base64_encode(&blob);
        for line in [
            mrml_runtime::mrml_format!("ssh-rsa {encoded} comment"),
            mrml_runtime::mrml_format!("git@example.invalid ssh-rsa {encoded}"),
            mrml_runtime::mrml_format!("example.invalid ssh-rsa {encoded}"),
        ] {
            assert_eq!(parse_rsa_public_line(&line).unwrap(), public);
        }
        assert!(parse_rsa_public_line("principal ssh-ed25519 AAAA").is_err());
    }
}
