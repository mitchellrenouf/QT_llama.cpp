use crate::{
    KeySchedule, RecordKeys, TLS_AES_128_GCM_SHA256, TlsError, Transcript, X25519_MLKEM768,
    finished_verify_data, hybrid_server_share,
};
use mrml_crypto::rsa_pss_sha256_sign;
use mrml_runtime::{TcpStream, Vector};

fn u16_push(out: &mut Vector<u8>, n: usize) -> Result<(), TlsError> {
    if n > u16::MAX as usize {
        return Err(TlsError::Handshake);
    }
    out.try_extend_from_slice(&(n as u16).to_be_bytes())
        .map_err(|_| TlsError::AllocationFailed)
}
fn u24_push(out: &mut Vector<u8>, n: usize) -> Result<(), TlsError> {
    if n > 0xff_ffff {
        return Err(TlsError::Handshake);
    }
    out.try_extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8])
        .map_err(|_| TlsError::AllocationFailed)
}
fn handshake(kind: u8, body: &[u8]) -> Result<Vector<u8>, TlsError> {
    let mut out = Vector::new();
    out.try_push(kind).map_err(|_| TlsError::AllocationFailed)?;
    u24_push(&mut out, body.len())?;
    out.try_extend_from_slice(body)
        .map_err(|_| TlsError::AllocationFailed)?;
    Ok(out)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], TlsError> {
        let end = self.at.checked_add(n).ok_or(TlsError::Handshake)?;
        let out = self.bytes.get(self.at..end).ok_or(TlsError::Handshake)?;
        self.at = end;
        Ok(out)
    }
    fn byte(&mut self) -> Result<u8, TlsError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, TlsError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn u24(&mut self) -> Result<usize, TlsError> {
        let b = self.take(3)?;
        Ok((b[0] as usize) << 16 | (b[1] as usize) << 8 | b[2] as usize)
    }
    fn done(&self) -> bool {
        self.at == self.bytes.len()
    }
}

fn der_length(c: &mut Cursor<'_>) -> Result<usize, TlsError> {
    let first = c.byte()?;
    if first & 0x80 == 0 {
        return Ok(first as usize);
    }
    let count = (first & 0x7f) as usize;
    if count == 0 || count > 4 {
        return Err(TlsError::Certificate);
    }
    let mut n = 0usize;
    for b in c.take(count)? {
        n = n
            .checked_mul(256)
            .and_then(|v| v.checked_add(*b as usize))
            .ok_or(TlsError::Certificate)?;
    }
    Ok(n)
}
fn der<'a>(c: &mut Cursor<'a>, tag: u8) -> Result<&'a [u8], TlsError> {
    if c.byte()? != tag {
        return Err(TlsError::Certificate);
    }
    let n = der_length(c)?;
    c.take(n).map_err(|_| TlsError::Certificate)
}
fn integer<'a>(c: &mut Cursor<'a>) -> Result<&'a [u8], TlsError> {
    let mut v = der(c, 2)?;
    while v.len() > 1 && v[0] == 0 {
        v = &v[1..];
    }
    Ok(v)
}

fn base64_value(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}
fn pem_blocks(input: &[u8], label: &[u8]) -> Result<Vector<Vector<u8>>, TlsError> {
    let mut begin = Vector::new();
    begin
        .try_extend_from_slice(b"-----BEGIN ")
        .map_err(|_| TlsError::AllocationFailed)?;
    begin
        .try_extend_from_slice(label)
        .map_err(|_| TlsError::AllocationFailed)?;
    begin
        .try_extend_from_slice(b"-----")
        .map_err(|_| TlsError::AllocationFailed)?;
    let mut end = Vector::new();
    end.try_extend_from_slice(b"-----END ")
        .map_err(|_| TlsError::AllocationFailed)?;
    end.try_extend_from_slice(label)
        .map_err(|_| TlsError::AllocationFailed)?;
    end.try_extend_from_slice(b"-----")
        .map_err(|_| TlsError::AllocationFailed)?;
    let mut blocks = Vector::new();
    let mut at = 0;
    while let Some(p) = input[at..]
        .windows(begin.len())
        .position(|w| w == &begin[..])
    {
        let start = at + p + begin.len();
        let stop = input[start..]
            .windows(end.len())
            .position(|w| w == &end[..])
            .ok_or(TlsError::Certificate)?
            + start;
        let mut out = Vector::new();
        let mut bits = 0u32;
        let mut count = 0u8;
        for &b in &input[start..stop] {
            if b == b'=' {
                break;
            }
            if let Some(v) = base64_value(b) {
                bits = (bits << 6) | v as u32;
                count += 1;
                if count == 4 {
                    out.try_extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8, bits as u8])
                        .map_err(|_| TlsError::AllocationFailed)?;
                    bits = 0;
                    count = 0;
                }
            }
        }
        if count == 2 {
            out.try_push((bits >> 4) as u8)
                .map_err(|_| TlsError::AllocationFailed)?;
        } else if count == 3 {
            out.try_extend_from_slice(&[(bits >> 10) as u8, (bits >> 2) as u8])
                .map_err(|_| TlsError::AllocationFailed)?;
        } else if count != 0 {
            return Err(TlsError::Certificate);
        }
        blocks
            .try_push(out)
            .map_err(|_| TlsError::AllocationFailed)?;
        at = stop + end.len();
    }
    Ok(blocks)
}

pub struct TlsServerConfig {
    certificates: Vector<Vector<u8>>,
    modulus: Vector<u8>,
    private_exponent: Vector<u8>,
}
impl TlsServerConfig {
    pub fn from_pem(certificate_pem: &[u8], private_key_pem: &[u8]) -> Result<Self, TlsError> {
        let certificates = pem_blocks(certificate_pem, b"CERTIFICATE")?;
        if certificates.is_empty() {
            return Err(TlsError::Certificate);
        }
        let pkcs8 = pem_blocks(private_key_pem, b"PRIVATE KEY")?;
        let pkcs1 = pem_blocks(private_key_pem, b"RSA PRIVATE KEY")?;
        let raw = if let Some(v) = pkcs1.first() {
            &v[..]
        } else if let Some(v) = pkcs8.first() {
            let mut top = Cursor::new(v);
            let seq = der(&mut top, 0x30)?;
            let mut s = Cursor::new(seq);
            integer(&mut s)?;
            der(&mut s, 0x30)?;
            der(&mut s, 0x04)?
        } else {
            return Err(TlsError::Certificate);
        };
        let mut top = Cursor::new(raw);
        let seq = der(&mut top, 0x30)?;
        let mut s = Cursor::new(seq);
        integer(&mut s)?;
        let modulus = integer(&mut s)?;
        integer(&mut s)?;
        let private_exponent = integer(&mut s)?;
        let mut n = Vector::new();
        n.try_extend_from_slice(modulus)
            .map_err(|_| TlsError::AllocationFailed)?;
        let mut d = Vector::new();
        d.try_extend_from_slice(private_exponent)
            .map_err(|_| TlsError::AllocationFailed)?;
        Ok(Self {
            certificates,
            modulus: n,
            private_exponent: d,
        })
    }
}

fn record(stream: &mut TcpStream) -> Result<Vector<u8>, TlsError> {
    let mut h = [0; 5];
    stream.read_exact(&mut h).map_err(|_| TlsError::Io)?;
    if h[1] != 3 || !(1..=3).contains(&h[2]) {
        return Err(TlsError::InvalidRecord);
    }
    let n = u16::from_be_bytes([h[3], h[4]]) as usize;
    if n > 16640 {
        return Err(TlsError::InvalidRecord);
    }
    let mut out = Vector::new();
    out.try_extend_from_slice(&h)
        .map_err(|_| TlsError::AllocationFailed)?;
    out.try_resize(5 + n, 0)
        .map_err(|_| TlsError::AllocationFailed)?;
    stream.read_exact(&mut out[5..]).map_err(|_| TlsError::Io)?;
    Ok(out)
}
fn send_plain(stream: &mut TcpStream, msg: &[u8]) -> Result<(), TlsError> {
    let h = [22, 3, 3, (msg.len() >> 8) as u8, msg.len() as u8];
    stream.write_all(&h).map_err(|_| TlsError::Io)?;
    stream.write_all(msg).map_err(|_| TlsError::Io)
}
fn client_share(msg: &[u8]) -> Result<(&[u8], &[u8]), TlsError> {
    let mut c = Cursor::new(msg);
    if c.byte()? != 1 || c.u24()? != msg.len() - 4 || c.take(2)? != [3, 3] {
        return Err(TlsError::Handshake);
    }
    c.take(32)?;
    let sid = c.byte()? as usize;
    let session = c.take(sid)?;
    let suites = c.u16()? as usize;
    let offered = c.take(suites)?;
    if !offered
        .chunks_exact(2)
        .any(|x| x == TLS_AES_128_GCM_SHA256.to_be_bytes())
    {
        return Err(TlsError::Handshake);
    }
    let compression = c.byte()? as usize;
    c.take(compression)?;
    let elen = c.u16()? as usize;
    let mut e = Cursor::new(c.take(elen)?);
    if !c.done() {
        return Err(TlsError::Handshake);
    }
    while !e.done() {
        let kind = e.u16()?;
        let n = e.u16()? as usize;
        let value = e.take(n)?;
        if kind == 51 {
            let mut k = Cursor::new(value);
            let total = k.u16()? as usize;
            let mut shares = Cursor::new(k.take(total)?);
            while !shares.done() {
                let group = shares.u16()?;
                let len = shares.u16()? as usize;
                let share = shares.take(len)?;
                if group == X25519_MLKEM768 && len == 1216 {
                    return Ok((session, share));
                }
            }
        }
    }
    Err(TlsError::Handshake)
}

pub struct TlsServerStream {
    stream: TcpStream,
    read_keys: RecordKeys,
    write_keys: RecordKeys,
    pending: Vector<u8>,
    at: usize,
}
impl TlsServerStream {
    pub fn accept(mut stream: TcpStream, config: &TlsServerConfig) -> Result<Self, TlsError> {
        stream
            .set_read_timeout_millis(30_000)
            .map_err(|_| TlsError::Io)?;
        stream
            .set_write_timeout_millis(30_000)
            .map_err(|_| TlsError::Io)?;
        let first = record(&mut stream)?;
        if first[0] != 22 {
            return Err(TlsError::Handshake);
        }
        let client_hello = &first[5..];
        let (session, share) = client_share(client_hello)?;
        let (server_share, mut shared) = hybrid_server_share(share)?;
        let mut random = [0; 32];
        mrml_runtime::fill_random(&mut random).map_err(|_| TlsError::AuthenticationFailed)?;
        let mut body = Vector::new();
        body.try_extend_from_slice(&[3, 3])
            .map_err(|_| TlsError::AllocationFailed)?;
        body.try_extend_from_slice(&random)
            .map_err(|_| TlsError::AllocationFailed)?;
        body.try_push(session.len() as u8)
            .map_err(|_| TlsError::AllocationFailed)?;
        body.try_extend_from_slice(session)
            .map_err(|_| TlsError::AllocationFailed)?;
        u16_push(&mut body, TLS_AES_128_GCM_SHA256 as usize)?;
        body.try_push(0).map_err(|_| TlsError::AllocationFailed)?;
        let mut ext = Vector::new();
        u16_push(&mut ext, 43)?;
        u16_push(&mut ext, 2)?;
        ext.try_extend_from_slice(&[3, 4])
            .map_err(|_| TlsError::AllocationFailed)?;
        u16_push(&mut ext, 51)?;
        u16_push(&mut ext, 4 + server_share.len())?;
        u16_push(&mut ext, X25519_MLKEM768 as usize)?;
        u16_push(&mut ext, server_share.len())?;
        ext.try_extend_from_slice(&server_share)
            .map_err(|_| TlsError::AllocationFailed)?;
        u16_push(&mut body, ext.len())?;
        body.try_extend_from_slice(&ext)
            .map_err(|_| TlsError::AllocationFailed)?;
        let server_hello = handshake(2, &body)?;
        send_plain(&mut stream, &server_hello)?;
        let mut transcript = Transcript::new();
        transcript.update(client_hello);
        transcript.update(&server_hello);
        let mut schedule = KeySchedule::new();
        schedule.mix_shared_secret(&shared);
        shared.fill(0);
        let client_hs = schedule.traffic_secret(false, &transcript.hash());
        let server_hs = schedule.traffic_secret(true, &transcript.hash());
        let mut read_hs = RecordKeys::from_traffic_secret(&client_hs);
        let mut write_hs = RecordKeys::from_traffic_secret(&server_hs);
        let extensions = handshake(8, &[0, 0])?;
        transcript.update(&extensions);
        let mut cert_body = Vector::new();
        cert_body
            .try_push(0)
            .map_err(|_| TlsError::AllocationFailed)?;
        let mut list = Vector::new();
        for cert in &config.certificates {
            u24_push(&mut list, cert.len())?;
            list.try_extend_from_slice(cert)
                .map_err(|_| TlsError::AllocationFailed)?;
            u16_push(&mut list, 0)?;
        }
        u24_push(&mut cert_body, list.len())?;
        cert_body
            .try_extend_from_slice(&list)
            .map_err(|_| TlsError::AllocationFailed)?;
        let certificate = handshake(11, &cert_body)?;
        transcript.update(&certificate);
        let mut signed = Vector::new();
        signed
            .try_extend_from_slice(&[0x20; 64])
            .map_err(|_| TlsError::AllocationFailed)?;
        signed
            .try_extend_from_slice(b"TLS 1.3, server CertificateVerify\0")
            .map_err(|_| TlsError::AllocationFailed)?;
        signed
            .try_extend_from_slice(&transcript.hash())
            .map_err(|_| TlsError::AllocationFailed)?;
        let mut salt = [0; 32];
        mrml_runtime::fill_random(&mut salt).map_err(|_| TlsError::AuthenticationFailed)?;
        let mut signature = Vector::new();
        signature
            .try_resize(config.modulus.len(), 0)
            .map_err(|_| TlsError::AllocationFailed)?;
        rsa_pss_sha256_sign(
            &config.modulus,
            &config.private_exponent,
            &signed,
            &salt,
            &mut signature,
        )
        .map_err(|_| TlsError::AuthenticationFailed)?;
        salt.fill(0);
        let mut verify_body = Vector::new();
        u16_push(&mut verify_body, 0x0804)?;
        u16_push(&mut verify_body, signature.len())?;
        verify_body
            .try_extend_from_slice(&signature)
            .map_err(|_| TlsError::AllocationFailed)?;
        let certificate_verify = handshake(15, &verify_body)?;
        transcript.update(&certificate_verify);
        let finished = handshake(20, &finished_verify_data(&server_hs, &transcript.hash()))?;
        transcript.update(&finished);
        for msg in [
            &extensions[..],
            &certificate[..],
            &certificate_verify[..],
            &finished[..],
        ] {
            let mut protected = Vector::new();
            write_hs.seal(22, msg, &mut protected)?;
            stream.write_all(&protected).map_err(|_| TlsError::Io)?;
        }
        let hash = transcript.hash();
        schedule.finish_handshake();
        let incoming = record(&mut stream)?;
        let mut plain = Vector::new();
        if read_hs.open(&incoming, &mut plain)? != 22 || plain.len() != 36 || plain[0] != 20 {
            return Err(TlsError::Handshake);
        }
        let expected = finished_verify_data(&client_hs, &hash);
        let mut difference = 0;
        for (a, b) in expected.iter().zip(&plain[4..]) {
            difference |= a ^ b
        }
        if difference != 0 {
            return Err(TlsError::AuthenticationFailed);
        }
        Ok(Self {
            stream,
            read_keys: RecordKeys::from_traffic_secret(
                &schedule.application_traffic_secret(false, &hash),
            ),
            write_keys: RecordKeys::from_traffic_secret(
                &schedule.application_traffic_secret(true, &hash),
            ),
            pending: Vector::new(),
            at: 0,
        })
    }
    pub fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError> {
        for chunk in data.chunks(1 << 14) {
            let mut out = Vector::new();
            self.write_keys.seal(23, chunk, &mut out)?;
            self.stream.write_all(&out).map_err(|_| TlsError::Io)?;
        }
        Ok(())
    }
    pub fn read(&mut self, out: &mut [u8]) -> Result<usize, TlsError> {
        if self.at < self.pending.len() {
            let n = out.len().min(self.pending.len() - self.at);
            out[..n].copy_from_slice(&self.pending[self.at..self.at + n]);
            self.at += n;
            return Ok(n);
        }
        loop {
            let record = record(&mut self.stream)?;
            let mut plain = Vector::new();
            let kind = self.read_keys.open(&record, &mut plain)?;
            if kind == 21 {
                return Ok(0);
            }
            if kind == 23 {
                let n = out.len().min(plain.len());
                out[..n].copy_from_slice(&plain[..n]);
                self.pending = plain;
                self.at = n;
                return Ok(n);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TlsClientStream;
    use mrml_runtime::{TcpListener, environment_variable, read_file};

    #[test]
    fn authenticated_hybrid_client_server_interoperate_when_configured() {
        let Some(cert_path) = environment_variable("MRML_TLS_SERVER_TEST_CERT") else {
            return;
        };
        let Some(key_path) = environment_variable("MRML_TLS_SERVER_TEST_KEY") else {
            return;
        };
        let cert = read_file(&cert_path).unwrap();
        let key = read_file(&key_path).unwrap();
        let config = mrml_runtime::Shared::new(TlsServerConfig::from_pem(&cert, &key).unwrap());
        let listener = TcpListener::bind([127, 0, 0, 1], 0).unwrap();
        let port = listener.local_port().unwrap();
        assert!(
            mrml_runtime::spawn_detached(move || {
                let socket = listener.accept().unwrap();
                let mut tls = TlsServerStream::accept(socket, &config).unwrap();
                let mut request = [0; 4];
                assert_eq!(tls.read(&mut request).unwrap(), 4);
                assert_eq!(&request, b"ping");
                tls.write_all(b"pong").unwrap();
            })
            .is_ok()
        );
        let mut client = TlsClientStream::connect("localhost", port).unwrap();
        assert_eq!(client.negotiated_group(), X25519_MLKEM768);
        client.write_all(b"ping").unwrap();
        let mut response = [0; 4];
        assert_eq!(client.read(&mut response).unwrap(), 4);
        assert_eq!(&response, b"pong");
    }
}
