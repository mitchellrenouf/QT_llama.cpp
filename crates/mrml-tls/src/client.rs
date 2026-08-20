use crate::{
    HybridClientSecret, KeySchedule, RecordKeys, TLS_AES_128_GCM_SHA256, TlsError, Transcript,
    X25519_MLKEM768, finished_verify_data, hybrid_client_share, verify_server_chain,
};
use mrml_runtime::{TcpStream, Vector};

fn push_u16(out: &mut Vector<u8>, value: usize) -> Result<(), TlsError> {
    if value > u16::MAX as usize {
        return Err(TlsError::Handshake);
    }
    out.try_extend_from_slice(&(value as u16).to_be_bytes())
        .map_err(|_| TlsError::AllocationFailed)
}
fn push_u24(out: &mut Vector<u8>, value: usize) -> Result<(), TlsError> {
    if value > 0xff_ffff {
        return Err(TlsError::Handshake);
    }
    out.try_extend_from_slice(&[(value >> 16) as u8, (value >> 8) as u8, value as u8])
        .map_err(|_| TlsError::AllocationFailed)
}
fn extension(out: &mut Vector<u8>, kind: u16, value: &[u8]) -> Result<(), TlsError> {
    push_u16(out, kind as usize)?;
    push_u16(out, value.len())?;
    out.try_extend_from_slice(value)
        .map_err(|_| TlsError::AllocationFailed)
}

fn client_hello(host: &str) -> Result<(Vector<u8>, HybridClientSecret), TlsError> {
    if host.is_empty() || host.len() > 253 || !host.is_ascii() {
        return Err(TlsError::Handshake);
    }
    let (share, secret) = hybrid_client_share()?;
    let mut random = [0u8; 32];
    let mut session = [0u8; 32];
    mrml_runtime::fill_random(&mut random).map_err(|_| TlsError::AuthenticationFailed)?;
    mrml_runtime::fill_random(&mut session).map_err(|_| TlsError::AuthenticationFailed)?;
    let mut extensions = Vector::new();
    let mut sni = Vector::new();
    push_u16(&mut sni, host.len() + 3)?;
    sni.try_push(0).map_err(|_| TlsError::AllocationFailed)?;
    push_u16(&mut sni, host.len())?;
    sni.try_extend_from_slice(host.as_bytes())
        .map_err(|_| TlsError::AllocationFailed)?;
    extension(&mut extensions, 0, &sni)?;
    extension(&mut extensions, 43, &[2, 3, 4])?;
    extension(&mut extensions, 13, &[0, 4, 8, 4, 4, 1])?;
    extension(&mut extensions, 10, &[0, 2, 0x11, 0xec])?;
    let mut key_share = Vector::new();
    push_u16(&mut key_share, 4 + share.len())?;
    push_u16(&mut key_share, X25519_MLKEM768 as usize)?;
    push_u16(&mut key_share, share.len())?;
    key_share
        .try_extend_from_slice(&share)
        .map_err(|_| TlsError::AllocationFailed)?;
    extension(&mut extensions, 51, &key_share)?;
    let mut body = Vector::new();
    body.try_extend_from_slice(&[3, 3])
        .map_err(|_| TlsError::AllocationFailed)?;
    body.try_extend_from_slice(&random)
        .map_err(|_| TlsError::AllocationFailed)?;
    body.try_push(32).map_err(|_| TlsError::AllocationFailed)?;
    body.try_extend_from_slice(&session)
        .map_err(|_| TlsError::AllocationFailed)?;
    push_u16(&mut body, 2)?;
    push_u16(&mut body, TLS_AES_128_GCM_SHA256 as usize)?;
    body.try_extend_from_slice(&[1, 0])
        .map_err(|_| TlsError::AllocationFailed)?;
    push_u16(&mut body, extensions.len())?;
    body.try_extend_from_slice(&extensions)
        .map_err(|_| TlsError::AllocationFailed)?;
    let mut handshake = Vector::new();
    handshake
        .try_push(1)
        .map_err(|_| TlsError::AllocationFailed)?;
    push_u24(&mut handshake, body.len())?;
    handshake
        .try_extend_from_slice(&body)
        .map_err(|_| TlsError::AllocationFailed)?;
    Ok((handshake, secret))
}

fn record(stream: &mut TcpStream) -> Result<(u8, Vector<u8>), TlsError> {
    let mut header = [0u8; 5];
    stream.read_exact(&mut header).map_err(|_| TlsError::Io)?;
    if header[1..3] != [3, 3] {
        return Err(TlsError::InvalidRecord);
    }
    let length = u16::from_be_bytes([header[3], header[4]]) as usize;
    if length > 16640 {
        return Err(TlsError::InvalidRecord);
    }
    let mut body = Vector::new();
    body.try_resize(length, 0)
        .map_err(|_| TlsError::AllocationFailed)?;
    stream.read_exact(&mut body).map_err(|_| TlsError::Io)?;
    Ok((header[0], body))
}
fn send_plain_handshake(stream: &mut TcpStream, message: &[u8]) -> Result<(), TlsError> {
    if message.len() > u16::MAX as usize {
        return Err(TlsError::Handshake);
    }
    let header = [22, 3, 1, (message.len() >> 8) as u8, message.len() as u8];
    stream.write_all(&header).map_err(|_| TlsError::Io)?;
    stream.write_all(message).map_err(|_| TlsError::Io)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], TlsError> {
        let end = self.position.checked_add(n).ok_or(TlsError::Handshake)?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(TlsError::Handshake)?;
        self.position = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, TlsError> {
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
        self.position == self.bytes.len()
    }
}

fn server_share(message: &[u8]) -> Result<&[u8], TlsError> {
    let mut c = Cursor::new(message);
    if c.u8()? != 2 {
        return Err(TlsError::Handshake);
    }
    let len = c.u24()?;
    if len != message.len() - 4 {
        return Err(TlsError::Handshake);
    }
    if c.take(2)? != [3, 3] {
        return Err(TlsError::Handshake);
    }
    c.take(32)?;
    let sid = c.u8()? as usize;
    c.take(sid)?;
    if c.u16() != Ok(TLS_AES_128_GCM_SHA256) || c.u8()? != 0 {
        return Err(TlsError::Handshake);
    }
    let ext_len = c.u16()? as usize;
    let mut e = Cursor::new(c.take(ext_len)?);
    if !c.done() {
        return Err(TlsError::Handshake);
    }
    let mut version = false;
    let mut share = None;
    while !e.done() {
        let kind = e.u16()?;
        let value_len = e.u16()? as usize;
        let value = e.take(value_len)?;
        match kind {
            43 if value == [3, 4] => version = true,
            51 => {
                let mut k = Cursor::new(value);
                if k.u16()? != X25519_MLKEM768 {
                    return Err(TlsError::Handshake);
                }
                let length = k.u16()? as usize;
                let bytes = k.take(length)?;
                if !k.done() || length != 1120 {
                    return Err(TlsError::Handshake);
                }
                share = Some(bytes)
            }
            _ => {}
        }
    }
    if !version {
        return Err(TlsError::Handshake);
    }
    share.ok_or(TlsError::Handshake)
}

fn handshake_message(buffer: &mut Vector<u8>) -> Result<Option<Vector<u8>>, TlsError> {
    if buffer.len() < 4 {
        return Ok(None);
    }
    let len = (buffer[1] as usize) << 16 | (buffer[2] as usize) << 8 | buffer[3] as usize;
    if len > 0xff_ffff {
        return Err(TlsError::Handshake);
    }
    if buffer.len() < 4 + len {
        return Ok(None);
    }
    let mut result = Vector::new();
    result
        .try_extend_from_slice(&buffer[..4 + len])
        .map_err(|_| TlsError::AllocationFailed)?;
    for _ in 0..4 + len {
        buffer.remove(0);
    }
    Ok(Some(result))
}

fn certificate_chain(message: &[u8], storage: &mut Vector<Vector<u8>>) -> Result<(), TlsError> {
    let mut c = Cursor::new(message);
    if c.u8()? != 11 {
        return Err(TlsError::Handshake);
    }
    let length = c.u24()?;
    if length != message.len() - 4 {
        return Err(TlsError::Handshake);
    }
    let context = c.u8()? as usize;
    if context != 0 {
        c.take(context)?;
        return Err(TlsError::Handshake);
    }
    let list_len = c.u24()?;
    let list = c.take(list_len)?;
    if !c.done() {
        return Err(TlsError::Handshake);
    }
    let mut entries = Cursor::new(list);
    while !entries.done() {
        let der_len = entries.u24()?;
        let der = entries.take(der_len)?;
        let mut owned = Vector::new();
        owned
            .try_extend_from_slice(der)
            .map_err(|_| TlsError::AllocationFailed)?;
        storage
            .try_push(owned)
            .map_err(|_| TlsError::AllocationFailed)?;
        let ext = entries.u16()? as usize;
        entries.take(ext)?;
    }
    if storage.is_empty() {
        Err(TlsError::Certificate)
    } else {
        Ok(())
    }
}

fn verify_certificate_message(
    message: &[u8],
    transcript_hash: &[u8; 32],
    leaf: &crate::Certificate<'_>,
) -> Result<(), TlsError> {
    let mut c = Cursor::new(message);
    if c.u8()? != 15 {
        return Err(TlsError::Handshake);
    }
    let len = c.u24()?;
    if len != message.len() - 4 || c.u16()? != 0x0804 {
        return Err(TlsError::Handshake);
    }
    let signature_len = c.u16()? as usize;
    let signature = c.take(signature_len)?;
    if !c.done() {
        return Err(TlsError::Handshake);
    }
    let mut signed = Vector::new();
    signed
        .try_extend_from_slice(&[0x20; 64])
        .map_err(|_| TlsError::AllocationFailed)?;
    signed
        .try_extend_from_slice(b"TLS 1.3, server CertificateVerify\0")
        .map_err(|_| TlsError::AllocationFailed)?;
    signed
        .try_extend_from_slice(transcript_hash)
        .map_err(|_| TlsError::AllocationFailed)?;
    leaf.verify_tls_pss(&signed, signature)
        .map_err(|_| TlsError::AuthenticationFailed)
}

pub struct TlsClientStream {
    stream: TcpStream,
    read_keys: RecordKeys,
    write_keys: RecordKeys,
    pending: Vector<u8>,
    pending_position: usize,
}
impl TlsClientStream {
    pub fn connect(host: &str, port: u16) -> Result<Self, TlsError> {
        let mut stream = TcpStream::connect_host(host, port).map_err(|_| TlsError::Io)?;
        stream
            .set_read_timeout_millis(30_000)
            .map_err(|_| TlsError::Io)?;
        stream
            .set_write_timeout_millis(30_000)
            .map_err(|_| TlsError::Io)?;
        let (client_hello, hybrid) = client_hello(host)?;
        send_plain_handshake(&mut stream, &client_hello)?;
        let (mut kind, mut server_hello) = record(&mut stream)?;
        while kind == 20 {
            (kind, server_hello) = record(&mut stream)?;
        }
        if kind != 22 {
            return Err(TlsError::Handshake);
        }
        let mut shared = hybrid.complete(server_share(&server_hello)?)?;
        let mut transcript = Transcript::new();
        transcript.update(&client_hello);
        transcript.update(&server_hello);
        let mut schedule = KeySchedule::new();
        schedule.mix_shared_secret(&shared);
        shared.fill(0);
        let client_hs = schedule.traffic_secret(false, &transcript.hash());
        let server_hs = schedule.traffic_secret(true, &transcript.hash());
        let mut read_hs = RecordKeys::from_traffic_secret(&server_hs);
        let mut write_hs = RecordKeys::from_traffic_secret(&client_hs);
        let mut handshakes = Vector::new();
        let mut certificates = Vector::new();
        let mut got_extensions = false;
        let mut got_certificate = false;
        let mut got_verify = false;
        loop {
            let (kind, encrypted) = record(&mut stream)?;
            if kind == 20 {
                continue;
            }
            if kind != 23 {
                return Err(TlsError::Handshake);
            }
            let mut complete = Vector::new();
            complete
                .try_extend_from_slice(&[
                    23,
                    3,
                    3,
                    (encrypted.len() >> 8) as u8,
                    encrypted.len() as u8,
                ])
                .map_err(|_| TlsError::AllocationFailed)?;
            complete
                .try_extend_from_slice(&encrypted)
                .map_err(|_| TlsError::AllocationFailed)?;
            let mut plain = Vector::new();
            if read_hs
                .open(&complete, &mut plain)
                .map_err(|_| TlsError::InvalidRecord)?
                != 22
            {
                return Err(TlsError::Handshake);
            }
            handshakes
                .try_extend_from_slice(&plain)
                .map_err(|_| TlsError::AllocationFailed)?;
            while let Some(message) = handshake_message(&mut handshakes)? {
                match message[0] {
                    8 => {
                        if got_extensions {
                            return Err(TlsError::Handshake);
                        }
                        got_extensions = true;
                        transcript.update(&message)
                    }
                    11 => {
                        if !got_extensions || got_certificate {
                            return Err(TlsError::Handshake);
                        }
                        certificate_chain(&message, &mut certificates)?;
                        let refs: Vector<&[u8]> = certificates.iter().map(|v| &v[..]).collect();
                        verify_server_chain(host, &refs).map_err(|_| TlsError::Certificate)?;
                        got_certificate = true;
                        transcript.update(&message)
                    }
                    15 => {
                        if !got_certificate || got_verify {
                            return Err(TlsError::Handshake);
                        }
                        let leaf = crate::Certificate::parse(&certificates[0])
                            .map_err(|_| TlsError::Certificate)?;
                        verify_certificate_message(&message, &transcript.hash(), &leaf)?;
                        got_verify = true;
                        transcript.update(&message)
                    }
                    20 => {
                        if !got_verify {
                            return Err(TlsError::Handshake);
                        }
                        let mut f = Cursor::new(&message);
                        f.u8()?;
                        if f.u24()? != 32 {
                            return Err(TlsError::Handshake);
                        }
                        let expected = finished_verify_data(&server_hs, &transcript.hash());
                        let received = f.take(32)?;
                        let mut difference = 0u8;
                        for (a, b) in expected.iter().zip(received) {
                            difference |= a ^ b;
                        }
                        if difference != 0 {
                            return Err(TlsError::AuthenticationFailed);
                        }
                        transcript.update(&message);
                        schedule.finish_handshake();
                        let application_hash = transcript.hash();
                        let read_keys = RecordKeys::from_traffic_secret(
                            &schedule.application_traffic_secret(true, &application_hash),
                        );
                        let write_keys = RecordKeys::from_traffic_secret(
                            &schedule.application_traffic_secret(false, &application_hash),
                        );
                        let verify = finished_verify_data(&client_hs, &transcript.hash());
                        let mut finished = Vector::new();
                        finished
                            .try_push(20)
                            .map_err(|_| TlsError::AllocationFailed)?;
                        push_u24(&mut finished, 32)?;
                        finished
                            .try_extend_from_slice(&verify)
                            .map_err(|_| TlsError::AllocationFailed)?;
                        let mut protected = Vector::new();
                        write_hs.seal(22, &finished, &mut protected)?;
                        stream.write_all(&protected).map_err(|_| TlsError::Io)?;
                        return Ok(Self {
                            stream,
                            read_keys,
                            write_keys,
                            pending: Vector::new(),
                            pending_position: 0,
                        });
                    }
                    _ => return Err(TlsError::Handshake),
                }
            }
        }
    }
    pub fn write_all(&mut self, data: &[u8]) -> Result<(), TlsError> {
        for chunk in data.chunks(1 << 14) {
            let mut record = Vector::new();
            self.write_keys.seal(23, chunk, &mut record)?;
            self.stream.write_all(&record).map_err(|_| TlsError::Io)?;
        }
        Ok(())
    }
    pub fn read(&mut self, output: &mut [u8]) -> Result<usize, TlsError> {
        if self.pending_position < self.pending.len() {
            let n = output.len().min(self.pending.len() - self.pending_position);
            output[..n]
                .copy_from_slice(&self.pending[self.pending_position..self.pending_position + n]);
            self.pending_position += n;
            if self.pending_position == self.pending.len() {
                self.pending.clear();
                self.pending_position = 0;
            }
            return Ok(n);
        }
        loop {
            let (kind, body) = record(&mut self.stream)?;
            if kind != 23 {
                return Err(TlsError::InvalidRecord);
            }
            let mut complete = Vector::new();
            complete
                .try_extend_from_slice(&[23, 3, 3, (body.len() >> 8) as u8, body.len() as u8])
                .map_err(|_| TlsError::AllocationFailed)?;
            complete
                .try_extend_from_slice(&body)
                .map_err(|_| TlsError::AllocationFailed)?;
            let inner = self.read_keys.open(&complete, &mut self.pending)?;
            match inner {
                23 => return self.read(output),
                21 => {
                    self.pending.clear();
                    return Ok(0);
                }
                22 => {
                    self.pending.clear();
                    continue;
                }
                _ => return Err(TlsError::InvalidRecord),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_authenticated_handshake_when_configured() {
        let Some(host) = mrml_runtime::environment_variable("MRML_TLS_LIVE_HOST") else {
            return;
        };
        let port = mrml_runtime::environment_variable("MRML_TLS_LIVE_PORT")
            .and_then(|p| p.parse().ok())
            .unwrap_or(443);
        let mut tls = TlsClientStream::connect(&host, port).unwrap();
        tls.write_all(b"HEAD / HTTP/1.1\r\nHost: huggingface.co\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = [0u8; 64];
        let read = tls.read(&mut response).unwrap();
        assert!(read >= 12);
        assert_eq!(&response[..5], b"HTTP/");
    }
}
