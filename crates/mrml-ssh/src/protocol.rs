use core::fmt;
use mrml_crypto::{Sha256, aes128_ctr_xor, hmac_sha256, x25519_public, x25519_shared};
use mrml_runtime::{Text, Vector};

const MAX_STRING: usize = 1024 * 1024;
const MAX_IDENTIFICATION_LINES: usize = 50;
const MAX_IDENTIFICATION_LINE: usize = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    Truncated,
    Length,
    InvalidUtf8,
    InvalidIdentification,
    InvalidNameList,
    NoCommonAlgorithm,
    InvalidPublicKey,
    InvalidPacket,
    Entropy,
    Authentication,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Identification {
    pub protocol: Text,
    pub software: Text,
    pub comments: Option<Text>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlgorithmProposal<'a> {
    pub key_exchange: &'a [&'a str],
    pub host_key: &'a [&'a str],
    pub cipher: &'a [&'a str],
    pub mac: &'a [&'a str],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExchangeKeys {
    pub client_public: [u8; 32],
    pub shared_secret: [u8; 32],
    pub exchange_hash: [u8; 32],
    pub client_iv: [u8; 16],
    pub server_iv: [u8; 16],
    pub client_key: [u8; 32],
    pub server_key: [u8; 32],
    pub client_mac: [u8; 32],
    pub server_mac: [u8; 32],
}

pub struct BinaryReader<'a> {
    source: &'a [u8],
    cursor: usize,
}

impl<'a> BinaryReader<'a> {
    pub const fn new(source: &'a [u8]) -> Self { Self { source, cursor: 0 } }
    pub fn remaining(&self) -> usize { self.source.len() - self.cursor }
    pub fn byte(&mut self) -> Result<u8, ProtocolError> {
        let value = *self.source.get(self.cursor).ok_or(ProtocolError::Truncated)?;
        self.cursor += 1;
        Ok(value)
    }
    pub fn boolean(&mut self) -> Result<bool, ProtocolError> { Ok(self.byte()? != 0) }
    pub fn u32(&mut self) -> Result<u32, ProtocolError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().map_err(|_| ProtocolError::Truncated)?;
        Ok(u32::from_be_bytes(bytes))
    }
    pub fn string(&mut self) -> Result<&'a [u8], ProtocolError> {
        let length = self.u32()? as usize;
        if length > MAX_STRING { return Err(ProtocolError::Length); }
        self.take(length)
    }
    pub fn text(&mut self) -> Result<&'a str, ProtocolError> {
        core::str::from_utf8(self.string()?).map_err(|_| ProtocolError::InvalidUtf8)
    }
    pub fn name_list(&mut self) -> Result<Vector<&'a str>, ProtocolError> {
        let text = self.text()?;
        if text.is_empty() { return Ok(Vector::new()); }
        let mut names = Vector::new();
        for name in text.split(',') {
            if name.is_empty() || name.bytes().any(|b| b <= 0x20 || b >= 0x7f) {
                return Err(ProtocolError::InvalidNameList);
            }
            names.push(name);
        }
        Ok(names)
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], ProtocolError> {
        let end = self.cursor.checked_add(length).ok_or(ProtocolError::Length)?;
        let value = self.source.get(self.cursor..end).ok_or(ProtocolError::Truncated)?;
        self.cursor = end;
        Ok(value)
    }
}

pub struct BinaryWriter { bytes: Vector<u8> }

impl BinaryWriter {
    pub fn new() -> Self { Self { bytes: Vector::new() } }
    pub fn byte(&mut self, value: u8) { self.bytes.push(value); }
    pub fn boolean(&mut self, value: bool) { self.byte(value as u8); }
    pub fn u32(&mut self, value: u32) { self.bytes.extend(value.to_be_bytes()); }
    pub fn string(&mut self, value: &[u8]) -> Result<(), ProtocolError> {
        if value.len() > MAX_STRING || value.len() > u32::MAX as usize { return Err(ProtocolError::Length); }
        self.u32(value.len() as u32);
        self.bytes.extend(value.iter().copied());
        Ok(())
    }
    pub fn name_list(&mut self, values: &[&str]) -> Result<(), ProtocolError> {
        let mut text = Text::new();
        for (index, value) in values.iter().enumerate() {
            if value.is_empty() || value.bytes().any(|b| b <= 0x20 || b >= 0x7f || b == b',') {
                return Err(ProtocolError::InvalidNameList);
            }
            if index != 0 { text.push(','); }
            text.push_str(value);
        }
        self.string(text.as_bytes())
    }
    pub fn finish(self) -> Vector<u8> { self.bytes }
}

pub fn parse_identification(source: &[u8]) -> Result<(Identification, usize), ProtocolError> {
    let mut start = 0usize;
    for _ in 0..MAX_IDENTIFICATION_LINES {
        let relative = source.get(start..).ok_or(ProtocolError::Truncated)?
            .windows(2).position(|pair| pair == b"\r\n").ok_or(ProtocolError::Truncated)?;
        let end = start.checked_add(relative).ok_or(ProtocolError::Length)?;
        if end - start > MAX_IDENTIFICATION_LINE { return Err(ProtocolError::InvalidIdentification); }
        let line = core::str::from_utf8(&source[start..end]).map_err(|_| ProtocolError::InvalidUtf8)?;
        start = end + 2;
        if let Some(rest) = line.strip_prefix("SSH-") {
            let (protocol, software_and_comments) = rest.split_once('-').ok_or(ProtocolError::InvalidIdentification)?;
            if protocol != "2.0" { return Err(ProtocolError::InvalidIdentification); }
            let (software, comments) = match software_and_comments.split_once(' ') {
                Some((software, comments)) => (software, Some(comments)), None => (software_and_comments, None),
            };
            if software.is_empty() || software.bytes().any(|b| b <= 0x20 || b >= 0x7f) { return Err(ProtocolError::InvalidIdentification); }
            return Ok((Identification { protocol: protocol.into(), software: software.into(), comments: comments.map(Into::into) }, start));
        }
    }
    Err(ProtocolError::InvalidIdentification)
}

pub fn negotiate<'a>(client: &'a [&'a str], server: &[&str]) -> Result<&'a str, ProtocolError> {
    client.iter().copied().find(|choice| server.contains(choice)).ok_or(ProtocolError::NoCommonAlgorithm)
}

/// Encodes the pre-NEWKEYS SSH binary packet form. The packet length excludes
/// its own four-byte field and the random padding is never exposed as payload.
pub fn encode_plain_packet(payload: &[u8], block_size: usize) -> Result<Vector<u8>, ProtocolError> {
    if payload.is_empty() || payload.len() > MAX_STRING || block_size < 8 || !block_size.is_power_of_two() {
        return Err(ProtocolError::InvalidPacket);
    }
    let body = payload.len().checked_add(5).ok_or(ProtocolError::Length)?;
    let aligned = body.checked_add(block_size - 1).ok_or(ProtocolError::Length)? & !(block_size - 1);
    let padding = aligned.checked_sub(payload.len() + 5).ok_or(ProtocolError::Length)?;
    let padding = if padding < 4 { padding + block_size } else { padding };
    if padding > u8::MAX as usize { return Err(ProtocolError::InvalidPacket); }
    let packet_length = payload.len().checked_add(padding + 1).ok_or(ProtocolError::Length)?;
    let mut output = Vector::new();
    output.extend((packet_length as u32).to_be_bytes());
    output.push(padding as u8);
    output.extend(payload.iter().copied());
    let start = output.len();
    output.try_resize(start + padding, 0).map_err(|_| ProtocolError::Length)?;
    mrml_runtime::fill_random(&mut output[start..]).map_err(|_| ProtocolError::Entropy)?;
    Ok(output)
}

pub fn decode_plain_packet(source: &[u8], block_size: usize) -> Result<(&[u8], usize), ProtocolError> {
    if block_size < 8 || !block_size.is_power_of_two() { return Err(ProtocolError::InvalidPacket); }
    let header: [u8;4] = source.get(..4).ok_or(ProtocolError::Truncated)?.try_into().map_err(|_| ProtocolError::Truncated)?;
    let length = u32::from_be_bytes(header) as usize;
    if length < 6 || length > MAX_STRING + 256 || (length + 4) % block_size != 0 { return Err(ProtocolError::InvalidPacket); }
    let total = length.checked_add(4).ok_or(ProtocolError::Length)?;
    let packet = source.get(..total).ok_or(ProtocolError::Truncated)?;
    let padding = packet[4] as usize;
    if padding < 4 || padding + 1 > length { return Err(ProtocolError::InvalidPacket); }
    let payload_end = total - padding;
    if payload_end <= 5 { return Err(ProtocolError::InvalidPacket); }
    Ok((&packet[5..payload_end], total))
}

pub struct EncryptedPacketWriter {
    key: [u8; 16],
    counter: [u8; 16],
    mac_key: [u8; 32],
    sequence: u32,
}

impl EncryptedPacketWriter {
    pub const fn new(key: [u8;16], counter: [u8;16], mac_key: [u8;32]) -> Self {
        Self { key, counter, mac_key, sequence: 0 }
    }
    pub fn encode(&mut self, payload: &[u8]) -> Result<Vector<u8>, ProtocolError> {
        let plain = encode_plain_packet(payload, 16)?;
        let mac = hmac_sha256(&self.mac_key, &[&self.sequence.to_be_bytes(), &plain]);
        let mut output = Vector::new();
        output.try_resize(plain.len(), 0).map_err(|_| ProtocolError::Length)?;
        self.counter = aes128_ctr_xor(&self.key, self.counter, &plain, &mut output).map_err(|_| ProtocolError::Length)?;
        output.extend(mac);
        self.sequence = self.sequence.wrapping_add(1);
        Ok(output)
    }
}

pub struct EncryptedPacketReader {
    key: [u8; 16],
    counter: [u8; 16],
    mac_key: [u8; 32],
    sequence: u32,
}

impl EncryptedPacketReader {
    pub const fn new(key: [u8;16], counter: [u8;16], mac_key: [u8;32]) -> Self {
        Self { key, counter, mac_key, sequence: 0 }
    }
    pub fn decode(&mut self, source: &[u8]) -> Result<(Vector<u8>, usize), ProtocolError> {
        if source.len() < 16 + 32 { return Err(ProtocolError::Truncated); }
        let mut first = [0u8;16];
        aes128_ctr_xor(&self.key, self.counter, &source[..16], &mut first).map_err(|_| ProtocolError::Length)?;
        let length = u32::from_be_bytes(first[..4].try_into().map_err(|_| ProtocolError::Truncated)?) as usize;
        let encrypted_len = length.checked_add(4).ok_or(ProtocolError::Length)?;
        if encrypted_len < 16 || encrypted_len > MAX_STRING + 260 || encrypted_len % 16 != 0 { return Err(ProtocolError::InvalidPacket); }
        let total = encrypted_len.checked_add(32).ok_or(ProtocolError::Length)?;
        if source.len() < total { return Err(ProtocolError::Truncated); }
        let mut plain = Vector::new();
        plain.try_resize(encrypted_len, 0).map_err(|_| ProtocolError::Length)?;
        let next = aes128_ctr_xor(&self.key, self.counter, &source[..encrypted_len], &mut plain).map_err(|_| ProtocolError::Length)?;
        let expected = hmac_sha256(&self.mac_key, &[&self.sequence.to_be_bytes(), &plain]);
        let mut difference = 0u8;
        for (left,right) in expected.iter().zip(&source[encrypted_len..total]) { difference |= left ^ right; }
        if difference != 0 { return Err(ProtocolError::Authentication); }
        let (payload, _) = decode_plain_packet(&plain, 16)?;
        let mut owned = Vector::new(); owned.extend(payload.iter().copied());
        self.counter = next;
        self.sequence = self.sequence.wrapping_add(1);
        Ok((owned,total))
    }
}

pub fn derive_exchange_keys(
    client_secret: [u8; 32], server_public: [u8; 32], transcript: &[&[u8]], session_id: Option<[u8; 32]>,
) -> Result<ExchangeKeys, ProtocolError> {
    let client_public = x25519_public(client_secret);
    let shared_secret = x25519_shared(client_secret, server_public).ok_or(ProtocolError::InvalidPublicKey)?;
    let mut hash = Sha256::new();
    for part in transcript { hash.update(part); }
    hash.update(&ssh_mpint(&shared_secret));
    let exchange_hash = hash.finalize();
    let session = session_id.unwrap_or(exchange_hash);
    Ok(ExchangeKeys {
        client_public, shared_secret, exchange_hash,
        client_iv: derive::<16>(&shared_secret, &exchange_hash, b'A', &session),
        server_iv: derive::<16>(&shared_secret, &exchange_hash, b'B', &session),
        client_key: derive::<32>(&shared_secret, &exchange_hash, b'C', &session),
        server_key: derive::<32>(&shared_secret, &exchange_hash, b'D', &session),
        client_mac: derive::<32>(&shared_secret, &exchange_hash, b'E', &session),
        server_mac: derive::<32>(&shared_secret, &exchange_hash, b'F', &session),
    })
}

fn derive<const N: usize>(shared: &[u8; 32], hash: &[u8; 32], letter: u8, session: &[u8; 32]) -> [u8; N] {
    let encoded = ssh_mpint(shared);
    let mut digest = Sha256::new();
    digest.update(&encoded);
    digest.update(hash);
    digest.update(&[letter]);
    digest.update(session);
    let material = digest.finalize();
    let mut result = [0u8; N];
    result.copy_from_slice(&material[..N]);
    result
}

fn ssh_mpint(value: &[u8]) -> Vector<u8> {
    let first = value.iter().position(|byte| *byte != 0).unwrap_or(value.len());
    let bytes = &value[first..];
    let extra = bytes.first().is_some_and(|byte| byte & 0x80 != 0) as usize;
    let mut output = Vector::new();
    output.extend(((bytes.len() + extra) as u32).to_be_bytes());
    if extra != 0 { output.push(0); }
    output.extend(bytes.iter().copied());
    output
}

impl Default for BinaryWriter { fn default() -> Self { Self::new() } }
impl fmt::Display for ProtocolError { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(match self { Self::Truncated => "truncated SSH message", Self::Length => "SSH field exceeds length limit", Self::InvalidUtf8 => "SSH text is not UTF-8", Self::InvalidIdentification => "invalid SSH identification", Self::InvalidNameList => "invalid SSH algorithm name-list", Self::NoCommonAlgorithm => "no mutually supported SSH algorithm", Self::InvalidPublicKey => "invalid SSH Curve25519 public key", Self::InvalidPacket => "invalid SSH binary packet", Self::Entropy => "operating-system entropy is unavailable", Self::Authentication => "SSH packet authentication failed" }) } }
impl core::error::Error for ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn binary_fields_round_trip_and_reject_truncation() { let mut w=BinaryWriter::new();w.byte(20);w.boolean(true);w.u32(7);w.string(b"hello").unwrap();w.name_list(&["curve25519-sha256","diffie-hellman-group14-sha256"]).unwrap();let bytes=w.finish();let mut r=BinaryReader::new(&bytes);assert_eq!(r.byte(),Ok(20));assert_eq!(r.boolean(),Ok(true));assert_eq!(r.u32(),Ok(7));assert_eq!(r.string(),Ok(&b"hello"[..]));assert_eq!(r.name_list().unwrap()[0],"curve25519-sha256");assert_eq!(r.remaining(),0);assert_eq!(BinaryReader::new(b"\0\0\0\x05x").string(),Err(ProtocolError::Truncated)); }
    #[test] fn identification_allows_banners_but_requires_ssh_two() { let (id,used)=parse_identification(b"notice\r\nSSH-2.0-mrml_0.4 test\r\nrest").unwrap();assert_eq!(id.software,"mrml_0.4");assert_eq!(id.comments.as_deref(),Some("test"));assert_eq!(used,31);assert_eq!(parse_identification(b"SSH-1.5-old\r\n"),Err(ProtocolError::InvalidIdentification)); }
    #[test] fn negotiation_respects_client_preference() { assert_eq!(negotiate(&["a","b"],&["b","a"]),Ok("a"));assert_eq!(negotiate(&["a"],&["b"]),Err(ProtocolError::NoCommonAlgorithm)); }
    #[test] fn curve25519_exchange_derives_matching_material() { let a=[7u8;32];let b=[9u8;32];let ap=x25519_public(a);let bp=x25519_public(b);let left=derive_exchange_keys(a,bp,&[b"client",b"server",&ap,&bp],None).unwrap();let shared=x25519_shared(b,ap).unwrap();assert_eq!(left.shared_secret,shared);assert_ne!(left.client_key,left.server_key); }
    #[test] fn plain_packets_round_trip_and_validate_padding() { let packet=encode_plain_packet(b"\x14hello",8).unwrap();assert_eq!(packet.len()%8,0);let(payload,used)=decode_plain_packet(&packet,8).unwrap();assert_eq!(payload,b"\x14hello");assert_eq!(used,packet.len());let mut bad=packet;bad[4]=3;assert_eq!(decode_plain_packet(&bad,8),Err(ProtocolError::InvalidPacket)); }
    #[test] fn encrypted_packets_authenticate_sequence_and_contents() { let key=[1;16];let iv=[2;16];let mac=[3;32];let mut writer=EncryptedPacketWriter::new(key,iv,mac);let first=writer.encode(b"first").unwrap();let second=writer.encode(b"second").unwrap();let mut reader=EncryptedPacketReader::new(key,iv,mac);assert_eq!(&*reader.decode(&first).unwrap().0,b"first");assert_eq!(&*reader.decode(&second).unwrap().0,b"second");let mut bad=first;let end=bad.len()-1;bad[end]^=1;let mut reader=EncryptedPacketReader::new(key,iv,mac);assert_eq!(reader.decode(&bad),Err(ProtocolError::Authentication)); }
}
