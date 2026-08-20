#![no_std]

use core::fmt;
use mrml_crypto::{Sha256, aes128_gcm_open, aes128_gcm_seal, hkdf_expand_label, hkdf_extract};
use mrml_runtime::Vector;

pub const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
pub const X25519_MLKEM768: u16 = 0x11ec;
const MAX_PLAINTEXT: usize = 1 << 14;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TlsError { InvalidRecord, AuthenticationFailed, SequenceExhausted, AllocationFailed }

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::InvalidRecord => "invalid TLS record", Self::AuthenticationFailed => "TLS record authentication failed", Self::SequenceExhausted => "TLS record sequence exhausted", Self::AllocationFailed => "TLS allocation failed" })
    }
}
impl core::error::Error for TlsError {}

pub struct Transcript(Sha256);
impl Transcript {
    pub const fn new() -> Self { Self(Sha256::new()) }
    pub fn update(&mut self, handshake: &[u8]) { self.0.update(handshake); }
    pub fn hash(&self) -> [u8; 32] { self.0.clone().finalize() }
}
impl Default for Transcript { fn default() -> Self { Self::new() } }

pub struct KeySchedule { secret: [u8; 32] }
impl KeySchedule {
    pub fn new() -> Self { Self { secret: hkdf_extract(&[0; 32], &[0; 32]) } }
    pub fn mix_shared_secret(&mut self, shared_secret: &[u8]) {
        let mut derived = [0u8; 32];
        let empty_hash = Sha256::digest(&[]);
        let _ = hkdf_expand_label(&self.secret, b"derived", &empty_hash, &mut derived);
        self.secret = hkdf_extract(&derived, shared_secret);
        derived.fill(0);
    }
    pub fn traffic_secret(&self, server: bool, transcript_hash: &[u8; 32]) -> [u8; 32] {
        let mut output = [0u8; 32];
        let label: &[u8] = if server { b"s hs traffic" } else { b"c hs traffic" };
        let _ = hkdf_expand_label(&self.secret, label, transcript_hash, &mut output);
        output
    }
}
impl Default for KeySchedule { fn default() -> Self { Self::new() } }
impl Drop for KeySchedule { fn drop(&mut self) { self.secret.fill(0); } }

pub struct RecordKeys { key: [u8; 16], iv: [u8; 12], sequence: u64 }
impl RecordKeys {
    pub fn from_traffic_secret(secret: &[u8; 32]) -> Self {
        let mut key = [0u8; 16]; let mut iv = [0u8; 12];
        let _ = hkdf_expand_label(secret, b"key", &[], &mut key);
        let _ = hkdf_expand_label(secret, b"iv", &[], &mut iv);
        Self { key, iv, sequence: 0 }
    }
    fn nonce(&self) -> [u8; 12] {
        let mut nonce = self.iv;
        let sequence = self.sequence.to_be_bytes();
        for i in 0..8 { nonce[4 + i] ^= sequence[i]; }
        nonce
    }
    fn advance(&mut self) -> Result<(), TlsError> { self.sequence = self.sequence.checked_add(1).ok_or(TlsError::SequenceExhausted)?; Ok(()) }

    pub fn seal(&mut self, content_type: u8, plaintext: &[u8], output: &mut Vector<u8>) -> Result<(), TlsError> {
        if plaintext.len() > MAX_PLAINTEXT { return Err(TlsError::InvalidRecord); }
        let encrypted_length = plaintext.len().checked_add(17).ok_or(TlsError::InvalidRecord)?;
        let header = [23, 3, 3, (encrypted_length >> 8) as u8, encrypted_length as u8];
        output.try_extend_from_slice(&header).map_err(|_| TlsError::AllocationFailed)?;
        let start = output.len();
        output.try_extend_from_slice(plaintext).map_err(|_| TlsError::AllocationFailed)?;
        output.try_push(content_type).map_err(|_| TlsError::AllocationFailed)?;
        output.try_extend_from_slice(&[0; 16]).map_err(|_| TlsError::AllocationFailed)?;
        let nonce = self.nonce();
        let mut tag = [0u8; 16];
        let mut inner = Vector::new(); inner.try_extend_from_slice(plaintext).map_err(|_| TlsError::AllocationFailed)?; inner.try_push(content_type).map_err(|_| TlsError::AllocationFailed)?;
        if !aes128_gcm_seal(&self.key, &nonce, &header, &inner, &mut output[start..start + inner.len()], &mut tag) { return Err(TlsError::InvalidRecord); }
        output[start + inner.len()..].copy_from_slice(&tag);
        self.advance()
    }

    pub fn open(&mut self, record: &[u8], output: &mut Vector<u8>) -> Result<u8, TlsError> {
        if record.len() < 5 + 17 || record[0] != 23 || record[1..3] != [3, 3] { return Err(TlsError::InvalidRecord); }
        let length = u16::from_be_bytes([record[3], record[4]]) as usize;
        if length != record.len() - 5 || length > MAX_PLAINTEXT + 256 { return Err(TlsError::InvalidRecord); }
        let ciphertext_length = length - 16; let start = output.len();
        output.try_extend_from_slice(&record[5..5 + ciphertext_length]).map_err(|_| TlsError::AllocationFailed)?;
        let tag: &[u8; 16] = record[5 + ciphertext_length..].try_into().map_err(|_| TlsError::InvalidRecord)?;
        let nonce = self.nonce();
        let mut plain = Vector::new(); plain.try_extend_from_slice(&record[5..5 + ciphertext_length]).map_err(|_| TlsError::AllocationFailed)?;
        if !aes128_gcm_open(&self.key, &nonce, &record[..5], &record[5..5 + ciphertext_length], tag, &mut plain) { output.truncate(start); return Err(TlsError::AuthenticationFailed); }
        let Some(position) = plain.iter().rposition(|byte| *byte != 0) else { output.truncate(start); return Err(TlsError::InvalidRecord); };
        let content_type = plain[position]; output.truncate(start); output.try_extend_from_slice(&plain[..position]).map_err(|_| TlsError::AllocationFailed)?;
        self.advance()?; Ok(content_type)
    }
}
impl Drop for RecordKeys { fn drop(&mut self) { self.key.fill(0); self.iv.fill(0); self.sequence = 0; } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn encrypted_records_round_trip_and_reject_tampering() {
        let secret = [7u8; 32]; let mut sender = RecordKeys::from_traffic_secret(&secret); let mut receiver = RecordKeys::from_traffic_secret(&secret);
        let mut record = Vector::new(); sender.seal(22, b"handshake", &mut record).unwrap();
        let mut plain = Vector::new(); assert_eq!(receiver.open(&record, &mut plain).unwrap(), 22); assert_eq!(&plain[..], b"handshake");
        let mut receiver = RecordKeys::from_traffic_secret(&secret); record[8] ^= 1; assert_eq!(receiver.open(&record, &mut Vector::new()), Err(TlsError::AuthenticationFailed));
    }
    #[test] fn key_schedule_separates_client_and_server() {
        let mut schedule = KeySchedule::new(); schedule.mix_shared_secret(&[9u8; 64]); let hash = Sha256::digest(b"transcript");
        assert_ne!(schedule.traffic_secret(false, &hash), schedule.traffic_secret(true, &hash));
    }
}
