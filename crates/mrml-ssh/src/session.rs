use mrml_runtime::{Text, Vector};
use crate::{BinaryReader, BinaryWriter, ProtocolError, negotiate};

pub const KEY_EXCHANGE: &[&str] = &["curve25519-sha256"];
pub const HOST_KEYS: &[&str] = &["rsa-sha2-256", "ssh-rsa"];
pub const CIPHERS: &[&str] = &["aes128-ctr"];
pub const MACS: &[&str] = &["hmac-sha2-256"];
pub const COMPRESSIONS: &[&str] = &["none"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KexAlgorithms {
    pub key_exchange: Vector<Text>,
    pub host_key: Vector<Text>,
    pub client_cipher: Vector<Text>,
    pub server_cipher: Vector<Text>,
    pub client_mac: Vector<Text>,
    pub server_mac: Vector<Text>,
    pub client_compression: Vector<Text>,
    pub server_compression: Vector<Text>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiatedAlgorithms {
    pub key_exchange: Text,
    pub host_key: Text,
    pub client_cipher: Text,
    pub server_cipher: Text,
    pub client_mac: Text,
    pub server_mac: Text,
}

pub fn build_kex_init() -> Result<Vector<u8>, ProtocolError> {
    let mut cookie = [0u8; 16];
    mrml_runtime::fill_random(&mut cookie).map_err(|_| ProtocolError::Entropy)?;
    let mut writer = BinaryWriter::new();
    writer.byte(20);
    for byte in cookie { writer.byte(byte); }
    writer.name_list(KEY_EXCHANGE)?;
    writer.name_list(HOST_KEYS)?;
    writer.name_list(CIPHERS)?;
    writer.name_list(CIPHERS)?;
    writer.name_list(MACS)?;
    writer.name_list(MACS)?;
    writer.name_list(COMPRESSIONS)?;
    writer.name_list(COMPRESSIONS)?;
    writer.name_list(&[])?;
    writer.name_list(&[])?;
    writer.boolean(false);
    writer.u32(0);
    Ok(writer.finish())
}

pub fn negotiate_kex_init(message: &[u8]) -> Result<(KexAlgorithms, NegotiatedAlgorithms), ProtocolError> {
    let mut reader = BinaryReader::new(message);
    if reader.byte()? != 20 { return Err(ProtocolError::InvalidPacket); }
    for _ in 0..16 { reader.byte()?; }
    let kex = owned(reader.name_list()?);
    let host = owned(reader.name_list()?);
    let c2s_cipher = owned(reader.name_list()?);
    let s2c_cipher = owned(reader.name_list()?);
    let c2s_mac = owned(reader.name_list()?);
    let s2c_mac = owned(reader.name_list()?);
    let c2s_compression = owned(reader.name_list()?);
    let s2c_compression = owned(reader.name_list()?);
    reader.name_list()?;
    reader.name_list()?;
    reader.boolean()?;
    reader.u32()?;
    if reader.remaining() != 0 { return Err(ProtocolError::InvalidPacket); }
    require_none(&c2s_compression)?;
    require_none(&s2c_compression)?;
    let algorithms = KexAlgorithms { key_exchange:kex, host_key:host, client_cipher:c2s_cipher, server_cipher:s2c_cipher, client_mac:c2s_mac, server_mac:s2c_mac, client_compression:c2s_compression, server_compression:s2c_compression };
    let selected = NegotiatedAlgorithms {
        key_exchange: choose(KEY_EXCHANGE, &algorithms.key_exchange)?.into(),
        host_key: choose(HOST_KEYS, &algorithms.host_key)?.into(),
        client_cipher: choose(CIPHERS, &algorithms.client_cipher)?.into(),
        server_cipher: choose(CIPHERS, &algorithms.server_cipher)?.into(),
        client_mac: choose(MACS, &algorithms.client_mac)?.into(),
        server_mac: choose(MACS, &algorithms.server_mac)?.into(),
    };
    Ok((algorithms, selected))
}

fn owned(values: Vector<&str>) -> Vector<Text> { values.into_iter().map(Into::into).collect() }
fn choose<'a>(ours: &'a [&'a str], theirs: &[Text]) -> Result<&'a str, ProtocolError> {
    let borrowed: Vector<&str> = theirs.iter().map(Text::as_str).collect();
    negotiate(ours, &borrowed)
}
fn require_none(values: &[Text]) -> Result<(), ProtocolError> {
    values.iter().any(|value| value == "none").then_some(()).ok_or(ProtocolError::NoCommonAlgorithm)
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn self_proposal_negotiates_complete_secure_profile() { let message=build_kex_init().unwrap();let(_,selected)=negotiate_kex_init(&message).unwrap();assert_eq!(selected.key_exchange,"curve25519-sha256");assert_eq!(selected.host_key,"rsa-sha2-256");assert_eq!(selected.client_cipher,"aes128-ctr");assert_eq!(selected.server_mac,"hmac-sha2-256"); }
    #[test] fn rejects_trailing_data() { let mut message=build_kex_init().unwrap();message.push(0);assert!(matches!(negotiate_kex_init(&message),Err(ProtocolError::InvalidPacket))); }
}
