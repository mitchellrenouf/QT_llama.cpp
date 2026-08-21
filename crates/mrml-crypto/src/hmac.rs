use crate::Sha256;

pub fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut normalized = [0u8; 64];
    if key.len() > normalized.len() {
        normalized[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        normalized[..key.len()].copy_from_slice(key);
    }
    let mut inner_key = normalized;
    let mut outer_key = normalized;
    for byte in &mut inner_key {
        *byte ^= 0x36;
    }
    for byte in &mut outer_key {
        *byte ^= 0x5c;
    }
    let mut inner = Sha256::new();
    inner.update(&inner_key);
    for part in parts {
        inner.update(part);
    }
    let inner_digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(&outer_key);
    outer.update(&inner_digest);
    outer.finalize()
}
