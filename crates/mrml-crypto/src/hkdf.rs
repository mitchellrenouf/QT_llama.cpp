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

pub fn hkdf_extract(salt: &[u8], input_key_material: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, &[input_key_material])
}

pub fn hkdf_expand(secret: &[u8], info: &[u8], output: &mut [u8]) -> bool {
    if output.len() > 255 * 32 {
        return false;
    }
    let mut previous = [0u8; 32];
    let mut previous_length = 0usize;
    for (index, chunk) in output.chunks_mut(32).enumerate() {
        let counter = [(index + 1) as u8];
        previous = hmac_sha256(secret, &[&previous[..previous_length], info, &counter]);
        chunk.copy_from_slice(&previous[..chunk.len()]);
        previous_length = 32;
    }
    true
}

pub fn hkdf_expand_label(secret: &[u8], label: &[u8], context: &[u8], output: &mut [u8]) -> bool {
    const PREFIX: &[u8] = b"tls13 ";
    if output.len() > u16::MAX as usize
        || PREFIX.len() + label.len() > u8::MAX as usize
        || context.len() > u8::MAX as usize
    {
        return false;
    }
    let mut info = mrml_runtime::Vector::new();
    if info
        .try_extend_from_slice(&(output.len() as u16).to_be_bytes())
        .is_err()
        || info.try_push((PREFIX.len() + label.len()) as u8).is_err()
        || info.try_extend_from_slice(PREFIX).is_err()
        || info.try_extend_from_slice(label).is_err()
        || info.try_push(context.len() as u8).is_err()
        || info.try_extend_from_slice(context).is_err()
    {
        return false;
    }
    hkdf_expand(secret, &info, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(hex: &str) -> mrml_runtime::Vector<u8> {
        (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
    #[test]
    fn rfc_5869_case_one() {
        let ikm = [0x0b; 22];
        let prk = hkdf_extract(&bytes("000102030405060708090a0b0c"), &ikm);
        assert_eq!(
            &prk,
            &bytes("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")[..]
        );
        assert_eq!(
            &hmac_sha256(&prk, &[&bytes("f0f1f2f3f4f5f6f7f8f9"), &[1]]),
            &bytes("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf")[..]
        );
        let mut output = [0u8; 42];
        assert!(hkdf_expand(
            &prk,
            &bytes("f0f1f2f3f4f5f6f7f8f9"),
            &mut output
        ));
        assert_eq!(
            &output,
            &bytes(
                "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
            )[..]
        );
    }
}
