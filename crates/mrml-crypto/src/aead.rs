use crate::{chacha20_block, chacha20_xor, poly1305};

fn authentication_input(aad: &[u8], ciphertext: &[u8]) -> Option<mrml_runtime::Vector<u8>> {
    let aad_padding = (16 - aad.len() % 16) % 16;
    let ciphertext_padding = (16 - ciphertext.len() % 16) % 16;
    let capacity = aad
        .len()
        .checked_add(aad_padding)?
        .checked_add(ciphertext.len())?
        .checked_add(ciphertext_padding)?
        .checked_add(16)?;
    let mut input = mrml_runtime::Vector::with_capacity(capacity).ok()?;
    input.try_extend_from_slice(aad).ok()?;
    for _ in 0..aad_padding {
        input.try_push(0).ok()?;
    }
    input.try_extend_from_slice(ciphertext).ok()?;
    for _ in 0..ciphertext_padding {
        input.try_push(0).ok()?;
    }
    input
        .try_extend_from_slice(&(aad.len() as u64).to_le_bytes())
        .ok()?;
    input
        .try_extend_from_slice(&(ciphertext.len() as u64).to_le_bytes())
        .ok()?;
    Some(input)
}

pub fn chacha20_poly1305_seal(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    message: &mut [u8],
) -> Option<[u8; 16]> {
    let block = chacha20_block(key, 0, nonce);
    let poly_key: &[u8; 32] = (&block[..32]).try_into().expect("Poly1305 key");
    chacha20_xor(key, nonce, 1, message).then_some(())?;
    Some(poly1305(poly_key, &authentication_input(aad, message)?))
}

pub fn chacha20_poly1305_open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    message: &mut [u8],
    tag: &[u8; 16],
) -> bool {
    let block = chacha20_block(key, 0, nonce);
    let poly_key: &[u8; 32] = (&block[..32]).try_into().expect("Poly1305 key");
    let Some(input) = authentication_input(aad, message) else {
        return false;
    };
    let expected = poly1305(poly_key, &input);
    let mut difference = 0u8;
    for (&left, &right) in expected.iter().zip(tag) {
        difference |= left ^ right;
    }
    if difference != 0 {
        return false;
    }
    chacha20_xor(key, nonce, 1, message)
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
    fn rfc_8439_aead_vector_and_rejects_tampering() {
        let key: [u8; 32] =
            bytes("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")[..]
                .try_into()
                .unwrap();
        let nonce: [u8; 12] = bytes("070000004041424344454647")[..].try_into().unwrap();
        let aad = bytes("50515253c0c1c2c3c4c5c6c7");
        let mut message = mrml_runtime::Vector::new();
        message.try_extend_from_slice(b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.").unwrap();
        let tag = chacha20_poly1305_seal(&key, &nonce, &aad, &mut message).unwrap();
        assert_eq!(
            &message[..],
            &bytes(
                "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116"
            )[..]
        );
        assert_eq!(&tag, &bytes("1ae10b594f09e26a7e902ecbd0600691")[..]);
        let ciphertext = message.clone();
        assert!(chacha20_poly1305_open(
            &key,
            &nonce,
            &aad,
            &mut message,
            &tag
        ));
        assert_eq!(&message[..],b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.");
        let mut altered = ciphertext;
        altered[0] ^= 1;
        assert!(!chacha20_poly1305_open(
            &key,
            &nonce,
            &aad,
            &mut altered,
            &tag
        ));
    }
}
