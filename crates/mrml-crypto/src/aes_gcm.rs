//! AES-128-GCM authenticated encryption for TLS 1.3.

fn xtime(x: u8) -> u8 {
    (x << 1) ^ (0x1b & (0u8.wrapping_sub(x >> 7)))
}

fn inverse(x: u8) -> u8 {
    if x == 0 {
        return 0;
    }
    let mut result = 1u8;
    let mut base = x;
    let mut exponent = 254u8;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = gf_mul(result, base);
        }
        base = gf_mul(base, base);
        exponent >>= 1;
    }
    result
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0;
    for _ in 0..8 {
        result ^= a & 0u8.wrapping_sub(b & 1);
        a = xtime(a);
        b >>= 1;
    }
    result
}

fn sbox(x: u8) -> u8 {
    let y = inverse(x);
    y ^ y.rotate_left(1) ^ y.rotate_left(2) ^ y.rotate_left(3) ^ y.rotate_left(4) ^ 0x63
}

fn expand_key(key: &[u8; 16]) -> [u8; 176] {
    let mut expanded = [0u8; 176];
    expanded[..16].copy_from_slice(key);
    let mut position = 16;
    let mut rcon = 1u8;
    while position < expanded.len() {
        let mut word = [
            expanded[position - 4],
            expanded[position - 3],
            expanded[position - 2],
            expanded[position - 1],
        ];
        if position & 15 == 0 {
            word = [
                sbox(word[1]) ^ rcon,
                sbox(word[2]),
                sbox(word[3]),
                sbox(word[0]),
            ];
            rcon = xtime(rcon);
        }
        for byte in word {
            expanded[position] = expanded[position - 16] ^ byte;
            position += 1;
        }
    }
    expanded
}

fn aes128_encrypt(key: &[u8; 16], input: &[u8; 16]) -> [u8; 16] {
    let keys = expand_key(key);
    let mut state = *input;
    add_round_key(&mut state, &keys[..16]);
    for round in 1..=10 {
        for byte in &mut state {
            *byte = sbox(*byte);
        }
        let old = state;
        for row in 0..4 {
            for column in 0..4 {
                state[4 * column + row] = old[4 * ((column + row) & 3) + row];
            }
        }
        if round != 10 {
            for column in 0..4 {
                let i = 4 * column;
                let a = [state[i], state[i + 1], state[i + 2], state[i + 3]];
                state[i] = xtime(a[0]) ^ (xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3];
                state[i + 1] = a[0] ^ xtime(a[1]) ^ (xtime(a[2]) ^ a[2]) ^ a[3];
                state[i + 2] = a[0] ^ a[1] ^ xtime(a[2]) ^ (xtime(a[3]) ^ a[3]);
                state[i + 3] = (xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ xtime(a[3]);
            }
        }
        add_round_key(&mut state, &keys[16 * round..16 * (round + 1)]);
    }
    state
}

fn add_round_key(state: &mut [u8; 16], key: &[u8]) {
    for i in 0..16 {
        state[i] ^= key[i];
    }
}

fn ghash_mul(x: u128, y: u128) -> u128 {
    let mut z = 0u128;
    let mut v = y;
    for bit in 0..128 {
        z ^= v & 0u128.wrapping_sub((x >> (127 - bit)) & 1);
        let low = v & 1;
        v = (v >> 1) ^ (0xe1000000000000000000000000000000u128 & 0u128.wrapping_sub(low));
    }
    z
}

fn ghash(h: u128, aad: &[u8], data: &[u8]) -> u128 {
    let mut state = 0u128;
    for source in [aad, data] {
        for block in source.chunks(16) {
            let mut padded = [0u8; 16];
            padded[..block.len()].copy_from_slice(block);
            state = ghash_mul(state ^ u128::from_be_bytes(padded), h);
        }
    }
    let lengths = ((aad.len() as u128 * 8) << 64) | data.len() as u128 * 8;
    ghash_mul(state ^ lengths, h)
}

fn counter_block(nonce: &[u8; 12], counter: u32) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[..12].copy_from_slice(nonce);
    block[12..].copy_from_slice(&counter.to_be_bytes());
    block
}

fn crypt(key: &[u8; 16], nonce: &[u8; 12], input: &[u8], output: &mut [u8]) {
    for (index, (source, target)) in input.chunks(16).zip(output.chunks_mut(16)).enumerate() {
        let stream = aes128_encrypt(key, &counter_block(nonce, 2 + index as u32));
        for i in 0..source.len() {
            target[i] = source[i] ^ stream[i];
        }
    }
}

pub fn aes128_gcm_seal(
    key: &[u8; 16],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
    ciphertext: &mut [u8],
    tag: &mut [u8; 16],
) -> bool {
    if ciphertext.len() != plaintext.len() {
        return false;
    }
    crypt(key, nonce, plaintext, ciphertext);
    let h = u128::from_be_bytes(aes128_encrypt(key, &[0; 16]));
    *tag = (ghash(h, aad, ciphertext)
        ^ u128::from_be_bytes(aes128_encrypt(key, &counter_block(nonce, 1))))
    .to_be_bytes();
    true
}

pub fn aes128_gcm_open(
    key: &[u8; 16],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
    plaintext: &mut [u8],
) -> bool {
    if plaintext.len() != ciphertext.len() {
        return false;
    }
    let h = u128::from_be_bytes(aes128_encrypt(key, &[0; 16]));
    let expected = (ghash(h, aad, ciphertext)
        ^ u128::from_be_bytes(aes128_encrypt(key, &counter_block(nonce, 1))))
    .to_be_bytes();
    let mut difference = 0u8;
    for i in 0..16 {
        difference |= expected[i] ^ tag[i];
    }
    if difference != 0 {
        return false;
    }
    crypt(key, nonce, ciphertext, plaintext);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hex<const N: usize>(s: &str) -> [u8; N] {
        let mut out = [0; N];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
    #[test]
    fn nist_empty_and_single_block_vectors() {
        let key = [0u8; 16];
        let nonce = [0u8; 12];
        let mut tag = [0u8; 16];
        assert!(aes128_gcm_seal(&key, &nonce, &[], &[], &mut [], &mut tag));
        assert_eq!(
            tag,
            [
                0x58, 0xe2, 0xfc, 0xce, 0xfa, 0x7e, 0x30, 0x61, 0x36, 0x7f, 0x1d, 0x57, 0xa4, 0xe7,
                0x45, 0x5a
            ]
        );
        let plain = [0u8; 16];
        let mut encrypted = [0u8; 16];
        aes128_gcm_seal(&key, &nonce, &[], &plain, &mut encrypted, &mut tag);
        assert_eq!(
            encrypted,
            [
                0x03, 0x88, 0xda, 0xce, 0x60, 0xb6, 0xa3, 0x92, 0xf3, 0x28, 0xc2, 0xb9, 0x71, 0xb2,
                0xfe, 0x78
            ]
        );
        assert_eq!(
            tag,
            [
                0xab, 0x6e, 0x47, 0xd4, 0x2c, 0xec, 0x13, 0xbd, 0xf5, 0x3a, 0x67, 0xb2, 0x12, 0x57,
                0xbd, 0xdf
            ]
        );
        let mut opened = [0u8; 16];
        assert!(aes128_gcm_open(
            &key,
            &nonce,
            &[],
            &encrypted,
            &tag,
            &mut opened
        ));
        assert_eq!(opened, plain);
        tag[0] ^= 1;
        assert!(!aes128_gcm_open(
            &key,
            &nonce,
            &[],
            &encrypted,
            &tag,
            &mut opened
        ));
    }
    #[test]
    fn nist_aad_and_partial_block_vector() {
        let key = hex("feffe9928665731c6d6a8f9467308308");
        let nonce = hex("cafebabefacedbaddecaf888");
        let aad = hex::<20>("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let plain = hex::<60>(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        );
        let expected = hex::<60>(
            "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091",
        );
        let mut encrypted = [0; 60];
        let mut tag = [0; 16];
        assert!(aes128_gcm_seal(
            &key,
            &nonce,
            &aad,
            &plain,
            &mut encrypted,
            &mut tag
        ));
        assert_eq!(encrypted, expected);
        assert_eq!(tag, hex("5bc94fbc3221a5db94fae95ae7121a47"));
    }
}
