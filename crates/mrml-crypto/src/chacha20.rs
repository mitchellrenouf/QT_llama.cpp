fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut initial = [
        0x6170_7865,
        0x3320_646e,
        0x7962_2d32,
        0x6b20_6574,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        counter,
        0,
        0,
        0,
    ];
    for index in 0..8 {
        initial[4 + index] =
            u32::from_le_bytes(key[index * 4..index * 4 + 4].try_into().expect("key word"));
    }
    for index in 0..3 {
        initial[13 + index] = u32::from_le_bytes(
            nonce[index * 4..index * 4 + 4]
                .try_into()
                .expect("nonce word"),
        );
    }
    let mut state = initial;
    for _ in 0..10 {
        quarter_round(&mut state, 0, 4, 8, 12);
        quarter_round(&mut state, 1, 5, 9, 13);
        quarter_round(&mut state, 2, 6, 10, 14);
        quarter_round(&mut state, 3, 7, 11, 15);
        quarter_round(&mut state, 0, 5, 10, 15);
        quarter_round(&mut state, 1, 6, 11, 12);
        quarter_round(&mut state, 2, 7, 8, 13);
        quarter_round(&mut state, 3, 4, 9, 14);
    }
    let mut output = [0u8; 64];
    for index in 0..16 {
        output[index * 4..index * 4 + 4]
            .copy_from_slice(&state[index].wrapping_add(initial[index]).to_le_bytes());
    }
    output
}

pub fn chacha20_xor(key: &[u8; 32], nonce: &[u8; 12], mut counter: u32, data: &mut [u8]) -> bool {
    for chunk in data.chunks_mut(64) {
        let block = chacha20_block(key, counter, nonce);
        for (byte, mask) in chunk.iter_mut().zip(block) {
            *byte ^= mask;
        }
        let Some(next) = counter.checked_add(1) else {
            return false;
        };
        counter = next;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode<const N: usize>(hex: &str) -> [u8; N] {
        let mut out = [0; N];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
    #[test]
    fn rfc_8439_block_vector() {
        let key = decode("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let nonce = decode("000000090000004a00000000");
        assert_eq!(
            chacha20_block(&key, 1, &nonce),
            decode(
                "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
            )
        );
    }
}
