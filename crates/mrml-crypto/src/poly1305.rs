pub fn poly1305(key: &[u8; 32], message: &[u8]) -> [u8; 16] {
    let load = |offset: usize| {
        u32::from_le_bytes(key[offset..offset + 4].try_into().expect("key word")) as u64
    };
    let r0 = load(0) & 0x3ffffff;
    let r1 = (load(3) >> 2) & 0x3ffff03;
    let r2 = (load(6) >> 4) & 0x3ffc0ff;
    let r3 = (load(9) >> 6) & 0x3f03fff;
    let r4 = (load(12) >> 8) & 0x00fffff;
    let (s1, s2, s3, s4) = (r1 * 5, r2 * 5, r3 * 5, r4 * 5);
    let (mut h0, mut h1, mut h2, mut h3, mut h4) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut remaining = message;
    while !remaining.is_empty() {
        let take = remaining.len().min(16);
        let mut block = [0u8; 17];
        block[..take].copy_from_slice(&remaining[..take]);
        block[take] = 1;
        let word = |offset: usize| {
            u32::from_le_bytes(block[offset..offset + 4].try_into().expect("block word")) as u64
        };
        h0 += word(0) & 0x3ffffff;
        h1 += (word(3) >> 2) & 0x3ffffff;
        h2 += (word(6) >> 4) & 0x3ffffff;
        h3 += (word(9) >> 6) & 0x3ffffff;
        h4 += ((word(12) >> 8) & 0x00ffffff) | ((block[16] as u64) << 24);
        let d0 = h0 * r0 + h1 * s4 + h2 * s3 + h3 * s2 + h4 * s1;
        let d1 = h0 * r1 + h1 * r0 + h2 * s4 + h3 * s3 + h4 * s2;
        let d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * s4 + h4 * s3;
        let d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * s4;
        let d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0;
        let mut carry = d0 >> 26;
        h0 = d0 & 0x3ffffff;
        let value = d1 + carry;
        carry = value >> 26;
        h1 = value & 0x3ffffff;
        let value = d2 + carry;
        carry = value >> 26;
        h2 = value & 0x3ffffff;
        let value = d3 + carry;
        carry = value >> 26;
        h3 = value & 0x3ffffff;
        let value = d4 + carry;
        carry = value >> 26;
        h4 = value & 0x3ffffff;
        h0 += carry * 5;
        carry = h0 >> 26;
        h0 &= 0x3ffffff;
        h1 += carry;
        remaining = &remaining[take..];
    }
    let mut carry = h1 >> 26;
    h1 &= 0x3ffffff;
    h2 += carry;
    carry = h2 >> 26;
    h2 &= 0x3ffffff;
    h3 += carry;
    carry = h3 >> 26;
    h3 &= 0x3ffffff;
    h4 += carry;
    carry = h4 >> 26;
    h4 &= 0x3ffffff;
    h0 += carry * 5;
    carry = h0 >> 26;
    h0 &= 0x3ffffff;
    h1 += carry;
    let mut g0 = h0 + 5;
    carry = g0 >> 26;
    g0 &= 0x3ffffff;
    let mut g1 = h1 + carry;
    carry = g1 >> 26;
    g1 &= 0x3ffffff;
    let mut g2 = h2 + carry;
    carry = g2 >> 26;
    g2 &= 0x3ffffff;
    let mut g3 = h3 + carry;
    carry = g3 >> 26;
    g3 &= 0x3ffffff;
    // The underflow bit selects h when h is below the modulus. Wrapping is
    // intentional and keeps the selection branch-free in checked builds too.
    let g4 = (h4 + carry).wrapping_sub(1 << 26);
    let mask = (g4 >> 63).wrapping_sub(1);
    let inverse = !mask;
    h0 = (h0 & inverse) | (g0 & mask);
    h1 = (h1 & inverse) | (g1 & mask);
    h2 = (h2 & inverse) | (g2 & mask);
    h3 = (h3 & inverse) | (g3 & mask);
    h4 = (h4 & inverse) | (g4 & mask);
    let f0 = (h0 | (h1 << 26)) & 0xffff_ffff;
    let f1 = ((h1 >> 6) | (h2 << 20)) & 0xffff_ffff;
    let f2 = ((h2 >> 12) | (h3 << 14)) & 0xffff_ffff;
    let f3 = ((h3 >> 18) | (h4 << 8)) & 0xffff_ffff;
    let pad =
        |offset: usize| u32::from_le_bytes(key[offset..offset + 4].try_into().expect("pad")) as u64;
    let mut output = [0u8; 16];
    let mut value = f0 + pad(16);
    output[..4].copy_from_slice(&(value as u32).to_le_bytes());
    value = (value >> 32) + f1 + pad(20);
    output[4..8].copy_from_slice(&(value as u32).to_le_bytes());
    value = (value >> 32) + f2 + pad(24);
    output[8..12].copy_from_slice(&(value as u32).to_le_bytes());
    value = (value >> 32) + f3 + pad(28);
    output[12..].copy_from_slice(&(value as u32).to_le_bytes());
    output
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
    fn rfc_8439_vector() {
        let key = decode("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b");
        assert_eq!(
            poly1305(&key, b"Cryptographic Forum Research Group"),
            decode("a8061dc1305136c6c22b8baf0c0127a9")
        );
    }
}
