use crate::Sha256;

const MAX_LIMBS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RsaError {
    InvalidKey,
    InvalidSignature,
    UnsupportedSize,
}

fn words(bytes: &[u8], out: &mut [u32; MAX_LIMBS]) -> Option<usize> {
    if bytes.is_empty() || bytes.len() > MAX_LIMBS * 4 {
        return None;
    }
    let n = bytes.len().div_ceil(4);
    for (index, byte) in bytes.iter().rev().enumerate() {
        out[index / 4] |= (*byte as u32) << (8 * (index & 3));
    }
    Some(n)
}

fn ge(a: &[u32; MAX_LIMBS], b: &[u32; MAX_LIMBS], n: usize) -> bool {
    for i in (0..n).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn subtract(a: &mut [u32; MAX_LIMBS], b: &[u32; MAX_LIMBS], n: usize) {
    let mut borrow = 0u64;
    for i in 0..n {
        let value = (1u64 << 32) + a[i] as u64 - b[i] as u64 - borrow;
        a[i] = value as u32;
        borrow = 1 - (value >> 32);
    }
}

fn double_mod(a: &mut [u32; MAX_LIMBS], modulus: &[u32; MAX_LIMBS], n: usize) {
    let mut carry = 0u64;
    for word in a[..n].iter_mut() {
        let value = (*word as u64) * 2 + carry;
        *word = value as u32;
        carry = value >> 32;
    }
    if carry != 0 || ge(a, modulus, n) {
        subtract(a, modulus, n);
    }
}

fn inverse32(odd: u32) -> u32 {
    let mut x = odd;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(odd.wrapping_mul(x)));
    }
    x.wrapping_neg()
}

fn montgomery(
    a: &[u32; MAX_LIMBS],
    b: &[u32; MAX_LIMBS],
    modulus: &[u32; MAX_LIMBS],
    n: usize,
    n0: u32,
) -> [u32; MAX_LIMBS] {
    let mut t = [0u32; MAX_LIMBS + 1];
    for i in 0..n {
        let mut carry = 0u64;
        for j in 0..n {
            let value = t[j] as u64 + a[j] as u64 * b[i] as u64 + carry;
            t[j] = value as u32;
            carry = value >> 32;
        }
        let top = t[n] as u64 + carry;
        t[n] = top as u32;
        let m = t[0].wrapping_mul(n0);
        carry = 0;
        for j in 0..n {
            let value = t[j] as u64 + m as u64 * modulus[j] as u64 + carry;
            if j != 0 {
                t[j - 1] = value as u32;
            }
            carry = value >> 32;
        }
        let value = t[n] as u64 + carry;
        t[n - 1] = value as u32;
        t[n] = (value >> 32) as u32;
    }
    let mut out = [0u32; MAX_LIMBS];
    out[..n].copy_from_slice(&t[..n]);
    if t[n] != 0 || ge(&out, modulus, n) {
        subtract(&mut out, modulus, n);
    }
    out
}

fn modular_power(
    modulus_bytes: &[u8],
    exponent: &[u8],
    input: &[u8],
    output: &mut [u8],
) -> Result<(), RsaError> {
    if modulus_bytes.len() != input.len()
        || output.len() != input.len()
        || modulus_bytes.last().copied().unwrap_or(0) & 1 == 0
    {
        return Err(RsaError::InvalidKey);
    }
    let mut modulus = [0u32; MAX_LIMBS];
    let n = words(modulus_bytes, &mut modulus).ok_or(RsaError::UnsupportedSize)?;
    let mut base = [0u32; MAX_LIMBS];
    words(input, &mut base).ok_or(RsaError::UnsupportedSize)?;
    if ge(&base, &modulus, n) {
        return Err(RsaError::InvalidSignature);
    }
    let n0 = inverse32(modulus[0]);
    let mut r = [0u32; MAX_LIMBS];
    r[0] = 1;
    for _ in 0..n * 32 {
        double_mod(&mut r, &modulus, n);
    }
    let mut r2 = r;
    for _ in 0..n * 32 {
        double_mod(&mut r2, &modulus, n);
    }
    let mut result = r;
    base = montgomery(&base, &r2, &modulus, n, n0);
    for byte in exponent {
        for bit in (0..8).rev() {
            result = montgomery(&result, &result, &modulus, n, n0);
            let multiplied = montgomery(&result, &base, &modulus, n, n0);
            let mask = 0u32.wrapping_sub(((byte >> bit) & 1) as u32);
            for index in 0..n {
                result[index] = (result[index] & !mask) | (multiplied[index] & mask);
            }
        }
    }
    let mut one = [0u32; MAX_LIMBS];
    one[0] = 1;
    result = montgomery(&result, &one, &modulus, n, n0);
    output.fill(0);
    for (index, byte) in output.iter_mut().rev().enumerate() {
        *byte = (result[index / 4] >> (8 * (index & 3))) as u8;
    }
    Ok(())
}

pub fn rsa_pkcs1_sha256_verify(
    modulus: &[u8],
    exponent: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), RsaError> {
    let mut encoded = mrml_runtime::Vector::new();
    encoded
        .try_resize(signature.len(), 0)
        .map_err(|_| RsaError::UnsupportedSize)?;
    modular_power(modulus, exponent, signature, &mut encoded)?;
    const PREFIX: &[u8] = &[
        0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01,
        0x05, 0x00, 0x04, 0x20,
    ];
    let digest = Sha256::digest(message);
    let padding = encoded
        .len()
        .checked_sub(3 + PREFIX.len() + digest.len())
        .ok_or(RsaError::InvalidSignature)?;
    let mut difference = (padding < 8) as u8 | encoded[0] ^ 0 | encoded[1] ^ 1;
    for byte in &encoded[2..2 + padding] {
        difference |= *byte ^ 0xff;
    }
    difference |= encoded[2 + padding];
    for (a, b) in encoded[3 + padding..3 + padding + PREFIX.len()]
        .iter()
        .zip(PREFIX)
    {
        difference |= a ^ b;
    }
    for (a, b) in encoded[3 + padding + PREFIX.len()..].iter().zip(digest) {
        difference |= a ^ b;
    }
    if difference == 0 {
        Ok(())
    } else {
        Err(RsaError::InvalidSignature)
    }
}

fn mgf1_sha256(seed: &[u8], output: &mut [u8]) {
    for (counter, chunk) in output.chunks_mut(32).enumerate() {
        let mut hash = Sha256::new();
        hash.update(seed);
        hash.update(&(counter as u32).to_be_bytes());
        let digest = hash.finalize();
        chunk.copy_from_slice(&digest[..chunk.len()]);
    }
}

pub fn rsa_pss_sha256_sign(
    modulus: &[u8],
    private_exponent: &[u8],
    message: &[u8],
    salt: &[u8; 32],
    signature: &mut [u8],
) -> Result<(), RsaError> {
    if modulus.is_empty() || signature.len() != modulus.len() {
        return Err(RsaError::InvalidKey);
    }
    let modulus_bits = modulus.len() * 8 - modulus[0].leading_zeros() as usize;
    let encoded_bits = modulus_bits.checked_sub(1).ok_or(RsaError::InvalidKey)?;
    let encoded_len = encoded_bits.div_ceil(8);
    if encoded_len < 66 || encoded_len > modulus.len() {
        return Err(RsaError::UnsupportedSize);
    }
    let mut encoded = mrml_runtime::Vector::new();
    encoded
        .try_resize(modulus.len(), 0)
        .map_err(|_| RsaError::UnsupportedSize)?;
    let start = modulus.len() - encoded_len;
    let database_len = encoded_len - 33;
    let padding_len = encoded_len - 66;
    encoded[start + padding_len] = 1;
    encoded[start + padding_len + 1..start + database_len].copy_from_slice(salt);
    let message_hash = Sha256::digest(message);
    let mut hash = Sha256::new();
    hash.update(&[0; 8]);
    hash.update(&message_hash);
    hash.update(salt);
    let h = hash.finalize();
    encoded[start + database_len..start + database_len + 32].copy_from_slice(&h);
    let mut mask = mrml_runtime::Vector::new();
    mask.try_resize(database_len, 0)
        .map_err(|_| RsaError::UnsupportedSize)?;
    mgf1_sha256(&h, &mut mask);
    for (byte, masking) in encoded[start..start + database_len].iter_mut().zip(mask) {
        *byte ^= masking;
    }
    let unused = encoded_len * 8 - encoded_bits;
    if unused != 0 {
        encoded[start] &= 0xff >> unused;
    }
    encoded[modulus.len() - 1] = 0xbc;
    modular_power(modulus, private_exponent, &encoded, signature)
}

pub fn rsa_pss_sha256_verify(
    modulus: &[u8],
    exponent: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), RsaError> {
    if modulus.is_empty() {
        return Err(RsaError::InvalidKey);
    }
    let leading = modulus[0].leading_zeros() as usize;
    let modulus_bits = modulus.len() * 8 - leading;
    let encoded_bits = modulus_bits.checked_sub(1).ok_or(RsaError::InvalidKey)?;
    let encoded_len = encoded_bits.div_ceil(8);
    if encoded_len < 66 || encoded_len > signature.len() {
        return Err(RsaError::InvalidSignature);
    }
    let mut full = mrml_runtime::Vector::new();
    full.try_resize(signature.len(), 0)
        .map_err(|_| RsaError::UnsupportedSize)?;
    modular_power(modulus, exponent, signature, &mut full)?;
    if full[..full.len() - encoded_len]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(RsaError::InvalidSignature);
    }
    let encoded = &full[full.len() - encoded_len..];
    let hash_start = encoded_len - 33;
    let mut difference = encoded[encoded_len - 1] ^ 0xbc;
    let unused = encoded_len * 8 - encoded_bits;
    if unused != 0 {
        difference |= encoded[0] & (0xff << (8 - unused));
    }
    let mut database = mrml_runtime::Vector::new();
    database
        .try_extend_from_slice(&encoded[..hash_start])
        .map_err(|_| RsaError::UnsupportedSize)?;
    let mut mask = mrml_runtime::Vector::new();
    mask.try_resize(hash_start, 0)
        .map_err(|_| RsaError::UnsupportedSize)?;
    mgf1_sha256(&encoded[hash_start..hash_start + 32], &mut mask);
    for (byte, masking) in database.iter_mut().zip(mask) {
        *byte ^= masking;
    }
    if unused != 0 {
        database[0] &= 0xff >> unused;
    }
    let padding_len = encoded_len - 32 - 32 - 2;
    for byte in &database[..padding_len] {
        difference |= *byte;
    }
    difference |= database[padding_len] ^ 1;
    let message_hash = Sha256::digest(message);
    let mut verify = Sha256::new();
    verify.update(&[0; 8]);
    verify.update(&message_hash);
    verify.update(&database[padding_len + 1..]);
    for (a, b) in verify
        .finalize()
        .iter()
        .zip(&encoded[hash_start..hash_start + 32])
    {
        difference |= a ^ b;
    }
    if difference == 0 {
        Ok(())
    } else {
        Err(RsaError::InvalidSignature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode(text: &str) -> mrml_runtime::Vector<u8> {
        (0..text.len() / 2)
            .map(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
    #[test]
    fn montgomery_power_matches_small_arithmetic() {
        let modulus = 3233u32.to_be_bytes();
        let input = 65u32.to_be_bytes();
        let mut output = [0u8; 4];
        modular_power(&modulus, &[17], &input, &mut output).unwrap();
        assert_eq!(u32::from_be_bytes(output), 2790);
        let encrypted = output;
        modular_power(&modulus, &[0x0a, 0xc1], &encrypted, &mut output).unwrap();
        assert_eq!(u32::from_be_bytes(output), 65);
    }
    #[test]
    fn verifies_independent_rsa_2048_pkcs1_signature() {
        let modulus = decode(
            "cb512e74cd40c33fe84009f2bb54563652c359828d41bd2fd3cbcb91a2ff4f1840872a4ecad30adfbd3cd4c4bb3a610d75d7ade9d93d767a7f7a8c4cee0c65d2536ecc5d1ef77824e159f661842dc2472cbb335528bd7c1dccce1a2b8feedbfe9744265c5e081defd981d49ddb5f20594ff0e9f76b82370e42cd66e6cea535d37e6992204c1c0ebdbf520873269904cbc1c02a75225bb0dd092a2f8d27b57bef46e4611d84f49dbc5b337b8ab796a6ea7a3c35e956da7be0a503ea409b0fa84cdb50ebaeda11bd921e9c9d3447bf667b38a1345eec5c4f363a85f623e97f081da2722009cfe0d619db9de642b85f118f2c9c73e98cb454167d0c90cbf09b9f0b",
        );
        let mut signature = decode(
            "052e3d262e5f2a268ed96bedc2839049ec6923eeb4cc73e2f1925b33b9b6a844d685e502daba4c4bed7a57b0bd0c36acf857d9cfd7a8009f7b5a10152bd23ab587da6c58fa12358c25995dface96dca6994e8dc45648aed2320fa3a805092560e1f7f0d7f47bc1cdff542fcaaf9f99118327656d1ccc5f28c5b13308b3b76ed2c9ca50dd995440b517a273a626e5247555352e12419247c4b6fc8bad7bbd78496750c87cf9c04f4b8cd9d17d0cb8c1ad42b0ea9483bc0d2ebfeb9ebe1d9c8aa7970899efe53d65c8dceebaad459f6299fd70eac1b17407644a7e02ef99b7391ae67bc679fa3b910f3695291f6e6e66fd5e4ba8aead4f2bca8c6cea008ec54ed2",
        );
        assert_eq!(
            rsa_pkcs1_sha256_verify(
                &modulus,
                &[1, 0, 1],
                b"mrml-rsa-verification-vector",
                &signature
            ),
            Ok(())
        );
        signature[100] ^= 1;
        assert_eq!(
            rsa_pkcs1_sha256_verify(
                &modulus,
                &[1, 0, 1],
                b"mrml-rsa-verification-vector",
                &signature
            ),
            Err(RsaError::InvalidSignature)
        );
    }
    #[test]
    fn verifies_independent_rsa_2048_pss_signature() {
        let modulus = decode(
            "b6ef7b80fdefe8b4dddae05a4a1c84689b0d710b917705149d26cd0adc6c53d0395dc9c7ef53e7331360e5b07e1595963e1bd5915b573c0db527b4bed755622fea869055a513fd542392fe8c958885ba840d742ffd8725c22965ce2a25d3000ad909938dc83e827c6817b3c4f1d6a59916d5db4a51ea8ef11ba946119f4331d470b895132845bf31f8c269a8364ece0137f8d2212194350920be8ea398d440a60c28b4bb83cf3a6356d83d298777348915acbd39c12a4a12fc7a2ba5faf9ed10aa7e30bdaab6043f56cc62a4a32b735e3181de1ea2e6611b60e8287a2e8f703bec9c9ff816b689bc5ec44df02f9824e0dc57d7a43d63970991e26a585a408efb",
        );
        let mut signature = decode(
            "15d206291399df9dee789a7cfc43ec4915c8ef402d28b7c7197eefbe195251a5142480cd35c5e22ef677ca0032bb65ac6c732cb891243d8664ac6f006b430580532dd96de421da397eefef3633def3479e1da63607dcef16587097d5b3714536828002f5c769627329f047538f7f6ee5a9b5c7d4d3ed51801b3cfa64a6ee00fe482d0d67ef25b359f89726cb25675c3d239bf528d08719f3449ee0b1feffe10c33a26ba36c481d08d9db67e0d65ea02414c77b6ce85f3d4fae73f7fd92a0d834d3b171357009399e56dd2513dd6059accdaacd4c718584eabb3dde08077355f39cef41016af710b47f58b1e3316f23f988e7d578b953cff5c3bfebfc2ab30993",
        );
        assert_eq!(
            rsa_pss_sha256_verify(&modulus, &[1, 0, 1], b"mrml-rsa-pss-vector", &signature),
            Ok(())
        );
        signature[0] ^= 1;
        assert_eq!(
            rsa_pss_sha256_verify(&modulus, &[1, 0, 1], b"mrml-rsa-pss-vector", &signature),
            Err(RsaError::InvalidSignature)
        );
    }
}
