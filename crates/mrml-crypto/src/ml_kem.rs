use crate::{Sha3_256, Sha3_512, Shake128, Shake256};

const Q: i32 = 3329;
const K: usize = 3;
const N: usize = 256;
const DU: usize = 10;
const DV: usize = 4;
pub const ML_KEM_768_ENCAPSULATION_KEY_BYTES: usize = 1184;
pub const ML_KEM_768_DECAPSULATION_KEY_BYTES: usize = 2400;
pub const ML_KEM_768_CIPHERTEXT_BYTES: usize = 1088;

#[derive(Clone, Copy)]
struct Poly([i16; N]);
type Matrix = [[Poly; K]; K];
const ZERO: Poly = Poly([0; N]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlKemError {
    Random,
    InvalidEncapsulationKey,
    InvalidDecapsulationKey,
}
impl core::fmt::Display for MlKemError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Random => "cryptographic randomness failed",
            Self::InvalidEncapsulationKey => "invalid ML-KEM-768 encapsulation key",
            Self::InvalidDecapsulationKey => "invalid ML-KEM-768 decapsulation key",
        })
    }
}
impl core::error::Error for MlKemError {}

#[derive(Clone)]
pub struct MlKem768EncapsulationKey(pub [u8; ML_KEM_768_ENCAPSULATION_KEY_BYTES]);
pub struct MlKem768DecapsulationKey(pub [u8; ML_KEM_768_DECAPSULATION_KEY_BYTES]);
#[derive(Clone)]
pub struct MlKem768Ciphertext(pub [u8; ML_KEM_768_CIPHERTEXT_BYTES]);

impl Drop for MlKem768DecapsulationKey {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

fn modulus(value: i64) -> i16 {
    let remainder = value % Q as i64;
    (remainder + ((remainder >> 63) & Q as i64)) as i16
}
fn bit_reverse7(mut value: usize) -> usize {
    let mut output = 0;
    for _ in 0..7 {
        output = (output << 1) | (value & 1);
        value >>= 1;
    }
    output
}
fn power(mut base: i32, mut exponent: usize) -> i16 {
    let mut output = 1i64;
    while exponent > 0 {
        if exponent & 1 != 0 {
            output = output * base as i64 % Q as i64;
        }
        base = (base as i64 * base as i64 % Q as i64) as i32;
        exponent >>= 1;
    }
    output as i16
}
fn zeta(index: usize) -> i16 {
    power(17, bit_reverse7(index))
}

fn ntt(mut poly: Poly) -> Poly {
    let mut index = 1;
    let mut length = 128;
    while length >= 2 {
        let mut start = 0;
        while start < N {
            let root = zeta(index) as i64;
            index += 1;
            for position in start..start + length {
                let product = root * poly.0[position + length] as i64;
                let value = poly.0[position] as i64;
                poly.0[position + length] = modulus(value - product);
                poly.0[position] = modulus(value + product);
            }
            start += 2 * length;
        }
        length /= 2;
    }
    poly
}
fn inverse_ntt(mut poly: Poly) -> Poly {
    let mut index = 127usize;
    let mut length = 2;
    while length <= 128 {
        let mut start = 0;
        while start < N {
            let root = zeta(index) as i64;
            index -= 1;
            for position in start..start + length {
                let value = poly.0[position] as i64;
                let other = poly.0[position + length] as i64;
                poly.0[position] = modulus(value + other);
                poly.0[position + length] = modulus(root * (other - value));
            }
            start += 2 * length;
        }
        length *= 2;
    }
    for value in &mut poly.0 {
        *value = modulus(*value as i64 * 3303);
    }
    poly
}
fn multiply_ntt(left: &Poly, right: &Poly) -> Poly {
    let mut output = ZERO;
    for index in 0..128 {
        let root = power(17, 2 * bit_reverse7(index) + 1) as i64;
        let a0 = left.0[2 * index] as i64;
        let a1 = left.0[2 * index + 1] as i64;
        let b0 = right.0[2 * index] as i64;
        let b1 = right.0[2 * index + 1] as i64;
        output.0[2 * index] = modulus(a0 * b0 + a1 * b1 * root);
        output.0[2 * index + 1] = modulus(a0 * b1 + a1 * b0);
    }
    output
}
fn add_assign(left: &mut Poly, right: &Poly) {
    for i in 0..N {
        left.0[i] = modulus(left.0[i] as i64 + right.0[i] as i64);
    }
}
fn sub(left: &Poly, right: &Poly) -> Poly {
    let mut output = ZERO;
    for i in 0..N {
        output.0[i] = modulus(left.0[i] as i64 - right.0[i] as i64);
    }
    output
}

fn encode<const D: usize>(poly: &Poly, output: &mut [u8]) {
    output.fill(0);
    for i in 0..N {
        let value = poly.0[i] as u16;
        for bit in 0..D {
            output[(i * D + bit) / 8] |= (((value >> bit) & 1) as u8) << ((i * D + bit) % 8);
        }
    }
}
fn decode<const D: usize>(input: &[u8]) -> Poly {
    let mut output = ZERO;
    let modulus_value = if D == 12 { Q } else { 1 << D };
    for i in 0..N {
        let mut value = 0i32;
        for bit in 0..D {
            value |= (((input[(i * D + bit) / 8] >> ((i * D + bit) % 8)) & 1) as i32) << bit;
        }
        output.0[i] = (value % modulus_value) as i16;
    }
    output
}
fn raw_12_is_canonical(input: &[u8]) -> bool {
    for i in 0..N {
        let mut value = 0u16;
        for bit in 0..12 {
            value |= (((input[(i * 12 + bit) / 8] >> ((i * 12 + bit) % 8)) & 1) as u16) << bit;
        }
        if value >= Q as u16 {
            return false;
        }
    }
    true
}
fn compress<const D: usize>(poly: &Poly) -> Poly {
    let mut output = ZERO;
    let scale = 1i64 << D;
    for i in 0..N {
        output.0[i] =
            ((((poly.0[i] as i64 * scale) + (Q as i64 / 2)) / Q as i64) & (scale - 1)) as i16;
    }
    output
}
fn decompress<const D: usize>(poly: &Poly) -> Poly {
    let mut output = ZERO;
    for i in 0..N {
        output.0[i] = ((Q as i64 * poly.0[i] as i64 + (1i64 << (D - 1))) >> D) as i16;
    }
    output
}

fn sample_ntt(seed: &[u8; 32], column: u8, row: u8) -> Poly {
    let mut input = [0u8; 34];
    input[..32].copy_from_slice(seed);
    input[32] = column;
    input[33] = row;
    let mut stream = [0u8; 4096];
    Shake128::digest(&input, &mut stream);
    let mut output = ZERO;
    let (mut position, mut count) = (0, 0);
    while count < N {
        let first = stream[position] as i32 + 256 * (stream[position + 1] as i32 & 15);
        let second = (stream[position + 1] as i32 >> 4) + 16 * stream[position + 2] as i32;
        position += 3;
        if first < Q {
            output.0[count] = first as i16;
            count += 1;
        }
        if second < Q && count < N {
            output.0[count] = second as i16;
            count += 1;
        }
    }
    output
}
fn sample_cbd(seed: &[u8; 32], nonce: u8) -> Poly {
    const ETA: usize = 2;
    let mut input = [0u8; 33];
    input[..32].copy_from_slice(seed);
    input[32] = nonce;
    let mut stream = [0u8; 128];
    Shake256::digest(&input, &mut stream);
    let mut output = ZERO;
    for i in 0..N {
        let mut left = 0;
        let mut right = 0;
        for bit in 0..ETA {
            left += ((stream[(2 * i * ETA + bit) / 8] >> ((2 * i * ETA + bit) % 8)) & 1) as i32;
            let at = 2 * i * ETA + ETA + bit;
            right += ((stream[at / 8] >> (at % 8)) & 1) as i32;
        }
        output.0[i] = modulus((left - right) as i64);
    }
    output
}
fn matrix(seed: &[u8; 32]) -> Matrix {
    let mut output = [[ZERO; K]; K];
    for row in 0..K {
        for column in 0..K {
            output[row][column] = sample_ntt(seed, column as u8, row as u8);
        }
    }
    output
}

fn pke_keygen(seed: &[u8; 32]) -> ([u8; 1184], [u8; 1152]) {
    let mut expanded_input = [0u8; 33];
    expanded_input[..32].copy_from_slice(seed);
    expanded_input[32] = K as u8;
    let expanded = Sha3_512::digest(&expanded_input);
    let rho: [u8; 32] = expanded[..32].try_into().unwrap();
    let sigma: [u8; 32] = expanded[32..].try_into().unwrap();
    let a = matrix(&rho);
    let mut secret = [ZERO; K];
    let mut error = [ZERO; K];
    for i in 0..K {
        secret[i] = ntt(sample_cbd(&sigma, i as u8));
        error[i] = ntt(sample_cbd(&sigma, (i + K) as u8));
    }
    let mut public = [ZERO; K];
    for row in 0..K {
        for column in 0..K {
            add_assign(
                &mut public[row],
                &multiply_ntt(&a[row][column], &secret[column]),
            );
        }
        add_assign(&mut public[row], &error[row]);
    }
    let mut ek = [0u8; 1184];
    let mut dk = [0u8; 1152];
    for i in 0..K {
        encode::<12>(&public[i], &mut ek[i * 384..(i + 1) * 384]);
        encode::<12>(&secret[i], &mut dk[i * 384..(i + 1) * 384]);
    }
    ek[1152..].copy_from_slice(&rho);
    (ek, dk)
}

fn pke_encrypt(ek: &[u8; 1184], message: &[u8; 32], randomness: &[u8; 32]) -> [u8; 1088] {
    let mut public = [ZERO; K];
    for i in 0..K {
        public[i] = decode::<12>(&ek[i * 384..(i + 1) * 384]);
    }
    let rho: [u8; 32] = ek[1152..].try_into().unwrap();
    let a = matrix(&rho);
    let mut y = [ZERO; K];
    let mut e1 = [ZERO; K];
    for i in 0..K {
        y[i] = ntt(sample_cbd(randomness, i as u8));
        e1[i] = sample_cbd(randomness, (i + K) as u8);
    }
    let e2 = sample_cbd(randomness, (2 * K) as u8);
    let mut u = [ZERO; K];
    for i in 0..K {
        for j in 0..K {
            add_assign(&mut u[i], &multiply_ntt(&a[j][i], &y[j]));
        }
        u[i] = inverse_ntt(u[i]);
        add_assign(&mut u[i], &e1[i]);
    }
    let mut v = ZERO;
    for i in 0..K {
        add_assign(&mut v, &multiply_ntt(&public[i], &y[i]));
    }
    v = inverse_ntt(v);
    add_assign(&mut v, &e2);
    for i in 0..N {
        let bit = ((message[i / 8] >> (i % 8)) & 1) as i64;
        v.0[i] = modulus(v.0[i] as i64 + bit * ((Q + 1) as i64 / 2));
    }
    let mut ciphertext = [0u8; 1088];
    for i in 0..K {
        encode::<DU>(
            &compress::<DU>(&u[i]),
            &mut ciphertext[i * 320..(i + 1) * 320],
        );
    }
    encode::<DV>(&compress::<DV>(&v), &mut ciphertext[960..]);
    ciphertext
}
fn pke_decrypt(dk: &[u8], ciphertext: &[u8; 1088]) -> [u8; 32] {
    let mut secret = [ZERO; K];
    let mut u = [ZERO; K];
    for i in 0..K {
        secret[i] = decode::<12>(&dk[i * 384..(i + 1) * 384]);
        u[i] = decompress::<DU>(&decode::<DU>(&ciphertext[i * 320..(i + 1) * 320]));
    }
    let v = decompress::<DV>(&decode::<DV>(&ciphertext[960..]));
    let mut product = ZERO;
    for i in 0..K {
        add_assign(&mut product, &multiply_ntt(&secret[i], &ntt(u[i])));
    }
    let difference = sub(&v, &inverse_ntt(product));
    let bits = compress::<1>(&difference);
    let mut message = [0u8; 32];
    encode::<1>(&bits, &mut message);
    message
}

fn keygen_internal(
    d: [u8; 32],
    z: [u8; 32],
) -> (MlKem768EncapsulationKey, MlKem768DecapsulationKey) {
    let (ek, pke_dk) = pke_keygen(&d);
    let mut dk = [0u8; 2400];
    dk[..1152].copy_from_slice(&pke_dk);
    dk[1152..2336].copy_from_slice(&ek);
    dk[2336..2368].copy_from_slice(&Sha3_256::digest(&ek));
    dk[2368..].copy_from_slice(&z);
    (MlKem768EncapsulationKey(ek), MlKem768DecapsulationKey(dk))
}
fn encapsulate_internal(
    ek: &MlKem768EncapsulationKey,
    message: [u8; 32],
) -> ([u8; 32], MlKem768Ciphertext) {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&message);
    input[32..].copy_from_slice(&Sha3_256::digest(&ek.0));
    let expanded = Sha3_512::digest(&input);
    let shared: [u8; 32] = expanded[..32].try_into().unwrap();
    let randomness: [u8; 32] = expanded[32..].try_into().unwrap();
    (
        shared,
        MlKem768Ciphertext(pke_encrypt(&ek.0, &message, &randomness)),
    )
}

pub fn ml_kem_768_keygen()
-> Result<(MlKem768EncapsulationKey, MlKem768DecapsulationKey), MlKemError> {
    let mut d = [0u8; 32];
    let mut z = [0u8; 32];
    mrml_runtime::fill_random(&mut d).map_err(|_| MlKemError::Random)?;
    mrml_runtime::fill_random(&mut z).map_err(|_| MlKemError::Random)?;
    let result = keygen_internal(d, z);
    d.fill(0);
    z.fill(0);
    Ok(result)
}
fn validate_ek(ek: &MlKem768EncapsulationKey) -> bool {
    (0..K).all(|i| raw_12_is_canonical(&ek.0[i * 384..(i + 1) * 384]))
}
pub fn ml_kem_768_encapsulate(
    ek: &MlKem768EncapsulationKey,
) -> Result<([u8; 32], MlKem768Ciphertext), MlKemError> {
    if !validate_ek(ek) {
        return Err(MlKemError::InvalidEncapsulationKey);
    }
    let mut message = [0u8; 32];
    mrml_runtime::fill_random(&mut message).map_err(|_| MlKemError::Random)?;
    let result = encapsulate_internal(ek, message);
    message.fill(0);
    Ok(result)
}
pub fn ml_kem_768_decapsulate(
    dk: &MlKem768DecapsulationKey,
    ciphertext: &MlKem768Ciphertext,
) -> Result<[u8; 32], MlKemError> {
    let ek = MlKem768EncapsulationKey(dk.0[1152..2336].try_into().unwrap());
    let expected_hash = Sha3_256::digest(&ek.0);
    let mut invalid = 0u8;
    for (a, b) in expected_hash.iter().zip(&dk.0[2336..2368]) {
        invalid |= a ^ b;
    }
    if invalid != 0 || !validate_ek(&ek) {
        return Err(MlKemError::InvalidDecapsulationKey);
    }
    let message = pke_decrypt(&dk.0[..1152], &ciphertext.0);
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&message);
    input[32..].copy_from_slice(&expected_hash);
    let expanded = Sha3_512::digest(&input);
    let mut candidate: [u8; 32] = expanded[..32].try_into().unwrap();
    let randomness: [u8; 32] = expanded[32..].try_into().unwrap();
    let expected_ciphertext = pke_encrypt(&ek.0, &message, &randomness);
    let mut difference = 0u8;
    for (a, b) in expected_ciphertext.iter().zip(&ciphertext.0) {
        difference |= a ^ b;
    }
    let mut rejection_input = mrml_runtime::Vector::with_capacity(32 + 1088)
        .map_err(|_| MlKemError::InvalidDecapsulationKey)?;
    rejection_input
        .try_extend_from_slice(&dk.0[2368..])
        .map_err(|_| MlKemError::InvalidDecapsulationKey)?;
    rejection_input
        .try_extend_from_slice(&ciphertext.0)
        .map_err(|_| MlKemError::InvalidDecapsulationKey)?;
    let mut rejection = [0u8; 32];
    Shake256::digest(&rejection_input, &mut rejection);
    let mask = 0u8.wrapping_sub((difference != 0) as u8);
    for i in 0..32 {
        candidate[i] = (candidate[i] & !mask) | (rejection[i] & mask);
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hex32(value: &str) -> [u8; 32] {
        let mut output = [0; 32];
        for (i, byte) in output.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16).unwrap();
        }
        output
    }
    #[test]
    fn ntt_round_trip() {
        let mut p = ZERO;
        for i in 0..N {
            p.0[i] = (i as i32 % Q) as i16;
        }
        assert_eq!(inverse_ntt(ntt(p)).0, p.0);
    }
    #[test]
    fn kem_round_trip_and_implicit_rejection() {
        let (ek, dk) = keygen_internal([7; 32], [9; 32]);
        assert!(validate_ek(&ek));
        let (shared, ciphertext) = encapsulate_internal(&ek, [11; 32]);
        assert_eq!(ml_kem_768_decapsulate(&dk, &ciphertext).unwrap(), shared);
        let mut altered = ciphertext.clone();
        altered.0[17] ^= 1;
        assert_ne!(ml_kem_768_decapsulate(&dk, &altered).unwrap(), shared);
    }
    #[test]
    fn rejects_noncanonical_public_coefficients() {
        let (mut ek, _) = keygen_internal([3; 32], [5; 32]);
        ek.0[0] = 0xff;
        ek.0[1] |= 0x0f;
        assert!(!validate_ek(&ek));
    }
    #[test]
    fn nist_acvp_keygen_case_26() {
        let d = hex32("E582B7D75E6C80B05AE392A1FC9F7153B12390FD99930368CC67A768BAEBC8A0");
        let z = hex32("1CDACB8740C0B87C4A379575F187B367CBFA3B300BF591B109F79816E9CBE8F0");
        let (ek, dk) = keygen_internal(d, z);
        assert_eq!(
            Sha3_256::digest(&ek.0),
            hex32("81e66ef5a7a221619f6a64039cc369843e10df5c859f6959cc3fd8e5272330fd")
        );
        assert_eq!(
            Sha3_256::digest(&dk.0),
            hex32("be81068c104cd6cf8efd800b294f4a15bb8a8050993fd54a2cc428841ef6ca44")
        );
    }
    #[test]
    fn external_nist_acvp_encapsulation_case() {
        let Some(directory) = mrml_runtime::environment_variable("MRML_ACVP_MLKEM768") else {
            return;
        };
        let read = |name: &str| {
            mrml_runtime::read_file(&mrml_runtime::join_path(&directory, name)).unwrap()
        };
        let ek = MlKem768EncapsulationKey(read("ek.bin")[..].try_into().unwrap());
        let message = read("m.bin")[..].try_into().unwrap();
        let expected_ciphertext = read("c.bin");
        let expected_key = read("k.bin");
        let (key, ciphertext) = encapsulate_internal(&ek, message);
        assert_eq!(&key, &expected_key[..]);
        assert_eq!(&ciphertext.0, &expected_ciphertext[..]);
    }
    #[test]
    fn external_nist_acvp_decapsulation_case() {
        let Some(directory) = mrml_runtime::environment_variable("MRML_ACVP_MLKEM768_DECAP") else {
            return;
        };
        let read = |name: &str| {
            mrml_runtime::read_file(&mrml_runtime::join_path(&directory, name)).unwrap()
        };
        let dk = MlKem768DecapsulationKey(read("dk.bin")[..].try_into().unwrap());
        let ciphertext = MlKem768Ciphertext(read("c.bin")[..].try_into().unwrap());
        let expected = read("k.bin");
        assert_eq!(
            &ml_kem_768_decapsulate(&dk, &ciphertext).unwrap(),
            &expected[..]
        );
    }
}
