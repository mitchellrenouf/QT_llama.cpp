use crate::Sha3_512;

pub const LAMPORT_SIGNATURE_BYTES: usize = 512 * 64;
pub const LAMPORT_PRIVATE_KEY_BYTES: usize = 2 * LAMPORT_SIGNATURE_BYTES;
pub const LAMPORT_PUBLIC_KEY_BYTES: usize = LAMPORT_PRIVATE_KEY_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LamportError {
    InvalidPrivateKey,
    InvalidPublicKey,
    InvalidSignature,
}

/// Derives a Lamport public key from 1,024 independent 512-bit secret values.
/// A private key must sign exactly one artifact for its entire lifetime.
pub fn lamport_public_key(private_key: &[u8], public_key: &mut [u8]) -> Result<(), LamportError> {
    if private_key.len() != LAMPORT_PRIVATE_KEY_BYTES
        || public_key.len() != LAMPORT_PUBLIC_KEY_BYTES
    {
        return Err(LamportError::InvalidPrivateKey);
    }
    for (secret, public) in private_key
        .chunks_exact(64)
        .zip(public_key.chunks_exact_mut(64))
    {
        public.copy_from_slice(&Sha3_512::digest(secret));
    }
    Ok(())
}

pub fn lamport_sign(
    private_key: &[u8],
    message: &[u8],
    signature: &mut [u8],
) -> Result<(), LamportError> {
    if private_key.len() != LAMPORT_PRIVATE_KEY_BYTES || signature.len() != LAMPORT_SIGNATURE_BYTES
    {
        return Err(LamportError::InvalidPrivateKey);
    }
    let digest = artifact_digest(message);
    for bit in 0..512 {
        let choice = ((digest[bit / 8] >> (bit % 8)) & 1) as usize;
        let source = (bit * 2 + choice) * 64;
        signature[bit * 64..(bit + 1) * 64].copy_from_slice(&private_key[source..source + 64]);
    }
    Ok(())
}

pub fn lamport_verify(
    public_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), LamportError> {
    if public_key.len() != LAMPORT_PUBLIC_KEY_BYTES {
        return Err(LamportError::InvalidPublicKey);
    }
    if signature.len() != LAMPORT_SIGNATURE_BYTES {
        return Err(LamportError::InvalidSignature);
    }
    let digest = artifact_digest(message);
    let mut difference = 0u8;
    for bit in 0..512 {
        let choice = ((digest[bit / 8] >> (bit % 8)) & 1) as usize;
        let expected = (bit * 2 + choice) * 64;
        let actual = Sha3_512::digest(&signature[bit * 64..(bit + 1) * 64]);
        for index in 0..64 {
            difference |= actual[index] ^ public_key[expected + index];
        }
    }
    if difference == 0 {
        Ok(())
    } else {
        Err(LamportError::InvalidSignature)
    }
}

fn artifact_digest(message: &[u8]) -> [u8; 64] {
    let mut hash = Sha3_512::new();
    hash.update(b"MRML-LAMPORT-SHA3-512-v1\0");
    hash.update(&(message.len() as u64).to_le_bytes());
    hash.update(message);
    hash.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_verifies_and_rejects_tampering() {
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(131).wrapping_add(17) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        lamport_sign(&private, b"kernel image", &mut signature).unwrap();
        assert_eq!(lamport_verify(&public, b"kernel image", &signature), Ok(()));
        assert_eq!(
            lamport_verify(&public, b"modified image", &signature),
            Err(LamportError::InvalidSignature)
        );
        signature[10] ^= 1;
        assert_eq!(
            lamport_verify(&public, b"kernel image", &signature),
            Err(LamportError::InvalidSignature)
        );
    }
}
