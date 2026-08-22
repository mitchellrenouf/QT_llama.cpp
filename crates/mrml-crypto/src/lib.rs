#![no_std]

#[cfg(feature = "runtime")]
mod aead;
mod aes_gcm;
mod chacha20;
#[cfg(feature = "runtime")]
mod hkdf;
mod hmac;
mod lamport;
#[cfg(feature = "runtime")]
mod ml_kem;
mod poly1305;
#[cfg(feature = "runtime")]
mod rsa;
mod sha256;
mod sha3;
mod x25519;

#[cfg(feature = "runtime")]
pub use aead::{chacha20_poly1305_open, chacha20_poly1305_seal};
pub use aes_gcm::{aes128_ctr_xor, aes128_gcm_open, aes128_gcm_seal};
pub use chacha20::{chacha20_block, chacha20_xor};
#[cfg(feature = "runtime")]
pub use hkdf::{hkdf_expand, hkdf_expand_label, hkdf_extract};
pub use hmac::hmac_sha256;
pub use lamport::{
    LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, LamportError,
    lamport_public_key, lamport_sign, lamport_verify,
};
#[cfg(feature = "runtime")]
pub use ml_kem::{
    MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError,
    ml_kem_768_decapsulate, ml_kem_768_encapsulate, ml_kem_768_keygen,
};
pub use poly1305::poly1305;
#[cfg(feature = "runtime")]
pub use rsa::{
    RsaError, rsa_pkcs1_sha256_sign, rsa_pkcs1_sha256_verify, rsa_pss_sha256_sign,
    rsa_pss_sha256_verify,
};
pub use sha3::{Sha3_256, Sha3_512, Shake128, Shake256};
pub use sha256::Sha256;
pub use x25519::{x25519, x25519_public, x25519_shared};
