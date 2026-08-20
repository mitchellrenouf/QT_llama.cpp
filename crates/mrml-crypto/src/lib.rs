#![no_std]

mod aead;
mod aes_gcm;
mod chacha20;
mod hkdf;
mod ml_kem;
mod poly1305;
mod rsa;
mod sha256;
mod sha3;
mod x25519;

pub use aead::{chacha20_poly1305_open, chacha20_poly1305_seal};
pub use aes_gcm::{aes128_gcm_open, aes128_gcm_seal};
pub use chacha20::{chacha20_block, chacha20_xor};
pub use hkdf::{hkdf_expand, hkdf_expand_label, hkdf_extract, hmac_sha256};
pub use ml_kem::{
    MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError,
    ml_kem_768_decapsulate, ml_kem_768_encapsulate, ml_kem_768_keygen,
};
pub use poly1305::poly1305;
pub use rsa::{RsaError, rsa_pkcs1_sha256_verify, rsa_pss_sha256_verify};
pub use sha3::{Sha3_256, Sha3_512, Shake128, Shake256};
pub use sha256::Sha256;
pub use x25519::{x25519, x25519_public, x25519_shared};
