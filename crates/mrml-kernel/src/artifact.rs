use mrml_crypto::{LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, Sha3_512, lamport_verify};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ArtifactKind {
    Kernel = 1,
    VmImage = 2,
    ServiceImage = 3,
    CudaKernelBundle = 4,
    LaunchPolicy = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    InvalidPublicKey,
    UntrustedPublicKey,
    InvalidSignature,
    RollbackDetected,
    EmptyArtifact,
}

pub struct TrustRoot {
    kind: ArtifactKind,
    public_key_digest: [u8; 64],
    minimum_version: u64,
}

impl TrustRoot {
    pub const fn new(
        kind: ArtifactKind,
        public_key_digest: [u8; 64],
        minimum_version: u64,
    ) -> Self {
        Self {
            kind,
            public_key_digest,
            minimum_version,
        }
    }

    pub fn verify(
        &self,
        version: u64,
        artifact: &[u8],
        public_key: &[u8],
        signature: &[u8],
    ) -> Result<VerifiedArtifact, ArtifactError> {
        if artifact.is_empty() {
            return Err(ArtifactError::EmptyArtifact);
        }
        if version < self.minimum_version {
            return Err(ArtifactError::RollbackDetected);
        }
        if public_key.len() != LAMPORT_PUBLIC_KEY_BYTES {
            return Err(ArtifactError::InvalidPublicKey);
        }
        let candidate = Sha3_512::digest(public_key);
        let mut difference = 0u8;
        for index in 0..64 {
            difference |= candidate[index] ^ self.public_key_digest[index];
        }
        if difference != 0 {
            return Err(ArtifactError::UntrustedPublicKey);
        }
        if signature.len() != LAMPORT_SIGNATURE_BYTES {
            return Err(ArtifactError::InvalidSignature);
        }
        let digest = Sha3_512::digest(artifact);
        let statement = artifact_statement(self.kind, version, artifact.len() as u64, digest);
        lamport_verify(public_key, &statement, signature)
            .map_err(|_| ArtifactError::InvalidSignature)?;
        Ok(VerifiedArtifact {
            kind: self.kind,
            version,
            digest,
        })
    }
}

pub struct VerifiedArtifact {
    kind: ArtifactKind,
    version: u64,
    digest: [u8; 64],
}

impl VerifiedArtifact {
    pub const fn kind(&self) -> ArtifactKind {
        self.kind
    }
    pub const fn version(&self) -> u64 {
        self.version
    }
    pub const fn digest(&self) -> &[u8; 64] {
        &self.digest
    }
}

pub fn artifact_statement(
    kind: ArtifactKind,
    version: u64,
    length: u64,
    digest: [u8; 64],
) -> [u8; 104] {
    let mut statement = [0u8; 104];
    statement[..17].copy_from_slice(b"MRML-ARTIFACT-v1\0");
    statement[17] = kind as u8;
    statement[24..32].copy_from_slice(&version.to_le_bytes());
    statement[32..40].copy_from_slice(&length.to_le_bytes());
    statement[40..].copy_from_slice(&digest);
    statement
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_crypto::{LAMPORT_PRIVATE_KEY_BYTES, lamport_public_key, lamport_sign};

    #[test]
    fn trust_root_binds_kind_version_length_key_and_content() {
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(73).wrapping_add(9) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        let artifact = b"measured CUDA PTX bundle";
        let statement = artifact_statement(
            ArtifactKind::CudaKernelBundle,
            4,
            artifact.len() as u64,
            Sha3_512::digest(artifact),
        );
        lamport_sign(&private, &statement, &mut signature).unwrap();
        let root = TrustRoot::new(ArtifactKind::CudaKernelBundle, Sha3_512::digest(&public), 4);
        assert!(root.verify(4, artifact, &public, &signature).is_ok());
        assert_eq!(
            root.verify(3, artifact, &public, &signature).err(),
            Some(ArtifactError::RollbackDetected)
        );
        assert_eq!(
            root.verify(4, b"changed", &public, &signature).err(),
            Some(ArtifactError::InvalidSignature)
        );
        let wrong_kind = TrustRoot::new(ArtifactKind::VmImage, Sha3_512::digest(&public), 4);
        assert_eq!(
            wrong_kind.verify(4, artifact, &public, &signature).err(),
            Some(ArtifactError::InvalidSignature)
        );
        public[0] ^= 1;
        assert_eq!(
            root.verify(4, artifact, &public, &signature).err(),
            Some(ArtifactError::UntrustedPublicKey)
        );
    }
}
