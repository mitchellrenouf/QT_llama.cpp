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
    MalformedManifest,
    MissingTrustRoot,
    ReusedTrustRoot,
    VersionExhausted,
    MalformedBootstrapState,
    UnauthenticatedBootstrapState,
    StateConflict,
    StateStorageFailure,
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

pub const RELEASE_MANIFEST_BYTES: usize = 408;

#[derive(Clone, Copy)]
pub struct ReleaseManifest {
    release: u64,
    next_root_digest: [u8; 64],
    artifact_roots: [[u8; 64]; 5],
}

impl ReleaseManifest {
    pub fn new(
        release: u64,
        next_root_digest: [u8; 64],
        artifact_roots: [[u8; 64]; 5],
    ) -> Result<Self, ArtifactError> {
        if release == 0
            || next_root_digest.iter().all(|byte| *byte == 0)
            || artifact_roots
                .iter()
                .any(|root| root.iter().all(|byte| *byte == 0))
        {
            return Err(ArtifactError::MissingTrustRoot);
        }
        Ok(Self {
            release,
            next_root_digest,
            artifact_roots,
        })
    }

    pub fn encode(self) -> [u8; RELEASE_MANIFEST_BYTES] {
        let mut output = [0u8; RELEASE_MANIFEST_BYTES];
        output[..16].copy_from_slice(b"MRML-RELEASE-v1\0");
        output[16..24].copy_from_slice(&self.release.to_le_bytes());
        output[24..88].copy_from_slice(&self.next_root_digest);
        for (index, root) in self.artifact_roots.iter().enumerate() {
            output[88 + index * 64..152 + index * 64].copy_from_slice(root);
        }
        output
    }

    pub fn decode(input: &[u8]) -> Result<Self, ArtifactError> {
        if input.len() != RELEASE_MANIFEST_BYTES || &input[..16] != b"MRML-RELEASE-v1\0" {
            return Err(ArtifactError::MalformedManifest);
        }
        let release = u64::from_le_bytes(input[16..24].try_into().unwrap());
        let next_root_digest = input[24..88].try_into().unwrap();
        let artifact_roots = core::array::from_fn(|index| {
            input[88 + index * 64..152 + index * 64].try_into().unwrap()
        });
        Self::new(release, next_root_digest, artifact_roots)
    }
}

pub struct BootstrapState {
    current_root_digest: [u8; 64],
    minimum_release: u64,
}

pub const BOOTSTRAP_STATE_BYTES: usize = 88;

/// Backend contract for TPM NV or an equivalently authenticated monotonic
/// store. Implementations must authenticate reads and make compare-and-store
/// atomic across power loss; ordinary files do not satisfy this contract.
pub trait MonotonicStateStore {
    fn load_authenticated(
        &self,
        output: &mut [u8; BOOTSTRAP_STATE_BYTES],
    ) -> Result<(), ArtifactError>;

    fn compare_and_store(
        &mut self,
        expected_minimum_release: u64,
        replacement: &[u8; BOOTSTRAP_STATE_BYTES],
    ) -> Result<(), ArtifactError>;
}

impl BootstrapState {
    pub const fn genesis(current_root_digest: [u8; 64], minimum_release: u64) -> Self {
        Self {
            current_root_digest,
            minimum_release,
        }
    }

    pub fn verify_release(
        &self,
        encoded: &[u8],
        public_key: &[u8],
        signature: &[u8],
    ) -> Result<(VerifiedRelease, Self), ArtifactError> {
        let manifest = ReleaseManifest::decode(encoded)?;
        TrustRoot::new(
            ArtifactKind::LaunchPolicy,
            self.current_root_digest,
            self.minimum_release,
        )
        .verify(manifest.release, encoded, public_key, signature)?;
        if constant_time_equal(&manifest.next_root_digest, &self.current_root_digest) {
            return Err(ArtifactError::ReusedTrustRoot);
        }
        let minimum_release = manifest
            .release
            .checked_add(1)
            .ok_or(ArtifactError::VersionExhausted)?;
        Ok((
            VerifiedRelease { manifest },
            Self {
                current_root_digest: manifest.next_root_digest,
                minimum_release,
            },
        ))
    }

    pub const fn minimum_release(&self) -> u64 {
        self.minimum_release
    }

    pub fn encode(self) -> [u8; BOOTSTRAP_STATE_BYTES] {
        let mut output = [0u8; BOOTSTRAP_STATE_BYTES];
        output[..16].copy_from_slice(b"MRML-BOOTSTATE1\0");
        output[16..24].copy_from_slice(&self.minimum_release.to_le_bytes());
        output[24..88].copy_from_slice(&self.current_root_digest);
        output
    }

    pub fn decode_authenticated(input: &[u8]) -> Result<Self, ArtifactError> {
        if input.len() != BOOTSTRAP_STATE_BYTES
            || &input[..16] != b"MRML-BOOTSTATE1\0"
            || input[24..88].iter().all(|byte| *byte == 0)
        {
            return Err(ArtifactError::MalformedBootstrapState);
        }
        let minimum_release = u64::from_le_bytes(input[16..24].try_into().unwrap());
        if minimum_release == 0 {
            return Err(ArtifactError::MalformedBootstrapState);
        }
        Ok(Self {
            current_root_digest: input[24..88].try_into().unwrap(),
            minimum_release,
        })
    }

    pub fn verify_and_commit<S: MonotonicStateStore>(
        store: &mut S,
        encoded_manifest: &[u8],
        root_public_key: &[u8],
        signature: &[u8],
    ) -> Result<VerifiedRelease, ArtifactError> {
        let mut encoded_state = [0u8; BOOTSTRAP_STATE_BYTES];
        store.load_authenticated(&mut encoded_state)?;
        let current = Self::decode_authenticated(&encoded_state)?;
        let (release, next) =
            current.verify_release(encoded_manifest, root_public_key, signature)?;
        store.compare_and_store(current.minimum_release, &next.encode())?;
        Ok(release)
    }
}

pub struct VerifiedRelease {
    manifest: ReleaseManifest,
}

impl VerifiedRelease {
    pub fn trust_root(&self, kind: ArtifactKind) -> TrustRoot {
        TrustRoot::new(
            kind,
            self.manifest.artifact_roots[kind as usize - 1],
            self.manifest.release,
        )
    }
    pub const fn release(&self) -> u64 {
        self.manifest.release
    }
}

fn constant_time_equal(left: &[u8; 64], right: &[u8; 64]) -> bool {
    let mut difference = 0;
    for index in 0..64 {
        difference |= left[index] ^ right[index];
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_crypto::{LAMPORT_PRIVATE_KEY_BYTES, lamport_public_key, lamport_sign};

    struct TestStateStore {
        state: [u8; BOOTSTRAP_STATE_BYTES],
        force_conflict: bool,
    }

    impl MonotonicStateStore for TestStateStore {
        fn load_authenticated(
            &self,
            output: &mut [u8; BOOTSTRAP_STATE_BYTES],
        ) -> Result<(), ArtifactError> {
            output.copy_from_slice(&self.state);
            Ok(())
        }

        fn compare_and_store(
            &mut self,
            expected_minimum_release: u64,
            replacement: &[u8; BOOTSTRAP_STATE_BYTES],
        ) -> Result<(), ArtifactError> {
            let current = BootstrapState::decode_authenticated(&self.state)?;
            if self.force_conflict || current.minimum_release() != expected_minimum_release {
                return Err(ArtifactError::StateConflict);
            }
            self.state.copy_from_slice(replacement);
            Ok(())
        }
    }

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

    #[test]
    fn signed_release_rotates_root_and_rejects_replay() {
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(29).wrapping_add(3) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        let manifest =
            ReleaseManifest::new(7, [9; 64], [[1; 64], [2; 64], [3; 64], [4; 64], [5; 64]])
                .unwrap()
                .encode();
        let statement = artifact_statement(
            ArtifactKind::LaunchPolicy,
            7,
            manifest.len() as u64,
            Sha3_512::digest(&manifest),
        );
        lamport_sign(&private, &statement, &mut signature).unwrap();
        let state = BootstrapState::genesis(Sha3_512::digest(&public), 7);
        let (release, next) = state
            .verify_release(&manifest, &public, &signature)
            .unwrap();
        assert_eq!(release.release(), 7);
        assert_eq!(next.minimum_release(), 8);
        assert_eq!(
            next.verify_release(&manifest, &public, &signature).err(),
            Some(ArtifactError::RollbackDetected)
        );

        let mut store = TestStateStore {
            state: BootstrapState::genesis(Sha3_512::digest(&public), 7).encode(),
            force_conflict: false,
        };
        let committed =
            BootstrapState::verify_and_commit(&mut store, &manifest, &public, &signature).unwrap();
        assert_eq!(committed.release(), 7);
        assert_eq!(
            BootstrapState::decode_authenticated(&store.state)
                .unwrap()
                .minimum_release(),
            8
        );
        assert_eq!(
            BootstrapState::verify_and_commit(&mut store, &manifest, &public, &signature).err(),
            Some(ArtifactError::RollbackDetected)
        );
    }

    #[test]
    fn bootstrap_state_rejects_corruption_and_atomic_conflicts() {
        let root = [7; 64];
        let encoded = BootstrapState::genesis(root, 3).encode();
        assert_eq!(
            BootstrapState::decode_authenticated(&encoded)
                .unwrap()
                .minimum_release(),
            3
        );

        let mut corrupt = encoded;
        corrupt[0] ^= 1;
        assert_eq!(
            BootstrapState::decode_authenticated(&corrupt).err(),
            Some(ArtifactError::MalformedBootstrapState)
        );
        let mut zero_version = encoded;
        zero_version[16..24].fill(0);
        assert_eq!(
            BootstrapState::decode_authenticated(&zero_version).err(),
            Some(ArtifactError::MalformedBootstrapState)
        );

        let mut store = TestStateStore {
            state: encoded,
            force_conflict: true,
        };
        assert_eq!(
            store.compare_and_store(3, &BootstrapState::genesis([8; 64], 4).encode()),
            Err(ArtifactError::StateConflict)
        );
        assert_eq!(store.state, encoded);
    }
}
