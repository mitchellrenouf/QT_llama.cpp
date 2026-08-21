#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BootValidationError {
    MissingEntropy,
    MissingMeasurement,
    SecureBootRequired,
    MeasuredBootRequired,
    RollbackDetected,
    MissingSignedArtifact,
    DuplicateSignedArtifact,
    MixedArtifactRelease,
    KernelMeasurementMismatch,
}

/// Evidence normalized by the architecture-specific UEFI loader. The kernel
/// treats firmware claims as evidence to validate, never as implicit trust.
pub struct BootEvidence {
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    image_version: u64,
    secure_boot: bool,
    measured_boot: bool,
    rollback_protected: bool,
}

impl BootEvidence {
    pub fn new(
        entropy: [u8; 32],
        image_measurement: [u8; 64],
        image_version: u64,
        secure_boot: bool,
        measured_boot: bool,
        rollback_protected: bool,
    ) -> Result<Self, BootValidationError> {
        if entropy.iter().all(|byte| *byte == 0) {
            return Err(BootValidationError::MissingEntropy);
        }
        if image_measurement.iter().all(|byte| *byte == 0) {
            return Err(BootValidationError::MissingMeasurement);
        }
        Ok(Self {
            entropy,
            image_measurement,
            image_version,
            secure_boot,
            measured_boot,
            rollback_protected,
        })
    }

    pub const fn entropy(&self) -> &[u8; 32] {
        &self.entropy
    }

    pub const fn image_measurement(&self) -> &[u8; 64] {
        &self.image_measurement
    }

    pub const fn image_version(&self) -> u64 {
        self.image_version
    }

    pub const fn secure_boot(&self) -> bool {
        self.secure_boot
    }

    pub const fn measured_boot(&self) -> bool {
        self.measured_boot
    }

    pub const fn rollback_protected(&self) -> bool {
        self.rollback_protected
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootPolicy {
    minimum_image_version: u64,
    require_secure_boot: bool,
    require_measured_boot: bool,
    require_rollback_protection: bool,
}

impl BootPolicy {
    pub const fn production(minimum_image_version: u64) -> Self {
        Self {
            minimum_image_version,
            require_secure_boot: true,
            require_measured_boot: true,
            require_rollback_protection: true,
        }
    }

    pub const fn development(minimum_image_version: u64) -> Self {
        Self {
            minimum_image_version,
            require_secure_boot: false,
            require_measured_boot: false,
            require_rollback_protection: false,
        }
    }

    pub const fn validate(self, evidence: &BootEvidence) -> Result<(), BootValidationError> {
        if self.require_secure_boot && !evidence.secure_boot {
            return Err(BootValidationError::SecureBootRequired);
        }
        if self.require_measured_boot && !evidence.measured_boot {
            return Err(BootValidationError::MeasuredBootRequired);
        }
        if self.require_rollback_protection && !evidence.rollback_protected {
            return Err(BootValidationError::RollbackDetected);
        }
        if evidence.image_version < self.minimum_image_version {
            return Err(BootValidationError::RollbackDetected);
        }
        Ok(())
    }

    pub fn validate_signed_chain(
        self,
        evidence: &BootEvidence,
        artifacts: &[&crate::VerifiedArtifact],
    ) -> Result<(), BootValidationError> {
        self.validate(evidence)?;
        let mut present = [false; 5];
        for artifact in artifacts {
            if artifact.version() < self.minimum_image_version {
                return Err(BootValidationError::RollbackDetected);
            }
            if artifact.version() != evidence.image_version {
                return Err(BootValidationError::MixedArtifactRelease);
            }
            let index = artifact.kind() as usize - 1;
            if present[index] {
                return Err(BootValidationError::DuplicateSignedArtifact);
            }
            present[index] = true;
            if artifact.kind() == crate::ArtifactKind::Kernel
                && artifact.digest() != evidence.image_measurement()
            {
                return Err(BootValidationError::KernelMeasurementMismatch);
            }
        }
        if present.iter().any(|value| !value) {
            return Err(BootValidationError::MissingSignedArtifact);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtifactKind, TrustRoot, artifact_statement};
    use mrml_crypto::{
        LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, Sha3_512,
        lamport_public_key, lamport_sign,
    };

    fn evidence(version: u64, secure: bool, measured: bool) -> BootEvidence {
        BootEvidence::new([1; 32], [2; 64], version, secure, measured, true).unwrap()
    }

    #[test]
    fn rejects_absent_entropy_and_measurement() {
        assert!(matches!(
            BootEvidence::new([0; 32], [1; 64], 1, true, true, true),
            Err(BootValidationError::MissingEntropy)
        ));
        assert!(matches!(
            BootEvidence::new([1; 32], [0; 64], 1, true, true, true),
            Err(BootValidationError::MissingMeasurement)
        ));
    }

    #[test]
    fn production_requires_secure_measured_non_rollback_boot() {
        let policy = BootPolicy::production(7);
        assert_eq!(
            policy.validate(&evidence(7, false, true)),
            Err(BootValidationError::SecureBootRequired)
        );
        assert_eq!(
            policy.validate(&evidence(7, true, false)),
            Err(BootValidationError::MeasuredBootRequired)
        );
        assert_eq!(
            policy.validate(&evidence(6, true, true)),
            Err(BootValidationError::RollbackDetected)
        );
        let no_counter = BootEvidence::new([1; 32], [2; 64], 7, true, true, false).unwrap();
        assert_eq!(
            policy.validate(&no_counter),
            Err(BootValidationError::RollbackDetected)
        );
        assert_eq!(policy.validate(&evidence(7, true, true)), Ok(()));
    }

    fn signed_artifact(
        kind: ArtifactKind,
        version: u64,
        content: &[u8],
        private: &[u8; LAMPORT_PRIVATE_KEY_BYTES],
        public: &[u8; LAMPORT_PUBLIC_KEY_BYTES],
    ) -> crate::VerifiedArtifact {
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        let statement = artifact_statement(
            kind,
            version,
            content.len() as u64,
            Sha3_512::digest(content),
        );
        lamport_sign(private, &statement, &mut signature).unwrap();
        TrustRoot::new(kind, Sha3_512::digest(public), version)
            .verify(version, content, public, &signature)
            .unwrap()
    }

    #[test]
    fn signed_chain_requires_one_measured_coherent_release() {
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(41).wrapping_add(17) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        let kernel = b"kernel release eight";
        let artifacts = [
            signed_artifact(ArtifactKind::Kernel, 8, kernel, &private, &public),
            signed_artifact(ArtifactKind::VmImage, 8, b"vm", &private, &public),
            signed_artifact(ArtifactKind::ServiceImage, 8, b"service", &private, &public),
            signed_artifact(
                ArtifactKind::CudaKernelBundle,
                8,
                b"cuda",
                &private,
                &public,
            ),
            signed_artifact(ArtifactKind::LaunchPolicy, 8, b"policy", &private, &public),
        ];
        let references = [
            &artifacts[0],
            &artifacts[1],
            &artifacts[2],
            &artifacts[3],
            &artifacts[4],
        ];
        let valid =
            BootEvidence::new([1; 32], Sha3_512::digest(kernel), 8, true, true, true).unwrap();
        assert_eq!(
            BootPolicy::production(8).validate_signed_chain(&valid, &references),
            Ok(())
        );

        let wrong_measurement = BootEvidence::new([1; 32], [9; 64], 8, true, true, true).unwrap();
        assert_eq!(
            BootPolicy::production(8).validate_signed_chain(&wrong_measurement, &references),
            Err(BootValidationError::KernelMeasurementMismatch)
        );

        let wrong_release =
            BootEvidence::new([1; 32], Sha3_512::digest(kernel), 9, true, true, true).unwrap();
        assert_eq!(
            BootPolicy::production(8).validate_signed_chain(&wrong_release, &references),
            Err(BootValidationError::MixedArtifactRelease)
        );

        let duplicate = [
            &artifacts[0],
            &artifacts[0],
            &artifacts[2],
            &artifacts[3],
            &artifacts[4],
        ];
        assert_eq!(
            BootPolicy::production(8).validate_signed_chain(&valid, &duplicate),
            Err(BootValidationError::DuplicateSignedArtifact)
        );
    }
}
