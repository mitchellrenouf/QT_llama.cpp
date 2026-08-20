#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BootValidationError {
    MissingEntropy,
    MissingMeasurement,
    SecureBootRequired,
    MeasuredBootRequired,
    RollbackDetected,
}

/// Evidence normalized by the architecture-specific UEFI loader. The kernel
/// treats firmware claims as evidence to validate, never as implicit trust.
pub struct BootEvidence {
    entropy: [u8; 32],
    image_measurement: [u8; 64],
    image_version: u64,
    secure_boot: bool,
    measured_boot: bool,
}

impl BootEvidence {
    pub fn new(
        entropy: [u8; 32],
        image_measurement: [u8; 64],
        image_version: u64,
        secure_boot: bool,
        measured_boot: bool,
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
        })
    }

    pub const fn entropy(&self) -> &[u8; 32] {
        &self.entropy
    }

    pub const fn image_measurement(&self) -> &[u8; 64] {
        &self.image_measurement
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootPolicy {
    minimum_image_version: u64,
    require_secure_boot: bool,
    require_measured_boot: bool,
}

impl BootPolicy {
    pub const fn production(minimum_image_version: u64) -> Self {
        Self {
            minimum_image_version,
            require_secure_boot: true,
            require_measured_boot: true,
        }
    }

    pub const fn development(minimum_image_version: u64) -> Self {
        Self {
            minimum_image_version,
            require_secure_boot: false,
            require_measured_boot: false,
        }
    }

    pub const fn validate(self, evidence: &BootEvidence) -> Result<(), BootValidationError> {
        if self.require_secure_boot && !evidence.secure_boot {
            return Err(BootValidationError::SecureBootRequired);
        }
        if self.require_measured_boot && !evidence.measured_boot {
            return Err(BootValidationError::MeasuredBootRequired);
        }
        if evidence.image_version < self.minimum_image_version {
            return Err(BootValidationError::RollbackDetected);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(version: u64, secure: bool, measured: bool) -> BootEvidence {
        BootEvidence::new([1; 32], [2; 64], version, secure, measured).unwrap()
    }

    #[test]
    fn rejects_absent_entropy_and_measurement() {
        assert!(matches!(
            BootEvidence::new([0; 32], [1; 64], 1, true, true),
            Err(BootValidationError::MissingEntropy)
        ));
        assert!(matches!(
            BootEvidence::new([1; 32], [0; 64], 1, true, true),
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
        assert_eq!(policy.validate(&evidence(7, true, true)), Ok(()));
    }
}
