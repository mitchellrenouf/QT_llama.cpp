use crate::{
    Capability, CapabilitySpace, ObjectId, Rights, VmBackend, VmError, VmId, VmRunError, VmState,
    VmTable,
};

const FIRST_EXTERNAL_VECTOR: u8 = 32;
const LAST_EXTERNAL_VECTOR: u8 = 254;

/// Explicit allowlist for externally injected interrupt vectors. CPU exception
/// vectors and the spurious-vector sentinel are never injectable through this
/// interface.
pub struct InterruptPolicy {
    words: [u64; 4],
}

impl InterruptPolicy {
    pub const fn deny_all() -> Self {
        Self { words: [0; 4] }
    }

    pub fn allow(&mut self, vector: u8) -> Result<(), VmError> {
        if !(FIRST_EXTERNAL_VECTOR..=LAST_EXTERNAL_VECTOR).contains(&vector) {
            return Err(VmError::InterruptDenied);
        }
        self.words[vector as usize / 64] |= 1u64 << (vector as usize % 64);
        Ok(())
    }

    pub fn deny(&mut self, vector: u8) {
        self.words[vector as usize / 64] &= !(1u64 << (vector as usize % 64));
    }

    pub fn permits(&self, vector: u8) -> bool {
        (FIRST_EXTERNAL_VECTOR..=LAST_EXTERNAL_VECTOR).contains(&vector)
            && self.words[vector as usize / 64] & (1u64 << (vector as usize % 64)) != 0
    }
}

impl Default for InterruptPolicy {
    fn default() -> Self {
        Self::deny_all()
    }
}

/// Delivers an interrupt only when the caller owns SIGNAL authority for the
/// VM's interrupt object and the vector is explicitly allowlisted.
pub fn inject_vm_interrupt<B: VmBackend, const V: usize, const C: usize>(
    backend: &mut B,
    vms: &mut VmTable<V>,
    vm: VmId,
    vcpu: u32,
    vector: u8,
    policy: &InterruptPolicy,
    capabilities: &CapabilitySpace<C>,
    capability: Capability,
    interrupt_object: ObjectId,
) -> Result<(), VmRunError<B::Error>> {
    if vms.state(vm).map_err(VmRunError::Policy)? != VmState::Running {
        return Err(VmRunError::Policy(VmError::InvalidVmState));
    }
    if !policy.permits(vector) {
        return Err(VmRunError::Policy(VmError::InterruptDenied));
    }
    let object = capabilities
        .authorize(capability, Rights::SIGNAL)
        .map_err(|error| VmRunError::Policy(VmError::Capability(error)))?;
    if object != interrupt_object {
        return Err(VmRunError::Policy(VmError::WrongObject));
    }
    if let Err(error) = backend.inject_interrupt(vcpu, vector) {
        let _ = vms.fail(vm);
        return Err(VmRunError::Backend(error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VmExit;

    struct Backend {
        injected: Option<(u32, u8)>,
        fail: bool,
    }

    impl VmBackend for Backend {
        type Error = ();
        fn run(&mut self, _vcpu: u32) -> Result<VmExit, Self::Error> {
            Err(())
        }
        fn read_guest(&self, _address: u64, _output: &mut [u8]) -> Result<(), Self::Error> {
            Err(())
        }
        fn write_guest(&mut self, _address: u64, _input: &[u8]) -> Result<(), Self::Error> {
            Err(())
        }
        fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
            if self.fail {
                return Err(());
            }
            self.injected = Some((vcpu, vector));
            Ok(())
        }
    }

    fn running_vm() -> (VmTable<1>, VmId) {
        let mut table = VmTable::new();
        let id = table.create(4).unwrap();
        table.mark_loaded(id).unwrap();
        table.start(id).unwrap();
        (table, id)
    }

    #[test]
    fn exception_and_spurious_vectors_cannot_be_allowed() {
        let mut policy = InterruptPolicy::deny_all();
        assert_eq!(policy.allow(31), Err(VmError::InterruptDenied));
        assert_eq!(policy.allow(255), Err(VmError::InterruptDenied));
        policy.allow(32).unwrap();
        policy.allow(254).unwrap();
        assert!(policy.permits(32));
        policy.deny(32);
        assert!(!policy.permits(32));
    }

    #[test]
    fn injection_requires_vector_and_exact_capability_object() {
        let object = ObjectId(7);
        let wrong = ObjectId(8);
        let mut capabilities = CapabilitySpace::<2>::new();
        let right_cap = capabilities.insert(object, Rights::SIGNAL).unwrap();
        let wrong_cap = capabilities.insert(wrong, Rights::SIGNAL).unwrap();
        let mut policy = InterruptPolicy::deny_all();
        policy.allow(48).unwrap();
        let (mut vms, id) = running_vm();
        let mut backend = Backend {
            injected: None,
            fail: false,
        };
        assert_eq!(
            inject_vm_interrupt(
                &mut backend,
                &mut vms,
                id,
                2,
                49,
                &policy,
                &capabilities,
                right_cap,
                object
            ),
            Err(VmRunError::Policy(VmError::InterruptDenied))
        );
        assert_eq!(
            inject_vm_interrupt(
                &mut backend,
                &mut vms,
                id,
                2,
                48,
                &policy,
                &capabilities,
                wrong_cap,
                object
            ),
            Err(VmRunError::Policy(VmError::WrongObject))
        );
        inject_vm_interrupt(
            &mut backend,
            &mut vms,
            id,
            2,
            48,
            &policy,
            &capabilities,
            right_cap,
            object,
        )
        .unwrap();
        assert_eq!(backend.injected, Some((2, 48)));
    }

    #[test]
    fn uncertain_backend_injection_fails_vm() {
        let object = ObjectId(7);
        let mut capabilities = CapabilitySpace::<1>::new();
        let cap = capabilities.insert(object, Rights::SIGNAL).unwrap();
        let mut policy = InterruptPolicy::deny_all();
        policy.allow(48).unwrap();
        let (mut vms, id) = running_vm();
        let mut backend = Backend {
            injected: None,
            fail: true,
        };
        assert_eq!(
            inject_vm_interrupt(
                &mut backend,
                &mut vms,
                id,
                0,
                48,
                &policy,
                &capabilities,
                cap,
                object
            ),
            Err(VmRunError::Backend(()))
        );
        assert_eq!(vms.state(id), Ok(VmState::Failed));
    }
}
