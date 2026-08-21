use crate::{
    CapabilitySpace, GuestMemory, HypercallRouter, RoutedHypercall, VmBackend, VmError, VmExit,
    VmId, VmRunError, VmTable, decode_hypercall_exit,
};

pub const MAX_IO_PORT_RULES: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoDirection {
    Read,
    Write,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoPortRule {
    port: u16,
    size: u8,
    direction: IoDirection,
}

impl IoPortRule {
    pub fn new(port: u16, size: u8, direction: IoDirection) -> Result<Self, VmError> {
        if !matches!(size, 1 | 2 | 4) {
            return Err(VmError::InvalidIo);
        }
        Ok(Self {
            port,
            size,
            direction,
        })
    }

    fn permits(self, port: u16, size: u8, write: bool) -> bool {
        self.port == port
            && self.size == size
            && matches!(
                (self.direction, write),
                (IoDirection::Both, _) | (IoDirection::Write, true) | (IoDirection::Read, false)
            )
    }
}

pub struct IoPortPolicy<const N: usize> {
    rules: [Option<IoPortRule>; N],
    count: usize,
}

impl<const N: usize> IoPortPolicy<N> {
    pub const fn deny_all() -> Self {
        Self {
            rules: [None; N],
            count: 0,
        }
    }

    pub fn allow(&mut self, rule: IoPortRule) -> Result<(), VmError> {
        if N > MAX_IO_PORT_RULES || self.count == N {
            return Err(VmError::IoPolicyFull);
        }
        if self.rules[..self.count]
            .iter()
            .flatten()
            .any(|existing| existing.port == rule.port && existing.size == rule.size)
        {
            return Err(VmError::InvalidIo);
        }
        self.rules[self.count] = Some(rule);
        self.count += 1;
        Ok(())
    }

    pub fn permits(&self, port: u16, size: u8, write: bool) -> bool {
        self.rules[..self.count]
            .iter()
            .flatten()
            .any(|rule| rule.permits(port, size, write))
    }
}

/// Runs one vCPU until its next exit and applies the common policy. Backend
/// execution errors fail the VM instance because register and device state may
/// no longer be safely resumable.
// Keep policy stores explicit at this security boundary so callers cannot
// accidentally substitute a partially initialized aggregate context.
#[allow(clippy::too_many_arguments)]
pub fn run_vm_once<B: VmBackend, const V: usize, const C: usize, const M: usize, const I: usize>(
    backend: &mut B,
    vcpu: u32,
    vms: &mut VmTable<V>,
    vm: VmId,
    router: &mut HypercallRouter,
    capabilities: &CapabilitySpace<C>,
    memory: &GuestMemory<M>,
    io: &IoPortPolicy<I>,
) -> Result<ExitDisposition, VmRunError<B::Error>> {
    let exit = match backend.run(vcpu) {
        Ok(exit) => exit,
        Err(error) => {
            let _ = vms.fail(vm);
            return Err(VmRunError::Backend(error));
        }
    };
    dispatch_vm_exit(&*backend, vms, vm, exit, router, capabilities, memory, io)
}

impl<const N: usize> Default for IoPortPolicy<N> {
    fn default() -> Self {
        Self::deny_all()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitDisposition {
    Resume,
    Yield,
    Tool {
        host_address: u64,
        length: u32,
        argument: u64,
    },
    Gpu {
        host_address: u64,
        length: u32,
        argument: u64,
    },
    Io {
        port: u16,
        size: u8,
        write: bool,
        value: u32,
    },
    Stopped,
}

/// Accounts and dispatches one translated backend exit. Any malformed,
/// unauthorized, unknown, or fault exit permanently fails this VM instance.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_vm_exit<
    B: VmBackend,
    const V: usize,
    const C: usize,
    const M: usize,
    const I: usize,
>(
    backend: &B,
    vms: &mut VmTable<V>,
    vm: VmId,
    exit: VmExit,
    router: &mut HypercallRouter,
    capabilities: &CapabilitySpace<C>,
    memory: &GuestMemory<M>,
    io: &IoPortPolicy<I>,
) -> Result<ExitDisposition, VmRunError<B::Error>> {
    if let Err(error) = vms.account_exit(vm) {
        return Err(VmRunError::Policy(error));
    }
    let result = dispatch_inner(backend, exit, router, capabilities, memory, io);
    let disposition = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = vms.fail(vm);
            return Err(error);
        }
    };
    if disposition == ExitDisposition::Stopped {
        vms.stop(vm).map_err(VmRunError::Policy)?;
    }
    Ok(disposition)
}

fn dispatch_inner<B: VmBackend, const C: usize, const M: usize, const I: usize>(
    backend: &B,
    exit: VmExit,
    router: &mut HypercallRouter,
    capabilities: &CapabilitySpace<C>,
    memory: &GuestMemory<M>,
    io: &IoPortPolicy<I>,
) -> Result<ExitDisposition, VmRunError<B::Error>> {
    match exit {
        VmExit::Hypercall { .. } => {
            let call = decode_hypercall_exit(backend, exit)?
                .ok_or(VmRunError::Policy(VmError::UnhandledExit))?;
            match router
                .route(call, capabilities, memory)
                .map_err(VmRunError::Policy)?
            {
                RoutedHypercall::Yield => Ok(ExitDisposition::Yield),
                RoutedHypercall::Tool {
                    host_address,
                    length,
                    argument,
                } => Ok(ExitDisposition::Tool {
                    host_address,
                    length,
                    argument,
                }),
                RoutedHypercall::Gpu {
                    host_address,
                    length,
                    argument,
                } => Ok(ExitDisposition::Gpu {
                    host_address,
                    length,
                    argument,
                }),
                RoutedHypercall::Shutdown => Ok(ExitDisposition::Stopped),
            }
        }
        VmExit::Io {
            port,
            size,
            write,
            value,
        } if io.permits(port, size, write) => Ok(ExitDisposition::Io {
            port,
            size,
            write,
            value,
        }),
        VmExit::Io { .. } => Err(VmRunError::Policy(VmError::IoDenied)),
        VmExit::Interrupted => Ok(ExitDisposition::Resume),
        VmExit::Halted => Ok(ExitDisposition::Stopped),
        VmExit::GuestMemoryFault { .. } | VmExit::Mmio { .. } | VmExit::Unknown { .. } => {
            Err(VmRunError::Policy(VmError::UnhandledExit))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hypercall, ObjectId, Rights, VmState};

    struct Backend {
        wire: [u8; crate::HYPERCALL_BYTES],
        address: u64,
    }

    impl VmBackend for Backend {
        type Error = ();
        fn run(&mut self, _vcpu: u32) -> Result<VmExit, Self::Error> {
            Ok(VmExit::Interrupted)
        }
        fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
            if address != self.address || output.len() != self.wire.len() {
                return Err(());
            }
            output.copy_from_slice(&self.wire);
            Ok(())
        }
        fn write_guest(&mut self, _address: u64, _input: &[u8]) -> Result<(), Self::Error> {
            Err(())
        }
        fn inject_interrupt(&mut self, _vcpu: u32, _vector: u8) -> Result<(), Self::Error> {
            Err(())
        }
    }

    fn running_vm() -> (VmTable<1>, VmId) {
        let mut table = VmTable::new();
        let id = table.create(8).unwrap();
        table.mark_loaded(id).unwrap();
        table.start(id).unwrap();
        (table, id)
    }

    #[test]
    fn io_is_deny_by_default_and_exactly_allowlisted() {
        let backend = Backend {
            wire: [0; crate::HYPERCALL_BYTES],
            address: 0,
        };
        let capabilities = CapabilitySpace::<1>::new();
        let memory = GuestMemory::<1>::new();
        let mut router = HypercallRouter::new(ObjectId(1), ObjectId(2), ObjectId(3));
        let (mut vms, id) = running_vm();
        assert_eq!(
            dispatch_vm_exit(
                &backend,
                &mut vms,
                id,
                VmExit::Io {
                    port: 0x3f8,
                    size: 1,
                    write: true,
                    value: 65
                },
                &mut router,
                &capabilities,
                &memory,
                &IoPortPolicy::<1>::deny_all()
            ),
            Err(VmRunError::Policy(VmError::IoDenied))
        );
        assert_eq!(vms.state(id), Ok(VmState::Failed));

        let (mut vms, id) = running_vm();
        let mut policy = IoPortPolicy::<1>::deny_all();
        policy
            .allow(IoPortRule::new(0x3f8, 1, IoDirection::Write).unwrap())
            .unwrap();
        assert_eq!(
            dispatch_vm_exit(
                &backend,
                &mut vms,
                id,
                VmExit::Io {
                    port: 0x3f8,
                    size: 1,
                    write: true,
                    value: 65
                },
                &mut router,
                &capabilities,
                &memory,
                &policy
            ),
            Ok(ExitDisposition::Io {
                port: 0x3f8,
                size: 1,
                write: true,
                value: 65
            })
        );
    }

    #[test]
    fn shutdown_requires_control_capability_and_stops_vm() {
        let control = ObjectId(3);
        let mut capabilities = CapabilitySpace::<1>::new();
        let cap = capabilities.insert(control, Rights::SIGNAL).unwrap();
        let call = Hypercall::shutdown(1, cap);
        let backend = Backend {
            wire: call.encode(),
            address: 0x1000,
        };
        let memory = GuestMemory::<1>::new();
        let mut router = HypercallRouter::new(ObjectId(1), ObjectId(2), control);
        let (mut vms, id) = running_vm();
        assert_eq!(
            dispatch_vm_exit(
                &backend,
                &mut vms,
                id,
                VmExit::Hypercall {
                    descriptor_address: 0x1000
                },
                &mut router,
                &capabilities,
                &memory,
                &IoPortPolicy::<0>::deny_all()
            ),
            Ok(ExitDisposition::Stopped)
        );
        assert_eq!(vms.state(id), Ok(VmState::Stopped));
    }

    #[test]
    fn faults_and_unknown_exits_fail_the_vm() {
        let backend = Backend {
            wire: Hypercall::yield_call(1).encode(),
            address: 0,
        };
        let capabilities = CapabilitySpace::<1>::new();
        let memory = GuestMemory::<1>::new();
        let mut router = HypercallRouter::new(ObjectId(1), ObjectId(2), ObjectId(3));
        let (mut vms, id) = running_vm();
        assert_eq!(
            dispatch_vm_exit(
                &backend,
                &mut vms,
                id,
                VmExit::Unknown { reason: 9 },
                &mut router,
                &capabilities,
                &memory,
                &IoPortPolicy::<0>::deny_all()
            ),
            Err(VmRunError::Policy(VmError::UnhandledExit))
        );
        assert_eq!(vms.state(id), Ok(VmState::Failed));
    }

    #[test]
    fn run_loop_accounts_translated_backend_exits() {
        let mut backend = Backend {
            wire: Hypercall::yield_call(1).encode(),
            address: 0,
        };
        let capabilities = CapabilitySpace::<1>::new();
        let memory = GuestMemory::<1>::new();
        let mut router = HypercallRouter::new(ObjectId(1), ObjectId(2), ObjectId(3));
        let (mut vms, id) = running_vm();
        assert_eq!(
            run_vm_once(
                &mut backend,
                0,
                &mut vms,
                id,
                &mut router,
                &capabilities,
                &memory,
                &IoPortPolicy::<0>::deny_all()
            ),
            Ok(ExitDisposition::Resume)
        );
        assert_eq!(vms.state(id), Ok(VmState::Running));
    }
}
