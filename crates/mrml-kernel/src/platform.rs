#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    RiscV64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Hypervisor {
    BareMetal,
    HyperV,
    Kvm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmRole {
    Inference,
    Tool,
    Device,
    Storage,
    Network,
}

/// Isolation may only be relaxed after measurement proves IPC is material and
/// a threat-model review shows the combined services share a trust domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsolationClass {
    Kernel,
    TrustedService,
    UntrustedService,
}
