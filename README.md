# MRML

MRML—the **Mitchell Renouf Machine Learning Library**—is an AI-led,
human-directed experimental computing stack, implemented from first principles
in dependency-free Rust and validated incrementally across UEFI, KVM, Hyper-V,
Windows, and Linux.

The project began as a native local-LLM runtime and is growing into a sovereign
software stack: inference, media codecs, development tools, cryptography,
network protocols, virtualization, verified boot, and an x86_64 microkernel are
implemented inside this repository. The objective is not merely to replace
individual libraries. It is to minimize trusted dependencies from source code
and signed builds through boot, isolated execution, and accelerated inference.

MRML is dedicated to the public domain under CC0. It is experimental research,
not a production-secure operating system, a complete Rust compiler, or a
drop-in replacement for mature codec and inference ecosystems.

## Project scope

| Area | Repository-owned implementation | Current maturity |
| --- | --- | --- |
| Local inference | GGUF loading, tokenizer, Gemma-family execution, CPU kernels, Rust-to-PTX CUDA kernels, KV cache, sampling, CLI, JSONL automation, and HTTPS/SSE service | Functional but narrower and less tuned than established runtimes |
| Microkernel | PE32+ x86_64 kernel, UEFI handoff, page tables, GDT/IDT/TSS, CPL3 tasks, APIC timer, preemption, capabilities, IPC, fault retirement, SMP, and service lifecycle | Booting research kernel with signed QEMU, KVM, and WHP probes; not production ready |
| Virtualization | Common VM policy plus native KVM and Hyper-V/WHP launchers, signed launch manifests, isolated guest memory, interrupt handling, and authenticated queues | Development and benchmark harnesses; a compromised host remains authoritative |
| Mediated GPU | Bounded authenticated command/completion rings, signed kernel bundles, capability checks, quotas, watchdogs, reset protocol, and CPU fallback design | Protocol and benchmark plumbing exist; production IOMMU containment and near-native in-VM CUDA remain unfinished |
| Rust compiler | Original lexer, parser, semantic analysis, constant evaluation, code generation, and source-to-object driver for a documented Rust subset | Experimental subset, not a complete `rustc` replacement |
| Git | Original repository, index, object, pack, ref, diff, checkout, commit, SSH transport, and signing functionality | Useful native client with documented unsupported Git features |
| AV1 | Original parser, transform, prediction, reconstruction, and encoder/decoder work | Active conformance work; do not claim complete AV1 interoperability yet |
| Opus | Original range coding, SILK/CELT primitives, packet handling, and encoder/decoder work | Active implementation and conformance work; not yet a production codec claim |
| Networking and security | Original HTTP, TLS, SSH, JSON, cryptography, signing, artifact admission, and rollback-policy components | Security-sensitive experimental code requiring independent audit |
| Data and training | ZIM, Zstandard, Wikipedia streaming, tokenizer training, compact transformer training, and GGUF export | Research pipeline; trainer is not yet a fully trained competitive LLM system |

Detailed microkernel architecture, trust boundaries, probe evidence, and
remaining milestones live in [docs/MICROKERNEL.md](docs/MICROKERNEL.md). Keeping
that detail outside this README makes the project overview readable while
retaining the evidence needed for engineering work.

## Design principles

- Own the critical implementation and keep the trusted dependency surface
  small.
- Put security first, measured performance second, and convenience third.
- Use narrow, auditable modules and explicit bounded storage at trust
  boundaries.
- Treat every parser, artifact, model, device, network peer, tool, and host
  input as untrusted.
- Enforce W^X, least privilege, capability-mediated access, generational
  identities, authenticated artifacts, and fail-closed state transitions.
- Measure claims on both Windows and Linux and distinguish controlled probes
  from production guarantees.
- State unsupported behavior and incomplete conformance explicitly.

All repository libraries and applications use `#![no_std]` and repository-owned
allocation and platform layers rather than importing Rust's `std` or global
`alloc` crates. Cargo's external test harness still supplies a host runtime for
`#[test]`; this does not mean model execution avoids all dynamic memory.

## Mandatory contribution rules

These rules apply equally to human and LLM/agent contributors. A change that
violates any rule must not be merged.

- Submit only original work that the contributor has the right to dedicate
  under CC0. Do not copy, translate, transcribe, or adapt third-party code or
  generated material with incompatible or uncertain provenance.
- All implementation code, build logic, tests, examples, benchmarks, and
  utilities must be Rust. Documentation, manifests, lockfiles, licenses, and
  required data formats may use their appropriate formats.
- Code may depend only on `core` and original crates in this workspace. Do not
  use `std`, `alloc`, crates.io packages, Git dependencies, or other sysroot
  crates.
- Keep code modular, cohesive, and easy to audit wherever that does not
  conflict with security or measured performance. Document non-obvious trust
  boundaries and invariants.
- Build and test every change on Windows and Linux. Windows supports only
  Rust's MinGW-based `x86_64-pc-windows-gnullvm` toolchain; do not require
  MSVC, Visual Studio, the Windows SDK, or a separate LLVM installation.
- Review unsafe code, FFI, integer and buffer arithmetic, parsers, filesystem
  and process operations, network input, concurrency, and failure paths. Add
  adversarial and regression tests appropriate to the change.
- Reject and rework security or measured performance regressions. Never waive
  a regression merely to make a feature pass.
- Record Windows and Linux test commands, security checks, and relevant
  benchmarks. Never claim a property or platform was verified when it was not.
- Compiler and codec work must clearly identify its supported subset and pass
  independent differential, conformance, malformed-input, and interoperability
  testing before broader compatibility is claimed.

## Trust model

The threat model assumes a hosted Windows or Linux OS and its VMM process may
be compromised. KVM and WHP are therefore development, compatibility, and
performance environments—not security boundaries against their hosts. A host
can inspect or alter guest RAM, CPU state, clocks, virtual devices, entropy,
keys, and GPU results regardless of in-guest authentication.

Bare-metal MRML currently trusts the CPU and microcode, chipset and memory
controller, RAM, selected hardware attestation root, firmware that establishes
the machine, and authenticated MRML loader/kernel/policy code. Devices, device
firmware, DMA, networks, models, tools, and external data are untrusted.
Compromised UEFI, SMM, platform security processors, hardware, or physical
access remain outside the enforceable boundary until a separately verified
dynamic root of trust and hardware reinitialization path exists.

MRML does not currently claim production verified boot, rollback resistance,
confidential computing, physical GPU isolation, side-channel resistance, or
formal noninterference. Those require hardware validation, TPM-backed monotonic
state, IOMMU/reset testing, reproducible release infrastructure, formal work,
and independent security review.

## Kernel state

The current x86_64 kernel has demonstrated:

- signed PE32+ loading through the repository-owned UEFI loader;
- bounded and normalized firmware handoff validation;
- private page tables with W^X and supervisor/user separation;
- GDT, 256-entry IDT, TSS, guarded privilege stacks, and double-fault IST;
- CPL3 entry, exact hardware trap-frame validation, and `IRETQ` restoration;
- local-APIC timer preemption across independent CR3 roots;
- capability spaces, bounded IPC, generational tasks, and lifecycle control;
- user-fault retirement with domain and capability revocation;
- resumption of a surviving CPL3 task after another task faults;
- multiprocessor startup, per-CPU state, IPIs, migration, and load balancing;
- signed execution probes under QEMU/UEFI, nested Linux KVM, and Windows WHP.

The kernel remains experimental. Storage, network, tool, inference, and GPU
service VMs; physical-hardware coverage; production rollback storage; complete
IOMMU containment; architecture ports; formal verification; and external audit
remain unfinished.

## Workspace layout

- `mrml-kernel`, `mrml-kernel-image`, `mrml-uefi`, `mrml-service-image`:
  microkernel, boot, and isolated-service execution.
- `mrml-kvm`, `mrml-kvm-run`, `mrml-whp`, `mrml-whp-run`: Linux and Windows
  virtualization backends and signed probes.
- `mrml-runtime`, `mrml-tensor`, `mrml-model`, `mrml-tokenizer`: inference and
  portable execution.
- `mrml-cli`, `mrml-machine`, `mrml-server`, `mrml-agent`, `mrml-tools`:
  interactive, automated, server, and agent interfaces.
- `mrml-rustc`, `mrml-rustc-driver`: original Rust language subset and object
  compiler.
- `mrml-git`, `mrml-ssh`: repository management, transport, and signing.
- `mrml-av1`, `mrml-opus`: native video and audio codec work.
- `mrml-http`, `mrml-tls`, `mrml-crypto`, `mrml-sign`: native protocol,
  cryptographic, and signed-artifact layers.
- `mrml-zim`, `mrml-zstd`, `mrml-wikipedia`, `mrml-trainer`: dataset ingestion
  and research training.
- `mrml-windows`, `mrml-linux`: minimal native platform interfaces.

## Build and run

MRML uses Rust 2024 and the pinned nightly in `rust-toolchain.toml`.

Windows requires the self-contained GNU/LLVM toolchain:

```powershell
winget install Rustlang.Rustup
rustup toolchain install nightly-x86_64-pc-windows-gnullvm --profile minimal `
  --component rust-src --component rustfmt --component clippy `
  --component rust-mingw --target nvptx64-nvidia-cuda `
  --target x86_64-unknown-uefi --target x86_64-unknown-none
rustup default nightly-x86_64-pc-windows-gnullvm
rustc -vV
```

Linux uses its native GNU host with the same Rust components and targets. CUDA
builds use Rust PTX plus the NVIDIA driver API and do not require the CUDA
Toolkit or `libcudart`.

```powershell
# CPU inference
cargo run --release -p mrml-cli --no-default-features -- `
  --model C:\path\to\model.gguf --prompt "Hello"

# CUDA inference
cargo run --release -p mrml-cli --features cuda -- `
  --model C:\path\to\model.gguf --ctx-size 8192 --prompt "Hello"

# Native Git client
cargo run --release -p mrml-git -- status

# Compile a supported Rust source subset to an object
cargo run --release -p mrml-rustc-driver --bin mrml-rustc -- `
  --emit coff --function answer input.rs -o answer.obj
```

Run a binary with `--help` for its supported interface. Do not infer complete
compatibility from familiar command names.

## Verification

The minimum repository gates are:

```powershell
cargo test --workspace --release --no-default-features
cargo clippy --workspace --all-targets --release --no-default-features -- `
  -D warnings
```

Run feature- and hardware-specific suites for CUDA, KVM, WHP, UEFI, codecs,
network protocols, and the compiler whenever those areas change. Performance
results must identify the exact artifact, backend, hardware, configuration,
warm-up policy, measurement boundary, and variation across runs. Codec and
compiler compatibility require independent conformance/differential evidence,
not only self-round-trip tests.

## Context for ChatGPT and Codex sessions

This section exists specifically to prevent a new AI session from overstating
the repository or undoing its invariants.

1. Read this README and [docs/MICROKERNEL.md](docs/MICROKERNEL.md) before
   changing kernel, VMM, signing, GPU, or trust-boundary code.
2. Treat the mandatory contribution rules above as hard acceptance gates.
3. Inspect the current worktree before editing. Preserve unrelated staged,
   unstaged, and untracked user work; stage only the intended change.
4. Do not describe a controlled probe as production functionality. Report the
   exact backend and observed evidence.
5. Never describe the Rust compiler as complete `rustc`, AV1/Opus as fully
   conforming, hosted virtualization as secure against its host, or the kernel
   as production ready until independent evidence supports those claims.
6. Update this overview when project scope changes. Put detailed experimental
   traces, threat analysis, and milestone evidence in the relevant document
   under `docs/`, not in an ever-growing README status narrative.
7. Commit and push only after proportional Windows and Linux testing and an
   explicit review of security and performance impact.

## License

MRML is dedicated to the public domain under [CC0 1.0 Universal](LICENSE)
(`CC0-1.0`). Anyone may use, study, modify, share, and redistribute it for any
purpose, including commercially. Where public-domain dedication is not legally
possible, CC0 provides its broad license and waiver fallback.
