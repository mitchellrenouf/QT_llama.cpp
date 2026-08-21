# MRML microkernel design

This document records design intent, not a claim of production readiness. The
kernel is new, original Rust intended for CC0 dedication. Research informs
requirements and trade-offs; no third-party implementation is copied or
adapted.

## Security boundary

The privileged kernel will contain only scheduling, address-space and interrupt
mechanisms, capability enforcement, IPC, and the minimum early console/timer
code needed to recover. Filesystems, networking, storage, GPU, CUDA mediation,
model loading, and tools remain isolated services. Each tool runs in a separate
VM by default. Messages carry attenuated, generational capabilities; names,
paths, process IDs, and VM IDs never confer authority.

Host directories are denied by default. A launch manifest explicitly lists
canonical directories as read-only or read-write. The Hyper-V or KVM host VMM
opens each directory without following a final symlink and lends an opaque
handle to a storage service. Guests never receive a host pathname API. Writable
grants are separate capabilities and cannot be derived from read-only grants.

Device assignment is also denied by default. CUDA passthrough is an optional
device-VM capability using the host's IOMMU and hypervisor assignment facility.
The device VM owns the GPU while assigned; CUDA command parsing does not enter
the kernel. Reset, DMA isolation, firmware trust, and peer-to-peer DMA must be
validated for each GPU/platform combination. Systems lacking trustworthy IOMMU
isolation use the CPU backend.

Passthrough grants are minted from a host-enumerated IOMMU topology rather than
from caller assertions. Every function in the target group must be assigned to
the same VM and have a verified reset path. System launch policy rejects reuse
of either a device or its IOMMU group across VMs. Hyper-V and KVM backends must
populate this topology from their native isolation APIs and fail closed when
the topology is incomplete.

### MRML virtual GPU

For Hyper-V systems where the host must retain the GPU, MRML uses an original
paravirtual interface rather than forwarding CUDA or NVIDIA driver APIs. An
isolated host GPU service owns the CUDA context. A guest receives opaque,
generational buffer IDs under a fixed byte quota and may dispatch only the 28
MRML kernels embedded in the same measured build. Requests contain checked
buffer ranges and bounded grid, block, and shared-memory values. They contain no
host pointers, arbitrary PTX, CUDA ioctls, firmware operations, or device MMIO.

The resource/session validator is implemented. The authenticated queue wire
format, Hyper-V transport, host CUDA executor, kernel-specific argument schemas,
copy staging, cancellation, watchdog reset, and end-to-end inference benchmarks
remain pending. Until those pieces exist and are audited, this is not a working
shared-CUDA Hyper-V device. It is intentionally MRML-specific instead of a
general `virtio-cuda` compatibility layer.

The CUDA service must verify the exact embedded PTX bundle as a
`CudaKernelBundle` artifact before module loading. Kernel IDs are enabled only
after this verification token exists; a digest mismatch or missing signature
leaves the virtual GPU unavailable and inference falls back to CPU.

## Portability and boot

The first target is x86_64 UEFI. Firmware-specific boot code hands a normalized
memory map, framebuffer/serial description, ACPI root, entropy, and signed boot
measurement to an architecture-neutral kernel entry point, then exits boot
services. Architecture modules contain page-table, interrupt, timer, context
switch, and virtualization primitives. AArch64 and RISC-V64 are represented in
public platform types now so x86 assumptions cannot leak into policy APIs.

Production boot policy requires firmware secure-boot evidence, a measured image
digest, and a monotonically enforced minimum image version. The UEFI loader must
source its 256-bit boot seed from the firmware RNG protocol or a documented
architecture entropy source; the portable kernel rejects absent evidence but
cannot independently prove firmware entropy quality. Development policy may
relax firmware requirements but must not be accepted by production launchers.

The native artifact chain uses an original SHA3-512 Lamport signature verifier.
Production policy requires separately typed signatures for the microkernel, VM
images, service images, launch policy, and embedded CUDA-kernel bundle. Each
trust root pins the SHA3-512 digest of its public key and minimum artifact
version. Signed statements bind artifact type, version, byte length, and content
digest, preventing a CUDA signature from authorizing a VM image or an older
image from satisfying rollback policy.

Lamport keys are one-time keys: reuse for two different artifacts can reveal
enough private material to permit forgery. MRML therefore requires an
independent pinned key for every signed artifact/version. The current primitive
is hash-based and designed to resist quantum preimage attacks, but it is not a
NIST-standard ML-DSA implementation and has not received independent review.
The large 64 KiB public keys and 32 KiB signatures are accepted for the small
verified-boot set, not general network signing. Production-security claims
remain prohibited until the signing workflow, key lifecycle, loaders, recovery
path, and verifier receive external cryptographic and platform audits.

Hyper-V and KVM are host backends, not code in the guest kernel. Both implement
the same VMM contract for vCPU lifecycle, guest memory, interrupts, virtio-like
queues, directory-handle grants, measured images, and optional IOMMU device
assignment. Bare metal uses native architecture modules and the CPU inference
backend until a separately isolated GPU service is available.

## Performance policy

Fast IPC alone is insufficient. MRML will measure call frequency, duplicate
state, cache misses, VM exits, TLB invalidation, context switches, model-token
latency, and throughput. Shared read-only rings carry bulk tensors and tool
payloads while capability-checked IPC remains the control plane. No shared
mutable kernel metadata is exposed to untrusted services. Isolation relaxation
requires a shared trust domain, benchmark evidence, and a documented threat
model.

This follows the useful lessons reported for Huawei's production HongMeng
microkernel: keep a minimal core and least-privileged services, reduce IPC
frequency and duplicate bookkeeping, separate control and data planes, and keep
paging mechanisms in the kernel while policy remains outside. MRML does not
adopt HongMeng's Linux ABI goal or its security/performance compromises without
independent justification. The paper reports that those compromises can add
attack surface, so they are not defaults here.

Primary references: [HongMeng production microkernel, OSDI
2024](https://www.usenix.org/conference/osdi24/presentation/chen-haibo) and
[functional verification of OpenHarmony LiteOS-M, FM
2026](https://doi.org/10.1007/978-3-032-26220-2_32). These are research inputs
only and contribute no source code to MRML.

## Verification gates

Each milestone requires Windows gnullvm and Arch Linux builds/tests, parser and
capability fuzz-style adversarial vectors, unsafe/FFI review, boot tests under
both hypervisors where applicable, and before/after latency and throughput
measurements. Production-security claims additionally require an external
audit, reproducible signed builds, verified boot, rollback protection, IOMMU
tests, side-channel analysis, and machine-checked capability/noninterference
properties. Until those exist, the kernel remains experimental.

Run the current release microbenchmarks with:

```text
cargo test --release -p mrml-kernel --test performance -- --ignored --nocapture
```

They report nanoseconds per capability authorization and scheduler selection.
The generous automated ceiling detects catastrophic regressions; comparative
optimization decisions require repeated samples on pinned hardware and a
recorded baseline.

Baseline recorded 2026-08-20 on the current development host, one million
iterations per sample:

| Environment | Capability authorization | Scheduler selection |
| --- | ---: | ---: |
| Windows `x86_64-pc-windows-gnullvm` | 559,400 ns total (559 ps/op) | 1,843,600 ns total (1,843 ps/op) |
| Arch Linux under WSL2 | 547,204 ns total (547 ps/op) | 1,841,754 ns total (1,841 ps/op) |

These measure optimized in-process policy operations, not VM exits, page-table
changes, context switches, or end-to-end IPC. Sub-nanosecond averages can result
from pipelining across independent iterations and must not be presented as
single-operation latency.

## Milestones

1. Capability model, explicit directory grants, IPC wire format, and property
   tests. The fixed-size policy model and stable, versioned cross-VM control
   encoding are implemented. Shared-memory transport and host-handle binding
   remain pending.
2. Original x86_64 UEFI loader, physical allocator, page tables, exceptions,
   timer, scheduler, and serial diagnostics under emulation.
3. User address spaces and synchronous IPC; isolated storage, network, tool,
   and inference services.
4. KVM VMM, then Hyper-V VMM, with identical measured launch manifests.
5. Shared-memory inference data plane and benchmark harness.
6. IOMMU-contained CUDA device VM, reset/recovery tests, and CPU fallback.
7. Bare-metal hardware matrix, formal specifications/proofs, external audit,
   then AArch64 and RISC-V64 ports.
