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

The resource/session validator and authenticated queue wire format are
implemented. Each fixed-size command is bound to a nonzero session and strict
sequence and carries an HMAC tag under a per-session key; tampering, replay,
cross-session substitution, noncanonical encoding, and zero keys are rejected
before resource state changes. HMAC-SHA-256 is conservatively treated as a
128-bit generic quantum-search target and is used only for ephemeral queue
authentication, not
artifact signing. Dispatches also have one fixed-size canonical encoding: they
contain an embedded kernel ID, bounded launch geometry, and up to 16 opaque
generational buffer ranges with explicit access modes. Unused slots must be
zero and there is no representation for PTX, pointers, driver calls, or
variable argument blobs. In-flight work uses a fixed-capacity watchdog table
with unique request IDs, kernel-minted generational dispatch IDs, explicit
cancellation, and deadline expiry. Expiry invalidates an ID before requesting
host recovery, so a late completion cannot affect a reused slot. The Hyper-V
transport, host CUDA executor, kernel-specific argument schemas, copy staging,
the platform-specific device-reset callback, and end-to-end inference benchmarks
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

The normalized UEFI-to-kernel handoff now has a fixed, versioned, little-endian
encoding and a bounded streaming parser. It carries boot flags, release,
entropy, kernel measurement, ACPI root, a required GOP framebuffer, and at most 128 normalized memory
regions. The parser does not allocate or dereference firmware pointers and
rejects unknown flags or region kinds, nonzero reserved bytes, noncanonical
lengths, unaligned or overflowing regions, and overlap before emitting data to
caller-owned early-boot storage. Completing and transferring this handoff to a
separate verified kernel image remains pending.

`mrml-loader.efi` is now a repository-owned PE/COFF UEFI application with raw
ABI definitions rather than a firmware crate dependency. It locates required
GOP and RNG protocols, rejects unsupported modes and zero entropy, obtains a
bounded memory map, retries the map-key-sensitive `ExitBootServices` transition,
and retains only copied framebuffer values afterward. The framebuffer changes
from blue before the transition to green after successful exit, providing a
visible bring-up signal without serial hardware. Build it with:

```text
cargo build --release -p mrml-uefi --bin mrml-loader --features uefi-image --target x86_64-unknown-uefi
```

This is the first executable boot stage, not yet a complete kernel loader: it
does not load and verify a separate kernel image, normalize the firmware map
into the kernel handoff, measure the loaded image, or transfer to a
kernel entry point. Those security-critical steps remain required before MRML
can be described as booting its microkernel.

ACPI RSDP discovery is now implemented before firmware exit. The loader prefers
the ACPI 2.0 configuration-table GUID, accepts ACPI 1.0 only as a fallback,
bounds the extended RSDP length, and validates both the legacy and extended
checksums. It does not assume configuration-table pointers retain legacy BIOS
alignment.

QEMU 11.1.0 interoperability was exercised on Windows with its bundled EDK2
firmware strictly as an external test oracle. With `virtio-rng-pci` and the EFI
image at `EFI/BOOT/BOOTX64.EFI`, a 1280x800 GOP capture after boot contained the
uniform post-exit marker `RGB(22,97,58)`. This demonstrates that QEMU loaded the
PE image and that GOP discovery, RNG, ACPI validation, memory-map retrieval,
and `ExitBootServices` completed. EDK2 is not linked, copied into repository
artifacts, or part of the planned MRML VMM firmware.

The loader now normalizes the variable-stride UEFI memory map after firmware
exit into at most 128 sorted, nonoverlapping MRML regions. Only conventional
memory is immediately free; loader and boot-service pages remain conservatively
reserved. Runtime firmware, ACPI, and MMIO receive distinct kinds, adjacent
equal regions are merged, and all page arithmetic is checked. Because GOP BARs
need not appear in the UEFI map, the firmware-authorized framebuffer is overlaid
as MMIO: it is inserted into an address gap or splits one containing region,
while cross-region conflicts fail closed. QEMU still reached the green marker
after this normalization was enabled.

The post-firmware path now enters code linked from `mrml-kernel`. A typed early
context revalidates nonzero entropy, the ACPI pointer, the sorted memory map,
and complete framebuffer MMIO containment before drawing. QEMU captured the
kernel marker as a white 64x8 rectangle at pixel `(0,0)` over the green
background; pixel `(0,8)` and the final framebuffer pixel remained
`RGB(22,97,58)`. Thus the marker cannot be confused with the loader's earlier
solid-color stages. This is a statically linked early kernel entry, not yet a
separately loaded and signed kernel image.

Host timing/output code moved to the `mrml-kernel-bench` crate. The UEFI target
dependency graph is now only `mrml-uefi -> mrml-kernel -> mrml-crypto`.
`mrml-crypto` exposes a fixed-storage boot feature path while its runtime-backed
TLS, RSA, and ML-KEM helpers remain enabled by default for applications.

GOP is the primary early console because contemporary physical machines cannot
be assumed to expose a usable serial port. Only the standard 32-bit RGB-reserved
and BGR-reserved pixel layouts are accepted. Geometry, stride, byte length,
address overflow, and containment inside a normalized MMIO region are checked
before the framebuffer is admitted. The safe renderer clips nothing silently:
out-of-range pixels and rectangles are rejected and row padding is never used
as visible storage. Serial remains an optional developer diagnostic path.

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

### Reproducible signing and bootstrap

`mrml-sign` is the original, dependency-free release signer. It runs after
compilation, never from `build.rs`, so Cargo and repository contents do not need
access to private keys:

```text
cargo build --release -p mrml-sign
mrml-sign keygen release.private release.public
mrml-sign sign kernel 1 mrml-kernel.efi release.private mrml-kernel.sig
mrml-sign key-digest release.public
```

After generating distinct artifact keys and a next-release root, create and
sign the canonical 408-byte release manifest:

```text
mrml-sign manifest 1 release.manifest NEXT_ROOT KERNEL VM SERVICE CUDA POLICY
mrml-sign sign policy 1 release.manifest genesis.private release.manifest.sig
```

Each digest argument is the 128-character SHA3-512 public-key digest printed by
`key-digest`. The kernel now decodes this exact representation, verifies it
against the current root, exposes typed artifact trust roots, advances the next
root, and raises the minimum acceptable release. Replaying the old manifest is
rejected after state advancement.

The signer refuses to overwrite keys or signatures, self-verifies before
writing a signature, and then consumes and removes the private-key file. Any
external backup of that one-time key must also be destroyed and must never sign
a second statement.

The bootstrap loader contains only a genesis public-key SHA3-512 digest and a
minimum release version, never a private key. That genesis key signs the first
canonical launch-policy artifact. The policy contains distinct one-time public
key digests for every kernel, VM, service, CUDA bundle, and the next policy
root. After successful verification, the next root digest and release counter
are sealed into TPM-backed monotonic state. Each release therefore authenticates
its successor without rebuilding trust from an unsigned file. Recovery uses a
separate offline root and must not lower the monotonic version. Until TPM/NVRAM
persistence and the canonical policy parser are implemented, this describes
the required bootstrap flow rather than an end-to-end completed secure boot.

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

Verified release state has a canonical 88-byte encoding containing the current
root digest and minimum accepted release. The kernel advances it only through
an authenticated monotonic-store compare-and-store operation, so concurrent or
stale updates fail instead of overwriting newer trust state. A conforming
platform backend must use TPM NV or an equivalent authenticated, power-loss
atomic facility; a normal host file is not a conforming backend. The encoding,
rotation, corruption, conflict, and replay paths have unit tests. Actual TPM NV
and UEFI variable backends remain pending and no rollback-resistance claim is
made until one is implemented and tested on physical hardware.

Boot admission also requires a coherent signed artifact set: exactly one
kernel, VM image, service image, CUDA kernel bundle, and launch policy from the
same release as the firmware evidence. The verified kernel content digest must
equal the measured-boot kernel digest. Individually valid artifacts from
different releases therefore cannot be spliced into an accepted boot chain.

Each milestone requires Windows gnullvm and Arch Linux builds/tests, parser and
capability fuzz-style adversarial vectors, unsafe/FFI review, boot tests under
both hypervisors where applicable, and before/after latency and throughput
measurements. Production-security claims additionally require an external
audit, reproducible signed builds, verified boot, rollback protection, IOMMU
tests, side-channel analysis, and machine-checked capability/noninterference
properties. Until those exist, the kernel remains experimental.

Run the current release microbenchmarks with:

```text
cargo test --release -p mrml-kernel-bench -- --ignored --nocapture
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
