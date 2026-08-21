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
caller-owned early-boot storage. The loader now emits that same canonical form
into bounded static storage and passes its address and exact length to a
separately built kernel PE entry point.

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

This remains a bring-up loader rather than a production boot chain. It loads,
authenticates, relocates, and transfers to a separate kernel image under fresh
loader-owned page tables. Persistent rollback state, measurement into a
hardware trust anchor, recovery policy, and kernel-owned interrupt tables
remain required.

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

The post-firmware path now enters the independent `mrml-kernel-pe.efi` image.
Its entry parses the canonical handoff again and revalidates nonzero entropy,
the ACPI pointer, the sorted memory map, and complete framebuffer MMIO
containment before drawing. On Windows QEMU 11.1, a freshly generated one-use
key signed the exact standalone image, the loader pinned that key digest, and a
GOP capture showed its gold `RGB(255,200,87)` 96x12 marker over
`RGB(11,59,90)`. The preceding loader stage was `RGB(32,64,128)`, so successful
PE materialization and entry transfer are independently observable.

The loader now resolves its own boot device through the raw UEFI Loaded Image
and Simple File System protocols and requires
`\EFI\MRML\KERNEL.SIGNED`. It opens the file read-only, accepts bounded file
metadata, allocates loader pages, handles short reads, and rejects empty,
oversized, truncated, or concurrently grown inputs. QEMU 11.1 with its bundled
EDK2 used only as an interoperability oracle successfully read the required
file and still reached the post-ExitBootServices marker: pixel `(0,0)` was
`RGB(242,242,242)` and pixel `(0,8)` was `RGB(22,97,58)`. This checkpoint proves
filesystem interoperability; the test file was a placeholder and was not
admitted as a signed kernel.

The loader now fails closed unless its own build embeds an exact SHA3-512
kernel public-key digest and nonzero minimum version through
`MRML_KERNEL_ROOT_DIGEST_HEX` and `MRML_KERNEL_MIN_VERSION`. These values become
part of the loader binary that conventional UEFI Secure Boot will eventually
authenticate; no private key enters the build. After reading, the loader
authenticates the complete typed Lamport container before interpreting any PE
offset. A disposable QEMU key accepted its matching version-1 fixture and
reached the kernel marker. A second independently valid version-1 fixture from
an unpinned key stopped at the blue verification-failure stage
`RGB(0,0,128)`. The one-time private files were consumed by the signer.

The admitted PE is copied into zeroed UEFI LoaderCode pages and only bounded
DIR64 relocations are applied. Each nonzero relocation must originally point
inside the preferred image and is converted to the corresponding checked RVA
at the actual 4 KiB-aligned load address. The entry must be inside a validated
executable, non-writable section, and the loader-owned CR3 enforces those final
section permissions in hardware.

The x86-64 transition now allocates a new zeroed four-level page-table tree and
a dedicated 64 KiB kernel stack before leaving firmware. It identity-maps only
the authenticated PE regions with their final read-only, writable/NX, or
read-only/executable permissions; the stack and GOP aperture writable/NX; the
canonical handoff read-only; and one page-aligned read-only/executable assembly
trampoline. The trampoline enables EFER.NXE and CR0.WP, replaces CR3, changes
to the dedicated stack, and jumps without returning. No writable alias of an
executable image page is retained. QEMU 11.1 reached the independent kernel
marker with `CR0=0x80010033`, `EFER=0xd00`, and the loader-created CR3 root.
The standalone kernel immediately installs an image-owned GDT preserving the
transition selectors and a complete image-owned IDT whose gates all target a
non-returning interrupt-disabled halt stub. Interrupts remain disabled; this is
deterministic fail-stop handling, not yet recoverable fault dispatch. A signed
`fault-probe` build executed `UD2` under QEMU and stopped with `HLT=1` at the
kernel handler while retaining the loader-created CR3 and kernel GDT/IDT bases.

Before materialization, the loader can now use the packed, raw
`EFI_TCG2_PROTOCOL` ABI to hash the already authenticated kernel PE into PCR 11
and append an `EV_IPL` event named `MRML authenticated kernel PE`. The canonical
handoff sets its measured-boot flag only after `HashLogExtendEvent` succeeds.
Set `MRML_REQUIRE_TPM=1` when building the loader to make a missing TCG2
protocol or failed extend boot-fatal; QEMU without a virtual TPM stopped at the
blue `RGB(0,0,128)` stage under that policy, while the optional development
policy still reached the kernel marker. PCR extension supplies attestation and
an event log, but is not a monotonic counter. TPM NV provisioning,
authorization, atomic version advancement, and recovery remain separate work
and no rollback-resistance claim is made yet.

An allocation-free TPM 2.0 NV backend emits bounded `NV_ReadPublic`, `NV_Read`,
and `NV_Increment` commands through TCG2 `SubmitCommand`. It rejects handles
outside the platform NV range, non-counter index types, sizes other than eight
bytes, mismatched public handles, malformed names, rollback below the stored
counter, oversized per-boot advancement, malformed response lengths, and
nonempty password-session responses. Set `MRML_TPM_NV_INDEX_HEX` to the exact
eight-hex-digit handle of a separately provisioned empty-auth counter. The
loader advances it only after signature verification, PE materialization,
page-table construction, and ACPI validation. `MRML_REQUIRE_ROLLBACK=1` makes a
missing configuration boot-fatal. Empty authorization permits unauthorized
increments (denial of service) but never decrement or rollback; a production
deployment should provision a policy-authorized index once policy sessions are
implemented.

Successful NV validation and advancement is now represented by a dedicated bit
in the canonical UEFI-to-kernel handoff rather than inferred from the image
version. The decoder rejects unknown bits, `BootEvidence` retains the result,
and production `BootPolicy` independently requires Secure Boot, measured boot,
and rollback protection. A version meeting the compiled floor is therefore not
enough when monotonic enforcement was skipped. A freshly signed version-5
kernel using the expanded evidence contract still reached the QEMU GOP marker.

The standalone image also has a `production-policy` feature. Such builds embed
`MRML_KERNEL_MIN_VERSION` independently of the loader and validate the decoded
handoff with `BootPolicy::production` before touching the framebuffer or
starting subsystems. Missing or malformed compile-time policy fails closed. A
properly hash-signed version-6 production image transferred under loader-owned
CR3/GDT/IDT but refused the deliberately unmeasured, non-Secure-Boot QEMU
handoff, leaving the loader's `RGB(20,33,53)` stage unchanged. All terminal
kernel paths now use interrupt-disabled `HLT` rather than consuming a CPU in a
spin loop.

The loader also reads the standard global `SecureBoot` and `SetupMode`
variables through its original raw runtime-services ABI. It reports secure boot
only when `SecureBoot` is exactly one and `SetupMode` is exactly zero; absent,
malformed, or contradictory values never become positive evidence. Set
`MRML_REQUIRE_SECURE_BOOT=1` to reject any other state. This verifies firmware
enforcement state but does not itself create an Authenticode signature. The
platform-compatible RSA/CMS signing path remains an optional build-tool task;
the inner hash-based MRML signature remains mandatory regardless of firmware
mode.

`mrml-sign authenticode-digest IMAGE.efi` now computes the SHA-256 PE image
digest used by conventional Authenticode. Its parser accepts only x86-64
PE32+, bounds the optional and section tables, sorts raw sections, rejects
overlap and overflow, requires a terminal eight-byte-aligned certificate table,
and excludes exactly the checksum field, security-directory entry, and
certificate bytes. This is hashing groundwork, not a signature generator:
RSA/PKCS#1 and CMS `SignedData` emission plus verification against an
independent platform tool remain necessary before claiming firmware-compatible
signing.

`mrml-sign attach-authenticode IMAGE.efi SIGNATURE.p7b OUTPUT.efi` now performs
the remaining PE container work for an externally generated CMS object. It
requires an unsigned eight-byte-aligned canonical image, appends exactly one
revision-2 `WIN_CERTIFICATE` of type PKCS signed data, zero-pads only the
certificate entry, updates the security directory, calculates the final PE
checksum while excluding its field, refuses overwrite or double attachment,
and recomputes the Authenticode digest before returning. On the real standalone
kernel PE, the digest remained
`5269d33d3834f99bcb5ea31acf908749b7eff2456b91015ddca0092b85eae59b`
before and after attachment on Windows. Certificate bytes are treated as an
opaque CMS input and are not claimed valid until the CMS generator and an
independent verifier accept them.

The original fixed-storage RSA implementation now also emits deterministic
PKCS#1 v1.5 SHA-256 signatures using the same constant-work exponent traversal
as RSA-PSS. It constructs the complete DigestInfo and requires at least eight
padding octets, rejects inconsistent modulus/signature sizes, and has a
disposable independently generated RSA-1024 private-key round-trip vector.
This primitive is required by Authenticode but is not itself CMS or a signed PE.

### OS executable format

PE32+ is the sole executable binary format for MRML on x86-64. The kernel, VM
boot images, service images, and tool executables must be x86-64 PE32+ payloads
inside their typed signed-artifact containers. Launch policies and CUDA bundles
are data and must never be admitted as executable images. Future ARM64 and
RISC-V ports retain PE32+ with their architecture-specific machine identifier,
keeping one bounded loader and signing representation across UEFI, Hyper-V,
KVM, and bare metal.

The original allocation-free parser accepts at most 32 sections and requires
an executable COFF image, the PE32+ optional header, NX compatibility,
power-of-two file and section alignment, exact in-file raw ranges, bounded
image ranges, nonoverlapping virtual and raw sections, and an entry point
contained in an executable section. Any writable-and-executable section is
rejected. The release signer validates these rules before consuming a one-time
key, and runtime admission repeats validation only after the outer artifact
signature succeeds. The loader core can now materialize a validated image into
an exact-size caller-owned buffer: it clears the entire image first, copies the
bounded headers and section bytes to their RVAs, leaves virtual tails and gaps
zeroed, and returns the checked preferred-base entry address. A separate load
plan describes read-only NX headers and the final W^X permission of every
section. Platform code must populate pages while NX is active and install the
final permissions only after copying completes. Mapped image size is capped by
artifact type: 64 MiB for the microkernel, 128 MiB for a service, and 512 MiB
for a VM boot image. Both the signer and verified runtime admission enforce the
same quota before allocation. Optional PE base relocation tables are fully
validated before loading, capped at 65,536 entries, and accept only padding and
x86-64 `DIR64` relocations; malformed blocks, unsupported relocation types,
out-of-image targets, arithmetic overflow, and rebasing a fixed image fail
closed. PE imports are rejected because they are not part of the standalone
MRML ABI. Page-table installation and transfer to a separately loaded image
remain pending.

The early frame allocator now reserves aligned physically contiguous runs
without crossing normalized firmware regions. PE admission can consume that
allocator to produce a fixed-capacity physical load plan containing one run for
the read-only NX headers and one run per section. Alignment padding and partial
allocations are never recycled during boot, preventing stale-frame aliasing;
failure is therefore fatal rather than rolled back. Physical table storage,
installation, and the final control transfer remain pending.

The x86-64 address-space layer now converts that physical plan into final user
or kernel mappings with permissions derived from the validated PE sections.
Headers remain read-only/NX, executable mappings are never writable, virtual
and physical alias checks are repeated, and all mapping identifiers are
generation-tagged. Installation is transactional: a conflict in any later
section removes every mapping installed earlier in the same attempt, leaving
no half-mapped executable. Hardware page-table construction and CR3 transfer
are separated behind this policy layer.

An original four-level x86-64 page-table builder now converts approved
mappings into hardware-format 4 KiB leaves through a minimal physical-table
storage trait. It allocates only zeroed table frames, validates every existing
intermediate entry, propagates user accessibility through all parent levels,
rejects huge-page collisions and duplicate leaves, and reuses the same NX/W^X
leaf constructor tested by the address-space policy. The builder returns its
root physical frame but does not write CR3 itself. A real UEFI physical-memory
backend, boot-time table population from the complete PE plan, TLB transition,
and final control transfer remain pending.

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
mrml-sign sign-bundle kernel 1 mrml-kernel.exe release.private mrml-kernel.signed
mrml-sign key-digest release.public
```

`sign-bundle` emits the sole boot-artifact representation: a fixed 112-byte
header, the 64 KiB public key, the 32 KiB signature, and the exact payload.
The header binds kind, nonzero version, payload length, and SHA3-512 digest.
The allocation-free decoder rejects unknown kinds, nonzero reserved bytes,
integer overflow, empty payloads, truncation, trailing bytes, digest mismatch,
wrong-type admission, rollback, unpinned keys, and invalid signatures. The
signer self-verifies before writing and removes the one-time private key only
after the completed bundle has been written successfully.

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

The implemented backend-neutral VM boundary uses a fixed 64-byte `MRMLHC01`
hypercall descriptor and a caller-owned copy buffer. Reserved bytes and
operation-specific unused fields must be zero, sequences are exact and advance
only after successful authorization, and tool, GPU, and shutdown operations
require distinct capability objects. Guest mappings are fixed-capacity,
page-aligned, non-overlapping in both guest and host address spaces, and never
writable and executable. A request buffer must fit wholly within one readable
mapping. Backend-owned pointers never enter the policy core. The common
`VmBackend` contract currently covers vCPU exits, bounded guest copies, and
interrupt injection; concrete KVM and Hyper-V adapters and shared-memory data
queues remain pending, so this is not yet a runnable hosted VM.

Guest mappings receive opaque monotonic identifiers and can be revoked or have
their permissions changed without exposing array slots. Revocation compacts the
bounded table but never reissues an old identifier. Permission changes retain
the W^X invariant, and translations fail immediately after revocation. This is
the common policy operation that concrete adapters must pair with their native
second-level page-table invalidation before allowing a vCPU to resume.

VM instances are tracked in a fixed-capacity lifecycle table with generational
identities. Only created images may become loaded, only loaded or cleanly
stopped VMs may run, and destroyed identifiers cannot control a replacement in
the same slot. Every run receives an explicit exit budget. Exceeding it moves
the VM to a failed state, bounding denial-of-service through pathological exit
storms and requiring an explicit destroy/recreate recovery path.

The common single-exit run path now translates one backend exit, charges the
VM's budget, and dispatches it through common policy. Port I/O is denied by
default and may be enabled only by an exact port, transfer-width, and direction
rule in a table capped at 32 entries. Unsupported exits and guest-memory faults
fail the VM instead of being reflected or silently resumed. A backend execution
error also fails the instance because its register and device state cannot be
assumed resumable. Concrete adapters remain responsible for translating native
exit structures into this deliberately small representation.

The first KVM adapter layer is implemented in the separate core-only
`mrml-kvm` crate. It decodes the kernel-owned `kvm_run` mapping as bytes instead
of creating references to its C union. I/O exits require one scalar transfer,
validated direction and width, and an in-bounds data offset. MMIO accepts only
1, 2, 4, or 8 byte accesses. MRML hypercalls use a dedicated number and require
all unused KVM arguments to be zero. Memory-slot encodings are page-aligned,
overflow checked, optionally read-only, and capped at 32 slots. Native file
descriptor, ioctl, mmap, register setup, and interrupt-chip integration remain
pending; this layer does not yet launch a KVM guest.

Interrupt injection is a separate capability-authorized path. The caller must
hold `SIGNAL` authority for the VM's dedicated interrupt object and the vector
must be present in a 256-bit allowlist. Architectural exception vectors 0--31
and vector 255 cannot be enabled. Injection is accepted only while the VM is
running; a backend injection failure moves it to the failed state because
delivery may have partially changed backend state.

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

| Environment | Capability authorization | Scheduler selection | VM exit accounting |
| --- | ---: | ---: | ---: |
| Windows `x86_64-pc-windows-gnullvm` | 548,200 ns total (548 ps/op) | 1,859,500 ns total (1,859 ps/op) | 729,600 ns total (729 ps/op) |
| Arch Linux under WSL2 | 552,195 ns total (552 ps/op) | 1,824,970 ns total (1,824 ps/op) | 876,643 ns total (876 ps/op) |

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
