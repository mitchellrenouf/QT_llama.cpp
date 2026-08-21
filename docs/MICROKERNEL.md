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

The resource/session validator, authenticated queue wire format, and bounded
kernel-owned FIFO are implemented. The FIFO accepts only exact-size messages,
never overwrites unread work, erases consumed slots, and safely wraps its
indices. Copying into owned storage closes the mutation-after-admission window;
the later shared-page transport must preserve that property. Monotonic
producer/consumer ownership state now models reserve-before-write,
commit-after-write, acquire-before-read, and release-after-read. It rejects
forged counters, producer lapping, double reservation, out-of-order release,
premature slot reuse, and counter exhaustion; positions never wrap within a
session. A 128-byte, 64-byte-aligned shared index structure now places producer
and consumer counters on separate cache lines and uses release compare-exchange
for publication/consumption with acquire loads at the opposite endpoint. The
remaining platform mapping layer must prove natural alignment and coherent
shared-memory semantics before exposing it. The common layout validator sizes
each ring from the 128-byte index block plus its bounded slots, rounds only with
checked arithmetic, requires page-aligned nonzero physical bases, and rejects
overlapping command/completion ranges. Each fixed-size command is bound to a nonzero session and strict
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
host recovery, so a late completion cannot affect a reused slot. The
completion direction has its own HMAC domain, fixed status encoding, session
and sequence state, and binds the original request ID to the exact generational
dispatch ID. It rejects tampering, replay, cross-direction substitution,
reserved bytes, unknown status, and stale handles before watchdog completion.
The executor-facing batch policy holds at most 32 ordered dispatches in fixed
storage and validates every buffer generation and range before admission. It
rejects empty/oversized batches, duplicate or zero request IDs, stale buffers,
and capacity overflow. This is the coarse submission unit intended for one
doorbell and one CUDA graph/stream sequence. Batch descriptors use a separate
generational `ControlBufferId`, never device `BufferId`. Admission caps shared
control bytes at 64 KiB and records their length and SHA3-512 digest; every use
rehashes before parsing, release erases the privileged digest, and stale IDs do
not revalidate. The authenticated resource queue has a canonical `SubmitBatch`
variant carrying only this typed ID. Canonical batch serialization is
implemented as
an exact-length `MRGB` header followed by one to 32 existing canonical dispatch
records. It contains no pointers or offsets, preserves order and independent
request IDs, rejects trailing/truncated/reserved encodings, and revalidates all
session resources while decoding. The service-side executor remains pending.
Dispatch and batch wire version 2 now add a fixed tail of at most 16 typed
32-bit scalar slots. Each slot identifies `u32`, `i32`, or raw IEEE-754 f32
bits, has zero-only reserved bytes, and unused slots must be entirely zero.
Oversized scalar lists, unknown kinds, nonzero padding, old-version records,
and alternate lengths fail closed. The original constructor still produces an
empty-scalar dispatch, so existing pointer-free operations remain explicit.
Before executor handoff, the dispatch watchdog now admits the entire batch
transactionally. Each request receives a generational `DispatchId`; capacity,
deadline, or duplicate failures cancel every ID minted by that attempt. A
service rejection can likewise cancel all still-live prepared entries, while
already completed identities remain invalid and are never revived.
The executor handoff is explicitly three-way. Accepted graphs retain watchdog
state; a definite rejection before GPU-visible action cancels the prepared
batch; an uncertain service error retains every identity for deadline expiry
and reset. Treating uncertainty as rejection would permit late GPU completions
to collide with reused state and is prohibited by the interface.
Raw prepared batches are no longer executor-visible. The executor accepts only
`ValidatedGpuBatch`, which owns the same watchdog identities plus a validated
launch proof for every entry. Only `VerifiedGpuKernelBundle::validate_batch`
can construct it, and one untrusted bundle, unsupported kernel schema, or ABI
mismatch rejects the whole conversion without GPU-visible effects.
The KVM adapter now consumes the common layout and registers two dedicated
memory slots transactionally from the caller's perspective: command memory is
guest-writable and completion memory uses KVM's read-only flag while remaining
host-service writable. A nested-KVM regression attaches both rings to the
signed high-half kernel guest, verifies directional writes, and still reaches
the authenticated framebuffer marker. The WHP adapter applies the same common
layout through independent GPA mappings; a live Windows Hyper-V regression
verifies command writes, completion write denial, and continued execution of a
verified PE guest. Platform cache-coherence validation, host CUDA executor,
kernel-specific argument schemas, operation-graph batching, completion queue,
IOMMU plumbing, the platform-specific device-reset callback, and end-to-end
inference benchmarks remain pending. Until those pieces exist and are audited, this is not a working
shared-CUDA Hyper-V device. It is intentionally MRML-specific instead of a
general `virtio-cuda` compatibility layer.

The CUDA service must verify the exact embedded PTX bundle as a
`CudaKernelBundle` artifact before module loading. Kernel IDs are enabled only
after this verification token exists; a digest mismatch or missing signature
leaves the virtual GPU unavailable and inference falls back to CPU.
`VerifiedGpuKernelBundle` now enforces this boundary: it accepts only the
unforgeable result of typed artifact verification and constant-time compares
that signed digest with the exact embedded bytes. A different artifact kind,
empty bundle, or changed PTX cannot mint the token even when a numeric
`KernelId` is otherwise valid.
The existing native CUDA runtime now provides the service boundary with a
read-only view of its build-embedded PTX and a bounded ID-to-symbol lookup for
all 28 compiled kernels. A non-hardware regression proves every accepted ID is
unique, nonempty, NUL-free, and identical to the runtime's fast lookup table,
while ID 28 and arbitrary names are rejected. This closes registry drift; it
does not by itself implement the service-side argument schemas or graph launcher.
The first typed executor schema is implemented for kernel ID 7 (`add_f32`). It
derives the scalar element count from three equal nonempty f32 buffer ranges,
requires read/read/write permissions and four-byte alignment, fixes the block
to 256 threads, verifies the corresponding grid, and forbids shared memory.
Kernel ID 0 (`gemm_q4_0_f32`) is also typed: positive signed rows, columns, and
batch dimensions must produce the exact Q4_0 weight length (18 bytes per 32
values), f32 input/output lengths, required alignments, access modes, and tiled
grid. Checked multiplication rejects dimension overflow before handoff.
Kernel IDs 3 and 4 reuse one checked GEMV policy while retaining distinct Q4_0
and Q8_0 storage sizes, block widths, and grid tiling. Substituting Q4 storage
for a Q8 dispatch or changing rows, columns, permissions, or output length is
rejected before the executor sees it.
Kernel IDs 9 and 10 share the proven three-buffer f32 elementwise policy with
ID 7. Kernel ID 8 validates a nonnegative signed token against the actual f32
embedding-table row count and binds the dimension, output bytes, alignment,
permissions, block width, and derived grid. A token at the row count is already
out of bounds and fails before launch.
Kernel ID 12 supports weighted and unweighted RMS normalization without a raw
nullable pointer: two accesses mean input/output and three mean
input/weight/output. Dimension and batch must exactly determine tensor and
weight bytes, while epsilon must be a typed, finite, positive f32. Grid,
alignment, permissions, and absence of dynamic shared memory are also fixed.
Kernel ID 11 validates in-place f32 RoPE. Position is a nonnegative signed
integer; head dimension is positive and even; head count is positive; frequency
base and scale are finite positive f32 values. Their product determines the
exact read/write buffer length, while head count and half-head width determine
the only accepted grid and block geometry.
Kernel ID 27 validates Q8_0 embedding rows using the kernel's exact 34-byte
storage per 32 values. Dimension must be divisible by 32, token must be inside
the row count derived from the table range, output must be one exact f32 row,
and output scale must be finite and positive. Alignment, permissions, grid,
block size, and shared-memory policy are fixed.
Kernel ID 13 binds positive model dimension, expert count, and batch to exact
f32 router-weight, input, and logits matrices and to its two-dimensional grid.
Kernel ID 14 requires at least eight experts, exact input logits, writable i32
top-8 IDs, writable f32 probabilities, and one single-thread block per token.
Checked dimension products and alignment checks precede every length test.
Kernel IDs 1 and 2 validate fused Q4_0 QKV GEMM and GEMV. Q, K, and V weight
ranges are independently sized from Q rows, KV rows, and the shared 32-value
column blocks. Input and packed output lengths are derived with checked batch
products. GEMM and GEMV retain their distinct block widths and row/token tiles.
Kernel IDs 5 and 6 validate fused Q4_0 GeGLU GEMM and GEMV. Gate and up weights
must each exactly match the same positive row/column dimensions, while checked
batch products determine the f32 input and output ranges. Their prefill and
decode variants retain distinct row tiles, token tiles, and block widths.
Kernel IDs 17 and 18 validate vocabulary top-k. Exact logits, validity mask,
recent i32 history, writable f32 scores, and writable i32 IDs are bound to
vocabulary, recent count, generated count, k, and partition count. Empty
partitions and k beyond a partition are rejected. The 256-thread specialized
kernel is accepted only when the maximum partition is at most 2048; larger
partitions must use the single-thread generic kernel.
Kernel ID 19 validates in-place QKV normalization, RoPE, and cache conversion.
It binds positive head counts and even head width to the exact packed QKV and
normalization ranges, requires grouped-query head divisibility, and rejects
position arithmetic that could overflow the kernel's signed indices. K and V
caches independently select only F16, Q8_0, or Q4_0 storage; quantized formats
require 32-value head blocks and every cache length is derived exactly from
capacity, KV heads, width, and format. Grid and block geometry are fixed from
the same dimensions.
Kernel ID 20 validates the bounded shared-memory F16 attention variant. Query
stride and dimensions determine the least readable query span, while capacity,
KV heads, and width determine both full cache ranges. Capacity must be a power
of two, head grouping must divide exactly, signed cache positions cannot
overflow, and output geometry is exact. The effective sliding-window key count
must be at most 8192 and determines the only accepted dynamic shared-memory
size; quantized cache formats cannot select this variant.
Kernel ID 21 reuses the checked attention dimensions for the zero-shared-memory
streaming variant and admits F16, Q8_0, or Q4_0 K and V cache layouts
independently. Quantized widths must use complete 32-value blocks. Variant
selection is exclusive, so a dispatch eligible for ID 20 cannot be relabeled
as ID 21. The mediated subset conservatively retains power-of-two cache
capacity and even head width.
All other IDs return `UnsupportedKernelSchema`, so adding an ID to the signed
registry alone cannot make it executable. The remaining 7 schemas and the
actual CUDA launch adapter are pending.

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
Descriptor construction now lives in the reusable x86_64 architecture module
rather than being duplicated in the PE image. It validates canonical handler
and table ranges, a ring-zero GDT selector that names an existing entry, the
exact 256-gate IDT size, and 16-bit descriptor limits before writing memory or
executing `LGDT` or `LIDT`. Unit tests check the exact 16-byte gate encoding and
prove malformed pointers, sizes, handlers, and selectors fail before privileged
instructions. The kernel image treats any installation error as a fail-stop.
Nested KVM initially reported a triple fault after `UD2`. A bounded post-exit
snapshot and four-level hardware page walk proved that CR2 named the valid,
present GDT code descriptor and that the vector-6 gate was exact. The descriptor
used type `0x9a`, leaving its architectural accessed bit clear; exception entry
therefore tried to set that bit on a deliberately read-only GDT page under
CR0.WP. The code descriptor is now encoded pre-accessed as type `0x9b`, and
installation rejects descriptors that are absent, non-code, non-ring-zero,
non-long-mode, default-operand-size, or not pre-accessed. A freshly signed,
high-half-relocated version-7 probe entered the kernel-owned invalid-opcode gate
and reached `HLT` under nested KVM without changing the framebuffer.

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
interrupt injection. KVM has a prepared launch path and Hyper-V has an initial
native lifecycle/memory path; register initialization, device models, and
shared-memory data queues are still incomplete, so this is not yet a runnable
hosted VM.

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
descriptor ownership, KVM API-version validation, VM and vCPU creation,
memory-slot registration, bounded `kvm_run` mapping, and exit execution are
also implemented with RAII cleanup. System admission queries the stable KVM
extension interface and rejects hosts without `KVM_CAP_USER_MEMORY` or at least
five memory slots before VM creation; API version alone is not treated as proof
of optional capability support. The adapter now owns fixed-capacity
anonymous guest RAM, rejects overlapping GPA regions and cross-region copies,
enforces read-only host writes, registers slots transactionally, validates its
single vCPU identity, implements the common `VmBackend` copy/run methods, and
injects validated vectors through `KVM_INTERRUPT`. Register setup and
interrupt-chip creation are now implemented: vCPU entry and stack addresses
must be canonical, CR3 must be page-aligned, general registers begin cleared,
RFLAGS bit 1 is set, and long mode enables paging, protected mode, PAE, NX, and
supervisor write protection with explicit 64-bit code and data segments. The
bootstrap adapter deliberately omits KVM's in-kernel IRQ chip until interrupt
delivery is configured. Guest page-table construction and
the backing store for it are now shared with the microkernel's page-table
builder. A fixed guest-RAM arena returns only freshly zeroed frames and rejects
reads or writes to unallocated tables, out-of-range entries, read-only RAM, and
arena exhaustion. KVM therefore inherits the same hardware W^X, NX, user-bit,
and duplicate-mapping checks as bare metal. PE materialization and final
section mapping are implemented, but their public entry points accept only a
`VerifiedExecutable` produced by signature verification. The exact image size
is zero-filled and relocated in owned RAM, every mapping is prevalidated before
hardware tables change, and final permissions come only from the validated PE
load plan. A late page-table storage failure discards the unpublished VM.
Canonical boot-handoff placement is also implemented. The complete handoff is
decoded and validated before guest memory changes, must occupy isolated
page-aligned storage, and has its unused page tail zeroed before being mapped
read-only and NX. This prevents a malformed record from partially replacing the
previous launch state. Host attestation remains part of the VMM trust boundary;
the handoff cannot make an untrusted host truthful. A fail-closed launch
transaction now composes these stages and publishes only `PreparedKvmGuest`:
it validates physical and virtual ranges, creates the VM and vCPU, registers up
to five disjoint memory slots, installs the signed image, immutable
handoff, guarded stack, and page tables, then writes CR3 and entry registers
last. The Microsoft x64 entry frame places the handoff pointer and length in
RCX/RDX and reserves a zero return slot with the required stack alignment.
Low-level construction and mutable backend extraction are not public. The Arch
WSL2 environment now exposes nested KVM through Hyper-V enlightened VMCS; API
version 12 and the 12,288-byte vCPU mapping query both succeed. A live regression
constructs and verifies a complete Lamport-signed VM artifact, materializes its
PE at a canonical supervisor high-half address, creates the VM and vCPU,
five memory slots, and hardware page tables, enters at the signed entry point,
writes a known pixel into the handoff-authenticated framebuffer, executes
`HLT`, and observes both `VmExit::Halted` and the pixel through a bounded guest
copy. Kernel launch derives the identity-mapped, supervisor-only, writable, NX
framebuffer solely from the validated handoff and rejects overlap with every
private launch region.
This proves the complete signed artifact-to-native-KVM execution and framebuffer
boundary. The core-only `mrml-kvm-run` host utility now verifies a release public
root and signed kernel bundle, prepares that same bounded launch environment,
and executes the real four-section, 8,704-byte standalone kernel PE at a
canonical supervisor high-half address. Nested KVM reaches the kernel-owned
framebuffer marker and reports its final halted state. Bootstrap launch omits an
in-kernel IRQ chip so the current interrupt-disabled `HLT` is observable; adding
an interrupt controller belongs with the timer and scheduler rather than this
one-shot boot proof.

The runner obtains 256 bits of fresh boot entropy from the host operating
system's cryptographic generator and places the verified artifact's actual
SHA3-512 digest in the handoff measurement field. It accepts an explicit nonzero
minimum release version and rejects older signed bundles before creating a VM.
It claims secure loading because signature and PE policy verification occurred,
but deliberately leaves measured-boot and rollback-protected evidence clear:
software hashing is not a hardware trust-anchor measurement, and a command-line
version floor is not persistent monotonic rollback state.

Successful launches report monotonic microsecond timings separately for
signature and PE verification, VM preparation, first kernel execution, and
total runner latency. These measurements are observational and never affect an
admission decision; they provide stable phase boundaries for performance
regression tracking without weakening fail-closed behavior.
On the current Arch WSL2 nested-KVM host, five consecutive release launches
measured 725--2,095 microseconds for verification, 1,065--1,986 microseconds for
VM preparation, 130--228 microseconds from `KVM_RUN` through the kernel marker
and halt, and 2,025--4,522 microseconds total. These are a recorded development
baseline, not a portable performance guarantee.

The reproducible integration sequence is:

```text
cargo build --release -p mrml-kernel-image --bin mrml-kernel-pe --features kernel-image --target x86_64-unknown-uefi
cargo build --release -p mrml-sign -p mrml-kvm-run
mrml-sign keygen release.private release.public
mrml-sign sign-bundle kernel 1 mrml-kernel-pe.efi release.private kernel.signed
mrml-kvm-run kernel.signed release.public 1 boot
```

Use the explicit `fault-probe` mode only with a signed kernel built with the
matching feature. It requires a halted VM and an untouched framebuffer; normal
boot requires the authenticated color marker. Unknown modes fail before file or
VM access. Unexpected exits capture only bounded architectural state and walk
the faulting address through guest-owned page tables for precise diagnostics.

The allocation-free `kvm_run` decoder validates the fixed x86 header before it
reads the kernel-owned union: padding is zero, readiness and IF fields are
boolean, only the defined SMM flag is accepted, and CR8 is bounded. Port I/O is
limited to one scalar transfer and its data offset cannot alias exit metadata.
MMIO requires zero padding and an exact supported width. MRML hypercalls require
the private number, one descriptor argument, zero unused arguments and return
state, zero padding, and a long-mode origin. Exception vectors above 31 fail
closed. These checks keep malformed kernel ABI state from becoming a broader
backend-neutral operation.

The separate core-only `mrml-whp` crate now provides the first Windows
Hypervisor Platform boundary without SDK bindings, import libraries, `std`, or
third-party crates. It dynamically resolves the documented C entry points from
`WinHvPlatform.dll`, validates the hypervisor-present capability and its exact
result size before allocating a partition, creates a one-vCPU partition in the required property,
setup, and vCPU order, and unwinds partial construction with owned guards.
Guest ranges must be page-aligned, nonempty, overflow-free, nonoverlapping, and
W^X. Fixed-capacity, zero-initialized host allocations are filled before being
mapped, are never exposed as mutable pointers, and are unmapped before the vCPU
and partition are deleted. Exit decoding reads the documented byte layout
instead of borrowing the C union: halt and cancellation translate directly,
I/O widths and reserved bits are checked, and memory exits preserve the exact
read/write/execute fault type rather than inventing an MMIO width. Unknown or
unfaithfully representable exits fail closed. The public native object is
`PreparedWhpPartition`; raw partitions and partially initialized vCPUs cannot
escape construction, and its lifetime is tied to the loaded DLL so resolved
function pointers cannot outlive their code. Hardened x64 register setup now
validates canonical entry/stack/handoff addresses, aligned CR3, Microsoft ABI
stack alignment, and bounded handoff length. It installs explicit long-mode,
PAE, supervisor write-protect, NX, RFLAGS, code/data segments, and RCX/RDX
handoff state through one register transaction. The launch transaction now
accepts only a `VerifiedExecutable`, validates the complete boot handoff before
native allocation, creates four disjoint table/image/handoff/stack mappings,
materializes and relocates the PE image, builds section-specific W^X page
tables, maps the handoff read-only and NX, writes a guarded ABI stack, and
programs CR3 and entry state last. Only `PreparedWhpGuest` is published after
all stages succeed. A live Windows test now creates and deletes an actual WHP
partition, vCPU, and guest mapping when the host reports Hyper-V present.
The partition enables WHP's xAPIC emulation before setup, and prepared guests
can request fixed, edge-triggered, physical-destination interrupts through the
documented 16-byte control record. Architectural vectors 0--31 and vector 255
are rejected before the host call. `PreparedWhpGuest` now implements the same
`VmBackend` run, bounded-copy, and interrupt contract as KVM and rejects every
vCPU identity except its single configured vCPU. Long-mode state also sets the
architectural MP, ET, and NE control bits and a 32/64-bit data-segment stack
width instead of relying on reset-state defaults. Launch now reserves a fifth
isolated page for a canonical two-entry GDT, maps that descriptor page read-only
and NX, supplies GDTR and the reset PAT value, and reads back all fifteen
critical general, segment, table, control, and MSR values after WHP accepts the
state. The readback tolerates only WHP's documented derivation of EFER.LMA;
every requested bit and all other values must match exactly. The initial probe's
real-mode exception-13 vector access at `0x34` was traced to a cached CS limit
of `0x000f_ffff`: the entry RIP at `0x20_0000` exceeded that limit before the
processor activated long mode. The cached segment limit is now
`0xffff_ffff`. A live regression guest builds hardware page tables, enters
x86-64 mode, executes an `INT3`, and returns the expected breakpoint vector
through WHP. Breakpoint exits are explicitly enabled in partition policy and
decoded with reserved-bit validation. The native launch path now records the
temporary writable image allocation, materializes the verified PE, removes the
temporary GPA view, and installs one second-level mapping per validated PE load
region. Headers become read-only, code becomes read/execute, data may become
read/write, and writable/executable sections are rejected. If any replacement
mapping fails, the already-installed fragments are removed and the original
non-executable writable mapping is restored so cleanup remains deterministic.
A live Windows regression constructs a complete Lamport-signed VM artifact,
verifies its trust root and statement, loads its PE32+ image, proves its code
page is immutable after sealing, enters long mode at the signed entry point,
executes `INT3`, and observes the expected breakpoint vector. This demonstrates
the complete verified-artifact-to-native-execution boundary on Hyper-V;
end-to-end UEFI-to-kernel boot remains a separate pending integration gate.
The WHP exit decoder validates the common x86 VP prefix before consuming any
exit union, including execution-state and reserved fields. Memory and port-I/O
exits bound the captured instruction length and require all ABI-reserved bytes
to remain zero. A claimed valid GVA must be canonical. Port-I/O forwarding is
limited to scalar 1-, 2-, or 4-byte operations; string and REP forms fail closed
because the backend-neutral exit type does not carry their RCX/RSI/RDI state,
and scalar RAX data is masked to the declared width before policy sees it.
Exception exits accept only architectural vectors 0--31, validate their
captured instruction and reserved fields, and reject an error code unless WHP
marks it valid. Host-cancellation exits accept only the defined user-cancel
reason, while interrupt-window exits accept only interrupt, NMI, or exception
delivery types. Undefined control reasons never become a generic interruption.

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

Baseline recorded 2026-08-20 and rechecked on Arch Linux 2026-08-21 on the current development host, one million
iterations per sample:

| Environment | Capability authorization | Scheduler selection | VM exit accounting |
| --- | ---: | ---: | ---: |
| Windows `x86_64-pc-windows-gnullvm` | 548,200 ns total (548 ps/op) | 1,859,500 ns total (1,859 ps/op) | 729,600 ns total (729 ps/op) |
| Arch Linux under WSL2 | 547,390 ns total (547 ps/op) | 1,862,494 ns total (1,862 ps/op) | 728,894 ns total (728 ps/op) |

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
