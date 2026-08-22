# MRML microkernel design

This document records design intent, not a claim of production readiness. The
kernel is new, original Rust intended for CC0 dedication. Research informs
requirements and trade-offs; no third-party implementation is copied or
adapted.

## Security boundary

MRML assumes a conventional Windows or Linux host and its VMM process may be
fully compromised. Consequently, hosted KVM/WHP execution is a development,
compatibility, and benchmark mode only: such a host controls guest memory,
page tables, scheduling, device emulation, launch evidence, and the current
host CUDA service. Cryptographic framing cannot protect a key that the host can
read from guest RAM or derive from a handoff it created. No hosted-mode result
establishes confidentiality, integrity, verified boot, or correct GPU execution
against that host.

The production architecture is therefore a bare-metal type-1
microkernel/monitor launched by measured UEFI. The kernel contains only
mechanisms: address spaces, scheduling, interrupt/IOMMU control, capabilities,
IPC, and VM entry/exit. A small, networkless management service owns policy and
domain lifecycle but receives no direct memory authority over ordinary service
VMs. Networking, storage, display composition, tools, model loading, and GPU
mediation run in mutually isolated least-authority service VMs. The GPU service
must run as an MRML-controlled domain, not as a process owned by an untrusted
Windows/Linux host.

The compartment model is also retained inside hosted MRML for defense in depth:
it can contain a compromised tool or model service from peer domains when the
host remains honest. It does not create a nested security boundary against a
malicious outer host. Hardware confidential-VM facilities could optionally
reduce that trust in the future, but MRML makes no such claim without measured
launch attestation, encrypted/integrity-protected memory, protected interrupt
and I/O paths, rollback protection, and an audited confidential-GPU channel.

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

For compatibility systems where the host retains the GPU, MRML uses an original
paravirtual interface rather than forwarding CUDA or NVIDIA driver APIs. An
isolated GPU service owns the CUDA context. In hosted KVM/WHP mode that service
is controlled by the untrusted host and provides no host-resistance; in the
production design it is a capability-confined MRML service domain. A guest receives opaque,
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
session resources while decoding. The `SubmitBatch` control path revalidates
the seal, decodes against current generations, admits watchdog identities,
validates signed schemas, executes, and publishes authenticated completions in
one fail-closed call. The resource service dequeues and erases one owned slot,
authenticates it, executes allocation/free through a narrow transactional
backend, or yields a typed `SubmitBatch` outcome. Invalid messages cannot wedge
the ring head. Successful allocation/free results use a distinct authenticated
`MRGR` response domain and monotonic sequence. The service reserves response
ring and sequence capacity before consuming the command, preventing an
unreportable resource mutation. Transactional backend failures produce a
domain-separated authenticated rejection containing only the request identity;
host error details are not exposed to the guest.
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
verified PE guest. The kernel-owned completion ring reserves capacity for a
whole validated batch before execution, receives authenticated results only
after synchronized completion, and erases slots as the VMM consumes them.
The platform-neutral VMM bridge now performs the missing bounded copy between
those mapped slots and the kernel-owned command/completion queues through the
common `VmBackend` interface. Monotonic tickets prevent reuse before
consumption, and any uncertain backend read or write permanently poisons the
bridge until VM teardown.
KVM can now create these mappings as part of the initial signed-kernel address
space rather than attaching unreachable GPAs after CR3 setup. Command pages are
writable/NX, completion pages are read-only/NX, and both resolve to their
dedicated identity GPAs. Host completion publication uses a separate
service-memory capability that does not weaken the guest-facing write check.
WHP applies the identical initial page-table policy and exposes a bounded
four-level diagnostic walk for verification. The live Windows test checks both
leaf translations and permission bits, guest write denial, service-only
completion publication, and continued execution of the signed PE.
The guest and isolated service derive their queue session and key from the
authenticated 256-bit boot entropy with a versioned SHA3-512 domain. This makes
the identity unique to each launch without placing a reusable symmetric key in
the signed benchmark image; zero entropy is rejected.
The signed guest workload can use a typed producer that fills one fixed slot
before release-publishing its monotonic position. Counter forgery poisons that
producer. Its VMM notification contract uses port `0x4d52`, and the diagnostic
`BenchmarkAdd` command admits only bounded element/iteration counts. It cannot
name memory, kernels, symbols, modules, or launch geometry.
Platform cache-coherence validation, CUDA graph capture, IOMMU plumbing, the
platform-specific physical
device-reset callback, and end-to-end inference benchmarks remain pending.
Until those pieces exist and are audited, this is not a working
shared-CUDA Hyper-V device. It is intentionally MRML-specific instead of a
general `virtio-cuda` compatibility layer.

The KVM diagnostic path now completes the narrower one-shot benchmark loop.
The signed `gpu-benchmark` kernel derives its queue identity from authenticated
boot entropy, publishes a bounded `BenchmarkAdd`, exits through the fixed
doorbell, and refuses to report success until it authenticates a nonzero timing
completion. The Linux launcher copies and authenticates the command, runs the
original Rust-PTX add kernel through the native CUDA driver interface, checks
the full output, publishes the authenticated completion through its
service-only mapping, resumes the same vCPU, and verifies the kernel's green
framebuffer marker. The 2026-08-21 live nested-KVM run on an RTX 5070 Ti measured
26.500 us per 4,194,304-element launch over 1,000 iterations (1,899.34 GB/s
effective traffic). This establishes an actual booted-guest/VMM/GPU round trip,
but hosted-mode integrity remains outside the threat model.

The Windows WHP launcher now completes the corresponding signed-kernel path.
It installs the normal high-half PE mappings, authenticated handoff, framebuffer,
and split GPU rings before vCPU entry. Scalar `OUT` exits are decoded strictly
and advanced using WHP's reported instruction length, avoiding repeated
doorbells. The service accepts only request doorbell value 1, executes and
checks the Rust-PTX add, and publishes through service-writable backing that is
still read-only under the guest-memory contract and guest page tables. After
authenticating the response and painting its marker, the WHP-specific benchmark
image emits fixed success value 2 rather than relying on implementation-specific
`HLT` wakeup behavior. A 2026-08-21 Windows release run on the RTX 5070 Ti
measured 28.429 us per 4,194,304-element launch over 1,000 iterations (1,770.41
GB/s effective traffic). This is mediated CUDA from a genuinely running guest,
not GPU assignment: the compromised host remains able to forge the measurement.

The live KVM and WHP launchers now enforce the signed-kernel-bundle boundary
rather than merely relying on the embedded PTX. Each runner can export its exact
compiled PTX with `--export-cuda-bundle`; release tooling signs those bytes as a
separate `CudaKernelBundle` using a one-use key. GPU benchmark mode requires the
signed bundle, its public key, and a nonzero minimum version. Before VM creation
or CUDA initialization, the launcher validates the artifact kind, Lamport
signature, rollback floor, and constant-time digest equality against the PTX
compiled into that runner. A live Windows mismatch test signed the kernel PE as
a valid CUDA-kind artifact and was rejected before WHP entry because its bytes
did not match. The successfully admitted Windows run measured 21.444 us and the
corresponding nested-KVM run measured 35.506 us per 4,194,304-element launch;
these single samples are functional gates, not updated performance baselines.

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
Kernel ID 15 binds FFN preparation's two full f32 input tensors, four
dimension-sized normalization vectors, and four distinct writable tensors to
one checked dimension/batch product and one block per token. Kernel ID 16 binds
the residual and dense inputs, read/write MoE accumulator, three normalization
vectors, and final output to the same geometry; its layer scale must be a typed
finite f32. Both require fixed 512-thread blocks and no dynamic shared memory.
`ValidatedExpertSelection` supplies the missing data-dependent-index boundary
for MoE kernels. It validates exact little-endian i32 ID and f32 weight ranges,
rejects negative, out-of-range, duplicate, non-finite, or zero-mass selections,
and caps active experts at 32. A domain-separated SHA3-512 digest binds both
buffers and their expert/active/batch dimensions; the GPU service must verify
the bytes again immediately before launch to close mutation after validation.
Kernel IDs 22 through 26 are implemented behind this proof-bearing API rather
than the ordinary shape-only validator. Generic gate/up and batched down,
single-token combined down, and fixed Gemma 4 26B variants bind exact Q4_0
matrices, selection buffers, scales, tensors, and geometry. Variant selection
is exclusive, and `ValidatedMoeKernelLaunch` retains the proof for immediate
service-side reverification.
`validate_batch_with_expert_selections` integrates these launches into mixed
batches atomically. It searches only supplied sealed proofs, rejects any MoE
entry without an exact buffer/dimension match, and stores the matched proof at
the same batch index. The ordinary batch admission path continues to reject
MoE IDs, so contextual provenance cannot be omitted accidentally.
`MediatedGpuExecutor` implements the host orchestration boundary in two passes.
The first asks the trusted isolated-memory backend to re-read and verify every
retained expert selection; any mismatch rejects before a launch. Only after the
entire preflight succeeds does the second pass lower validated entries in order.
Backend errors during that pass retain uncertain-acceptance watchdog semantics.
`MediatedCudaBackend` is the concrete CUDA adapter. Its fixed-capacity table
maps generational IDs to service-owned device allocations, checks every range,
rereads bounded MoE proof bytes, and constructs driver arguments only from
resolved addresses and typed scalar bits. The validated kernel ID selects the
build-embedded symbol and all geometry comes from the validated dispatch. Raw
binding is unsafe with explicit lifetime and non-aliasing obligations.
`MediatedCudaService` couples this table to `VirtualGpuSession` quotas. Allocate
first reserves a generational handle and rolls it back if CUDA allocation or
binding fails. Free releases only the matching service-owned CUDA allocation
before returning quota; raw external bindings cannot enter that path. Dropping
the backend best-effort releases all remaining owned allocations.
After lowering the ordered batch, the backend synchronizes its CUDA stream
before the executor reports acceptance. Barrier failure is an uncertain result,
not a clean rejection, so the watchdog retains every in-flight identity until
completion or reset recovery resolves device state.
The lifecycle submitter validates every watchdog identity and reserves enough
completion sequence space before calling the backend. Once synchronization
succeeds, it retires the identities and produces ordered authenticated success
frames. Clean rejection cancels all identities; uncertain errors leave all
identities live for deadline or reset handling.
Deadline expiry emits authenticated `TimedOut` frames in stable slot order;
device recovery emits authenticated `DeviceReset` frames for every remaining
entry. Both paths reserve completion sequence space before invalidating any
generational identity, preventing partial retirement on sequence exhaustion.
All other IDs return `UnsupportedKernelSchema`, so adding an ID to the signed
registry alone cannot make it executable. All embedded kernel schemas,
including the proof-bearing MoE variants, are integrated with the concrete CUDA
launch adapter.

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
hardware trust anchor, and recovery policy remain required. The admitted kernel
now installs and owns its interrupt tables independently of firmware.

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
The successful `ExitBootServices` call is treated as an irreversible state
transition. All subsequent memory-map, handoff, and launch failures enter the
loader's fail-stop loop rather than returning a status through invalidated UEFI
state.
Its entry parses the canonical handoff again and revalidates nonzero entropy,
the ACPI pointer, the sorted memory map, and complete framebuffer MMIO
containment before drawing. It retains every emitted region in fixed bounded
storage and constructs the same architecture-neutral `EarlyKernelContext` used
by later initialization, so the standalone image does not bypass the common
early-boot admission boundary. On Windows QEMU 11.1, a freshly generated one-use
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

The admitted PE is copied into fully erased UEFI LoaderCode pages, including
the padding after `SizeOfImage` in its final mapped page, and only bounded DIR64
relocations are applied. Each nonzero relocation must originally point
inside the preferred image and is converted to the corresponding checked RVA
at the actual 4 KiB-aligned load address. The entry must be inside a validated
executable, non-writable section, and the loader-owned CR3 enforces those final
section permissions in hardware.

The x86-64 transition now allocates a new zeroed four-level page-table tree and
a fully erased 128 KiB kernel stack arena before leaving firmware. The fixed
arena contains a six-page early stack, an omitted guard page, a sixteen-page
ring-transition stack, a second omitted guard page, and an eight-page
double-fault IST stack. One additional lower allocation page is omitted as an
early-stack underflow guard. The loader passes both protected stack tops through
the PE entry ABI; the kernel rejects invalid tops before installing its TSS and
retains no static fallback privilege stack inside the image. It identity-maps only
the authenticated PE regions with their final read-only, writable/NX, or
read-only/executable permissions; the three stack regions and GOP aperture writable/NX; the
canonical handoff read-only; and one page-aligned read-only/executable assembly
trampoline. The trampoline enables EFER.NXE and CR0.WP, replaces CR3, changes
to the dedicated stack, and jumps without returning. No writable alias of an
executable image page is retained. The handoff has a dedicated, statically
size-checked 4 KiB page whose complete contents are erased immediately before
the canonical prefix is encoded. The transition rejects a trampoline whose linker
symbols do not prove that it fits entirely within its dedicated page. Thus
neither restricted mapping silently grants access to adjacent loader data or
code. QEMU 11.1 reached the independent kernel
marker with `CR0=0x80010033`, `EFER=0xd00`, and the loader-created CR3 root.
The same signed cross-root timer-preemption images ran under KVM and WHP with
the launcher-provisioned supervisor stacks; isolated service roots omit both
internal guard pages.
Before the loader's first GOP write, the shared framebuffer validator now
rejects misaligned bases, unsupported geometry or stride, undersized or
overflowing apertures, and lengths above Rust's `isize::MAX` raw-slice bound.
The kernel applies the same validation again when decoding the authenticated
handoff.

The standalone kernel immediately installs an image-owned GDT preserving the
transition selectors and a complete image-owned IDT. Architectural exceptions
enter vector-specific stubs; unassigned external vectors retain a non-returning
interrupt-disabled fallback. A signed
`fault-probe` build executed `UD2` under QEMU and stopped with `HLT=1` at the
kernel handler while retaining the loader-created CR3 and kernel GDT/IDT bases.
Descriptor construction now lives in the reusable x86_64 architecture module
rather than being duplicated in the PE image. It validates canonical handler
addresses and table ranges, a ring-zero GDT selector that names an existing
entry, the exact 256-gate IDT size, 16-bit descriptor limits, IST indices, and
gate privilege before writing memory or executing `LGDT` or `LIDT`.
The architecture module also defines the exact 176-byte trap-frame contract
used by the assembly entry stubs. Its fail-closed dispatcher validates
the exception vector, privilege transition, canonical RIP and user RSP, fixed
RFLAGS bits, and whether that vector architecturally carries an error code.
Validated ordinary user exceptions terminate only the current task; NMI,
double fault, machine check, every kernel exception, and every malformed frame
halt the kernel. Page faults additionally require a canonical captured CR2.
The image installs 32 distinct original assembly stubs. They normalize vectors with
and without CPU error codes, preserve every general register, clear DF, and
enter the Rust dispatcher through an explicit SysV64 boundary. IDT construction
validates every handler before modifying the live table, exposes only breakpoint
and overflow gates to ring three, and retains the fail-stop fallback for vectors
32--255. A signed nested-KVM `fault-probe` build executed `UD2`, emitted the
dispatcher's exact vector-6 proof through diagnostic port `0x4d53`, resumed,
and halted cleanly in 156 microseconds. Later signed CPL3 probes prove the same
path revokes a faulted task domain and restores a replacement through
`CR3`/`iretq` rather than merely failing the kernel.

The image-owned GDT now also contains ring-three code and data descriptors and
an exact 104-byte x86-64 task-state segment. Initialization validates canonical,
nonzero, 16-byte-aligned `RSP0` and IST1 stack tops, writes the two-slot available
TSS descriptor before loading the GDT, loads `TR`, and disables the TSS I/O
bitmap by placing its base immediately beyond the segment limit. Vector 8 alone
uses IST1, so a double fault does not depend on the interrupted stack. Windows
and Linux unit tests verify the packed offsets and descriptor encoding. A fresh
signed nested-KVM `fault-probe` reached its expected vector-6 halt after `LTR`
in 157 microseconds (`verify=694us`, `prepare=1156us`, `total=4830us`). The
entry and double-fault stacks now come from the fixed launcher-owned 128 KiB
arena. They are separate sixteen- and eight-page supervisor mappings with an
absent guard below each; the kernel image contains no static fallback privilege
stack. The first guarded two-syscall service run faulted at
`CR2=...60006d88`, proving the former 16 KiB transition stack overflowed into
its newly absent guard. Enlarging only the mapped stack regions retained both
guards and made the same signed path complete on KVM and WHP.

External vector installation is now a separate validated operation restricted
to vectors 32--254 and performed only while interrupts are disabled. The
`timer-probe` image constructs one runnable scheduler task, enables the local
APIC software bit, lowers task priority, masks the LVT while programming a
periodic vector/count/divisor, and observes a real counter wrap while IF is
clear. It then emits a second marker and enables interrupts. Its assembly entry
preserves all general registers before calling the Rust handler. The handler
advances `KernelScheduler` exactly one tick, acknowledges EOI, emits a distinct
proof, and fails closed. KVM creates its in-kernel interrupt controller before
the vCPU, installs the bounded supported CPUID set, and maps the xAPIC page
supervisor-only when x2APIC is unavailable. No host interrupt injection is used.
This APIC setup is restricted to timer guests so it cannot change halt behavior
for other launch modes. A freshly signed nested-KVM run completed guest
execution in 1,856 microseconds (`verify=744us`, `prepare=10296us`,
`total=15834us`). The corresponding WHP proof uses Hyper-V's xAPIC emulation
and the same supervisor-only page-table policy;
the freshly signed identical kernel reached its tick in 8,232 microseconds
(`verify=758us`, `prepare=2971us`, `total=13998us`). A live timer-driven switch
now also runs in the signed `preemption-probe`: vector 32 arrives from task A
with `CS=0x23`; the kernel reconstructs A's complete context, rejects any CR3
that differs from A's bound address space, commits the quantum switch, selects
B, acknowledges EOI, and restores B through the common `iretq` transition. B
raises a checked CPL3 breakpoint rather than using forbidden port I/O. The
final artifact completed in 2,073 microseconds on nested KVM and 8,315
microseconds on WHP. The signed `service-preemption-probe` repeats the path
across independently materialized service roots: A is interrupted under
`CR3=0xc00000`, its frame remains bound to that domain, and B is restored under
`CR3=0xd00000`. Each root maps the supervisor kernel and only its own user
image/stack; timer-enabled roots add the APIC page supervisor-only. Final
execution measured 2,063 microseconds on nested KVM and 8,413 microseconds on
WHP.

The `user-probe` diagnostic now gives the context and TSS work a live privilege-
transition proof. Its signed PE is relocated into a bounded lower-half layout;
the kernel loads the ring-three data segments and an exact `SS:RSP`, RFLAGS,
`CS:RIP` frame, then executes `IRETQ`. The embedded CPL3 instruction deliberately
raises invalid opcode. Hardware switches to TSS `RSP0`, the vector-specific
entry captures the user RSP/SS tail, and checked policy admits exactly
`TerminateUser { vector: 6, address: None }` before emitting proof. A freshly
signed nested-KVM run completed in 178 microseconds (`verify=758us`,
`prepare=946us`, `total=4669us`). This diagnostic intentionally marks all PE
sections and its stack user-accessible and halts at the first fault; it is not a
security boundary and cannot ship as a service configuration. The production
path still requires separately authenticated service pages, a private guarded
user stack, a kernel-only higher-half mapping, CR3/PCID switching, and revocation
before another context can run.

User entry is no longer encoded in the diagnostic image. The reusable x86-64
transition consumes the fixed 152-byte `UserContext` contract, writes its
page-aligned physical root to CR3, builds the exact five-word privilege frame,
sets DS/ES/FS/GS to the ring-three data selector, restores all fifteen general
registers, and executes `IRETQ`. Compile-time Rust layout plus unit tests bind
every assembly offset to the context representation. Its safety contract
requires the new root to retain the transition page until the CR3 write, map
the target RIP user-executable and RSP user-writable, and retain a kernel-only
TSS `RSP0`. Windows and Linux now pass 129 kernel tests. A fresh signed KVM
probe exercised this reusable CR3/register path and the checked return path in
210 microseconds (`verify=729us`, `prepare=1231us`, `total=5403us`). The probe
used the same root before and after CR3 as an earlier transition proof; the
two-root service probe below now proves address-space switching as well.

`TaskRuntime` now binds each scheduler identity to exactly one saved
`UserContext` and one fixed-capacity `CapabilitySpace`. Recoverable fault
handling first takes the complete domain—making its context and all task-local
handles unreachable—then calls `terminate_current` and exposes only the
resulting replacement outcome. Missing domains are integrity failures, kernel
fault dispositions cannot enter this path, and generation changes prevent the
retired task identity from addressing a reused slot. Windows and Linux pass 115
kernel tests, including fail-closed retirement ordering. The signed CPL3 probe
now creates a real runtime task and emits success only after vector 6 has removed
its domain. The probe now contains a second runtime task with a distinct entry
and stack. After the first task raises `#UD`, the exception handler revokes it,
selects the second identity, and calls the same full CR3/register/`IRETQ`
transition directly from the exception stack. The replacement executes `INT3`;
only its checked vector-3 retirement reaching `Idle` emits success. A freshly
signed nested-KVM run completed both CPL3 entries and recoveries in 179
microseconds (`verify=821us`, `prepare=1032us`, `total=6465us`). Both tasks still
share the deliberately permissive diagnostic PE mapping, so this proves live
replacement mechanics but not cross-address-space service isolation.

The runtime now owns cross-task IPC routing as well. `send_ipc` requires two
distinct live generational task identities, derives only explicitly requested
rights from the sender domain into the receiver domain, then authorizes the
sender's exact SIGNAL capability against the endpoint object. If authorization
fails after derivation, every tentative receiver capability is revoked before
the error becomes visible; failure to perform that rollback is an integrity
failure. Same-task routing is rejected so callers cannot use this interface to
bypass a domain boundary. Windows and Linux pass 116 kernel tests, including
attenuation and failed-authorization rollback. Live syscall entry and blocking
endpoint queues are now exercised by the two-root service proof below.

The initial x86 user-call ABI is deliberately pointer-free. Exactly IDT vector
`0x80` receives a DPL3 interrupt gate; external interrupt gates remain DPL0 and
all other fallback vectors remain inaccessible to `INT`. Operation zero is
yield and accepts no nonzero reserved argument. Operation one carries an
endpoint capability token, a generational receiver task token, a length no
larger than 24, and three payload words in registers r10/r8/r9. Decoding zeros
the unused payload tail and rejects unknown operations, generation-zero tokens,
and oversized lengths without reading guest memory. Operation two receives or
blocks, and operation three voluntarily exits with no arguments. Exit removes
the complete context/capability domain before replacement selection, so an
exited identity cannot retain authority or resume. Windows and Linux pass 134
kernel tests, including exact gate privilege/vector placement and canonical ABI
decoding. The live entry now preserves all fifteen registers in an exact
160-byte `UserCallFrame`, validates the ring-three CS/SS, lower-half RIP/RSP,
and a complete RFLAGS whitelist that excludes IOPL, NT, VM, and reserved bits.
It dispatches inline send through `TaskRuntime::send_ipc`, returns status in RAX
and sequence in RDX, restores the frame, and executes `IRETQ`. The signed KVM
probe sends `ping` from task A to task B with sequence one, emits proof only
after endpoint authorization and message construction, returns to CPL3, then
continues through the existing `#UD` revocation and task-B breakpoint recovery.
A fresh run completed the combined path in 192 microseconds (`verify=783us`,
`prepare=1125us`, `total=6426us`). Blocking receive, wakeup, and production
service-image entry were subsequently completed by the two-root service proof.

`ServiceAddressSpace` now defines the production mapping boundary independently
of the permissive probe layout. Its public constructor accepts only a
`VerifiedExecutable` whose authenticated kind is `ServiceImage`. Each instance
copies only explicitly supplied higher-half supervisor mappings, maps the
validated PE allocation plan as lower-half user pages with final per-section
W^X permissions, maps a user-writable/NX stack, proves the immediately lower
page is absent, and can materialize the plan only through a fresh
`PageTableBuilder` root. Low supervisor mappings, user-marked kernel mappings,
physical aliases, overlap with the guard, overflow, and a non-service artifact
fail closed. Windows now passes 122 kernel tests. Live loading of separately
signed service bytes and switching between independently materialized roots is
now implemented in the KVM integration path. `PreparedKvmGuest` maps service
image, stack, and table arenas into separate KVM slots, maps the kernel PE
supervisor-only into a fresh root, maps the authenticated service PE user-only
at its signed preferred base, leaves the lower stack guard absent, and exposes
the root and entry only after all mappings succeed. The diagnostic kernel uses
root `0xc00000`, service entry `0x140001000`, and a two-page stack ending at
`0x702000`.

The first isolated run exposed an important CR3 transition invariant: both the
saved context and the stack used to build the `IRETQ` frame must remain mapped
after CR3 changes. Mapping the old boot stack would have broadened every service
root, so the architecture now provides `enter_user_context_on_stack`, which
moves first to the kernel-only per-CPU transition stack shared by both roots and
then performs the existing CR3/register transition. With the context stored in
kernel-only image data, a freshly signed kernel and independently signed 2 KiB
service PE entered CPL3 and returned through vector 3 under nested KVM in 208
microseconds (`verify=11987us`, `prepare=1149us`, `total=18464us`). The extra
verification time includes two independent Lamport/SHA3 artifact checks. WHP
now has parity: `PreparedWhpGuest::attach_isolated_service` creates separate
WHP GPA mappings, seals the service PE after relocation/materialization, and
builds the same supervisor-kernel/user-service root. The service partition does
not enable WHP's test-only breakpoint exit bitmap, so vector 3 reaches the guest
IDT instead of stopping in the host. A signed Windows run completed in 216
microseconds (`verify=1579us`, `prepare=2805us`, `total=6521us`). The runner
stops at the explicit kernel proof because re-entering an interrupt-disabled
HLT is not a portable WHP completion mechanism. Service syscalls and persistent
service scheduling were then advanced by making the independently signed PE
issue operation-zero through `INT 0x80` with every reserved register cleared.
The kernel validates the complete call frame, returns zero status, restores all
registers, and executes `IRETQ`; the service reaches its subsequent breakpoint
only if that return succeeds. Fresh signed runs completed the call, return, and
breakpoint chain in 260 microseconds on KVM (`verify=5908us`, `prepare=1325us`,
`total=11398us`) and 190 microseconds on WHP (`verify=1565us`, `prepare=3535us`,
`total=7117us`). A yield with one runnable service legitimately continues that
service. The next increment materializes the verified service PE twice with
distinct physical images, guarded stacks, and roots (`0xc00000` and
`0xd00000`). Receiver A executes operation two, and the kernel validates and
captures its exact post-interrupt continuation before blocking it. Sender B
runs under the second CR3, uses its exact SIGNAL capability and A's generational
task token to deliver `ping`, wakes A, and exits. The kernel revokes B before it
dequeues into A's
saved registers and restores A under its own root; A validates RAX, RDX, and
R10 before raising its completion breakpoint. Clean independently signed runs
completed the clean-exit chain in 326 microseconds on WHP and 491 microseconds
on nested KVM. One authenticated service
artifact supplies both instances; task identities, address spaces, physical
copies, stacks, and table arenas remain separate.

`ServiceSupervisor` now binds each service object to exactly one live
generational task identity. The signed exit syscall must resolve the current
task through this table before `TaskRuntime` removes its context and capability
space. Clean exit records `Exited`; checked user-fault retirement records
`Faulted`. Restart accepts neither state without exact `CONTROL` authority for
the recorded object and a newly supplied validated `UserContext`. It creates a
new task first, then advances the service generation and publishes the
replacement, so allocation failure leaves the stopped record unchanged and a
stale service ID never regains control. Windows and Linux tests cover wrong
object authority, clean and fault retirement, successful restart, and stale-ID
rejection. The current signed probe exercises supervised clean exit.

WHP and KVM now provide a platform reset primitive for the two isolated service
slots. The launcher privately records the admitted `ServiceImage` digest, image
size, physical/virtual placement, stack range, and page-table root. Reprovision
is permitted only after the caller observes kernel retirement with the vCPU
stopped, and only for the exact same verified digest and size. It withdraws the
entry and root before mutation, zeroes the entire image and stack backing,
rematerializes the verified PE (including initialized data and zeroed BSS), and
republishes only after every step succeeds. The unchanged page tables remain
safe for this operation because the exact image identity and mapping contract
cannot change. Both signed live runners reject substitution with the valid
kernel executable before publication changes, contaminate the second service
stack, reset it, and verify its bytes, entry, and root. The 2026-08-21 release
runs now connect reset to kernel-supervised restart. The receiver fault is
retired through `ServiceSupervisor`, the host rebuilds both stopped instances,
and exact kernel-held `CONTROL` capabilities authorize fresh service and task
generations. New task-local endpoint authority is issued to the replacement
sender, stale receiver/task tokens are not reused, and scheduler creation order
preserves the receiver-first blocking invariant despite its retained
round-robin cursor. Checked stages prove both fresh generations exist and the
replacement receiver is selected. The rebuilt pair then repeats the complete
block, send, clean sender exit, receiver wakeup, CPL3 breakpoint, and fault
retirement chain. The final bounded-policy artifact completed both generations
in 390 microseconds on WHP (`verify=1910us`, `prepare=3428us`, `total=7630us`)
and 570 microseconds on nested KVM (`verify=4018us`, `prepare=1358us`,
`total=9797us`) after the CPU-private descriptor refactor.
Every registration now binds a validated restart policy containing a nonzero
maximum restart count, base delay, and maximum delay. Retirement computes the
next eligible scheduler tick with checked arithmetic; delay doubles after each
completed restart and saturates at the configured maximum. Exact `CONTROL`
authorization precedes budget/timing disclosure, an early attempt neither
allocates a task nor advances a generation, and exhausting the count is
permanent for that service record. Time overflow occurs only after the retiring
domain has been revoked and fails closed. Windows and Linux tests cover invalid
policies, unauthorized probes, early retry, saturation, exhaustion, and time
overflow. The signed two-service policy permits one immediate restart and its
terminal marker is emitted only after the kernel proves a second restart is
rejected. A general policy/configuration service remains future work.
As elsewhere, these host-visible markers are test evidence, not attestation
against a compromised host; protecting guest memory from that host requires the
confidential-computing boundary described in the threat model.

Each `TaskRuntime` domain now owns a fixed two-message inbox in addition to its
context and capability space. `receive_or_block_current` dequeues immediately
or marks only the empty receiver blocked and selects a replacement.
`deliver_ipc` checks capacity before authorization, sequence advancement, or
capability derivation; after transactional `send_ipc` succeeds it publishes at
the tail and wakes the receiver. FIFO head/tail/count invariants make overwrite
impossible, and a full inbox returns without consuming endpoint sequence or
creating receiver capabilities. Windows and Linux pass 129 kernel tests,
including block, wake, FIFO, overflow behavior, syscall continuation capture,
and voluntary switching. The two-root signed probe exercises live delivery,
wakeup, and syscall-visible receive results on WHP and KVM.

Unit tests check the exact 16-byte gate encoding and prove malformed pointers,
sizes, handlers, and selectors fail before privileged instructions. The kernel
image treats any installation error as a fail-stop.
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
MRML ABI. The UEFI loader materializes the separately built kernel into
LoaderCode pages, builds final W^X page tables, exits boot services, switches
CR3 and stack in a page-aligned trampoline, and transfers to the validated
entry without return.

The early frame allocator now reserves aligned physically contiguous runs
without crossing normalized firmware regions. PE admission can consume that
allocator to produce a fixed-capacity physical load plan containing one run for
the read-only NX headers and one run per section. Alignment padding and partial
allocations are never recycled during boot, preventing stale-frame aliasing;
failure is therefore fatal rather than rolled back. This generic physical-plan
path remains available for non-UEFI loaders; the active UEFI loader uses its
firmware page allocator directly.

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
root physical frame without performing privileged state changes. The UEFI
adapter supplies a real firmware page store, maps the PE, stack, handoff, GOP
aperture, and transition trampoline, then the assembly transition enables NX
and write protection, writes CR3, installs the erased stack, and jumps to the
authenticated entry.

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

The kernel scheduler now has a bounded timer-driven owner above the existing
weighted round-robin policy. Timer frequency is restricted to 10--100,000 Hz,
the quantum must be nonzero and no longer than one second, tick overflow fails
closed, and preemption occurs only at an exact quantum boundary. Blocking and
termination clear the current identity before selecting a replacement; removal
advances its generation so a faulted task can never be woken or resumed through
a stale handle. Windows and Linux tests cover preemption, idle wakeup, task
termination, stale identities, and invalid timer policies. A separate release
microbenchmark now measures tick accounting. Validated x2APIC/xAPIC timer
programming and acknowledgement primitives now execute in the signed KVM guest;
the same primitives also execute in the signed WHP guest. Timer-driven context
capture and restoration now execute across distinct service CR3 roots on both.

The x86_64 architecture layer now defines a fixed user-context record and a
generational task-to-context table. New contexts require a nonzero page-aligned
CR3, canonical lower-half entry point, nonzero 16-byte-aligned initial stack,
fixed ring-three code/data selectors, and a minimal initial RFLAGS value.
Contexts captured from traps must carry those exact selectors and may contain
only explicitly enumerated user-modifiable status flags; IOPL, NT, VM, RF,
reserved high bits, kernel addresses, and zero roots fail closed. Binding,
replacement, lookup, and revocation use the complete generational `TaskId`, so
a context belonging to a terminated task cannot attach to a reused scheduler
slot. Windows and Linux tests exercise selector, flag, address, CR3, duplicate,
revocation, and stale-generation rejection. GDT user descriptors, TSS/RSP0,
assembly restore, CR3 switching, live ring-three entry, cross-root timer
preemption, and guarded privilege stacks are exercised by signed KVM and WHP
probes. Privilege-stack planning is now CPU-indexed for 1--256 CPUs. Its checked
stride reserves disjoint physical and virtual 128 KiB arenas, including both
internal guard pages, and rejects invalid counts, undersized strides, range
wrap, addresses beyond the 52-bit physical limit, noncanonical virtual
coverage, and invalid indices before page-table
construction. CPU 0 now uses that allocator in UEFI, KVM, and WHP launches.
Its GDT, complete IDT, TSS, RSP0/IST1 ownership, CPU identity, and transition
stack are held in one 4 KiB-aligned non-copyable `CpuDescriptorState`; table
installation refuses an out-of-range owner and repeat initialization. The
signed service lifecycle still completes under WHP and nested KVM after this
privileged-path refactor. A QEMU 11.1 TCG/UEFI boot of the plain kernel produced
1,022,848 background pixels at `RGB(11,59,90)` plus the exact 1,152-pixel gold
kernel marker at `RGB(255,200,87)`. Application-processor discovery/startup,
per-CPU interrupt routing, and live multi-vCPU scheduling remain pending, so
MRML does not yet claim SMP execution. Platform-backed service page
erasure/reprovisioning and one bounded restart are complete for the hosted
WHP/KVM proof.

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
