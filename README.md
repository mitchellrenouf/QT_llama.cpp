# MRML

MRML—the **Mitchell Renouf Machine Learning Library**—is an experimental,
CC0-licensed Rust workspace for running Gemma 4 GGUF models locally. It combines
a native inference engine with terminal, machine-readable, and
OpenAI-compatible HTTP interfaces plus agent tools.

MRML does **not** link to llama.cpp. Its tensor operations, GGUF loader, CPU
kernels, CUDA kernels, KV cache, and generation loop live in this repository.
That makes MRML useful as a compact inference-engine project, but it does not
yet have llama.cpp's model coverage, hardware coverage, or years of kernel
tuning.

## Architecture and design intent

MRML is a self-contained local inference runtime. Every library, application,
example, benchmark, CUDA kernel, and build script is compiled with `#![no_std]`
and does not import Rust's global `alloc` crate. Repository source—including
test modules—contains no direct `std` or `alloc` crate imports. Cargo's external
test harness still supplies its own host runtime when executing `#[test]`. This does not
mean inference uses no dynamic memory. MRML instead makes allocation explicit
through its own platform and accelerator layers:

- `mrml-tensor` contains portable tensor and inference primitives.
- `mrml-model` implements model behavior and generation.
- `mrml-windows` and `mrml-linux` provide operating-system memory, file,
  synchronization, timing, and process interfaces.
- CUDA memory is managed directly through the NVIDIA driver API.
- CLI, server, and tool crates contain application policy and user interfaces.

The goals are predictable latency, controlled memory behavior, minimal
third-party dependencies, portability, and a small auditable runtime. Hot paths
use bounded storage, caller-provided buffers, arenas, memory mapping, and device
memory instead of implicit general-purpose allocation where practical.

When analyzing or extending MRML, preserve the separation between portable
inference code, platform services, accelerator code, and application policy.
Do not interpret `no_std` or the removal of `alloc` as an embedded-only
limitation or as a claim that model execution never allocates memory.

## Mandatory contribution rules

These rules apply equally to human contributors and LLM/agent contributors.
Changes that do not satisfy every rule must not be merged:

- Submit only original work that the contributor has the right to dedicate
  under CC0. Do not copy, translate, transcribe, or adapt code from third-party
  projects, generated output with incompatible terms, or other sources whose
  provenance prevents legal release under CC0.
- All repository code must be Rust. Configuration files, Cargo manifests,
  lockfiles, licenses, and documentation may use their necessary data or prose
  formats, but implementations, build logic, tests, examples, benchmarks,
  utilities, and generated source must be Rust.
- Code may depend only on `core` and original crates contained in this
  workspace. Do not use `std`, `alloc`, third-party crates, or other crates
  shipped with Rust. Adding a dependency from crates.io, Git, the sysroot, or
  any external source is prohibited.
- Keep implementations modular, cohesive, and easy to audit and read wherever
  that does not conflict with security or measured performance. Security comes
  first, measured performance second, and readability and convenience follow;
  split large components behind narrow interfaces and document non-obvious
  invariants at trust boundaries.
- Every change must be built and tested on both Windows and Linux before it is
  accepted. Windows support is limited to Rust's MinGW-based native GNU/LLVM
  toolchain (`x86_64-pc-windows-gnullvm`) and facilities available through
  that toolchain; do not require MSVC, Visual Studio, the Windows SDK, or a
  separately installed LLVM. Linux code must likewise keep its native system
  interface and toolchain requirements minimal.
- Treat security as a release requirement. Review unsafe code, FFI boundaries,
  integer and buffer arithmetic, parsers, file and process operations, network
  input, and error paths; add adversarial and regression tests appropriate to
  the change. A change with an unresolved security weakness is rejected.
- Assume any Windows or Linux host operating system and host VMM process are
  compromised. Hosted KVM/WHP mode may be used for development, compatibility,
  and performance measurement, but it must never be presented as protecting
  guest confidentiality, integrity, keys, verified boot, or CUDA results from
  that host. Production security must root in measured bare-metal MRML code;
  secrets and signing authority must not be entrusted to a hosted VMM.
- Measure performance-sensitive changes against an appropriate existing
  baseline. Any security or performance regression is rejected and must be
  corrected and retested before the change is reconsidered; do not waive a
  regression merely to make a proposed implementation pass.
- Record the Windows and Linux commands, security checks, and relevant
  benchmark results used for verification in the contribution or commit
  evidence. LLM/agent contributors must report limitations honestly and must
  not claim that a platform, security property, or performance result was
  verified when it was not actually tested.

## Platform trust boundaries

On a bare-metal boot, the OS currently trusts the CPU and its documented
execution and privilege behavior, CPU microcode, the chipset and memory
controller, RAM, the TPM or other hardware attestation root when used, and the
UEFI firmware that loads and describes the machine. It also trusts the
repository-controlled MRML loader, kernel, policy, and embedded verification
keys after their signatures and measurements have been accepted. Peripheral
devices, device firmware, DMA input, network peers, model files, tools, and all
other external input are untrusted and must be validated and capability
restricted. Until MRML implements and verifies a dynamic root of trust and
independent hardware reinitialization, a compromised UEFI, SMM, platform
security processor, CPU, microcode, chipset, TPM, or physical machine is outside
the security boundary; ordinary verified boot cannot defend against those
components modifying the system after verification.

When booted by the development VMM, the OS additionally trusts that VMM and its
host operating system for guest memory, virtual CPU state, clocks, interrupts,
virtual devices, entropy, persistence, availability, and GPU computation. A
host-controlled VMM can read or alter the guest regardless of in-guest queue
authentication or image signatures. Consequently, the current KVM and WHP
paths provide functional isolation between cooperative guests and are suitable
for development and benchmarking, but provide no confidentiality, integrity,
verified-boot, attestation, or signing-key guarantee against a compromised
host. Such guarantees would require a separately validated hardware-protected
confidential-computing path and attestation rooted below the VMM; MRML does not
currently claim that support.

## Status

The experimental microkernel foundation and its security model are documented
in [`docs/MICROKERNEL.md`](docs/MICROKERNEL.md). It is not production-secure or
ready for bare-metal deployment. The original x86-64 PE32+ kernel does now boot
through the original UEFI loader under QEMU and directly under nested KVM. It
validates the bounded handoff, installs its own GDT and 256-entry fail-stop IDT,
retains the bounded normalized regions in fixed storage, constructs the common
validated early-kernel context, uses launcher-provisioned guarded privilege
stacks and guarded page tables, and renders a GOP framebuffer
marker. Before its first framebuffer write, the loader validates the complete
GOP geometry, page-aligned physical base, non-overflowing allocation, and
Rust raw-slice length bound through the same core validator used by the kernel.
Before installing the guarded address space, the loader erases the
entire page-rounded kernel allocation and all 128 KiB of the kernel stack arena,
so no firmware-era tail or stack data enters the kernel mapping. The arena has
a six-page early stack, a sixteen-page ring-transition stack, and an eight-page
double-fault IST stack. Pages below the arena and between each stack class are
deliberately absent from the new page tables, so underflow and cross-stack
overflow fault instead of reaching another stack. KVM and WHP use the same
fixed layout, and isolated service roots map the protected stacks
supervisor-only while omitting both internal guard pages. The
canonical handoff occupies its own compile-time-checked 4 KiB-aligned page, so
making it read-only cannot expose neighboring loader statics. The loader owns
and erases every byte of that page immediately before encoding the bounded
handoff, preventing uninitialized tail data from crossing the boundary. The
kernel now has a validated x86_64 trap-frame and user-fault termination policy,
32 vector-specific assembly entries, and a live signed nested-KVM invalid-opcode
dispatch proof. A task runtime now owns scheduler identity, saved context, and
task-local capability space as one revocation domain. The live CPL3 probe proves
a checked user fault removes that complete domain and retires the current
generational task. The same signed run then selects a second task, restores its
validated context through CR3/`iretq`, and observes that replacement's distinct
CPL3 breakpoint before reaching idle. This proves exception-path replacement
restoration with a shared diagnostic address space. Separate production-style
service address spaces are exercised by the live service probe below.
The architecture layer now has a service-address-space constructor that accepts
only an already verified `ServiceImage`. It creates a fresh mapping policy with
explicit supervisor-only higher-half kernel mappings, user PE sections with
their final W^X permissions, one user RW/NX stack, and an absent lower guard
page, then materializes those mappings into a newly allocated page-table root.
The KVM path verifies the kernel and service with separate
one-time keys, materializes a fresh service root, retains only supervisor kernel
PE mappings plus the service PE and guarded stack, switches CR3, and enters the
service at CPL3. A separately built 2 KiB service PE raises its signed breakpoint
and returns through the kernel IDT successfully. The Windows WHP path now uses
the same artifact and mapping contract, disables its test-only breakpoint
intercept for this partition, and completes the same guest-IDT proof. The
current probe materializes the authenticated service PE twice under distinct
roots (`0xc00000` and `0xd00000`) with different physical image, stack, and
table arenas and no shared user mappings. Receiver A issues pointer-free
receive and blocks. The scheduler switches CR3 to sender B; B sends `ping`
using its exact SIGNAL capability, wakes A, then exits. The kernel atomically
removes B's complete context/capability domain before selecting A, dequeues
the message into A's saved registers and restores A's post-`INT 0x80` context
under A's root. Fresh signed runs completed this chain in 301 microseconds on
WHP and 404 microseconds on nested KVM.
Service task ownership is now recorded by a fixed-capacity generational
supervisor. The signed clean-exit path resolves the current task through that
owner before revocation. A stopped or faulted instance can be restarted only
with exact `CONTROL` authority for its service object and a freshly supplied
validated context; restart advances the service generation, so old management
handles cannot control the replacement. Tests on Windows and Linux cover wrong
object authority, clean exit, fault retirement, restart, and stale identity
rejection. The WHP and KVM lifecycle managers now retain a private record of
each verified service instance. After the kernel has retired the task and the
vCPU has stopped, reprovision accepts only the same `ServiceImage` digest and
image size, withdraws the published entry/root, erases the complete image and
guarded-stack backing, rematerializes the verified PE, and only then republishes
the instance. A different validly signed executable is rejected before
publication changes. Live signed Windows WHP and nested-KVM runs exercise
rejection, deliberate stack contamination, complete reset, and clean
republication. The signed service probe now connects that primitive to the
kernel supervisor: both retired instances receive fresh service and task
generations through exact `CONTROL` capabilities, receive new task-local IPC
authority, and repeat the complete block/send/exit/fault path from rebuilt
memory. Checked lifecycle markers expose each transition to the live WHP and
KVM runners. Each supervisor registration now carries a validated maximum
restart count and bounded exponential backoff in scheduler ticks. Unauthorized
callers are rejected before learning budget/timing state; early retries do not
consume a generation, and retirement-time overflow fails after revocation. The
signed two-service policy permits exactly one immediate restart and proves a
second restart is denied. Policy parsing and a general management service remain
outside this fixed live proof.
Kernel task domains now contain a two-message, allocation-free inbox. Receiving
from an empty inbox blocks only the current task and immediately selects a
replacement; capability-authorized delivery enqueues in FIFO order and wakes
the receiver. A full inbox is rejected before endpoint sequence or capability
state changes, so overflow cannot grant authority or create a replay gap. This
is tested on Windows and Linux and exercised by the two-root live service flow.
Both instances use one independently signed service artifact; they are separate
task/address-space instances, not two differently signed binaries.
Task-to-task IPC is now routed through those runtime domains. It rejects
self-routing, requires the sender's exact endpoint capability, attenuates every
transferred right, and transactionally revokes all receiver capabilities if
endpoint authorization fails. This path is covered on Windows and Linux but is
invoked by the live sender's pointer-free user syscall.
The x86 syscall boundary now reserves only DPL3 interrupt vector `0x80` and
defines a pointer-free register ABI. Yield, receive, and exit require every reserved
register to be zero; inline send carries generational endpoint/task tokens and at most 24
payload bytes by value in registers. Unknown operations, malformed tokens, and
oversized payloads fail before any user address can be dereferenced. The call
gate now has a live assembly dispatcher. A signed nested-KVM run enters vector
`0x80` from CPL3, validates the complete return frame and RFLAGS whitelist,
authorizes a four-byte `ping` through the sender's endpoint capability, creates
the receiver message, returns status and sequence through registers, restores
all user registers, and executes `iretq`. The resumed task then faults and the
existing recovery path restores its replacement. The two-root service probe
additionally proves blocking receive, wakeup, cross-root scheduling, and
syscall-visible message delivery.
Bounded timer-driven scheduler policy and faulted-task retirement are
implemented. A signed nested-KVM probe now proves the booted kernel enables its
local APIC, programs a periodic timer, observes the hardware counter wrap,
accepts vector 32 only after enabling interrupts, advances its scheduler by
exactly one tick, and acknowledges EOI. The timer path uses a KVM in-kernel
interrupt controller and a supervisor-only xAPIC mapping when x2APIC is not
available; it does not use host interrupt injection. The APIC-specific KVM
configuration is confined to timer guests so ordinary proof guests preserve
their host-visible halt behavior. The latest signed run completed in 1,856
microseconds (`verify=744us`, `prepare=10296us`, `total=15834us`) under nested
KVM. The corresponding live WHP timer proof is also complete: the freshly
signed identical kernel reached the scheduler
tick under WHP in 8,232 microseconds (`verify=758us`, `prepare=2971us`,
`total=13998us`). A separate signed `preemption-probe` now enters task A at
CPL3, receives vector 32 with `CS=0x23`, reconstructs and binds A's complete
interrupted context to its current CR3, advances the scheduler quantum, selects
task B, acknowledges EOI, and restores B through `iretq`. B's checked CPL3
breakpoint proves execution without granting user I/O rights. The finalized
artifact completed in 2,073 microseconds on nested KVM and 8,315 microseconds
on WHP. The signed `service-preemption-probe` repeats the complete path across
independently materialized service roots: A runs under `CR3=0xc00000`, its
validated frame is saved only into A's domain, and B is restored under
`CR3=0xd00000`. Each root contains the supervisor kernel plus only its own user
image and guarded stack; timer-enabled roots additionally receive the
supervisor-only APIC page. The final cross-root proof completed in 2,063
microseconds on nested KVM and 8,413 microseconds on WHP. Validated generational ring-three context
storage is implemented. The live image now installs ring-three code/data
descriptors plus a validated 64-bit TSS, loads `TR`, disables its I/O bitmap,
supplies `RSP0`, and routes double fault through a dedicated IST stack. A
freshly signed nested-KVM exception probe ran successfully after this setup.
The launcher-owned transition and double-fault stacks have absent lower guard
pages and no static image fallback. The first guarded service IPC run exposed
an otherwise silent transition-stack overflow at `CR2=...60006d88`; the fixed
arena now provides 64 KiB for syscall/interrupt entry and 32 KiB for IST1. A
separate signed diagnostic build performs a live
`iretq` transition to CPL3, executes an invalid opcode, returns through TSS
`RSP0`, validates the privilege-transition frame, and reaches user-task
termination policy under nested KVM. That proof temporarily maps the entire
diagnostic PE in the lower half with user permissions and is therefore not an
acceptable service isolation design. Separate signed user mappings and
distinct CR3 roots are now exercised by both the service IPC and timer probes.
Privilege-stack allocation is now CPU-indexed for 1--256 x86-64 CPUs. A checked
stride reserves one complete 128 KiB guarded arena per CPU in both physical and
virtual address spaces; invalid counts, short strides, arithmetic wrap,
addresses beyond x86-64's 52-bit physical limit, noncanonical virtual coverage,
and out-of-range CPU indices fail before mapping. CPU 0
now owns its GDT, complete IDT, TSS, RSP0/IST1, and transition-stack pointer in
one aligned non-copyable `CpuDescriptorState` rather than unrelated globals.
WHP, KVM, and UEFI launch paths all obtain CPU 0 through the common allocator.
The architecture layer now parses a complete, loader-copied ACPI MADT into a
bounded topology of at most 256 enabled legacy APIC or x2APIC CPUs. It verifies
the table signature, exact length, checksum, entry sizes, reserved flags,
unique APIC and firmware identities, and a single aligned local-APIC override
before admitting the topology. A generational AP-startup state machine then
enforces `offline -> INIT sent -> SIPI sent -> online`, rejects acknowledgements
from the wrong APIC, and permanently invalidates stale attempts after failure.
Bounded ACPI discovery now validates both RSDP checksums, chooses XSDT or RSDT,
checks the complete root and candidate MADT, rejects missing or duplicate MADTs,
and copies the unique table into caller-owned memory with an independent final
checksum. The canonical handoff can now append that snapshot with an explicit
flag and bounded length; both encoder and kernel decoder recheck its APIC
signature, exact internal length, checksum, and whole-message length while
legacy handoffs remain canonical. The UEFI loader now traverses only ranges
admitted as ACPI by the final post-`ExitBootServices` memory map, emits the
snapshot from loader-owned storage, and maps the enlarged handoff read-only.
Kernel entry parses the snapshot into the bounded topology before continuing.
The local-APIC layer now builds exact directed INIT and SIPI ICR commands for
xAPIC and x2APIC, rejects broadcast/reserved destinations and a zero trampoline
vector, and uses bounded pre/post publication polling so a stuck controller
fails rather than hanging forever. Startup timing now requires an invariant TSC
with an architectural CPUID frequency; it provides the 10 ms post-INIT and
200 microsecond post-SIPI waits and refuses guessed clocks. A bounded 4 KiB
trampoline image now performs the real-mode, PAE, long-mode, CPU-index, private
stack, and kernel-entry transition. It accepts only a 4 KiB-aligned SIPI page
below 1 MiB, a 32-bit identity-mapped CR3, and canonical aligned destinations,
and the lifecycle binds the exact image page to its SIPI vector. Installation
requires an initially read/write and NX page, copies the image, changes it to
read/execute and non-writable, verifies those permissions, and revokes and
zeroes the page on every failed write, protection, or verification step. Only
the opaque installed-page result can advance the lifecycle to SIPI. WHP now
creates a dedicated low GPA page, writes it while guest-RW/NX, atomically
replaces the mapping with guest-RX, and verifies host reads still match while
guest writes are denied. This passed against the live Windows hypervisor.
For in-kernel sealing, the page-table layer now preflights an entire existing
mapping against its exact physical frames and old permissions before replacing
leaf permissions; missing, aliased, huge, or already-modified entries fail
before the first logical update. A dedicated low-supervisor RX permission keeps
the SIPI page executable without making it user-accessible or writable.
The SMP handoff extension carries an indivisible pair containing the exact
firmware-selected low page and the bounded AP privilege-stack arena; it rejects
absent, unaligned, zero-vector, one-megabyte-and-above trampoline addresses,
and invalid physical stack bases. UEFI reserves one loader-owned low page with
`AllocateMaxAddress`, zeroes it, maps it supervisor-RW/NX, and provisions all
256 bounded CPU slots with disjoint early, entry, and double-fault stacks while
leaving both internal guard pages in every slot unmapped. It identity-maps every
page-table frame actually allocated
(including frames allocated while adding those mappings), so the relocated BSP
can perform the checked seal. An `ActivePageTables` view opens CR3 without
allocating, applies the same exact-match transition through identity-mapped
frames, preserves hardware Accessed/Dirty state, and invalidates every affected
TLB entry before publication. The in-kernel adapter now inspects the actual
active leaf, accepts only the
exact identity-mapped supervisor RW/NX page, copies only in that state, seals
it RX through the checked transition, and on a failed installation reverses
RX, zeroes the page, unmaps the exact leaf, invalidates it, and verifies that
the mapping is absent. After an acknowledged AP has left the page, the opaque
installed token can also transition that exact RX mapping back to RW/NX and
zero it before binding the same scarce low page to the next AP. WHP exercises
the complete seal, denied-write, zeroed-rearm, and restored-staging sequence
against the live hypervisor. The PE kernel's BSP now launches each discovered
AP through directed INIT/SIPI, a sealed low trampoline, and a generational
acknowledgement. Each AP receives its CPU index, startup generation, and private
stack base, enables NX before adopting the shared CR3, installs a CPU-private
GDT/IDT/TSS with guarded entry and double-fault stacks, and only then publishes
online. The BSP zeroes and unmaps the trampoline after the final
acknowledgement. A freshly signed two-vCPU QEMU 11.1 TCG/UEFI image completed
this sequence on Windows; both CPUs halted in kernel-owned code with distinct
descriptor tables and the same CR3. A freshly signed two-vCPU nested-KVM image
also completed the sequence on Arch Linux under WSL2. The VMM gives each vCPU
a normalized APIC identity, holds the AP in KVM's architectural uninitialized
state until INIT/SIPI makes it runnable, maps the authenticated low trampoline
and identity-mapped privilege-stack arena, and checks independent terminal
proofs from CPU 0 and CPU 1. The measured successful run spent 914 microseconds
verifying, 11,061 microseconds preparing the VM, and 4,181 microseconds executing
the guest. The Windows WHP backend now proves the same two-vCPU sequence while
also enforcing second-level W^X: it switches the installed trampoline GPA from
RW/NX to RX before VP1 runs, uses a pre-acknowledgement exit to restore RW/NX,
then permits the BSP to zero and revoke the page. The trampoline GDT descriptors
are pre-accessed so segment loading never attempts a hardware write to RX code.
The signed WHP run completed in 64,164 microseconds. A separate signed
`smp-scheduler-probe` now runs on both KVM and WHP: CPU 1 owns a disjoint bounded
scheduler, programs its own local-APIC timer, receives vector 32 through its
private IDT and privilege stack, advances exactly one local tick, acknowledges
EOI, and emits a CPU-indexed proof while CPU 0 independently completes startup
and trampoline revocation. The measured runs completed in 52,565 microseconds
on nested KVM and 63,092 microseconds on WHP. Cross-CPU task migration,
load balancing, and interprocessor reschedule interrupts remain unfinished.
Platform-backed
writable-memory reprovisioning and one bounded supervised restart are part of
the live WHP/KVM proof. Clean user-requested
service exit, generational ownership, and domain revocation are now
part of the signed two-root IPC proof. The live
probe now uses the reusable context transition that writes its validated CR3,
sanitizes DS/ES/FS/GS, restores all fifteen general registers, and constructs
the exact ring-three `iretq` frame rather than probe-specific entry assembly.
Service VMs and bare-metal validation also remain unfinished.
The successful `ExitBootServices` call is a one-way boundary: memory-map
normalization, handoff construction, and launch failures after it halt locally
and can never return into terminated firmware services.
Under the mandatory compromised-host threat model, the current KVM and WHP
launchers are test harnesses, not security boundaries. Their host can inspect
or modify guest RAM and the entropy-derived queue key. Authenticated queues
still test canonical protocol enforcement and isolate ordinary peer guests,
but they cannot authenticate a host that owns both endpoints.

The KVM path now has a live, signed in-VM CUDA benchmark. A kernel image built
with `gpu-benchmark` publishes one authenticated, pointer-free `BenchmarkAdd`
request from inside the booted guest, rings port `0x4d52`, waits on its
read-only completion page, authenticates the returned GPU duration, paints a
distinct success marker, and halts. `mrml-kvm-run` admits only the exact
doorbell and bounded request, executes the repository's Rust-PTX add kernel,
checks every returned element, publishes an authenticated completion, resumes
the vCPU, and verifies the guest marker. On an RTX 5070 Ti exposed to Arch Linux
under WSL2, the 2026-08-21 release run processed 4,194,304 f32 elements for
1,000 iterations in 26,499,503 ns: 26.500 us per launch and 1,899.34 GB/s of
effective traffic. This is a functional/performance result under a compromised
host, not evidence that the host cannot forge the result.

The Windows WHP path now completes the same signed, booted-guest round trip.
`mrml-whp-run` maps the framebuffer and distinct command/completion pages before
vCPU entry, accepts only the fixed authenticated request doorbell, runs the
Rust-PTX kernel through the Windows CUDA driver, and resumes the guest. The
guest authenticates the completion, paints the green marker, and emits a second
fixed success doorbell so the VMM never depends on host-specific `HLT` wakeup
semantics. A 2026-08-21 release run on the same RTX 5070 Ti processed 4,194,304
f32 elements for 1,000 iterations in 28,429,300 ns: 28.429 us per launch and
1,770.41 GB/s effective traffic. CUDA execution is mediated by the host service;
the guest does not own or directly pass through the GPU. The compromised host
can still forge memory, timings, or responses, so this is a functional and
performance benchmark, not a hostile-host security proof.

Reproduce the KVM benchmark after generating a one-use release key and signing
the benchmark PE as described in `docs/MICROKERNEL.md`. Export the exact PTX
embedded in the final runner, sign that file as `cuda` with a separate one-use
key, and supply both independently signed artifacts:

```bash
cargo run --release -p mrml-kvm-run -- --export-cuda-bundle target/cuda.ptx
target/release/mrml-sign keygen target/cuda.private target/cuda.public
target/release/mrml-sign sign-bundle cuda 1 target/cuda.ptx \
  target/cuda.private target/cuda.signed
cargo run --release -p mrml-kvm-run -- \
  target/gpu-bench.signed target/gpu-bench.public 1 gpu-benchmark \
  target/cuda.signed target/cuda.public 1
```

On Windows, build the PE with `whp-gpu-benchmark`, sign it with a fresh one-use
release key, build `mrml-whp-run` with the GNU/LLVM Rust toolchain, then export
and independently sign that binary's embedded PTX before running:

```text
target\release\mrml-whp-run.exe --export-cuda-bundle target\cuda.ptx
target\release\mrml-sign.exe keygen target\cuda.private target\cuda.public
target\release\mrml-sign.exe sign-bundle cuda 1 target\cuda.ptx target\cuda.private target\cuda.signed
target\release\mrml-whp-run.exe KERNEL.signed RELEASE.public 1 target\cuda.signed target\cuda.public 1
```

Both launchers verify the CUDA artifact's kind, signature, minimum version, and
constant-time digest equality with their embedded PTX before creating the VM or
CUDA device. A correctly signed artifact containing different bytes is rejected.
Rebuilding either runner requires exporting and signing its embedded PTX again.

### Secure mediated CUDA design

MRML's preferred VM accelerator path is an inference-specific paravirtual
device, not forwarding NVIDIA ioctls or exposing GPU MMIO to an untrusted VM.
An isolated GPU service owns the real CUDA context and keeps model weights and
KV cache resident. Guests submit coarse tensor operations in batches so GPU
execution dominates transport cost and VM exits are amortized. The contract is:

- Buffers use opaque generational IDs with fixed per-session byte quotas. Wire
  messages never contain host pointers or guest-selected device addresses.
- Only kernel IDs from the measured, release-signed MRML CUDA bundle are
  dispatchable. Arbitrary PTX, runtime compilation, firmware operations, and
  raw CUDA driver calls are rejected.
- Every access includes a checked buffer range and mode. Grid, block, shared
  memory, argument count, request lifetime, and concurrent work are bounded.
- Queue messages have one fixed canonical encoding, an independent session
  identity, a monotonic sequence, and an HMAC tag. Replays, mutation, cross-VM
  use, malformed padding, stale handles, and duplicate requests fail closed.
- Commands are copied into a bounded kernel-owned FIFO before consumption, so
  an untrusted shared-memory producer cannot change admitted bytes. Full queues
  apply backpressure and never overwrite unread work; consumed slots are erased.
- Dispatch IDs are generational and protected by deadlines. Timeout handling
  invalidates the ID before reset or recovery begins, preventing stale
  completion from affecting a reused slot.
- The eventual GPU service must use IOMMU-confined pinned/shared pages, deny
  peer-to-peer DMA by default, validate the signed CUDA bundle before creating
  a context, and expose completion through capability-authorized interrupts.

Implemented today are the core-only buffer/session policy, canonical resource
and dispatch encodings, authenticated sender/receiver state, bounded FIFO,
monotonic producer/consumer ownership state, cache-line-separated atomic
publication indices with acquire/release ordering, embedded-kernel allowlist,
launch validation, an independently authenticated completion protocol bound to
generational dispatch IDs, bounded ordered batches of up to 32 prevalidated
dispatches, and a separate generational control-buffer namespace capped at
64 KiB whose shared bytes are SHA3-512 sealed and rechecked on use. The queue
has a canonical `SubmitBatch` command; device buffers cannot be substituted
for control descriptors. Batches have an exact-length pointer-free `MRGB`
encoding containing up to 32 ordered canonical dispatches and are fully
resource-revalidated after decoding. Whole batches are admitted to the
watchdog transactionally: every dispatch receives a generational identity or
all partial admissions are invalidated before service handoff. The executor
contract distinguishes acceptance, definite pre-accept rejection, and
uncertain failure; uncertain work remains tracked until timeout/reset. A
launch-enabling token can only be minted from a verified `CudaKernelBundle`
whose digest exactly matches the PTX embedded in the service. Wrong artifact
types or changed PTX fail closed. The native CUDA runtime now exposes that
compiled PTX only as a read-only byte slice for hashing and exposes the exact
28-entry kernel registry only through bounded numeric lookup. It does not add
an arbitrary module loader or guest-controlled symbol lookup. A fail-closed
executor ABI validator is implemented for `add_f32`: it requires exactly three
equal, nonempty, four-byte-aligned ranges with read/read/write rights, fixed
launch geometry, and no dynamic shared memory. The other kernel schemas remain
disabled until specified. Dispatch wire version 2 carries at most 16 explicitly
typed 32-bit scalar arguments (`u32`, `i32`, or IEEE-754 bits); unused slots and
per-scalar reserved bytes must be zero, and version-1 records are rejected.
This provides bounded scalar transport without pointers or variable blobs.
Kernel 0 (`gemm_q4_0_f32`) is enabled only when its positive signed
rows/columns/batch scalars exactly match Q4_0 weight bytes, f32 input/output
bytes, alignment, permissions, and fixed tiled launch geometry.
Decode kernels 3 and 4 apply the same binding to Q4_0 and Q8_0 GEMV while
enforcing their distinct 18-byte and 34-byte quantized blocks and launch tiles.
SwiGLU and GeGLU reuse the exact three-f32-buffer elementwise proof. The f32
embedding schema additionally proves that the signed nonnegative token is
inside the table row count and that its dimension equals the exact output row.
The Q8_0 embedding variant additionally requires dimensions divisible by 32,
34-byte quantized blocks, and a finite positive output scale.
MoE router logits bind expert, model, and batch dimensions to exact dense
weight/input/output ranges. Top-8 selection requires at least eight experts and
exactly sized writable i32 ID and f32 probability outputs.
Fused QKV GEMM and GEMV bind three separately sized Q4_0 matrices to one shared
column width and one packed Q/K/V output, with distinct batch and launch tiling.
Fused GeGLU GEMM and GEMV similarly require exact gate and up Q4_0 matrices,
shared dimensions, f32 input/output tensors, and decode/prefill launch tiling.
Vocabulary top-k binds logits, validity bytes, recent-token history, candidate
scores/IDs, and partition scalars, and selects the specialized or generic
kernel exclusively from the proven maximum partition size.
RMS normalization represents an absent optional weight structurally rather
than with a guest pointer, binds dimensions to exact tensor lengths, and
requires epsilon to be a finite positive f32.
RoPE binds its nonnegative position, even head dimension, head count, positive
finite frequency parameters, exact in-place buffer size, and per-head launch
geometry; the buffer must be explicitly read/write.
QKV post-processing binds head grouping, positions, normalization vectors, and
the packed in-place QKV tensor to exact dimensions. Its K and V cache ranges
independently permit only F16, Q8_0, or Q4_0 layouts, with checked capacity and
format-specific byte calculations; signed position overflow is rejected.
The shared-memory F16 attention schema independently derives the least query
span, full cache ranges, output tensor, effective key count, and dynamic shared
memory. It requires a power-of-two cache, grouped-query head divisibility, and
the exact specialized launch conditions; quantized or oversized attention is
kept on the still-fail-closed streaming path.
The streaming attention schema accepts independently selected F16, Q8_0, and
Q4_0 caches with their exact layouts and no dynamic shared memory. Variant
selection is exclusive: a dispatch eligible for bounded shared F16 attention
cannot be relabeled as streaming. The mediated interface currently keeps the
streaming subset power-of-two and even-width for conservative cache indexing.
FFN preparation binds two full input tensors, four normalization vectors, and
four separate outputs to the same checked dimension and batch. FFN completion
requires the MoE accumulator to be read/write, keeps the dense branch
read-only, validates three normalization vectors and the output, and rejects a
non-finite layer scale.
MoE expert selections now have a bounded sealed proof type. It validates exact
i32 ID and f32 weight ranges, rejects negative/out-of-range or duplicate IDs,
limits active experts to 32, requires finite weights in `[0, 1]` with positive
per-token mass, and binds the checked bytes and dimensions with domain-separated
SHA3-512. The service must reverify that digest immediately before a
data-dependent launch, detecting mutation after admission.
All five MoE projection schemas now require that retained proof through a
distinct `ValidatedMoeKernelLaunch`. Generic batched, single-token combined,
and fixed Gemma 4 26B variants exclusively bind exact Q4 storage, activation
tensors, scales, outputs, and geometry. Shape-only validation still rejects
IDs 22–26, preventing callers from discarding expert provenance.
Mixed validated batches now carry a per-entry optional expert proof. Ordinary
batch admission still fails closed on MoE IDs, while contextual admission
matches each MoE dispatch to a supplied sealed selection and preserves that
selection alongside the launch for service-side reverification.
`MediatedGpuExecutor` now performs that reverification as an allocation-free
two-phase submission. It preflights every MoE proof before making any launch
GPU-visible, rejects mutation with zero launches, then submits validated entries
in their original order through a narrow trusted backend. Backend failure after
the launch phase starts remains uncertain and uses watchdog/reset recovery.
The CUDA runtime implements that backend with a fixed-capacity generational
binding table. It rejects stale IDs and range overflow, rereads sealed MoE
controls from device memory, and lowers resolved addresses plus typed scalar
bits to the exact embedded ID-to-symbol entry with validated launch geometry.
Unsafe raw binding requires retained allocation ownership; device addresses
never enter the guest protocol.
`MediatedCudaService` synchronizes virtual quota allocation with owned CUDA
memory transactionally: CUDA allocation failure rolls back the guest handle,
free requires the matching owned generation before releasing quota, externally
owned raw bindings cannot be freed through the owned path, and drop releases
any remaining owned allocations.
Mediated submission now synchronizes the CUDA stream after the ordered launch
pass and reports acceptance only after that barrier succeeds. A synchronization
error is not treated as rejection: watchdog identities remain live because GPU
effects may have occurred and reset recovery is required.
The complete submission path preflights watchdog identities and authenticated
completion-sequence capacity before execution. After the barrier succeeds it
retires each generational identity and emits one ordered, session-bound success
frame; rejection cancels the batch, while uncertain execution retains it for
timeout or reset recovery.
Watchdog expiry and device reset likewise produce ordered authenticated status
frames while invalidating the corresponding generational IDs. Completion
sequence exhaustion is checked before mutation, so recovery cannot silently
retire work that the guest was not told about.
The executor trait accepts only a `ValidatedGpuBatch`. That type can be created
only by combining watchdog-bound identities, the verified embedded-bundle
token, and successful ABI validation of every dispatch; mixed batches reject
atomically before service submission.
Also implemented are the dispatch watchdog and
adversarial unit tests. The cross-VM
queue layout is also implemented: command and completion rings occupy separate,
page-aligned, overflow-checked physical ranges sized from a bounded slot count.
KVM and Hyper-V/WHP now attach both ranges independently, make the completion
ring guest-read-only, and have live backend regressions that preserve verified
guest execution. The kernel-owned completion transport preflights capacity for
the whole batch before GPU-visible work, publishes authenticated results under
an exclusive producer borrow, and erases consumed slots.
A common VMM queue bridge now copies guest-published command slots into
kernel-owned storage and publishes service completions through the protected
ring using the `VmBackend` contract shared by KVM and WHP. It enforces monotonic
slot ownership and permanently poisons itself after uncertain backend I/O, so a
VM reset is required instead of risking duplicate GPU work.
The KVM GPU-guest constructor now installs both rings in the initial kernel
page tables as identity-mapped NX memory: commands are writable and
completions are read-only. A separate `GpuVmmMemory` authority lets only the
isolated host service update read-only completion backing; ordinary guest-write
APIs continue to reject that operation.
WHP now provides the same initial queue mapping and an independently inspected
x86-64 page walk. Its live Hyper-V test proves command writable/NX and
completion read-only/NX translations, preserves guest write denial, and proves
that the isolated service authority can still publish completion bytes.
Queue authentication no longer needs a reusable benchmark secret: a
domain-separated SHA3-512 derivation produces the session ID and 256-bit key
from the authenticated, per-launch boot entropy, and rejects absent entropy.
The guest-side producer copies a complete authenticated command before its
release publication and permanently poisons itself if shared counters are
forged. A fixed `0x4d52` doorbell and canonical bounded `BenchmarkAdd` request
carry only element and iteration counts; pointers, symbols, PTX, and
guest-selected launch geometry cannot be encoded.
The `SubmitBatch` service path now joins sealed-control revalidation, canonical
decode, current buffer-generation checks, transactional watchdog admission,
signed schema validation, execution, and completion publication in one API.
The resource service consumes one kernel-owned ring slot, authenticates it,
executes allocation/free through the transactional CUDA resource backend, or
returns a typed sealed-batch request. Invalid authenticated input is consumed
without changing resource state, preventing a poisoned head from wedging the
queue. Successful allocation/free commands now produce a distinct `MRGR`
response authenticated under its own HMAC domain, session, and monotonic
sequence. Response-ring and sequence capacity are checked before command
consumption, so resource state cannot change without publication capacity.
Backend allocation/free failures emit authenticated rejection frames carrying
only the original request identity; backend details remain host-local and
transactional backends leave resource state unchanged. Platform cache-coherence
validation, CUDA graph lowering, IOMMU plumbing, and end-to-end performance
measurements are still pending. Consequently MRML
does not yet claim passthrough-equivalent VM CUDA performance. The design aims
to approach it for long-running LLM inference by avoiding copies, per-kernel VM
exits, and repeated context setup; arbitrary CUDA applications are out of scope.

MRML is under active development and currently specializes in
`ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`.

Working today:

- Interactive and one-shot terminal clients.
- Versioned JSONL interface for automation and regression tests.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` HTTPS/SSE server.
- Native authenticated TLS 1.3 with hybrid X25519MLKEM768 key agreement.
- Native streaming HTTPS model downloads with resumable SHA3-512 integrity sidecars.
- Memory-mapped GGUF access with native Q4_0/Q8_0 CPU operations.
- NVIDIA CUDA decode and batched prompt prefill.
- F16, Q8_0, and Q4_0 KV-cache storage.
- General, coder, and automatic agent modes.
- Workspace, Git, shell, browser, desktop, media, and stdio MCP tools.
- Conversation state reuse and automatic compaction.

Important limitations:

- CUDA is the only native GPU backend. Vulkan, ROCm, and SYCL choices are
  placeholders and do not execute inference.
- GPU execution currently targets device 0. Partial layer offload, multi-GPU
  execution, and the `--n-gpu-layers` control are not implemented end to end.
- A CUDA build tries to make the complete model resident. Insufficient free
  VRAM can disable the fastest resident path and cause a large slowdown.
- CPU inference supports a much narrower set of quantizations and instruction
  sets than llama.cpp and is primarily a correctness/fallback path.
- Batched CUDA prefill requires the complete prompt to fit in the GPU KV cache.
  If it cannot, the current fallback evaluates only the most recent 32 tokens.
  Do not interpret the 256K cache-sizing limit as verified full-prompt 256K
  inference.
- Q4 KV cache saves VRAM but is slower than F16 on the current 8K CUDA
  workload. `auto` therefore selects F16 below 128K context.
- N-gram proposal code exists, but generation does not yet load a draft/MTP
  model or perform batched speculative verification.

## Quick start

MRML uses Rust 2024 and pins nightly in `rust-toolchain.toml`. On Windows, MRML
supports only Rust's native GNU/LLVM (`x86_64-pc-windows-gnullvm`) host. It does
not require or support the MSVC Rust target, Visual Studio, the MSVC compiler,
the Windows SDK, or a separate LLVM installation. Install rustup, the
self-contained Rust MinGW component, and the Rust CUDA PTX target:

```powershell
# Windows PowerShell
winget install Rustlang.Rustup
rustup toolchain install nightly-x86_64-pc-windows-gnullvm --profile minimal `
  --component rust-src --component rustfmt --component clippy `
  --component rust-mingw --target nvptx64-nvidia-cuda `
  --target x86_64-unknown-uefi
rustup default nightly-x86_64-pc-windows-gnullvm

rustc -vV
# host must print: x86_64-pc-windows-gnullvm
```

The workspace selects Rust's bundled `rust-lld` and statically links its MinGW
runtime. Do not install `LLVM.LLVM`, Visual Studio Build Tools, or the Windows
SDK for MRML. `rust-mingw` supplies the Windows import libraries required by
the GNU/LLVM target.

```bash
# Linux/WSL2 uses its normal native GNU host (not the Windows gnullvm target).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly --component rust-src,rustfmt,clippy
rustup target add nvptx64-nvidia-cuda --toolchain nightly
rustup target add x86_64-unknown-uefi --toolchain nightly
```

CUDA builds require only a compatible NVIDIA display driver plus the Rust
`nvptx64-nvidia-cuda` target installed above. They do **not** require the CUDA
Toolkit, CUDA compiler, or `libcudart`. MRML compiles its kernels directly to
PTX with nightly Rust and dynamically loads the driver's `nvcuda.dll` on Windows or
`libcuda.so.1` on Linux. In WSL2, install the current NVIDIA Windows driver;
Microsoft's WSL bridge exposes its CUDA driver to Linux. Do not install a Linux
NVIDIA display driver inside WSL2.

CPU-only CLI:

```powershell
cargo run --release -p mrml-cli --no-default-features -- `
  --model C:\path\to\model.gguf --prompt "Hello"
```

CUDA CLI:

```powershell
cargo run --release -p mrml-cli --features cuda -- `
  --model C:\path\to\model.gguf --ctx-size 8192 --prompt "Hello"
```

If `--model` is omitted, MRML looks for the default Gemma 4 Q4_0 model in its
Hugging Face cache and may offer to download it. The model is large; check disk
space, network use, and VRAM before accepting a download.

## Applications

| Crate | Binary | Purpose |
| --- | --- | --- |
| `mrml-cli` | `mrml-cli` | Interactive and one-shot terminal frontend |
| `mrml-git` | `mrml-git` | Dependency-free Git workspace client |
| `mrml-machine` | `mrml-machine` | Stable JSONL automation and benchmark frontend |
| `mrml-server` | `mrml-server` | OpenAI-compatible HTTPS/SSE server |
| `mrml-uefi` | `mrml-loader.efi` | Minimal original x86_64 UEFI boot stage |
| `mrml-trainer` | `mrml-trainer` | Wikipedia ZIM training and GGUF export |
| `mrml-agent` | — | Agent orchestration, configuration, rules, and model resolution |
| `mrml-model` | — | Application-facing model and streaming adapter |
| `mrml-tensor` | — | GGUF execution, tensor math, CPU kernels, and CUDA kernels |

Common commands:

```powershell
# Interactive terminal
cargo run --release -p mrml-cli --features cuda -- --model C:\path\to\model.gguf

# Show the current branch, HEAD, and staged/unstaged workspace pulse
cargo run --release -p mrml-git

# OpenAI-compatible HTTPS server. The PEM certificate must contain the full
# chain and the unencrypted key must be PKCS #8 or PKCS #1 RSA.
$env:MRML_TLS_CERT = "C:\path\to\fullchain.pem"
$env:MRML_TLS_KEY = "C:\path\to\private-key.pem"
$env:MRML_API_TOKEN = "replace-with-at-least-32-random-ascii-bytes"
cargo run --release -p mrml-server --features cuda -- `
  --model C:\path\to\model.gguf --port 8080
```

### MRML Git client

`mrml-git` is a small, original porcelain client built on MRML's native
runtime. Its default workspace-pulse view separates staged, unstaged, and
dual-lane changes and safely parses Git's NUL-delimited porcelain format.

```powershell
mrml-git status
mrml-git -C C:\path\to\another-repo status
mrml-git log 20
mrml-git diff --staged
mrml-git show HEAD~2
mrml-git stage src/main.rs
mrml-git unstage src/main.rs
mrml-git restore src/main.rs
mrml-git branch feature/name
mrml-git switch main
mrml-git commit "Describe the change"
mrml-git fetch origin
mrml-git pull origin main
mrml-git push origin main
mrml-git stash push "checkpoint"
```

SSH remotes and SSH-format Git signatures use the host Git and OpenSSH
processes. MRML validates remote structure before configuration, accepts SCP
style (`git@host:owner/repo.git`) and `ssh://` URLs, rejects embedded passwords,
and provides a read-only remote access check:

```powershell
mrml-git ssh add origin git@github.com:owner/repository.git
mrml-git ssh info origin
mrml-git ssh check origin
mrml-git signing configure C:\keys\id_ed25519 C:\keys\allowed_signers
mrml-git commit --sign "Signed change"
mrml-git tag-sign v1.0.0 "Signed release"
mrml-git signing verify HEAD
mrml-git signing verify-tag v1.0.0
```

Signing configuration is local to the repository. `signing auto` enables
automatic commit and tag signing, while `signing off` disables both. A
compromised host can read or misuse host keys, alter Git/OpenSSH, or forge the
displayed result; this client does not treat hosted signing as a bare-metal
MRML security boundary.

Mutating commands deliberately map to narrow Git operations: `stage` inserts
`--` before paths, `unstage` uses `restore --staged`, `branch <name>` creates
and switches in one operation, and `commit` requires an explicit message.
`pull` always uses `--ff-only`, `fetch` prunes deleted remote references, and
the dashboard reports the repository root, upstream, and ahead/behind counts.

The HTTPS server requires X25519MLKEM768 and rejects clients that do not offer
the standardized hybrid group. Its certificate handshake uses TLS 1.3
RSA-PSS/SHA-256. Model downloads prefer the same hybrid group, retain X25519
interoperability for HTTPS origins that have not deployed it, validate the
server certificate and hostname against the native trust store, and store a
SHA3-512 sidecar beside each completed model. A mismatched model or resume
checkpoint is rejected and downloaded again.
The server listens only on `127.0.0.1`, requires an exact
`Authorization: Bearer $MRML_API_TOKEN` header on every request, disables
cross-origin browser access, and marks responses as non-cacheable.

Run a binary with `--help` for its complete set of options. Backend and GPU
layer arguments are exposed for interface compatibility, but the limitations
above describe what the native engine currently honors.

## Machine interface

`mrml-machine` is the preferred non-interactive interface for regression tests
and external automation. Records use schema version 1 and the
`MRML_MACHINE_JSON=` prefix so structured results remain identifiable beside
CUDA diagnostics.

```powershell
cargo run --release -p mrml-machine --features cuda -- `
  --model C:\path\to\model.gguf chat --prompt "What time is it?"

cargo run --release -p mrml-machine --features cuda -- `
  --model C:\path\to\model.gguf session
```

Session operations are `chat`, `health`, `reset`, and `exit`. Chat results
include content, separated reasoning, tool events, finish reason, token count,
generation time, wall time, and generated tokens per second.

## Performance and benchmarking

Always build with `--release` when measuring inference. Debug builds are not
representative.

MRML reports generated-token throughput separately from end-to-end wall time.
Generation throughput excludes model loading and prompt initialization, so it
must not be compared with a benchmark that includes those phases. Prompt
processing and token generation are different workloads and should be reported
separately, as llama.cpp does with `pp` and `tg` tests.

CUDA device initialization eagerly loads the Rust PTX module and resolves every
kernel entry point before MRML reports the model ready. This moves PTX JIT and
CUDA lazy-loader work out of first-token latency without generating a synthetic
warm-up response or modifying model state. GPU clock selection remains under
the NVIDIA driver; MRML does not require privileged clock locking.

Enable per-token phase timing with:

```powershell
$env:MRML_PROFILE = "1"
cargo run --release -p mrml-machine --features cuda -- `
  --model C:\path\to\model.gguf chat --prompt "Benchmark prompt"
```

Run the deterministic kernel benchmarks with:

```powershell
# CPU hot paths
cargo bench -p mrml-tensor --bench hot_paths --no-default-features --features runtime

# CUDA correctness and prefill benchmark
cargo run --release -p mrml-tensor --example cuda_prefill_primitives --features cuda
```

After changing inference kernels, also run a deterministic end-to-end quality
smoke test. It prints the raw model text so channel markers, repetition, and
numerical mistakes cannot be hidden by the chat frontend:

```powershell
cargo run --release -p mrml-tensor --example prefill_model_check --features cuda -- `
  C:\path\to\model.gguf "What is 48 + 57? Give the number and a short explanation."
```

Use several fixed prompts covering factual recall, arithmetic, and coherent
prose, and save their exact outputs alongside performance results. Low
temperature makes regressions reproducible, but it does not prove strong
instruction following: validate exact-format requirements separately through
`mrml-machine`.

The CUDA example includes a 2816×2816, batch-128 Q4_0 projection matching the
default Gemma hidden width. On the development RTX 5070 Ti, token-tiled weight
reuse reduced the median time for that isolated projection from approximately
0.891 ms to 0.525 ms (about 1.70× throughput). This is a kernel microbenchmark,
not a claim about complete model tokens/second.

For a useful MRML-versus-llama.cpp comparison, keep all of the following equal:

- Exact GGUF file and quantization.
- Prompt tokens, generated tokens, and context depth.
- K/V cache formats.
- GPU layer residency and device selection.
- Warm-up policy and number of repetitions.
- Whether tokenization and sampling are included.
- Driver, CUDA version, clocks, power limit, and competing GPU workloads.

Report model load time, prompt tokens/second, generated tokens/second, wall
latency, and variation across runs. A single warmed generated-token number is
not enough to characterize the engine.

### Performance troubleshooting

1. Confirm the command uses `--release` and `--features cuda`.
2. Read the startup lines for the execution plan and fully resident layer count.
   The fastest path requires all transformer layers to be resident.
3. Check free VRAM and GPU utilization with `nvidia-smi`. Close other GPU-heavy
   applications before loading the model.
4. Start with `--ctx-size 8192 --cache-type-k f16 --cache-type-v f16`.
5. Set `MRML_PROFILE=1` to distinguish QKV, attention, output-projection, FFN,
   and sampling time.
6. Compare prompt processing and generation separately. Slow first-token time
   with normal generation usually points to prefill; declining generation speed
   at long context usually points to attention/KV traffic.
7. Use an identical local GGUF when comparing with llama.cpp. Repository labels
   such as `Q4_0` do not guarantee byte-identical model files.

## Testing

```powershell
cargo test --workspace --release --no-default-features
cargo test -p mrml-tensor --release --features cuda
```

CUDA tests require compatible NVIDIA hardware and a CUDA-enabled build.

## Wikipedia ZIM datasets

`mrml-zim`, `mrml-zstd`, and `mrml-wikipedia` provide a dependency-free,
`no_std` path for streaming Kiwix Wikipedia archives. The native decoder handles
modern Zstandard clusters, including Huffman literals, FSE sequences, repeated
entropy tables and overlapping match copies. Only the current decoded cluster
is cached, so a multi-gigabyte ZIM is never loaded into memory.

```rust
let mut articles = mrml_wikipedia::ArticleReader::open("wikipedia_en_all_nopic.zim")?;
while let Some(article) = articles.next_article()? {
    train_on_document(&article.title, &article.text)?;
}
```

To validate a local archive, set `MRML_TEST_ZIM` to its path. Add
`MRML_TEST_ZIM_ALL=1` to decode every compressed cluster rather than sampling
the beginning, middle, and end:

```powershell
$env:MRML_TEST_ZIM = "C:\path\to\wikipedia.zim"
$env:MRML_TEST_ZIM_ALL = "1"
cargo test --release -p mrml-wikipedia external_ -- --nocapture
```

Train a small from-scratch next-token research baseline from one substantive
article and export it as GGUF:

```powershell
cargo run --release -p mrml-trainer -- `
  --zim C:\path\to\wikipedia.zim `
  --output target\wikipedia-one-article.gguf `
  --article 0 --articles 1000 --vocab 384 `
  --steps 20000 --learning-rate 0.02 --prompt hello
```

The trainer reports extraction, tokenizer training, model construction, GGUF
export, validation, and total wall time separately. The resulting
`mrml_transformer` checkpoint is a compact causal decoder with token and
position embeddings, four-head causal attention, residual/RMS normalization,
GELU feed-forward layers, and a vocabulary head. Its first fast training stage
fits the output projection from corpus transition statistics while attention
and feed-forward weights use deterministic initialization. It is a research
bootstrap checkpoint. Contextual cross-entropy updates train the vocabulary
projection (`--steps` and `--learning-rate`); the remaining transformer tensors
are deterministic initializations pending full backpropagation support. It is
not yet a fully gradient-trained competitive LLM.

## Development roadmap

The staged [third-party Rust crate removal plan](docs/dependency-removal-plan.md)
defines the compatibility, quality, and inference-performance gates for moving
the workspace to repository-owned Rust code.

## License

MRML is dedicated to the public domain under
[CC0 1.0 Universal](LICENSE) (`CC0-1.0`). Anyone may use, study, modify, share,
and redistribute it for any purpose, including commercially. Where a
public-domain dedication is not legally possible, CC0 provides its broad
license and waiver fallback.
