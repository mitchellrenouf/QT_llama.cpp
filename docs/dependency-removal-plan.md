# Third-party Rust crate removal plan

> Status: complete. The lockfile contains only local `mrml-*` packages, and
> production, example, benchmark, CUDA, and build-script targets are `no_std`
> without Rust's global `alloc` crate. Test-only modules may use `std` through
> the Rust test harness to compare native behavior.

## Goal and boundary

The end state is a workspace whose manifests contain only local `mrml-*` path
dependencies. No package from crates.io may appear in `cargo metadata`. CUDA
and operating-system APIs may remain native build/runtime dependencies; their
Rust bindings must live in this repository.

This is a compatibility project, not permission to delete features. GGUF
inference, chat templates, JSON protocols, HTTP/SSE, WebSockets, MCP, browser
tools and the CLI must keep working unless a separate change explicitly
deprecates them.

## Non-negotiable gates

Each removal must pass the affected crate's tests and the workspace release
tests. Changes touching inference must additionally pass:

- CUDA numerical tests.
- The deterministic arithmetic and factual quality prompts documented in the
  README.
- Three 128-token full-residency runs with the local Gemma Q4_0 model.
- No statistically meaningful regression from the current approximately
  74 generated tokens/second control median on the development RTX 5070 Ti.

Protocol replacements need golden byte-for-byte fixtures. Parser replacements
need malformed-input, nesting-depth, size-limit, and Unicode tests. Unsafe FFI
replacements require focused ownership and error-path tests.

## Current dependency groups

| Group | Crates | Replacement direction |
|---|---|---|
| Errors | `anyhow`, `thiserror` | Workspace error enum, boxed source errors, local context helpers |
| Synchronization and parallelism | `parking_lot`, `crossbeam-channel`, `rayon` | `std::sync`, `std::sync::mpsc`, scoped worker pool |
| Files and platform paths | `memmap2`, `dirs`, `walkdir`, `chrono` | Small Windows/Unix FFI modules, iterative directory walk, `SystemTime` formatting |
| Build | `cc`, `cmake`, build-time `walkdir` | Direct `Command` invocation of NVCC/CMake/compiler plus rerun directives |
| Data model | `serde`, `serde_json` | Local bounded JSON value/parser/writer and explicit protocol codecs |
| Templates | `minijinja`, `minijinja-contrib` | Gemma chat-template interpreter limited to constructs present in supported GGUF files |
| Async/runtime | `tokio`, `futures-util`, `async-trait` | Threads, blocking sockets/processes, explicit state machines and channels |
| CLI/output | `clap`, `colored`, `termimad` | Local argument parser, ANSI helpers, Markdown renderer |
| Text/HTML | `regex`, `similar`, `scraper`, `urlencoding` | Purpose-built scanners, diff implementation, bounded HTML tokenizer, percent codec |
| Transport | `tokio-tungstenite`, `tungstenite` | Local RFC 6455 framing over the internal socket layer |
| Browser/media | `chromiumoxide`, `base64` | Local CDP client and base64 codec |

## Migration sequence

### 1. Establish enforcement and local foundations

Add a CI check that rejects registry dependencies in `cargo metadata`. Create
small local modules for errors, JSON, platform paths, time, base64, percent
encoding, and ANSI output. Initially validate them against the existing crates
with differential tests; remove the old crate only after callers migrate.

### 2. Make `mrml-tensor` dependency-free first

This protects the performance-critical core and provides a small proving
ground. Replace, in order:

1. `thiserror` and `anyhow` with concrete tensor/GGUF/CUDA errors.
2. `dirs` and `chrono` with the local platform/time modules.
3. `parking_lot` with poison-tolerant `std::sync` wrappers.
4. `crossbeam-channel` and the limited `tokio::mpsc` use with standard channels.
5. `rayon` with a persistent scoped CPU worker pool; benchmark CPU fallback
   before and after.
6. `memmap2` with a read-only native mapping abstraction and buffered-file
   fallback.
7. `cc` with a deterministic build script invoking NVCC and the host compiler.

Exit criterion: `mrml-tensor` has no registry dependencies in normal, test,
benchmark, or build dependency sections.

### 3. Replace shared serialization before network stacks

Implement a bounded JSON parser/writer supporting the exact value model used
by tool schemas, JSONL, OpenAI-compatible requests, and MCP.
Replace derive-generated serialization with explicit codecs. Preserve field
names, omitted/null behavior, number handling, escaping, and stable JSONL/SSE
output through golden fixtures.

Remove `serde` and `serde_json` workspace-wide only after every protocol has
golden compatibility coverage.

### 4. Replace model templating

Collect the chat templates from every supported GGUF fixture and define the
required language subset. Implement and fuzz a bounded interpreter for that
subset, including loops, conditionals, filters, whitespace control, and the
Python-compatibility behavior currently supplied by `minijinja-contrib`.

Do not replace templates with a hard-coded Gemma prompt: that would narrow
model compatibility and invalidate quality comparisons.

### 5. Remove the async ecosystem

Move inference streaming to standard bounded channels. Convert tool execution,
MCP child-process I/O and HTTP serving to owned worker threads
and blocking I/O with explicit cancellation. Replace trait macro expansion
with synchronous traits or boxed state-machine futures implemented locally.

Remove `tokio`, `futures-util`, and `async-trait` together only after shutdown,
backpressure, cancellation, and concurrent-client tests pass.

### 6. Replace user-interface and utility crates

Migrate argument parsing, color, Markdown, regex-specific scanners, directory
walking, diffs, URL encoding, HTML extraction, and base64. Prefer narrow
parsers for known formats over general-purpose reimplementations.

### 7. Replace transports and integrations

Implement strict WebSocket framing and handshake validation for the server
path. Implement the required Chrome DevTools Protocol subset over that layer.
The browser downloader should become an explicit external setup step or a
small platform download implementation with checksum verification.

### 8. Lock the workspace

Delete all registry dependency declarations, regenerate `Cargo.lock`, enforce
offline builds in CI, and audit the source tree for copied third-party code and
license obligations. Run the full functional, protocol, CUDA quality, and
performance matrix before declaring the migration complete.

## Recommended first implementation slice

Start with `mrml-tensor`: local error types, `std::sync` migration, and native
read-only file mapping. These changes are isolated from public JSON/network
protocols and expose performance regressions early. Do not begin with
`serde_json`, `tokio`, templates, or Chromium; those need dedicated
compatibility fixtures before implementation.
