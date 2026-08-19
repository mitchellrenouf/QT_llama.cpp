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

MRML is a self-contained local inference runtime. Its portable crates are being
migrated away from Rust's `std` and global `alloc` interfaces. This does not
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

## Status

MRML is under active development and currently specializes in
`ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`.

Working today:

- Interactive and one-shot terminal clients.
- Versioned JSONL interface for automation and regression tests.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` HTTP/SSE server.
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
  --component rust-mingw --target nvptx64-nvidia-cuda
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
| `mrml-machine` | `mrml-machine` | Stable JSONL automation and benchmark frontend |
| `mrml-server` | `mrml-server` | OpenAI-compatible HTTP/SSE server |
| `mrml-core` | — | Agent, configuration, tools, rules, and model resolution |
| `mrml-model` | — | Application-facing model and streaming adapter |
| `mrml-tensor` | — | GGUF execution, tensor math, CPU kernels, and CUDA kernels |

Common commands:

```powershell
# Interactive terminal
cargo run --release -p mrml-cli --features cuda -- --model C:\path\to\model.gguf

# OpenAI-compatible server
cargo run --release -p mrml-server --features cuda -- `
  --model C:\path\to\model.gguf --port 8080
```

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

Enable per-token phase timing with:

```powershell
$env:MRML_PROFILE = "1"
cargo run --release -p mrml-machine --features cuda -- `
  --model C:\path\to\model.gguf chat --prompt "Benchmark prompt"
```

Run the deterministic kernel benchmarks with:

```powershell
# CPU hot paths
cargo bench -p mrml-tensor --bench hot_paths --no-default-features

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
