# MRML

MRML—the **Mitchell Renouf Machine Learning Library**—is an experimental,
CC0-licensed Rust workspace for running Gemma 4 GGUF models locally. It combines
a native inference engine with terminal, desktop, machine-readable, and
OpenAI-compatible HTTP interfaces plus agent tools.

MRML does **not** link to llama.cpp. Its tensor operations, GGUF loader, CPU
kernels, CUDA kernels, KV cache, and generation loop live in this repository.
That makes MRML useful as a compact inference-engine project, but it does not
yet have llama.cpp's model coverage, hardware coverage, or years of kernel
tuning.

## Status

MRML is under active development and currently specializes in
`ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`.

Working today:

- Native Qt 6 desktop application on Windows and Linux.
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

You need stable Rust. CUDA builds additionally require an NVIDIA driver and
CUDA Toolkit 13.3. The desktop application also requires Qt 6.

CPU-only CLI:

```powershell
cargo run --release -p mrml-cli --no-default-features -- `
  --model C:\path\to\model.gguf --prompt "Hello"
```

CUDA CLI:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
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
| `mrml-qt` | `mrml-qt` | Native Qt 6 desktop frontend |
| `mrml-core` | — | Agent, configuration, tools, rules, and model resolution |
| `mrml-model` | — | Application-facing model and streaming adapter |
| `mrml-tensor` | — | GGUF execution, tensor math, CPU kernels, and CUDA kernels |

Common commands:

```powershell
# Interactive terminal
cargo run --release -p mrml-cli --features cuda -- --model C:\path\to\model.gguf

# Desktop application
cargo run --release -p mrml-qt --features cuda -- --model C:\path\to\model.gguf

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

## Building the desktop application

On Windows, install Visual Studio Build Tools with the Desktop development with
C++ workload, Rust MSVC, a Qt 6 `msvc2022_64` kit, and CUDA 13.3:

```powershell
$env:Path = "C:\Qt\6.8.3\msvc2022_64\bin;$env:Path"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
qmake -query QT_VERSION
nvcc --version
cargo build --release --locked -p mrml-qt --no-default-features --features cuda
```

On Linux, install Qt 6, a C++ toolchain, Rust, the NVIDIA driver, and CUDA 13.3:

```bash
export CUDA_PATH=/opt/cuda
export PATH="$CUDA_PATH/bin:$PATH"
qmake6 -query QT_VERSION
nvcc --version
cargo build --release --locked -p mrml-qt --no-default-features --features cuda
```

## Flatpak

The Flatpak application ID is `dev.mitchellrenouf.mrml`. Its manifest builds
the CUDA variant and relies on the host NVIDIA driver for driver libraries.

```bash
./scripts/build-flatpak.sh
flatpak run dev.mitchellrenouf.mrml
```

The application requests broad host filesystem and device access for workspace
automation and local GPU inference. Review the manifest before installation.

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
