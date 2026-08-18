# MRML

MRML is the **Mitchell Renouf Machine Learning Library**, a CC0-licensed,
free-as-in-speech machine learning library. It is a Rust workspace for local
GGUF inference, AI-agent tooling, and native desktop, terminal, machine, and
HTTP interfaces. The project is hosted at
[github.com/mitchellrenouf/mrml](https://github.com/mitchellrenouf/mrml).

MRML currently focuses on Gemma 4 GGUF models. Its inference implementation is
native to this repository and does not link to or require llama.cpp. The
`mrml-tensor` crate supplies quantized CPU operations and custom NVIDIA CUDA
kernels; `mrml-model` exposes the model and chat interface used by the apps.

## Current status

Working today:

- Native Qt 6 desktop application on Windows and Linux.
- Interactive and one-shot terminal CLI.
- Versioned JSONL machine CLI for ChatGPT and automated regression tests.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` HTTP/SSE server.
- Memory-mapped GGUF loading and native Q4_0/Q8_0 CPU operations.
- CUDA Q4_0/Q8_0 projections, attention, MoE routing, CUDA graphs, GPU-resident
  activation buffers, and F16/Q8_0/Q4_0 KV-cache storage.
- General, coder, and automatic agent modes.
- Workspace, Git, shell, browser, desktop, media, and stdio MCP tools.
- Configurable context size and automatic conversation compaction.

Performance depends on prompt length, context size, quantization, GPU clocks,
available VRAM, and full layer residency. On the current RTX 5070 Ti test
system, Gemma 4 26B A4B Q4_0 sustains approximately **60 tokens/second** in the
8K CUDA machine-CLI workload after warm-up. This is a measured reference, not
a universal performance guarantee.

### Known limitations

- CUDA is the only implemented native GPU backend. Vulkan, ROCm, and SYCL
  feature/backend selections are placeholders and do not yet execute inference.
- GPU allocation is best-effort. Other GPU processes can prevent full model
  residency and cause a substantial slowdown.
- The configured 256K maximum is supported by cache sizing and compaction, but
  prompt initialization currently evaluates only the most recent prompt tail;
  it does not yet evaluate an arbitrary 256K-token prompt in full.
- Q4 KV storage saves VRAM but is slower than F16 at 8K on the current CUDA
  kernels. The automatic default therefore remains F16 for shorter contexts.
- The speculative module has n-gram proposal logic, but generation does not yet
  load a draft/MTP model or perform batched speculative verification.

## Workspace crates and binaries

| Crate | Purpose | Binary |
| --- | --- | --- |
| `mrml-core` | Agent, configuration, tools, rules, and Hugging Face integration | — |
| `mrml-model` | Application-facing GGUF model/chat adapter | — |
| `mrml-tensor` | Tensor math, GGUF execution, KV cache, CPU and CUDA kernels | — |
| `mrml-cli` | Interactive and one-shot terminal frontend | `mrml-cli` |
| `mrml-machine` | JSONL automation and ChatGPT regression frontend | `mrml-machine` |
| `mrml-server` | OpenAI-compatible HTTP/SSE frontend | `mrml-server` |
| `mrml-qt` | Native Qt 6 desktop frontend | `mrml-qt` |

## Requirements

- Stable Rust and Cargo.
- Qt 6 development files, Qt Widgets/UiTools, and `qmake` for `mrml-qt`.
- A C++ compiler compatible with the installed Qt build.
- NVIDIA CUDA Toolkit 13.3 for CUDA builds.

On Windows, install Visual Studio Build Tools with the Desktop development with
C++ workload, Rust MSVC, a Qt 6 `msvc2022_64` kit, and CUDA 13.3:

```powershell
$env:Path = "C:\Qt\6.8.3\msvc2022_64\bin;$env:Path"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
qmake -query QT_VERSION
nvcc --version
cargo build -p mrml-qt --release --locked --no-default-features --features cuda
```

On Linux, install Qt 6, a C++ toolchain, Rust, the NVIDIA driver, and CUDA 13.3:

```bash
export CUDA_PATH=/opt/cuda
export PATH="$CUDA_PATH/bin:$PATH"
qmake6 -query QT_VERSION
nvcc --version
cargo build -p mrml-qt --release --locked --no-default-features --features cuda
```

CPU-only terminal builds do not require CUDA or Qt:

```bash
cargo build -p mrml-cli --release --locked --no-default-features
```

## Running

The default model resolves
`ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`. Pass `--model` to select a local GGUF.

```bash
# Desktop
cargo run --release --bin mrml-qt --features cuda -- --model /path/to/model.gguf

# Interactive terminal
cargo run --release --bin mrml-cli --features cuda -- --model /path/to/model.gguf

# One-shot terminal prompt
cargo run --release --bin mrml-cli --features cuda -- --prompt "Hello" --ctx-size 8192 --model /path/to/model.gguf

# HTTP/SSE server
cargo run --release --bin mrml-server --features cuda -- --port 8080 --model /path/to/model.gguf
```

### Machine interface

`mrml-machine` is the stable non-interactive interface for ChatGPT and test
harnesses. Records use schema version 1 and the `MRML_MACHINE_JSON=` prefix so
they remain identifiable alongside CUDA diagnostics.

```bash
cargo run --release --bin mrml-machine --features cuda -- chat --prompt "What time is it?"
printf '{"op":"chat","id":1,"prompt":"Hi"}\n{"op":"exit","id":2}\n' | cargo run --release --bin mrml-machine --features cuda -- session
```

Session operations are `chat`, `health`, `reset`, and `exit`. Chat results
include content, separated reasoning, tool events, finish reason, token count,
generation time, wall time, and tokens per second.

Run `mrml-cli --help` or `mrml-machine --help` for all model, cache, backend,
context, workspace, and MCP options.

## Flatpak

The Flatpak application ID is `dev.mitchellrenouf.mrml`. The manifest compiles
MRML with CUDA 13.3, removes the toolkit after the build, and retains only the
CUDA runtime required by the application. The host NVIDIA driver supplies the
driver libraries.

```bash
./scripts/build-flatpak.sh
flatpak run dev.mitchellrenouf.mrml
```

The application requests broad host filesystem and device access for workspace
automation and local GPU inference. Review the manifest before installation.

## License

MRML is free as in speech. It is dedicated to the public domain under
[CC0 1.0 Universal](LICENSE) (`CC0-1.0`), so anyone may use, study, modify,
share, and redistribute it for any purpose, including commercially, without
asking permission. Where a public-domain dedication is not legally possible,
CC0 provides the broadest possible waiver and license fallback.
