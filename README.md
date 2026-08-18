# RustLlama

RustLlama is a local AI assistant written in Rust. It provides a native Qt 6 desktop interface, a terminal interface, an OpenAI-compatible HTTP/SSE server, workspace tools, browser automation, and an in-process GGUF inference engine named `qtensor`.

The native engine currently targets Gemma 4 GGUF models and supports CPU inference plus custom NVIDIA CUDA kernels. It does not use or require llama.cpp. Vulkan, ROCm, and SYCL appear as future-facing feature/backend choices in parts of the codebase, but they do not currently provide native GPU inference.

## Current state

- Native Qt 6 Widgets GUI on Windows and Linux.
- Interactive CLI (`--cli`) and one-shot prompts (`--cli --prompt "..."`).
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` server (`--serve`).
- Memory-mapped GGUF loading and native Q4_0/Q8_0 CPU operations.
- CUDA Q4_0/Q8_0 projections, attention, MoE routing, and GPU-resident decode buffers.
- General, coder, and automatic agent modes.
- Built-in workspace, Git, shell, browser, desktop, and media tools.
- Optional external tools through stdio Model Context Protocol servers.
- Context selection through `--ctx-size` and automatic conversation compaction through `--max-context-tokens`.

Performance depends heavily on the model, quantization, GPU, available VRAM, prompt length, and whether every required layer remains GPU-resident. No fixed token-rate claim is made. As a reference, Gemma 4 26B A4B Q4_0 currently decodes at approximately 57 tokens/second on an RTX 5070 Ti in the tested short-context CLI workload.

### Known limitations

- CUDA is the only implemented native GPU backend. The Vulkan, ROCm, and SYCL feature flags are placeholders.
- The native CUDA KV cache currently stores `f32` values and caps its resident capacity at 1,024 tokens. The cache-type CLI options are accepted for configuration/display but are not yet implemented as alternate native storage formats.
- The engine currently evaluates only the most recent prompt tail during initialization. A configured 256K maximum therefore does not mean that 256K tokens are resident or evaluated.
- The speculative-decoding module contains n-gram proposal logic, but generation does not yet perform MTP/draft-model loading or batched speculative verification.
- GPU allocation is best-effort. Insufficient VRAM can leave a layer without its full resident CUDA path and reduce performance considerably.

## Build requirements

- Stable Rust with Cargo
- Qt 6 development files, Qt Widgets/UiTools, and `qmake`
- A C++ compiler compatible with the installed Qt build
- NVIDIA CUDA Toolkit 13.3 for CUDA builds

### Windows

Install Visual Studio Build Tools with Desktop development with C++, a Qt 6 `msvc2022_64` kit, Rust MSVC, and CUDA 13.3. Then ensure Qt and CUDA are discoverable:

```powershell
$env:Path = "C:\Qt\6.8.3\msvc2022_64\bin;$env:Path"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
qmake -query QT_VERSION
nvcc --version
cargo build --release --locked --no-default-features --features cuda
```

The Windows GitHub Actions runner installs Qt and CUDA and uses that same explicit CUDA feature selection.

### Linux

Install Qt 6, a C++ toolchain, Rust, the NVIDIA driver, and CUDA 13.3. Package names vary by distribution. A source build is:

```bash
export CUDA_PATH=/opt/cuda
export PATH="$CUDA_PATH/bin:$PATH"
qmake6 -query QT_VERSION
nvcc --version
cargo build --release --locked --no-default-features --features cuda
```

For a CPU-only build:

```bash
cargo build --release --locked --no-default-features
```

## Running

The default model setting downloads/resolves `ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0`. A local GGUF can be selected explicitly.

```bash
# Desktop GUI
cargo run --release --no-default-features --features cuda -- --model /path/to/model.gguf

# Interactive terminal
cargo run --release --no-default-features --features cuda -- --cli --model /path/to/model.gguf

# One-shot CLI prompt with an 8K configured context
cargo run --release --no-default-features --features cuda -- --cli --prompt "Hello" --ctx-size 8192 --model /path/to/model.gguf

# HTTP/SSE server
cargo run --release --no-default-features --features cuda -- --serve --port 8080 --model /path/to/model.gguf
```

Useful options include `--max-tokens`, `--temperature`, `--gpu-layers`, `--workspace-root`, `--mode`, and `--mcp-server`. Run `rustllama --help` for the complete current list.

## Flatpak

The Flatpak application ID is `dev.mitchellrenouf.rustllama`. Its main manifest installs CUDA 13.3 for compilation and builds RustLlama with `--no-default-features --features cuda`; CUDA is therefore compiled into the Linux package rather than relying on feature autodetection. After compilation, the toolkit, headers, compiler, and unused libraries are removed. The finished package retains only `libcudart`, while the host NVIDIA driver provides `libcuda`.

```bash
./scripts/build-flatpak.sh
flatpak run dev.mitchellrenouf.rustllama
```

The Flatpak requests broad host filesystem and device access because the application provides workspace automation and local GPU inference. Review the manifest permissions before installation.

## Repository layout

- `src/`: application, GUI, CLI, server, and tools
- `crates/qtensor/`: GGUF reader, tensor operations, model execution, and CUDA kernels
- `crates/llama-rs/`: application-facing inference wrapper (despite its historical name, it does not link llama.cpp)
- `flatpak/`: Flatpak manifests and desktop metadata
- `.github/workflows/flatpak-build.yml`: Linux Flatpak and Windows CUDA release builds

## License

MIT. See [LICENSE](LICENSE).
