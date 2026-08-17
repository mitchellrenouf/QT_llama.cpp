# RustLlama: Enterprise AI Assistant & Vibe-Coder (Pure Rust qtensor Engine, Native Qt6 GUI, Chromiumoxide & Flatpak)

A ultra-high-performance, autonomous AI assistant written in pure Rust with **in-process `qtensor` GGUF inference & CUDA GPU acceleration**. It runs completely local on **Linux and Windows** with zero external server dependencies, instant token streaming ($\ge 110\text{--}120+$ tk/s), and direct DevTools browser automation.

Functioning like **Gemini directly on your desktop and terminal**, RustLlama launches a native **Qt6 Widgets** interface by default. Its UI and application logic are authored in Rust; the Qt binding layer is generated during the build. The application supports terminal (`--cli`) and OpenAI-compatible HTTP API (`--serve`) modes as well as General-Purpose, Vibe-Coding, and Autonomous Inner Monologue modes.

---

## ⚡ Why RustLlama & qtensor Are So Blazing Fast

`RustLlama` achieves industry-leading token generation throughput ($\ge 112\text{--}121+$ tokens/sec on consumer GPUs like the RTX 5070 Ti) through low-level systems optimizations engineered directly in Rust and CUDA:

1. **100% Compute & I/O Overlap (Asynchronous GPU Pipelining)**:
   - Traditional inference runtimes sequentially wait for the GPU to evaluate a token before formatting and transmitting it to the user.
   - `qtensor` launches the next token's CUDA forward pass **asynchronously on the GPU hardware stream first**. While tensor cores execute the matrix math in hardware, the CPU concurrently streams the current token piece over Tokio channels with **zero GPU idle stall time**.

2. **Q8_0 Quantized KV Cache & Sliding Window Attention (SWA)**:
   - LLM generation is strictly **memory-bandwidth bound**. Quantizing the Key/Value cache from FP16 to **Q8_0** halves memory bandwidth pressure per layer.
   - Gemma 4's alternating Sliding Window Attention ($W=4096$) is managed via ring-buffer eviction, preventing memory bus saturation and supporting contexts up to **256,000 tokens**.

3. **Zero-Allocation Autoregressive Hot-Loop**:
   - Eliminates all dynamic memory allocations (`malloc`/`free`, `Vec::clone`, string allocations) during generation by using pre-allocated single-token batches and stack-allocated piece buffers.

4. **Warp-Level CUDA Kernel Vectorization**:
   - Custom CUDA kernels in `crates/qtensor` utilize **Warp Shuffle Intrinsics (`__shfl_down_sync`)** to reduce token activations across 32 threads entirely in register space without touching slow GPU shared or global memory.
   - 16-element Q4_0 and Q8_0 dot products are vectorized directly into fast FP16 half-precision tensor instructions.

5. **Direct Zero-Copy GGUF Memory Mapping (No Python / No GIL)**:
   - 100% pure Rust compiled directly to native machine code with zero Python interpreter overhead, zero Global Interpreter Lock (GIL) contention, and zero IPC RPC marshalling hops.

6. **Speculative Decoding & Prompt Prefix Caching**:
   - **Speculative Decoding**: N-Gram candidate drafting and batched forward verification deliver $1.5\times\text{--}2.0\times$ throughput acceleration.
   - **Prompt Prefix Caching**: Matches invariant system prompts and project rules (`GEMMA.md`), reducing time-to-first-token (TTFT) to **0 ms**.

7. **Console Display Buffer Jitter Elimination**:
   - Avoids synchronous Windows console / terminal display buffer blocking on every character by buffering chunked stdout streaming flushes.

---

## 🌟 Highlights & Architecture

- **Pure Rust `qtensor` Engine**: Native quantized tensor mathematics (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`, `BF16`), zero-copy GGUF v2/v3 parsing, and native CUDA kernel execution.
- **Multi-Device & RAM Scaling**: Automatically estimates VRAM limits across GPU 0, secondary GPUs, and host CPU memory, distributing transformer layers dynamically.
- **OpenAI-Compatible HTTP / SSE API Server**: Embedded `/v1/models` and `/v1/chat/completions` API server (`--serve --port 8080`) supporting streaming Server-Sent Events (SSE).
- **Model Context Protocol (MCP) Client**: Connects to external stdio MCP tool servers (`--mcp-server "<command>"`) to dynamically discover and execute remote tools.
- **Pure Chromiumoxide Web Engine**: 100% headless Chromium DevTools Protocol (CDP) for all browser automation, searches (`web_search`), and live web extractions (`web_fetch`). Completely eliminates `reqwest` and HTTP client dependencies.
- **Native Qt6 Widgets Desktop Interface (Default)**: A polished, platform-adaptive desktop workspace with real-time token streaming, chat history, mode and backend controls, model downloads, and speech toggles. It has no QML runtime or `qml` executable dependency.
- **Flatpak Packaging (KDE Platform 6.11)**: Bundled with the Rust stable extension, Qt 6 Widgets runtime, Vulkan stack, and AppStream metadata.
- **Autonomous Inner Monologue Mode (`/automatic` | `--mode automatic`)**: Gemma 4 26B maintains a continuous internal monologue (`🧠 Inner Monologue...`), reflecting step-by-step on goals, context, tool choices, error recovery, and self-correction before taking action.
- **22 Built-in Tools**: Full workspace file management, diff editing, shell commands, Git checkpoints, web search, browser automation, and desktop controls.

---

## 🚀 Quick Start

### 1. Install build dependencies

RustLlama uses native Qt 6 Widgets from Rust. It needs Qt's development headers, `qmake`, and a C++ compiler in addition to Rust. QML is not used or required.

#### Windows: Visual Studio 2026, Qt 6, Rust, and CUDA

Install Visual Studio 2026 Build Tools with **Desktop development with C++**, the MSVC x64/x86 toolset, and a Windows SDK. Install Qt 6's `msvc2022_64` kit and QtUiTools using the Qt Online Installer. MSVC v145 (VS 2026) is ABI-compatible with the prebuilt MSVC 2022 Qt kit.

Install Rust with `rustup` and select the MSVC target:

```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc
rustup default stable-x86_64-pc-windows-msvc
```

For NVIDIA acceleration, install the NVIDIA driver and CUDA Toolkit. Add the selected Qt kit's `bin` directory to your user `PATH` once (this lets Cargo find `qmake` while compiling and lets Windows find the Qt DLLs when running). RustLlama's Cargo configuration discovers and initializes the installed Visual Studio C++ tools automatically, so an ordinary PowerShell window is sufficient:

```powershell
# Once per PowerShell session, or add this directory in Windows Environment Variables
$env:Path = "C:\Qt\6.11.1\msvc2022_64\bin;$env:Path"
qmake -query QT_VERSION
nvcc --version                 # optional, required for CUDA
cargo run --release
```

Use the installed Qt version in place of `6.11.1`. Cargo supplies the required MSVC C++17 compatibility options for the Rust Qt binding; do not set `CXXFLAGS` manually. `nvcc` enables CUDA autodetection; omit CUDA if you want a CPU-only build. The build script checks Qt, the compiler, and CUDA availability before the application itself is built, and reports a direct setup error if Qt or the C++ tools are absent.

#### Arch Linux: Qt 6, Rust, and CUDA

Install the native build tools, Qt Widgets/UiTools, Rust, and Vulkan headers:

```bash
sudo pacman -S --needed base-devel rustup qt6-base qt6-tools vulkan-headers
rustup default stable
qmake6 -query QT_VERSION
```

For NVIDIA CUDA acceleration, also install the proprietary NVIDIA driver and CUDA toolkit:

```bash
sudo pacman -S --needed nvidia nvidia-utils cuda
export PATH=/opt/cuda/bin:$PATH
export CUDA_PATH=/opt/cuda
nvcc --version
```

Then launch with `cargo run --release`. For a CPU-only build, use `cargo run --release --no-default-features`. Install the driver package appropriate to your kernel (for example, `linux-lts-nvidia` on an LTS kernel) rather than blindly mixing kernel modules.

### 2. Build from Source
```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/mitchellrenouf/QT_llama.cpp.git
cd QT_llama.cpp

# Build and launch the optimized native application (auto-detects CUDA/Vulkan)
cargo run --release
```

### GPU Acceleration

By default, the build system **auto-detects** available GPU backends. If `nvcc` is found in your PATH or at `/opt/cuda`, CUDA is enabled automatically. If Vulkan headers are found, Vulkan is enabled.

You can also explicitly select a backend with Cargo features:

```bash
# Force CUDA only (NVIDIA GPUs)
cargo build --release --features cuda --no-default-features

# Force Vulkan only (AMD, Intel, NVIDIA — any Vulkan-capable GPU)
cargo build --release --features vulkan --no-default-features

# Both CUDA + Vulkan
cargo build --release --features cuda,vulkan --no-default-features

# AMD ROCm / HIP
cargo build --release --features hipblas --no-default-features

# Intel SYCL / oneAPI
cargo build --release --features sycl --no-default-features
```

Environment variables also work:
```bash
LLAMA_CUDA=1 cargo build --release    # Enable CUDA
LLAMA_VULKAN=1 cargo build --release  # Enable Vulkan
```

| Backend | GPU Vendor | Feature Flag | Auto-Detected? |
|---------|-----------|--------------|----------------|
| CUDA (cuBLAS) | NVIDIA | `--features cuda` | ✅ Yes (`nvcc` in PATH) |
| Vulkan | AMD, Intel, NVIDIA | `--features vulkan` | ✅ Yes (`vulkan.h` found) |
| HIP (ROCm) | AMD | `--features hipblas` | ❌ Explicit only |
| SYCL (oneAPI) | Intel | `--features sycl` | ❌ Explicit only |
| CPU only | Any | `--no-default-features` | — |

### 3. Launch GUI Interface (Default)
```bash
cargo run --release -- --model /path/to/gemma-4-26b-it-q4_0.gguf
```

### 4. Launch Terminal CLI Interface
```bash
cargo run --release -- --cli --model /path/to/gemma-4-26b-it-q4_0.gguf
```

---

## 🛠️ Operating Modes

- **General Mode** (`--mode general`): Natural conversation, writing, math, web research, desktop control, and coding.
- **Coder Mode** (`--mode coder`): Autonomous software development, testing, refactoring, and debugging.
- **Automatic Mode** (`--mode automatic` or `/automatic`): Autonomous multi-turn task execution with internal monologue reasoning.

---

## 📦 Flatpak Build & Installation

```bash
# Build Flatpak bundle with Freedesktop SDK 25.08
./scripts/build-flatpak.sh

# Run Flatpak application
flatpak run dev.mitchellrenouf.QT_llama
```
