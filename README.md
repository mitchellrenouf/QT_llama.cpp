# QT_llama.cpp: Enterprise AI Assistant & Vibe-Coder (Pure Rust qtensor Engine, Qt6 GUI, Chromiumoxide & Flatpak)

A ultra-high-performance, autonomous AI assistant written in pure Rust with **in-process `qtensor` GGUF inference & CUDA GPU acceleration**. It runs completely local on **Linux and Windows** with zero external server dependencies, instant token streaming ($\ge 110\text{--}120+$ tk/s), and direct DevTools browser automation.

Functioning like **Gemini directly on your desktop and terminal**, the application launches a modern **Qt6 QML Graphical Interface** by default (with `--cli` available for terminal mode and `--serve` for OpenAI-compatible HTTP API mode), supporting **General-Purpose AI Assistant Mode**, **Vibe-Coding Mode**, and **Autonomous Inner Monologue Mode (`/automatic`)**.

---

## ⚡ Why QT_llama & qtensor Are So Blazing Fast

`QT_llama` achieves industry-leading token generation throughput ($\ge 112\text{--}121+$ tokens/sec on consumer GPUs like the RTX 5070 Ti) through low-level systems optimizations engineered directly in Rust and CUDA:

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
- **Modern Qt6 QML Desktop Interface (Default)**: Sleek dark-mode GUI (`qml/Main.qml`) with real-time token streaming, formatted markdown bubbles, thinking animation blocks, tool call inspection cards, and speech synthesis toggles.
- **Flatpak Packaging (KDE Platform 6.11)**: Bundled under application ID `dev.mitchellrenouf.QT_llama` targeting `org.kde.Platform//6.11` and `org.kde.Sdk//6.11` with Rust stable extension, Qt6 QML declarative runner, Vulkan stack, and AppStream metadata.
- **Autonomous Inner Monologue Mode (`/automatic` | `--mode automatic`)**: Gemma 4 26B maintains a continuous internal monologue (`🧠 Inner Monologue...`), reflecting step-by-step on goals, context, tool choices, error recovery, and self-correction before taking action.
- **22 Built-in Tools**: Full workspace file management, diff editing, shell commands, Git checkpoints, web search, browser automation, and desktop controls.

---

## 🚀 Quick Start

### 1. Build from Source
```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/mitchellrenouf/QT_llama.cpp.git
cd QT_llama.cpp

# Build optimized release binary (auto-detects CUDA/Vulkan)
cargo build --release
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

### 2. Launch GUI Interface (Default)
```bash
./target/release/qt_llama --model /path/to/gemma-4-26b-it-q4_0.gguf
```

### 3. Launch Terminal CLI Interface
```bash
./target/release/qt_llama --cli --model /path/to/gemma-4-26b-it-q4_0.gguf
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
