# QT_llama.cpp: Enterprise AI Assistant & Vibe-Coder (In-Process llama.cpp, Qt6 GUI, Chromiumoxide & Flatpak)

A high-performance, autonomous AI assistant written in Rust with **in-process `llama.cpp` GGUF inference bindings**. It runs completely local on **Linux** (Arch Linux, Fedora, Ubuntu, or Freedesktop Flatpak Sandbox) with zero external server dependencies, instant token streaming, and direct DevTools browser automation.

Functioning like **Gemini directly on your Linux desktop and terminal**, the application launches a modern **Qt6 QML Graphical Interface** by default (with `--cli` available for terminal mode), supporting **General-Purpose AI Assistant Mode**, **Vibe-Coding Mode**, and **Autonomous Inner Monologue Mode (`/automatic`)**.

---

## 🌟 Highlights & Architecture

- **In-Process `llama.cpp` Inference Engine**: Direct C/C++ FFI bindings via the `llama-cpp-binding` crate. Loads GGUF models directly in memory with KV caching, batch evaluation, and fast token generation with zero HTTP network overhead.
- **Git Submodule Integration**: `llama.cpp` is linked directly as a Git submodule (`llama.cpp/`) and built statically via CMake during compilation.
- **Pure Chromiumoxide Web Engine**: 100% headless Chromium DevTools Protocol (CDP) for all browser automation, searches (`web_search`), and live web extractions (`web_fetch`). Completely eliminates `reqwest` and HTTP client dependencies.
- **Modern Qt6 QML Desktop Interface (Default)**: Sleek dark-mode GUI (`qml/Main.qml`) with real-time token streaming, formatted markdown bubbles, thinking animation blocks, tool call inspection cards, and speech synthesis toggles.
- **Flatpak Packaging (Freedesktop SDK 26.08 Beta)**: Bundled under application ID `org.gemma.GemmaAgent` targeting `org.freedesktop.Platform//26.08` and `org.freedesktop.Sdk//26.08` with Rust stable extension and AppStream metadata.
- **Automated Flatpak Update Infrastructure**: Configured with `flatpak-external-data-checker` (`update-checker.json`), GitHub Actions automated release pipeline (`.github/workflows/flatpak-build.yml`), and local update scripts (`./scripts/update-flatpak.sh`).
- **Autonomous Inner Monologue Mode (`/automatic` | `--mode automatic`)**: Gemma 4 26B maintains a continuous, human-like internal monologue (`🧠 Inner Monologue...`), reflecting step-by-step on goals, context, tool choices, error recovery, and self-correction before taking action.
- **Desktop Screenshot & Multimodal Perception**: Native Linux screenshot capture (`spectacle`, `grim`, `scrot`, `ffmpeg`) fed directly into multimodal vision inspection.
- **Speech Synthesis & Audio Recording**: Integrated `/speech` command for desktop text-to-speech (`spd-say`, `espeak-ng`) and microphone recording (`ffmpeg`, `pw-record`).
- **22 Built-in Tools**: Full workspace file management, diff editing, shell commands, Git checkpoints, web search, browser automation, and desktop controls.

---

## 🚀 Quick Start

### 1. Build from Source
```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/mitchell/gemma.git
cd gemma

# Build optimized release binary
cargo build --release
```

### 2. Launch GUI Interface (Default)
```bash
./target/release/gemma-agent --model /path/to/gemma-4-26b-it-q4_0.gguf
```

### 3. Launch Terminal CLI Interface
```bash
./target/release/gemma-agent --cli --model /path/to/gemma-4-26b-it-q4_0.gguf
```

---

## 🛠️ Operating Modes

- **General Mode** (`--mode general`): Natural conversation, writing, math, web research, desktop control, and coding.
- **Coder Mode** (`--mode coder`): Autonomous software development, testing, refactoring, and debugging.
- **Automatic Mode** (`--mode automatic` or `/automatic`): Autonomous multi-turn task execution with internal monologue reasoning.

---

## 📦 Flatpak Build & Installation

```bash
# Build Flatpak bundle with Freedesktop SDK 26.08 Beta
./scripts/build-flatpak.sh

# Run Flatpak application
flatpak run org.gemma.GemmaAgent
```
