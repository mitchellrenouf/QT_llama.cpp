# Enterprise Gemma 4 26B AI Assistant & Vibe-Coder (Pure Linux, Qt6 & Flatpak SDK 26.08 Beta)

A high-performance, autonomous AI assistant written in Rust. It interfaces directly with **Gemma 4 26B** (or compatible GGUF models) running locally on **Linux** (Arch Linux, Fedora, Ubuntu, or Freedesktop Flatpak Sandbox) via `llama-server`'s OpenAI-compatible API (`http://localhost:8080/v1`).

Functioning like **Gemini directly on your Linux desktop and terminal**, the application provides both a rich **CLI REPL** and a modern **Qt6 QML Graphical Interface**, supporting **General-Purpose AI Assistant Mode**, **Vibe-Coding Mode**, and **Autonomous Inner Monologue Mode (`/automatic`)**.

Equipped with **Pure Linux Standards (Zero Windows Code)**, **Modern Qt6 QML Desktop Interface**, **Headless Chromium CDP Browser Automation (`--headless=new`)**, **Flatpak Packaging with Freedesktop SDK 26.08 (Beta)**, **Automated Flatpak Updates & CI/CD**, **Direct Headless DOM Text Extraction (`browser_get_content`)**, **Infinite Tool Loop Protection**, **Toggleable Text-to-Speech (`/speech`)**, **PipeWire/PulseAudio Audio Recording**, **Desktop Screenshot Vision Perception**, and 22 integrated workspace, git, web, desktop, and media tools.

---

## 🌟 Highlights & Platform Features

- **Pure Linux Architecture**: 100% native Linux / Freedesktop implementation with complete elimination of legacy Windows/PowerShell dependencies.
- **Modern Qt6 QML Desktop Interface**: Sleek dark-mode GUI (`qml/Main.qml`) with real-time streaming tokens, formatted markdown chat bubbles, thinking animation blocks, tool call inspection cards, and speech synthesis toggles.
- **Flatpak Packaging (Freedesktop SDK 26.08 Beta)**: Bundled under application ID `org.gemma.GemmaAgent` targeting the latest `org.freedesktop.Platform//26.08` and `org.freedesktop.Sdk//26.08` with Rust stable extension and AppStream metadata.
- **Automated Flatpak Update Infrastructure**: Configured with `flatpak-external-data-checker` (`update-checker.json`), GitHub Actions automated release pipeline (`.github/workflows/flatpak-build.yml`), and local update scripts (`./scripts/update-flatpak.sh`).
- **Headless Chromium & Brave Browser Automation**: Direct support for Flatpak Brave (`flatpak run com.brave.Browser`), native Brave (`/usr/bin/brave-origin`), Chromium, and Google Chrome with isolated profile management avoiding `SingletonLock` collisions.
- **Fix for Server 500 `Invalid uri-encoded base64 value` Error**: Removed automatic conversion of `DATA_URI:` strings into base64 image objects for text-only GGUF models running under `llama-server`. Completely eliminates server 500 errors and keeps token context light.
- **Direct Headless DOM Text Extraction (`browser_get_content`)**: Prioritizes direct DOM text parsing over visual screenshots. Extracting clean `innerText` and structured headlines directly from headless browser sessions executes in **<100ms** and consumes **0 image tokens**.
- **Headless Browser Mode (`--headless=new`, On by Default)**: `browser_open` launches Brave / Edge with `--headless=new` enabled by default, ensuring all web page browsing, DOM rendering, and web element automation execute silently in the background without popping up GUI windows over your screen.
- **Infinite Tool Loop Protection**: Detects when a tool is called repeatedly with identical arguments (max 3 consecutive identical calls) or when a turn exceeds 10 execution steps. Automatically breaks the loop and prompts the model to consult the user.
- **Smart UI Element Clicker (`browser_click_element`)**: Locates buttons, links, search bars, and input fields on screen by their visible text or name (e.g., `"Add to Cart"`, `"Search Amazon"`, `"Proceed to checkout"`).
- **Toggleable Text-to-Speech (`/speech` Command, Disabled by Default)**: Text-to-speech audio output (`speak_text`) is **disabled by default** to avoid unexpected audio playback. Users can toggle TTS audio output on and off anytime in the CLI using the `/speech` command (`🔊 Text-to-Speech (TTS) enabled` / `🔇 Text-to-Speech (TTS) disabled`).
- **Native Rust Web Page Fetching (`web_fetch`)**: High-speed, robust DOM parser implemented in native Rust (`reqwest` + `scraper`). Filters out JavaScript noise tags (`<script>`, `<style>`, `<noscript>`, `<svg>`) to extract clean, readable news headlines and text from complex Single-Page Application sites (e.g., AP News, TechCrunch, Reuters, BBC).
- **Google Search Integration (`web_search`)**: Queries Google Search (`https://www.google.com/search?q=...&gbv=1`), extracting direct result URLs, titles, and snippets.
- **Default Browser Profile & Custom CLI Options (`--browser-exe`, `--browser-profile`)**: `browser_open` automatically detects the user's default browser profile (`User Data` / `~/.config/BraveSoftware/Brave-Origin`) and passes `--user-data-dir` and `--profile-directory="Default"` when running in `--headless=new` or GUI mode. Users can also configure custom executable paths or profile directories via CLI arguments (`--browser-exe`, `--browser-profile`) or environment variables (`BROWSER_EXE`, `BROWSER_PROFILE`).
- **Visual Output Partition Banners**: Clear visual divider (`─── 🎨 Rich Formatted Output ───`) separating live streaming text tokens from the `termimad` rich formatted markdown view.
- **Enhanced `/help` Command Menu**: Command reference accessible anytime via `/help`, also displayed in the startup banner.
- **Autonomous Inner Monologue Mode (`/automatic` | `/mode automatic`)**: Gemma 4 26B maintains a continuous, human-like internal monologue (`🧠 Inner Monologue...`), reflecting step-by-step on goals, context, tool choices, error recovery, and self-correction before taking action.
- **Audio Capabilities (Speech Synthesis & Mic Recording)**:
  - `speak_text`: Speaks text messages aloud (`spd-say`, `espeak-ng`, or Windows Speech Synthesis; requires `/speech` toggle enabled).
  - `record_audio`: Records microphone audio into `.gemma/audio/recording.wav` via PipeWire/PulseAudio (`ffmpeg`/`pw-record`) on Linux or MCI on Windows.
- **Video Capabilities (Webcam Frame & Screen Video Capture)**:
  - `capture_webcam`: Captures a snapshot video frame from the webcam and passes it to Gemma 4's multimodal vision engine.
  - `record_screen_video`: Records a multi-frame video keyframe sequence across N seconds for video motion analysis.
- **Multimodal Vision Engine**: Gemma 4 26B visually inspects desktop screenshots, browser windows, UI layouts, and images passed directly via base64 `image_url` data URIs.
- **Desktop Screenshot & Control**:
  - `take_screenshot`: Capture active monitor screens using native tools (`spectacle`, `grim`, `scrot`, or PowerShell) and feed base64 images to Gemma's multimodal vision engine.
  - `open_app`: Launch any desktop application, document, or URL (`brave-origin`, `dolphin`, `kate`, `msedge`, `notepad`, `calc`).
- **Interactive Browser Automation**:
  - `browser_open`: Launch Brave Origin (`/usr/bin/brave-origin`) / Brave Nightly / Brave Stable / Chromium / Chrome / Microsoft Edge with target web pages (Headless by default).
  - `browser_get_content`: Extract clean DOM text content (Preferred over screenshots).
  - `browser_screenshot`: Capture visual page snapshots for visual inspection.
  - `browser_click_element`: Find and click buttons/links by visible text (e.g. `'Add to Cart'`).
  - `browser_click`: Click screen coordinates (x, y) on interactive buttons/links.
  - `browser_type`: Type text into search inputs or web forms.
- **Dynamic Search-Based Fact Verification**: Automatically triggers `web_search` for real-world factual queries regarding public figures, world leaders, news events, or popes.
- **Rich Terminal Markdown Formatting (`termimad`)**: Renders markdown headers, bold/italics, bullet lists (`•`), and boxed code cards natively in terminal.
- **Triple Agent Modes (`--mode general` | `--mode coder` | `--mode automatic`)**:
  - **`General Mode` (Default)**: Functions like Gemini in a browser. Natural conversation, writing essays/emails, answering questions, math, web research, desktop control, audio/video tools, and coding.
  - **`Coder Mode`**: Focused strictly on autonomous software development, testing, and debugging.
  - **`Automatic Mode`**: Continuous human-like inner monologue, planning, reflection, and autonomous execution.
- **OpenAI-Compatible Streaming Client**: Asynchronous `reqwest` client with SSE stream parsing for text tokens, reasoning (`<think>`) tokens, and assembled `tool_call` deltas.
- **Deep Reasoning Display**: Renders Gemma 4 26B thinking tokens live in dim styling (`🧠 Thinking...`).
- **Live Terminal Diff Previews**: Displays colorized unified diffs (`+` additions in green, `-` deletions in red, line context in cyan) whenever Gemma modifies workspace files (`write_file`, `replace_file_content`).
- **Git Checkpoint & Rollback Tools**:
  - `git_checkpoint`: Creates a snapshot before major refactors.
  - `git_rollback`: Restores workspace to a clean state if compiler errors occur.
  - `git_diff`: Views uncommitted working tree modifications.
- **Automatic Project Rules Loader**: Discovers `GEMMA.md`, `AGENTS.md`, `.gemma/rules`, `.agent/rules`, or `CLAUDE.md` in your project root.
- **Session Persistence**: `/save [name]`, `/load [name]`, `/sessions`.
- **Context Auto-Compaction & Telemetry**: `/compact [threshold]`, `/status`, `/reset` or `/clear`.
- **22 Integrated Tools**:
  - `speak_text`, `record_audio`, `capture_webcam`, `record_screen_video`
  - `take_screenshot`, `open_app`, `browser_open`, `browser_get_content`, `browser_screenshot`, `browser_click_element`, `browser_click`, `browser_type`
  - `web_search`, `web_fetch`, `view_file`, `write_file`, `replace_file_content`, `list_dir`, `grep_search`, `run_command`, `git_checkpoint`, `git_rollback`, `git_diff`.

---

## 🚀 Quick Start

### 1. Start `llama-server`

Run `llama-server` on Linux or Windows:

```bash
# On Arch Linux / Linux
llama-server \
  -m "path/to/gemma-4-26B-A4B-it-Q4_0.gguf" \
  -c 16384 \
  -ngl 99 \
  --port 8080 \
  --api-key "mitchell" \
  --chat-template gemma
```

### 2. Build & Run the Rust Agent

```bash
# Run unit tests
cargo test

# Build release binary
cargo build --release

# Run in General-Purpose Gemini Mode (Default)
./target/release/gemma-agent

# Run in Autonomous Inner Monologue Mode
./target/release/gemma-agent --mode automatic

# Or run in Vibe-Coding Mode
./target/release/gemma-agent --mode coder
```

---

## 💬 REPL Command Reference

- `/speech` — Toggle Text-to-Speech (TTS) audio output on/off (Disabled by default).
- `/help` — Show command help menu.
- `/automatic` or `/auto` — Switch instantly into Autonomous Inner Monologue Mode.
- `/mode [general|coder|automatic]` — Switch between General, Coding, and Automatic Inner Monologue Modes.
- `/status` — View connection telemetry, active mode, token count & loaded workspace rules.
- `/save [name]` — Save session history to `.gemma/sessions/`.
- `/load [name]` — Restore a saved session file.
- `/sessions` — List all saved session files.
- `/reset` or `/clear` — Reset context back to clean system prompt.
- `/compact [tokens]` — Compact & summarize past conversation history.
- `/exit` or `/quit` — Exit the agent.
