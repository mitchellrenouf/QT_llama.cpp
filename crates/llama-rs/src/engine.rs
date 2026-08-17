use anyhow::{anyhow, Result};
use qtensor::QTensorEngine;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

enum EngineBackend {
    LlamaCpp(Arc<LlamaCppServer>),
    QTensor(Arc<QTensorEngine>),
}

struct LlamaCppServer {
    port: u16,
    child: Mutex<Child>,
}

impl Drop for LlamaCppServer {
    fn drop(&mut self) {
        if let Ok(child) = self.child.get_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[derive(Clone)]
pub struct LlamaEngine {
    inner: Arc<EngineBackend>,
}

impl LlamaEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, n_gpu_layers: i32, ctx_size: u32, backend: Option<&str>) -> Result<Self> {
        let model_path = model_path.as_ref();
        if !matches!(backend, Some("cpu")) {
            if let Some(server_path) = find_llama_server() {
                match LlamaCppServer::start(&server_path, model_path, n_gpu_layers, ctx_size) {
                    Ok(server) => {
                        eprintln!("[llama.cpp] CUDA graph backend active on port {}", server.port);
                        return Ok(Self { inner: Arc::new(EngineBackend::LlamaCpp(Arc::new(server))) });
                    }
                    Err(error) => eprintln!("[llama.cpp] optimized backend unavailable: {error}; using qtensor"),
                }
            }
        }

        let inner = QTensorEngine::new(model_path, ctx_size as usize)?;
        Ok(Self { inner: Arc::new(EngineBackend::QTensor(Arc::new(inner))) })
    }

    pub fn generate_stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        match self.inner.as_ref() {
            EngineBackend::LlamaCpp(server) => server.generate_stream(prompt, max_tokens, temperature),
            EngineBackend::QTensor(engine) => engine.generate_stream(prompt, max_tokens, temperature),
        }
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        match self.inner.as_ref() {
            EngineBackend::QTensor(engine) => engine.tokenize(text, add_special),
            EngineBackend::LlamaCpp(_) => Ok(text.as_bytes().iter().map(|byte| *byte as i32).collect()),
        }
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        match self.inner.as_ref() {
            EngineBackend::QTensor(engine) => engine.token_to_piece(token),
            EngineBackend::LlamaCpp(_) => Ok(char::from_u32(token as u32).unwrap_or_default().to_string()),
        }
    }

    pub fn is_eog(&self, token: i32) -> bool {
        match self.inner.as_ref() {
            EngineBackend::QTensor(engine) => engine.is_eog(token),
            EngineBackend::LlamaCpp(_) => token == 1 || token == 2,
        }
    }
}

impl LlamaCppServer {
    fn start(executable: &Path, model: &Path, n_gpu_layers: i32, ctx_size: u32) -> Result<Self> {
        let port = TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port();
        let gpu_layers = if n_gpu_layers < 0 { 99 } else { n_gpu_layers };
        let mut command = Command::new(executable);
        command
            .args(["--model", &model.to_string_lossy()])
            .args(["--n-gpu-layers", &gpu_layers.to_string()])
            .args(["--ctx-size", &ctx_size.to_string()])
            .args(["--flash-attn", "on"])
            .args(["--host", "127.0.0.1"])
            .args(["--port", &port.to_string()])
            .arg("--no-webui")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        let mut child = command.spawn()?;

        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if server_ready(port) {
                break;
            }
            if let Some(status) = child.try_wait()? {
                return Err(anyhow!("llama-server exited during startup with {status}"));
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                return Err(anyhow!("timed out loading llama.cpp model"));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(Self { port, child: Mutex::new(child) })
    }

    fn generate_stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(4096);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel = cancelled.clone();
        let port = self.port;
        let prompt = prompt.to_owned();
        std::thread::spawn(move || {
            if let Err(error) = stream_completion(port, &prompt, max_tokens, temperature, &cancel, &tx) {
                let _ = tx.blocking_send(Err(error));
            }
        });
        (rx, cancelled)
    }
}

fn server_ready(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream.write_all(format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n").as_bytes()).is_err() {
        return false;
    }
    let mut status = String::new();
    BufReader::new(stream).read_line(&mut status).is_ok() && status.contains(" 200 ")
}

fn stream_completion(port: u16, prompt: &str, max_tokens: usize, temperature: f32, cancelled: &AtomicBool, tx: &mpsc::Sender<Result<String>>) -> Result<()> {
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": true,
        "cache_prompt": true
    }).to_string();
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.write_all(format!(
        "POST /completion HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    ).as_bytes())?;
    stream.flush()?;

    for line in BufReader::new(stream).lines() {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            break;
        }
        let event: serde_json::Value = serde_json::from_str(data)?;
        if let Some(content) = event.get("content").and_then(|value| value.as_str()) {
            if !content.is_empty() && tx.blocking_send(Ok(content.to_owned())).is_err() {
                break;
            }
        }
    }
    Ok(())
}

fn find_llama_server() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LLAMA_SERVER_EXE").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let executable_name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    let cwd = std::env::current_dir().ok()?;
    [
        cwd.join("llama.cpp/build-codex-ninja/bin").join(executable_name),
        cwd.join("../llama.cpp/build-codex-ninja/bin").join(executable_name),
        cwd.join("build-codex-ninja/bin").join(executable_name),
    ].into_iter().find(|path| path.is_file())
}
