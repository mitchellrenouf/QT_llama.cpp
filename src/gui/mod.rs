#[allow(dead_code)]
pub mod bridge;

#[allow(unused_imports, dead_code)]
pub use bridge::*;

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncBufReadExt;

pub fn find_qml_entrypoint(workspace_root: &Path) -> Option<PathBuf> {
    let candidates = [
        workspace_root.join("qml").join("Main.qml"),
        PathBuf::from("qml/Main.qml"),
        PathBuf::from("/app/share/gemma/qml/Main.qml"),
        PathBuf::from("/usr/share/gemma/qml/Main.qml"),
    ];

    for candidate in candidates {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn find_qml_runner() -> Option<PathBuf> {
    let candidate_names = [
        "qml6",
        "qml",
        "/usr/lib/qt6/bin/qml",
        "qmlscene",
        "/usr/lib/qt6/bin/qmlscene",
    ];

    for name in candidate_names {
        if let Some(p) = crate::tools::desktop::is_executable_in_path(name) {
            return Some(p);
        }
        let p = PathBuf::from(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

pub fn is_display_available() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok() || std::env::var("DISPLAY").is_ok()
}

pub async fn launch_qt_gui(workspace_root: &Path) -> Result<()> {
    if !is_display_available() {
        return Err(anyhow!("No graphical display server found (WAYLAND_DISPLAY or DISPLAY is unset)."));
    }

    let qml_file = find_qml_entrypoint(workspace_root)
        .ok_or_else(|| anyhow!("Could not find qml/Main.qml application file."))?;

    let qml_runner = find_qml_runner()
        .ok_or_else(|| anyhow!("Could not locate Qt6 QML runner (checked: qml6, qml, /usr/lib/qt6/bin/qml, qmlscene)."))?;

    println!("🎨 Launching Gemma 4 Qt6 Interface ({}) with {}...", qml_file.display(), qml_runner.display());

    let mut child = tokio::process::Command::new(qml_runner)
        .arg(&qml_file)
        .env("QT_QUICK_CONTROLS_STYLE", "Basic")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn Qt6 QML application: {}", e))?;

    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Some(json_str) = line.strip_prefix("JSON_IPC:") {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                        let cmd_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if cmd_type == "load_hf_model" {
                            if let Some(spec_str) = val.get("spec").and_then(|v| v.as_str()) {
                                if let Ok(spec) = crate::hf::HfModelSpec::parse(spec_str) {
                                    tokio::spawn(async move {
                                        let _ = crate::hf::resolve_or_fetch_hf_model(&spec, |msg, _p, _idx, _tot| {
                                            println!("[Hugging Face Sync] {}", msg);
                                        }).await;
                                    });
                                }
                            }
                        }
                    }
                } else if !line.trim().is_empty() {
                    println!("{}", line);
                }
            }
        });
    }

    let status = child.wait().await?;
    if !status.success() {
        return Err(anyhow!("Qt6 QML application exited with code: {:?}", status.code()));
    }

    Ok(())
}
