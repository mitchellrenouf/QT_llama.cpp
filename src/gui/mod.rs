#[allow(dead_code)]
pub mod bridge;

#[allow(unused_imports, dead_code)]
pub use bridge::*;

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

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

    let mut child = Command::new(qml_runner)
        .arg(&qml_file)
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn Qt6 QML application: {}", e))?;

    let status = child.wait()?;
    if !status.success() {
        return Err(anyhow!("Qt6 QML application exited with code: {:?}", status.code()));
    }

    Ok(())
}
