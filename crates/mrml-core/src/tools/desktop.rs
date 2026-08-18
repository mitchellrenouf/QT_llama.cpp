use crate::tools::Tool;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn is_executable_in_path(cmd: &str) -> Option<PathBuf> {
    if let Ok(path_var) = std::env::var("PATH") {
        let separator = if cfg!(windows) { ';' } else { ':' };
        for dir in path_var.split(separator) {
            let p = Path::new(dir).join(cmd);
            if p.is_file() {
                return Some(p);
            }
            if cfg!(windows) && !cmd.ends_with(".exe") {
                let p_exe = Path::new(dir).join(format!("{}.exe", cmd));
                if p_exe.is_file() {
                    return Some(p_exe);
                }
            }
        }
    }
    None
}

pub struct TakeScreenshotTool;
#[async_trait]
impl Tool for TakeScreenshotTool {
    fn name(&self) -> &'static str {
        "take_screenshot"
    }

    fn description(&self) -> &'static str {
        "Capture a downscaled, compressed screenshot of the active desktop monitor for vision analysis (optimized context size)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, workspace_root: &Path, _args: serde_json::Value) -> Result<String> {
        let shot_dir = workspace_root.join(".mrml").join("screenshots");
        fs::create_dir_all(&shot_dir)?;

        let timestamp = crate::platform::local_timestamp_string();
        let file_path = shot_dir.join(format!("screenshot_{}.jpg", timestamp));
        let path_str = file_path.to_string_lossy().to_string();

        let temp_png = shot_dir.join(format!("temp_screenshot_{}.png", timestamp));
        let temp_png_str = temp_png.to_string_lossy().to_string();

        let mut captured = false;
        if is_executable_in_path("spectacle").is_some() {
            let out = Command::new("spectacle")
                .args(["-b", "-n", "-o", &temp_png_str])
                .output();
            if let Ok(o) = out {
                if o.status.success() && temp_png.exists() {
                    captured = true;
                }
            }
        }

        if !captured && is_executable_in_path("grim").is_some() {
            let out = Command::new("grim").arg(&temp_png_str).output();
            if let Ok(o) = out {
                if o.status.success() && temp_png.exists() {
                    captured = true;
                }
            }
        }

        if !captured && is_executable_in_path("scrot").is_some() {
            let out = Command::new("scrot").arg(&temp_png_str).output();
            if let Ok(o) = out {
                if o.status.success() && temp_png.exists() {
                    captured = true;
                }
            }
        }

        if !captured && is_executable_in_path("maim").is_some() {
            let out = Command::new("maim").arg(&temp_png_str).output();
            if let Ok(o) = out {
                if o.status.success() && temp_png.exists() {
                    captured = true;
                }
            }
        }

        if !captured && is_executable_in_path("import").is_some() {
            let out = Command::new("import")
                .args(["-window", "root", &temp_png_str])
                .output();
            if let Ok(o) = out {
                if o.status.success() && temp_png.exists() {
                    captured = true;
                }
            }
        }

        if !captured {
            return Err(anyhow!(
                "No supported Linux screenshot utility found (checked: spectacle, grim, scrot, maim, import). Please install one via your package manager."
            ));
        }

        if is_executable_in_path("ffmpeg").is_some() {
            let _ = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-i",
                    &temp_png_str,
                    "-vf",
                    "scale='min(1024,iw)':-2",
                    "-q:v",
                    "4",
                    &path_str,
                ])
                .output();
            let _ = fs::remove_file(&temp_png);
        } else {
            let _ = fs::rename(&temp_png, &file_path);
        }

        if !file_path.exists() {
            return Err(anyhow!("Screenshot file was not created at {}", path_str));
        }

        let img_bytes = fs::read(&file_path)?;
        let base64_str = base64::engine::general_purpose::STANDARD.encode(&img_bytes);
        let data_uri = format!("data:image/jpeg;base64,{}", base64_str);

        Ok(format!(
            "Screenshot captured at '{}' (compressed JPEG). Base64 Data URI length: {} bytes.\nDATA_URI:{}",
            path_str,
            base64_str.len(),
            data_uri
        ))
    }
}

pub struct OpenAppTool;
#[async_trait]
impl Tool for OpenAppTool {
    fn name(&self) -> &'static str {
        "open_app"
    }

    fn description(&self) -> &'static str {
        "Launch any desktop application, file, or system utility on Linux (e.g., 'dolphin', 'kate', 'flatpak run com.brave.Browser')."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name or path of the application or Flatpak command (e.g. 'dolphin', 'flatpak run com.brave.Browser')"
                }
            },
            "required": ["app_name"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let app_name = args["app_name"].as_str().ok_or_else(|| anyhow!("Missing app_name"))?;

        if app_name.starts_with("flatpak run ") {
            let parts: Vec<&str> = app_name.split_whitespace().collect();
            Command::new("flatpak")
                .args(&parts[1..])
                .spawn()
                .map_err(|e| anyhow!("Failed to launch Flatpak app '{}': {}", app_name, e))?;
            return Ok(format!("Successfully launched Flatpak app '{}'.", app_name));
        }

        if is_executable_in_path(app_name).is_some() || Path::new(app_name).is_file() {
            Command::new(app_name)
                .spawn()
                .map_err(|e| anyhow!("Failed to spawn application '{}': {}", app_name, e))?;
            Ok(format!("Successfully launched application '{}'.", app_name))
        } else if is_executable_in_path("xdg-open").is_some() {
            let output = Command::new("xdg-open").arg(app_name).spawn();
            match output {
                Ok(_) => Ok(format!("Successfully opened '{}' via xdg-open.", app_name)),
                Err(e) => Err(anyhow!("Failed to open '{}' via xdg-open: {}", app_name, e)),
            }
        } else {
            Command::new(app_name)
                .spawn()
                .map_err(|e| anyhow!("Failed to launch application '{}': {}", app_name, e))?;
            Ok(format!("Successfully launched application '{}'.", app_name))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_executable_in_path() {
        let exe = if cfg!(windows) { "cargo" } else { "sh" };
        let found = is_executable_in_path(exe);
        assert!(found.is_some());
    }
}


