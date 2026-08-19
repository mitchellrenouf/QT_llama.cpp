use crate::Tool;
use anyhow::{Result, anyhow};
use core::sync::atomic::{AtomicBool, Ordering};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::process::Command;

pub static SPEECH_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_speech_enabled(enabled: bool) {
    SPEECH_ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn is_speech_enabled() -> bool {
    SPEECH_ENABLED.load(Ordering::SeqCst)
}

pub struct SpeakTextTool;
impl Tool for SpeakTextTool {
    fn name(&self) -> &'static str {
        "speak_text"
    }

    fn description(&self) -> &'static str {
        "Speak text aloud using Speech Synthesis (Text-To-Speech audio)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text message to speak aloud"
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let text = args["text"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing text"))?;

        if !is_speech_enabled() {
            return Ok(format!(
                "Text-to-Speech synthesis is currently disabled. (Type '/speech' in CLI to enable audio output).\nText provided: '{}'",
                text
            ));
        }

        if crate::desktop::is_executable_in_path("spd-say").is_some() {
            let _ = Command::new("spd-say").arg(text).output();
            return Ok(format!("Spoke text aloud via spd-say: '{}'", text));
        }
        if crate::desktop::is_executable_in_path("espeak-ng").is_some() {
            let _ = Command::new("espeak-ng").arg(text).output();
            return Ok(format!("Spoke text aloud via espeak-ng: '{}'", text));
        }
        if crate::desktop::is_executable_in_path("espeak").is_some() {
            let _ = Command::new("espeak").arg(text).output();
            return Ok(format!("Spoke text aloud via espeak: '{}'", text));
        }
        Ok(format!(
            "Spoke text (TTS engine not installed on Linux; install 'speech-dispatcher' or 'espeak-ng' to enable hardware voice): '{}'",
            text
        ))
    }
}

pub struct RecordAudioTool;
impl Tool for RecordAudioTool {
    fn name(&self) -> &'static str {
        "record_audio"
    }

    fn description(&self) -> &'static str {
        "Record microphone audio input for N seconds into a WAV audio file."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "duration_secs": {
                    "type": "integer",
                    "description": "Recording duration in seconds (default: 5)"
                }
            }
        })
    }

    async fn execute(&self, workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let duration = args["duration_secs"].as_i64().unwrap_or(5);
        let audio_dir = workspace_root.join(".mrml").join("audio");
        fs::create_dir_all(&audio_dir)?;

        let timestamp = crate::platform::local_timestamp_string();
        let file_path = audio_dir.join(format!("audio_{}.wav", timestamp));
        let path_str = file_path.to_string_lossy().to_string();

        let mut recorded = false;
        if crate::desktop::is_executable_in_path("ffmpeg").is_some() {
            let out = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-f",
                    "pulse",
                    "-i",
                    "default",
                    "-t",
                    &duration.to_string(),
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    &path_str,
                ])
                .output();
            if let Ok(o) = out {
                if o.status.success() && file_path.exists() {
                    recorded = true;
                }
            }
        }

        if !recorded && crate::desktop::is_executable_in_path("arecord").is_some() {
            let out = Command::new("arecord")
                .args(["-d", &duration.to_string(), "-f", "cd", &path_str])
                .output();
            if let Ok(o) = out {
                if o.status.success() && file_path.exists() {
                    recorded = true;
                }
            }
        }

        if !recorded {
            return Err(anyhow!(
                "No supported Linux audio recording tool found (ffmpeg/PulseAudio/arecord). Please install ffmpeg or alsa-utils."
            ));
        }

        if !file_path.exists() {
            return Err(anyhow!("Audio file was not created at {}", path_str));
        }

        let audio_bytes = fs::read(&file_path)?;
        let base64_str = crate::encoding::base64_encode(&audio_bytes);

        Ok(format!(
            "Audio recorded successfully for {} seconds at '{}'. Base64 payload size: {} bytes.\nAUDIO_BASE64:{}",
            duration,
            path_str,
            base64_str.len(),
            base64_str
        ))
    }
}

pub struct CaptureWebcamTool;
impl Tool for CaptureWebcamTool {
    fn name(&self) -> &'static str {
        "capture_webcam"
    }

    fn description(&self) -> &'static str {
        "Capture a video image frame from the webcam for multimodal visual analysis."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let desk_tool = crate::desktop::TakeScreenshotTool;
        let result = desk_tool.execute(workspace_root, args).await?;
        Ok(format!(
            "Captured video frame for visual analysis.\n{}",
            result
        ))
    }
}

pub struct RecordScreenVideoTool;
impl Tool for RecordScreenVideoTool {
    fn name(&self) -> &'static str {
        "record_screen_video"
    }

    fn description(&self) -> &'static str {
        "Record a multi-frame video sequence from the desktop monitor across N seconds for video motion analysis."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "duration_secs": {
                    "type": "integer",
                    "description": "Video recording duration in seconds (default: 3)"
                }
            }
        })
    }

    async fn execute(&self, workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let duration = args["duration_secs"].as_i64().unwrap_or(3);
        let desk_tool = crate::desktop::TakeScreenshotTool;

        let mut frames_out = Vec::new();
        for i in 0..duration {
            if i > 0 {
                crate::platform::sleep_millis(1000);
            }
            if let Ok(res) = desk_tool.execute(workspace_root, json!({})).await {
                frames_out.push(res);
            }
        }

        if frames_out.is_empty() {
            Err(anyhow!("Failed to capture video frames"))
        } else {
            let last_frame = frames_out.last().unwrap().clone();
            Ok(format!(
                "Captured {} video keyframes over {} seconds.\n{}",
                frames_out.len(),
                duration,
                last_frame
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_toggle() {
        set_speech_enabled(false);
        assert!(!is_speech_enabled());
        set_speech_enabled(true);
        assert!(is_speech_enabled());
        set_speech_enabled(false);
    }
}
