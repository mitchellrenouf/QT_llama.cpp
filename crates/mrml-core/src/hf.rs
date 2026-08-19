use anyhow::{anyhow, Result};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfModelSpec {
    pub repo_id: String,
    pub user: String,
    pub model: String,
    pub quant: String,
}

#[derive(Debug, Clone)]
pub struct HfModelFiles {
    pub primary_entry_file: PathBuf,
    pub shard_files: Vec<PathBuf>,
    pub mmproj_file: Option<PathBuf>,
    pub speedup_draft_file: Option<PathBuf>,
}

impl HfModelSpec {
    pub fn parse(input: &str) -> Result<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("Empty HuggingFace model specification"));
        }

        let (repo_part, quant_part) = if let Some(idx) = trimmed.find(':') {
            (&trimmed[..idx], &trimmed[idx + 1..])
        } else {
            (trimmed, "Q4_0")
        };

        let parts: Vec<&str> = repo_part.split('/').collect();
        let (user, model) = match parts.len() {
            1 => ("ggml-org", parts[0]),
            2 => (parts[0], parts[1]),
            _ => {
                return Err(anyhow!(
                    "Invalid HuggingFace repo format. Expected 'user/model' or 'user/model:quant'"
                ))
            }
        };

        let repo_id = format!("{}/{}", user, model);
        let quant = if quant_part.is_empty() {
            "Q4_0".to_string()
        } else {
            quant_part.to_uppercase()
        };

        Ok(Self {
            repo_id,
            user: user.to_string(),
            model: model.to_string(),
            quant,
        })
    }

    pub fn default_gemma_4_26b() -> Self {
        Self {
            repo_id: "ggml-org/gemma-4-26B-A4B-it-GGUF".to_string(),
            user: "ggml-org".to_string(),
            model: "gemma-4-26B-A4B-it-GGUF".to_string(),
            quant: "Q4_0".to_string(),
        }
    }

    pub fn cache_dir() -> PathBuf {
        if let Ok(path) = std::env::var("HF_HUB_CACHE") {
            return PathBuf::from(path);
        }
        if let Ok(hf_home) = std::env::var("HF_HOME") {
            return PathBuf::from(hf_home).join("hub");
        }
        if let Ok(mrml_cache) = std::env::var("MRML_CACHE") {
            return PathBuf::from(mrml_cache);
        }

        #[cfg(windows)]
        {
            if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
                let p = PathBuf::from(local_appdata).join("huggingface").join("hub");
                return p;
            }
        }

        if let Some(home) = crate::platform::home_dir() {
            return home.join(".cache").join("huggingface").join("hub");
        }

        PathBuf::from(".cache").join("huggingface").join("hub")
    }

    pub fn get_model_dir(&self) -> PathBuf {
        let repo_slug = format!("models--{}--{}", self.user, self.model);
        Self::cache_dir().join(repo_slug)
    }

    pub fn is_cached(&self) -> bool {
        let model_dir = self.get_model_dir();
        if !model_dir.is_dir() {
            return false;
        }

        let target_quant_lower = self.quant.to_lowercase();
        for path in crate::fs_walk::paths(&model_dir) {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if name.ends_with(".gguf")
                && name.contains(&target_quant_lower)
                && !name.ends_with(".part")
                && !name.contains("mmproj")
                && !name.contains("mtp")
            {
                if let Ok(meta) = path.metadata() {
                    if meta.len() > 10 * 1024 * 1024 {
                        // > 10MB
                        return true;
                    }
                }
            }
        }
        false
    }
}

pub fn render_progress_bar(percent: f32, width: usize) -> String {
    let filled = ((percent.clamp(0.0, 1.0) * width as f32).round() as usize).min(width);
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

pub async fn query_hf_api_siblings(spec: &HfModelSpec) -> Result<Vec<String>> {
    let api_url = format!(
        "https://huggingface.co/api/models/{}/{}",
        spec.user, spec.model
    );

    let mut cmd = std::process::Command::new("curl");
    cmd.arg("-s").arg("-L").arg(&api_url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        cmd.arg("-H")
            .arg(format!("Authorization: Bearer {}", token));
    }

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(anyhow!("Failed to query Hugging Face API at {}", api_url));
    }

    let body = String::from_utf8_lossy(&output.stdout);
    let val: serde_json::Value = serde_json::from_str(&body)?;

    let mut filenames = Vec::new();
    if let Some(siblings) = val.get("siblings").and_then(|s| s.as_array()) {
        for item in siblings {
            if let Some(rfilename) = item.get("rfilename").and_then(|r| r.as_str()) {
                if rfilename.ends_with(".gguf") {
                    filenames.push(rfilename.to_string());
                }
            }
        }
    }

    Ok(filenames)
}

pub async fn resolve_or_fetch_hf_model<F>(
    spec: &HfModelSpec,
    mut progress_cb: F,
) -> Result<HfModelFiles>
where
    F: FnMut(&str, f32, usize, usize) + Send + 'static,
{
    // 1. First check if model already exists in HF hub cache (~/.cache/huggingface/hub/) or local cache
    if let Some(existing_primary) =
        crate::client::find_model_file(&format!("{}:{}", spec.repo_id, spec.quant))
            .or_else(|| crate::client::find_model_file(&spec.model))
    {
        progress_cb(
            &format!(
                "✓ Found local cached weights at {}",
                existing_primary
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ),
            1.0,
            1,
            1,
        );
        return Ok(HfModelFiles {
            primary_entry_file: existing_primary.clone(),
            shard_files: vec![existing_primary],
            mmproj_file: None,
            speedup_draft_file: None,
        });
    }

    let model_dir = spec.get_model_dir();
    std::fs::create_dir_all(&model_dir)?;

    progress_cb(
        &format!(
            "Querying Hugging Face for model shards, mmproj & MTP speedup in {}...",
            spec.repo_id
        ),
        0.05,
        0,
        1,
    );

    let siblings = query_hf_api_siblings(spec).await.unwrap_or_default();

    let target_quant_lower = spec.quant.to_lowercase();
    let mut matching_shards = Vec::new();
    let mut mmproj_file_opt = None;
    let mut speedup_file_opt = None;

    for fname in &siblings {
        let fname_lower = fname.to_lowercase();
        if fname_lower.contains("mmproj") {
            if mmproj_file_opt.is_none() || fname_lower.contains(&target_quant_lower) {
                mmproj_file_opt = Some(fname.clone());
            }
        } else if fname_lower.contains("mtp") || fname_lower.contains("dflash") {
            if speedup_file_opt.is_none() || fname_lower.contains(&target_quant_lower) {
                speedup_file_opt = Some(fname.clone());
            }
        } else if fname_lower.contains(&target_quant_lower) {
            matching_shards.push(fname.clone());
        }
    }

    matching_shards.sort();

    // Fallbacks if not found via API
    if matching_shards.is_empty() {
        if spec.quant.eq_ignore_ascii_case("Q8_0") {
            for shard_idx in 1..=4 {
                matching_shards.push(format!(
                    "{}-{}-{:05}-of-00004.gguf",
                    spec.model.to_lowercase(),
                    spec.quant.to_lowercase(),
                    shard_idx
                ));
            }
        } else {
            matching_shards.push(format!(
                "{}-{}.gguf",
                spec.model.to_lowercase(),
                spec.quant.to_lowercase()
            ));
        }
    }

    let mut download_queue = matching_shards.clone();
    if let Some(ref mmproj) = mmproj_file_opt {
        if !download_queue.contains(mmproj) {
            download_queue.push(mmproj.clone());
        }
    }
    if let Some(ref speedup) = speedup_file_opt {
        if !download_queue.contains(speedup) {
            download_queue.push(speedup.clone());
        }
    }

    let total_files = download_queue.len();
    let mut downloaded_shard_paths = Vec::new();
    let mut resolved_mmproj_path = None;
    let mut resolved_speedup_path = None;

    for (idx, filename) in download_queue.iter().enumerate() {
        let file_num = idx + 1;
        let dest_path = model_dir.join(filename);
        let part_path = model_dir.join(format!("{}.part", filename));

        if dest_path.exists()
            && dest_path
                .metadata()
                .map(|m| m.len() > 10 * 1024 * 1024)
                .unwrap_or(false)
        {
            let msg = format!(
                "✓ [File {}/{}] {} (Cached)",
                file_num, total_files, filename
            );
            println!("{}", msg);
            progress_cb(
                &msg,
                file_num as f32 / total_files as f32,
                file_num,
                total_files,
            );

            if filename.contains("mmproj") {
                resolved_mmproj_path = Some(dest_path);
            } else if filename.contains("mtp") || filename.contains("dflash") {
                resolved_speedup_path = Some(dest_path);
            } else {
                downloaded_shard_paths.push(dest_path);
            }
            continue;
        }

        let download_url = format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            spec.user, spec.model, filename
        );

        let initial_resume_bytes = if part_path.exists() {
            part_path.metadata().map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        if initial_resume_bytes > 0 {
            let mb = initial_resume_bytes as f64 / (1024.0 * 1024.0);
            println!(
                "🔄 [File {}/{}] Resuming {} from {:.1} MB...",
                file_num, total_files, filename, mb
            );
        } else {
            println!(
                "⬇️ [File {}/{}] Downloading {}...",
                file_num, total_files, filename
            );
        }

        // Run real streaming curl download with resume support
        let mut curl_cmd = std::process::Command::new("curl");
        curl_cmd
            .arg("-f")
            .arg("-sS")
            .arg("-L")
            .arg("-C")
            .arg("-") // Auto-resume from partial byte offset
            .arg("-o")
            .arg(&part_path)
            .arg(&download_url);

        if let Ok(token) = std::env::var("HF_TOKEN") {
            curl_cmd
                .arg("-H")
                .arg(format!("Authorization: Bearer {}", token));
        }

        let mut child = curl_cmd
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()?;

        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    return Err(anyhow!("Failed to download {}. Check network or HF_TOKEN.", filename));
                }
                break;
            }
            crate::platform::sleep_millis(500);
            if let Ok(meta) = part_path.metadata() {
                let cur_len = meta.len();
                let mb = cur_len as f64 / (1024.0 * 1024.0);
                let msg = format!("⬇️ [File {}/{}] {} ({:.1} MB downloaded)...", file_num, total_files, filename, mb);
                let file_pct = (mb / 4000.0).clamp(0.05, 0.95) as f32;
                let overall = ((file_num as f32 - 1.0) + file_pct) / total_files as f32;
                progress_cb(&msg, overall, file_num, total_files);
            }
        }

        // Successfully downloaded: promote .part to final .gguf
        if part_path.exists() {
            std::fs::rename(&part_path, &dest_path)?;
        }

        let done_msg = format!(
            "✓ [File {}/{}] {} (Downloaded)",
            file_num, total_files, filename
        );
        println!("{}", done_msg);
        progress_cb(
            &done_msg,
            file_num as f32 / total_files as f32,
            file_num,
            total_files,
        );

        if filename.contains("mmproj") {
            resolved_mmproj_path = Some(dest_path);
        } else if filename.contains("mtp") || filename.contains("dflash") {
            resolved_speedup_path = Some(dest_path);
        } else {
            downloaded_shard_paths.push(dest_path);
        }
    }

    let primary_entry_file = downloaded_shard_paths.first().cloned().unwrap_or_else(|| {
        model_dir.join(format!(
            "{}-{}.gguf",
            spec.model.to_lowercase(),
            spec.quant.to_lowercase()
        ))
    });

    let complete_msg = format!(
        "✨ All {} model weights ready in {}",
        total_files,
        model_dir.display()
    );
    println!("{}", complete_msg);
    progress_cb(&complete_msg, 1.0, total_files, total_files);

    Ok(HfModelFiles {
        primary_entry_file,
        shard_files: downloaded_shard_paths,
        mmproj_file: resolved_mmproj_path,
        speedup_draft_file: resolved_speedup_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hf_spec_full() {
        let spec = HfModelSpec::parse("ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0").unwrap();
        assert_eq!(spec.user, "ggml-org");
        assert_eq!(spec.model, "gemma-4-26B-A4B-it-GGUF");
        assert_eq!(spec.quant, "Q4_0");
    }

    #[test]
    fn test_parse_hf_spec_sharded_q8() {
        let spec = HfModelSpec::parse("ggml-org/gemma-4-26B-A4B-it-GGUF:Q8_0").unwrap();
        assert_eq!(spec.user, "ggml-org");
        assert_eq!(spec.model, "gemma-4-26B-A4B-it-GGUF");
        assert_eq!(spec.quant, "Q8_0");
    }

    #[test]
    fn test_render_progress_bar() {
        let bar_half = render_progress_bar(0.5, 10);
        assert_eq!(bar_half, "[█████░░░░░]");
        let bar_full = render_progress_bar(1.0, 10);
        assert_eq!(bar_full, "[██████████]");
    }
}
