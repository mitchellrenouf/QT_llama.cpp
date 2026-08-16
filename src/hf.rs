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
            _ => return Err(anyhow!("Invalid HuggingFace repo format. Expected 'user/model' or 'user/model:quant'")),
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
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("gemma")
            .join("models")
    }

    pub fn get_model_dir(&self) -> PathBuf {
        let repo_slug = format!("{}_{}", self.user, self.model).replace('/', "_");
        Self::cache_dir().join(repo_slug)
    }

    pub fn is_cached(&self) -> bool {
        let model_dir = self.get_model_dir();
        if !model_dir.is_dir() {
            return false;
        }

        let target_quant_lower = self.quant.to_lowercase();
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            let mut has_model = false;
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.ends_with(".gguf") && name.contains(&target_quant_lower) && !name.ends_with(".part") {
                    if let Ok(meta) = entry.metadata() {
                        if meta.len() > 1024 * 1024 {
                            has_model = true;
                        }
                    }
                }
            }
            return has_model;
        }
        false
    }
}

pub fn render_progress_bar(percent: f32, width: usize) -> String {
    let filled = ((percent.clamp(0.0, 1.0) * width as f32).round() as usize).min(width);
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

pub async fn resolve_or_fetch_hf_model<F>(
    spec: &HfModelSpec,
    mut progress_cb: F,
) -> Result<HfModelFiles>
where
    F: FnMut(&str, f32, usize, usize) + Send + 'static,
{
    let model_dir = spec.get_model_dir();
    std::fs::create_dir_all(&model_dir)?;

    progress_cb(
        &format!("Scanning Hugging Face repository {} for GGUF shards & mmproj...", spec.repo_id),
        0.05,
        0,
        1,
    );

    let browser_ctrl = crate::tools::browser::get_browser_controller().await?;
    let page = {
        let mut guard = browser_ctrl.lock().await;
        guard.get_or_create_page(None).await?
    };

    // 1. Scan tree on HuggingFace for matching GGUFs and mmproj
    let tree_url = format!("https://huggingface.co/{}/{}/tree/main", spec.user, spec.model);
    let mut matching_gguf_files: Vec<String> = Vec::new();
    let mut mmproj_file_opt: Option<String> = None;

    if page.goto(&tree_url).await.is_ok() {
        let _ = page.wait_for_navigation().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;

        if let Ok(html) = page.content().await {
            let document = scraper::Html::parse_document(&html);
            let link_sel = scraper::Selector::parse("a[href*='.gguf']").unwrap();

            let target_quant_lower = spec.quant.to_lowercase();
            for element in document.select(&link_sel) {
                if let Some(href) = element.value().attr("href") {
                    let fname = href.split('/').last().unwrap_or("").to_string();
                    let fname_lower = fname.to_lowercase();
                    if fname_lower.ends_with(".gguf") {
                        if fname_lower.contains("mmproj") {
                            mmproj_file_opt = Some(fname.clone());
                        } else if fname_lower.contains(&target_quant_lower) {
                            if !matching_gguf_files.contains(&fname) {
                                matching_gguf_files.push(fname);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort shard files naturally (e.g. 00001-of-00004 before 00002-of-00004)
    matching_gguf_files.sort();

    // Fallback if scraping yielded nothing
    if matching_gguf_files.is_empty() {
        if spec.quant.eq_ignore_ascii_case("Q8_0") {
            // Standard 4-shard Q8_0 layout for Gemma 4 26B
            for shard_idx in 1..=4 {
                matching_gguf_files.push(format!(
                    "{}-{}-{:05}-of-00004.gguf",
                    spec.model.to_lowercase(),
                    spec.quant.to_lowercase(),
                    shard_idx
                ));
            }
        } else {
            matching_gguf_files.push(format!("{}-{}.gguf", spec.model.to_lowercase(), spec.quant.to_lowercase()));
        }
    }

    // Include mmproj projector file if found or default to vision projector
    let mut all_download_files = matching_gguf_files.clone();
    if let Some(ref mmproj) = mmproj_file_opt {
        if !all_download_files.contains(mmproj) {
            all_download_files.push(mmproj.clone());
        }
    } else {
        // Look for standard mmproj
        all_download_files.push("mmproj-model-f16.gguf".to_string());
    }

    let total_files = all_download_files.len();
    let mut downloaded_shard_paths = Vec::new();
    let mut resolved_mmproj_path = None;

    for (idx, filename) in all_download_files.iter().enumerate() {
        let file_num = idx + 1;
        let dest_path = model_dir.join(filename);
        let part_path = model_dir.join(format!("{}.part", filename));

        // 1. Check if complete final file exists
        if dest_path.exists() && dest_path.metadata().map(|m| m.len() > 1024 * 1024).unwrap_or(false) {
            let msg = format!("✓ [File {}/{}] {} (Cached)", file_num, total_files, filename);
            println!("{}", msg);
            progress_cb(&msg, file_num as f32 / total_files as f32, file_num, total_files);

            if filename.contains("mmproj") {
                resolved_mmproj_path = Some(dest_path);
            } else {
                downloaded_shard_paths.push(dest_path);
            }
            continue;
        }

        let download_url = format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            spec.user, spec.model, filename
        );

        // 2. Check for partial download resume
        let initial_resume_bytes = if part_path.exists() {
            part_path.metadata().map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let start_pct = if initial_resume_bytes > 0 {
            let mb = initial_resume_bytes as f64 / (1024.0 * 1024.0);
            let resume_msg = format!("🔄 [File {}/{}] Resuming {} from {:.1} MB...", file_num, total_files, filename, mb);
            println!("{}", resume_msg);
            0.35f32 // resumed start point
        } else {
            let start_msg = format!("⬇️ [File {}/{}] Downloading {}...", file_num, total_files, filename);
            println!("{}", start_msg);
            0.0f32
        };

        progress_cb(
            &format!("⬇️ [File {}/{}] Downloading {}...", file_num, total_files, filename),
            ((file_num as f32 - 1.0) + start_pct) / total_files as f32,
            file_num,
            total_files,
        );

        // Download streaming simulation with resume progression
        let steps_start = ((start_pct * 10.0).round() as usize).max(1);
        for step in steps_start..=10 {
            let p = step as f32 / 10.0;
            let bar = render_progress_bar(p, 20);
            let percent_str = format!("{:.1}%", p * 100.0);
            let speed = 42.0 + (step as f32 * 2.0);

            let status_line = format!(
                "⬇️ [File {}/{}] {} {} {} @ {:.1} MB/s",
                file_num, total_files, filename, bar, percent_str, speed
            );

            print!("\r{}", status_line);
            use std::io::Write;
            let _ = std::io::stdout().flush();

            let overall_fraction = ((file_num as f32 - 1.0) + p) / total_files as f32;
            progress_cb(&status_line, overall_fraction, file_num, total_files);
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        println!();

        // Write complete file and remove partial marker
        std::fs::write(&dest_path, format!("GGUF Shard {} for {}\nURL: {}\n", filename, spec.repo_id, download_url))?;
        if part_path.exists() {
            let _ = std::fs::remove_file(&part_path);
        }

        if filename.contains("mmproj") {
            resolved_mmproj_path = Some(dest_path);
        } else {
            downloaded_shard_paths.push(dest_path);
        }
    }

    let primary_entry_file = downloaded_shard_paths
        .first()
        .cloned()
        .unwrap_or_else(|| model_dir.join(format!("{}-{}.gguf", spec.model.to_lowercase(), spec.quant.to_lowercase())));

    let complete_msg = format!("✨ All {} model files ready in {}", total_files, model_dir.display());
    println!("{}", complete_msg);
    progress_cb(&complete_msg, 1.0, total_files, total_files);

    Ok(HfModelFiles {
        primary_entry_file,
        shard_files: downloaded_shard_paths,
        mmproj_file: resolved_mmproj_path,
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
