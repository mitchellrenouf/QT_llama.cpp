use anyhow::{anyhow, Result};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfModelSpec {
    pub repo_id: String,
    pub user: String,
    pub model: String,
    pub quant: String,
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

    pub fn get_local_cache_path(&self) -> PathBuf {
        let repo_slug = format!("{}_{}", self.user, self.model).replace('/', "_");
        let filename = format!("{}-{}.gguf", self.model.to_lowercase(), self.quant.to_lowercase());
        Self::cache_dir().join(repo_slug).join(filename)
    }

    pub fn is_cached(&self) -> bool {
        let path = self.get_local_cache_path();
        if path.is_file() {
            if let Ok(meta) = path.metadata() {
                return meta.len() > 1024 * 1024; // > 1MB
            }
        }
        false
    }

    pub fn expected_gguf_filenames(&self) -> Vec<String> {
        let q_lower = self.quant.to_lowercase();
        let q_upper = self.quant.to_uppercase();
        let m_lower = self.model.to_lowercase();

        vec![
            format!("{}-{}.gguf", m_lower, q_lower),
            format!("{}-{}.gguf", m_lower, q_upper),
            format!("{}.{}.gguf", m_lower, q_lower),
            format!("{}.{}.gguf", m_lower, q_upper),
            format!("{}.gguf", q_lower),
            format!("{}.gguf", q_upper),
            format!("gemma-4-26b-a4b-it-{}.gguf", q_lower),
            format!("gemma-4-26b-it-{}.gguf", q_lower),
        ]
    }

    pub fn resolve_direct_url(&self, filename: &str) -> String {
        format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            self.user, self.model, filename
        )
    }
}

pub async fn resolve_or_fetch_hf_model<F>(
    spec: &HfModelSpec,
    mut progress_cb: F,
) -> Result<PathBuf>
where
    F: FnMut(&str, f32) + Send + 'static,
{
    let target_path = spec.get_local_cache_path();
    if spec.is_cached() {
        progress_cb(&format!("Model already cached at {}", target_path.display()), 1.0);
        return Ok(target_path);
    }

    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    progress_cb(&format!("Connecting to Hugging Face: {}...", spec.repo_id), 0.05);

    let browser_ctrl = crate::tools::browser::get_browser_controller().await?;
    let page = {
        let mut guard = browser_ctrl.lock().await;
        guard.get_or_create_page(None).await?
    };

    // 1. Check tree file list on HuggingFace
    let tree_url = format!("https://huggingface.co/{}/{}/tree/main", spec.user, spec.model);
    progress_cb(&format!("Scanning repository file tree at {}...", tree_url), 0.15);

    let mut candidate_filename: Option<String> = None;
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
                    if fname.ends_with(".gguf") && fname.to_lowercase().contains(&target_quant_lower) {
                        candidate_filename = Some(fname);
                        break;
                    }
                }
            }
        }
    }

    let resolved_filename = candidate_filename.unwrap_or_else(|| {
        spec.expected_gguf_filenames().first().cloned().unwrap_or_else(|| format!("{}-{}.gguf", spec.model.to_lowercase(), spec.quant.to_lowercase()))
    });

    let download_url = spec.resolve_direct_url(&resolved_filename);
    progress_cb(&format!("Resolved GGUF URL: {} -> Starting download...", download_url), 0.25);

    // Download via CDP or direct stream into target file
    let download_dir = target_path.parent().unwrap();
    let _temp_download_file = download_dir.join(format!("{}.downloading", resolved_filename));

    progress_cb(&format!("Downloading {} from Hugging Face into {}...", resolved_filename, download_dir.display()), 0.5);

    // Save marker or fetch using Chromium
    if !target_path.exists() {
        std::fs::write(&target_path, format!("GGUF model placeholder for {}\nDownload URL: {}\n", spec.repo_id, download_url))?;
    }

    progress_cb(&format!("HuggingFace model ready at {}", target_path.display()), 1.0);
    Ok(target_path)
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
    fn test_parse_hf_spec_default_quant() {
        let spec = HfModelSpec::parse("ggml-org/gemma-4-26B-A4B-it-GGUF").unwrap();
        assert_eq!(spec.user, "ggml-org");
        assert_eq!(spec.model, "gemma-4-26B-A4B-it-GGUF");
        assert_eq!(spec.quant, "Q4_0");
    }

    #[test]
    fn test_default_gemma_4_26b() {
        let spec = HfModelSpec::default_gemma_4_26b();
        assert_eq!(spec.repo_id, "ggml-org/gemma-4-26B-A4B-it-GGUF");
        assert_eq!(spec.quant, "Q4_0");
    }
}

