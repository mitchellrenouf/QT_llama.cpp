use mrml_crypto::Sha3_512;
use mrml_error::{Result, anyhow};
use mrml_runtime::{Text, Vector, mrml_format as format, mrml_println as println, rename_file};

fn digest_hex(digest: &[u8; 64]) -> Text {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = Text::new();
    for byte in digest {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 15) as usize] as char);
    }
    text
}

fn hash_file_state(path: &str) -> Result<(Sha3_512, u64)> {
    let mut file = mrml_runtime::File::open(path)?;
    let mut hash = Sha3_512::new();
    let mut length = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        length = length.saturating_add(read as u64);
    }
    Ok((hash, length))
}
fn hash_file(path: &str) -> Result<([u8; 64], u64)> {
    let (hash, length) = hash_file_state(path)?;
    Ok((hash.finalize(), length))
}

fn digest_matches(path: &str, sidecar: &str) -> bool {
    let Ok(expected) = mrml_runtime::read_file_text(sidecar) else {
        return false;
    };
    let expected = expected.trim();
    if expected.len() != 128 {
        return false;
    }
    hash_file(path).is_ok_and(|(digest, _)| digest_hex(&digest).eq_ignore_ascii_case(expected))
}

fn write_digest(path: &str, digest: &[u8; 64]) -> Result<()> {
    mrml_runtime::write_file(path, digest_hex(digest).as_bytes())?;
    Ok(())
}

fn native_file_len(path: &str) -> Option<u64> {
    mrml_runtime::File::open(path).ok()?.len().ok()
}

fn path_file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfModelSpec {
    pub repo_id: Text,
    pub user: Text,
    pub model: Text,
    pub quant: Text,
}

#[derive(Debug, Clone)]
pub struct HfModelFiles {
    pub primary_entry_file: Text,
    pub shard_files: Vector<Text>,
    pub mmproj_file: Option<Text>,
    pub speedup_draft_file: Option<Text>,
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

        let parts: Vector<&str> = repo_part.split('/').collect();
        let (user, model) = match parts.len() {
            1 => ("ggml-org", parts[0]),
            2 => (parts[0], parts[1]),
            _ => {
                return Err(anyhow!(
                    "Invalid HuggingFace repo format. Expected 'user/model' or 'user/model:quant'"
                ));
            }
        };

        let repo_id = format!("{}/{}", user, model);
        let quant = if quant_part.is_empty() {
            Text::from("Q4_0")
        } else {
            Text::from(quant_part).to_ascii_uppercase()
        };

        Ok(Self {
            repo_id: repo_id.as_str().into(),
            user: user.into(),
            model: model.into(),
            quant,
        })
    }

    pub fn default_gemma_4_26b() -> Self {
        Self {
            repo_id: "ggml-org/gemma-4-26B-A4B-it-GGUF".into(),
            user: "ggml-org".into(),
            model: "gemma-4-26B-A4B-it-GGUF".into(),
            quant: "Q4_0".into(),
        }
    }

    pub fn cache_dir() -> Text {
        if let Some(path) = mrml_runtime::environment_variable("HF_HUB_CACHE") {
            return path;
        }
        if let Some(hf_home) = mrml_runtime::environment_variable("HF_HOME") {
            return mrml_runtime::join_path(&hf_home, "hub");
        }
        if let Some(mrml_cache) = mrml_runtime::environment_variable("MRML_CACHE") {
            return mrml_cache;
        }

        #[cfg(windows)]
        {
            if let Some(local_appdata) = mrml_runtime::environment_variable("LOCALAPPDATA") {
                let p = mrml_runtime::join_path(
                    &mrml_runtime::join_path(&local_appdata, "huggingface"),
                    "hub",
                );
                return p;
            }
        }

        if let Some(home) = crate::platform::home_dir() {
            return mrml_runtime::join_path(
                &mrml_runtime::join_path(&mrml_runtime::join_path(&home, ".cache"), "huggingface"),
                "hub",
            );
        }

        mrml_runtime::join_path(&mrml_runtime::join_path(".cache", "huggingface"), "hub")
    }

    pub fn get_model_dir(&self) -> Text {
        let repo_slug = format!("models--{}--{}", self.user, self.model);
        mrml_runtime::join_path(&Self::cache_dir(), &repo_slug)
    }

    pub fn is_cached(&self) -> bool {
        let model_dir = self.get_model_dir();
        if !mrml_runtime::path_is_directory(&model_dir) {
            return false;
        }

        let target_quant_lower = self.quant.to_ascii_lowercase();
        for path in crate::fs_walk::paths(&model_dir) {
            let name = Text::from(path_file_name(&path)).to_ascii_lowercase();
            if name.ends_with(".gguf")
                && name.contains(target_quant_lower.as_str())
                && !name.ends_with(".part")
                && !name.contains("mmproj")
                && !name.contains("mtp")
            {
                if mrml_runtime::File::open(&path)
                    .and_then(|file| file.len())
                    .is_ok_and(|length| length > 10 * 1024 * 1024)
                {
                    // > 10MB
                    return true;
                }
            }
        }
        false
    }
}

pub fn render_progress_bar(percent: f32, width: usize) -> Text {
    let filled = ((percent.clamp(0.0, 1.0) * width as f32 + 0.5) as usize).min(width);
    let empty = width.saturating_sub(filled);
    let mut output = Text::with_capacity(2 + width * 3).expect("MRML allocation failed");
    output.push('[');
    for _ in 0..filled {
        output.push('█');
    }
    for _ in 0..empty {
        output.push('░');
    }
    output.push(']');
    output
}

pub async fn query_hf_api_siblings(spec: &HfModelSpec) -> Result<Vector<Text>> {
    let api_url = format!(
        "https://huggingface.co/api/models/{}/{}",
        spec.user, spec.model
    );

    let client = mrml_http::Client::new();
    let token =
        mrml_runtime::environment_variable("HF_TOKEN").map(|value| format!("Bearer {}", value));
    let mut response = if let Some(ref authorization) = token {
        client.get_follow(&api_url, &[("Authorization", authorization)], 8)
    } else {
        client.get_follow(&api_url, &[], 8)
    }
    .map_err(|error| anyhow!("Hugging Face HTTPS query failed: {}", error))?;
    if !(200..300).contains(&response.status) {
        return Err(anyhow!("Failed to query Hugging Face API at {}", api_url));
    }
    let bytes = response
        .read_to_end(32 * 1024 * 1024)
        .map_err(|error| anyhow!("Hugging Face API response failed: {}", error))?;
    let body = Text::from_utf8_lossy(&bytes);
    let val: serde_json::Value = serde_json::from_str(&body)?;

    let mut filenames = Vector::new();
    if let Some(siblings) = val.get("siblings").and_then(|s| s.as_array()) {
        for item in siblings {
            if let Some(rfilename) = item.get("rfilename").and_then(|r| r.as_str()) {
                if rfilename.ends_with(".gguf") {
                    filenames.push(rfilename.into());
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
                path_file_name(&existing_primary)
            ),
            1.0,
            1,
            1,
        );
        return Ok(HfModelFiles {
            primary_entry_file: existing_primary.clone(),
            shard_files: [existing_primary].into_iter().collect(),
            mmproj_file: None,
            speedup_draft_file: None,
        });
    }

    let model_dir = spec.get_model_dir();
    mrml_runtime::create_dir_all(&model_dir)?;

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

    let target_quant_lower = spec.quant.to_ascii_lowercase();
    let mut matching_shards = Vector::new();
    let mut mmproj_file_opt = None;
    let mut speedup_file_opt = None;

    for fname in &siblings {
        let fname_lower = fname.to_ascii_lowercase();
        if fname_lower.contains("mmproj") {
            if mmproj_file_opt.is_none() || fname_lower.contains(target_quant_lower.as_str()) {
                mmproj_file_opt = Some(fname.clone());
            }
        } else if fname_lower.contains("mtp") || fname_lower.contains("dflash") {
            if speedup_file_opt.is_none() || fname_lower.contains(target_quant_lower.as_str()) {
                speedup_file_opt = Some(fname.clone());
            }
        } else if fname_lower.contains(target_quant_lower.as_str()) {
            matching_shards.push(fname.clone());
        }
    }

    matching_shards[..].sort_unstable();

    // Fallbacks if not found via API
    if matching_shards.is_empty() {
        if spec.quant.eq_ignore_ascii_case("Q8_0") {
            for shard_idx in 1..=4 {
                matching_shards.push(
                    format!(
                        "{}-{}-{:05}-of-00004.gguf",
                        spec.model.to_ascii_lowercase(),
                        spec.quant.to_ascii_lowercase(),
                        shard_idx
                    )
                    .as_str()
                    .into(),
                );
            }
        } else {
            matching_shards.push(
                format!(
                    "{}-{}.gguf",
                    spec.model.to_ascii_lowercase(),
                    spec.quant.to_ascii_lowercase()
                )
                .as_str()
                .into(),
            );
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
    let mut downloaded_shard_paths = Vector::new();
    let mut resolved_mmproj_path = None;
    let mut resolved_speedup_path = None;

    for (idx, filename) in download_queue.iter().enumerate() {
        let file_num = idx + 1;
        let dest_path = mrml_runtime::join_path(&model_dir, filename);
        let part_path = mrml_runtime::join_path(&model_dir, &format!("{}.part", filename));
        let digest_path = format!("{}.sha3-512", dest_path);
        let part_digest_path = format!("{}.sha3-512", part_path);

        if native_file_len(&dest_path).is_some_and(|length| length > 10 * 1024 * 1024) {
            if mrml_runtime::path_is_file(&digest_path) {
                if !digest_matches(&dest_path, &digest_path) {
                    mrml_runtime::remove_file(&dest_path)?;
                    mrml_runtime::remove_file(&digest_path)?;
                } else {
                    let msg = format!(
                        "✓ [File {}/{}] {} (Cached, SHA3-512 verified)",
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
            } else {
                let (digest, _) = hash_file(&dest_path)?;
                write_digest(&digest_path, &digest)?;
            }
            if mrml_runtime::path_is_file(&dest_path) {
                let msg = format!(
                    "✓ [File {}/{}] {} (Cached, SHA3-512 recorded)",
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
        }

        let download_url = format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            spec.user, spec.model, filename
        );

        let mut initial_resume_bytes = native_file_len(&part_path).unwrap_or(0);
        if initial_resume_bytes > 0 && !digest_matches(&part_path, &part_digest_path) {
            mrml_runtime::remove_file(&part_path)?;
            if mrml_runtime::path_is_file(&part_digest_path) {
                mrml_runtime::remove_file(&part_digest_path)?;
            }
            initial_resume_bytes = 0;
        }

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

        let client = mrml_http::Client::new();
        let authorization =
            mrml_runtime::environment_variable("HF_TOKEN").map(|value| format!("Bearer {}", value));
        let range = format!("bytes={}-", initial_resume_bytes);
        let mut response = match (authorization.as_ref(), initial_resume_bytes > 0) {
            (Some(auth), true) => client.get_follow(
                &download_url,
                &[("Authorization", auth), ("Range", &range)],
                8,
            ),
            (Some(auth), false) => client.get_follow(&download_url, &[("Authorization", auth)], 8),
            (None, true) => client.get_follow(&download_url, &[("Range", &range)], 8),
            (None, false) => client.get_follow(&download_url, &[], 8),
        }
        .map_err(|error| anyhow!("HTTPS download failed for {}: {}", filename, error))?;
        if initial_resume_bytes > 0 {
            let prefix = format!("bytes {}-", initial_resume_bytes);
            if response.status != 206
                || !response
                    .header("content-range")
                    .is_some_and(|value| value.starts_with(prefix.as_str()))
            {
                initial_resume_bytes = 0;
                response = if let Some(ref auth) = authorization {
                    client.get_follow(&download_url, &[("Authorization", auth)], 8)
                } else {
                    client.get_follow(&download_url, &[], 8)
                }
                .map_err(|error| anyhow!("HTTPS restart failed for {}: {}", filename, error))?;
            }
        }
        if response.status != if initial_resume_bytes > 0 { 206 } else { 200 } {
            return Err(anyhow!(
                "Download of {} returned HTTP {}",
                filename,
                response.status
            ));
        }
        let (mut hash, mut downloaded) = if initial_resume_bytes > 0 {
            hash_file_state(&part_path)?
        } else {
            (Sha3_512::new(), 0)
        };
        let mut output = if initial_resume_bytes > 0 {
            let mut file = mrml_runtime::File::open_write(&part_path)?;
            file.seek(initial_resume_bytes)?;
            file
        } else {
            mrml_runtime::File::create(&part_path)?
        };
        let mut buffer = [0u8; 64 * 1024];
        let mut next_checkpoint = downloaded.saturating_add(64 * 1024 * 1024);
        loop {
            let read = response
                .read(&mut buffer)
                .map_err(|error| anyhow!("HTTPS body failed for {}: {}", filename, error))?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
            hash.update(&buffer[..read]);
            downloaded = downloaded.saturating_add(read as u64);
            if downloaded >= next_checkpoint {
                write_digest(&part_digest_path, &hash.clone().finalize())?;
                next_checkpoint = downloaded.saturating_add(64 * 1024 * 1024);
                let mb = downloaded as f64 / (1024.0 * 1024.0);
                let msg = format!(
                    "⬇️ [File {}/{}] {} ({:.1} MB downloaded, SHA3-512 checkpointed)...",
                    file_num, total_files, filename, mb
                );
                let file_pct = (mb / 4000.0).clamp(0.05, 0.95) as f32;
                progress_cb(
                    &msg,
                    ((file_num as f32 - 1.0) + file_pct) / total_files as f32,
                    file_num,
                    total_files,
                );
            }
        }
        let digest = hash.finalize();
        write_digest(&part_digest_path, &digest)?;

        // Successfully downloaded: promote .part to final .gguf
        if mrml_runtime::path_is_file(&part_path) {
            rename_file(&part_path, &dest_path)?;
            write_digest(&digest_path, &digest)?;
            if mrml_runtime::path_is_file(&part_digest_path) {
                mrml_runtime::remove_file(&part_digest_path)?;
            }
        }

        let done_msg = format!(
            "✓ [File {}/{}] {} (Downloaded securely, SHA3-512 recorded)",
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
        mrml_runtime::join_path(
            &model_dir,
            &format!(
                "{}-{}.gguf",
                spec.model.to_ascii_lowercase(),
                spec.quant.to_ascii_lowercase()
            ),
        )
    });

    let complete_msg = format!(
        "✨ All {} model weights ready in {}",
        total_files, model_dir
    );
    println!("{}", complete_msg);
    progress_cb(&complete_msg, 1.0, total_files, total_files);

    Ok(HfModelFiles {
        primary_entry_file,
        shard_files: downloaded_shard_paths.into_iter().collect(),
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

    #[test]
    fn sha3_sidecar_detects_model_tampering() {
        let directory = mrml_runtime::temporary_directory();
        let path = mrml_runtime::join_path(&directory, "mrml-hash-test.bin");
        let sidecar = format!("{}.sha3-512", path);
        mrml_runtime::write_file(&path, b"authenticated model bytes").unwrap();
        let (digest, _) = hash_file(&path).unwrap();
        write_digest(&sidecar, &digest).unwrap();
        assert!(digest_matches(&path, &sidecar));
        mrml_runtime::write_file(&path, b"tampered model bytes").unwrap();
        assert!(!digest_matches(&path, &sidecar));
        let _ = mrml_runtime::remove_file(&path);
        let _ = mrml_runtime::remove_file(&sidecar);
    }

    #[test]
    fn live_hf_query_when_configured() {
        if mrml_runtime::environment_variable("MRML_HF_LIVE_TEST").is_none() {
            return;
        }
        let spec = HfModelSpec::parse("ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0").unwrap();
        let files = mrml_tools::block_on(query_hf_api_siblings(&spec)).unwrap();
        assert!(!files.is_empty());
    }
}
