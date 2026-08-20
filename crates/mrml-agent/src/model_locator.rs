use mrml_runtime::{Text, Vector, mrml_format as format};

type Vec<T> = Vector<T>;

pub fn get_model_cache_roots() -> Vec<Text> {
    let mut roots = Vec::new();

    if let Some(p) = mrml_runtime::environment_variable("HF_HUB_CACHE") {
        roots.push(p);
    }
    if let Some(p) = mrml_runtime::environment_variable("HF_HOME") {
        roots.push(mrml_runtime::join_path(&p, "hub"));
    }
    if let Some(p) = mrml_runtime::environment_variable("MRML_CACHE") {
        roots.push(p);
    }

    #[cfg(windows)]
    {
        if let Some(local_appdata) = mrml_runtime::environment_variable("LOCALAPPDATA") {
            roots.push(mrml_runtime::join_path(
                &mrml_runtime::join_path(&local_appdata, "huggingface"),
                "hub",
            ));
        }
        if let Some(userprofile) = mrml_runtime::environment_variable("USERPROFILE") {
            roots.push(mrml_runtime::join_path(
                &mrml_runtime::join_path(
                    &mrml_runtime::join_path(&userprofile, ".cache"),
                    "huggingface",
                ),
                "hub",
            ));
        }
    }

    if let Some(home) = crate::platform::home_dir() {
        roots.push(mrml_runtime::join_path(
            &mrml_runtime::join_path(&mrml_runtime::join_path(&home, ".cache"), "huggingface"),
            "hub",
        ));
        roots.push(mrml_runtime::join_path(
            &mrml_runtime::join_path(&mrml_runtime::join_path(&home, ".cache"), "gemma"),
            "models",
        ));
    }

    let mut unique_roots = Vec::new();
    for r in roots {
        if mrml_runtime::path_is_directory(&r) && !unique_roots.contains(&r) {
            unique_roots.push(r);
        }
    }
    unique_roots
}

pub fn find_model_file(model_arg: &str) -> Option<Text> {
    if crate::platform::path_is_file(model_arg) {
        return Some(model_arg.into());
    }

    let cache_roots = get_model_cache_roots();

    if let Ok(spec) = crate::hf::HfModelSpec::parse(model_arg) {
        let repo_slug = format!("models--{}--{}", spec.user, spec.model);
        let target_quant = spec.quant.to_ascii_lowercase();

        // 1. Search for matching repo slug in Hugging Face cache directories
        for root in &cache_roots {
            let repo_dir = mrml_runtime::join_path(root, &repo_slug);
            if mrml_runtime::path_is_directory(&repo_dir) {
                let mut best_match = None;
                for path in crate::fs_walk::paths(&repo_dir) {
                    let name = Text::from(path.rsplit(['/', '\\']).next().unwrap_or(&path))
                        .to_ascii_lowercase();
                    if name.ends_with(".gguf")
                        && !name.ends_with(".part")
                        && !name.contains("mmproj")
                        && !name.contains("mtp")
                    {
                        if name.contains(target_quant.as_str()) {
                            return Some(path);
                        }
                        if best_match.is_none() {
                            best_match = Some(path);
                        }
                    }
                }
                if let Some(m) = best_match {
                    return Some(m);
                }
            }

            // Legacy folder name check (e.g. user_model)
            let legacy_dir =
                mrml_runtime::join_path(root, &format!("{}_{}", spec.user, spec.model));
            if mrml_runtime::path_is_directory(&legacy_dir) {
                for path in crate::fs_walk::paths(&legacy_dir) {
                    let name = Text::from(path.rsplit(['/', '\\']).next().unwrap_or(&path))
                        .to_ascii_lowercase();
                    if name.ends_with(".gguf")
                        && !name.ends_with(".part")
                        && !name.contains("mmproj")
                        && !name.contains("mtp")
                    {
                        if name.contains(target_quant.as_str()) {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }

    // 2. Scan whole cache roots for matching model file
    for root in &cache_roots {
        for path in crate::fs_walk::paths(root) {
            let name =
                Text::from(path.rsplit(['/', '\\']).next().unwrap_or(&path)).to_ascii_lowercase();
            if name.ends_with(".gguf")
                && !name.ends_with(".part")
                && !name.contains("mmproj")
                && !name.contains("mtp")
            {
                if name.contains("gemma-4") || name.contains("gemma") {
                    return Some(path);
                }
            }
        }
    }

    let separator = model_arg.rfind(['/', '\\']);
    let extension = model_arg
        .rfind('.')
        .filter(|dot| separator.is_none_or(|separator| *dot > separator));
    let with_gguf = extension
        .map(|dot| format!("{}.gguf", &model_arg[..dot]))
        .unwrap_or_else(|| format!("{}.gguf", model_arg));
    let candidates: [Text; 5] = [
        mrml_runtime::join_path("models", model_arg),
        with_gguf.as_str().into(),
        mrml_runtime::join_path(
            &mrml_runtime::join_path(
                &crate::platform::home_dir().unwrap_or_default(),
                ".cache/gemma",
            ),
            model_arg,
        ),
        mrml_runtime::join_path(
            &crate::platform::home_dir().unwrap_or_default(),
            ".cache/gemma/gemma-4-26b-it-q4_0.gguf",
        ),
        "/models/gemma-4-26b-it-q4_0.gguf".into(),
    ];

    for c in candidates {
        if crate::platform::path_is_file(&c) {
            return Some(c);
        }
    }

    None
}
