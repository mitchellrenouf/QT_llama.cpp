use std::path::PathBuf;

fn local_data_dir() -> PathBuf {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    } else if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(path)
    } else {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".local")
            .join("share")
    }
}

#[test]
fn test_generation_outputs() {
    let mut path = local_data_dir();
    path.push("huggingface");
    path.push("hub");
    path.push("models--ggml-org--gemma-4-26B-A4B-it-GGUF");
    path.push("gemma-4-26B-A4B-it-Q4_0.gguf");

    if !path.exists() {
        println!("Model file not found at {:?}, skipping", path);
        return;
    }

    let model = mrml_tensor::model::MrmlModel::load_from_gguf(&path, 8192).expect("Load model");

    let prompts = [
        "<|turn>user\nHi\n<turn|>\n<|turn>model\n",
        "<|turn>user\nWhat is 2+2?\n<turn|>\n<|turn>model\n",
    ];

    for (p_idx, p) in prompts.iter().enumerate() {
        println!("\n=================== PROMPT {} ===================", p_idx);
        println!("Prompt: {:?}", p);
        let tokens = model.tokenize(p);
        println!("Tokenized into {} tokens: {:?}", tokens.len(), tokens);

        let mut state = model.init_generation_state(&tokens);
        print!("Generated: ");
        for step in 0..15 {
            let next_tok = model.step_generation(&mut state, 0.7);
            let piece = model.token_to_piece(next_tok);
            print!("{}", piece);
            if model.is_eog_token(next_tok) && step >= 2 {
                println!(" [EOG]");
                break;
            }
        }
        println!();
    }
}
