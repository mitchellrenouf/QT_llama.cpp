#![no_std]

use mrml_runtime::{Text, environment_variable, join_path, mrml_print as print, mrml_println as println};

fn local_data_dir() -> Text {
    if cfg!(windows) {
        environment_variable("LOCALAPPDATA").unwrap_or_else(|| Text::from("."))
    } else if let Some(path) = environment_variable("XDG_DATA_HOME") {
        path
    } else {
        let home = environment_variable("HOME").unwrap_or_else(|| Text::from("."));
        join_path(&join_path(&home, ".local"), "share")
    }
}

#[test]
fn test_generation_outputs() {
    let path = join_path(
        &join_path(
            &join_path(
                &join_path(&local_data_dir(), "huggingface"),
                "hub",
            ),
            "models--ggml-org--gemma-4-26B-A4B-it-GGUF",
        ),
        "gemma-4-26B-A4B-it-Q4_0.gguf",
    );

    if !mrml_runtime::path_is_file(&path) {
        println!("Model file not found at {:?}, skipping", path);
        return;
    }

    let model = mrml_tensor::model::MrmlModel::load_from_gguf(&path, 8192)
        .expect("Load model");

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
