#![no_std]
#![no_main]

use mrml_runtime::{
    Text as String, Vector as Vec, command_arguments, mrml_format as format,
    mrml_println as println,
};
use mrml_tensor::MrmlModel;

fn application_main() -> mrml_tensor::error::Result<()> {
    let arguments = command_arguments();
    let path = arguments
        .get(1)
        .expect("usage: prefill_model_check <model.gguf> [prompt]");
    let user_prompt = arguments.get(2).map_or("hi", |value| value.as_str());
    let model = MrmlModel::load_from_gguf(&path, 8192)?;
    let prompt = format!("<bos><|turn>user\n{user_prompt}<turn|>\n<|turn>model\n");
    let tokens = model.tokenize(&prompt);
    let mut state = model.init_generation_state(&tokens);
    let mut ids = Vec::new();
    let mut text = String::new();
    for _ in 0..128 {
        let token = model.step_generation(&mut state, 0.0);
        if model.is_eog_token(token) {
            break;
        }
        ids.push(token);
        text.push_str(&model.token_to_piece(token));
    }
    println!(
        "prompt_tokens={}\ngenerated_ids={ids:?}\ntext={text:?}",
        tokens.len()
    );
    Ok(())
}

mrml_runtime::mrml_entrypoint!(application_main);
