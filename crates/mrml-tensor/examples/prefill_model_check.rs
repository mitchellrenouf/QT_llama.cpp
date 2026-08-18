use mrml_tensor::MrmlModel;

fn main() -> mrml_tensor::anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: prefill_model_check <model.gguf> [prompt]");
    let user_prompt = std::env::args().nth(2).unwrap_or_else(|| "hi".to_string());
    let model = MrmlModel::load_from_gguf(path, 8192)?;
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
