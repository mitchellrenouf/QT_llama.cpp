use qtensor::QTensorModel;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: prefill_model_check <model.gguf>");
    let model = QTensorModel::load_from_gguf(path, 8192)?;
    let prompt = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n";
    let tokens = model.tokenize(prompt);
    let mut state = model.init_generation_state(&tokens);
    let mut ids = Vec::new();
    for _ in 0..8 {
        ids.push(model.step_generation(&mut state, 0.0));
    }
    println!("tokens={tokens:?}\ngenerated={ids:?}");
    Ok(())
}
