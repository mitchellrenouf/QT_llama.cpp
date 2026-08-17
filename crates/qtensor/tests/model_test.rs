use qtensor::gguf::GgufFile;
use std::path::PathBuf;

#[test]
fn test_inspect_gemma_gguf() {
    let mut path = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push("huggingface");
    path.push("hub");
    path.push("models--ggml-org--gemma-4-26B-A4B-it-GGUF");
    path.push("gemma-4-26B-A4B-it-Q4_0.gguf");

    if !path.exists() {
        println!("Model file not found at {:?}, skipping", path);
        return;
    }

    let gguf = GgufFile::open(&path).expect("Failed to open GGUF");
    println!("GGUF version: {}", gguf.version);
    let model = qtensor::model::QTensorModel::load_from_gguf(&path, 8192).expect("Load model");
    let prompt_tokens = model.tokenize("Hello!");
    println!("Prompt tokens: {:?}", prompt_tokens);

    let special_tags = ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<|turn>", "<turn|>"];
    for tag in special_tags {
        let t = model.tokenize(tag);
        println!("Tag '{}' tokens: {:?}", tag, t);
    }
    let mut state = model.init_generation_state(&prompt_tokens);
    println!("Hidden state norm: {}", state.hidden.iter().map(|x| x * x).sum::<f32>().sqrt());
    
    // Inspect dot products of specific words: "How", "can", "there", "world", "the", "en"
    let test_words = [" Hello", " How", " can", " I", " you", " there", " world", " the", "en"];
    for w in test_words {
        let t = model.tokenize(w);
        println!("Word '{}' tokens: {:?}", w, t);
    }
    
    for step in 0..5 {
        let tok = model.step_generation(&mut state, 0.7);
        let piece = model.token_to_piece(tok);
        println!("Step {}: token {} -> '{}'", step, tok, piece);
    }
}
