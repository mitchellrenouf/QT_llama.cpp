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
    println!("Metadata count: {}", gguf.metadata.len());
    println!("Tensors count: {}", gguf.tensors.len());

    let mut names: Vec<&String> = gguf.tensors.keys().collect();
    names.sort();

    println!("First 20 tensors:");
    for name in names.iter().take(20) {
        let info = &gguf.tensors[*name];
        println!("  - {} {:?} {:?}", info.name, info.shape.dims, info.dtype);
    }
}
