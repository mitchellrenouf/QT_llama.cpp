use mrml_tensor::gguf::GgufFile;

fn main() -> mrml_tensor::anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: gguf_inspect <model.gguf>");
    let gguf = GgufFile::open(path)?;
    let mut metadata: Vec<_> = gguf.metadata.iter().collect();
    metadata.sort_unstable_by_key(|(key, _)| *key);
    for (key, value) in metadata {
        if !key.starts_with("tokenizer.") {
            println!("{key} = {value:?}");
        }
    }
    let mut tensors: Vec<_> = gguf.tensors.values().collect();
    tensors.sort_unstable_by_key(|tensor| &tensor.name);
    for tensor in tensors {
        println!("{} {:?} {:?}", tensor.name, tensor.shape, tensor.dtype);
    }
    Ok(())
}
