#![no_std]
#![no_main]

use mrml_runtime::{Vector as Vec, command_arguments, mrml_println as println};
use mrml_tensor::gguf::GgufFile;

fn application_main() -> mrml_tensor::anyhow::Result<()> {
    let arguments = command_arguments();
    let path = arguments
        .get(1)
        .expect("usage: gguf_inspect <model.gguf>");
    let gguf = GgufFile::open(&path)?;
    let mut metadata: Vec<_> = gguf.metadata.iter().collect();
    metadata.sort_unstable_by_key(|(key, _)| *key);
    for (key, value) in metadata {
        if !key.starts_with("tokenizer.") {
            println!("{key} = {value:?}");
        }
    }
    let mut tensors: Vec<_> = gguf.tensors.iter().map(|(_, tensor)| tensor).collect();
    tensors.sort_unstable_by_key(|tensor| &tensor.name);
    for tensor in tensors {
        println!("{} {:?} {:?}", tensor.name, tensor.shape, tensor.dtype);
    }
    Ok(())
}

mrml_runtime::mrml_entrypoint!(application_main);
