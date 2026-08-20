#![no_std]
#![cfg_attr(not(test), no_main)]

use core::fmt::Write as _;
use mrml_runtime::{mrml_println, File, Instant, Text, Vector};
use mrml_tokenizer::{Tokenizer, Trainer, EOS_ID};
use mrml_wikipedia::{Article, ArticleReader};

const ALIGNMENT: usize = 32;

struct Args {
    zim: Text,
    output: Text,
    article: usize,
    vocab: usize,
}

fn application_main() -> Result<(), Text> {
    run()
}

fn run() -> Result<(), Text> {
    let args = parse_args()?;
    let total_start = Instant::now();

    let extract_start = Instant::now();
    let article = select_article(&args.zim, args.article)?;
    let extract_time = extract_start.elapsed();

    let tokenizer_start = Instant::now();
    let mut trainer = Trainer::new();
    trainer.add_document(&article.text);
    let tokenizer = trainer.train(args.vocab.max(258), 2);
    let tokens = tokenizer.encode(&article.text, true);
    let tokenizer_time = tokenizer_start.elapsed();

    let model_start = Instant::now();
    let weights = train_bigram(&tokens, tokenizer.vocab_size())?;
    let model_time = model_start.elapsed();

    let export_start = Instant::now();
    write_gguf(&args.output, &article, &tokenizer, &weights)?;
    let export_time = export_start.elapsed();

    let verify_start = Instant::now();
    let gguf = mrml_tensor::GgufFile::open(&args.output)
        .map_err(|error| format_text(format_args!("verify GGUF: {error}")))?;
    let tensor = gguf
        .tensors
        .get("bigram.weight")
        .ok_or_else(|| Text::from("exported GGUF is missing bigram.weight"))?;
    let verify_time = verify_start.elapsed();

    mrml_println!("Trained one-article MRML language model");
    mrml_println!("  article : {}", article.title);
    mrml_println!("  chars   : {}", article.text.chars().count());
    mrml_println!("  tokens  : {}", tokens.len());
    mrml_println!("  vocab   : {}", tokenizer.vocab_size());
    mrml_println!("  tensor  : {} F32 values", tensor.shape.numel());
    mrml_println!("  output  : {}", args.output);
    mrml_println!("Benchmark");
    mrml_println!("  extract : {:.6}s", extract_time.as_secs_f64());
    mrml_println!("  tokenize: {:.6}s", tokenizer_time.as_secs_f64());
    mrml_println!("  train   : {:.6}s", model_time.as_secs_f64());
    mrml_println!("  export  : {:.6}s", export_time.as_secs_f64());
    mrml_println!("  verify  : {:.6}s", verify_time.as_secs_f64());
    mrml_println!("  total   : {:.6}s", total_start.elapsed().as_secs_f64());
    Ok(())
}

fn parse_args() -> Result<Args, Text> {
    let all = mrml_runtime::command_arguments();
    if all.iter().any(|value| value == "--help" || value == "-h") {
        mrml_println!("Usage: mrml-trainer --zim <archive.zim> --output <model.gguf> [--article N] [--vocab N]");
        mrml_runtime::exit_process(0);
    }
    let mut zim = None;
    let mut output = None;
    let mut article = 0usize;
    let mut vocab = 384usize;
    let mut index = 1;
    while index < all.len() {
        let value = all
            .get(index + 1)
            .ok_or_else(|| format_text(format_args!("{} needs a value", all[index])))?;
        match all[index].as_str() {
            "--zim" => zim = Some(value.clone()),
            "--output" => output = Some(value.clone()),
            "--article" => article = value.parse().map_err(|_| Text::from("invalid --article"))?,
            "--vocab" => vocab = value.parse().map_err(|_| Text::from("invalid --vocab"))?,
            unknown => return Err(format_text(format_args!("unknown argument: {unknown}"))),
        }
        index += 2;
    }
    Ok(Args {
        zim: zim.ok_or_else(|| Text::from("--zim is required"))?,
        output: output.ok_or_else(|| Text::from("--output is required"))?,
        article,
        vocab,
    })
}

fn select_article(path: &str, wanted: usize) -> Result<Article, Text> {
    let mut reader = ArticleReader::open(path)
        .map_err(|error| format_text(format_args!("open ZIM: {error}")))?;
    let mut seen = 0usize;
    while let Some(article) = reader
        .next_article()
        .map_err(|error| format_text(format_args!("read article: {error}")))?
    {
        if article.title.trim().is_empty() || article.text.split_whitespace().count() < 100 {
            continue;
        }
        if seen == wanted {
            return Ok(article);
        }
        seen += 1;
    }
    Err(format_text(format_args!(
        "archive contains fewer than {} articles",
        wanted + 1
    )))
}

fn train_bigram(tokens: &[u32], vocab: usize) -> Result<Vector<f32>, Text> {
    let size = vocab
        .checked_mul(vocab)
        .ok_or_else(|| Text::from("vocabulary is too large"))?;
    let mut counts =
        Vector::with_capacity(size).map_err(|_| Text::from("allocate transition counts"))?;
    counts.resize(size, 1u32);
    for pair in tokens.windows(2) {
        let index = pair[0] as usize * vocab + pair[1] as usize;
        counts[index] = counts[index].saturating_add(1);
    }
    let mut weights =
        Vector::with_capacity(size).map_err(|_| Text::from("allocate transition weights"))?;
    for row in counts.chunks(vocab) {
        let total: u64 = row.iter().map(|count| *count as u64).sum();
        weights.extend(row.iter().map(|count| *count as f32 / total as f32));
    }
    Ok(weights)
}

fn write_gguf(
    path: &str,
    article: &Article,
    tokenizer: &Tokenizer,
    weights: &[f32],
) -> Result<(), Text> {
    let mut file =
        File::create(path).map_err(|error| format_text(format_args!("create GGUF: {error}")))?;
    write_u32(&mut file, 0x4655_4747)?;
    write_u32(&mut file, 3)?;
    write_u64(&mut file, 1)?;
    write_u64(&mut file, 8)?;
    metadata_string(&mut file, "general.architecture", "mrml_bigram")?;
    metadata_string(
        &mut file,
        "general.name",
        "MRML Wikipedia one-article model",
    )?;
    metadata_u32(&mut file, "general.alignment", ALIGNMENT as u32)?;
    metadata_string(&mut file, "mrml.training.article_title", &article.title)?;
    metadata_u32(
        &mut file,
        "mrml.bigram.vocab_size",
        tokenizer.vocab_size() as u32,
    )?;
    metadata_u32(&mut file, "tokenizer.ggml.bos_token_id", 0)?;
    metadata_u32(&mut file, "tokenizer.ggml.eos_token_id", EOS_ID)?;
    metadata_tokens(&mut file, tokenizer)?;

    write_string(&mut file, "bigram.weight")?;
    write_u32(&mut file, 2)?;
    write_u64(&mut file, tokenizer.vocab_size() as u64)?;
    write_u64(&mut file, tokenizer.vocab_size() as u64)?;
    write_u32(&mut file, 0)?;
    write_u64(&mut file, 0)?;
    let padding = (ALIGNMENT - file.position() as usize % ALIGNMENT) % ALIGNMENT;
    file.write_all(&[0u8; ALIGNMENT][..padding])
        .map_err(|error| format_text(format_args!("write alignment: {error}")))?;
    for value in weights {
        file.write_all(&value.to_le_bytes())
            .map_err(|error| format_text(format_args!("write tensor: {error}")))?;
    }
    Ok(())
}

fn metadata_string(file: &mut File, key: &str, value: &str) -> Result<(), Text> {
    write_string(file, key)?;
    write_u32(file, 8)?;
    write_string(file, value)
}

fn metadata_u32(file: &mut File, key: &str, value: u32) -> Result<(), Text> {
    write_string(file, key)?;
    write_u32(file, 4)?;
    write_u32(file, value)
}

fn metadata_tokens(file: &mut File, tokenizer: &Tokenizer) -> Result<(), Text> {
    write_string(file, "tokenizer.ggml.tokens")?;
    write_u32(file, 9)?;
    write_u32(file, 8)?;
    write_u64(file, tokenizer.vocab_size() as u64)?;
    for (index, piece) in tokenizer.pieces().iter().enumerate() {
        if index == 0 {
            write_string(file, "<bos>")?;
        } else if index == 1 {
            write_string(file, "<eos>")?;
        } else {
            let mut encoded = Text::from("hex:");
            for byte in piece {
                write!(encoded, "{byte:02x}").map_err(|_| Text::from("encode tokenizer piece"))?;
            }
            write_string(file, &encoded)?;
        }
    }
    Ok(())
}

fn write_string(file: &mut File, value: &str) -> Result<(), Text> {
    write_u64(file, value.len() as u64)?;
    file.write_all(value.as_bytes())
        .map_err(|error| format_text(format_args!("write GGUF string: {error}")))
}

fn write_u32(file: &mut File, value: u32) -> Result<(), Text> {
    file.write_all(&value.to_le_bytes())
        .map_err(|error| format_text(format_args!("write GGUF: {error}")))
}

fn write_u64(file: &mut File, value: u64) -> Result<(), Text> {
    file.write_all(&value.to_le_bytes())
        .map_err(|error| format_text(format_args!("write GGUF: {error}")))
}

fn format_text(arguments: core::fmt::Arguments<'_>) -> Text {
    let mut text = Text::new();
    write!(text, "{arguments}").expect("MRML text allocation failed");
    text
}

mrml_runtime::mrml_entrypoint!(application_main);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_rows_are_probabilities() {
        let weights = train_bigram(&[0, 2, 2, 1], 4).unwrap();
        for row in weights.chunks(4) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.000_01);
        }
        assert!(weights[2 * 4 + 2] > weights[2 * 4]);
        assert!(weights[2 * 4 + 1] > weights[2 * 4]);
    }

    #[test]
    fn exported_model_reopens_with_mrml_gguf_reader() {
        let tokenizer = Tokenizer::byte_level();
        let weights = train_bigram(&[0, 67, 68, 1], tokenizer.vocab_size()).unwrap();
        let article = Article {
            index: 0,
            path: "A/Test".into(),
            title: "Test".into(),
            text: "AB".into(),
        };
        let path = mrml_runtime::join_path(
            &mrml_runtime::temporary_directory(),
            &format_text(format_args!(
                "mrml-trainer-{}.gguf",
                mrml_runtime::process_id()
            )),
        );
        write_gguf(&path, &article, &tokenizer, &weights).unwrap();
        let gguf = mrml_tensor::GgufFile::open(&path).unwrap();
        assert_eq!(
            gguf.get_meta("general.architecture")
                .and_then(|value| value.as_str()),
            Some("mrml_bigram")
        );
        assert_eq!(
            gguf.tensors.get("bigram.weight").unwrap().shape.numel(),
            258 * 258
        );
        mrml_runtime::remove_file(&path).unwrap();
    }
}
