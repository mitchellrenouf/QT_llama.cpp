#![no_std]
#![cfg_attr(not(test), no_main)]

use core::fmt::Write as _;
use mrml_runtime::{File, Instant, Text, Vector, mrml_println};
use mrml_tokenizer::{EOS_ID, Tokenizer, Trainer};
use mrml_wikipedia::{Article, ArticleReader};

mod transformer;
use transformer::{CONTEXT, DIM, FFN, HEADS, Transformer};

const ALIGNMENT: usize = 32;

struct Args {
    zim: Text,
    output: Text,
    article: usize,
    articles: usize,
    vocab: usize,
    prompt: Text,
    steps: usize,
    learning_rate: f32,
}

fn application_main() -> Result<(), Text> {
    run()
}

fn run() -> Result<(), Text> {
    let args = parse_args()?;
    let total_start = Instant::now();

    let extract_start = Instant::now();
    let article = select_articles(&args.zim, args.article, args.articles)?;
    let extract_time = extract_start.elapsed();

    let tokenizer_start = Instant::now();
    let mut trainer = Trainer::new();
    trainer.add_document(&article.text);
    let tokenizer = trainer.train(args.vocab.max(258), 2);
    let tokens = tokenizer.encode(&article.text, true);
    let tokenizer_time = tokenizer_start.elapsed();

    let model_start = Instant::now();
    let transitions = train_bigram(&tokens, tokenizer.vocab_size())?;
    let mut model = Transformer::from_transitions(&transitions, tokenizer.vocab_size());
    let training = model.train_output(&tokens, args.steps, args.learning_rate);
    let model_time = model_start.elapsed();

    let export_start = Instant::now();
    write_gguf(&args.output, &article, &tokenizer, &model)?;
    let export_time = export_start.elapsed();

    let verify_start = Instant::now();
    let gguf = mrml_tensor::GgufFile::open(&args.output)
        .map_err(|error| format_text(format_args!("verify GGUF: {error}")))?;
    gguf.tensors
        .get("output.weight")
        .ok_or_else(|| Text::from("exported GGUF is missing output.weight"))?;
    let parameters: usize = gguf
        .tensors
        .iter()
        .map(|(_, tensor)| tensor.shape.numel())
        .sum();
    let verify_time = verify_start.elapsed();

    let response = model.generate(&tokenizer, &args.prompt, 80);
    mrml_println!("Trained MRML causal transformer");
    mrml_println!("  source  : {}", article.title);
    mrml_println!("  articles: {}", args.articles);
    mrml_println!("  chars   : {}", article.text.chars().count());
    mrml_println!("  tokens  : {}", tokens.len());
    mrml_println!("  vocab   : {}", tokenizer.vocab_size());
    mrml_println!("  model   : 1 layer, {DIM} dim, {HEADS} heads, {FFN} FFN, {CONTEXT} context");
    mrml_println!(
        "  updates : {} contextual cross-entropy steps",
        training.steps
    );
    mrml_println!("  accuracy: {:.2}%", training.accuracy * 100.0);
    mrml_println!(
        "  params  : {} F32 values across {} tensors",
        parameters,
        gguf.tensors.len()
    );
    mrml_println!("  output  : {}", args.output);
    mrml_println!("Benchmark");
    mrml_println!("  extract : {:.6}s", extract_time.as_secs_f64());
    mrml_println!("  tokenize: {:.6}s", tokenizer_time.as_secs_f64());
    mrml_println!("  train   : {:.6}s", model_time.as_secs_f64());
    mrml_println!("  export  : {:.6}s", export_time.as_secs_f64());
    mrml_println!("  verify  : {:.6}s", verify_time.as_secs_f64());
    mrml_println!("  total   : {:.6}s", total_start.elapsed().as_secs_f64());
    mrml_println!("Prompt: {}", args.prompt);
    mrml_println!("Model: {response}");
    Ok(())
}

fn parse_args() -> Result<Args, Text> {
    let all = mrml_runtime::command_arguments();
    if all.iter().any(|value| value == "--help" || value == "-h") {
        mrml_println!(
            "Usage: mrml-trainer --zim <archive.zim> --output <model.gguf> [--article N] [--articles N] [--vocab N] [--steps N] [--learning-rate F] [--prompt TEXT]"
        );
        mrml_runtime::exit_process(0);
    }
    let mut zim = None;
    let mut output = None;
    let mut article = 0usize;
    let mut articles = 1usize;
    let mut vocab = 384usize;
    let mut prompt = Text::from("hello");
    let mut steps = 20_000usize;
    let mut learning_rate = 0.02f32;
    let mut index = 1;
    while index < all.len() {
        let value = all
            .get(index + 1)
            .ok_or_else(|| format_text(format_args!("{} needs a value", all[index])))?;
        match all[index].as_str() {
            "--zim" => zim = Some(value.clone()),
            "--output" => output = Some(value.clone()),
            "--article" => article = value.parse().map_err(|_| Text::from("invalid --article"))?,
            "--articles" => {
                articles = value
                    .parse()
                    .map_err(|_| Text::from("invalid --articles"))?
            }
            "--vocab" => vocab = value.parse().map_err(|_| Text::from("invalid --vocab"))?,
            "--prompt" => prompt = value.clone(),
            "--steps" => steps = value.parse().map_err(|_| Text::from("invalid --steps"))?,
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| Text::from("invalid --learning-rate"))?
            }
            unknown => return Err(format_text(format_args!("unknown argument: {unknown}"))),
        }
        index += 2;
    }
    Ok(Args {
        zim: zim.ok_or_else(|| Text::from("--zim is required"))?,
        output: output.ok_or_else(|| Text::from("--output is required"))?,
        article,
        articles: articles.max(1),
        vocab,
        prompt,
        steps,
        learning_rate,
    })
}

fn select_articles(path: &str, wanted: usize, count: usize) -> Result<Article, Text> {
    let mut reader = ArticleReader::open(path)
        .map_err(|error| format_text(format_args!("open ZIM: {error}")))?;
    reader.set_cluster_cache_capacity(256);
    let mut seen = 0usize;
    let mut selected = 0usize;
    let mut text = Text::new();
    let mut first_title = Text::new();
    while let Some(article) = reader
        .next_article()
        .map_err(|error| format_text(format_args!("read article: {error}")))?
    {
        if article.title.trim().is_empty() || article.text.split_whitespace().count() < 100 {
            continue;
        }
        if seen >= wanted {
            if selected == 0 {
                first_title = article.title.clone();
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&article.text);
            selected += 1;
            if selected == count {
                return Ok(Article {
                    index: article.index,
                    path: Text::new(),
                    title: format_text(format_args!("{first_title} + {} more", selected - 1)),
                    text,
                });
            }
        }
        seen += 1;
    }
    Err(format_text(format_args!(
        "archive contains fewer than {} requested substantive articles",
        wanted + count
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
    model: &Transformer,
) -> Result<(), Text> {
    let mut file =
        File::create(path).map_err(|error| format_text(format_args!("create GGUF: {error}")))?;
    write_u32(&mut file, 0x4655_4747)?;
    write_u32(&mut file, 3)?;
    write_u64(&mut file, 9)?;
    write_u64(&mut file, 13)?;
    metadata_string(&mut file, "general.architecture", "mrml_transformer")?;
    metadata_string(
        &mut file,
        "general.name",
        "MRML Wikipedia causal transformer",
    )?;
    metadata_u32(&mut file, "general.alignment", ALIGNMENT as u32)?;
    metadata_string(&mut file, "mrml.training.article_title", &article.title)?;
    metadata_u32(
        &mut file,
        "mrml_transformer.vocab_size",
        tokenizer.vocab_size() as u32,
    )?;
    metadata_u32(&mut file, "mrml_transformer.block_count", 1)?;
    metadata_u32(&mut file, "mrml_transformer.embedding_length", DIM as u32)?;
    metadata_u32(
        &mut file,
        "mrml_transformer.attention.head_count",
        HEADS as u32,
    )?;
    metadata_u32(&mut file, "mrml_transformer.context_length", CONTEXT as u32)?;
    metadata_u32(&mut file, "tokenizer.ggml.bos_token_id", 0)?;
    metadata_u32(&mut file, "tokenizer.ggml.eos_token_id", EOS_ID)?;
    metadata_tokens(&mut file, tokenizer)?;
    metadata_merges(&mut file, tokenizer)?;

    let tensors: [(&str, &[usize], &[f32]); 9] = [
        ("token_embd.weight", &[DIM, model.vocab], &model.token_embd),
        (
            "position_embd.weight",
            &[DIM, CONTEXT],
            &model.position_embd,
        ),
        ("blk.0.attn_q.weight", &[DIM, DIM], &model.attn_q),
        ("blk.0.attn_k.weight", &[DIM, DIM], &model.attn_k),
        ("blk.0.attn_v.weight", &[DIM, DIM], &model.attn_v),
        ("blk.0.attn_output.weight", &[DIM, DIM], &model.attn_output),
        ("blk.0.ffn_up.weight", &[DIM, FFN], &model.ffn_up),
        ("blk.0.ffn_down.weight", &[FFN, DIM], &model.ffn_down),
        ("output.weight", &[DIM, model.vocab], &model.output),
    ];
    let mut offset = 0usize;
    for (name, shape, values) in &tensors {
        write_tensor_descriptor(&mut file, name, shape, offset as u64)?;
        offset = align_up(offset + values.len() * 4);
    }
    let padding = (ALIGNMENT - file.position() as usize % ALIGNMENT) % ALIGNMENT;
    file.write_all(&[0u8; ALIGNMENT][..padding])
        .map_err(|error| format_text(format_args!("write alignment: {error}")))?;
    for (_, _, values) in tensors {
        for value in values {
            file.write_all(&value.to_le_bytes())
                .map_err(|error| format_text(format_args!("write tensor: {error}")))?;
        }
        let padding = (ALIGNMENT - file.position() as usize % ALIGNMENT) % ALIGNMENT;
        file.write_all(&[0u8; ALIGNMENT][..padding])
            .map_err(|error| format_text(format_args!("write tensor alignment: {error}")))?;
    }
    Ok(())
}

fn write_tensor_descriptor(
    file: &mut File,
    name: &str,
    shape: &[usize],
    offset: u64,
) -> Result<(), Text> {
    write_string(file, name)?;
    write_u32(file, shape.len() as u32)?;
    for dimension in shape {
        write_u64(file, *dimension as u64)?;
    }
    write_u32(file, 0)?;
    write_u64(file, offset)
}

const fn align_up(value: usize) -> usize {
    (value + ALIGNMENT - 1) & !(ALIGNMENT - 1)
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
        write_string(file, &piece_name(index, piece)?)?;
    }
    Ok(())
}

fn metadata_merges(file: &mut File, tokenizer: &Tokenizer) -> Result<(), Text> {
    write_string(file, "tokenizer.ggml.merges")?;
    write_u32(file, 9)?;
    write_u32(file, 8)?;
    write_u64(file, tokenizer.merges().len() as u64)?;
    for merge in tokenizer.merges() {
        let left = piece_name(
            merge.left as usize,
            &tokenizer.pieces()[merge.left as usize],
        )?;
        let right = piece_name(
            merge.right as usize,
            &tokenizer.pieces()[merge.right as usize],
        )?;
        write_string(file, &format_text(format_args!("{left} {right}")))?;
    }
    Ok(())
}

fn piece_name(index: usize, piece: &[u8]) -> Result<Text, Text> {
    if index == 0 {
        return Ok("<bos>".into());
    }
    if index == 1 {
        return Ok("<eos>".into());
    }
    let mut encoded = Text::from("hex:");
    for byte in piece {
        write!(encoded, "{byte:02x}").map_err(|_| Text::from("encode tokenizer piece"))?;
    }
    Ok(encoded)
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
        let transitions = train_bigram(&[0, 67, 68, 1], tokenizer.vocab_size()).unwrap();
        let model = Transformer::from_transitions(&transitions, tokenizer.vocab_size());
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
        write_gguf(&path, &article, &tokenizer, &model).unwrap();
        let gguf = mrml_tensor::GgufFile::open(&path).unwrap();
        assert_eq!(
            gguf.get_meta("general.architecture")
                .and_then(|value| value.as_str()),
            Some("mrml_transformer")
        );
        assert_eq!(
            gguf.tensors.get("output.weight").unwrap().shape.numel(),
            258 * DIM
        );
        assert!(gguf.get_meta("tokenizer.ggml.merges").is_some());
        mrml_runtime::remove_file(&path).unwrap();
    }
}
