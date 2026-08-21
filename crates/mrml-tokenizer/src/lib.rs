#![no_std]

use mrml_runtime::Vector;

pub const BOS_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BYTE_TOKEN_OFFSET: u32 = 2;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    pub token: u32,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vector<Vector<u8>>,
    merges: Vector<Merge>,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        let mut pieces = Vector::with_capacity(258).expect("MRML allocation failed");
        pieces.push(vector_from_slice(b"<bos>"));
        pieces.push(vector_from_slice(b"<eos>"));
        for byte in 0..=255 {
            pieces.push(vector_from_slice(&[byte]));
        }
        Self {
            pieces,
            merges: Vector::new(),
        }
    }
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub fn pieces(&self) -> &[Vector<u8>] {
        &self.pieces
    }
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }
    pub fn encode(&self, text: &str, boundaries: bool) -> Vector<u32> {
        let mut tokens = Vector::with_capacity(text.len() + usize::from(boundaries) * 2).unwrap();
        if boundaries {
            tokens.push(BOS_ID);
        }
        tokens.extend(
            text.as_bytes()
                .iter()
                .map(|byte| *byte as u32 + BYTE_TOKEN_OFFSET),
        );
        for merge in &self.merges {
            apply_merge(&mut tokens, *merge);
        }
        if boundaries {
            tokens.push(EOS_ID);
        }
        tokens
    }
    pub fn decode(&self, tokens: &[u32]) -> Vector<u8> {
        let capacity = tokens
            .iter()
            .filter_map(|token| self.pieces.get(*token as usize))
            .map(Vector::len)
            .sum();
        let mut bytes = Vector::with_capacity(capacity).unwrap();
        for token in tokens {
            if *token == BOS_ID || *token == EOS_ID {
                continue;
            }
            if let Some(piece) = self.pieces.get(*token as usize) {
                bytes.try_extend_from_slice(piece).unwrap();
            }
        }
        bytes
    }
}

pub struct Trainer {
    documents: Vector<Vector<u32>>,
    tokenizer: Tokenizer,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            documents: Vector::new(),
            tokenizer: Tokenizer::byte_level(),
        }
    }
    pub fn add_document(&mut self, text: &str) {
        self.documents.push(self.tokenizer.encode(text, false));
    }
    pub fn train(mut self, target_vocab_size: usize, minimum_frequency: u64) -> Tokenizer {
        let matrix_size = target_vocab_size
            .checked_mul(target_vocab_size)
            .unwrap_or(0);
        let mut counts = Vector::with_capacity(matrix_size).expect("MRML allocation failed");
        counts.resize(matrix_size, 0u64);
        while self.tokenizer.vocab_size() < target_vocab_size {
            counts.fill(0);
            for document in &self.documents {
                for pair in document.windows(2) {
                    let index = pair[0] as usize * target_vocab_size + pair[1] as usize;
                    if let Some(count) = counts.get_mut(index) {
                        *count = count.saturating_add(1);
                    }
                }
            }
            let Some((index, frequency)) = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, count)| **count)
                .map(|(index, count)| (index, *count))
            else {
                break;
            };
            if frequency < minimum_frequency {
                break;
            }
            let left = (index / target_vocab_size) as u32;
            let right = (index % target_vocab_size) as u32;
            let token = self.tokenizer.pieces.len() as u32;
            let mut piece = self.tokenizer.pieces[left as usize].clone();
            piece
                .try_extend_from_slice(&self.tokenizer.pieces[right as usize])
                .unwrap();
            self.tokenizer.pieces.push(piece);
            let merge = Merge { left, right, token };
            self.tokenizer.merges.push(merge);
            for document in &mut self.documents {
                apply_merge(document, merge);
            }
        }
        self.tokenizer
    }
}

fn apply_merge(tokens: &mut Vector<u32>, merge: Merge) {
    let mut read = 0;
    let mut write = 0;
    while read < tokens.len() {
        if read + 1 < tokens.len() && tokens[read] == merge.left && tokens[read + 1] == merge.right
        {
            tokens[write] = merge.token;
            read += 2;
        } else {
            tokens[write] = tokens[read];
            read += 1;
        }
        write += 1;
    }
    tokens.truncate(write);
}

fn vector_from_slice<T: Clone>(slice: &[T]) -> Vector<T> {
    let mut output = Vector::with_capacity(slice.len()).unwrap();
    output.try_extend_from_slice(slice).unwrap();
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_fallback_round_trips_all_utf8() {
        let tokenizer = Tokenizer::byte_level();
        let input = "Wikipedia: café, 安全, 🦀";
        assert_eq!(
            core::str::from_utf8(&tokenizer.decode(&tokenizer.encode(input, true))).unwrap(),
            input
        );
    }
    #[test]
    fn training_learns_frequent_pairs_and_shortens_text() {
        let mut trainer = Trainer::new();
        trainer.add_document("research research research");
        trainer.add_document("research improves research");
        let tokenizer = trainer.train(280, 2);
        let encoded = tokenizer.encode("research research", false);
        assert!(tokenizer.vocab_size() > 258);
        assert!(encoded.len() < "research research".len());
        assert_eq!(
            core::str::from_utf8(&tokenizer.decode(&encoded)).unwrap(),
            "research research"
        );
    }
}
