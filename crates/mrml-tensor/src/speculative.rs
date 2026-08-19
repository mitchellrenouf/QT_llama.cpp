use mrml_runtime::{OrderedMap, Vector};

/// Speculative Decoding Engine using fast N-Gram candidate prediction and batched verification
pub struct SpeculativeDecoder {
    /// N-Gram sequence window (e.g. 3 or 4)
    n_gram_order: usize,
    /// Number of speculative draft tokens to propose per step (e.g. 3 to 5)
    num_draft_tokens: usize,
    /// N-gram transitions map: Prefix [u32] -> Next candidate tokens
    ngram_table: OrderedMap<Vector<i32>, Vector<i32>>,
    /// Acceptance counter for performance metrics
    total_proposed: usize,
    total_accepted: usize,
}

impl SpeculativeDecoder {
    pub fn new(n_gram_order: usize, num_draft_tokens: usize) -> Self {
        Self {
            n_gram_order: n_gram_order.max(2),
            num_draft_tokens: num_draft_tokens.max(1),
            ngram_table: OrderedMap::new(),
            total_proposed: 0,
            total_accepted: 0,
        }
    }

    /// Record observed token sequence to learn local patterns for draft prediction
    pub fn record_sequence(&mut self, tokens: &[i32]) {
        if tokens.len() <= self.n_gram_order {
            return;
        }

        for i in 0..tokens.len() - self.n_gram_order {
            let key: Vector<i32> = tokens[i..i + self.n_gram_order].iter().copied().collect();
            let next_tok = tokens[i + self.n_gram_order];

            if let Some(candidates) = self.ngram_table.get_mut(&key) {
                if !candidates.contains(&next_tok) {
                    candidates.push(next_tok);
                }
            } else {
                self.ngram_table.insert(key, [next_tok].into());
            }
        }
    }

    /// Generate $K$ speculative draft tokens from current context history
    pub fn propose_draft_tokens(&mut self, context: &[i32]) -> Vector<i32> {
        let mut draft =
            Vector::with_capacity(self.num_draft_tokens).expect("MRML allocation failed");
        let mut virtual_ctx: Vector<i32> = context.iter().copied().collect();

        for _ in 0..self.num_draft_tokens {
            if virtual_ctx.len() < self.n_gram_order {
                break;
            }

            let start = virtual_ctx.len() - self.n_gram_order;
            let key = &virtual_ctx[start..];

            if let Some(next_tokens) = self.ngram_table.get(key) {
                if let Some(&predicted) = next_tokens.first() {
                    draft.push(predicted);
                    virtual_ctx.push(predicted);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.total_proposed += draft.len();
        draft
    }

    /// Verify draft tokens against target model outputs and return accepted tokens
    pub fn verify_draft(&mut self, draft: &[i32], actual_candidates: &[i32]) -> usize {
        let mut accepted = 0;
        for (d, a) in draft.iter().zip(actual_candidates.iter()) {
            if d == a {
                accepted += 1;
            } else {
                break;
            }
        }
        self.total_accepted += accepted;
        accepted
    }

    /// Acceptance rate metric for speed profiling
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_proposed as f32
        }
    }
}
