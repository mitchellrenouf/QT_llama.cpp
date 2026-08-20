use mrml_runtime::{Text, Vector};

pub const DIM: usize = 64;
pub const HEADS: usize = 4;
pub const CONTEXT: usize = 32;
pub const FFN: usize = 128;

pub struct Transformer {
    pub vocab: usize,
    pub token_embd: Vector<f32>,
    pub position_embd: Vector<f32>,
    pub attn_q: Vector<f32>,
    pub attn_k: Vector<f32>,
    pub attn_v: Vector<f32>,
    pub attn_output: Vector<f32>,
    pub ffn_up: Vector<f32>,
    pub ffn_down: Vector<f32>,
    pub output: Vector<f32>,
}

pub struct TrainingReport {
    pub steps: usize,
    pub accuracy: f32,
}

impl Transformer {
    pub fn from_transitions(transitions: &[f32], vocab: usize) -> Self {
        let mut token_embd = filled(vocab * DIM, 0.0);
        for token in 0..vocab {
            for dim in 0..DIM {
                token_embd[token * DIM + dim] = signed_hash(token as u64, dim as u64) * 0.125;
            }
        }
        let mut position_embd = filled(CONTEXT * DIM, 0.0);
        for position in 0..CONTEXT {
            for dim in 0..DIM {
                position_embd[position * DIM + dim] =
                    signed_hash(0x706f_7369_7469_6f6e ^ position as u64, dim as u64) * 0.01;
            }
        }
        let mut attn_q = identity(DIM, 1.6);
        let mut attn_k = identity(DIM, 1.6);
        let attn_v = identity(DIM, 1.0);
        let attn_output = identity(DIM, 0.35);
        for row in 0..DIM {
            for column in 0..DIM {
                if row != column {
                    attn_q[row * DIM + column] =
                        signed_hash(11 + row as u64, column as u64) * 0.002;
                    attn_k[row * DIM + column] =
                        signed_hash(29 + row as u64, column as u64) * 0.002;
                }
            }
        }
        let mut ffn_up = filled(FFN * DIM, 0.0);
        let mut ffn_down = filled(DIM * FFN, 0.0);
        for row in 0..FFN {
            for column in 0..DIM {
                ffn_up[row * DIM + column] = signed_hash(101 + row as u64, column as u64) * 0.015;
                ffn_down[column * FFN + row] = signed_hash(211 + column as u64, row as u64) * 0.003;
            }
        }
        // Project the empirical next-token distributions into the embedding
        // basis. This is the corpus-trained portion of the first transformer
        // checkpoint and provides a fast initialization for later SGD.
        let mut output = filled(vocab * DIM, 0.0);
        for source in 0..vocab {
            let embedding = &token_embd[source * DIM..(source + 1) * DIM];
            for target in 0..vocab {
                let probability = transitions[source * vocab + target];
                for dim in 0..DIM {
                    output[target * DIM + dim] += probability * embedding[dim];
                }
            }
        }
        Self {
            vocab,
            token_embd,
            position_embd,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_up,
            ffn_down,
            output,
        }
    }

    pub fn generate(
        &self,
        tokenizer: &mrml_tokenizer::Tokenizer,
        prompt: &str,
        maximum: usize,
    ) -> Text {
        let mut tokens = tokenizer.encode(prompt, false);
        let mut seed = prompt
            .as_bytes()
            .iter()
            .fold(0x9e37_79b9u64, |state, byte| {
                state.rotate_left(5) ^ *byte as u64
            });
        let mut generated = Vector::new();
        for _ in 0..maximum {
            let logits = self.forward(&tokens);
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let next = sample_logits(&logits, seed);
            if next == mrml_tokenizer::EOS_ID {
                break;
            }
            tokens.push(next);
            generated.push(next);
        }
        Text::from_utf8_lossy(&tokenizer.decode(&generated))
    }

    pub fn train_output(
        &mut self,
        tokens: &[u32],
        steps: usize,
        learning_rate: f32,
    ) -> TrainingReport {
        if tokens.len() < 2 || steps == 0 {
            return TrainingReport {
                steps: 0,
                accuracy: 0.0,
            };
        }
        let actual_steps = steps.min(tokens.len().saturating_mul(8));
        let mut correct = 0usize;
        for step in 0..actual_steps {
            let position = 1 + step.wrapping_mul(1_000_003) % (tokens.len() - 1);
            let start = position.saturating_sub(CONTEXT);
            let hidden = self.hidden(&tokens[start..position]);
            let logits = matvec(&self.output, &hidden, self.vocab, DIM);
            let predicted = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(token, _)| token)
                .unwrap_or(0);
            let target = tokens[position] as usize % self.vocab;
            correct += usize::from(predicted == target);
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities =
                Vector::with_capacity(self.vocab).expect("MRML allocation failed");
            let mut sum = 0.0;
            for logit in logits {
                let value = mrml_math::exp(logit - maximum);
                probabilities.push(value);
                sum += value;
            }
            let rate = learning_rate / (1.0 + step as f32 / actual_steps as f32);
            for token in 0..self.vocab {
                let gradient = probabilities[token] / sum - if token == target { 1.0 } else { 0.0 };
                for dim in 0..DIM {
                    self.output[token * DIM + dim] -= rate * gradient * hidden[dim];
                }
            }
        }
        TrainingReport {
            steps: actual_steps,
            accuracy: correct as f32 / actual_steps as f32,
        }
    }

    fn forward(&self, tokens: &[u32]) -> Vector<f32> {
        let hidden = self.hidden(tokens);
        matvec(&self.output, &hidden, self.vocab, DIM)
    }

    fn hidden(&self, tokens: &[u32]) -> Vector<f32> {
        let start = tokens.len().saturating_sub(CONTEXT);
        let active = &tokens[start..];
        let length = active.len().max(1);
        let mut states = filled(length * DIM, 0.0);
        for position in 0..length {
            let token = active.get(position).copied().unwrap_or(0) as usize % self.vocab;
            for dim in 0..DIM {
                states[position * DIM + dim] =
                    self.token_embd[token * DIM + dim] + self.position_embd[position * DIM + dim];
            }
            rms_norm(&mut states[position * DIM..(position + 1) * DIM]);
        }
        let last = length - 1;
        let query = matvec(
            &self.attn_q,
            &states[last * DIM..(last + 1) * DIM],
            DIM,
            DIM,
        );
        let mut attended = filled(DIM, 0.0);
        let head_dim = DIM / HEADS;
        for head in 0..HEADS {
            let offset = head * head_dim;
            let mut scores = filled(length, 0.0);
            let mut maximum = f32::NEG_INFINITY;
            for position in 0..length {
                let key = matvec(
                    &self.attn_k,
                    &states[position * DIM..(position + 1) * DIM],
                    DIM,
                    DIM,
                );
                let mut score = 0.0;
                for dim in 0..head_dim {
                    score += query[offset + dim] * key[offset + dim];
                }
                score /= mrml_math::sqrt(head_dim as f32);
                scores[position] = score;
                maximum = maximum.max(score);
            }
            let mut sum = 0.0;
            for score in &mut scores {
                *score = mrml_math::exp(*score - maximum);
                sum += *score;
            }
            for position in 0..length {
                let value = matvec(
                    &self.attn_v,
                    &states[position * DIM..(position + 1) * DIM],
                    DIM,
                    DIM,
                );
                let probability = scores[position] / sum;
                for dim in 0..head_dim {
                    attended[offset + dim] += probability * value[offset + dim];
                }
            }
        }
        let projected = matvec(&self.attn_output, &attended, DIM, DIM);
        let mut hidden = filled(DIM, 0.0);
        for dim in 0..DIM {
            hidden[dim] = states[last * DIM + dim] + projected[dim];
        }
        rms_norm(&mut hidden);
        let mut expanded = matvec(&self.ffn_up, &hidden, FFN, DIM);
        for value in &mut expanded {
            *value = gelu(*value);
        }
        let reduced = matvec(&self.ffn_down, &expanded, DIM, FFN);
        for dim in 0..DIM {
            hidden[dim] += reduced[dim];
        }
        rms_norm(&mut hidden);
        hidden
    }
}

fn sample_logits(logits: &[f32], seed: u64) -> u32 {
    const TOP_K: usize = 8;
    let mut candidates = Vector::with_capacity(TOP_K).expect("MRML allocation failed");
    for (token, logit) in logits.iter().copied().enumerate() {
        let insertion = candidates
            .iter()
            .position(|(_, value): &(usize, f32)| logit > *value)
            .unwrap_or(candidates.len());
        if insertion < TOP_K {
            candidates
                .try_insert(insertion, (token, logit))
                .expect("MRML allocation failed");
            if candidates.len() > TOP_K {
                candidates.pop();
            }
        }
    }
    let maximum = candidates.first().map(|(_, value)| *value).unwrap_or(0.0);
    let mut probabilities =
        Vector::with_capacity(candidates.len()).expect("MRML allocation failed");
    let mut sum = 0.0;
    for (_, logit) in &candidates {
        let value = mrml_math::exp((*logit - maximum) / 0.55);
        probabilities.push(value);
        sum += value;
    }
    let target = (seed >> 32) as u32 as f32 / u32::MAX as f32 * sum;
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += *probability;
        if target <= cumulative {
            return candidates[index].0 as u32;
        }
    }
    candidates
        .last()
        .map(|(token, _)| *token as u32)
        .unwrap_or(0)
}

fn matvec(matrix: &[f32], input: &[f32], rows: usize, columns: usize) -> Vector<f32> {
    let mut output = filled(rows, 0.0);
    for row in 0..rows {
        for column in 0..columns {
            output[row] += matrix[row * columns + column] * input[column];
        }
    }
    output
}

fn rms_norm(values: &mut [f32]) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = 1.0 / mrml_math::sqrt(mean + 1e-5);
    for value in values {
        *value *= scale;
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + mrml_math::tanh(0.797_884_6 * (value + 0.044_715 * value * value * value)))
}

fn identity(size: usize, diagonal: f32) -> Vector<f32> {
    let mut values = filled(size * size, 0.0);
    for index in 0..size {
        values[index * size + index] = diagonal;
    }
    values
}

fn filled<T: Clone>(length: usize, value: T) -> Vector<T> {
    let mut values = Vector::with_capacity(length).expect("MRML allocation failed");
    values.resize(length, value);
    values
}

fn signed_hash(left: u64, right: u64) -> f32 {
    let mut value =
        left.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ right.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    ((value >> 40) as i32 - 8_388_608) as f32 / 8_388_608.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_attention_changes_logits_with_history() {
        let vocab = 258;
        let mut transitions = filled(vocab * vocab, 1.0 / vocab as f32);
        transitions[2 * vocab + 3] = 1.0;
        let model = Transformer::from_transitions(&transitions, vocab);
        let short = model.forward(&[2]);
        let contextual = model.forward(&[7, 2]);
        assert!(short
            .iter()
            .zip(&contextual)
            .any(|(left, right)| (left - right).abs() > 1e-5));
    }

    #[test]
    fn contextual_cross_entropy_training_learns_repetition() {
        let vocab = 258;
        let transitions = filled(vocab * vocab, 1.0 / vocab as f32);
        let mut model = Transformer::from_transitions(&transitions, vocab);
        let mut tokens = Vector::new();
        for _ in 0..128 {
            tokens.extend([2, 3]);
        }
        let report = model.train_output(&tokens, 500, 0.03);
        assert!(report.accuracy > 0.5, "accuracy={}", report.accuracy);
    }
}
