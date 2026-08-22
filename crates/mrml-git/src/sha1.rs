use core::fmt;
use mrml_runtime::Text;

const INITIAL: [u32; 5] = [
    0x6745_2301,
    0xefcd_ab89,
    0x98ba_dcfe,
    0x1032_5476,
    0xc3d2_e1f0,
];

#[derive(Clone)]
pub struct Sha1 {
    state: [u32; 5],
    block: [u8; 64],
    used: usize,
    length: u64,
}

impl Sha1 {
    pub const fn new() -> Self {
        Self {
            state: INITIAL,
            block: [0; 64],
            used: 0,
            length: 0,
        }
    }

    pub fn update(&mut self, mut input: &[u8]) {
        self.length = self.length.wrapping_add(input.len() as u64);
        if self.used != 0 {
            let take = input.len().min(64 - self.used);
            self.block[self.used..self.used + take].copy_from_slice(&input[..take]);
            self.used += take;
            input = &input[take..];
            if self.used != 64 {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.used = 0;
        }
        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().expect("fixed SHA-1 block");
            self.compress(block);
            input = &input[64..];
        }
        self.block[..input.len()].copy_from_slice(input);
        self.used = input.len();
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut words = [0u32; 80];
        for (word, bytes) in words[..16].iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes(*bytes);
        }
        for index in 16..80 {
            words[index] = (words[index - 3]
                ^ words[index - 8]
                ^ words[index - 14]
                ^ words[index - 16])
                .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for (index, word) in words.into_iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e]) {
            *state = state.wrapping_add(value);
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let bit_length = self.length.wrapping_mul(8);
        self.block[self.used] = 0x80;
        self.used += 1;
        if self.used > 56 {
            self.block[self.used..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.used = 0;
        }
        self.block[self.used..56].fill(0);
        self.block[56..].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.block;
        self.compress(&block);
        let mut output = [0u8; 20];
        for (bytes, word) in output.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        output
    }

    pub fn digest(input: &[u8]) -> [u8; 20] {
        let mut hash = Self::new();
        hash.update(input);
        hash.finalize()
    }
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ObjectId(pub [u8; 20]);

impl ObjectId {
    pub fn parse(hex: &str) -> Option<Self> {
        if hex.len() != 40 {
            return None;
        }
        let mut bytes = [0; 20];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = decode_hex(hex.as_bytes()[index * 2])?
                .checked_mul(16)?
                .checked_add(decode_hex(hex.as_bytes()[index * 2 + 1])?)?;
        }
        Some(Self(bytes))
    }

    pub fn to_hex(self) -> Text {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = Text::new();
        for byte in self.0 {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 15) as usize] as char);
        }
        output
    }

    pub fn blob(contents: &[u8]) -> Self {
        let mut hash = Sha1::new();
        let header = mrml_runtime::mrml_format!("blob {}\0", contents.len());
        hash.update(header.as_bytes());
        hash.update(contents);
        Self(hash.finalize())
    }
}

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_sha1_vectors() {
        assert_eq!(
            ObjectId(Sha1::digest(b"")).to_hex(),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            ObjectId(Sha1::digest(b"abc")).to_hex(),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
    }

    #[test]
    fn git_empty_blob_vector() {
        assert_eq!(
            ObjectId::blob(b"").to_hex(),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        );
    }

    #[test]
    fn split_updates_match_one_shot() {
        let mut hash = Sha1::new();
        hash.update(&[b'a'; 60]);
        hash.update(&[b'a'; 40]);
        assert_eq!(hash.finalize(), Sha1::digest(&[b'a'; 100]));
    }
}
