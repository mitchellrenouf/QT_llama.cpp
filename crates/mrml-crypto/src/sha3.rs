const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

const ROTATIONS: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

fn permute(state: &mut [u64; 25]) {
    for round_constant in ROUND_CONSTANTS {
        let mut parity = [0u64; 5];
        for x in 0..5 {
            parity[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        let mut delta = [0u64; 5];
        for x in 0..5 {
            delta[x] = parity[(x + 4) % 5] ^ parity[(x + 1) % 5].rotate_left(1);
        }
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] ^= delta[x];
            }
        }

        let mut rotated = [0u64; 25];
        for y in 0..5 {
            for x in 0..5 {
                rotated[y + 5 * ((2 * x + 3 * y) % 5)] =
                    state[x + 5 * y].rotate_left(ROTATIONS[x + 5 * y]);
            }
        }

        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] = rotated[x + 5 * y]
                    ^ ((!rotated[(x + 1) % 5 + 5 * y]) & rotated[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= round_constant;
    }
}

#[derive(Clone)]
struct Sponge<const RATE: usize> {
    state: [u64; 25],
    position: usize,
}

impl<const RATE: usize> Sponge<RATE> {
    const fn new() -> Self {
        Self {
            state: [0; 25],
            position: 0,
        }
    }

    fn update(&mut self, input: &[u8]) {
        for &byte in input {
            let lane = self.position / 8;
            let shift = (self.position % 8) * 8;
            self.state[lane] ^= (byte as u64) << shift;
            self.position += 1;
            if self.position == RATE {
                permute(&mut self.state);
                self.position = 0;
            }
        }
    }

    fn finalize(mut self, domain: u8, output: &mut [u8]) {
        let lane = self.position / 8;
        let shift = (self.position % 8) * 8;
        self.state[lane] ^= (domain as u64) << shift;
        let final_position = RATE - 1;
        self.state[final_position / 8] ^= 0x80u64 << ((final_position % 8) * 8);
        permute(&mut self.state);

        let mut position = 0usize;
        for byte in output {
            if position == RATE {
                permute(&mut self.state);
                position = 0;
            }
            *byte = (self.state[position / 8] >> ((position % 8) * 8)) as u8;
            position += 1;
        }
    }
}

macro_rules! fixed_hash {
    ($name:ident, $rate:expr, $size:expr) => {
        #[derive(Clone)]
        pub struct $name(Sponge<$rate>);

        impl $name {
            pub const fn new() -> Self {
                Self(Sponge::new())
            }
            pub fn update(&mut self, input: &[u8]) {
                self.0.update(input);
            }
            pub fn finalize(self) -> [u8; $size] {
                let mut output = [0u8; $size];
                self.0.finalize(0x06, &mut output);
                output
            }
            pub fn digest(input: &[u8]) -> [u8; $size] {
                let mut hash = Self::new();
                hash.update(input);
                hash.finalize()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

macro_rules! extendable_hash {
    ($name:ident, $rate:expr) => {
        pub struct $name(Sponge<$rate>);

        impl $name {
            pub const fn new() -> Self {
                Self(Sponge::new())
            }
            pub fn update(&mut self, input: &[u8]) {
                self.0.update(input);
            }
            pub fn finalize(self, output: &mut [u8]) {
                self.0.finalize(0x1f, output);
            }
            pub fn digest(input: &[u8], output: &mut [u8]) {
                let mut hash = Self::new();
                hash.update(input);
                hash.finalize(output);
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

fixed_hash!(Sha3_256, 136, 32);
fixed_hash!(Sha3_512, 72, 64);
extendable_hash!(Shake128, 168);
extendable_hash!(Shake256, 136);

#[cfg(test)]
mod tests {
    use super::*;

    fn decode<const N: usize>(hex: &str) -> [u8; N] {
        let mut output = [0u8; N];
        for (index, byte) in output.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
        }
        output
    }

    #[test]
    fn sha3_empty_vectors() {
        assert_eq!(
            Sha3_256::digest(b""),
            decode("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
        );
        assert_eq!(
            Sha3_512::digest(b""),
            decode(
                "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
            )
        );
    }

    #[test]
    fn shake_empty_vectors() {
        let mut shake128 = [0u8; 32];
        Shake128::digest(b"", &mut shake128);
        assert_eq!(
            shake128,
            decode("7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26")
        );

        let mut shake256 = [0u8; 64];
        Shake256::digest(b"", &mut shake256);
        assert_eq!(
            shake256,
            decode(
                "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
            )
        );
    }

    #[test]
    fn incremental_absorption_matches_one_shot() {
        let mut hash = Sha3_512::new();
        hash.update(b"model ");
        hash.update(b"weights");
        assert_eq!(hash.finalize(), Sha3_512::digest(b"model weights"));
    }
}
