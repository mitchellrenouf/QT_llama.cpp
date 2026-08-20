const MASK: u64 = (1u64 << 51) - 1;

#[derive(Clone, Copy)]
struct Field([u64; 5]);

impl Field {
    const ZERO: Self = Self([0; 5]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u64; 5];
        for bit in 0..255 {
            limbs[bit / 51] |= (((bytes[bit / 8] >> (bit % 8)) & 1) as u64) << (bit % 51);
        }
        Self(limbs)
    }

    fn carry(&mut self) {
        for index in 0..4 {
            let carry = self.0[index] >> 51;
            self.0[index] &= MASK;
            self.0[index + 1] += carry;
        }
        let carry = self.0[4] >> 51;
        self.0[4] &= MASK;
        self.0[0] += carry * 19;
        let carry = self.0[0] >> 51;
        self.0[0] &= MASK;
        self.0[1] += carry;
    }

    fn add(self, other: Self) -> Self {
        let mut output = Self([0; 5]);
        for index in 0..5 {
            output.0[index] = self.0[index] + other.0[index];
        }
        output.carry();
        output
    }

    fn sub(self, other: Self) -> Self {
        let mut output = Self([0; 5]);
        output.0[0] = self.0[0] + (1u64 << 52) - 38 - other.0[0];
        for index in 1..5 {
            output.0[index] = self.0[index] + (1u64 << 52) - 2 - other.0[index];
        }
        output.carry();
        output
    }

    fn mul(self, other: Self) -> Self {
        let a = self.0;
        let b = other.0;
        let mut wide = [0u128; 5];
        wide[0] = a[0] as u128 * b[0] as u128
            + 19 * (a[1] as u128 * b[4] as u128
                + a[2] as u128 * b[3] as u128
                + a[3] as u128 * b[2] as u128
                + a[4] as u128 * b[1] as u128);
        wide[1] = a[0] as u128 * b[1] as u128
            + a[1] as u128 * b[0] as u128
            + 19 * (a[2] as u128 * b[4] as u128
                + a[3] as u128 * b[3] as u128
                + a[4] as u128 * b[2] as u128);
        wide[2] = a[0] as u128 * b[2] as u128
            + a[1] as u128 * b[1] as u128
            + a[2] as u128 * b[0] as u128
            + 19 * (a[3] as u128 * b[4] as u128 + a[4] as u128 * b[3] as u128);
        wide[3] = a[0] as u128 * b[3] as u128
            + a[1] as u128 * b[2] as u128
            + a[2] as u128 * b[1] as u128
            + a[3] as u128 * b[0] as u128
            + 19 * a[4] as u128 * b[4] as u128;
        wide[4] = a[0] as u128 * b[4] as u128
            + a[1] as u128 * b[3] as u128
            + a[2] as u128 * b[2] as u128
            + a[3] as u128 * b[1] as u128
            + a[4] as u128 * b[0] as u128;
        for index in 0..4 {
            let carry = wide[index] >> 51;
            wide[index] &= MASK as u128;
            wide[index + 1] += carry;
        }
        let carry = wide[4] >> 51;
        wide[4] &= MASK as u128;
        wide[0] += carry * 19;
        let carry = wide[0] >> 51;
        wide[0] &= MASK as u128;
        wide[1] += carry;
        let mut output = Self([
            wide[0] as u64,
            wide[1] as u64,
            wide[2] as u64,
            wide[3] as u64,
            wide[4] as u64,
        ]);
        output.carry();
        output
    }

    fn square(self) -> Self {
        self.mul(self)
    }
    fn mul_small(self, value: u64) -> Self {
        self.mul(Self([value, 0, 0, 0, 0]))
    }

    fn invert(self) -> Self {
        let exponent = [
            0xeb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ];
        let mut result = Self::ONE;
        for bit in (0..255).rev() {
            result = result.square();
            if (exponent[bit / 8] >> (bit % 8)) & 1 != 0 {
                result = result.mul(self);
            }
        }
        result
    }

    fn conditional_swap(left: &mut Self, right: &mut Self, swap: u64) {
        let mask = 0u64.wrapping_sub(swap);
        for index in 0..5 {
            let difference = mask & (left.0[index] ^ right.0[index]);
            left.0[index] ^= difference;
            right.0[index] ^= difference;
        }
    }

    fn to_bytes(mut self) -> [u8; 32] {
        self.carry();
        self.carry();
        let modulus = [MASK - 18, MASK, MASK, MASK, MASK];
        let mut reduced = [0u64; 5];
        let mut borrow = 0u64;
        for index in 0..5 {
            let subtrahend = modulus[index] + borrow;
            reduced[index] = self.0[index].wrapping_sub(subtrahend) & MASK;
            borrow = (self.0[index] < subtrahend) as u64;
        }
        let select = 0u64.wrapping_sub(borrow ^ 1);
        let reject = !select;
        for index in 0..5 {
            self.0[index] = (self.0[index] & reject) | (reduced[index] & select);
        }
        let mut output = [0u8; 32];
        for bit in 0..255 {
            output[bit / 8] |= (((self.0[bit / 51] >> (bit % 51)) & 1) as u8) << (bit % 8);
        }
        output
    }
}

pub fn x25519(mut scalar: [u8; 32], point: [u8; 32]) -> [u8; 32] {
    scalar[0] &= 248;
    scalar[31] &= 127;
    scalar[31] |= 64;
    let x1 = Field::from_bytes(&point);
    let mut x2 = Field::ONE;
    let mut z2 = Field::ZERO;
    let mut x3 = x1;
    let mut z3 = Field::ONE;
    let mut swap = 0u64;
    for position in (0..255).rev() {
        let bit = ((scalar[position / 8] >> (position % 8)) & 1) as u64;
        swap ^= bit;
        Field::conditional_swap(&mut x2, &mut x3, swap);
        Field::conditional_swap(&mut z2, &mut z3, swap);
        swap = bit;
        let a = x2.add(z2);
        let aa = a.square();
        let b = x2.sub(z2);
        let bb = b.square();
        let e = aa.sub(bb);
        let c = x3.add(z3);
        let d = x3.sub(z3);
        let da = d.mul(a);
        let cb = c.mul(b);
        x3 = da.add(cb).square();
        z3 = x1.mul(da.sub(cb).square());
        x2 = aa.mul(bb);
        z2 = e.mul(aa.add(e.mul_small(121665)));
    }
    Field::conditional_swap(&mut x2, &mut x3, swap);
    Field::conditional_swap(&mut z2, &mut z3, swap);
    x2.mul(z2.invert()).to_bytes()
}

pub fn x25519_public(secret: [u8; 32]) -> [u8; 32] {
    let mut base = [0u8; 32];
    base[0] = 9;
    x25519(secret, base)
}

pub fn x25519_shared(secret: [u8; 32], peer: [u8; 32]) -> Option<[u8; 32]> {
    let shared = x25519(secret, peer);
    let mut nonzero = 0u8;
    for byte in shared {
        nonzero |= byte;
    }
    (nonzero != 0).then_some(shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode(hex: &str) -> [u8; 32] {
        let mut out = [0; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
    #[test]
    fn rfc_7748_alice_public_key() {
        let secret = decode("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        assert_eq!(
            x25519_public(secret),
            decode("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
        );
    }
    #[test]
    fn rejects_all_zero_shared_secret() {
        assert!(x25519_shared([7; 32], [0; 32]).is_none());
    }
    #[test]
    fn rfc_7748_iteration_vector() {
        let mut nine = [0u8; 32];
        nine[0] = 9;
        assert_eq!(
            x25519(nine, nine),
            decode("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")
        );
    }
    #[test]
    fn independent_keypairs_agree() {
        let left = [0x39; 32];
        let right = [0xa7; 32];
        let a = x25519_shared(left, x25519_public(right)).unwrap();
        let b = x25519_shared(right, x25519_public(left)).unwrap();
        assert_eq!(a, b);
    }
}
