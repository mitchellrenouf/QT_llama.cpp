#![no_std]

// Original bootstrap conformance cases for the MRML Rust compiler. These are
// intentionally small language observations, not copies of upstream tests.

#[test]
fn integer_precedence_and_wrapping_are_stable() {
    assert_eq!(2 + 3 * 4, 14);
    assert_eq!((2 + 3) * 4, 20);
    assert_eq!(u8::MAX.wrapping_add(1), 0);
    assert_eq!(1u32 << 31, 0x8000_0000);
}

#[test]
fn references_and_mutable_reborrows_preserve_identity() {
    fn increment(value: &mut u32) {
        *value += 1;
    }

    let mut value = 40;
    increment(&mut value);
    increment(&mut value);
    assert_eq!(value, 42);
}

#[test]
fn enums_match_payloads_and_guards() {
    enum Value {
        Empty,
        Number(u32),
        Pair(u16, u16),
    }

    fn evaluate(value: Value) -> u32 {
        match value {
            Value::Number(number) if number > 10 => number,
            Value::Pair(left, right) => u32::from(left) + u32::from(right),
            Value::Empty | Value::Number(_) => 0,
        }
    }

    assert_eq!(evaluate(Value::Empty), 0);
    assert_eq!(evaluate(Value::Number(42)), 42);
    assert_eq!(evaluate(Value::Pair(20, 22)), 42);
}

#[test]
fn generics_and_associated_types_dispatch() {
    trait Source {
        type Item;
        fn get(&self) -> Self::Item;
    }

    struct Constant<T: Copy>(T);

    impl<T: Copy> Source for Constant<T> {
        type Item = T;

        fn get(&self) -> Self::Item {
            self.0
        }
    }

    fn read<S: Source>(source: &S) -> S::Item {
        source.get()
    }

    assert_eq!(read(&Constant(42u32)), 42);
}

#[test]
fn const_evaluation_builds_values() {
    const fn triangular(limit: usize) -> usize {
        let mut value = 0;
        let mut next = 1;
        while next <= limit {
            value += next;
            next += 1;
        }
        value
    }

    const VALUE: usize = triangular(9);
    const ARRAY: [u8; VALUE] = [7; VALUE];
    assert_eq!(ARRAY.len(), 45);
    assert_eq!(ARRAY[44], 7);
}

#[test]
fn closures_capture_by_reference_and_value() {
    let offset = 2;
    let borrowed = |value| value + offset;
    assert_eq!(borrowed(40), 42);

    let owned = move |value| value + offset;
    assert_eq!(owned(40), 42);
}

#[test]
fn slices_and_iterators_observe_bounds() {
    let values = [10, 20, 12];
    assert_eq!(values.get(3), None);
    assert_eq!(values.iter().copied().sum::<u32>(), 42);
    assert_eq!(&values[1..], &[20, 12]);
}

#[test]
fn representation_has_expected_core_invariants() {
    assert_eq!(core::mem::size_of::<u64>(), 8);
    assert_eq!(core::mem::align_of::<u64>(), 8);
    assert_eq!(
        core::mem::size_of::<Option<&u8>>(),
        core::mem::size_of::<&u8>()
    );
    assert_eq!(core::mem::size_of::<[u16; 4]>(), 8);
}

#[test]
fn question_mark_propagates_errors() {
    fn add(left: Result<u32, u8>, right: Result<u32, u8>) -> Result<u32, u8> {
        Ok(left? + right?)
    }

    assert_eq!(add(Ok(20), Ok(22)), Ok(42));
    assert_eq!(add(Err(7), Ok(22)), Err(7));
}

#[unsafe(no_mangle)]
pub extern "C" fn mrml_conformance_identity(value: u64) -> u64 {
    value
}

#[test]
fn explicit_c_abi_export_preserves_integer_values() {
    assert_eq!(mrml_conformance_identity(42), 42);
}
