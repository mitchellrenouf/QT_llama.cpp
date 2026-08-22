#![no_std]

// Original MRML cases mapped to the actual rustc test suite at the installed
// nightly commit. No upstream source or expected-output text is reproduced.
//
// Reference commit: f7d782a3be46d6bb4b9792fe69a61db389ba1769
// Reference cases:
// - tests/ui/numbers-arithmetic/arith-unsigned.rs
// - tests/ui/numbers-arithmetic/div-mod.rs
// - tests/ui/numbers-arithmetic/shift.rs
// - tests/ui/numbers-arithmetic/bitwise-ops-platform.rs
// - tests/ui/numbers-arithmetic/overflowing-lsh-4.rs
// - tests/ui/numbers-arithmetic/overflowing-rsh-4.rs
// - tests/ui/numbers-arithmetic/overflowing-add.rs
// - tests/ui/numbers-arithmetic/overflowing-sub.rs
// - tests/ui/numbers-arithmetic/overflowing-mul.rs
// - tests/ui/numbers-arithmetic/overflowing-neg-nonzero.rs
// - tests/ui/numbers-arithmetic/i128-min-literal-parses.rs
// - tests/ui/consts/control-flow/short-circuit.rs
// - tests/ui/consts/control-flow/basics.rs
// - tests/ui/consts/return-in-const-fn.rs
// - tests/ui/for-loop-while/while.rs
// - tests/ui/for-loop-while/long-while.rs
// - tests/ui/for-loop-while/while-with-break.rs
// - tests/ui/for-loop-while/break.rs
// - tests/ui/for-loop-while/while-cont.rs
// - tests/ui/for-loop-while/loop-break-cont-1.rs
// - tests/ui/for-loop-while/loop-break-cont.rs
// - tests/ui/for-loop-while/loop-break-value.rs
// - tests/ui/for-loop-while/for-loop-has-unit-body.rs
// - tests/ui/expr/if-generic.rs
// - tests/ui/expr/weird-exprs.rs
// - tests/ui/binop/binops.rs
// - tests/ui/consts/min_const_fn/min_const_fn.rs
// - tests/ui/consts/const-meth-pattern.rs
// - tests/ui/consts/const_let_eq.rs
// - tests/ui/consts/const-eval/stable-metric/ctfe-simple-loop.rs
// - tests/ui/lint/unused/issue-117142-invalid-remove-parens.rs
// - tests/ui/liveness/var-defined-after-early-return.rs
// - tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs
// - tests/ui/expr/if/if-check.rs
// - tests/ui/consts/const-eval/issue-64970.rs
// - tests/ui/binding/if-let.rs
// - tests/ui/drop/dynamic-drop.rs
// - tests/ui/structs-enums/class-impl-very-parameterized-trait.rs
// - tests/ui/drop/drop-on-ret.rs
// - tests/ui/consts/const-fn-const-eval.rs
// - tests/ui/consts/const-extern-fn/const-extern-fn.rs
// - tests/ui/consts/const-negative.rs
// - tests/ui/consts/const-binops.rs
// - tests/ui/cast/constant-expression-cast-9942.rs
// - tests/ui/consts/const-block-items/assert-pass.rs
// - tests/ui/consts/const-block.rs
// - tests/ui/consts/const-blocks/const-block-in-array-size.rs
// - tests/ui/inline-const/referencing-local-variables.rs
// - tests/ui/consts/const-blocks/fn-call-in-const.rs
// - tests/ui/inline-const/const-expr-basic.rs
// - tests/ui/consts/consts-in-patterns.rs
// - tests/ui/binding/match-range.rs
// - tests/ui/binding/match-range-static.rs
// - tests/ui/match/guards.rs
// - tests/ui/match/guard-pattern-ordering-14865.rs
// - tests/ui/match/guard-arm-and-or-arm.rs
// - tests/ui/match/range-arm-and-ref-arm.rs
// - tests/ui/binding/match-pattern-bindings.rs
// - tests/ui/or-patterns/bindings-runpass-2.rs
// - tests/ui/or-patterns/missing-bindings.rs
// - tests/ui/or-patterns/inner-or-pat.rs
// - tests/ui/or-patterns/or-patterns-syntactic-pass.rs
// - tests/ui/match/large-match-mir-gen.rs
// - tests/ui/or-patterns/exhaustiveness-pass.rs
// - tests/ui/or-patterns/exhaustiveness-non-exhaustive.rs
// - tests/ui/error-codes/E0030.rs
// - tests/ui/match/match-range-fail-2.rs
// - tests/ui/consts/const-eval/const_signed_pat.rs
// - tests/ui/match/overeager-sub-match-pruning-13027.rs
// - tests/ui/half-open-range-patterns/range_pat_interactions0.rs
// - tests/ui/lint/unused_braces.rs
// - tests/ui/consts/const-fn-nested.rs
// - tests/codegen-llvm/overflow-checks.rs
// - tests/codegen-llvm/integer-overflow.rs
// - tests/codegen-llvm/abi-x86_64_sysv.rs

use mrml_rustc::{
    CodegenErrorKind, CodegenOptions, CompileErrorKind, ConstEvalError, ExecutionError,
    ExpressionErrorKind, ObjectFormat, TargetLayout, compile_source_function,
    compile_source_function_with_options,
};

fn compile(source: &str, format: ObjectFormat) -> Result<(), CompileErrorKind> {
    compile_source_function::<2048, 512, 4, 2, 4, 32>(source, "probe", format, TargetLayout::X86_64)
        .map(|_| ())
        .map_err(|error| error.kind)
}

fn compile_wide(source: &str, format: ObjectFormat) -> Result<(), CompileErrorKind> {
    compile_source_function::<4096, 2048, 4, 8, 8, 64>(
        source,
        "probe",
        format,
        TargetLayout::X86_64,
    )
    .map(|_| ())
    .map_err(|error| error.kind)
}

#[test]
fn rustc_binding_if_let_assignment_order_has_a_scalar_replacement() {
    // This original scalar case maps only the pinned test's ordered assignment
    // behavior. It does not claim support for `if let` or pattern matching.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { let mut clause: usize = 0; if value == 1 { clause = 1; } else if value == 2 { clause = 2; } else if value == 3 { clause = 3; } else { clause = 4; } clause }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_dynamic_drop_has_an_ordered_scalar_mutation_replacement() {
    // This original scalar case maps only source-ordered mutations in a
    // selected branch. It does not claim ownership, drop, allocation,
    // deferred initialization, tuples, unwinding, or coroutine support.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { let mut result = value; if value == 0 { result = 20; value + 1; result *= 2; result += 2; } else if value == 1 { result = 100; value * 3; result -= 58; } else { result = 126 / value; value + 10; result += 0; } result }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_weird_exprs_has_a_guarded_discarded_expression_replacement() {
    // The pinned test contains discarded expressions in unusual block
    // contexts. This original scalar replacement covers their ordered,
    // value-discarding behavior inside selected conditional branches only.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { let mut result = value; if value == 0 { result = 40; value + 1; result += 2; } else if value == 1 { result = 40; 84 / value; result += 2; } else { result = 40; value == 3; result += 2; } result }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_parameterized_class_has_a_scalar_mutate_then_return_replacement() {
    // The pinned method performs an expression, mutation, and return in each
    // selected branch. This original replacement covers only the scalar action
    // order; it does not claim structs, methods, generics, formatting, or
    // ownership behavior.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> bool { let mut hunger: i32 = value; if hunger > 0 { hunger + 10; hunger -= 2; return hunger == value - 2; } else { hunger - 10; hunger += 2; return hunger == value + 2; } false }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_arith_unsigned_and_div_mod_runtime_subset_reaches_native_objects() {
    let probes = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8, divisor: u8) -> u8 { value / divisor }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u16, divisor: u16) -> u16 { value % divisor }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u32, right: u32) -> u32 { left + right * 3 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, divisor: u64) -> u64 { value / divisor }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, divisor: u64) -> u64 { value % divisor }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u64, right: u64) -> u64 { left + right * 3 }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_arith_unsigned_comparisons_emit_typed_bool_results() {
    let probes = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u8, right: u8) -> bool { left < right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u16, right: u16) -> bool { left <= right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u32, right: u32) -> bool { left > right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u64, right: u64) -> bool { left >= right }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_shift_and_platform_bitwise_runtime_subset_reaches_native_objects() {
    let probes = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, distance: u64) -> u64 { value << distance }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, distance: u64) -> u64 { value >> distance }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u64, right: u64) -> u64 { !(left ^ right) & 255 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: usize, right: usize) -> usize { left | right }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_signed_division_arithmetic_and_bitwise_cases_reach_native_objects() {
    let probes = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: i8, right: i8) -> i8 { left / right + left % right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: i16, right: i16) -> i16 { left - right * 3 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: i32, right: i32) -> i32 { !(left ^ right) }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: i64, right: i64) -> i64 { left >> right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: isize, right: isize) -> isize { left | right }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_negative_signed_constants_reach_native_objects() {
    let probes = [
        "static OFFSET: isize = -1; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: isize) -> isize { value + OFFSET }",
        "const MINIMUM: i8 = -128; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { value | MINIMUM }",
        "const OFFSET: i32 = -2; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + OFFSET }",
        "static OFFSET: isize = -4 + 3; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: isize) -> isize { value + OFFSET }",
        "static FACTOR: i32 = -3 * 3; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + FACTOR }",
        "static QUOTIENT: i16 = 3 / -1; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i16) -> i16 { value + QUOTIENT }",
        "static MASK: i8 = -8 | 3; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { value & MASK }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }

    let mismatch = "const OFFSET: i32 = -1; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i64) -> i64 { value + OFFSET }";
    assert_eq!(
        compile(mismatch, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            mrml_rustc::CodegenErrorKind::RuntimeTypeMismatch
        ))
    );
}

#[test]
fn rustc_overflow_checks_codegen_modes_emit_distinct_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + 1 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { value + 1 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i16) -> i16 { -value }",
    ];
    for source in sources {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            let checked = compile_source_function_with_options::<2048, 512, 4, 2, 4, 32>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
                CodegenOptions::CHECKED,
            )
            .unwrap();
            let wrapping = compile_source_function_with_options::<2048, 512, 4, 2, 4, 32>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
                CodegenOptions::WRAPPING,
            )
            .unwrap();
            assert!(checked.len() > wrapping.len());
            assert_ne!(checked.bytes(), wrapping.bytes());
        }
    }
}

#[test]
fn rustc_unchecked_shift_masks_distance_to_the_integer_width() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8, distance: u8) -> u8 { value << distance }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u16, distance: u16) -> u16 { value >> distance }",
    ];
    for source in sources {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert!(
                compile_source_function_with_options::<2048, 512, 4, 2, 4, 32>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                    CodegenOptions::WRAPPING,
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn rustc_boolean_short_circuit_cases_reach_both_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> bool { true || (1 / 0 > 0) }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> bool { false && (1 / 0 > 0) }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { left && !right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, guard: bool) -> bool { guard || (10 / value > 1) }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_shift_static_initializers_reach_both_native_objects() {
    let source = "static SHIFTED: usize = 10_usize << 4_usize; #[unsafe(no_mangle)] pub extern \"C\" fn probe() -> usize { SHIFTED }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_x86_64_abi_stack_arguments_reach_native_objects() {
    let windows = "#[unsafe(no_mangle)] pub extern \"C\" fn fifth(a: u64, b: u64, c: u64, d: u64, value: u64) -> u64 { value + a }";
    assert!(
        compile_source_function::<2048, 512, 4, 8, 4, 32>(
            windows,
            "fifth",
            ObjectFormat::Coff,
            TargetLayout::X86_64,
        )
        .is_ok()
    );

    let system_v = "#[unsafe(no_mangle)] pub extern \"C\" fn seventh(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, value: u64) -> u64 { value + a }";
    assert!(
        compile_source_function::<2048, 512, 4, 8, 4, 32>(
            system_v,
            "seventh",
            ObjectFormat::Elf64,
            TargetLayout::X86_64,
        )
        .is_ok()
    );
}

#[test]
fn rustc_shift_cast_distance_reaches_both_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8, distance: u8) -> u8 { value >> distance as usize }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> usize { -1000isize as usize >> 3usize }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_div_mod_typed_locals_reach_both_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> isize { let x: isize = 15; let y: isize = 4; x / y + x % y }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_parameterized_typed_locals_reach_both_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64) -> u64 { let doubled: u64 = value * 2; let adjusted: u64 = doubled + 1; adjusted + value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, enabled: bool) -> bool { let positive: bool = value > 0; enabled && positive }",
    ];
    for source in sources {
        assert!(
            compile_source_function::<2048, 1024, 4, 4, 4, 64>(
                source,
                "probe",
                ObjectFormat::Elf64,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
        assert!(
            compile_source_function::<2048, 1024, 4, 4, 4, 64>(
                source,
                "probe",
                ObjectFormat::Coff,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_mutable_bitwise_locals_reach_both_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> isize { let mut a: isize = 1; let mut b: isize = 2; a ^= b; b ^= a; a = a ^ b; a | b }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64) -> u64 { let mut x: u64 = value; x ^= 255; x += 1; x }",
    ];
    for source in sources {
        assert!(
            compile_source_function::<2048, 1024, 4, 4, 4, 64>(
                source,
                "probe",
                ObjectFormat::Elf64,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
        assert!(
            compile_source_function::<2048, 1024, 4, 4, 4, 64>(
                source,
                "probe",
                ObjectFormat::Coff,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_increment_decrement_and_inferred_local_cases_reach_native_objects() {
    // Mapped to i8-incr.rs, u8-incr.rs, u8-incr-decr.rs, u32-decr.rs,
    // i32-sub.rs, and the inferred binding form in short-circuit-let.rs.
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> i8 { let mut x: i8 = -12; x = x + 1; x = x - 1; x }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u8 { let mut x: u8 = 12; x = x + 1; x = x - 1; x }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u8 { let mut x: u8 = 19; let mut y: u8 = 35; x = x + 7; y = y - 9; x }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let mut word: u32 = 200000; word = word - 1; word }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> i32 { let mut x: i32 = -400; x = 0 - x; x }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> i32 { let mut x = 0; x += 1; x }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_const_control_flow_if_expressions_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: u32, right: u32) -> u32 { if left < right { right - left } else { left - right } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, guard: bool) -> u64 { if guard { value } else { 10 / value } }",
        "const X: u32 = 4; const Y: u32 = 5; #[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { if X < Y { Y - X } else { X - Y } }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_explicit_tail_return_reaches_native_objects() {
    let source =
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { return value; }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_const_control_flow_conditional_return_reaches_native_objects() {
    // Maps the base-case conditional return shape in consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64, stop: bool) -> u64 { let adjusted: u64 = value + 1; if stop { return adjusted; } adjusted + 1 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_scalar_while_mutation_reaches_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: isize) -> isize { let mut y: isize = 0; while y < limit { y = y + 1; } y }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: isize) -> isize { let mut i: isize = 0; while i < limit { i += 1; } i }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_long_while_loop_local_has_an_executable_replacement() {
    // Original scalar replacement for the per-iteration local declaration in
    // tests/ui/for-loop-while/long-while.rs at the pinned nightly commit. The
    // upstream print-free unused binding is strengthened here by using the
    // scoped value before each backedge.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { let current: u32 = i + 1; current + 10; i = current; if i % 2 == 0 { continue; } total += current; if i == stop { break; } } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1024, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_while_with_break_conditional_block_has_an_executable_replacement() {
    // Original scalar replacement for the local/action/break ordering in
    // tests/ui/for-loop-while/while-with-break.rs at the pinned nightly commit.
    // Allocation, vectors, destructors, printing, and drop elaboration are not
    // claimed by this scalar control-flow slice.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { let current: u32 = i + 1; i = current; if i % 2 == 0 { let skipped: u32 = current; skipped + 10; continue; } if i == stop { let selected: u32 = current; selected + 20; total += selected; break; } total += current; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_loop_guarded_fallthrough_actions_have_an_executable_replacement() {
    // Original scalar replacement for the guarded mutation and subsequent
    // fallthrough order in tests/ui/for-loop-while/loop-no-reinit-needed-post-bot.rs
    // at the pinned nightly commit. Moves, destructors, user-defined calls, and
    // bottom-typed expressions are outside this control-flow slice.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { i += 1; if i % 3 == 0 { let selected: u32 = i; total += selected; } total += 1; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_loop_if_else_actions_have_an_executable_replacement() {
    // Original scalar replacement for the if/else loop ordering in
    // tests/ui/for-loop-while/loop-no-reinit-needed-post-bot.rs at the pinned
    // nightly commit. Moves, destructors, user-defined calls, and bottom-typed
    // expressions remain outside this control-flow slice.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { i += 1; if i % 3 == 0 { let selected: u32 = i; total += selected; } else { let fallback: u32 = 1; total += fallback; } total += 1; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_loop_else_if_chain_has_an_executable_replacement() {
    // Original scalar replacement for ordered alternative loop arms in
    // tests/ui/for-loop-while/loop-no-reinit-needed-post-bot.rs at the pinned
    // nightly commit. Moves, destructors, calls, and bottom types are excluded.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { i += 1; if i % 3 == 0 { let selected: u32 = i; total += selected; } else if i % 2 == 0 { let even: u32 = 2; total += even; } else { let fallback: u32 = 1; total += fallback; } } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_loop_alternative_terminals_have_an_executable_replacement() {
    // Original scalar replacement for conditional break/continue ordering in
    // tests/ui/for-loop-while/loop-break-cont.rs and the alternative continue
    // edge in loop-no-reinit-needed-post-bot.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, mode: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { i += 1; if mode == 0 { let selected: u32 = i; total += selected; break; } else if mode == 1 { let repeated: u32 = 2; total += repeated; continue; } else if mode == 2 { let stopped: u32 = 3; total += stopped; break; } else { let returned: u32 = i + 40; return returned; } } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_nested_unit_loop_has_an_executable_replacement() {
    // Original executable replacement for the inner `loop { break; }` unit
    // behavior in tests/ui/for-loop-while/nested-loop-break-unit.rs and the
    // immediate exit in loop-break-cont-1.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut i: u32 = 0; while i < limit { loop { break; } i += 1; } i }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<1024, 768, 2, 2, 2, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_nested_loop_actions_have_an_executable_replacement() {
    // Original scalar extension of the inner-break scope observed in pinned
    // tests/ui/for-loop-while/nested-loop-break-unit.rs. The inner body uses
    // only MRML's bounded scalar local, expression, and assignment subset.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; while i < limit { loop { let selected: u32 = i + 1; selected + 10; total += selected; break; } i += 1; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_repeated_inner_loop_has_an_executable_replacement() {
    // Original bounded scalar extension of the nested exit targeting in pinned
    // tests/ui/for-loop-while/nested-loop-break-unit.rs. Each outer iteration
    // runs three inner iterations before the inner conditional break.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < limit { loop { let selected: u32 = inner + 1; inner = selected; total += selected; if selected == 3 { break; } } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_inner_loop_continue_has_an_executable_replacement() {
    // Original scalar replacement for the nested continue/break targeting in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs. Even inner values take
    // the inner backedge before the later accumulator and break operations.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < limit { loop { let selected: u32 = inner + 1; inner = selected; if selected % 2 == 0 { continue; } total += selected; if selected == 5 { break; } } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_nested_while_has_an_executable_replacement() {
    // Original bounded scalar replacement for nested while flow in pinned
    // tests/ui/for-loop-while/while.rs and nested-loop-break-unit.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < limit { while inner < 3 { inner += 1; total += inner; } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_nested_while_controls_have_an_executable_replacement() {
    // Original scalar replacement combining nested while, continue, and break
    // flow from pinned tests/ui/for-loop-while/while.rs and loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(outer_limit: u32, inner_limit: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < outer_limit { while inner < inner_limit { let selected: u32 = inner + 1; inner = selected; if selected % 2 == 0 { continue; } total += selected; if selected == 5 { break; } } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_unconditional_break_in_nested_while_has_an_executable_replacement() {
    // Original scalar replacement for the nested unconditional exit targeting
    // in pinned tests/ui/for-loop-while/break.rs and nested-loop-break-unit.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, enter: bool) -> u32 { let mut i: u32 = 0; while i < limit { while enter { break; } i += 1; } i }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<1536, 1024, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_return_from_nested_while_has_an_executable_replacement() {
    // Original scalar replacement for nested return cleanup in pinned
    // tests/ui/for-loop-while/loop-no-reinit-needed-post-bot.rs and the return
    // control family already mapped from consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, enter: bool) -> u32 { let mut i: u32 = 0; while i < limit { while enter { let selected: u32 = i + 40; return selected; } i += 1; } i }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<1536, 1024, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_unit_return_from_nested_while_has_an_executable_replacement() {
    // Original nested-loop replacement for the valueless return shape in
    // pinned tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, enter: bool) { let mut i: u32 = 0; while i < limit { while enter { let selected: u32 = i + 1; selected + 10; return; } i += 1; } }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<1536, 1024, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_conditional_unit_return_from_nested_while_has_an_executable_replacement() {
    // Original bounded scalar replacement for the conditional unit return in
    // pinned tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) { let mut outer: u32 = 0; let mut inner: u32 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == stop { return; } } outer += 1; inner = 0; } }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_conditional_typed_return_from_nested_while_has_an_executable_replacement() {
    // Original scalar replacement for nested cleanup and guarded return in
    // pinned tests/ui/liveness/loop-no-reinit-needed-post-bot.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == stop { return outer + 40; } } outer += 1; inner = 0; } outer }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_ordered_conditional_returns_in_nested_while_have_an_executable_replacement() {
    // Original bounded replacement for ordered guarded return edges in pinned
    // tests/ui/liveness/loop-no-reinit-needed-post-bot.rs and
    // tests/ui/consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, first: u32, second: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == first { return outer + 40; } if inner == second { return outer + 50; } } outer += 1; inner = 0; } outer }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_nested_continue_precedes_later_return() {
    // Original scalar replacement for ordered continue/return control in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs and nested cleanup in
    // tests/ui/liveness/loop-no-reinit-needed-post-bot.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, skip: u32, stop: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == skip { continue; } if inner == stop { return outer + 40; } } outer += 1; inner = 0; } outer }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_multiple_ordered_nested_continues_have_an_executable_replacement() {
    // Original scalar replacement for repeated ordered continue edges in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, first: u32, second: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < limit { while inner < 5 { inner += 1; if inner == first { continue; } if inner == second { continue; } total += inner; } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_positioned_nested_breaks_have_an_executable_replacement() {
    // Original scalar replacement for ordered conditional break edges in
    // pinned tests/ui/for-loop-while/break.rs and loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, first: u32, second: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; let mut total: u32 = 0; while outer < limit { while inner < 5 { inner += 1; if inner == first { break; } total += inner; if inner == second { break; } } outer += 1; inner = 0; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_unconditional_nested_continue_has_an_executable_replacement() {
    // Original scalar replacement for the unconditional inner continue edge in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut outer: u32 = 0; let mut inner: u32 = 0; while outer < limit { while inner < 3 { let selected: u32 = inner + 1; inner = selected; continue; let unreachable: u32 = 10 / 0; unreachable; return unreachable; } outer += 1; inner = 0; } outer }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_inner_loop_with_conditional_only_exits_has_an_executable_replacement() {
    // Original scalar replacement for conditional backedges and return exits in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool, stop: u32) -> u32 { let mut inner: u32 = 0; while enter { loop { inner += 1; if inner < stop { continue; } if inner == stop { return inner; } } } 0 }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 1536, 4, 4, 4, 64>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_diverging_empty_inner_loop_has_an_executable_replacement() {
    // Original scalar replacement for the diverging empty loop admitted by
    // pinned tests/ui/for-loop-while/loop-break-cont.rs.
    for source in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u32 { while enter { loop {} } 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u32 { while enter { loop { 1; } } 42 }",
    ] {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert!(
                compile_source_function::<1536, 1024, 4, 4, 4, 48>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn rustc_empty_while_and_top_level_diverging_loops_have_executable_replacements() {
    // Original scalar replacements for empty and diverging loop bodies in
    // pinned tests/ui/for-loop-while/loop-break-cont.rs.
    for source in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u32 { while enter {} 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { loop {} }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { loop { 1; } }",
    ] {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert!(
                compile_source_function::<1536, 1024, 4, 4, 4, 48>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn rustc_labeled_statement_loop_controls_have_an_executable_replacement() {
    // Original scalar replacement for same-loop labels in pinned
    // tests/ui/for-loop-while/loop-break-value.rs and loop-break-cont.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) -> u32 { let mut i: u32 = 0; let mut total: u32 = 0; 'count: while i < limit { i += 1; if i % 2 == 0 { continue 'count; } else if i == stop { break 'count; } total += i; } 'once: loop { break 'once; } total }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<3072, 2048, 4, 4, 4, 96>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_cross_nested_labeled_controls_have_an_executable_replacement() {
    // Original scalar replacement for nested label targeting in pinned
    // tests/ui/for-loop-while/loop-break-value.rs and loop-break-cont.rs.
    for source in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32, stop: u32) -> u32 { let mut outer: u32 = 0; let mut total: u32 = 0; 'outer: while outer < limit { outer += 1; 'inner: loop { let selected: u32 = outer; if selected == 0 { continue 'inner; } if selected % 2 == 0 { continue 'outer; } if selected == stop { break 'outer; } break 'inner; } total += outer; } total }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut outer: u32 = 0; 'outer: while outer < limit { outer += 1; 'inner: loop { continue 'outer; } } outer }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u32 { let mut outer: u32 = 0; 'outer: while enter { outer += 1; 'inner: loop { break 'outer; } } outer }",
    ] {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert!(
                compile_source_function::<4096, 3072, 4, 4, 4, 128>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn rustc_const_function_declarations_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(value: usize) -> usize { return value; }",
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(left: usize, right: usize) -> usize { left + right }",
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(value: u8) -> u8 { value + 1 }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_integer_const_function_calls_feed_native_objects() {
    let sources = [
        "const fn add(left: usize, right: usize) -> usize { left + right } const SUM: usize = add(40, 2); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { value + SUM }",
        "const fn difference(left: u32, right: u32) -> u32 { left - right } const DELTA: u32 = difference(50, 8); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + DELTA }",
        "const fn combine(a: u8, b: u8) -> u8 { a + b } const TOTAL: u8 = combine(30, 12); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + TOTAL }",
        "const fn adjust(value: i32, offset: i32) -> i32 { value + offset } const TOTAL: i32 = adjust(-8, 50); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + TOTAL }",
        "const fn minimum(value: i8) -> i8 { value } const MINIMUM: i8 = minimum(-128); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { value | MINIMUM }",
        "const fn sub(left: u32, right: u32) -> u32 { left - right } const RESULT: u32 = sub(sub(88, 44), 22); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + RESULT }",
        "const fn add(left: u8, right: u8) -> u8 { left + right } const RESULT: u8 = if true { add(20, 1) * 2 } else { add(255, 1) }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + RESULT }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_boolean_const_function_calls_feed_native_objects() {
    let sources = [
        "const fn invert(value: bool) -> bool { !value } const FLAG: bool = invert(false); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value && FLAG }",
        "const fn both(left: bool, right: bool) -> bool { left && right } const FLAG: bool = both(true, true); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value && FLAG }",
        "const fn positive(value: i32) -> bool { value > 0 } const FLAG: bool = positive(42); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value && FLAG }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
    let choose = "const fn choose(condition: bool, left: u8, right: u8) -> u8 { if condition { left } else { right } } const VALUE: u8 = choose(true, 42, 0); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + VALUE }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 512, 4, 4, 4, 32>(
                choose,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_recursive_integer_const_function_call_feeds_native_objects() {
    let source = "const fn gcd(left: u32, right: u32) -> u32 { if right == 0 { left } else { gcd(right, left % right) } } const VALUE: u32 = gcd(48, 18); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_statement_bodied_const_function_call_feeds_native_objects() {
    let sources = [
        "const fn identity(value: usize) -> usize { return value; } const VALUE: usize = identity(42); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { value + VALUE }",
        "const fn adjust(value: i32) -> i32 { let offset: i32 = 2; if value < 0 { return -value + offset; } value + offset } const VALUE: i32 = adjust(-40); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + VALUE }",
        "const fn enabled(value: bool) -> bool { let inverted: bool = !value; if inverted { return true; } false } const FLAG: bool = enabled(false); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value && FLAG }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_mutable_scalar_const_function_locals_feed_native_objects() {
    let sources = [
        "const fn mutate(value: u32) -> u32 { let mut result: u32 = value; result += 2; result *= 2; result -= 42; result } const VALUE: u32 = mutate(40); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn toggle(value: bool) -> bool { let mut result: bool = value; result ^= true; result } const FLAG: bool = toggle(false); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value && FLAG }",
        "const fn swap() -> u8 { let mut left: u8 = 1; let mut right: u8 = 2; left ^= right; right ^= left; left ^= right; left } const VALUE: u8 = swap(); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + VALUE }",
    ];
    for source in sources {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert!(
                compile_source_function::<2048, 512, 4, 4, 4, 32>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                )
                .is_ok()
            );
        }
    }
}

#[test]
fn rustc_const_function_statement_loops_feed_native_objects() {
    let sources = [
        "const fn count(limit: u32) -> u32 { let mut index: u32 = 0; while index < limit { index += 1; } index } const VALUE: u32 = count(35); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn even_sum(limit: u32) -> u32 { let mut index: u32 = 0; let mut total: u32 = 0; while index < limit { index += 1; if index % 2 != 0 { continue; } total += index; if total >= 42 { break; } } total } const VALUE: u32 = even_sum(20); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn until() -> u32 { let mut value: u32 = 0; loop { value += 1; if value == 42 { break; } } value } const VALUE: u32 = until(); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn prime(value: u32) -> bool { let mut divisor: u32 = 3; if value % 2 == 0 { return false; } loop { if value % divisor == 0 { return false; } if divisor * divisor > value { return true; } divisor += 2; } false } const VALUE: u32 = if prime(113) { 42 } else { 0 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn immediate() -> u32 { loop { return 42; } } const VALUE: u32 = immediate(); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn staged(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } value *= 2; value } const VALUE: u32 = staged(21); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
        "const fn classify(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } if value == 42 { return 7; } value += 1; value } const VALUE: u32 = classify(42); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + VALUE }",
    ];
    for source in sources {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            let result = compile_source_function::<2048, 512, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            );
            assert!(result.is_ok(), "{source}: {result:?}");
        }
    }
}

#[test]
fn runtime_interleaved_statement_order_reaches_native_objects() {
    for source in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } value *= 2; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } if value == 42 { return 7; } value += 1; value }",
    ] {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_sequential_while_loops_reach_native_objects() {
    // Original scalar probe for the two consecutive while-loop shape exercised
    // by tests/ui/for-loop-while/while.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: u32, second: u32) -> u32 { let mut value: u32 = 0; while value < first { value += 1; } while value < second { value += 1; } value }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_post_loop_local_binding_reaches_native_objects() {
    // Original scalar replacement for the loop-then-binding statement order
    // present in tests/ui/for-loop-while/break.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } let offset: u32 = value + 2; offset * 2 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_guarded_local_binding_reaches_native_objects() {
    // Original scalar replacement for the early-return-then-binding order in
    // tests/ui/liveness/var-defined-after-early-return.rs and the scalar const
    // shape in tests/ui/consts/control-flow/basics.rs at the pinned commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, stop: bool) -> u32 { if stop { return 7; } let adjusted: u32 = value + 1; adjusted * 2 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_drop_on_return_scoped_local_replacement() {
    // Original scalar replacement for the branch-local-before-return order in
    // tests/ui/drop/drop-on-ret.rs at the pinned nightly commit. Strings,
    // allocation, destructors, and drop elaboration are outside this slice.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, stop: bool) -> u32 { if stop { let selected: u32 = value + 1; selected + 10; return selected; } value * 2 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_unconditional_early_return_reaches_native_objects() {
    // Original scalar replacement for the return-before-later-declarations
    // control-flow boundary in tests/ui/liveness/var-defined-after-early-return.rs
    // at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { let adjusted: u32 = value + 1; return adjusted; let unreachable: u32 = value + 2; unreachable }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_valueless_loop_return_reaches_native_objects() {
    // Original scalar replacement for the terminating loop branch in
    // tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs at the
    // pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(stop: bool) { loop { if stop { return; } } }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_exhaustive_conditional_returns_reach_native_objects() {
    // Original scalar replacement for the exhaustive else-if return chain in
    // tests/ui/expr/if/if-check.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { if value == 0 { return 42; } else if value == 1 { return value + 41; } else if value == 2 { return 84 / value; } else { return 126 / value; } }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_non_exhaustive_else_if_returns_reach_native_objects() {
    // Original scalar replacement for the returning else-if chain with
    // fallthrough in tests/ui/expr/weird-exprs.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { if value == 0 { return 42; } else if value == 1 { return 42 / value; } 84 / value }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_conditional_mutation_reaches_native_objects() {
    // Original scalar replacement for the guarded assignment in
    // tests/ui/consts/const-eval/issue-64970.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, select: bool) -> u32 { let mut result = value; if select { result = 42; } else { result = 1 / value; } if !select { result += 40; } result }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 768, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_guarded_mutation_reaches_native_objects() {
    // Original scalar replacement for the guard-before-mutation ordering in
    // tests/ui/consts/control-flow/basics.rs at the pinned nightly commit.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, stop: bool) -> u32 { let mut adjusted: u32 = value; if stop { return 7; } adjusted += 1; adjusted * 2 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_guarded_fibonacci_loop_reaches_native_objects() {
    // Original scalar replacement for the base-case and iterative Fibonacci
    // control flow in tests/ui/consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(n: u32) -> u32 { if n == 0 { return 0; } let mut previous: u32 = 0; let mut current: u32 = 1; let mut index: u32 = 1; while index < n { current += previous; previous = current - previous; index += 1; } current }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 512, 4, 4, 4, 32>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_alternating_guards_and_mutations_reach_native_objects() {
    // Original scalar replacement for repeated early-return decisions and
    // state changes represented in tests/ui/consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, first: bool, second: bool) -> u32 { let mut result: u32 = value; if first { return 1; } result += 1; if second { return 2; } result *= 2; result }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 512, 4, 4, 4, 32>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_post_loop_alternating_guards_reach_native_objects() {
    // Original scalar replacement combining the loop and repeated return
    // decisions represented in tests/ui/consts/control-flow/basics.rs.
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, limit: u32, first: bool, second: bool) -> u32 { let mut index: u32 = 0; let mut result: u32 = value; while index < limit { index += 1; } if first && index == limit { return 1; } result += 1; if second && index == limit { return 2; } result *= 2; result }";
    for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
        assert!(
            compile_source_function::<2048, 768, 4, 4, 4, 48>(
                source,
                "probe",
                format,
                TargetLayout::X86_64,
            )
            .is_ok()
        );
    }
}

#[test]
fn rustc_scalar_block_expressions_reach_native_objects() {
    // Original scalar replacements for the integral, unit, condition, and call
    // operand blocks exercised by tests/ui/consts/const-block.rs and
    // tests/ui/lint/unused_braces.rs at the pinned nightly commit.
    let probes = [
        "const BLOCK: u32 = { 40 + 2 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { { value + BLOCK } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, guard: bool) -> u32 { if { guard } { { value } } else { { 84 / value } } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { {} }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + { let mut offset: u32 = 1; offset += 1; offset } }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_scalar_constant_patterns_reach_native_objects() {
    // A bounded executable replacement for the scalar constant-pattern behavior
    // in tests/ui/consts/consts-in-patterns.rs at the pinned nightly commit.
    let probes = [
        "const SELECTED: u32 = 7; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { SELECTED => { let mut answer: u32 = 40; answer += 2; answer }, _ => 84 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { -7i8 => 42i8, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { true => 42, _ => 0 } }",
        "const ZERO: u32 = 0; const SEVEN: u32 = 7; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { ZERO => 42, SEVEN => 84 / (value - 5), 9 => { let answer = 42; answer } _ => 420 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { match value { 1usize..=5usize => 42usize, _ => 0usize } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { match value { 1usize..5usize => 42usize, _ => 0usize } }",
        "const START: isize = 1; const END: isize = 42; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: isize) -> isize { match value { START..=END => 42isize, _ => 0isize } }",
        "const MINIMUM: i8 = -5; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { MINIMUM..=-1i8 => 42i8, _ => 0i8 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 0 | 1..=10 => 42, 12 | 13 => 84 / (value - 10), _ => 420 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { -7..0 | 1 => 42, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { -5.. => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { ..-7 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { ..=-7 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { ..-7 | -5.. => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 1 if false => 1 / 0, 1..=2 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, guard: bool) -> u32 { match value { 0 | 1..=10 if guard => 42, _ => 84 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 1 if 1 / (value - 2) > 0 => 0, 2 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { 'a'..='z' => '*', _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { '\\t' | '\\n' => '*', _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { '\\u{3b1}'..='\\u{3c9}' => '*', _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { bound if bound < 7 => 84 / bound, selected if selected < 11 => selected + 32, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> bool { match value { bound if bound == '\\u{3bb}' => true, _ => false } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, choose_first: bool) -> u32 { match value { _ if choose_first => 84 / value, 7 => 42, _ if value == 0 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { _ if value != 0 && 84 / value == 42 => 42, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { number @ 1..=100 if number == 50 => number - 8, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ 42 => selected, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { symbol @ 'a'..='z' => symbol, _ => '*' } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ _ if selected == 42 => selected, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ 1 | selected @ 2 => selected + 40, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ 1..=3 | selected @ 7..=9 if selected == 7 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 42..=42 => 42, 41..42 => 41, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { 'm'..='m' => '*', _ => value } }",
        "const START: u32 = 42; const END: u32 = 42; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { START..=END => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { match value { 0..=255 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { -128..=127 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { match value { 0..=18446744073709551615 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { (1 | 2) => value + 40, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { ((1 | 2) | (3..=4)) => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, accept: bool) -> u32 { match value { (| 1 | (5..=10)) if accept => 42, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ (1 | 2) => selected + 40, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, accept: bool) -> u32 { match value { selected @ ((1..=3) | (7 | 9)) if accept => selected + 35, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected @ (| 1 | _) if selected == 42 => selected, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { true => 42, false => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { false => 0, true => 42, } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { selected => selected } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { selected @ (false | true) => if selected { 42 } else { 0 } } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { ; value + 1; let mut result: u32 = value; if false { 1 / 0 } else { 0 }; result += 2; result }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == true; !value; value }",
        "const fn adjust(value: u32) -> u32 { ; value + 1; let mut result: u32 = value; result += 2; result } const ANSWER: u32 = adjust(40); #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { ANSWER + value }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
    for source in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 1 => 1 / 0, 2 => 1 / 0, 3 => 1 / 0, 4 => 1 / 0, 5 => 1 / 0, 6 => 1 / 0, 7 => 1 / 0, 8 => 42, _ => value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, accept: bool) -> u32 { match value { 1 if 1 / (value - 8) > 0 => 0, 2 => 1 / 0, 3 => 1 / 0, 4 => 1 / 0, 5 => 1 / 0, 6 => 1 / 0, 7 => 1 / 0, 8 if accept => 42, _ => value } }",
    ] {
        for format in [ObjectFormat::Elf64, ObjectFormat::Coff] {
            assert_eq!(
                compile_source_function::<4096, 1024, 4, 2, 4, 64>(
                    source,
                    "probe",
                    format,
                    TargetLayout::X86_64,
                )
                .map(|_| ())
                .map_err(|error| error.kind),
                Ok(())
            );
        }
    }

    let non_boolean_guard = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 1 if 2 => 42, _ => 0 } }";
    assert_eq!(
        compile(non_boolean_guard, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeTypeMismatch
        ))
    );
    let mixed_character_integer = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 'a' => 42, _ => 0 } }";
    assert_eq!(
        compile(mixed_character_integer, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeTypeMismatch
        ))
    );
    let inconsistent_binding = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { bound | 1 => 42, _ => 0 } }";
    assert_eq!(
        compile(inconsistent_binding, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
            ExpressionErrorKind::InconsistentMatchBindings,
        )))
    );
    let mismatched_bindings = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { left @ 1 | right @ 2 => 42, _ => 0 } }";
    assert_eq!(
        compile(mismatched_bindings, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
            ExpressionErrorKind::InconsistentMatchBindings,
        )))
    );
    let nested_mismatched_bindings = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { (left @ 1 | (right @ 2)) => 42, _ => 0 } }";
    assert_eq!(
        compile(nested_mismatched_bindings, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
            ExpressionErrorKind::InconsistentMatchBindings,
        )))
    );
    let non_boolean_wildcard_guard = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { _ if value => 42, _ => 0 } }";
    assert_eq!(
        compile(non_boolean_wildcard_guard, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeTypeMismatch
        ))
    );
    for non_exhaustive in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { true => 42 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { true => 42, false if value => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 0 => 42, 1 => 0 } }",
    ] {
        assert_eq!(
            compile(non_exhaustive, ObjectFormat::Elf64),
            Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
                ExpressionErrorKind::NonExhaustiveMatch,
            )))
        );
    }
    for invalid_range in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { 6..=1 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { 0..0 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: char) -> char { match value { 'z'..='a' => '*', _ => value } }",
    ] {
        assert_eq!(
            compile(invalid_range, ObjectFormat::Elf64),
            Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
                ExpressionErrorKind::InvalidRangeBounds,
            )))
        );
    }
    let boolean_range = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { false..=true => 42, _ => 0 } }";
    assert_eq!(
        compile(boolean_range, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Expression(
            ExpressionErrorKind::InvalidRangeType,
        )))
    );
    for invalid_named_range in [
        "const START: u32 = 6; const END: u32 = 1; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { START..=END => 42, _ => 0 } }",
        "const START: i32 = -1; const END: i32 = -5; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { match value { START..=END => 42, _ => 0 } }",
        "const BOUND: u32 = 7; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { match value { BOUND..BOUND => 42, _ => 0 } }",
    ] {
        assert_eq!(
            compile(invalid_named_range, ObjectFormat::Elf64),
            Err(CompileErrorKind::Codegen(
                CodegenErrorKind::InvalidRangeBounds
            ))
        );
    }
    let named_boolean_range = "const LOW: bool = false; const HIGH: bool = true; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> u32 { match value { LOW..=HIGH => 42, _ => 0 } }";
    assert_eq!(
        compile(named_boolean_range, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::InvalidRangeType
        ))
    );
    for overflowing_endpoint in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { match value { 1..257 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { match value { 1..=256 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { 0..129 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { -129..0 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { match value { -10000.. => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u64) -> u64 { match value { 0..=99999999999999999999 => 42, _ => 0 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { match value { 0..=18446744073709551616 => 42, _ => 0 } }",
    ] {
        assert_eq!(
            compile(overflowing_endpoint, ObjectFormat::Elf64),
            Err(CompileErrorKind::Codegen(
                CodegenErrorKind::RangeEndpointOutOfRange
            ))
        );
    }
}

#[test]
fn rustc_closed_scalar_inline_const_blocks_reach_native_objects() {
    // Original runtime and declaration probes for the pinned nightly's inline
    // const-block syntax. Module constants resolve inside the distinct inline
    // const boundary, while runtime parameters and locals are rejected.
    let probes = [
        "const fn offset() -> u32 { 2 } const OFFSET: u32 = const { offset() }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, guard: bool) -> u32 { if guard { value + const { OFFSET } } else { 84 / value + OFFSET - 2 } }",
        "const fn enabled() -> bool { true } const FLAG: bool = const { enabled() }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == const { true == true } && FLAG }",
        "const fn offset() -> u32 { 2 } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { offset() } }",
        "const fn enabled() -> bool { true } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == const { enabled() } }",
        "const fn answer() -> u32 { 42 } #[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { const { answer() } }",
        "const fn add(left: u32, right: u32) -> u32 { left + right } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { add(1, 1) } }",
        "const fn less(left: u32, right: u32) -> bool { left < right } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == const { less(1, 2) } }",
        "const fn adjust(value: i32, offset: i32) -> i32 { value + offset } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + const { adjust(-8i32, 50i32) } }",
        "const OFFSET: u32 = const { let first = 1; let second = first + 1; second }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { let base = 6; base / 3 } + OFFSET - 2 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> i32 { const { let x = 5 + 10; x / 3 } }",
        "const OFFSET: u32 = const { let mut value = 1; value += 1; value }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { let mut base = 3; base *= 2; base /= 3; base } + OFFSET - 2 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == const { let mut flag = false; flag |= true; flag } }",
        "const OFFSET: u8 = const { let value: u8 = 2; value }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u8) -> u8 { value + const { let mut base: u8 = 40; base += OFFSET; base } - 40 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { value == const { let flag: bool = 1 < 2; flag } }",
        "const OFFSET: u32 = const { let mut value = 1; value += 1; let scale = 10; value *= scale; let adjustment = 22; value += adjustment; value }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { let mut base: u32 = 1; base += 1; let factor: u32 = 21; base *= factor; base } + OFFSET - 42 }",
        "const OFFSET: u32 = const { 20 + 22; 2 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { if false { 1 / 0 } else { 20 + 22 }; let factor = 1; 2 * factor } + OFFSET - 2 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { const { let mut value = 40; value += 2; value; } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { if false { 1 / 0; }; 2 } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, guard: bool) -> u32 { if guard { value + const { if true { let mut selected: u32 = 40; selected += 2; selected } else { 1 / 0 } } - 40 } else { 84 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32, guard: bool) -> u32 { if guard { value + const { if false { 1 / 0 } else if true { let selected = 2; selected } else { 1 / 0 } } } else { 84 / value } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { const {} }",
    ];
    for source in probes {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }

    let parameter_capture =
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { const { value } }";
    assert_eq!(
        compile(parameter_capture, ObjectFormat::Coff),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeExpressionUnsupported
        ))
    );
    let local_capture = "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let value: u32 = 42; const { value } }";
    assert_eq!(
        compile(local_capture, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeExpressionUnsupported
        ))
    );
    let non_const_call = "fn offset() -> u32 { 2 } #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { offset() } }";
    assert_eq!(
        compile(non_const_call, ObjectFormat::Coff),
        Err(CompileErrorKind::Codegen(
            CodegenErrorKind::RuntimeExpressionUnsupported
        ))
    );
    let invalid_inline_const = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> u32 { value + const { 1 / 0 } }";
    assert_eq!(
        compile(invalid_inline_const, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Execution(
            ExecutionError::Arithmetic(ConstEvalError::DivisionByZero)
        )))
    );
    let overflowing_ascription = "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u8 { const { let value: u8 = 256; value } }";
    assert_eq!(
        compile(overflowing_ascription, ObjectFormat::Coff),
        Err(CompileErrorKind::Codegen(CodegenErrorKind::Execution(
            ExecutionError::Arithmetic(ConstEvalError::Overflow)
        )))
    );
    for cross_boundary_control in [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { const { return } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { const { break } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { const { continue } }",
    ] {
        assert_eq!(
            compile(cross_boundary_control, ObjectFormat::Coff),
            Err(CompileErrorKind::Codegen(
                CodegenErrorKind::RuntimeExpressionUnsupported
            ))
        );
    }
}

#[test]
fn runtime_loop_return_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(value: u32) -> bool { let mut divisor: u32 = 3; loop { if value % divisor == 0 { return false; } divisor += 2; } false }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_conditional_while_break_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: isize) -> isize { let mut i: isize = 0; while i < limit { i += 1; if i == 10 { break; } } i }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_explicit_continue_and_unconditional_loop_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: isize) -> isize { let mut i: isize = limit; while i > 0 { i -= 1; continue; } i }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> usize { let value: usize = 42; loop { break; } value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> usize { let mut i: usize = 0; loop { i += 1; if i == 10 { break; } } i }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_interleaved_loop_control_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: usize) -> usize { let mut i: usize = 0; let mut is_even: bool = false; loop { if i == limit { break; } is_even = false; i += 1; if i % 2 != 0 { continue; } is_even = true; } if is_even { i } else { 0 } }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_immediate_break_loop_values_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> isize { let value: isize = loop { break 13; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { let value: bool = loop { break input; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: u32) -> u32 { let value: u32 = loop { break input + 1; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: u32) -> u32 { let value: u32 = 'value: loop { break 'value input + 1; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool) -> u32 { let value: () = 'value: loop { if first { break 'value; } else { break 'value (); } }; value; 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: u32) -> u32 { let value: u32 = 'outer: loop { break 'outer 'inner: loop { break 'inner input + 1; }; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(exit_outer: bool) -> u32 { let value: u32 = 'outer: loop { 'inner: loop { if exit_outer { break 'outer 42; } else { break 'inner false; } }; break 'outer 99; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let first: u32 = 'outer1: loop { 'inner: while break 'inner { 1 / 0; } break 'outer1 123; }; let second: u32 = 'outer2: loop { while break 'outer2 567 { 1 / 0; } }; first + second }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let first: () = loop { break (); break; }; first; let second: () = loop { break; break (); }; second; 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let first: () = loop { if true { break; } else { break break Default::default(); } }; first; let second: () = loop { if true { break Default::default(); } else { break; } }; second; 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let third: () = loop { break if true { Default::default() } else { break; }; }; third; 42 }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { let integer: u32 = Default::default(); let boolean: bool = Default::default(); integer; if boolean { 1 / 0 } else { integer + 42 } }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()), "{source}");
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()), "{source}");
    }
}

#[test]
fn rustc_loop_break_value_array_literals_have_a_scalar_index_replacement() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { [1u32, 3, 5][1] }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u32 { [17u32][0] }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_loop_break_value_contextual_array_defaults_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u64 { (loop { break if true { break Default::default() } else { break [13u64, 14] }; })[0] }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u64 { (loop { if false { break [1 / 0, 14] } else { break Default::default() } })[1] }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u64 { (loop { break if false { break Default::default() } else { [42u64, 43] }; })[1] }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_loop_break_value_fixed_array_locals_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> u64 { let values: [u64; 3] = [13, 42, 99]; values[1] }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(select: bool) -> u64 { let values: [u64; 2] = loop { break if select { break [13, 14] } else { break Default::default() }; }; let after: u64 = 14; values[0] + values[1] + after }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_loop_break_value_runtime_array_indexes_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(index: usize) -> usize { [13usize, 42, 99][index] }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(index: usize, divisor: usize) -> usize { let values = [84 / divisor, 42, 99]; values[index] }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_loop_break_value_mutable_array_assignment_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(index: usize) -> usize { let mut values = [13usize, 42, 99]; let after = 1usize; values = [values[2], values[0], values[1]]; values[index] + after }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_loop_break_value_mutable_array_element_assignment_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(index: usize, value: usize) -> usize { let mut values = [13usize, 42, 99]; let after = 1usize; values[index] += value; values[index] + after }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_fixed_array_repeat_literals_reach_native_objects() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: usize) -> usize { let mut values = [input + 1; 3usize]; values[1] += 2; values[0] + values[1] + values[2] }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_zero_length_array_repeat_still_evaluates_its_operand() {
    let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(input: usize, divisor: usize) -> usize { let values: [usize; 0] = [input / divisor; 0]; 42 }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_const_let_eq_one_word_array_parameter_reaches_native_objects() {
    let source = "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(values: [usize; 1], after: usize) -> usize { let copied = values; copied[0] + after }";
    assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
    assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
}

#[test]
fn rustc_copy_out_of_array_multiword_parameter_reaches_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(before: usize, values: [usize; 2], after: usize) -> usize { before + values[0] * 10 + values[1] + after }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(before: usize, values: [usize; 3], after: usize) -> usize { before + values[0] + values[1] + values[2] + after }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(a: usize, b: usize, c: usize, d: usize, e: usize, values: [usize; 2], after: usize) -> usize { a + b * 2 + c * 3 + d * 4 + e * 5 + values[0] * 6 + values[1] * 7 + after * 8 }",
    ];
    for source in sources {
        assert_eq!(compile_wide(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile_wide(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_competing_break_loop_values_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool, input: isize) -> isize { let value: isize = loop { if first { break input + 1; } break 84 / input; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool) -> bool { let value: bool = loop { if first { break true; } break false; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool, input: isize) -> isize { let value: isize = 'value: loop { if first { break 'value input + 1; } break 'value 84 / input; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool, input: isize) -> isize { let value: isize = 'value: loop { if first { break 'value input + 1; } if input == 0 { break 'value 42; } if input == 1 { break 'value 43; } break 'value 84 / input; }; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(first: bool, input: isize) -> isize { let value: isize = 'value: loop { if first { break 'value input + 1; } else if input == 0 { break 'value 42; } else if input == 1 { break 'value 43; } else { break 'value 84 / input; } }; value }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_unit_loop_bodies_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { () }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() { loop { break; } }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> () { let value: () = loop { break (); }; value }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_implicit_unit_tails_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() {}",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() { let value: usize = 1; }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() { loop { break; } }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_boolean_and_unit_equality_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { left == right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { left != right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> bool { () == () }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_boolean_ordering_and_bitwise_ops_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { left < right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { left >= right }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe() -> bool { () <= () }",
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(left: bool, right: bool) -> bool { left & right }",
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(left: bool, right: bool) -> bool { left | right }",
        "#[unsafe(no_mangle)] pub const extern \"C\" fn probe(left: bool, right: bool) -> bool { left ^ right }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_boolean_constants_and_integer_const_comparisons_reach_native_objects() {
    let sources = [
        "const FLAG: bool = true && false; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { FLAG || input }",
        "static ORDERED: bool = -1 < 2; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { ORDERED && input }",
        "const SAME: bool = 3 == 3; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { SAME ^ input }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }

    let mismatch = "const FLAG: bool = true; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: u8) -> u8 { input + FLAG }";
    assert_eq!(
        compile(mismatch, ObjectFormat::Elf64),
        Err(CompileErrorKind::Codegen(
            mrml_rustc::CodegenErrorKind::RuntimeTypeMismatch
        ))
    );
}

#[test]
fn rustc_integer_constant_casts_and_signed_const_if_reach_native_objects() {
    let sources = [
        "const OFFSET: i32 = if -1 < 2 { -1i8 as i32 } else { 1 / 0 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i32) -> i32 { value + OFFSET }",
        "const MASK: i8 = 255u16 as i8; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: i8) -> i8 { value & MASK }",
        "const ENABLED: bool = if false { 1 == 2 } else { -3 < 1 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: bool) -> bool { ENABLED && value }",
        "const SIZE: usize = 23 as usize; #[unsafe(no_mangle)] pub extern \"C\" fn probe(value: usize) -> usize { value + SIZE }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_compound_constant_comparisons_reach_native_objects() {
    let sources = [
        "const SUM: i32 = 2 + 2; const PASS: bool = SUM == 4; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { PASS && input }",
        "const BASE: i32 = -4; const PASS: bool = BASE + 3 == -1; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { PASS && input }",
        "const LIMIT: u8 = 9; const PASS: bool = (LIMIT - 1) * 2 > 15; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { PASS && input }",
        "const BASE: i16 = -4; const PASS: bool = if BASE < 0 { BASE * 2 <= -8 } else { 1 / 0 == 0 }; #[unsafe(no_mangle)] pub extern \"C\" fn probe(input: bool) -> bool { PASS && input }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}

#[test]
fn rustc_boolean_compound_assignments_reach_native_objects() {
    let sources = [
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { let mut value: bool = left; value &= right; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { let mut value: bool = left; value |= right; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(left: bool, right: bool) -> bool { let mut value: bool = left; value ^= right; value }",
        "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u64) -> bool { let mut i: u64 = 0; let mut value: bool = false; while i < limit { value ^= true; i += 1; } value }",
    ];
    for source in sources {
        assert_eq!(compile(source, ObjectFormat::Elf64), Ok(()));
        assert_eq!(compile(source, ObjectFormat::Coff), Ok(()));
    }
}
