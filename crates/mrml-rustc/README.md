# MRML Rust compiler

`mrml-rustc` is an original, dependency-free Rust compiler implementation for
MRML. It is being bootstrapped against the observable behavior of the pinned
Rust nightly while remaining suitable for MRML's `no_std`, no-global-`alloc`
environment.

The initial reference compiler is `rustc 1.100.0-nightly`, commit
`f7d782a3be46d6bb4b9792fe69a61db389ba1769` (2026-08-19). The rustup `rust-src`
component does not contain the rustc compiler suites, so the matching official
`rust-lang/rust` commit is the test source of truth. Consequently:

- `tests/conformance.rs` records original black-box language observations used
  to bootstrap and later regression-test the MRML compiler.
- `tests/rustc_nightly_replacements.rs` maps original executable MRML probes to
  exact files in the pinned rustc UI and LLVM-codegen suites. The first slice
  covers unsigned arithmetic, division/remainder, shifts, platform-width bit
  operations, overflow checks, and x86-64 ABI boundaries.
- Unit tests beside compiler modules test bounded internal behavior.
- No upstream Rust source or test text is copied into this crate.
- The current upstream nightly is only the stage-zero oracle. Each test must
  eventually run through an MRML-owned parser, type checker, lowering pipeline,
  code generator, and test runner.

The implemented bootstrap grammar currently recognizes module-level `fn`, `const fn`,
typed `const`, and immutable typed `static` items, optional `pub`, named typed parameters, optional return
types, nested generic and tuple/array type syntax, balanced function bodies, and
bounded constant initializers. Function bodies additionally parse a bounded
prefix of typed or inferred `let` bindings, including `let mut`, followed by bounded simple
or compound assignment statements, up to eight conditional-action
statements, conditional-return statements, up to eight
standalone value-return statements, up to eight scalar expression statements,
and one tail expression. Empty `;` statements
occupy the same bounded stream as explicit no-ops. Expression results are
discarded only after their lazy or checked evaluation completes, and statements
may interleave with locals and assignments before or after a loop sequence. The
AST borrows source text and uses
caller-selected fixed item and parameter capacities; nesting is capped at 64
delimiters. Capacity and malformed-input failures carry bounded source spans.

Conditional action blocks retain up to four source-ordered local declarations,
assignments, discarded scalar expressions, or value returns in each selected
`if`, `else if`, or `else` arm. Branch locals may shadow enclosing locals, are
visible to later actions in the same arm, and are removed at the arm's native
fallthrough edge and from the const evaluator before subsequent statements.
They cannot leak into another arm or the enclosing function body. Assignments
reuse the ordinary mutability, width, type, and operator checks. Conditions must
be Boolean and only the selected arm is evaluated. The fixed conditional-action
arena holds eight records independently of the top-level local-binding capacity.
Exhausting the shared runtime-local arena fails with `TooManyRuntimeLocals`
before indexing or emitting an object.

Standalone `return value;` statements retain their position among declarations,
mutations, loops, and discarded expressions. Runtime emission checks the value
against the declared return type and restores the complete native stack frame
before returning. Later unreachable source remains parsed and type-checked.
Bounded const-function evaluation terminates immediately with the returned
integer or Boolean value. The original replacement is mapped to pinned nightly
`tests/ui/liveness/var-defined-after-early-return.rs`; no upstream test text is
copied.
Valueless `return;` uses the same fixed record with an explicit unit value. It
works as a top-level statement and as a conditional or unconditional loop
operation, and returning unit from an integer function is rejected. The
original loop replacement maps to pinned nightly
`tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs`.

The expression layer uses a fixed-capacity AST arena and implements Rust's
precedence ordering for unary operations and the arithmetic, shift, comparison,
bitwise, and short-circuit logical binary operators. It decodes decimal, binary,
octal, and hexadecimal integer tokens with the built-in integer suffix names.
Its current constant evaluator uses a `u128` bit representation for layout-like
expressions: it checks overflow, division by zero, and shift ranges, and resolves
names through a caller-supplied trait without allocation. Built-in integer `as`
casts carry explicit signedness and width, including an IR-selected pointer
width. It is not yet Rust's complete type-directed constant evaluator;
references, aggregates, and unrestricted const function calls remain
unsupported.
Bounded `if condition { expression } else { expression }` values are represented
directly in the expression arena. Conditions must be Boolean and both branches
must have compatible types. Direct evaluation, lowered IR, and native code all
select only the taken branch; malformed delimiters and missing `else` clauses
produce bounded diagnostics.
Scalar block expressions `{ expression }` transparently retain the inner value
and type without consuming another arena node. Empty `{}` blocks produce the
existing unit value. Blocks nest under the same depth bound and work in
constants, conditions, branches, locals, return values, IR, and native code;
statement-bearing blocks reuse the same scoped parser as conditional branches.
They may introduce four typed or inferred scalar bindings, perform eight
assignments, interleave eight expression statements, shadow outer names, and
produce either an explicit tail value or implicit unit. Their names do not leak
past the closing brace, while enclosing runtime identifiers remain available.
Scalar inline const blocks `const { expression }` and `const {}` retain a
distinct fixed-arena AST boundary while using the same value, type, IR, and
native-code paths. Prior module constants resolve inside that boundary. Runtime
parameters and locals are rejected rather than captured, including during
bounded const-function evaluation. An inline const used as a module-constant
initializer may call the existing bounded prior scalar `const fn` subset,
including Boolean results. Closed zero-argument prior Rust `const fn` calls are
also folded when used directly in a runtime function, with their integer or
Boolean type retained through IR and native emission. Up to four closed scalar
arguments are evaluated in source order, range-checked and normalized against
the callee's declared integer or Boolean parameter types, then bound in the
existing fixed-capacity const evaluator. Arguments that depend on runtime
parameters or locals remain rejected. Inline const blocks additionally accept
up to four source-ordered scalar `let` statements followed by a value
expression. Bindings may be inferred or explicitly annotated with a supported
integer or Boolean scalar type. An explicit ascription is retained as its own
checked expression node rather than treated as a truncating cast, and each
subsequent assignment retains that declared type. Bindings may be mutable; up
to eight direct or arithmetic,
remainder, shift, and bitwise compound assignments create successive checked
expression versions in source order. Immutable targets are rejected. Bindings
are substituted only within their current inline boundary, so nested inline
consts cannot capture them; dependent bindings retain declaration order. Let
declarations may also follow assignments, so mutation, later declarations,
and scalar shadowing are folded in source order under the same four-binding
and eight-assignment limits. Up to eight scalar expression statements may be
interleaved with those declarations and mutations. An explicit sequence node
evaluates each statement in lexical order, discards only its completed value,
and retains compile-time failures before evaluating the tail. Lowered IR uses
an explicit checked stack pop for the same boundary. Non-scalar annotations,
general control-flow statements, and items inside inline const remain separate
work. A trailing semicolon after any supported statement sequence supplies an
implicit unit tail, including after declarations and mutations. `return`,
`break`, and `continue` do not cross the inline-const boundary and fail closed.
Scalar branch blocks accept up to eight expression statements and an implicit
unit tail. Consequently an `if` without `else` is accepted only when its then
branch is unit-valued; a value tail is rejected rather than silently discarded.
The conditional retains lazy evaluation through direct folding, IR, and native
emission.
Each scalar conditional branch may also introduce its own four bounded typed or
inferred bindings, apply up to eight assignments, and interleave expression
statements. Branch-local substitution preserves shadowing and does not leak a
name into the enclosing inline block. Outer closed bindings remain visible in
branch initializers, and untaken initializers remain unevaluated.
Rust `else if` spelling is represented as a nested bounded conditional rather
than flattened or eagerly selected. Every link retains its own scoped branch
budgets, type checks, and lazy alternatives under the common expression-depth
limit. A final no-`else` link remains restricted to a unit-valued branch.
Bounded scalar `match` expressions accept up to eight source-ordered integer,
Boolean, signed-integer, or named constant patterns. A final `_` fallback is
required unless fixed-capacity coverage analysis proves both Boolean literals
are covered by unguarded arms or finds an unguarded irrefutable scalar binding
or wildcard alternative. In a proven match, the final arm becomes the lowering
default without synthesizing an unreachable expression. Guards never count
toward coverage, and incomplete supported matches produce
`NonExhaustiveMatch`. Integer patterns may also be inclusive `start..=end` or exclusive
`start..end` ranges with literal, signed, or named constant bounds. Ranges lower
to typed lower/upper comparisons. Open-ended `start..`, `..end`, and `..=end`
integer patterns use the corresponding single comparison; a completely
unbounded `..` arm is rejected rather than conflated with `_`. Each arm may
join up to four exact or range patterns with `|`; their conditions short-circuit
in source order. Top-level scalar alternatives may be grouped in recursively
nested parentheses or begin with `|`; groups flatten into the same bounded
four-alternative representation and therefore cannot bypass its capacity or
expression-depth limits. An optional Boolean `if` guard executes only after at
least one alternative matches; a false guard continues to the next arm. The combined
condition then enters the existing lazy conditional representation. Only the
first matching, guard-accepting arm executes through direct evaluation, IR, and
native code. Any arm may use a scoped scalar statement block, and the comma
after such a block arm is optional as in Rust. Character literals are a
distinct scalar type rather than integer aliases. ASCII, standard escaped,
two-digit hexadecimal, and validated Unicode-scalar spellings work in exact,
alternative, and bounded character range patterns; invalid Unicode scalar
values are rejected. A lowercase scalar identifier may bind an entire
scrutinee for one arm; the binding is substituted only into that arm's guard
and body, and successive irrefutable bindings retain source-ordered guard
fallthrough. Uppercase identifiers remain named constant patterns. Binding
alternatives require the same binding name in every branch. Guarded `_` arms
may appear before the required final unguarded
fallback; they participate in the same source-ordered, lazy guard selection,
and a match containing only the final `_` is valid. A scalar `name @ pattern`
may bind an exact, range, character, or wildcard subpattern and exposes the
scrutinee only to that arm's guard and body. Non-identifier left sides, nested
`@`, and inconsistent alternatives are rejected. Scalar `|` alternatives may
each bind the same name, including exact and range `@` patterns; a missing or
differently named binding produces a dedicated diagnostic before lowering.
Parenthesized alternatives inside a scalar `name @ (...)` subpattern flatten
into individually bound alternatives, so the same name is available in the
guard and body regardless of which exact, range, or wildcard branch matched.
Nested `@` bindings within such a group fail closed. Floating-point ranges,
binding modes, aggregate destructuring, and matches beyond eight non-fallback
arms remain separate work. Exhaustiveness for integer or character ranges,
named constants, and aggregates also remains separate type-aware work. Literal range
ordering is validated during parsing; named
integer endpoints retain one of thirty-two fixed-capacity validation records and
are checked after constant resolution with their declared signedness and target
pointer width. Reversed inclusive ranges and empty or reversed exclusive ranges
produce `InvalidRangeBounds`, while a one-value inclusive range remains valid.
The ordering check handles unsigned, signed, and character literals without
conflating their representations. Literal or named Boolean ranges are rejected
with `InvalidRangeType` rather than inheriting MRML's separately supported
Boolean comparison operators. The same metadata retains the scrutinee and both
closed or open endpoints, allowing code generation to reject a literal outside
the matched `u8` through `u64`, `i8` through `i64`, or 64-bit pointer-sized
integer type with `RangeEndpointOutOfRange` before emitting partial code.
An immediate value-producing loop expression of the form
`loop { break expression; }` is represented in the same fixed-capacity arena.
Its operand retains its integer or Boolean type through direct evaluation, IR
lowering, local initialization, and native emission. Up to four source-ordered
`if condition { break value; }` exits may precede the required fallback break.
The same exits may use a structured `if`/`else if`/`else` chain, matching the
scalar control-flow shape in the pinned `loop-break-value.rs` oracle. Every
condition must be Boolean, all five possible values unify to one type, and only
the selected operand is evaluated. These forms accept one
lifetime-style loop label and matching labeled breaks; missing colons, unknown
break labels, and a fifth conditional exit fail with dedicated expression
diagnostics. General iterating loop expressions, larger break-value graphs,
coercion and truly diverging nested loops remain separate work. A directly
nested labeled loop may use up to four conditional breaks targeting its
enclosing value loop, then complete through its own typed break before an outer
fallback. The inner completion value is evaluated and discarded rather than
unified with the outer exits; wrong-level targets receive
`InvalidLoopBreakTarget`. A Boolean inner value and `u32` outer values emitted
148-byte COFF and 544-byte ELF64 objects whose callers passed both the direct
outer exit and inner-completion/fallback paths. A terminating labeled value
loop may also be the operand of an enclosing labeled break, preserving both
labels, the inner value's type, and lazy inner alternatives through evaluation,
IR, and native code. Its original probe emitted 139-byte COFF and 536-byte ELF64
objects and passed zero, ordinary, and million-valued inputs through independent
native callers. The two pinned forms whose `while` condition itself is an
unconditional labeled `break` are also represented: a unit break can terminate
the current inner `while` before an outer fallback, while a valued break can
exit the enclosing value loop. The unreachable body is consumed with bounded,
balanced-brace scanning and is never evaluated; a non-unit value targeting the
inner `while` is rejected with `InvalidLoopBreakTarget`. Their combined probe
emitted 93-byte COFF and 488-byte ELF64 objects and returned 690 through
independent nightly-built callers on Windows gnullvm and Arch Linux WSL. Both
bodies contain division by zero, so executing either unreachable edge would
trap. Up to five source-ordered immediate breaks may also occupy the same value
loop. A constant-true internal edge preserves the first unconditional result
while retaining later unreachable operands for ordinary branch type checking;
the sixth break fails with `TooManyLoopBreakBranches`. This covers the pinned
unit pairs `break (); break;` and `break; break ();` without discarding their
second operands. Their combined probe emitted 169-byte COFF and 560-byte ELF64
objects and returned 42 through independent Windows gnullvm and Arch Linux WSL
callers. A separate regression rejects an unreachable Boolean value after an
integer break, and a trapping later integer operand remains unevaluated. An
associated `Default::default()` expression now retains an unresolved default
type until a scalar unit, integer, or Boolean context is available. It lowers
to the corresponding zero representation; an unconstrained inferred local is
rejected rather than silently choosing a type, and malformed method names or
arguments fail during parsing. Nested unlabeled `break` operands share their
single terminating semicolon with the enclosing break, while a value
immediately before the closing brace accepts Rust's semicolonless tail form.
Together these features
cover the oracle's three remaining scalar unit/default shapes, including
`break break Default::default()` and a break whose operand is an `if` with a
nested valueless break. Their combined probe emitted 207-byte COFF and 600-byte
ELF64 objects and returned 42 through independent Windows gnullvm and Arch
Linux WSL callers. Unconstrained aggregate defaults and general trait method
dispatch remain outside this scalar implementation. An original labeled
four-guard probe and
its structured-alternative counterpart each emitted
deterministic 407-byte COFF and 800-byte ELF64 objects. Independent
pinned-nightly callers selected every exit, covered signed results, and passed
a selected zero input that would trap if fallback division were evaluated.
Fixed-size scalar array literals now retain up to sixteen homogeneous elements in
the fixed expression arena, accept a trailing comma, and support constant or
runtime `usize` postfix indexes. Array construction evaluates every element in
source order before indexing, while an enclosing conditional array expression
still evaluates only its selected arm. A compile-time out-of-bounds index or a
seventeenth element fails with a bounded error; runtime out-of-bounds indexes reach
`UD2`. Runtime typing preserves the element type and length, rejects mixed
scalar element types, and rejects non-`usize` variable indexes. Index extraction
lowers through IR and native code without representing the aggregate as a
packed integer. The pinned `loop-break-value.rs` array literals therefore have
an original scalar-index replacement: its 93-byte COFF and 488-byte ELF64
objects returned the middle value `3` through independent Windows gnullvm and
Arch Linux WSL callers. Array-valued conditional loop-break branches now also
unify a `Default::default()` arm with a same-length fixed scalar array arm.
Scalar repeat literals `[expression; length]` accept an unsuffixed or `usize`
literal length up to sixteen and reuse the same fixed-array typing,
indexing, local storage, and mutation paths. Other integer suffixes and a
seventeenth element are rejected before lowering. An original bounded replacement covers
the first repeat-and-mutate shape from pinned nightly
`tests/ui/array-slice-vec/mutability-inherits-through-fixed-length-vec.rs`;
iteration over array references remains outside this slice. The exact pinned
run-pass test compiled and ran unchanged on both hosts. Repeat operands have a
distinct expression representation and are evaluated exactly once before their
value is copied, including when the length is zero. The pinned intentional
failure `tests/ui/array-slice-vec/eval-empty-array-expr.rs` printed its expected
panic on both hosts. MRML's zero-length division replacement emitted 131-byte
COFF and 528-byte ELF64 objects; an independent Linux call with a zero divisor
terminated with `SIGILL`. The bounded repeat-and-mutate replacement emitted
438-byte COFF and 832-byte ELF64 objects after single-evaluation materialization;
independent callers observed 5, 35, and 3,000,005 for zero, ordinary, and
million-valued inputs on Windows gnullvm and Arch Linux WSL.
Constant indexing preserves conditional-arm laziness, including a trapping
element in an unselected array arm, and materializes a selected default element
as zero. A combined probe covering both branch orders and a semicolonless tail
`break` emitted
93-byte COFF and 488-byte ELF64 objects; independent pinned-nightly callers on
both hosts observed 43. Immutable fixed-array locals now retain each of their
up to sixteen scalar elements in a distinct stack slot. Both explicit
`[scalar; N]` annotations and inferred literal types are accepted, including a
typed contextual default or loop-break array initializer. Constant indexes load
the selected slot while preserving parameter offsets, surrounding scalar
locals, scoped cleanup, and both native ABIs. A contextual loop-array probe
emitted 208-byte COFF and 600-byte ELF64 objects; independent pinned-nightly
callers observed 14 for its default path and 41 for its selected path on both
hosts. Runtime indexes evaluate once and dispatch across the bounded temporary
or local slots. A conditional-array probe emitted 352-byte COFF and 744-byte
ELF64 objects; independent callers observed 42, 42, 13, and 99 across both arms
and three indexes, including a zero divisor hidden in the unselected arm. An
out-of-bounds Linux call terminated with `SIGILL`. Mutable fixed-array locals
also support whole-array `=` assignment. The complete right-hand array is
materialized before any destination slot is overwritten, so self-referential
permutations retain Rust's value semantics; assignment from either conditional
branch follows the same path. Compound aggregate assignment is rejected. A
probe that permuted `[13, 42, 99]`, retained a following scalar local, and then
used a runtime index emitted 373-byte COFF and 768-byte ELF64 objects;
independent callers observed 100, 14, and 43 on both hosts. Indexed element
assignment now accepts constant or runtime `usize` indexes and all scalar
assignment operators supported by the element type. The index is evaluated
once before the right-hand side, compound assignment loads and updates the
selected element, a constant out-of-bounds target is rejected, and a runtime
out-of-bounds store reaches `UD2`. A mutable-array
compound-assignment probe emitted 452-byte COFF and 848-byte ELF64 objects;
independent callers observed 24, 48, and 101 across all three elements on
Windows gnullvm and Arch Linux WSL. Out-of-bounds calls terminated with
illegal-instruction status `0xC000001D` on Windows and `SIGILL` on Linux.
One-element fixed scalar array parameters now use the shared one-eightbyte
x86-64 ABI class. Integer, Boolean, and character elements retain their scalar
normalization through direct indexing and copies into fixed-array locals on
both Windows and System V. The exact pinned nightly
`tests/ui/consts/const_let_eq.rs` compiled and ran unchanged on both hosts. An
original `const fn` replacement emitted 128-byte COFF and 520-byte ELF64
objects; independent pinned-nightly callers passed `[0]`, `[13]`, and
`[1_000_000]` with following scalar arguments. Two-to-sixteen-element 64-bit
integer arrays additionally follow their platform ABI classes: Windows copies
elements from the caller-provided indirect argument, while System V consumes a
two-eightbyte array from registers when both are available and otherwise rolls
the whole aggregate back to its by-value stack area. Larger arrays use that
stack area directly. Following scalar arguments retain their independent
register allocation. The exact pinned nightly
`tests/ui/array-slice-vec/copy-out-of-array-1.rs` compiled and ran unchanged on
both hosts. Independent two-word probes emitted 216-byte COFF and 600-byte ELF64
objects; three-word indirect/by-value probes emitted 244-byte COFF and 640-byte
ELF64 objects. A register-exhaustion probe placed a two-word array pointer in a
Windows stack argument and rolled the same array back to the System V stack
while retaining the following sixth integer register; its 459-byte COFF and
840-byte ELF64 objects produced 204 and 120 on both hosts. Fixed-array returns
use the corresponding result classes: one scalar element returns in `RAX`, a
two-word System V array returns in `RAX`/`RDX`, and Windows multiword or System V
three-to-sixteen-word arrays use a preserved hidden result pointer while shifting
ordinary parameters to their ABI-defined locations. Tail values, explicit
returns, and conditional returns share the same materialization path. The exact
pinned nightly `tests/ui/parser/block-expr-statement-vs-expr.rs` compiled and ran
unchanged on both hosts. Independent one-, two-, and three-word return probes
emitted 137-, 167-, and 168-byte COFF objects and 528-, 544-, and 552-byte ELF64
objects; pinned-nightly callers observed `[42]`, `[13, 14]`, and `[13, 42, 99]`
on both hosts. Narrow integer, Boolean, and character arrays use their compact
one-, two-, or four-byte element layout at the ABI boundary while retaining one
auditable internal stack slot per element. Windows packs total sizes of one,
two, four, or eight bytes into a direct integer argument/result and otherwise
uses its indirect aggregate class. System V packs up to two eightbytes into
registers and rolls the complete value back to its stack class when necessary.
The exact pinned nightly `tests/ui/array-slice-vec/vec-fixed-length.rs` compiled
and ran unchanged on both hosts. Independent mutate-and-return callers passed a
four-byte `[u8; 4]`, a ten-byte `[u16; 5]`, and a three-byte `[bool; 3]` on both
ABIs. Their COFF objects were 553, 608, and 411 bytes; their ELF64 objects were
944, 1,048, and 824 bytes. Zero-length arrays and arrays of unit elements now
follow their zero-sized ABI classes without transporting result bytes. System V
omits zero-sized arguments from register allocation, while Windows preserves
their positional argument slots; later scalar arguments therefore arrive in
the same locations as pinned nightly on each platform. Array expressions still
evaluate normally, and unit elements retain their auditable internal slots.
Independent callers passed zero-length and three-unit parameters between scalar
arguments and consumed both zero-sized return forms on both hosts. The four
COFF objects were 195, 201, 99, and 119 bytes; their ELF64 counterparts were
584, 592, 488, and 512 bytes. Whole-array `Default::default()` values acquire
their fixed array type from returns and typed locals, then use the same direct
or indirect ABI materialization as explicit literals. A conditional `[u16; 5]`
default-return probe emitted 439-byte COFF and 856-byte ELF64 objects;
independent callers observed both the all-zero default and the explicitly
initialized alternative on both hosts. The fixed-array arena, runtime indexes,
locals, assignments, parameter classes, and return classes now share a
sixteen-element bound. Array cleanup switches to a 32-bit stack adjustment at
the exact 128-byte boundary. An independent `[u8; 16]` caller exercised
parameter transport, a copied mutable local, last-element assignment, and
return transport on both ABIs; the resulting objects were 1,602-byte COFF and
2,184-byte ELF64. Immutable references to supported integer scalars, `bool`,
and `char` now use a typed thin-pointer runtime representation. Reference
parameters occupy their ordinary pointer-sized ABI slot, inferred locals can
copy the pointer without truncation, and unary `*` emits width- and
signedness-correct pointee loads. Independent callers observed 42 through a
copied `&u64` between surrounding scalar arguments, preserved `-1234i16`, and
loaded `true` plus crab emoji `char` values on both hosts. The four COFF objects
were 156, 111, 111, and 110 bytes; their ELF64 counterparts were 544, 496, 504,
and 504 bytes. Explicitly typed scalar-reference locals preserve the same
pointer identity. `&mut T` parameters and typed locals share that representation
for dereference and scalar pointee mutation. Plain and compound assignments
store through mutable references with the pointee's exact integer, Boolean, or
character width; immutable references reject stores. Independent callers
mutated `40u64` to 42 and observed both the returned value and changed memory
through 156-byte COFF and 544-byte ELF64 objects. Defaults without a
constraining type remain separate work. Scalar locals and saved scalar
parameters now support shared address-taking, mutable locals support mutable
address-taking, and existing scalar references support `&*value` and
`&mut *value` reborrows with mutability checks. An independent caller used a
mutable local reference and then a mutable reborrow to change external memory
from 40 to 42 through 199-byte COFF and 584-byte ELF64 objects. Aggregate
reference parameters now include fixed arrays up to sixteen elements. Shared
and mutable fixed-array references remain one-word pointers, retain their
element/count metadata through typed local copies and reborrows, and support
bounds-checked constant or runtime indexing plus exact-width mutable element
stores. Independent callers changed `[40usize, 10, 20, 30]` through a runtime
index and observed `[40, 10, 22, 30]` plus the returned sum 62 through 237-byte
COFF and 624-byte ELF64 objects. References can also target MRML's internal
expanded fixed-array locals. Runtime metadata distinguishes their reversed
eight-byte stack stride from native contiguous arrays and preserves it through
typed copies and reborrows. Independent callers mutated indexes 0, 2, and 3
and observed 42, 22, and 32 through 272-byte COFF and 656-byte ELF64 objects.
Shared and mutable scalar-element slice parameters use their native two-word
fat-pointer representation. The Windows prologue follows pinned rustc by
dereferencing an indirect pair, while System V captures separate data and
length argument words. Typed copies and reborrows preserve both words;
constant and runtime indexing checks the saved length before exact-width reads
or mutable stores. Independent callers passed a four-element mutable slice,
changed index 2 from 20 to 22, and observed the changed backing array plus the
returned sum 62 through 337-byte COFF and 720-byte ELF64 objects. The postfix
`len()` method reads the preserved runtime length from shared
or mutable slice references, including typed copies and mutable reborrows. A
native four-element probe added the copied and original lengths and returned 8
through 146-byte COFF and 528-byte ELF64 objects. The `is_empty()` method tests
that same runtime length and normalizes the result to Rust `bool`. Independent
callers observed true for an empty slice and false for a one-element slice
through 130-byte COFF and 520-byte ELF64 objects.
Constant `start..end`, `start..=end`, `..end`, `start..`, and `..` ranges over
native fixed-array references create shared or mutable slice fat pointers.
Bounds and mutability are checked before code emission; typed locals retain the
adjusted data address and derived length. Independent callers sliced elements
1 through 2 from a four-element mutable array reference, changed 20 to 22, and
observed length 2 plus the changed backing array through 309-byte COFF and
696-byte ELF64 objects. Runtime range endpoints, ranges over expanded array
locals, and general aggregate ABI transport remain separate work.
The zero-sized unit value `()` is a distinct runtime expression type. It flows
through condition branches, locals, immediate loop breaks, IR, and native code.
Value-producing loops treat a valueless `break;` as the same unit type as
`break ();`, including mixed guarded and structured alternative exits with
labels. This covers the pinned oracle's scalar `regular_break` forms. A
132-byte COFF and 528-byte ELF64 probe passed both alternatives through
independent native callers. Both explicit `-> ()` and an omitted function
return type select a C-ABI void result. Empty, declaration-only, and supported
statement-ending bodies receive an implicit unit tail with a zero-width source
span at the body boundary. Unit is not silently compatible with integers or
Booleans.
Boolean and unit operands support Rust's equality and ordering comparisons.
Boolean operands additionally support eager `&`, `|`, and `^`; unlike `&&` and
`||`, IR evaluates both operands. Results remain Boolean through direct
evaluation, IR, and native code. Mixed Boolean, unit, and integer operands are
rejected.
Mutable Boolean locals support `&=`, `|=`, and `^=` in straight-line and loop
bodies using the same eager bitwise semantics. Arithmetic, remainder, and shift
compound assignments on Boolean targets remain type errors.
An explicit tail `return expression;` is also represented in the arena and
flows through the same typing, IR, and native paths as an implicit tail value.
Function bodies may additionally contain bounded prefix statements of the form
`if condition { return expression; }`. Their conditions and return values are
type-checked, and each native return removes the complete saved parameter/local
frame before `RET`. Exhaustive `if`/`else if` return chains retain up to four
condition-and-value branches plus a final `else` value in a separate bounded
record. Native and const paths
evaluate only the selected branch, and an exhaustive return is accepted as a
diverging body without a synthetic fallback value. Typed conditional and unconditional returns are supported
inside bounded loops. Returns in arbitrary nested blocks, `else` statement arms,
other than the exhaustive form above, and arbitrary statement blocks remain
deliberately rejected.
The same bounded chain may omit its final `else` when it contains at least two
returning conditions. If every condition is false, native execution and const
evaluation continue with subsequent statements and the function tail. A
single conditional return retains the smaller pre-existing record.
Bounded scalar `while condition { ... }` and `loop { ... }` statements retain
up to eight source-ordered operations, including typed or inferred local
declarations, discarded scalar expressions, up to four mutations, and
unconditional or conditional `break`, `continue`, and typed function returns.
While and control conditions must be Boolean, each assignment target must be
mutable, and assignment types are checked exactly as for straight-line
statements. Each iteration receives a lexical local scope: declarations may
shadow enclosing names, remain visible to later operations, and are removed
before every backedge, taken break, or taken continue. A return includes the
live iteration slots in its complete frame cleanup. Const evaluation restores
the same scope after every iteration while retaining mutations to enclosing
locals. Native emission uses checked forward exits and signed 32-bit backedges.
Capacity overflow is diagnosed before object emission. Empty and action-only
unconditional inner loops are retained as diverging backedges; const evaluation
stops them at the shared 65,536-iteration limit rather than treating natural
fallthrough as an exit. The exact unit-valued inner form `loop { break; }` is
accepted as a nested loop operation.
The same divergence rules apply to top-level loop statements. Empty `while`
bodies recheck their Boolean head, and empty or action-only unconditional loops
emit a signed backedge without requiring a synthetic exit. Source after a
diverging loop remains parsed and type-checked. Const evaluation reports the
same loop-limit diagnostic for an entered nonterminating body.
Up to four inner loops may each perform up to four scoped scalar locals,
assignments, or discarded expressions before an optional control. Inner
locals are removed while mutations to enclosing locals survive. Dedicated
capacity diagnostics reject a fifth inner action or block. The inner break is
consumed locally and cannot exit the containing loop.
The inner exit may instead be a terminal `if condition { break; }`. A false
condition cleans the iteration's inner locals and repeats from the inner action
block; a true condition performs the same cleanup and resumes the containing
loop after the inner operation. Conditions are Boolean, may read inner locals,
and use the existing 65,536-iteration const-evaluation bound.
Up to four ordered `if condition { continue; }` edges may split the inner action
block. A taken edge cleans all inner locals and targets the inner backedge; a
false condition retains them for the remaining actions and controls. Continue
and return edges share exact source ordering. A fifth conditional continue is
rejected before lowering.
Up to four ordered `if condition { break; }` edges may likewise occur before,
between, or after inner actions. A true guard cleans the inner scope and exits
only the nested loop; a false guard preserves that scope for later actions and
controls. Break, continue, and return edges share one exact source order, and a
fifth conditional break is rejected before lowering.
An inner block may instead use a Boolean `while` head. The condition is checked
before every inner iteration, false exits only the inner loop, and natural body
fallthrough cleans inner locals before returning to the inner head. The same
bounded actions and optional conditional controls remain available in its body.
An explicit unconditional `break;` in that body is retained separately from
natural closing-brace fallthrough: the former exits after one entered iteration,
while the latter returns to the head. This distinction prevents a true-headed
`while { break; }` from being lowered as an infinite backedge.
A nested `loop` or `while` body may terminate with a typed function `return`.
Its value is checked and emitted while inner locals remain live, then the normal
function epilogue removes the complete inner and outer frame. A false inner
`while` head bypasses the return and resumes after the nested operation.
An unconditional terminal `continue;` cleans the complete inner iteration
scope and targets the nested loop's own head. For an inner `while`, the head
condition is rechecked before the next iteration; it never skips the remaining
operations of the containing loop.
Unconditional inner `break`, `continue`, and function-return controls retain
their exact position in the same bounded control stream. Source after a taken
edge remains parsed and native-type-checked as unreachable Rust source, while
const evaluation and emitted execution do not evaluate it. Up to four such
controls are retained so later unreachable controls receive the same checking;
a fifth produces a dedicated capacity diagnostic before lowering.
Up to four conditional function returns may occur at ordered positions among
the inner actions. Each false guard advances to the next operation without
changing the inner scope; the first true guard returns after full-frame cleanup.
A fifth guarded return is rejected before lowering.
Conditional returns also retain their order relative to the inner conditional
continue. If both guards at one action position are true, the source-earlier
control edge wins; a taken continue therefore suppresses every later return in
that iteration.
An unconditional inner `loop` may use only conditional continue and function
return controls. False guards reach the inner backedge rather than escaping the
loop, while the first true return performs full-frame cleanup. This preserves
Rust's diverging-loop behavior without requiring a synthetic break edge.
Statement-level `while` and `loop`, plus their directly nested `while` or
`loop`, may carry Rust lifetime-style labels. Matching labeled `break` and
`continue` work in direct operations and bounded `if`/`else if`/`else` control
arms. A nested control may target either its own label or the enclosing
statement loop's label, with Rust-style inner-label shadowing; missing colons
and unknown labels fail with dedicated bounded diagnostics. General deeper
statement-loop nesting and break values in statement-loop operations remain
unsupported. Three native probes cover conditional inner and outer targets,
direct outer continue, and direct outer break. They emitted deterministic 439-,
198-, and 181-byte COFF objects and 832-, 592-, and 576-byte ELF64 objects;
independent pinned-nightly callers passed zero-iteration, ordinary, early-exit,
and 60,000-iteration paths on both hosts.
Conditional loop-control blocks may perform up to four local declarations,
assignments, or discarded scalar expressions before a terminal `break`,
`continue`, or typed function return. Their actions are lazy and receive a
nested lexical scope. Taken break/continue edges remove both the conditional
block's locals and the containing iteration's locals; a return cleans the full
live frame. False conditions allocate no block slots and continue with later
loop operations. A fifth prefix action fails with
`TooManyConditionalLoopActions` before lowering.
The terminal control operation may be omitted. A selected action-only block
cleans its lexical locals and falls through to the next loop operation; an
unselected block performs no actions or stack changes.
Loop conditionals may additionally carry a bounded chain of up to four
`else if` or final `else` arms with the same action and terminal-control forms.
The first matching arm is selected lazily, and each has an independent lexical
scope. The compact alternative-arm arena keeps the common `WhileLoop` record
bounded, and a fifth arm fails with
`TooManyConditionalLoopElseArms` before lowering.
An unconditional loop with no break edge is a diverging body tail, so a
non-unit function ending in a loop return needs no synthetic fallback value.
Const qualification is retained on both Rust-ABI and `extern "C"` function
items and is distinct from a module constant declaration. Qualifying an
otherwise supported exported C function does not change its runtime ABI or
machine code. The bounded scalar call evaluator described below interprets a
restricted subset of const-function bodies; complete const-safety analysis is
not implied by parsing the modifier.

The first semantic pass builds a fixed-capacity constant table, rejects duplicate
module names across functions and constants, resolves dependencies in declaration
order, translates initializer diagnostics back into module source spans, and
checks values against their declared integer type. Root positive and negative
signed literals are range-checked against that declared width and stored as
their low-width two's-complement bit pattern. Signed add, subtract, multiply,
divide, remainder, bitwise operations, and shifts execute in the declared type
with checked overflow, zero-divisor, and shift-distance failures. `usize` and
`isize` range checking uses an explicit validated target pointer width instead
of inheriting the compiler host's layout. Boolean constants support literals,
named prior Boolean constants, short-circuit `&&` and `||`, eager Boolean
bitwise operations, and integer or Boolean comparisons. Signed constant `if`
expressions select only the taken branch, and integer `as` casts preserve source
signedness while applying the destination width. Constant calls and
floating-point values remain unsupported except for a bounded integer scalar
call boundary. A root constant initializer may call a prior Rust-ABI
`const fn` with at most four signed or unsigned integer arguments when every
parameter and the return type match exactly. Arguments and results are
range-checked in their declared widths. Calls may be nested directly as call
arguments, and a function body may directly return a call to an earlier
function. Calls also compose with checked same-typed integer arithmetic,
integer casts, lazy `if` branches, and integer equality or ordering conditions.
Evaluation uses a fixed 64-binding environment and rejects a ninth nested call.
Boolean parameters and return values retain their type in the same environment;
Boolean calls compose with lazy or eager logical operations, integer predicates,
and integer-returning conditionals. A function body may resolve itself as well
as earlier declarations, enabling terminating direct recursion within the same
eight-call limit. Forward mutual recursion, non-const calls, methods, and
runtime calls are rejected.
Const-function bodies use a separate 16-node expression arena and the bounded
body parser. A fixed 32-entry statement stream records locals, assignments,
conditional returns, and loops in source order while compatibility views remain
available for the native backend migration. They may introduce up to the
caller-selected local capacity of
typed or inferred integer and Boolean locals, perform source-ordered
conditional returns, or use an explicit tail `return`. Mutable scalar locals
support direct assignment, checked arithmetic/remainder/shift compound
assignments, integer bitwise compound assignments, and Boolean bitwise compound
assignments. Immutable targets and type mismatches fail before mutation; an
arithmetic failure is transactional because the new value is committed only
after successful evaluation. Bounded `while` and `loop` statements reevaluate
their Boolean condition, execute scalar assignments in source order, and honor
conditional or unconditional `break` and `continue`. Each loop permits at most
65,536 body iterations; exceeding that budget reports
`ConstLoopLimitExceeded`. A loop may conditionally or unconditionally return an
integer or Boolean value of the const function's declared type. Nested statement loops remain
outside the parser boundary. This design avoids treating statement text as a
whole expression and keeps each recursive call frame's stack use bounded.
Const evaluation consumes the ordered stream and supports a scalar assignment
after a loop. Native code generation uses the same statement indices to emit
every pre-loop local, assignment, and conditional return directly in source
order, then dispatches post-loop locals, assignments, and conditional returns
on their respective control-flow edges. Consecutive bounded loop statements retain their source order in both
const evaluation and native emission. A typed or inferred local binding may
also immediately follow the loop sequence; its initializer observes the final
loop state and its native stack slot is created only on that fallthrough edge.
A bounded assignment may follow a conditional return. Const evaluation and
native emission execute it only on the fallthrough edge and preserve checked
arithmetic behavior. Before the first loop, bounded locals, assignments, and
conditional returns may alternate repeatedly; parsing and native emission both
consume the same ordered statement stream. The identical bounded sequencer is
used after the loop sequence, so fallthrough locals, mutations, and guarded
returns may likewise alternate without category-based reordering.
Comparison operands may be named or compound integer expressions. Unsuffixed
literals inherit the other operand's concrete type; incompatible concrete
widths and narrow arithmetic overflow are rejected before comparison.

Expressions lower into a separate fixed-capacity, target-independent stack IR.
The IR has checked arithmetic, explicit constant loads, normalization, and
forward conditional branches; `&&` and `||` therefore retain short-circuit
behavior after lowering. A bounded interpreter provides the bootstrap execution
backend and independently checks stack overflow, underflow, invalid branches,
unknown constants, and malformed final state. Constant semantic analysis now
uses the parse → AST → IR → interpreter path, while differential tests compare
that path with direct AST evaluation.

The first native backend accepts expression-bodied functions returning `u8`,
`u16`, `u32`, `u64`, or 64-bit `usize`. Native object export requires the honest
stable boundary `#[unsafe(no_mangle)] pub extern "C" fn`; ordinary Rust-ABI
functions are rejected because Rust symbol mangling and its native ABI are not
stable contracts. Zero-parameter functions resolve and execute their checked IR
at compile time, validate the declared return range, and emit the canonical
11-byte x86-64 `MOV RAX, imm64; RET` body. Parameter expressions use an
ABI-specific prologue that saves up to four Windows x64 or six System V AMD64
register arguments in bounded stack storage. The typed runtime backend accepts
`u8`, `u16`, `u32`, `u64`, 64-bit `usize`, `i8`, `i16`, `i32`, `i64`, and 64-bit
`isize` parameters and returns. It emits literals, same-typed named signed or
unsigned module constants, integer negation and bitwise not, add, subtract, multiply, divide,
remainder, bit operations, shifts, and signed or unsigned comparisons. Boolean
parameters and returns support logical not plus branch-based short-circuit
`&&` and `||`; mixed integer comparisons and Boolean guards are type-checked.
Runtime integer casts preserve their concrete target type, reject mismatched
function returns and unsupported 128-bit targets, and emit x86-64 zero/sign
extension. Shift right-hand operands may use a different integer cast type,
matching the selected upstream `shift.rs` cases.
Narrow
ABI inputs are explicitly zero- or sign-extended and results are normalized to
their declared width. The minimum signed literal is accepted through unary
negation. Comparison expressions are type-checked as `bool`; returning them from
an integer function is rejected. Checked arithmetic is the default: overflow,
division by zero, `MIN / -1`, `MIN % -1`, and an
out-of-range shift branch to one shared `UD2` trap. `CodegenOptions::WRAPPING`
and the driver spelling `-C overflow-checks=no` disable add, subtract, and
multiply overflow traps and emit width-correct wrapping results. In that mode,
shift distances are masked to the low `log2(type_width)` bits, matching rustc's
`overflowing-lsh-4.rs` and `overflowing-rsh-4.rs` requirements. Output storage is fixed-capacity, and
unsupported signatures,
expressions, names, ranges, ABIs, operators, and output sizes carry source spans.
Up to 16 parameters are supported. Arguments beyond the first four Windows x64
or first six System V AMD64 register arguments are loaded from their ABI stack
slots with checked displacement arithmetic and saved into the same bounded
evaluation frame. Larger signatures are rejected before emission.
Zero-parameter integer functions may use typed or inferred locals. Initializers
are parsed, type-checked, lowered, and interpreted in declaration order; each
binding becomes visible only after successful evaluation. Duplicate bindings,
forward references, unsupported types, and out-of-range values are rejected.
Parameterized functions emit same-width integer locals and Boolean locals into
bounded stack slots. Later initializers and the tail can reload those slots;
parameter offsets incorporate saved locals and temporary expression depth, and
the epilogue checks and removes the complete combined frame. Mutable integer
and array frames use signed-eight-bit stack displacements only through offset
127 and switch to checked 32-bit displacements at 128 bytes. Array unpacking
uses volatile `R10` scratch storage on both x86-64 ABIs so it cannot overwrite a
later unsaved argument register. Independent callers exercised both an
indirect eight-word array with a copied local at a 136-byte load and a packed
four-byte array followed by a scalar argument. They observed 42 on both hosts;
the corrected deep-array objects were 367-byte COFF and 784-byte ELF64, while
the packed objects were 235 and 624 bytes.
Mutable integer
locals support `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, and
`>>=` with the same checked or wrapping arithmetic policy as expressions.
Boolean locals support simple assignment. Assignment to an immutable or unknown
binding, type mismatch, invalid operation, and out-of-range constant result are
reported with bounded source spans.
Local annotations may be omitted. Explicit integer suffixes and Boolean
expressions determine their own types; unsuffixed integer initializers use the
function's supported integer context. A conflicting suffix or native frame
width is rejected rather than silently converted. This is a bounded bootstrap
subset of Rust's constraint-based type inference, not a claim of complete
inference.

The object layer wraps one emitted x86-64 function in deterministic ELF64 or
AMD64 COFF relocatable bytes. Both writers emit the native file header, `.text`,
external function symbol, symbol/string tables, and the required section table
without filesystem access or allocation. ELF section offsets are aligned and
COFF supports both inline eight-byte names and long string-table names. All
offset conversion, output capacity, empty names, and embedded NULs are checked
before an artifact is returned. Relocations and multi-function objects are not
yet supported because the current native function has no external references.

Statements, other item kinds, attributes, generics, full name resolution and
type checking, typed constant evaluation, general function/control-flow IR,
runtime machine-code generation, object relocations and multi-section emission,
linking, and Cargo-compatible package orchestration remain future milestones and
are not represented as complete.

## Verification evidence

The initial compiler slice was verified on 2026-08-21 against pinned nightly
`rustc 1.100.0-nightly (f7d782a3b 2026-08-19)`.

The official `rust-lang/rust` repository was sparse-checked out at the full
matching commit. Its `tests/ui` tree contains 7,401 files marked `run-pass` or
`check-pass`. The first oracle slice executed these upstream `run-pass` files
unchanged with that nightly under native Linux: `arith-unsigned.rs`,
`div-mod.rs`, `shift.rs`, and `bitwise-ops-platform.rs` from
`tests/ui/numbers-arithmetic/`, plus
`tests/ui/consts/control-flow/short-circuit.rs`; all five exited successfully. Original MRML
probes mapped to those cases and to `tests/codegen-llvm/overflow-checks.rs`,
`integer-overflow.rs`, and `abi-x86_64_sysv.rs` run through both MRML object
backends. Unsupported narrow and signed cases are asserted as explicit errors,
not counted as compatibility passes.
The next oracle slice ran `i8-incr.rs`, `u8-incr.rs`, `u8-incr-decr.rs`,
`u32-decr.rs`, and `i32-sub.rs` unchanged with the same pinned Linux nightly;
all five also exited successfully before their executable MRML replacements
were added.
The complete upstream `tests/ui/consts/control-flow/basics.rs` also compiled and
ran unchanged under the pinned nightly. The current MRML replacement covers its
module-constant absolute-difference `if` expression and its recursive scalar
Euclidean GCD behavior. An original scalar iterative Fibonacci replacement now
covers the same early base case and loop state transitions without copying the
upstream tuple representation. Recursive loops, tuples, macros, and general
matches remain separate future slices.
The complete upstream `tests/ui/consts/return-in-const-fn.rs` likewise compiled
and ran unchanged under the pinned nightly before its explicit tail-return
replacement was added.
The complete upstream `tests/ui/for-loop-while/while.rs` and `long-while.rs`
also compiled and ran unchanged. MRML replacements cover their scalar counter
mutation, million-iteration termination behavior, and bounded per-iteration
scalar local scope. Printing remains outside this slice.
The complete upstream `tests/ui/consts/const-fn-const-eval.rs` and
`tests/ui/consts/const-extern-fn/const-extern-fn.rs` also compiled and ran
unchanged. MRML replacements cover const-qualified scalar declarations and
runtime C-ABI execution, but not their compile-time calls, arrays, references,
unsafe functions, closures, or floating-point cases.
The complete upstream `tests/ui/for-loop-while/while-with-break.rs` and
`break.rs` also ran unchanged. The MRML replacement covers the scalar `while`
portion with a post-mutation conditional break; allocation/drop behavior,
iterators, and general statement sequencing remain outside that slice.
The complete upstream `tests/ui/for-loop-while/while-cont.rs` and
`loop-break-cont-1.rs` then compiled and ran unchanged. Original MRML probes
cover condition rechecking after a final explicit `continue`, immediate
unconditional loop exit, and mutation inside an unconditional loop followed by
a conditional break. The complete upstream `loop-break-cont.rs` subsequently
compiled and ran unchanged. Its interleaved conditional-break, mutation,
conditional-continue, and post-continue mutation shape now has an original MRML
replacement; printing and assertions remain oracle-side behavior.
The complete upstream `tests/ui/for-loop-while/loop-break-value.rs` also compiled
and ran unchanged under the pinned nightly. Original MRML replacements cover
its immediate scalar integer, Boolean, and valueless or explicit-unit
break-value forms, including matching labels. The oracle file's array-valued
breaks, references, trait coercions, never type, matches, and more general
nested control-flow graphs are not claimed by this slice. Its fixed scalar
array literals now have bounded direct-index coverage, including contextual
array defaults selected through conditional loop-break branches and retained in
fixed-array locals, plus bounded runtime `usize` indexes and whole-array
or indexed-element assignment for mutable locals. Replacements
cover bounded cross-nested value exits, terminating nested value-loop operands,
up to five compatible competing scalar values, both sequential and structured
alternative syntax, lazy selection of the taken break edge, and the oracle's
labeled `break`-as-`while`-condition cases plus its two source orders for
unreachable unit sibling breaks, its three scalar
`Default::default()`/nested-break forms,
and the three directly indexed array/default branch shapes. General `while`
condition expressions, aggregate parameters or returns, unconstrained
aggregate defaults, and arbitrary nested value-loop graphs remain outside this
bounded slice.
The complete upstream `tests/ui/for-loop-while/for-loop-has-unit-body.rs` and
`loop-break-cont-1.rs` also compiled and ran unchanged for the unit slice.
Original MRML probes cover unit expressions, unit locals, unit-valued immediate
breaks, an actual unit-returning unconditional loop function, and implicit unit
tails after empty, local-only, and loop-statement bodies; `for` iterator
desugaring and the oracle's `std::mem::zeroed` call remain separate work.
The complete upstream `tests/ui/expr/if-generic.rs` and `weird-exprs.rs` then
compiled and ran unchanged. Original scalar replacements cover their Boolean
equality and unit-equality behavior; generics, closures, references, function
calls, macros, and the remaining unusual expression forms are not claimed.
The complete upstream run-pass `tests/ui/binop/binops.rs` also compiled and ran
unchanged. Its Boolean ordering and bitwise truth tables and unit ordering have
original scalar MRML replacements. The pinned diagnostic test
`tests/ui/consts/min_const_fn/min_const_fn.rs` was rejected by rustc at its
annotated invalid const-function cases as expected; MRML replacements map only
its accepted const-qualified Boolean `&`, `|`, and `^` declarations.
The exact pinned check-pass
`tests/ui/lint/unused/issue-117142-invalid-remove-parens.rs` compiled and ran
unchanged (with its ordinary unused-value warnings), establishing the Boolean
`|=` oracle. Original replacements cover all three Boolean compound assignments
and an ordered-loop Boolean parity mutation.
The complete upstream run-pass `tests/ui/consts/const-negative.rs` compiled and
ran unchanged under the pinned Linux nightly. Original MRML replacements cover
negative `isize` and `i32` constants, the exact `i8` minimum, both x86-64 object
formats, and rejection when a constant's signed width differs from its runtime
expression context. The complete upstream `tests/ui/consts/const-binops.rs`
also compiled and ran unchanged. MRML replacements cover its signed integer
add, subtract, multiply, divide, bitwise, and shift subset. A following
replacement covers Boolean `&&`/`||`, Boolean names and bitwise operations, and
the oracle's signed and unsigned integer equality and ordering constants.
Floating-point declarations remain outside the claimed coverage.
The pinned run-pass `tests/ui/cast/constant-expression-cast-9942.rs` compiled
and ran unchanged, as did the previously selected complete const-control-flow
oracle. Original MRML replacements cover integer constant casts, signed and
Boolean lazy `if` values, truncation, sign extension, and an untaken invalid
arithmetic branch. Pointer, character, enum, and floating-point casts are not
claimed.
The pinned check-pass `tests/ui/consts/const-block-items/assert-pass.rs`
compiled and ran unchanged. Its arithmetic equality observation has an original
module-constant replacement, extended with named signed and narrow unsigned
compound comparison operands. Const-block item syntax and `assert!` expansion
remain outside this slice.
The complete pinned run-pass `tests/ui/consts/const-fn.rs` and
`const-fn-const-eval.rs` compiled and ran unchanged. Original MRML replacements
now execute prior integer scalar `const fn` declarations with checked argument
binding and feed their results into both native object formats. Generics,
arrays, and unsafe calls remain outside this call boundary.
The pinned run-pass `tests/ui/consts/const-fn-nested.rs` compiled and ran
unchanged. Its call-as-argument shape has an original MRML replacement, along
with direct prior-function calls from a const-function body and fail-closed
coverage for the eight-call nesting limit. Calls embedded in arithmetic and
general call statements are not claimed.
An additional original black-box observation compiled and ran with the same
pinned nightly before its MRML replacement was added. It covers calls as
checked `u8` arithmetic operands, a call comparison selecting a lazy branch,
an integer cast followed by wider arithmetic, and a Boolean comparison result.
The corresponding adversarial cases reject intermediate narrow overflow and
skip an overflowing call in an untaken branch.
The Boolean call observation was also compiled and run under the pinned
nightly after confirming the accepted Boolean const-function declarations in
`tests/ui/consts/min_const_fn/min_const_fn.rs` and the Boolean const result in
the run-pass `tests/ui/consts/const-meth-pattern.rs`. Original replacements
cover Boolean parameters and results, integer predicates returning Boolean,
lazy nested calls, Boolean-to-integer function composition, exact mismatch
rejection, and an untaken invalid arithmetic path. Associated methods and
method patterns remain outside the claimed MRML boundary.
The pinned `tests/ui/consts/control-flow/basics.rs` was rerun unchanged for the
direct-recursion slice. Its Euclidean GCD call now has an original MRML
replacement. Additional regressions cover signed countdown and Boolean parity
recursion, stop nontermination at the eight-call bound, and reject forward
mutual recursion before evaluating its cycle.
The pinned run-pass `tests/ui/consts/return-in-const-fn.rs` was rerun unchanged
for the statement-body slice. Its explicit return has an original MRML
replacement alongside integer and Boolean locals and conditional returns. A
regression source that previously exhausted the Windows debug driver's default
process stack now compiles normally. The Windows driver records an 8 MiB PE
stack reserve through a linker directive while retaining its 256-node public
expression capacity; the reserve is demand-committed by the operating system.
The complete pinned run-pass `tests/ui/consts/const_let_eq.rs` also compiled and
ran unchanged. Original MRML replacements cover its scalar direct and compound
assignment boundary, plus Boolean mutation and an integer XOR swap. Deferred
initialization, projections, and aggregates remain outside this slice. Its
empty and scalar no-effect statements now have original bounded replacements
in the shared source-ordered body stream. Adversarial cases reject immutable assignment and preserve
checked overflow and division-by-zero failures.
The pinned file was rerun unchanged on both hosts for this statement slice. An
additional original oracle interleaves an empty statement, discarded integer,
Boolean, and lazy conditional expressions, a local declaration, and a later
assignment. The same ordered representation executes inside bounded const
functions and the runtime backend. MRML's 363-byte COFF and 768-byte ELF objects
passed independent `no_std` callers on three paths, including zero-divisor paths
protected by short-circuit and conditional selection. A ninth expression
statement produces `TooManyExpressionStatements`.
The pinned check-pass
`tests/ui/consts/const-eval/stable-metric/ctfe-simple-loop.rs` compiled and ran
unchanged. Its mutable scalar `while` behavior has an original MRML replacement,
extended with ordered conditional continue/break and an unconditional `loop`.
An endless `while true` regression stops at the exact MRML iteration budget,
and a loop overflow fails before committing the assignment.
The complete pinned run-pass `tests/ui/consts/consts-in-patterns.rs` also
compiled and ran unchanged. Its signed zero-argument call has an original MRML
replacement alongside negative-argument signed arithmetic and exact `i8::MIN`
binding. A subsequent original oracle covers a named scalar constant pattern,
signed literal pattern, wildcard fallback, omitted block-arm comma, and lazy
arm selection. General pattern forms and reference constants are not claimed.

Windows used the supported native GNU/LLVM host:

```powershell
$env:RUSTFLAGS = '-C link-arg=/stack:16777216'
cargo +nightly-x86_64-pc-windows-gnullvm test `
  -p mrml-rustc -p mrml-rustc-driver `
  --target x86_64-pc-windows-gnullvm --offline
cargo +nightly-x86_64-pc-windows-gnullvm clippy `
  -p mrml-rustc -p mrml-rustc-driver --all-targets --no-deps --offline -- `
  -D warnings
cargo +nightly-x86_64-pc-windows-gnullvm check -p mrml-rustc `
  --target x86_64-unknown-uefi --offline
cargo +nightly-x86_64-pc-windows-gnullvm check -p mrml-rustc `
  --target nvptx64-nvidia-cuda --offline
```

The 301 Windows library, conformance, rustc-nightly-replacement, and driver
tests passed.
A release driver emitted a 93-byte COFF object. Rust's bundled `rust-lld`
accepted it as the sole input to a 1 KiB PE executable with `/entry:answer
/subsystem:console /nodefaultlib`; the resulting executable returned status 42.

The runtime-expression increment additionally emitted a 179-byte COFF object
for `(left * 3 + right) / 2`. A separate nightly-produced `no_std` caller object
linked with it using only bundled `rust-lld`; the executable validated
`calculate(20, 24) == 42` and exited normally. An independently linked
`value + 1` artifact invoked with `u64::MAX` terminated with Windows status
`0xC000001D`, confirming the generated overflow edge reaches `UD2`.

A 196-byte COFF narrow-integer artifact executed the rustc arithmetic probe
`255u8 / 10 + 255u8 % 10 == 30`. Checked `255u8 + 1` reached `UD2`, while the
130-byte object emitted with `-C overflow-checks=no` returned the wrapped value
zero through an independently compiled C-ABI caller.
Unchecked `1u8 << 17` returned 2 after distance masking, and an exported
`left: u8, right: u8 -> bool` comparison returned both true and false correctly
through independent callers.

A 353-byte signed COFF artifact produced `-10` for the combined negative
division, remainder, and arithmetic-shift probe. A wrapping signed-add object
returned `i8::MIN` for `i8::MAX + 1`; the corresponding checked object and an
unchecked `i8::MIN / -1` object both reached the shared `UD2` trap.

Separate `guard || dangerous_rhs` and `guard && dangerous_rhs` COFF objects
returned correctly for true and false inputs. Both skipped a right-hand
`10 / 0` expression when the left operand determined the result, while ordinary
nonzero right-hand evaluations still produced the expected Boolean values.
Zero-parameter `true || dangerous_rhs` and `false && dangerous_rhs` objects
also linked against an independently nightly-compiled caller and exited
normally, covering Boolean literals and literal-driven short circuiting.
An immutable `static SHIFTED: usize = 10_usize << 4_usize` from the upstream
shift-test pattern was evaluated through bounded IR; the emitted function
returned 160 through an independently nightly-compiled caller.
A fifth-argument Windows x64 function consumed its first stack argument and
returned 42 through a separately compiled caller. The backend also rejects a
seventeenth argument and uses non-sign-extending 32-bit stack cleanup for the
maximum 16-argument frame.
The typed-local `isize` probe derived from `div-mod.rs` evaluated
`15 / 4 + 15 % 4`, emitted a 107-byte COFF object, and returned 6 through a
separate caller.
Parameterized runtime-local probes emitted 185-byte arithmetic and 153-byte
Boolean COFF objects. Independent calls returned 31 for input 10 and exercised
all true/false Boolean-local paths successfully.
The upstream shift pattern `value >> distance as usize` emitted a 139-byte COFF
object and returned 10 for `(160u8, 4u8)` through a separate caller.
Preserved literal suffixes also drive signed unary semantics: the upstream
platform-bitwise pattern `-1000isize as usize >> 3usize` emitted a 113-byte COFF
object and returned 2,305,843,009,213,693,827.
The mutable-local XOR-swap sequence selected from `bitwise-ops-platform.rs`
emitted a 93-byte COFF object and returned 3. A parameterized mutable-local
probe using `^=` and checked `+=` emitted a 157-byte object and returned 241 for
input 15 through an independently nightly-compiled caller.
Replacement probes now also map the mutation forms from `i8-incr.rs`,
`u8-incr.rs`, `u8-incr-decr.rs`, `u32-decr.rs`, and `i32-sub.rs`, plus the
inferred binding form exercised by `short-circuit-let.rs`. An inferred `i32`
mutation object was 255 bytes and returned 42 through an independently
nightly-compiled Windows caller.
Runtime `if` probes emitted 212-byte absolute-difference and 165-byte lazy-choice
COFF objects. An independent caller exercised both comparison directions,
skipped an untaken division by zero, evaluated the ordinary division branch,
and exited normally.
The explicit `usize` tail-return probe emitted a 105-byte COFF object and
preserved both 42 and `usize::MAX` through an independently compiled caller.
A conditional-return probe derived from the base-case shape in
`consts/control-flow/basics.rs` emitted a 190-byte COFF object. Its independently
compiled caller observed 41 through the early path and 42 through the fallthrough
path, including cleanup of one saved local and both parameters.
A scalar counter loop emitted a 172-byte COFF object. Its independent caller
validated zero iterations, one iteration, and one million checked increments.
A two-local summation loop with ordered `sum += i; i += 1;` mutations emitted a
209-byte COFF object and returned 0, 45, and 499,500 for limits 0, 10, and 1,000.
A const-qualified C-ABI addition function emitted a 122-byte COFF object and
returned 42 through an independently compiled caller.
A conditional-break counter emitted a 224-byte COFF object. Independent calls
proved zero-entry exit, natural exit at 5, and break exit at 10 despite a limit
of 100.
Explicit-continue, immediate-loop-break, and conditional-loop-break probes
emitted 200-byte, 114-byte, and 190-byte COFF objects. Their independent caller
proved zero and one-million iteration condition rechecking, immediate return of
42, and loop termination at 10.
An ordered interleaved-control probe emitted a 337-byte COFF object. Its caller
covered zero entry, odd and even exits at 21 and 22, a taken continue that skips
the later mutation, and one million iterations.
Immediate integer and Boolean break-value expressions emitted 144-byte and
126-byte COFF objects. The independent caller observed integer results 1 and 42
and both Boolean values through the native ABI.
A competing-break expression emitted a 202-byte COFF object. Its caller selected
the first value with a zero input without evaluating a trapping division on the
fallback edge, then selected the fallback value and returned 42.
A unit-returning loop function with no written tail expression emitted a
106-byte COFF object and returned normally through an independently compiled
native void call.
Boolean equality, Boolean inequality, and unit equality emitted 127-byte,
131-byte, and 109-byte COFF objects. An independent caller exercised equal and
unequal truth-table rows and observed unit equality as true.
Boolean ordering, `&`, `|`, and `^` emitted 126-byte, 122-byte, 121-byte, and
122-byte COFF objects. The independent caller passed representative true and
false rows for each operation; AST and IR adversarial tests separately prove
bitwise operands are evaluated eagerly.
A chained Boolean compound-assignment function and a loop-parity mutation
emitted 200-byte and 231-byte COFF objects. Independent calls exercised all four
input rows plus zero, one, and one million loop iterations.
The Boolean-constant replacement emitted a 145-byte COFF object. A separately
compiled `no_std` caller exercised both input rows through named false and true
compound constants and exited normally.
The cast/conditional slice emitted 165-byte signed and 129-byte Boolean COFF
objects. An independent `no_std` caller observed 42 from the sign-extended
selected branch and passed both Boolean input rows.
The compound-comparison replacement emitted a 157-byte COFF object. Its
independent `no_std` caller exercised both Boolean input rows after signed and
narrow unsigned compile-time comparisons.
The first const-call replacement emitted a 127-byte COFF object. An independent
`no_std` caller observed the called `add(40, 2)` result through two runtime
inputs.
The signed const-call replacement emitted a 169-byte COFF object; its caller
observed 42 and -42 from a constant produced by `adjust(-8, 50)`.
The nested const-call replacement emitted a 149-byte COFF object. Its
independently compiled `no_std` caller observed 22 and 42 and exited normally.
The composed const-call replacement emitted a 152-byte COFF object. Its
independent `no_std` caller observed 42 and 43 after a call comparison and
checked call-result multiplication selected at compile time.
The Boolean const-call replacement emitted a 151-byte COFF object. Its
independent `no_std` caller observed 42 and 43 after Boolean calls selected an
integer result at compile time.
The recursive GCD replacement emitted a 152-byte COFF object. Its independent
`no_std` caller observed 6 and 42 after bounded compile-time recursion.
The statement-bodied const-call replacement emitted a 172-byte COFF object.
Its independent `no_std` caller observed 42 and -42. The debug driver compiled
the same source successfully with an inspected 8,388,608-byte PE stack reserve.
The mutable-local const-call replacement emitted a 150-byte COFF object. Its
independent caller observed 42 and 43 after three checked compile-time
assignments.
The const-loop replacement emitted a 147-byte COFF object. Its independent
caller observed 42 and 43 after compile-time mutation, continue, and break.
The conditional-return const-loop replacement, derived from the scalar prime
shape in `consts/control-flow/basics.rs`, emitted a 154-byte COFF object. Its
independent caller observed 42 and 43; the unchanged pinned rustc test also
compiled and ran successfully.
The corresponding runtime prime emitted a 433-byte COFF object. Its independent
`no_std` caller exercised a loop return of true for 113 and an earlier loop
return of false for 117, including complete parameter/local frame cleanup.
An unconditional runtime loop-return probe with no tail expression emitted a
146-byte COFF object and returned 42 from inside the loop through a separately
compiled caller. A const variant maps the same form back to the pinned
explicit-return oracle without inventing an unreachable fallback value.
An original black-box source containing both const and runtime tail-free loop
returns compiled and ran with the pinned nightly before this behavior was
claimed for MRML.
The first ordered-statement const replacement executes a loop before a final
`*=` mutation. It emitted a 150-byte COFF object whose independent caller
observed 42 and 43. An original pinned-nightly observation independently
confirmed that the post-loop mutation produces 42 rather than the reordered 21.
The runtime form emitted a 266-byte COFF object. Its independent caller observed
zero for zero iterations and 42 for 21 iterations, directly proving the final
mutation executes after the loop.
A post-loop conditional-return replacement followed by another assignment
emitted a 308-byte COFF object. Its caller observed fallthrough value 4 and
returned value 7 after 42 iterations, covering `loop → condition → return or
assignment` order. The matching original black-box source passed both
assertions with the pinned nightly before MRML coverage was claimed.
A consecutive-loop replacement maps the two-`while` shape in the unchanged
pinned `tests/ui/for-loop-while/while.rs` oracle. It emitted a 296-byte COFF
object; an independent caller observed 42 with increasing bounds, 42 when the
second loop began past its bound, and zero when both loops had zero iterations.
The loop-local replacement maps the per-iteration binding in unchanged pinned
`tests/ui/for-loop-while/long-while.rs`. That oracle compiled and ran with the
exact dated nightly on both hosts. MRML's 425-byte COFF object passed an
independent nightly-built caller across zero and one iteration, conditional
continue, conditional break, ordinary termination, and 60,000 iterations. Its
discarded expression consumes the scoped binding before each control edge;
post-loop lookup of that binding is rejected, and runtime-local capacity
exhaustion fails before emission.
The conditional-loop-block replacement maps the local/action/break order in
unchanged pinned `tests/ui/for-loop-while/while-with-break.rs`; vectors,
allocation, destructors, printing, and drop elaboration are not claimed. The
oracle compiled and ran with the exact dated nightly on both hosts. MRML's
525-byte COFF object passed an independent nightly-built caller through false
conditions, a taken action-bearing continue, a taken action-bearing break,
ordinary completion, and 60,000 iterations. Unit and const coverage additionally
exercise action-bearing conditional returns and reject a fifth prefix action.
An action-only conditional-loop replacement maps the guarded mutation and
subsequent fallthrough order in pinned
`tests/ui/for-loop-while/loop-no-reinit-needed-post-bot.rs`; moves, destructors,
user-defined calls, and bottom-typed expressions are not claimed. Its native
probe emitted 377-byte COFF and 768-byte ELF64 objects. Independent
nightly-built callers exercise selected and unselected arms, scoped locals,
post-arm fallthrough, and 60,000 iterations on both supported hosts. The exact
unchanged pinned oracle also compiled and ran successfully on both hosts.
A loop `if`/`else` replacement maps the two-arm ordering from the same pinned
oracle. MRML's 440-byte COFF and 832-byte ELF64 objects passed independent
nightly-built callers across true and false arms, scoped locals, post-arm
fallthrough, and 60,000 iterations. Const evaluation covers both arms, while a
capacity regression rejects a fifth else arm.
A chained-alternative replacement extends that pinned ordering with an original
three-way scalar probe. Its 534-byte COFF and 928-byte ELF64 objects passed
independent nightly-built callers across first-, second-, and fallback-arm
selection, zero iterations, and 60,000 iterations. Conditions are evaluated in
source order and later arms remain lazy after the first match.
An alternative-terminal replacement maps conditional control ordering in
pinned `tests/ui/for-loop-while/loop-break-cont.rs` and the alternative
continue edge in `loop-no-reinit-needed-post-bot.rs`. Its 576-byte COFF and
968-byte ELF64 objects passed independent nightly-built callers through a
primary break, else-if continue, later else-if break, fallback return, zero
iterations, and 60,000 continues. Const evaluation exercises the same edges.
The immediate nested-unit-loop replacement maps unchanged pinned
`tests/ui/for-loop-while/nested-loop-break-unit.rs` and the immediate exit in
`loop-break-cont-1.rs`. The exact oracle compiled and ran on both hosts. MRML's
193-byte COFF and 584-byte ELF64 objects passed independent nightly-built
callers at zero, one, 42, and 60,000 outer iterations, proving the inner break
does not target the outer exit.
A scoped inner-action extension emitted 334-byte COFF and 728-byte ELF64
objects. Independent nightly-built callers observed triangular sums at zero,
one, five, and 60,000 outer iterations, proving inner locals are cleaned after
each break while enclosing accumulator mutations persist. Const evaluation and
capacity regressions cover the same scope and fixed-arena boundaries.
A repeated-inner-loop extension emitted 373-byte COFF and 768-byte ELF64
objects. Independent nightly-built callers exercised three inner iterations per
outer iteration at outer counts zero, one, five, and 60,000. The conditional
break reads the scoped selected value, while each false backedge and true exit
cleans that slot without discarding enclosing mutations. Const evaluation
produced the same zero, six, and thirty results for its bounded cases.
The inner-continue replacement maps nested continue/break targeting from pinned
`tests/ui/for-loop-while/loop-break-cont.rs`. Its 454-byte COFF and 848-byte
ELF64 objects passed independent nightly-built callers through even-value
continues, odd-value fallthrough, later conditional break, zero iterations, and
300,000 inner iterations. Const evaluation produced zero, nine, and forty-five
for the corresponding bounded cases.
The positioned inner-break replacement maps ordered nested break edges from
pinned `tests/ui/for-loop-while/break.rs` and `loop-break-cont.rs`. Its 442-byte
COFF and 832-byte ELF64 objects passed independent nightly-built callers through
early, middle, late, natural, zero-iteration, and 60,000-outer-iteration paths.
Const evaluation produced zero, three, six, and fifteen for the bounded cases.
The unconditional inner-continue replacement maps the nested backedge from
pinned `tests/ui/for-loop-while/loop-break-cont.rs`. Its 319-byte COFF and
712-byte ELF64 objects passed independent nightly-built callers at zero, one,
five, and 60,000 outer iterations while rechecking the inner `while` head and
cleaning a scoped local before every backedge. Const evaluation produced the
same bounded results.
The ordered-unreachable-control extension retains a local, discarded division
by zero, and return after that taken continue. Its 385-byte COFF and 776-byte
ELF64 objects passed the same independent callers without evaluating the
unreachable division, while a separate invalid Boolean local after the edge was
still rejected during native type checking. Const evaluation likewise skipped
the unreachable overflow and return; parser regressions cover the four-control
capacity boundary.
The conditional-only inner-loop replacement maps the backedge/return shape from
pinned `tests/ui/for-loop-while/loop-break-cont.rs`. Its 270-byte COFF and
664-byte ELF64 objects passed independent nightly-built callers through false
outer entry, immediate return, two conditional continues, and 60,000 inner
iterations. Const evaluation independently covered false entry and one- and
200-iteration returns.
The diverging-inner-loop replacement maps empty and action-only loop bodies from
pinned `tests/ui/for-loop-while/loop-break-cont.rs`. Its 125-byte COFF and
520-byte ELF64 objects passed independent nightly-built callers through the
false enclosing condition, returning 42. Machine-code inspection confirms a
signed backward jump, and const evaluation reports the exact loop-limit error
for both entered shapes instead of allowing fallthrough.
The top-level empty-loop replacement extends the same pinned oracle mapping to
empty `while`, empty `loop`, and action-only `loop` statements. Its 120-byte
COFF and 512-byte ELF64 empty-while objects passed independent nightly-built
callers through the false head and returned 42. Both object backends contain
checked negative backedges for the diverging forms, and const evaluation stops
them at the exact shared iteration limit.
The same-loop-label replacement maps labels from pinned
`tests/ui/for-loop-while/loop-break-value.rs` and `loop-break-cont.rs`. Its
379-byte COFF and 776-byte ELF64 objects passed independent nightly-built
callers through labeled conditional continue, labeled else-if break, a labeled
immediate loop exit, zero iterations, and two 60,000-iteration paths. Const
evaluation produced one and nine for early and complete bounded cases; parser
regressions reject a missing colon and unknown direct or conditional labels.
The nested-while replacement maps nested scalar while flow from pinned
`tests/ui/for-loop-while/while.rs` and the nested targeting observed in
`nested-loop-break-unit.rs`. Its 361-byte COFF and 752-byte ELF64 objects passed
independent nightly-built callers at zero, one, five, and 60,000 outer
iterations, with three condition-headed inner iterations each. Const evaluation
produced matching zero, six, and thirty results.
A combined nested-while-control replacement emitted 489-byte COFF and 880-byte
ELF64 objects. Independent nightly-built callers covered immediate head exit,
natural head exhaustion, taken and false conditional continue, later
conditional break, zero outer iterations, and 60,000 outer repetitions. The
const evaluator produced matching zero, four, nine, and forty-five path results.
The unconditional nested-while-break regression maps pinned `break.rs` and
`nested-loop-break-unit.rs`. Its 211-byte COFF and 608-byte ELF64 objects passed
independent nightly-built callers with false and true heads through 60,000 outer
iterations. Const coverage independently distinguishes false-head exit from a
true-head unconditional break.
The nested-while-return replacement maps nested cleanup and return control from
pinned `loop-no-reinit-needed-post-bot.rs` and the existing const-control-flow
oracle family. Its 269-byte COFF and 664-byte ELF64 objects passed independent
nightly-built callers through false-head fallthrough, true-head typed return,
zero outer iterations, and 60,000 fallthrough iterations. Const evaluation
returned five and forty on the corresponding paths.
The valueless nested-while-return replacement maps the unit-return shape from
pinned `tests/ui/codegen/issue-88043-bb-does-not-have-terminator.rs`. Its
300-byte COFF and 696-byte ELF64 objects passed independent nightly-built
callers through zero-iteration, false-head, entered-return, and 60,000-iteration
paths.
The conditional valueless-return extension maps that pinned test's guarded
return shape without calls or heap values. It emitted a 344-byte COFF object
and a 736-byte ELF64 object. A pinned-nightly Linux caller exercised immediate
outer exit, returns on the first and third inner iterations, and 60,000 natural
inner-loop exits.
The typed conditional nested-return replacement maps guarded nested cleanup in
pinned `tests/ui/liveness/loop-no-reinit-needed-post-bot.rs`. Its 388-byte COFF
and 784-byte ELF64 objects passed pinned-nightly callers through zero outer
iterations, returns on the first and third inner iterations, and 60,000 natural
inner-loop exits. The Windows oracle was a separate `no_std` object linked with
bundled `rust-lld`, the generated object, and no CRT or SDK libraries.
The ordered nested-return extension emitted 470-byte COFF and 864-byte ELF64
objects. Pinned-nightly native callers proved zero-iteration fallthrough,
first- and second-guard selection, source-order priority for equal guards, and
60,000 natural inner-loop exits. The Windows caller was again `no_std` and
linked without CRT or SDK libraries.
The nested continue-before-return regression emitted 429-byte COFF and 824-byte
ELF64 objects. Pinned-nightly callers proved that an earlier taken continue
suppresses a simultaneously true later return, while false and differently
positioned continue guards preserve the later return. Zero and 60,000-iteration
fallthrough paths also passed; the Windows caller used no CRT or SDK libraries.
The multiple-continue extension emitted 442-byte COFF and 832-byte ELF64
objects. Pinned-nightly callers distinguished first-only, second-only, and
duplicate continue guards and passed a 60,000-outer-iteration path. Const
evaluation independently produced matching 12, 13, 5, and 40 results across
multiple continues and ordered continue/return edges. A fifth continue has a
dedicated pre-lowering capacity diagnostic.
A post-loop local-binding replacement emitted a 291-byte COFF object. Its
independent caller observed 4 on the zero-iteration path and 42 after 19
iterations, proving the initializer reads the loop's final value instead of a
hoisted value.
A guarded local-binding replacement maps the statement order in the pinned
`tests/ui/liveness/var-defined-after-early-return.rs` run-pass test and the
scalar const shape in `tests/ui/consts/control-flow/basics.rs`. It emitted a
226-byte COFF object. Its independent caller returned 7 before an otherwise
overflowing initializer and returned 42 through the ordinary fallthrough path.
A guarded-mutation replacement emitted a 241-byte COFF object. Its independent
caller returned 7 without executing an overflowing `+=` and returned 42 after
the ordinary mutation path.
The guarded iterative-Fibonacci replacement emitted a 374-byte COFF object.
Its independent caller observed 0, 1, 55, and 6,765 for inputs 0, 1, 10, and
20, covering the early return, zero-iteration fallthrough, and dependent
three-local loop mutations.
An alternating two-guard replacement emitted a 289-byte COFF object. Its
independent caller observed the first return, the second return after one
mutation, and the final value 42 after both mutations. Supplying `u32::MAX` to
the first-return path proves the skipped mutation cannot be hoisted.
A post-loop alternating-guard replacement emitted a 458-byte COFF object. Its
independent caller covered three loop iterations followed by the first return,
the second return, or final value 42, plus a zero-iteration fallthrough. The
first path again used `u32::MAX` to prove the post-loop mutation remains lazy.
A scalar-block replacement emitted a 206-byte COFF object. Its independent
caller observed 42 through both selected branches and passed a zero divisor on
the untaken division edge. Unit and nested value blocks also compile through
both object writers.
An original scoped-block oracle then passed under the pinned nightly with a
runtime operand, typed mutable local, nested shadowing, and explicit tails.
MRML's runtime replacement emitted a 244-byte COFF object and 632-byte ELF
object. Independent callers observed the same selected, fallback, and
zero-divisor-skipping paths, while a separate regression proves inner names do
not escape their closing brace.
A separate scalar-match oracle passed under the same pinned nightly with a
named constant pattern, a signed literal pattern, a wildcard fallback, and an
omitted comma after a block arm. MRML emitted a 256-byte COFF object and a
648-byte ELF object for the runtime replacement. Independent callers observed
42 through both the selected and fallback arms; the selected input would make
the unselected division denominator zero, directly exercising lazy arm
selection on both hosts.
The follow-on ordered-match oracle exercised three distinct patterns plus the
wildcard and returned 42 through every path under the pinned nightly. MRML's
corresponding 351-byte COFF and 744-byte ELF objects passed independent callers
on all four paths. The first zero-valued pattern also skipped a wildcard
division by zero, while middle-arm execution proved earlier nonmatches do not
select or evaluate their bodies.
The unchanged pinned `tests/ui/binding/match-range.rs`, `match-range-static.rs`,
and `tests/ui/consts/const-eval/const_signed_pat.rs` then passed with the
reference nightly. MRML replacements cover their inclusive and exclusive
integer boundaries, an exact arm before a range, signed bounds, and named
constant bounds. Character coverage is recorded separately below; floating-point
ranges are not claimed. An
original seven-path oracle passed the same nightly. Its 390-byte COFF and
784-byte ELF replacements passed independent callers at both inclusive bounds,
both exclusive interior edges, the exclusive end fallback, an exact arm, and a
zero input that skips a dividing fallback.
The unchanged pinned
`tests/ui/match/overeager-sub-match-pruning-13027.rs` and
`tests/ui/half-open-range-patterns/range_pat_interactions0.rs` then ran
successfully under the reference nightly. Original MRML replacements map their
scalar exact-or-range alternatives and overlap ordering; bindings and
collections are not claimed. A nine-path oracle
passed the same nightly, and MRML's 470-byte COFF and 864-byte ELF objects passed
independent callers through every exact and range alternative plus the
wildcard. The zero alternative skipped a dividing fallback on both hosts.
The open-ended subset from `range_pat_interactions0.rs` now has direct MRML
coverage for exclusive and inclusive upper bounds, lower-only bounds, and an
open range joined with an exact alternative. An original eight-path oracle
passed the pinned nightly. MRML's 659-byte COFF and 1,056-byte ELF objects passed
independent callers across each boundary and the wildcard fallback on both
hosts.
Scalar guards from `overeager-sub-match-pruning-13027.rs` now have original
MRML coverage over exact, range, and alternative patterns. A companion oracle
passed the pinned nightly with true guards, false-guard fallthrough, overlapping
later arms, and a mismatched pattern whose guard would divide by zero. MRML's
635-byte COFF and 1,032-byte ELF objects passed all five paths through
independent callers on both hosts. A non-Boolean guard is rejected during type
checking.
The character-range case in the unchanged pinned `match-range.rs` oracle now
has a distinct MRML `char` implementation. An original oracle extended it with
ASCII boundaries, standard escapes, Unicode-scalar escapes, alternatives, and
a fallback. MRML's 362-byte COFF and 752-byte ELF objects passed all six paths
through independently nightly-compiled callers on both hosts. The caller
explicitly acknowledges rustc's `improper_ctypes` warning for `char` in a C ABI;
MRML nevertheless keeps `char` type-distinct from `u32` and rejects mixed
comparisons.
The unchanged pinned `tests/ui/match/guards.rs` then compiled and ran on both
hosts. Its scalar whole-scrutinee binding case has an original three-path MRML
replacement with two successive guarded bindings and a wildcard fallback. The
first accepted arm, false-guard fallthrough into the second binding, and final
fallback passed independently compiled callers against MRML's 302-byte COFF
and 704-byte ELF objects. A binding is scoped to its own guard and body, while
an attempted binding alternative is rejected rather than given inconsistent
Rust semantics. Struct destructuring from the pinned test is not claimed.
The unchanged pinned `guard-pattern-ordering-14865.rs` and
`guard-arm-and-or-arm.rs` tests also compiled and ran on both hosts. Their
guarded-wildcard ordering has an original four-path scalar replacement: a true
leading guard, false-guard fallthrough to an exact arm, a later guarded
wildcard, and the final fallback. MRML's 304-byte COFF and 704-byte ELF objects
passed independent callers on all paths. The replacement additionally checks
short-circuiting inside a wildcard guard and rejects a non-Boolean guard. Enum
construction, enum destructuring, and enum alternatives from the pinned tests
are not claimed.
The unchanged pinned `range-arm-and-ref-arm.rs` and
`binding/match-pattern-bindings.rs` tests then compiled and ran on both hosts.
MRML maps their scalar `name @ range` and `name @ _` behavior with an original
five-path replacement covering an accepted guarded range, false-guard
fallthrough, an exact binding, a guarded wildcard binding, and the final
fallback. Its 358-byte COFF and 760-byte ELF objects passed independent callers
on every path. Exact and character-range `@` forms also reach both object
formats in the replacement suite. Reference binding modes, `Option` patterns,
string patterns, and `let` patterns from the pinned tests are not claimed.
The pinned `or-patterns/bindings-runpass-2.rs` test then compiled and ran on
both hosts, while pinned `missing-bindings.rs` produced its expected E0408
diagnostic under the specified 2018 edition. An original six-path scalar oracle
covers identical exact bindings, identical range bindings, a guard accepted at
each end of separated ranges, false-guard fallthrough, and the final fallback.
MRML's 457-byte COFF and 856-byte ELF objects passed independent callers on
both hosts. Original negative replacements reject both a missing binding and
differently named bindings with `InconsistentMatchBindings`; nested aggregate
and parenthesized or-patterns from the pinned tests are not claimed.
The pinned `error-codes/E0030.rs` and `match/match-range-fail-2.rs` tests then
produced their expected E0030 diagnostics on both hosts. Original MRML negative
replacements reject reversed unsigned, reversed signed, reversed character,
and equal exclusive literal ranges. A three-path boundary oracle proves an
equal inclusive range selects its arm, a one-value exclusive interval selects
its arm, and the adjacent fallback remains reachable. MRML's 276-byte COFF and
680-byte ELF objects passed independent callers on both hosts. Validation of
target-width endpoint overflow is covered separately below.
The unchanged pinned `binding/match-range-static.rs` named-bound test then ran
on both hosts. Original MRML replacements accept equal inclusive named bounds
and reject reversed unsigned, reversed signed, equal exclusive, and Boolean
named bounds after resolution. A companion three-path named-bound oracle passed
independent callers against 276-byte COFF and 680-byte ELF objects on both
hosts. The validation metadata is fixed at thirty-two records and cannot allocate.
The pinned `match/validate-range-endpoints.rs` test then produced its expected
unsigned, signed, and E0030 diagnostics on both hosts. Original MRML negative
replacements reject overflowing `u8`, `i8`, `u64`, and 64-bit `usize`
endpoints, including negative and open-ended cases, with
`RangeEndpointOutOfRange`. Valid full-width `u8`, `i8`, and `usize` ranges reach
both object formats. A maximum-`u8` three-path oracle passed independent callers
against MRML's 187-byte COFF and 592-byte ELF objects on both hosts.
The unchanged pinned `or-patterns/inner-or-pat.rs` `or1` revision and
`or-patterns/or-patterns-syntactic-pass.rs` then compiled with the exact nightly
on Windows and compiled and ran under that nightly on Arch Linux WSL. MRML's
original scalar replacement recursively groups four exact/range alternatives,
accepts a leading `|`, applies a lazy Boolean guard, and proves both a later arm
and the final fallback. Its 402-byte COFF and 800-byte ELF objects passed
independent `no_std` callers on six paths. Empty and trailing-alternative groups,
a missing close delimiter, inconsistent nested bindings, and a fifth flattened
alternative fail closed. Aggregate patterns, strings, and grouped alternatives
inside aggregate subpatterns from the pinned tests are not claimed.
An original follow-up oracle exercises a scalar binding around recursively
grouped exact and range alternatives, a leading `|` plus wildcard group, guard
fallthrough, and binding use in both the guard and result. It passed the exact
pinned nightly on both hosts. MRML's 417-byte COFF and 816-byte ELF objects then
passed independent `no_std` callers on seven paths. A nested second `@` and a
fifth flattened bound alternative are rejected before lowering.
The unchanged pinned `match/large-match-mir-gen.rs` regression compiled with the
exact nightly on Windows and compiled and ran under that nightly on Arch Linux
WSL. MRML does not claim its string and enum surface; an original eight-arm
scalar replacement instead proves source order, two arms sharing a value with
guard fallthrough, selection of the eighth guarded arm, the final fallback, and
lazy earlier division bodies. Its 680-byte COFF and 1,080-byte ELF objects passed
independent `no_std` callers on eight paths. A ninth non-fallback arm produces
`TooManyMatchPatterns`, and the coupled range-validation arena now admits the
maximum eight arms times four alternatives while rejecting a thirty-third
range record.
The unchanged pinned `or-patterns/exhaustiveness-pass.rs` test compiled with the
exact nightly on Windows and compiled and ran on Arch Linux WSL; the unchanged
`exhaustiveness-non-exhaustive.rs` test produced E0004 on both hosts. Original
MRML replacements accept both source orders of exhaustive Boolean literals,
grouped `@` Boolean alternatives, and an irrefutable scalar binding without a
wildcard fallback. Missing Boolean variants, guarded-only coverage, and partial
integer literals fail with `NonExhaustiveMatch`. A two-path oracle passed the
exact nightly, and MRML's 151-byte COFF and 552-byte ELF objects passed
independent `no_std` callers for both Boolean values.
A closed inline-const replacement emitted a 277-byte COFF object. Its
independent caller observed 42 through both selected branches and passed zero
through an untaken division edge. The updated object resolves its offset from a
module constant inside the inline const boundary. The unchanged pinned
`tests/ui/consts/const-blocks/const-block-in-array-size.rs` test also compiled
and ran successfully under the reference nightly. Pinned
`tests/ui/inline-const/referencing-local-variables.rs` produced its expected
E0435 rejection; original MRML regressions reject both parameter and local
captures. The unchanged pinned
`tests/ui/consts/const-blocks/fn-call-in-const.rs` run-pass test also executed
successfully; original integer and Boolean replacements evaluate a prior scalar
const call inside the inline boundary before feeding its result to native code.
The same 277-byte COFF and 664-byte ELF objects now fold `offset()` directly
from the exported runtime body; their independent callers pass both branches
and the untaken zero-divisor case. A zero-parameter exported probe separately
exercises the IR call-load path, and non-const calls fail closed. The final
argument-bearing objects fold `add(1, 1)` directly; IR regression coverage
proves two operands retain source order. An original integer, Boolean, and
signed-argument oracle passed under the pinned nightly before this behavior was
claimed.
The unchanged pinned `tests/ui/inline-const/const-expr-basic.rs` run-pass test
also executed successfully. Its immutable `let` plus division shape has an
original MRML replacement, extended with two dependent bindings and a lazy
tail. Final compile-time folding restores the 277-byte COFF and 664-byte ELF
objects; an inline division by zero is rejected during compilation instead of
being deferred to a runtime trap.
An original mutable inline-const oracle then passed under the pinned nightly.
MRML replacements cover ordered integer assignment and arithmetic plus Boolean
bitwise mutation. Their final folded artifacts remain 277-byte COFF and
664-byte ELF objects and pass the same independent selected, fallback, and
untaken-division paths.
An original typed inline-const oracle then passed under that nightly on both
the Windows compile boundary and Linux execution boundary. Narrow integer and
Boolean annotations, plus mutation that retains the declared narrow type, have
original MRML replacements. The pinned nightly independently rejected both an
out-of-range `u8` initializer and an integer initializer ascribed as `bool`.
MRML reports the corresponding bounded compile-time errors. The resulting
typed replacement again emitted 277-byte COFF and 664-byte ELF objects; both
independent native callers passed all three branch and lazy-division cases.
An original interleaved-statement oracle then produced 42 through both later
declarations and scalar shadowing under the pinned nightly. The MRML
replacement folds alternating declarations and mutations in lexical order.
Its 277-byte COFF and 664-byte ELF objects passed the same independent native
callers without introducing runtime storage or allocation.
An original expression-statement oracle passed under the pinned nightly with a
discarded scalar const call and a lazy conditional statement. The MRML
replacement preserves the same evaluated/discarded sequencing and rejects an
executed division by zero before reaching the tail. Final folding again emitted
277-byte COFF and 664-byte ELF objects whose independent callers passed every
selected, fallback, and skipped-division path.
An original trailing-semicolon oracle then confirmed the pinned nightly returns
unit after declarations, mutations, and a discarded path statement. The
unchanged pinned `tests/ui/inline-const/expr-with-block.rs` check-pass test also
compiled and ran. Its accompanying cross-control-flow test produced the
expected errors for `return`, `break`, and `continue` attempting to escape the
inline boundary; original MRML adversarial probes fail closed for the same
spellings. The unit replacement emitted a 103-byte COFF object and 496-byte ELF
object. Independent callers linked and returned normally on both hosts.
An original unit-if oracle then passed under the pinned nightly with both an
untaken division and a taken discarded arithmetic statement. A companion
oracle produced E0317 for a value-valued branch without `else`. MRML preserves
the same acceptance boundary. The selected replacement emitted a 277-byte COFF
object and 664-byte ELF object; independent callers passed selected, fallback,
and skipped-division paths on both hosts.
An original scoped-branch oracle then produced 42 through an outer binding,
branch-local shadowing, and mutation while leaving a division-by-zero
initializer untaken. MRML reproduces that lexical scope and laziness. Final
folding emitted a 206-byte COFF object and 592-byte ELF object; both independent
callers passed all three runtime paths.
An original three-link `else if` oracle selected its third branch under the
pinned nightly while skipping division-by-zero expressions and an invalid
initializer in every other branch. MRML's nested replacement produced the same
42 and retained scoped locals. Its 206-byte COFF and 592-byte ELF objects passed
all independent caller paths.
Signed compound constants emitted 168-byte add and 130-byte mask COFF objects.
A separately compiled `no_std` caller linked them with bundled `rust-lld` and
observed `43 + (-4 + 3) == 42` and `-1 & (-8 | 3) == -5`. This native probe
initially exposed and then regression-tested the required narrow signed-constant
sign extension at the code-generation boundary.
An exhaustive-return replacement mapped to pinned
`tests/ui/expr/if/if-check.rs` now follows its `else if` shape. It emitted a
deterministic 345-byte COFF object and 736-byte ELF object. Independent `no_std`
callers selected the first, both middle, and final return paths on their native
hosts. The zero-valued first path would divide by zero if selection leaked into
either later division, proving ordered lazy execution. The linked Windows and
Linux processes both exited successfully.
A non-exhaustive chain mapped to the returning `else if` fallthrough in pinned
`tests/ui/expr/weird-exprs.rs` emitted a deterministic 262-byte COFF object and
656-byte ELF object. Independent `no_std` callers selected the first return,
the second return, and the ordinary tail. Passing zero to the first branch
would trap if either later division executed, proving that successful early
return prevents both later branch and tail evaluation on both hosts.
A guarded-mutation replacement mapped to pinned
`tests/ui/consts/const-eval/issue-64970.rs` emitted a deterministic 259-byte
COFF object and 656-byte ELF object. Independent `no_std` callers selected the
true assignment, false assignment, and a later conditional compound mutation.
The zero-valued true path would divide by zero if the false assignment executed;
all three paths returned the expected values on both native hosts.
A bounded mutation-chain replacement maps the ordered assignment behavior in
pinned `tests/ui/binding/if-let.rs` to original scalar Boolean conditions; it
does not claim `if let` or pattern support. Up to four guarded assignments and
an optional final `else` are represented without allocation. The unchanged
pinned test compiled on Windows and compiled and ran on Arch Linux WSL. MRML's
554-byte COFF and 944-byte ELF objects passed independent `no_std` callers on
the first, two middle, fallback, and two later fallthrough-mutation paths. The
zero-valued first path proves later division expressions remain unevaluated,
and a fifth guarded assignment is rejected with
`TooManyConditionalAssignmentBranches`.
Conditional assignment branches now contain up to four ordered scalar
assignments rather than exactly one. An original replacement maps this narrow
sequencing property to pinned `tests/ui/drop/dynamic-drop.rs`; allocation,
ownership, destructors, deferred initialization, tuples, unwinding, and
coroutines are explicitly not claimed. The unchanged pinned editions 2015 and
2018 revisions compiled and ran on Windows, and edition 2015 compiled and ran
on Arch Linux WSL. MRML's 868-byte COFF and 1,264-byte ELF objects passed
independent `no_std` callers on six paths, covering ordered noncommutative
updates in first, middle, fallback, and later fallthrough blocks. The zero path
would divide by zero if an unselected later block executed.
The same branch storage now preserves a bounded source-ordered mixture of
assignments and discarded scalar expression statements. A fifth mixed action fails with
`TooManyConditionalBranchActions`. An additional original replacement maps
this narrow behavior to the unusual discarded block expressions in pinned
`tests/ui/expr/weird-exprs.rs`; the unchanged test compiled and ran with the
exact dated nightly on both hosts. MRML's 996-byte COFF and 1,392-byte ELF
objects passed independent `no_std` callers on six paths. The zero path has no
dangerous assignment, but would divide by zero if a later branch's discarded
expression leaked across selection. A selected discarded division-by-zero
expression is separately rejected during const evaluation, proving these
statements execute rather than disappear.
Value-return actions now share that ordered branch stream. Native lowering
checks each return against the declared function type and emits the normal
frame epilogue; const evaluation immediately propagates the selected value.
An original scalar replacement maps the expression/mutation/return order in
pinned `tests/ui/structs-enums/class-impl-very-parameterized-trait.rs`; structs,
methods, generics, formatting, and ownership are not claimed. The unchanged
pinned run-pass test compiled and ran with the exact dated nightly on both
hosts. MRML's 602-byte COFF and 992-byte ELF objects passed independent
`no_std` callers through first, middle, later-return, and ordinary fallthrough
paths. Passing zero would trap if either the unselected later expression or the
post-conditional tail executed. A Boolean return in an integer function is
rejected at native code generation.
Branch-local declarations now participate in that same action stream without
requiring an assignment action. Native lowering permits lexical shadowing,
addresses the newest binding, includes live branch slots in return cleanup, and
removes them before fallthrough; const evaluation applies the same scope
boundary while preserving mutations to enclosing locals. An original scalar
replacement maps the local-before-return order in pinned
`tests/ui/drop/drop-on-ret.rs`; strings, allocation, destructors, and drop
elaboration are explicitly not claimed. The unchanged pinned run-pass test
compiled and ran with the exact dated nightly on both hosts. MRML's 339-byte
COFF and 736-byte ELF objects passed independent nightly-built callers through
a shadowing early return and two ordinary fallthrough values. The selected
zero-valued call additionally proves the discarded division in the unselected
arm remains lazy.

Linux used native Arch Linux WSL2 `x86_64-unknown-linux-gnu` with the identical
nightly commit:

```bash
cargo test -p mrml-rustc -p mrml-rustc-driver --offline
cargo clippy -p mrml-rustc -p mrml-rustc-driver \
  --all-targets --no-deps --offline -- -D warnings
cargo run --release -p mrml-rustc-driver --bin mrml-rustc --offline -- \
  --emit elf64 --function answer answer.rs -o answer.o
$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld \
  -flavor gnu -shared -o libanswer.so answer.o
readelf -h -S -s answer.o
```

The 301 Linux library, conformance, rustc-nightly-replacement, and driver tests
passed. The driver emitted a 496-byte ELF64 relocatable object;
the bundled linker accepted it as shared-object input. `readelf` independently
reported five canonical sections, a global 11-byte `answer` function in `.text`,
and valid linked symbol/string tables.

The runtime-expression source emitted a 568-byte ELF64 object. Native nightly
linked that object into a Linux caller, which executed
`calculate(20, 24) == 42` successfully. This independently exercises System V
argument preservation and generated multiply, checked add, and divide paths.
Separate 600-byte checked and 520-byte wrapping narrow-integer ELF objects also
linked and executed successfully, returning 30 and zero for the corresponding
rustc-derived probes.
Unchecked narrow shift and typed Boolean comparison ELF objects were also linked
into native callers and executed successfully.
A 752-byte signed ELF artifact returned `-10` for the same negative arithmetic
probe, and a 528-byte wrapping artifact returned `i8::MIN` successfully.
The signed compound-constant slice emitted 560-byte add and 520-byte mask ELF
objects; an independent nightly-built Linux caller linked and executed both,
observing the same 42 and -5 results as Windows.
The Boolean-constant replacement emitted a 536-byte ELF object. Its independent
nightly-built caller exercised both false and true inputs successfully.
The cast/conditional slice emitted 552-byte signed and 520-byte Boolean ELF
objects. A separate nightly-built caller linked and executed both successfully.
The compound-comparison replacement emitted a 552-byte ELF object and passed
both rows through an independent nightly-built Linux caller.
The first const-call replacement emitted a 520-byte ELF object and produced 42
and 50 through an independent nightly-built caller.
The signed const-call replacement emitted a 560-byte ELF object and produced 42
and -42 through an independent nightly-built caller.
The nested const-call replacement emitted a 544-byte ELF object. Its
independent nightly-built caller observed 22 and 42 and exited normally.
The composed const-call replacement emitted a 544-byte ELF object and passed
the same 42 and 43 assertions through an independent native caller.
The Boolean const-call replacement emitted a 544-byte ELF object and passed
the same assertions through an independent native caller.
The recursive GCD replacement emitted a 544-byte ELF object and passed native
results 6 and 42.
The statement-bodied const-call replacement emitted a 560-byte ELF object and
passed results 42 and -42. The native Linux debug driver also compiled the same
source successfully.
The mutable-local replacement emitted a 544-byte ELF object and passed the same
42 and 43 native results.
The const-loop replacement emitted a 544-byte ELF object and passed the same
native results.
The conditional-return const-loop replacement emitted a 552-byte ELF object
and passed native results 42 and 43 through an independent nightly-built caller.
The runtime prime emitted an 824-byte ELF object and passed the same 113/117
Boolean return-path assertions through an independent System V caller.
The unconditional runtime loop-return probe emitted a 536-byte ELF object and
returned 42 through an independent nightly-built System V caller.
The ordered-statement const replacement emitted a 544-byte ELF object and
passed the same independent 42/43 assertions.
The runtime ordered-statement replacement emitted a 656-byte ELF object and
passed the same zero/42 sequencing assertions through a System V caller.
The post-loop return/assignment replacement emitted a 704-byte ELF object and
passed the same 4/7 path assertions through an independent System V caller.
The consecutive-loop replacement emitted a 688-byte ELF object and passed the
same increasing-bound, already-past, and zero-iteration assertions through an
independent System V caller. The unchanged pinned nightly oracle also compiled
and ran successfully in this verification pass.
The post-loop local-binding replacement emitted a 688-byte ELF object and
passed the same zero-iteration and iterated assertions through an independent
System V caller. An original scalar source passed the same assertions with the
pinned nightly before MRML coverage was claimed.
The guarded local-binding replacement emitted a 616-byte ELF object and passed
the same early-return trap-skipping and ordinary fallthrough assertions. The
original scalar oracle produced the same results with the pinned nightly.
The guarded-mutation replacement emitted a 632-byte ELF object and passed the
same trap-skipping and fallthrough assertions. Both its original scalar oracle
and the unchanged pinned `consts/control-flow/basics.rs` test passed under the
reference nightly.
The guarded iterative-Fibonacci replacement emitted a 768-byte ELF object and
passed the identical four results through an independent System V caller. Its
original const oracle also compiled, evaluated, and ran under the pinned
nightly.
The alternating-guard replacement emitted a 680-byte ELF object and passed the
same three control-flow paths through an independent System V caller. Its
compile-time oracle produced 1, 2, and 42 under the pinned nightly.
The post-loop alternating-guard replacement emitted an 840-byte ELF object and
passed the identical four branch/iteration cases. Its compile-time oracle
produced the same results under the pinned nightly.
The scalar-block replacement emitted a 592-byte ELF object and passed the same
lazy branch assertions. The original scalar oracle and unchanged pinned
`tests/ui/consts/const-block.rs` run-pass test both executed successfully;
MRML claims only their scalar integral and unit block forms.
The closed inline-const replacement emitted a 664-byte ELF object and passed
the same selected, fallback, and skipped-division assertions through an
independent nightly-built Linux caller.
The paired short-circuit ELF objects passed the same division-by-zero skip and
ordinary evaluation probes through a native Linux caller.
The literal-driven pair likewise linked into and passed a native Linux caller.
The immutable-static shift object also linked into a native caller and returned
160.
A seventh-argument System V function consumed its first stack argument and
returned 42 through a native caller.
The cast-distance shift emitted a 528-byte ELF64 object and returned 10 through
a native caller.
The suffix-aware negative cast/shift emitted a 504-byte ELF64 object and
returned the same platform-width value through that caller.
The typed-local div/remainder probe emitted a 496-byte ELF64 object and returned
6 through a native caller.
Parameterized runtime-local probes emitted 576-byte arithmetic and 544-byte
Boolean ELF64 objects and passed the identical native caller assertions.
The mutable-local XOR-swap and parameterized mutation probes emitted 496-byte
and 560-byte ELF64 objects. A native nightly-compiled Linux caller observed the
same results, 3 and 241.
The inferred `i32` mutation probe emitted a 648-byte ELF64 object and returned
42 through an independently nightly-compiled Linux caller.
The corresponding runtime `if` probes emitted 616-byte and 552-byte ELF64
objects and passed the identical four native branch assertions.
The tail-return probe emitted a 496-byte ELF64 object and preserved the same two
values through a native Linux caller.
The conditional-return probe emitted a 584-byte ELF64 object and passed both
native branch assertions with the System V frame layout.
The scalar counter loop emitted a 576-byte ELF64 object and passed the identical
zero, one, and million-iteration assertions.
The loop-local action probe emitted an 816-byte ELF64 object and passed the same
zero-, one-, break-, continue-, completion-, and 60,000-iteration assertions as
its Windows counterpart through an independent nightly-built System V caller.
The conditional-loop-block probe emitted a 920-byte ELF64 object and passed the
same conditional continue, conditional break, completion, and long-iteration
assertions through an independent nightly-built System V caller.
The ordered summation loop emitted a 608-byte ELF64 object and passed the same
three summation assertions.
The const-qualified addition function emitted a 512-byte ELF64 object and
returned 42 through a native Linux caller.
The conditional-break counter emitted a 616-byte ELF64 object and passed the
same three exit-path assertions.
The corresponding explicit-continue, immediate-loop-break, and
conditional-loop-break objects were 592, 504, and 576 bytes. A native
nightly-compiled caller passed the same zero, million-iteration, immediate-exit,
and conditional-exit assertions.
The ordered interleaved-control ELF64 object was 728 bytes and passed the same
zero, odd, even, skipped-mutation, and million-iteration assertions.
The corresponding integer and Boolean break-value ELF64 objects were 528 and
512 bytes and passed the identical independent caller assertions.
The competing-break ELF64 object was 592 bytes and passed the same lazy
division-by-zero skip and selected-fallback assertions.
The corresponding implicit-unit loop object was 496 bytes and returned normally
through the independent Linux void call.
The corresponding equality objects were 520, 528, and 496 bytes on ELF64 and
passed the identical native truth-table assertions.
The corresponding Boolean ordering and bitwise ELF64 objects were 520, 512,
512, and 512 bytes and passed the same native truth-table assertions.
The Boolean compound and loop-parity ELF64 objects were 584 and 624 bytes and
passed the identical truth-table and iteration assertions.

Security review searched both crates for direct `std`/`alloc` imports and unsafe
blocks and found none. `cargo tree -p mrml-rustc-driver --offline` showed only
original workspace crates. Adversarial tests cover truncated output, capacity
exhaustion, nesting limits, malformed delimiters and literals, integer and
offset overflow, invalid names, unresolved symbols, invalid shifts, division by
zero, duplicate definitions, and source-bounded diagnostics. The driver caps
source input at 1 MiB and refuses an already existing output path. No claim is
made that a compromised host protects compiler input or output.

There was no prior MRML compiler baseline. A release-mode one-function smoke
compile, including process startup and filesystem I/O, took 22.050 ms on the
Windows development host and 0.005 seconds under Arch Linux WSL2. These are
single observations, not throughput benchmarks or regression thresholds.
