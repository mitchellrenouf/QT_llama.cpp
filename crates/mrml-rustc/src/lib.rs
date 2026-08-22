#![no_std]
#![forbid(unsafe_code)]

//! The MRML-owned Rust compiler.
//!
//! This crate begins at the lexical compatibility boundary. It deliberately
//! contains no source copied from the upstream compiler or standard library;
//! its tests describe independently observed language behavior.

pub mod codegen;
pub mod expression;
pub mod ir;
pub mod lexer;
pub mod object;
pub mod parser;
pub mod pipeline;
pub mod semantics;

pub use codegen::{
    CodegenError, CodegenErrorKind, CodegenOptions, MachineCode, X86_64Abi,
    compile_x86_64_constant_function, compile_x86_64_function,
    compile_x86_64_function_with_options,
};
pub use expression::{
    BinaryOperator, ConstEvalError, ConstantResolver, Expr, ExprId, ExprKind, ExpressionError,
    ExpressionErrorKind, ExpressionParser, ExpressionTree, IntegerLiteral, IntegerType,
    NoConstants, ScalarType, UnaryOperator,
};
pub use ir::{
    ExecutionError, Instruction, IrError, IrErrorKind, IrProgram, lower_expression,
    lower_expression_with_pointer_bits,
};
pub use lexer::{LexError, LexErrorKind, Lexer, Span, Token, TokenKind};
pub use object::{ObjectError, ObjectFile, emit_elf64_x86_64, emit_x86_64_coff};
pub use parser::{
    Assignment, AssignmentOperator, BodyStatement, ConditionalAssignment,
    ConditionalAssignmentAction, ConditionalAssignmentBranch, ConditionalLoopAction,
    ConditionalLoopArm, ConditionalLoopBlock, ConditionalLoopControl, ConditionalLoopTerminal,
    ConditionalReturn, ConditionalReturnElse, ConstItem, ExpressionStatement, Function,
    FunctionAbi, FunctionBody, Item, LocalBinding, LoopOperation, LoopReturn,
    MAX_BODY_CONDITIONAL_ASSIGNMENTS, MAX_BODY_EXPRESSION_STATEMENTS, MAX_BODY_RETURNS,
    MAX_BODY_STATEMENTS, MAX_CONDITIONAL_ASSIGNMENT_BRANCHES, MAX_CONDITIONAL_BRANCH_ACTIONS,
    MAX_CONDITIONAL_LOOP_ACTIONS, MAX_CONDITIONAL_LOOP_ELSE_ARMS, MAX_CONDITIONAL_RETURN_BRANCHES,
    MAX_NESTED_LOOP_ACTIONS, MAX_NESTED_LOOP_BLOCKS, MAX_NESTED_LOOP_CONDITIONAL_BREAKS,
    MAX_NESTED_LOOP_CONDITIONAL_CONTINUES, MAX_NESTED_LOOP_CONDITIONAL_RETURNS, Module,
    NestedLoopBlock, Parameter, ParseError, ParseErrorKind, Parser, TypeRef, WhileLoop,
};
pub use pipeline::{
    CompileError, CompileErrorKind, ObjectFormat, compile_source_function,
    compile_source_function_with_options,
};
pub use semantics::{
    Constant, ConstantTable, MAX_CONST_CALL_DEPTH, MAX_CONST_FUNCTION_EXPRESSION_NODES,
    MAX_CONST_LOOP_ITERATIONS, MAX_CONSTANT_IR_INSTRUCTIONS, MAX_CONSTANT_STACK_VALUES,
    SemanticError, SemanticErrorKind, TargetLayout, analyze_constants,
};
