use crate::expression::{
    MAX_CALL_ARGUMENTS, ascribe_integer, cast_integer, evaluate_binary, negate_signed_magnitude,
};
use crate::{
    BinaryOperator, ConstEvalError, ConstantResolver, ExprId, ExprKind, ExpressionTree, Span,
    UnaryOperator,
};

const MAX_LOWERING_DEPTH: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction<'source> {
    PushInteger(u128),
    LoadConstant(&'source str),
    CallConstant(&'source str, usize),
    Unary(UnaryOperator),
    BooleanNot,
    SignedLiteralNegate(crate::IntegerType, u8),
    Cast(crate::IntegerType, u8),
    AscribeInteger(crate::IntegerType, u8),
    Binary(BinaryOperator),
    NormalizeBool,
    Pop,
    BranchIfFalse(usize),
    BranchIfTrue(usize),
    JumpIfFalse(usize),
    Jump(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IrErrorKind {
    MissingExpression,
    InvalidExpressionTree,
    TooManyInstructions,
    NestingLimitExceeded,
    UnsupportedCall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IrError {
    pub kind: IrErrorKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    UnknownConstant,
    StackOverflow,
    StackUnderflow,
    InvalidBranch,
    InvalidFinalStack,
    Arithmetic(ConstEvalError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IrProgram<'source, const MAX_INSTRUCTIONS: usize> {
    instructions: [Option<Instruction<'source>>; MAX_INSTRUCTIONS],
    instruction_count: usize,
}

impl<'source, const MAX_INSTRUCTIONS: usize> IrProgram<'source, MAX_INSTRUCTIONS> {
    pub fn instructions(&self) -> &[Option<Instruction<'source>>] {
        &self.instructions[..self.instruction_count]
    }

    pub const fn len(&self) -> usize {
        self.instruction_count
    }

    pub const fn is_empty(&self) -> bool {
        self.instruction_count == 0
    }

    pub fn execute<R: ConstantResolver, const MAX_STACK: usize>(
        &self,
        resolver: &R,
    ) -> Result<u128, ExecutionError> {
        let mut stack = [0u128; MAX_STACK];
        let mut stack_len = 0usize;
        let mut program_counter = 0usize;
        while program_counter < self.instruction_count {
            let instruction =
                self.instructions[program_counter].ok_or(ExecutionError::InvalidBranch)?;
            match instruction {
                Instruction::PushInteger(value) => push(&mut stack, &mut stack_len, value)?,
                Instruction::LoadConstant(name) => {
                    let value = resolver
                        .resolve(name)
                        .ok_or(ExecutionError::UnknownConstant)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::CallConstant(name, argument_count) => {
                    if argument_count > MAX_CALL_ARGUMENTS {
                        return Err(ExecutionError::InvalidFinalStack);
                    }
                    let mut arguments = [0u128; MAX_CALL_ARGUMENTS];
                    for index in (0..argument_count).rev() {
                        arguments[index] = pop(&stack, &mut stack_len)?;
                    }
                    let value = resolver
                        .resolve_call(name, &arguments[..argument_count])
                        .ok_or(ExecutionError::UnknownConstant)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::Unary(operator) => {
                    let value = pop(&stack, &mut stack_len)?;
                    let result = match operator {
                        UnaryOperator::Negate => 0u128
                            .checked_sub(value)
                            .ok_or(ExecutionError::Arithmetic(ConstEvalError::Overflow))?,
                        UnaryOperator::Not => !value,
                        UnaryOperator::Dereference
                        | UnaryOperator::AddressOf
                        | UnaryOperator::AddressOfMut => {
                            return Err(ExecutionError::Arithmetic(
                                ConstEvalError::InvalidExpressionTree,
                            ));
                        }
                    };
                    push(&mut stack, &mut stack_len, result)?;
                }
                Instruction::BooleanNot => {
                    let value = pop(&stack, &mut stack_len)?;
                    push(&mut stack, &mut stack_len, u128::from(value == 0))?;
                }
                Instruction::SignedLiteralNegate(target, pointer_bits) => {
                    let magnitude = pop(&stack, &mut stack_len)?;
                    let value = negate_signed_magnitude(magnitude, target, pointer_bits)
                        .map_err(ExecutionError::Arithmetic)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::Cast(target, pointer_bits) => {
                    let value = pop(&stack, &mut stack_len)?;
                    let value = cast_integer(value, target, pointer_bits)
                        .map_err(ExecutionError::Arithmetic)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::AscribeInteger(target, pointer_bits) => {
                    let value = pop(&stack, &mut stack_len)?;
                    let value = ascribe_integer(value, target, pointer_bits)
                        .map_err(ExecutionError::Arithmetic)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::Binary(operator) => {
                    let right = pop(&stack, &mut stack_len)?;
                    let left = pop(&stack, &mut stack_len)?;
                    let value = evaluate_binary(operator, left, right)
                        .map_err(ExecutionError::Arithmetic)?;
                    push(&mut stack, &mut stack_len, value)?;
                }
                Instruction::NormalizeBool => {
                    let value = pop(&stack, &mut stack_len)?;
                    push(&mut stack, &mut stack_len, u128::from(value != 0))?;
                }
                Instruction::Pop => {
                    pop(&stack, &mut stack_len)?;
                }
                Instruction::BranchIfFalse(target) => {
                    let value = pop(&stack, &mut stack_len)?;
                    if value == 0 {
                        validate_branch(program_counter, target, self.instruction_count)?;
                        push(&mut stack, &mut stack_len, 0)?;
                        program_counter = target;
                        continue;
                    }
                }
                Instruction::BranchIfTrue(target) => {
                    let value = pop(&stack, &mut stack_len)?;
                    if value != 0 {
                        validate_branch(program_counter, target, self.instruction_count)?;
                        push(&mut stack, &mut stack_len, 1)?;
                        program_counter = target;
                        continue;
                    }
                }
                Instruction::JumpIfFalse(target) => {
                    let value = pop(&stack, &mut stack_len)?;
                    if value == 0 {
                        validate_branch(program_counter, target, self.instruction_count)?;
                        program_counter = target;
                        continue;
                    }
                }
                Instruction::Jump(target) => {
                    validate_branch(program_counter, target, self.instruction_count)?;
                    program_counter = target;
                    continue;
                }
            }
            program_counter += 1;
        }
        if stack_len != 1 {
            return Err(ExecutionError::InvalidFinalStack);
        }
        Ok(stack[0])
    }
}

fn validate_branch(current: usize, target: usize, length: usize) -> Result<(), ExecutionError> {
    if target <= current || target > length {
        Err(ExecutionError::InvalidBranch)
    } else {
        Ok(())
    }
}

fn push<const MAX_STACK: usize>(
    stack: &mut [u128; MAX_STACK],
    length: &mut usize,
    value: u128,
) -> Result<(), ExecutionError> {
    let slot = stack
        .get_mut(*length)
        .ok_or(ExecutionError::StackOverflow)?;
    *slot = value;
    *length += 1;
    Ok(())
}

fn pop<const MAX_STACK: usize>(
    stack: &[u128; MAX_STACK],
    length: &mut usize,
) -> Result<u128, ExecutionError> {
    *length = length
        .checked_sub(1)
        .ok_or(ExecutionError::StackUnderflow)?;
    Ok(stack[*length])
}

struct Lowerer<'tree, 'source, const MAX_NODES: usize, const MAX_INSTRUCTIONS: usize> {
    tree: &'tree ExpressionTree<'source, MAX_NODES>,
    pointer_bits: u8,
    program: IrProgram<'source, MAX_INSTRUCTIONS>,
}

impl<'source, const MAX_NODES: usize, const MAX_INSTRUCTIONS: usize>
    Lowerer<'_, 'source, MAX_NODES, MAX_INSTRUCTIONS>
{
    fn push(&mut self, instruction: Instruction<'source>, span: Span) -> Result<usize, IrError> {
        if self.program.instruction_count == MAX_INSTRUCTIONS {
            return Err(IrError {
                kind: IrErrorKind::TooManyInstructions,
                span,
            });
        }
        let index = self.program.instruction_count;
        self.program.instructions[index] = Some(instruction);
        self.program.instruction_count += 1;
        Ok(index)
    }

    fn patch(&mut self, index: usize, instruction: Instruction<'source>) {
        self.program.instructions[index] = Some(instruction);
    }

    fn lower(&mut self, id: ExprId, depth: usize) -> Result<(), IrError> {
        let expression = self.tree.expression(id).ok_or(IrError {
            kind: IrErrorKind::MissingExpression,
            span: Span { start: 0, end: 0 },
        })?;
        if depth == MAX_LOWERING_DEPTH {
            return Err(IrError {
                kind: IrErrorKind::NestingLimitExceeded,
                span: expression.span,
            });
        }
        match expression.kind {
            ExprKind::Unit | ExprKind::DefaultValue => {
                self.push(Instruction::PushInteger(0), expression.span)?;
            }
            ExprKind::Array { .. } | ExprKind::ArrayRepeat { .. } => {
                return Err(IrError {
                    kind: IrErrorKind::InvalidExpressionTree,
                    span: expression.span,
                });
            }
            ExprKind::RangeIndex { .. }
            | ExprKind::SliceLen { .. }
            | ExprKind::SliceIsEmpty { .. }
            | ExprKind::StrAsBytes { .. }
            | ExprKind::StrIsCharBoundary { .. }
            | ExprKind::ReferenceAsPointer { .. }
            | ExprKind::RawPointerIsNull { .. }
            | ExprKind::RawPointerAddress { .. }
            | ExprKind::RawPointerWithAddress { .. }
            | ExprKind::RawPointerOffset { .. }
            | ExprKind::RawPointerDifference { .. } => {
                return Err(IrError {
                    kind: IrErrorKind::InvalidExpressionTree,
                    span: expression.span,
                });
            }
            ExprKind::Index { base, index } => {
                let index_expression = self.tree.expression(index).ok_or(IrError {
                    kind: IrErrorKind::MissingExpression,
                    span: expression.span,
                })?;
                let ExprKind::Integer(index_literal) = index_expression.kind else {
                    return Err(IrError {
                        kind: IrErrorKind::InvalidExpressionTree,
                        span: index_expression.span,
                    });
                };
                let index = usize::try_from(index_literal.value).map_err(|_| IrError {
                    kind: IrErrorKind::InvalidExpressionTree,
                    span: index_expression.span,
                })?;
                self.lower_array_element(base, index, depth + 1)?;
            }
            ExprKind::Integer(literal) => {
                self.push(Instruction::PushInteger(literal.value), expression.span)?;
            }
            ExprKind::Bool(value) => {
                self.push(Instruction::PushInteger(u128::from(value)), expression.span)?;
            }
            ExprKind::Char(value) => {
                self.push(Instruction::PushInteger(u128::from(value)), expression.span)?;
            }
            ExprKind::Identifier(name) => {
                self.push(Instruction::LoadConstant(name), expression.span)?;
            }
            ExprKind::Call {
                callee,
                arguments,
                argument_count,
            } => {
                for argument in arguments[..argument_count].iter() {
                    self.lower(
                        argument.ok_or(IrError {
                            kind: IrErrorKind::MissingExpression,
                            span: expression.span,
                        })?,
                        depth + 1,
                    )?;
                }
                self.push(
                    Instruction::CallConstant(callee, argument_count),
                    expression.span,
                )?;
            }
            ExprKind::Cast { operand, target } => {
                self.lower(operand, depth + 1)?;
                let crate::CastType::Integer(target) = target else {
                    return Err(IrError {
                        kind: IrErrorKind::InvalidExpressionTree,
                        span: expression.span,
                    });
                };
                self.push(
                    Instruction::Cast(target, self.pointer_bits),
                    expression.span,
                )?;
            }
            ExprKind::Ascribe { operand, target } => {
                self.lower(operand, depth + 1)?;
                match target {
                    crate::ScalarType::Integer(target) => {
                        self.push(
                            Instruction::AscribeInteger(target, self.pointer_bits),
                            expression.span,
                        )?;
                    }
                    crate::ScalarType::Bool => {
                        self.push(Instruction::NormalizeBool, expression.span)?;
                    }
                }
            }
            ExprKind::Unary { operator, operand } => {
                self.lower(operand, depth + 1)?;
                let signed_literal_target = self.tree.expression(operand).and_then(|operand| {
                    if let ExprKind::Integer(literal) = operand.kind {
                        literal
                            .suffix
                            .and_then(crate::IntegerType::from_name)
                            .filter(|target| target.is_signed())
                    } else {
                        None
                    }
                });
                let instruction = if let (UnaryOperator::Negate, Some(target)) =
                    (operator, signed_literal_target)
                {
                    Instruction::SignedLiteralNegate(target, self.pointer_bits)
                } else if operator == UnaryOperator::Not
                    && self.tree.is_boolean_expression(operand, depth + 1)
                {
                    Instruction::BooleanNot
                } else {
                    Instruction::Unary(operator)
                };
                self.push(instruction, expression.span)?;
            }
            ExprKind::Binary {
                operator,
                left,
                right,
            } if matches!(
                operator,
                BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr
            ) =>
            {
                self.lower(left, depth + 1)?;
                let branch = self.push(
                    if operator == BinaryOperator::LogicalAnd {
                        Instruction::BranchIfFalse(0)
                    } else {
                        Instruction::BranchIfTrue(0)
                    },
                    expression.span,
                )?;
                self.lower(right, depth + 1)?;
                self.push(Instruction::NormalizeBool, expression.span)?;
                let target = self.program.instruction_count;
                self.patch(
                    branch,
                    if operator == BinaryOperator::LogicalAnd {
                        Instruction::BranchIfFalse(target)
                    } else {
                        Instruction::BranchIfTrue(target)
                    },
                );
            }
            ExprKind::Binary {
                operator,
                left,
                right,
            } => {
                self.lower(left, depth + 1)?;
                self.lower(right, depth + 1)?;
                self.push(Instruction::Binary(operator), expression.span)?;
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            }
            | ExprKind::LoopBreakIf {
                condition,
                then_branch,
                else_branch,
            } => {
                self.lower(condition, depth + 1)?;
                let false_branch = self.push(Instruction::JumpIfFalse(0), expression.span)?;
                self.lower(then_branch, depth + 1)?;
                let end_branch = self.push(Instruction::Jump(0), expression.span)?;
                let else_target = self.program.instruction_count;
                self.patch(false_branch, Instruction::JumpIfFalse(else_target));
                self.lower(else_branch, depth + 1)?;
                let end_target = self.program.instruction_count;
                self.patch(end_branch, Instruction::Jump(end_target));
            }
            ExprKind::Return { operand } | ExprKind::LoopBreak { operand } => {
                self.lower(operand, depth + 1)?
            }
            ExprKind::InlineConst { operand } => self.lower(operand, depth + 1)?,
            ExprKind::Sequence { first, then } => {
                self.lower(first, depth + 1)?;
                self.push(Instruction::Pop, expression.span)?;
                self.lower(then, depth + 1)?;
            }
        }
        Ok(())
    }

    fn lower_array_element(
        &mut self,
        id: ExprId,
        index: usize,
        depth: usize,
    ) -> Result<(), IrError> {
        let expression = self.tree.expression(id).ok_or(IrError {
            kind: IrErrorKind::MissingExpression,
            span: Span { start: 0, end: 0 },
        })?;
        if depth == MAX_LOWERING_DEPTH {
            return Err(IrError {
                kind: IrErrorKind::NestingLimitExceeded,
                span: expression.span,
            });
        }
        match expression.kind {
            ExprKind::DefaultValue => {
                self.push(Instruction::PushInteger(0), expression.span)?;
            }
            ExprKind::Array {
                elements,
                element_count,
            } => {
                if index >= element_count {
                    return Err(IrError {
                        kind: IrErrorKind::InvalidExpressionTree,
                        span: expression.span,
                    });
                }
                for (element_index, element) in elements[..element_count].iter().enumerate() {
                    let element = element.ok_or(IrError {
                        kind: IrErrorKind::InvalidExpressionTree,
                        span: expression.span,
                    })?;
                    self.lower(element, depth + 1)?;
                    if element_index != index {
                        self.push(Instruction::Pop, expression.span)?;
                    }
                }
            }
            ExprKind::ArrayRepeat { element, count } => {
                if index >= count {
                    return Err(IrError {
                        kind: IrErrorKind::InvalidExpressionTree,
                        span: expression.span,
                    });
                }
                self.lower(element, depth + 1)?;
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            }
            | ExprKind::LoopBreakIf {
                condition,
                then_branch,
                else_branch,
            } => {
                self.lower(condition, depth + 1)?;
                let false_branch = self.push(Instruction::JumpIfFalse(0), expression.span)?;
                self.lower_array_element(then_branch, index, depth + 1)?;
                let end_branch = self.push(Instruction::Jump(0), expression.span)?;
                let else_start = self.program.instruction_count;
                self.patch(false_branch, Instruction::JumpIfFalse(else_start));
                self.lower_array_element(else_branch, index, depth + 1)?;
                let end = self.program.instruction_count;
                self.patch(end_branch, Instruction::Jump(end));
            }
            ExprKind::InlineConst { operand }
            | ExprKind::LoopBreak { operand }
            | ExprKind::Return { operand } => {
                self.lower_array_element(operand, index, depth + 1)?;
            }
            ExprKind::Sequence { first, then } => {
                self.lower(first, depth + 1)?;
                self.push(Instruction::Pop, expression.span)?;
                self.lower_array_element(then, index, depth + 1)?;
            }
            _ => {
                return Err(IrError {
                    kind: IrErrorKind::InvalidExpressionTree,
                    span: expression.span,
                });
            }
        }
        Ok(())
    }
}

pub fn lower_expression<'source, const MAX_INSTRUCTIONS: usize, const MAX_NODES: usize>(
    tree: &ExpressionTree<'source, MAX_NODES>,
) -> Result<IrProgram<'source, MAX_INSTRUCTIONS>, IrError> {
    lower_expression_with_pointer_bits(tree, 64)
}

pub fn lower_expression_with_pointer_bits<
    'source,
    const MAX_INSTRUCTIONS: usize,
    const MAX_NODES: usize,
>(
    tree: &ExpressionTree<'source, MAX_NODES>,
    pointer_bits: u8,
) -> Result<IrProgram<'source, MAX_INSTRUCTIONS>, IrError> {
    let mut lowerer = Lowerer {
        tree,
        pointer_bits,
        program: IrProgram {
            instructions: [None; MAX_INSTRUCTIONS],
            instruction_count: 0,
        },
    };
    lowerer.lower(tree.root(), 0)?;
    Ok(lowerer.program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpressionParser, NoConstants};

    fn execute(source: &str) -> Result<u128, ExecutionError> {
        let tree = ExpressionParser::<64>::new(source).parse().unwrap();
        let program = lower_expression::<96, 64>(&tree).unwrap();
        program.execute::<_, 32>(&NoConstants)
    }

    #[test]
    fn lowered_ir_matches_ast_evaluation() {
        for source in [
            "2 + 3 * 4",
            "(2 + 3) * 4",
            "8 >> 2 + 1",
            "7 & 3 == 3",
            "1 < 2 && 3 < 4",
            "0 || 9",
        ] {
            let tree = ExpressionParser::<64>::new(source).parse().unwrap();
            let expected = tree.evaluate(&NoConstants).unwrap();
            let program = lower_expression::<96, 64>(&tree).unwrap();
            assert_eq!(program.execute::<_, 32>(&NoConstants), Ok(expected));
        }
    }

    #[test]
    fn branches_preserve_short_circuit_behavior() {
        assert_eq!(execute("0 && UNKNOWN"), Ok(0));
        assert_eq!(execute("1 || UNKNOWN"), Ok(1));
        assert_eq!(
            execute("1 && UNKNOWN"),
            Err(ExecutionError::UnknownConstant)
        );
        assert_eq!(execute("false && UNKNOWN"), Ok(0));
        assert_eq!(execute("true || UNKNOWN"), Ok(1));
    }

    #[test]
    fn boolean_bitwise_ops_evaluate_both_operands() {
        assert_eq!(
            execute("false & UNKNOWN"),
            Err(ExecutionError::UnknownConstant)
        );
        assert_eq!(
            execute("true | UNKNOWN"),
            Err(ExecutionError::UnknownConstant)
        );
        assert_eq!(
            execute("false ^ UNKNOWN"),
            Err(ExecutionError::UnknownConstant)
        );
    }

    #[test]
    fn branches_preserve_lazy_if_expression_behavior() {
        assert_eq!(execute("if true { 42 } else { UNKNOWN }"), Ok(42));
        assert_eq!(execute("if false { UNKNOWN } else { 7 }"), Ok(7));
        assert_eq!(execute("if 3 > 2 { 8 } else { 9 }"), Ok(8));
    }

    #[test]
    fn array_indexes_evaluate_selected_arm_elements_in_source_order() {
        assert_eq!(
            execute("[1 / 0, 42][1]"),
            Err(ExecutionError::Arithmetic(ConstEvalError::DivisionByZero))
        );
        assert_eq!(
            execute("(if true { [13, 42] } else { [1 / 0, 0] })[1]"),
            Ok(42)
        );
    }

    #[test]
    fn sequence_instruction_discards_only_completed_statement_values() {
        assert_eq!(execute("const { 20 + 22; 7 }"), Ok(7));
        assert_eq!(
            execute("const { 1 / 0; 7 }"),
            Err(ExecutionError::Arithmetic(ConstEvalError::DivisionByZero))
        );
        let tree = ExpressionParser::<16>::new("const { 1; 2 }")
            .parse()
            .unwrap();
        let program = lower_expression::<16, 16>(&tree).unwrap();
        assert!(
            program
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction, Some(Instruction::Pop)))
        );
    }

    #[test]
    fn constant_call_instruction_preserves_argument_order() {
        struct Resolver;
        impl ConstantResolver for Resolver {
            fn resolve(&self, _: &str) -> Option<u128> {
                None
            }

            fn resolve_call(&self, name: &str, arguments: &[u128]) -> Option<u128> {
                (name == "combine" && arguments == [4, 2]).then_some(42)
            }
        }
        let tree = ExpressionParser::<16>::new("const { combine(4, 2) }")
            .parse()
            .unwrap();
        let program = lower_expression::<16, 16>(&tree).unwrap();
        assert_eq!(program.execute::<_, 8>(&Resolver), Ok(42));
        assert!(matches!(
            program.instructions()[2],
            Some(Instruction::CallConstant("combine", 2))
        ));
    }

    #[test]
    fn cast_instruction_uses_explicit_target_pointer_width() {
        let tree = ExpressionParser::<8>::new("0xffff_ffff_ffff_ffffu64 as usize")
            .parse()
            .unwrap();
        let program = lower_expression_with_pointer_bits::<8, 8>(&tree, 32).unwrap();
        assert_eq!(
            program.execute::<_, 4>(&NoConstants),
            Ok(u128::from(u32::MAX))
        );
        assert!(matches!(
            program.instructions()[1],
            Some(Instruction::Cast(crate::IntegerType::Usize, 32))
        ));
    }

    #[test]
    fn lowering_enforces_instruction_capacity() {
        let tree = ExpressionParser::<16>::new("1 + 2 * 3").parse().unwrap();
        let error = lower_expression::<4, 16>(&tree).unwrap_err();
        assert_eq!(error.kind, IrErrorKind::TooManyInstructions);
    }

    #[test]
    fn execution_enforces_stack_capacity_and_arithmetic_checks() {
        let tree = ExpressionParser::<16>::new("1 + 2").parse().unwrap();
        let program = lower_expression::<8, 16>(&tree).unwrap();
        assert_eq!(
            program.execute::<_, 1>(&NoConstants),
            Err(ExecutionError::StackOverflow)
        );
        assert_eq!(
            execute("1 / 0"),
            Err(ExecutionError::Arithmetic(ConstEvalError::DivisionByZero))
        );
    }

    #[test]
    fn exposes_auditable_instruction_stream() {
        let tree = ExpressionParser::<8>::new("VALUE + 1").parse().unwrap();
        let program = lower_expression::<8, 8>(&tree).unwrap();
        assert_eq!(
            program.instructions(),
            &[
                Some(Instruction::LoadConstant("VALUE")),
                Some(Instruction::PushInteger(1)),
                Some(Instruction::Binary(BinaryOperator::Add)),
            ]
        );
    }
}
