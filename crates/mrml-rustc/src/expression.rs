use crate::{LexError, Lexer, Span, Token, TokenKind};

const MAX_EXPRESSION_DEPTH: usize = 64;
pub const MAX_CALL_ARGUMENTS: usize = 4;
const MAX_INLINE_CONST_BINDINGS: usize = 4;
const MAX_INLINE_CONST_ASSIGNMENTS: usize = 8;
const MAX_INLINE_CONST_EXPRESSION_STATEMENTS: usize = 8;
const MAX_MATCH_PATTERNS: usize = 8;
const MAX_MATCH_ALTERNATIVES: usize = 4;
const MAX_RANGE_VALIDATIONS: usize = MAX_MATCH_PATTERNS * MAX_MATCH_ALTERNATIVES;
const MAX_LOOP_BREAK_BRANCHES: usize = 4;

type InlineConstBinding<'source> = (&'source str, ExprKind<'source>, bool, Option<ScalarType>);

#[derive(Clone, Copy)]
enum MatchPattern<'source> {
    Wildcard,
    Binding(&'source str),
    Value(ExprId),
    Range {
        start: Option<ExprId>,
        end: Option<ExprId>,
        inclusive: bool,
    },
    BindingAt {
        name: &'source str,
        pattern: MatchPatternBody,
    },
}

#[derive(Clone, Copy)]
enum MatchPatternBody {
    Wildcard,
    Value(ExprId),
    Range {
        start: Option<ExprId>,
        end: Option<ExprId>,
        inclusive: bool,
    },
}

#[derive(Clone, Copy)]
enum PatternOrderValue {
    Unsigned(u128),
    Signed(i128),
    Character(u32),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ValueLoopControlTarget {
    Current,
    Enclosing,
}

impl PatternOrderValue {
    fn compare(self, other: Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;

        Some(match (self, other) {
            (Self::Unsigned(left), Self::Unsigned(right)) => left.cmp(&right),
            (Self::Signed(left), Self::Signed(right)) => left.cmp(&right),
            (Self::Signed(left), Self::Unsigned(right)) => {
                if left < 0 {
                    Ordering::Less
                } else {
                    (left as u128).cmp(&right)
                }
            }
            (Self::Unsigned(left), Self::Signed(right)) => {
                if right < 0 {
                    Ordering::Greater
                } else {
                    left.cmp(&(right as u128))
                }
            }
            (Self::Character(left), Self::Character(right)) => left.cmp(&right),
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct MatchArmPattern<'source> {
    alternatives: [Option<MatchPattern<'source>>; MAX_MATCH_ALTERNATIVES],
    alternative_count: usize,
    guard: Option<ExprId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RangeValidation {
    pub(crate) scrutinee: ExprId,
    pub(crate) start: Option<ExprId>,
    pub(crate) end: Option<ExprId>,
    pub(crate) inclusive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExprId(usize);

impl ExprId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOperator {
    LogicalOr,
    LogicalAnd,
    BitOr,
    BitXor,
    BitAnd,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    ShiftLeft,
    ShiftRight,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerType {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarType {
    Integer(IntegerType),
    Bool,
}

impl ScalarType {
    fn from_name(name: &str) -> Option<Self> {
        if name == "bool" {
            Some(Self::Bool)
        } else {
            IntegerType::from_name(name).map(Self::Integer)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerLiteral<'source> {
    pub value: u128,
    pub suffix: Option<&'source str>,
}

impl IntegerType {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "u128" => Self::U128,
            "usize" => Self::Usize,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "i128" => Self::I128,
            "isize" => Self::Isize,
            _ => return None,
        })
    }

    pub const fn bits(self, pointer_bits: u8) -> Option<u8> {
        Some(match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 => 16,
            Self::U32 | Self::I32 => 32,
            Self::U64 | Self::I64 => 64,
            Self::U128 | Self::I128 => 128,
            Self::Usize | Self::Isize if matches!(pointer_bits, 16 | 32 | 64) => pointer_bits,
            Self::Usize | Self::Isize => return None,
        })
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::Isize
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
            Self::Usize => "usize",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I128 => "i128",
            Self::Isize => "isize",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExprKind<'source> {
    Unit,
    DefaultValue,
    Integer(IntegerLiteral<'source>),
    Bool(bool),
    Char(u32),
    Identifier(&'source str),
    Call {
        callee: &'source str,
        arguments: [Option<ExprId>; MAX_CALL_ARGUMENTS],
        argument_count: usize,
    },
    InlineConst {
        operand: ExprId,
    },
    Ascribe {
        operand: ExprId,
        target: ScalarType,
    },
    Sequence {
        first: ExprId,
        then: ExprId,
    },
    Cast {
        operand: ExprId,
        target: IntegerType,
    },
    Unary {
        operator: UnaryOperator,
        operand: ExprId,
    },
    Binary {
        operator: BinaryOperator,
        left: ExprId,
        right: ExprId,
    },
    If {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: ExprId,
    },
    Return {
        operand: ExprId,
    },
    LoopBreak {
        operand: ExprId,
    },
    LoopBreakIf {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: ExprId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Expr<'source> {
    pub kind: ExprKind<'source>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpressionErrorKind {
    Lexical(LexError),
    ExpectedExpression,
    ExpectedColon,
    ExpectedCloseParen,
    ExpectedOpenBrace,
    ExpectedCloseBrace,
    ExpectedIdentifier,
    ExpectedEquals,
    ExpectedSemicolon,
    ExpectedInlineConstType,
    UnsupportedInlineConstType,
    ExpectedElse,
    ExpectedMatchPattern,
    ExpectedFatArrow,
    ExpectedComma,
    ExpectedWildcardPattern,
    NonExhaustiveMatch,
    TooManyMatchPatterns,
    TooManyMatchAlternatives,
    TooManyRangeValidations,
    InconsistentMatchBindings,
    TrailingToken,
    InvalidInteger,
    InvalidIntegerSuffix,
    InvalidCharacter,
    InvalidRangeType,
    InvalidRangeBounds,
    ExpectedCastType,
    UnsupportedCastType,
    IntegerOverflow,
    TooManyNodes,
    TooManyCallArguments,
    TooManyInlineConstBindings,
    TooManyInlineConstAssignments,
    TooManyInlineConstExpressionStatements,
    TooManyLoopBreakBranches,
    ImmutableInlineConstAssignment,
    UnsupportedInlineConstAssignment,
    NestingLimitExceeded,
    InvalidExpressionTree,
    InvalidLoopBreakTarget,
    UnknownLoopLabel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpressionError {
    pub kind: ExpressionErrorKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpressionTree<'source, const MAX_NODES: usize> {
    nodes: [Option<Expr<'source>>; MAX_NODES],
    node_count: usize,
    range_validations: [Option<RangeValidation>; MAX_RANGE_VALIDATIONS],
    range_validation_count: usize,
    root: ExprId,
}

impl<'source, const MAX_NODES: usize> ExpressionTree<'source, MAX_NODES> {
    pub fn nodes(&self) -> &[Option<Expr<'source>>] {
        &self.nodes[..self.node_count]
    }

    pub const fn root(&self) -> ExprId {
        self.root
    }

    pub(crate) fn range_validations(&self) -> &[Option<RangeValidation>] {
        &self.range_validations[..self.range_validation_count]
    }

    pub fn expression(&self, id: ExprId) -> Option<&Expr<'source>> {
        self.nodes.get(id.index()).and_then(Option::as_ref)
    }

    pub(crate) fn is_boolean_expression(&self, id: ExprId, depth: usize) -> bool {
        if depth == MAX_EXPRESSION_DEPTH {
            return false;
        }
        let Some(expression) = self.expression(id) else {
            return false;
        };
        match expression.kind {
            ExprKind::Bool(_) => true,
            ExprKind::Unit
            | ExprKind::DefaultValue
            | ExprKind::Integer(_)
            | ExprKind::Char(_)
            | ExprKind::Identifier(_)
            | ExprKind::Call { .. }
            | ExprKind::Cast { .. } => false,
            ExprKind::Ascribe {
                operand,
                target: ScalarType::Bool,
            } => self.is_boolean_expression(operand, depth + 1),
            ExprKind::Ascribe {
                target: ScalarType::Integer(_),
                ..
            } => false,
            ExprKind::Sequence { then, .. } => self.is_boolean_expression(then, depth + 1),
            ExprKind::If {
                then_branch,
                else_branch,
                ..
            }
            | ExprKind::LoopBreakIf {
                then_branch,
                else_branch,
                ..
            } => {
                self.is_boolean_expression(then_branch, depth + 1)
                    && self.is_boolean_expression(else_branch, depth + 1)
            }
            ExprKind::Return { operand } | ExprKind::LoopBreak { operand } => {
                self.is_boolean_expression(operand, depth + 1)
            }
            ExprKind::InlineConst { operand } => self.is_boolean_expression(operand, depth + 1),
            ExprKind::Unary {
                operator: UnaryOperator::Not,
                operand,
            } => self.is_boolean_expression(operand, depth + 1),
            ExprKind::Unary {
                operator: UnaryOperator::Negate,
                ..
            } => false,
            ExprKind::Binary { operator, .. } => matches!(
                operator,
                BinaryOperator::LogicalOr
                    | BinaryOperator::LogicalAnd
                    | BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::Less
                    | BinaryOperator::LessEqual
                    | BinaryOperator::Greater
                    | BinaryOperator::GreaterEqual
            ),
        }
    }

    pub fn evaluate<R: ConstantResolver>(&self, resolver: &R) -> Result<u128, ConstEvalError> {
        self.validate_ranges(resolver, 64)?;
        self.evaluate_node(self.root, resolver, 0)
    }

    pub(crate) fn evaluate_at<R: ConstantResolver>(
        &self,
        id: ExprId,
        resolver: &R,
    ) -> Result<u128, ConstEvalError> {
        self.evaluate_node(id, resolver, 0)
    }

    pub(crate) fn validate_ranges<R: ConstantResolver>(
        &self,
        resolver: &R,
        pointer_bits: u8,
    ) -> Result<(), ConstEvalError> {
        for validation in self.range_validations[..self.range_validation_count]
            .iter()
            .flatten()
        {
            let (Some(start_id), Some(end_id)) = (validation.start, validation.end) else {
                continue;
            };
            let start = self.resolved_pattern_order_value(start_id, resolver, pointer_bits)?;
            let end = self.resolved_pattern_order_value(end_id, resolver, pointer_bits)?;
            let (Some(start), Some(end)) = (start, end) else {
                continue;
            };
            let Some(order) = start.compare(end) else {
                return Err(ConstEvalError::InvalidRangeType);
            };
            let invalid = if validation.inclusive {
                order == core::cmp::Ordering::Greater
            } else {
                order != core::cmp::Ordering::Less
            };
            if invalid {
                return Err(ConstEvalError::InvalidRangeBounds);
            }
        }
        Ok(())
    }

    fn resolved_pattern_order_value<R: ConstantResolver>(
        &self,
        id: ExprId,
        resolver: &R,
        pointer_bits: u8,
    ) -> Result<Option<PatternOrderValue>, ConstEvalError> {
        let expression = self
            .expression(id)
            .ok_or(ConstEvalError::InvalidExpressionTree)?;
        match expression.kind {
            ExprKind::Integer(literal) => {
                if literal
                    .suffix
                    .and_then(IntegerType::from_name)
                    .is_some_and(IntegerType::is_signed)
                {
                    Ok(i128::try_from(literal.value)
                        .ok()
                        .map(PatternOrderValue::Signed))
                } else {
                    Ok(Some(PatternOrderValue::Unsigned(literal.value)))
                }
            }
            ExprKind::Unary {
                operator: UnaryOperator::Negate,
                operand,
            } => {
                let Some(ExprKind::Integer(literal)) =
                    self.expression(operand).map(|operand| operand.kind)
                else {
                    return Ok(None);
                };
                let minimum_magnitude = (i128::MAX as u128) + 1;
                let value = if literal.value == minimum_magnitude {
                    i128::MIN
                } else {
                    -i128::try_from(literal.value).map_err(|_| ConstEvalError::Overflow)?
                };
                Ok(Some(PatternOrderValue::Signed(value)))
            }
            ExprKind::Char(value) => Ok(Some(PatternOrderValue::Character(value))),
            ExprKind::Bool(_) => Err(ConstEvalError::InvalidRangeType),
            ExprKind::Identifier(name) if resolver.resolves_bool(name) => {
                Err(ConstEvalError::InvalidRangeType)
            }
            ExprKind::Identifier(name) => {
                let Some(ty) = resolver.resolve_type(name) else {
                    return Ok(None);
                };
                let Some(value) = resolver.resolve(name) else {
                    return Ok(None);
                };
                if ty.is_signed() {
                    let bits = ty.bits(pointer_bits).ok_or(ConstEvalError::InvalidCast)?;
                    let signed = if bits == 128 {
                        value as i128
                    } else {
                        let mask = (1u128 << bits) - 1;
                        let value = value & mask;
                        if value & (1u128 << (bits - 1)) == 0 {
                            value as i128
                        } else {
                            (value | !mask) as i128
                        }
                    };
                    Ok(Some(PatternOrderValue::Signed(signed)))
                } else {
                    Ok(Some(PatternOrderValue::Unsigned(value)))
                }
            }
            _ => Ok(None),
        }
    }

    fn evaluate_node<R: ConstantResolver>(
        &self,
        id: ExprId,
        resolver: &R,
        depth: usize,
    ) -> Result<u128, ConstEvalError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(ConstEvalError::NestingLimitExceeded);
        }
        let expression = self
            .expression(id)
            .ok_or(ConstEvalError::InvalidExpressionTree)?;
        match expression.kind {
            ExprKind::Unit | ExprKind::DefaultValue => Ok(0),
            ExprKind::Integer(literal) => Ok(literal.value),
            ExprKind::Bool(value) => Ok(u128::from(value)),
            ExprKind::Char(value) => Ok(u128::from(value)),
            ExprKind::Identifier(name) => resolver
                .resolve(name)
                .ok_or(ConstEvalError::UnknownIdentifier),
            ExprKind::Call {
                callee,
                arguments,
                argument_count,
            } => {
                let mut values = [0u128; MAX_CALL_ARGUMENTS];
                for (index, argument) in arguments[..argument_count].iter().enumerate() {
                    values[index] = self.evaluate_node(
                        argument.ok_or(ConstEvalError::InvalidExpressionTree)?,
                        resolver,
                        depth + 1,
                    )?;
                }
                resolver
                    .resolve_call(callee, &values[..argument_count])
                    .ok_or(ConstEvalError::UnsupportedCall)
            }
            ExprKind::Cast { operand, target } => {
                let value = self.evaluate_node(operand, resolver, depth + 1)?;
                cast_integer(value, target, 64)
            }
            ExprKind::Ascribe { operand, target } => {
                let value = self.evaluate_node(operand, resolver, depth + 1)?;
                match target {
                    ScalarType::Integer(target) => ascribe_integer(value, target, 64),
                    ScalarType::Bool if self.is_boolean_expression(operand, depth + 1) => {
                        Ok(u128::from(value != 0))
                    }
                    ScalarType::Bool => Err(ConstEvalError::InvalidCast),
                }
            }
            ExprKind::Unary { operator, operand } => {
                let value = self.evaluate_node(operand, resolver, depth + 1)?;
                match operator {
                    UnaryOperator::Negate
                        if let Some(target) = self.expression(operand).and_then(|expression| {
                            match expression.kind {
                                ExprKind::Integer(literal) => literal
                                    .suffix
                                    .and_then(IntegerType::from_name)
                                    .filter(|target| target.is_signed()),
                                _ => None,
                            }
                        }) =>
                    {
                        negate_signed_magnitude(value, target, 64)
                    }
                    UnaryOperator::Negate => {
                        0u128.checked_sub(value).ok_or(ConstEvalError::Overflow)
                    }
                    UnaryOperator::Not if self.is_boolean_expression(operand, depth + 1) => {
                        Ok(u128::from(value == 0))
                    }
                    UnaryOperator::Not => Ok(!value),
                }
            }
            ExprKind::Binary {
                operator,
                left,
                right,
            } => {
                let left = self.evaluate_node(left, resolver, depth + 1)?;
                if operator == BinaryOperator::LogicalAnd && left == 0 {
                    return Ok(0);
                }
                if operator == BinaryOperator::LogicalOr && left != 0 {
                    return Ok(1);
                }
                let right = self.evaluate_node(right, resolver, depth + 1)?;
                evaluate_binary(operator, left, right)
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
                if self.evaluate_node(condition, resolver, depth + 1)? != 0 {
                    self.evaluate_node(then_branch, resolver, depth + 1)
                } else {
                    self.evaluate_node(else_branch, resolver, depth + 1)
                }
            }
            ExprKind::Return { operand } | ExprKind::LoopBreak { operand } => {
                self.evaluate_node(operand, resolver, depth + 1)
            }
            ExprKind::InlineConst { operand } => self.evaluate_node(operand, resolver, depth + 1),
            ExprKind::Sequence { first, then } => {
                self.evaluate_node(first, resolver, depth + 1)?;
                self.evaluate_node(then, resolver, depth + 1)
            }
        }
    }
}

pub trait ConstantResolver {
    fn resolve(&self, name: &str) -> Option<u128>;

    fn resolve_type(&self, _: &str) -> Option<IntegerType> {
        None
    }

    fn resolves_bool(&self, _: &str) -> bool {
        false
    }

    fn resolve_call(&self, _: &str, _: &[u128]) -> Option<u128> {
        None
    }

    fn resolve_call_type(&self, _: &str, _: usize) -> Option<IntegerType> {
        None
    }

    fn call_resolves_bool(&self, _: &str, _: usize) -> bool {
        false
    }
}

pub struct NoConstants;

impl ConstantResolver for NoConstants {
    fn resolve(&self, _name: &str) -> Option<u128> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstEvalError {
    UnknownIdentifier,
    DivisionByZero,
    Overflow,
    InvalidShift,
    InvalidCast,
    InvalidExpressionTree,
    NestingLimitExceeded,
    UnsupportedCall,
    InvalidRangeType,
    InvalidRangeBounds,
}

pub(crate) fn cast_integer(
    value: u128,
    target: IntegerType,
    pointer_bits: u8,
) -> Result<u128, ConstEvalError> {
    let bits = target
        .bits(pointer_bits)
        .ok_or(ConstEvalError::InvalidCast)?;
    let mask = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    let truncated = value & mask;
    if target.is_signed() && bits != 128 && truncated & (1u128 << (bits - 1)) != 0 {
        Ok(truncated | !mask)
    } else {
        Ok(truncated)
    }
}

pub(crate) fn ascribe_integer(
    value: u128,
    target: IntegerType,
    pointer_bits: u8,
) -> Result<u128, ConstEvalError> {
    let normalized = cast_integer(value, target, pointer_bits)?;
    if normalized == value {
        Ok(value)
    } else {
        Err(ConstEvalError::Overflow)
    }
}

pub(crate) fn negate_signed_magnitude(
    magnitude: u128,
    target: IntegerType,
    pointer_bits: u8,
) -> Result<u128, ConstEvalError> {
    if !target.is_signed() {
        return Err(ConstEvalError::InvalidCast);
    }
    let bits = target
        .bits(pointer_bits)
        .ok_or(ConstEvalError::InvalidCast)?;
    let limit = 1u128 << (bits - 1);
    if magnitude > limit {
        return Err(ConstEvalError::Overflow);
    }
    cast_integer(0u128.wrapping_sub(magnitude), target, pointer_bits)
}

pub(crate) fn evaluate_binary(
    operator: BinaryOperator,
    left: u128,
    right: u128,
) -> Result<u128, ConstEvalError> {
    match operator {
        BinaryOperator::LogicalOr => Ok(u128::from(left != 0 || right != 0)),
        BinaryOperator::LogicalAnd => Ok(u128::from(left != 0 && right != 0)),
        BinaryOperator::BitOr => Ok(left | right),
        BinaryOperator::BitXor => Ok(left ^ right),
        BinaryOperator::BitAnd => Ok(left & right),
        BinaryOperator::Equal => Ok(u128::from(left == right)),
        BinaryOperator::NotEqual => Ok(u128::from(left != right)),
        BinaryOperator::Less => Ok(u128::from(left < right)),
        BinaryOperator::LessEqual => Ok(u128::from(left <= right)),
        BinaryOperator::Greater => Ok(u128::from(left > right)),
        BinaryOperator::GreaterEqual => Ok(u128::from(left >= right)),
        BinaryOperator::ShiftLeft => left
            .checked_shl(u32::try_from(right).map_err(|_| ConstEvalError::InvalidShift)?)
            .ok_or(ConstEvalError::InvalidShift),
        BinaryOperator::ShiftRight => left
            .checked_shr(u32::try_from(right).map_err(|_| ConstEvalError::InvalidShift)?)
            .ok_or(ConstEvalError::InvalidShift),
        BinaryOperator::Add => left.checked_add(right).ok_or(ConstEvalError::Overflow),
        BinaryOperator::Subtract => left.checked_sub(right).ok_or(ConstEvalError::Overflow),
        BinaryOperator::Multiply => left.checked_mul(right).ok_or(ConstEvalError::Overflow),
        BinaryOperator::Divide => left
            .checked_div(right)
            .ok_or(ConstEvalError::DivisionByZero),
        BinaryOperator::Remainder => left
            .checked_rem(right)
            .ok_or(ConstEvalError::DivisionByZero),
    }
}

pub struct ExpressionParser<'source, const MAX_NODES: usize> {
    lexer: Lexer<'source>,
    lookahead: Option<Token<'source>>,
    nodes: [Option<Expr<'source>>; MAX_NODES],
    node_count: usize,
    range_validations: [Option<RangeValidation>; MAX_RANGE_VALIDATIONS],
    range_validation_count: usize,
    source_len: usize,
}

impl<'source, const MAX_NODES: usize> ExpressionParser<'source, MAX_NODES> {
    pub const fn new(source: &'source str) -> Self {
        Self {
            lexer: Lexer::new(source),
            lookahead: None,
            nodes: [None; MAX_NODES],
            node_count: 0,
            range_validations: [None; MAX_RANGE_VALIDATIONS],
            range_validation_count: 0,
            source_len: source.len(),
        }
    }

    fn lexical(error: LexError) -> ExpressionError {
        ExpressionError {
            kind: ExpressionErrorKind::Lexical(error),
            span: error.span,
        }
    }

    fn peek(&mut self) -> Result<Option<Token<'source>>, ExpressionError> {
        if self.lookahead.is_none() {
            self.lookahead = self.lexer.next_token().map_err(Self::lexical)?;
        }
        Ok(self.lookahead)
    }

    fn take(&mut self) -> Result<Option<Token<'source>>, ExpressionError> {
        if let Some(token) = self.lookahead.take() {
            Ok(Some(token))
        } else {
            self.lexer.next_token().map_err(Self::lexical)
        }
    }

    fn error(&self, kind: ExpressionErrorKind, token: Option<Token<'source>>) -> ExpressionError {
        ExpressionError {
            kind,
            span: token.map_or(
                Span {
                    start: self.source_len,
                    end: self.source_len,
                },
                |value| value.span,
            ),
        }
    }

    fn push(&mut self, expression: Expr<'source>) -> Result<ExprId, ExpressionError> {
        if self.node_count == MAX_NODES {
            return Err(ExpressionError {
                kind: ExpressionErrorKind::TooManyNodes,
                span: expression.span,
            });
        }
        let id = ExprId(self.node_count);
        self.nodes[self.node_count] = Some(expression);
        self.node_count += 1;
        Ok(id)
    }

    fn record_range_validation(
        &mut self,
        scrutinee: ExprId,
        start: Option<ExprId>,
        end: Option<ExprId>,
        inclusive: bool,
    ) -> Result<(), ExpressionError> {
        if self.range_validation_count == MAX_RANGE_VALIDATIONS {
            return Err(ExpressionError {
                kind: ExpressionErrorKind::TooManyRangeValidations,
                span: Span {
                    start: start
                        .or(end)
                        .and_then(|bound| self.node_span(bound).ok())
                        .map_or(0, |span| span.start),
                    end: end
                        .or(start)
                        .and_then(|bound| self.node_span(bound).ok())
                        .map_or(self.source_len, |span| span.end),
                },
            });
        }
        self.range_validations[self.range_validation_count] = Some(RangeValidation {
            scrutinee,
            start,
            end,
            inclusive,
        });
        self.range_validation_count += 1;
        Ok(())
    }

    fn node_span(&self, id: ExprId) -> Result<Span, ExpressionError> {
        self.nodes
            .get(id.index())
            .and_then(Option::as_ref)
            .map(|expression| expression.span)
            .ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: Span {
                    start: self.source_len,
                    end: self.source_len,
                },
            })
    }

    fn pattern_order_value(&self, id: ExprId) -> Option<PatternOrderValue> {
        let expression = self.nodes.get(id.index()).and_then(Option::as_ref)?;
        match expression.kind {
            ExprKind::Integer(literal) => {
                if literal
                    .suffix
                    .and_then(IntegerType::from_name)
                    .is_some_and(IntegerType::is_signed)
                {
                    i128::try_from(literal.value)
                        .ok()
                        .map(PatternOrderValue::Signed)
                } else {
                    Some(PatternOrderValue::Unsigned(literal.value))
                }
            }
            ExprKind::Unary {
                operator: UnaryOperator::Negate,
                operand,
            } => {
                let operand = self.nodes.get(operand.index()).and_then(Option::as_ref)?;
                let ExprKind::Integer(literal) = operand.kind else {
                    return None;
                };
                if literal
                    .suffix
                    .and_then(IntegerType::from_name)
                    .is_some_and(|ty| !ty.is_signed())
                {
                    return None;
                }
                let minimum_magnitude = (i128::MAX as u128) + 1;
                let value = if literal.value == minimum_magnitude {
                    i128::MIN
                } else {
                    -i128::try_from(literal.value).ok()?
                };
                Some(PatternOrderValue::Signed(value))
            }
            ExprKind::Char(value) => Some(PatternOrderValue::Character(value)),
            _ => None,
        }
    }

    fn node_is_unit(&self, id: ExprId, depth: usize) -> bool {
        if depth == MAX_EXPRESSION_DEPTH {
            return false;
        }
        let Some(expression) = self.nodes.get(id.index()).and_then(Option::as_ref) else {
            return false;
        };
        match expression.kind {
            ExprKind::Unit => true,
            ExprKind::DefaultValue => false,
            ExprKind::InlineConst { operand }
            | ExprKind::Return { operand }
            | ExprKind::LoopBreak { operand } => self.node_is_unit(operand, depth + 1),
            ExprKind::Sequence { then, .. } => self.node_is_unit(then, depth + 1),
            ExprKind::If {
                then_branch,
                else_branch,
                ..
            }
            | ExprKind::LoopBreakIf {
                then_branch,
                else_branch,
                ..
            } => {
                self.node_is_unit(then_branch, depth + 1)
                    && self.node_is_unit(else_branch, depth + 1)
            }
            _ => false,
        }
    }

    fn atom(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        let token = self.take()?;
        match token {
            Some(token) if token.text == "break" => {
                if self
                    .peek()?
                    .is_some_and(|next| next.kind == TokenKind::CloseBrace)
                {
                    return self.push(Expr {
                        kind: ExprKind::Identifier(token.text),
                        span: token.span,
                    });
                }
                let operand = self.nested_loop_break_operand(depth + 1)?;
                let operand_span = self.node_span(operand)?;
                self.push(Expr {
                    kind: ExprKind::LoopBreak { operand },
                    span: Span {
                        start: token.span.start,
                        end: operand_span.end,
                    },
                })
            }
            Some(token) if token.kind == TokenKind::Integer => {
                let literal = decode_integer(token)?;
                self.push(Expr {
                    kind: ExprKind::Integer(literal),
                    span: token.span,
                })
            }
            Some(token) if token.kind == TokenKind::Character => self.push(Expr {
                kind: ExprKind::Char(decode_character(token)?),
                span: token.span,
            }),
            Some(token) if token.text == "if" => self.if_expression(token, depth + 1),
            Some(token) if token.text == "match" => self.match_expression(token, depth + 1),
            Some(token) if token.text == "loop" => {
                self.loop_break_expression(token, None, depth + 1)
            }
            Some(label) if label.kind == TokenKind::Lifetime => {
                let colon = self.take()?;
                if !colon.is_some_and(|token| token.kind == TokenKind::Colon) {
                    return Err(self.error(ExpressionErrorKind::ExpectedColon, colon));
                }
                let loop_token = self.take()?;
                let Some(loop_token) = loop_token.filter(|token| token.text == "loop") else {
                    return Err(self.error(ExpressionErrorKind::ExpectedExpression, loop_token));
                };
                self.loop_break_expression(loop_token, Some(label), depth + 1)
            }
            Some(token) if token.text == "const" => self.const_block_expression(token, depth + 1),
            Some(token) if token.kind == TokenKind::Identifier => {
                if token.text == "Default"
                    && self
                        .peek()?
                        .is_some_and(|next| next.kind == TokenKind::PathSeparator)
                {
                    self.take()?;
                    let method = self.take()?;
                    if !method.is_some_and(|method| method.text == "default") {
                        return Err(self.error(ExpressionErrorKind::ExpectedExpression, method));
                    }
                    let open = self.take()?;
                    if !open.is_some_and(|open| open.kind == TokenKind::OpenParen) {
                        return Err(self.error(ExpressionErrorKind::ExpectedExpression, open));
                    }
                    let close = self.take()?;
                    let Some(close) = close.filter(|close| close.kind == TokenKind::CloseParen)
                    else {
                        return Err(self.error(ExpressionErrorKind::ExpectedCloseParen, close));
                    };
                    return self.push(Expr {
                        kind: ExprKind::DefaultValue,
                        span: Span {
                            start: token.span.start,
                            end: close.span.end,
                        },
                    });
                }
                if matches!(token.text, "true" | "false") {
                    return self.push(Expr {
                        kind: ExprKind::Bool(token.text == "true"),
                        span: token.span,
                    });
                }
                if !self
                    .peek()?
                    .is_some_and(|next| next.kind == TokenKind::OpenParen)
                {
                    return self.push(Expr {
                        kind: ExprKind::Identifier(token.text),
                        span: token.span,
                    });
                }
                self.take()?;
                let mut arguments = [None; MAX_CALL_ARGUMENTS];
                let mut argument_count = 0usize;
                if !self
                    .peek()?
                    .is_some_and(|next| next.kind == TokenKind::CloseParen)
                {
                    loop {
                        if argument_count == MAX_CALL_ARGUMENTS {
                            let token = self.peek()?;
                            return Err(
                                self.error(ExpressionErrorKind::TooManyCallArguments, token)
                            );
                        }
                        arguments[argument_count] = Some(self.expression(0, depth + 1)?);
                        argument_count += 1;
                        if !self
                            .peek()?
                            .is_some_and(|next| next.kind == TokenKind::Comma)
                        {
                            break;
                        }
                        self.take()?;
                    }
                }
                let close = self.take()?;
                let Some(close) = close.filter(|close| close.kind == TokenKind::CloseParen) else {
                    return Err(self.error(ExpressionErrorKind::ExpectedCloseParen, close));
                };
                self.push(Expr {
                    kind: ExprKind::Call {
                        callee: token.text,
                        arguments,
                        argument_count,
                    },
                    span: Span {
                        start: token.span.start,
                        end: close.span.end,
                    },
                })
            }
            Some(token) if token.kind == TokenKind::OpenParen => {
                if self
                    .peek()?
                    .is_some_and(|value| value.kind == TokenKind::CloseParen)
                {
                    let close = self.take()?.ok_or_else(|| {
                        self.error(ExpressionErrorKind::ExpectedCloseParen, self.lookahead)
                    })?;
                    return self.push(Expr {
                        kind: ExprKind::Unit,
                        span: Span {
                            start: token.span.start,
                            end: close.span.end,
                        },
                    });
                }
                let inner = self.expression(0, depth + 1)?;
                let close = self.take()?;
                if !close.is_some_and(|value| value.kind == TokenKind::CloseParen) {
                    return Err(self.error(ExpressionErrorKind::ExpectedCloseParen, close));
                }
                Ok(inner)
            }
            Some(token) if token.kind == TokenKind::OpenBrace => {
                self.block_expression(token, depth + 1)
            }
            _ => Err(self.error(ExpressionErrorKind::ExpectedExpression, token)),
        }
    }

    fn block_expression(
        &mut self,
        _open: Token<'source>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        let (inner, _) = self.scalar_branch_block(depth + 1)?;
        Ok(inner)
    }

    fn const_block_expression(
        &mut self,
        const_token: Token<'source>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        let open = self.take()?;
        let Some(open) = open.filter(|token| token.kind == TokenKind::OpenBrace) else {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        };
        let mut bindings: [Option<InlineConstBinding<'source>>; MAX_INLINE_CONST_BINDINGS] =
            [None; MAX_INLINE_CONST_BINDINGS];
        let mut binding_count = 0usize;
        while self.peek()?.is_some_and(|token| token.text == "let") {
            self.inline_const_binding(&mut bindings, &mut binding_count, depth + 1)?;
        }
        if binding_count == 0
            && self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::CloseBrace)
        {
            let inner = self.block_expression(open, depth + 1)?;
            return self.push(Expr {
                kind: ExprKind::InlineConst { operand: inner },
                span: Span {
                    start: const_token.span.start,
                    end: self.node_span(inner)?.end,
                },
            });
        }
        let mut assignment_count = 0usize;
        let mut statements = [None; MAX_INLINE_CONST_EXPRESSION_STATEMENTS];
        let mut statement_count = 0usize;
        let tail = loop {
            if let Some(close) = self
                .peek()?
                .filter(|token| token.kind == TokenKind::CloseBrace)
            {
                break self.push(Expr {
                    kind: ExprKind::Unit,
                    span: Span {
                        start: close.span.start,
                        end: close.span.start,
                    },
                })?;
            }
            if self.peek()?.is_some_and(|token| token.text == "let") {
                self.inline_const_binding(&mut bindings, &mut binding_count, depth + 1)?;
                continue;
            }
            if self.inline_const_assignment(
                &mut bindings,
                binding_count,
                &mut assignment_count,
                depth + 1,
            )? {
                continue;
            }
            let expression = self.expression(0, depth + 1)?;
            for (name, replacement, _, _) in bindings[..binding_count].iter().flatten().rev() {
                self.substitute_identifier(expression, name, *replacement, depth + 1)?;
            }
            if self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Semicolon)
            {
                let semicolon = self.take()?;
                if statement_count == MAX_INLINE_CONST_EXPRESSION_STATEMENTS {
                    return Err(self.error(
                        ExpressionErrorKind::TooManyInlineConstExpressionStatements,
                        semicolon,
                    ));
                }
                statements[statement_count] = Some(expression);
                statement_count += 1;
            } else {
                break expression;
            }
        };
        let inner = {
            let mut inner = tail;
            let close = self.take()?;
            if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
            }
            for first in statements[..statement_count].iter().flatten().rev() {
                let first_span = self.node_span(*first)?;
                let then_span = self.node_span(inner)?;
                inner = self.push(Expr {
                    kind: ExprKind::Sequence {
                        first: *first,
                        then: inner,
                    },
                    span: Span {
                        start: first_span.start,
                        end: then_span.end,
                    },
                })?;
            }
            inner
        };
        let inner_span = self.node_span(inner)?;
        self.push(Expr {
            kind: ExprKind::InlineConst { operand: inner },
            span: Span {
                start: const_token.span.start,
                end: inner_span.end,
            },
        })
    }

    fn inline_const_binding(
        &mut self,
        bindings: &mut [Option<InlineConstBinding<'source>>; MAX_INLINE_CONST_BINDINGS],
        binding_count: &mut usize,
        depth: usize,
    ) -> Result<(), ExpressionError> {
        let let_token = self.take()?;
        if *binding_count == MAX_INLINE_CONST_BINDINGS {
            return Err(self.error(ExpressionErrorKind::TooManyInlineConstBindings, let_token));
        }
        let mutable = if self.peek()?.is_some_and(|token| token.text == "mut") {
            self.take()?;
            true
        } else {
            false
        };
        let name = self.take()?;
        let Some(name) = name.filter(|token| token.kind == TokenKind::Identifier) else {
            return Err(self.error(ExpressionErrorKind::ExpectedIdentifier, name));
        };
        let annotation = if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Colon)
        {
            self.take()?;
            let ty = self.take()?;
            let Some(ty) = ty.filter(|token| token.kind == TokenKind::Identifier) else {
                return Err(self.error(ExpressionErrorKind::ExpectedInlineConstType, ty));
            };
            Some(ScalarType::from_name(ty.text).ok_or(ExpressionError {
                kind: ExpressionErrorKind::UnsupportedInlineConstType,
                span: ty.span,
            })?)
        } else {
            None
        };
        let equals = self.take()?;
        if !equals.is_some_and(|token| token.kind == TokenKind::Operator && token.text == "=") {
            return Err(self.error(ExpressionErrorKind::ExpectedEquals, equals));
        }
        let initializer = self.expression(0, depth + 1)?;
        let semicolon = self.take()?;
        if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
            return Err(self.error(ExpressionErrorKind::ExpectedSemicolon, semicolon));
        }
        for (binding_name, replacement, _, _) in bindings[..*binding_count].iter().flatten().rev() {
            self.substitute_identifier(initializer, binding_name, *replacement, depth + 1)?;
        }
        let mut replacement = self
            .nodes
            .get(initializer.index())
            .and_then(Option::as_ref)
            .map(|expression| expression.kind)
            .ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: name.span,
            })?;
        if let Some(target) = annotation {
            let operand = self.push(Expr {
                kind: replacement,
                span: self.node_span(initializer)?,
            })?;
            let ascribed = self.push(Expr {
                kind: ExprKind::Ascribe { operand, target },
                span: Span {
                    start: name.span.start,
                    end: self.node_span(initializer)?.end,
                },
            })?;
            replacement = self.nodes[ascribed.index()]
                .as_ref()
                .ok_or(ExpressionError {
                    kind: ExpressionErrorKind::InvalidExpressionTree,
                    span: name.span,
                })?
                .kind;
        }
        bindings[*binding_count] = Some((name.text, replacement, mutable, annotation));
        *binding_count += 1;
        Ok(())
    }

    fn inline_const_assignment(
        &mut self,
        bindings: &mut [Option<InlineConstBinding<'source>>; MAX_INLINE_CONST_BINDINGS],
        binding_count: usize,
        assignment_count: &mut usize,
        depth: usize,
    ) -> Result<bool, ExpressionError> {
        let saved_lexer = self.lexer.clone();
        let saved_lookahead = self.lookahead;
        let Some(name) = self
            .take()?
            .filter(|token| token.kind == TokenKind::Identifier)
        else {
            self.lexer = saved_lexer;
            self.lookahead = saved_lookahead;
            return Ok(false);
        };
        let Some(binding_index) = bindings[..binding_count]
            .iter()
            .rposition(|binding| binding.is_some_and(|binding| binding.0 == name.text))
        else {
            self.lexer = saved_lexer;
            self.lookahead = saved_lookahead;
            return Ok(false);
        };
        let operator = self.take()?;
        let Some(operator) = operator.filter(|token| {
            token.kind == TokenKind::Operator
                && matches!(
                    token.text,
                    "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>="
                )
        }) else {
            self.lexer = saved_lexer;
            self.lookahead = saved_lookahead;
            return Ok(false);
        };
        if *assignment_count == MAX_INLINE_CONST_ASSIGNMENTS {
            return Err(ExpressionError {
                kind: ExpressionErrorKind::TooManyInlineConstAssignments,
                span: operator.span,
            });
        }
        let (_, previous, mutable, annotation) =
            bindings[binding_index].ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: name.span,
            })?;
        if !mutable {
            return Err(ExpressionError {
                kind: ExpressionErrorKind::ImmutableInlineConstAssignment,
                span: name.span,
            });
        }
        let value = self.expression(0, depth + 1)?;
        let semicolon = self.take()?;
        if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
            return Err(self.error(ExpressionErrorKind::ExpectedSemicolon, semicolon));
        }
        for (binding_name, replacement, _, _) in bindings[..binding_count].iter().flatten().rev() {
            self.substitute_identifier(value, binding_name, *replacement, depth + 1)?;
        }
        let value_kind = self
            .nodes
            .get(value.index())
            .and_then(Option::as_ref)
            .map(|expression| expression.kind)
            .ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: name.span,
            })?;
        let mut replacement = if operator.text == "=" {
            value_kind
        } else {
            let binary_operator = match operator.text {
                "+=" => BinaryOperator::Add,
                "-=" => BinaryOperator::Subtract,
                "*=" => BinaryOperator::Multiply,
                "/=" => BinaryOperator::Divide,
                "%=" => BinaryOperator::Remainder,
                "&=" => BinaryOperator::BitAnd,
                "|=" => BinaryOperator::BitOr,
                "^=" => BinaryOperator::BitXor,
                "<<=" => BinaryOperator::ShiftLeft,
                ">>=" => BinaryOperator::ShiftRight,
                _ => {
                    return Err(ExpressionError {
                        kind: ExpressionErrorKind::UnsupportedInlineConstAssignment,
                        span: operator.span,
                    });
                }
            };
            let previous = self.push(Expr {
                kind: previous,
                span: name.span,
            })?;
            let value_span = self.node_span(value)?;
            let updated = self.push(Expr {
                kind: ExprKind::Binary {
                    operator: binary_operator,
                    left: previous,
                    right: value,
                },
                span: Span {
                    start: name.span.start,
                    end: value_span.end,
                },
            })?;
            self.nodes[updated.index()]
                .as_ref()
                .map(|expression| expression.kind)
                .ok_or(ExpressionError {
                    kind: ExpressionErrorKind::InvalidExpressionTree,
                    span: name.span,
                })?
        };
        if let Some(target) = annotation {
            let operand = self.push(Expr {
                kind: replacement,
                span: self.node_span(value)?,
            })?;
            let ascribed = self.push(Expr {
                kind: ExprKind::Ascribe { operand, target },
                span: Span {
                    start: name.span.start,
                    end: self.node_span(value)?.end,
                },
            })?;
            replacement = self.nodes[ascribed.index()]
                .as_ref()
                .ok_or(ExpressionError {
                    kind: ExpressionErrorKind::InvalidExpressionTree,
                    span: name.span,
                })?
                .kind;
        }
        bindings[binding_index] = Some((name.text, replacement, mutable, annotation));
        *assignment_count += 1;
        Ok(true)
    }

    fn substitute_identifier(
        &mut self,
        id: ExprId,
        name: &str,
        replacement: ExprKind<'source>,
        depth: usize,
    ) -> Result<(), ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(ExpressionError {
                kind: ExpressionErrorKind::NestingLimitExceeded,
                span: self.node_span(id)?,
            });
        }
        let expression = self
            .nodes
            .get(id.index())
            .and_then(Option::as_ref)
            .copied()
            .ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: Span {
                    start: self.source_len,
                    end: self.source_len,
                },
            })?;
        match expression.kind {
            ExprKind::Identifier(identifier) if identifier == name => {
                self.nodes[id.index()] = Some(Expr {
                    kind: replacement,
                    span: expression.span,
                });
            }
            ExprKind::Call {
                arguments,
                argument_count,
                ..
            } => {
                for argument in arguments[..argument_count].iter().flatten() {
                    self.substitute_identifier(*argument, name, replacement, depth + 1)?;
                }
            }
            ExprKind::Cast { operand, .. }
            | ExprKind::Ascribe { operand, .. }
            | ExprKind::Unary { operand, .. }
            | ExprKind::Return { operand }
            | ExprKind::LoopBreak { operand } => {
                self.substitute_identifier(operand, name, replacement, depth + 1)?;
            }
            ExprKind::Binary { left, right, .. } => {
                self.substitute_identifier(left, name, replacement, depth + 1)?;
                self.substitute_identifier(right, name, replacement, depth + 1)?;
            }
            ExprKind::Sequence { first, then } => {
                self.substitute_identifier(first, name, replacement, depth + 1)?;
                self.substitute_identifier(then, name, replacement, depth + 1)?;
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
                self.substitute_identifier(condition, name, replacement, depth + 1)?;
                self.substitute_identifier(then_branch, name, replacement, depth + 1)?;
                self.substitute_identifier(else_branch, name, replacement, depth + 1)?;
            }
            ExprKind::InlineConst { .. }
            | ExprKind::Unit
            | ExprKind::DefaultValue
            | ExprKind::Integer(_)
            | ExprKind::Bool(_)
            | ExprKind::Char(_)
            | ExprKind::Identifier(_) => {}
        }
        Ok(())
    }

    fn if_expression(
        &mut self,
        if_token: Token<'source>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(if_token)));
        }
        let condition = self.expression(0, depth + 1)?;
        let open = self.take()?;
        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        }
        let (then_branch, then_end) = self.scalar_branch_block(depth + 1)?;
        let has_else = self.peek()?.is_some_and(|token| token.text == "else");
        if !has_else && !self.node_is_unit(then_branch, depth + 1) {
            let token = self.peek()?;
            return Err(self.error(ExpressionErrorKind::ExpectedElse, token));
        }
        let (else_branch, end) = if has_else {
            self.take()?;
            if let Some(nested_if) = self.peek()?.filter(|token| token.text == "if") {
                self.take()?;
                let branch = self.if_expression(nested_if, depth + 1)?;
                (branch, self.node_span(branch)?.end)
            } else {
                let open = self.take()?;
                if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                    return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
                }
                self.scalar_branch_block(depth + 1)?
            }
        } else {
            let unit = self.push(Expr {
                kind: ExprKind::Unit,
                span: Span {
                    start: then_end,
                    end: then_end,
                },
            })?;
            (unit, then_end)
        };
        self.push(Expr {
            kind: ExprKind::If {
                condition,
                then_branch,
                else_branch,
            },
            span: Span {
                start: if_token.span.start,
                end,
            },
        })
    }

    fn match_pattern_bound(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        let token = self.take()?;
        match token {
            Some(token) if token.kind == TokenKind::Integer => self.push(Expr {
                kind: ExprKind::Integer(decode_integer(token)?),
                span: token.span,
            }),
            Some(token) if token.kind == TokenKind::Character => self.push(Expr {
                kind: ExprKind::Char(decode_character(token)?),
                span: token.span,
            }),
            Some(token) if matches!(token.text, "true" | "false") => self.push(Expr {
                kind: ExprKind::Bool(token.text == "true"),
                span: token.span,
            }),
            Some(token)
                if token.kind == TokenKind::Identifier
                    && token.text != "_"
                    && token.text != "mut" =>
            {
                self.push(Expr {
                    kind: ExprKind::Identifier(token.text),
                    span: token.span,
                })
            }
            Some(minus) if minus.kind == TokenKind::Operator && minus.text == "-" => {
                let literal = self.take()?;
                let Some(literal) = literal.filter(|token| token.kind == TokenKind::Integer) else {
                    return Err(self.error(ExpressionErrorKind::ExpectedMatchPattern, literal));
                };
                let operand = self.push(Expr {
                    kind: ExprKind::Integer(decode_integer(literal)?),
                    span: literal.span,
                })?;
                self.push(Expr {
                    kind: ExprKind::Unary {
                        operator: UnaryOperator::Negate,
                        operand,
                    },
                    span: Span {
                        start: minus.span.start,
                        end: literal.span.end,
                    },
                })
            }
            _ => Err(self.error(ExpressionErrorKind::ExpectedMatchPattern, token)),
        }
    }

    fn match_pattern(&mut self, depth: usize) -> Result<MatchPattern<'source>, ExpressionError> {
        if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Identifier && token.text == "_")
        {
            self.take()?;
            return Ok(MatchPattern::Wildcard);
        }
        let start = if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Dot)
        {
            None
        } else {
            Some(self.match_pattern_bound(depth + 1)?)
        };
        let has_at = start.is_some()
            && self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Operator && token.text == "@");
        if let Some(start) = start.filter(|_| has_at) {
            let name = self
                .nodes
                .get(start.index())
                .and_then(Option::as_ref)
                .and_then(|expression| match expression.kind {
                    ExprKind::Identifier(name) => Some(name),
                    _ => None,
                })
                .ok_or_else(|| {
                    self.error(ExpressionErrorKind::ExpectedIdentifier, self.lookahead)
                })?;
            self.take()?;
            let pattern = match self.match_pattern(depth + 1)? {
                MatchPattern::Wildcard => MatchPatternBody::Wildcard,
                MatchPattern::Value(value) => MatchPatternBody::Value(value),
                MatchPattern::Range {
                    start,
                    end,
                    inclusive,
                } => MatchPatternBody::Range {
                    start,
                    end,
                    inclusive,
                },
                MatchPattern::Binding(_) | MatchPattern::BindingAt { .. } => {
                    return Err(
                        self.error(ExpressionErrorKind::ExpectedMatchPattern, self.lookahead)
                    );
                }
            };
            return Ok(MatchPattern::BindingAt { name, pattern });
        }
        if start.is_some()
            && !self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Dot)
        {
            let start = start.ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: Span {
                    start: self.source_len,
                    end: self.source_len,
                },
            })?;
            if let Some(name) = self
                .nodes
                .get(start.index())
                .and_then(Option::as_ref)
                .and_then(|expression| match expression.kind {
                    ExprKind::Identifier(name) => Some(name),
                    _ => None,
                })
                .filter(|name| {
                    name.as_bytes()
                        .first()
                        .is_some_and(|byte| byte.is_ascii_lowercase())
                })
            {
                return Ok(MatchPattern::Binding(name));
            }
            return Ok(MatchPattern::Value(start));
        }
        self.take()?;
        let second_dot = self.take()?;
        if !second_dot.is_some_and(|token| token.kind == TokenKind::Dot) {
            return Err(self.error(ExpressionErrorKind::ExpectedMatchPattern, second_dot));
        }
        let inclusive = if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Operator && token.text == "=")
        {
            self.take()?;
            true
        } else {
            false
        };
        let end = if !inclusive
            && self.peek()?.is_some_and(|token| {
                token.kind == TokenKind::FatArrow
                    || (token.kind == TokenKind::Operator && token.text == "|")
            }) {
            None
        } else {
            Some(self.match_pattern_bound(depth + 1)?)
        };
        if start.into_iter().chain(end).any(|bound| {
            self.nodes
                .get(bound.index())
                .and_then(Option::as_ref)
                .is_some_and(|expression| matches!(expression.kind, ExprKind::Bool(_)))
        }) {
            let start_span = start
                .and_then(|bound| self.node_span(bound).ok())
                .or_else(|| end.and_then(|bound| self.node_span(bound).ok()));
            let end_span = end
                .and_then(|bound| self.node_span(bound).ok())
                .or(start_span);
            return Err(ExpressionError {
                kind: ExpressionErrorKind::InvalidRangeType,
                span: Span {
                    start: start_span.map_or(0, |span| span.start),
                    end: end_span.map_or(self.source_len, |span| span.end),
                },
            });
        }
        if let (Some(start), Some(end), Some(order)) = (
            start,
            end,
            start
                .and_then(|start| self.pattern_order_value(start))
                .zip(end.and_then(|end| self.pattern_order_value(end)))
                .and_then(|(start, end)| start.compare(end)),
        ) {
            let invalid = if inclusive {
                order == core::cmp::Ordering::Greater
            } else {
                order != core::cmp::Ordering::Less
            };
            if invalid {
                return Err(ExpressionError {
                    kind: ExpressionErrorKind::InvalidRangeBounds,
                    span: Span {
                        start: self.node_span(start)?.start,
                        end: self.node_span(end)?.end,
                    },
                });
            }
        }
        Ok(MatchPattern::Range {
            start,
            end,
            inclusive,
        })
    }

    fn append_match_pattern_alternatives(
        &mut self,
        alternatives: &mut [Option<MatchPattern<'source>>; MAX_MATCH_ALTERNATIVES],
        alternative_count: &mut usize,
        depth: usize,
    ) -> Result<(), ExpressionError> {
        self.append_match_pattern_alternatives_with_binding(
            alternatives,
            alternative_count,
            depth,
            None,
        )
    }

    fn append_match_pattern_alternatives_with_binding(
        &mut self,
        alternatives: &mut [Option<MatchPattern<'source>>; MAX_MATCH_ALTERNATIVES],
        alternative_count: &mut usize,
        depth: usize,
        binding: Option<&'source str>,
    ) -> Result<(), ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        if binding.is_none() {
            let saved_lexer = self.lexer.clone();
            let saved_lookahead = self.lookahead;
            let candidate = self.take()?;
            let at = self.take()?;
            let grouped_binding = candidate
                .filter(|token| token.kind == TokenKind::Identifier && token.text != "_")
                .filter(|_| {
                    at.is_some_and(|token| token.kind == TokenKind::Operator && token.text == "@")
                })
                .filter(|_| {
                    self.peek()
                        .ok()
                        .flatten()
                        .is_some_and(|token| token.kind == TokenKind::OpenParen)
                });
            if let Some(candidate) = grouped_binding {
                return self.append_match_pattern_alternatives_with_binding(
                    alternatives,
                    alternative_count,
                    depth + 1,
                    Some(candidate.text),
                );
            }
            self.lexer = saved_lexer;
            self.lookahead = saved_lookahead;
        }
        if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::OpenParen)
        {
            self.take()?;
            if self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Operator && token.text == "|")
            {
                self.take()?;
            }
            loop {
                if self
                    .peek()?
                    .is_some_and(|token| token.kind == TokenKind::CloseParen)
                {
                    return Err(
                        self.error(ExpressionErrorKind::ExpectedMatchPattern, self.lookahead)
                    );
                }
                self.append_match_pattern_alternatives_with_binding(
                    alternatives,
                    alternative_count,
                    depth + 1,
                    binding,
                )?;
                if !self
                    .peek()?
                    .is_some_and(|token| token.kind == TokenKind::Operator && token.text == "|")
                {
                    break;
                }
                self.take()?;
            }
            let close = self.take()?;
            if !close.is_some_and(|token| token.kind == TokenKind::CloseParen) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseParen, close));
            }
            return Ok(());
        }
        if *alternative_count == MAX_MATCH_ALTERNATIVES {
            return Err(self.error(
                ExpressionErrorKind::TooManyMatchAlternatives,
                self.lookahead,
            ));
        }
        let pattern = self.match_pattern(depth + 1)?;
        alternatives[*alternative_count] = Some(if let Some(name) = binding {
            let pattern = match pattern {
                MatchPattern::Wildcard => MatchPatternBody::Wildcard,
                MatchPattern::Value(value) => MatchPatternBody::Value(value),
                MatchPattern::Range {
                    start,
                    end,
                    inclusive,
                } => MatchPatternBody::Range {
                    start,
                    end,
                    inclusive,
                },
                MatchPattern::Binding(_) | MatchPattern::BindingAt { .. } => {
                    return Err(
                        self.error(ExpressionErrorKind::ExpectedMatchPattern, self.lookahead)
                    );
                }
            };
            MatchPattern::BindingAt { name, pattern }
        } else {
            pattern
        });
        *alternative_count += 1;
        Ok(())
    }

    fn match_pattern_condition(
        &mut self,
        scrutinee: ExprId,
        pattern: MatchPattern<'source>,
    ) -> Result<ExprId, ExpressionError> {
        match pattern {
            MatchPattern::Wildcard => self.push(Expr {
                kind: ExprKind::Bool(true),
                span: self.node_span(scrutinee)?,
            }),
            MatchPattern::Binding(_) => self.push(Expr {
                kind: ExprKind::Bool(true),
                span: self.node_span(scrutinee)?,
            }),
            MatchPattern::BindingAt { pattern, .. } => {
                let pattern = match pattern {
                    MatchPatternBody::Wildcard => MatchPattern::Wildcard,
                    MatchPatternBody::Value(value) => MatchPattern::Value(value),
                    MatchPatternBody::Range {
                        start,
                        end,
                        inclusive,
                    } => MatchPattern::Range {
                        start,
                        end,
                        inclusive,
                    },
                };
                self.match_pattern_condition(scrutinee, pattern)
            }
            MatchPattern::Value(pattern) => self.push(Expr {
                kind: ExprKind::Binary {
                    operator: BinaryOperator::Equal,
                    left: scrutinee,
                    right: pattern,
                },
                span: Span {
                    start: self.node_span(scrutinee)?.start,
                    end: self.node_span(pattern)?.end,
                },
            }),
            MatchPattern::Range {
                start,
                end,
                inclusive,
            } => {
                self.record_range_validation(scrutinee, start, end, inclusive)?;
                match (start, end) {
                    (Some(start), Some(end)) => {
                        let lower = self.push(Expr {
                            kind: ExprKind::Binary {
                                operator: BinaryOperator::GreaterEqual,
                                left: scrutinee,
                                right: start,
                            },
                            span: Span {
                                start: self.node_span(scrutinee)?.start,
                                end: self.node_span(start)?.end,
                            },
                        })?;
                        let upper = self.push(Expr {
                            kind: ExprKind::Binary {
                                operator: if inclusive {
                                    BinaryOperator::LessEqual
                                } else {
                                    BinaryOperator::Less
                                },
                                left: scrutinee,
                                right: end,
                            },
                            span: Span {
                                start: self.node_span(scrutinee)?.start,
                                end: self.node_span(end)?.end,
                            },
                        })?;
                        self.push(Expr {
                            kind: ExprKind::Binary {
                                operator: BinaryOperator::LogicalAnd,
                                left: lower,
                                right: upper,
                            },
                            span: Span {
                                start: self.node_span(lower)?.start,
                                end: self.node_span(upper)?.end,
                            },
                        })
                    }
                    (Some(start), None) => self.push(Expr {
                        kind: ExprKind::Binary {
                            operator: BinaryOperator::GreaterEqual,
                            left: scrutinee,
                            right: start,
                        },
                        span: Span {
                            start: self.node_span(scrutinee)?.start,
                            end: self.node_span(start)?.end,
                        },
                    }),
                    (None, Some(end)) => self.push(Expr {
                        kind: ExprKind::Binary {
                            operator: if inclusive {
                                BinaryOperator::LessEqual
                            } else {
                                BinaryOperator::Less
                            },
                            left: scrutinee,
                            right: end,
                        },
                        span: Span {
                            start: self.node_span(scrutinee)?.start,
                            end: self.node_span(end)?.end,
                        },
                    }),
                    (None, None) => Err(ExpressionError {
                        kind: ExpressionErrorKind::ExpectedMatchPattern,
                        span: self.node_span(scrutinee)?,
                    }),
                }
            }
        }
    }

    fn match_patterns_cover_supported_domain(
        &self,
        patterns: &[Option<MatchArmPattern<'source>>; MAX_MATCH_PATTERNS],
        pattern_count: usize,
    ) -> bool {
        let mut covers_false = false;
        let mut covers_true = false;
        for pattern in patterns[..pattern_count].iter().flatten() {
            if pattern.guard.is_some() {
                continue;
            }
            for alternative in pattern.alternatives[..pattern.alternative_count]
                .iter()
                .flatten()
            {
                let body = match alternative {
                    MatchPattern::Wildcard | MatchPattern::Binding(_) => return true,
                    MatchPattern::BindingAt { pattern, .. } => Some(*pattern),
                    MatchPattern::Value(value) => {
                        if let Some(Expr {
                            kind: ExprKind::Bool(value),
                            ..
                        }) = self.nodes.get(value.index()).and_then(Option::as_ref)
                        {
                            if *value {
                                covers_true = true;
                            } else {
                                covers_false = true;
                            }
                        }
                        None
                    }
                    MatchPattern::Range { .. } => None,
                };
                match body {
                    Some(MatchPatternBody::Wildcard) => return true,
                    Some(MatchPatternBody::Value(value)) => {
                        if let Some(Expr {
                            kind: ExprKind::Bool(value),
                            ..
                        }) = self.nodes.get(value.index()).and_then(Option::as_ref)
                        {
                            if *value {
                                covers_true = true;
                            } else {
                                covers_false = true;
                            }
                        }
                    }
                    Some(MatchPatternBody::Range { .. }) | None => {}
                }
            }
        }
        covers_false && covers_true
    }

    fn match_expression(
        &mut self,
        match_token: Token<'source>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(match_token)));
        }
        let scrutinee = self.expression(0, depth + 1)?;
        let open = self.take()?;
        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        }
        let mut patterns = [None; MAX_MATCH_PATTERNS];
        let mut branches = [None; MAX_MATCH_PATTERNS];
        let mut pattern_count = 0usize;
        let mut exhaustive_fallback = None;
        let mut exhaustive_close = None;
        loop {
            let leading_wildcard = self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Identifier && token.text == "_");
            if leading_wildcard {
                self.take()?.ok_or(ExpressionError {
                    kind: ExpressionErrorKind::InvalidExpressionTree,
                    span: match_token.span,
                })?;
            }
            if leading_wildcard && !self.peek()?.is_some_and(|token| token.text == "if") {
                break;
            }
            if pattern_count == MAX_MATCH_PATTERNS {
                return Err(self.error(ExpressionErrorKind::TooManyMatchPatterns, self.lookahead));
            }
            let mut alternatives = [None; MAX_MATCH_ALTERNATIVES];
            let mut alternative_count = 0usize;
            if leading_wildcard {
                alternatives[0] = Some(MatchPattern::Wildcard);
                alternative_count = 1;
            } else {
                loop {
                    self.append_match_pattern_alternatives(
                        &mut alternatives,
                        &mut alternative_count,
                        depth + 1,
                    )?;
                    if !self
                        .peek()?
                        .is_some_and(|token| token.kind == TokenKind::Operator && token.text == "|")
                    {
                        break;
                    }
                    self.take()?;
                }
            }
            let binding = alternatives[..alternative_count]
                .iter()
                .flatten()
                .find_map(|pattern| match pattern {
                    MatchPattern::Binding(name) | MatchPattern::BindingAt { name, .. } => {
                        Some(*name)
                    }
                    MatchPattern::Wildcard
                    | MatchPattern::Value(_)
                    | MatchPattern::Range { .. } => None,
                });
            if let Some(binding) = binding {
                let consistent = alternatives[..alternative_count].iter().flatten().all(
                    |pattern| match pattern {
                        MatchPattern::Binding(name) | MatchPattern::BindingAt { name, .. } => {
                            *name == binding
                        }
                        MatchPattern::Wildcard
                        | MatchPattern::Value(_)
                        | MatchPattern::Range { .. } => false,
                    },
                );
                if !consistent {
                    return Err(self.error(
                        ExpressionErrorKind::InconsistentMatchBindings,
                        self.lookahead,
                    ));
                }
            }
            let guard = if self.peek()?.is_some_and(|token| token.text == "if") {
                self.take()?;
                Some(self.expression(0, depth + 1)?)
            } else {
                None
            };
            let arrow = self.take()?;
            if !arrow.is_some_and(|token| token.kind == TokenKind::FatArrow) {
                return Err(self.error(ExpressionErrorKind::ExpectedFatArrow, arrow));
            }
            let block_arm = self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::OpenBrace);
            let branch = self.expression(0, depth + 1)?;
            if let Some(binding) = binding {
                let scrutinee_kind = self
                    .nodes
                    .get(scrutinee.index())
                    .and_then(Option::as_ref)
                    .map(|expression| expression.kind)
                    .ok_or(ExpressionError {
                        kind: ExpressionErrorKind::InvalidExpressionTree,
                        span: match_token.span,
                    })?;
                if let Some(guard) = guard {
                    self.substitute_identifier(guard, binding, scrutinee_kind, depth + 1)?;
                }
                self.substitute_identifier(branch, binding, scrutinee_kind, depth + 1)?;
            }
            patterns[pattern_count] = Some(MatchArmPattern {
                alternatives,
                alternative_count,
                guard,
            });
            branches[pattern_count] = Some(branch);
            pattern_count += 1;
            let has_comma = self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Comma);
            if has_comma {
                self.take()?;
            }
            let closes_match = self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::CloseBrace);
            if closes_match {
                if !self.match_patterns_cover_supported_domain(&patterns, pattern_count) {
                    return Err(self.error(ExpressionErrorKind::NonExhaustiveMatch, self.lookahead));
                }
                exhaustive_fallback = branches[pattern_count - 1].take();
                pattern_count -= 1;
                exhaustive_close = self.take()?;
                break;
            }
            if !has_comma && !block_arm {
                return Err(self.error(ExpressionErrorKind::ExpectedComma, self.lookahead));
            }
        }
        let (else_branch, close) =
            if let (Some(else_branch), Some(close)) = (exhaustive_fallback, exhaustive_close) {
                (else_branch, close)
            } else {
                let arrow = self.take()?;
                if !arrow.is_some_and(|token| token.kind == TokenKind::FatArrow) {
                    return Err(self.error(ExpressionErrorKind::ExpectedFatArrow, arrow));
                }
                let else_branch = self.expression(0, depth + 1)?;
                if self
                    .peek()?
                    .is_some_and(|token| token.kind == TokenKind::Comma)
                {
                    self.take()?;
                }
                let close = self.take()?;
                let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
                    return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
                };
                (else_branch, close)
            };
        let mut decision = else_branch;
        for index in (0..pattern_count).rev() {
            let pattern = patterns[index].ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: match_token.span,
            })?;
            let then_branch = branches[index].ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: match_token.span,
            })?;
            let mut condition = None;
            for alternative in pattern.alternatives[..pattern.alternative_count]
                .iter()
                .flatten()
                .copied()
            {
                let alternative = self.match_pattern_condition(scrutinee, alternative)?;
                condition = Some(if let Some(previous) = condition {
                    self.push(Expr {
                        kind: ExprKind::Binary {
                            operator: BinaryOperator::LogicalOr,
                            left: previous,
                            right: alternative,
                        },
                        span: Span {
                            start: self.node_span(previous)?.start,
                            end: self.node_span(alternative)?.end,
                        },
                    })?
                } else {
                    alternative
                });
            }
            let condition = condition.ok_or(ExpressionError {
                kind: ExpressionErrorKind::InvalidExpressionTree,
                span: match_token.span,
            })?;
            let condition = if let Some(guard) = pattern.guard {
                self.push(Expr {
                    kind: ExprKind::Binary {
                        operator: BinaryOperator::LogicalAnd,
                        left: condition,
                        right: guard,
                    },
                    span: Span {
                        start: self.node_span(condition)?.start,
                        end: self.node_span(guard)?.end,
                    },
                })?
            } else {
                condition
            };
            decision = self.push(Expr {
                kind: ExprKind::If {
                    condition,
                    then_branch,
                    else_branch: decision,
                },
                span: Span {
                    start: match_token.span.start,
                    end: close.span.end,
                },
            })?;
        }
        Ok(decision)
    }

    fn scalar_branch_block(&mut self, depth: usize) -> Result<(ExprId, usize), ExpressionError> {
        let mut bindings: [Option<InlineConstBinding<'source>>; MAX_INLINE_CONST_BINDINGS] =
            [None; MAX_INLINE_CONST_BINDINGS];
        let mut binding_count = 0usize;
        let mut assignment_count = 0usize;
        let mut statements = [None; MAX_INLINE_CONST_EXPRESSION_STATEMENTS];
        let mut statement_count = 0usize;
        let tail = loop {
            if let Some(close) = self
                .peek()?
                .filter(|token| token.kind == TokenKind::CloseBrace)
            {
                break self.push(Expr {
                    kind: ExprKind::Unit,
                    span: Span {
                        start: close.span.start,
                        end: close.span.start,
                    },
                })?;
            }
            if self.peek()?.is_some_and(|token| token.text == "let") {
                self.inline_const_binding(&mut bindings, &mut binding_count, depth + 1)?;
                continue;
            }
            if self.inline_const_assignment(
                &mut bindings,
                binding_count,
                &mut assignment_count,
                depth + 1,
            )? {
                continue;
            }
            let expression = self.expression(0, depth + 1)?;
            for (name, replacement, _, _) in bindings[..binding_count].iter().flatten().rev() {
                self.substitute_identifier(expression, name, *replacement, depth + 1)?;
            }
            if self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Semicolon)
            {
                let semicolon = self.take()?;
                if statement_count == MAX_INLINE_CONST_EXPRESSION_STATEMENTS {
                    return Err(self.error(
                        ExpressionErrorKind::TooManyInlineConstExpressionStatements,
                        semicolon,
                    ));
                }
                statements[statement_count] = Some(expression);
                statement_count += 1;
            } else {
                break expression;
            }
        };
        let close = self.take()?;
        let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
            return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
        };
        let mut branch = tail;
        for first in statements[..statement_count].iter().flatten().rev() {
            branch = self.push(Expr {
                kind: ExprKind::Sequence {
                    first: *first,
                    then: branch,
                },
                span: Span {
                    start: self.node_span(*first)?.start,
                    end: self.node_span(branch)?.end,
                },
            })?;
        }
        Ok((branch, close.span.end))
    }

    fn loop_break_expression(
        &mut self,
        loop_token: Token<'source>,
        label: Option<Token<'source>>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(loop_token)));
        }
        let open = self.take()?;
        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        }
        if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Lifetime)
        {
            return self.cross_nested_loop_break_expression(loop_token, label, depth + 1);
        }
        if self.peek()?.is_some_and(|token| token.text == "while") {
            self.take()?;
            return self.while_break_condition_expression(loop_token, label, None, depth + 1);
        }
        let mut conditions = [None; MAX_LOOP_BREAK_BRANCHES];
        let mut branches = [None; MAX_LOOP_BREAK_BRANCHES];
        let mut branch_count = 0usize;
        let mut control = self.take()?;
        let (operand, close) = loop {
            if !control.is_some_and(|token| token.text == "if") {
                if !control.is_some_and(|token| token.text == "break") {
                    return Err(self.error(ExpressionErrorKind::ExpectedExpression, control));
                }
                let break_token = control.ok_or(ExpressionError {
                    kind: ExpressionErrorKind::InvalidExpressionTree,
                    span: Span {
                        start: loop_token.span.start,
                        end: loop_token.span.end,
                    },
                })?;
                self.loop_break_label(label)?;
                let operand = self.loop_break_operand(depth + 1)?;
                if self.peek()?.is_some_and(|token| token.text == "break") {
                    if branch_count == MAX_LOOP_BREAK_BRANCHES {
                        return Err(self.error(
                            ExpressionErrorKind::TooManyLoopBreakBranches,
                            Some(break_token),
                        ));
                    }
                    let condition = self.push(Expr {
                        kind: ExprKind::Bool(true),
                        span: break_token.span,
                    })?;
                    conditions[branch_count] = Some(condition);
                    branches[branch_count] = Some(operand);
                    branch_count += 1;
                    control = self.take()?;
                    continue;
                }
                let close = self.take()?;
                let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
                    return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
                };
                break (operand, close);
            }
            if branch_count == MAX_LOOP_BREAK_BRANCHES {
                return Err(self.error(ExpressionErrorKind::TooManyLoopBreakBranches, control));
            }
            let condition = self.expression(0, depth + 1)?;
            let open = self.take()?;
            if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
            }
            let break_token = self.take()?;
            if !break_token.is_some_and(|token| token.text == "break") {
                return Err(self.error(ExpressionErrorKind::ExpectedExpression, break_token));
            }
            self.loop_break_label(label)?;
            let then_branch = self.loop_break_operand(depth + 1)?;
            let close = self.take()?;
            if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
            }
            conditions[branch_count] = Some(condition);
            branches[branch_count] = Some(then_branch);
            branch_count += 1;
            control = self.take()?;
            if !control.is_some_and(|token| token.text == "else") {
                continue;
            }
            let alternative = self.take()?;
            if alternative.is_some_and(|token| token.text == "if") {
                control = alternative;
                continue;
            }
            if !alternative.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, alternative));
            }
            let fallback_break = self.take()?;
            if !fallback_break.is_some_and(|token| token.text == "break") {
                return Err(self.error(ExpressionErrorKind::ExpectedExpression, fallback_break));
            }
            self.loop_break_label(label)?;
            let operand = self.loop_break_operand(depth + 1)?;
            let alternative_close = self.take()?;
            if !alternative_close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, alternative_close));
            }
            let close = self.take()?;
            let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
            };
            break (operand, close);
        };
        let span = Span {
            start: label.map_or(loop_token.span.start, |label| label.span.start),
            end: close.span.end,
        };
        if branch_count == 0 {
            return self.push(Expr {
                kind: ExprKind::LoopBreak { operand },
                span,
            });
        }
        let mut result = operand;
        for index in (0..branch_count).rev() {
            result = self.push(Expr {
                kind: ExprKind::LoopBreakIf {
                    condition: conditions[index].ok_or(ExpressionError {
                        kind: ExpressionErrorKind::InvalidExpressionTree,
                        span,
                    })?,
                    then_branch: branches[index].ok_or(ExpressionError {
                        kind: ExpressionErrorKind::InvalidExpressionTree,
                        span,
                    })?,
                    else_branch: result,
                },
                span,
            })?;
        }
        Ok(result)
    }

    fn cross_nested_loop_break_expression(
        &mut self,
        outer_loop: Token<'source>,
        outer_label: Option<Token<'source>>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(outer_loop)));
        }
        let inner_label = self
            .take()?
            .ok_or_else(|| self.error(ExpressionErrorKind::ExpectedExpression, self.lookahead))?;
        let colon = self.take()?;
        if !colon.is_some_and(|token| token.kind == TokenKind::Colon) {
            return Err(self.error(ExpressionErrorKind::ExpectedColon, colon));
        }
        let inner_loop = self.take()?;
        if inner_loop.is_some_and(|token| token.text == "while") {
            return self.while_break_condition_expression(
                outer_loop,
                outer_label,
                Some(inner_label),
                depth + 1,
            );
        }
        if !inner_loop.is_some_and(|token| token.text == "loop") {
            return Err(self.error(ExpressionErrorKind::ExpectedExpression, inner_loop));
        }
        let open = self.take()?;
        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        }

        let mut conditions = [None; MAX_LOOP_BREAK_BRANCHES];
        let mut outer_branches = [None; MAX_LOOP_BREAK_BRANCHES];
        let mut branch_count = 0usize;
        let mut control = self.take()?;
        let inner_operand = loop {
            if !control.is_some_and(|token| token.text == "if") {
                if !control.is_some_and(|token| token.text == "break") {
                    return Err(self.error(ExpressionErrorKind::ExpectedExpression, control));
                }
                if self.value_loop_break_target(Some(inner_label), outer_label)?
                    != ValueLoopControlTarget::Current
                {
                    return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, control));
                }
                let operand = self.loop_break_operand(depth + 1)?;
                let close = self.take()?;
                if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                    return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
                }
                break operand;
            }
            if branch_count == MAX_LOOP_BREAK_BRANCHES {
                return Err(self.error(ExpressionErrorKind::TooManyLoopBreakBranches, control));
            }
            let condition = self.expression(0, depth + 1)?;
            let open = self.take()?;
            if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
            }
            let break_token = self.take()?;
            if !break_token.is_some_and(|token| token.text == "break") {
                return Err(self.error(ExpressionErrorKind::ExpectedExpression, break_token));
            }
            if self.value_loop_break_target(Some(inner_label), outer_label)?
                != ValueLoopControlTarget::Enclosing
            {
                return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, break_token));
            }
            let branch = self.loop_break_operand(depth + 1)?;
            let close = self.take()?;
            if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
            }
            conditions[branch_count] = Some(condition);
            outer_branches[branch_count] = Some(branch);
            branch_count += 1;
            control = self.take()?;
            if !control.is_some_and(|token| token.text == "else") {
                continue;
            }
            let alternative = self.take()?;
            if alternative.is_some_and(|token| token.text == "if") {
                control = alternative;
                continue;
            }
            if !alternative.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, alternative));
            }
            let break_token = self.take()?;
            if !break_token.is_some_and(|token| token.text == "break") {
                return Err(self.error(ExpressionErrorKind::ExpectedExpression, break_token));
            }
            if self.value_loop_break_target(Some(inner_label), outer_label)?
                != ValueLoopControlTarget::Current
            {
                return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, break_token));
            }
            let operand = self.loop_break_operand(depth + 1)?;
            let alternative_close = self.take()?;
            if !alternative_close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, alternative_close));
            }
            let inner_close = self.take()?;
            if !inner_close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, inner_close));
            }
            break operand;
        };

        let semicolon = self.take()?;
        if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
            return Err(self.error(ExpressionErrorKind::TrailingToken, semicolon));
        }
        let outer_break = self.take()?;
        if !outer_break.is_some_and(|token| token.text == "break") {
            return Err(self.error(ExpressionErrorKind::ExpectedExpression, outer_break));
        }
        if self.value_loop_break_target(outer_label, None)? != ValueLoopControlTarget::Current {
            return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, outer_break));
        }
        let fallback = self.loop_break_operand(depth + 1)?;
        let close = self.take()?;
        let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
            return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
        };
        let span = Span {
            start: outer_label.map_or(outer_loop.span.start, |label| label.span.start),
            end: close.span.end,
        };
        let mut result = self.push(Expr {
            kind: ExprKind::Sequence {
                first: inner_operand,
                then: fallback,
            },
            span,
        })?;
        for index in (0..branch_count).rev() {
            result = self.push(Expr {
                kind: ExprKind::LoopBreakIf {
                    condition: conditions[index].ok_or(ExpressionError {
                        kind: ExpressionErrorKind::InvalidExpressionTree,
                        span,
                    })?,
                    then_branch: outer_branches[index].ok_or(ExpressionError {
                        kind: ExpressionErrorKind::InvalidExpressionTree,
                        span,
                    })?,
                    else_branch: result,
                },
                span,
            })?;
        }
        Ok(result)
    }

    fn while_break_condition_expression(
        &mut self,
        outer_loop: Token<'source>,
        outer_label: Option<Token<'source>>,
        inner_label: Option<Token<'source>>,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(outer_loop)));
        }
        let break_token = self.take()?;
        if !break_token.is_some_and(|token| token.text == "break") {
            return Err(self.error(ExpressionErrorKind::ExpectedExpression, break_token));
        }
        let target = self.value_loop_break_target(inner_label, outer_label)?;
        let operand = self.loop_break_condition_operand(depth + 1)?;
        if target == ValueLoopControlTarget::Current && !self.node_is_unit(operand, depth + 1) {
            return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, break_token));
        }
        let open = self.take()?;
        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
            return Err(self.error(ExpressionErrorKind::ExpectedOpenBrace, open));
        }
        self.skip_unreachable_brace_body()?;

        let span_start = outer_label.map_or(outer_loop.span.start, |label| label.span.start);
        if target == ValueLoopControlTarget::Enclosing {
            let close = self.take()?;
            let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
            };
            return self.push(Expr {
                kind: ExprKind::LoopBreak { operand },
                span: Span {
                    start: span_start,
                    end: close.span.end,
                },
            });
        }

        let outer_break = self.take()?;
        if !outer_break.is_some_and(|token| token.text == "break") {
            return Err(self.error(ExpressionErrorKind::ExpectedExpression, outer_break));
        }
        if self.value_loop_break_target(outer_label, None)? != ValueLoopControlTarget::Current {
            return Err(self.error(ExpressionErrorKind::InvalidLoopBreakTarget, outer_break));
        }
        let fallback = self.loop_break_operand(depth + 1)?;
        let close = self.take()?;
        let Some(close) = close.filter(|token| token.kind == TokenKind::CloseBrace) else {
            return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, close));
        };
        self.push(Expr {
            kind: ExprKind::Sequence {
                first: operand,
                then: fallback,
            },
            span: Span {
                start: span_start,
                end: close.span.end,
            },
        })
    }

    fn loop_break_condition_operand(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if let Some(open) = self
            .peek()?
            .filter(|token| token.kind == TokenKind::OpenBrace)
        {
            return self.push(Expr {
                kind: ExprKind::Unit,
                span: Span {
                    start: open.span.start,
                    end: open.span.start,
                },
            });
        }
        self.expression(0, depth + 1)
    }

    fn skip_unreachable_brace_body(&mut self) -> Result<(), ExpressionError> {
        let mut depth = 1usize;
        while depth != 0 {
            let token = self.take()?;
            let Some(token) = token else {
                return Err(self.error(ExpressionErrorKind::ExpectedCloseBrace, None));
            };
            if token.kind == TokenKind::OpenBrace {
                if depth == MAX_EXPRESSION_DEPTH {
                    return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, Some(token)));
                }
                depth += 1;
            } else if token.kind == TokenKind::CloseBrace {
                depth -= 1;
            }
        }
        Ok(())
    }

    fn loop_break_operand(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if let Some(semicolon) = self
            .peek()?
            .filter(|token| token.kind == TokenKind::Semicolon)
        {
            self.take()?;
            return self.push(Expr {
                kind: ExprKind::Unit,
                span: Span {
                    start: semicolon.span.start,
                    end: semicolon.span.start,
                },
            });
        }
        let operand = self.expression(0, depth + 1)?;
        let semicolon = self.take()?;
        if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
            return Err(self.error(ExpressionErrorKind::TrailingToken, semicolon));
        }
        Ok(operand)
    }

    fn nested_loop_break_operand(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if let Some(semicolon) = self
            .peek()?
            .filter(|token| token.kind == TokenKind::Semicolon)
        {
            return self.push(Expr {
                kind: ExprKind::Unit,
                span: Span {
                    start: semicolon.span.start,
                    end: semicolon.span.start,
                },
            });
        }
        self.expression(0, depth + 1)
    }

    fn loop_break_label(
        &mut self,
        expected: Option<Token<'source>>,
    ) -> Result<(), ExpressionError> {
        let _ = self.value_loop_break_target(expected, None)?;
        Ok(())
    }

    fn value_loop_break_target(
        &mut self,
        current: Option<Token<'source>>,
        enclosing: Option<Token<'source>>,
    ) -> Result<ValueLoopControlTarget, ExpressionError> {
        let Some(actual) = self
            .peek()?
            .filter(|token| token.kind == TokenKind::Lifetime)
        else {
            return Ok(ValueLoopControlTarget::Current);
        };
        self.take()?;
        if current.is_some_and(|expected| expected.text == actual.text) {
            Ok(ValueLoopControlTarget::Current)
        } else if enclosing.is_some_and(|expected| expected.text == actual.text) {
            Ok(ValueLoopControlTarget::Enclosing)
        } else {
            Err(self.error(ExpressionErrorKind::UnknownLoopLabel, Some(actual)))
        }
    }

    fn unary(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        if let Some(token) = self.peek()?
            && matches!(token.text, "-" | "!")
        {
            self.take()?;
            let operator = if token.text == "-" {
                UnaryOperator::Negate
            } else {
                UnaryOperator::Not
            };
            let operand = self.unary(depth + 1)?;
            let end = self.node_span(operand)?.end;
            return self.push(Expr {
                kind: ExprKind::Unary { operator, operand },
                span: Span {
                    start: token.span.start,
                    end,
                },
            });
        }
        self.atom(depth + 1)
    }

    fn cast(&mut self, depth: usize) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        let mut operand = self.unary(depth + 1)?;
        while self.peek()?.is_some_and(|token| token.text == "as") {
            self.take()?;
            let target = self.take()?;
            let Some(target) = target else {
                return Err(self.error(ExpressionErrorKind::ExpectedCastType, None));
            };
            if target.kind != TokenKind::Identifier {
                return Err(self.error(ExpressionErrorKind::ExpectedCastType, Some(target)));
            }
            let Some(integer_type) = IntegerType::from_name(target.text) else {
                return Err(self.error(ExpressionErrorKind::UnsupportedCastType, Some(target)));
            };
            let span = Span {
                start: self.node_span(operand)?.start,
                end: target.span.end,
            };
            operand = self.push(Expr {
                kind: ExprKind::Cast {
                    operand,
                    target: integer_type,
                },
                span,
            })?;
        }
        Ok(operand)
    }

    fn expression(
        &mut self,
        minimum_precedence: u8,
        depth: usize,
    ) -> Result<ExprId, ExpressionError> {
        if depth == MAX_EXPRESSION_DEPTH {
            return Err(self.error(ExpressionErrorKind::NestingLimitExceeded, self.lookahead));
        }
        let mut left = self.cast(depth + 1)?;
        while let Some(token) = self.peek()? {
            let Some((operator, precedence)) = binary_operator(token.text) else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            self.take()?;
            let right = self.expression(precedence + 1, depth + 1)?;
            let span = Span {
                start: self.node_span(left)?.start,
                end: self.node_span(right)?.end,
            };
            left = self.push(Expr {
                kind: ExprKind::Binary {
                    operator,
                    left,
                    right,
                },
                span,
            })?;
        }
        Ok(left)
    }

    pub fn parse(mut self) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        let root = if self.peek()?.is_some_and(|token| token.text == "return") {
            let keyword = self.take()?.ok_or_else(|| {
                self.error(ExpressionErrorKind::ExpectedExpression, self.lookahead)
            })?;
            let operand = self.expression(0, 1)?;
            let end = self.node_span(operand)?.end;
            if self
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::Semicolon)
            {
                self.take()?;
            }
            self.push(Expr {
                kind: ExprKind::Return { operand },
                span: Span {
                    start: keyword.span.start,
                    end,
                },
            })?
        } else {
            self.expression(0, 0)?
        };
        if let Some(token) = self.take()? {
            return Err(self.error(ExpressionErrorKind::TrailingToken, Some(token)));
        }
        Ok(ExpressionTree {
            nodes: self.nodes,
            node_count: self.node_count,
            range_validations: self.range_validations,
            range_validation_count: self.range_validation_count,
            root,
        })
    }
}

fn binary_operator(text: &str) -> Option<(BinaryOperator, u8)> {
    Some(match text {
        "||" => (BinaryOperator::LogicalOr, 1),
        "&&" => (BinaryOperator::LogicalAnd, 2),
        "|" => (BinaryOperator::BitOr, 3),
        "^" => (BinaryOperator::BitXor, 4),
        "&" => (BinaryOperator::BitAnd, 5),
        "==" => (BinaryOperator::Equal, 6),
        "!=" => (BinaryOperator::NotEqual, 6),
        "<" => (BinaryOperator::Less, 7),
        "<=" => (BinaryOperator::LessEqual, 7),
        ">" => (BinaryOperator::Greater, 7),
        ">=" => (BinaryOperator::GreaterEqual, 7),
        "<<" => (BinaryOperator::ShiftLeft, 8),
        ">>" => (BinaryOperator::ShiftRight, 8),
        "+" => (BinaryOperator::Add, 9),
        "-" => (BinaryOperator::Subtract, 9),
        "*" => (BinaryOperator::Multiply, 10),
        "/" => (BinaryOperator::Divide, 10),
        "%" => (BinaryOperator::Remainder, 10),
        _ => return None,
    })
}

fn decode_integer(token: Token<'_>) -> Result<IntegerLiteral<'_>, ExpressionError> {
    let bytes = token.text.as_bytes();
    let (radix, mut position) = match bytes {
        [b'0', b'x' | b'X', ..] => (16u32, 2usize),
        [b'0', b'o' | b'O', ..] => (8, 2),
        [b'0', b'b' | b'B', ..] => (2, 2),
        _ => (10, 0),
    };
    let mut value = 0u128;
    let mut digit_count = 0usize;
    while let Some(&byte) = bytes.get(position) {
        if byte == b'_' {
            position += 1;
            continue;
        }
        let Some(digit) = (byte as char).to_digit(radix) else {
            break;
        };
        value = value
            .checked_mul(u128::from(radix))
            .and_then(|current| current.checked_add(u128::from(digit)))
            .ok_or(ExpressionError {
                kind: ExpressionErrorKind::IntegerOverflow,
                span: token.span,
            })?;
        digit_count += 1;
        position += 1;
    }
    if digit_count == 0 {
        return Err(ExpressionError {
            kind: ExpressionErrorKind::InvalidInteger,
            span: token.span,
        });
    }
    let suffix = &token.text[position..];
    if !suffix.is_empty()
        && !matches!(
            suffix,
            "u8" | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
        )
    {
        return Err(ExpressionError {
            kind: ExpressionErrorKind::InvalidIntegerSuffix,
            span: token.span,
        });
    }
    Ok(IntegerLiteral {
        value,
        suffix: (!suffix.is_empty()).then_some(suffix),
    })
}

fn decode_character(token: Token<'_>) -> Result<u32, ExpressionError> {
    let invalid = || ExpressionError {
        kind: ExpressionErrorKind::InvalidCharacter,
        span: token.span,
    };
    let inner = token
        .text
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
        .ok_or_else(invalid)?;
    if let Some(escaped) = inner.strip_prefix('\\') {
        return match escaped {
            "0" => Ok(0),
            "t" => Ok(u32::from(b'\t')),
            "n" => Ok(u32::from(b'\n')),
            "r" => Ok(u32::from(b'\r')),
            "\\" => Ok(u32::from(b'\\')),
            "'" => Ok(u32::from(b'\'')),
            "\"" => Ok(u32::from(b'\"')),
            value if value.starts_with('x') && value.len() == 3 => {
                u32::from_str_radix(&value[1..], 16).map_err(|_| invalid())
            }
            value if value.starts_with("u{") && value.ends_with('}') => {
                let scalar =
                    u32::from_str_radix(&value[2..value.len() - 1], 16).map_err(|_| invalid())?;
                char::from_u32(scalar).map(u32::from).ok_or_else(invalid)
            }
            _ => Err(invalid()),
        };
    }
    let mut characters = inner.chars();
    let character = characters.next().ok_or_else(invalid)?;
    if characters.next().is_some() {
        return Err(invalid());
    }
    Ok(u32::from(character))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(source: &str) -> Result<u128, ConstEvalError> {
        ExpressionParser::<64>::new(source)
            .parse()
            .unwrap()
            .evaluate(&NoConstants)
    }

    #[test]
    fn honors_rust_operator_precedence_and_parentheses() {
        assert_eq!(evaluate("2 + 3 * 4"), Ok(14));
        assert_eq!(evaluate("(2 + 3) * 4"), Ok(20));
        assert_eq!(evaluate("1 << 4 + 1"), Ok(32));
        assert_eq!(evaluate("1 | 2 == 3 && 7 > 2"), Ok(1));
    }

    #[test]
    fn decodes_radices_separators_and_suffixes() {
        assert_eq!(evaluate("0xff_u16 + 0b1_010 + 0o7"), Ok(272));
        let typed = ExpressionParser::<2>::new("10_usize").parse().unwrap();
        assert!(matches!(
            typed
                .expression(typed.root())
                .map(|expression| expression.kind),
            Some(ExprKind::Integer(IntegerLiteral {
                value: 10,
                suffix: Some("usize")
            }))
        ));
        let error = ExpressionParser::<8>::new("12bananas").parse().unwrap_err();
        assert_eq!(error.kind, ExpressionErrorKind::InvalidIntegerSuffix);
    }

    #[test]
    fn parses_and_evaluates_integer_casts_with_rust_precedence() {
        assert_eq!(evaluate("300 as u8"), Ok(44));
        assert_eq!(evaluate("255 as i8 as u16"), Ok(u128::from(u16::MAX)));
        assert_eq!(evaluate("1 + 2 as u8 * 3"), Ok(7));
        assert_eq!(
            evaluate("-1000isize as usize >> 3usize"),
            Ok(u128::from(u64::MAX - 999) >> 3)
        );
        let unsupported = ExpressionParser::<8>::new("1 as bool").parse().unwrap_err();
        assert_eq!(unsupported.kind, ExpressionErrorKind::UnsupportedCastType);
        let missing = ExpressionParser::<8>::new("1 as").parse().unwrap_err();
        assert_eq!(missing.kind, ExpressionErrorKind::ExpectedCastType);
    }

    #[test]
    fn resolves_named_constants() {
        struct Resolver;
        impl ConstantResolver for Resolver {
            fn resolve(&self, name: &str) -> Option<u128> {
                (name == "PAGE_SIZE").then_some(4096)
            }
        }
        let tree = ExpressionParser::<8>::new("PAGE_SIZE * 2").parse().unwrap();
        assert_eq!(tree.evaluate(&Resolver), Ok(8192));
        let unknown = ExpressionParser::<2>::new("UNKNOWN").parse().unwrap();
        assert_eq!(
            unknown.evaluate(&Resolver),
            Err(ConstEvalError::UnknownIdentifier)
        );
    }

    #[test]
    fn reports_arithmetic_failures_without_panicking() {
        assert_eq!(evaluate("1 / 0"), Err(ConstEvalError::DivisionByZero));
        assert_eq!(evaluate("0 - 1"), Err(ConstEvalError::Overflow));
        assert_eq!(evaluate("1 << 128"), Err(ConstEvalError::InvalidShift));
        assert_eq!(
            evaluate("340282366920938463463374607431768211455 + 1"),
            Err(ConstEvalError::Overflow)
        );
    }

    #[test]
    fn short_circuits_logical_operations() {
        assert_eq!(evaluate("0 && UNKNOWN"), Ok(0));
        assert_eq!(evaluate("1 || UNKNOWN"), Ok(1));
        assert_eq!(evaluate("false && UNKNOWN"), Ok(0));
        assert_eq!(evaluate("true || UNKNOWN"), Ok(1));
        assert_eq!(evaluate("!false"), Ok(1));
    }

    #[test]
    fn parses_and_evaluates_if_expressions_lazily() {
        assert_eq!(evaluate("if 2 < 3 { 40 + 2 } else { 1 / 0 }"), Ok(42));
        assert_eq!(evaluate("if false { UNKNOWN } else { 7 }"), Ok(7));
        assert_eq!(evaluate("if true { false } else { true }"), Ok(0));
        struct Resolver;
        impl ConstantResolver for Resolver {
            fn resolve(&self, name: &str) -> Option<u128> {
                match name {
                    "X" => Some(4),
                    "Y" => Some(5),
                    _ => None,
                }
            }
        }
        let upstream = ExpressionParser::<24>::new("if X < Y { Y - X } else { X - Y }")
            .parse()
            .unwrap();
        assert_eq!(upstream.evaluate(&Resolver), Ok(1));

        let missing_else = ExpressionParser::<16>::new("if true { 1 }")
            .parse()
            .unwrap_err();
        assert_eq!(missing_else.kind, ExpressionErrorKind::ExpectedElse);
        let missing_brace = ExpressionParser::<16>::new("if true 1 else { 2 }")
            .parse()
            .unwrap_err();
        assert_eq!(missing_brace.kind, ExpressionErrorKind::ExpectedOpenBrace);
    }

    #[test]
    fn parses_explicit_tail_return_only() {
        assert_eq!(evaluate("return 40 + 2;"), Ok(42));
        assert_eq!(evaluate("return true"), Ok(1));
        let nested = ExpressionParser::<16>::new("1 + return 2")
            .parse()
            .unwrap_err();
        assert_eq!(nested.kind, ExpressionErrorKind::TrailingToken);
        assert!(matches!(
            ExpressionParser::<16>::new("if true { return 1 } else { 2 }").parse(),
            Err(ExpressionError {
                kind: ExpressionErrorKind::ExpectedCloseBrace,
                ..
            })
        ));
    }

    #[test]
    fn parses_immediate_break_loop_values() {
        assert_eq!(evaluate("loop { break 13; }"), Ok(13));
        assert_eq!(evaluate("loop { break; }"), Ok(0));
        let unit = ExpressionParser::<8>::new("'value: loop { break 'value; }")
            .parse()
            .unwrap();
        assert_eq!(unit.evaluate(&NoConstants), Ok(0));
        assert_eq!(
            evaluate(
                "'outer: loop { break 'outer 'inner: loop { if false { break 'inner 1 / 0; } else { break 'inner 42; } }; }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "'outer: loop { 'inner: loop { if true { break 'outer 42; } else { break 'inner 17; } }; break 'outer 99; }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "'outer: loop { 'inner: loop { if false { break 'outer 1 / 0; } break 'inner 17; }; break 'outer 99; }"
            ),
            Ok(99)
        );
        assert_eq!(
            evaluate(
                "'outer: loop { 'inner: while break 'inner { 1 / 0; { 2 / 0; } } break 'outer 123; }"
            ),
            Ok(123)
        );
        assert_eq!(
            evaluate("'outer: loop { while break 'outer 567 { 1 / 0; } }"),
            Ok(567)
        );
        assert_eq!(
            ExpressionParser::<16>::new(
                "'outer: loop { 'inner: while break 'inner 1 { } break 'outer 2; }"
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::InvalidLoopBreakTarget
        );
        assert_eq!(evaluate("'value: loop { break 'value 13; }"), Ok(13));
        assert_eq!(evaluate("loop { break (); break; }"), Ok(0));
        assert_eq!(evaluate("loop { break; break (); }"), Ok(0));
        assert_eq!(evaluate("loop { break 42; break 1 / 0; }"), Ok(42));
        assert_eq!(
            evaluate("loop { if true { break; } else { break break Default::default(); } }"),
            Ok(0)
        );
        assert_eq!(
            evaluate("loop { if true { break Default::default(); } else { break; } }"),
            Ok(0)
        );
        assert_eq!(
            evaluate("loop { break if true { Default::default() } else { break; }; }"),
            Ok(0)
        );
        assert_eq!(evaluate("Default::default()"), Ok(0));
        for source in ["Default::other()", "Default::default(1)"] {
            assert!(ExpressionParser::<8>::new(source).parse().is_err());
        }
        assert_eq!(
            ExpressionParser::<24>::new(
                "loop { break 1; break 2; break 3; break 4; break 5; break 6; }"
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::TooManyLoopBreakBranches
        );
        let boolean = ExpressionParser::<8>::new("loop { break true; }")
            .parse()
            .unwrap();
        assert!(boolean.is_boolean_expression(boolean.root(), 0));
        assert_eq!(boolean.evaluate(&NoConstants), Ok(1));

        let missing_semicolon = ExpressionParser::<8>::new("loop { break 13 }")
            .parse()
            .unwrap_err();
        assert_eq!(missing_semicolon.kind, ExpressionErrorKind::TrailingToken);
        let missing_break = ExpressionParser::<8>::new("loop { 13 }")
            .parse()
            .unwrap_err();
        assert_eq!(missing_break.kind, ExpressionErrorKind::ExpectedExpression);

        assert_eq!(
            evaluate("loop { if true { break 13; } break 1 / 0; }"),
            Ok(13)
        );
        assert_eq!(
            evaluate("loop { if false { break 1 / 0; } break 42; }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("'value: loop { if false { break 'value 1 / 0; } break 'value 42; }"),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "'value: loop { if false { break 'value 1 / 0; } if true { break 'value 42; } if true { break 'value 1 / 0; } break 'value 1 / 0; }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "loop { if false { break 1; } if false { break 2; } if false { break 3; } if true { break 4; } break 5; }"
            ),
            Ok(4)
        );
        assert_eq!(
            evaluate(
                "'value: loop { if false { break 'value 1 / 0; } else if true { break 'value 42; } else { break 'value 1 / 0; } }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate("loop { if true { break 13; } else { break 1 / 0; } }"),
            Ok(13)
        );
        let mixed_unit = ExpressionParser::<24>::new(
            "'value: loop { if false { break 'value 1 / 0; } else if true { break 'value; } else { break 'value (); } }",
        )
        .parse()
        .unwrap();
        assert_eq!(mixed_unit.evaluate(&NoConstants), Ok(0));
        assert_eq!(
            ExpressionParser::<32>::new(
                "loop { if false { break 1; } if false { break 2; } if false { break 3; } if false { break 4; } if true { break 5; } break 6; }"
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::TooManyLoopBreakBranches
        );
        assert_eq!(
            ExpressionParser::<16>::new("loop { if true { break 1; } else break 2; }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedOpenBrace
        );
        assert_eq!(
            ExpressionParser::<8>::new("'value loop { break 13; }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedColon
        );
        for source in [
            "loop { break 'value 13; }",
            "loop { break 'value; }",
            "'value: loop { break 'other 13; }",
            "'value: loop { if true { break 'other 13; } break 42; }",
        ] {
            assert_eq!(
                ExpressionParser::<16>::new(source)
                    .parse()
                    .unwrap_err()
                    .kind,
                ExpressionErrorKind::UnknownLoopLabel
            );
        }
        assert_eq!(
            ExpressionParser::<24>::new(
                "'outer: loop { 'inner: loop { if true { break 'inner 13; } break 'inner 17; }; break 'outer 42; }",
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::InvalidLoopBreakTarget
        );
        let conditional_bool =
            ExpressionParser::<16>::new("loop { if true { break false; } break true; }")
                .parse()
                .unwrap();
        assert!(conditional_bool.is_boolean_expression(conditional_bool.root(), 0));
    }

    #[test]
    fn parses_unit_values() {
        assert_eq!(evaluate("()"), Ok(0));
        assert_eq!(evaluate("if true { () } else { () }"), Ok(0));
        assert_eq!(evaluate("loop { break (); }"), Ok(0));
        let tree = ExpressionParser::<8>::new("()").parse().unwrap();
        assert!(!tree.is_boolean_expression(tree.root(), 0));
        assert_eq!(evaluate("true == false"), Ok(0));
        assert_eq!(evaluate("true != false"), Ok(1));
        assert_eq!(evaluate("() == ()"), Ok(1));
        assert_eq!(evaluate("false < true"), Ok(1));
        assert_eq!(evaluate("true >= false"), Ok(1));
        assert_eq!(evaluate("() <= ()"), Ok(1));
        assert_eq!(evaluate("true & false"), Ok(0));
        assert_eq!(evaluate("true | false"), Ok(1));
        assert_eq!(evaluate("true ^ true"), Ok(0));
        assert_eq!(
            evaluate("false & (1 / 0 > 0)"),
            Err(ConstEvalError::DivisionByZero)
        );
        assert_eq!(
            evaluate("true | (1 / 0 > 0)"),
            Err(ConstEvalError::DivisionByZero)
        );
    }

    #[test]
    fn parses_scalar_block_expressions() {
        assert_eq!(evaluate("{ 40 + 2 }"), Ok(42));
        assert_eq!(evaluate("{{ 40 + 2 }}"), Ok(42));
        assert_eq!(
            evaluate("if { true } { { 42 } } else { { 1 / 0 } }"),
            Ok(42)
        );
        assert_eq!(evaluate("{}"), Ok(0));
        assert_eq!(evaluate("{ let value = 42; value }"), Ok(42));
        assert_eq!(
            evaluate("{ let mut value: u8 = 40; value += 2; value }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("{ let outer = 40; { let outer = outer + 2; outer } }"),
            Ok(42)
        );
        assert_eq!(evaluate("{ let value = 42; value; }"), Ok(0));
        assert_eq!(
            evaluate("{ { let local = 42; local }; local }"),
            Err(ConstEvalError::UnknownIdentifier)
        );
        assert_eq!(
            evaluate("if false { { let invalid = 1 / 0; invalid } } else { 42 }"),
            Ok(42)
        );
        let empty = ExpressionParser::<2>::new("{}").parse().unwrap();
        assert!(matches!(
            empty
                .expression(empty.root())
                .map(|expression| expression.kind),
            Some(ExprKind::Unit)
        ));
        assert_eq!(
            ExpressionParser::<8>::new("{ 42").parse().unwrap_err().kind,
            ExpressionErrorKind::ExpectedCloseBrace
        );
    }

    #[test]
    fn parses_bounded_scalar_match_expressions() {
        assert_eq!(evaluate("match 7 { 7 => 42, _ => 1 / 0 }"), Ok(42));
        assert_eq!(evaluate("match 8 { 7 => 1 / 0, _ => 42, }"), Ok(42));
        assert_eq!(
            evaluate("match 7 { 7 => { let value = 42; value } _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match -7i8 { -7i8 => { let value = 40; value + 2 }, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match false { true => 1 / 0, _ => 42 }"), Ok(42));
        assert_eq!(
            evaluate("match false { true => 1 / 0, false => 42 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match true { false => 1 / 0, true => 42, }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 42 { value => value }"), Ok(42));
        assert_eq!(
            evaluate("match true { selected @ (false | true) => if selected { 42 } else { 0 } }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 42 { selected @ (1 | _) => selected }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 3 { 1 => 1 / 0, 2 => 1 / 0, 3 => 42, 4 => 1 / 0, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "match 8 { 1 => 1 / 0, 2 => 1 / 0, 3 => 1 / 0, 4 => 1 / 0, 5 => 1 / 0, 6 => 1 / 0, 7 => 1 / 0, 8 => 42, _ => 0 }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "match 8 { 1 if 1 / 0 > 0 => 0, 2 => 1 / 0, 3 => 1 / 0, 4 => 1 / 0, 5 => 1 / 0, 6 => 1 / 0, 7 => 1 / 0, 8 if true => 42, _ => 0 }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 9 { 1 => 1 / 0, 2 => 1 / 0, 3 => 1 / 0, _ => 42 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 5 { 1..=5 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 1 { 1..5 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 5 { 1..5 => 1 / 0, _ => 42 }"), Ok(42));
        assert_eq!(
            evaluate("match 5 { 1 => 1 / 0, 2..=6 => 42, _ => 1 / 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 0 { 0 | 1..=10 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 5 { 0 | 1..=10 => 42, _ => 0 }"), Ok(42));
        assert_eq!(
            evaluate("match 12 { 0 | 1..=10 => 1 / 0, 12 | 13 => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 2 { (1 | 2) => 42, _ => 0 }"), Ok(42));
        assert_eq!(
            evaluate("match 3 { ((1 | 2) | (3 | 4)) => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 9 { (| 1 | (5..=10)) if true => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 99 { (1 | (_)) if false => 1 / 0, _ => 42 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 9 { 5.. => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 4 { ..5 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 5 { ..5 => 1 / 0, _ => 42 }"), Ok(42));
        assert_eq!(evaluate("match 5 { ..=5 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 42 { 42..=42 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 41 { 41..42 => 42, _ => 0 }"), Ok(42));
        assert_eq!(evaluate("match 7 { ..3 | 7.. => 42, _ => 0 }"), Ok(42));
        assert_eq!(
            evaluate("match 1 { 1 if false => 1 / 0, 1..=2 => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 2 { 1 if 1 / 0 > 0 => 0, 2 if true => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 3 { 1..=3 if true => 42, _ => 1 / 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 5 { _ => 42 }"), Ok(42));
        assert_eq!(
            evaluate("match 5 { _ if false => 1 / 0, 5 => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 5 { _ if true => 42, 5 => 1 / 0, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 0 { _ if false && 1 / 0 > 0 => 0, _ if true => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 10 { x if x < 7 => 1 / 0, y if y < 11 => y + 32, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match '\\u{3bb}' { scalar if scalar == '\\u{3bb}' => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 50 { number @ 1..=100 if number == 50 => number - 8, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 42 { value @ _ if value == 42 => value, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match '\u{6d}' { symbol @ 'a'..='z' => symbol, _ => '*' }"),
            Ok(109)
        );
        assert_eq!(
            evaluate("match 2 { selected @ 1 | selected @ 2 => selected + 40, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 7 { value @ 1..=3 | value @ 7..=9 if value == 7 => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 2 { selected @ (1 | 2) => selected + 40, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 7 { selected @ ((1..=3) | (7 | 9)) if selected == 7 => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match 42 { selected @ (| 1 | _) if selected == 42 => selected, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("match 'c' { 'a'..='z' => 42, _ => 0 }"), Ok(42));
        assert_eq!(
            evaluate("match '\\n' { '\\t' | '\\n' => 42, _ => 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("match '\\u{3bb}' { '\u{3b1}'..='\u{3c9}' => 42, _ => 0 }"),
            Ok(42)
        );

        let missing_fallback = ExpressionParser::<16>::new("match 1 { 1 => 42 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            missing_fallback.kind,
            ExpressionErrorKind::NonExhaustiveMatch
        );
        let missing_wildcard = ExpressionParser::<24>::new("match 1 { 1 => 42, 2 => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            missing_wildcard.kind,
            ExpressionErrorKind::NonExhaustiveMatch
        );
        for source in [
            "match true { true => 42 }",
            "match false { true => 1, false if false => 2 }",
            "match true { true if true => 42, false => 0 }",
        ] {
            assert_eq!(
                ExpressionParser::<32>::new(source)
                    .parse()
                    .unwrap_err()
                    .kind,
                ExpressionErrorKind::NonExhaustiveMatch
            );
        }
        let too_many = ExpressionParser::<128>::new(
            "match 1 { 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 => 7, 8 => 8, 9 => 9, _ => 0 }",
        )
        .parse()
        .unwrap_err();
        assert_eq!(too_many.kind, ExpressionErrorKind::TooManyMatchPatterns);
        let malformed_range = ExpressionParser::<24>::new("match 1 { 0.1 => 1, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            malformed_range.kind,
            ExpressionErrorKind::ExpectedMatchPattern
        );
        let too_many_alternatives =
            ExpressionParser::<64>::new("match 1 { 0 | 1 | 2 | 3 | 4 => 42, _ => 0 }")
                .parse()
                .unwrap_err();
        assert_eq!(
            too_many_alternatives.kind,
            ExpressionErrorKind::TooManyMatchAlternatives
        );
        let too_many_nested_alternatives =
            ExpressionParser::<64>::new("match 1 { ((0 | 1) | (2 | (3 | 4))) => 42, _ => 0 }")
                .parse()
                .unwrap_err();
        assert_eq!(
            too_many_nested_alternatives.kind,
            ExpressionErrorKind::TooManyMatchAlternatives
        );
        let too_many_grouped_binding_alternatives = ExpressionParser::<64>::new(
            "match 1 { value @ ((0 | 1) | (2 | (3 | 4))) => value, _ => 0 }",
        )
        .parse()
        .unwrap_err();
        assert_eq!(
            too_many_grouped_binding_alternatives.kind,
            ExpressionErrorKind::TooManyMatchAlternatives
        );
        let empty_group = ExpressionParser::<24>::new("match 1 { () => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(empty_group.kind, ExpressionErrorKind::ExpectedMatchPattern);
        let trailing_alternative = ExpressionParser::<32>::new("match 1 { (1 |) => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            trailing_alternative.kind,
            ExpressionErrorKind::ExpectedMatchPattern
        );
        let missing_group_close = ExpressionParser::<32>::new("match 1 { (1 | 2 => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            missing_group_close.kind,
            ExpressionErrorKind::ExpectedCloseParen
        );
        let unbounded = ExpressionParser::<24>::new("match 1 { .. => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(unbounded.kind, ExpressionErrorKind::ExpectedMatchPattern);
        for source in [
            "match 5 { 6..=1 => 42, _ => 0 }",
            "match 0 { 0..0 => 42, _ => 0 }",
            "match -3 { -1..=-5 => 42, _ => 0 }",
            "match 'm' { 'z'..='a' => 42, _ => 0 }",
        ] {
            let invalid = ExpressionParser::<32>::new(source).parse().unwrap_err();
            assert_eq!(invalid.kind, ExpressionErrorKind::InvalidRangeBounds);
        }
        for source in [
            "match false { false..=true => 42, _ => 0 }",
            "match true { false.. => 42, _ => 0 }",
        ] {
            let invalid = ExpressionParser::<24>::new(source).parse().unwrap_err();
            assert_eq!(invalid.kind, ExpressionErrorKind::InvalidRangeType);
        }
        struct RangeResolver;
        impl ConstantResolver for RangeResolver {
            fn resolve(&self, name: &str) -> Option<u128> {
                match name {
                    "START" => Some(6),
                    "END" => Some(1),
                    _ => None,
                }
            }

            fn resolve_type(&self, name: &str) -> Option<IntegerType> {
                matches!(name, "START" | "END").then_some(IntegerType::U32)
            }
        }
        let named_invalid = ExpressionParser::<32>::new("match 5 { START..=END => 42, _ => 0 }")
            .parse()
            .unwrap();
        assert_eq!(
            named_invalid.evaluate(&RangeResolver),
            Err(ConstEvalError::InvalidRangeBounds)
        );
        let too_many_range_validations = concat!(
            "{ ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, 1..=1 => 0, 2..=2 => 0, 3..=3 => 0, _ => 0 }; ",
            "match 0 { 0..=0 => 0, _ => 0 }; 0 }",
        );
        assert_eq!(
            ExpressionParser::<512>::new(too_many_range_validations)
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::TooManyRangeValidations
        );
        let inconsistent_binding =
            ExpressionParser::<24>::new("match 1 { value | 1 => 42, _ => 0 }")
                .parse()
                .unwrap_err();
        assert_eq!(
            inconsistent_binding.kind,
            ExpressionErrorKind::InconsistentMatchBindings
        );
        let mismatched_bindings =
            ExpressionParser::<40>::new("match 1 { left @ 1 | right @ 2 => 42, _ => 0 }")
                .parse()
                .unwrap_err();
        assert_eq!(
            mismatched_bindings.kind,
            ExpressionErrorKind::InconsistentMatchBindings
        );
        let nested_mismatched_bindings =
            ExpressionParser::<48>::new("match 1 { (left @ 1 | (right @ 2)) => 42, _ => 0 }")
                .parse()
                .unwrap_err();
        assert_eq!(
            nested_mismatched_bindings.kind,
            ExpressionErrorKind::InconsistentMatchBindings
        );
        let nested_at_in_group = ExpressionParser::<48>::new(
            "match 1 { outer @ (inner @ 1 | inner @ 2) => outer, _ => 0 }",
        )
        .parse()
        .unwrap_err();
        assert_eq!(
            nested_at_in_group.kind,
            ExpressionErrorKind::ExpectedMatchPattern
        );
        let non_identifier_at = ExpressionParser::<24>::new("match 1 { 1 @ 1 => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(
            non_identifier_at.kind,
            ExpressionErrorKind::ExpectedIdentifier
        );
        let nested_at = ExpressionParser::<32>::new("match 1 { outer @ inner @ 1 => 42, _ => 0 }")
            .parse()
            .unwrap_err();
        assert_eq!(nested_at.kind, ExpressionErrorKind::ExpectedMatchPattern);
        let invalid_scalar = ExpressionParser::<8>::new("'\\u{d800}'")
            .parse()
            .unwrap_err();
        assert_eq!(invalid_scalar.kind, ExpressionErrorKind::InvalidCharacter);
    }

    #[test]
    fn parses_closed_scalar_inline_const_blocks() {
        assert_eq!(evaluate("const { 40 + 2 }"), Ok(42));
        assert_eq!(evaluate("const { { 42 } }"), Ok(42));
        assert_eq!(evaluate("if true { const { 42 } } else { 1 / 0 }"), Ok(42));
        assert_eq!(evaluate("const {}"), Ok(0));
        assert_eq!(
            ExpressionParser::<8>::new("const 42")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedOpenBrace
        );
        struct Resolver;
        impl ConstantResolver for Resolver {
            fn resolve(&self, name: &str) -> Option<u128> {
                (name == "ITEM").then_some(42)
            }
        }
        let named = ExpressionParser::<8>::new("const { ITEM }")
            .parse()
            .unwrap();
        assert_eq!(named.evaluate(&Resolver), Ok(42));
    }

    #[test]
    fn parses_immutable_scalar_bindings_inside_inline_const() {
        assert_eq!(evaluate("const { let x = 5 + 10; x / 3 }"), Ok(5));
        assert_eq!(
            evaluate("const { let x = 20; let y = x + 1; y * 2 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("const { let x = 40; if true { x + 2 } else { 1 / 0 } }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("const { let x = 1; const { x } }"),
            Err(ConstEvalError::UnknownIdentifier)
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { let = 1; 1 }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedIdentifier
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { let x 1; x }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedEquals
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { let x = 1 x }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedSemicolon
        );
        assert_eq!(
            ExpressionParser::<32>::new(
                "const { let a = 1; let b = 2; let c = 3; let d = 4; let e = 5; e }",
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::TooManyInlineConstBindings
        );
    }

    #[test]
    fn evaluates_mutable_scalar_bindings_inside_inline_const() {
        assert_eq!(evaluate("const { let mut x = 5; x += 10; x / 3 }"), Ok(5));
        assert_eq!(
            evaluate("const { let mut x = 2; let y = 10; x *= y; x += 22; x }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("const { let mut flag = false; flag |= true; flag }"),
            Ok(1)
        );
        assert_eq!(evaluate("const { let mut x = 1; x = 42; x }"), Ok(42));
        assert_eq!(
            ExpressionParser::<24>::new("const { let x = 1; x += 1; x }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ImmutableInlineConstAssignment
        );
        assert_eq!(
            evaluate("const { let mut x = 340282366920938463463374607431768211455; x += 1; x }"),
            Err(ConstEvalError::Overflow)
        );
    }

    #[test]
    fn preserves_scalar_type_ascriptions_inside_inline_const() {
        assert_eq!(evaluate("const { let value: u8 = 42; value }"), Ok(42));
        assert_eq!(
            evaluate("const { let mut value: u8 = 40; value += 2; value }"),
            Ok(42)
        );
        assert_eq!(evaluate("const { let value: bool = 1 < 2; value }"), Ok(1));
        assert_eq!(
            evaluate("const { let value: u8 = 256; value }"),
            Err(ConstEvalError::Overflow)
        );
        assert_eq!(
            evaluate("const { let mut value: u8 = 250; value += 6; value }"),
            Err(ConstEvalError::Overflow)
        );
        assert_eq!(
            evaluate("const { let value: bool = 1; value }"),
            Err(ConstEvalError::InvalidCast)
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { let value: = 1; value }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedInlineConstType
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { let value: String = 1; value }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::UnsupportedInlineConstType
        );
    }

    #[test]
    fn preserves_interleaved_inline_const_statement_order() {
        assert_eq!(
            evaluate(
                "const { let mut value = 1; value += 1; let scale = 10; value *= scale; let offset = 22; value += offset; value }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "const { let mut value: u8 = 1; value += 1; let mut value: u8 = value * 10; value += 22; value }"
            ),
            Ok(42)
        );
        assert_eq!(
            ExpressionParser::<64>::new(
                "const { let mut value = 1; value += 1; let value = 2; value += 1; value }",
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::ImmutableInlineConstAssignment
        );
        assert_eq!(
            ExpressionParser::<64>::new(
                "const { let a = 1; let b = 2; a; let c = 3; let d = 4; let e = 5; e }",
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::TooManyInlineConstBindings
        );
        assert_eq!(
            ExpressionParser::<64>::new(
                "const { let mut a = 1; a += 1; let b = 2; let c = 3; let d = 4; let e = 5; e }",
            )
            .parse()
            .unwrap_err()
            .kind,
            ExpressionErrorKind::TooManyInlineConstBindings
        );
    }

    #[test]
    fn evaluates_inline_const_expression_statements_in_order() {
        assert_eq!(evaluate("const { 1 + 1; 42 }"), Ok(42));
        assert_eq!(
            evaluate(
                "const { let mut value = 1; value += 1; value; let scale = 21; value * scale }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate("const { 1 / 0; 42 }"),
            Err(ConstEvalError::DivisionByZero)
        );
        assert_eq!(
            evaluate("const { if false { 1 / 0 } else { 1 }; 42 }"),
            Ok(42)
        );
        assert_eq!(
            ExpressionParser::<64>::new("const { 0; 1; 2; 3; 4; 5; 6; 7; 8; 9 }",)
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::TooManyInlineConstExpressionStatements
        );
    }

    #[test]
    fn inline_const_trailing_semicolons_produce_unit() {
        assert_eq!(evaluate("const { 42; }"), Ok(0));
        assert_eq!(evaluate("const { let value = 42; value; }"), Ok(0));
        assert_eq!(
            evaluate("const { let mut value = 40; value += 2; value; }"),
            Ok(0)
        );
        let tree = ExpressionParser::<32>::new("const { true; }")
            .parse()
            .unwrap();
        assert!(!tree.is_boolean_expression(tree.root(), 0));
        assert_eq!(
            ExpressionParser::<16>::new("const { return }")
                .parse()
                .unwrap()
                .evaluate(&NoConstants),
            Err(ConstEvalError::UnknownIdentifier)
        );
        assert_eq!(
            ExpressionParser::<16>::new("const { break }")
                .parse()
                .unwrap()
                .evaluate(&NoConstants),
            Err(ConstEvalError::UnknownIdentifier)
        );
    }

    #[test]
    fn parses_unit_if_statements_without_else() {
        assert_eq!(evaluate("if false { 1 / 0; }"), Ok(0));
        assert_eq!(evaluate("if true { 20 + 22; }"), Ok(0));
        assert_eq!(evaluate("if true {}"), Ok(0));
        assert_eq!(evaluate("const { if false { 1 / 0; }; 42 }"), Ok(42));
        assert_eq!(
            ExpressionParser::<16>::new("if true { 42 }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedElse
        );
        assert_eq!(
            ExpressionParser::<32>::new("if true { 0; 1; 2; 3; 4; 5; 6; 7; 8; }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::TooManyInlineConstExpressionStatements
        );
    }

    #[test]
    fn evaluates_scoped_statements_inside_if_branches() {
        assert_eq!(
            evaluate("if true { let mut value: u8 = 40; value += 2; value } else { 1 / 0 }"),
            Ok(42)
        );
        assert_eq!(
            evaluate("if false { 1 / 0 } else { let first = 20; let second = 22; first + second }"),
            Ok(42)
        );
        assert_eq!(
            evaluate(
                "const { let outer = 40; if true { let outer = outer + 2; outer } else { 0 } }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate("const { if true { let local = 42; local } else { 0 }; 7 }"),
            Ok(7)
        );
        assert_eq!(
            evaluate("const { if true { let local = 42; local } else { 0 }; local }"),
            Err(ConstEvalError::UnknownIdentifier)
        );
        assert_eq!(evaluate("if false { let value = 1 / 0; value; }"), Ok(0));
    }

    #[test]
    fn evaluates_else_if_chains_lazily() {
        assert_eq!(
            evaluate(
                "if false { 1 / 0 } else if true { let mut value = 40; value += 2; value } else { 1 / 0 }"
            ),
            Ok(42)
        );
        assert_eq!(
            evaluate("if false { 1 } else if false { 1 / 0 } else if true { 42 } else { 1 / 0 }"),
            Ok(42)
        );
        assert_eq!(evaluate("if false {} else if true { 20 + 22; }"), Ok(0));
        assert_eq!(
            ExpressionParser::<32>::new("if false { 0 } else if true { 42 }")
                .parse()
                .unwrap_err()
                .kind,
            ExpressionErrorKind::ExpectedElse
        );
    }

    #[test]
    fn enforces_arena_and_nesting_limits() {
        let arena = ExpressionParser::<2>::new("1 + 2").parse().unwrap_err();
        assert_eq!(arena.kind, ExpressionErrorKind::TooManyNodes);
        let nested = ExpressionParser::<128>::new(
            "((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((1))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))",
        )
        .parse()
        .unwrap_err();
        assert_eq!(nested.kind, ExpressionErrorKind::NestingLimitExceeded);
    }

    #[test]
    fn rejects_trailing_and_incomplete_syntax() {
        let trailing = ExpressionParser::<8>::new("1 2").parse().unwrap_err();
        assert_eq!(trailing.kind, ExpressionErrorKind::TrailingToken);
        let incomplete = ExpressionParser::<8>::new("1 +").parse().unwrap_err();
        assert_eq!(incomplete.kind, ExpressionErrorKind::ExpectedExpression);
        let group = ExpressionParser::<8>::new("(1 + 2").parse().unwrap_err();
        assert_eq!(group.kind, ExpressionErrorKind::ExpectedCloseParen);
    }
}
