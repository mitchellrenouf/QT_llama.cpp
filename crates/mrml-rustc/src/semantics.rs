use crate::{
    BinaryOperator, ConstEvalError, ConstantResolver, ExecutionError, ExprId, ExprKind,
    ExpressionErrorKind, ExpressionTree, IrErrorKind, Item, Module, Span, UnaryOperator,
    lower_expression_with_pointer_bits,
};

pub const MAX_CONSTANT_IR_INSTRUCTIONS: usize = 256;
pub const MAX_CONSTANT_STACK_VALUES: usize = 64;
pub const MAX_CONST_CALL_DEPTH: usize = 8;
pub const MAX_CONST_FUNCTION_EXPRESSION_NODES: usize = 16;
pub const MAX_CONST_LOOP_ITERATIONS: usize = 65_536;
const MAX_CONST_CALL_BINDINGS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetLayout {
    pointer_bits: u8,
}

impl TargetLayout {
    pub const X86_64: Self = Self { pointer_bits: 64 };
    pub const NVPTX64: Self = Self { pointer_bits: 64 };

    pub const fn new(pointer_bits: u8) -> Option<Self> {
        if matches!(pointer_bits, 16 | 32 | 64) {
            Some(Self { pointer_bits })
        } else {
            None
        }
    }

    pub const fn pointer_bits(self) -> u8 {
        self.pointer_bits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Constant<'source> {
    pub name: &'source str,
    pub ty: &'source str,
    pub value: u128,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstantTable<'source, const MAX_CONSTANTS: usize> {
    constants: [Option<Constant<'source>>; MAX_CONSTANTS],
    count: usize,
}

impl<'source, const MAX_CONSTANTS: usize> ConstantTable<'source, MAX_CONSTANTS> {
    pub fn constants(&self) -> &[Option<Constant<'source>>] {
        &self.constants[..self.count]
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, name: &str) -> Option<&Constant<'source>> {
        self.constants()
            .iter()
            .flatten()
            .find(|constant| constant.name == name)
    }
}

impl<const MAX_CONSTANTS: usize> ConstantResolver for ConstantTable<'_, MAX_CONSTANTS> {
    fn resolve(&self, name: &str) -> Option<u128> {
        self.get(name).map(|constant| constant.value)
    }

    fn resolve_type(&self, name: &str) -> Option<crate::IntegerType> {
        self.get(name)
            .and_then(|constant| crate::IntegerType::from_name(constant.ty))
    }

    fn resolves_bool(&self, name: &str) -> bool {
        self.get(name).is_some_and(|constant| constant.ty == "bool")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticErrorKind {
    DuplicateDefinition,
    TooManyConstants,
    UnsupportedConstantType,
    ConstantOutOfRange,
    Expression(ExpressionErrorKind),
    Lowering(IrErrorKind),
    Execution(ExecutionError),
    UnsupportedConstCall,
    ConstLoopLimitExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticError {
    pub kind: SemanticErrorKind,
    pub span: Span,
}

pub fn analyze_constants<
    'source,
    const MAX_CONSTANTS: usize,
    const MAX_EXPRESSION_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    module: &Module<'source, MAX_ITEMS, MAX_PARAMETERS>,
    target: TargetLayout,
) -> Result<ConstantTable<'source, MAX_CONSTANTS>, SemanticError> {
    let mut symbols = [None; MAX_ITEMS];
    let mut table = ConstantTable {
        constants: [None; MAX_CONSTANTS],
        count: 0,
    };

    for (symbol_count, item) in module.items().iter().flatten().enumerate() {
        let (name, name_span) = match item {
            Item::Function(function) => (function.name, function.name_span),
            Item::Const(constant) | Item::Static(constant) => (constant.name, constant.name_span),
        };
        if symbols[..symbol_count].contains(&Some(name)) {
            return Err(SemanticError {
                kind: SemanticErrorKind::DuplicateDefinition,
                span: name_span,
            });
        }
        symbols[symbol_count] = Some(name);

        let constant = match item {
            Item::Const(constant) | Item::Static(constant) => constant,
            Item::Function(_) => continue,
        };
        if table.count == MAX_CONSTANTS {
            return Err(SemanticError {
                kind: SemanticErrorKind::TooManyConstants,
                span: constant.name_span,
            });
        }
        if constant.ty.text == "bool" {
            let tree = constant
                .parse_initializer::<MAX_EXPRESSION_NODES>()
                .map_err(|error| SemanticError {
                    kind: SemanticErrorKind::Expression(error.kind),
                    span: Span {
                        start: constant.initializer_span.start + error.span.start,
                        end: constant.initializer_span.start + error.span.end,
                    },
                })?;
            let value = if expression_contains_call(&tree, tree.root(), 0) {
                let resolver =
                    ConstCallResolver::from_constants(&table).map_err(|kind| SemanticError {
                        kind,
                        span: constant.initializer_span,
                    })?;
                evaluate_call_aware_boolean(
                    &ConstCallContext { module, target },
                    symbol_count,
                    &tree,
                    tree.root(),
                    &resolver,
                    ConstEvalDepth::ROOT,
                )
                .map_err(|kind| SemanticError {
                    kind,
                    span: constant.initializer_span,
                })?
            } else {
                evaluate_boolean_constant(&tree, tree.root(), &table, 0).map_err(|error| {
                    SemanticError {
                        kind: SemanticErrorKind::Execution(error),
                        span: constant.initializer_span,
                    }
                })?
            };
            table.constants[table.count] = Some(Constant {
                name: constant.name,
                ty: constant.ty.text,
                value: u128::from(value),
                span: Span {
                    start: constant.name_span.start,
                    end: constant.initializer_span.end,
                },
            });
            table.count += 1;
            continue;
        }
        let Some(integer_type) = crate::IntegerType::from_name(constant.ty.text) else {
            return Err(SemanticError {
                kind: SemanticErrorKind::UnsupportedConstantType,
                span: constant.ty.span,
            });
        };
        let tree = constant
            .parse_initializer::<MAX_EXPRESSION_NODES>()
            .map_err(|error| SemanticError {
                kind: SemanticErrorKind::Expression(error.kind),
                span: Span {
                    start: constant.initializer_span.start + error.span.start,
                    end: constant.initializer_span.start + error.span.end,
                },
            })?;
        let call_value = if expression_contains_call(&tree, tree.root(), 0) {
            let resolver =
                ConstCallResolver::from_constants(&table).map_err(|kind| SemanticError {
                    kind,
                    span: constant.initializer_span,
                })?;
            let context = ConstCallContext { module, target };
            Some(
                evaluate_integer_const_expression(
                    &context,
                    symbol_count,
                    &tree,
                    tree.root(),
                    integer_type,
                    &resolver,
                    ConstEvalDepth::ROOT,
                )
                .map_err(|kind| SemanticError {
                    kind: if kind
                        == SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                            ConstEvalError::Overflow,
                        )) {
                        SemanticErrorKind::ConstantOutOfRange
                    } else {
                        kind
                    },
                    span: constant.initializer_span,
                })?,
            )
        } else {
            None
        };
        let value = if let Some(value) = call_value {
            value
        } else if integer_type.is_signed() {
            evaluate_signed_integer(&tree, tree.root(), integer_type, target, &table, 0).map_err(
                |error| SemanticError {
                    kind: if error == ExecutionError::Arithmetic(ConstEvalError::Overflow) {
                        SemanticErrorKind::ConstantOutOfRange
                    } else {
                        SemanticErrorKind::Execution(error)
                    },
                    span: constant.initializer_span,
                },
            )?
        } else {
            let program = lower_expression_with_pointer_bits::<
                MAX_CONSTANT_IR_INSTRUCTIONS,
                MAX_EXPRESSION_NODES,
            >(&tree, target.pointer_bits())
            .map_err(|error| SemanticError {
                kind: SemanticErrorKind::Lowering(error.kind),
                span: Span {
                    start: constant.initializer_span.start + error.span.start,
                    end: constant.initializer_span.start + error.span.end,
                },
            })?;
            program
                .execute::<_, MAX_CONSTANT_STACK_VALUES>(&table)
                .map_err(|error| SemanticError {
                    kind: SemanticErrorKind::Execution(error),
                    span: constant.initializer_span,
                })?
        };
        let in_range = if integer_type.is_signed() {
            let bits = integer_type
                .bits(target.pointer_bits())
                .ok_or(SemanticError {
                    kind: SemanticErrorKind::UnsupportedConstantType,
                    span: constant.ty.span,
                })?;
            value & signed_mask(bits) == value
        } else {
            value <= maximum_unsigned_value(constant.ty.text, target)
        };
        if !in_range {
            return Err(SemanticError {
                kind: SemanticErrorKind::ConstantOutOfRange,
                span: constant.initializer_span,
            });
        }
        table.constants[table.count] = Some(Constant {
            name: constant.name,
            ty: constant.ty.text,
            value,
            span: Span {
                start: constant.name_span.start,
                end: constant.initializer_span.end,
            },
        });
        table.count += 1;
    }
    Ok(table)
}

#[derive(Clone, Copy)]
enum ConstCallType {
    Integer(crate::IntegerType),
    Bool,
}

#[derive(Clone, Copy)]
struct ConstCallArgument<'source> {
    name: &'source str,
    ty: ConstCallType,
    value: u128,
    mutable: bool,
}

#[derive(Clone, Copy)]
struct ConstCallResolver<'source> {
    arguments: [Option<ConstCallArgument<'source>>; MAX_CONST_CALL_BINDINGS],
    count: usize,
    constant_count: usize,
}

impl<'source> ConstCallResolver<'source> {
    fn from_constants<const MAX_CONSTANTS: usize>(
        constants: &ConstantTable<'source, MAX_CONSTANTS>,
    ) -> Result<Self, SemanticErrorKind> {
        let mut resolver = Self {
            arguments: [None; MAX_CONST_CALL_BINDINGS],
            count: 0,
            constant_count: 0,
        };
        for constant in constants.constants().iter().flatten() {
            let ty = if constant.ty == "bool" {
                ConstCallType::Bool
            } else if let Some(ty) = crate::IntegerType::from_name(constant.ty) {
                ConstCallType::Integer(ty)
            } else {
                continue;
            };
            resolver.push(ConstCallArgument {
                name: constant.name,
                ty,
                value: constant.value,
                mutable: false,
            })?;
        }
        resolver.constant_count = resolver.count;
        Ok(resolver)
    }

    fn push(&mut self, argument: ConstCallArgument<'source>) -> Result<(), SemanticErrorKind> {
        if self.count == MAX_CONST_CALL_BINDINGS {
            return Err(SemanticErrorKind::UnsupportedConstCall);
        }
        self.arguments[self.count] = Some(argument);
        self.count += 1;
        Ok(())
    }

    fn binding(&self, name: &str) -> Option<ConstCallArgument<'source>> {
        self.arguments[..self.count]
            .iter()
            .rev()
            .flatten()
            .find(|argument| argument.name == name)
            .copied()
    }

    fn has_nonconstant_binding(&self, name: &str) -> bool {
        self.arguments[self.constant_count..self.count]
            .iter()
            .flatten()
            .any(|argument| argument.name == name)
    }

    fn has_module_constant(&self, name: &str) -> bool {
        self.arguments[..self.constant_count]
            .iter()
            .rev()
            .flatten()
            .any(|argument| argument.name == name)
    }

    fn assign(&mut self, name: &str, value: u128) -> Result<(), SemanticErrorKind> {
        let binding = self.arguments[..self.count]
            .iter_mut()
            .rev()
            .flatten()
            .find(|argument| argument.name == name)
            .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
        if !binding.mutable {
            return Err(SemanticErrorKind::UnsupportedConstCall);
        }
        binding.value = value;
        Ok(())
    }

    fn truncate(&mut self, count: usize) -> Result<(), SemanticErrorKind> {
        if count < self.constant_count || count > self.count {
            return Err(SemanticErrorKind::UnsupportedConstCall);
        }
        for argument in &mut self.arguments[count..self.count] {
            *argument = None;
        }
        self.count = count;
        Ok(())
    }
}

impl ConstantResolver for ConstCallResolver<'_> {
    fn resolve(&self, name: &str) -> Option<u128> {
        self.arguments[..self.count]
            .iter()
            .rev()
            .flatten()
            .find(|argument| argument.name == name)
            .map(|argument| argument.value)
    }

    fn resolve_type(&self, name: &str) -> Option<crate::IntegerType> {
        self.arguments[..self.count]
            .iter()
            .rev()
            .flatten()
            .find(|argument| argument.name == name)
            .and_then(|argument| match argument.ty {
                ConstCallType::Integer(ty) => Some(ty),
                ConstCallType::Bool => None,
            })
    }

    fn resolves_bool(&self, name: &str) -> bool {
        self.arguments[..self.count]
            .iter()
            .rev()
            .flatten()
            .find(|argument| argument.name == name)
            .is_some_and(|argument| matches!(argument.ty, ConstCallType::Bool))
    }
}

struct ConstCallContext<'module, 'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize> {
    module: &'module Module<'source, MAX_ITEMS, MAX_PARAMETERS>,
    target: TargetLayout,
}

pub(crate) fn evaluate_scalar_const_function<
    'source,
    const MAX_CONSTANTS: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    module: &Module<'source, MAX_ITEMS, MAX_PARAMETERS>,
    constants: &ConstantTable<'source, MAX_CONSTANTS>,
    target: TargetLayout,
    name: &str,
    arguments: &[u128],
) -> Option<u128> {
    let (index, function) =
        module
            .items()
            .iter()
            .enumerate()
            .find_map(|(index, item)| match item.as_ref()? {
                Item::Function(function) if function.name == name => Some((index, function)),
                _ => None,
            })?;
    if !function.constant
        || function.abi != crate::FunctionAbi::Rust
        || function.parameter_count() != arguments.len()
    {
        return None;
    }
    let context = ConstCallContext { module, target };
    let mut resolver = ConstCallResolver::from_constants(constants).ok()?;
    for (parameter, value) in function.parameters().iter().flatten().zip(arguments) {
        let (ty, value) = if parameter.ty.text == "bool" {
            if *value > 1 {
                return None;
            }
            (ConstCallType::Bool, *value)
        } else {
            let ty = crate::IntegerType::from_name(parameter.ty.text)?;
            let bits = ty.bits(target.pointer_bits())?;
            let mask = signed_mask(bits);
            let raw = *value;
            let value = raw & mask;
            if ty.is_signed() {
                let sign_extended = if value & (1u128 << (bits - 1)) != 0 {
                    value | !mask
                } else {
                    value
                };
                if raw != value && raw != sign_extended {
                    return None;
                }
            } else if raw != value {
                return None;
            }
            (ConstCallType::Integer(ty), value)
        };
        resolver
            .push(ConstCallArgument {
                name: parameter.name,
                ty,
                value,
                mutable: false,
            })
            .ok()?;
    }
    let symbol_count = index.checked_add(1)?;
    if function
        .return_type
        .is_some_and(|return_type| return_type.text == "bool")
    {
        let value = evaluate_boolean_const_function_body(
            &context,
            symbol_count,
            function,
            &mut resolver,
            ConstEvalDepth::ROOT.enter_call().ok()?,
        )
        .ok()?;
        return Some(u128::from(value));
    }
    let ty = function
        .return_type
        .and_then(|return_type| crate::IntegerType::from_name(return_type.text))?;
    let value = evaluate_integer_const_function_body(
        &context,
        symbol_count,
        function,
        ty,
        &mut resolver,
        ConstEvalDepth::ROOT.enter_call().ok()?,
    )
    .ok()?;
    Some(value)
}

#[derive(Clone, Copy)]
struct ConstEvalDepth {
    calls: usize,
    expressions: usize,
}

impl ConstEvalDepth {
    const ROOT: Self = Self {
        calls: 0,
        expressions: 0,
    };

    fn enter_call(self) -> Result<Self, SemanticErrorKind> {
        if self.calls == MAX_CONST_CALL_DEPTH {
            Err(SemanticErrorKind::UnsupportedConstCall)
        } else {
            Ok(Self {
                calls: self.calls + 1,
                expressions: self.expressions,
            })
        }
    }

    fn enter_expression(self) -> Result<Self, SemanticErrorKind> {
        if self.expressions == 64 {
            Err(SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                ConstEvalError::NestingLimitExceeded,
            )))
        } else {
            Ok(Self {
                calls: self.calls,
                expressions: self.expressions + 1,
            })
        }
    }
}

fn evaluate_integer_const_call<
    const MAX_EXPRESSION_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    call_tree: &ExpressionTree<'_, MAX_EXPRESSION_NODES>,
    call_id: ExprId,
    declared_type: crate::IntegerType,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<u128, SemanticErrorKind> {
    let depth = depth.enter_call()?;
    let Some(ExprKind::Call {
        callee,
        arguments,
        argument_count,
    }) = call_tree
        .expression(call_id)
        .map(|expression| expression.kind)
    else {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    };
    let (function_index, function) = context.module.items()[..symbol_count]
        .iter()
        .enumerate()
        .find_map(|(index, item)| match item.as_ref()? {
            Item::Function(function) if function.name == callee => Some((index, function)),
            _ => None,
        })
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    let return_type = function
        .return_type
        .and_then(|return_type| crate::IntegerType::from_name(return_type.text))
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    if !function.constant
        || function.abi != crate::FunctionAbi::Rust
        || function.parameter_count() != argument_count
        || return_type != declared_type
    {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    let mut values = [None; MAX_PARAMETERS];
    for (index, parameter) in function.parameters().iter().flatten().enumerate() {
        let argument = arguments[index].ok_or(SemanticErrorKind::UnsupportedConstCall)?;
        let (ty, value, in_range) = if parameter.ty.text == "bool" {
            let value = evaluate_boolean_const_expression(
                context,
                symbol_count,
                call_tree,
                argument,
                constants,
                depth,
            )?;
            (ConstCallType::Bool, u128::from(value), true)
        } else {
            let ty = crate::IntegerType::from_name(parameter.ty.text)
                .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
            let value = evaluate_integer_const_expression(
                context,
                symbol_count,
                call_tree,
                argument,
                ty,
                constants,
                depth,
            )?;
            let in_range = if ty.is_signed() {
                let bits = ty
                    .bits(context.target.pointer_bits())
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                value & signed_mask(bits) == value
            } else {
                value <= maximum_unsigned_value(ty.name(), context.target)
            };
            (ConstCallType::Integer(ty), value, in_range)
        };
        if !in_range {
            return Err(SemanticErrorKind::ConstantOutOfRange);
        }
        values[index] = Some(ConstCallArgument {
            name: parameter.name,
            ty,
            value,
            mutable: false,
        });
    }
    let mut resolver = *constants;
    for value in values[..argument_count].iter().flatten() {
        resolver.push(*value)?;
    }
    let body_symbol_count = function_index
        .checked_add(1)
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    let value = evaluate_integer_const_function_body::<MAX_ITEMS, MAX_PARAMETERS>(
        context,
        body_symbol_count,
        function,
        return_type,
        &mut resolver,
        depth,
    )?;
    if !return_type.is_signed()
        && value > maximum_unsigned_value(return_type.name(), context.target)
    {
        return Err(SemanticErrorKind::ConstantOutOfRange);
    }
    Ok(value)
}

fn evaluate_boolean_const_call<
    const MAX_EXPRESSION_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    call_tree: &ExpressionTree<'_, MAX_EXPRESSION_NODES>,
    call_id: ExprId,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<bool, SemanticErrorKind> {
    let depth = depth.enter_call()?;
    let Some(ExprKind::Call {
        callee,
        arguments,
        argument_count,
    }) = call_tree
        .expression(call_id)
        .map(|expression| expression.kind)
    else {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    };
    let (function_index, function) = context.module.items()[..symbol_count]
        .iter()
        .enumerate()
        .find_map(|(index, item)| match item.as_ref()? {
            Item::Function(function) if function.name == callee => Some((index, function)),
            _ => None,
        })
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    if !function.constant
        || function.abi != crate::FunctionAbi::Rust
        || function.parameter_count() != argument_count
        || !function
            .return_type
            .is_some_and(|return_type| return_type.text == "bool")
    {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    let mut values = [None; MAX_PARAMETERS];
    for (index, parameter) in function.parameters().iter().flatten().enumerate() {
        let argument = arguments[index].ok_or(SemanticErrorKind::UnsupportedConstCall)?;
        let (ty, value) = if parameter.ty.text == "bool" {
            (
                ConstCallType::Bool,
                u128::from(evaluate_boolean_const_expression(
                    context,
                    symbol_count,
                    call_tree,
                    argument,
                    constants,
                    depth,
                )?),
            )
        } else {
            let ty = crate::IntegerType::from_name(parameter.ty.text)
                .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
            let value = evaluate_integer_const_expression(
                context,
                symbol_count,
                call_tree,
                argument,
                ty,
                constants,
                depth,
            )?;
            let in_range = if ty.is_signed() {
                let bits = ty
                    .bits(context.target.pointer_bits())
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                value & signed_mask(bits) == value
            } else {
                value <= maximum_unsigned_value(ty.name(), context.target)
            };
            if !in_range {
                return Err(SemanticErrorKind::ConstantOutOfRange);
            }
            (ConstCallType::Integer(ty), value)
        };
        values[index] = Some(ConstCallArgument {
            name: parameter.name,
            ty,
            value,
            mutable: false,
        });
    }
    let mut resolver = *constants;
    for value in values[..argument_count].iter().flatten() {
        resolver.push(*value)?;
    }
    let body_symbol_count = function_index
        .checked_add(1)
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    evaluate_boolean_const_function_body::<MAX_ITEMS, MAX_PARAMETERS>(
        context,
        body_symbol_count,
        function,
        &mut resolver,
        depth,
    )
}

fn evaluate_integer_const_function_body<
    'source,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    function: &crate::Function<'source, MAX_PARAMETERS>,
    return_type: crate::IntegerType,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<u128, SemanticErrorKind> {
    let body = function
        .parse_body::<MAX_PARAMETERS>()
        .map_err(|_| SemanticErrorKind::UnsupportedConstCall)?;
    if let Some(value) = evaluate_const_body_statements(
        context,
        symbol_count,
        &body,
        ConstCallType::Integer(return_type),
        resolver,
        depth,
    )? {
        return Ok(value);
    }
    if body.tail_diverges {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    let tail = body
        .parse_tail::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    evaluate_integer_const_expression(
        context,
        symbol_count,
        &tail,
        tail.root(),
        return_type,
        resolver,
        depth,
    )
}

fn evaluate_boolean_const_function_body<
    'source,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    function: &crate::Function<'source, MAX_PARAMETERS>,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<bool, SemanticErrorKind> {
    let body = function
        .parse_body::<MAX_PARAMETERS>()
        .map_err(|_| SemanticErrorKind::UnsupportedConstCall)?;
    if let Some(value) = evaluate_const_body_statements(
        context,
        symbol_count,
        &body,
        ConstCallType::Bool,
        resolver,
        depth,
    )? {
        return Ok(value != 0);
    }
    if body.tail_diverges {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    let tail = body
        .parse_tail::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    evaluate_boolean_const_expression(context, symbol_count, &tail, tail.root(), resolver, depth)
}

fn evaluate_const_body_statements<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    body: &crate::FunctionBody<'source, MAX_PARAMETERS>,
    return_type: ConstCallType,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<Option<u128>, SemanticErrorKind> {
    for statement in body.statements().iter().flatten() {
        match statement {
            crate::BodyStatement::Local(index) => {
                let local = body.locals()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                evaluate_const_local(context, symbol_count, local, resolver, depth)?;
            }
            crate::BodyStatement::Assignment(index) => {
                let assignment = body.assignments()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                evaluate_const_assignment(context, symbol_count, assignment, resolver, depth)?;
            }
            crate::BodyStatement::ConditionalReturn(index) => {
                let conditional = body.conditional_returns()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                let condition = conditional
                    .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                    .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                if evaluate_boolean_const_expression(
                    context,
                    symbol_count,
                    &condition,
                    condition.root(),
                    resolver,
                    depth,
                )? {
                    let value = conditional
                        .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    return Ok(Some(match return_type {
                        ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
                            context,
                            symbol_count,
                            &value,
                            value.root(),
                            resolver,
                            depth,
                        )?),
                        ConstCallType::Integer(ty) => evaluate_integer_const_expression(
                            context,
                            symbol_count,
                            &value,
                            value.root(),
                            ty,
                            resolver,
                            depth,
                        )?,
                    }));
                }
            }
            crate::BodyStatement::ConditionalReturnElse(index) => {
                let conditional = body.conditional_return_elses()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                let mut selected = None;
                for branch in conditional.branches().iter().flatten() {
                    let condition = branch
                        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    if evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &condition,
                        condition.root(),
                        resolver,
                        depth,
                    )? {
                        selected = Some(crate::LoopReturn {
                            value: branch.value,
                            value_span: branch.value_span,
                        });
                        break;
                    }
                }
                let selected = match selected.as_ref().or(conditional.else_value.as_ref()) {
                    Some(selected) => selected,
                    None => continue,
                };
                let value = selected
                    .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                    .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                return Ok(Some(match return_type {
                    ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &value,
                        value.root(),
                        resolver,
                        depth,
                    )?),
                    ConstCallType::Integer(ty) => evaluate_integer_const_expression(
                        context,
                        symbol_count,
                        &value,
                        value.root(),
                        ty,
                        resolver,
                        depth,
                    )?,
                }));
            }
            crate::BodyStatement::ConditionalAssignment(index) => {
                let conditional = body.conditional_assignments()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                let mut selected = false;
                for branch in conditional.branches().iter().flatten() {
                    let condition = branch
                        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    if evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &condition,
                        condition.root(),
                        resolver,
                        depth,
                    )? {
                        let binding_checkpoint = resolver.count;
                        for action in branch.actions().iter().flatten() {
                            match action {
                                crate::ConditionalAssignmentAction::Local(local) => {
                                    evaluate_const_local(
                                        context,
                                        symbol_count,
                                        local,
                                        resolver,
                                        depth,
                                    )?;
                                }
                                crate::ConditionalAssignmentAction::Assignment(assignment) => {
                                    evaluate_const_assignment(
                                        context,
                                        symbol_count,
                                        assignment,
                                        resolver,
                                        depth,
                                    )?;
                                }
                                crate::ConditionalAssignmentAction::Expression(statement) => {
                                    evaluate_const_expression_statement(
                                        context,
                                        symbol_count,
                                        statement,
                                        resolver,
                                        depth,
                                    )?;
                                }
                                crate::ConditionalAssignmentAction::Return(return_statement) => {
                                    return Ok(Some(evaluate_const_return_statement(
                                        context,
                                        symbol_count,
                                        return_statement,
                                        return_type,
                                        resolver,
                                        depth,
                                    )?));
                                }
                            }
                        }
                        resolver.truncate(binding_checkpoint)?;
                        selected = true;
                        break;
                    }
                }
                if !selected {
                    let binding_checkpoint = resolver.count;
                    for action in conditional.else_actions().iter().flatten() {
                        match action {
                            crate::ConditionalAssignmentAction::Local(local) => {
                                evaluate_const_local(
                                    context,
                                    symbol_count,
                                    local,
                                    resolver,
                                    depth,
                                )?;
                            }
                            crate::ConditionalAssignmentAction::Assignment(assignment) => {
                                evaluate_const_assignment(
                                    context,
                                    symbol_count,
                                    assignment,
                                    resolver,
                                    depth,
                                )?;
                            }
                            crate::ConditionalAssignmentAction::Expression(statement) => {
                                evaluate_const_expression_statement(
                                    context,
                                    symbol_count,
                                    statement,
                                    resolver,
                                    depth,
                                )?;
                            }
                            crate::ConditionalAssignmentAction::Return(return_statement) => {
                                return Ok(Some(evaluate_const_return_statement(
                                    context,
                                    symbol_count,
                                    return_statement,
                                    return_type,
                                    resolver,
                                    depth,
                                )?));
                            }
                        }
                    }
                    resolver.truncate(binding_checkpoint)?;
                }
            }
            crate::BodyStatement::Loop(index) => {
                let loop_statement = body.while_loops()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                if let Some(value) = evaluate_const_loop(
                    context,
                    symbol_count,
                    loop_statement,
                    return_type,
                    resolver,
                    depth,
                )? {
                    return Ok(Some(value));
                }
            }
            crate::BodyStatement::Expression(index) => {
                let statement = body.expression_statements()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                evaluate_const_expression_statement(
                    context,
                    symbol_count,
                    statement,
                    resolver,
                    depth,
                )?;
            }
            crate::BodyStatement::Return(index) => {
                let return_statement = body.returns()[*index]
                    .as_ref()
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                let value = return_statement
                    .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                    .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                return Ok(Some(match return_type {
                    ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &value,
                        value.root(),
                        resolver,
                        depth,
                    )?),
                    ConstCallType::Integer(ty) => evaluate_integer_const_expression(
                        context,
                        symbol_count,
                        &value,
                        value.root(),
                        ty,
                        resolver,
                        depth,
                    )?,
                }));
            }
        }
    }
    Ok(None)
}

fn evaluate_const_expression_statement<
    'source,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    statement: &crate::ExpressionStatement<'source>,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<(), SemanticErrorKind> {
    if statement.expression.is_empty() {
        return Ok(());
    }
    let tree = statement
        .parse::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    if tree.is_boolean_expression(tree.root(), 0) {
        evaluate_boolean_const_expression(
            context,
            symbol_count,
            &tree,
            tree.root(),
            resolver,
            depth,
        )?;
    } else if matches!(
        tree.expression(tree.root())
            .map(|expression| expression.kind),
        Some(ExprKind::Unit)
    ) {
        tree.evaluate(resolver)
            .map_err(|error| SemanticErrorKind::Execution(ExecutionError::Arithmetic(error)))?;
    } else {
        let ty = integer_expression_type(context, symbol_count, &tree, tree.root(), resolver)
            .unwrap_or(crate::IntegerType::I32);
        evaluate_integer_const_expression(
            context,
            symbol_count,
            &tree,
            tree.root(),
            ty,
            resolver,
            depth,
        )?;
    }
    Ok(())
}

fn evaluate_const_return_statement<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    return_statement: &crate::LoopReturn<'source>,
    return_type: ConstCallType,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<u128, SemanticErrorKind> {
    let value = return_statement
        .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    match return_type {
        ConstCallType::Bool => Ok(u128::from(evaluate_boolean_const_expression(
            context,
            symbol_count,
            &value,
            value.root(),
            resolver,
            depth,
        )?)),
        ConstCallType::Integer(ty) => evaluate_integer_const_expression(
            context,
            symbol_count,
            &value,
            value.root(),
            ty,
            resolver,
            depth,
        ),
    }
}

fn evaluate_const_local<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    local: &crate::LocalBinding<'source>,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<(), SemanticErrorKind> {
    let tree = local
        .parse_initializer::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    let ty = match local.ty.map(|ty| ty.text) {
        Some("bool") => ConstCallType::Bool,
        Some(name) => ConstCallType::Integer(
            crate::IntegerType::from_name(name).ok_or(SemanticErrorKind::UnsupportedConstCall)?,
        ),
        None if tree.is_boolean_expression(tree.root(), 0) => ConstCallType::Bool,
        None => ConstCallType::Integer(
            integer_expression_type(context, symbol_count, &tree, tree.root(), resolver)
                .unwrap_or(crate::IntegerType::I32),
        ),
    };
    let value = match ty {
        ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
            context,
            symbol_count,
            &tree,
            tree.root(),
            resolver,
            depth,
        )?),
        ConstCallType::Integer(ty) => evaluate_integer_const_expression(
            context,
            symbol_count,
            &tree,
            tree.root(),
            ty,
            resolver,
            depth,
        )?,
    };
    resolver.push(ConstCallArgument {
        name: local.name,
        ty,
        value,
        mutable: local.mutable,
    })
}

fn evaluate_const_assignment<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    assignment: &crate::Assignment<'source>,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<(), SemanticErrorKind> {
    let binding = resolver
        .binding(assignment.name)
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    if !binding.mutable {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    let tree = assignment
        .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    let value = match binding.ty {
        ConstCallType::Bool => {
            let right = evaluate_boolean_const_expression(
                context,
                symbol_count,
                &tree,
                tree.root(),
                resolver,
                depth,
            )?;
            let left = binding.value != 0;
            u128::from(match assignment.operator {
                crate::AssignmentOperator::Assign => right,
                crate::AssignmentOperator::BitAnd => left && right,
                crate::AssignmentOperator::BitOr => left || right,
                crate::AssignmentOperator::BitXor => left != right,
                _ => return Err(SemanticErrorKind::UnsupportedConstCall),
            })
        }
        ConstCallType::Integer(ty) => {
            let right = evaluate_integer_const_expression(
                context,
                symbol_count,
                &tree,
                tree.root(),
                ty,
                resolver,
                depth,
            )?;
            if assignment.operator == crate::AssignmentOperator::Assign {
                right
            } else {
                let operator = match assignment.operator {
                    crate::AssignmentOperator::Add => BinaryOperator::Add,
                    crate::AssignmentOperator::Subtract => BinaryOperator::Subtract,
                    crate::AssignmentOperator::Multiply => BinaryOperator::Multiply,
                    crate::AssignmentOperator::Divide => BinaryOperator::Divide,
                    crate::AssignmentOperator::Remainder => BinaryOperator::Remainder,
                    crate::AssignmentOperator::BitAnd => BinaryOperator::BitAnd,
                    crate::AssignmentOperator::BitOr => BinaryOperator::BitOr,
                    crate::AssignmentOperator::BitXor => BinaryOperator::BitXor,
                    crate::AssignmentOperator::ShiftLeft => BinaryOperator::ShiftLeft,
                    crate::AssignmentOperator::ShiftRight => BinaryOperator::ShiftRight,
                    crate::AssignmentOperator::Assign => BinaryOperator::Add,
                };
                let bits = ty
                    .bits(context.target.pointer_bits())
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                if ty.is_signed() {
                    evaluate_call_aware_signed_binary(operator, binding.value, right, bits)
                } else {
                    evaluate_call_aware_unsigned_binary(operator, binding.value, right, bits)
                }
                .map_err(|error| SemanticErrorKind::Execution(ExecutionError::Arithmetic(error)))?
            }
        }
    };
    resolver.assign(assignment.name, value)
}

fn evaluate_const_loop<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
    context: &ConstCallContext<'_, 'source, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    loop_statement: &crate::WhileLoop<'source>,
    return_type: ConstCallType,
    resolver: &mut ConstCallResolver<'source>,
    depth: ConstEvalDepth,
) -> Result<Option<u128>, SemanticErrorKind> {
    let condition = loop_statement
        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
    let mut iterations = 0usize;
    loop {
        if let Some(condition) = condition.as_ref()
            && !evaluate_boolean_const_expression(
                context,
                symbol_count,
                condition,
                condition.root(),
                resolver,
                depth,
            )?
        {
            break;
        }
        if iterations == MAX_CONST_LOOP_ITERATIONS {
            return Err(SemanticErrorKind::ConstLoopLimitExceeded);
        }
        iterations = iterations
            .checked_add(1)
            .ok_or(SemanticErrorKind::ConstLoopLimitExceeded)?;
        let binding_checkpoint = resolver.count;
        let mut break_loop = false;
        for operation in loop_statement.operations().iter().flatten() {
            match operation {
                crate::LoopOperation::Local(local) => {
                    evaluate_const_local(context, symbol_count, local, resolver, depth)?;
                }
                crate::LoopOperation::Assignment(assignment) => {
                    evaluate_const_assignment(context, symbol_count, assignment, resolver, depth)?;
                }
                crate::LoopOperation::Expression(statement) => {
                    evaluate_const_expression_statement(
                        context,
                        symbol_count,
                        statement,
                        resolver,
                        depth,
                    )?;
                }
                crate::LoopOperation::ConditionalBlock(index) => {
                    let block = loop_statement.conditional_blocks()[*index]
                        .as_ref()
                        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                    let condition = block
                        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    if evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &condition,
                        condition.root(),
                        resolver,
                        depth,
                    )? {
                        let block_checkpoint = resolver.count;
                        for action in block.actions().iter().flatten() {
                            match action {
                                crate::ConditionalLoopAction::Local(local) => {
                                    evaluate_const_local(
                                        context,
                                        symbol_count,
                                        local,
                                        resolver,
                                        depth,
                                    )?;
                                }
                                crate::ConditionalLoopAction::Assignment(assignment) => {
                                    evaluate_const_assignment(
                                        context,
                                        symbol_count,
                                        assignment,
                                        resolver,
                                        depth,
                                    )?;
                                }
                                crate::ConditionalLoopAction::Expression(statement) => {
                                    evaluate_const_expression_statement(
                                        context,
                                        symbol_count,
                                        statement,
                                        resolver,
                                        depth,
                                    )?;
                                }
                            }
                        }
                        match block.terminal {
                            crate::ConditionalLoopTerminal::Break => break_loop = true,
                            crate::ConditionalLoopTerminal::Continue => {}
                            crate::ConditionalLoopTerminal::Return(return_statement) => {
                                return Ok(Some(evaluate_const_return_statement(
                                    context,
                                    symbol_count,
                                    &return_statement,
                                    return_type,
                                    resolver,
                                    depth,
                                )?));
                            }
                        }
                        resolver.truncate(block_checkpoint)?;
                        break;
                    }
                }
                crate::LoopOperation::Break => {
                    break_loop = true;
                    break;
                }
                crate::LoopOperation::Continue => break,
                crate::LoopOperation::ConditionalBreak(conditional)
                | crate::LoopOperation::ConditionalContinue(conditional) => {
                    let tree = conditional
                        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    if evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &tree,
                        tree.root(),
                        resolver,
                        depth,
                    )? {
                        if matches!(operation, crate::LoopOperation::ConditionalBreak(_)) {
                            break_loop = true;
                        }
                        break;
                    }
                }
                crate::LoopOperation::ConditionalReturn(conditional) => {
                    let condition = conditional
                        .parse_condition::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    if evaluate_boolean_const_expression(
                        context,
                        symbol_count,
                        &condition,
                        condition.root(),
                        resolver,
                        depth,
                    )? {
                        let value = conditional
                            .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                            .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                        return Ok(Some(match return_type {
                            ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
                                context,
                                symbol_count,
                                &value,
                                value.root(),
                                resolver,
                                depth,
                            )?),
                            ConstCallType::Integer(ty) => evaluate_integer_const_expression(
                                context,
                                symbol_count,
                                &value,
                                value.root(),
                                ty,
                                resolver,
                                depth,
                            )?,
                        }));
                    }
                }
                crate::LoopOperation::Return(loop_return) => {
                    let value = loop_return
                        .parse_value::<MAX_CONST_FUNCTION_EXPRESSION_NODES>()
                        .map_err(|error| SemanticErrorKind::Expression(error.kind))?;
                    return Ok(Some(match return_type {
                        ConstCallType::Bool => u128::from(evaluate_boolean_const_expression(
                            context,
                            symbol_count,
                            &value,
                            value.root(),
                            resolver,
                            depth,
                        )?),
                        ConstCallType::Integer(ty) => evaluate_integer_const_expression(
                            context,
                            symbol_count,
                            &value,
                            value.root(),
                            ty,
                            resolver,
                            depth,
                        )?,
                    }));
                }
            }
        }
        resolver.truncate(binding_checkpoint)?;
        if break_loop {
            break;
        }
    }
    Ok(None)
}

fn evaluate_boolean_const_expression<
    const MAX_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<bool, SemanticErrorKind> {
    if inline_const_has_invalid_capture(tree, id, constants, false, 0) {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    if expression_contains_call(tree, id, 0) {
        evaluate_call_aware_boolean(context, symbol_count, tree, id, constants, depth)
    } else {
        evaluate_boolean_constant(tree, id, constants, 0).map_err(SemanticErrorKind::Execution)
    }
}

fn evaluate_integer_const_expression<
    const MAX_EXPRESSION_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    tree: &ExpressionTree<'_, MAX_EXPRESSION_NODES>,
    id: ExprId,
    ty: crate::IntegerType,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<u128, SemanticErrorKind> {
    if inline_const_has_invalid_capture(tree, id, constants, false, 0) {
        return Err(SemanticErrorKind::UnsupportedConstCall);
    }
    if matches!(
        tree.expression(id).map(|expression| expression.kind),
        Some(ExprKind::Call { .. })
    ) {
        return evaluate_integer_const_call::<MAX_EXPRESSION_NODES, MAX_ITEMS, MAX_PARAMETERS>(
            context,
            symbol_count,
            tree,
            id,
            ty,
            constants,
            depth,
        );
    }
    if expression_contains_call(tree, id, 0) {
        return evaluate_call_aware_integer(context, symbol_count, tree, id, ty, constants, depth);
    }
    if ty.is_signed() {
        evaluate_signed_integer(tree, id, ty, context.target, constants, 0)
            .map_err(SemanticErrorKind::Execution)
    } else if id != tree.root() {
        tree.evaluate_at(id, constants)
            .map_err(|error| SemanticErrorKind::Execution(ExecutionError::Arithmetic(error)))
    } else {
        let program = lower_expression_with_pointer_bits::<
            MAX_CONSTANT_IR_INSTRUCTIONS,
            MAX_EXPRESSION_NODES,
        >(tree, context.target.pointer_bits())
        .map_err(|error| SemanticErrorKind::Lowering(error.kind))?;
        program
            .execute::<_, MAX_CONSTANT_STACK_VALUES>(constants)
            .map_err(SemanticErrorKind::Execution)
    }
}

fn inline_const_has_invalid_capture<const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &ConstCallResolver<'_>,
    inside_inline_const: bool,
    depth: usize,
) -> bool {
    if depth == 64 {
        return true;
    }
    let Some(expression) = tree.expression(id) else {
        return true;
    };
    let recurse = |operand, inside| {
        inline_const_has_invalid_capture(tree, operand, constants, inside, depth + 1)
    };
    match expression.kind {
        ExprKind::Identifier(name) if inside_inline_const => {
            constants.has_nonconstant_binding(name) || !constants.has_module_constant(name)
        }
        ExprKind::Call {
            arguments,
            argument_count,
            ..
        } => arguments[..argument_count]
            .iter()
            .flatten()
            .any(|argument| recurse(*argument, inside_inline_const)),
        ExprKind::InlineConst { operand } => recurse(operand, true),
        ExprKind::Cast { operand, .. }
        | ExprKind::Ascribe { operand, .. }
        | ExprKind::Unary { operand, .. }
        | ExprKind::Return { operand }
        | ExprKind::LoopBreak { operand } => recurse(operand, inside_inline_const),
        ExprKind::Binary { left, right, .. } => {
            recurse(left, inside_inline_const) || recurse(right, inside_inline_const)
        }
        ExprKind::Sequence { first, then } => {
            recurse(first, inside_inline_const) || recurse(then, inside_inline_const)
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
            recurse(condition, inside_inline_const)
                || recurse(then_branch, inside_inline_const)
                || recurse(else_branch, inside_inline_const)
        }
        ExprKind::Unit
        | ExprKind::Integer(_)
        | ExprKind::Bool(_)
        | ExprKind::Char(_)
        | ExprKind::Identifier(_) => false,
    }
}

fn expression_contains_call<const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    depth: usize,
) -> bool {
    if depth == 64 {
        return true;
    }
    let Some(expression) = tree.expression(id) else {
        return true;
    };
    let recurse = |operand| expression_contains_call(tree, operand, depth + 1);
    match expression.kind {
        ExprKind::Call { .. } => true,
        ExprKind::Cast { operand, .. }
        | ExprKind::Ascribe { operand, .. }
        | ExprKind::Unary { operand, .. }
        | ExprKind::Return { operand }
        | ExprKind::LoopBreak { operand }
        | ExprKind::InlineConst { operand } => recurse(operand),
        ExprKind::Binary { left, right, .. } => recurse(left) || recurse(right),
        ExprKind::Sequence { first, then } => recurse(first) || recurse(then),
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        }
        | ExprKind::LoopBreakIf {
            condition,
            then_branch,
            else_branch,
        } => recurse(condition) || recurse(then_branch) || recurse(else_branch),
        ExprKind::Unit
        | ExprKind::Integer(_)
        | ExprKind::Bool(_)
        | ExprKind::Char(_)
        | ExprKind::Identifier(_) => false,
    }
}

fn evaluate_call_aware_integer<
    const MAX_EXPRESSION_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    tree: &ExpressionTree<'_, MAX_EXPRESSION_NODES>,
    id: ExprId,
    ty: crate::IntegerType,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<u128, SemanticErrorKind> {
    let depth = depth.enter_expression()?;
    let bits = ty
        .bits(context.target.pointer_bits())
        .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
    let mask = signed_mask(bits);
    let maximum = maximum_unsigned_value(ty.name(), context.target);
    let expression =
        tree.expression(id)
            .ok_or(SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                ConstEvalError::InvalidExpressionTree,
            )))?;
    let arithmetic_error = |error| SemanticErrorKind::Execution(ExecutionError::Arithmetic(error));
    let recurse = |operand| {
        evaluate_call_aware_integer(context, symbol_count, tree, operand, ty, constants, depth)
    };
    let value = match expression.kind {
        ExprKind::Integer(literal) => {
            if literal.suffix.is_some_and(|suffix| suffix != ty.name())
                || (!ty.is_signed() && literal.value > maximum)
                || (ty.is_signed() && literal.value > signed_maximum(bits) as u128)
            {
                return Err(arithmetic_error(ConstEvalError::Overflow));
            }
            literal.value
        }
        ExprKind::Identifier(name) => {
            if constants.resolve_type(name) != Some(ty) {
                return Err(arithmetic_error(ConstEvalError::InvalidCast));
            }
            constants.resolve(name).ok_or(SemanticErrorKind::Execution(
                ExecutionError::UnknownConstant,
            ))?
        }
        ExprKind::Call { .. } => {
            evaluate_integer_const_call(context, symbol_count, tree, id, ty, constants, depth)?
        }
        ExprKind::Unary {
            operator: UnaryOperator::Negate,
            operand,
        } if ty.is_signed() => {
            if let Some(ExprKind::Integer(literal)) =
                tree.expression(operand).map(|operand| operand.kind)
            {
                let limit = 1u128 << (bits - 1);
                if literal.suffix.is_some_and(|suffix| suffix != ty.name()) || literal.value > limit
                {
                    return Err(arithmetic_error(ConstEvalError::Overflow));
                }
                0u128.wrapping_sub(literal.value) & mask
            } else {
                let value = decode_signed(recurse(operand)?, bits);
                encode_signed(
                    value
                        .checked_neg()
                        .ok_or(arithmetic_error(ConstEvalError::Overflow))?,
                    bits,
                )
                .map_err(|error| match error {
                    ExecutionError::Arithmetic(error) => arithmetic_error(error),
                    other => SemanticErrorKind::Execution(other),
                })?
            }
        }
        ExprKind::Unary {
            operator: UnaryOperator::Not,
            operand,
        } => !recurse(operand)? & mask,
        ExprKind::Cast { operand, target } => {
            if target != ty {
                return Err(arithmetic_error(ConstEvalError::InvalidCast));
            }
            let source_type =
                integer_expression_type(context, symbol_count, tree, operand, constants)
                    .unwrap_or(ty);
            let source = evaluate_call_aware_integer(
                context,
                symbol_count,
                tree,
                operand,
                source_type,
                constants,
                depth,
            )?;
            crate::expression::cast_integer(source, ty, context.target.pointer_bits())
                .map_err(arithmetic_error)?
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = if expression_contains_call(tree, condition, 0) {
                evaluate_call_aware_boolean(
                    context,
                    symbol_count,
                    tree,
                    condition,
                    constants,
                    depth,
                )?
            } else {
                evaluate_boolean_constant(tree, condition, constants, 0)
                    .map_err(SemanticErrorKind::Execution)?
            };
            if condition {
                recurse(then_branch)?
            } else {
                recurse(else_branch)?
            }
        }
        ExprKind::Return { operand }
        | ExprKind::LoopBreak { operand }
        | ExprKind::InlineConst { operand } => recurse(operand)?,
        ExprKind::Binary {
            operator,
            left,
            right,
        } => {
            let left_bits = recurse(left)?;
            let right_bits = recurse(right)?;
            if ty.is_signed() {
                evaluate_call_aware_signed_binary(operator, left_bits, right_bits, bits)
                    .map_err(arithmetic_error)?
            } else {
                evaluate_call_aware_unsigned_binary(operator, left_bits, right_bits, bits)
                    .map_err(arithmetic_error)?
            }
        }
        _ => return Err(arithmetic_error(ConstEvalError::InvalidCast)),
    };
    if (!ty.is_signed() && value > maximum) || (ty.is_signed() && value & mask != value) {
        return Err(arithmetic_error(ConstEvalError::Overflow));
    }
    Ok(value)
}

fn integer_expression_type<
    const MAX_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &ConstCallResolver<'_>,
) -> Option<crate::IntegerType> {
    match tree.expression(id)?.kind {
        ExprKind::InlineConst { operand } => {
            integer_expression_type(context, symbol_count, tree, operand, constants)
        }
        ExprKind::Call { callee, .. } => context.module.items()[..symbol_count]
            .iter()
            .flatten()
            .find_map(|item| match item {
                Item::Function(function)
                    if function.name == callee
                        && function.constant
                        && function.abi == crate::FunctionAbi::Rust =>
                {
                    function
                        .return_type
                        .and_then(|return_type| crate::IntegerType::from_name(return_type.text))
                }
                _ => None,
            }),
        _ => expression_integer_type(tree, id, constants),
    }
}

fn evaluate_call_aware_boolean<
    const MAX_NODES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
>(
    context: &ConstCallContext<'_, '_, MAX_ITEMS, MAX_PARAMETERS>,
    symbol_count: usize,
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &ConstCallResolver<'_>,
    depth: ConstEvalDepth,
) -> Result<bool, SemanticErrorKind> {
    let depth = depth.enter_expression()?;
    let expression =
        tree.expression(id)
            .ok_or(SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                ConstEvalError::InvalidExpressionTree,
            )))?;
    let recurse = |operand| {
        evaluate_call_aware_boolean(context, symbol_count, tree, operand, constants, depth)
    };
    match expression.kind {
        ExprKind::Bool(value) => Ok(value),
        ExprKind::Identifier(name) if constants.resolves_bool(name) => constants
            .resolve(name)
            .map(|value| value != 0)
            .ok_or(SemanticErrorKind::Execution(
                ExecutionError::UnknownConstant,
            )),
        ExprKind::Call { .. } => {
            evaluate_boolean_const_call(context, symbol_count, tree, id, constants, depth)
        }
        ExprKind::InlineConst { operand } => recurse(operand),
        ExprKind::Unary {
            operator: UnaryOperator::Not,
            operand,
        } => Ok(!recurse(operand)?),
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if recurse(condition)? {
                recurse(then_branch)
            } else {
                recurse(else_branch)
            }
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } if matches!(
            operator,
            BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr
                | BinaryOperator::BitAnd
                | BinaryOperator::BitOr
                | BinaryOperator::BitXor
        ) =>
        {
            let left = recurse(left)?;
            if operator == BinaryOperator::LogicalAnd && !left {
                return Ok(false);
            }
            if operator == BinaryOperator::LogicalOr && left {
                return Ok(true);
            }
            let right = recurse(right)?;
            Ok(match operator {
                BinaryOperator::LogicalAnd | BinaryOperator::BitAnd => left && right,
                BinaryOperator::LogicalOr | BinaryOperator::BitOr => left || right,
                BinaryOperator::BitXor => left != right,
                _ => false,
            })
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } if matches!(
            operator,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
        ) =>
        {
            let left_type = integer_expression_type(context, symbol_count, tree, left, constants);
            let right_type = integer_expression_type(context, symbol_count, tree, right, constants);
            if left_type.is_some() && right_type.is_some() && left_type != right_type {
                return Err(SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                    ConstEvalError::InvalidCast,
                )));
            }
            let ty = left_type.or(right_type).unwrap_or(crate::IntegerType::I32);
            let left = evaluate_call_aware_integer(
                context,
                symbol_count,
                tree,
                left,
                ty,
                constants,
                depth,
            )?;
            let right = evaluate_call_aware_integer(
                context,
                symbol_count,
                tree,
                right,
                ty,
                constants,
                depth,
            )?;
            let ordering = if ty.is_signed() {
                let bits = ty
                    .bits(context.target.pointer_bits())
                    .ok_or(SemanticErrorKind::UnsupportedConstCall)?;
                decode_signed(left, bits).cmp(&decode_signed(right, bits))
            } else {
                left.cmp(&right)
            };
            Ok(compare_ordering(operator, ordering))
        }
        _ => Err(SemanticErrorKind::Execution(ExecutionError::Arithmetic(
            ConstEvalError::InvalidCast,
        ))),
    }
}

fn evaluate_call_aware_unsigned_binary(
    operator: BinaryOperator,
    left: u128,
    right: u128,
    bits: u8,
) -> Result<u128, ConstEvalError> {
    let maximum = signed_mask(bits);
    let checked = |value: Option<u128>| {
        value
            .filter(|value| *value <= maximum)
            .ok_or(ConstEvalError::Overflow)
    };
    match operator {
        BinaryOperator::Add => checked(left.checked_add(right)),
        BinaryOperator::Subtract => checked(left.checked_sub(right)),
        BinaryOperator::Multiply => checked(left.checked_mul(right)),
        BinaryOperator::Divide if right == 0 => Err(ConstEvalError::DivisionByZero),
        BinaryOperator::Divide => Ok(left / right),
        BinaryOperator::Remainder if right == 0 => Err(ConstEvalError::DivisionByZero),
        BinaryOperator::Remainder => Ok(left % right),
        BinaryOperator::BitAnd => Ok(left & right),
        BinaryOperator::BitOr => Ok(left | right),
        BinaryOperator::BitXor => Ok(left ^ right),
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            let distance = u32::try_from(right).map_err(|_| ConstEvalError::InvalidShift)?;
            if distance >= u32::from(bits) {
                return Err(ConstEvalError::InvalidShift);
            }
            if operator == BinaryOperator::ShiftLeft {
                checked(left.checked_shl(distance))
            } else {
                Ok(left >> distance)
            }
        }
        _ => Err(ConstEvalError::InvalidCast),
    }
}

fn evaluate_call_aware_signed_binary(
    operator: BinaryOperator,
    left_bits: u128,
    right_bits: u128,
    bits: u8,
) -> Result<u128, ConstEvalError> {
    let left = decode_signed(left_bits, bits);
    let right = decode_signed(right_bits, bits);
    let encode = |value: Option<i128>| {
        encode_signed(value.ok_or(ConstEvalError::Overflow)?, bits).map_err(|error| match error {
            ExecutionError::Arithmetic(error) => error,
            _ => ConstEvalError::InvalidExpressionTree,
        })
    };
    match operator {
        BinaryOperator::Add => encode(left.checked_add(right)),
        BinaryOperator::Subtract => encode(left.checked_sub(right)),
        BinaryOperator::Multiply => encode(left.checked_mul(right)),
        BinaryOperator::Divide if right == 0 => Err(ConstEvalError::DivisionByZero),
        BinaryOperator::Divide => encode(left.checked_div(right)),
        BinaryOperator::Remainder if right == 0 => Err(ConstEvalError::DivisionByZero),
        BinaryOperator::Remainder => encode(left.checked_rem(right)),
        BinaryOperator::BitAnd => Ok(left_bits & right_bits),
        BinaryOperator::BitOr => Ok(left_bits | right_bits),
        BinaryOperator::BitXor => Ok(left_bits ^ right_bits),
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            let distance = u32::try_from(right).map_err(|_| ConstEvalError::InvalidShift)?;
            if distance >= u32::from(bits) {
                return Err(ConstEvalError::InvalidShift);
            }
            if operator == BinaryOperator::ShiftLeft {
                encode(left.checked_shl(distance))
            } else {
                encode(left.checked_shr(distance))
            }
        }
        _ => Err(ConstEvalError::InvalidCast),
    }
}

#[derive(Clone, Copy)]
enum ComparisonInteger {
    Signed(i128),
    Unsigned(u128),
}

fn evaluate_boolean_constant<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &R,
    depth: usize,
) -> Result<bool, ExecutionError> {
    if depth == 64 {
        return Err(ExecutionError::Arithmetic(
            ConstEvalError::NestingLimitExceeded,
        ));
    }
    let expression = tree.expression(id).ok_or(ExecutionError::Arithmetic(
        ConstEvalError::InvalidExpressionTree,
    ))?;
    match expression.kind {
        ExprKind::Bool(value) => Ok(value),
        ExprKind::InlineConst { operand } => {
            evaluate_boolean_constant(tree, operand, constants, depth + 1)
        }
        ExprKind::Identifier(name) if constants.resolves_bool(name) => constants
            .resolve(name)
            .map(|value| value != 0)
            .ok_or(ExecutionError::UnknownConstant),
        ExprKind::Unary {
            operator: UnaryOperator::Not,
            operand,
        } => Ok(!evaluate_boolean_constant(
            tree,
            operand,
            constants,
            depth + 1,
        )?),
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if evaluate_boolean_constant(tree, condition, constants, depth + 1)? {
                evaluate_boolean_constant(tree, then_branch, constants, depth + 1)
            } else {
                evaluate_boolean_constant(tree, else_branch, constants, depth + 1)
            }
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } if matches!(
            operator,
            BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr
                | BinaryOperator::BitAnd
                | BinaryOperator::BitOr
                | BinaryOperator::BitXor
        ) =>
        {
            let left = evaluate_boolean_constant(tree, left, constants, depth + 1)?;
            if operator == BinaryOperator::LogicalAnd && !left {
                return Ok(false);
            }
            if operator == BinaryOperator::LogicalOr && left {
                return Ok(true);
            }
            let right = evaluate_boolean_constant(tree, right, constants, depth + 1)?;
            Ok(match operator {
                BinaryOperator::LogicalAnd | BinaryOperator::BitAnd => left && right,
                BinaryOperator::LogicalOr | BinaryOperator::BitOr => left || right,
                BinaryOperator::BitXor => left != right,
                _ => false,
            })
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } if matches!(
            operator,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
        ) =>
        {
            if let (Ok(left), Ok(right)) = (
                evaluate_boolean_constant(tree, left, constants, depth + 1),
                evaluate_boolean_constant(tree, right, constants, depth + 1),
            ) {
                return Ok(compare_ordering(operator, left.cmp(&right)));
            }
            let left_type = expression_integer_type(tree, left, constants);
            let right_type = expression_integer_type(tree, right, constants);
            if left_type.is_some() && right_type.is_some() && left_type != right_type {
                return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast));
            }
            let inferred_type = left_type.or(right_type).unwrap_or(crate::IntegerType::I32);
            let left = evaluate_comparison_integer(tree, left, inferred_type, constants)?;
            let right = evaluate_comparison_integer(tree, right, inferred_type, constants)?;
            let ordering = match (left, right) {
                (ComparisonInteger::Signed(left), ComparisonInteger::Signed(right)) => {
                    left.cmp(&right)
                }
                (ComparisonInteger::Unsigned(left), ComparisonInteger::Unsigned(right)) => {
                    left.cmp(&right)
                }
                _ => return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
            };
            Ok(compare_ordering(operator, ordering))
        }
        _ => Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
    }
}

fn compare_ordering(operator: BinaryOperator, ordering: core::cmp::Ordering) -> bool {
    match operator {
        BinaryOperator::Equal => ordering.is_eq(),
        BinaryOperator::NotEqual => !ordering.is_eq(),
        BinaryOperator::Less => ordering.is_lt(),
        BinaryOperator::LessEqual => !ordering.is_gt(),
        BinaryOperator::Greater => ordering.is_gt(),
        BinaryOperator::GreaterEqual => !ordering.is_lt(),
        _ => false,
    }
}

fn evaluate_comparison_integer<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    inferred_type: crate::IntegerType,
    constants: &R,
) -> Result<ComparisonInteger, ExecutionError> {
    let expression = tree.expression(id).ok_or(ExecutionError::Arithmetic(
        ConstEvalError::InvalidExpressionTree,
    ))?;
    match expression.kind {
        ExprKind::InlineConst { operand } => {
            evaluate_comparison_integer(tree, operand, inferred_type, constants)
        }
        ExprKind::Integer(literal) => {
            if literal
                .suffix
                .and_then(crate::IntegerType::from_name)
                .is_some_and(|ty| ty.is_signed())
            {
                i128::try_from(literal.value)
                    .map(ComparisonInteger::Signed)
                    .map_err(|_| ExecutionError::Arithmetic(ConstEvalError::Overflow))
            } else if literal.suffix.is_none() {
                if inferred_type.is_signed() {
                    i128::try_from(literal.value)
                        .map(ComparisonInteger::Signed)
                        .map_err(|_| ExecutionError::Arithmetic(ConstEvalError::Overflow))
                } else {
                    Ok(ComparisonInteger::Unsigned(literal.value))
                }
            } else {
                Ok(ComparisonInteger::Unsigned(literal.value))
            }
        }
        ExprKind::Unary {
            operator: UnaryOperator::Negate,
            operand,
        } => match tree.expression(operand).map(|operand| operand.kind) {
            Some(ExprKind::Integer(literal)) => {
                let magnitude = i128::try_from(literal.value)
                    .map_err(|_| ExecutionError::Arithmetic(ConstEvalError::Overflow))?;
                Ok(ComparisonInteger::Signed(-magnitude))
            }
            _ => Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
        },
        ExprKind::Identifier(name) => {
            let ty = constants
                .resolve_type(name)
                .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
            let value = constants
                .resolve(name)
                .ok_or(ExecutionError::UnknownConstant)?;
            if ty.is_signed() {
                let bits = ty
                    .bits(64)
                    .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
                Ok(ComparisonInteger::Signed(decode_signed(value, bits)))
            } else {
                Ok(ComparisonInteger::Unsigned(value))
            }
        }
        ExprKind::Binary { .. } | ExprKind::If { .. } | ExprKind::Cast { .. } => {
            let ty = expression_integer_type(tree, id, constants).unwrap_or(inferred_type);
            if ty.is_signed() {
                let value =
                    evaluate_signed_integer(tree, id, ty, TargetLayout::X86_64, constants, 0)?;
                let bits = ty
                    .bits(64)
                    .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
                Ok(ComparisonInteger::Signed(decode_signed(value, bits)))
            } else {
                let value = tree
                    .evaluate_at(id, constants)
                    .map_err(ExecutionError::Arithmetic)?;
                let bits = ty
                    .bits(64)
                    .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
                if value > signed_mask(bits) {
                    return Err(ExecutionError::Arithmetic(ConstEvalError::Overflow));
                }
                Ok(ComparisonInteger::Unsigned(value))
            }
        }
        _ => Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
    }
}

fn evaluate_signed_integer<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    ty: crate::IntegerType,
    target: TargetLayout,
    constants: &R,
    depth: usize,
) -> Result<u128, ExecutionError> {
    if depth == 64 {
        return Err(ExecutionError::Arithmetic(
            ConstEvalError::NestingLimitExceeded,
        ));
    }
    let bits = ty
        .bits(target.pointer_bits())
        .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
    let expression = tree.expression(id).ok_or(ExecutionError::Arithmetic(
        ConstEvalError::InvalidExpressionTree,
    ))?;
    let recurse =
        |operand| evaluate_signed_integer(tree, operand, ty, target, constants, depth + 1);
    let value = match expression.kind {
        ExprKind::InlineConst { operand } => recurse(operand)?,
        ExprKind::Integer(literal) => {
            if literal.suffix.is_some_and(|suffix| suffix != ty.name())
                || literal.value > signed_maximum(bits) as u128
            {
                return Err(ExecutionError::Arithmetic(ConstEvalError::Overflow));
            }
            literal.value
        }
        ExprKind::Identifier(name) => {
            if constants.resolve_type(name) != Some(ty) {
                return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast));
            }
            constants
                .resolve(name)
                .ok_or(ExecutionError::UnknownConstant)?
        }
        ExprKind::Unary {
            operator: UnaryOperator::Negate,
            operand,
        } => {
            if let Some(ExprKind::Integer(literal)) =
                tree.expression(operand).map(|operand| operand.kind)
            {
                let limit = 1u128 << (bits - 1);
                if literal.suffix.is_some_and(|suffix| suffix != ty.name()) || literal.value > limit
                {
                    return Err(ExecutionError::Arithmetic(ConstEvalError::Overflow));
                }
                0u128.wrapping_sub(literal.value) & signed_mask(bits)
            } else {
                let operand = decode_signed(recurse(operand)?, bits);
                encode_signed(
                    operand
                        .checked_neg()
                        .ok_or(ExecutionError::Arithmetic(ConstEvalError::Overflow))?,
                    bits,
                )?
            }
        }
        ExprKind::Unary {
            operator: UnaryOperator::Not,
            operand,
        } => !recurse(operand)? & signed_mask(bits),
        ExprKind::Cast {
            operand,
            target: cast,
        } => {
            if cast != ty {
                return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast));
            }
            let source_type = expression_integer_type(tree, operand, constants).unwrap_or(ty);
            if source_type.is_signed() {
                let operand = evaluate_signed_integer(
                    tree,
                    operand,
                    source_type,
                    target,
                    constants,
                    depth + 1,
                )?;
                let source_bits = source_type
                    .bits(target.pointer_bits())
                    .ok_or(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))?;
                encode_signed(decode_signed(operand, source_bits), bits)?
            } else {
                let operand = match tree.expression(operand).map(|operand| operand.kind) {
                    Some(ExprKind::Integer(literal)) => literal.value,
                    Some(ExprKind::Identifier(name))
                        if constants.resolve_type(name) == Some(source_type) =>
                    {
                        constants
                            .resolve(name)
                            .ok_or(ExecutionError::UnknownConstant)?
                    }
                    _ => return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
                };
                operand & signed_mask(bits)
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if evaluate_boolean_constant(tree, condition, constants, depth + 1)? {
                recurse(then_branch)?
            } else {
                recurse(else_branch)?
            }
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } => {
            let left_bits = recurse(left)?;
            let right_bits = recurse(right)?;
            let left = decode_signed(left_bits, bits);
            let right = decode_signed(right_bits, bits);
            let arithmetic = |result: Option<i128>| {
                encode_signed(
                    result.ok_or(ExecutionError::Arithmetic(ConstEvalError::Overflow))?,
                    bits,
                )
            };
            match operator {
                BinaryOperator::Add => arithmetic(left.checked_add(right))?,
                BinaryOperator::Subtract => arithmetic(left.checked_sub(right))?,
                BinaryOperator::Multiply => arithmetic(left.checked_mul(right))?,
                BinaryOperator::Divide => {
                    if right == 0 {
                        return Err(ExecutionError::Arithmetic(ConstEvalError::DivisionByZero));
                    }
                    arithmetic(left.checked_div(right))?
                }
                BinaryOperator::Remainder => {
                    if right == 0 {
                        return Err(ExecutionError::Arithmetic(ConstEvalError::DivisionByZero));
                    }
                    arithmetic(left.checked_rem(right))?
                }
                BinaryOperator::BitAnd => left_bits & right_bits,
                BinaryOperator::BitOr => left_bits | right_bits,
                BinaryOperator::BitXor => left_bits ^ right_bits,
                BinaryOperator::ShiftLeft => {
                    let distance = u32::try_from(right)
                        .map_err(|_| ExecutionError::Arithmetic(ConstEvalError::InvalidShift))?;
                    if distance >= u32::from(bits) {
                        return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidShift));
                    }
                    arithmetic(left.checked_shl(distance))?
                }
                BinaryOperator::ShiftRight => {
                    let distance = u32::try_from(right)
                        .map_err(|_| ExecutionError::Arithmetic(ConstEvalError::InvalidShift))?;
                    if distance >= u32::from(bits) {
                        return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidShift));
                    }
                    arithmetic(left.checked_shr(distance))?
                }
                _ => {
                    return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast));
                }
            }
        }
        _ => return Err(ExecutionError::Arithmetic(ConstEvalError::InvalidCast)),
    };
    Ok(value & signed_mask(bits))
}

fn expression_integer_type<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &ExpressionTree<'_, MAX_NODES>,
    id: ExprId,
    constants: &R,
) -> Option<crate::IntegerType> {
    match tree.expression(id)?.kind {
        ExprKind::InlineConst { operand } => expression_integer_type(tree, operand, constants),
        ExprKind::Integer(literal) => literal.suffix.and_then(crate::IntegerType::from_name),
        ExprKind::Identifier(name) => constants.resolve_type(name),
        ExprKind::Cast { target, .. } => Some(target),
        ExprKind::Unary { operand, .. } => expression_integer_type(tree, operand, constants),
        ExprKind::Binary { left, right, .. } => expression_integer_type(tree, left, constants)
            .or_else(|| expression_integer_type(tree, right, constants)),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => expression_integer_type(tree, then_branch, constants)
            .or_else(|| expression_integer_type(tree, else_branch, constants)),
        _ => None,
    }
}

const fn signed_mask(bits: u8) -> u128 {
    if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

const fn signed_maximum(bits: u8) -> i128 {
    if bits == 128 {
        i128::MAX
    } else {
        (1i128 << (bits - 1)) - 1
    }
}

fn decode_signed(value: u128, bits: u8) -> i128 {
    if bits == 128 {
        value as i128
    } else {
        ((value << (128 - bits)) as i128) >> (128 - bits)
    }
}

fn encode_signed(value: i128, bits: u8) -> Result<u128, ExecutionError> {
    let minimum = if bits == 128 {
        i128::MIN
    } else {
        -(1i128 << (bits - 1))
    };
    if value < minimum || value > signed_maximum(bits) {
        return Err(ExecutionError::Arithmetic(ConstEvalError::Overflow));
    }
    Ok(value as u128 & signed_mask(bits))
}

fn maximum_unsigned_value(ty: &str, target: TargetLayout) -> u128 {
    match ty {
        "u8" => u128::from(u8::MAX),
        "u16" => u128::from(u16::MAX),
        "u32" => u128::from(u32::MAX),
        "u64" => u128::from(u64::MAX),
        "u128" => u128::MAX,
        "usize" => u128::MAX >> (128 - target.pointer_bits),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;

    #[test]
    fn resolves_constants_in_declaration_order() {
        let source = "const PAGE: usize = 4096; const ARENA: usize = PAGE * 8; fn boot() {}";
        let module = Parser::new(source).parse_module::<4, 2>().unwrap();
        let table = analyze_constants::<4, 16, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table.get("PAGE").unwrap().value, 4096);
        assert_eq!(table.get("ARENA").unwrap().value, 32768);
        assert_eq!(table.resolve("boot"), None);
    }

    #[test]
    fn rejects_duplicate_names_across_item_kinds() {
        let source = "const VALUE: u32 = 1; fn VALUE() {}";
        let module = Parser::new(source).parse_module::<4, 2>().unwrap();
        let error = analyze_constants::<4, 8, 4, 2>(&module, TargetLayout::X86_64).unwrap_err();
        assert_eq!(error.kind, SemanticErrorKind::DuplicateDefinition);
        assert_eq!(&source[error.span.start..error.span.end], "VALUE");
    }

    #[test]
    fn rejects_forward_references_and_unsupported_types() {
        let forward = Parser::new("const A: usize = B; const B: usize = 1;")
            .parse_module::<4, 2>()
            .unwrap();
        let error = analyze_constants::<4, 8, 4, 2>(&forward, TargetLayout::X86_64).unwrap_err();
        assert_eq!(
            error.kind,
            SemanticErrorKind::Execution(ExecutionError::UnknownConstant)
        );

        let unsupported = Parser::new("const A: f32 = 1;")
            .parse_module::<2, 2>()
            .unwrap();
        let error =
            analyze_constants::<2, 8, 2, 2>(&unsupported, TargetLayout::X86_64).unwrap_err();
        assert_eq!(error.kind, SemanticErrorKind::UnsupportedConstantType);
    }

    #[test]
    fn resolves_signed_literal_constants_with_declared_ranges() {
        let module = Parser::new(
            "const NEGATIVE: i8 = -128; static POSITIVE: i32 = 2147483647; const WORD: isize = -1;",
        )
        .parse_module::<4, 2>()
        .unwrap();
        let values = analyze_constants::<4, 16, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("NEGATIVE"), Some(128));
        assert_eq!(values.resolve("POSITIVE"), Some(i32::MAX as u128));
        assert_eq!(values.resolve("WORD"), Some(u64::MAX as u128));
        assert_eq!(
            values.resolve_type("NEGATIVE"),
            Some(crate::IntegerType::I8)
        );

        for source in ["const BAD: i8 = -129;", "const BAD: i8 = 128;"] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            assert_eq!(
                analyze_constants::<2, 8, 2, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                SemanticErrorKind::ConstantOutOfRange
            );
        }
    }

    #[test]
    fn evaluates_signed_constant_arithmetic_in_the_declared_type() {
        let module = Parser::new(
            "const A: isize = -4 + 3; const B: isize = 3 - 4; const C: isize = -3 * 3; const D: isize = 3 / -1; const E: isize = 1024 >> 4; const F: i8 = -8 | 3;",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<8, 32, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("A"), Some(u64::MAX as u128));
        assert_eq!(values.resolve("B"), Some(u64::MAX as u128));
        assert_eq!(values.resolve("C"), Some((u64::MAX - 8) as u128));
        assert_eq!(values.resolve("D"), Some((u64::MAX - 2) as u128));
        assert_eq!(values.resolve("E"), Some(64));
        assert_eq!(values.resolve("F"), Some(251));

        for source in [
            "const BAD: i8 = 127 + 1;",
            "const BAD: i8 = -128 - 1;",
            "const BAD: i8 = 64 * 2;",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            assert_eq!(
                analyze_constants::<2, 16, 2, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                SemanticErrorKind::ConstantOutOfRange
            );
        }
    }

    #[test]
    fn evaluates_boolean_constants_and_integer_comparisons() {
        let module = Parser::new(
            "const A: bool = true && false; const B: bool = true || A; const C: bool = -1 < 2; const D: bool = 3 == 3; const E: bool = B ^ C;",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<8, 32, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("A"), Some(0));
        assert_eq!(values.resolve("B"), Some(1));
        assert_eq!(values.resolve("C"), Some(1));
        assert_eq!(values.resolve("D"), Some(1));
        assert_eq!(values.resolve("E"), Some(0));
        assert!(values.resolves_bool("E"));

        let invalid = Parser::new("const BAD: bool = 1 + 2;")
            .parse_module::<2, 2>()
            .unwrap();
        assert_eq!(
            analyze_constants::<2, 8, 2, 2>(&invalid, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::Execution(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))
        );
    }

    #[test]
    fn evaluates_signed_constant_casts_and_conditionals_lazily() {
        let module = Parser::new(
            "const WIDE: i32 = -1i8 as i32; const NARROW: i8 = 255u16 as i8; const CHOICE: i32 = if -1 < 2 { WIDE - 6 } else { 1 / 0 }; const FLAG: bool = if false { 1 == 2 } else { CHOICE == -7 };",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<8, 48, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("WIDE"), Some(u32::MAX as u128));
        assert_eq!(values.resolve("NARROW"), Some(u8::MAX as u128));
        assert_eq!(values.resolve("CHOICE"), Some((u32::MAX - 6) as u128));
        assert_eq!(values.resolve("FLAG"), Some(1));

        let mismatch = Parser::new("const BAD: i16 = 1i8 as i32;")
            .parse_module::<2, 2>()
            .unwrap();
        assert_eq!(
            analyze_constants::<2, 8, 2, 2>(&mismatch, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::Execution(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))
        );
    }

    #[test]
    fn compares_named_and_compound_typed_constant_operands() {
        let module = Parser::new(
            "const BASE: i32 = -4; const LIMIT: u8 = 9; const A: bool = BASE + 3 == -1; const B: bool = (LIMIT - 1) * 2 > 15; const C: bool = 1 < LIMIT; const D: bool = if A { BASE * 2 <= -8 } else { 1 / 0 == 0 };",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<8, 64, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        for name in ["A", "B", "C", "D"] {
            assert_eq!(values.resolve(name), Some(1));
        }

        let mixed = Parser::new("const BAD: bool = 1u8 < 2u16;")
            .parse_module::<2, 2>()
            .unwrap();
        assert_eq!(
            analyze_constants::<2, 8, 2, 2>(&mixed, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::Execution(ExecutionError::Arithmetic(ConstEvalError::InvalidCast))
        );

        let overflow = Parser::new("const BAD: bool = 255u8 + 1 > 0;")
            .parse_module::<2, 2>()
            .unwrap();
        assert_eq!(
            analyze_constants::<2, 16, 2, 2>(&overflow, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::Execution(ExecutionError::Arithmetic(ConstEvalError::Overflow))
        );
    }

    #[test]
    fn evaluates_bounded_integer_const_function_calls() {
        let module = Parser::new(
            "const fn add(left: usize, right: usize) -> usize { left + right } const BASE: usize = 40; const SUM: usize = add(BASE, 2);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values = analyze_constants::<4, 32, 4, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("SUM"), Some(42));

        for source in [
            "fn add(a: usize, b: usize) -> usize { a + b } const BAD: usize = add(1, 2);",
            "const BAD: usize = add(1, 2); const fn add(a: usize, b: usize) -> usize { a + b }",
            "const fn add(a: usize, b: usize) -> usize { a + b } const BAD: usize = add(1);",
            "const fn signed(a: i32) -> i32 { a } const BAD: i64 = signed(1);",
        ] {
            let module = Parser::new(source).parse_module::<4, 4>().unwrap();
            assert_eq!(
                analyze_constants::<4, 32, 4, 4>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                SemanticErrorKind::UnsupportedConstCall
            );
        }

        let signed = Parser::new(
            "const fn adjust(value: i32, offset: i32) -> i32 { value + offset } const ANSWER: i32 = adjust(-8, 50); const fn minimum(value: i8) -> i8 { value } const MINIMUM: i8 = minimum(-128);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values = analyze_constants::<4, 32, 4, 4>(&signed, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("ANSWER"), Some(42));
        assert_eq!(values.resolve("MINIMUM"), Some(128));
    }

    #[test]
    fn inline_const_resolves_items_but_rejects_const_function_captures() {
        let module = Parser::new(
            "const ITEM: u32 = 42; const fn item() -> u32 { const { ITEM } } const RESULT: u32 = item();",
        )
        .parse_module::<4, 2>()
        .unwrap();
        let values = analyze_constants::<4, 32, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("RESULT"), Some(42));

        for source in [
            "const fn captured(value: u32) -> u32 { const { value } } const BAD: u32 = captured(42);",
            "const fn captured() -> u32 { let value: u32 = 42; const { value } } const BAD: u32 = captured();",
        ] {
            let module = Parser::new(source).parse_module::<4, 2>().unwrap();
            assert_eq!(
                analyze_constants::<4, 32, 4, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                SemanticErrorKind::UnsupportedConstCall
            );
        }
    }

    #[test]
    fn inline_const_evaluates_bounded_scalar_const_function_calls() {
        let module = Parser::new(
            "const fn offset() -> u32 { 2 } const fn enabled() -> bool { true } const OFFSET: u32 = const { offset() }; const ENABLED: bool = const { enabled() };",
        )
        .parse_module::<6, 2>()
        .unwrap();
        let values = analyze_constants::<4, 48, 6, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("OFFSET"), Some(2));
        assert_eq!(values.resolve("ENABLED"), Some(1));

        let non_const =
            Parser::new("fn offset() -> u32 { 2 } const BAD: u32 = const { offset() };")
                .parse_module::<4, 2>()
                .unwrap();
        assert_eq!(
            analyze_constants::<4, 32, 4, 2>(&non_const, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::UnsupportedConstCall
        );
    }

    #[test]
    fn evaluates_nested_prior_integer_const_function_calls_with_a_depth_bound() {
        let module = Parser::new(
            "const fn sub(left: u32, right: u32) -> u32 { left - right } const fn halve(value: u32) -> u32 { sub(value, 22) } const DIRECT: u32 = sub(sub(88, 44), 22); const THROUGH_BODY: u32 = halve(44);",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 64, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("DIRECT"), Some(22));
        assert_eq!(values.resolve("THROUGH_BODY"), Some(22));

        let shadowing = Parser::new(
            "const value: u32 = 99; const fn identity(value: u32) -> u32 { value } const RESULT: u32 = identity(42);",
        )
        .parse_module::<4, 2>()
        .unwrap();
        let values = analyze_constants::<4, 32, 4, 2>(&shadowing, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("RESULT"), Some(42));

        let too_deep = Parser::new(
            "const fn id(value: u8) -> u8 { value } const BAD: u8 = id(id(id(id(id(id(id(id(id(1)))))))));",
        )
        .parse_module::<4, 2>()
        .unwrap();
        assert_eq!(
            analyze_constants::<2, 64, 4, 2>(&too_deep, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::UnsupportedConstCall
        );
    }

    #[test]
    fn evaluates_integer_const_calls_as_composed_expression_operands() {
        let module = Parser::new(
            "const fn add(left: u8, right: u8) -> u8 { left + right } const fn adjust(value: i32, offset: i32) -> i32 { value + offset } const COMPOSED: u8 = add(20, 1) * 2; const LAZY: u8 = if add(1, 1) < 3 { add(40, 2) } else { add(255, 1) }; const SIGNED: i32 = adjust(-8, 50) - 84; const WIDE: u16 = add(254, 1) as u16 + 1; const PASS: bool = add(20, 22) == 42;",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<8, 96, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("COMPOSED"), Some(42));
        assert_eq!(values.resolve("LAZY"), Some(42));
        assert_eq!(values.resolve("SIGNED"), Some((u32::MAX - 41) as u128));
        assert_eq!(values.resolve("WIDE"), Some(256));
        assert_eq!(values.resolve("PASS"), Some(1));

        let overflow =
            Parser::new("const fn maximum() -> u8 { 255 } const BAD: u8 = maximum() + 1;")
                .parse_module::<4, 2>()
                .unwrap();
        assert_eq!(
            analyze_constants::<2, 32, 4, 2>(&overflow, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::ConstantOutOfRange
        );
    }

    #[test]
    fn evaluates_unconditional_early_returns_in_const_functions() {
        let module = Parser::new(
            "const fn integer(value: u8) -> u8 { let adjusted = value + 1; return adjusted; let ignored = 255 / 0; ignored } const fn boolean() -> bool { return true; false } const VALUE: u8 = integer(41); const FLAG: bool = boolean();",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<4, 64, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("VALUE"), Some(42));
        assert_eq!(values.resolve("FLAG"), Some(1));
    }

    #[test]
    fn evaluates_exhaustive_conditional_returns_lazily_in_const_functions() {
        let module = Parser::new(
            "const fn choose(value: u8) -> u8 { if value == 0 { return 42; } else if value == 1 { return value + 41; } else if value == 2 { return 84 / value; } else { return 126 / value; } } const fn invert(select: bool) -> bool { if select { return false; } else { return true; } } const FIRST: u8 = choose(0); const MIDDLE: u8 = choose(1); const LATER: u8 = choose(2); const FALLBACK: u8 = choose(3); const FLAG: bool = invert(false);",
        )
        .parse_module::<10, 2>()
        .unwrap();
        let values = analyze_constants::<8, 96, 10, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FIRST"), Some(42));
        assert_eq!(values.resolve("MIDDLE"), Some(42));
        assert_eq!(values.resolve("LATER"), Some(42));
        assert_eq!(values.resolve("FALLBACK"), Some(42));
        assert_eq!(values.resolve("FLAG"), Some(1));
    }

    #[test]
    fn evaluates_non_exhaustive_return_chains_and_fallthrough_in_const_functions() {
        let module = Parser::new(
            "const fn choose(value: u8) -> u8 { if value == 0 { return 42; } else if value == 1 { return 42 / value; } 84 / value } const FIRST: u8 = choose(0); const SECOND: u8 = choose(1); const FALLTHROUGH: u8 = choose(2);",
        )
        .parse_module::<6, 2>()
        .unwrap();
        let values = analyze_constants::<4, 64, 6, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FIRST"), Some(42));
        assert_eq!(values.resolve("SECOND"), Some(42));
        assert_eq!(values.resolve("FALLTHROUGH"), Some(42));
    }

    #[test]
    fn evaluates_conditional_const_assignments_lazily() {
        let module = Parser::new(
            "const fn choose(value: u8) -> u8 { let mut result = value; if value == 0 { result = 40; value + 1; result += 2; return result; } else if value == 1 { result = 40; 84 / value; result += 2; return result; } else if value == 2 { result = 40; 42 / value; result += 2; return result; } else { result = 40; value + 10; result += 2; return result; } 1 / value } const FIRST: u8 = choose(0); const MIDDLE: u8 = choose(1); const LATER: u8 = choose(2); const FALLBACK: u8 = choose(3); const fn optional(value: u8) -> u8 { let mut result = 40; if value == 0 { result += 1; value + 1; result += 1; } else if value == 1 { result += 3; value + 1; result -= 1; } result } const UNSELECTED: u8 = optional(2);",
        )
        .parse_module::<8, 2>()
        .unwrap();
        let values = analyze_constants::<8, 96, 8, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FIRST"), Some(42));
        assert_eq!(values.resolve("MIDDLE"), Some(42));
        assert_eq!(values.resolve("LATER"), Some(42));
        assert_eq!(values.resolve("FALLBACK"), Some(42));
        assert_eq!(values.resolve("UNSELECTED"), Some(40));

        let selected_failure = Parser::new(
            "const fn fail(value: u8) -> u8 { let mut result = value; if value == 0 { result = 1; 1 / value; result += 1; } result } const BAD: u8 = fail(0);",
        )
        .parse_module::<4, 2>()
        .unwrap();
        assert!(analyze_constants::<2, 48, 4, 2>(&selected_failure, TargetLayout::X86_64).is_err());
    }

    #[test]
    fn evaluates_scoped_conditional_locals_in_const_functions() {
        let module = Parser::new(
            "const fn choose(value: u8, select: bool) -> u8 { let mut result = value; if select { let mut result: u8 = 40; result += 2; return result; } else { let selected: u8 = 84 / value; selected + 1; result = selected; } result } const SELECTED: u8 = choose(0, true); const FALLBACK: u8 = choose(2, false);",
        )
        .parse_module::<6, 4>()
        .unwrap();
        let values = analyze_constants::<4, 64, 6, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("SELECTED"), Some(42));
        assert_eq!(values.resolve("FALLBACK"), Some(42));
    }

    #[test]
    fn evaluates_scoped_loop_locals_and_expressions_in_const_functions() {
        let module = Parser::new(
            "const fn count(limit: u8, stop: u8) -> u8 { let mut i: u8 = 0; let mut total: u8 = 0; while i < limit { let current: u8 = i + 1; current + 10; i = current; if i % 2 == 0 { let skipped: u8 = current; skipped + 10; continue; } if i == stop { let selected: u8 = current; selected + 20; total += selected; break; } total += current; } total } const STOPPED: u8 = count(5, 3); const COMPLETE: u8 = count(4, 99); const fn choose(value: u8) -> u8 { loop { if value == 0 { let selected: u8 = 42; selected + 1; return selected; } return value; } } const RETURNED: u8 = choose(0);",
        )
        .parse_module::<6, 4>()
        .unwrap();
        let values = analyze_constants::<4, 64, 6, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("STOPPED"), Some(4));
        assert_eq!(values.resolve("COMPLETE"), Some(4));
        assert_eq!(values.resolve("RETURNED"), Some(42));
    }

    #[test]
    fn evaluates_bounded_boolean_const_function_calls_and_parameters() {
        let module = Parser::new(
            "const fn invert(value: bool) -> bool { !value } const fn both(left: bool, right: bool) -> bool { left && right } const fn positive(value: i32) -> bool { value > 0 } const fn choose(condition: bool, left: u8, right: u8) -> u8 { if condition { left } else { right } } const FLAG: bool = invert(false) && both(true, positive(42)); const VALUE: u8 = choose(FLAG, 42, 0);",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 96, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FLAG"), Some(1));
        assert_eq!(values.resolve("VALUE"), Some(42));

        let lazy = Parser::new(
            "const fn dangerous(value: u8) -> bool { value / 0 == 0 } const SAFE: bool = true || dangerous(1);",
        )
        .parse_module::<4, 2>()
        .unwrap();
        assert_eq!(
            analyze_constants::<2, 48, 4, 2>(&lazy, TargetLayout::X86_64)
                .unwrap()
                .resolve("SAFE"),
            Some(1)
        );

        for (source, expected) in [
            (
                "const fn integer() -> u8 { 1 } const BAD: bool = integer();",
                SemanticErrorKind::UnsupportedConstCall,
            ),
            (
                "const fn boolean() -> bool { true } const BAD: u8 = boolean();",
                SemanticErrorKind::UnsupportedConstCall,
            ),
            (
                "const fn boolean(value: bool) -> bool { value } const BAD: bool = boolean(1);",
                SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                    ConstEvalError::InvalidCast,
                )),
            ),
        ] {
            let module = Parser::new(source).parse_module::<4, 2>().unwrap();
            assert_eq!(
                analyze_constants::<2, 32, 4, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                expected
            );
        }
    }

    #[test]
    fn evaluates_terminating_self_recursive_const_functions_with_a_hard_limit() {
        let module = Parser::new(
            "const fn gcd(left: u32, right: u32) -> u32 { if right == 0 { left } else { gcd(right, left % right) } } const fn descend(value: i32) -> i32 { if value == 0 { -42 } else { descend(value - 1) } } const fn even(value: u8) -> bool { if value == 0 { true } else { !even(value - 1) } } const GCD: u32 = gcd(48, 18); const SIGNED: i32 = descend(4); const EVEN: bool = even(4);",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 128, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("GCD"), Some(6));
        assert_eq!(values.resolve("SIGNED"), Some((u32::MAX - 41) as u128));
        assert_eq!(values.resolve("EVEN"), Some(1));

        for source in [
            "const fn forever(value: u8) -> u8 { forever(value) } const BAD: u8 = forever(1);",
            "const fn first(value: u8) -> u8 { second(value) } const fn second(value: u8) -> u8 { first(value) } const BAD: u8 = first(1);",
        ] {
            let module = Parser::new(source).parse_module::<4, 2>().unwrap();
            assert_eq!(
                analyze_constants::<2, 64, 4, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                SemanticErrorKind::UnsupportedConstCall
            );
        }
    }

    #[test]
    fn evaluates_const_function_locals_and_conditional_returns() {
        let module = Parser::new(
            "const fn adjust(value: i32) -> i32 { let offset: i32 = 2; if value < 0 { return -value + offset; } value + offset } const fn enabled(value: bool) -> bool { let inverted: bool = !value; if inverted { return true; } false } const fn identity(value: usize) -> usize { return value; } const ADJUSTED: i32 = adjust(-40); const ENABLED: bool = enabled(false); const IDENTITY: usize = identity(42);",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 96, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("ADJUSTED"), Some(42));
        assert_eq!(values.resolve("ENABLED"), Some(1));
        assert_eq!(values.resolve("IDENTITY"), Some(42));

        let guarded_local = Parser::new(
            "const fn guarded(value: u8, stop: bool) -> u8 { if stop { return 7; } let adjusted: u8 = value + 1; adjusted * 2 } const EARLY: u8 = guarded(255, true); const FALLTHROUGH: u8 = guarded(20, false);",
        )
        .parse_module::<3, 2>()
        .unwrap();
        let values =
            analyze_constants::<2, 64, 3, 2>(&guarded_local, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("EARLY"), Some(7));
        assert_eq!(values.resolve("FALLTHROUGH"), Some(42));

        let guarded_assignment = Parser::new(
            "const fn guarded(value: u8, stop: bool) -> u8 { let mut adjusted: u8 = value; if stop { return 7; } adjusted += 1; adjusted * 2 } const EARLY: u8 = guarded(255, true); const FALLTHROUGH: u8 = guarded(20, false);",
        )
        .parse_module::<3, 2>()
        .unwrap();
        let values =
            analyze_constants::<2, 64, 3, 2>(&guarded_assignment, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("EARLY"), Some(7));
        assert_eq!(values.resolve("FALLTHROUGH"), Some(42));

        let fibonacci = Parser::new(
            "const fn fibonacci(n: u32) -> u32 { if n == 0 { return 0; } let mut previous: u32 = 0; let mut current: u32 = 1; let mut index: u32 = 1; while index < n { current += previous; previous = current - previous; index += 1; } current } const ZERO: u32 = fibonacci(0); const ONE: u32 = fibonacci(1); const TEN: u32 = fibonacci(10);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values = analyze_constants::<4, 96, 4, 4>(&fibonacci, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("ZERO"), Some(0));
        assert_eq!(values.resolve("ONE"), Some(1));
        assert_eq!(values.resolve("TEN"), Some(55));

        let alternating_guards = Parser::new(
            "const fn classify(value: u32, first: bool, second: bool) -> u32 { let mut result: u32 = value; if first { return 1; } result += 1; if second { return 2; } result *= 2; result } const FIRST: u32 = classify(4294967295, true, false); const SECOND: u32 = classify(20, false, true); const FALLTHROUGH: u32 = classify(20, false, false);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values =
            analyze_constants::<4, 96, 4, 4>(&alternating_guards, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FIRST"), Some(1));
        assert_eq!(values.resolve("SECOND"), Some(2));
        assert_eq!(values.resolve("FALLTHROUGH"), Some(42));

        let assignment = Parser::new(
            "const fn mutate() -> u8 { let mut value: u8 = 1; value += 1; value } const fn toggle() -> bool { let mut value: bool = false; value ^= true; value } const fn swap() -> u8 { let mut left: u8 = 1; let mut right: u8 = 2; left ^= right; right ^= left; left ^= right; left } const MUTATED: u8 = mutate(); const TOGGLED: bool = toggle(); const SWAPPED: u8 = swap();",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 64, 8, 4>(&assignment, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("MUTATED"), Some(2));
        assert_eq!(values.resolve("TOGGLED"), Some(1));
        assert_eq!(values.resolve("SWAPPED"), Some(2));

        for (source, expected) in [
            (
                "const fn immutable() -> u8 { let value: u8 = 1; value += 1; value } const BAD: u8 = immutable();",
                SemanticErrorKind::UnsupportedConstCall,
            ),
            (
                "const fn overflow() -> u8 { let mut value: u8 = 255; value += 1; value } const BAD: u8 = overflow();",
                SemanticErrorKind::ConstantOutOfRange,
            ),
            (
                "const fn divide() -> i32 { let mut value: i32 = 42; value /= 0; value } const BAD: i32 = divide();",
                SemanticErrorKind::Execution(ExecutionError::Arithmetic(
                    ConstEvalError::DivisionByZero,
                )),
            ),
        ] {
            let module = Parser::new(source).parse_module::<4, 2>().unwrap();
            assert_eq!(
                analyze_constants::<2, 48, 4, 2>(&module, TargetLayout::X86_64)
                    .unwrap_err()
                    .kind,
                expected
            );
        }
    }

    #[test]
    fn evaluates_bounded_const_function_statement_loops() {
        let module = Parser::new(
            "const fn count(limit: u32) -> u32 { let mut index: u32 = 0; while index < limit { index += 1; } index } const fn even_sum(limit: u32) -> u32 { let mut index: u32 = 0; let mut total: u32 = 0; while index < limit { index += 1; if index % 2 != 0 { continue; } total += index; if total >= 42 { break; } } total } const fn until() -> u32 { let mut value: u32 = 0; loop { value += 1; if value == 42 { break; } } value } const fn immediate() -> u32 { loop { return 42; } } const COUNT: u32 = count(35); const SUM: u32 = even_sum(20); const UNTIL: u32 = until(); const IMMEDIATE: u32 = immediate();",
        )
        .parse_module::<8, 4>()
        .unwrap();
        let values = analyze_constants::<4, 96, 8, 4>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("COUNT"), Some(35));
        assert_eq!(values.resolve("SUM"), Some(42));
        assert_eq!(values.resolve("UNTIL"), Some(42));
        assert_eq!(values.resolve("IMMEDIATE"), Some(42));

        let staged = Parser::new(
            "const fn staged(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } value *= 2; value } const STAGED: u32 = staged(21);",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let values = analyze_constants::<2, 64, 2, 2>(&staged, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("STAGED"), Some(42));

        let post_loop_return = Parser::new(
            "const fn classify(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } if value == 42 { return 7; } value += 1; value } const HIT: u32 = classify(42); const MISS: u32 = classify(3);",
        )
        .parse_module::<3, 2>()
        .unwrap();
        let values =
            analyze_constants::<2, 64, 3, 2>(&post_loop_return, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("HIT"), Some(7));
        assert_eq!(values.resolve("MISS"), Some(4));

        let post_loop_guards = Parser::new(
            "const fn classify(value: u32, limit: u32, first: bool, second: bool) -> u32 { let mut index: u32 = 0; let mut result: u32 = value; while index < limit { index += 1; } if first && index == limit { return 1; } result += 1; if second && index == limit { return 2; } result *= 2; result } const FIRST: u32 = classify(4294967295, 3, true, false); const SECOND: u32 = classify(20, 3, false, true); const FALLTHROUGH: u32 = classify(20, 3, false, false);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values =
            analyze_constants::<4, 96, 4, 4>(&post_loop_guards, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FIRST"), Some(1));
        assert_eq!(values.resolve("SECOND"), Some(2));
        assert_eq!(values.resolve("FALLTHROUGH"), Some(42));

        let sequential = Parser::new(
            "const fn traverse(first: u32, second: u32) -> u32 { let mut value: u32 = 0; while value < first { value += 1; } while value < second { value += 1; } value } const FORWARD: u32 = traverse(17, 42); const ALREADY_PAST: u32 = traverse(42, 17);",
        )
        .parse_module::<3, 2>()
        .unwrap();
        let values = analyze_constants::<2, 64, 3, 2>(&sequential, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FORWARD"), Some(42));
        assert_eq!(values.resolve("ALREADY_PAST"), Some(42));

        let post_loop_local = Parser::new(
            "const fn finish(limit: u32) -> u32 { let mut value: u32 = 0; while value < limit { value += 1; } let offset: u32 = value + 2; offset * 2 } const FINISHED: u32 = finish(19);",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let values =
            analyze_constants::<2, 64, 2, 2>(&post_loop_local, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("FINISHED"), Some(42));

        let primes = Parser::new(
            "const fn prime(value: u32) -> bool { let mut divisor: u32 = 3; if value % 2 == 0 { return false; } loop { if value % divisor == 0 { return false; } if divisor * divisor > value { return true; } divisor += 2; } false } const PRIME: bool = prime(113); const COMPOSITE: bool = prime(117);",
        )
        .parse_module::<4, 4>()
        .unwrap();
        let values = analyze_constants::<4, 96, 4, 4>(&primes, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("PRIME"), Some(1));
        assert_eq!(values.resolve("COMPOSITE"), Some(0));

        let endless = Parser::new(
            "const fn endless() -> u8 { let mut value: u8 = 0; while true { value += 0; } value } const BAD: u8 = endless();",
        )
        .parse_module::<4, 2>()
        .unwrap();
        assert_eq!(
            analyze_constants::<2, 48, 4, 2>(&endless, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::ConstLoopLimitExceeded
        );

        let overflow = Parser::new(
            "const fn overflow() -> u8 { let mut value: u8 = 255; while true { value += 1; } value } const BAD: u8 = overflow();",
        )
        .parse_module::<4, 2>()
        .unwrap();
        assert_eq!(
            analyze_constants::<2, 48, 4, 2>(&overflow, TargetLayout::X86_64)
                .unwrap_err()
                .kind,
            SemanticErrorKind::ConstantOutOfRange
        );
    }

    #[test]
    fn translates_expression_errors_to_module_spans() {
        let source = "const VALUE: usize = 1 + ;";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let error = analyze_constants::<2, 8, 2, 2>(&module, TargetLayout::X86_64).unwrap_err();
        assert_eq!(
            error.kind,
            SemanticErrorKind::Expression(ExpressionErrorKind::ExpectedExpression)
        );
        let Some(Item::Const(constant)) = module.items()[0] else {
            panic!("expected constant")
        };
        assert_eq!(error.span.start, constant.initializer_span.end);
    }

    #[test]
    fn enforces_constant_capacity() {
        let module = Parser::new("const A: u8 = 1; const B: u8 = 2;")
            .parse_module::<2, 2>()
            .unwrap();
        let error = analyze_constants::<1, 8, 2, 2>(&module, TargetLayout::X86_64).unwrap_err();
        assert_eq!(error.kind, SemanticErrorKind::TooManyConstants);
    }

    #[test]
    fn resolves_immutable_statics_with_constants() {
        let module = Parser::new(
            "const BASE: usize = { 10 }; static SHIFTED: usize = { BASE << 4 }; const SELECTED: u32 = if { true } { { 42 } } else { { 1 / 0 } };",
        )
        .parse_module::<4, 2>()
        .unwrap();
        let values = analyze_constants::<4, 24, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        assert_eq!(values.resolve("BASE"), Some(10));
        assert_eq!(values.resolve("SHIFTED"), Some(160));
        assert_eq!(values.resolve("SELECTED"), Some(42));
    }

    #[test]
    fn enforces_declared_integer_and_target_pointer_ranges() {
        let byte = Parser::new("const VALUE: u8 = 256;")
            .parse_module::<2, 2>()
            .unwrap();
        let error = analyze_constants::<2, 8, 2, 2>(&byte, TargetLayout::X86_64).unwrap_err();
        assert_eq!(error.kind, SemanticErrorKind::ConstantOutOfRange);

        let pointer = Parser::new("const VALUE: usize = 4294967296;")
            .parse_module::<2, 2>()
            .unwrap();
        let target = TargetLayout::new(32).unwrap();
        let error = analyze_constants::<2, 8, 2, 2>(&pointer, target).unwrap_err();
        assert_eq!(error.kind, SemanticErrorKind::ConstantOutOfRange);
        assert_eq!(TargetLayout::new(24), None);
    }
}
