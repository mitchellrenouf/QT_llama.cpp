use crate::expression::MAX_CALL_ARGUMENTS;
use crate::{
    ConstantResolver, ExecutionError, ExprKind, ExpressionErrorKind, Function, FunctionAbi,
    IrErrorKind, LoopOperation, ParseErrorKind, Span, lower_expression,
};

const MAX_FUNCTION_IR_INSTRUCTIONS: usize = 256;
const MAX_FUNCTION_STACK_VALUES: usize = 64;
const X86_64_RETURN_CONSTANT_BYTES: usize = 11;
const MAX_X86_64_ABI_PARAMETERS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86_64Abi {
    SystemV,
    Windows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodegenOptions {
    pub overflow_checks: bool,
}

impl CodegenOptions {
    pub const CHECKED: Self = Self {
        overflow_checks: true,
    };
    pub const WRAPPING: Self = Self {
        overflow_checks: false,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MachineCode<const MAX_BYTES: usize> {
    bytes: [u8; MAX_BYTES],
    length: usize,
}

impl<const MAX_BYTES: usize> MachineCode<MAX_BYTES> {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodegenErrorKind {
    StableExportRequired,
    ParametersUnsupported,
    TooManyAbiParameters,
    UnsupportedParameterType,
    UnsupportedRuntimeType,
    UnsupportedRuntimeOperator,
    UnknownRuntimeName,
    RuntimeTypeMismatch,
    InvalidRangeType,
    InvalidRangeBounds,
    RangeEndpointOutOfRange,
    RuntimeExpressionUnsupported,
    MissingReturnType,
    UnsupportedReturnType,
    Expression(ExpressionErrorKind),
    Body(ParseErrorKind),
    Lowering(IrErrorKind),
    Execution(ExecutionError),
    ValueOutOfRange,
    OutputTooSmall,
    DuplicateLocal,
    ImmutableAssignment,
}

#[derive(Clone, Copy)]
struct ConstantLocal<'source> {
    name: &'source str,
    value: u128,
    ty: crate::IntegerType,
    mutable: bool,
}

struct LocalResolver<'values, 'source, R, const MAX_LOCALS: usize> {
    outer: &'values R,
    values: &'values [Option<ConstantLocal<'source>>; MAX_LOCALS],
    count: usize,
}

impl<R: ConstantResolver, const MAX_LOCALS: usize> ConstantResolver
    for LocalResolver<'_, '_, R, MAX_LOCALS>
{
    fn resolve(&self, name: &str) -> Option<u128> {
        self.values[..self.count]
            .iter()
            .flatten()
            .find(|value| value.name == name)
            .map(|value| value.value)
            .or_else(|| self.outer.resolve(name))
    }

    fn resolve_type(&self, name: &str) -> Option<crate::IntegerType> {
        self.outer.resolve_type(name)
    }

    fn resolves_bool(&self, name: &str) -> bool {
        self.outer.resolves_bool(name)
    }

    fn resolve_call(&self, name: &str, arguments: &[u128]) -> Option<u128> {
        self.outer.resolve_call(name, arguments)
    }

    fn resolve_call_type(&self, name: &str, argument_count: usize) -> Option<crate::IntegerType> {
        self.outer.resolve_call_type(name, argument_count)
    }

    fn call_resolves_bool(&self, name: &str, argument_count: usize) -> bool {
        self.outer.call_resolves_bool(name, argument_count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodegenError {
    pub kind: CodegenErrorKind,
    pub span: Span,
}

pub fn compile_x86_64_constant_function<
    'source,
    R: ConstantResolver,
    const MAX_BYTES: usize,
    const MAX_PARAMETERS: usize,
    const MAX_EXPRESSION_NODES: usize,
>(
    function: &Function<'source, MAX_PARAMETERS>,
    resolver: &R,
) -> Result<MachineCode<MAX_BYTES>, CodegenError> {
    if !function.public || function.abi != FunctionAbi::C || !function.no_mangle {
        return Err(CodegenError {
            kind: CodegenErrorKind::StableExportRequired,
            span: function.name_span,
        });
    }
    if function.parameter_count() != 0 {
        return Err(CodegenError {
            kind: CodegenErrorKind::ParametersUnsupported,
            span: function.name_span,
        });
    }
    let return_type = function.return_type.ok_or(CodegenError {
        kind: CodegenErrorKind::MissingReturnType,
        span: function.name_span,
    })?;
    runtime_width(return_type.text).ok_or(CodegenError {
        kind: CodegenErrorKind::UnsupportedReturnType,
        span: return_type.span,
    })?;
    let return_integer_type =
        crate::IntegerType::from_name(return_type.text).ok_or(CodegenError {
            kind: CodegenErrorKind::UnsupportedReturnType,
            span: return_type.span,
        })?;
    let body = function
        .parse_body::<MAX_PARAMETERS>()
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Body(error.kind),
            span: translate_span(function.body_expression_span.start, error.span),
        })?;
    let mut local_values: [Option<ConstantLocal<'source>>; MAX_PARAMETERS] = [None; MAX_PARAMETERS];
    let mut local_count = 0usize;
    for local in body.locals().iter().flatten() {
        if local_values[..local_count]
            .iter()
            .flatten()
            .any(|value| value.name == local.name)
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::DuplicateLocal,
                span: translate_span(function.body_expression_span.start, local.name_span),
            });
        }
        let tree = local
            .parse_initializer::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(
                    function.body_expression_span.start + local.initializer_span.start,
                    error.span,
                ),
            })?;
        let expression_type = runtime_expression_type(function, resolver, &tree, tree.root(), 0)
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(function.body_expression_span.start, local.initializer_span),
            })?;
        let local_type = if let Some(ty) = local.ty {
            let integer_type = crate::IntegerType::from_name(ty.text).ok_or(CodegenError {
                kind: CodegenErrorKind::UnsupportedRuntimeType,
                span: translate_span(function.body_expression_span.start, ty.span),
            })?;
            if runtime_width(ty.text).is_none() {
                return Err(CodegenError {
                    kind: CodegenErrorKind::UnsupportedRuntimeType,
                    span: translate_span(function.body_expression_span.start, ty.span),
                });
            }
            integer_type
        } else {
            match expression_type {
                RuntimeExpressionType::Unit => {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            local.initializer_span,
                        ),
                    });
                }
                RuntimeExpressionType::Integer(Some(ty)) => ty,
                RuntimeExpressionType::Integer(None) => return_integer_type,
                RuntimeExpressionType::Bool | RuntimeExpressionType::Char => {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            local.initializer_span,
                        ),
                    });
                }
            }
        };
        if !runtime_types_compatible(
            expression_type,
            RuntimeExpressionType::Integer(Some(local_type)),
        ) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(function.body_expression_span.start, local.initializer_span),
            });
        }
        let program = lower_expression::<MAX_FUNCTION_IR_INSTRUCTIONS, MAX_EXPRESSION_NODES>(&tree)
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Lowering(error.kind),
                span: translate_span(
                    function.body_expression_span.start + local.initializer_span.start,
                    error.span,
                ),
            })?;
        let resolver = LocalResolver {
            outer: resolver,
            values: &local_values,
            count: local_count,
        };
        let value = program
            .execute::<_, MAX_FUNCTION_STACK_VALUES>(&resolver)
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Execution(error),
                span: translate_span(function.body_expression_span.start, local.initializer_span),
            })?;
        if crate::expression::cast_integer(value, local_type, 64) != Ok(value) {
            return Err(CodegenError {
                kind: CodegenErrorKind::ValueOutOfRange,
                span: translate_span(function.body_expression_span.start, local.initializer_span),
            });
        }
        local_values[local_count] = Some(ConstantLocal {
            name: local.name,
            value,
            ty: local_type,
            mutable: local.mutable,
        });
        local_count += 1;
    }
    for assignment in body.assignments().iter().flatten() {
        let index = local_values[..local_count]
            .iter()
            .position(|value| value.is_some_and(|value| value.name == assignment.name))
            .ok_or(CodegenError {
                kind: CodegenErrorKind::UnknownRuntimeName,
                span: translate_span(function.body_expression_span.start, assignment.name_span),
            })?;
        let target = local_values[index].ok_or(CodegenError {
            kind: CodegenErrorKind::UnknownRuntimeName,
            span: translate_span(function.body_expression_span.start, assignment.name_span),
        })?;
        if !target.mutable {
            return Err(CodegenError {
                kind: CodegenErrorKind::ImmutableAssignment,
                span: translate_span(function.body_expression_span.start, assignment.name_span),
            });
        }
        let tree = assignment
            .parse_value::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(
                    function.body_expression_span.start + assignment.value_span.start,
                    error.span,
                ),
            })?;
        let expression_type = runtime_expression_type(function, resolver, &tree, tree.root(), 0)
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            })?;
        if !runtime_types_compatible(
            expression_type,
            RuntimeExpressionType::Integer(Some(target.ty)),
        ) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            });
        }
        let program = lower_expression::<MAX_FUNCTION_IR_INSTRUCTIONS, MAX_EXPRESSION_NODES>(&tree)
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Lowering(error.kind),
                span: translate_span(
                    function.body_expression_span.start + assignment.value_span.start,
                    error.span,
                ),
            })?;
        let local_resolver = LocalResolver {
            outer: resolver,
            values: &local_values,
            count: local_count,
        };
        let right = program
            .execute::<_, MAX_FUNCTION_STACK_VALUES>(&local_resolver)
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Execution(error),
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            })?;
        let value = if assignment.operator == crate::AssignmentOperator::Assign {
            right
        } else {
            crate::expression::evaluate_binary(
                assignment_binary_operator(assignment.operator).ok_or(CodegenError {
                    kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                    span: translate_span(function.body_expression_span.start, assignment.name_span),
                })?,
                target.value,
                right,
            )
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Execution(ExecutionError::Arithmetic(error)),
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            })?
        };
        if crate::expression::cast_integer(value, target.ty, 64) != Ok(value) {
            return Err(CodegenError {
                kind: CodegenErrorKind::ValueOutOfRange,
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            });
        }
        local_values[index] = Some(ConstantLocal { value, ..target });
    }
    let tree = body
        .parse_tail::<MAX_EXPRESSION_NODES>()
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Expression(error.kind),
            span: translate_span(
                function.body_expression_span.start + body.tail_span.start,
                error.span,
            ),
        })?;
    let expression_type = runtime_expression_type(function, resolver, &tree, tree.root(), 0)
        .map_err(|kind| CodegenError {
            kind,
            span: function.body_expression_span,
        })?;
    let expected_type =
        RuntimeExpressionType::Integer(crate::IntegerType::from_name(return_type.text));
    if !runtime_types_compatible(expression_type, expected_type) {
        return Err(CodegenError {
            kind: CodegenErrorKind::RuntimeTypeMismatch,
            span: function.body_expression_span,
        });
    }
    let program = lower_expression::<MAX_FUNCTION_IR_INSTRUCTIONS, MAX_EXPRESSION_NODES>(&tree)
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Lowering(error.kind),
            span: translate_span(function.body_expression_span.start, error.span),
        })?;
    let resolver = LocalResolver {
        outer: resolver,
        values: &local_values,
        count: local_count,
    };
    let value = program
        .execute::<_, MAX_FUNCTION_STACK_VALUES>(&resolver)
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Execution(error),
            span: function.body_expression_span,
        })?;
    if crate::expression::cast_integer(value, return_integer_type, 64) != Ok(value) {
        return Err(CodegenError {
            kind: CodegenErrorKind::ValueOutOfRange,
            span: function.body_expression_span,
        });
    }
    if MAX_BYTES < X86_64_RETURN_CONSTANT_BYTES {
        return Err(CodegenError {
            kind: CodegenErrorKind::OutputTooSmall,
            span: function.body_expression_span,
        });
    }

    let mut bytes = [0u8; MAX_BYTES];
    // REX.W + MOV r64, imm64; RET. The System V and Windows x64 ABIs both
    // return an integer no wider than 64 bits in RAX.
    bytes[0] = 0x48;
    bytes[1] = 0xb8;
    bytes[2..10].copy_from_slice(&(value as u64).to_le_bytes());
    bytes[10] = 0xc3;
    Ok(MachineCode {
        bytes,
        length: X86_64_RETURN_CONSTANT_BYTES,
    })
}

pub fn compile_x86_64_function<
    'source,
    R: ConstantResolver,
    const MAX_BYTES: usize,
    const MAX_PARAMETERS: usize,
    const MAX_EXPRESSION_NODES: usize,
>(
    function: &Function<'source, MAX_PARAMETERS>,
    resolver: &R,
    abi: X86_64Abi,
) -> Result<MachineCode<MAX_BYTES>, CodegenError> {
    compile_x86_64_function_with_options::<R, MAX_BYTES, MAX_PARAMETERS, MAX_EXPRESSION_NODES>(
        function,
        resolver,
        abi,
        CodegenOptions::CHECKED,
    )
}

pub fn compile_x86_64_function_with_options<
    'source,
    R: ConstantResolver,
    const MAX_BYTES: usize,
    const MAX_PARAMETERS: usize,
    const MAX_EXPRESSION_NODES: usize,
>(
    function: &Function<'source, MAX_PARAMETERS>,
    resolver: &R,
    abi: X86_64Abi,
    options: CodegenOptions,
) -> Result<MachineCode<MAX_BYTES>, CodegenError> {
    let parsed_body = function
        .parse_body::<MAX_PARAMETERS>()
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Body(error.kind),
            span: translate_span(function.body_expression_span.start, error.span),
        })?;
    let body_requires_runtime = parsed_body.assignment_count() != 0
        || parsed_body.conditional_return_count() != 0
        || parsed_body.conditional_return_else_count() != 0
        || parsed_body.conditional_assignment_count() != 0
        || parsed_body.while_loop_count() != 0
        || parsed_body.expression_statement_count() != 0
        || parsed_body.return_count() != 0;
    if function.parameter_count() == 0
        && !body_requires_runtime
        && function
            .return_type
            .and_then(|return_type| runtime_width(return_type.text))
            .is_some()
    {
        return compile_x86_64_constant_function::<
            R,
            MAX_BYTES,
            MAX_PARAMETERS,
            MAX_EXPRESSION_NODES,
        >(function, resolver);
    }
    validate_stable_export(function)?;
    if function.parameter_count() > MAX_X86_64_ABI_PARAMETERS {
        return Err(CodegenError {
            kind: CodegenErrorKind::TooManyAbiParameters,
            span: function.name_span,
        });
    }
    let return_type_text = function.return_type.map_or("()", |ty| ty.text);
    let return_type_span = function
        .return_type
        .map_or(function.name_span, |ty| ty.span);
    let returns_unit = return_type_text == "()";
    let returns_bool = return_type_text == "bool";
    let returns_char = return_type_text == "char";
    if !returns_unit && !returns_bool && !returns_char && runtime_width(return_type_text).is_none()
    {
        return Err(CodegenError {
            kind: CodegenErrorKind::UnsupportedReturnType,
            span: return_type_span,
        });
    }
    let mut integer_operand_type = None;
    for parameter in function.parameters().iter().flatten() {
        if parameter.ty.text == "bool" {
            continue;
        }
        if runtime_width(parameter.ty.text).is_none()
            || (!returns_unit && !returns_bool && parameter.ty.text != return_type_text)
            || integer_operand_type.is_some_and(|ty| ty != parameter.ty.text)
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedParameterType,
                span: parameter.ty.span,
            });
        }
        integer_operand_type = Some(parameter.ty.text);
    }
    let operand_type = if returns_bool || returns_unit {
        integer_operand_type.unwrap_or("u64")
    } else {
        return_type_text
    };
    let expected_type = if returns_unit {
        RuntimeExpressionType::Unit
    } else if returns_bool {
        RuntimeExpressionType::Bool
    } else if returns_char {
        RuntimeExpressionType::Char
    } else {
        RuntimeExpressionType::Integer(crate::IntegerType::from_name(return_type_text))
    };
    let width = runtime_width(operand_type).ok_or(CodegenError {
        kind: CodegenErrorKind::UnsupportedRuntimeType,
        span: function.name_span,
    })?;
    let body = function
        .parse_body::<MAX_PARAMETERS>()
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Body(error.kind),
            span: translate_span(function.body_expression_span.start, error.span),
        })?;
    let mut emitter = RuntimeEmitter::<_, MAX_BYTES, MAX_PARAMETERS> {
        bytes: [0; MAX_BYTES],
        length: 0,
        trap_patches: [0; MAX_BYTES],
        trap_count: 0,
        function,
        resolver,
        abi,
        width,
        overflow_checks: options.overflow_checks,
        saved_parameters: function.parameter_count(),
        locals: [None; MAX_PARAMETERS],
        saved_locals: 0,
        evaluation_depth: 0,
    };
    emitter.emit_prologue()?;
    for statement in body.statements().iter().flatten() {
        match statement {
            crate::BodyStatement::Loop(_) => break,
            crate::BodyStatement::Local(index) => {
                let local = body.locals()[*index].unwrap();
                emitter.emit_local::<MAX_EXPRESSION_NODES>(
                    &local,
                    operand_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Assignment(index) => {
                let assignment = body.assignments()[*index].unwrap();
                emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                    &assignment,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalReturn(index) => {
                let conditional = body.conditional_returns()[*index].unwrap();
                emitter.emit_conditional_return::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalReturnElse(index) => {
                let conditional = body.conditional_return_elses()[*index].unwrap();
                emitter.emit_conditional_return_else::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalAssignment(index) => {
                let conditional = body.conditional_assignments()[*index].unwrap();
                emitter.emit_conditional_assignment::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Expression(index) => {
                let statement = body.expression_statements()[*index].unwrap();
                emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                    &statement,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Return(index) => {
                let return_statement = body.returns()[*index].unwrap();
                emitter.emit_return::<MAX_EXPRESSION_NODES>(
                    &return_statement,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
        }
    }
    for loop_statement in body.while_loops().iter().flatten() {
        let condition_tree = loop_statement
            .parse_condition::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(
                    function.body_expression_span.start + loop_statement.condition_span.start,
                    error.span,
                ),
            })?;
        if let Some(tree) = condition_tree.as_ref() {
            let condition_type = runtime_expression_type_with_locals(
                function,
                emitter.resolver,
                &emitter.locals[..emitter.saved_locals],
                tree,
                tree.root(),
                0,
            )
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(
                    function.body_expression_span.start,
                    loop_statement.condition_span,
                ),
            })?;
            if condition_type != RuntimeExpressionType::Bool {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(
                        function.body_expression_span.start,
                        loop_statement.condition_span,
                    ),
                });
            }
        }
        let loop_start = emitter.length;
        let mut exit_patches = [None; 9];
        let mut exit_count = 0usize;
        if let Some(tree) = condition_tree.as_ref() {
            emitter.emit_expression(tree, tree.root(), 0)?;
            emitter.emit(&[0x48, 0x85, 0xc0])?;
            exit_patches[exit_count] = Some(emitter.emit_forward_branch(0x84)?);
            exit_count += 1;
        }
        let mut ends_with_unconditional_control = false;
        for operation in loop_statement.operations().iter().flatten() {
            if let LoopOperation::Return(loop_return) = operation {
                let value_tree =
                    loop_return
                        .parse_value::<MAX_EXPRESSION_NODES>()
                        .map_err(|error| CodegenError {
                            kind: CodegenErrorKind::Expression(error.kind),
                            span: translate_span(
                                function.body_expression_span.start + loop_return.value_span.start,
                                error.span,
                            ),
                        })?;
                let value_type = runtime_expression_type_with_locals(
                    function,
                    emitter.resolver,
                    &emitter.locals[..emitter.saved_locals],
                    &value_tree,
                    value_tree.root(),
                    0,
                )
                .map_err(|kind| CodegenError {
                    kind,
                    span: translate_span(
                        function.body_expression_span.start,
                        loop_return.value_span,
                    ),
                })?;
                if !runtime_types_compatible(value_type, expected_type) {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            loop_return.value_span,
                        ),
                    });
                }
                emitter.emit_expression(&value_tree, value_tree.root(), 0)?;
                emitter.emit_epilogue()?;
                ends_with_unconditional_control = true;
                break;
            }
            if let LoopOperation::ConditionalReturn(control) = operation {
                let condition_tree =
                    control
                        .parse_condition::<MAX_EXPRESSION_NODES>()
                        .map_err(|error| CodegenError {
                            kind: CodegenErrorKind::Expression(error.kind),
                            span: translate_span(
                                function.body_expression_span.start + control.condition_span.start,
                                error.span,
                            ),
                        })?;
                let condition_type = runtime_expression_type_with_locals(
                    function,
                    emitter.resolver,
                    &emitter.locals[..emitter.saved_locals],
                    &condition_tree,
                    condition_tree.root(),
                    0,
                )
                .map_err(|kind| CodegenError {
                    kind,
                    span: translate_span(
                        function.body_expression_span.start,
                        control.condition_span,
                    ),
                })?;
                if condition_type != RuntimeExpressionType::Bool {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            control.condition_span,
                        ),
                    });
                }
                let value_tree =
                    control
                        .parse_value::<MAX_EXPRESSION_NODES>()
                        .map_err(|error| CodegenError {
                            kind: CodegenErrorKind::Expression(error.kind),
                            span: translate_span(
                                function.body_expression_span.start + control.value_span.start,
                                error.span,
                            ),
                        })?;
                let value_type = runtime_expression_type_with_locals(
                    function,
                    emitter.resolver,
                    &emitter.locals[..emitter.saved_locals],
                    &value_tree,
                    value_tree.root(),
                    0,
                )
                .map_err(|kind| CodegenError {
                    kind,
                    span: translate_span(function.body_expression_span.start, control.value_span),
                })?;
                if !runtime_types_compatible(value_type, expected_type) {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            control.value_span,
                        ),
                    });
                }
                emitter.emit_expression(&condition_tree, condition_tree.root(), 0)?;
                emitter.emit(&[0x48, 0x85, 0xc0])?;
                let skip_return = emitter.emit_forward_branch(0x84)?;
                emitter.emit_expression(&value_tree, value_tree.root(), 0)?;
                emitter.emit_epilogue()?;
                emitter.patch_forward_branch(skip_return)?;
                ends_with_unconditional_control = false;
                continue;
            }
            let LoopOperation::Assignment(assignment) = operation else {
                let (conditional, is_break) = match operation {
                    LoopOperation::ConditionalBreak(control) => (Some(control), true),
                    LoopOperation::ConditionalContinue(control) => (Some(control), false),
                    LoopOperation::Break => (None, true),
                    LoopOperation::Continue => (None, false),
                    LoopOperation::ConditionalReturn(_) => continue,
                    LoopOperation::Return(_) => continue,
                    LoopOperation::Assignment(_) => continue,
                };
                if let Some(control) = conditional {
                    let tree =
                        control
                            .parse_condition::<MAX_EXPRESSION_NODES>()
                            .map_err(|error| CodegenError {
                                kind: CodegenErrorKind::Expression(error.kind),
                                span: translate_span(
                                    function.body_expression_span.start
                                        + control.condition_span.start,
                                    error.span,
                                ),
                            })?;
                    let ty = runtime_expression_type_with_locals(
                        function,
                        emitter.resolver,
                        &emitter.locals[..emitter.saved_locals],
                        &tree,
                        tree.root(),
                        0,
                    )
                    .map_err(|kind| CodegenError {
                        kind,
                        span: translate_span(
                            function.body_expression_span.start,
                            control.condition_span,
                        ),
                    })?;
                    if ty != RuntimeExpressionType::Bool {
                        return Err(CodegenError {
                            kind: CodegenErrorKind::RuntimeTypeMismatch,
                            span: translate_span(
                                function.body_expression_span.start,
                                control.condition_span,
                            ),
                        });
                    }
                    emitter.emit_expression(&tree, tree.root(), 0)?;
                    emitter.emit(&[0x48, 0x85, 0xc0])?;
                    if is_break {
                        exit_patches[exit_count] = Some(emitter.emit_forward_branch(0x85)?);
                        exit_count += 1;
                    } else {
                        let skip_continue = emitter.emit_forward_branch(0x84)?;
                        emitter.emit_backward_branch(loop_start)?;
                        emitter.patch_forward_branch(skip_continue)?;
                    }
                    ends_with_unconditional_control = false;
                } else {
                    if is_break {
                        exit_patches[exit_count] =
                            Some(emitter.emit_unconditional_forward_branch()?);
                        exit_count += 1;
                    } else {
                        emitter.emit_backward_branch(loop_start)?;
                    }
                    ends_with_unconditional_control = true;
                }
                continue;
            };
            ends_with_unconditional_control = false;
            let index = emitter.locals[..emitter.saved_locals]
                .iter()
                .position(|local| local.is_some_and(|local| local.name == assignment.name))
                .ok_or(CodegenError {
                    kind: CodegenErrorKind::UnknownRuntimeName,
                    span: translate_span(function.body_expression_span.start, assignment.name_span),
                })?;
            let target = emitter.locals[index].ok_or(CodegenError {
                kind: CodegenErrorKind::UnknownRuntimeName,
                span: translate_span(function.body_expression_span.start, assignment.name_span),
            })?;
            if !target.mutable {
                return Err(CodegenError {
                    kind: CodegenErrorKind::ImmutableAssignment,
                    span: translate_span(function.body_expression_span.start, assignment.name_span),
                });
            }
            let value_tree = assignment
                .parse_value::<MAX_EXPRESSION_NODES>()
                .map_err(|error| CodegenError {
                    kind: CodegenErrorKind::Expression(error.kind),
                    span: translate_span(
                        function.body_expression_span.start + assignment.value_span.start,
                        error.span,
                    ),
                })?;
            let value_type = runtime_expression_type_with_locals(
                function,
                emitter.resolver,
                &emitter.locals[..emitter.saved_locals],
                &value_tree,
                value_tree.root(),
                0,
            )
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(function.body_expression_span.start, assignment.value_span),
            })?;
            if !runtime_types_compatible(value_type, target.ty) {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(
                        function.body_expression_span.start,
                        assignment.value_span,
                    ),
                });
            }
            if assignment.operator == crate::AssignmentOperator::Assign {
                emitter.emit_expression(&value_tree, value_tree.root(), 0)?;
            } else {
                if target.ty == RuntimeExpressionType::Bool
                    && !matches!(
                        assignment.operator,
                        crate::AssignmentOperator::BitAnd
                            | crate::AssignmentOperator::BitOr
                            | crate::AssignmentOperator::BitXor
                    )
                {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            assignment.name_span,
                        ),
                    });
                }
                emitter.emit_identifier(assignment.name)?;
                emitter.emit(&[0x50])?;
                emitter.evaluation_depth += 1;
                emitter.emit_expression(&value_tree, value_tree.root(), 0)?;
                emitter.evaluation_depth -= 1;
                emitter.emit(&[0x59])?;
                emitter.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                    CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(
                            function.body_expression_span.start,
                            assignment.name_span,
                        ),
                    },
                )?)?;
            }
            emitter.emit_store_stack_slot(emitter.saved_locals - 1 - index)?;
        }
        if !ends_with_unconditional_control {
            emitter.emit_backward_branch(loop_start)?;
        }
        for patch in exit_patches[..exit_count].iter().flatten().copied() {
            emitter.patch_forward_branch(patch)?;
        }
    }
    let mut saw_loop_statement = false;
    for statement in body.statements().iter().flatten() {
        match statement {
            crate::BodyStatement::Loop(_) => saw_loop_statement = true,
            crate::BodyStatement::Local(index) if saw_loop_statement => {
                let local = body.locals()[*index].unwrap();
                emitter.emit_local::<MAX_EXPRESSION_NODES>(
                    &local,
                    operand_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Assignment(index) if saw_loop_statement => {
                let assignment = body.assignments()[*index].unwrap();
                emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                    &assignment,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalReturn(index) if saw_loop_statement => {
                let conditional = body.conditional_returns()[*index].unwrap();
                emitter.emit_conditional_return::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalReturnElse(index) if saw_loop_statement => {
                let conditional = body.conditional_return_elses()[*index].unwrap();
                emitter.emit_conditional_return_else::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::ConditionalAssignment(index) if saw_loop_statement => {
                let conditional = body.conditional_assignments()[*index].unwrap();
                emitter.emit_conditional_assignment::<MAX_EXPRESSION_NODES>(
                    &conditional,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Expression(index) if saw_loop_statement => {
                let statement = body.expression_statements()[*index].unwrap();
                emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                    &statement,
                    function.body_expression_span.start,
                )?;
            }
            crate::BodyStatement::Return(index) if saw_loop_statement => {
                let return_statement = body.returns()[*index].unwrap();
                emitter.emit_return::<MAX_EXPRESSION_NODES>(
                    &return_statement,
                    expected_type,
                    function.body_expression_span.start,
                )?;
            }
            _ => {}
        }
    }
    if body.tail_diverges {
        return emitter.finish();
    }
    let tree = body
        .parse_tail::<MAX_EXPRESSION_NODES>()
        .map_err(|error| CodegenError {
            kind: CodegenErrorKind::Expression(error.kind),
            span: translate_span(
                function.body_expression_span.start + body.tail_span.start,
                error.span,
            ),
        })?;
    let expression_type = runtime_expression_type_with_locals(
        function,
        emitter.resolver,
        &emitter.locals[..emitter.saved_locals],
        &tree,
        tree.root(),
        0,
    )
    .map_err(|kind| CodegenError {
        kind,
        span: translate_span(function.body_expression_span.start, body.tail_span),
    })?;
    if !runtime_types_compatible(expression_type, expected_type) {
        return Err(CodegenError {
            kind: CodegenErrorKind::RuntimeTypeMismatch,
            span: translate_span(function.body_expression_span.start, body.tail_span),
        });
    }
    emitter.emit_expression(&tree, tree.root(), 0)?;
    emitter.emit_epilogue()?;
    emitter.finish()
}

fn validate_stable_export<const MAX_PARAMETERS: usize>(
    function: &Function<'_, MAX_PARAMETERS>,
) -> Result<(), CodegenError> {
    if !function.public || function.abi != FunctionAbi::C || !function.no_mangle {
        Err(CodegenError {
            kind: CodegenErrorKind::StableExportRequired,
            span: function.name_span,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeExpressionType {
    Unit,
    Integer(Option<crate::IntegerType>),
    Bool,
    Char,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeLocal<'source> {
    name: &'source str,
    ty: RuntimeExpressionType,
    mutable: bool,
}

fn runtime_types_compatible(left: RuntimeExpressionType, right: RuntimeExpressionType) -> bool {
    match (left, right) {
        (RuntimeExpressionType::Unit, RuntimeExpressionType::Unit) => true,
        (RuntimeExpressionType::Bool, RuntimeExpressionType::Bool) => true,
        (RuntimeExpressionType::Char, RuntimeExpressionType::Char) => true,
        (RuntimeExpressionType::Integer(left), RuntimeExpressionType::Integer(right)) => {
            left.is_none() || right.is_none() || left == right
        }
        _ => false,
    }
}

fn unify_integer_types(
    left: RuntimeExpressionType,
    right: RuntimeExpressionType,
) -> Result<Option<crate::IntegerType>, CodegenErrorKind> {
    let (RuntimeExpressionType::Integer(left), RuntimeExpressionType::Integer(right)) =
        (left, right)
    else {
        return Err(CodegenErrorKind::RuntimeTypeMismatch);
    };
    if left.is_some() && right.is_some() && left != right {
        return Err(CodegenErrorKind::RuntimeTypeMismatch);
    }
    Ok(left.or(right))
}

fn evaluate_runtime_call_arguments<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
    arguments: [Option<crate::ExprId>; MAX_CALL_ARGUMENTS],
    argument_count: usize,
    resolver: &R,
) -> Result<[u128; MAX_CALL_ARGUMENTS], CodegenErrorKind> {
    let mut values = [0u128; MAX_CALL_ARGUMENTS];
    for (index, argument) in arguments[..argument_count].iter().enumerate() {
        values[index] = tree
            .evaluate_at(
                argument.ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)?,
                resolver,
            )
            .map_err(|error| CodegenErrorKind::Execution(ExecutionError::Arithmetic(error)))?;
    }
    Ok(values)
}

fn validate_runtime_inline_const<
    R: ConstantResolver,
    const MAX_PARAMETERS: usize,
    const MAX_NODES: usize,
>(
    function: &Function<'_, MAX_PARAMETERS>,
    resolver: &R,
    locals: &[Option<RuntimeLocal<'_>>],
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
    id: crate::ExprId,
    depth: usize,
) -> Result<(), CodegenErrorKind> {
    if depth == 64 {
        return Err(CodegenErrorKind::Lowering(
            IrErrorKind::NestingLimitExceeded,
        ));
    }
    let expression = tree
        .expression(id)
        .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)?;
    let recurse = |operand| {
        validate_runtime_inline_const(function, resolver, locals, tree, operand, depth + 1)
    };
    match expression.kind {
        ExprKind::Identifier(name) => {
            let captures_local = locals.iter().flatten().any(|local| local.name == name)
                || function
                    .parameters()
                    .iter()
                    .flatten()
                    .any(|parameter| parameter.name == name);
            if captures_local
                || (!resolver.resolves_bool(name) && resolver.resolve_type(name).is_none())
            {
                return Err(CodegenErrorKind::RuntimeExpressionUnsupported);
            }
        }
        ExprKind::Call {
            callee,
            arguments,
            argument_count,
            ..
        } if evaluate_runtime_call_arguments(tree, arguments, argument_count, resolver)
            .and_then(|values| {
                resolver
                    .resolve_call(callee, &values[..argument_count])
                    .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)
            })
            .is_ok()
            && (resolver.call_resolves_bool(callee, argument_count)
                || resolver.resolve_call_type(callee, argument_count).is_some()) => {}
        ExprKind::Call { .. } => return Err(CodegenErrorKind::RuntimeExpressionUnsupported),
        ExprKind::Cast { operand, .. }
        | ExprKind::Ascribe { operand, .. }
        | ExprKind::Unary { operand, .. }
        | ExprKind::Return { operand }
        | ExprKind::LoopBreak { operand }
        | ExprKind::InlineConst { operand } => recurse(operand)?,
        ExprKind::Binary { left, right, .. } => {
            recurse(left)?;
            recurse(right)?;
        }
        ExprKind::Sequence { first, then } => {
            recurse(first)?;
            recurse(then)?;
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
            recurse(condition)?;
            recurse(then_branch)?;
            recurse(else_branch)?;
        }
        ExprKind::Unit | ExprKind::Integer(_) | ExprKind::Bool(_) | ExprKind::Char(_) => {}
    }
    Ok(())
}

fn runtime_expression_type<
    R: ConstantResolver,
    const MAX_PARAMETERS: usize,
    const MAX_NODES: usize,
>(
    function: &Function<'_, MAX_PARAMETERS>,
    resolver: &R,
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
    id: crate::ExprId,
    depth: usize,
) -> Result<RuntimeExpressionType, CodegenErrorKind> {
    runtime_expression_type_with_locals(function, resolver, &[], tree, id, depth)
}

fn validate_runtime_range_endpoints<
    R: ConstantResolver,
    const MAX_PARAMETERS: usize,
    const MAX_NODES: usize,
>(
    function: &Function<'_, MAX_PARAMETERS>,
    resolver: &R,
    locals: &[Option<RuntimeLocal<'_>>],
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
) -> Result<(), CodegenErrorKind> {
    for validation in tree.range_validations().iter().flatten() {
        let width = match runtime_expression_type_with_locals(
            function,
            resolver,
            locals,
            tree,
            validation.scrutinee,
            1,
        )? {
            RuntimeExpressionType::Integer(Some(ty)) => {
                runtime_width(ty.name()).ok_or(CodegenErrorKind::UnsupportedRuntimeType)?
            }
            RuntimeExpressionType::Char => {
                runtime_width("char").ok_or(CodegenErrorKind::UnsupportedRuntimeType)?
            }
            RuntimeExpressionType::Integer(None) => continue,
            RuntimeExpressionType::Bool => return Err(CodegenErrorKind::InvalidRangeType),
            RuntimeExpressionType::Unit => return Err(CodegenErrorKind::RuntimeTypeMismatch),
        };
        for endpoint in [validation.start, validation.end].into_iter().flatten() {
            let expression = tree
                .expression(endpoint)
                .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)?;
            let fits = match expression.kind {
                ExprKind::Integer(literal) => literal.value <= u128::from(width.maximum),
                ExprKind::Unary {
                    operator: crate::UnaryOperator::Negate,
                    operand,
                } => match tree.expression(operand).map(|operand| operand.kind) {
                    Some(ExprKind::Integer(literal)) => {
                        width.signed && literal.value <= width.minimum.unsigned_abs().into()
                    }
                    _ => true,
                },
                ExprKind::Identifier(name) => match resolver.resolve_type(name) {
                    Some(ty) => {
                        let Some(endpoint_width) = runtime_width(ty.name()) else {
                            return Err(CodegenErrorKind::UnsupportedRuntimeType);
                        };
                        endpoint_width == width
                    }
                    None => true,
                },
                ExprKind::Char(value) => u64::from(value) <= width.maximum,
                ExprKind::Bool(_) => return Err(CodegenErrorKind::InvalidRangeType),
                _ => true,
            };
            if !fits {
                return Err(CodegenErrorKind::RangeEndpointOutOfRange);
            }
        }
    }
    Ok(())
}

fn runtime_expression_type_with_locals<
    R: ConstantResolver,
    const MAX_PARAMETERS: usize,
    const MAX_NODES: usize,
>(
    function: &Function<'_, MAX_PARAMETERS>,
    resolver: &R,
    locals: &[Option<RuntimeLocal<'_>>],
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
    id: crate::ExprId,
    depth: usize,
) -> Result<RuntimeExpressionType, CodegenErrorKind> {
    if depth == 0 {
        tree.validate_ranges(resolver, 64)
            .map_err(|error| match error {
                crate::ConstEvalError::InvalidRangeType => CodegenErrorKind::InvalidRangeType,
                crate::ConstEvalError::InvalidRangeBounds => CodegenErrorKind::InvalidRangeBounds,
                error => CodegenErrorKind::Execution(ExecutionError::Arithmetic(error)),
            })?;
        validate_runtime_range_endpoints(function, resolver, locals, tree)?;
    }
    if depth == 64 {
        return Err(CodegenErrorKind::Lowering(
            IrErrorKind::NestingLimitExceeded,
        ));
    }
    let expression = tree
        .expression(id)
        .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)?;
    match expression.kind {
        ExprKind::Unit => Ok(RuntimeExpressionType::Unit),
        ExprKind::Integer(literal) => Ok(RuntimeExpressionType::Integer(
            literal.suffix.and_then(crate::IntegerType::from_name),
        )),
        ExprKind::Bool(_) => Ok(RuntimeExpressionType::Bool),
        ExprKind::Char(_) => Ok(RuntimeExpressionType::Char),
        ExprKind::Identifier(name) => Ok(locals
            .iter()
            .flatten()
            .rev()
            .find(|local| local.name == name)
            .map(|local| local.ty)
            .or_else(|| {
                function
                    .parameters()
                    .iter()
                    .flatten()
                    .find(|parameter| parameter.name == name)
                    .map(|parameter| {
                        if parameter.ty.text == "bool" {
                            RuntimeExpressionType::Bool
                        } else if parameter.ty.text == "char" {
                            RuntimeExpressionType::Char
                        } else {
                            RuntimeExpressionType::Integer(crate::IntegerType::from_name(
                                parameter.ty.text,
                            ))
                        }
                    })
            })
            .unwrap_or_else(|| {
                if resolver.resolves_bool(name) {
                    RuntimeExpressionType::Bool
                } else {
                    RuntimeExpressionType::Integer(resolver.resolve_type(name))
                }
            })),
        ExprKind::Call {
            callee,
            argument_count,
            ..
        } if resolver.call_resolves_bool(callee, argument_count) => Ok(RuntimeExpressionType::Bool),
        ExprKind::Call {
            callee,
            argument_count,
            ..
        } => resolver
            .resolve_call_type(callee, argument_count)
            .map(|ty| RuntimeExpressionType::Integer(Some(ty)))
            .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported),
        ExprKind::InlineConst { operand } => {
            validate_runtime_inline_const(function, resolver, locals, tree, operand, depth + 1)?;
            runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                operand,
                depth + 1,
            )
        }
        ExprKind::Cast { operand, target } => {
            if !matches!(
                runtime_expression_type_with_locals(
                    function,
                    resolver,
                    locals,
                    tree,
                    operand,
                    depth + 1
                )?,
                RuntimeExpressionType::Integer(_)
            ) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            if runtime_width(target.name()).is_none() {
                return Err(CodegenErrorKind::UnsupportedRuntimeType);
            }
            Ok(RuntimeExpressionType::Integer(Some(target)))
        }
        ExprKind::Ascribe { operand, target } => {
            let operand_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                operand,
                depth + 1,
            )?;
            match target {
                crate::ScalarType::Bool if operand_type == RuntimeExpressionType::Bool => {
                    Ok(RuntimeExpressionType::Bool)
                }
                crate::ScalarType::Integer(target)
                    if matches!(operand_type, RuntimeExpressionType::Integer(_)) =>
                {
                    Ok(RuntimeExpressionType::Integer(Some(target)))
                }
                _ => Err(CodegenErrorKind::RuntimeTypeMismatch),
            }
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
            if runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                condition,
                depth + 1,
            )? != RuntimeExpressionType::Bool
            {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            let then_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                then_branch,
                depth + 1,
            )?;
            let else_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                else_branch,
                depth + 1,
            )?;
            if !runtime_types_compatible(then_type, else_type) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            Ok(match (then_type, else_type) {
                (RuntimeExpressionType::Integer(None), resolved)
                | (resolved, RuntimeExpressionType::Integer(None)) => resolved,
                (resolved, _) => resolved,
            })
        }
        ExprKind::Return { operand } | ExprKind::LoopBreak { operand } => {
            runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                operand,
                depth + 1,
            )
        }
        ExprKind::Sequence { first, then } => {
            runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                first,
                depth + 1,
            )?;
            runtime_expression_type_with_locals(function, resolver, locals, tree, then, depth + 1)
        }
        ExprKind::Unary {
            operator: crate::UnaryOperator::Not,
            operand,
        } => runtime_expression_type_with_locals(
            function,
            resolver,
            locals,
            tree,
            operand,
            depth + 1,
        ),
        ExprKind::Unary {
            operator: crate::UnaryOperator::Negate,
            operand,
        } => {
            let operand_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                operand,
                depth + 1,
            )?;
            if matches!(operand_type, RuntimeExpressionType::Integer(_)) {
                Ok(operand_type)
            } else {
                Err(CodegenErrorKind::RuntimeTypeMismatch)
            }
        }
        ExprKind::Binary {
            operator,
            left,
            right,
        } => {
            if matches!(
                operator,
                crate::BinaryOperator::LogicalAnd | crate::BinaryOperator::LogicalOr
            ) {
                if runtime_expression_type_with_locals(
                    function,
                    resolver,
                    locals,
                    tree,
                    left,
                    depth + 1,
                )? == RuntimeExpressionType::Bool
                    && runtime_expression_type_with_locals(
                        function,
                        resolver,
                        locals,
                        tree,
                        right,
                        depth + 1,
                    )? == RuntimeExpressionType::Bool
                {
                    return Ok(RuntimeExpressionType::Bool);
                }
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            let left_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                left,
                depth + 1,
            )?;
            let right_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                right,
                depth + 1,
            )?;
            if matches!(
                operator,
                crate::BinaryOperator::Equal
                    | crate::BinaryOperator::NotEqual
                    | crate::BinaryOperator::Less
                    | crate::BinaryOperator::LessEqual
                    | crate::BinaryOperator::Greater
                    | crate::BinaryOperator::GreaterEqual
            ) && matches!(
                (left_type, right_type),
                (RuntimeExpressionType::Bool, RuntimeExpressionType::Bool)
                    | (RuntimeExpressionType::Unit, RuntimeExpressionType::Unit)
                    | (RuntimeExpressionType::Char, RuntimeExpressionType::Char)
            ) {
                return Ok(RuntimeExpressionType::Bool);
            }
            if matches!(
                operator,
                crate::BinaryOperator::BitAnd
                    | crate::BinaryOperator::BitOr
                    | crate::BinaryOperator::BitXor
            ) && matches!(
                (left_type, right_type),
                (RuntimeExpressionType::Bool, RuntimeExpressionType::Bool)
            ) {
                return Ok(RuntimeExpressionType::Bool);
            }
            if matches!(
                operator,
                crate::BinaryOperator::ShiftLeft | crate::BinaryOperator::ShiftRight
            ) {
                if !matches!(right_type, RuntimeExpressionType::Integer(_)) {
                    return Err(CodegenErrorKind::RuntimeTypeMismatch);
                }
                return match left_type {
                    RuntimeExpressionType::Integer(_) => Ok(left_type),
                    RuntimeExpressionType::Unit
                    | RuntimeExpressionType::Bool
                    | RuntimeExpressionType::Char => Err(CodegenErrorKind::RuntimeTypeMismatch),
                };
            }
            let integer_type = unify_integer_types(left_type, right_type)?;
            if matches!(
                operator,
                crate::BinaryOperator::Equal
                    | crate::BinaryOperator::NotEqual
                    | crate::BinaryOperator::Less
                    | crate::BinaryOperator::LessEqual
                    | crate::BinaryOperator::Greater
                    | crate::BinaryOperator::GreaterEqual
            ) {
                Ok(RuntimeExpressionType::Bool)
            } else {
                Ok(RuntimeExpressionType::Integer(integer_type))
            }
        }
    }
}

struct RuntimeEmitter<'tree, 'source, R, const MAX_BYTES: usize, const MAX_PARAMETERS: usize> {
    bytes: [u8; MAX_BYTES],
    length: usize,
    trap_patches: [usize; MAX_BYTES],
    trap_count: usize,
    function: &'tree Function<'source, MAX_PARAMETERS>,
    resolver: &'tree R,
    abi: X86_64Abi,
    width: RuntimeWidth,
    overflow_checks: bool,
    saved_parameters: usize,
    locals: [Option<RuntimeLocal<'source>>; MAX_PARAMETERS],
    saved_locals: usize,
    evaluation_depth: usize,
}

impl<'tree, 'source, R: ConstantResolver, const MAX_BYTES: usize, const MAX_PARAMETERS: usize>
    RuntimeEmitter<'tree, 'source, R, MAX_BYTES, MAX_PARAMETERS>
{
    fn emit(&mut self, bytes: &[u8]) -> Result<(), CodegenError> {
        let capacity_error = self.error(CodegenErrorKind::OutputTooSmall);
        let end = self.length.checked_add(bytes.len()).ok_or(capacity_error)?;
        let output = self.bytes.get_mut(self.length..end).ok_or(capacity_error)?;
        output.copy_from_slice(bytes);
        self.length = end;
        Ok(())
    }

    fn error(&self, kind: CodegenErrorKind) -> CodegenError {
        CodegenError {
            kind,
            span: self.function.body_expression_span,
        }
    }

    fn emit_prologue(&mut self) -> Result<(), CodegenError> {
        for index in 0..self.saved_parameters {
            if let Some(encoding) = parameter_push_encoding(self.abi, index) {
                self.emit(encoding)?;
            } else {
                self.emit_stack_parameter_push(index)?;
            }
        }
        Ok(())
    }

    fn emit_stack_parameter_push(&mut self, index: usize) -> Result<(), CodegenError> {
        let (register_count, first_stack_offset) = match self.abi {
            X86_64Abi::Windows => (4usize, 40usize),
            X86_64Abi::SystemV => (6, 8),
        };
        let stack_index = index
            .checked_sub(register_count)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        let original_offset = stack_index
            .checked_mul(8)
            .and_then(|offset| first_stack_offset.checked_add(offset))
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        let current_stack_delta = index
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        let displacement = original_offset
            .checked_add(current_stack_delta)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        if let Ok(displacement) = u8::try_from(displacement) {
            self.emit(&[0x48, 0x8b, 0x44, 0x24, displacement])?;
        } else {
            let displacement = u32::try_from(displacement)
                .map_err(|_| self.error(CodegenErrorKind::TooManyAbiParameters))?;
            self.emit(&[0x48, 0x8b, 0x84, 0x24])?;
            self.emit(&displacement.to_le_bytes())?;
        }
        self.emit(&[0x50])
    }

    fn emit_epilogue(&mut self) -> Result<(), CodegenError> {
        let bytes = self
            .saved_parameters
            .checked_add(self.saved_locals)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if bytes != 0 {
            if bytes <= i8::MAX as usize {
                self.emit(&[0x48, 0x83, 0xc4, bytes as u8])?;
            } else {
                let bytes = u32::try_from(bytes)
                    .map_err(|_| self.error(CodegenErrorKind::TooManyAbiParameters))?;
                self.emit(&[0x48, 0x81, 0xc4])?;
                self.emit(&bytes.to_le_bytes())?;
            }
        }
        self.emit(&[0xc3])
    }

    fn emit_trap_branch(&mut self, opcode: u8) -> Result<(), CodegenError> {
        self.emit(&[0x0f, opcode, 0, 0, 0, 0])?;
        let patch = self.length - 4;
        let capacity_error = self.error(CodegenErrorKind::OutputTooSmall);
        let slot = self
            .trap_patches
            .get_mut(self.trap_count)
            .ok_or(capacity_error)?;
        *slot = patch;
        self.trap_count += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<MachineCode<MAX_BYTES>, CodegenError> {
        if self.trap_count != 0 {
            let trap = self.length;
            self.emit(&[0x0f, 0x0b])?;
            for index in 0..self.trap_count {
                let patch = self.trap_patches[index];
                let displacement = trap as isize - (patch + 4) as isize;
                let displacement = i32::try_from(displacement)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?;
                self.bytes[patch..patch + 4].copy_from_slice(&displacement.to_le_bytes());
            }
        }
        Ok(MachineCode {
            bytes: self.bytes,
            length: self.length,
        })
    }

    fn emit_expression<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        id: crate::ExprId,
        depth: usize,
    ) -> Result<(), CodegenError> {
        if depth == 64 {
            return Err(self.error(CodegenErrorKind::Lowering(
                IrErrorKind::NestingLimitExceeded,
            )));
        }
        let expression = *tree
            .expression(id)
            .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
        match expression.kind {
            ExprKind::Unit => self.emit(&[0x31, 0xc0])?,
            ExprKind::Integer(literal) => {
                if literal.value > u128::from(self.width.maximum) {
                    return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                }
                let value = u64::try_from(literal.value)
                    .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
                self.emit(&[0x48, 0xb8])?;
                self.emit(&value.to_le_bytes())?;
            }
            ExprKind::Bool(value) => {
                self.emit(&[0x48, 0xb8])?;
                self.emit(&u64::from(value).to_le_bytes())?;
            }
            ExprKind::Char(value) => {
                self.emit(&[0x48, 0xb8])?;
                self.emit(&u64::from(value).to_le_bytes())?;
            }
            ExprKind::Identifier(name) => self.emit_identifier(name)?,
            ExprKind::Call {
                callee,
                arguments,
                argument_count,
            } => {
                let arguments =
                    evaluate_runtime_call_arguments(tree, arguments, argument_count, self.resolver)
                        .map_err(|kind| self.error(kind))?;
                let value = self
                    .resolver
                    .resolve_call(callee, &arguments[..argument_count])
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                let resolved_type = self.resolver.resolve_call_type(callee, argument_count);
                if let Some(ty) = resolved_type
                    && (ty.bits(64) != Some(self.width.bits) || ty.is_signed() != self.width.signed)
                {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                }
                let maximum_bits = if self.width.bits == 64 {
                    u128::from(u64::MAX)
                } else {
                    (1u128 << self.width.bits) - 1
                };
                if value > maximum_bits {
                    return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                }
                let mut value = u64::try_from(value)
                    .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
                if resolved_type.is_some_and(crate::IntegerType::is_signed) && self.width.bits < 64
                {
                    let sign_bit = 1u64 << (self.width.bits - 1);
                    if value & sign_bit != 0 {
                        value |= !((1u64 << self.width.bits) - 1);
                    }
                }
                self.emit(&[0x48, 0xb8])?;
                self.emit(&value.to_le_bytes())?;
            }
            ExprKind::Cast { operand, target } => {
                self.emit_expression(tree, operand, depth + 1)?;
                let width = runtime_width(target.name())
                    .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
                self.emit_normalize_width(width)?;
            }
            ExprKind::Ascribe { operand, target } => {
                self.emit_expression(tree, operand, depth + 1)?;
                match target {
                    crate::ScalarType::Integer(target) => {
                        let width = runtime_width(target.name())
                            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
                        self.emit_normalize_width(width)?;
                    }
                    crate::ScalarType::Bool => {
                        self.emit(&[0x48, 0x85, 0xc0, 0x0f, 0x95, 0xc0, 0x0f, 0xb6, 0xc0])?;
                    }
                }
            }
            ExprKind::Unary {
                operator: crate::UnaryOperator::Not,
                operand,
            } => {
                let operand_type = runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    operand,
                    0,
                )
                .map_err(|kind| self.error(kind))?;
                self.emit_expression(tree, operand, depth + 1)?;
                if operand_type == RuntimeExpressionType::Bool {
                    self.emit(&[0x48, 0x85, 0xc0, 0x0f, 0x94, 0xc0, 0x0f, 0xb6, 0xc0])?;
                } else {
                    self.emit(&[0x48, 0xf7, 0xd0])?;
                    self.emit_normalize()?;
                }
            }
            ExprKind::Unary {
                operator: crate::UnaryOperator::Negate,
                operand,
            } => {
                if !self.width.signed {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                }
                if let Some(ExprKind::Integer(literal)) =
                    tree.expression(operand).map(|expression| expression.kind)
                {
                    let magnitude_limit = u128::from(self.width.maximum) + 1;
                    if literal.value > magnitude_limit {
                        return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                    }
                    self.emit(&[0x48, 0xb8])?;
                    self.emit(&(literal.value as u64).to_le_bytes())?;
                } else {
                    self.emit_expression(tree, operand, depth + 1)?;
                }
                self.emit(&[0x48, 0xf7, 0xd8])?;
                if self.overflow_checks {
                    if self.width.bits == 64 {
                        self.emit_trap_branch(0x80)?;
                    } else {
                        self.emit_range_check()?;
                    }
                }
                self.emit_normalize()?;
            }
            ExprKind::Binary {
                operator,
                left,
                right,
            } => {
                if matches!(
                    operator,
                    crate::BinaryOperator::LogicalAnd | crate::BinaryOperator::LogicalOr
                ) {
                    self.emit_expression(tree, left, depth + 1)?;
                    self.emit(&[0x48, 0x85, 0xc0])?;
                    let branch = self.emit_forward_branch(
                        if operator == crate::BinaryOperator::LogicalAnd {
                            0x84
                        } else {
                            0x85
                        },
                    )?;
                    self.emit_expression(tree, right, depth + 1)?;
                    self.patch_forward_branch(branch)?;
                    return Ok(());
                }
                self.emit_expression(tree, left, depth + 1)?;
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
                self.emit_expression(tree, right, depth + 1)?;
                self.evaluation_depth -= 1;
                self.emit(&[0x59])?;
                self.emit_binary(operator)?;
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
                self.emit_expression(tree, condition, depth + 1)?;
                self.emit(&[0x48, 0x85, 0xc0])?;
                let else_patch = self.emit_forward_branch(0x84)?;
                self.emit_expression(tree, then_branch, depth + 1)?;
                let end_patch = self.emit_unconditional_forward_branch()?;
                self.patch_forward_branch(else_patch)?;
                self.emit_expression(tree, else_branch, depth + 1)?;
                self.patch_forward_branch(end_patch)?;
            }
            ExprKind::Return { operand } | ExprKind::LoopBreak { operand } => {
                self.emit_expression(tree, operand, depth + 1)?;
            }
            ExprKind::Sequence { first, then } => {
                self.emit_expression(tree, first, depth + 1)?;
                self.emit_expression(tree, then, depth + 1)?;
            }
            ExprKind::InlineConst { operand } => {
                let expression_type = runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    operand,
                    depth + 1,
                )
                .map_err(|kind| self.error(kind))?;
                let value = tree.evaluate_at(operand, self.resolver).map_err(|error| {
                    self.error(CodegenErrorKind::Execution(ExecutionError::Arithmetic(
                        error,
                    )))
                })?;
                match expression_type {
                    RuntimeExpressionType::Unit => self.emit(&[0x31, 0xc0])?,
                    RuntimeExpressionType::Bool => {
                        if value > 1 {
                            return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                        }
                        self.emit(&[0x48, 0xb8])?;
                        self.emit(&(value as u64).to_le_bytes())?;
                    }
                    RuntimeExpressionType::Char => {
                        if value > u128::from(u32::MAX) || char::from_u32(value as u32).is_none() {
                            return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                        }
                        self.emit(&[0x48, 0xb8])?;
                        self.emit(&(value as u64).to_le_bytes())?;
                    }
                    RuntimeExpressionType::Integer(resolved_type) => {
                        if let Some(ty) = resolved_type
                            && (ty.bits(64) != Some(self.width.bits)
                                || ty.is_signed() != self.width.signed)
                        {
                            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                        }
                        let maximum_bits = if self.width.bits == 64 {
                            u128::from(u64::MAX)
                        } else {
                            (1u128 << self.width.bits) - 1
                        };
                        let low_value = value & maximum_bits;
                        if resolved_type.is_some_and(crate::IntegerType::is_signed) {
                            let sign_extended = if low_value & (1u128 << (self.width.bits - 1)) != 0
                            {
                                low_value | !maximum_bits
                            } else {
                                low_value
                            };
                            if value != low_value && value != sign_extended {
                                return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                            }
                        } else if value != low_value {
                            return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                        }
                        let mut value = low_value as u64;
                        if resolved_type.is_some_and(crate::IntegerType::is_signed)
                            && self.width.bits < 64
                        {
                            let sign_bit = 1u64 << (self.width.bits - 1);
                            if value & sign_bit != 0 {
                                value |= !((1u64 << self.width.bits) - 1);
                            }
                        }
                        self.emit(&[0x48, 0xb8])?;
                        self.emit(&value.to_le_bytes())?;
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_forward_branch(&mut self, opcode: u8) -> Result<usize, CodegenError> {
        self.emit(&[0x0f, opcode, 0, 0, 0, 0])?;
        Ok(self.length - 4)
    }

    fn emit_unconditional_forward_branch(&mut self) -> Result<usize, CodegenError> {
        self.emit(&[0xe9, 0, 0, 0, 0])?;
        Ok(self.length - 4)
    }

    fn emit_backward_branch(&mut self, target: usize) -> Result<(), CodegenError> {
        let instruction_end = self
            .length
            .checked_add(5)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        let displacement = isize::try_from(target)
            .ok()
            .and_then(|target| {
                isize::try_from(instruction_end)
                    .ok()
                    .and_then(|end| target.checked_sub(end))
            })
            .and_then(|value| i32::try_from(value).ok())
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        self.emit(&[0xe9])?;
        self.emit(&displacement.to_le_bytes())
    }

    fn patch_forward_branch(&mut self, patch: usize) -> Result<(), CodegenError> {
        let displacement = self.length as isize - (patch + 4) as isize;
        let displacement = i32::try_from(displacement)
            .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?;
        let capacity_error = self.error(CodegenErrorKind::OutputTooSmall);
        let target = self.bytes.get_mut(patch..patch + 4).ok_or(capacity_error)?;
        target.copy_from_slice(&displacement.to_le_bytes());
        Ok(())
    }

    fn emit_identifier(&mut self, name: &str) -> Result<(), CodegenError> {
        if let Some(index) = self.locals[..self.saved_locals]
            .iter()
            .flatten()
            .position(|local| local.name == name)
        {
            let local =
                self.locals[index].ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
            let slots = self.saved_locals - 1 - index + self.evaluation_depth;
            self.emit_stack_slot(slots)?;
            if local.ty == RuntimeExpressionType::Bool {
                self.emit(&[0x0f, 0xb6, 0xc0])
            } else {
                self.emit_normalize()
            }
        } else if let Some(index) = self
            .function
            .parameters()
            .iter()
            .flatten()
            .position(|parameter| parameter.name == name)
        {
            let is_bool = self
                .function
                .parameters()
                .iter()
                .flatten()
                .nth(index)
                .is_some_and(|parameter| parameter.ty.text == "bool");
            let slots =
                self.saved_locals + self.saved_parameters - 1 - index + self.evaluation_depth;
            self.emit_stack_slot(slots)?;
            if is_bool {
                self.emit(&[0x0f, 0xb6, 0xc0])
            } else {
                self.emit_normalize()
            }
        } else {
            let value = self
                .resolver
                .resolve(name)
                .ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
            let resolved_type = self.resolver.resolve_type(name);
            if let Some(ty) = resolved_type {
                if ty.bits(64) != Some(self.width.bits) || ty.is_signed() != self.width.signed {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                }
                let maximum_bits = if self.width.bits == 64 {
                    u128::from(u64::MAX)
                } else {
                    (1u128 << self.width.bits) - 1
                };
                if value > maximum_bits {
                    return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                }
            }
            let mut value =
                u64::try_from(value).map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
            if resolved_type.is_some_and(crate::IntegerType::is_signed) && self.width.bits < 64 {
                let sign_bit = 1u64 << (self.width.bits - 1);
                if value & sign_bit != 0 {
                    value |= !((1u64 << self.width.bits) - 1);
                }
            }
            if resolved_type.is_none() && value > self.width.maximum {
                return Err(self.error(CodegenErrorKind::ValueOutOfRange));
            }
            self.emit(&[0x48, 0xb8])?;
            self.emit(&value.to_le_bytes())
        }
    }

    fn emit_stack_slot(&mut self, slots: usize) -> Result<(), CodegenError> {
        let displacement = slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if displacement <= usize::from(u8::MAX) {
            self.emit(&[0x48, 0x8b, 0x44, 0x24, displacement as u8])
        } else {
            self.emit(&[0x48, 0x8b, 0x84, 0x24])?;
            self.emit(
                &u32::try_from(displacement)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?
                    .to_le_bytes(),
            )
        }
    }

    fn emit_store_stack_slot(&mut self, slots: usize) -> Result<(), CodegenError> {
        let displacement = slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if displacement <= usize::from(u8::MAX) {
            self.emit(&[0x48, 0x89, 0x44, 0x24, displacement as u8])
        } else {
            self.emit(&[0x48, 0x89, 0x84, 0x24])?;
            self.emit(
                &u32::try_from(displacement)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?
                    .to_le_bytes(),
            )
        }
    }

    fn emit_local<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        local: &crate::LocalBinding<'source>,
        operand_type: &str,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        if self
            .function
            .parameters()
            .iter()
            .flatten()
            .any(|parameter| parameter.name == local.name)
            || self.locals[..self.saved_locals]
                .iter()
                .flatten()
                .any(|existing| existing.name == local.name)
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::DuplicateLocal,
                span: translate_span(body_start, local.name_span),
            });
        }
        let tree = local
            .parse_initializer::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + local.initializer_span.start, error.span),
            })?;
        let initializer_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &tree,
            tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, local.initializer_span),
        })?;
        let local_type = if let Some(ty) = local.ty {
            if ty.text == "()" {
                RuntimeExpressionType::Unit
            } else if ty.text == "bool" {
                RuntimeExpressionType::Bool
            } else if ty.text == "char" {
                RuntimeExpressionType::Char
            } else {
                let integer_type = crate::IntegerType::from_name(ty.text).ok_or(CodegenError {
                    kind: CodegenErrorKind::UnsupportedRuntimeType,
                    span: translate_span(body_start, ty.span),
                })?;
                if runtime_width(ty.text) != Some(self.width) {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(body_start, ty.span),
                    });
                }
                RuntimeExpressionType::Integer(Some(integer_type))
            }
        } else {
            match initializer_type {
                RuntimeExpressionType::Unit => RuntimeExpressionType::Unit,
                RuntimeExpressionType::Bool => RuntimeExpressionType::Bool,
                RuntimeExpressionType::Char => RuntimeExpressionType::Char,
                RuntimeExpressionType::Integer(Some(ty)) => {
                    if ty.bits(64) != Some(self.width.bits) {
                        return Err(CodegenError {
                            kind: CodegenErrorKind::RuntimeTypeMismatch,
                            span: translate_span(body_start, local.initializer_span),
                        });
                    }
                    RuntimeExpressionType::Integer(Some(ty))
                }
                RuntimeExpressionType::Integer(None) => {
                    RuntimeExpressionType::Integer(crate::IntegerType::from_name(operand_type))
                }
            }
        };
        if !runtime_types_compatible(initializer_type, local_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, local.initializer_span),
            });
        }
        self.emit_expression(&tree, tree.root(), 0)?;
        self.emit(&[0x50])?;
        self.locals[self.saved_locals] = Some(RuntimeLocal {
            name: local.name,
            ty: local_type,
            mutable: local.mutable,
        });
        self.saved_locals += 1;
        Ok(())
    }

    fn emit_expression_statement<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        statement: &crate::ExpressionStatement<'_>,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        if statement.expression.is_empty() {
            return Ok(());
        }
        let tree = statement
            .parse::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + statement.span.start, error.span),
            })?;
        runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &tree,
            tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, statement.span),
        })?;
        self.emit_expression(&tree, tree.root(), 0)
    }

    fn emit_assignment<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        let index = self.locals[..self.saved_locals]
            .iter()
            .position(|local| local.is_some_and(|local| local.name == assignment.name))
            .ok_or(CodegenError {
                kind: CodegenErrorKind::UnknownRuntimeName,
                span: translate_span(body_start, assignment.name_span),
            })?;
        let target = self.locals[index].ok_or(CodegenError {
            kind: CodegenErrorKind::UnknownRuntimeName,
            span: translate_span(body_start, assignment.name_span),
        })?;
        if !target.mutable {
            return Err(CodegenError {
                kind: CodegenErrorKind::ImmutableAssignment,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        if target.ty == RuntimeExpressionType::Bool
            && !matches!(
                assignment.operator,
                crate::AssignmentOperator::Assign
                    | crate::AssignmentOperator::BitAnd
                    | crate::AssignmentOperator::BitOr
                    | crate::AssignmentOperator::BitXor
            )
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let tree = assignment
            .parse_value::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + assignment.value_span.start, error.span),
            })?;
        let value_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &tree,
            tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, assignment.value_span),
        })?;
        if !runtime_types_compatible(value_type, target.ty) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, assignment.value_span),
            });
        }
        if assignment.operator == crate::AssignmentOperator::Assign {
            self.emit_expression(&tree, tree.root(), 0)?;
        } else {
            self.emit_identifier(assignment.name)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_expression(&tree, tree.root(), 0)?;
            self.evaluation_depth -= 1;
            self.emit(&[0x59])?;
            self.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                CodegenError {
                    kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                    span: translate_span(body_start, assignment.name_span),
                },
            )?)?;
        }
        self.emit_store_stack_slot(self.saved_locals - 1 - index)
    }

    fn emit_conditional_return<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        conditional: &crate::ConditionalReturn<'_>,
        expected_type: RuntimeExpressionType,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        let condition_tree = conditional
            .parse_condition::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + conditional.condition_span.start, error.span),
            })?;
        let condition_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &condition_tree,
            condition_tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, conditional.condition_span),
        })?;
        if condition_type != RuntimeExpressionType::Bool {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, conditional.condition_span),
            });
        }
        let value_tree = conditional
            .parse_value::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + conditional.value_span.start, error.span),
            })?;
        let value_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &value_tree,
            value_tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, conditional.value_span),
        })?;
        if !runtime_types_compatible(value_type, expected_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, conditional.value_span),
            });
        }
        self.emit_expression(&condition_tree, condition_tree.root(), 0)?;
        self.emit(&[0x48, 0x85, 0xc0])?;
        let skip_return = self.emit_forward_branch(0x84)?;
        self.emit_expression(&value_tree, value_tree.root(), 0)?;
        self.emit_epilogue()?;
        self.patch_forward_branch(skip_return)
    }

    fn emit_conditional_assignment<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        conditional: &crate::ConditionalAssignment<'_>,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        let mut end_patches = [None; crate::MAX_CONDITIONAL_ASSIGNMENT_BRANCHES];
        let mut end_count = 0usize;
        for branch in conditional.branches().iter().flatten() {
            let condition_tree =
                branch
                    .parse_condition::<MAX_EXPRESSION_NODES>()
                    .map_err(|error| CodegenError {
                        kind: CodegenErrorKind::Expression(error.kind),
                        span: translate_span(body_start + branch.condition_span.start, error.span),
                    })?;
            let condition_type = runtime_expression_type_with_locals(
                self.function,
                self.resolver,
                &self.locals[..self.saved_locals],
                &condition_tree,
                condition_tree.root(),
                0,
            )
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(body_start, branch.condition_span),
            })?;
            if condition_type != RuntimeExpressionType::Bool {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(body_start, branch.condition_span),
                });
            }
            self.emit_expression(&condition_tree, condition_tree.root(), 0)?;
            self.emit(&[0x48, 0x85, 0xc0])?;
            let false_branch = self.emit_forward_branch(0x84)?;
            for action in branch.actions().iter().flatten() {
                match action {
                    crate::ConditionalAssignmentAction::Assignment(assignment) => {
                        self.emit_assignment::<MAX_EXPRESSION_NODES>(assignment, body_start)?;
                    }
                    crate::ConditionalAssignmentAction::Expression(statement) => {
                        self.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                            statement, body_start,
                        )?;
                    }
                }
            }
            end_patches[end_count] = Some(self.emit_unconditional_forward_branch()?);
            end_count += 1;
            self.patch_forward_branch(false_branch)?;
        }
        for action in conditional.else_actions().iter().flatten() {
            match action {
                crate::ConditionalAssignmentAction::Assignment(assignment) => {
                    self.emit_assignment::<MAX_EXPRESSION_NODES>(assignment, body_start)?;
                }
                crate::ConditionalAssignmentAction::Expression(statement) => {
                    self.emit_expression_statement::<MAX_EXPRESSION_NODES>(statement, body_start)?;
                }
            }
        }
        for patch in end_patches[..end_count].iter().flatten().copied() {
            self.patch_forward_branch(patch)?;
        }
        Ok(())
    }

    fn emit_return<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        return_statement: &crate::LoopReturn<'_>,
        expected_type: RuntimeExpressionType,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        let value_tree = return_statement
            .parse_value::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + return_statement.value_span.start, error.span),
            })?;
        let value_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &value_tree,
            value_tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, return_statement.value_span),
        })?;
        if !runtime_types_compatible(value_type, expected_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, return_statement.value_span),
            });
        }
        self.emit_expression(&value_tree, value_tree.root(), 0)?;
        self.emit_epilogue()
    }

    fn emit_conditional_return_else<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        conditional: &crate::ConditionalReturnElse<'_>,
        expected_type: RuntimeExpressionType,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        for branch in conditional.branches().iter().flatten() {
            let condition_tree =
                branch
                    .parse_condition::<MAX_EXPRESSION_NODES>()
                    .map_err(|error| CodegenError {
                        kind: CodegenErrorKind::Expression(error.kind),
                        span: translate_span(body_start + branch.condition_span.start, error.span),
                    })?;
            let condition_type = runtime_expression_type_with_locals(
                self.function,
                self.resolver,
                &self.locals[..self.saved_locals],
                &condition_tree,
                condition_tree.root(),
                0,
            )
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(body_start, branch.condition_span),
            })?;
            if condition_type != RuntimeExpressionType::Bool {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(body_start, branch.condition_span),
                });
            }
            let value_tree = branch
                .parse_value::<MAX_EXPRESSION_NODES>()
                .map_err(|error| CodegenError {
                    kind: CodegenErrorKind::Expression(error.kind),
                    span: translate_span(body_start + branch.value_span.start, error.span),
                })?;
            let value_type = runtime_expression_type_with_locals(
                self.function,
                self.resolver,
                &self.locals[..self.saved_locals],
                &value_tree,
                value_tree.root(),
                0,
            )
            .map_err(|kind| CodegenError {
                kind,
                span: translate_span(body_start, branch.value_span),
            })?;
            if !runtime_types_compatible(value_type, expected_type) {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(body_start, branch.value_span),
                });
            }
            self.emit_expression(&condition_tree, condition_tree.root(), 0)?;
            self.emit(&[0x48, 0x85, 0xc0])?;
            let next_branch = self.emit_forward_branch(0x84)?;
            self.emit_expression(&value_tree, value_tree.root(), 0)?;
            self.emit_epilogue()?;
            self.patch_forward_branch(next_branch)?;
        }
        let Some(else_value) = conditional.else_value.as_ref() else {
            return Ok(());
        };
        let else_tree = else_value
            .parse_value::<MAX_EXPRESSION_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + else_value.value_span.start, error.span),
            })?;
        let else_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &else_tree,
            else_tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, else_value.value_span),
        })?;
        if !runtime_types_compatible(else_type, expected_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, else_value.value_span),
            });
        }
        self.emit_expression(&else_tree, else_tree.root(), 0)?;
        self.emit_epilogue()
    }

    fn emit_normalize(&mut self) -> Result<(), CodegenError> {
        self.emit_normalize_width(self.width)
    }

    fn emit_normalize_width(&mut self, width: RuntimeWidth) -> Result<(), CodegenError> {
        match (width.signed, width.bits) {
            (false, 8) => self.emit(&[0x0f, 0xb6, 0xc0]),
            (false, 16) => self.emit(&[0x0f, 0xb7, 0xc0]),
            (false, 32) => self.emit(&[0x89, 0xc0]),
            (true, 8) => self.emit(&[0x48, 0x0f, 0xbe, 0xc0]),
            (true, 16) => self.emit(&[0x48, 0x0f, 0xbf, 0xc0]),
            (true, 32) => self.emit(&[0x48, 0x63, 0xc0]),
            (_, 64) => Ok(()),
            _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
    }

    fn emit_range_check(&mut self) -> Result<(), CodegenError> {
        if self.width.bits == 64 {
            return Ok(());
        }
        self.emit(&[0x48, 0xba])?;
        self.emit(&self.width.maximum.to_le_bytes())?;
        self.emit(&[0x48, 0x39, 0xd0])?;
        self.emit_trap_branch(if self.width.signed { 0x8f } else { 0x87 })?;
        if self.width.signed {
            self.emit(&[0x48, 0xba])?;
            self.emit(&(self.width.minimum as u64).to_le_bytes())?;
            self.emit(&[0x48, 0x39, 0xd0])?;
            self.emit_trap_branch(0x8c)?;
        }
        Ok(())
    }

    fn emit_binary(&mut self, operator: crate::BinaryOperator) -> Result<(), CodegenError> {
        use crate::BinaryOperator::*;
        match operator {
            Add => {
                self.emit(&[0x48, 0x01, 0xc8])?;
                if self.overflow_checks {
                    if self.width.bits == 64 {
                        self.emit_trap_branch(if self.width.signed { 0x80 } else { 0x82 })
                    } else {
                        self.emit_range_check()
                    }
                } else {
                    self.emit_normalize()
                }
            }
            Subtract => {
                self.emit(&[0x48, 0x29, 0xc1])?;
                if self.overflow_checks && self.width.bits == 64 {
                    self.emit_trap_branch(if self.width.signed { 0x80 } else { 0x82 })?;
                }
                self.emit(&[0x48, 0x89, 0xc8])?;
                if self.overflow_checks && self.width.bits < 64 {
                    self.emit_range_check()
                } else {
                    self.emit_normalize()
                }
            }
            Multiply => {
                self.emit(&[0x48, 0x87, 0xc1])?;
                if self.width.signed {
                    self.emit(&[0x48, 0x0f, 0xaf, 0xc1])?;
                } else {
                    self.emit(&[0x48, 0xf7, 0xe1, 0x48, 0x85, 0xd2])?;
                }
                if self.overflow_checks && self.width.bits == 64 {
                    self.emit_trap_branch(if self.width.signed { 0x80 } else { 0x85 })
                } else if self.overflow_checks {
                    self.emit_range_check()
                } else {
                    self.emit_normalize()
                }
            }
            Divide | Remainder => {
                self.emit(&[0x48, 0x87, 0xc1, 0x48, 0x85, 0xc9])?;
                self.emit_trap_branch(0x84)?;
                if self.width.signed {
                    // Rust traps for MIN / -1 and MIN % -1 in every overflow
                    // mode. Skip the second comparison unless the divisor is -1.
                    self.emit(&[0x48, 0x83, 0xf9, 0xff, 0x75, 19])?;
                    self.emit(&[0x48, 0xba])?;
                    self.emit(&(self.width.minimum as u64).to_le_bytes())?;
                    self.emit(&[0x48, 0x39, 0xd0])?;
                    self.emit_trap_branch(0x84)?;
                    self.emit(&[0x48, 0x99, 0x48, 0xf7, 0xf9])?;
                } else {
                    self.emit(&[0x48, 0x31, 0xd2, 0x48, 0xf7, 0xf1])?;
                }
                if operator == Remainder {
                    self.emit(&[0x48, 0x89, 0xd0])?;
                }
                self.emit_normalize()
            }
            BitAnd => self.emit(&[0x48, 0x21, 0xc8]),
            BitOr => self.emit(&[0x48, 0x09, 0xc8]),
            BitXor => self.emit(&[0x48, 0x31, 0xc8]),
            ShiftLeft | ShiftRight => {
                self.emit(&[0x48, 0x87, 0xc1])?;
                if self.overflow_checks {
                    self.emit(&[0x48, 0x83, 0xf9, self.width.bits])?;
                    self.emit_trap_branch(0x83)?;
                } else {
                    self.emit(&[0x83, 0xe1, self.width.bits - 1])?;
                }
                self.emit(if operator == ShiftLeft {
                    &[0x48, 0xd3, 0xe0]
                } else if self.width.signed {
                    &[0x48, 0xd3, 0xf8]
                } else {
                    &[0x48, 0xd3, 0xe8]
                })?;
                self.emit_normalize()
            }
            Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual => {
                let condition = match (self.width.signed, operator) {
                    (_, Equal) => 0x94,
                    (_, NotEqual) => 0x95,
                    (false, Less) => 0x92,
                    (false, LessEqual) => 0x96,
                    (false, Greater) => 0x97,
                    (false, GreaterEqual) => 0x93,
                    (true, Less) => 0x9c,
                    (true, LessEqual) => 0x9e,
                    (true, Greater) => 0x9f,
                    (true, GreaterEqual) => 0x9d,
                    _ => unreachable!(),
                };
                self.emit(&[0x48, 0x39, 0xc1, 0x0f, condition, 0xc0, 0x0f, 0xb6, 0xc0])
            }
            LogicalAnd | LogicalOr => Err(self.error(CodegenErrorKind::UnsupportedRuntimeOperator)),
        }
    }
}

fn parameter_push_encoding(abi: X86_64Abi, index: usize) -> Option<&'static [u8]> {
    const RCX: &[u8] = &[0x51];
    const RDX: &[u8] = &[0x52];
    const RDI: &[u8] = &[0x57];
    const RSI: &[u8] = &[0x56];
    const R8: &[u8] = &[0x41, 0x50];
    const R9: &[u8] = &[0x41, 0x51];
    match (abi, index) {
        (X86_64Abi::Windows, 0) | (X86_64Abi::SystemV, 3) => Some(RCX),
        (X86_64Abi::Windows, 1) | (X86_64Abi::SystemV, 2) => Some(RDX),
        (X86_64Abi::Windows, 2) | (X86_64Abi::SystemV, 4) => Some(R8),
        (X86_64Abi::Windows, 3) | (X86_64Abi::SystemV, 5) => Some(R9),
        (X86_64Abi::SystemV, 0) => Some(RDI),
        (X86_64Abi::SystemV, 1) => Some(RSI),
        _ => None,
    }
}

fn assignment_binary_operator(
    operator: crate::AssignmentOperator,
) -> Option<crate::BinaryOperator> {
    use crate::{AssignmentOperator as Assignment, BinaryOperator as Binary};
    Some(match operator {
        Assignment::Assign => return None,
        Assignment::Add => Binary::Add,
        Assignment::Subtract => Binary::Subtract,
        Assignment::Multiply => Binary::Multiply,
        Assignment::Divide => Binary::Divide,
        Assignment::Remainder => Binary::Remainder,
        Assignment::BitAnd => Binary::BitAnd,
        Assignment::BitOr => Binary::BitOr,
        Assignment::BitXor => Binary::BitXor,
        Assignment::ShiftLeft => Binary::ShiftLeft,
        Assignment::ShiftRight => Binary::ShiftRight,
    })
}

fn translate_span(base: usize, relative: Span) -> Span {
    Span {
        start: base + relative.start,
        end: base + relative.end,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeWidth {
    bits: u8,
    signed: bool,
    minimum: i64,
    maximum: u64,
}

fn runtime_width(ty: &str) -> Option<RuntimeWidth> {
    Some(match ty {
        "u8" => RuntimeWidth {
            bits: 8,
            signed: false,
            minimum: 0,
            maximum: u64::from(u8::MAX),
        },
        "u16" => RuntimeWidth {
            bits: 16,
            signed: false,
            minimum: 0,
            maximum: u64::from(u16::MAX),
        },
        "u32" | "char" => RuntimeWidth {
            bits: 32,
            signed: false,
            minimum: 0,
            maximum: u64::from(u32::MAX),
        },
        "u64" | "usize" => RuntimeWidth {
            bits: 64,
            signed: false,
            minimum: 0,
            maximum: u64::MAX,
        },
        "i8" => RuntimeWidth {
            bits: 8,
            signed: true,
            minimum: i64::from(i8::MIN),
            maximum: i8::MAX as u64,
        },
        "i16" => RuntimeWidth {
            bits: 16,
            signed: true,
            minimum: i64::from(i16::MIN),
            maximum: i16::MAX as u64,
        },
        "i32" => RuntimeWidth {
            bits: 32,
            signed: true,
            minimum: i64::from(i32::MIN),
            maximum: i32::MAX as u64,
        },
        "i64" | "isize" => RuntimeWidth {
            bits: 64,
            signed: true,
            minimum: i64::MIN,
            maximum: i64::MAX as u64,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Item, NoConstants, Parser, TargetLayout, analyze_constants};

    #[test]
    fn emits_a_windows_and_system_v_compatible_integer_return() {
        let source = "const PAGE: u64 = 4096; #[unsafe(no_mangle)] pub extern \"C\" fn arena_bytes() -> u64 { PAGE * 8 + 16 }";
        let module = Parser::new(source).parse_module::<4, 2>().unwrap();
        let constants = analyze_constants::<4, 16, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        let Some(Item::Function(function)) = module.items()[1] else {
            panic!("expected function")
        };
        let code = compile_x86_64_constant_function::<_, 11, 2, 16>(&function, &constants).unwrap();
        assert_eq!(
            code.bytes(),
            &[
                0x48, 0xb8, 0x10, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc3
            ]
        );
    }

    #[test]
    fn emits_typed_negative_module_constants() {
        let source = "const OFFSET: i32 = -2; #[unsafe(no_mangle)] pub extern \"C\" fn value(input: i32) -> i32 { input + OFFSET }";
        let module = Parser::new(source).parse_module::<4, 2>().unwrap();
        let constants = analyze_constants::<4, 16, 4, 2>(&module, TargetLayout::X86_64).unwrap();
        let Some(Item::Function(function)) = module.items()[1] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            let code =
                compile_x86_64_function::<_, 128, 2, 16>(&function, &constants, abi).unwrap();
            assert!(code.bytes().windows(10).any(|bytes| {
                bytes == [0x48, 0xb8, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
            }));
        }
    }

    #[test]
    fn reports_source_relative_expression_errors() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn broken() -> u64 { 1 + }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_constant_function::<_, 16, 2, 8>(&function, &NoConstants).unwrap_err();
        assert_eq!(
            error.kind,
            CodegenErrorKind::Expression(ExpressionErrorKind::ExpectedExpression)
        );
        assert_eq!(error.span.start, function.body_expression_span.end);
    }

    #[test]
    fn rejects_unsupported_signatures_and_overflow() {
        let parameters = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn identity(value: u64) -> u64 { value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = parameters.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_constant_function::<_, 16, 2, 8>(&function, &NoConstants).unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::ParametersUnsupported);

        let overflow = Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn byte() -> u8 { 256 }")
            .parse_module::<2, 2>()
            .unwrap();
        let Some(Item::Function(function)) = overflow.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_constant_function::<_, 16, 2, 8>(&function, &NoConstants).unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::ValueOutOfRange);
    }

    #[test]
    fn enforces_output_capacity_without_partial_artifacts() {
        let module = Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { 42 }")
            .parse_module::<2, 2>()
            .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_constant_function::<_, 10, 2, 8>(&function, &NoConstants).unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::OutputTooSmall);
    }

    #[test]
    fn refuses_to_export_an_unstable_rust_abi_symbol() {
        let module = Parser::new("pub fn answer() -> u64 { 42 }")
            .parse_module::<2, 2>()
            .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_constant_function::<_, 16, 2, 8>(&function, &NoConstants).unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::StableExportRequired);
    }

    #[test]
    fn emits_abi_specific_parameter_returns() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn identity(value: u64) -> u64 { value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let windows =
            compile_x86_64_function::<_, 16, 2, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert_eq!(
            windows.bytes(),
            &[0x51, 0x48, 0x8b, 0x44, 0x24, 0, 0x48, 0x83, 0xc4, 8, 0xc3]
        );
        let system_v =
            compile_x86_64_function::<_, 16, 2, 8>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert_eq!(system_v.bytes()[0], 0x57);
        assert_eq!(&system_v.bytes()[1..], &windows.bytes()[1..]);
    }

    #[test]
    fn saves_and_reads_stack_passed_parameters_for_both_abis() {
        let windows_module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn fifth(a: u64, b: u64, c: u64, d: u64, value: u64) -> u64 { value }",
        )
        .parse_module::<2, 8>()
        .unwrap();
        let Some(Item::Function(windows_function)) = windows_module.items()[0] else {
            panic!("expected function")
        };
        let windows = compile_x86_64_function::<_, 96, 8, 8>(
            &windows_function,
            &NoConstants,
            X86_64Abi::Windows,
        )
        .unwrap();
        assert!(
            windows
                .bytes()
                .windows(6)
                .any(|bytes| bytes == [0x48, 0x8b, 0x44, 0x24, 72, 0x50])
        );

        let system_v_module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn seventh(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, value: u64) -> u64 { value }",
        )
        .parse_module::<2, 8>()
        .unwrap();
        let Some(Item::Function(system_v_function)) = system_v_module.items()[0] else {
            panic!("expected function")
        };
        let system_v = compile_x86_64_function::<_, 128, 8, 8>(
            &system_v_function,
            &NoConstants,
            X86_64Abi::SystemV,
        )
        .unwrap();
        assert!(
            system_v
                .bytes()
                .windows(6)
                .any(|bytes| bytes == [0x48, 0x8b, 0x44, 0x24, 56, 0x50])
        );
    }

    #[test]
    fn bounds_abi_parameters_and_uses_wide_stack_cleanup() {
        let sixteen = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn last(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64, i: u64, j: u64, k: u64, l: u64, m: u64, n: u64, o: u64, value: u64) -> u64 { value }",
        )
        .parse_module::<2, 17>()
        .unwrap();
        let Some(Item::Function(function)) = sixteen.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 256, 17, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(
            code.bytes()
                .windows(7)
                .any(|bytes| bytes == [0x48, 0x81, 0xc4, 128, 0, 0, 0])
        );

        let seventeen = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn excess(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64, i: u64, j: u64, k: u64, l: u64, m: u64, n: u64, o: u64, p: u64, value: u64) -> u64 { value }",
        )
        .parse_module::<2, 17>()
        .unwrap();
        let Some(Item::Function(function)) = seventeen.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_function::<_, 256, 17, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::TooManyAbiParameters);
    }

    #[test]
    fn emits_checked_runtime_arithmetic_and_a_shared_trap() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn increment(value: u64) -> u64 { value + 1 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 64, 2, 8>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x48, 0x01, 0xc8])
        );
        assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x82]));
        assert_eq!(&code.bytes()[code.len() - 2..], &[0x0f, 0x0b]);
    }

    #[test]
    fn emits_width_normalization_and_narrow_overflow_checks() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn increment(value: u8) -> u8 { value + 1 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 96, 2, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x0f, 0xb6, 0xc0])
        );
        assert!(
            code.bytes()
                .windows(8)
                .any(|bytes| bytes == u64::from(u8::MAX).to_le_bytes())
        );
        assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x87]));

        let wrapping = compile_x86_64_function_with_options::<_, 96, 2, 8>(
            &function,
            &NoConstants,
            X86_64Abi::Windows,
            CodegenOptions::WRAPPING,
        )
        .unwrap();
        assert!(
            !wrapping
                .bytes()
                .windows(2)
                .any(|bytes| bytes == [0x0f, 0x82])
        );
        assert!(
            !wrapping
                .bytes()
                .windows(2)
                .any(|bytes| bytes == [0x0f, 0x87])
        );
        assert_eq!(wrapping.bytes()[wrapping.len() - 1], 0xc3);
    }

    #[test]
    fn types_unsigned_comparisons_as_bool() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn less(left: u8, right: u8) -> bool { left < right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 96, 2, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x0f, 0x92, 0xc0])
        );

        let invalid = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn less(left: u8, right: u8) -> u8 { left < right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = invalid.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_function::<_, 96, 2, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::RuntimeTypeMismatch);
    }

    #[test]
    fn emits_signed_division_negation_and_comparisons() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn quotient(left: i8, right: i8) -> i8 { -left / right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 256, 2, 16>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(
            code.bytes()
                .windows(4)
                .any(|bytes| bytes == [0x48, 0x0f, 0xbe, 0xc0])
        );
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x48, 0xf7, 0xf9])
        );
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x48, 0xf7, 0xd8])
        );

        let comparison = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn less(left: i32, right: i32) -> bool { left < right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = comparison.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 128, 2, 8>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x0f, 0x9c, 0xc0])
        );

        let minimum =
            Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn minimum() -> i8 { -128i8 }")
                .parse_module::<2, 2>()
                .unwrap();
        let Some(Item::Function(function)) = minimum.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 64, 2, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert_eq!(&code.bytes()[2..10], &(-128i64 as u64).to_le_bytes());
    }

    #[test]
    fn emits_typed_boolean_not_and_short_circuit_branches() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn logic(left: bool, right: bool) -> bool { left && !right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x84]));
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x0f, 0x94, 0xc0])
        );

        let invalid = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn logic(left: u8, right: u8) -> bool { left && right }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = invalid.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::RuntimeTypeMismatch);

        let literals = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn constant_guard() -> bool { true || (1 / 0 > 0) }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = literals.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 160, 2, 16>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x85]));
    }

    #[test]
    fn emits_integer_cast_normalization_for_shift_distances() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn shifted(value: u8, distance: u8) -> u8 { value >> distance as usize }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(
            code.bytes()
                .windows(3)
                .any(|bytes| bytes == [0x48, 0xd3, 0xe8])
        );
    }

    #[test]
    fn rejects_cast_results_that_do_not_match_the_return_type() {
        let module =
            Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn mismatch() -> u8 { 1 as u64 }")
                .parse_module::<2, 1>()
                .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_function::<_, 64, 1, 8>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::RuntimeTypeMismatch);
    }

    #[test]
    fn folds_typed_locals_in_declaration_order() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn local_math() -> isize { let x: isize = 15; let y: isize = 4; x / y + x % y }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 64, 2, 16>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert_eq!(&code.bytes()[2..10], &6u64.to_le_bytes());

        let duplicate = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn duplicate() -> usize { let x: usize = 1; let x: usize = 2; x }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = duplicate.items()[0] else {
            panic!("expected function")
        };
        let error =
            compile_x86_64_function::<_, 64, 2, 16>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err();
        assert_eq!(error.kind, CodegenErrorKind::DuplicateLocal);
    }

    #[test]
    fn emits_parameterized_integer_and_boolean_locals() {
        let arithmetic = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn local_runtime(value: u64) -> u64 { let doubled: u64 = value * 2; let adjusted: u64 = doubled + 1; adjusted + value }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = arithmetic.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 256, 4, 32>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(code.bytes().iter().filter(|byte| **byte == 0x50).count() >= 4);
        assert!(
            code.bytes()
                .windows(4)
                .any(|bytes| bytes == [0x48, 0x83, 0xc4, 24])
        );

        let boolean = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn local_guard(value: u64, enabled: bool) -> bool { let positive: bool = value > 0; enabled && positive }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = boolean.items()[0] else {
            panic!("expected function")
        };
        assert!(
            compile_x86_64_function::<_, 256, 4, 32>(&function, &NoConstants, X86_64Abi::SystemV,)
                .is_ok()
        );
    }

    #[test]
    fn evaluates_and_emits_mutable_local_assignments() {
        let constant = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn swap() -> isize { let mut a: isize = 1; let mut b: isize = 2; a ^= b; b ^= a; a = a ^ b; a | b }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = constant.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 256, 4, 32>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(code.bytes().len() > X86_64_RETURN_CONSTANT_BYTES);

        let runtime = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn mutate(value: u64) -> u64 { let mut x: u64 = value; x ^= 255; x += 1; x }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = runtime.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 256, 4, 32>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(
            code.bytes()
                .windows(5)
                .any(|bytes| bytes == [0x48, 0x89, 0x44, 0x24, 0])
        );

        let immutable = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn invalid(value: u64) -> u64 { let x: u64 = value; x = 1; x }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = immutable.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment
        );
    }

    #[test]
    fn infers_local_integer_types_from_context_and_suffixes() {
        let inferred = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn inferred() -> i32 { let mut x = 0; x += 42; x }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = inferred.items()[0] else {
            panic!("expected function")
        };
        assert!(
            compile_x86_64_function::<_, 256, 2, 16>(&function, &NoConstants, X86_64Abi::Windows,)
                .is_ok()
        );

        let mismatch = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn mismatch(value: u64) -> u64 { let x = 1u32; x + value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 256, 2, 16>(&function, &NoConstants, X86_64Abi::SystemV,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_typed_lazy_if_expressions() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64, guard: bool) -> u64 { if guard { value } else { 10 / value } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let code =
            compile_x86_64_function::<_, 512, 2, 32>(&function, &NoConstants, X86_64Abi::SystemV)
                .unwrap();
        assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x84]));
        assert!(code.bytes().contains(&0xe9));

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { if 1 { 2 } else { 3 } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(flag: bool) -> u64 { if flag { 2 } else { false } }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 512, 2, 32>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                CodegenErrorKind::RuntimeTypeMismatch
            );
        }
    }

    #[test]
    fn emits_explicit_tail_return_values() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn identity(value: usize) -> usize { return value; }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert!(
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::Windows,)
                .is_ok()
        );
        assert!(
            compile_x86_64_function::<_, 128, 2, 16>(&function, &NoConstants, X86_64Abi::SystemV,)
                .is_ok()
        );
    }

    #[test]
    fn emits_conditional_returns_with_local_frame_cleanup() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn early(value: u64, stop: bool) -> u64 { let adjusted: u64 = value + 1; if stop { return adjusted; } adjusted + 1 }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            let code =
                compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi).unwrap();
            assert!(code.bytes().iter().filter(|byte| **byte == 0xc3).count() >= 2);
        }

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64) -> u64 { if value { return 1; } value }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64, stop: bool) -> u64 { if stop { return false; } value }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 512, 2, 32>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                CodegenErrorKind::RuntimeTypeMismatch
            );
        }
    }

    #[test]
    fn emits_exhaustive_conditional_returns_with_lazy_branches() {
        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64, select: bool) -> u64 { let adjusted = value + 1; if select { return adjusted; } else { return value / value; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn choose_chain(value: u64) -> u64 { if value == 0 { return 42; } else if value == 1 { return value + 41; } else if value == 2 { return 84 / value; } else { return 126 / value; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn choose_unit(select: bool) { if select { return; } else { return (); } }",
        ] {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let mismatch = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(select: bool) -> u64 { if select { return 1; } else { return false; } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_non_exhaustive_return_chains_with_fallthrough() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64) -> u64 { if value == 0 { return 42; } else if value == 1 { return 42 / value; } 84 / value }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(compile_x86_64_function::<_, 512, 2, 32>(&function, &NoConstants, abi).is_ok());
        }
    }

    #[test]
    fn emits_lazy_conditional_assignments() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64) -> u64 { let mut result = value; if value == 0 { result = 40; value + 1; result += 2; } else if value == 1 { result = 40; 84 / value; result += 2; } else if value == 2 { result = 40; 42 / value; result += 2; } else { result = 40; value + 10; result += 2; } if value == 4 { result += 1; value * 3; result *= 2; } else if value == 5 { result += 2; value * 3; result *= 2; } result }";
        let module = Parser::new(source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(
                compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, abi).is_ok()
            );
        }

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64, select: bool) -> u64 { let result = value; if select { result = 1; } result }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64) -> u64 { let mut result = value; if value { result = 1; } result }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64, select: bool) -> u64 { let mut result = value; if select { result = false; } result }",
        ] {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert!(
                compile_x86_64_function::<_, 1024, 4, 48>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn emits_bounded_scalar_while_loop_backedges() {
        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn count(limit: u64) -> u64 { let mut i: u64 = 0; let mut sum: u64 = 0; while i < limit { sum += i; i += 1; if i == 10 { break; } } sum }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            let code =
                compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi).unwrap();
            assert!(code.bytes().contains(&0xe9));
            assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x84]));
        }

        let immutable = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(limit: u64) -> u64 { let i: u64 = 0; while i < limit { i += 1; } i }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = immutable.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment
        );

        let invalid_break = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(limit: u64) -> u64 { let mut i: u64 = 0; while i < limit { i += 1; if i { break; } } i }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = invalid_break.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_bounded_unconditional_loops_and_explicit_continue() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn once() -> u64 { let mut value: u64 = 0; loop { value += 1; if value == 1 { break; } } value }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn count(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { value += 1; continue; } value }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn immediate() -> u64 { loop { break; } 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn classify(limit: usize) -> usize { let mut i: usize = 0; let mut is_even: bool = false; loop { if i == limit { break; } is_even = false; i += 1; if i % 2 != 0 { continue; } is_even = true; } if is_even { i } else { 0 } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let endless = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn endless() -> u64 { let mut value: u64 = 0; loop { value += 1; continue; } value }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = endless.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<4>().unwrap_err().kind,
            crate::ParseErrorKind::ExpectedBody
        );
    }

    #[test]
    fn emits_typed_returns_inside_loops() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn find(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { if value == 42 { return value; } value += 1; } limit }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn divisible(value: u32) -> bool { let mut divisor: u32 = 2; loop { if value % divisor == 0 { return true; } divisor += 1; } false }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn immediate(value: u64) -> u64 { loop { return value + 1; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn immediate(value: bool) -> bool { loop { return !value; } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let code =
                    compile_x86_64_function::<_, 768, 4, 48>(&function, &NoConstants, abi).unwrap();
                assert!(code.bytes().contains(&0xc3));
            }
        }

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64) -> u64 { loop { if value { return value; } } 0 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64) -> u64 { loop { if value == 0 { return false; } } 0 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { loop { return false; } }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 512, 2, 32>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                CodegenErrorKind::RuntimeTypeMismatch
            );
        }
    }

    #[test]
    fn emits_typed_immediate_break_loop_values() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let result: u64 = loop { break input + 1; }; result }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: bool) -> bool { let result: bool = loop { break input; }; result }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let result: u64 = loop { break 13; }; result }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(choose_first: bool, input: u64) -> u64 { let result: u64 = loop { if choose_first { break input + 1; } break 84 / input; }; result }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let mismatch = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { let result: u64 = loop { break false; }; result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );

        let branch_mismatch = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(flag: bool) -> u64 { let result = loop { if flag { break 1; } break false; }; result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = branch_mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_unit_returning_functions() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn unit() -> () { () }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn implicit_unit() {}",
            "#[unsafe(no_mangle)] pub extern \"C\" fn early_unit(value: u64) { let adjusted = value + 1; return; let unreachable = adjusted + 1; unreachable; }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn conditional_unit(stop: bool) { if stop { return; } () }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn loop_return_unit(stop: bool) { loop { if stop { return; } } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn loop_unit() -> () { loop { break; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn loop_value_unit() -> () { let value: () = loop { break (); }; value }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let mismatch = Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> () { 42 }")
            .parse_module::<2, 4>()
            .unwrap();
        let Some(Item::Function(function)) = mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );

        let mismatch =
            Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { return; }")
                .parse_module::<2, 4>()
                .unwrap();
        let Some(Item::Function(function)) = mismatch.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<4>().unwrap();
        assert_eq!(body.returns()[0].unwrap().value, "()");
        assert!(body.tail_diverges);
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_boolean_and_unit_comparisons_and_boolean_bitwise_ops() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn equal(left: bool, right: bool) -> bool { left == right }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn different(left: bool, right: bool) -> bool { left != right }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn unit_equal() -> bool { () == () }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn ordered(left: bool, right: bool) -> bool { left < right }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn unit_ordered() -> bool { () <= () }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn bitwise(left: bool, right: bool) -> bool { (left & right) | (left ^ right) }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let mixed = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(left: bool) -> bool { left == 1 }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = mixed.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_boolean_compound_assignments() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn mutate(left: bool, right: bool) -> bool { let mut value: bool = left; value &= right; value ^= true; value |= left; value }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn parity(limit: u64) -> bool { let mut i: u64 = 0; let mut value: bool = false; while i < limit { value ^= true; i += 1; } value }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 768, 4, 64>(&function, &NoConstants, abi,).is_ok()
                );
            }
        }

        let arithmetic = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> bool { let mut value: bool = false; value += true; value }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = arithmetic.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }
}
