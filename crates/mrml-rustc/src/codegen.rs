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
    TooManyRuntimeLocals,
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
                RuntimeExpressionType::Unit
                | RuntimeExpressionType::Default
                | RuntimeExpressionType::Array { .. }
                | RuntimeExpressionType::Reference { .. }
                | RuntimeExpressionType::RawPointer { .. } => {
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
        if assignment.index().is_some() {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeExpressionUnsupported,
                span: translate_span(function.body_expression_span.start, assignment.name_span),
            });
        }
        let index = local_values[..local_count]
            .iter()
            .position(|value| value.is_some_and(|value| value.name == assignment.binding_name()))
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
    let mut body_has_array_local = false;
    for local in parsed_body.locals().iter().flatten() {
        if local
            .ty
            .and_then(|ty| runtime_array_type(ty.text))
            .is_some()
        {
            body_has_array_local = true;
            break;
        }
        if local.initializer.as_bytes().contains(&b'[') {
            body_has_array_local = true;
            break;
        }
        if let Ok(tree) = local.parse_initializer::<MAX_EXPRESSION_NODES>()
            && runtime_expression_type(function, resolver, &tree, tree.root(), 0)
                .is_ok_and(|ty| matches!(ty, RuntimeExpressionType::Array { .. }))
        {
            body_has_array_local = true;
            break;
        }
    }
    let body_requires_runtime = body_has_array_local
        || parsed_body.assignment_count() != 0
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
    let return_array = runtime_array_type(return_type_text);
    let return_reference = runtime_reference_type(return_type_text);
    let return_pointer = runtime_raw_pointer_type(return_type_text);
    let returns_zero_sized_array = return_array
        .and_then(runtime_array_abi_layout)
        .is_some_and(|(_, bytes, _)| bytes == 0);
    let returns_value_free = returns_unit || returns_zero_sized_array;
    if !returns_unit
        && !returns_bool
        && !returns_char
        && runtime_width(return_type_text).is_none()
        && return_array.is_none()
        && return_reference.is_none()
        && return_pointer.is_none()
    {
        return Err(CodegenError {
            kind: CodegenErrorKind::UnsupportedReturnType,
            span: return_type_span,
        });
    }
    if let Some(RuntimeExpressionType::Array { element, .. }) = return_array
        && !matches!(
            element,
            RuntimeArrayElementType::Unit
                | RuntimeArrayElementType::Bool
                | RuntimeArrayElementType::Char
                | RuntimeArrayElementType::Integer(Some(_))
        )
    {
        return Err(CodegenError {
            kind: CodegenErrorKind::UnsupportedReturnType,
            span: return_type_span,
        });
    }
    if let Some(RuntimeExpressionType::Reference { target, .. }) = return_reference
        && !matches!(
            target,
            RuntimeReferenceTarget::Scalar(
                RuntimeArrayElementType::Bool
                    | RuntimeArrayElementType::Char
                    | RuntimeArrayElementType::Integer(Some(_))
            ) | RuntimeReferenceTarget::Array {
                element: RuntimeArrayElementType::Bool
                    | RuntimeArrayElementType::Char
                    | RuntimeArrayElementType::Integer(Some(_)),
                ..
            } | RuntimeReferenceTarget::Slice(
                RuntimeArrayElementType::Bool
                    | RuntimeArrayElementType::Char
                    | RuntimeArrayElementType::Integer(Some(_))
            ) | RuntimeReferenceTarget::Str
        )
    {
        return Err(CodegenError {
            kind: CodegenErrorKind::UnsupportedReturnType,
            span: return_type_span,
        });
    }
    let returns_boolean_like = returns_bool
        || matches!(
            return_array,
            Some(RuntimeExpressionType::Array {
                element: RuntimeArrayElementType::Bool,
                ..
            })
        );
    let operand_type = match return_array {
        Some(RuntimeExpressionType::Array {
            element: RuntimeArrayElementType::Integer(Some(element)),
            ..
        }) => element.name(),
        Some(RuntimeExpressionType::Array {
            element: RuntimeArrayElementType::Char,
            ..
        }) => "char",
        Some(_) => "u64",
        None if returns_bool
            || returns_unit
            || return_reference.is_some()
            || return_pointer.is_some() =>
        {
            "u64"
        }
        None => return_type_text,
    };
    let mut integer_operand_type = None;
    for parameter in function.parameters().iter().flatten() {
        if runtime_raw_pointer_type(parameter.ty.text).is_some() {
            continue;
        }
        if let Some(RuntimeExpressionType::Reference { target, .. }) =
            runtime_reference_type(parameter.ty.text)
        {
            let element = match target {
                RuntimeReferenceTarget::Scalar(element)
                | RuntimeReferenceTarget::Slice(element)
                | RuntimeReferenceTarget::Array { element, .. } => element,
                RuntimeReferenceTarget::Str => continue,
            };
            match element {
                RuntimeArrayElementType::Bool | RuntimeArrayElementType::Char => continue,
                RuntimeArrayElementType::Integer(Some(pointee)) => {
                    if integer_operand_type.is_none() {
                        integer_operand_type = Some(pointee.name());
                    }
                    continue;
                }
                _ => {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::UnsupportedParameterType,
                        span: parameter.ty.span,
                    });
                }
            }
        }
        if let Some(RuntimeExpressionType::Array { element, count }) =
            runtime_array_type(parameter.ty.text)
        {
            if runtime_array_element_bytes(element) == Some(0) || count == 0 {
                continue;
            }
            match element {
                RuntimeArrayElementType::Bool | RuntimeArrayElementType::Char => continue,
                RuntimeArrayElementType::Integer(Some(element)) => {
                    if integer_operand_type.is_none() {
                        integer_operand_type = Some(element.name());
                    }
                    continue;
                }
                _ => {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::UnsupportedParameterType,
                        span: parameter.ty.span,
                    });
                }
            }
        }
        if parameter.ty.text == "bool" {
            continue;
        }
        if runtime_width(parameter.ty.text).is_none() {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedParameterType,
                span: parameter.ty.span,
            });
        }
        if integer_operand_type.is_none() {
            integer_operand_type = Some(parameter.ty.text);
        }
    }
    let operand_type =
        if (returns_boolean_like || returns_value_free) && integer_operand_type.is_some() {
            integer_operand_type.unwrap_or("u64")
        } else {
            operand_type
        };
    let expected_type = if returns_unit {
        RuntimeExpressionType::Unit
    } else if returns_bool {
        RuntimeExpressionType::Bool
    } else if returns_char {
        RuntimeExpressionType::Char
    } else if let Some(array) = return_array {
        array
    } else if let Some(reference) = return_reference {
        reference
    } else if let Some(pointer) = return_pointer {
        pointer
    } else {
        RuntimeExpressionType::Integer(crate::IntegerType::from_name(return_type_text))
    };
    let width = runtime_width(operand_type).ok_or(CodegenError {
        kind: CodegenErrorKind::UnsupportedRuntimeType,
        span: function.name_span,
    })?;
    let returns_fat_reference = matches!(
        return_reference,
        Some(RuntimeExpressionType::Reference {
            target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
            ..
        })
    );
    let uses_sret = (abi == X86_64Abi::Windows && returns_fat_reference)
        || return_array.and_then(runtime_array_abi_layout).is_some_and(
            |(_, bytes, words)| match abi {
                X86_64Abi::Windows => bytes != 0 && !matches!(bytes, 1 | 2 | 4 | 8),
                X86_64Abi::SystemV => words > 2,
            },
        );
    let saved_parameter_slots = function
        .parameters()
        .iter()
        .flatten()
        .try_fold(0usize, |total, parameter| {
            let slots = runtime_array_type(parameter.ty.text)
                .or_else(|| runtime_reference_type(parameter.ty.text))
                .or_else(|| runtime_raw_pointer_type(parameter.ty.text))
                .map_or(1, runtime_type_stack_slots);
            total.checked_add(slots)
        })
        .and_then(|slots| slots.checked_add(usize::from(uses_sret)))
        .ok_or(CodegenError {
            kind: CodegenErrorKind::TooManyAbiParameters,
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
        uses_sret,
        saved_parameter_slots,
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
                    false,
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
                    expected_type,
                    operand_type,
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
        let mut exit_patches = [None;
            1 + crate::parser::MAX_LOOP_OPERATIONS
                * (crate::MAX_NESTED_LOOP_CONDITIONAL_BREAKS
                    + crate::MAX_NESTED_LOOP_UNCONDITIONAL_CONTROLS)];
        let mut exit_count = 0usize;
        if let Some(tree) = condition_tree.as_ref() {
            emitter.emit_expression(tree, tree.root(), 0)?;
            emitter.emit(&[0x48, 0x85, 0xc0])?;
            exit_patches[exit_count] = Some(emitter.emit_forward_branch(0x84)?);
            exit_count += 1;
        }
        let local_checkpoint = emitter.saved_locals;
        let mut ends_with_unconditional_control = false;
        for operation in loop_statement.operations().iter().flatten() {
            if let LoopOperation::Local(local) = operation {
                emitter.emit_local::<MAX_EXPRESSION_NODES>(
                    local,
                    operand_type,
                    function.body_expression_span.start,
                    true,
                )?;
                ends_with_unconditional_control = false;
                continue;
            }
            if let LoopOperation::Expression(statement) = operation {
                emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                    statement,
                    function.body_expression_span.start,
                )?;
                ends_with_unconditional_control = false;
                continue;
            }
            if let LoopOperation::NestedBlock(index) = operation {
                let block = loop_statement.nested_blocks()[*index].ok_or(CodegenError {
                    kind: CodegenErrorKind::Body(ParseErrorKind::ExpectedBody),
                    span: function.body_expression_span,
                })?;
                let nested_checkpoint = emitter.saved_locals;
                let nested_start = emitter.length;
                let entry_exit = if let Some(condition) = block.entry_condition {
                    let tree = condition
                        .parse_condition::<MAX_EXPRESSION_NODES>()
                        .map_err(|error| CodegenError {
                            kind: CodegenErrorKind::Expression(error.kind),
                            span: translate_span(
                                function.body_expression_span.start
                                    + condition.condition_span.start,
                                error.span,
                            ),
                        })?;
                    let condition_type = runtime_expression_type_with_locals(
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
                            condition.condition_span,
                        ),
                    })?;
                    if condition_type != RuntimeExpressionType::Bool {
                        return Err(CodegenError {
                            kind: CodegenErrorKind::RuntimeTypeMismatch,
                            span: translate_span(
                                function.body_expression_span.start,
                                condition.condition_span,
                            ),
                        });
                    }
                    emitter.emit_expression(&tree, tree.root(), 0)?;
                    emitter.emit(&[0x48, 0x85, 0xc0])?;
                    Some(emitter.emit_forward_branch(0x84)?)
                } else {
                    None
                };
                let mut nested_exit_patches = [None;
                    crate::MAX_NESTED_LOOP_CONDITIONAL_BREAKS
                        + crate::MAX_NESTED_LOOP_UNCONDITIONAL_CONTROLS];
                let mut nested_exit_count = 0usize;
                for action_index in 0..=block.action_count() {
                    let control_count = block
                        .conditional_returns()
                        .len()
                        .checked_add(block.conditional_continues().len())
                        .and_then(|count| count.checked_add(block.conditional_breaks().len()))
                        .and_then(|count| count.checked_add(block.unconditional_controls().len()))
                        .ok_or(CodegenError {
                            kind: CodegenErrorKind::OutputTooSmall,
                            span: function.body_expression_span,
                        })?;
                    for control_order in 0..control_count {
                        for ((control, control_action_index), stored_control_order) in block
                            .unconditional_controls()
                            .iter()
                            .flatten()
                            .zip(block.unconditional_control_action_indices())
                            .zip(block.unconditional_control_orders())
                        {
                            if action_index != *control_action_index
                                || control_order != *stored_control_order
                            {
                                continue;
                            }
                            match control {
                                crate::NestedLoopUnconditionalControl::Break(target) => {
                                    if *target == crate::NestedLoopControlTarget::Outer {
                                        emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                        exit_patches[exit_count] =
                                            Some(emitter.emit_unconditional_forward_branch()?);
                                        exit_count += 1;
                                    } else {
                                        emitter.emit_stack_cleanup_to(nested_checkpoint)?;
                                        nested_exit_patches[nested_exit_count] =
                                            Some(emitter.emit_unconditional_forward_branch()?);
                                        nested_exit_count += 1;
                                    }
                                }
                                crate::NestedLoopUnconditionalControl::Continue(target) => {
                                    if *target == crate::NestedLoopControlTarget::Outer {
                                        emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                        emitter.emit_backward_branch(loop_start)?;
                                    } else {
                                        emitter.emit_stack_cleanup_to(nested_checkpoint)?;
                                        emitter.emit_backward_branch(nested_start)?;
                                    }
                                }
                                crate::NestedLoopUnconditionalControl::Return(value) => {
                                    emitter.emit_return::<MAX_EXPRESSION_NODES>(
                                        value,
                                        expected_type,
                                        function.body_expression_span.start,
                                    )?;
                                }
                            }
                        }
                        for (((condition, break_action_index), break_control_order), target) in
                            block
                                .conditional_breaks()
                                .iter()
                                .flatten()
                                .zip(block.conditional_break_action_indices())
                                .zip(block.conditional_break_control_orders())
                                .zip(block.conditional_break_targets())
                        {
                            if action_index != *break_action_index
                                || control_order != *break_control_order
                            {
                                continue;
                            }
                            let tree = condition
                                .parse_condition::<MAX_EXPRESSION_NODES>()
                                .map_err(|error| CodegenError {
                                    kind: CodegenErrorKind::Expression(error.kind),
                                    span: translate_span(
                                        function.body_expression_span.start
                                            + condition.condition_span.start,
                                        error.span,
                                    ),
                                })?;
                            let condition_type = runtime_expression_type_with_locals(
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
                                    condition.condition_span,
                                ),
                            })?;
                            if condition_type != RuntimeExpressionType::Bool {
                                return Err(CodegenError {
                                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                                    span: translate_span(
                                        function.body_expression_span.start,
                                        condition.condition_span,
                                    ),
                                });
                            }
                            emitter.emit_expression(&tree, tree.root(), 0)?;
                            emitter.emit(&[0x48, 0x85, 0xc0])?;
                            let skip_break = emitter.emit_forward_branch(0x84)?;
                            if *target == crate::NestedLoopControlTarget::Outer {
                                emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                exit_patches[exit_count] =
                                    Some(emitter.emit_unconditional_forward_branch()?);
                                exit_count += 1;
                            } else {
                                emitter.emit_stack_cleanup_to(nested_checkpoint)?;
                                nested_exit_patches[nested_exit_count] =
                                    Some(emitter.emit_unconditional_forward_branch()?);
                                nested_exit_count += 1;
                            }
                            emitter.patch_forward_branch(skip_break)?;
                        }
                        for ((conditional, return_action_index), return_control_order) in block
                            .conditional_returns()
                            .iter()
                            .flatten()
                            .zip(block.conditional_return_action_indices())
                            .zip(block.conditional_return_control_orders())
                        {
                            if action_index == *return_action_index
                                && control_order == *return_control_order
                            {
                                emitter.emit_conditional_return::<MAX_EXPRESSION_NODES>(
                                    conditional,
                                    expected_type,
                                    function.body_expression_span.start,
                                )?;
                            }
                        }
                        for (
                            ((condition, continue_action_index), continue_control_order),
                            target,
                        ) in block
                            .conditional_continues()
                            .iter()
                            .flatten()
                            .zip(block.conditional_continue_action_indices())
                            .zip(block.conditional_continue_control_orders())
                            .zip(block.conditional_continue_targets())
                        {
                            if action_index != *continue_action_index
                                || control_order != *continue_control_order
                            {
                                continue;
                            }
                            let tree = condition
                                .parse_condition::<MAX_EXPRESSION_NODES>()
                                .map_err(|error| CodegenError {
                                    kind: CodegenErrorKind::Expression(error.kind),
                                    span: translate_span(
                                        function.body_expression_span.start
                                            + condition.condition_span.start,
                                        error.span,
                                    ),
                                })?;
                            let condition_type = runtime_expression_type_with_locals(
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
                                    condition.condition_span,
                                ),
                            })?;
                            if condition_type != RuntimeExpressionType::Bool {
                                return Err(CodegenError {
                                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                                    span: translate_span(
                                        function.body_expression_span.start,
                                        condition.condition_span,
                                    ),
                                });
                            }
                            emitter.emit_expression(&tree, tree.root(), 0)?;
                            emitter.emit(&[0x48, 0x85, 0xc0])?;
                            let skip_continue = emitter.emit_forward_branch(0x84)?;
                            if *target == crate::NestedLoopControlTarget::Outer {
                                emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                emitter.emit_backward_branch(loop_start)?;
                            } else {
                                emitter.emit_stack_cleanup_to(nested_checkpoint)?;
                                emitter.emit_backward_branch(nested_start)?;
                            }
                            emitter.patch_forward_branch(skip_continue)?;
                        }
                    }
                    if action_index == block.action_count() {
                        break;
                    }
                    let action = block.actions()[action_index].ok_or(CodegenError {
                        kind: CodegenErrorKind::Body(ParseErrorKind::ExpectedBody),
                        span: function.body_expression_span,
                    })?;
                    match action {
                        crate::ConditionalLoopAction::Local(local) => {
                            emitter.emit_local::<MAX_EXPRESSION_NODES>(
                                &local,
                                operand_type,
                                function.body_expression_span.start,
                                true,
                            )?;
                        }
                        crate::ConditionalLoopAction::Assignment(assignment) => {
                            emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                                &assignment,
                                function.body_expression_span.start,
                            )?;
                        }
                        crate::ConditionalLoopAction::Expression(statement) => {
                            emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                                &statement,
                                function.body_expression_span.start,
                            )?;
                        }
                    }
                }
                emitter.emit_stack_cleanup_to(nested_checkpoint)?;
                emitter.truncate_scoped_locals(nested_checkpoint)?;
                if block.unconditional_controls().is_empty() {
                    emitter.emit_backward_branch(nested_start)?;
                }
                for patch in nested_exit_patches[..nested_exit_count].iter().flatten() {
                    emitter.patch_forward_branch(*patch)?;
                }
                if let Some(entry_exit) = entry_exit {
                    emitter.patch_forward_branch(entry_exit)?;
                }
                ends_with_unconditional_control = false;
                continue;
            }
            if matches!(operation, LoopOperation::NestedUnitLoop) {
                ends_with_unconditional_control = false;
                continue;
            }
            if let LoopOperation::ConditionalBlock(index) = operation {
                let block = loop_statement.conditional_blocks()[*index].ok_or(CodegenError {
                    kind: CodegenErrorKind::Body(ParseErrorKind::ExpectedBody),
                    span: function.body_expression_span,
                })?;
                let condition_tree =
                    block
                        .parse_condition::<MAX_EXPRESSION_NODES>()
                        .map_err(|error| CodegenError {
                            kind: CodegenErrorKind::Expression(error.kind),
                            span: translate_span(
                                function.body_expression_span.start + block.condition_span.start,
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
                    span: translate_span(function.body_expression_span.start, block.condition_span),
                })?;
                if condition_type != RuntimeExpressionType::Bool {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(
                            function.body_expression_span.start,
                            block.condition_span,
                        ),
                    });
                }
                emitter.emit_expression(&condition_tree, condition_tree.root(), 0)?;
                emitter.emit(&[0x48, 0x85, 0xc0])?;
                let skip_block = emitter.emit_forward_branch(0x84)?;
                let block_checkpoint = emitter.saved_locals;
                for action in block.actions().iter().flatten() {
                    match action {
                        crate::ConditionalLoopAction::Local(local) => {
                            emitter.emit_local::<MAX_EXPRESSION_NODES>(
                                local,
                                operand_type,
                                function.body_expression_span.start,
                                true,
                            )?;
                        }
                        crate::ConditionalLoopAction::Assignment(assignment) => {
                            emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                                assignment,
                                function.body_expression_span.start,
                            )?;
                        }
                        crate::ConditionalLoopAction::Expression(statement) => {
                            emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                                statement,
                                function.body_expression_span.start,
                            )?;
                        }
                    }
                }
                match block.terminal {
                    Some(crate::ConditionalLoopTerminal::Break) => {
                        emitter.emit_stack_cleanup_to(local_checkpoint)?;
                        exit_patches[exit_count] =
                            Some(emitter.emit_unconditional_forward_branch()?);
                        exit_count += 1;
                    }
                    Some(crate::ConditionalLoopTerminal::Continue) => {
                        emitter.emit_stack_cleanup_to(local_checkpoint)?;
                        emitter.emit_backward_branch(loop_start)?;
                    }
                    Some(crate::ConditionalLoopTerminal::Return(return_statement)) => {
                        emitter.emit_return::<MAX_EXPRESSION_NODES>(
                            &return_statement,
                            expected_type,
                            function.body_expression_span.start,
                        )?;
                    }
                    None => emitter.emit_stack_cleanup_to(block_checkpoint)?,
                }
                emitter.truncate_scoped_locals(block_checkpoint)?;
                let mut end_patches = [None; crate::MAX_CONDITIONAL_LOOP_ELSE_ARMS + 1];
                let mut end_patch_count = 0usize;
                if block.else_arm.is_some() {
                    end_patches[end_patch_count] =
                        Some(emitter.emit_unconditional_forward_branch()?);
                    end_patch_count += 1;
                }
                emitter.patch_forward_branch(skip_block)?;
                if let Some(else_index) = block.else_arm {
                    for else_arm in loop_statement.conditional_else_arms()
                        [else_index..else_index + block.else_arm_count]
                        .iter()
                        .flatten()
                    {
                        let skip_arm = if let Some(condition_tree) = else_arm
                            .parse_condition::<MAX_EXPRESSION_NODES>()
                            .map_err(|error| CodegenError {
                                kind: CodegenErrorKind::Expression(error.kind),
                                span: translate_span(
                                    function.body_expression_span.start
                                        + else_arm.condition_span.start,
                                    error.span,
                                ),
                            })? {
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
                                    else_arm.condition_span,
                                ),
                            })?;
                            if condition_type != RuntimeExpressionType::Bool {
                                return Err(CodegenError {
                                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                                    span: translate_span(
                                        function.body_expression_span.start,
                                        else_arm.condition_span,
                                    ),
                                });
                            }
                            emitter.emit_expression(&condition_tree, condition_tree.root(), 0)?;
                            emitter.emit(&[0x48, 0x85, 0xc0])?;
                            Some(emitter.emit_forward_branch(0x84)?)
                        } else {
                            None
                        };
                        let else_checkpoint = emitter.saved_locals;
                        for action in else_arm.actions().iter().flatten() {
                            match action {
                                crate::ConditionalLoopAction::Local(local) => {
                                    emitter.emit_local::<MAX_EXPRESSION_NODES>(
                                        local,
                                        operand_type,
                                        function.body_expression_span.start,
                                        true,
                                    )?;
                                }
                                crate::ConditionalLoopAction::Assignment(assignment) => {
                                    emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                                        assignment,
                                        function.body_expression_span.start,
                                    )?;
                                }
                                crate::ConditionalLoopAction::Expression(statement) => {
                                    emitter.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                                        statement,
                                        function.body_expression_span.start,
                                    )?;
                                }
                            }
                        }
                        match else_arm.terminal {
                            Some(crate::ConditionalLoopTerminal::Break) => {
                                emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                exit_patches[exit_count] =
                                    Some(emitter.emit_unconditional_forward_branch()?);
                                exit_count += 1;
                            }
                            Some(crate::ConditionalLoopTerminal::Continue) => {
                                emitter.emit_stack_cleanup_to(local_checkpoint)?;
                                emitter.emit_backward_branch(loop_start)?;
                            }
                            Some(crate::ConditionalLoopTerminal::Return(return_statement)) => {
                                emitter.emit_return::<MAX_EXPRESSION_NODES>(
                                    &return_statement,
                                    expected_type,
                                    function.body_expression_span.start,
                                )?;
                            }
                            None => emitter.emit_stack_cleanup_to(else_checkpoint)?,
                        }
                        emitter.truncate_scoped_locals(else_checkpoint)?;
                        end_patches[end_patch_count] =
                            Some(emitter.emit_unconditional_forward_branch()?);
                        end_patch_count += 1;
                        if let Some(skip_arm) = skip_arm {
                            emitter.patch_forward_branch(skip_arm)?;
                        } else {
                            break;
                        }
                    }
                    for patch in end_patches[..end_patch_count].iter().flatten() {
                        emitter.patch_forward_branch(*patch)?;
                    }
                }
                ends_with_unconditional_control = false;
                continue;
            }
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
                emitter.emit_typed_return(&value_tree, value_tree.root(), expected_type)?;
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
                emitter.emit_typed_return(&value_tree, value_tree.root(), expected_type)?;
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
                    LoopOperation::ConditionalBlock(_) => continue,
                    LoopOperation::Return(_) => continue,
                    LoopOperation::Local(_) => continue,
                    LoopOperation::Expression(_) => continue,
                    LoopOperation::NestedUnitLoop => continue,
                    LoopOperation::NestedBlock(_) => continue,
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
                    let skip_control = emitter.emit_forward_branch(0x84)?;
                    emitter.emit_stack_cleanup_to(local_checkpoint)?;
                    if is_break {
                        exit_patches[exit_count] =
                            Some(emitter.emit_unconditional_forward_branch()?);
                        exit_count += 1;
                    } else {
                        emitter.emit_backward_branch(loop_start)?;
                    }
                    emitter.patch_forward_branch(skip_control)?;
                    ends_with_unconditional_control = false;
                } else {
                    emitter.emit_stack_cleanup_to(local_checkpoint)?;
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
            emitter.emit_assignment::<MAX_EXPRESSION_NODES>(
                assignment,
                function.body_expression_span.start,
            )?;
        }
        if !ends_with_unconditional_control {
            emitter.emit_stack_cleanup_to(local_checkpoint)?;
            emitter.emit_backward_branch(loop_start)?;
        }
        emitter.truncate_scoped_locals(local_checkpoint)?;
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
                    false,
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
                    expected_type,
                    operand_type,
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
    emitter.emit_typed_return(&tree, tree.root(), expected_type)?;
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
    Default,
    Integer(Option<crate::IntegerType>),
    Bool,
    Char,
    Reference {
        target: RuntimeReferenceTarget,
        mutable: bool,
    },
    RawPointer {
        pointee: RuntimeArrayElementType,
        mutable: bool,
    },
    Array {
        element: RuntimeArrayElementType,
        count: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeArrayElementType {
    Unit,
    Default,
    Integer(Option<crate::IntegerType>),
    Bool,
    Char,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeReferenceTarget {
    Scalar(RuntimeArrayElementType),
    Slice(RuntimeArrayElementType),
    Str,
    Array {
        element: RuntimeArrayElementType,
        count: usize,
        layout: RuntimeReferenceArrayLayout,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeReferenceArrayLayout {
    Native,
}

fn runtime_reference_target_type(
    target: RuntimeReferenceTarget,
) -> Result<RuntimeExpressionType, CodegenErrorKind> {
    match target {
        RuntimeReferenceTarget::Scalar(element) => Ok(runtime_type_from_array_element(element)),
        RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str => {
            Err(CodegenErrorKind::RuntimeExpressionUnsupported)
        }
        RuntimeReferenceTarget::Array { element, count, .. } => {
            Ok(RuntimeExpressionType::Array { element, count })
        }
    }
}

fn runtime_reference_target_from_type(
    ty: RuntimeExpressionType,
) -> Result<RuntimeReferenceTarget, CodegenErrorKind> {
    match ty {
        RuntimeExpressionType::Array { element, count } => Ok(RuntimeReferenceTarget::Array {
            element,
            count,
            layout: RuntimeReferenceArrayLayout::Native,
        }),
        RuntimeExpressionType::Reference { .. } | RuntimeExpressionType::RawPointer { .. } => {
            Err(CodegenErrorKind::RuntimeExpressionUnsupported)
        }
        scalar => Ok(RuntimeReferenceTarget::Scalar(runtime_array_element_type(
            scalar,
        )?)),
    }
}

fn runtime_array_element_type(
    ty: RuntimeExpressionType,
) -> Result<RuntimeArrayElementType, CodegenErrorKind> {
    match ty {
        RuntimeExpressionType::Unit => Ok(RuntimeArrayElementType::Unit),
        RuntimeExpressionType::Default => Ok(RuntimeArrayElementType::Default),
        RuntimeExpressionType::Integer(ty) => Ok(RuntimeArrayElementType::Integer(ty)),
        RuntimeExpressionType::Bool => Ok(RuntimeArrayElementType::Bool),
        RuntimeExpressionType::Char => Ok(RuntimeArrayElementType::Char),
        RuntimeExpressionType::Array { .. }
        | RuntimeExpressionType::Reference { .. }
        | RuntimeExpressionType::RawPointer { .. } => Err(CodegenErrorKind::RuntimeTypeMismatch),
    }
}

fn runtime_type_from_array_element(element: RuntimeArrayElementType) -> RuntimeExpressionType {
    match element {
        RuntimeArrayElementType::Unit => RuntimeExpressionType::Unit,
        RuntimeArrayElementType::Default => RuntimeExpressionType::Default,
        RuntimeArrayElementType::Integer(ty) => RuntimeExpressionType::Integer(ty),
        RuntimeArrayElementType::Bool => RuntimeExpressionType::Bool,
        RuntimeArrayElementType::Char => RuntimeExpressionType::Char,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeLocal<'source> {
    name: &'source str,
    ty: RuntimeExpressionType,
    mutable: bool,
    stack_slots: usize,
}

fn runtime_array_type(text: &str) -> Option<RuntimeExpressionType> {
    let body = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    let (element, count) = body.split_once(';')?;
    let count = count.trim().parse::<usize>().ok()?;
    if count > crate::expression::MAX_ARRAY_ELEMENTS {
        return None;
    }
    let element = match element.trim() {
        "()" => RuntimeArrayElementType::Unit,
        "bool" => RuntimeArrayElementType::Bool,
        "char" => RuntimeArrayElementType::Char,
        name => RuntimeArrayElementType::Integer(Some(crate::IntegerType::from_name(name)?)),
    };
    Some(RuntimeExpressionType::Array { element, count })
}

fn runtime_reference_type(text: &str) -> Option<RuntimeExpressionType> {
    let target = text.trim().strip_prefix('&')?.trim();
    let (target, mutable) = target
        .strip_prefix("mut ")
        .map_or((target, false), |target| (target.trim(), true));
    let target = match runtime_array_type(target) {
        Some(RuntimeExpressionType::Array { element, count }) => RuntimeReferenceTarget::Array {
            element,
            count,
            layout: RuntimeReferenceArrayLayout::Native,
        },
        _ if target == "str" => RuntimeReferenceTarget::Str,
        _ if target.starts_with('[') && target.ends_with(']') => {
            let element = match target.strip_prefix('[')?.strip_suffix(']')?.trim() {
                "()" => RuntimeArrayElementType::Unit,
                "bool" => RuntimeArrayElementType::Bool,
                "char" => RuntimeArrayElementType::Char,
                name => {
                    RuntimeArrayElementType::Integer(Some(crate::IntegerType::from_name(name)?))
                }
            };
            RuntimeReferenceTarget::Slice(element)
        }
        _ => RuntimeReferenceTarget::Scalar(match target {
            "bool" => RuntimeArrayElementType::Bool,
            "char" => RuntimeArrayElementType::Char,
            name => RuntimeArrayElementType::Integer(Some(crate::IntegerType::from_name(name)?)),
        }),
    };
    Some(RuntimeExpressionType::Reference { target, mutable })
}

fn runtime_raw_pointer_type(text: &str) -> Option<RuntimeExpressionType> {
    let target = text.trim().strip_prefix('*')?.trim();
    let (target, mutable) = if let Some(target) = target.strip_prefix("const ") {
        (target.trim(), false)
    } else {
        (target.strip_prefix("mut ")?.trim(), true)
    };
    let pointee = match target {
        "bool" => RuntimeArrayElementType::Bool,
        "char" => RuntimeArrayElementType::Char,
        name => RuntimeArrayElementType::Integer(Some(crate::IntegerType::from_name(name)?)),
    };
    Some(RuntimeExpressionType::RawPointer { pointee, mutable })
}

fn runtime_type_stack_slots(ty: RuntimeExpressionType) -> usize {
    match ty {
        RuntimeExpressionType::Array { count, .. } => count,
        RuntimeExpressionType::Reference {
            target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
            ..
        } => 2,
        _ => 1,
    }
}

fn runtime_local_stack_slots(ty: RuntimeExpressionType) -> usize {
    let RuntimeExpressionType::Array { element, count } = ty else {
        return runtime_type_stack_slots(ty);
    };
    let Some(element_bytes @ (1 | 2 | 4)) = runtime_array_element_bytes(element) else {
        return count;
    };
    element_bytes
        .checked_mul(count)
        .and_then(|bytes| bytes.checked_add(7))
        .map_or(count, |bytes| bytes / 8)
}

fn runtime_expression_width(ty: RuntimeExpressionType) -> Option<RuntimeWidth> {
    match ty {
        RuntimeExpressionType::Integer(Some(integer_type)) => runtime_width(integer_type.name()),
        RuntimeExpressionType::Char => runtime_width("char"),
        RuntimeExpressionType::Array {
            element: RuntimeArrayElementType::Integer(Some(integer_type)),
            ..
        } => runtime_width(integer_type.name()),
        RuntimeExpressionType::Array {
            element: RuntimeArrayElementType::Char,
            ..
        } => runtime_width("char"),
        _ => None,
    }
}

fn runtime_array_element_bytes(element: RuntimeArrayElementType) -> Option<usize> {
    match element {
        RuntimeArrayElementType::Unit | RuntimeArrayElementType::Default => Some(0),
        RuntimeArrayElementType::Bool => Some(1),
        RuntimeArrayElementType::Char => Some(4),
        RuntimeArrayElementType::Integer(Some(ty)) => Some(usize::from(ty.bits(64)?) / 8),
        RuntimeArrayElementType::Integer(None) => None,
    }
}

fn runtime_array_abi_layout(ty: RuntimeExpressionType) -> Option<(usize, usize, usize)> {
    let RuntimeExpressionType::Array { element, count } = ty else {
        return None;
    };
    let element_bytes = runtime_array_element_bytes(element)?;
    let total_bytes = element_bytes.checked_mul(count)?;
    let words = total_bytes.checked_add(7)? / 8;
    Some((element_bytes, total_bytes, words))
}

fn runtime_array_elements_in_word(ty: RuntimeExpressionType, word: usize) -> Option<usize> {
    let RuntimeExpressionType::Array { element, count } = ty else {
        return None;
    };
    let element_bytes = runtime_array_element_bytes(element)?;
    Some(
        (0..count)
            .filter(|index| index * element_bytes / 8 == word)
            .count(),
    )
}

fn runtime_types_compatible(left: RuntimeExpressionType, right: RuntimeExpressionType) -> bool {
    match (left, right) {
        (RuntimeExpressionType::Default, _) | (_, RuntimeExpressionType::Default) => true,
        (RuntimeExpressionType::Unit, RuntimeExpressionType::Unit) => true,
        (RuntimeExpressionType::Bool, RuntimeExpressionType::Bool) => true,
        (RuntimeExpressionType::Char, RuntimeExpressionType::Char) => true,
        (
            RuntimeExpressionType::Reference {
                target: left,
                mutable: left_mutable,
            },
            RuntimeExpressionType::Reference {
                target: right,
                mutable: right_mutable,
            },
        ) => {
            let targets_compatible = match (left, right) {
                (RuntimeReferenceTarget::Scalar(left), RuntimeReferenceTarget::Scalar(right)) => {
                    left == right
                }
                (RuntimeReferenceTarget::Slice(left), RuntimeReferenceTarget::Slice(right)) => {
                    left == right
                }
                (RuntimeReferenceTarget::Str, RuntimeReferenceTarget::Str) => true,
                (
                    RuntimeReferenceTarget::Array {
                        element: left,
                        count: left_count,
                        ..
                    },
                    RuntimeReferenceTarget::Array {
                        element: right,
                        count: right_count,
                        ..
                    },
                ) => left == right && left_count == right_count,
                _ => false,
            };
            targets_compatible
                && (left_mutable == right_mutable || (left_mutable && !right_mutable))
        }
        (
            RuntimeExpressionType::RawPointer {
                pointee: left,
                mutable: left_mutable,
            },
            RuntimeExpressionType::RawPointer {
                pointee: right,
                mutable: right_mutable,
            },
        ) => left == right && (left_mutable == right_mutable || (left_mutable && !right_mutable)),
        (RuntimeExpressionType::Integer(left), RuntimeExpressionType::Integer(right)) => {
            left.is_none() || right.is_none() || left == right
        }
        (
            RuntimeExpressionType::Array {
                element: left,
                count: left_count,
            },
            RuntimeExpressionType::Array {
                element: right,
                count: right_count,
            },
        ) => {
            left_count == right_count
                && (left == RuntimeArrayElementType::Default
                    || right == RuntimeArrayElementType::Default
                    || left == right
                    || matches!(
                        (left, right),
                        (
                            RuntimeArrayElementType::Integer(None),
                            RuntimeArrayElementType::Integer(Some(_))
                        ) | (
                            RuntimeArrayElementType::Integer(Some(_)),
                            RuntimeArrayElementType::Integer(None)
                        )
                    ))
        }
        _ => false,
    }
}

fn reference_source_name<'source, const MAX_NODES: usize>(
    tree: &crate::ExpressionTree<'source, MAX_NODES>,
    id: crate::ExprId,
) -> Option<&'source str> {
    match tree.expression(id)?.kind {
        ExprKind::Identifier(name) => Some(name),
        ExprKind::StrAsBytes { base } => reference_source_name(tree, base),
        ExprKind::Unary {
            operator: crate::UnaryOperator::AddressOf | crate::UnaryOperator::AddressOfMut,
            operand,
        } => match tree.expression(operand)?.kind {
            ExprKind::Unary {
                operator: crate::UnaryOperator::Dereference,
                operand,
            } => reference_source_name(tree, operand),
            _ => None,
        },
        _ => None,
    }
}

fn constant_range_bounds<R: ConstantResolver, const MAX_NODES: usize>(
    tree: &crate::ExpressionTree<'_, MAX_NODES>,
    start: Option<crate::ExprId>,
    end: Option<crate::ExprId>,
    inclusive: bool,
    count: usize,
    resolver: &R,
) -> Result<(usize, usize), CodegenErrorKind> {
    let evaluate = |id| {
        tree.evaluate_at(id, resolver)
            .map_err(|error| match error {
                crate::ConstEvalError::UnknownIdentifier => {
                    CodegenErrorKind::RuntimeExpressionUnsupported
                }
                error => CodegenErrorKind::Execution(ExecutionError::Arithmetic(error)),
            })
            .and_then(|value| usize::try_from(value).map_err(|_| CodegenErrorKind::ValueOutOfRange))
    };
    let start = start.map_or(Ok(0), evaluate)?;
    let mut end = end.map_or_else(
        || {
            if inclusive {
                Err(CodegenErrorKind::RuntimeExpressionUnsupported)
            } else {
                Ok(count)
            }
        },
        evaluate,
    )?;
    if inclusive {
        end = end
            .checked_add(1)
            .ok_or(CodegenErrorKind::ValueOutOfRange)?;
    }
    if start > end || end > count {
        return Err(CodegenErrorKind::ValueOutOfRange);
    }
    Ok((start, end))
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
        ExprKind::Array {
            elements,
            element_count,
        } => {
            for element in elements[..element_count].iter().flatten() {
                recurse(*element)?;
            }
        }
        ExprKind::ArrayRepeat { element, .. } => recurse(element)?,
        ExprKind::Index { base, index } => {
            recurse(base)?;
            recurse(index)?;
        }
        ExprKind::RangeIndex {
            base, start, end, ..
        } => {
            recurse(base)?;
            if let Some(start) = start {
                recurse(start)?;
            }
            if let Some(end) = end {
                recurse(end)?;
            }
        }
        ExprKind::SliceLen { base } => recurse(base)?,
        ExprKind::SliceIsEmpty { base } => recurse(base)?,
        ExprKind::StrAsBytes { base } => recurse(base)?,
        ExprKind::StrIsCharBoundary { base, index } => {
            recurse(base)?;
            recurse(index)?;
        }
        ExprKind::ReferenceAsPointer { base, .. } => recurse(base)?,
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
        ExprKind::Unit
        | ExprKind::DefaultValue
        | ExprKind::Integer(_)
        | ExprKind::Bool(_)
        | ExprKind::Char(_) => {}
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
            RuntimeExpressionType::Unit | RuntimeExpressionType::Default => {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            RuntimeExpressionType::Array { .. }
            | RuntimeExpressionType::Reference { .. }
            | RuntimeExpressionType::RawPointer { .. } => {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
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
        ExprKind::DefaultValue => Ok(RuntimeExpressionType::Default),
        ExprKind::Array {
            elements,
            element_count,
        } => {
            let mut element_type = RuntimeArrayElementType::Default;
            for element in elements[..element_count].iter().flatten() {
                let current = runtime_array_element_type(runtime_expression_type_with_locals(
                    function,
                    resolver,
                    locals,
                    tree,
                    *element,
                    depth + 1,
                )?)?;
                if element_type == RuntimeArrayElementType::Default {
                    element_type = current;
                } else if current != RuntimeArrayElementType::Default && current != element_type {
                    let compatible_integer = matches!(
                        (element_type, current),
                        (
                            RuntimeArrayElementType::Integer(None),
                            RuntimeArrayElementType::Integer(Some(_))
                        ) | (
                            RuntimeArrayElementType::Integer(Some(_)),
                            RuntimeArrayElementType::Integer(None)
                        )
                    );
                    if !compatible_integer {
                        return Err(CodegenErrorKind::RuntimeTypeMismatch);
                    }
                    if matches!(element_type, RuntimeArrayElementType::Integer(None)) {
                        element_type = current;
                    }
                }
            }
            Ok(RuntimeExpressionType::Array {
                element: element_type,
                count: element_count,
            })
        }
        ExprKind::ArrayRepeat { element, count } => Ok(RuntimeExpressionType::Array {
            element: runtime_array_element_type(runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                element,
                depth + 1,
            )?)?,
            count,
        }),
        ExprKind::Index { base, index } => {
            let index_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                index,
                depth + 1,
            )?;
            if !matches!(
                index_type,
                RuntimeExpressionType::Integer(None)
                    | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
            ) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            let base_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                base,
                depth + 1,
            )?;
            let element = match base_type {
                RuntimeExpressionType::Array { element, .. }
                | RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Array { element, .. },
                    ..
                }
                | RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(element),
                    ..
                } => element,
                _ => return Err(CodegenErrorKind::RuntimeTypeMismatch),
            };
            Ok(runtime_type_from_array_element(element))
        }
        ExprKind::RangeIndex { .. } => Err(CodegenErrorKind::RuntimeExpressionUnsupported),
        ExprKind::SliceLen { base } => {
            let base_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                base,
                depth + 1,
            )?;
            if !matches!(
                base_type,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                }
            ) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            Ok(RuntimeExpressionType::Integer(Some(
                crate::IntegerType::Usize,
            )))
        }
        ExprKind::SliceIsEmpty { base } => {
            let base_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                base,
                depth + 1,
            )?;
            if !matches!(
                base_type,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                }
            ) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            Ok(RuntimeExpressionType::Bool)
        }
        ExprKind::StrAsBytes { base } => {
            let base_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                base,
                depth + 1,
            )?;
            let RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Str,
                ..
            } = base_type
            else {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            };
            Ok(RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Slice(RuntimeArrayElementType::Integer(Some(
                    crate::IntegerType::U8,
                ))),
                mutable: false,
            })
        }
        ExprKind::StrIsCharBoundary { base, index } => {
            if !matches!(
                runtime_expression_type_with_locals(
                    function,
                    resolver,
                    locals,
                    tree,
                    base,
                    depth + 1,
                )?,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Str,
                    ..
                }
            ) || !matches!(
                runtime_expression_type_with_locals(
                    function,
                    resolver,
                    locals,
                    tree,
                    index,
                    depth + 1,
                )?,
                RuntimeExpressionType::Integer(None)
                    | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
            ) {
                return Err(CodegenErrorKind::RuntimeTypeMismatch);
            }
            Ok(RuntimeExpressionType::Bool)
        }
        ExprKind::ReferenceAsPointer { base, mutable } => {
            let base_type = runtime_expression_type_with_locals(
                function,
                resolver,
                locals,
                tree,
                base,
                depth + 1,
            )?;
            let (pointee, base_mutable) = match base_type {
                RuntimeExpressionType::Reference {
                    target:
                        RuntimeReferenceTarget::Slice(pointee)
                        | RuntimeReferenceTarget::Array {
                            element: pointee, ..
                        },
                    mutable,
                } => (pointee, mutable),
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Str,
                    mutable,
                } => (
                    RuntimeArrayElementType::Integer(Some(crate::IntegerType::U8)),
                    mutable,
                ),
                _ => return Err(CodegenErrorKind::RuntimeTypeMismatch),
            };
            if mutable && !base_mutable {
                return Err(CodegenErrorKind::ImmutableAssignment);
            }
            Ok(RuntimeExpressionType::RawPointer { pointee, mutable })
        }
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
                        if let Some(array) = runtime_array_type(parameter.ty.text) {
                            array
                        } else if let Some(reference) = runtime_reference_type(parameter.ty.text) {
                            reference
                        } else if let Some(pointer) = runtime_raw_pointer_type(parameter.ty.text) {
                            pointer
                        } else if parameter.ty.text == "bool" {
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
                (RuntimeExpressionType::Default, resolved)
                | (resolved, RuntimeExpressionType::Default) => resolved,
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
        ExprKind::Unary {
            operator: crate::UnaryOperator::Dereference,
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
            match operand_type {
                RuntimeExpressionType::Reference { target, .. } => {
                    runtime_reference_target_type(target)
                }
                RuntimeExpressionType::RawPointer { pointee, .. } => {
                    Ok(runtime_type_from_array_element(pointee))
                }
                _ => Err(CodegenErrorKind::RuntimeTypeMismatch),
            }
        }
        ExprKind::Unary {
            operator:
                operator @ (crate::UnaryOperator::AddressOf | crate::UnaryOperator::AddressOfMut),
            operand,
        } => {
            let operand_kind = tree
                .expression(operand)
                .map(|expression| expression.kind)
                .ok_or(CodegenErrorKind::RuntimeExpressionUnsupported)?;
            let target = match operand_kind {
                ExprKind::Identifier(name) => {
                    let local = locals
                        .iter()
                        .flatten()
                        .rev()
                        .find(|local| local.name == name);
                    if let Some(local) = local {
                        if operator == crate::UnaryOperator::AddressOfMut && !local.mutable {
                            return Err(CodegenErrorKind::ImmutableAssignment);
                        }
                    } else {
                        function
                            .parameters()
                            .iter()
                            .flatten()
                            .find(|parameter| parameter.name == name)
                            .ok_or(CodegenErrorKind::UnknownRuntimeName)?;
                        if operator == crate::UnaryOperator::AddressOfMut {
                            return Err(CodegenErrorKind::ImmutableAssignment);
                        }
                    }
                    runtime_reference_target_from_type(runtime_expression_type_with_locals(
                        function,
                        resolver,
                        locals,
                        tree,
                        operand,
                        depth + 1,
                    )?)?
                }
                ExprKind::Unary {
                    operator: crate::UnaryOperator::Dereference,
                    operand: reference,
                } => {
                    let reference_type = runtime_expression_type_with_locals(
                        function,
                        resolver,
                        locals,
                        tree,
                        reference,
                        depth + 1,
                    )?;
                    let RuntimeExpressionType::Reference { target, mutable } = reference_type
                    else {
                        return Err(CodegenErrorKind::RuntimeTypeMismatch);
                    };
                    if operator == crate::UnaryOperator::AddressOfMut && !mutable {
                        return Err(CodegenErrorKind::ImmutableAssignment);
                    }
                    target
                }
                ExprKind::RangeIndex {
                    base,
                    start,
                    end,
                    inclusive,
                } => {
                    let base_type = runtime_expression_type_with_locals(
                        function,
                        resolver,
                        locals,
                        tree,
                        base,
                        depth + 1,
                    )?;
                    let (target, count, mutable) = match base_type {
                        RuntimeExpressionType::Reference {
                            target:
                                RuntimeReferenceTarget::Array {
                                    element,
                                    count,
                                    layout: RuntimeReferenceArrayLayout::Native,
                                },
                            mutable,
                        } => (RuntimeReferenceTarget::Slice(element), Some(count), mutable),
                        RuntimeExpressionType::Reference {
                            target: RuntimeReferenceTarget::Slice(element),
                            mutable,
                        } => (RuntimeReferenceTarget::Slice(element), None, mutable),
                        RuntimeExpressionType::Reference {
                            target: RuntimeReferenceTarget::Str,
                            mutable,
                        } => (RuntimeReferenceTarget::Str, None, mutable),
                        RuntimeExpressionType::Array { element, count } => {
                            let Some(ExprKind::Identifier(name)) =
                                tree.expression(base).map(|expression| expression.kind)
                            else {
                                return Err(CodegenErrorKind::RuntimeTypeMismatch);
                            };
                            let local = locals
                                .iter()
                                .flatten()
                                .rev()
                                .find(|local| local.name == name)
                                .ok_or(CodegenErrorKind::RuntimeTypeMismatch)?;
                            (
                                RuntimeReferenceTarget::Slice(element),
                                Some(count),
                                local.mutable,
                            )
                        }
                        _ => return Err(CodegenErrorKind::RuntimeTypeMismatch),
                    };
                    if operator == crate::UnaryOperator::AddressOfMut && !mutable {
                        return Err(CodegenErrorKind::ImmutableAssignment);
                    }
                    if inclusive && end.is_none() {
                        return Err(CodegenErrorKind::RuntimeExpressionUnsupported);
                    }
                    for endpoint in [start, end].into_iter().flatten() {
                        if !matches!(
                            runtime_expression_type_with_locals(
                                function,
                                resolver,
                                locals,
                                tree,
                                endpoint,
                                depth + 1,
                            )?,
                            RuntimeExpressionType::Integer(None)
                                | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
                        ) {
                            return Err(CodegenErrorKind::RuntimeTypeMismatch);
                        }
                    }
                    if let Some(count) = count {
                        match constant_range_bounds(tree, start, end, inclusive, count, resolver) {
                            Ok(_) | Err(CodegenErrorKind::RuntimeExpressionUnsupported) => {}
                            Err(error) => return Err(error),
                        }
                    }
                    target
                }
                ExprKind::Index { base, index } => {
                    if !matches!(
                        runtime_expression_type_with_locals(
                            function,
                            resolver,
                            locals,
                            tree,
                            index,
                            depth + 1,
                        )?,
                        RuntimeExpressionType::Integer(None)
                            | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
                    ) {
                        return Err(CodegenErrorKind::RuntimeTypeMismatch);
                    }
                    let base_type = runtime_expression_type_with_locals(
                        function,
                        resolver,
                        locals,
                        tree,
                        base,
                        depth + 1,
                    )?;
                    let (element, mutable) = match base_type {
                        RuntimeExpressionType::Reference {
                            target:
                                RuntimeReferenceTarget::Array { element, .. }
                                | RuntimeReferenceTarget::Slice(element),
                            mutable,
                        } => (element, mutable),
                        RuntimeExpressionType::Array { element, .. } => {
                            let Some(ExprKind::Identifier(name)) =
                                tree.expression(base).map(|expression| expression.kind)
                            else {
                                return Err(CodegenErrorKind::RuntimeTypeMismatch);
                            };
                            let mutable = locals
                                .iter()
                                .flatten()
                                .rev()
                                .find(|local| local.name == name)
                                .is_some_and(|local| local.mutable);
                            (element, mutable)
                        }
                        _ => return Err(CodegenErrorKind::RuntimeTypeMismatch),
                    };
                    if operator == crate::UnaryOperator::AddressOfMut && !mutable {
                        return Err(CodegenErrorKind::ImmutableAssignment);
                    }
                    runtime_reference_target_from_type(runtime_type_from_array_element(element))?
                }
                _ => return Err(CodegenErrorKind::RuntimeExpressionUnsupported),
            };
            Ok(RuntimeExpressionType::Reference {
                target,
                mutable: operator == crate::UnaryOperator::AddressOfMut,
            })
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
                    | RuntimeExpressionType::Default
                    | RuntimeExpressionType::Array { .. }
                    | RuntimeExpressionType::Reference { .. }
                    | RuntimeExpressionType::RawPointer { .. }
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
    uses_sret: bool,
    saved_parameter_slots: usize,
    locals: [Option<RuntimeLocal<'source>>; MAX_PARAMETERS],
    saved_locals: usize,
    evaluation_depth: usize,
}

impl<'tree, 'source, R: ConstantResolver, const MAX_BYTES: usize, const MAX_PARAMETERS: usize>
    RuntimeEmitter<'tree, 'source, R, MAX_BYTES, MAX_PARAMETERS>
{
    fn local_stack_slots(&self) -> Result<usize, CodegenError> {
        self.locals[..self.saved_locals]
            .iter()
            .flatten()
            .try_fold(0usize, |total, local| {
                total
                    .checked_add(local.stack_slots)
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))
            })
    }

    fn local_stack_slots_from(&self, index: usize) -> Result<usize, CodegenError> {
        self.locals[index..self.saved_locals]
            .iter()
            .flatten()
            .try_fold(0usize, |total, local| {
                total
                    .checked_add(local.stack_slots)
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))
            })
    }

    fn parameter_stack_slots_from(&self, index: usize) -> Result<usize, CodegenError> {
        self.function
            .parameters()
            .iter()
            .flatten()
            .skip(index)
            .try_fold(0usize, |total, parameter| {
                let slots = runtime_array_type(parameter.ty.text)
                    .or_else(|| runtime_reference_type(parameter.ty.text))
                    .map_or(1, runtime_type_stack_slots);
                total
                    .checked_add(slots)
                    .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))
            })
    }

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
        match self.abi {
            X86_64Abi::Windows => {
                let mut pushed_slots = usize::from(self.uses_sret);
                if self.uses_sret {
                    self.emit(&[0x51])?;
                }
                for (index, parameter) in self.function.parameters().iter().flatten().enumerate() {
                    let abi_index = index + usize::from(self.uses_sret);
                    if matches!(
                        runtime_reference_type(parameter.ty.text),
                        Some(RuntimeExpressionType::Reference {
                            target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                            ..
                        })
                    ) {
                        self.emit_windows_parameter_pointer(abi_index, pushed_slots)?;
                        self.emit(&[0x4c, 0x8b, 0x10, 0x41, 0x52])?;
                        self.emit(&[0x4c, 0x8b, 0x50, 0x08, 0x41, 0x52])?;
                        pushed_slots += 2;
                    } else if let Some(array) = runtime_array_type(parameter.ty.text) {
                        let (_, bytes, _) = runtime_array_abi_layout(array)
                            .ok_or(self.error(CodegenErrorKind::UnsupportedParameterType))?;
                        let RuntimeExpressionType::Array { count, .. } = array else {
                            unreachable!()
                        };
                        if bytes == 0 {
                            self.emit_zero_array_slots(count)?;
                        } else if matches!(bytes, 1 | 2 | 4 | 8) {
                            self.emit_parameter_register_or_stack_load(
                                X86_64Abi::Windows,
                                abi_index,
                                40,
                                pushed_slots,
                            )?;
                            self.emit_unpack_array_word(array, 0)?;
                        } else {
                            self.emit_windows_parameter_pointer(abi_index, pushed_slots)?;
                            self.emit_compact_pointer_array_pushes(array)?;
                        }
                        pushed_slots += count;
                    } else {
                        self.emit_windows_parameter_word(abi_index, pushed_slots)?;
                        pushed_slots += 1;
                    }
                }
            }
            X86_64Abi::SystemV => {
                let mut register = usize::from(self.uses_sret);
                let mut stack_offset = 8usize;
                let mut pushed_slots = usize::from(self.uses_sret);
                if self.uses_sret {
                    self.emit(&[0x57])?;
                }
                for parameter in self.function.parameters().iter().flatten() {
                    if matches!(
                        runtime_reference_type(parameter.ty.text),
                        Some(RuntimeExpressionType::Reference {
                            target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                            ..
                        })
                    ) {
                        if register + 2 <= 6 {
                            for _ in 0..2 {
                                self.emit_parameter_register_load(X86_64Abi::SystemV, register)?;
                                self.emit(&[0x50])?;
                                register += 1;
                                pushed_slots += 1;
                            }
                        } else {
                            for word in 0..2 {
                                self.emit_original_stack_word_push(
                                    stack_offset + word * 8,
                                    pushed_slots,
                                )?;
                                pushed_slots += 1;
                            }
                            stack_offset += 16;
                        }
                    } else if let Some(array) = runtime_array_type(parameter.ty.text) {
                        let (_, _, words) = runtime_array_abi_layout(array)
                            .ok_or(self.error(CodegenErrorKind::UnsupportedParameterType))?;
                        let RuntimeExpressionType::Array { count, .. } = array else {
                            unreachable!()
                        };
                        if words == 0 {
                            self.emit_zero_array_slots(count)?;
                            pushed_slots += count;
                        } else if words <= 2 && register + words <= 6 {
                            for word in 0..words {
                                self.emit_parameter_register_load(X86_64Abi::SystemV, register)?;
                                self.emit_unpack_array_word(array, word)?;
                                pushed_slots += runtime_array_elements_in_word(array, word).ok_or(
                                    self.error(CodegenErrorKind::UnsupportedParameterType),
                                )?;
                                register += 1;
                            }
                        } else {
                            for word in 0..words {
                                let original = stack_offset
                                    .checked_add(word * 8)
                                    .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
                                self.emit_original_stack_word_load(original, pushed_slots)?;
                                self.emit_unpack_array_word(array, word)?;
                                pushed_slots += runtime_array_elements_in_word(array, word).ok_or(
                                    self.error(CodegenErrorKind::UnsupportedParameterType),
                                )?;
                            }
                            stack_offset = stack_offset
                                .checked_add(words * 8)
                                .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
                        }
                    } else if register < 6 {
                        self.emit(
                            parameter_push_encoding(X86_64Abi::SystemV, register)
                                .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?,
                        )?;
                        register += 1;
                        pushed_slots += 1;
                    } else {
                        self.emit_original_stack_word_push(stack_offset, pushed_slots)?;
                        pushed_slots += 1;
                        stack_offset = stack_offset
                            .checked_add(8)
                            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_windows_parameter_word(
        &mut self,
        index: usize,
        pushed_slots: usize,
    ) -> Result<(), CodegenError> {
        if let Some(encoding) = parameter_push_encoding(X86_64Abi::Windows, index) {
            self.emit(encoding)
        } else {
            let original = 40usize
                .checked_add((index - 4) * 8)
                .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
            self.emit_original_stack_word_push(original, pushed_slots)
        }
    }

    fn emit_parameter_register_or_stack_load(
        &mut self,
        abi: X86_64Abi,
        index: usize,
        first_stack_offset: usize,
        pushed_slots: usize,
    ) -> Result<(), CodegenError> {
        let register_count = match abi {
            X86_64Abi::Windows => 4,
            X86_64Abi::SystemV => 6,
        };
        if index < register_count {
            return self.emit_parameter_register_load(abi, index);
        }
        let original = first_stack_offset
            .checked_add((index - register_count) * 8)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        self.emit_original_stack_word_load(original, pushed_slots)
    }

    fn emit_parameter_register_load(
        &mut self,
        abi: X86_64Abi,
        index: usize,
    ) -> Result<(), CodegenError> {
        let encoding: &[u8] = match (abi, index) {
            (X86_64Abi::Windows, 0) | (X86_64Abi::SystemV, 3) => &[0x48, 0x89, 0xc8],
            (X86_64Abi::Windows, 1) | (X86_64Abi::SystemV, 2) => &[0x48, 0x89, 0xd0],
            (X86_64Abi::Windows, 2) | (X86_64Abi::SystemV, 4) => &[0x4c, 0x89, 0xc0],
            (X86_64Abi::Windows, 3) | (X86_64Abi::SystemV, 5) => &[0x4c, 0x89, 0xc8],
            (X86_64Abi::SystemV, 0) => &[0x48, 0x89, 0xf8],
            (X86_64Abi::SystemV, 1) => &[0x48, 0x89, 0xf0],
            _ => return Err(self.error(CodegenErrorKind::TooManyAbiParameters)),
        };
        self.emit(encoding)
    }

    fn emit_unpack_array_word(
        &mut self,
        array: RuntimeExpressionType,
        word: usize,
    ) -> Result<(), CodegenError> {
        let RuntimeExpressionType::Array { element, count } = array else {
            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
        };
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedParameterType))?;
        for index in 0..count {
            let byte_offset = index * element_bytes;
            if byte_offset / 8 != word {
                continue;
            }
            self.emit(&[0x49, 0x89, 0xc2])?;
            let shift = ((byte_offset % 8) * 8) as u8;
            if shift != 0 {
                self.emit(&[0x49, 0xc1, 0xea, shift])?;
            }
            match element_bytes {
                1 => self.emit(&[0x45, 0x0f, 0xb6, 0xd2])?,
                2 => self.emit(&[0x45, 0x0f, 0xb7, 0xd2])?,
                4 | 8 => {}
                _ => return Err(self.error(CodegenErrorKind::UnsupportedParameterType)),
            }
            self.emit(&[0x41, 0x52])?;
        }
        Ok(())
    }

    fn emit_zero_array_slots(&mut self, count: usize) -> Result<(), CodegenError> {
        if count != 0 {
            self.emit(&[0x31, 0xc0])?;
        }
        for _ in 0..count {
            self.emit(&[0x50])?;
        }
        Ok(())
    }

    fn emit_compact_pointer_array_pushes(
        &mut self,
        array: RuntimeExpressionType,
    ) -> Result<(), CodegenError> {
        let RuntimeExpressionType::Array { element, count } = array else {
            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
        };
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedParameterType))?;
        for index in 0..count {
            let displacement = u8::try_from(index * element_bytes)
                .map_err(|_| self.error(CodegenErrorKind::TooManyAbiParameters))?;
            match element_bytes {
                1 => self.emit(&[0x44, 0x0f, 0xb6, 0x50, displacement])?,
                2 => self.emit(&[0x44, 0x0f, 0xb7, 0x50, displacement])?,
                4 => self.emit(&[0x44, 0x8b, 0x50, displacement])?,
                8 if displacement == 0 => self.emit(&[0x4c, 0x8b, 0x10])?,
                8 => self.emit(&[0x4c, 0x8b, 0x50, displacement])?,
                _ => return Err(self.error(CodegenErrorKind::UnsupportedParameterType)),
            }
            self.emit(&[0x41, 0x52])?;
        }
        Ok(())
    }

    fn emit_windows_parameter_pointer(
        &mut self,
        index: usize,
        pushed_slots: usize,
    ) -> Result<(), CodegenError> {
        match index {
            0 => self.emit(&[0x48, 0x89, 0xc8]),
            1 => self.emit(&[0x48, 0x89, 0xd0]),
            2 => self.emit(&[0x4c, 0x89, 0xc0]),
            3 => self.emit(&[0x4c, 0x89, 0xc8]),
            _ => {
                let original = 40usize
                    .checked_add((index - 4) * 8)
                    .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
                self.emit_original_stack_word_load(original, pushed_slots)
            }
        }
    }

    fn emit_original_stack_word_push(
        &mut self,
        original_offset: usize,
        pushed_slots: usize,
    ) -> Result<(), CodegenError> {
        self.emit_original_stack_word_load(original_offset, pushed_slots)?;
        self.emit(&[0x50])
    }

    fn emit_original_stack_word_load(
        &mut self,
        original_offset: usize,
        pushed_slots: usize,
    ) -> Result<(), CodegenError> {
        let current_stack_delta = pushed_slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        let displacement = original_offset
            .checked_add(current_stack_delta)
            .ok_or(self.error(CodegenErrorKind::TooManyAbiParameters))?;
        if displacement <= i8::MAX as usize {
            self.emit(&[0x48, 0x8b, 0x44, 0x24, displacement as u8])?;
        } else {
            let displacement = u32::try_from(displacement)
                .map_err(|_| self.error(CodegenErrorKind::TooManyAbiParameters))?;
            self.emit(&[0x48, 0x8b, 0x84, 0x24])?;
            self.emit(&displacement.to_le_bytes())?;
        }
        Ok(())
    }

    fn emit_epilogue(&mut self) -> Result<(), CodegenError> {
        let bytes = self
            .saved_parameter_slots
            .checked_add(self.local_stack_slots()?)
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

    fn emit_stack_cleanup_bytes(&mut self, bytes: usize) -> Result<(), CodegenError> {
        if bytes == 0 {
            return Ok(());
        }
        if bytes <= i8::MAX as usize {
            self.emit(&[0x48, 0x83, 0xc4, bytes as u8])
        } else {
            let bytes =
                u32::try_from(bytes).map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?;
            self.emit(&[0x48, 0x81, 0xc4])?;
            self.emit(&bytes.to_le_bytes())
        }
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
        let previous_width = self.width;
        let expression_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            tree,
            id,
            depth,
        )
        .map_err(|kind| self.error(kind))?;
        let operation_type = if expression_type == RuntimeExpressionType::Bool {
            match tree.expression(id).map(|expression| expression.kind) {
                Some(ExprKind::Binary { left, .. }) => runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    left,
                    depth + 1,
                )
                .map_err(|kind| self.error(kind))?,
                _ => expression_type,
            }
        } else {
            expression_type
        };
        if let RuntimeExpressionType::Integer(Some(integer_type)) = operation_type {
            self.width = runtime_width(integer_type.name())
                .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        } else if operation_type == RuntimeExpressionType::Char {
            self.width = runtime_width("char")
                .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        }
        let result = self.emit_expression_with_width(tree, id, depth);
        self.width = previous_width;
        result
    }

    fn emit_expression_with_width<const MAX_NODES: usize>(
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
            ExprKind::Unit | ExprKind::DefaultValue => self.emit(&[0x31, 0xc0])?,
            ExprKind::Array { .. } | ExprKind::ArrayRepeat { .. } => {
                return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
            }
            ExprKind::Index { base, index } => {
                let base_type = runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    base,
                    depth + 1,
                )
                .map_err(|kind| self.error(kind))?;
                if let RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(element),
                    ..
                } = base_type
                {
                    let source = reference_source_name(tree, base)
                        .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                    self.emit_slice_runtime_index(tree, base, index, element, source, depth + 1)?;
                    return Ok(());
                }
                if let RuntimeExpressionType::Reference {
                    target:
                        RuntimeReferenceTarget::Array {
                            element,
                            count,
                            layout,
                        },
                    ..
                } = base_type
                {
                    match tree.evaluate_at(index, self.resolver) {
                        Ok(index) => {
                            let index = usize::try_from(index)
                                .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
                            if index >= count {
                                return Err(self.error(CodegenErrorKind::Execution(
                                    ExecutionError::Arithmetic(
                                        crate::ConstEvalError::ArrayIndexOutOfBounds,
                                    ),
                                )));
                            }
                            self.emit_reference_array_constant_index(
                                tree,
                                base,
                                element,
                                index,
                                layout,
                                depth + 1,
                            )?;
                        }
                        Err(crate::ConstEvalError::UnknownIdentifier) => {
                            self.emit_reference_array_runtime_index(
                                tree,
                                base,
                                index,
                                RuntimeReferenceTarget::Array {
                                    element,
                                    count,
                                    layout,
                                },
                                depth + 1,
                            )?;
                        }
                        Err(error) => {
                            return Err(self.error(CodegenErrorKind::Execution(
                                ExecutionError::Arithmetic(error),
                            )));
                        }
                    }
                    return Ok(());
                }
                match tree.evaluate_at(index, self.resolver) {
                    Ok(index) => {
                        let index = usize::try_from(index)
                            .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
                        let RuntimeExpressionType::Array { count, .. } =
                            runtime_expression_type_with_locals(
                                self.function,
                                self.resolver,
                                &self.locals[..self.saved_locals],
                                tree,
                                base,
                                depth + 1,
                            )
                            .map_err(|kind| self.error(kind))?
                        else {
                            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                        };
                        if index >= count {
                            return Err(self.error(CodegenErrorKind::Execution(
                                ExecutionError::Arithmetic(
                                    crate::ConstEvalError::ArrayIndexOutOfBounds,
                                ),
                            )));
                        }
                        self.emit_constant_array_index(tree, base, index, count, depth + 1)?;
                    }
                    Err(crate::ConstEvalError::UnknownIdentifier) => {
                        let RuntimeExpressionType::Array { count, .. } =
                            runtime_expression_type_with_locals(
                                self.function,
                                self.resolver,
                                &self.locals[..self.saved_locals],
                                tree,
                                base,
                                depth + 1,
                            )
                            .map_err(|kind| self.error(kind))?
                        else {
                            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                        };
                        self.emit_runtime_array_index(tree, base, index, count, depth + 1)?;
                    }
                    Err(error) => {
                        return Err(self.error(CodegenErrorKind::Execution(
                            ExecutionError::Arithmetic(error),
                        )));
                    }
                }
            }
            ExprKind::RangeIndex { .. } => {
                return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
            }
            ExprKind::SliceLen { base } => {
                let base_type = runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    base,
                    depth + 1,
                )
                .map_err(|kind| self.error(kind))?;
                if !matches!(
                    base_type,
                    RuntimeExpressionType::Reference {
                        target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                        ..
                    }
                ) {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                }
                let source = reference_source_name(tree, base)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                self.emit_slice_length(source)?;
            }
            ExprKind::SliceIsEmpty { base } => {
                let base_type = runtime_expression_type_with_locals(
                    self.function,
                    self.resolver,
                    &self.locals[..self.saved_locals],
                    tree,
                    base,
                    depth + 1,
                )
                .map_err(|kind| self.error(kind))?;
                if !matches!(
                    base_type,
                    RuntimeExpressionType::Reference {
                        target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                        ..
                    }
                ) {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                }
                let source = reference_source_name(tree, base)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                self.emit_slice_length(source)?;
                self.emit(&[0x48, 0x85, 0xc0, 0x0f, 0x94, 0xc0, 0x0f, 0xb6, 0xc0])?;
            }
            ExprKind::StrAsBytes { base } => {
                self.emit_expression(tree, base, depth + 1)?;
            }
            ExprKind::StrIsCharBoundary { base, index } => {
                self.emit_string_is_char_boundary(tree, base, index, depth + 1)?;
            }
            ExprKind::ReferenceAsPointer { base, .. } => {
                self.emit_expression(tree, base, depth + 1)?;
            }
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
            ExprKind::Unary {
                operator: crate::UnaryOperator::Dereference,
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
                let pointee = match operand_type {
                    RuntimeExpressionType::Reference {
                        target: RuntimeReferenceTarget::Scalar(pointee),
                        ..
                    }
                    | RuntimeExpressionType::RawPointer { pointee, .. } => pointee,
                    RuntimeExpressionType::Reference { .. } => {
                        return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
                    }
                    _ => return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch)),
                };
                self.emit_expression(tree, operand, depth + 1)?;
                self.emit_reference_load(pointee)?;
            }
            ExprKind::Unary {
                operator: crate::UnaryOperator::AddressOf | crate::UnaryOperator::AddressOfMut,
                operand,
            } => match tree.expression(operand).map(|expression| expression.kind) {
                Some(ExprKind::Identifier(name)) => self.emit_identifier_address(name)?,
                Some(ExprKind::Unary {
                    operator: crate::UnaryOperator::Dereference,
                    operand: reference,
                }) => self.emit_expression(tree, reference, depth + 1)?,
                Some(ExprKind::RangeIndex {
                    base,
                    start,
                    end,
                    inclusive,
                }) => {
                    let base_type = runtime_expression_type_with_locals(
                        self.function,
                        self.resolver,
                        &self.locals[..self.saved_locals],
                        tree,
                        base,
                        depth + 1,
                    )
                    .map_err(|kind| self.error(kind))?;
                    let RuntimeExpressionType::Reference {
                        target:
                            RuntimeReferenceTarget::Array {
                                element,
                                count,
                                layout: RuntimeReferenceArrayLayout::Native,
                            },
                        ..
                    } = base_type
                    else {
                        return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                    };
                    let (start, _) =
                        constant_range_bounds(tree, start, end, inclusive, count, self.resolver)
                            .map_err(|kind| self.error(kind))?;
                    self.emit_expression(tree, base, depth + 1)?;
                    let offset =
                        start
                            .checked_mul(runtime_array_element_bytes(element).ok_or(
                                self.error(CodegenErrorKind::RuntimeExpressionUnsupported),
                            )?)
                            .ok_or(self.error(CodegenErrorKind::ValueOutOfRange))?;
                    if offset != 0 {
                        let offset = u32::try_from(offset)
                            .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
                        self.emit(&[0x48, 0x05])?;
                        self.emit(&offset.to_le_bytes())?;
                    }
                }
                Some(ExprKind::Index { base, index }) => {
                    self.emit_index_address(tree, base, index, depth + 1)?;
                }
                _ => {
                    return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
                }
            },
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
                    RuntimeExpressionType::Unit | RuntimeExpressionType::Default => {
                        self.emit(&[0x31, 0xc0])?;
                    }
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
                    RuntimeExpressionType::Array { .. } => {
                        return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
                    }
                    RuntimeExpressionType::Reference { .. } => {
                        return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
                    }
                    RuntimeExpressionType::RawPointer { .. } => {
                        return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_array_element<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        id: crate::ExprId,
        index: usize,
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
            ExprKind::DefaultValue => self.emit(&[0x31, 0xc0]),
            ExprKind::Array {
                elements,
                element_count,
            } => {
                let element = elements
                    .get(index)
                    .filter(|_| index < element_count)
                    .and_then(|element| *element)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                self.emit_expression(tree, element, depth + 1)
            }
            ExprKind::ArrayRepeat { element, count } => {
                if index >= count {
                    return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                }
                self.emit_expression(tree, element, depth + 1)
            }
            ExprKind::Identifier(name) => {
                if let Some((local_index, local)) = self.locals[..self.saved_locals]
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(local_index, local)| {
                        local
                            .filter(|local| local.name == name)
                            .map(|local| (local_index, local))
                    })
                {
                    let RuntimeExpressionType::Array { element, count } = local.ty else {
                        return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                    };
                    if index >= count {
                        return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                    }
                    let element_bytes = runtime_array_element_bytes(element)
                        .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
                    if matches!(element_bytes, 1 | 2 | 4) {
                        let slots = self
                            .local_stack_slots_from(local_index + 1)?
                            .checked_add(self.evaluation_depth)
                            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                        self.emit_stack_slot_address(slots)?;
                        let offset = index
                            .checked_mul(element_bytes)
                            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                        if offset != 0 {
                            self.emit(&[0x48, 0x83, 0xc0, offset as u8])?;
                        }
                        return self.emit_reference_load(element);
                    }
                    let slots = self
                        .local_stack_slots_from(local_index + 1)?
                        .checked_add(index)
                        .and_then(|slots| slots.checked_add(self.evaluation_depth))
                        .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                    self.emit_stack_slot(slots)?;
                    return self.emit_normalize_array_element(element);
                }
                let (parameter_index, parameter) = self
                    .function
                    .parameters()
                    .iter()
                    .flatten()
                    .enumerate()
                    .find(|(_, parameter)| parameter.name == name)
                    .ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
                let RuntimeExpressionType::Array { element, count } =
                    runtime_array_type(parameter.ty.text)
                        .ok_or(self.error(CodegenErrorKind::RuntimeTypeMismatch))?
                else {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                };
                if index >= count {
                    return Err(self.error(CodegenErrorKind::ValueOutOfRange));
                }
                let slots = self
                    .local_stack_slots()?
                    .checked_add(self.parameter_stack_slots_from(parameter_index + 1)?)
                    .and_then(|slots| slots.checked_add(count - 1 - index))
                    .and_then(|slots| slots.checked_add(self.evaluation_depth))
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                self.emit_stack_slot(slots)?;
                self.emit_normalize_array_element(element)
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
                self.emit_array_element(tree, then_branch, index, depth + 1)?;
                let end_patch = self.emit_unconditional_forward_branch()?;
                self.patch_forward_branch(else_patch)?;
                self.emit_array_element(tree, else_branch, index, depth + 1)?;
                self.patch_forward_branch(end_patch)
            }
            ExprKind::InlineConst { operand }
            | ExprKind::LoopBreak { operand }
            | ExprKind::Return { operand } => {
                self.emit_array_element(tree, operand, index, depth + 1)
            }
            ExprKind::Sequence { first, then } => {
                self.emit_expression(tree, first, depth + 1)?;
                self.emit_array_element(tree, then, index, depth + 1)
            }
            _ => Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported)),
        }
    }

    fn emit_normalize_array_element(
        &mut self,
        element: RuntimeArrayElementType,
    ) -> Result<(), CodegenError> {
        match element {
            RuntimeArrayElementType::Bool => self.emit(&[0x0f, 0xb6, 0xc0]),
            RuntimeArrayElementType::Integer(_) => self.emit_normalize(),
            RuntimeArrayElementType::Char => self.emit_normalize_width(
                runtime_width("char")
                    .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?,
            ),
            RuntimeArrayElementType::Unit | RuntimeArrayElementType::Default => Ok(()),
        }
    }

    fn emit_runtime_array_index<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: crate::ExprId,
        count: usize,
        depth: usize,
    ) -> Result<(), CodegenError> {
        if count > crate::expression::MAX_ARRAY_ELEMENTS {
            return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
        }
        self.emit_array_to_stack(tree, base, count)?;
        self.emit_expression(tree, index, depth + 1)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        let mut end_patches = [0usize; crate::expression::MAX_ARRAY_ELEMENTS];
        for (candidate, end_patch) in end_patches[..count].iter_mut().enumerate() {
            self.emit_stack_slot(0)?;
            self.emit(&[0x48, 0x83, 0xf8, candidate as u8])?;
            let next = self.emit_forward_branch(0x85)?;
            self.emit_stack_slot(1 + candidate)?;
            *end_patch = self.emit_unconditional_forward_branch()?;
            self.patch_forward_branch(next)?;
        }
        self.emit(&[0x0f, 0x0b])?;
        for patch in end_patches[..count].iter().copied() {
            self.patch_forward_branch(patch)?;
        }
        self.evaluation_depth -= count + 1;
        let bytes = (count + 1)
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        self.emit_stack_cleanup_bytes(bytes)
    }

    fn emit_index_address<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: crate::ExprId,
        depth: usize,
    ) -> Result<(), CodegenError> {
        let base_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            tree,
            base,
            depth + 1,
        )
        .map_err(|kind| self.error(kind))?;
        let (element, count, slice_source, local_name) = match base_type {
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Array { element, count, .. },
                ..
            } => (element, Some(count), None, None),
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Slice(element),
                ..
            } => {
                let source = reference_source_name(tree, base)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                (element, None, Some(source), None)
            }
            RuntimeExpressionType::Array { element, count } => {
                let Some(ExprKind::Identifier(name)) =
                    tree.expression(base).map(|expression| expression.kind)
                else {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                };
                (element, Some(count), None, Some(name))
            }
            _ => return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch)),
        };
        if let Some(name) = local_name {
            self.emit_identifier_address(name)?;
        } else {
            self.emit_expression(tree, base, depth + 1)?;
        }
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        if let Some(source) = slice_source {
            self.emit_slice_length(source)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_expression(tree, index, depth + 1)?;
            self.emit(&[0x48, 0x3b, 0x04, 0x24])?;
            self.emit_trap_branch(0x83)?;
            self.emit(&[0x48, 0x89, 0xc1, 0x48, 0x8b, 0x44, 0x24, 0x08])?;
            self.emit_stack_cleanup_bytes(16)?;
            self.evaluation_depth -= 2;
        } else {
            self.emit_expression(tree, index, depth + 1)?;
            self.emit(&[0x48, 0x83, 0xf8])?;
            self.emit(&[u8::try_from(
                count.ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?,
            )
            .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?])?;
            self.emit_trap_branch(0x83)?;
            self.emit(&[0x48, 0x89, 0xc1, 0x58])?;
            self.evaluation_depth -= 1;
        }
        match runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?
        {
            1 => self.emit(&[0x48, 0x01, 0xc8]),
            2 => self.emit(&[0x48, 0x8d, 0x04, 0x48]),
            4 => self.emit(&[0x48, 0x8d, 0x04, 0x88]),
            8 => self.emit(&[0x48, 0x8d, 0x04, 0xc8]),
            _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
    }

    fn emit_reference_array_constant_index<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        element: RuntimeArrayElementType,
        index: usize,
        layout: RuntimeReferenceArrayLayout,
        depth: usize,
    ) -> Result<(), CodegenError> {
        self.emit_expression(tree, base, depth + 1)?;
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        let stride = if layout == RuntimeReferenceArrayLayout::Native {
            element_bytes
        } else {
            8
        };
        let displacement = stride
            .checked_mul(index)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if displacement != 0 {
            if displacement <= i8::MAX as usize {
                self.emit(&[0x48, 0x83, 0xc0, displacement as u8])?;
            } else {
                self.emit(&[0x48, 0x05])?;
                self.emit(
                    &u32::try_from(displacement)
                        .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?
                        .to_le_bytes(),
                )?;
            }
        }
        if element_bytes == 0 {
            self.emit(&[0x31, 0xc0])
        } else {
            self.emit_reference_load(element)
        }
    }

    fn emit_slice_runtime_index<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: crate::ExprId,
        element: RuntimeArrayElementType,
        source: &str,
        depth: usize,
    ) -> Result<(), CodegenError> {
        self.emit_expression(tree, base, depth + 1)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_slice_length(source)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_expression(tree, index, depth + 1)?;
        self.emit(&[0x48, 0x3b, 0x04, 0x24])?;
        let in_bounds = self.emit_forward_branch(0x82)?;
        self.emit(&[0x0f, 0x0b])?;
        self.patch_forward_branch(in_bounds)?;
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        if element_bytes == 0 {
            self.emit(&[0x48, 0x83, 0xc4, 16, 0x31, 0xc0])?;
            self.evaluation_depth -= 2;
            return Ok(());
        }
        self.emit(&[0x48, 0x89, 0xc1, 0x48, 0x8b, 0x44, 0x24, 0x08])?;
        self.emit(&[0x48, 0x83, 0xc4, 16])?;
        self.evaluation_depth -= 2;
        match element_bytes {
            1 => self.emit(&[0x48, 0x01, 0xc8])?,
            2 => self.emit(&[0x48, 0x8d, 0x04, 0x48])?,
            4 => self.emit(&[0x48, 0x8d, 0x04, 0x88])?,
            8 => self.emit(&[0x48, 0x8d, 0x04, 0xc8])?,
            _ => return Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
        self.emit_reference_load(element)
    }

    fn emit_reference_array_runtime_index<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: crate::ExprId,
        target: RuntimeReferenceTarget,
        depth: usize,
    ) -> Result<(), CodegenError> {
        let RuntimeReferenceTarget::Array {
            element,
            count,
            layout: _,
        } = target
        else {
            return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
        };
        self.emit_expression(tree, base, depth + 1)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_expression(tree, index, depth + 1)?;
        self.emit(&[0x48, 0x83, 0xf8, count as u8])?;
        let in_bounds = self.emit_forward_branch(0x82)?;
        self.emit(&[0x0f, 0x0b])?;
        self.patch_forward_branch(in_bounds)?;
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        if element_bytes == 0 {
            self.emit(&[0x48, 0x83, 0xc4, 0x08, 0x31, 0xc0])?;
            self.evaluation_depth -= 1;
            return Ok(());
        }
        self.emit(&[0x48, 0x89, 0xc1, 0x58])?;
        self.evaluation_depth -= 1;
        match element_bytes {
            1 => self.emit(&[0x48, 0x01, 0xc8])?,
            2 => self.emit(&[0x48, 0x8d, 0x04, 0x48])?,
            4 => self.emit(&[0x48, 0x8d, 0x04, 0x88])?,
            8 => self.emit(&[0x48, 0x8d, 0x04, 0xc8])?,
            _ => return Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
        self.emit_reference_load(element)
    }

    fn emit_constant_array_index<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: usize,
        count: usize,
        _depth: usize,
    ) -> Result<(), CodegenError> {
        if count > crate::expression::MAX_ARRAY_ELEMENTS || index >= count {
            return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
        }
        self.emit_array_to_stack(tree, base, count)?;
        self.emit_stack_slot(index)?;
        self.evaluation_depth -= count;
        let bytes = count
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        self.emit_stack_cleanup_bytes(bytes)
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
            .enumerate()
            .rev()
            .find_map(|(index, local)| local.filter(|local| local.name == name).map(|_| index))
        {
            let local =
                self.locals[index].ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
            if matches!(local.ty, RuntimeExpressionType::Array { .. }) {
                return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
            }
            let slots = self
                .local_stack_slots_from(index + 1)?
                .checked_add(self.evaluation_depth)
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
            let slots = if matches!(
                local.ty,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                }
            ) {
                slots
                    .checked_add(1)
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?
            } else {
                slots
            };
            self.emit_stack_slot(slots)?;
            if local.ty == RuntimeExpressionType::Bool {
                self.emit(&[0x0f, 0xb6, 0xc0])
            } else if matches!(
                local.ty,
                RuntimeExpressionType::Reference { .. } | RuntimeExpressionType::RawPointer { .. }
            ) {
                Ok(())
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
            let parameter_type = self
                .function
                .parameters()
                .iter()
                .flatten()
                .nth(index)
                .map(|parameter| parameter.ty.text)
                .ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
            let slots = self
                .local_stack_slots()?
                .checked_add(self.parameter_stack_slots_from(index + 1)?)
                .and_then(|slots| slots.checked_add(self.evaluation_depth))
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
            let reference_type = runtime_reference_type(parameter_type);
            let pointer_type = runtime_raw_pointer_type(parameter_type);
            let slots = if matches!(
                reference_type,
                Some(RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                })
            ) {
                slots
                    .checked_add(1)
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?
            } else {
                slots
            };
            self.emit_stack_slot(slots)?;
            if parameter_type == "bool" {
                self.emit(&[0x0f, 0xb6, 0xc0])
            } else if reference_type.is_some() || pointer_type.is_some() {
                Ok(())
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

    fn emit_slice_length(&mut self, name: &str) -> Result<(), CodegenError> {
        let slots = if let Some(index) = self.locals[..self.saved_locals]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, local)| {
                local
                    .filter(|local| {
                        local.name == name
                            && matches!(
                                local.ty,
                                RuntimeExpressionType::Reference {
                                    target: RuntimeReferenceTarget::Slice(_)
                                        | RuntimeReferenceTarget::Str,
                                    ..
                                }
                            )
                    })
                    .map(|_| index)
            }) {
            self.local_stack_slots_from(index + 1)?
                .checked_add(self.evaluation_depth)
        } else if let Some(index) =
            self.function
                .parameters()
                .iter()
                .flatten()
                .position(|parameter| {
                    parameter.name == name
                        && matches!(
                            runtime_reference_type(parameter.ty.text),
                            Some(RuntimeExpressionType::Reference {
                                target: RuntimeReferenceTarget::Slice(_)
                                    | RuntimeReferenceTarget::Str,
                                ..
                            })
                        )
                })
        {
            self.local_stack_slots()?
                .checked_add(self.parameter_stack_slots_from(index + 1)?)
                .and_then(|slots| slots.checked_add(self.evaluation_depth))
        } else {
            None
        }
        .ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
        self.emit_stack_slot(slots)
    }

    fn emit_range_slice_to_stack<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        id: crate::ExprId,
        depth: usize,
    ) -> Result<bool, CodegenError> {
        let Some(ExprKind::Unary {
            operator: crate::UnaryOperator::AddressOf | crate::UnaryOperator::AddressOfMut,
            operand,
        }) = tree.expression(id).map(|expression| expression.kind)
        else {
            return Ok(false);
        };
        let Some(ExprKind::RangeIndex {
            base,
            start,
            end,
            inclusive,
        }) = tree.expression(operand).map(|expression| expression.kind)
        else {
            return Ok(false);
        };
        let base_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            tree,
            base,
            depth + 1,
        )
        .map_err(|kind| self.error(kind))?;
        let (element, count, local_name, slice_name, string_slice) = match base_type {
            RuntimeExpressionType::Reference {
                target:
                    RuntimeReferenceTarget::Array {
                        element,
                        count,
                        layout: RuntimeReferenceArrayLayout::Native,
                    },
                ..
            } => (element, Some(count), None, None, false),
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Slice(element),
                ..
            } => {
                let source = reference_source_name(tree, base)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                (element, None, None, Some(source), false)
            }
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Str,
                ..
            } => {
                let source = reference_source_name(tree, base)
                    .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
                (
                    RuntimeArrayElementType::Integer(Some(crate::IntegerType::U8)),
                    None,
                    None,
                    Some(source),
                    true,
                )
            }
            RuntimeExpressionType::Array { element, count } => {
                let Some(ExprKind::Identifier(name)) =
                    tree.expression(base).map(|expression| expression.kind)
                else {
                    return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
                };
                (element, Some(count), Some(name), None, false)
            }
            _ => return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch)),
        };

        if let Some(start) = start {
            self.emit_expression(tree, start, depth + 1)?;
        } else {
            self.emit(&[0x31, 0xc0])?;
        }
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        if let Some(end) = end {
            self.emit_expression(tree, end, depth + 1)?;
        } else if let Some(source) = slice_name {
            self.emit_slice_length(source)?;
        } else {
            let count = u64::try_from(
                count.ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?,
            )
            .map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
            self.emit(&[0x48, 0xb8])?;
            self.emit(&count.to_le_bytes())?;
        }
        if inclusive {
            self.emit(&[0x48, 0x83, 0xc0, 0x01])?;
            self.emit_trap_branch(0x82)?;
        }
        self.emit(&[0x48, 0x8b, 0x0c, 0x24, 0x48, 0x39, 0xc1])?;
        self.emit_trap_branch(0x87)?;
        if let Some(count) = count {
            let count =
                u64::try_from(count).map_err(|_| self.error(CodegenErrorKind::ValueOutOfRange))?;
            self.emit(&[0x48, 0xba])?;
            self.emit(&count.to_le_bytes())?;
        } else {
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_slice_length(
                slice_name.ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?,
            )?;
            self.emit(&[0x48, 0x89, 0xc2, 0x58])?;
            self.evaluation_depth -= 1;
        }
        self.emit(&[0x48, 0x39, 0xd0])?;
        self.emit_trap_branch(0x87)?;
        self.emit(&[0x48, 0x29, 0xc8, 0x50])?;
        self.evaluation_depth += 1;
        if let Some(name) = local_name {
            self.emit_identifier_address(name)?;
        } else {
            self.emit_expression(tree, base, depth + 1)?;
        }
        if string_slice {
            self.emit_string_range_boundary_checks(
                slice_name.ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?,
            )?;
        }
        self.emit(&[0x48, 0x8b, 0x4c, 0x24, 0x08])?;
        match runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?
        {
            0 | 1 => {}
            2 => self.emit(&[0x48, 0xc1, 0xe1, 0x01])?,
            4 => self.emit(&[0x48, 0xc1, 0xe1, 0x02])?,
            8 => self.emit(&[0x48, 0xc1, 0xe1, 0x03])?,
            _ => return Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
        self.emit(&[
            0x48, 0x01, 0xc8, 0x48, 0x89, 0xc2, 0x59, 0x48, 0x83, 0xc4, 0x08, 0x52, 0x51,
        ])?;
        self.evaluation_depth -= 2;
        Ok(true)
    }

    fn emit_string_range_boundary_checks(&mut self, source: &str) -> Result<(), CodegenError> {
        self.emit(&[0x49, 0x89, 0xc0])?;
        self.emit_slice_length(source)?;
        self.emit(&[0x48, 0x89, 0xc2, 0x4c, 0x89, 0xc0])?;
        self.emit(&[0x48, 0x8b, 0x4c, 0x24, 0x08])?;
        self.emit_string_boundary_check()?;
        self.emit(&[0x48, 0x8b, 0x4c, 0x24, 0x08, 0x48, 0x03, 0x0c, 0x24])?;
        self.emit_string_boundary_check()
    }

    fn emit_string_is_char_boundary<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        base: crate::ExprId,
        index: crate::ExprId,
        depth: usize,
    ) -> Result<(), CodegenError> {
        let source = reference_source_name(tree, base)
            .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
        self.emit_expression(tree, base, depth + 1)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_slice_length(source)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_expression(tree, index, depth + 1)?;
        self.emit(&[0x48, 0x3b, 0x04, 0x24])?;
        let out_of_bounds = self.emit_forward_branch(0x87)?;
        let at_end = self.emit_forward_branch(0x84)?;
        self.emit(&[0x48, 0x85, 0xc0])?;
        let at_start = self.emit_forward_branch(0x84)?;
        self.emit(&[0x48, 0x89, 0xc1, 0x48, 0x8b, 0x44, 0x24, 0x08])?;
        self.emit(&[0x0f, 0xb6, 0x04, 0x08, 0x25, 0xc0, 0x00, 0x00, 0x00])?;
        self.emit(&[
            0x3d, 0x80, 0x00, 0x00, 0x00, 0x0f, 0x95, 0xc0, 0x0f, 0xb6, 0xc0,
        ])?;
        let value_end = self.emit_unconditional_forward_branch()?;
        self.patch_forward_branch(at_end)?;
        self.patch_forward_branch(at_start)?;
        self.emit(&[0xb8, 0x01, 0x00, 0x00, 0x00])?;
        let true_end = self.emit_unconditional_forward_branch()?;
        self.patch_forward_branch(out_of_bounds)?;
        self.emit(&[0x31, 0xc0])?;
        self.patch_forward_branch(value_end)?;
        self.patch_forward_branch(true_end)?;
        self.emit_stack_cleanup_bytes(16)?;
        self.evaluation_depth -= 2;
        Ok(())
    }

    fn emit_string_boundary_check(&mut self) -> Result<(), CodegenError> {
        self.emit(&[0x48, 0x85, 0xc9])?;
        let at_start = self.emit_forward_branch(0x84)?;
        self.emit(&[0x48, 0x39, 0xd1])?;
        let at_end = self.emit_forward_branch(0x84)?;
        self.emit(&[0x44, 0x0f, 0xb6, 0x04, 0x08])?;
        self.emit(&[0x41, 0x80, 0xe0, 0xc0, 0x41, 0x80, 0xf8, 0x80])?;
        self.emit_trap_branch(0x84)?;
        self.patch_forward_branch(at_start)?;
        self.patch_forward_branch(at_end)
    }

    fn emit_slice_value_to_stack<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        id: crate::ExprId,
        depth: usize,
    ) -> Result<(), CodegenError> {
        if let Some(ExprKind::If {
            condition,
            then_branch,
            else_branch,
        }) = tree.expression(id).map(|expression| expression.kind)
        {
            self.emit_expression(tree, condition, depth + 1)?;
            self.emit(&[0x48, 0x85, 0xc0])?;
            let else_patch = self.emit_forward_branch(0x84)?;
            self.emit_slice_value_to_stack(tree, then_branch, depth + 1)?;
            let end_patch = self.emit_unconditional_forward_branch()?;
            self.patch_forward_branch(else_patch)?;
            self.emit_slice_value_to_stack(tree, else_branch, depth + 1)?;
            self.patch_forward_branch(end_patch)?;
            return Ok(());
        }
        if self.emit_range_slice_to_stack(tree, id, depth)? {
            return Ok(());
        }
        let source = reference_source_name(tree, id)
            .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
        self.emit_expression(tree, id, depth + 1)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_slice_length(source)?;
        self.evaluation_depth -= 1;
        self.emit(&[0x50])
    }

    fn emit_reference_load(
        &mut self,
        pointee: RuntimeArrayElementType,
    ) -> Result<(), CodegenError> {
        match pointee {
            RuntimeArrayElementType::Bool => self.emit(&[0x0f, 0xb6, 0x00]),
            RuntimeArrayElementType::Char => self.emit(&[0x8b, 0x00]),
            RuntimeArrayElementType::Integer(Some(ty)) => {
                let bits = ty
                    .bits(64)
                    .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
                match (bits, ty.is_signed()) {
                    (8, false) => self.emit(&[0x0f, 0xb6, 0x00]),
                    (8, true) => self.emit(&[0x48, 0x0f, 0xbe, 0x00]),
                    (16, false) => self.emit(&[0x0f, 0xb7, 0x00]),
                    (16, true) => self.emit(&[0x48, 0x0f, 0xbf, 0x00]),
                    (32, false) => self.emit(&[0x8b, 0x00]),
                    (32, true) => self.emit(&[0x48, 0x63, 0x00]),
                    (64, _) => self.emit(&[0x48, 0x8b, 0x00]),
                    _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
                }
            }
            RuntimeArrayElementType::Unit
            | RuntimeArrayElementType::Default
            | RuntimeArrayElementType::Integer(None) => {
                Err(self.error(CodegenErrorKind::UnsupportedRuntimeType))
            }
        }
    }

    fn emit_reference_store(
        &mut self,
        pointee: RuntimeArrayElementType,
    ) -> Result<(), CodegenError> {
        match pointee {
            RuntimeArrayElementType::Bool => self.emit(&[0x88, 0x02]),
            RuntimeArrayElementType::Char => self.emit(&[0x89, 0x02]),
            RuntimeArrayElementType::Integer(Some(ty)) => {
                match ty
                    .bits(64)
                    .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?
                {
                    8 => self.emit(&[0x88, 0x02]),
                    16 => self.emit(&[0x66, 0x89, 0x02]),
                    32 => self.emit(&[0x89, 0x02]),
                    64 => self.emit(&[0x48, 0x89, 0x02]),
                    _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
                }
            }
            RuntimeArrayElementType::Unit
            | RuntimeArrayElementType::Default
            | RuntimeArrayElementType::Integer(None) => {
                Err(self.error(CodegenErrorKind::UnsupportedRuntimeType))
            }
        }
    }

    fn emit_identifier_address(&mut self, name: &str) -> Result<(), CodegenError> {
        let slots = if let Some((index, _local)) = self.locals[..self.saved_locals]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, local)| {
                local
                    .filter(|local| local.name == name)
                    .map(|local| (index, local))
            }) {
            self.local_stack_slots_from(index + 1)?
                .checked_add(self.evaluation_depth)
        } else if let Some(index) = self
            .function
            .parameters()
            .iter()
            .flatten()
            .position(|parameter| parameter.name == name)
        {
            self.local_stack_slots()?
                .checked_add(self.parameter_stack_slots_from(index + 1)?)
                .and_then(|slots| slots.checked_add(self.evaluation_depth))
        } else {
            None
        }
        .ok_or(self.error(CodegenErrorKind::UnknownRuntimeName))?;
        self.emit_stack_slot_address(slots)
    }

    fn emit_stack_slot_address(&mut self, slots: usize) -> Result<(), CodegenError> {
        let displacement = slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if displacement <= i8::MAX as usize {
            self.emit(&[0x48, 0x8d, 0x44, 0x24, displacement as u8])
        } else {
            self.emit(&[0x48, 0x8d, 0x84, 0x24])?;
            self.emit(
                &u32::try_from(displacement)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?
                    .to_le_bytes(),
            )
        }
    }

    fn emit_stack_slot(&mut self, slots: usize) -> Result<(), CodegenError> {
        let displacement = slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if displacement <= i8::MAX as usize {
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
        if displacement <= i8::MAX as usize {
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
        allow_shadowing: bool,
    ) -> Result<(), CodegenError> {
        if self.saved_locals == MAX_PARAMETERS {
            return Err(CodegenError {
                kind: CodegenErrorKind::TooManyRuntimeLocals,
                span: translate_span(body_start, local.name_span),
            });
        }
        if !allow_shadowing
            && (self
                .function
                .parameters()
                .iter()
                .flatten()
                .any(|parameter| parameter.name == local.name)
                || self.locals[..self.saved_locals]
                    .iter()
                    .flatten()
                    .any(|existing| existing.name == local.name))
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
            if let Some(array) = runtime_array_type(ty.text) {
                array
            } else if let Some(reference) = runtime_reference_type(ty.text) {
                reference
            } else if let Some(pointer) = runtime_raw_pointer_type(ty.text) {
                pointer
            } else if ty.text == "()" {
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
                RuntimeExpressionType::Integer(Some(integer_type))
            }
        } else {
            match initializer_type {
                RuntimeExpressionType::Unit => RuntimeExpressionType::Unit,
                RuntimeExpressionType::Default => {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::RuntimeTypeMismatch,
                        span: translate_span(body_start, local.initializer_span),
                    });
                }
                RuntimeExpressionType::Bool => RuntimeExpressionType::Bool,
                RuntimeExpressionType::Char => RuntimeExpressionType::Char,
                RuntimeExpressionType::Integer(Some(ty)) => {
                    RuntimeExpressionType::Integer(Some(ty))
                }
                RuntimeExpressionType::Integer(None) => {
                    RuntimeExpressionType::Integer(crate::IntegerType::from_name(operand_type))
                }
                RuntimeExpressionType::Array { .. }
                | RuntimeExpressionType::Reference { .. }
                | RuntimeExpressionType::RawPointer { .. } => initializer_type,
            }
        };
        if !runtime_types_compatible(initializer_type, local_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, local.initializer_span),
            });
        }
        let stack_slots = runtime_local_stack_slots(local_type);
        let previous_width = self.width;
        self.width = runtime_expression_width(local_type).unwrap_or(previous_width);
        let emission = (|| {
            if let RuntimeExpressionType::Array { element, count } = local_type {
                self.emit_array_to_stack(&tree, tree.root(), count)?;
                self.pack_temporary_array_for_local(element, count)?;
                self.evaluation_depth -= stack_slots;
            } else if matches!(
                local_type,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                }
            ) {
                self.emit_slice_value_to_stack(&tree, tree.root(), 0)?;
            } else {
                self.emit_expression(&tree, tree.root(), 0)?;
                self.emit(&[0x50])?;
            }
            Ok(())
        })();
        self.width = previous_width;
        emission?;
        self.locals[self.saved_locals] = Some(RuntimeLocal {
            name: local.name,
            ty: local_type,
            mutable: local.mutable,
            stack_slots,
        });
        self.saved_locals += 1;
        Ok(())
    }

    fn emit_scoped_local_cleanup(&mut self, checkpoint: usize) -> Result<(), CodegenError> {
        self.emit_stack_cleanup_to(checkpoint)?;
        self.truncate_scoped_locals(checkpoint)
    }

    fn emit_stack_cleanup_to(&mut self, checkpoint: usize) -> Result<(), CodegenError> {
        if checkpoint > self.saved_locals {
            return Err(self.error(CodegenErrorKind::OutputTooSmall));
        }
        let bytes = self
            .local_stack_slots_from(checkpoint)?
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        if bytes != 0 {
            if bytes <= i8::MAX as usize {
                self.emit(&[0x48, 0x83, 0xc4, bytes as u8])?;
            } else {
                let bytes = u32::try_from(bytes)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?;
                self.emit(&[0x48, 0x81, 0xc4])?;
                self.emit(&bytes.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn truncate_scoped_locals(&mut self, checkpoint: usize) -> Result<(), CodegenError> {
        if checkpoint > self.saved_locals {
            return Err(self.error(CodegenErrorKind::OutputTooSmall));
        }
        for local in &mut self.locals[checkpoint..self.saved_locals] {
            *local = None;
        }
        self.saved_locals = checkpoint;
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
        if assignment.dereference {
            return self
                .emit_dereference_assignment::<MAX_EXPRESSION_NODES>(assignment, body_start);
        }
        if assignment.index().is_some() {
            let target_type = self.locals[..self.saved_locals]
                .iter()
                .rev()
                .flatten()
                .find(|local| local.name == assignment.binding_name())
                .map(|local| local.ty)
                .or_else(|| {
                    self.function
                        .parameters()
                        .iter()
                        .flatten()
                        .find(|parameter| parameter.name == assignment.binding_name())
                        .and_then(|parameter| runtime_reference_type(parameter.ty.text))
                });
            if let Some(RuntimeExpressionType::Reference {
                target:
                    RuntimeReferenceTarget::Array {
                        element,
                        count,
                        layout,
                    },
                mutable,
            }) = target_type
            {
                return self.emit_reference_array_assignment::<MAX_EXPRESSION_NODES>(
                    assignment, body_start, element, count, layout, mutable,
                );
            }
            if let Some(RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Slice(element),
                mutable,
            }) = target_type
            {
                return self.emit_slice_assignment::<MAX_EXPRESSION_NODES>(
                    assignment, body_start, element, mutable,
                );
            }
        }
        let index = self.locals[..self.saved_locals]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, local)| {
                local
                    .filter(|local| local.name == assignment.binding_name())
                    .map(|_| index)
            })
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
        if assignment.index().is_some() {
            return self.emit_indexed_assignment::<MAX_EXPRESSION_NODES>(
                assignment, body_start, index, target,
            );
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
        let previous_width = self.width;
        self.width = runtime_expression_width(target.ty).unwrap_or(previous_width);
        let emission = (|| {
            if matches!(
                target.ty,
                RuntimeExpressionType::Reference {
                    target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                    ..
                }
            ) {
                if assignment.operator != crate::AssignmentOperator::Assign {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    });
                }
                self.emit_slice_value_to_stack(&tree, tree.root(), 0)?;
                let target_length_slot = self
                    .local_stack_slots_from(index + 1)?
                    .checked_add(2)
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                self.emit_stack_slot(0)?;
                self.emit_store_stack_slot(target_length_slot)?;
                self.emit_stack_slot(1)?;
                self.emit_store_stack_slot(
                    target_length_slot
                        .checked_add(1)
                        .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?,
                )?;
                return self.emit_stack_cleanup_bytes(16);
            }
            if let RuntimeExpressionType::Array { element, count } = target.ty {
                if assignment.operator != crate::AssignmentOperator::Assign {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    });
                }
                return self.emit_array_assignment(&tree, tree.root(), index, element, count);
            }
            if assignment.operator == crate::AssignmentOperator::Assign {
                self.emit_expression(&tree, tree.root(), 0)?;
            } else {
                self.emit_identifier(assignment.binding_name())?;
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
            self.emit_store_stack_slot(self.local_stack_slots_from(index + 1)?)
        })();
        self.width = previous_width;
        emission
    }

    fn emit_reference_array_assignment<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
        element: RuntimeArrayElementType,
        count: usize,
        layout: RuntimeReferenceArrayLayout,
        mutable: bool,
    ) -> Result<(), CodegenError> {
        if !mutable {
            return Err(CodegenError {
                kind: CodegenErrorKind::ImmutableAssignment,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let element_type = runtime_type_from_array_element(element);
        if element_type == RuntimeExpressionType::Bool
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
        if matches!(
            element_type,
            RuntimeExpressionType::Unit | RuntimeExpressionType::Char
        ) && assignment.operator != crate::AssignmentOperator::Assign
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let previous_width = self.width;
        self.width = runtime_expression_width(element_type).unwrap_or(previous_width);
        let emission = (|| {
            self.emit_indexed_assignment_index::<MAX_NODES>(assignment, body_start, count)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_identifier(assignment.binding_name())?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            let element_bytes = runtime_array_element_bytes(element)
                .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.emit_reference_array_address_from_stack_index(element_bytes, layout, 1)?;
                if element_bytes == 0 {
                    self.emit(&[0x31, 0xc0])?;
                } else {
                    self.emit_reference_load(element)?;
                }
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
            }
            self.emit_indexed_assignment_value::<MAX_NODES>(assignment, body_start, element_type)?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.evaluation_depth -= 1;
                self.emit(&[0x59])?;
                self.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                    CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    },
                )?)?;
            }
            if element_bytes != 0 {
                self.emit(&[0x48, 0x8b, 0x14, 0x24])?;
                self.emit_reference_array_store_address_from_stack_index(element_bytes, layout, 1)?;
                self.emit_reference_store(element)?;
            }
            self.emit(&[0x48, 0x83, 0xc4, 16])?;
            self.evaluation_depth -= 2;
            Ok(())
        })();
        self.width = previous_width;
        emission
    }

    fn emit_slice_assignment<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
        element: RuntimeArrayElementType,
        mutable: bool,
    ) -> Result<(), CodegenError> {
        if !mutable {
            return Err(CodegenError {
                kind: CodegenErrorKind::ImmutableAssignment,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let element_type = runtime_type_from_array_element(element);
        if element_type == RuntimeExpressionType::Bool
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
        if matches!(
            element_type,
            RuntimeExpressionType::Unit | RuntimeExpressionType::Char
        ) && assignment.operator != crate::AssignmentOperator::Assign
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let previous_width = self.width;
        self.width = runtime_expression_width(element_type).unwrap_or(previous_width);
        let emission = (|| {
            self.emit_slice_assignment_index::<MAX_NODES>(assignment, body_start)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_identifier(assignment.binding_name())?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            let element_bytes = runtime_array_element_bytes(element)
                .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.emit_reference_array_address_from_stack_index(
                    element_bytes,
                    RuntimeReferenceArrayLayout::Native,
                    1,
                )?;
                if element_bytes == 0 {
                    self.emit(&[0x31, 0xc0])?;
                } else {
                    self.emit_reference_load(element)?;
                }
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
            }
            self.emit_indexed_assignment_value::<MAX_NODES>(assignment, body_start, element_type)?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.evaluation_depth -= 1;
                self.emit(&[0x59])?;
                self.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                    CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    },
                )?)?;
            }
            if element_bytes != 0 {
                self.emit(&[0x48, 0x8b, 0x14, 0x24])?;
                self.emit_reference_array_store_address_from_stack_index(
                    element_bytes,
                    RuntimeReferenceArrayLayout::Native,
                    1,
                )?;
                self.emit_reference_store(element)?;
            }
            self.emit(&[0x48, 0x83, 0xc4, 16])?;
            self.evaluation_depth -= 2;
            Ok(())
        })();
        self.width = previous_width;
        emission
    }

    fn emit_slice_assignment_index<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        let index_span = assignment.index_span().ok_or(CodegenError {
            kind: CodegenErrorKind::RuntimeExpressionUnsupported,
            span: translate_span(body_start, assignment.name_span),
        })?;
        let tree = assignment
            .parse_index::<MAX_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + index_span.start, error.span),
            })?;
        let index_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &tree,
            tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, index_span),
        })?;
        if !matches!(
            index_type,
            RuntimeExpressionType::Integer(None)
                | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
        ) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, index_span),
            });
        }
        self.emit_expression(&tree, tree.root(), 0)?;
        self.emit(&[0x50])?;
        self.evaluation_depth += 1;
        self.emit_slice_length(assignment.binding_name())?;
        self.emit(&[0x48, 0x8b, 0x0c, 0x24, 0x48, 0x39, 0xc1])?;
        let in_bounds = self.emit_forward_branch(0x82)?;
        self.emit(&[0x0f, 0x0b])?;
        self.patch_forward_branch(in_bounds)?;
        self.emit(&[0x58])?;
        self.evaluation_depth -= 1;
        Ok(())
    }

    fn emit_reference_array_address_from_stack_index(
        &mut self,
        element_bytes: usize,
        _layout: RuntimeReferenceArrayLayout,
        index_slot: u8,
    ) -> Result<(), CodegenError> {
        self.emit(&[0x48, 0x8b, 0x4c, 0x24, index_slot * 8])?;
        match element_bytes {
            0 => Ok(()),
            1 => self.emit(&[0x48, 0x01, 0xc8]),
            2 => self.emit(&[0x48, 0x8d, 0x04, 0x48]),
            4 => self.emit(&[0x48, 0x8d, 0x04, 0x88]),
            8 => self.emit(&[0x48, 0x8d, 0x04, 0xc8]),
            _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
    }

    fn emit_reference_array_store_address_from_stack_index(
        &mut self,
        element_bytes: usize,
        _layout: RuntimeReferenceArrayLayout,
        index_slot: u8,
    ) -> Result<(), CodegenError> {
        self.emit(&[0x48, 0x8b, 0x4c, 0x24, index_slot * 8])?;
        match element_bytes {
            1 => self.emit(&[0x48, 0x01, 0xca]),
            2 => self.emit(&[0x48, 0x8d, 0x14, 0x4a]),
            4 => self.emit(&[0x48, 0x8d, 0x14, 0x8a]),
            8 => self.emit(&[0x48, 0x8d, 0x14, 0xca]),
            _ => Err(self.error(CodegenErrorKind::UnsupportedRuntimeType)),
        }
    }

    fn emit_dereference_assignment<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
    ) -> Result<(), CodegenError> {
        if assignment.index().is_some() {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeExpressionUnsupported,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let name = assignment.binding_name();
        let target_type = self.locals[..self.saved_locals]
            .iter()
            .rev()
            .flatten()
            .find(|local| local.name == name)
            .map(|local| local.ty)
            .or_else(|| {
                self.function
                    .parameters()
                    .iter()
                    .flatten()
                    .find(|parameter| parameter.name == name)
                    .and_then(|parameter| {
                        runtime_reference_type(parameter.ty.text)
                            .or_else(|| runtime_raw_pointer_type(parameter.ty.text))
                    })
            })
            .ok_or(CodegenError {
                kind: CodegenErrorKind::UnknownRuntimeName,
                span: translate_span(body_start, assignment.name_span),
            })?;
        let (pointee, mutable) = match target_type {
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Scalar(pointee),
                mutable,
            }
            | RuntimeExpressionType::RawPointer { pointee, mutable } => (pointee, mutable),
            _ => {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(body_start, assignment.name_span),
                });
            }
        };
        if !mutable {
            return Err(CodegenError {
                kind: CodegenErrorKind::ImmutableAssignment,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let value_type = runtime_type_from_array_element(pointee);
        if value_type == RuntimeExpressionType::Bool
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
        if value_type == RuntimeExpressionType::Char
            && assignment.operator != crate::AssignmentOperator::Assign
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                span: translate_span(body_start, assignment.name_span),
            });
        }

        let previous_width = self.width;
        self.width = runtime_expression_width(value_type).unwrap_or(previous_width);
        let emission = (|| {
            self.emit_identifier(name)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.emit_reference_load(pointee)?;
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
            }
            let tree = assignment
                .parse_value::<MAX_EXPRESSION_NODES>()
                .map_err(|error| CodegenError {
                    kind: CodegenErrorKind::Expression(error.kind),
                    span: translate_span(body_start + assignment.value_span.start, error.span),
                })?;
            let actual_type = runtime_expression_type_with_locals(
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
            if !runtime_types_compatible(actual_type, value_type) {
                return Err(CodegenError {
                    kind: CodegenErrorKind::RuntimeTypeMismatch,
                    span: translate_span(body_start, assignment.value_span),
                });
            }
            self.emit_expression(&tree, tree.root(), 0)?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.evaluation_depth -= 1;
                self.emit(&[0x59])?;
                self.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                    CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    },
                )?)?;
            }
            self.emit(&[0x48, 0x8b, 0x14, 0x24])?;
            self.emit_reference_store(pointee)?;
            self.emit(&[0x48, 0x83, 0xc4, 0x08])?;
            self.evaluation_depth -= 1;
            Ok(())
        })();
        self.width = previous_width;
        emission
    }

    fn emit_indexed_assignment<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
        local_index: usize,
        target: RuntimeLocal<'source>,
    ) -> Result<(), CodegenError> {
        let RuntimeExpressionType::Array { element, count } = target.ty else {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, assignment.name_span),
            });
        };
        let element_type = runtime_type_from_array_element(element);
        if element_type == RuntimeExpressionType::Bool
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
        if matches!(
            element_type,
            RuntimeExpressionType::Unit | RuntimeExpressionType::Char
        ) && assignment.operator != crate::AssignmentOperator::Assign
        {
            return Err(CodegenError {
                kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                span: translate_span(body_start, assignment.name_span),
            });
        }
        let previous_width = self.width;
        self.width = runtime_expression_width(element_type).unwrap_or(previous_width);
        let emission = (|| {
            self.emit_indexed_assignment_index::<MAX_NODES>(assignment, body_start, count)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.emit_array_local_element(local_index, element, count, 0)?;
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
            }
            self.emit_indexed_assignment_value::<MAX_NODES>(assignment, body_start, element_type)?;
            if assignment.operator != crate::AssignmentOperator::Assign {
                self.evaluation_depth -= 1;
                self.emit(&[0x59])?;
                self.emit_binary(assignment_binary_operator(assignment.operator).ok_or(
                    CodegenError {
                        kind: CodegenErrorKind::UnsupportedRuntimeOperator,
                        span: translate_span(body_start, assignment.name_span),
                    },
                )?)?;
            }
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            self.emit_array_local_element_store(local_index, element, count, 1)?;
            self.evaluation_depth -= 2;
            self.emit(&[0x48, 0x83, 0xc4, 16])
        })();
        self.width = previous_width;
        emission
    }

    fn emit_indexed_assignment_index<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
        count: usize,
    ) -> Result<(), CodegenError> {
        let index_span = assignment.index_span().ok_or(CodegenError {
            kind: CodegenErrorKind::RuntimeExpressionUnsupported,
            span: translate_span(body_start, assignment.name_span),
        })?;
        let tree = assignment
            .parse_index::<MAX_NODES>()
            .map_err(|error| CodegenError {
                kind: CodegenErrorKind::Expression(error.kind),
                span: translate_span(body_start + index_span.start, error.span),
            })?;
        let index_type = runtime_expression_type_with_locals(
            self.function,
            self.resolver,
            &self.locals[..self.saved_locals],
            &tree,
            tree.root(),
            0,
        )
        .map_err(|kind| CodegenError {
            kind,
            span: translate_span(body_start, index_span),
        })?;
        if !matches!(
            index_type,
            RuntimeExpressionType::Integer(None)
                | RuntimeExpressionType::Integer(Some(crate::IntegerType::Usize))
        ) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, index_span),
            });
        }
        match tree.evaluate_at(tree.root(), self.resolver) {
            Ok(index) => {
                let index = usize::try_from(index).map_err(|_| CodegenError {
                    kind: CodegenErrorKind::ValueOutOfRange,
                    span: translate_span(body_start, index_span),
                })?;
                if index >= count {
                    return Err(CodegenError {
                        kind: CodegenErrorKind::Execution(ExecutionError::Arithmetic(
                            crate::ConstEvalError::ArrayIndexOutOfBounds,
                        )),
                        span: translate_span(body_start, index_span),
                    });
                }
            }
            Err(crate::ConstEvalError::UnknownIdentifier) => {}
            Err(error) => {
                return Err(CodegenError {
                    kind: CodegenErrorKind::Execution(ExecutionError::Arithmetic(error)),
                    span: translate_span(body_start, index_span),
                });
            }
        }
        self.emit_expression(&tree, tree.root(), 0)
    }

    fn emit_indexed_assignment_value<const MAX_NODES: usize>(
        &mut self,
        assignment: &crate::Assignment<'_>,
        body_start: usize,
        element_type: RuntimeExpressionType,
    ) -> Result<(), CodegenError> {
        let tree = assignment
            .parse_value::<MAX_NODES>()
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
        if !runtime_types_compatible(value_type, element_type) {
            return Err(CodegenError {
                kind: CodegenErrorKind::RuntimeTypeMismatch,
                span: translate_span(body_start, assignment.value_span),
            });
        }
        self.emit_expression(&tree, tree.root(), 0)
    }

    fn emit_array_local_element(
        &mut self,
        local_index: usize,
        element: RuntimeArrayElementType,
        count: usize,
        index_slot: usize,
    ) -> Result<(), CodegenError> {
        let later_slots = self.local_stack_slots_from(local_index + 1)?;
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        if matches!(element_bytes, 1 | 2 | 4) {
            self.emit_stack_slot(index_slot)?;
            self.emit(&[0x48, 0x89, 0xc1])?;
            self.emit_stack_slot_address(self.evaluation_depth + later_slots)?;
            match element_bytes {
                1 => self.emit(&[0x48, 0x01, 0xc8])?,
                2 => self.emit(&[0x48, 0x8d, 0x04, 0x48])?,
                4 => self.emit(&[0x48, 0x8d, 0x04, 0x88])?,
                _ => unreachable!(),
            }
            return self.emit_reference_load(element);
        }
        let mut end_patches = [0usize; crate::expression::MAX_ARRAY_ELEMENTS];
        for (candidate, end_patch) in end_patches[..count].iter_mut().enumerate() {
            self.emit_stack_slot(index_slot)?;
            self.emit(&[0x48, 0x83, 0xf8, candidate as u8])?;
            let next = self.emit_forward_branch(0x85)?;
            self.emit_stack_slot(self.evaluation_depth + later_slots + candidate)?;
            *end_patch = self.emit_unconditional_forward_branch()?;
            self.patch_forward_branch(next)?;
        }
        self.emit(&[0x0f, 0x0b])?;
        for patch in end_patches[..count].iter().copied() {
            self.patch_forward_branch(patch)?;
        }
        Ok(())
    }

    fn emit_array_local_element_store(
        &mut self,
        local_index: usize,
        element: RuntimeArrayElementType,
        count: usize,
        index_slot: usize,
    ) -> Result<(), CodegenError> {
        let later_slots = self.local_stack_slots_from(local_index + 1)?;
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        if matches!(element_bytes, 1 | 2 | 4) {
            self.emit_stack_slot(index_slot)?;
            self.emit(&[0x48, 0x89, 0xc1])?;
            self.emit_stack_slot_address(self.evaluation_depth + later_slots)?;
            self.emit(&[0x48, 0x89, 0xc2])?;
            match element_bytes {
                1 => self.emit(&[0x48, 0x01, 0xca])?,
                2 => self.emit(&[0x48, 0x8d, 0x14, 0x4a])?,
                4 => self.emit(&[0x48, 0x8d, 0x14, 0x8a])?,
                _ => unreachable!(),
            }
            self.emit_stack_slot(0)?;
            return self.emit_reference_store(element);
        }
        let mut end_patches = [0usize; crate::expression::MAX_ARRAY_ELEMENTS];
        for (candidate, end_patch) in end_patches[..count].iter_mut().enumerate() {
            self.emit_stack_slot(index_slot)?;
            self.emit(&[0x48, 0x83, 0xf8, candidate as u8])?;
            let next = self.emit_forward_branch(0x85)?;
            self.emit_stack_slot(0)?;
            self.emit_store_stack_slot(self.evaluation_depth + later_slots + candidate)?;
            *end_patch = self.emit_unconditional_forward_branch()?;
            self.patch_forward_branch(next)?;
        }
        self.emit(&[0x0f, 0x0b])?;
        for patch in end_patches[..count].iter().copied() {
            self.patch_forward_branch(patch)?;
        }
        Ok(())
    }

    fn emit_array_assignment<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        root: crate::ExprId,
        local_index: usize,
        element: RuntimeArrayElementType,
        count: usize,
    ) -> Result<(), CodegenError> {
        if count > crate::expression::MAX_ARRAY_ELEMENTS {
            return Err(self.error(CodegenErrorKind::RuntimeExpressionUnsupported));
        }
        self.emit_array_to_stack(tree, root, count)?;
        let stored_slots = self.pack_temporary_array_for_local(element, count)?;
        let later_slots = self.local_stack_slots_from(local_index + 1)?;
        for slot in 0..stored_slots {
            self.emit_stack_slot(slot)?;
            let destination = stored_slots
                .checked_add(later_slots)
                .and_then(|slots| slots.checked_add(slot))
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
            self.emit_store_stack_slot(destination)?;
        }
        self.evaluation_depth -= stored_slots;
        let bytes = stored_slots
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        self.emit_stack_cleanup_bytes(bytes)?;
        Ok(())
    }

    fn emit_array_to_stack<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        root: crate::ExprId,
        count: usize,
    ) -> Result<(), CodegenError> {
        let expression = tree
            .expression(root)
            .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
        if let ExprKind::ArrayRepeat {
            element,
            count: repeat,
        } = expression.kind
        {
            if repeat != count {
                return Err(self.error(CodegenErrorKind::RuntimeTypeMismatch));
            }
            self.emit_expression(tree, element, 0)?;
            if count == 0 {
                return Ok(());
            }
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
            for _ in 1..count {
                self.emit_stack_slot(0)?;
                self.emit(&[0x50])?;
                self.evaluation_depth += 1;
            }
            return Ok(());
        }
        for element in 0..count {
            self.emit_array_element(tree, root, element, 0)?;
            self.emit(&[0x50])?;
            self.evaluation_depth += 1;
        }
        for left in 0..count / 2 {
            let right = count - 1 - left;
            self.emit_stack_slot(left)?;
            self.emit(&[0x48, 0x89, 0xc2])?;
            self.emit_stack_slot(right)?;
            self.emit_store_stack_slot(left)?;
            self.emit(&[0x48, 0x89, 0xd0])?;
            self.emit_store_stack_slot(right)?;
        }
        Ok(())
    }

    fn pack_temporary_array_for_local(
        &mut self,
        element: RuntimeArrayElementType,
        count: usize,
    ) -> Result<usize, CodegenError> {
        let local_slots =
            runtime_local_stack_slots(RuntimeExpressionType::Array { element, count });
        if local_slots == count {
            return Ok(count);
        }
        let element_bytes = runtime_array_element_bytes(element)
            .ok_or(self.error(CodegenErrorKind::UnsupportedRuntimeType))?;
        for word in (0..local_slots).rev() {
            self.emit_pack_array_word(element, count, element_bytes, word)?;
            let destination = count
                .checked_sub(local_slots)
                .and_then(|slots| slots.checked_add(word))
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
            self.emit_store_stack_slot(destination)?;
        }
        let removed_slots = count - local_slots;
        self.evaluation_depth -= removed_slots;
        self.emit_stack_cleanup_bytes(
            removed_slots
                .checked_mul(8)
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?,
        )?;
        Ok(local_slots)
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
        self.emit_typed_return(&value_tree, value_tree.root(), expected_type)?;
        self.patch_forward_branch(skip_return)
    }

    fn emit_conditional_assignment<const MAX_EXPRESSION_NODES: usize>(
        &mut self,
        conditional: &crate::ConditionalAssignment<'source>,
        expected_type: RuntimeExpressionType,
        operand_type: &str,
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
            let local_checkpoint = self.saved_locals;
            for action in branch.actions().iter().flatten() {
                match action {
                    crate::ConditionalAssignmentAction::Local(local) => {
                        self.emit_local::<MAX_EXPRESSION_NODES>(
                            local,
                            operand_type,
                            body_start,
                            true,
                        )?;
                    }
                    crate::ConditionalAssignmentAction::Assignment(assignment) => {
                        self.emit_assignment::<MAX_EXPRESSION_NODES>(assignment, body_start)?;
                    }
                    crate::ConditionalAssignmentAction::Expression(statement) => {
                        self.emit_expression_statement::<MAX_EXPRESSION_NODES>(
                            statement, body_start,
                        )?;
                    }
                    crate::ConditionalAssignmentAction::Return(return_statement) => {
                        self.emit_return::<MAX_EXPRESSION_NODES>(
                            return_statement,
                            expected_type,
                            body_start,
                        )?;
                    }
                }
            }
            self.emit_scoped_local_cleanup(local_checkpoint)?;
            end_patches[end_count] = Some(self.emit_unconditional_forward_branch()?);
            end_count += 1;
            self.patch_forward_branch(false_branch)?;
        }
        let local_checkpoint = self.saved_locals;
        for action in conditional.else_actions().iter().flatten() {
            match action {
                crate::ConditionalAssignmentAction::Local(local) => {
                    self.emit_local::<MAX_EXPRESSION_NODES>(local, operand_type, body_start, true)?;
                }
                crate::ConditionalAssignmentAction::Assignment(assignment) => {
                    self.emit_assignment::<MAX_EXPRESSION_NODES>(assignment, body_start)?;
                }
                crate::ConditionalAssignmentAction::Expression(statement) => {
                    self.emit_expression_statement::<MAX_EXPRESSION_NODES>(statement, body_start)?;
                }
                crate::ConditionalAssignmentAction::Return(return_statement) => {
                    self.emit_return::<MAX_EXPRESSION_NODES>(
                        return_statement,
                        expected_type,
                        body_start,
                    )?;
                }
            }
        }
        self.emit_scoped_local_cleanup(local_checkpoint)?;
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
        self.emit_typed_return(&value_tree, value_tree.root(), expected_type)
    }

    fn emit_typed_return<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        root: crate::ExprId,
        expected_type: RuntimeExpressionType,
    ) -> Result<(), CodegenError> {
        if matches!(
            expected_type,
            RuntimeExpressionType::Reference {
                target: RuntimeReferenceTarget::Slice(_) | RuntimeReferenceTarget::Str,
                ..
            }
        ) {
            return self.emit_slice_return(tree, root);
        }
        let RuntimeExpressionType::Array { element, count } = expected_type else {
            self.emit_expression(tree, root, 0)?;
            return self.emit_epilogue();
        };
        let (element_bytes, _, words) = runtime_array_abi_layout(expected_type)
            .ok_or(self.error(CodegenErrorKind::UnsupportedReturnType))?;
        self.emit_array_to_stack(tree, root, count)?;
        if words == 0 {
            // Zero-sized aggregates have no result transport. Their elements are
            // still evaluated and materialized above so expression semantics stay
            // identical to non-zero-sized arrays.
        } else if self.uses_sret {
            let pointer_slot = self
                .evaluation_depth
                .checked_add(self.local_stack_slots()?)
                .and_then(|slots| slots.checked_add(self.saved_parameter_slots - 1))
                .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
            self.emit_stack_slot(pointer_slot)?;
            self.emit(&[0x48, 0x89, 0xc2])?;
            for index in 0..count {
                self.emit_stack_slot(index)?;
                let displacement = u8::try_from(index * element_bytes)
                    .map_err(|_| self.error(CodegenErrorKind::OutputTooSmall))?;
                match element_bytes {
                    1 => self.emit(&[0x88, 0x42, displacement])?,
                    2 => self.emit(&[0x66, 0x89, 0x42, displacement])?,
                    4 => self.emit(&[0x89, 0x42, displacement])?,
                    8 if displacement == 0 => self.emit(&[0x48, 0x89, 0x02])?,
                    8 => self.emit(&[0x48, 0x89, 0x42, displacement])?,
                    _ => return Err(self.error(CodegenErrorKind::UnsupportedReturnType)),
                }
            }
            self.emit(&[0x48, 0x89, 0xd0])?;
        } else if words == 1 {
            self.emit_pack_array_word(element, count, element_bytes, 0)?;
        } else if self.abi == X86_64Abi::SystemV && words == 2 {
            self.emit_pack_array_word(element, count, element_bytes, 1)?;
            self.emit(&[0x48, 0x89, 0xc1])?;
            self.emit_pack_array_word(element, count, element_bytes, 0)?;
            self.emit(&[0x48, 0x89, 0xca])?;
        } else {
            return Err(self.error(CodegenErrorKind::UnsupportedReturnType));
        }
        self.evaluation_depth -= count;
        let bytes = count
            .checked_mul(8)
            .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
        self.emit_stack_cleanup_bytes(bytes)?;
        self.emit_epilogue()
    }

    fn emit_slice_return<const MAX_NODES: usize>(
        &mut self,
        tree: &crate::ExpressionTree<'_, MAX_NODES>,
        root: crate::ExprId,
    ) -> Result<(), CodegenError> {
        if let Some(ExprKind::If {
            condition,
            then_branch,
            else_branch,
        }) = tree.expression(root).map(|expression| expression.kind)
        {
            self.emit_expression(tree, condition, 0)?;
            self.emit(&[0x48, 0x85, 0xc0])?;
            let else_patch = self.emit_forward_branch(0x84)?;
            self.emit_slice_return(tree, then_branch)?;
            self.patch_forward_branch(else_patch)?;
            return self.emit_slice_return(tree, else_branch);
        }
        let stacked = self.emit_range_slice_to_stack(tree, root, 0)?;
        if stacked {
            self.emit_stack_slot(1)?;
            self.emit(&[0x48, 0x89, 0xc1])?;
            self.emit_stack_slot(0)?;
        } else {
            let source = reference_source_name(tree, root)
                .ok_or(self.error(CodegenErrorKind::RuntimeExpressionUnsupported))?;
            self.emit_expression(tree, root, 0)?;
            self.emit(&[0x48, 0x89, 0xc1])?;
            self.emit_slice_length(source)?;
        }
        match self.abi {
            X86_64Abi::SystemV => {
                self.emit(&[0x48, 0x89, 0xc2, 0x48, 0x89, 0xc8])?;
            }
            X86_64Abi::Windows => {
                self.emit(&[0x49, 0x89, 0xc0])?;
                let local_slots = self.local_stack_slots()?;
                let pointer_slot = usize::from(stacked)
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_add(self.evaluation_depth))
                    .and_then(|slots| slots.checked_add(local_slots))
                    .and_then(|slots| slots.checked_add(self.saved_parameter_slots - 1))
                    .ok_or(self.error(CodegenErrorKind::OutputTooSmall))?;
                self.emit_stack_slot(pointer_slot)?;
                self.emit(&[0x48, 0x89, 0x08, 0x4c, 0x89, 0x40, 0x08])?;
            }
        }
        if stacked {
            self.emit_stack_cleanup_bytes(16)?;
        }
        self.emit_epilogue()
    }

    fn emit_pack_array_word(
        &mut self,
        _element: RuntimeArrayElementType,
        count: usize,
        element_bytes: usize,
        word: usize,
    ) -> Result<(), CodegenError> {
        self.emit(&[0x31, 0xd2])?;
        for index in 0..count {
            let byte_offset = index * element_bytes;
            if byte_offset / 8 != word {
                continue;
            }
            self.emit_stack_slot(index)?;
            match element_bytes {
                1 => self.emit(&[0x0f, 0xb6, 0xc0])?,
                2 => self.emit(&[0x0f, 0xb7, 0xc0])?,
                4 => self.emit(&[0x89, 0xc0])?,
                8 => {}
                _ => return Err(self.error(CodegenErrorKind::UnsupportedReturnType)),
            }
            let shift = ((byte_offset % 8) * 8) as u8;
            if shift != 0 {
                self.emit(&[0x48, 0xc1, 0xe0, shift])?;
            }
            self.emit(&[0x48, 0x09, 0xc2])?;
        }
        self.emit(&[0x48, 0x89, 0xd0])
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
            self.emit_typed_return(&value_tree, value_tree.root(), expected_type)?;
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
        self.emit_typed_return(&else_tree, else_tree.root(), expected_type)
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
    fn uses_wide_displacements_for_deep_array_stack_slots() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn deep(values: [u64; 8], after: u64) -> u64 { let copied = values; copied[7] + values[0] + after }";
        let module = Parser::new(source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            let code =
                compile_x86_64_function::<_, 2048, 4, 64>(&function, &NoConstants, abi).unwrap();
            assert!(
                code.bytes()
                    .windows(4)
                    .any(|bytes| bytes == [0x48, 0x8b, 0x84, 0x24])
            );
        }

        let packed_source = "#[unsafe(no_mangle)] pub extern \"C\" fn packed(values: [u8; 4], after: u8) -> u8 { values[0] + after }";
        let packed_module = Parser::new(packed_source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(packed_function)) = packed_module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(
                compile_x86_64_function::<_, 512, 4, 32>(&packed_function, &NoConstants, abi,)
                    .is_ok()
            );
        }
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
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64) -> u64 { let mut result = value; if value == 0 { result = 40; value + 1; result += 2; return result; } else if value == 1 { result = 40; 84 / value; result += 2; return result; } else if value == 2 { result = 40; 42 / value; result += 2; return result; } else { result = 40; value + 10; result += 2; return result; } 1 / value }";
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
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(value: u64, select: bool) -> u64 { let mut result = value; if select { result += 1; return false; } result }",
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
    fn emits_scoped_locals_in_conditional_action_blocks() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn choose(value: u64, select: bool) -> u64 { let mut result = value; if select { let mut result: u64 = 40; result += 2; return result; } else { let selected: u64 = 84 / value; selected + 1; result = selected; } result }";
        let module = Parser::new(source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(
                compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, abi,).is_ok()
            );
        }

        let leaking = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(select: bool) -> u64 { let mut result: u64 = 0; if select { let scoped: u64 = 42; result = scoped; } scoped }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = leaking.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::UnknownRuntimeName
        );

        let crowded = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn crowded(select: bool) -> u64 { let one: u64 = 1; let two: u64 = 2; let three: u64 = 3; let four: u64 = 4; if select { let fifth: u64 = 5; return fifth; } one + two + three + four }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::TooManyRuntimeLocals
        );
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
    fn emits_scoped_loop_locals_across_control_edges() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn count(limit: u64, stop: u64) -> u64 { let mut i: u64 = 0; let mut total: u64 = 0; while i < limit { loop { break; } let current: u64 = i + 1; current + 10; i = current; if i % 3 == 0 { let selected: u64 = current; total += selected; } else if i % 2 == 0 { let even: u64 = 2; total += even; } else { let fallback: u64 = 1; total += fallback; } if i % 2 == 0 { let skipped: u64 = current; skipped + 10; continue; } if i == stop { let selected: u64 = current; selected + 20; break; } total + current; } total }";
        let module = Parser::new(source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            let result = compile_x86_64_function::<_, 4096, 4, 64>(&function, &NoConstants, abi);
            assert!(result.is_ok(), "{:#?}", result.err());
        }
        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn count(limit: u64) -> u64 { let mut i: u64 = 0; while i < limit { let current: u64 = i + 1; i = current; continue; } i }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn once() -> u64 { loop { let selected: u64 = 42; selected + 1; break; } 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn return_local() -> u64 { loop { let selected: u64 = 42; return selected; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn conditional_return(value: u64) -> u64 { loop { if value == 0 { let selected: u64 = 42; selected + 1; return selected; } return value; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_unit_return(limit: u64, enter: bool) { let mut i: u64 = 0; while i < limit { while enter { let selected: u64 = i + 1; selected + 10; return; } i += 1; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_conditional_unit_return(limit: u64, stop: u64) { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == stop { return; } } outer += 1; inner = 0; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_conditional_return(limit: u64, stop: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == stop { return outer + 40; } } outer += 1; inner = 0; } outer }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_ordered_returns(limit: u64, first: u64, second: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == first { return outer + 40; } if inner == second { return outer + 50; } } outer += 1; inner = 0; } outer }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_continue_then_return(limit: u64, skip: u64, stop: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { inner += 1; if inner == skip { continue; } if inner == stop { return outer + 40; } } outer += 1; inner = 0; } outer }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_multiple_continues(limit: u64, first: u64, second: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; let mut total: u64 = 0; while outer < limit { while inner < 5 { inner += 1; if inner == first { continue; } if inner == second { continue; } total += inner; } outer += 1; inner = 0; } total }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_multiple_breaks(limit: u64, first: u64, second: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; let mut total: u64 = 0; while outer < limit { while inner < 5 { inner += 1; if inner == first { break; } total += inner; if inner == second { break; } } outer += 1; inner = 0; } total }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_unconditional_continue(limit: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { let selected: u64 = inner + 1; inner = selected; continue; let unreachable: u64 = 10 / 0; unreachable; } outer += 1; inner = 0; } outer }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_conditional_only_loop(enter: bool, stop: u64) -> u64 { let mut inner: u64 = 0; while enter { loop { inner += 1; if inner < stop { continue; } if inner == stop { return inner; } } } 0 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_empty_loop(enter: bool) -> u64 { while enter { loop {} } 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn nested_action_only_loop(enter: bool) -> u64 { while enter { loop { 1; } } 42 }",
        ] {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 1536, 4, 48>(&function, &NoConstants, abi,)
                        .is_ok()
                );
            }
        }

        let unreachable_type_error = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(limit: u64) -> u64 { let mut outer: u64 = 0; let mut inner: u64 = 0; while outer < limit { while inner < 3 { inner += 1; continue; let invalid: bool = 1; invalid; } outer += 1; inner = 0; } outer }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = unreachable_type_error.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 1536, 4, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );

        let empty_inner_loop = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn spin(enter: bool) -> u64 { while enter { loop {} } 42 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = empty_inner_loop.items()[0] else {
            panic!("expected function")
        };
        let machine =
            compile_x86_64_function::<_, 512, 2, 32>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap();
        assert!(machine.bytes().windows(5).any(|instruction| {
            instruction[0] == 0xe9
                && i32::from_le_bytes([
                    instruction[1],
                    instruction[2],
                    instruction[3],
                    instruction[4],
                ]) < 0
        }));

        let leaking = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(limit: u64) -> u64 { let mut i: u64 = 0; while i < limit { let scoped: u64 = i; i += 1; } scoped }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = leaking.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::UnknownRuntimeName
        );

        let crowded = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn crowded(limit: u64) -> u64 { let mut i: u64 = 0; let one: u64 = 1; let two: u64 = 2; let three: u64 = 3; while i < limit { let fifth: u64 = i; i += fifth; } one + two + three }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 1024, 4, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::TooManyRuntimeLocals
        );
    }

    #[test]
    fn emits_empty_while_and_top_level_diverging_loop_backedges() {
        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u64 { while enter {} 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn spin() -> u64 { loop {} }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn spin() -> u64 { loop { 1; } }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let machine =
                    compile_x86_64_function::<_, 512, 2, 32>(&function, &NoConstants, abi).unwrap();
                assert!(machine.bytes().windows(5).any(|instruction| {
                    instruction[0] == 0xe9
                        && i32::from_le_bytes([
                            instruction[1],
                            instruction[2],
                            instruction[3],
                            instruction[4],
                        ]) < 0
                }));
            }
        }
    }

    #[test]
    fn emits_matching_labels_on_statement_loop_controls() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u64, stop: u64) -> u64 { let mut i: u64 = 0; let mut total: u64 = 0; 'count: while i < limit { i += 1; if i % 2 == 0 { continue 'count; } else if i == stop { break 'count; } total += i; } 'once: loop { break 'once; } total }";
        let module = Parser::new(source).parse_module::<2, 4>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(
                compile_x86_64_function::<_, 2048, 4, 64>(&function, &NoConstants, abi).is_ok()
            );
        }
    }

    #[test]
    fn emits_labeled_controls_targeting_inner_and_outer_loops() {
        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u64, stop: u64) -> u64 { let mut outer: u64 = 0; let mut total: u64 = 0; 'outer: while outer < limit { outer += 1; 'inner: loop { let selected: u64 = outer; if selected == 0 { continue 'inner; } if selected % 2 == 0 { continue 'outer; } if selected == stop { break 'outer; } break 'inner; } total += outer; } total }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn probe(limit: u64) -> u64 { let mut outer: u64 = 0; 'outer: while outer < limit { outer += 1; 'inner: loop { continue 'outer; } } outer }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn probe(enter: bool) -> u64 { let mut outer: u64 = 0; 'outer: while enter { outer += 1; 'inner: loop { break 'outer; } } outer }",
        ] {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 3072, 4, 96>(&function, &NoConstants, abi,)
                        .is_ok()
                );
            }
        }
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
        let body = function.parse_body::<4>().unwrap();
        assert_eq!(body.tail_expression, "value");
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi).is_ok());
        }
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
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let first: () = loop { break (); break; }; first; let second: () = loop { break; break (); }; second; 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let first: () = loop { if true { break; } else { break break Default::default(); } }; first; let second: () = loop { if true { break Default::default(); } else { break; } }; second; let third: () = loop { break if true { Default::default() } else { break; }; }; third; 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let integer: u64 = Default::default(); let boolean: bool = Default::default(); integer; if boolean { 1 / 0 } else { integer + 42 } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
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

        let unreachable_mismatch = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { let result = loop { break 1; break false; }; result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = unreachable_mismatch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );

        let unconstrained_default = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { let result = Default::default(); result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = unconstrained_default.items()[0] else {
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
    fn emits_bounded_scalar_array_literal_indexes() {
        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { [13u64, 42, 99][1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u32 { [1u32, 3, 5][2] }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                assert!(
                    compile_x86_64_function::<_, 128, 2, 24>(&function, &NoConstants, abi).is_ok()
                );
            }
        }

        let mixed = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { [1u64, false][0] }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = mixed.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 128, 2, 24>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn emits_contextual_array_defaults_from_loop_breaks() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { (loop { break if true { break Default::default() } else { break [13u64, 14] }; })[0] + (loop { if false { break [1 / 0, 14] } else { break Default::default() } })[1] + (loop { break if false { break Default::default() } else { [42, 43] }; })[1] }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
            assert!(compile_x86_64_function::<_, 256, 2, 64>(&function, &NoConstants, abi).is_ok());
        }
    }

    #[test]
    fn stores_fixed_array_locals_for_scalar_indexes_and_assignment() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let values: [u64; 3] = [13, 42, 99]; values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let before: u64 = input; let values = [13u64, 42, 99]; let after: u64 = 1; before + values[1] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let values: [u64; 2] = Default::default(); values[0] + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let values = [input + 1; 3]; values[0] + values[1] + values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> bool { let values = [select; 2usize]; values[0] & values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64, divisor: u64) -> u64 { let values: [u64; 0] = [input / divisor; 0]; 42 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> u64 { let values: [u64; 2] = loop { break if select { break [13, 14] } else { break Default::default() }; }; values[0] + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> u64 { let choice: bool = select; let values = if choice { [13u64, 14] } else { [20, 22] }; values[0] + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let mut values = [13u64, 14]; values = [20, 22]; values[0] + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let mut values = [13u64, 42]; values = [values[1], values[0]]; values[0] * 100 + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> u64 { let mut values = [1u64, 2]; if select { values = [20, 22]; } else { values = [13, 14]; } values[0] + values[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 64>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let compound = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad() -> u64 { let mut values: [u64; 2] = [1, 2]; values += [3, 4]; values[0] }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = compound.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 128, 2, 32>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err()
                .kind,
            CodegenErrorKind::UnsupportedRuntimeOperator
        );
    }

    #[test]
    fn emits_bounded_runtime_array_indexes() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) -> usize { [13usize, 42, 99][index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) -> usize { let values = [13usize, 42, 99]; values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize, divisor: usize) -> usize { [84 / divisor, 42][index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool, index: usize, divisor: usize) -> usize { (if select { [84 / divisor, 42] } else { [13, 14] })[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) -> bool { let values = [false, true]; values[index] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let code = compile_x86_64_function::<_, 1024, 4, 64>(&function, &NoConstants, abi)
                    .unwrap();
                assert!(code.bytes().windows(2).any(|bytes| bytes == [0x0f, 0x0b]));
            }
        }

        let wrong_index = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn bad(index: u64) -> u64 { [13u64, 42][index] }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = wrong_index.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 256, 2, 32>(&function, &NoConstants, X86_64Abi::Windows)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch
        );
    }

    #[test]
    fn stores_fixed_array_elements_through_constant_and_runtime_indexes() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let mut values = [13u64, 42, 99]; values[1] = 20; values[0] + values[1] + values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) -> usize { let mut values = [13usize, 42, 99]; let after = 1usize; values[index] += after; values[index] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) -> bool { let mut values = [false, true]; values[index] ^= true; values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> char { let mut values = ['a', 'b']; values[0] = 'z'; values[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: usize) { let mut values = [(), ()]; values[index] = (); values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool, index: usize) -> usize { let mut values = [1usize, 2]; if select { values[index] = 20; } else { values[index] = 13; } values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> usize { let mut values = [1usize, 2]; let mut index = 0usize; while index < 2 { values[index] += 1; index += 1; } values[0] + values[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1536, 4, 64>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let invalid = [
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let values = [1u64, 2]; values[0] = 3; values[0] }",
                CodegenErrorKind::ImmutableAssignment,
            ),
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let mut value = 1u64; value[0] = 3; value }",
                CodegenErrorKind::RuntimeTypeMismatch,
            ),
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value(index: u64) -> u64 { let mut values = [1u64, 2]; values[index] = 3; values[0] }",
                CodegenErrorKind::RuntimeTypeMismatch,
            ),
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { let mut values = [1u64, 2]; values[2] = 3; values[0] }",
                CodegenErrorKind::Execution(ExecutionError::Arithmetic(
                    crate::ConstEvalError::ArrayIndexOutOfBounds,
                )),
            ),
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value() -> char { let mut values = ['a', 'b']; values[0] += 'c'; values[0] }",
                CodegenErrorKind::UnsupportedRuntimeOperator,
            ),
        ];
        for (source, expected) in invalid {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 512, 2, 64>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                expected,
                "{source}",
            );
        }
    }

    #[test]
    fn loads_one_word_fixed_array_parameters() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u64; 1], after: u64) -> u64 { values[0] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u64; 1]) -> u64 { let copied = values; copied[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [bool; 1]) -> bool { values[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [char; 1]) -> char { values[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u64; 2], after: u64) -> u64 { values[0] + values[1] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(before: u64, values: [u64; 3], after: u64) -> u64 { before + values[0] + values[1] + values[2] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(a: u64, b: u64, c: u64, d: u64, e: u64, values: [u64; 2], after: u64) -> u64 { a + b + c + d + e + values[0] + values[1] + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u8; 4]) -> u8 { values[0] + values[1] + values[2] + values[3] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u16; 5]) -> u16 { values[0] + values[4] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u32; 2]) -> u32 { values[0] + values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [i16; 3]) -> i16 { values[0] + values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [char; 2]) -> char { values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [bool; 3]) -> bool { values[0] ^ values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u32; 2]) -> u64 { values[0] as u64 }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(a: u64, empty: [u8; 0], b: u64, c: u64) -> u64 { a + b + c }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(a: u64, units: [(); 3], b: u64, c: u64) -> u64 { a + b + c }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 8>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 768, 8, 64>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn loads_immutable_scalar_reference_parameters() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(before: u64, input: &u64, after: u64) -> u64 { let copied = input; before + *copied + after }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &i16) -> i16 { *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &bool) -> bool { *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &char) -> char { *input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 48>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn returns_thin_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u16) -> &u16 { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut i32) -> &mut i32 { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u8; 4]) -> &[u8; 4] { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u16) -> &u16 { let copied: &u16 = input; copied }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u16) -> &u16 { &*input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut u16) -> &mut u16 { &mut *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut u16) -> &u16 { input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 768, 4, 96>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let source =
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u16) -> &mut u16 { input }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch,
        );
    }

    #[test]
    fn returns_slice_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16]) -> &[u16] { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [i32]) -> &mut [i32] { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u8]) -> &[u8] { let copied: &[u8] = input; copied }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16]) -> &[u16] { &input[..] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], start: usize, end: usize) -> &[u16] { &input[start..end] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16], start: usize, end: usize) -> &mut [u16] { &mut input[start..end] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16]) -> &[u16] { input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn creates_and_returns_indexed_element_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], index: usize) -> &u16 { &input[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [i32], index: usize) -> &mut i32 { &mut input[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u8; 4], index: usize) -> &u8 { &input[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16; 4], index: usize) -> &mut u16 { &mut input[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u16, index: usize) -> u16 { let mut values = [input, 40u16]; let selected: &mut u16 = &mut values[index]; *selected += 2; values[index] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], index: usize) -> &mut u16 { &mut input[index] }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );

        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16]) -> &u16 { &input[true] }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch,
        );
    }

    #[test]
    fn returns_conditional_slice_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], select: bool) -> &[u16] { if select { &input[..1] } else { &input[1..] } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16], select: bool) -> &mut [u16] { if select { &mut input[..1] } else { &mut input[1..] } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16], select: bool) -> &[u16] { if select { input } else { &input[1..] } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 3072, 4, 192>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn stores_conditional_slice_references_in_locals() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], select: bool) -> &[u16] { let selected: &[u16] = if select { &input[..1] } else { &input[1..] }; selected }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16], select: bool) -> u16 { let selected: &mut [u16] = if select { &mut input[..1] } else { &mut input[1..] }; selected[0] += 2; selected[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], select: bool) -> usize { let selected = if select { &input[..1] } else { &input[1..] }; selected.len() }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 4096, 4, 224>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn replaces_slice_reference_locals() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16], select: bool) -> &[u16] { let mut selected: &[u16] = &input[..1]; if select { selected = &input[1..]; } selected }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16], select: bool) -> u16 { let mut selected: &mut [u16] = &mut input[..1]; if select { selected = &mut input[1..]; } selected[0] += 2; selected[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16]) -> usize { let mut selected: &[u16] = input; selected = &input[1..]; selected.len() }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 4096, 4, 224>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn supports_string_slice_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> usize { input.len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> bool { input.is_empty() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> &str { input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, select: bool) -> &str { let selected: &str = if select { input } else { input }; selected }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut str) -> &str { input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let source =
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> u8 { input[0] }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch,
        );
    }

    #[test]
    fn creates_utf8_checked_string_ranges() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, start: usize, end: usize) -> &str { &input[start..end] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, end: usize) -> &str { &input[..end] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, start: usize) -> &str { &input[start..] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> &str { &input[..] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, start: usize, end: usize) -> &str { &input[start..=end] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut str, start: usize, end: usize) -> &mut str { &mut input[start..end] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 3072, 4, 192>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn converts_string_slices_to_bytes() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> &[u8] { input.as_bytes() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, index: usize) -> u8 { let bytes: &[u8] = input.as_bytes(); bytes[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> usize { input.as_bytes().len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut str) -> &[u8] { input.as_bytes() }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut str) -> u8 { let bytes: &mut [u8] = input.as_bytes(); bytes[0] }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch,
        );
    }

    #[test]
    fn checks_string_character_boundaries() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, index: usize) -> bool { input.is_char_boundary(index) }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> bool { input.is_char_boundary(0) && input.is_char_boundary(input.len()) }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str, index: usize) -> bool { let copied: &str = input; copied.is_char_boundary(index) }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u8], index: usize) -> bool { input.is_char_boundary(index) }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> bool { input.is_char_boundary(true) }",
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 768, 2, 80>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                CodegenErrorKind::RuntimeTypeMismatch,
            );
        }
    }

    #[test]
    fn exposes_reference_data_pointers() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16]) -> *const u16 { input.as_ptr() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut [u16]) -> *mut u16 { input.as_mut_ptr() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &str) -> *const u8 { input.as_ptr() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *mut u16) -> *const u16 { let copied: *const u16 = input; copied }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        for (source, expected) in [
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &[u16]) -> *mut u16 { input.as_mut_ptr() }",
                CodegenErrorKind::ImmutableAssignment,
            ),
            (
                "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *const u16) -> *mut u16 { input }",
                CodegenErrorKind::RuntimeTypeMismatch,
            ),
        ] {
            let module = Parser::new(source).parse_module::<2, 2>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 768, 2, 80>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                expected,
            );
        }
    }

    #[test]
    fn dereferences_scalar_raw_pointers() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *const u16) -> u16 { *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *mut u16) -> u16 { *input = 42; *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *mut i16) -> i16 { *input += 2; *input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let source =
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: *const u16) { *input = 42; }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 768, 2, 80>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn preserves_typed_and_mutable_scalar_reference_pointers() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u64) -> u64 { let copied: &u64 = input; *copied }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut u64) -> u64 { let copied: &mut u64 = input; *copied }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 48>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn mutates_scalar_reference_pointees() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut u64) -> u64 { *input = 40; *input += 2; *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut i16) -> i16 { let copied: &mut i16 = input; *copied += 2; *copied }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut bool) -> bool { *input ^= true; *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut char) -> char { *input = 'z'; *input }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 48>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u64) -> u64 { *input = 42; *input }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 256, 2, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn creates_and_reborrows_scalar_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let reference: &u64 = &input; *reference }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: i16) -> i16 { let mut local = input; let reference: &mut i16 = &mut local; *reference += 2; local }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &mut u64) -> u64 { let reference: &mut u64 = &mut *input; *reference += 2; *input }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: &u64) -> u64 { let reference: &u64 = &*input; *reference }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 512, 4, 48>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let reference: &mut u64 = &mut input; *reference }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 256, 2, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn indexes_and_mutates_fixed_array_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4], index: usize) -> usize { let copied: &[usize; 4] = values; copied[index] + copied[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize; 4], index: usize) -> usize { let copied: &mut [usize; 4] = &mut *values; copied[index] += 2; values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [i16; 3]) -> i16 { values[1] = -12; values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [bool; 3]) -> bool { values[2] ^= true; values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [char; 2]) -> char { values[1] = 'z'; values[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 768, 4, 64>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[u64; 2]) -> u64 { values[0] = 42; values[0] }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 256, 2, 32>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn takes_references_to_expanded_fixed_array_locals() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: usize, index: usize) -> usize { let values = [input, 10, 20, 30]; let reference: &[usize; 4] = &values; reference[index] + reference[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: usize, index: usize) -> usize { let mut values = [input, 10, 20, 30]; let reference: &mut [usize; 4] = &mut values; let copied: &mut [usize; 4] = &mut *reference; copied[index] += 2; copied[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: i16) -> i16 { let mut values = [input, 0, 0]; let reference: &mut [i16; 3] = &mut values; reference[1] = -12; reference[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: bool) -> bool { let mut values = [input, false, true]; let reference: &mut [bool; 3] = &mut values; reference[1] ^= true; reference[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: char) -> char { let mut values = [input, 'a']; let reference: &mut [char; 2] = &mut values; reference[1] = 'z'; reference[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 8>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1024, 8, 96>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn indexes_and_mutates_slice_references() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize], index: usize) -> usize { let copied: &[usize] = values; copied[index] + copied[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize], index: usize) -> usize { let copied: &mut [usize] = &mut *values; copied[index] += 2; values[index] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize]) -> usize { values.len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize]) -> usize { let copied: &mut [usize] = &mut *values; copied.len() + values.len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize]) -> bool { values.is_empty() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize]) -> bool { let copied: &mut [usize] = &mut *values; copied.is_empty() || values.is_empty() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [i16]) -> i16 { values[1] = -12; values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [bool]) -> bool { values[2] ^= true; values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [char]) -> char { values[1] = 'z'; values[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1024, 4, 96>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[u64]) -> u64 { values[0] = 42; values[0] }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 384, 2, 48>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn creates_slices_from_fixed_array_reference_ranges() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[1..3]; slice.len() + slice[0] + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize; 4]) -> usize { let slice: &mut [usize] = &mut values[1..=2]; slice[1] += 2; values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[..2]; slice.len() + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[2..]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[..]; slice.len() + slice[3] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1024, 4, 128>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }

        for source in [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[3..2]; slice.len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &[usize] = &values[..5]; slice.len() }",
        ] {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            assert_eq!(
                compile_x86_64_function::<_, 512, 4, 96>(
                    &function,
                    &NoConstants,
                    X86_64Abi::Windows,
                )
                .unwrap_err()
                .kind,
                CodegenErrorKind::ValueOutOfRange,
            );
        }

        let module = Parser::new(
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4]) -> usize { let slice: &mut [usize] = &mut values[1..3]; slice.len() }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 4, 96>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::ImmutableAssignment,
        );
    }

    #[test]
    fn creates_slices_from_runtime_fixed_array_ranges() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [usize; 4], start: usize, end: usize) -> usize { let slice: &mut [usize] = &mut values[start..end]; slice[0] += 2; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4], start: usize, end: usize) -> usize { let slice: &[usize] = &values[start..=end]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4], end: usize) -> usize { let slice: &[usize] = &values[..end]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize; 4], start: usize) -> usize { let slice: &[usize] = &values[start..]; slice.len() + slice[0] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1536, 4, 128>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn creates_subslices_from_slice_ranges() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &mut [u16], start: usize, end: usize) -> u16 { let slice: &mut [u16] = &mut values[start..end]; slice[0] += 2; slice[0] + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize], start: usize, end: usize) -> usize { let slice: &[usize] = &values[start..=end]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize], end: usize) -> usize { let slice: &[usize] = &values[..end]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize], start: usize) -> usize { let slice: &[usize] = &values[start..]; slice.len() + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: &[usize]) -> usize { let slice: &[usize] = &values[..]; slice.len() + slice[0] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 4, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn creates_slices_from_contiguous_fixed_array_locals() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: usize, start: usize, end: usize) -> usize { let values = [input, 10, 20, 30]; let slice: &[usize] = &values[start..end]; slice.len() + slice[0] + values[2] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: usize, start: usize, end: usize) -> usize { let mut values = [input, 10, 20, 30]; let slice: &mut [usize] = &mut values[start..=end]; slice[1] += 2; values[2] + slice.len() }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: i64) -> i64 { let values = [input, 10, 20, 30]; let slice: &[i64] = &values[1..3]; slice[0] + slice[1] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 8>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 2048, 8, 160>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn creates_slices_from_packed_fixed_array_locals() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> u8 { let mut values = [input, 10, 20, 30]; let slice: &mut [u8] = &mut values[1..=2]; slice[1] += 2; values[2] + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u16) -> u16 { let mut values = [input, 1000, 2000, 3000, 4000]; let slice: &mut [u16] = &mut values[1..4]; slice[1] += 2; values[2] + slice[0] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u32) -> u32 { let mut values = [input, 100000, 200000]; values = [input, values[1], values[2]]; let slice: &[u32] = &values[..2]; slice[0] + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: bool) -> bool { let mut values = [input, false, true]; let slice: &mut [bool] = &mut values[1..]; slice[0] ^= true; values[1] && slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: char) -> char { let mut values = [input, 'x', 'y']; let slice: &mut [char] = &mut values[1..3]; slice[0] = 'z'; values[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(start: usize, end: usize) -> u8 { let mut values = [7u8, 10, 20, 30]; let slice: &mut [u8] = &mut values[start..end]; slice[0] += 2; values[start] + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(start: usize) -> u8 { let end: usize = start + 2; let mut values = [7u8, 10, 20, 30]; let slice: &mut [u8] = &mut values[start..end]; slice[0] += 2; values[start] + slice[1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(start: usize) -> u8 { let end = start + 2; if end == 3 { 42u8 } else { 1 } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(start: usize) -> u8 { let end: usize = start + 2; if end == 3 { 42u8 } else { 1 } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 8>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 3072, 8, 192>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn keeps_auxiliary_usize_parameters_out_of_narrow_arithmetic() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, index: usize) -> u8 { input + index }";
        let module = Parser::new(source).parse_module::<2, 2>().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            compile_x86_64_function::<_, 512, 2, 64>(&function, &NoConstants, X86_64Abi::Windows,)
                .unwrap_err()
                .kind,
            CodegenErrorKind::RuntimeTypeMismatch,
        );
    }

    #[test]
    fn supports_independent_scalar_integer_widths() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, wide: u64, signed: i16) -> u8 { let adjusted = wide + 2; let negative: i16 = signed + 2; if adjusted == 258 && negative == -1 { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> u8 { let wide: u64 = 256; let signed: i16 = -3; if wide + 2 == 258 && signed + 2 == -1 { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> u8 { let values: [u16; 3] = [256, 257, 258]; if values[1] == 257 { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> u8 { let mut wide: u64 = 256; wide += 2; let mut signed: i16 = -3; signed += 2; let mut values: [u16; 3] = [256, 257, 258]; values[1] += 1; if wide == 258 && signed == -1 && values[1] == 258 { input + 1 } else { input } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1536, 4, 128>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn supports_independent_aggregate_parameter_widths() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, values: [u16; 3]) -> u8 { if values[1] == 257 { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, values: &[u16; 3], slice: &[i32]) -> u8 { if values[1] == 257 && slice[0] == -3 { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, flag: &bool, character: &char) -> u8 { if *flag && *character == 'z' { input + 1 } else { input } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8, wide: &mut u64, values: &mut [u16; 3], slice: &mut [i16]) -> u8 { *wide += 2; values[1] += 1; slice[0] += 2; if *wide == 258 && values[1] == 258 && slice[0] == -1 { input + 1 } else { input } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 1536, 4, 128>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn returns_fixed_arrays_through_x86_64_abi_classes() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> [u64; 1] { [input + 1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: bool) -> [bool; 1] { [input] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: char) -> [char; 1] { [input] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> [u64; 2] { [input, input + 1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64, after: u64) -> [u64; 3] { [input, 42, after] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> [u64; 3] { let values = [input, input + 1, input + 2]; return values; }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> [u64; 3] { if select { return [13u64, 42, 99]; } [1, 2, 3] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(select: bool) -> [u64; 2] { if select { return [13u64, 42]; } else { return [1, 2]; } }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> [u8; 4] { [input, input + 1, input + 2, input + 3] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u16) -> [u16; 5] { [input, 1, 2, 3, input + 4] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u32) -> [u32; 2] { [input, input + 1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: i16) -> [i16; 3] { [input, -2, input + 1] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: char) -> [char; 2] { [input, 'z'] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: bool) -> [bool; 3] { [input, false, true] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> [u8; 0] { [] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u8) -> [(); 3] { [(); 3] }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u16) -> [u16; 5] { if input == 0 { Default::default() } else { [input, 0, 0, 0, input] } }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result = compile_x86_64_function::<_, 768, 4, 64>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
    }

    #[test]
    fn supports_sixteen_element_fixed_arrays() {
        let sources = [
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(values: [u8; 16]) -> [u8; 16] { let mut copied = values; copied[15] += 1; copied }",
            "#[unsafe(no_mangle)] pub extern \"C\" fn value(input: u64) -> u64 { let mut values = [input; 16]; values[15] += 1; values[15] }",
        ];
        for source in sources {
            let module = Parser::new(source).parse_module::<2, 4>().unwrap();
            let Some(Item::Function(function)) = module.items()[0] else {
                panic!("expected function")
            };
            for abi in [X86_64Abi::Windows, X86_64Abi::SystemV] {
                let result =
                    compile_x86_64_function::<_, 4096, 4, 96>(&function, &NoConstants, abi);
                assert!(result.is_ok(), "{source}: {result:?}");
            }
        }
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
