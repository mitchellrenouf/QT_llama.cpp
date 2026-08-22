use crate::semantics::evaluate_scalar_const_function;
use crate::{
    CodegenErrorKind, CodegenOptions, ConstantResolver, ConstantTable, Item, Module, ObjectError,
    ObjectFile, ParseErrorKind, Parser, SemanticErrorKind, Span, TargetLayout, X86_64Abi,
    analyze_constants, compile_x86_64_function_with_options, emit_elf64_x86_64, emit_x86_64_coff,
};

struct PipelineResolver<
    'module,
    'source,
    const MAX_CONSTANTS: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
> {
    module: &'module Module<'source, MAX_ITEMS, MAX_PARAMETERS>,
    constants: &'module ConstantTable<'source, MAX_CONSTANTS>,
    target: TargetLayout,
}

impl<const MAX_CONSTANTS: usize, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>
    ConstantResolver for PipelineResolver<'_, '_, MAX_CONSTANTS, MAX_ITEMS, MAX_PARAMETERS>
{
    fn resolve(&self, name: &str) -> Option<u128> {
        self.constants.resolve(name)
    }

    fn resolve_type(&self, name: &str) -> Option<crate::IntegerType> {
        self.constants.resolve_type(name)
    }

    fn resolves_bool(&self, name: &str) -> bool {
        self.constants.resolves_bool(name)
    }

    fn resolve_call(&self, name: &str, arguments: &[u128]) -> Option<u128> {
        evaluate_scalar_const_function(self.module, self.constants, self.target, name, arguments)
    }

    fn resolve_call_type(&self, name: &str, argument_count: usize) -> Option<crate::IntegerType> {
        self.module
            .items()
            .iter()
            .flatten()
            .find_map(|item| match item {
                Item::Function(function)
                    if function.name == name
                        && function.constant
                        && function.abi == crate::FunctionAbi::Rust
                        && function.parameter_count() == argument_count =>
                {
                    function
                        .return_type
                        .and_then(|return_type| crate::IntegerType::from_name(return_type.text))
                }
                _ => None,
            })
    }

    fn call_resolves_bool(&self, name: &str, argument_count: usize) -> bool {
        self.module.items().iter().flatten().any(|item| {
            matches!(item, Item::Function(function)
                if function.name == name
                    && function.constant
                    && function.abi == crate::FunctionAbi::Rust
                    && function.parameter_count() == argument_count
                    && function.return_type.is_some_and(|return_type| return_type.text == "bool"))
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectFormat {
    Elf64,
    Coff,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompileErrorKind {
    Parse(ParseErrorKind),
    Semantic(SemanticErrorKind),
    FunctionNotFound,
    Codegen(CodegenErrorKind),
    Object(ObjectError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileError {
    pub kind: CompileErrorKind,
    pub span: Option<Span>,
}

impl CompileError {
    pub const fn message(&self) -> &'static str {
        match self.kind {
            CompileErrorKind::Parse(_) => "source parsing failed",
            CompileErrorKind::Semantic(_) => "semantic analysis failed",
            CompileErrorKind::FunctionNotFound => "requested function was not found",
            CompileErrorKind::Codegen(_) => "machine-code generation failed",
            CompileErrorKind::Object(_) => "object emission failed",
        }
    }
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.message())?;
        if let Some(span) = self.span {
            write!(formatter, " at bytes {}..{}", span.start, span.end)?;
        }
        Ok(())
    }
}

pub fn compile_source_function<
    const MAX_OBJECT_BYTES: usize,
    const MAX_CODE_BYTES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
    const MAX_CONSTANTS: usize,
    const MAX_EXPRESSION_NODES: usize,
>(
    source: &str,
    function_name: &str,
    format: ObjectFormat,
    target: TargetLayout,
) -> Result<ObjectFile<MAX_OBJECT_BYTES>, CompileError> {
    compile_source_function_with_options::<
        MAX_OBJECT_BYTES,
        MAX_CODE_BYTES,
        MAX_ITEMS,
        MAX_PARAMETERS,
        MAX_CONSTANTS,
        MAX_EXPRESSION_NODES,
    >(
        source,
        function_name,
        format,
        target,
        CodegenOptions::CHECKED,
    )
}

pub fn compile_source_function_with_options<
    const MAX_OBJECT_BYTES: usize,
    const MAX_CODE_BYTES: usize,
    const MAX_ITEMS: usize,
    const MAX_PARAMETERS: usize,
    const MAX_CONSTANTS: usize,
    const MAX_EXPRESSION_NODES: usize,
>(
    source: &str,
    function_name: &str,
    format: ObjectFormat,
    target: TargetLayout,
    options: CodegenOptions,
) -> Result<ObjectFile<MAX_OBJECT_BYTES>, CompileError> {
    let module = Parser::new(source)
        .parse_module::<MAX_ITEMS, MAX_PARAMETERS>()
        .map_err(|error| CompileError {
            kind: CompileErrorKind::Parse(error.kind),
            span: Some(error.span),
        })?;
    let constants =
        analyze_constants::<MAX_CONSTANTS, MAX_EXPRESSION_NODES, MAX_ITEMS, MAX_PARAMETERS>(
            &module, target,
        )
        .map_err(|error| CompileError {
            kind: CompileErrorKind::Semantic(error.kind),
            span: Some(error.span),
        })?;
    let function = module
        .items()
        .iter()
        .flatten()
        .find_map(|item| match item {
            Item::Function(function) if function.name == function_name => Some(function),
            _ => None,
        })
        .ok_or(CompileError {
            kind: CompileErrorKind::FunctionNotFound,
            span: None,
        })?;
    let abi = match format {
        ObjectFormat::Elf64 => X86_64Abi::SystemV,
        ObjectFormat::Coff => X86_64Abi::Windows,
    };
    let resolver = PipelineResolver {
        module: &module,
        constants: &constants,
        target,
    };
    let code = compile_x86_64_function_with_options::<
        _,
        MAX_CODE_BYTES,
        MAX_PARAMETERS,
        MAX_EXPRESSION_NODES,
    >(function, &resolver, abi, options)
    .map_err(|error| CompileError {
        kind: CompileErrorKind::Codegen(error.kind),
        span: Some(error.span),
    })?;
    match format {
        ObjectFormat::Elf64 => emit_elf64_x86_64(function.name, &code),
        ObjectFormat::Coff => emit_x86_64_coff(function.name, &code),
    }
    .map_err(|error| CompileError {
        kind: CompileErrorKind::Object(error),
        span: Some(function.name_span),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type Artifact = ObjectFile<1024>;

    fn compile(source: &str, name: &str, format: ObjectFormat) -> Result<Artifact, CompileError> {
        compile_source_function::<1024, 256, 8, 4, 8, 32>(
            source,
            name,
            format,
            TargetLayout::X86_64,
        )
    }

    #[test]
    fn compiles_source_to_both_native_object_formats() {
        let source = "const BASE: u64 = 40; #[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { BASE + 2 }";
        let elf = compile(source, "answer", ObjectFormat::Elf64).unwrap();
        assert_eq!(&elf.bytes()[..4], b"\x7fELF");
        let coff = compile(source, "answer", ObjectFormat::Coff).unwrap();
        assert_eq!(&coff.bytes()[..2], &0x8664u16.to_le_bytes());
    }

    #[test]
    fn reports_each_pipeline_boundary() {
        let parse = compile("fn broken(", "broken", ObjectFormat::Elf64).unwrap_err();
        assert!(matches!(parse.kind, CompileErrorKind::Parse(_)));

        let semantic = compile(
            "const VALUE: f32 = 1; #[unsafe(no_mangle)] pub extern \"C\" fn value() -> u64 { 1 }",
            "value",
            ObjectFormat::Elf64,
        )
        .unwrap_err();
        assert!(matches!(semantic.kind, CompileErrorKind::Semantic(_)));

        let missing = compile(
            "#[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { 42 }",
            "other",
            ObjectFormat::Elf64,
        )
        .unwrap_err();
        assert_eq!(missing.kind, CompileErrorKind::FunctionNotFound);

        let codegen = compile(
            "#[unsafe(no_mangle)] pub extern \"C\" fn answer(value: f64) -> f64 { value + 1 }",
            "answer",
            ObjectFormat::Elf64,
        )
        .unwrap_err();
        assert!(matches!(codegen.kind, CompileErrorKind::Codegen(_)));
    }

    #[test]
    fn selects_the_object_formats_native_calling_convention() {
        let source =
            "#[unsafe(no_mangle)] pub extern \"C\" fn identity(value: u64) -> u64 { value }";
        let elf = compile(source, "identity", ObjectFormat::Elf64).unwrap();
        assert_eq!(
            &elf.bytes()[64..75],
            &[0x57, 0x48, 0x8b, 0x44, 0x24, 0, 0x48, 0x83, 0xc4, 8, 0xc3]
        );
        let coff = compile(source, "identity", ObjectFormat::Coff).unwrap();
        assert_eq!(
            &coff.bytes()[60..71],
            &[0x51, 0x48, 0x8b, 0x44, 0x24, 0, 0x48, 0x83, 0xc4, 8, 0xc3]
        );
    }

    #[test]
    fn diagnostics_include_bounded_source_locations() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { 1 + }";
        let error = compile(source, "answer", ObjectFormat::Elf64).unwrap_err();
        let span = error.span.unwrap();
        assert!(span.start <= span.end);
        assert!(span.end <= source.len());
        assert_eq!(error.message(), "machine-code generation failed");
    }
}
