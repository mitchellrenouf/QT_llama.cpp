use crate::{
    ExpressionError, ExpressionParser, ExpressionTree, LexError, Lexer, Span, Token, TokenKind,
};

const MAX_DELIMITER_DEPTH: usize = 64;
const MAX_LOOP_ASSIGNMENTS: usize = 4;
const MAX_LOOP_OPERATIONS: usize = 8;
pub const MAX_CONDITIONAL_LOOP_ACTIONS: usize = 4;
pub const MAX_CONDITIONAL_LOOP_ELSE_ARMS: usize = 4;
pub const MAX_BODY_STATEMENTS: usize = 32;
pub const MAX_BODY_EXPRESSION_STATEMENTS: usize = 8;
pub const MAX_BODY_RETURNS: usize = 8;
pub const MAX_CONDITIONAL_RETURN_BRANCHES: usize = 4;
pub const MAX_BODY_CONDITIONAL_ASSIGNMENTS: usize = 8;
pub const MAX_CONDITIONAL_ASSIGNMENT_BRANCHES: usize = 4;
pub const MAX_CONDITIONAL_BRANCH_ACTIONS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypeRef<'source> {
    pub text: &'source str,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parameter<'source> {
    pub name: &'source str,
    pub span: Span,
    pub ty: TypeRef<'source>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FunctionAbi {
    Rust,
    C,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Function<'source, const MAX_PARAMETERS: usize> {
    pub public: bool,
    pub constant: bool,
    pub abi: FunctionAbi,
    pub no_mangle: bool,
    pub name: &'source str,
    pub name_span: Span,
    parameters: [Option<Parameter<'source>>; MAX_PARAMETERS],
    parameter_count: usize,
    pub return_type: Option<TypeRef<'source>>,
    pub body: Span,
    pub body_expression: &'source str,
    pub body_expression_span: Span,
}

impl<'source, const MAX_PARAMETERS: usize> Function<'source, MAX_PARAMETERS> {
    pub fn parameters(&self) -> &[Option<Parameter<'source>>] {
        &self.parameters[..self.parameter_count]
    }

    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub fn parse_body_expression<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.body_expression).parse()
    }

    pub fn parse_body<const MAX_LOCALS: usize>(
        &self,
    ) -> Result<FunctionBody<'source, MAX_LOCALS>, ParseError> {
        BodyParser::new(self.body_expression).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalBinding<'source> {
    pub mutable: bool,
    pub name: &'source str,
    pub name_span: Span,
    pub ty: Option<TypeRef<'source>>,
    pub initializer: &'source str,
    pub initializer_span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssignmentOperator {
    Assign,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
}

impl AssignmentOperator {
    fn from_text(text: &str) -> Option<Self> {
        Some(match text {
            "=" => Self::Assign,
            "+=" => Self::Add,
            "-=" => Self::Subtract,
            "*=" => Self::Multiply,
            "/=" => Self::Divide,
            "%=" => Self::Remainder,
            "&=" => Self::BitAnd,
            "|=" => Self::BitOr,
            "^=" => Self::BitXor,
            "<<=" => Self::ShiftLeft,
            ">>=" => Self::ShiftRight,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Assignment<'source> {
    pub name: &'source str,
    pub name_span: Span,
    pub operator: AssignmentOperator,
    pub value: &'source str,
    pub value_span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionalAssignmentAction<'source> {
    Local(LocalBinding<'source>),
    Assignment(Assignment<'source>),
    Expression(ExpressionStatement<'source>),
    Return(LoopReturn<'source>),
}

type ConditionalAssignmentBlock<'source> = (
    [Option<ConditionalAssignmentAction<'source>>; MAX_CONDITIONAL_BRANCH_ACTIONS],
    usize,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalAssignment<'source> {
    branches: [Option<ConditionalAssignmentBranch<'source>>; MAX_CONDITIONAL_ASSIGNMENT_BRANCHES],
    branch_count: usize,
    else_actions: [Option<ConditionalAssignmentAction<'source>>; MAX_CONDITIONAL_BRANCH_ACTIONS],
    else_action_count: usize,
}

impl<'source> ConditionalAssignment<'source> {
    pub fn branches(&self) -> &[Option<ConditionalAssignmentBranch<'source>>] {
        &self.branches[..self.branch_count]
    }

    pub const fn branch_count(&self) -> usize {
        self.branch_count
    }

    pub fn else_actions(&self) -> &[Option<ConditionalAssignmentAction<'source>>] {
        &self.else_actions[..self.else_action_count]
    }

    pub const fn else_action_count(&self) -> usize {
        self.else_action_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalAssignmentBranch<'source> {
    pub condition: &'source str,
    pub condition_span: Span,
    actions: [Option<ConditionalAssignmentAction<'source>>; MAX_CONDITIONAL_BRANCH_ACTIONS],
    action_count: usize,
}

impl<'source> ConditionalAssignmentBranch<'source> {
    pub fn actions(&self) -> &[Option<ConditionalAssignmentAction<'source>>] {
        &self.actions[..self.action_count]
    }

    pub const fn action_count(&self) -> usize {
        self.action_count
    }

    pub fn parse_condition<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.condition).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalReturn<'source> {
    pub condition: &'source str,
    pub condition_span: Span,
    pub value: &'source str,
    pub value_span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalReturnElse<'source> {
    branches: [Option<ConditionalReturn<'source>>; MAX_CONDITIONAL_RETURN_BRANCHES],
    branch_count: usize,
    pub else_value: Option<LoopReturn<'source>>,
}

impl<'source> ConditionalReturnElse<'source> {
    pub fn branches(&self) -> &[Option<ConditionalReturn<'source>>] {
        &self.branches[..self.branch_count]
    }

    pub const fn branch_count(&self) -> usize {
        self.branch_count
    }
}

impl<'source> ConditionalReturn<'source> {
    pub fn parse_condition<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.condition).parse()
    }

    pub fn parse_value<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.value).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalLoopControl<'source> {
    pub condition: &'source str,
    pub condition_span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionalLoopAction<'source> {
    Local(LocalBinding<'source>),
    Assignment(Assignment<'source>),
    Expression(ExpressionStatement<'source>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionalLoopTerminal<'source> {
    Break,
    Continue,
    Return(LoopReturn<'source>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalLoopArm<'source> {
    actions: [Option<ConditionalLoopAction<'source>>; MAX_CONDITIONAL_LOOP_ACTIONS],
    action_count: usize,
    pub terminal: Option<ConditionalLoopTerminal<'source>>,
}

impl<'source> ConditionalLoopArm<'source> {
    pub fn actions(&self) -> &[Option<ConditionalLoopAction<'source>>] {
        &self.actions[..self.action_count]
    }

    pub const fn action_count(&self) -> usize {
        self.action_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalLoopBlock<'source> {
    pub condition: &'source str,
    pub condition_span: Span,
    actions: [Option<ConditionalLoopAction<'source>>; MAX_CONDITIONAL_LOOP_ACTIONS],
    action_count: usize,
    pub terminal: Option<ConditionalLoopTerminal<'source>>,
    pub else_arm: Option<usize>,
}

impl<'source> ConditionalLoopBlock<'source> {
    pub fn actions(&self) -> &[Option<ConditionalLoopAction<'source>>] {
        &self.actions[..self.action_count]
    }

    pub const fn action_count(&self) -> usize {
        self.action_count
    }

    pub fn parse_condition<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.condition).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopReturn<'source> {
    pub value: &'source str,
    pub value_span: Span,
}

impl<'source> LoopReturn<'source> {
    pub fn parse_value<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.value).parse()
    }
}

impl<'source> ConditionalLoopControl<'source> {
    pub fn parse_condition<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.condition).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopOperation<'source> {
    Local(LocalBinding<'source>),
    Assignment(Assignment<'source>),
    Expression(ExpressionStatement<'source>),
    Break,
    Continue,
    ConditionalBreak(ConditionalLoopControl<'source>),
    ConditionalContinue(ConditionalLoopControl<'source>),
    ConditionalReturn(ConditionalReturn<'source>),
    ConditionalBlock(usize),
    Return(LoopReturn<'source>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BodyStatement {
    Local(usize),
    Assignment(usize),
    ConditionalReturn(usize),
    Loop(usize),
    Expression(usize),
    Return(usize),
    ConditionalReturnElse(usize),
    ConditionalAssignment(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpressionStatement<'source> {
    pub expression: &'source str,
    pub span: Span,
}

impl<'source> ExpressionStatement<'source> {
    pub fn parse<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.expression).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhileLoop<'source> {
    pub condition: Option<&'source str>,
    pub condition_span: Span,
    operations: [Option<LoopOperation<'source>>; MAX_LOOP_OPERATIONS],
    operation_count: usize,
    assignment_count: usize,
    conditional_blocks: [Option<ConditionalLoopBlock<'source>>; MAX_LOOP_OPERATIONS],
    conditional_block_count: usize,
    conditional_else_arms: [Option<ConditionalLoopArm<'source>>; MAX_CONDITIONAL_LOOP_ELSE_ARMS],
    conditional_else_arm_count: usize,
}

impl<'source> WhileLoop<'source> {
    pub fn parse_condition<const MAX_NODES: usize>(
        &self,
    ) -> Result<Option<ExpressionTree<'source, MAX_NODES>>, ExpressionError> {
        self.condition
            .map(|condition| ExpressionParser::new(condition).parse())
            .transpose()
    }

    pub fn operations(&self) -> &[Option<LoopOperation<'source>>] {
        &self.operations[..self.operation_count]
    }

    pub const fn assignment_count(&self) -> usize {
        self.assignment_count
    }

    pub const fn operation_count(&self) -> usize {
        self.operation_count
    }

    pub fn conditional_blocks(&self) -> &[Option<ConditionalLoopBlock<'source>>] {
        &self.conditional_blocks[..self.conditional_block_count]
    }

    pub const fn conditional_block_count(&self) -> usize {
        self.conditional_block_count
    }

    pub fn conditional_else_arms(&self) -> &[Option<ConditionalLoopArm<'source>>] {
        &self.conditional_else_arms[..self.conditional_else_arm_count]
    }
}

impl<'source> Assignment<'source> {
    pub fn parse_value<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.value).parse()
    }
}

impl<'source> LocalBinding<'source> {
    pub fn parse_initializer<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.initializer).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FunctionBody<'source, const MAX_LOCALS: usize> {
    statements: [Option<BodyStatement>; MAX_BODY_STATEMENTS],
    statement_count: usize,
    locals: [Option<LocalBinding<'source>>; MAX_LOCALS],
    local_count: usize,
    assignments: [Option<Assignment<'source>>; MAX_LOCALS],
    assignment_count: usize,
    conditional_returns: [Option<ConditionalReturn<'source>>; MAX_LOCALS],
    conditional_return_count: usize,
    conditional_return_elses: [Option<ConditionalReturnElse<'source>>; MAX_LOCALS],
    conditional_return_else_count: usize,
    conditional_assignments:
        [Option<ConditionalAssignment<'source>>; MAX_BODY_CONDITIONAL_ASSIGNMENTS],
    conditional_assignment_count: usize,
    while_loops: [Option<WhileLoop<'source>>; MAX_LOCALS],
    while_loop_count: usize,
    expression_statements: [Option<ExpressionStatement<'source>>; MAX_BODY_EXPRESSION_STATEMENTS],
    expression_statement_count: usize,
    returns: [Option<LoopReturn<'source>>; MAX_BODY_RETURNS],
    return_count: usize,
    pub tail_expression: &'source str,
    pub tail_span: Span,
    pub implicit_unit: bool,
    pub tail_diverges: bool,
}

impl<'source, const MAX_LOCALS: usize> FunctionBody<'source, MAX_LOCALS> {
    pub fn statements(&self) -> &[Option<BodyStatement>] {
        &self.statements[..self.statement_count]
    }

    pub const fn statement_count(&self) -> usize {
        self.statement_count
    }

    fn push_statement(&mut self, statement: BodyStatement) -> bool {
        let Some(slot) = self.statements.get_mut(self.statement_count) else {
            return false;
        };
        *slot = Some(statement);
        self.statement_count += 1;
        true
    }

    pub fn locals(&self) -> &[Option<LocalBinding<'source>>] {
        &self.locals[..self.local_count]
    }

    pub const fn local_count(&self) -> usize {
        self.local_count
    }

    pub fn assignments(&self) -> &[Option<Assignment<'source>>] {
        &self.assignments[..self.assignment_count]
    }

    pub const fn assignment_count(&self) -> usize {
        self.assignment_count
    }

    pub fn conditional_returns(&self) -> &[Option<ConditionalReturn<'source>>] {
        &self.conditional_returns[..self.conditional_return_count]
    }

    pub const fn conditional_return_count(&self) -> usize {
        self.conditional_return_count
    }

    pub fn conditional_return_elses(&self) -> &[Option<ConditionalReturnElse<'source>>] {
        &self.conditional_return_elses[..self.conditional_return_else_count]
    }

    pub const fn conditional_return_else_count(&self) -> usize {
        self.conditional_return_else_count
    }

    pub fn conditional_assignments(&self) -> &[Option<ConditionalAssignment<'source>>] {
        &self.conditional_assignments[..self.conditional_assignment_count]
    }

    pub const fn conditional_assignment_count(&self) -> usize {
        self.conditional_assignment_count
    }

    pub fn while_loops(&self) -> &[Option<WhileLoop<'source>>] {
        &self.while_loops[..self.while_loop_count]
    }

    pub const fn while_loop_count(&self) -> usize {
        self.while_loop_count
    }

    pub fn expression_statements(&self) -> &[Option<ExpressionStatement<'source>>] {
        &self.expression_statements[..self.expression_statement_count]
    }

    pub const fn expression_statement_count(&self) -> usize {
        self.expression_statement_count
    }

    pub fn returns(&self) -> &[Option<LoopReturn<'source>>] {
        &self.returns[..self.return_count]
    }

    pub const fn return_count(&self) -> usize {
        self.return_count
    }

    pub fn parse_tail<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.tail_expression).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstItem<'source> {
    pub public: bool,
    pub name: &'source str,
    pub name_span: Span,
    pub ty: TypeRef<'source>,
    pub initializer: &'source str,
    pub initializer_span: Span,
}

impl<'source> ConstItem<'source> {
    pub fn parse_initializer<const MAX_NODES: usize>(
        &self,
    ) -> Result<ExpressionTree<'source, MAX_NODES>, ExpressionError> {
        ExpressionParser::new(self.initializer).parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Item<'source, const MAX_PARAMETERS: usize> {
    Function(Function<'source, MAX_PARAMETERS>),
    Const(ConstItem<'source>),
    Static(ConstItem<'source>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Module<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize> {
    items: [Option<Item<'source, MAX_PARAMETERS>>; MAX_ITEMS],
    item_count: usize,
}

impl<'source, const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>
    Module<'source, MAX_ITEMS, MAX_PARAMETERS>
{
    pub fn items(&self) -> &[Option<Item<'source, MAX_PARAMETERS>>] {
        &self.items[..self.item_count]
    }

    pub const fn item_count(&self) -> usize {
        self.item_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseErrorKind {
    Lexical(LexError),
    ExpectedItem,
    UnsupportedAttribute,
    UnsupportedAbi,
    ExpectedAttributeDelimiter,
    ExpectedIdentifier,
    ExpectedParameterSeparator,
    ExpectedType,
    ExpectedEquals,
    ExpectedInitializer,
    ExpectedSemicolon,
    ExpectedBody,
    UnexpectedClosingDelimiter,
    UnterminatedDelimiter,
    NestingLimitExceeded,
    TooManyItems,
    TooManyParameters,
    TooManyLocals,
    TooManyLoopAssignments,
    TooManyLoopOperations,
    TooManyConditionalLoopActions,
    TooManyConditionalLoopElseArms,
    TooManyExpressionStatements,
    TooManyReturns,
    TooManyConditionalReturnBranches,
    TooManyConditionalAssignments,
    TooManyConditionalAssignmentBranches,
    TooManyConditionalBranchActions,
    ExpectedTailExpression,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}

pub struct Parser<'source> {
    source: &'source str,
    lexer: Lexer<'source>,
    lookahead: Option<Token<'source>>,
}

#[derive(Clone)]
struct BodyParser<'source> {
    source: &'source str,
    lexer: Lexer<'source>,
    lookahead: Option<Token<'source>>,
}

impl<'source> BodyParser<'source> {
    const fn new(source: &'source str) -> Self {
        Self {
            source,
            lexer: Lexer::new(source),
            lookahead: None,
        }
    }

    fn lexical(error: LexError) -> ParseError {
        ParseError {
            kind: ParseErrorKind::Lexical(error),
            span: error.span,
        }
    }

    fn peek(&mut self) -> Result<Option<Token<'source>>, ParseError> {
        if self.lookahead.is_none() {
            self.lookahead = self.lexer.next_token().map_err(Self::lexical)?;
        }
        Ok(self.lookahead)
    }

    fn take(&mut self) -> Result<Option<Token<'source>>, ParseError> {
        if let Some(token) = self.lookahead.take() {
            Ok(Some(token))
        } else {
            self.lexer.next_token().map_err(Self::lexical)
        }
    }

    fn error(&self, kind: ParseErrorKind, token: Option<Token<'source>>) -> ParseError {
        ParseError {
            kind,
            span: token.map_or(
                Span {
                    start: self.source.len(),
                    end: self.source.len(),
                },
                |token| token.span,
            ),
        }
    }

    fn delimited_until(
        &mut self,
        terminator: &str,
        missing: ParseErrorKind,
    ) -> Result<(Span, Token<'source>), ParseError> {
        let mut depth = 0usize;
        let mut first = None;
        let mut end = 0usize;
        loop {
            let token = self.take()?.ok_or_else(|| self.error(missing, None))?;
            if depth == 0 && token.text == terminator {
                let start = first.ok_or_else(|| self.error(missing, Some(token)))?;
                return Ok((Span { start, end }, token));
            }
            match token.kind {
                TokenKind::OpenParen | TokenKind::OpenBrace | TokenKind::OpenBracket => {
                    if depth == MAX_DELIMITER_DEPTH {
                        return Err(self.error(ParseErrorKind::NestingLimitExceeded, Some(token)));
                    }
                    depth = depth.checked_add(1).ok_or_else(|| {
                        self.error(ParseErrorKind::NestingLimitExceeded, Some(token))
                    })?;
                }
                TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        self.error(ParseErrorKind::UnexpectedClosingDelimiter, Some(token))
                    })?;
                }
                _ => {}
            }
            first.get_or_insert(token.span.start);
            end = token.span.end;
        }
    }

    fn return_value(&mut self) -> Result<(&'source str, Span), ParseError> {
        if let Some(semicolon) = self
            .peek()?
            .filter(|token| token.kind == TokenKind::Semicolon)
        {
            self.take()?;
            return Ok((
                "()",
                Span {
                    start: semicolon.span.start,
                    end: semicolon.span.start,
                },
            ));
        }
        let (span, _) = self.delimited_until(";", ParseErrorKind::ExpectedSemicolon)?;
        Ok((&self.source[span.start..span.end], span))
    }

    fn parse_local<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        if !self.peek()?.is_some_and(|token| token.text == "let") {
            return Ok(false);
        }
        let let_token = self
            .peek()?
            .ok_or_else(|| self.error(ParseErrorKind::ExpectedIdentifier, None))?;
        if body.local_count == MAX_LOCALS {
            return Err(self.error(ParseErrorKind::TooManyLocals, Some(let_token)));
        }
        let Some(local) = self.local_record()? else {
            return Ok(false);
        };
        if !body.push_statement(BodyStatement::Local(body.local_count)) {
            return Err(self.error(ParseErrorKind::TooManyLocals, Some(let_token)));
        }
        body.locals[body.local_count] = Some(local);
        body.local_count += 1;
        Ok(true)
    }

    fn local_record(&mut self) -> Result<Option<LocalBinding<'source>>, ParseError> {
        if !self.peek()?.is_some_and(|token| token.text == "let") {
            return Ok(None);
        }
        self.take()?;
        let mutable = self.peek()?.is_some_and(|token| token.text == "mut");
        if mutable {
            self.take()?;
        }
        let name = self.take()?;
        let Some(name) = name.filter(|token| token.kind == TokenKind::Identifier) else {
            return Err(self.error(ParseErrorKind::ExpectedIdentifier, name));
        };
        let ty = if self
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::Colon)
        {
            self.take()?;
            let (ty_span, _) = self.delimited_until("=", ParseErrorKind::ExpectedEquals)?;
            Some(TypeRef {
                text: &self.source[ty_span.start..ty_span.end],
                span: ty_span,
            })
        } else {
            let equals = self.take()?;
            if !equals.is_some_and(|token| token.text == "=") {
                return Err(self.error(ParseErrorKind::ExpectedEquals, equals));
            }
            None
        };
        let (initializer_span, _) = self.delimited_until(";", ParseErrorKind::ExpectedSemicolon)?;
        Ok(Some(LocalBinding {
            mutable,
            name: name.text,
            name_span: name.span,
            ty,
            initializer: &self.source[initializer_span.start..initializer_span.end],
            initializer_span,
        }))
    }

    fn parse_assignment<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        let Some(name) = self.peek()? else {
            return Ok(false);
        };
        if name.kind != TokenKind::Identifier
            || matches!(name.text, "if" | "while" | "loop" | "let")
        {
            return Ok(false);
        }
        let mut probe = self.clone();
        probe.take()?;
        let Some(operator_token) = probe.peek()? else {
            return Ok(false);
        };
        let Some(operator) = AssignmentOperator::from_text(operator_token.text) else {
            return Ok(false);
        };
        probe.take()?;
        if body.assignment_count == MAX_LOCALS {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(name)));
        }
        let value_span = match probe.delimited_until(";", ParseErrorKind::ExpectedSemicolon) {
            Ok((span, _)) => span,
            Err(error) if error.kind == ParseErrorKind::ExpectedSemicolon => return Ok(false),
            Err(error) => return Err(error),
        };
        let assignment = Assignment {
            name: name.text,
            name_span: name.span,
            operator,
            value: &self.source[value_span.start..value_span.end],
            value_span,
        };
        if !body.push_statement(BodyStatement::Assignment(body.assignment_count)) {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(name)));
        }
        body.assignments[body.assignment_count] = Some(assignment);
        body.assignment_count += 1;
        *self = probe;
        Ok(true)
    }

    fn assignment_record(&mut self) -> Result<Option<Assignment<'source>>, ParseError> {
        let Some(name) = self.peek()? else {
            return Ok(None);
        };
        if name.kind != TokenKind::Identifier {
            return Ok(None);
        }
        let mut probe = self.clone();
        probe.take()?;
        let Some(operator_token) = probe.take()? else {
            return Ok(None);
        };
        let Some(operator) = AssignmentOperator::from_text(operator_token.text) else {
            return Ok(None);
        };
        let (value_span, _) = probe.delimited_until(";", ParseErrorKind::ExpectedSemicolon)?;
        *self = probe;
        Ok(Some(Assignment {
            name: name.text,
            name_span: name.span,
            operator,
            value: &self.source[value_span.start..value_span.end],
            value_span,
        }))
    }

    fn expression_statement_record(
        &mut self,
    ) -> Result<Option<ExpressionStatement<'source>>, ParseError> {
        let Some(first) = self.peek()? else {
            return Ok(None);
        };
        if first.kind == TokenKind::Semicolon {
            self.take()?;
            return Ok(Some(ExpressionStatement {
                expression: "",
                span: Span {
                    start: first.span.start,
                    end: first.span.start,
                },
            }));
        }
        if matches!(
            first.text,
            "let" | "while" | "loop" | "return" | "break" | "continue"
        ) {
            return Ok(None);
        }
        let mut probe = self.clone();
        let span = match probe.delimited_until(";", ParseErrorKind::ExpectedSemicolon) {
            Ok((span, _)) => span,
            Err(error)
                if matches!(
                    error.kind,
                    ParseErrorKind::ExpectedSemicolon | ParseErrorKind::UnexpectedClosingDelimiter
                ) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        *self = probe;
        Ok(Some(ExpressionStatement {
            expression: &self.source[span.start..span.end],
            span,
        }))
    }

    fn conditional_loop_arm(
        &mut self,
        assignment_count: &mut usize,
    ) -> Result<ConditionalLoopArm<'source>, ParseError> {
        let mut actions = [None; MAX_CONDITIONAL_LOOP_ACTIONS];
        let mut action_count = 0usize;
        loop {
            let next = self.peek()?;
            if next.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                self.take()?;
                return Ok(ConditionalLoopArm {
                    actions,
                    action_count,
                    terminal: None,
                });
            }
            if next.is_some_and(|token| matches!(token.text, "break" | "continue" | "return")) {
                let control = self
                    .take()?
                    .ok_or_else(|| self.error(ParseErrorKind::ExpectedBody, self.lookahead))?;
                let terminal = if control.text == "return" {
                    let (value, value_span) = self.return_value()?;
                    ConditionalLoopTerminal::Return(LoopReturn { value, value_span })
                } else {
                    let semicolon = self.take()?;
                    if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
                        return Err(self.error(ParseErrorKind::ExpectedSemicolon, semicolon));
                    }
                    if control.text == "break" {
                        ConditionalLoopTerminal::Break
                    } else {
                        ConditionalLoopTerminal::Continue
                    }
                };
                let close = self.take()?;
                if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                    return Err(self.error(ParseErrorKind::UnexpectedClosingDelimiter, close));
                }
                return Ok(ConditionalLoopArm {
                    actions,
                    action_count,
                    terminal: Some(terminal),
                });
            }
            if action_count == MAX_CONDITIONAL_LOOP_ACTIONS {
                return Err(self.error(ParseErrorKind::TooManyConditionalLoopActions, next));
            }
            let action = if let Some(local) = self.local_record()? {
                ConditionalLoopAction::Local(local)
            } else {
                let assignment_name = self.peek()?;
                if let Some(assignment) = self.assignment_record()? {
                    if *assignment_count == MAX_LOOP_ASSIGNMENTS {
                        return Err(
                            self.error(ParseErrorKind::TooManyLoopAssignments, assignment_name)
                        );
                    }
                    *assignment_count += 1;
                    ConditionalLoopAction::Assignment(assignment)
                } else if let Some(expression) = self.expression_statement_record()? {
                    ConditionalLoopAction::Expression(expression)
                } else {
                    return Err(self.error(ParseErrorKind::ExpectedBody, next));
                }
            };
            actions[action_count] = Some(action);
            action_count += 1;
        }
    }

    fn return_record(&mut self) -> Result<Option<LoopReturn<'source>>, ParseError> {
        if !self.peek()?.is_some_and(|token| token.text == "return") {
            return Ok(None);
        }
        let mut probe = self.clone();
        probe.take()?;
        let (value, value_span) = match probe.return_value() {
            Ok(value) => value,
            Err(error) if error.kind == ParseErrorKind::ExpectedSemicolon => return Ok(None),
            Err(error) => return Err(error),
        };
        *self = probe;
        Ok(Some(LoopReturn { value, value_span }))
    }

    fn conditional_assignment_block(
        &mut self,
    ) -> Result<Option<ConditionalAssignmentBlock<'source>>, ParseError> {
        let mut actions = [None; MAX_CONDITIONAL_BRANCH_ACTIONS];
        let mut action_count = 0usize;
        loop {
            let next = self.peek()?;
            if next.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                if action_count == 0 {
                    return Ok(None);
                }
                self.take()?;
                return Ok(Some((actions, action_count)));
            }
            if action_count == MAX_CONDITIONAL_BRANCH_ACTIONS {
                return Err(self.error(ParseErrorKind::TooManyConditionalBranchActions, next));
            }
            let action = if let Some(local) = self.local_record()? {
                ConditionalAssignmentAction::Local(local)
            } else if let Some(assignment) = self.assignment_record()? {
                ConditionalAssignmentAction::Assignment(assignment)
            } else if let Some(return_statement) = self.return_record()? {
                ConditionalAssignmentAction::Return(return_statement)
            } else if let Some(expression) = self.expression_statement_record()? {
                ConditionalAssignmentAction::Expression(expression)
            } else {
                return Ok(None);
            };
            actions[action_count] = Some(action);
            action_count += 1;
        }
    }

    fn parse_conditional_assignment<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        let Some(if_token) = self.peek()?.filter(|token| token.text == "if") else {
            return Ok(false);
        };
        let mut probe = self.clone();
        probe.take()?;
        let mut branches = [None; MAX_CONDITIONAL_ASSIGNMENT_BRANCHES];
        let mut branch_count = 0usize;
        let else_actions;
        let else_action_count;
        loop {
            if branch_count == MAX_CONDITIONAL_ASSIGNMENT_BRANCHES {
                let next = probe.peek()?;
                return Err(probe.error(ParseErrorKind::TooManyConditionalAssignmentBranches, next));
            }
            if probe
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::OpenBrace)
            {
                return Ok(false);
            }
            let (condition_span, _) = probe.delimited_until("{", ParseErrorKind::ExpectedBody)?;
            let Some((actions, action_count)) = probe.conditional_assignment_block()? else {
                return Ok(false);
            };
            branches[branch_count] = Some(ConditionalAssignmentBranch {
                condition: &self.source[condition_span.start..condition_span.end],
                condition_span,
                actions,
                action_count,
            });
            branch_count += 1;
            if !probe.peek()?.is_some_and(|token| token.text == "else") {
                else_actions = [None; MAX_CONDITIONAL_BRANCH_ACTIONS];
                else_action_count = 0;
                break;
            }
            probe.take()?;
            if probe.peek()?.is_some_and(|token| token.text == "if") {
                probe.take()?;
                continue;
            }
            let else_open = probe.take()?;
            if !else_open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Ok(false);
            }
            let Some((actions, action_count)) = probe.conditional_assignment_block()? else {
                return Ok(false);
            };
            else_actions = actions;
            else_action_count = action_count;
            break;
        }
        if body.conditional_assignment_count == MAX_BODY_CONDITIONAL_ASSIGNMENTS {
            return Err(self.error(
                ParseErrorKind::TooManyConditionalAssignments,
                Some(if_token),
            ));
        }
        if !body.push_statement(BodyStatement::ConditionalAssignment(
            body.conditional_assignment_count,
        )) {
            return Err(self.error(
                ParseErrorKind::TooManyConditionalAssignments,
                Some(if_token),
            ));
        }
        body.conditional_assignments[body.conditional_assignment_count] =
            Some(ConditionalAssignment {
                branches,
                branch_count,
                else_actions,
                else_action_count,
            });
        body.conditional_assignment_count += 1;
        *self = probe;
        Ok(true)
    }

    fn parse_conditional_return<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        if !self.peek()?.is_some_and(|token| token.text == "if") {
            return Ok(false);
        }
        let mut probe = self.clone();
        probe.take()?;
        if probe
            .peek()?
            .is_some_and(|token| token.kind == TokenKind::OpenBrace)
        {
            return Ok(false);
        }
        let (condition_span, _) = probe.delimited_until("{", ParseErrorKind::ExpectedBody)?;
        let Some(return_token) = probe.take()? else {
            return Ok(false);
        };
        if return_token.text != "return" {
            return Ok(false);
        }
        let (value, value_span) = probe.return_value()?;
        let close = probe.take()?;
        if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
            return Err(probe.error(ParseErrorKind::UnexpectedClosingDelimiter, close));
        }
        if probe.peek()?.is_some_and(|token| token.text == "else") {
            return Ok(false);
        }
        if body.conditional_return_count == MAX_LOCALS {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(return_token)));
        }
        let conditional_return = ConditionalReturn {
            condition: &self.source[condition_span.start..condition_span.end],
            condition_span,
            value,
            value_span,
        };
        if !body.push_statement(BodyStatement::ConditionalReturn(
            body.conditional_return_count,
        )) {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(return_token)));
        }
        body.conditional_returns[body.conditional_return_count] = Some(conditional_return);
        body.conditional_return_count += 1;
        *self = probe;
        Ok(true)
    }

    fn parse_conditional_return_else<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        if !self.peek()?.is_some_and(|token| token.text == "if") {
            return Ok(false);
        }
        let mut probe = self.clone();
        let if_token = probe.take()?.unwrap();
        let mut branches = [None; MAX_CONDITIONAL_RETURN_BRANCHES];
        let mut branch_count = 0usize;
        let else_value;
        loop {
            if branch_count == MAX_CONDITIONAL_RETURN_BRANCHES {
                let next = probe.peek()?;
                return Err(probe.error(ParseErrorKind::TooManyConditionalReturnBranches, next));
            }
            if probe
                .peek()?
                .is_some_and(|token| token.kind == TokenKind::OpenBrace)
            {
                return Ok(false);
            }
            let (condition_span, _) = probe.delimited_until("{", ParseErrorKind::ExpectedBody)?;
            let Some(_branch_return) = probe.take()?.filter(|token| token.text == "return") else {
                return Ok(false);
            };
            let (value, value_span) = probe.return_value()?;
            let close = probe.take()?;
            if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(probe.error(ParseErrorKind::UnexpectedClosingDelimiter, close));
            }
            branches[branch_count] = Some(ConditionalReturn {
                condition: &self.source[condition_span.start..condition_span.end],
                condition_span,
                value,
                value_span,
            });
            branch_count += 1;
            if !probe.peek()?.is_some_and(|token| token.text == "else") {
                if branch_count == 1 {
                    return Ok(false);
                }
                else_value = None;
                break;
            }
            probe.take()?;
            if probe.peek()?.is_some_and(|token| token.text == "if") {
                probe.take()?;
                continue;
            }
            let else_open = probe.take()?;
            if !else_open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                return Err(probe.error(ParseErrorKind::ExpectedBody, else_open));
            }
            let Some(_else_return) = probe.take()?.filter(|token| token.text == "return") else {
                return Ok(false);
            };
            let (value, value_span) = probe.return_value()?;
            let else_close = probe.take()?;
            if !else_close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                return Err(probe.error(ParseErrorKind::UnexpectedClosingDelimiter, else_close));
            }
            else_value = Some(LoopReturn { value, value_span });
            break;
        }
        if body.conditional_return_else_count == MAX_LOCALS {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(if_token)));
        }
        if !body.push_statement(BodyStatement::ConditionalReturnElse(
            body.conditional_return_else_count,
        )) {
            return Err(probe.error(ParseErrorKind::TooManyLocals, Some(if_token)));
        }
        body.conditional_return_elses[body.conditional_return_else_count] =
            Some(ConditionalReturnElse {
                branches,
                branch_count,
                else_value,
            });
        body.conditional_return_else_count += 1;
        *self = probe;
        Ok(true)
    }

    fn parse_expression_statement<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        let Some(first) = self.peek()? else {
            return Ok(false);
        };
        if first.kind == TokenKind::Semicolon {
            if body.expression_statement_count == MAX_BODY_EXPRESSION_STATEMENTS {
                return Err(self.error(ParseErrorKind::TooManyExpressionStatements, Some(first)));
            }
            self.take()?;
            if !body.push_statement(BodyStatement::Expression(body.expression_statement_count)) {
                return Err(self.error(ParseErrorKind::TooManyExpressionStatements, Some(first)));
            }
            body.expression_statements[body.expression_statement_count] =
                Some(ExpressionStatement {
                    expression: "",
                    span: Span {
                        start: first.span.start,
                        end: first.span.start,
                    },
                });
            body.expression_statement_count += 1;
            return Ok(true);
        }
        if matches!(
            first.text,
            "let" | "while" | "loop" | "return" | "break" | "continue"
        ) {
            return Ok(false);
        }
        if first.kind == TokenKind::Identifier {
            let mut assignment_probe = self.clone();
            assignment_probe.take()?;
            if assignment_probe
                .peek()?
                .is_some_and(|token| AssignmentOperator::from_text(token.text).is_some())
            {
                return Ok(false);
            }
        }
        let mut probe = self.clone();
        let span = match probe.delimited_until(";", ParseErrorKind::ExpectedSemicolon) {
            Ok((span, _)) => span,
            Err(error) if error.kind == ParseErrorKind::ExpectedSemicolon => return Ok(false),
            Err(error) => return Err(error),
        };
        if body.expression_statement_count == MAX_BODY_EXPRESSION_STATEMENTS {
            return Err(self.error(ParseErrorKind::TooManyExpressionStatements, Some(first)));
        }
        if !body.push_statement(BodyStatement::Expression(body.expression_statement_count)) {
            return Err(self.error(ParseErrorKind::TooManyExpressionStatements, Some(first)));
        }
        body.expression_statements[body.expression_statement_count] = Some(ExpressionStatement {
            expression: &self.source[span.start..span.end],
            span,
        });
        body.expression_statement_count += 1;
        *self = probe;
        Ok(true)
    }

    fn parse_return<const MAX_LOCALS: usize>(
        &mut self,
        body: &mut FunctionBody<'source, MAX_LOCALS>,
    ) -> Result<bool, ParseError> {
        let Some(return_token) = self.peek()?.filter(|token| token.text == "return") else {
            return Ok(false);
        };
        let mut probe = self.clone();
        probe.take()?;
        let (value, value_span) = match probe.return_value() {
            Ok(value) => value,
            Err(error) if error.kind == ParseErrorKind::ExpectedSemicolon => return Ok(false),
            Err(error) => return Err(error),
        };
        if body.return_count == MAX_BODY_RETURNS {
            return Err(self.error(ParseErrorKind::TooManyReturns, Some(return_token)));
        }
        if !body.push_statement(BodyStatement::Return(body.return_count)) {
            return Err(self.error(ParseErrorKind::TooManyReturns, Some(return_token)));
        }
        body.returns[body.return_count] = Some(LoopReturn { value, value_span });
        body.return_count += 1;
        *self = probe;
        Ok(true)
    }

    fn parse<const MAX_LOCALS: usize>(
        mut self,
    ) -> Result<FunctionBody<'source, MAX_LOCALS>, ParseError> {
        let mut body = FunctionBody {
            statements: [None; MAX_BODY_STATEMENTS],
            statement_count: 0,
            locals: [None; MAX_LOCALS],
            local_count: 0,
            assignments: [None; MAX_LOCALS],
            assignment_count: 0,
            conditional_returns: [None; MAX_LOCALS],
            conditional_return_count: 0,
            conditional_return_elses: [None; MAX_LOCALS],
            conditional_return_else_count: 0,
            conditional_assignments: [None; MAX_BODY_CONDITIONAL_ASSIGNMENTS],
            conditional_assignment_count: 0,
            while_loops: [None; MAX_LOCALS],
            while_loop_count: 0,
            expression_statements: [None; MAX_BODY_EXPRESSION_STATEMENTS],
            expression_statement_count: 0,
            returns: [None; MAX_BODY_RETURNS],
            return_count: 0,
            tail_expression: "",
            tail_span: Span { start: 0, end: 0 },
            implicit_unit: false,
            tail_diverges: false,
        };
        loop {
            let before = body.statement_count;
            while self.parse_local(&mut body)? {}
            while self.parse_assignment(&mut body)? {}
            while self.parse_conditional_return_else(&mut body)? {}
            while self.parse_conditional_return(&mut body)? {}
            while self.parse_conditional_assignment(&mut body)? {}
            while self.parse_return(&mut body)? {}
            while self.parse_expression_statement(&mut body)? {}
            if body.statement_count == before {
                break;
            }
        }
        let mut tail = self.peek()?;
        while self
            .peek()?
            .is_some_and(|token| matches!(token.text, "while" | "loop"))
        {
            let mut probe = self.clone();
            let loop_token = probe
                .take()?
                .ok_or_else(|| probe.error(ParseErrorKind::ExpectedBody, probe.lookahead))?;
            let condition_span = if loop_token.text == "while" {
                probe.delimited_until("{", ParseErrorKind::ExpectedBody)?.0
            } else {
                let open = probe.take()?;
                if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                    return Err(probe.error(ParseErrorKind::ExpectedBody, open));
                }
                Span {
                    start: loop_token.span.end,
                    end: loop_token.span.end,
                }
            };
            let mut operations = [None; MAX_LOOP_OPERATIONS];
            let mut operation_count = 0usize;
            let mut assignment_count = 0usize;
            let mut conditional_blocks = [None; MAX_LOOP_OPERATIONS];
            let mut conditional_block_count = 0usize;
            let mut conditional_else_arms = [None; MAX_CONDITIONAL_LOOP_ELSE_ARMS];
            let mut conditional_else_arm_count = 0usize;
            let mut has_break = false;
            loop {
                let next = probe.peek()?;
                if next.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                    probe.take()?;
                    break;
                }
                if operation_count == MAX_LOOP_OPERATIONS {
                    return Err(probe.error(ParseErrorKind::TooManyLoopOperations, next));
                }
                if next.is_some_and(|token| token.text == "if") {
                    probe.take()?;
                    let (condition_span, _) =
                        probe.delimited_until("{", ParseErrorKind::ExpectedBody)?;
                    let mut actions = [None; MAX_CONDITIONAL_LOOP_ACTIONS];
                    let mut action_count = 0usize;
                    let (control, terminal) = loop {
                        let next = probe.peek()?;
                        if next.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                            probe.take()?;
                            break (None, None);
                        }
                        if next.is_some_and(|token| {
                            matches!(token.text, "break" | "continue" | "return")
                        }) {
                            let control = probe.take()?.ok_or_else(|| {
                                probe.error(ParseErrorKind::ExpectedBody, probe.lookahead)
                            })?;
                            let terminal = if control.text == "return" {
                                let (value, value_span) = probe.return_value()?;
                                ConditionalLoopTerminal::Return(LoopReturn { value, value_span })
                            } else {
                                let semicolon = probe.take()?;
                                if !semicolon
                                    .is_some_and(|token| token.kind == TokenKind::Semicolon)
                                {
                                    return Err(
                                        probe.error(ParseErrorKind::ExpectedSemicolon, semicolon)
                                    );
                                }
                                if control.text == "break" {
                                    ConditionalLoopTerminal::Break
                                } else {
                                    ConditionalLoopTerminal::Continue
                                }
                            };
                            break (Some(control), Some(terminal));
                        }
                        if action_count == MAX_CONDITIONAL_LOOP_ACTIONS {
                            return Err(
                                probe.error(ParseErrorKind::TooManyConditionalLoopActions, next)
                            );
                        }
                        let action = if let Some(local) = probe.local_record()? {
                            ConditionalLoopAction::Local(local)
                        } else {
                            let assignment_name = probe.peek()?;
                            if let Some(assignment) = probe.assignment_record()? {
                                if assignment_count == MAX_LOOP_ASSIGNMENTS {
                                    return Err(probe.error(
                                        ParseErrorKind::TooManyLoopAssignments,
                                        assignment_name,
                                    ));
                                }
                                assignment_count += 1;
                                ConditionalLoopAction::Assignment(assignment)
                            } else if let Some(expression) = probe.expression_statement_record()? {
                                ConditionalLoopAction::Expression(expression)
                            } else {
                                return Err(probe.error(ParseErrorKind::ExpectedBody, next));
                            }
                        };
                        actions[action_count] = Some(action);
                        action_count += 1;
                    };
                    if control.is_some() {
                        let close = probe.take()?;
                        if !close.is_some_and(|token| token.kind == TokenKind::CloseBrace) {
                            return Err(
                                probe.error(ParseErrorKind::UnexpectedClosingDelimiter, close)
                            );
                        }
                    }
                    let else_arm_record = if probe.peek()?.is_some_and(|token| token.text == "else")
                    {
                        probe.take()?;
                        let open = probe.take()?;
                        if !open.is_some_and(|token| token.kind == TokenKind::OpenBrace) {
                            return Err(probe.error(ParseErrorKind::ExpectedBody, open));
                        }
                        Some(probe.conditional_loop_arm(&mut assignment_count)?)
                    } else {
                        None
                    };
                    let conditional = ConditionalLoopControl {
                        condition: &self.source[condition_span.start..condition_span.end],
                        condition_span,
                    };
                    has_break |=
                        control.is_some_and(|control| matches!(control.text, "break" | "return"));
                    has_break |= else_arm_record.is_some_and(|arm| {
                        matches!(
                            arm.terminal,
                            Some(
                                ConditionalLoopTerminal::Break | ConditionalLoopTerminal::Return(_)
                            )
                        )
                    });
                    let else_arm = if let Some(arm) = else_arm_record {
                        if conditional_else_arm_count == MAX_CONDITIONAL_LOOP_ELSE_ARMS {
                            return Err(probe.error(
                                ParseErrorKind::TooManyConditionalLoopElseArms,
                                probe.lookahead,
                            ));
                        }
                        conditional_else_arms[conditional_else_arm_count] = Some(arm);
                        let index = conditional_else_arm_count;
                        conditional_else_arm_count += 1;
                        Some(index)
                    } else {
                        None
                    };
                    operations[operation_count] =
                        Some(if action_count != 0 || else_arm.is_some() {
                            conditional_blocks[conditional_block_count] =
                                Some(ConditionalLoopBlock {
                                    condition: conditional.condition,
                                    condition_span: conditional.condition_span,
                                    actions,
                                    action_count,
                                    terminal,
                                    else_arm,
                                });
                            let index = conditional_block_count;
                            conditional_block_count += 1;
                            LoopOperation::ConditionalBlock(index)
                        } else if control.is_some_and(|control| control.text == "break") {
                            LoopOperation::ConditionalBreak(conditional)
                        } else if control.is_some_and(|control| control.text == "continue") {
                            LoopOperation::ConditionalContinue(conditional)
                        } else {
                            let Some(ConditionalLoopTerminal::Return(LoopReturn {
                                value,
                                value_span,
                            })) = terminal
                            else {
                                return Err(
                                    probe.error(ParseErrorKind::ExpectedInitializer, control)
                                );
                            };
                            LoopOperation::ConditionalReturn(ConditionalReturn {
                                condition: conditional.condition,
                                condition_span: conditional.condition_span,
                                value,
                                value_span,
                            })
                        });
                    operation_count += 1;
                    continue;
                }
                if next.is_some_and(|token| matches!(token.text, "break" | "continue" | "return")) {
                    let control = probe.take()?.ok_or_else(|| {
                        probe.error(ParseErrorKind::ExpectedBody, probe.lookahead)
                    })?;
                    if control.text == "return" {
                        let (value, value_span) = probe.return_value()?;
                        has_break = true;
                        operations[operation_count] =
                            Some(LoopOperation::Return(LoopReturn { value, value_span }));
                    } else {
                        let semicolon = probe.take()?;
                        if !semicolon.is_some_and(|token| token.kind == TokenKind::Semicolon) {
                            return Err(probe.error(ParseErrorKind::ExpectedSemicolon, semicolon));
                        }
                        has_break |= control.text == "break";
                        operations[operation_count] = Some(if control.text == "break" {
                            LoopOperation::Break
                        } else {
                            LoopOperation::Continue
                        });
                    }
                    operation_count += 1;
                    continue;
                }
                if let Some(local) = probe.local_record()? {
                    operations[operation_count] = Some(LoopOperation::Local(local));
                    operation_count += 1;
                    continue;
                }
                let assignment_name = probe.peek()?;
                if let Some(assignment) = probe.assignment_record()? {
                    if assignment_count == MAX_LOOP_ASSIGNMENTS {
                        return Err(
                            probe.error(ParseErrorKind::TooManyLoopAssignments, assignment_name)
                        );
                    }
                    operations[operation_count] = Some(LoopOperation::Assignment(assignment));
                    operation_count += 1;
                    assignment_count += 1;
                    continue;
                }
                if let Some(expression) = probe.expression_statement_record()? {
                    operations[operation_count] = Some(LoopOperation::Expression(expression));
                    operation_count += 1;
                    continue;
                }
                let name = probe.peek()?;
                return Err(probe.error(ParseErrorKind::ExpectedIdentifier, name));
            }
            if operation_count == 0 {
                return Err(probe.error(ParseErrorKind::ExpectedBody, Some(loop_token)));
            }
            if loop_token.text == "loop" && !has_break {
                return Err(probe.error(ParseErrorKind::ExpectedBody, Some(loop_token)));
            }
            if body.while_loop_count == MAX_LOCALS {
                return Err(probe.error(ParseErrorKind::TooManyLocals, Some(loop_token)));
            }
            let loop_statement = WhileLoop {
                condition: (loop_token.text == "while")
                    .then_some(&self.source[condition_span.start..condition_span.end]),
                condition_span,
                operations,
                operation_count,
                assignment_count,
                conditional_blocks,
                conditional_block_count,
                conditional_else_arms,
                conditional_else_arm_count,
            };
            if !body.push_statement(BodyStatement::Loop(body.while_loop_count)) {
                return Err(probe.error(ParseErrorKind::TooManyLocals, Some(loop_token)));
            }
            body.while_loops[body.while_loop_count] = Some(loop_statement);
            body.while_loop_count += 1;
            self = probe;
        }
        if body.while_loop_count != 0 {
            loop {
                let before = body.statement_count;
                while self.parse_local(&mut body)? {}
                while self.parse_assignment(&mut body)? {}
                while self.parse_conditional_return_else(&mut body)? {}
                while self.parse_conditional_return(&mut body)? {}
                while self.parse_conditional_assignment(&mut body)? {}
                while self.parse_return(&mut body)? {}
                while self.parse_expression_statement(&mut body)? {}
                if body.statement_count == before {
                    break;
                }
            }
            tail = self.peek()?;
        }
        let tail_end = self.source.trim_end().len();
        if let Some(tail) = tail {
            body.tail_span = Span {
                start: tail.span.start,
                end: tail_end,
            };
            body.tail_expression = &self.source[body.tail_span.start..body.tail_span.end];
        } else {
            body.tail_span = Span {
                start: tail_end,
                end: tail_end,
            };
            let last_loop_diverges = body.while_loops[..body.while_loop_count]
                .iter()
                .flatten()
                .next_back()
                .is_some_and(|loop_statement| {
                    loop_statement.condition.is_none()
                        && !loop_statement
                            .operations()
                            .iter()
                            .flatten()
                            .any(|operation| {
                                matches!(
                                    operation,
                                    LoopOperation::Break | LoopOperation::ConditionalBreak(_)
                                ) || matches!(
                                    operation,
                                    LoopOperation::ConditionalBlock(index)
                                        if loop_statement.conditional_blocks()[*index]
                                            .is_some_and(|block| {
                                                block.terminal
                                                    == Some(ConditionalLoopTerminal::Break)
                                            })
                                )
                            })
                });
            let exhaustive_chain_diverges = body.conditional_return_elses
                [..body.conditional_return_else_count]
                .iter()
                .flatten()
                .any(|conditional| conditional.else_value.is_some());
            if body.return_count != 0 || exhaustive_chain_diverges || last_loop_diverges {
                body.tail_expression = "";
                body.tail_diverges = true;
            } else {
                body.tail_expression = "()";
                body.implicit_unit = true;
            }
        }
        Ok(body)
    }
}

impl<'source> Parser<'source> {
    pub const fn new(source: &'source str) -> Self {
        Self {
            source,
            lexer: Lexer::new(source),
            lookahead: None,
        }
    }

    fn lexical(error: LexError) -> ParseError {
        ParseError {
            kind: ParseErrorKind::Lexical(error),
            span: error.span,
        }
    }

    fn peek(&mut self) -> Result<Option<Token<'source>>, ParseError> {
        if self.lookahead.is_none() {
            self.lookahead = self.lexer.next_token().map_err(Self::lexical)?;
        }
        Ok(self.lookahead)
    }

    fn take(&mut self) -> Result<Option<Token<'source>>, ParseError> {
        if let Some(token) = self.lookahead.take() {
            Ok(Some(token))
        } else {
            self.lexer.next_token().map_err(Self::lexical)
        }
    }

    fn eof_span(&self) -> Span {
        Span {
            start: self.source.len(),
            end: self.source.len(),
        }
    }

    fn error(&self, kind: ParseErrorKind, token: Option<Token<'source>>) -> ParseError {
        ParseError {
            kind,
            span: token.map_or_else(|| self.eof_span(), |value| value.span),
        }
    }

    fn take_text(&mut self, text: &str) -> Result<Option<Token<'source>>, ParseError> {
        if self.peek()?.is_some_and(|token| token.text == text) {
            self.take()
        } else {
            Ok(None)
        }
    }

    fn identifier(&mut self) -> Result<Token<'source>, ParseError> {
        let token = self.take()?;
        match token {
            Some(value) if value.kind == TokenKind::Identifier => Ok(value),
            _ => Err(self.error(ParseErrorKind::ExpectedIdentifier, token)),
        }
    }

    fn expect_kind(
        &mut self,
        kind: TokenKind,
        error: ParseErrorKind,
    ) -> Result<Token<'source>, ParseError> {
        let token = self.take()?;
        match token {
            Some(value) if value.kind == kind => Ok(value),
            _ => Err(self.error(error, token)),
        }
    }

    fn type_until(&mut self, terminators: &[TokenKind]) -> Result<TypeRef<'source>, ParseError> {
        let Some(first) = self.peek()? else {
            return Err(self.error(ParseErrorKind::ExpectedType, None));
        };
        if terminators.contains(&first.kind) {
            return Err(self.error(ParseErrorKind::ExpectedType, Some(first)));
        }
        let start = first.span.start;
        let mut end = start;
        let mut angle_depth = 0usize;
        let mut delimiter_depth = 0usize;
        loop {
            let Some(token) = self.peek()? else {
                return Err(self.error(ParseErrorKind::UnterminatedDelimiter, None));
            };
            if angle_depth == 0 && delimiter_depth == 0 && terminators.contains(&token.kind) {
                break;
            }
            match token.text {
                "<" => angle_depth = angle_depth.saturating_add(1),
                ">" if angle_depth != 0 => angle_depth -= 1,
                _ => {}
            }
            match token.kind {
                TokenKind::OpenParen | TokenKind::OpenBracket => {
                    delimiter_depth = delimiter_depth.saturating_add(1);
                }
                TokenKind::CloseParen | TokenKind::CloseBracket if delimiter_depth != 0 => {
                    delimiter_depth -= 1;
                }
                _ => {}
            }
            end = token.span.end;
            self.take()?;
        }
        let text = self.source[start..end].trim_end();
        Ok(TypeRef {
            text,
            span: Span {
                start,
                end: start + text.len(),
            },
        })
    }

    fn type_until_text(&mut self, terminator: &str) -> Result<TypeRef<'source>, ParseError> {
        let Some(first) = self.peek()? else {
            return Err(self.error(ParseErrorKind::ExpectedType, None));
        };
        if first.text == terminator {
            return Err(self.error(ParseErrorKind::ExpectedType, Some(first)));
        }
        let start = first.span.start;
        let mut end = start;
        let mut angle_depth = 0usize;
        let mut delimiters = [TokenKind::OpenParen; 64];
        let mut delimiter_depth = 0usize;
        loop {
            let Some(token) = self.peek()? else {
                return Err(self.error(ParseErrorKind::UnterminatedDelimiter, None));
            };
            if angle_depth == 0 && delimiter_depth == 0 && token.text == terminator {
                break;
            }
            match token.text {
                "<" => angle_depth = angle_depth.saturating_add(1),
                ">" if angle_depth != 0 => angle_depth -= 1,
                _ => {}
            }
            let open = match token.kind {
                TokenKind::OpenParen => Some(TokenKind::OpenParen),
                TokenKind::OpenBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(open) = open {
                if delimiter_depth == delimiters.len() {
                    return Err(self.error(ParseErrorKind::NestingLimitExceeded, Some(token)));
                }
                delimiters[delimiter_depth] = open;
                delimiter_depth += 1;
            }
            let expected = match token.kind {
                TokenKind::CloseParen => Some(TokenKind::OpenParen),
                TokenKind::CloseBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(expected) = expected {
                if delimiter_depth == 0 || delimiters[delimiter_depth - 1] != expected {
                    return Err(self.error(ParseErrorKind::UnexpectedClosingDelimiter, Some(token)));
                }
                delimiter_depth -= 1;
            }
            end = token.span.end;
            self.take()?;
        }
        if angle_depth != 0 || delimiter_depth != 0 {
            let token = self.peek()?;
            return Err(self.error(ParseErrorKind::UnterminatedDelimiter, token));
        }
        let text = self.source[start..end].trim_end();
        Ok(TypeRef {
            text,
            span: Span {
                start,
                end: start + text.len(),
            },
        })
    }

    fn initializer(&mut self) -> Result<(&'source str, Span), ParseError> {
        let Some(first) = self.peek()? else {
            return Err(self.error(ParseErrorKind::ExpectedInitializer, None));
        };
        if first.kind == TokenKind::Semicolon {
            return Err(self.error(ParseErrorKind::ExpectedInitializer, Some(first)));
        }
        let start = first.span.start;
        let mut end = start;
        let mut delimiters = [TokenKind::OpenParen; 64];
        let mut depth = 0usize;
        loop {
            let Some(token) = self.peek()? else {
                return Err(self.error(ParseErrorKind::ExpectedSemicolon, None));
            };
            if depth == 0 && token.kind == TokenKind::Semicolon {
                self.take()?;
                let text = self.source[start..end].trim_end();
                return Ok((
                    text,
                    Span {
                        start,
                        end: start + text.len(),
                    },
                ));
            }
            let open = match token.kind {
                TokenKind::OpenParen => Some(TokenKind::OpenParen),
                TokenKind::OpenBrace => Some(TokenKind::OpenBrace),
                TokenKind::OpenBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(open) = open {
                if depth == delimiters.len() {
                    return Err(self.error(ParseErrorKind::NestingLimitExceeded, Some(token)));
                }
                delimiters[depth] = open;
                depth += 1;
            }
            let expected = match token.kind {
                TokenKind::CloseParen => Some(TokenKind::OpenParen),
                TokenKind::CloseBrace => Some(TokenKind::OpenBrace),
                TokenKind::CloseBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(expected) = expected {
                if depth == 0 || delimiters[depth - 1] != expected {
                    return Err(self.error(ParseErrorKind::UnexpectedClosingDelimiter, Some(token)));
                }
                depth -= 1;
            }
            end = token.span.end;
            self.take()?;
        }
    }

    fn body(&mut self) -> Result<Span, ParseError> {
        let open = self.expect_kind(TokenKind::OpenBrace, ParseErrorKind::ExpectedBody)?;
        let mut delimiters = [TokenKind::OpenBrace; 64];
        let mut depth = 1usize;
        while let Some(token) = self.take()? {
            let matching_open = match token.kind {
                TokenKind::OpenParen => Some(TokenKind::OpenParen),
                TokenKind::OpenBrace => Some(TokenKind::OpenBrace),
                TokenKind::OpenBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(kind) = matching_open {
                if depth == delimiters.len() {
                    return Err(self.error(ParseErrorKind::NestingLimitExceeded, Some(token)));
                }
                delimiters[depth] = kind;
                depth += 1;
                continue;
            }
            let expected = match token.kind {
                TokenKind::CloseParen => Some(TokenKind::OpenParen),
                TokenKind::CloseBrace => Some(TokenKind::OpenBrace),
                TokenKind::CloseBracket => Some(TokenKind::OpenBracket),
                _ => None,
            };
            if let Some(expected) = expected {
                if delimiters[depth - 1] != expected {
                    return Err(self.error(ParseErrorKind::UnexpectedClosingDelimiter, Some(token)));
                }
                depth -= 1;
                if depth == 0 {
                    return Ok(Span {
                        start: open.span.start,
                        end: token.span.end,
                    });
                }
            }
        }
        Err(self.error(ParseErrorKind::UnterminatedDelimiter, None))
    }

    fn function<const MAX_PARAMETERS: usize>(
        &mut self,
        public: bool,
        constant: bool,
        abi: FunctionAbi,
        no_mangle: bool,
    ) -> Result<Function<'source, MAX_PARAMETERS>, ParseError> {
        let name = self.identifier()?;
        self.expect_kind(
            TokenKind::OpenParen,
            ParseErrorKind::ExpectedParameterSeparator,
        )?;
        let mut parameters = [None; MAX_PARAMETERS];
        let mut parameter_count = 0usize;
        if self
            .peek()?
            .is_some_and(|token| token.kind != TokenKind::CloseParen)
        {
            loop {
                if parameter_count == MAX_PARAMETERS {
                    let token = self.peek()?;
                    return Err(self.error(ParseErrorKind::TooManyParameters, token));
                }
                let parameter_name = self.identifier()?;
                self.expect_kind(TokenKind::Colon, ParseErrorKind::ExpectedParameterSeparator)?;
                let ty = self.type_until(&[TokenKind::Comma, TokenKind::CloseParen])?;
                parameters[parameter_count] = Some(Parameter {
                    name: parameter_name.text,
                    span: Span {
                        start: parameter_name.span.start,
                        end: ty.span.end,
                    },
                    ty,
                });
                parameter_count += 1;
                if self.take_text(",")?.is_some() {
                    if self
                        .peek()?
                        .is_some_and(|token| token.kind == TokenKind::CloseParen)
                    {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.expect_kind(
            TokenKind::CloseParen,
            ParseErrorKind::ExpectedParameterSeparator,
        )?;
        let return_type = if self.take_text("->")?.is_some() {
            Some(self.type_until(&[TokenKind::OpenBrace])?)
        } else {
            None
        };
        let body = self.body()?;
        let untrimmed = &self.source[body.start + 1..body.end - 1];
        let body_expression = untrimmed.trim();
        let body_expression_start =
            body.start + 1 + (untrimmed.len() - untrimmed.trim_start().len());
        Ok(Function {
            public,
            constant,
            abi,
            no_mangle,
            name: name.text,
            name_span: name.span,
            parameters,
            parameter_count,
            return_type,
            body,
            body_expression,
            body_expression_span: Span {
                start: body_expression_start,
                end: body_expression_start + body_expression.len(),
            },
        })
    }

    fn no_mangle_attribute(&mut self) -> Result<bool, ParseError> {
        if self.take_text("#")?.is_none() {
            return Ok(false);
        }
        self.expect_kind(
            TokenKind::OpenBracket,
            ParseErrorKind::ExpectedAttributeDelimiter,
        )?;
        let unsafe_keyword = self.identifier()?;
        if unsafe_keyword.text != "unsafe" {
            return Err(self.error(ParseErrorKind::UnsupportedAttribute, Some(unsafe_keyword)));
        }
        self.expect_kind(
            TokenKind::OpenParen,
            ParseErrorKind::ExpectedAttributeDelimiter,
        )?;
        let attribute = self.identifier()?;
        if attribute.text != "no_mangle" {
            return Err(self.error(ParseErrorKind::UnsupportedAttribute, Some(attribute)));
        }
        self.expect_kind(
            TokenKind::CloseParen,
            ParseErrorKind::ExpectedAttributeDelimiter,
        )?;
        self.expect_kind(
            TokenKind::CloseBracket,
            ParseErrorKind::ExpectedAttributeDelimiter,
        )?;
        Ok(true)
    }

    fn const_item(&mut self, public: bool) -> Result<ConstItem<'source>, ParseError> {
        let name = self.identifier()?;
        self.expect_kind(TokenKind::Colon, ParseErrorKind::ExpectedType)?;
        let ty = self.type_until_text("=")?;
        if self.take_text("=")?.is_none() {
            let token = self.peek()?;
            return Err(self.error(ParseErrorKind::ExpectedEquals, token));
        }
        let (initializer, initializer_span) = self.initializer()?;
        Ok(ConstItem {
            public,
            name: name.text,
            name_span: name.span,
            ty,
            initializer,
            initializer_span,
        })
    }

    pub fn parse_module<const MAX_ITEMS: usize, const MAX_PARAMETERS: usize>(
        mut self,
    ) -> Result<Module<'source, MAX_ITEMS, MAX_PARAMETERS>, ParseError> {
        let mut module = Module {
            items: [None; MAX_ITEMS],
            item_count: 0,
        };
        while self.peek()?.is_some() {
            if module.item_count == MAX_ITEMS {
                let token = self.peek()?;
                return Err(self.error(ParseErrorKind::TooManyItems, token));
            }
            let no_mangle = self.no_mangle_attribute()?;
            let public = self.take_text("pub")?.is_some();
            let keyword = self.take()?;
            let item = match keyword.map(|token| token.text) {
                Some("fn") => {
                    Item::Function(self.function(public, false, FunctionAbi::Rust, no_mangle)?)
                }
                Some("extern") => {
                    let abi = self.take()?;
                    if !abi.is_some_and(|token| {
                        token.kind == TokenKind::String && token.text == "\"C\""
                    }) {
                        return Err(self.error(ParseErrorKind::UnsupportedAbi, abi));
                    }
                    let function_keyword = self.take()?;
                    if !function_keyword.is_some_and(|token| token.text == "fn") {
                        return Err(self.error(ParseErrorKind::ExpectedItem, function_keyword));
                    }
                    Item::Function(self.function(public, false, FunctionAbi::C, no_mangle)?)
                }
                Some("const") => match self.peek()?.map(|token| token.text) {
                    Some("fn") => {
                        self.take()?;
                        Item::Function(self.function(public, true, FunctionAbi::Rust, no_mangle)?)
                    }
                    Some("extern") => {
                        self.take()?;
                        let abi = self.take()?;
                        if !abi.is_some_and(|token| {
                            token.kind == TokenKind::String && token.text == "\"C\""
                        }) {
                            return Err(self.error(ParseErrorKind::UnsupportedAbi, abi));
                        }
                        let function_keyword = self.take()?;
                        if !function_keyword.is_some_and(|token| token.text == "fn") {
                            return Err(self.error(ParseErrorKind::ExpectedItem, function_keyword));
                        }
                        Item::Function(self.function(public, true, FunctionAbi::C, no_mangle)?)
                    }
                    _ if !no_mangle => Item::Const(self.const_item(public)?),
                    _ => {
                        let token = self.peek()?;
                        return Err(self.error(ParseErrorKind::ExpectedItem, token));
                    }
                },
                Some("static") if !no_mangle => Item::Static(self.const_item(public)?),
                _ => return Err(self.error(ParseErrorKind::ExpectedItem, keyword)),
            };
            module.items[module.item_count] = Some(item);
            module.item_count += 1;
        }
        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestModule<'source> = Module<'source, 4, 4>;

    #[test]
    fn parses_function_signatures_without_allocation() {
        let source = "pub fn add(left: u32, right: &u32) -> core::primitive::u32 { left + *right }";
        let module: TestModule<'_> = Parser::new(source).parse_module().unwrap();
        assert_eq!(module.item_count(), 1);
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert!(function.public);
        assert!(!function.constant);
        assert_eq!(function.abi, FunctionAbi::Rust);
        assert!(!function.no_mangle);
        assert_eq!(function.name, "add");
        assert_eq!(function.parameter_count(), 2);
        assert_eq!(function.parameters()[0].unwrap().ty.text, "u32");
        assert_eq!(function.parameters()[1].unwrap().ty.text, "&u32");
        assert_eq!(function.return_type.unwrap().text, "core::primitive::u32");
        assert_eq!(
            &source[function.body.start..function.body.end],
            "{ left + *right }"
        );
        assert_eq!(function.body_expression, "left + *right");
        assert_eq!(
            &source[function.body_expression_span.start..function.body_expression_span.end],
            function.body_expression
        );
    }

    #[test]
    fn parses_explicit_unmangled_c_exports() {
        let source = "#[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { 42 }";
        let module: TestModule<'_> = Parser::new(source).parse_module().unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert!(function.public);
        assert_eq!(function.abi, FunctionAbi::C);
        assert!(function.no_mangle);
        assert_eq!(function.name, "answer");
    }

    #[test]
    fn parses_const_rust_and_c_abi_functions() {
        let source = "const fn identity(value: usize) -> usize { return value; } #[unsafe(no_mangle)] pub const extern \"C\" fn exported(value: u8) -> u8 { value + 1 }";
        let module: TestModule<'_> = Parser::new(source).parse_module().unwrap();
        let Some(Item::Function(identity)) = module.items()[0] else {
            panic!("expected function")
        };
        assert!(identity.constant);
        assert_eq!(identity.abi, FunctionAbi::Rust);
        let Some(Item::Function(exported)) = module.items()[1] else {
            panic!("expected function")
        };
        assert!(exported.constant);
        assert_eq!(exported.abi, FunctionAbi::C);
        assert!(exported.no_mangle);
    }

    #[test]
    fn rejects_unknown_attributes_and_abis() {
        let attribute = Parser::new("#[unsafe(export_name)] fn answer() {}")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(attribute.kind, ParseErrorKind::UnsupportedAttribute);
        let abi = Parser::new("extern \"system\" fn answer() {}")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(abi.kind, ParseErrorKind::UnsupportedAbi);
    }

    #[test]
    fn accepts_trailing_parameter_comma_and_nested_delimiters() {
        let module: TestModule<'_> =
            Parser::new("fn f(value: Option<(u8, u8)>,) { call([value]); }")
                .parse_module()
                .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parameters()[0].unwrap().ty.text,
            "Option<(u8, u8)>"
        );
    }

    #[test]
    fn parses_and_evaluates_typed_constants() {
        let source = "pub const BUFFER_BYTES: usize = (4 * 1024) + 16; fn ready() {}";
        let module: TestModule<'_> = Parser::new(source).parse_module().unwrap();
        assert_eq!(module.item_count(), 2);
        let Some(Item::Const(constant)) = module.items()[0] else {
            panic!("expected constant")
        };
        assert!(constant.public);
        assert_eq!(constant.name, "BUFFER_BYTES");
        assert_eq!(constant.ty.text, "usize");
        assert_eq!(constant.initializer, "(4 * 1024) + 16");
        assert_eq!(
            &source[constant.initializer_span.start..constant.initializer_span.end],
            constant.initializer
        );
        let expression = constant.parse_initializer::<16>().unwrap();
        assert_eq!(expression.evaluate(&crate::NoConstants), Ok(4112));
    }

    #[test]
    fn parses_immutable_static_initializers() {
        let source = "static SHIFTED: usize = 10_usize << 4_usize;";
        let module: TestModule<'_> = Parser::new(source).parse_module().unwrap();
        let Some(Item::Static(value)) = module.items()[0] else {
            panic!("expected static")
        };
        assert!(!value.public);
        assert_eq!(value.name, "SHIFTED");
        assert_eq!(value.ty.text, "usize");
        assert_eq!(
            value
                .parse_initializer::<8>()
                .unwrap()
                .evaluate(&crate::NoConstants),
            Ok(160)
        );
    }

    #[test]
    fn parses_typed_local_bindings_and_tail_expression() {
        let module: TestModule<'_> =
            Parser::new("fn arithmetic() -> usize { let x: usize = 15; let y: usize = 4; x / y }")
                .parse_module()
                .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.local_count(), 2);
        let locals = body.locals();
        assert_eq!(locals[0].unwrap().name, "x");
        assert_eq!(locals[0].unwrap().ty.unwrap().text, "usize");
        assert_eq!(locals[0].unwrap().initializer, "15");
        assert_eq!(locals[1].unwrap().name, "y");
        assert_eq!(body.tail_expression, "x / y");
    }

    #[test]
    fn bounds_and_validates_typed_local_bindings() {
        let module: TestModule<'_> =
            Parser::new("fn bounded() { let x: usize = 1; let y: usize = 2; x }")
                .parse_module()
                .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyLocals
        );

        let implicit_unit: TestModule<'_> = Parser::new("fn empty() { let x: usize = 1; }")
            .parse_module()
            .unwrap();
        let Some(Item::Function(function)) = implicit_unit.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<1>().unwrap();
        assert!(body.implicit_unit);
        assert_eq!(body.tail_expression, "()");
        assert_eq!(body.tail_span.start, body.tail_span.end);

        let empty: TestModule<'_> = Parser::new("fn empty() {}").parse_module().unwrap();
        let Some(Item::Function(function)) = empty.items()[0] else {
            panic!("expected function")
        };
        assert!(function.parse_body::<1>().unwrap().implicit_unit);
    }

    #[test]
    fn parses_mutable_locals_and_assignment_statements() {
        let module: TestModule<'_> = Parser::new(
            "fn swaps() -> isize { let mut a: isize = 1; let mut b: isize = 2; a ^= b; b ^= a; a = a ^ b; a | b }",
        )
        .parse_module()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<4>().unwrap();
        assert!(body.locals()[0].unwrap().mutable);
        assert_eq!(body.assignment_count(), 3);
        assert_eq!(
            body.assignments()[0].unwrap().operator,
            AssignmentOperator::BitXor
        );
        assert_eq!(
            body.assignments()[2].unwrap().operator,
            AssignmentOperator::Assign
        );
        assert_eq!(body.assignments()[2].unwrap().value, "a ^ b");
        assert_eq!(body.tail_expression, "a | b");
    }

    #[test]
    fn parses_bounded_conditional_assignments_in_body_order() {
        let module = Parser::new(
            "fn choose(value: u8, select: bool) -> u8 { let mut result = value; if select { let mut selected: u8 = result; selected += 1; selected + 10; return selected; } else if value == 1 { result -= 1; result + value; result += 2; return result; } else if value == 2 { result = 42; } else { result *= 2; result == value; result += 1; return result; } if result == 0 { result = 42; } result }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.conditional_assignment_count(), 2);
        let first = body.conditional_assignments()[0].unwrap();
        assert_eq!(first.branch_count(), 3);
        let first_branch = first.branches()[0].unwrap();
        assert_eq!(first_branch.condition, "select");
        assert_eq!(first_branch.action_count(), 4);
        let Some(ConditionalAssignmentAction::Local(first_local)) = first_branch.actions()[0]
        else {
            panic!("expected local")
        };
        assert_eq!(first_local.name, "selected");
        assert!(first_local.mutable);
        let Some(ConditionalAssignmentAction::Assignment(first_assignment)) =
            first_branch.actions()[1]
        else {
            panic!("expected assignment")
        };
        assert_eq!(first_assignment.operator, AssignmentOperator::Add);
        let Some(ConditionalAssignmentAction::Expression(expression)) = first_branch.actions()[2]
        else {
            panic!("expected expression")
        };
        assert_eq!(expression.expression, "selected + 10");
        let Some(ConditionalAssignmentAction::Return(return_statement)) = first_branch.actions()[3]
        else {
            panic!("expected return")
        };
        assert_eq!(return_statement.value, "selected");
        assert_eq!(first.branches()[1].unwrap().condition, "value == 1");
        let Some(ConditionalAssignmentAction::Assignment(assignment)) =
            first.branches()[1].unwrap().actions()[0]
        else {
            panic!("expected assignment")
        };
        assert_eq!(assignment.operator, AssignmentOperator::Subtract);
        assert_eq!(first.branches()[2].unwrap().condition, "value == 2");
        assert_eq!(first.else_action_count(), 4);
        let Some(ConditionalAssignmentAction::Assignment(assignment)) = first.else_actions()[0]
        else {
            panic!("expected assignment")
        };
        assert_eq!(assignment.operator, AssignmentOperator::Multiply);
        assert_eq!(
            body.conditional_assignments()[1]
                .unwrap()
                .else_action_count(),
            0
        );
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(
            body.statements()[1],
            Some(BodyStatement::ConditionalAssignment(0))
        );

        let crowded = Parser::new(
            "fn crowded(value: u8) -> u8 { let mut result = value; if true { result = 0; } if true { result = 1; } if true { result = 2; } if true { result = 3; } if true { result = 4; } if true { result = 5; } if true { result = 6; } if true { result = 7; } if true { result = 8; } result }",
        )
        .parse_module::<2, 1>()
        .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalAssignments
        );

        let too_many_branches = Parser::new(
            "fn crowded(value: u8) -> u8 { let mut result = value; if value == 0 { result = 0; } else if value == 1 { result = 1; } else if value == 2 { result = 2; } else if value == 3 { result = 3; } else if value == 4 { result = 4; } else { result = 5; } result }",
        )
        .parse_module::<2, 1>()
        .unwrap();
        let Some(Item::Function(function)) = too_many_branches.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalAssignmentBranches
        );

        let too_many_in_branch = Parser::new(
            "fn crowded(value: u8) -> u8 { let mut result = value; if true { result = 0; result = 1; result = 2; result = 3; result = 4; } result }",
        )
        .parse_module::<2, 1>()
        .unwrap();
        let Some(Item::Function(function)) = too_many_in_branch.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalBranchActions
        );
    }

    #[test]
    fn parses_inferred_local_bindings() {
        let module = Parser::new("fn value() -> i32 { let mut x = 0; x += 1; x }")
            .parse_module::<2, 1>()
            .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let local = body.locals()[0].unwrap();
        assert!(local.ty.is_none());
        assert!(local.mutable);
        assert_eq!(local.initializer, "0");
    }

    #[test]
    fn separates_conditional_returns_from_tail_if_expressions() {
        let module = Parser::new(
            "fn choose(value: u64, guard: bool) -> u64 { if guard { return value; } value + 1 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.conditional_return_count(), 1);
        let conditional = body.conditional_returns()[0].unwrap();
        assert_eq!(conditional.condition, "guard");
        assert_eq!(conditional.value, "value");
        assert_eq!(body.tail_expression, "value + 1");

        let tail_if = Parser::new(
            "fn choose(value: u64, guard: bool) -> u64 { if guard { value } else { value + 1 } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = tail_if.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.conditional_return_count(), 0);
        assert!(body.tail_expression.starts_with("if guard"));

        let guarded_local = Parser::new(
            "fn guarded(value: u64, stop: bool) -> u64 { if stop { return 7; } let adjusted: u64 = value + 1; adjusted * 2 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = guarded_local.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(
            body.statements()[0],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[1], Some(BodyStatement::Local(0)));
        assert_eq!(body.locals()[0].unwrap().initializer, "value + 1");
        assert_eq!(body.tail_expression, "adjusted * 2");

        let guarded_assignment = Parser::new(
            "fn guarded(value: u64, stop: bool) -> u64 { let mut adjusted: u64 = value; if stop { return 7; } adjusted += 1; adjusted * 2 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = guarded_assignment.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(
            body.statements()[1],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[2], Some(BodyStatement::Assignment(0)));
        assert_eq!(
            body.assignments()[0].unwrap().operator,
            AssignmentOperator::Add
        );

        let guarded_loop = Parser::new(
            "fn fibonacci(n: u32) -> u32 { if n == 0 { return 0; } let mut previous: u32 = 0; let mut current: u32 = 1; let mut index: u32 = 1; while index < n { current += previous; previous = current - previous; index += 1; } current }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = guarded_loop.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<4>().unwrap();
        assert_eq!(
            body.statements()[0],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[1], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Local(1)));
        assert_eq!(body.statements()[3], Some(BodyStatement::Local(2)));
        assert_eq!(body.statements()[4], Some(BodyStatement::Loop(0)));
        assert_eq!(body.while_loops()[0].unwrap().operation_count(), 3);

        let alternating_guards = Parser::new(
            "fn classify(value: u32, first: bool, second: bool) -> u32 { let mut result: u32 = value; if first { return 1; } result += 1; if second { return 2; } result *= 2; result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = alternating_guards.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<4>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(
            body.statements()[1],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[2], Some(BodyStatement::Assignment(0)));
        assert_eq!(
            body.statements()[3],
            Some(BodyStatement::ConditionalReturn(1))
        );
        assert_eq!(body.statements()[4], Some(BodyStatement::Assignment(1)));
    }

    #[test]
    fn parses_bounded_scalar_while_loops() {
        let module = Parser::new(
            "fn count(limit: u64) -> u64 { let mut i: u64 = 0; let mut total: u64 = 0; while i < limit { let current: u64 = i + 1; current + 10; total += current; i = current; if i == 10 { let marker: u64 = i; marker + 1; total += marker; break; } } total }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statement_count(), 3);
        assert!(core::mem::size_of::<BodyStatement>() <= 2 * core::mem::size_of::<usize>());
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Local(1)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Loop(0)));
        assert_eq!(body.while_loop_count(), 1);
        let loop_statement = body.while_loops()[0].unwrap();
        assert_eq!(loop_statement.condition, Some("i < limit"));
        assert_eq!(loop_statement.assignment_count(), 3);
        assert_eq!(loop_statement.operation_count(), 5);
        let Some(LoopOperation::Local(current)) = loop_statement.operations()[0] else {
            panic!("expected loop local")
        };
        assert_eq!(current.name, "current");
        assert_eq!(current.initializer, "i + 1");
        let Some(LoopOperation::Expression(expression)) = loop_statement.operations()[1] else {
            panic!("expected loop expression")
        };
        assert_eq!(expression.expression, "current + 10");
        let Some(LoopOperation::Assignment(total)) = loop_statement.operations()[2] else {
            panic!("expected total assignment")
        };
        let Some(LoopOperation::Assignment(increment)) = loop_statement.operations()[3] else {
            panic!("expected increment assignment")
        };
        assert_eq!(total.name, "total");
        assert_eq!(increment.name, "i");
        assert_eq!(increment.operator, AssignmentOperator::Assign);
        assert_eq!(increment.value, "current");
        let Some(LoopOperation::ConditionalBlock(control_index)) = loop_statement.operations()[4]
        else {
            panic!("expected conditional block")
        };
        let control = loop_statement.conditional_blocks()[control_index].unwrap();
        assert_eq!(control.condition, "i == 10");
        assert_eq!(control.action_count(), 3);
        assert!(matches!(
            control.actions()[0],
            Some(ConditionalLoopAction::Local(local)) if local.name == "marker"
        ));
        assert!(matches!(
            control.actions()[1],
            Some(ConditionalLoopAction::Expression(expression))
                if expression.expression == "marker + 1"
        ));
        assert!(matches!(
            control.actions()[2],
            Some(ConditionalLoopAction::Assignment(assignment))
                if assignment.name == "total"
        ));
        assert_eq!(control.terminal, Some(ConditionalLoopTerminal::Break));
        assert_eq!(body.tail_expression, "total");

        let action_only = Parser::new(
            "fn guarded(limit: u64) -> u64 { let mut i: u64 = 0; let mut total: u64 = 0; while i < limit { i += 1; if i % 2 == 0 { let selected: u64 = i; total += selected; } else { let fallback: u64 = 1; total += fallback; } total += 1; } total }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = action_only.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let loop_statement = body.while_loops()[0].unwrap();
        let Some(LoopOperation::ConditionalBlock(index)) = loop_statement.operations()[1] else {
            panic!("expected action-only conditional block")
        };
        let block = loop_statement.conditional_blocks()[index].unwrap();
        assert_eq!(block.action_count(), 2);
        assert_eq!(block.terminal, None);
        let else_index = block.else_arm.expect("expected else arm");
        let else_arm = loop_statement.conditional_else_arms()[else_index].unwrap();
        assert_eq!(else_arm.action_count(), 2);
        assert_eq!(else_arm.terminal, None);

        let too_many_conditional_actions = Parser::new(
            "fn crowded(limit: u64) -> u64 { let mut i: u64 = 0; while i < limit { i += 1; if i == limit { i + 1; i + 2; i + 3; i + 4; i + 5; break; } } i }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = too_many_conditional_actions.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<2>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalLoopActions
        );

        let too_many_else_arms = Parser::new(
            "fn crowded() -> u64 { loop { if true { 1; } else { 2; } if true { 3; } else { 4; } if true { 5; } else { 6; } if true { 7; } else { 8; } if true { 9; } else { 10; } break; } 0 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = too_many_else_arms.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<2>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalLoopElseArms
        );

        let post_loop = Parser::new(
            "fn staged(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { value += 1; } value *= 2; value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = post_loop.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Loop(0)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Assignment(0)));
        assert_eq!(
            body.assignments()[0].unwrap().operator,
            AssignmentOperator::Multiply
        );

        let post_loop_return = Parser::new(
            "fn classify(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { value += 1; } if value == 42 { return 7; } value += 1; value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = post_loop_return.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Loop(0)));
        assert_eq!(
            body.statements()[2],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[3], Some(BodyStatement::Assignment(0)));

        let post_loop_guards = Parser::new(
            "fn classify(value: u32, limit: u32, first: bool, second: bool) -> u32 { let mut index: u32 = 0; let mut result: u32 = value; while index < limit { index += 1; } if first && index == limit { return 1; } result += 1; if second && index == limit { return 2; } result *= 2; result }",
        )
        .parse_module::<2, 4>()
        .unwrap();
        let Some(Item::Function(function)) = post_loop_guards.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<4>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Local(1)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Loop(0)));
        assert_eq!(
            body.statements()[3],
            Some(BodyStatement::ConditionalReturn(0))
        );
        assert_eq!(body.statements()[4], Some(BodyStatement::Assignment(0)));
        assert_eq!(
            body.statements()[5],
            Some(BodyStatement::ConditionalReturn(1))
        );
        assert_eq!(body.statements()[6], Some(BodyStatement::Assignment(1)));

        let sequential_loops = Parser::new(
            "fn traverse(first: u64, second: u64) -> u64 { let mut value: u64 = 0; while value < first { value += 1; } while value < second { value += 1; } value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = sequential_loops.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.while_loop_count(), 2);
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Loop(0)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Loop(1)));
        assert_eq!(
            body.while_loops()[0].unwrap().condition,
            Some("value < first")
        );
        assert_eq!(
            body.while_loops()[1].unwrap().condition,
            Some("value < second")
        );

        let post_loop_local = Parser::new(
            "fn finish(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { value += 1; } let offset: u64 = value + 2; offset * 2 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = post_loop_local.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Loop(0)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Local(1)));
        assert_eq!(body.locals()[1].unwrap().initializer, "value + 2");
        assert_eq!(body.tail_expression, "offset * 2");

        let unconditional = Parser::new(
            "fn once() -> u64 { let mut value: u64 = 0; loop { value += 1; if value == 1 { break; } } value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = unconditional.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let loop_statement = body.while_loops()[0].unwrap();
        assert_eq!(loop_statement.condition, None);
        assert!(matches!(
            loop_statement.operations()[1],
            Some(LoopOperation::ConditionalBreak(control)) if control.condition == "value == 1"
        ));

        let returning = Parser::new(
            "fn classify(value: u64) -> bool { loop { if value == 42 { return true; } if value != 42 { return false; } } false }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = returning.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let loop_statement = body.while_loops()[0].unwrap();
        assert!(matches!(
            loop_statement.operations()[0],
            Some(LoopOperation::ConditionalReturn(value))
                if value.condition == "value == 42" && value.value == "true"
        ));

        let immediate_return = Parser::new("fn answer() -> u64 { loop { return 42; } }")
            .parse_module::<2, 1>()
            .unwrap();
        let Some(Item::Function(function)) = immediate_return.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<1>().unwrap();
        assert!(body.tail_diverges);
        assert!(!body.implicit_unit);
        assert_eq!(body.tail_expression, "");
        assert!(matches!(
            body.while_loops()[0].unwrap().operations()[0],
            Some(LoopOperation::Return(value)) if value.value == "42"
        ));

        let controls = Parser::new(
            "fn controls(limit: u64) -> u64 { let mut value: u64 = 0; while value < limit { value += 1; continue; } loop { break; } value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = controls.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert!(matches!(
            body.while_loops()[0].unwrap().operations()[1],
            Some(LoopOperation::Continue)
        ));
        assert!(matches!(
            body.while_loops()[1].unwrap().operations()[0],
            Some(LoopOperation::Break)
        ));

        let oversized = Parser::new(
            "fn too_many(limit: u64) -> u64 { let mut i: u64 = 0; while i < limit { i += 1; i += 1; i += 1; i += 1; i += 1; } i }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = oversized.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<2>().unwrap_err().kind,
            ParseErrorKind::TooManyLoopAssignments
        );

        let too_many_operations = Parser::new(
            "fn crowded() -> u64 { loop { if true { continue; } if true { continue; } if true { continue; } if true { continue; } if true { continue; } if true { continue; } if true { continue; } if true { continue; } break; } 0 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = too_many_operations.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<2>().unwrap_err().kind,
            ParseErrorKind::TooManyLoopOperations
        );
    }

    #[test]
    fn const_types_and_initializers_preserve_nested_delimiters() {
        let module: TestModule<'_> = Parser::new("const VALUE: Option<[u8; 4]> = (1 + (2 * 3));")
            .parse_module()
            .unwrap();
        let Some(Item::Const(constant)) = module.items()[0] else {
            panic!("expected constant")
        };
        assert_eq!(constant.ty.text, "Option<[u8; 4]>");
        assert_eq!(
            constant
                .parse_initializer::<16>()
                .unwrap()
                .evaluate(&crate::NoConstants),
            Ok(7)
        );
    }

    #[test]
    fn rejects_empty_unterminated_and_mismatched_const_initializers() {
        let empty = Parser::new("const VALUE: usize = ;")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(empty.kind, ParseErrorKind::ExpectedInitializer);
        let unterminated = Parser::new("const VALUE: usize = 1")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(unterminated.kind, ParseErrorKind::ExpectedSemicolon);
        let mismatched = Parser::new("const VALUE: usize = (1];")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(mismatched.kind, ParseErrorKind::UnexpectedClosingDelimiter);
    }

    #[test]
    fn rejects_mismatched_and_unterminated_bodies() {
        let mismatch = Parser::new("fn f() { (] }")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(mismatch.kind, ParseErrorKind::UnexpectedClosingDelimiter);
        let unterminated = Parser::new("fn f() {").parse_module::<2, 2>().unwrap_err();
        assert_eq!(unterminated.kind, ParseErrorKind::UnterminatedDelimiter);
    }

    #[test]
    fn rejects_body_nesting_beyond_the_fixed_limit() {
        let source = "fn f() {((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((x))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))}";
        let error = Parser::new(source).parse_module::<2, 2>().unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::NestingLimitExceeded);
        assert!(error.span.start < source.len());
    }

    #[test]
    fn preserves_bounded_expression_statements_in_body_order() {
        let module = Parser::new(
            "fn probe(value: u32) -> u32 { ; value + 1; let next = value + 2; next; value + 3 }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.statements()[0], Some(BodyStatement::Expression(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Expression(1)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[3], Some(BodyStatement::Expression(2)));
        assert_eq!(body.expression_statements()[0].unwrap().expression, "");
        assert_eq!(
            body.expression_statements()[1].unwrap().expression,
            "value + 1"
        );
        assert_eq!(body.expression_statements()[2].unwrap().expression, "next");
        assert_eq!(body.tail_expression, "value + 3");

        let crowded = Parser::new("fn crowded() -> u32 { 0; 1; 2; 3; 4; 5; 6; 7; 8; 42 }")
            .parse_module::<2, 1>()
            .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyExpressionStatements
        );
    }

    #[test]
    fn preserves_bounded_early_returns_in_body_order() {
        let module = Parser::new(
            "fn probe(value: u32) -> u32 { let adjusted = value + 1; return adjusted; let unreachable = value + 2; unreachable }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.return_count(), 1);
        assert_eq!(body.returns()[0].unwrap().value, "adjusted");
        assert_eq!(body.statements()[0], Some(BodyStatement::Local(0)));
        assert_eq!(body.statements()[1], Some(BodyStatement::Return(0)));
        assert_eq!(body.statements()[2], Some(BodyStatement::Local(1)));
        assert!(!body.tail_diverges);

        let unit =
            Parser::new("fn stop(flag: bool) { if flag { return; } loop { if flag { return; } } }")
                .parse_module::<2, 2>()
                .unwrap();
        let Some(Item::Function(function)) = unit.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.conditional_returns()[0].unwrap().value, "()");
        assert!(matches!(
            body.while_loops()[0].unwrap().operations()[0],
            Some(LoopOperation::ConditionalReturn(value)) if value.value == "()"
        ));

        let unit = Parser::new("fn stop() { return; let unreachable = (); unreachable }")
            .parse_module::<2, 1>()
            .unwrap();
        let Some(Item::Function(function)) = unit.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<1>().unwrap();
        assert_eq!(body.returns()[0].unwrap().value, "()");

        let module = Parser::new(
            "fn probe() -> u8 { return 0; return 1; return 2; return 3; return 4; return 5; return 6; return 7; return 8; }",
        )
        .parse_module::<2, 1>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyReturns
        );
    }

    #[test]
    fn parses_bounded_exhaustive_conditional_returns() {
        let module = Parser::new(
            "fn choose(value: u32, select: bool) -> u32 { let adjusted = value + 1; if select { return adjusted; } else { return value; } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        assert_eq!(body.conditional_return_else_count(), 1);
        let conditional = body.conditional_return_elses()[0].unwrap();
        assert_eq!(conditional.branch_count(), 1);
        assert_eq!(conditional.branches()[0].unwrap().condition, "select");
        assert_eq!(conditional.branches()[0].unwrap().value, "adjusted");
        assert_eq!(conditional.else_value.unwrap().value, "value");
        assert_eq!(
            body.statements()[1],
            Some(BodyStatement::ConditionalReturnElse(0))
        );
        assert!(body.tail_diverges);

        let chain = Parser::new(
            "fn choose(value: u8) -> u8 { if value == 0 { return 10; } else if value == 1 { return 20; } else if value == 2 { return 30; } else { return 40; } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = chain.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let conditional = body.conditional_return_elses()[0].unwrap();
        assert_eq!(conditional.branch_count(), 3);
        assert_eq!(conditional.branches()[1].unwrap().condition, "value == 1");
        assert_eq!(conditional.branches()[2].unwrap().value, "30");
        assert_eq!(conditional.else_value.unwrap().value, "40");

        let fallthrough = Parser::new(
            "fn choose(value: u8) -> u8 { if value == 0 { return 42; } else if value == 1 { return 42 / value; } 84 / value }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = fallthrough.items()[0] else {
            panic!("expected function")
        };
        let body = function.parse_body::<2>().unwrap();
        let conditional = body.conditional_return_elses()[0].unwrap();
        assert_eq!(conditional.branch_count(), 2);
        assert!(conditional.else_value.is_none());
        assert_eq!(body.tail_expression, "84 / value");
        assert!(!body.tail_diverges);

        let crowded = Parser::new(
            "fn choose(first: bool, second: bool) -> u8 { if first { return 1; } else { return 2; } if second { return 3; } else { return 4; } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<1>().unwrap_err().kind,
            ParseErrorKind::TooManyLocals
        );

        let crowded = Parser::new(
            "fn choose(value: u8) -> u8 { if value == 0 { return 0; } else if value == 1 { return 1; } else if value == 2 { return 2; } else if value == 3 { return 3; } else if value == 4 { return 4; } else { return 5; } }",
        )
        .parse_module::<2, 2>()
        .unwrap();
        let Some(Item::Function(function)) = crowded.items()[0] else {
            panic!("expected function")
        };
        assert_eq!(
            function.parse_body::<2>().unwrap_err().kind,
            ParseErrorKind::TooManyConditionalReturnBranches
        );
    }

    #[test]
    fn enforces_item_and_parameter_capacity() {
        let items = Parser::new("fn a() {} fn b() {}")
            .parse_module::<1, 1>()
            .unwrap_err();
        assert_eq!(items.kind, ParseErrorKind::TooManyItems);
        let parameters = Parser::new("fn a(a: u8, b: u8) {}")
            .parse_module::<1, 1>()
            .unwrap_err();
        assert_eq!(parameters.kind, ParseErrorKind::TooManyParameters);
    }

    #[test]
    fn rejects_non_items_and_missing_types() {
        let item = Parser::new("struct Value;")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(item.kind, ParseErrorKind::ExpectedItem);
        let ty = Parser::new("fn f(value:) {}")
            .parse_module::<2, 2>()
            .unwrap_err();
        assert_eq!(ty.kind, ParseErrorKind::ExpectedType);
    }
}
