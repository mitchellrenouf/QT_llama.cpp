#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenKind {
    Identifier,
    Lifetime,
    Integer,
    String,
    Character,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    Semicolon,
    Comma,
    Dot,
    Colon,
    PathSeparator,
    Arrow,
    FatArrow,
    Pound,
    Bang,
    Question,
    Operator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<'source> {
    pub kind: TokenKind,
    pub text: &'source str,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LexErrorKind {
    UnterminatedBlockComment,
    UnterminatedCharacter,
    UnterminatedString,
    UnexpectedByte,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub span: Span,
}

#[derive(Clone)]
pub struct Lexer<'source> {
    source: &'source str,
    position: usize,
}

impl<'source> Lexer<'source> {
    pub const fn new(source: &'source str) -> Self {
        Self {
            source,
            position: 0,
        }
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    fn bytes(&self) -> &'source [u8] {
        self.source.as_bytes()
    }

    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            while self
                .bytes()
                .get(self.position)
                .is_some_and(u8::is_ascii_whitespace)
            {
                self.position += 1;
            }
            if self.bytes().get(self.position..self.position + 2) == Some(b"//") {
                self.position += 2;
                while self
                    .bytes()
                    .get(self.position)
                    .is_some_and(|byte| *byte != b'\n')
                {
                    self.position += 1;
                }
                continue;
            }
            if self.bytes().get(self.position..self.position + 2) == Some(b"/*") {
                let start = self.position;
                self.position += 2;
                let mut depth = 1usize;
                while depth != 0 {
                    match self.bytes().get(self.position..self.position + 2) {
                        Some(b"/*") => {
                            depth = depth.checked_add(1).ok_or(LexError {
                                kind: LexErrorKind::UnterminatedBlockComment,
                                span: Span {
                                    start,
                                    end: self.position,
                                },
                            })?;
                            self.position += 2;
                        }
                        Some(b"*/") => {
                            depth -= 1;
                            self.position += 2;
                        }
                        _ if self.position < self.bytes().len() => self.position += 1,
                        _ => {
                            return Err(LexError {
                                kind: LexErrorKind::UnterminatedBlockComment,
                                span: Span {
                                    start,
                                    end: self.position,
                                },
                            });
                        }
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    fn finish(&self, start: usize, kind: TokenKind) -> Token<'source> {
        Token {
            kind,
            text: &self.source[start..self.position],
            span: Span {
                start,
                end: self.position,
            },
        }
    }

    fn quoted(
        &mut self,
        start: usize,
        quote: u8,
        kind: TokenKind,
    ) -> Result<Token<'source>, LexError> {
        self.position += 1;
        while let Some(&byte) = self.bytes().get(self.position) {
            if byte == quote {
                self.position += 1;
                return Ok(self.finish(start, kind));
            }
            if byte == b'\\' {
                self.position += 1;
                if self.position == self.bytes().len() {
                    break;
                }
            }
            self.position += 1;
        }
        Err(LexError {
            kind: if quote == b'"' {
                LexErrorKind::UnterminatedString
            } else {
                LexErrorKind::UnterminatedCharacter
            },
            span: Span {
                start,
                end: self.position,
            },
        })
    }

    pub fn next_token(&mut self) -> Result<Option<Token<'source>>, LexError> {
        self.skip_trivia()?;
        let start = self.position;
        let Some(&byte) = self.bytes().get(start) else {
            return Ok(None);
        };

        if byte.is_ascii_alphabetic() || byte == b'_' {
            self.position += 1;
            while self
                .bytes()
                .get(self.position)
                .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
            {
                self.position += 1;
            }
            return Ok(Some(self.finish(start, TokenKind::Identifier)));
        }
        if byte.is_ascii_digit() {
            self.position += 1;
            while self
                .bytes()
                .get(self.position)
                .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
            {
                self.position += 1;
            }
            return Ok(Some(self.finish(start, TokenKind::Integer)));
        }
        if byte == b'\''
            && self
                .bytes()
                .get(start + 1)
                .is_some_and(|value| value.is_ascii_alphabetic() || *value == b'_')
            && self.bytes().get(start + 2) != Some(&b'\'')
        {
            self.position += 2;
            while self
                .bytes()
                .get(self.position)
                .is_some_and(|value| value.is_ascii_alphanumeric() || *value == b'_')
            {
                self.position += 1;
            }
            return Ok(Some(self.finish(start, TokenKind::Lifetime)));
        }
        if byte == b'"' {
            return self.quoted(start, byte, TokenKind::String).map(Some);
        }
        if byte == b'\'' {
            return self.quoted(start, byte, TokenKind::Character).map(Some);
        }

        if matches!(self.bytes().get(start..start + 3), Some(b"<<=" | b">>=")) {
            self.position += 3;
            return Ok(Some(self.finish(start, TokenKind::Operator)));
        }
        let pair = self.bytes().get(start..start + 2);
        let paired_kind = match pair {
            Some(b"::") => Some(TokenKind::PathSeparator),
            Some(b"->") => Some(TokenKind::Arrow),
            Some(b"=>") => Some(TokenKind::FatArrow),
            Some(
                b"==" | b"!=" | b"<=" | b">=" | b"&&" | b"||" | b"<<" | b">>" | b"+=" | b"-="
                | b"*=" | b"/=" | b"%=" | b"&=" | b"|=" | b"^=",
            ) => Some(TokenKind::Operator),
            _ => None,
        };
        if let Some(kind) = paired_kind {
            self.position += 2;
            return Ok(Some(self.finish(start, kind)));
        }
        let kind = match byte {
            b'(' => TokenKind::OpenParen,
            b')' => TokenKind::CloseParen,
            b'{' => TokenKind::OpenBrace,
            b'}' => TokenKind::CloseBrace,
            b'[' => TokenKind::OpenBracket,
            b']' => TokenKind::CloseBracket,
            b';' => TokenKind::Semicolon,
            b',' => TokenKind::Comma,
            b'.' => TokenKind::Dot,
            b':' => TokenKind::Colon,
            b'#' => TokenKind::Pound,
            b'!' => TokenKind::Bang,
            b'?' => TokenKind::Question,
            b'+' | b'-' | b'*' | b'/' | b'%' | b'=' | b'<' | b'>' | b'&' | b'|' | b'^' | b'~'
            | b'@' => TokenKind::Operator,
            _ => {
                return Err(LexError {
                    kind: LexErrorKind::UnexpectedByte,
                    span: Span {
                        start,
                        end: start + 1,
                    },
                });
            }
        };
        self.position += 1;
        Ok(Some(self.finish(start, kind)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tokens(source: &str, expected: &[(TokenKind, &str)]) {
        let mut lexer = Lexer::new(source);
        for &(kind, text) in expected {
            let token = lexer.next_token().unwrap().expect("missing token");
            assert_eq!((token.kind, token.text), (kind, text));
            assert_eq!(&source[token.span.start..token.span.end], text);
        }
        assert_eq!(lexer.next_token(), Ok(None));
    }

    #[test]
    fn tokenizes_a_minimal_rust_item() {
        assert_tokens(
            "fn main() { let answer = 42_u8; }",
            &[
                (TokenKind::Identifier, "fn"),
                (TokenKind::Identifier, "main"),
                (TokenKind::OpenParen, "("),
                (TokenKind::CloseParen, ")"),
                (TokenKind::OpenBrace, "{"),
                (TokenKind::Identifier, "let"),
                (TokenKind::Identifier, "answer"),
                (TokenKind::Operator, "="),
                (TokenKind::Integer, "42_u8"),
                (TokenKind::Semicolon, ";"),
                (TokenKind::CloseBrace, "}"),
            ],
        );
    }

    #[test]
    fn separates_lifetimes_from_character_literals() {
        assert_tokens(
            "&'input str 'x' '\\n'",
            &[
                (TokenKind::Operator, "&"),
                (TokenKind::Lifetime, "'input"),
                (TokenKind::Identifier, "str"),
                (TokenKind::Character, "'x'"),
                (TokenKind::Character, "'\\n'"),
            ],
        );
    }

    #[test]
    fn skips_nested_comments_and_line_comments() {
        assert_tokens(
            "a /* outer /* inner */ end */ // rest\n b",
            &[(TokenKind::Identifier, "a"), (TokenKind::Identifier, "b")],
        );
    }

    #[test]
    fn reports_bounded_unterminated_input() {
        let mut lexer = Lexer::new("/* open");
        assert_eq!(
            lexer.next_token(),
            Err(LexError {
                kind: LexErrorKind::UnterminatedBlockComment,
                span: Span { start: 0, end: 7 },
            })
        );
        let mut lexer = Lexer::new("\"open");
        assert_eq!(
            lexer.next_token().unwrap_err().kind,
            LexErrorKind::UnterminatedString
        );
    }

    #[test]
    fn recognizes_multi_byte_punctuation_before_single_byte_forms() {
        assert_tokens(
            "a::b -> c => d == e",
            &[
                (TokenKind::Identifier, "a"),
                (TokenKind::PathSeparator, "::"),
                (TokenKind::Identifier, "b"),
                (TokenKind::Arrow, "->"),
                (TokenKind::Identifier, "c"),
                (TokenKind::FatArrow, "=>"),
                (TokenKind::Identifier, "d"),
                (TokenKind::Operator, "=="),
                (TokenKind::Identifier, "e"),
            ],
        );
    }
}
