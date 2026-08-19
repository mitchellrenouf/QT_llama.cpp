#![no_std]

#[cfg(test)]
extern crate std;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Token<'a> {
    Null,
    Bool(bool),
    Number(&'a str),
    String(&'a str),
    ArrayStart,
    ArrayEnd,
    ObjectStart,
    ObjectEnd,
    Colon,
    Comma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenError {
    pub message: &'static str,
    pub offset: usize,
}

impl core::fmt::Display for TokenError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{} at byte {}", self.message, self.offset)
    }
}

impl core::error::Error for TokenError {}

pub struct Tokens<'a> {
    source: &'a str,
    position: usize,
}

impl<'a> Tokens<'a> {
    pub const fn new(source: &'a str) -> Self {
        Self {
            source,
            position: 0,
        }
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    fn error(&self, message: &'static str) -> TokenError {
        TokenError {
            message,
            offset: self.position,
        }
    }

    fn string(&mut self) -> Result<Token<'a>, TokenError> {
        let bytes = self.source.as_bytes();
        self.position += 1;
        let start = self.position;
        while let Some(&byte) = bytes.get(self.position) {
            match byte {
                b'"' => {
                    let value = &self.source[start..self.position];
                    self.position += 1;
                    return Ok(Token::String(value));
                }
                b'\\' => {
                    self.position += 1;
                    let escaped = *bytes
                        .get(self.position)
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    if escaped == b'u' {
                        for _ in 0..4 {
                            self.position += 1;
                            if !bytes.get(self.position).is_some_and(u8::is_ascii_hexdigit) {
                                return Err(self.error("invalid unicode escape"));
                            }
                        }
                    } else if !matches!(
                        escaped,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                    ) {
                        return Err(self.error("invalid escape"));
                    }
                    self.position += 1;
                }
                0..=0x1f => return Err(self.error("control character in string")),
                _ => self.position += 1,
            }
        }
        Err(self.error("unterminated string"))
    }

    fn number(&mut self) -> Result<Token<'a>, TokenError> {
        let bytes = self.source.as_bytes();
        let start = self.position;
        if bytes.get(self.position) == Some(&b'-') {
            self.position += 1;
        }
        if bytes.get(self.position) == Some(&b'0') {
            self.position += 1;
            if bytes.get(self.position).is_some_and(u8::is_ascii_digit) {
                return Err(self.error("leading zero"));
            }
        } else {
            self.digits()?;
        }
        if bytes.get(self.position) == Some(&b'.') {
            self.position += 1;
            self.digits()?;
        }
        if matches!(bytes.get(self.position), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(bytes.get(self.position), Some(b'+' | b'-')) {
                self.position += 1;
            }
            self.digits()?;
        }
        Ok(Token::Number(&self.source[start..self.position]))
    }

    fn digits(&mut self) -> Result<(), TokenError> {
        let start = self.position;
        while self
            .source
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_digit)
        {
            self.position += 1;
        }
        if self.position == start {
            Err(self.error("expected digit"))
        } else {
            Ok(())
        }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Result<Token<'a>, TokenError>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.source.as_bytes();
        while matches!(bytes.get(self.position), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
        let byte = *bytes.get(self.position)?;
        let single = match byte {
            b'[' => Some(Token::ArrayStart),
            b']' => Some(Token::ArrayEnd),
            b'{' => Some(Token::ObjectStart),
            b'}' => Some(Token::ObjectEnd),
            b':' => Some(Token::Colon),
            b',' => Some(Token::Comma),
            _ => None,
        };
        if let Some(token) = single {
            self.position += 1;
            return Some(Ok(token));
        }
        if byte == b'"' {
            return Some(self.string());
        }
        if byte == b'-' || byte.is_ascii_digit() {
            return Some(self.number());
        }
        for (literal, token) in [
            ("null", Token::Null),
            ("true", Token::Bool(true)),
            ("false", Token::Bool(false)),
        ] {
            if self.source[self.position..].starts_with(literal) {
                self.position += literal.len();
                return Some(Ok(token));
            }
        }
        Some(Err(self.error("unexpected JSON token")))
    }
}

pub fn write_escaped_string(output: &mut impl core::fmt::Write, value: &str) -> core::fmt::Result {
    output.write_char('"')?;
    for character in value.chars() {
        match character {
            '"' => output.write_str("\\\"")?,
            '\\' => output.write_str("\\\\")?,
            '\u{08}' => output.write_str("\\b")?,
            '\u{0c}' => output.write_str("\\f")?,
            '\n' => output.write_str("\\n")?,
            '\r' => output.write_str("\\r")?,
            '\t' => output.write_str("\\t")?,
            character if character < '\u{20}' => write!(output, "\\u{:04x}", character as u32)?,
            character => output.write_char(character)?,
        }
    }
    output.write_char('"')
}

use core::fmt::{self, Write};
use core::ops::{Index, IndexMut};
use mrml_runtime::{OrderedMap as BTreeMap, Text as String, Vector as Vec};

trait ToString {
    fn to_string(&self) -> String;
}

impl<T: fmt::Display + ?Sized> ToString for T {
    fn to_string(&self) -> String {
        let mut output = String::new();
        write!(output, "{self}").expect("MRML allocation failed");
        output
    }
}

fn formatted(arguments: fmt::Arguments<'_>) -> String {
    let mut output = String::new();
    output.write_fmt(arguments).expect("MRML allocation failed");
    output
}

pub type Map = BTreeMap<String, Value>;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl PartialEq<&str> for Value {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}

impl PartialEq<String> for Value {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}

pub fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

pub fn array<const N: usize>(values: [Value; N]) -> Value {
    Value::Array(values.into())
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}
impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::Number(value.to_string())
    }
}
impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::Number(value.to_string())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(value: Option<T>) -> Self {
        value.map(Into::into).unwrap_or(Self::Null)
    }
}

impl Value {
    pub fn text(value: impl AsRef<str>) -> Self {
        Self::String(value.as_ref().into())
    }

    pub fn optional_text<T: AsRef<str>>(value: Option<T>) -> Self {
        value.map(Self::text).unwrap_or(Self::Null)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.as_object_mut()?.get_mut(key)
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        if let Self::Object(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, Value>> {
        if let Self::Object(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        if let Self::Array(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        if let Self::Array(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        if let Self::Number(value) = self {
            value.parse().ok()
        } else {
            None
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        if let Self::Number(value) = self {
            value.parse().ok()
        } else {
            None
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        if let Self::Number(value) = self {
            value.parse().ok()
        } else {
            None
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }
}

static NULL: Value = Value::Null;

impl Index<&str> for Value {
    type Output = Value;
    fn index(&self, key: &str) -> &Self::Output {
        self.get(key).unwrap_or(&NULL)
    }
}

impl IndexMut<&str> for Value {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        if !matches!(self, Self::Object(_)) {
            *self = Self::Object(BTreeMap::new());
        }
        let object = self.as_object_mut().unwrap();
        if object.get(key).is_none() {
            object.insert(key.into(), Self::Null);
        }
        object.get_mut(key).unwrap()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&stringify(self))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    message: String,
    offset: usize,
}

impl Error {
    fn new(message: impl fmt::Display, offset: usize) -> Self {
        Self {
            message: formatted(format_args!("{message}")),
            offset,
        }
    }

    pub fn message(message: impl fmt::Display) -> Self {
        Self::new(message, 0)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} at byte {}", self.message, self.offset)
    }
}

impl core::error::Error for Error {}

pub fn parse(source: &str) -> Result<Value, Error> {
    let mut parser = Parser {
        bytes: source.as_bytes(),
        position: 0,
    };
    let value = parser.value()?;
    parser.whitespace();
    if parser.position != parser.bytes.len() {
        return Err(Error::new("trailing characters", parser.position));
    }
    Ok(value)
}

pub trait FromJson: Sized {
    fn from_json(value: Value) -> Result<Self, Error>;
}

impl FromJson for Value {
    fn from_json(value: Value) -> Result<Self, Error> {
        Ok(value)
    }
}

pub fn from_str<T: FromJson>(source: &str) -> Result<T, Error> {
    T::from_json(parse(source)?)
}

pub fn from_slice<T: FromJson>(source: &[u8]) -> Result<T, Error> {
    from_str(core::str::from_utf8(source).map_err(|_| Error::message("JSON is not UTF-8"))?)
}

pub fn to_string(value: &Value) -> Result<String, Error> {
    Ok(stringify(value))
}

pub fn string(value: &str) -> String {
    stringify(&Value::String(value.into()))
}

#[macro_export]
macro_rules! json {
    (null) => { $crate::Value::Null };
    ([ $($value:tt),* $(,)? ]) => { $crate::array([$($crate::json!($value)),*]) };
    ({ $($key:literal : $value:tt),* $(,)? }) => {
        $crate::object([$(($key, $crate::json!($value))),*])
    };
    ($value:expr) => { $crate::Value::from($value) };
}

pub fn stringify(value: &Value) -> String {
    let mut output = String::new();
    write_value(value, &mut output);
    output
}

fn write_value(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(value),
        Value::String(value) => write_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_value(value, output);
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_string(key, output);
                output.push(':');
                write_value(value, output);
            }
            output.push('}');
        }
    }
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character < '\u{20}' => {
                write!(output, "\\u{:04x}", character as u32).expect("MRML allocation failed")
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Parser<'_> {
    fn value(&mut self) -> Result<Value, Error> {
        self.whitespace();
        match self.peek() {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => Ok(Value::Number(self.number()?)),
            _ => Err(Error::new("expected JSON value", self.position)),
        }
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.position += 1;
        let mut values = Vec::new();
        self.whitespace();
        if self.take(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.whitespace();
            if self.take(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(Value::Array(values))
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.position += 1;
        let mut values = BTreeMap::new();
        self.whitespace();
        if self.take(b'}') {
            return Ok(Value::Object(values));
        }
        loop {
            self.whitespace();
            if self.peek() != Some(b'"') {
                return Err(Error::new("expected object key", self.position));
            }
            let key = self.string()?;
            self.whitespace();
            self.expect(b':')?;
            values.insert(key, self.value()?);
            self.whitespace();
            if self.take(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(Value::Object(values))
    }

    fn string(&mut self) -> Result<String, Error> {
        self.expect(b'"')?;
        let mut output = String::new();
        let mut start = self.position;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.push_utf8(start, self.position, &mut output)?;
                    self.position += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.push_utf8(start, self.position, &mut output)?;
                    self.position += 1;
                    let escaped = self
                        .peek()
                        .ok_or_else(|| Error::new("unterminated escape", self.position))?;
                    self.position += 1;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{08}'),
                        b'f' => output.push('\u{0c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => self.unicode_escape(&mut output)?,
                        _ => return Err(Error::new("invalid escape", self.position - 1)),
                    }
                    start = self.position;
                }
                0..=0x1f => {
                    return Err(Error::new("control character in string", self.position));
                }
                _ => self.position += 1,
            }
        }
        Err(Error::new("unterminated string", self.position))
    }

    fn unicode_escape(&mut self, output: &mut String) -> Result<(), Error> {
        let first = self.hex4()?;
        let scalar = if (0xd800..=0xdbff).contains(&first) {
            if !self.take(b'\\') || !self.take(b'u') {
                return Err(Error::new("missing low surrogate", self.position));
            }
            let second = self.hex4()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(Error::new("invalid low surrogate", self.position - 4));
            }
            0x10000 + (((first - 0xd800) as u32) << 10) + (second - 0xdc00) as u32
        } else {
            first as u32
        };
        output.push(
            char::from_u32(scalar)
                .ok_or_else(|| Error::new("invalid unicode scalar", self.position))?,
        );
        Ok(())
    }

    fn hex4(&mut self) -> Result<u16, Error> {
        let mut value = 0u16;
        for _ in 0..4 {
            let byte = self
                .peek()
                .ok_or_else(|| Error::new("short unicode escape", self.position))?;
            self.position += 1;
            value = value * 16
                + match byte {
                    b'0'..=b'9' => (byte - b'0') as u16,
                    b'a'..=b'f' => (byte - b'a' + 10) as u16,
                    b'A'..=b'F' => (byte - b'A' + 10) as u16,
                    _ => return Err(Error::new("invalid hex digit", self.position - 1)),
                };
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<String, Error> {
        let start = self.position;
        self.take(b'-');
        if self.take(b'0') {
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(Error::new("leading zero", self.position));
            }
        } else {
            self.digits()?;
        }
        if self.take(b'.') {
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            self.digits()?;
        }
        Ok(core::str::from_utf8(&self.bytes[start..self.position])
            .unwrap()
            .into())
    }

    fn digits(&mut self) -> Result<(), Error> {
        let start = self.position;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        if start == self.position {
            Err(Error::new("expected digit", self.position))
        } else {
            Ok(())
        }
    }

    fn push_utf8(&self, start: usize, end: usize, output: &mut String) -> Result<(), Error> {
        output.push_str(
            core::str::from_utf8(&self.bytes[start..end])
                .map_err(|_| Error::new("invalid UTF-8", start))?,
        );
        Ok(())
    }

    fn literal(&mut self, literal: &[u8]) -> Result<(), Error> {
        if self.bytes.get(self.position..self.position + literal.len()) == Some(literal) {
            self.position += literal.len();
            Ok(())
        } else {
            Err(Error::new("invalid literal", self.position))
        }
    }
    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, byte: u8) -> Result<(), Error> {
        if self.take(byte) {
            Ok(())
        } else {
            Err(Error::new(
                formatted(format_args!("expected '{}'", byte as char)),
                self.position,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nested_values_and_unicode() {
        let source = r#"{"array":[null,true,-12.5e2,"hi\n\u263a\ud83d\ude80"]}"#;
        let value = parse(source).unwrap();
        assert_eq!(parse(&stringify(&value)).unwrap(), value);
        assert_eq!(
            value.get("array").unwrap().as_array().unwrap()[3].as_str(),
            Some("hi\n☺🚀")
        );
    }

    #[test]
    fn rejects_trailing_data_bad_numbers_and_bad_surrogates() {
        for source in ["null x", "01", "1.", r#""\ud800""#] {
            assert!(parse(source).is_err(), "accepted {source}");
        }
    }
}
#[cfg(test)]
mod portable_tests {
    use super::*;
    use core::fmt::Write;

    struct Buffer {
        bytes: [u8; 96],
        len: usize,
    }

    impl Write for Buffer {
        fn write_str(&mut self, value: &str) -> core::fmt::Result {
            let end = self.len + value.len();
            if end > self.bytes.len() {
                return Err(core::fmt::Error);
            }
            self.bytes[self.len..end].copy_from_slice(value.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    #[test]
    fn tokenizes_without_allocating() {
        let tokens = Tokens::new(r#"{"name":"MRML\n","fast":true,"rate":60.2}"#)
            .collect::<std::vec::Vec<_>>();
        assert_eq!(tokens.len(), 13);
        assert_eq!(tokens[0], Ok(Token::ObjectStart));
        assert_eq!(tokens[2], Ok(Token::Colon));
        assert_eq!(tokens[3], Ok(Token::String("MRML\\n")));
        assert_eq!(tokens[11], Ok(Token::Number("60.2")));
        assert_eq!(tokens[12], Ok(Token::ObjectEnd));
    }

    #[test]
    fn writes_escaped_strings_into_caller_storage() {
        let mut output = Buffer {
            bytes: [0; 96],
            len: 0,
        };
        write_escaped_string(&mut output, "MRML\n✓").unwrap();
        assert_eq!(&output.bytes[..output.len], "\"MRML\\n✓\"".as_bytes());
    }
}
