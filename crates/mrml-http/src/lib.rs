#![no_std]

use core::fmt;
use mrml_runtime::{Text, Vector};
use mrml_tls::TlsClientStream;

const MAX_HEADER: usize = 64 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpError {
    InvalidUrl,
    UnsupportedScheme,
    InvalidResponse,
    HeaderTooLarge,
    BodyTooLarge,
    RedirectLimit,
    Tls,
    Io,
    Allocation,
}
impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidUrl => "invalid URL",
            Self::UnsupportedScheme => "only HTTPS URLs are supported",
            Self::InvalidResponse => "invalid HTTP response",
            Self::HeaderTooLarge => "HTTP headers exceed limit",
            Self::BodyTooLarge => "HTTP body exceeds limit",
            Self::RedirectLimit => "HTTP redirect limit exceeded",
            Self::Tls => "HTTPS transport failed",
            Self::Io => "HTTP I/O failed",
            Self::Allocation => "HTTP allocation failed",
        })
    }
}
impl core::error::Error for HttpError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Url {
    pub host: Text,
    pub port: u16,
    pub target: Text,
}
impl Url {
    pub fn parse(input: &str) -> Result<Self, HttpError> {
        let rest = input
            .strip_prefix("https://")
            .ok_or(if input.contains("://") {
                HttpError::UnsupportedScheme
            } else {
                HttpError::InvalidUrl
            })?;
        let split = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..split];
        if authority.is_empty() || authority.contains(['@', '[', ']']) {
            return Err(HttpError::InvalidUrl);
        }
        let (mut host, mut port) = (authority, 443);
        if let Some(colon) = authority.rfind(':') {
            host = &authority[..colon];
            port = authority[colon + 1..]
                .parse()
                .map_err(|_| HttpError::InvalidUrl)?;
            if port == 0 {
                return Err(HttpError::InvalidUrl);
            }
        }
        if host.is_empty()
            || !host.is_ascii()
            || host.bytes().any(|b| b.is_ascii_control() || b == b' ')
        {
            return Err(HttpError::InvalidUrl);
        }
        let path = &rest[split..];
        if path.contains('#') {
            return Err(HttpError::InvalidUrl);
        }
        let target = if path.is_empty() {
            Text::from("/")
        } else if path.starts_with('?') {
            let mut t = Text::from("/");
            t.push_str(path);
            t
        } else {
            path.into()
        };
        Ok(Self {
            host: Text::from(host).to_ascii_lowercase(),
            port,
            target,
        })
    }
    pub fn authority(&self) -> Text {
        if self.port == 443 {
            self.host.clone()
        } else {
            mrml_runtime::mrml_format!("{}:{}", self.host, self.port)
        }
    }
    pub fn resolve(&self, location: &str) -> Result<Self, HttpError> {
        if location.starts_with("https://") {
            return Self::parse(location);
        }
        if location.contains("://") {
            return Err(HttpError::UnsupportedScheme);
        }
        let target = if location.starts_with('/') {
            location.into()
        } else {
            let base = self.target.split('?').next().unwrap_or("/");
            let directory = base.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            mrml_runtime::mrml_format!("{}/{}", directory, location)
        };
        Ok(Self {
            host: self.host.clone(),
            port: self.port,
            target,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub name: Text,
    pub value: Text,
}
enum Framing {
    Length(u64),
    Chunked { remaining: usize, finished: bool },
    Close,
}
pub struct Response {
    stream: TlsClientStream,
    pending: Vector<u8>,
    position: usize,
    pub status: u16,
    pub headers: Vector<Header>,
    framing: Framing,
    read: u64,
}
impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|h| h.name.eq_ignore_ascii_case(name))
            .map(|h| h.value.as_str())
    }
    fn raw(&mut self, out: &mut [u8]) -> Result<usize, HttpError> {
        if self.position < self.pending.len() {
            let n = out.len().min(self.pending.len() - self.position);
            out[..n].copy_from_slice(&self.pending[self.position..self.position + n]);
            self.position += n;
            if self.position == self.pending.len() {
                self.pending.clear();
                self.position = 0
            }
            return Ok(n);
        }
        self.stream.read(out).map_err(|_| HttpError::Tls)
    }
    fn raw_exact(&mut self, mut out: &mut [u8]) -> Result<(), HttpError> {
        while !out.is_empty() {
            let n = self.raw(out)?;
            if n == 0 {
                return Err(HttpError::InvalidResponse);
            }
            out = &mut out[n..]
        }
        Ok(())
    }
    fn line(&mut self) -> Result<Text, HttpError> {
        let mut line = Vector::new();
        loop {
            let mut byte = [0];
            self.raw_exact(&mut byte)?;
            line.try_push(byte[0]).map_err(|_| HttpError::Allocation)?;
            if line.len() > 8192 {
                return Err(HttpError::InvalidResponse);
            }
            if line.ends_with(b"\r\n") {
                line.truncate(line.len() - 2);
                return Text::try_from_str(
                    core::str::from_utf8(&line).map_err(|_| HttpError::InvalidResponse)?,
                )
                .map_err(|_| HttpError::Allocation);
            }
        }
    }
    pub fn read(&mut self, out: &mut [u8]) -> Result<usize, HttpError> {
        if out.is_empty() {
            return Ok(0);
        }
        let mut framing = core::mem::replace(&mut self.framing, Framing::Close);
        let result = self.read_framed(out, &mut framing);
        self.framing = framing;
        result
    }
    fn read_framed(&mut self, out: &mut [u8], framing: &mut Framing) -> Result<usize, HttpError> {
        match framing {
            Framing::Length(total) => {
                let left = total.saturating_sub(self.read);
                if left == 0 {
                    return Ok(0);
                }
                let limit = out.len().min(left as usize);
                let n = self.raw(&mut out[..limit])?;
                if n == 0 {
                    return Err(HttpError::InvalidResponse);
                }
                self.read += n as u64;
                Ok(n)
            }
            Framing::Close => {
                let n = self.raw(out)?;
                self.read += n as u64;
                Ok(n)
            }
            Framing::Chunked {
                remaining,
                finished,
            } => {
                if *finished {
                    return Ok(0);
                }
                if *remaining == 0 {
                    let line = self.line()?;
                    let size_text = line.split(';').next().ok_or(HttpError::InvalidResponse)?;
                    let size = usize::from_str_radix(size_text.trim(), 16)
                        .map_err(|_| HttpError::InvalidResponse)?;
                    if size > 1 << 30 {
                        return Err(HttpError::BodyTooLarge);
                    }
                    if size == 0 {
                        loop {
                            if self.line()?.is_empty() {
                                break;
                            }
                        }
                        *finished = true;
                        return Ok(0);
                    }
                    *remaining = size
                }
                let limit = out.len().min(*remaining);
                let n = self.raw(&mut out[..limit])?;
                if n == 0 {
                    return Err(HttpError::InvalidResponse);
                }
                *remaining -= n;
                self.read += n as u64;
                if *remaining == 0 {
                    let mut end = [0; 2];
                    self.raw_exact(&mut end)?;
                    if end != *b"\r\n" {
                        return Err(HttpError::InvalidResponse);
                    }
                }
                Ok(n)
            }
        }
    }
    pub fn read_to_end(&mut self, limit: usize) -> Result<Vector<u8>, HttpError> {
        let mut out = Vector::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = self.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            if out.len().checked_add(n).is_none_or(|v| v > limit) {
                return Err(HttpError::BodyTooLarge);
            }
            out.try_extend_from_slice(&buffer[..n])
                .map_err(|_| HttpError::Allocation)?
        }
        Ok(out)
    }
}

pub struct Client {
    user_agent: Text,
}
impl Client {
    pub fn new() -> Self {
        Self {
            user_agent: "mrml/0.4".into(),
        }
    }
    pub fn user_agent(mut self, value: &str) -> Self {
        self.user_agent = value.into();
        self
    }
    pub fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<Response, HttpError> {
        let url = Url::parse(url)?;
        self.request(&url, "GET", headers)
    }
    pub fn get_follow(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        redirects: usize,
    ) -> Result<Response, HttpError> {
        let mut current = Url::parse(url)?;
        for _ in 0..=redirects {
            let response = self.request(&current, "GET", headers)?;
            if matches!(response.status, 301 | 302 | 303 | 307 | 308) {
                let location = Text::from(
                    response
                        .header("location")
                        .ok_or(HttpError::InvalidResponse)?,
                );
                current = current.resolve(&location)?;
                continue;
            }
            return Ok(response);
        }
        Err(HttpError::RedirectLimit)
    }
    fn request(
        &self,
        url: &Url,
        method: &str,
        headers: &[(&str, &str)],
    ) -> Result<Response, HttpError> {
        let mut stream =
            TlsClientStream::connect(&url.host, url.port).map_err(|_| HttpError::Tls)?;
        let mut request = mrml_runtime::mrml_format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nAccept: */*\r\nConnection: close\r\n",
            method,
            url.target,
            url.authority(),
            self.user_agent
        );
        for (name, value) in headers {
            if !valid_header(name) || !valid_value(value) {
                return Err(HttpError::InvalidUrl);
            }
            request.push_str(name);
            request.push_str(": ");
            request.push_str(value);
            request.push_str("\r\n")
        }
        request.push_str("\r\n");
        stream
            .write_all(request.as_bytes())
            .map_err(|_| HttpError::Tls)?;
        let mut received = Vector::new();
        let header_end = loop {
            if let Some(p) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                break p + 4;
            }
            if received.len() >= MAX_HEADER {
                return Err(HttpError::HeaderTooLarge);
            }
            let mut buffer = [0u8; 4096];
            let n = stream.read(&mut buffer).map_err(|_| HttpError::Tls)?;
            if n == 0 {
                return Err(HttpError::InvalidResponse);
            }
            received
                .try_extend_from_slice(&buffer[..n])
                .map_err(|_| HttpError::Allocation)?
        };
        let head = core::str::from_utf8(&received[..header_end])
            .map_err(|_| HttpError::InvalidResponse)?;
        let mut lines = head[..head.len() - 4].split("\r\n");
        let status_line = lines.next().ok_or(HttpError::InvalidResponse)?;
        let mut parts = status_line.splitn(3, ' ');
        if parts.next() != Some("HTTP/1.1") {
            return Err(HttpError::InvalidResponse);
        }
        let status = parts
            .next()
            .ok_or(HttpError::InvalidResponse)?
            .parse()
            .map_err(|_| HttpError::InvalidResponse)?;
        if !(100..=599).contains(&status) {
            return Err(HttpError::InvalidResponse);
        }
        let mut parsed = Vector::new();
        for line in lines {
            if line.starts_with([' ', '\t']) {
                return Err(HttpError::InvalidResponse);
            }
            let (name, value) = line.split_once(':').ok_or(HttpError::InvalidResponse)?;
            if !valid_header(name) || !valid_value(value) {
                return Err(HttpError::InvalidResponse);
            }
            parsed
                .try_push(Header {
                    name: Text::from(name).to_ascii_lowercase(),
                    value: value.trim().into(),
                })
                .map_err(|_| HttpError::Allocation)?
        }
        let transfer = header_value(&parsed, "transfer-encoding").map(Text::from);
        let mut content_length = None;
        for header in parsed.iter().filter(|h| h.name == "content-length") {
            let value: u64 = header
                .value
                .parse()
                .map_err(|_| HttpError::InvalidResponse)?;
            if content_length.is_some_and(|old| old != value) {
                return Err(HttpError::InvalidResponse);
            }
            content_length = Some(value)
        }
        let framing = if let Some(value) = transfer {
            if !value.eq_ignore_ascii_case("chunked") {
                return Err(HttpError::InvalidResponse);
            }
            Framing::Chunked {
                remaining: 0,
                finished: false,
            }
        } else if let Some(value) = content_length {
            Framing::Length(value)
        } else {
            Framing::Close
        };
        let mut pending = Vector::new();
        pending
            .try_extend_from_slice(&received[header_end..])
            .map_err(|_| HttpError::Allocation)?;
        Ok(Response {
            stream,
            pending,
            position: 0,
            status,
            headers: parsed,
            framing,
            read: 0,
        })
    }
}
impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}
fn header_value<'a>(headers: &'a [Header], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|h| h.name == name)
        .map(|h| h.value.as_str())
}
fn valid_header(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}
fn valid_value(value: &str) -> bool {
    value.bytes().all(|b| b == b'\t' || b >= 0x20 && b != 0x7f) && !value.contains(['\r', '\n'])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_and_resolves_https_urls() {
        let url = Url::parse("https://Example.COM:8443/a/b?q=1").unwrap();
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 8443);
        assert_eq!(url.resolve("next").unwrap().target, "/a/next");
        assert_eq!(
            Url::parse("http://example.com"),
            Err(HttpError::UnsupportedScheme)
        );
    }
    #[test]
    fn live_get_when_configured() {
        let Some(url) = mrml_runtime::environment_variable("MRML_HTTP_LIVE_URL") else {
            return;
        };
        let mut response = Client::new().get_follow(&url, &[], 5).unwrap();
        assert_eq!(response.status, 200);
        assert!(!response.read_to_end(2 * 1024 * 1024).unwrap().is_empty());
    }
}
