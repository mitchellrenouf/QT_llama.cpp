use crate::Tool;
use anyhow::{Result, anyhow};
use mrml_runtime::{OnceCell, Shared, SpinMutex};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
static BROWSER_INSTANCE: OnceCell<Shared<SpinMutex<EdgeController>>> = OnceCell::new();

struct CdpSocket {
    stream: TcpStream,
    next_id: u64,
}
impl CdpSocket {
    fn connect(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("ws://")
            .ok_or_else(|| anyhow!("Unsupported DevTools URL: {}", url))?;
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let stream = TcpStream::connect(authority)?;
        stream.set_read_timeout(Some(Duration::from_secs(15)))?;
        stream.set_write_timeout(Some(Duration::from_secs(15)))?;
        let mut s = Self { stream, next_id: 1 };
        write!(
            s.stream,
            "GET /{} HTTP/1.1\r\nHost: {}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: bXJtbC1lZGdlLWNkcA==\r\nSec-WebSocket-Version: 13\r\n\r\n",
            path, authority
        )?;
        let h = read_http_header(&mut s.stream)?;
        if !h.starts_with("HTTP/1.1 101") {
            return Err(anyhow!(
                "DevTools WebSocket upgrade failed: {}",
                h.lines().next().unwrap_or("empty response")
            ));
        }
        Ok(s)
    }
    fn command(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_frame(
            1,
            json!({"id":id,"method":method,"params":params})
                .to_string()
                .as_bytes(),
        )?;
        loop {
            let (op, payload) = self.read_message()?;
            if op == 8 {
                return Err(anyhow!("Edge closed DevTools"));
            }
            if op == 9 {
                self.write_frame(10, &payload)?;
                continue;
            }
            if op != 1 {
                continue;
            }
            let v: Value = serde_json::from_slice(&payload)?;
            if v.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(e) = v.get("error") {
                return Err(anyhow!("CDP {} failed: {}", method, e));
            }
            return Ok(v.get("result").cloned().unwrap_or(Value::Null));
        }
    }
    fn write_frame(&mut self, op: u8, p: &[u8]) -> Result<()> {
        let mut f = Vec::with_capacity(p.len() + 14);
        f.push(0x80 | op);
        if p.len() < 126 {
            f.push(0x80 | p.len() as u8)
        } else if p.len() <= 65535 {
            f.extend([0x80 | 126]);
            f.extend_from_slice(&(p.len() as u16).to_be_bytes())
        } else {
            f.extend([0x80 | 127]);
            f.extend_from_slice(&(p.len() as u64).to_be_bytes())
        }
        let k = (self.next_id as u32).wrapping_mul(0x9e3779b9).to_be_bytes();
        f.extend_from_slice(&k);
        f.extend(p.iter().enumerate().map(|(i, b)| b ^ k[i % 4]));
        self.stream.write_all(&f)?;
        Ok(())
    }
    fn read_message(&mut self) -> Result<(u8, Vec<u8>)> {
        let mut all = Vec::new();
        let mut first = 0;
        loop {
            let mut h = [0; 2];
            self.stream.read_exact(&mut h)?;
            let fin = h[0] & 128 != 0;
            let op = h[0] & 15;
            if first == 0 && op != 0 {
                first = op
            }
            let mut n = (h[1] & 127) as u64;
            if n == 126 {
                let mut b = [0; 2];
                self.stream.read_exact(&mut b)?;
                n = u16::from_be_bytes(b) as u64
            } else if n == 127 {
                let mut b = [0; 8];
                self.stream.read_exact(&mut b)?;
                n = u64::from_be_bytes(b)
            }
            if n > 64 * 1024 * 1024 {
                return Err(anyhow!("Oversized DevTools message"));
            }
            let masked = h[1] & 128 != 0;
            let mut k = [0; 4];
            if masked {
                self.stream.read_exact(&mut k)?
            }
            let start = all.len();
            all.resize(start + n as usize, 0);
            self.stream.read_exact(&mut all[start..])?;
            if masked {
                for (i, b) in all[start..].iter_mut().enumerate() {
                    *b ^= k[i % 4]
                }
            }
            if fin {
                return Ok((first, all));
            }
        }
    }
}

pub struct EdgeController {
    child: Child,
    port: u16,
    browser: CdpSocket,
    page: Option<CdpSocket>,
    profile: PathBuf,
}
impl EdgeController {
    pub fn ensure_latest_and_launch() -> Result<Self> {
        Self::launch()
    }
    fn launch() -> Result<Self> {
        let exe = find_browser()
            .ok_or_else(|| anyhow!("No installed Edge/Chrome/Chromium found; set BROWSER_EXE"))?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let profile =
            std::env::temp_dir().join(format!("mrml-edge-{}-{}", std::process::id(), port));
        mrml_runtime::create_dir_all(
            profile
                .to_str()
                .ok_or_else(|| anyhow!("Browser profile path is not valid UTF-8"))?,
        )?;
        let child = Command::new(&exe)
            .args([
                "--headless=new",
                "--no-first-run",
                "--no-default-browser-check",
            ])
            .arg(format!("--remote-debugging-port={}", port))
            .arg(format!("--user-data-dir={}", profile.display()))
            .arg("about:blank")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("Failed to launch '{}': {}", exe.display(), e))?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Ok(version) = http_json(port, "GET", "/json/version") {
                if let Some(ws) = version["webSocketDebuggerUrl"].as_str() {
                    if let Ok(browser) = CdpSocket::connect(ws) {
                        return Ok(Self {
                            child,
                            port,
                            browser,
                            page: None,
                            profile,
                        });
                    }
                }
            }
            crate::platform::sleep_millis(50)
        }
        Err(anyhow!("Edge did not expose DevTools on port {}", port))
    }
    pub fn get_or_create_page(&mut self, url: Option<&str>) -> Result<&mut Self> {
        if self.page.is_none() {
            let deadline = Instant::now() + Duration::from_secs(3);
            let target = loop {
                match http_json(self.port, "PUT", "/json/new?about%3Ablank") {
                    Ok(target) => break target,
                    Err(error) if Instant::now() < deadline => {
                        let _ = error;
                        crate::platform::sleep_millis(50);
                    }
                    Err(error) => return Err(error),
                }
            };
            let ws = target["webSocketDebuggerUrl"]
                .as_str()
                .ok_or_else(|| anyhow!("Missing page debugger URL"))?;
            let mut page = CdpSocket::connect(ws)?;
            page.command("Page.enable", json!({}))?;
            page.command("Runtime.enable", json!({}))?;
            self.page = Some(page)
        }
        if let Some(u) = url {
            self.navigate(u)?
        }
        Ok(self)
    }
    fn cdp(&mut self, m: &str, p: Value) -> Result<Value> {
        self.page
            .as_mut()
            .ok_or_else(|| anyhow!("No active browser page"))?
            .command(m, p)
    }
    fn navigate(&mut self, url: &str) -> Result<()> {
        self.cdp("Page.navigate", json!({"url":url}))?;
        let end = Instant::now() + Duration::from_secs(15);
        while Instant::now() < end {
            if self
                .eval("document.readyState")?
                .as_str()
                .is_some_and(|s| s == "complete" || s == "interactive")
            {
                break;
            }
            crate::platform::sleep_millis(40)
        }
        Ok(())
    }
    fn eval(&mut self, e: &str) -> Result<Value> {
        Ok(self.cdp(
            "Runtime.evaluate",
            json!({"expression":e,"returnByValue":true,"awaitPromise":true}),
        )?["result"]["value"]
            .clone())
    }
    pub fn content(&mut self) -> Result<String> {
        Ok(self
            .eval("document.documentElement.outerHTML")?
            .as_str()
            .unwrap_or_default()
            .to_string())
    }
    fn title(&mut self) -> Result<String> {
        Ok(self
            .eval("document.title")?
            .as_str()
            .unwrap_or_default()
            .to_string())
    }
    fn url(&mut self) -> Result<String> {
        Ok(self
            .eval("location.href")?
            .as_str()
            .unwrap_or_default()
            .to_string())
    }
}
impl Drop for EdgeController {
    fn drop(&mut self) {
        let _ = self.browser.command("Browser.close", json!({}));
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.profile);
    }
}

fn find_browser() -> Option<PathBuf> {
    if let Some(p) = mrml_runtime::environment_variable("BROWSER_EXE")
        .map(|value| PathBuf::from(value.as_str()))
        .filter(|p| crate::platform::path_is_file(p))
    {
        return Some(p);
    }
    let mut c = Vec::new();
    for root in [
        mrml_runtime::environment_variable("ProgramFiles(x86)"),
        mrml_runtime::environment_variable("ProgramFiles"),
        mrml_runtime::environment_variable("LOCALAPPDATA"),
    ]
    .into_iter()
    .flatten()
    {
        let r = PathBuf::from(root.as_str());
        c.push(r.join("Microsoft/Edge/Application/msedge.exe"));
        c.push(r.join("Google/Chrome/Application/chrome.exe"))
    }
    c.extend([
        PathBuf::from("/usr/bin/microsoft-edge"),
        PathBuf::from("/usr/bin/google-chrome"),
        PathBuf::from("/usr/bin/chromium"),
    ]);
    c.into_iter().find(|p| crate::platform::path_is_file(p))
}
fn read_http_header(s: &mut TcpStream) -> Result<String> {
    let mut v = Vec::new();
    let mut b = [0];
    while v.len() < 65536 {
        s.read_exact(&mut b)?;
        v.push(b[0]);
        if v.ends_with(b"\r\n\r\n") {
            return Ok(String::from_utf8_lossy(&v).into_owned());
        }
    }
    Err(anyhow!("HTTP header too large"))
}
fn http_json(port: u16, method: &str, path: &str) -> Result<Value> {
    let mut s = TcpStream::connect(("127.0.0.1", port))?;
    s.set_read_timeout(Some(Duration::from_secs(2)))?;
    write!(
        s,
        "{} {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
        method, path, port
    )?;
    let h = read_http_header(&mut s)?;
    if !h.starts_with("HTTP/1.1 200") {
        return Err(anyhow!(
            "DevTools HTTP failed: {}",
            h.lines().next().unwrap_or("empty response")
        ));
    }
    let length = h
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .ok_or_else(|| anyhow!("DevTools response omitted Content-Length"))?;
    let mut body = vec![0; length];
    s.read_exact(&mut body)?;
    Ok(serde_json::from_slice(&body)?)
}
pub async fn get_browser_controller() -> Result<Shared<SpinMutex<EdgeController>>> {
    if let Some(controller) = BROWSER_INSTANCE.get() {
        return Ok(controller.clone());
    }
    let controller = Shared::new(SpinMutex::new(EdgeController::ensure_latest_and_launch()?));
    let _ = BROWSER_INSTANCE.set(controller.clone());
    Ok(BROWSER_INSTANCE.get().cloned().unwrap_or(controller))
}

pub struct BrowserOpenTool;
impl Tool for BrowserOpenTool {
    fn name(&self) -> &'static str {
        "browser_open"
    }
    fn description(&self) -> &'static str {
        "Open a URL in the installed browser running headlessly."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"url":{"type":"string"}},"required":["url"]})
    }
    async fn execute(&self, _: &Path, a: Value) -> Result<String> {
        let u = a["url"].as_str().ok_or_else(|| anyhow!("Missing url"))?;
        let x = get_browser_controller().await?;
        let mut c = x.lock();
        c.get_or_create_page(Some(u))?;
        Ok(format!(
            "Opened '{}' in headless Edge (Title: '{}', URL: '{}').",
            u,
            c.title()?,
            c.url()?
        ))
    }
}
pub struct BrowserGetContentTool;
impl Tool for BrowserGetContentTool {
    fn name(&self) -> &'static str {
        "browser_get_content"
    }
    fn description(&self) -> &'static str {
        "Read DOM content from the active headless browser page or URL."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"url":{"type":"string"}}})
    }
    async fn execute(&self, _: &Path, a: Value) -> Result<String> {
        let x = get_browser_controller().await?;
        let mut c = x.lock();
        c.get_or_create_page(a["url"].as_str())?;
        let title = c.title()?;
        let text = crate::html::visible_text(&c.content()?);
        let d = if text.len() > 6000 {
            format!(
                "{}... (truncated)",
                crate::markdown::truncate_utf8(&text, 6000)
            )
        } else {
            text.to_string()
        };
        Ok(format!(
            "[Headless Edge DOM Content - '{}']:\n{}",
            title,
            if d.trim().is_empty() {
                "(No readable text found)"
            } else {
                d.trim()
            }
        ))
    }
}
pub struct BrowserScreenshotTool;
impl Tool for BrowserScreenshotTool {
    fn name(&self) -> &'static str {
        "browser_screenshot"
    }
    fn description(&self) -> &'static str {
        "Capture a JPEG screenshot from the active headless browser."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    async fn execute(&self, r: &Path, _: Value) -> Result<String> {
        let x = get_browser_controller().await?;
        let mut c = x.lock();
        c.get_or_create_page(None)?;
        let data = c.cdp(
            "Page.captureScreenshot",
            json!({"format":"jpeg","quality":75}),
        )?["data"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let bytes = crate::encoding::base64_decode(&data)
            .map_err(|e| anyhow!("Invalid screenshot: {}", e))?;
        let dir = r.join(".mrml/screenshots");
        mrml_runtime::create_dir_all(
            dir.to_str()
                .ok_or_else(|| anyhow!("Screenshot directory is not valid UTF-8"))?,
        )?;
        let p = dir.join(format!(
            "browser_screenshot_{}.jpg",
            crate::platform::local_timestamp_string()
        ));
        mrml_runtime::write_file(
            p.to_str().ok_or_else(|| anyhow!("Screenshot path is not valid UTF-8"))?,
            &bytes,
        )?;
        Ok(format!(
            "Screenshot captured at '{}' ({} bytes).\nDATA_URI:data:image/jpeg;base64,{}",
            p.display(),
            bytes.len(),
            data
        ))
    }
}
pub struct BrowserClickElementTool;
impl Tool for BrowserClickElementTool {
    fn name(&self) -> &'static str {
        "browser_click_element"
    }
    fn description(&self) -> &'static str {
        "Click an element by CSS selector or visible text."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"target":{"type":"string"}},"required":["target"]})
    }
    async fn execute(&self, _: &Path, a: Value) -> Result<String> {
        let t = a["target"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing target"))?;
        let x = get_browser_controller().await?;
        let mut c = x.lock();
        c.get_or_create_page(None)?;
        let q = serde_json::string(t);
        let js = format!(
            "(()=>{{const t={q};let e;try{{e=document.querySelector(t)}}catch(_){{}}if(!e)e=[...document.querySelectorAll('a,button,input,[role=button],[onclick]')].find(x=>(x.innerText||x.value||'').trim().includes(t));if(!e)return false;e.scrollIntoView({{block:'center'}});e.click();return true}})()"
        );
        if c.eval(&js)?.as_bool() == Some(true) {
            Ok(format!("Clicked '{}'.", t))
        } else {
            Err(anyhow!("Could not locate '{}'", t))
        }
    }
}
pub struct BrowserClickTool;
impl Tool for BrowserClickTool {
    fn name(&self) -> &'static str {
        "browser_click"
    }
    fn description(&self) -> &'static str {
        "Click viewport coordinates on the active page."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"]})
    }
    async fn execute(&self, _: &Path, a: Value) -> Result<String> {
        let x = a["x"].as_f64().ok_or_else(|| anyhow!("Missing x"))?;
        let y = a["y"].as_f64().ok_or_else(|| anyhow!("Missing y"))?;
        let ctl = get_browser_controller().await?;
        let mut c = ctl.lock();
        c.get_or_create_page(None)?;
        for (k, b) in [
            ("mouseMoved", false),
            ("mousePressed", true),
            ("mouseReleased", true),
        ] {
            let mut p = json!({"type":k,"x":x,"y":y});
            if b {
                p["button"] = json!("left");
                p["clickCount"] = json!(1)
            }
            c.cdp("Input.dispatchMouseEvent", p)?;
        }
        Ok(format!("Clicked at ({}, {}).", x, y))
    }
}
pub struct BrowserTypeTool;
impl Tool for BrowserTypeTool {
    fn name(&self) -> &'static str {
        "browser_type"
    }
    fn description(&self) -> &'static str {
        "Type into the focused element on the active page."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})
    }
    async fn execute(&self, _: &Path, a: Value) -> Result<String> {
        let t = a["text"].as_str().ok_or_else(|| anyhow!("Missing text"))?;
        let x = get_browser_controller().await?;
        let mut c = x.lock();
        c.get_or_create_page(None)?;
        c.cdp("Input.insertText", json!({"text":t}))?;
        Ok(format!("Typed '{}' into the browser page.", t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn definitions() {
        assert_eq!(BrowserOpenTool.name(), "browser_open");
        assert_eq!(BrowserGetContentTool.name(), "browser_get_content");
        assert_eq!(BrowserScreenshotTool.name(), "browser_screenshot");
        assert_eq!(BrowserClickTool.name(), "browser_click");
        assert_eq!(BrowserTypeTool.name(), "browser_type");
    }
    #[test]
    fn installed_browser_is_found() {
        if cfg!(windows) {
            assert!(find_browser().is_some())
        }
    }

    #[test]
    fn installed_browser_navigates_types_clicks_and_screenshots() {
        if find_browser().is_none() {
            return;
        }
        crate::block_on(async {
            let mut browser = EdgeController::ensure_latest_and_launch().unwrap();
            browser.get_or_create_page(Some("data:text/html,<title>MRML%20Browser%20Test</title><input%20id='name'><button%20id='go'%20onclick=\"document.title=document.querySelector('%23name').value\">Go</button>")).unwrap();
            assert_eq!(browser.title().unwrap(), "MRML Browser Test");
            browser
                .eval("document.querySelector('#name').focus()")
                .unwrap();
            browser
                .cdp("Input.insertText", json!({"text":"typed correctly"}))
                .unwrap();
            assert_eq!(
                browser
                    .eval("document.querySelector('#name').value")
                    .unwrap(),
                "typed correctly"
            );
            assert_eq!(
                browser
                    .eval("document.querySelector('#go').click(); document.title")
                    .unwrap(),
                "typed correctly"
            );
            let shot = browser
                .cdp(
                    "Page.captureScreenshot",
                    json!({"format":"jpeg","quality":50}),
                )
                .unwrap();
            assert!(shot["data"].as_str().is_some_and(|data| data.len() > 100));
        });
    }
}
