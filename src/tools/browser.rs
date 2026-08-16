use crate::tools::Tool;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::input::{
    DispatchMouseEventParams, DispatchMouseEventType, InsertTextParams, MouseButton,
};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::fetcher::{BrowserFetcher, BrowserFetcherOptions};
use chromiumoxide::page::ScreenshotParams;
use futures_util::StreamExt;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OnceCell};
use tokio::time::sleep;

static BROWSER_INSTANCE: OnceCell<Arc<Mutex<ChromiumController>>> = OnceCell::const_new();

pub struct ChromiumController {
    pub browser: Browser,
    pub active_page: Option<chromiumoxide::Page>,
}

impl ChromiumController {
    pub async fn ensure_latest_and_launch() -> Result<Self> {
        // 1. First check if a system browser binary is already present
        let system_candidates = [
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/brave",
            "/usr/bin/brave-browser",
            "/usr/bin/microsoft-edge",
            "/app/bin/chromium",
            "/app/bin/chrome",
            "/app/extra/bin/chromium",
        ];

        let mut exec_path: Option<PathBuf> = None;
        for candidate in system_candidates {
            let p = PathBuf::from(candidate);
            if p.is_file() {
                exec_path = Some(p);
                break;
            }
        }

        if exec_path.is_none() {
            for bin_name in ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "brave", "brave-browser"] {
                if let Ok(output) = std::process::Command::new("which").arg(bin_name).output() {
                    if output.status.success() {
                        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        let p = PathBuf::from(&path_str);
                        if p.is_file() {
                            exec_path = Some(p);
                            break;
                        }
                    }
                }
            }
        }

        let final_exec_path = match exec_path {
            Some(p) => p,
            None => {
                let options = BrowserFetcherOptions::default()
                    .map_err(|e| anyhow!("Failed to create default browser fetcher options: {}", e))?;
                let fetcher = BrowserFetcher::new(options);
                let info = fetcher
                    .fetch()
                    .await
                    .map_err(|e| anyhow!("Failed to fetch latest Chromium binary via Chromiumoxide: {}", e))?;
                info.executable_path
            }
        };

        // 2. Configure headless Chromium
        let config = BrowserConfig::builder()
            .chrome_executable(final_exec_path)
            .no_sandbox()
            .viewport(None)
            .arg("--disable-dev-shm-usage")
            .arg("--disable-gpu")
            .arg("--headless=new")
            .build()
            .map_err(|e| anyhow!("Failed to build browser config: {}", e))?;

        let (browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| anyhow!("Failed to launch Chromium: {}", e))?;

        // 3. Spawn background handler to poll DevTools WebSocket events
        tokio::task::spawn(async move {
            while let Some(event) = handler.next().await {
                if event.is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            browser,
            active_page: None,
        })
    }

    pub async fn get_or_create_page(&mut self, url_opt: Option<&str>) -> Result<chromiumoxide::Page> {
        if let Some(page) = &self.active_page {
            if let Some(url) = url_opt {
                page.goto(url)
                    .await
                    .map_err(|e| anyhow!("Failed to navigate to '{}': {}", url, e))?;
                let _ = page.wait_for_navigation().await;
            }
            return Ok(page.clone());
        }

        let target_url = url_opt.unwrap_or("about:blank");
        let page = self
            .browser
            .new_page(target_url)
            .await
            .map_err(|e| anyhow!("Failed to create new Chromium page: {}", e))?;

        let _ = page.wait_for_navigation().await;
        self.active_page = Some(page.clone());
        Ok(page)
    }
}

pub async fn get_browser_controller() -> Result<Arc<Mutex<ChromiumController>>> {
    let controller = BROWSER_INSTANCE
        .get_or_try_init(|| async {
            let ctrl = ChromiumController::ensure_latest_and_launch().await?;
            Ok::<Arc<Mutex<ChromiumController>>, anyhow::Error>(Arc::new(Mutex::new(ctrl)))
        })
        .await?;
    Ok(controller.clone())
}

pub struct BrowserOpenTool;
#[async_trait]
impl Tool for BrowserOpenTool {
    fn name(&self) -> &'static str {
        "browser_open"
    }

    fn description(&self) -> &'static str {
        "Open and navigate to a web URL using the latest headless Chromium (Chromiumoxide CDP engine)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full web page URL (e.g. 'https://www.youtube.com')"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let url = args["url"].as_str().ok_or_else(|| anyhow!("Missing url"))?;

        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(Some(url)).await?;
        sleep(Duration::from_millis(1500)).await;

        let page_title = page.get_title().await.unwrap_or(None).unwrap_or_default();
        let page_url = page.url().await.unwrap_or(None).unwrap_or_else(|| url.to_string());

        Ok(format!(
            "Successfully opened URL '{}' in latest Chromium (Title: '{}', URL: '{}').",
            url, page_title, page_url
        ))
    }
}

pub struct BrowserGetContentTool;
#[async_trait]
impl Tool for BrowserGetContentTool {
    fn name(&self) -> &'static str {
        "browser_get_content"
    }

    fn description(&self) -> &'static str {
        "Extract and return clean plain-text DOM content from the active Chromium page or target URL."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Optional web page URL to navigate and read DOM text from (if omitted, reads active page)"
                }
            }
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let url_opt = args["url"].as_str();

        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(url_opt).await?;
        sleep(Duration::from_millis(500)).await;

        let page_title = page.get_title().await.unwrap_or(None).unwrap_or_default();

        let html = page
            .content()
            .await
            .map_err(|e| anyhow!("Failed to extract page HTML: {}", e))?;

        let display = {
            let document = scraper::Html::parse_document(&html);
            let noise_selector =
                scraper::Selector::parse("script, style, noscript, svg, iframe, nav, footer").unwrap();
            let noise_ids: Vec<_> = document.select(&noise_selector).map(|e| e.id()).collect();

            let content_selector =
                scraper::Selector::parse("h1, h2, h3, h4, h5, p, article, section, li, a, span").unwrap();
            let mut extracted_lines = Vec::new();

            for element in document.select(&content_selector) {
                let mut current = element;
                let mut in_noise = false;
                while let Some(parent) = current.parent().and_then(scraper::ElementRef::wrap) {
                    if noise_ids.contains(&parent.id()) {
                        in_noise = true;
                        break;
                    }
                    current = parent;
                }

                if !in_noise {
                    let text = element.text().collect::<Vec<_>>().join(" ");
                    let trimmed = text.trim();
                    if !trimmed.is_empty() && trimmed.len() > 2 && !extracted_lines.contains(&trimmed.to_string()) {
                        extracted_lines.push(trimmed.to_string());
                    }
                }
            }

            let full_text = extracted_lines.join("\n");
            if full_text.len() > 6000 {
                format!("{}... (truncated)", crate::markdown::truncate_utf8(&full_text, 6000))
            } else {
                full_text
            }
        };

        Ok(format!(
            "[Chromium Page DOM Content - '{}']:\n{}",
            page_title,
            if display.trim().is_empty() { "(No readable text found on page)" } else { display.trim() }
        ))
    }
}

pub struct BrowserScreenshotTool;
#[async_trait]
impl Tool for BrowserScreenshotTool {
    fn name(&self) -> &'static str {
        "browser_screenshot"
    }

    fn description(&self) -> &'static str {
        "Capture a compressed JPEG screenshot directly from the active Chromium page via Chrome DevTools Protocol."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, workspace_root: &Path, _args: serde_json::Value) -> Result<String> {
        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(None).await?;
        sleep(Duration::from_millis(500)).await;

        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Jpeg)
            .quality(75)
            .build();

        let img_bytes = page
            .screenshot(params)
            .await
            .map_err(|e| anyhow!("Failed to capture Chromium screenshot via CDP: {}", e))?;

        let shot_dir = workspace_root.join(".gemma").join("screenshots");
        fs::create_dir_all(&shot_dir)?;

        let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let file_path = shot_dir.join(format!("browser_screenshot_{}.jpg", timestamp));
        fs::write(&file_path, &img_bytes)?;

        let base64_str = base64::engine::general_purpose::STANDARD.encode(&img_bytes);
        let data_uri = format!("data:image/jpeg;base64,{}", base64_str);

        Ok(format!(
            "Chromium screenshot captured at '{}' ({} bytes). Base64 Data URI length: {}.\nDATA_URI:{}",
            file_path.display(),
            img_bytes.len(),
            base64_str.len(),
            data_uri
        ))
    }
}

pub struct BrowserClickElementTool;
#[async_trait]
impl Tool for BrowserClickElementTool {
    fn name(&self) -> &'static str {
        "browser_click_element"
    }

    fn description(&self) -> &'static str {
        "Find and click an interactive element on the Chromium page by visible text or CSS selector."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Visible text or CSS selector of the button/link/element to click (e.g. 'Add to Cart' or 'button#submit')"
                }
            },
            "required": ["target"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let target = args["target"].as_str().ok_or_else(|| anyhow!("Missing target"))?;

        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(None).await?;

        // 1. Try finding by CSS selector first
        if let Ok(elem) = page.find_element(target).await {
            elem.click()
                .await
                .map_err(|e| anyhow!("Failed to click element '{}': {}", target, e))?;
            return Ok(format!("Successfully clicked element matching selector '{}'.", target));
        }

        // 2. Try finding by XPath containing visible text
        let xpath = format!("//*[contains(text(), '{}')]", target);
        if let Ok(elem) = page.find_element(xpath.as_str()).await {
            elem.click()
                .await
                .map_err(|e| anyhow!("Failed to click element with text '{}': {}", target, e))?;
            return Ok(format!("Successfully clicked element with text '{}'.", target));
        }

        Err(anyhow!(
            "Could not locate element matching selector or text '{}' on the active Chromium page.",
            target
        ))
    }
}

pub struct BrowserClickTool;
#[async_trait]
impl Tool for BrowserClickTool {
    fn name(&self) -> &'static str {
        "browser_click"
    }

    fn description(&self) -> &'static str {
        "Perform a mouse click at specific viewport coordinates (X, Y) on the active Chromium page."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Horizontal pixel coordinate (X)"
                },
                "y": {
                    "type": "integer",
                    "description": "Vertical pixel coordinate (Y)"
                }
            },
            "required": ["x", "y"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let x = args["x"].as_f64().ok_or_else(|| anyhow!("Missing x coordinate"))?;
        let y = args["y"].as_f64().ok_or_else(|| anyhow!("Missing y coordinate"))?;

        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(None).await?;

        // Dispatch CDP mouse click event at (x, y)
        let move_params = DispatchMouseEventParams::builder()
            .r#type(DispatchMouseEventType::MouseMoved)
            .x(x)
            .y(y)
            .build()
            .map_err(|e| anyhow!("Failed to build mouse move params: {}", e))?;

        let press_params = DispatchMouseEventParams::builder()
            .r#type(DispatchMouseEventType::MousePressed)
            .x(x)
            .y(y)
            .button(MouseButton::Left)
            .click_count(1)
            .build()
            .map_err(|e| anyhow!("Failed to build mouse press params: {}", e))?;

        let release_params = DispatchMouseEventParams::builder()
            .r#type(DispatchMouseEventType::MouseReleased)
            .x(x)
            .y(y)
            .button(MouseButton::Left)
            .click_count(1)
            .build()
            .map_err(|e| anyhow!("Failed to build mouse release params: {}", e))?;

        page.execute(move_params).await?;
        page.execute(press_params).await?;
        page.execute(release_params).await?;

        Ok(format!("Successfully clicked at coordinates ({}, {}) on Chromium page.", x, y))
    }
}

pub struct BrowserTypeTool;
#[async_trait]
impl Tool for BrowserTypeTool {
    fn name(&self) -> &'static str {
        "browser_type"
    }

    fn description(&self) -> &'static str {
        "Type text into the currently focused input element on the active Chromium page."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to type into the focused element"
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let text = args["text"].as_str().ok_or_else(|| anyhow!("Missing text"))?;

        let controller_arc = get_browser_controller().await?;
        let mut ctrl = controller_arc.lock().await;

        let page = ctrl.get_or_create_page(None).await?;

        let insert_params = InsertTextParams::builder()
            .text(text)
            .build()
            .map_err(|e| anyhow!("Failed to build insert text params: {}", e))?;

        page.execute(insert_params)
            .await
            .map_err(|e| anyhow!("Failed to insert text into Chromium page: {}", e))?;

        Ok(format!("Successfully typed '{}' into Chromium page.", text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tools_definitions() {
        let open_tool = BrowserOpenTool;
        assert_eq!(open_tool.name(), "browser_open");

        let content_tool = BrowserGetContentTool;
        assert_eq!(content_tool.name(), "browser_get_content");

        let shot_tool = BrowserScreenshotTool;
        assert_eq!(shot_tool.name(), "browser_screenshot");

        let click_tool = BrowserClickTool;
        assert_eq!(click_tool.name(), "browser_click");

        let type_tool = BrowserTypeTool;
        assert_eq!(type_tool.name(), "browser_type");
    }
}
