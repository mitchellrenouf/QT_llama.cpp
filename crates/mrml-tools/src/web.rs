use crate::Tool;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::path::Path;

pub async fn fetch_http_text(url: &str) -> Result<String> {
    // 1. Try headless Chromium if browser controller is available
    if let Ok(browser_ctrl) = crate::browser::get_browser_controller().await {
        let mut guard = browser_ctrl
            .lock()
            .map_err(|_| anyhow!("Browser controller lock poisoned"))?;
        if let Ok(page) = guard.get_or_create_page(Some(url)) {
            crate::platform::sleep_millis(1200);
            if let Ok(html) = page.content() {
                if !html.is_empty() {
                    return Ok(html);
                }
            }
        }
    }

    // 2. Direct HTTP fallback using curl
    if let Ok(output) = std::process::Command::new("curl")
        .arg("-sL")
        .arg("--max-time")
        .arg("10")
        .arg("-A")
        .arg("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0")
        .arg(url)
        .output()
    {
        if output.status.success() {
            let body = String::from_utf8_lossy(&output.stdout).to_string();
            if !body.is_empty() {
                return Ok(body);
            }
        }
    }

    // 3. Fallback to wget
    if let Ok(output) = std::process::Command::new("wget")
        .arg("-qO-")
        .arg("--timeout=10")
        .arg("-U")
        .arg("Mozilla/5.0 (X11; Linux x86_64)")
        .arg(url)
        .output()
    {
        if output.status.success() {
            let body = String::from_utf8_lossy(&output.stdout).to_string();
            if !body.is_empty() {
                return Ok(body);
            }
        }
    }

    Err(anyhow!(
        "Failed to fetch web content from '{}' via headless browser, curl, or wget.",
        url
    ))
}

pub struct WebSearchTool;

impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn description(&self) -> &'static str {
        "Search the internet for live web results, technical documentation, Wikipedia articles, images, and news."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query keywords (e.g. 'cute fruit bat', 'Rust async-trait', 'Bicycle Race Queen')"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing query"))?;
        let mut results = Vec::new();

        let _is_image_query = query.to_lowercase().contains("image")
            || query.to_lowercase().contains("picture")
            || query.to_lowercase().contains("photo");

        // 1. First search Wikipedia directly
        let clean_query = query
            .replace("find an image of a ", "")
            .replace("find an image of ", "")
            .replace("show an image of a ", "")
            .replace("show an image of ", "")
            .replace("image of a ", "")
            .replace("image of ", "")
            .replace("picture of ", "")
            .replace("photo of ", "");

        let wiki_url = format!(
            "https://en.wikipedia.org/wiki/Special:Search?search={}&go=Go",
            urlencoding::encode(&clean_query)
        );

        if let Ok(html) = fetch_http_text(&wiki_url).await {
            let heading = crate::html::elements(&html, "h1", None)
                .into_iter()
                .next()
                .map(|element| element.text)
                .unwrap_or_else(|| query.to_string());

            let mut img_url = None;
            for element in crate::html::elements(&html, "meta", None)
                .into_iter()
                .chain(crate::html::elements(&html, "img", None))
            {
                if let Some(content) = element.attributes.get("content") {
                    if content.starts_with("http") {
                        img_url = Some(content.to_string());
                        break;
                    }
                }
                if let Some(src) = element.attributes.get("src") {
                    let full = if src.starts_with("//") {
                        format!("https:{}", src)
                    } else {
                        src.to_string()
                    };
                    if full.starts_with("http")
                        && !full.contains("static/favicon")
                        && !full.contains("apple-touch")
                    {
                        img_url = Some(full);
                        break;
                    }
                }
            }

            let p_text = crate::html::elements(&html, "p", None)
                .into_iter()
                .take(4)
                .map(|element| element.text)
                .filter(|t| !t.is_empty() && t.len() > 20)
                .collect::<Vec<_>>()
                .join("\n\n");

            if !p_text.is_empty() || img_url.is_some() {
                let mut entry = format!("- **{}** (Wikipedia)\n  URL: {}\n", heading, wiki_url);
                if let Some(ref img) = img_url {
                    entry.push_str(&format!(
                        "  Image URL: {}\n  Markdown Image: ![{}]({})\n",
                        img, heading, img
                    ));
                }
                if !p_text.is_empty() {
                    entry.push_str(&format!(
                        "  Summary: {}\n",
                        crate::markdown::truncate_utf8(&p_text, 1000)
                    ));
                }
                results.push(entry);
            }
        }

        // 2. Search DuckDuckGo HTML
        let ddg_url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding::encode(query)
        );
        if let Ok(html) = fetch_http_text(&ddg_url).await {
            let links = crate::html::elements(&html, "a", Some("result__a"));
            let snippets = crate::html::elements(&html, "a", Some("result__snippet"));
            for (index, link) in links.into_iter().take(6).enumerate() {
                let title = link.text;
                let raw_href = link
                    .attributes
                    .get("href")
                    .map(String::as_str)
                    .unwrap_or("");
                let snippet = snippets
                    .get(index)
                    .map(|item| item.text.clone())
                    .unwrap_or_default();

                let clean_url = if let Some(pos) = raw_href.find("uddg=") {
                    let after = &raw_href[pos + 5..];
                    let raw_encoded = after.split('&').next().unwrap_or(after);
                    urlencoding::decode(raw_encoded)
                        .unwrap_or(std::borrow::Cow::Borrowed(raw_encoded))
                        .to_string()
                } else if raw_href.starts_with("http") {
                    raw_href.to_string()
                } else {
                    continue;
                };

                if !title.is_empty()
                    && clean_url.starts_with("http")
                    && !clean_url.contains("duckduckgo.com")
                {
                    if !snippet.is_empty() {
                        results.push(format!(
                            "- **{}**\n  URL: {}\n  Summary: {}",
                            title, clean_url, snippet
                        ));
                    } else {
                        results.push(format!("- **{}**\n  URL: {}", title, clean_url));
                    }
                }
            }
        }

        if results.is_empty() {
            Ok(format!("Web search executed for '{}', but no public search results were returned. Try searching with more specific keywords or use browser_open.", query))
        } else {
            Ok(results.join("\n\n"))
        }
    }
}

pub struct WebFetchTool;

impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        "web_fetch"
    }

    fn description(&self) -> &'static str {
        "Fetch and dump clean, rendered DOM page text from a web URL using headless browser or HTTP fallback."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full web page URL to fetch and dump text from"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let url_str = args["url"].as_str().ok_or_else(|| anyhow!("Missing url"))?;
        let html = fetch_http_text(url_str).await?;

        let full_text = crate::html::visible_text(&html);
        if full_text.is_empty() {
            Ok(format!(
                "Successfully fetched URL '{}', but no clean body text was extracted.",
                url_str
            ))
        } else if full_text.len() > 10000 {
            Ok(format!(
                "{}... (truncated at 10,000 characters)",
                crate::markdown::truncate_utf8(&full_text, 10000)
            ))
        } else {
            Ok(full_text)
        }
    }
}
