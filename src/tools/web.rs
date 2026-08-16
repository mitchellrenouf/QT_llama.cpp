use crate::tools::Tool;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;

pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn description(&self) -> &'static str {
        "Search the internet for live web results, technical documentation, Wikipedia articles, and news using headless Chromium."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query keywords (e.g. 'Mark Carney', 'Rust async-trait', 'Bicycle Race Queen')"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, _workspace_root: &Path, args: serde_json::Value) -> Result<String> {
        let query = args["query"].as_str().ok_or_else(|| anyhow!("Missing query"))?;

        let browser_ctrl = crate::tools::browser::get_browser_controller().await?;
        let page = {
            let mut guard = browser_ctrl.lock().await;
            guard.get_or_create_page(None).await?
        };

        let mut results = Vec::new();
        let ddg_url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));

        if page.goto(&ddg_url).await.is_ok() {
            let _ = page.wait_for_navigation().await;
            tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
            if let Ok(html) = page.content().await {
                let document = scraper::Html::parse_document(&html);
                let result_block_sel = scraper::Selector::parse(".result").unwrap();
                let title_sel = scraper::Selector::parse(".result__a").unwrap();
                let snippet_sel = scraper::Selector::parse(".result__snippet").unwrap();

                for block in document.select(&result_block_sel).take(8) {
                    let title = block
                        .select(&title_sel)
                        .next()
                        .map(|e| e.text().collect::<Vec<_>>().join(" ").trim().to_string())
                        .unwrap_or_default();

                    let raw_href = block
                        .select(&title_sel)
                        .next()
                        .and_then(|e| e.value().attr("href"))
                        .unwrap_or("");

                    let snippet = block
                        .select(&snippet_sel)
                        .next()
                        .map(|e| e.text().collect::<Vec<_>>().join(" ").trim().to_string())
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

                    if !title.is_empty() && clean_url.starts_with("http") && !clean_url.contains("duckduckgo.com") {
                        if !snippet.is_empty() {
                            results.push(format!("- **{}**\n  URL: {}\n  Summary: {}", title, clean_url, snippet));
                        } else {
                            results.push(format!("- **{}**\n  URL: {}", title, clean_url));
                        }
                    }
                }
            }
        }

        // Fallback: If DDG gave no results or query requests images, query Wikipedia directly via Chromium
        let is_image_query = query.to_lowercase().contains("image")
            || query.to_lowercase().contains("picture")
            || query.to_lowercase().contains("photo");

        if results.is_empty() || is_image_query {
            let clean_query = query
                .replace("find an image of a ", "")
                .replace("find an image of ", "")
                .replace("image of a ", "")
                .replace("image of ", "")
                .replace("picture of ", "")
                .replace("photo of ", "");
            let wiki_url = format!(
                "https://en.wikipedia.org/wiki/Special:Search?search={}&go=Go",
                urlencoding::encode(&clean_query)
            );
            if page.goto(&wiki_url).await.is_ok() {
                let _ = page.wait_for_navigation().await;
                tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
                if let Ok(html) = page.content().await {
                    let document = scraper::Html::parse_document(&html);
                    let title_sel = scraper::Selector::parse("#firstHeading, h1").unwrap();
                    let p_sel = scraper::Selector::parse("p").unwrap();
                    let img_sel = scraper::Selector::parse(".infobox img, .mw-file-element, figure img, .thumbimage, .image img").unwrap();

                    let heading = document
                        .select(&title_sel)
                        .next()
                        .map(|e| e.text().collect::<Vec<_>>().join(" ").trim().to_string())
                        .unwrap_or_else(|| query.to_string());

                    let img_url = document
                        .select(&img_sel)
                        .next()
                        .and_then(|e| e.value().attr("src"))
                        .map(|src| {
                            if src.starts_with("//") {
                                format!("https:{}", src)
                            } else {
                                src.to_string()
                            }
                        });

                    let p_text = document
                        .select(&p_sel)
                        .take(3)
                        .map(|p| p.text().collect::<Vec<_>>().join(" ").trim().to_string())
                        .filter(|t| !t.is_empty() && t.len() > 20)
                        .collect::<Vec<_>>()
                        .join("\n\n");

                    if !p_text.is_empty() || img_url.is_some() {
                        let mut entry = format!(
                            "- **{}** (Wikipedia)\n  URL: {}\n",
                            heading, wiki_url
                        );
                        if let Some(ref img) = img_url {
                            entry.push_str(&format!("  Image URL: {}\n  Markdown Embedded: ![{}]({})\n", img, heading, img));
                        }
                        if !p_text.is_empty() {
                            entry.push_str(&format!("  Summary: {}\n", crate::markdown::truncate_utf8(&p_text, 1000)));
                        }
                        results.insert(0, entry);
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

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        "web_fetch"
    }

    fn description(&self) -> &'static str {
        "Fetch and dump clean, rendered DOM page text from a web URL using headless Chromium."
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

        let browser_ctrl = crate::tools::browser::get_browser_controller().await?;
        let page = {
            let mut guard = browser_ctrl.lock().await;
            guard.get_or_create_page(None).await?
        };

        page.goto(url_str)
            .await
            .map_err(|e| anyhow!("Failed to navigate Chromium to '{}': {}", url_str, e))?;

        let _ = page.wait_for_navigation().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;

        let html = page
            .content()
            .await
            .map_err(|e| anyhow!("Failed to retrieve DOM content from Chromium: {}", e))?;

        let document = scraper::Html::parse_document(&html);

        // Remove script, style, noscript, svg, iframe tags
        let noise_selector = scraper::Selector::parse("script, style, noscript, svg, iframe, nav, footer").unwrap();
        let noise_ids: Vec<_> = document.select(&noise_selector).map(|e| e.id()).collect();

        let content_selector = scraper::Selector::parse("h1, h2, h3, h4, h5, p, article, section, li, a, span").unwrap();

        let mut lines = Vec::new();
        for element in document.select(&content_selector) {
            if noise_ids.contains(&element.id()) {
                continue;
            }

            let text = element.text().collect::<Vec<_>>().join(" ");
            let trimmed = text.trim();

            if trimmed.len() >= 15
                && !trimmed.contains("function(")
                && !trimmed.contains("const ")
                && !trimmed.contains("var ")
                && !trimmed.contains("window.")
                && !trimmed.contains("document.")
                && !trimmed.contains("freestar")
                && !trimmed.contains("{")
            {
                let cleaned_line = trimmed.to_string();
                if !lines.contains(&cleaned_line) {
                    lines.push(cleaned_line);
                }
            }
        }

        let full_text = lines.join("\n");
        if full_text.is_empty() {
            Ok(format!("Successfully fetched URL '{}', but no clean body text was extracted.", url_str))
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
