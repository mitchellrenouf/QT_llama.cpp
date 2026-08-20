#![no_std]

use core::fmt;
use mrml_runtime::Text;
use mrml_zim::{Archive, ClusterDecoder, DirectoryEntry, EntryLocation};

#[derive(Debug)]
pub enum Error {
    Zim(mrml_zim::Error),
    NotArticle,
    NotHtml,
    InvalidUtf8,
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zim(error) => error.fmt(output),
            Self::NotArticle => output.write_str("ZIM entry is not a Wikipedia article"),
            Self::NotHtml => output.write_str("Wikipedia entry is not HTML"),
            Self::InvalidUtf8 => output.write_str("Wikipedia article is not valid UTF-8"),
        }
    }
}

impl core::error::Error for Error {}
impl From<mrml_zim::Error> for Error {
    fn from(error: mrml_zim::Error) -> Self {
        Self::Zim(error)
    }
}
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Article {
    pub index: u32,
    pub path: Text,
    pub title: Text,
    pub text: Text,
}

pub struct ArticleReader {
    archive: Archive,
    next_index: u32,
}

impl ArticleReader {
    pub fn open(path: &str) -> Result<Self> {
        Ok(Self {
            archive: Archive::open(path)?,
            next_index: 0,
        })
    }

    pub fn set_cluster_cache_capacity(&mut self, capacity: usize) {
        self.archive.set_cluster_cache_capacity(capacity);
    }

    pub fn next_article(&mut self) -> Result<Option<Article>> {
        let count = self.archive.header().entry_count;
        while self.next_index < count {
            let index = self.next_index;
            self.next_index += 1;
            let entry = self.archive.entry(index)?;
            if !is_article(&entry) || self.archive.mime_type(entry.mime_type) != Some("text/html") {
                continue;
            }
            let text = entry_text(&mut self.archive, &entry)?;
            return Ok(Some(Article {
                index,
                path: entry.path,
                title: entry.title,
                text,
            }));
        }
        Ok(None)
    }
}

struct NativeZstd;

impl ClusterDecoder for NativeZstd {
    fn decode_zstd(
        &mut self,
        compressed: &[u8],
    ) -> core::result::Result<mrml_runtime::Vector<u8>, ()> {
        mrml_zstd::decode(compressed).map_err(|_| ())
    }
}

/// Reads one article and converts its HTML body to normalized training text.
/// The archive stays open so callers can stream entries from multi-gigabyte
/// Wikipedia files without loading the ZIM into memory.
pub fn article_text(archive: &mut Archive, index: u32) -> Result<Text> {
    let entry = archive.entry(index)?;
    if !is_article(&entry) {
        return Err(Error::NotArticle);
    }
    if archive.mime_type(entry.mime_type) != Some("text/html") {
        return Err(Error::NotHtml);
    }
    entry_text(archive, &entry)
}

fn entry_text(archive: &mut Archive, entry: &DirectoryEntry) -> Result<Text> {
    let EntryLocation::Blob { cluster, blob } = entry.location else {
        return Err(Error::NotArticle);
    };
    let bytes = archive.read_blob_with(cluster, blob, &mut NativeZstd)?;
    let html = core::str::from_utf8(&bytes).map_err(|_| Error::InvalidUtf8)?;
    Ok(html_to_text(html))
}

/// Returns true for content entries that can contribute natural-language text.
pub fn is_article(entry: &DirectoryEntry) -> bool {
    matches!(entry.location, EntryLocation::Blob { .. })
        && matches!(entry.namespace, b'A' | b'C')
        && !entry.path.starts_with("Special:")
        && !entry.path.starts_with("Category:")
        && !entry.path.starts_with("File:")
        && !entry.path.starts_with("Template:")
        && !entry.path.starts_with("User:")
}

/// Converts a Wikipedia HTML document into normalized training text.
///
/// Script, style, template, table, navigation, and citation markup is omitted.
/// Block elements become paragraph boundaries and entities are decoded without
/// allocating intermediary strings.
pub fn html_to_text(html: &str) -> Text {
    let bytes = html.as_bytes();
    let mut output =
        Text::with_capacity(html.len().min(64 * 1024)).expect("MRML allocation failed");
    let mut index = 0;
    let mut suppressed_depth = 0usize;
    let mut pending_space = false;
    let mut pending_paragraph = false;

    while index < bytes.len() {
        if bytes[index] == b'<' {
            let Some(relative_end) = html[index + 1..].find('>') else {
                break;
            };
            let end = index + 1 + relative_end;
            let raw = html[index + 1..end].trim();
            let closing = raw.starts_with('/');
            let name_start = usize::from(closing);
            let name_end = raw[name_start..]
                .find(|character: char| character.is_ascii_whitespace() || character == '/')
                .map(|offset| name_start + offset)
                .unwrap_or(raw.len());
            let name = &raw[name_start..name_end];
            let suppressed = matches_ignore_ascii_case(
                name,
                &[
                    "script", "style", "table", "nav", "footer", "header", "aside", "math",
                ],
            );
            if suppressed {
                if closing {
                    suppressed_depth = suppressed_depth.saturating_sub(1);
                } else if !raw.ends_with('/') {
                    suppressed_depth += 1;
                }
            } else if suppressed_depth == 0 {
                if matches_ignore_ascii_case(
                    name,
                    &[
                        "p",
                        "div",
                        "section",
                        "article",
                        "h1",
                        "h2",
                        "h3",
                        "h4",
                        "h5",
                        "h6",
                        "li",
                        "br",
                        "hr",
                        "blockquote",
                        "pre",
                    ],
                ) {
                    pending_paragraph = true;
                } else {
                    pending_space = true;
                }
            }
            index = end + 1;
            continue;
        }
        if suppressed_depth != 0 {
            index += 1;
            continue;
        }
        if bytes[index] == b'&' {
            if let Some(relative_end) = html[index..].find(';').filter(|end| *end <= 12) {
                let entity = &html[index + 1..index + relative_end];
                if let Some(character) = decode_entity(entity) {
                    push_normalized(
                        &mut output,
                        character,
                        &mut pending_space,
                        &mut pending_paragraph,
                    );
                    index += relative_end + 1;
                    continue;
                }
            }
        }
        let character = html[index..].chars().next().unwrap();
        push_normalized(
            &mut output,
            character,
            &mut pending_space,
            &mut pending_paragraph,
        );
        index += character.len_utf8();
    }
    while output.ends_with(' ') || output.ends_with('\n') {
        output.pop();
    }
    output
}

fn push_normalized(
    output: &mut Text,
    character: char,
    pending_space: &mut bool,
    pending_paragraph: &mut bool,
) {
    if character.is_whitespace() {
        *pending_space = true;
        return;
    }
    if *pending_paragraph && !output.is_empty() {
        while output.ends_with(' ') {
            output.pop();
        }
        if !output.ends_with("\n\n") {
            output.push_str(if output.ends_with('\n') { "\n" } else { "\n\n" });
        }
    } else if *pending_space
        && !output.is_empty()
        && !output.ends_with('\n')
        && !output.ends_with(' ')
    {
        output.push(' ');
    }
    *pending_paragraph = false;
    *pending_space = false;
    output.push(character);
}

fn matches_ignore_ascii_case(value: &str, choices: &[&str]) -> bool {
    choices
        .iter()
        .any(|choice| value.eq_ignore_ascii_case(choice))
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some(' '),
        "ndash" => Some('–'),
        "mdash" => Some('—'),
        "hellip" => Some('…'),
        value if value.starts_with("#x") || value.starts_with("#X") => {
            u32::from_str_radix(&value[2..], 16)
                .ok()
                .and_then(char::from_u32)
        }
        value if value.starts_with('#') => value[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_readable_wikipedia_text() {
        let html = "<html><style>bad{}</style><h1>Rust &amp; Safety</h1><p>Rust is a <b>systems</b> language.</p><table><tr><td>noise</td></tr></table><p>It prevents&nbsp;many bugs.</p><script>noise()</script></html>";
        assert_eq!(
            html_to_text(html),
            "Rust & Safety\n\nRust is a systems language.\n\nIt prevents many bugs."
        );
    }

    #[test]
    fn decodes_numeric_entities_and_normalizes_spacing() {
        assert_eq!(html_to_text("<p>A  &#x2014;  B &#65;</p>"), "A — B A");
    }

    #[test]
    fn filters_non_article_namespaces_and_paths() {
        let article = DirectoryEntry {
            mime_type: 0,
            namespace: b'C',
            revision: 0,
            path: "Rust".into(),
            title: "Rust".into(),
            location: EntryLocation::Blob {
                cluster: 0,
                blob: 0,
            },
        };
        assert!(is_article(&article));
        let mut category = article.clone();
        category.path = "Category:Languages".into();
        assert!(!is_article(&category));
        let mut redirect = article;
        redirect.location = EntryLocation::Redirect(1);
        assert!(!is_article(&redirect));
    }

    #[test]
    fn decodes_external_kiwix_cluster_when_configured() {
        let Some(path) = mrml_runtime::environment_variable("MRML_TEST_ZIM") else {
            return;
        };
        let mut archive = Archive::open(&path).expect("open configured ZIM archive");
        let count = archive.header().cluster_count;
        let mut decoded_clusters = 0;
        let exhaustive =
            mrml_runtime::environment_variable("MRML_TEST_ZIM_ALL").as_deref() == Some("1");
        let mut samples = mrml_runtime::Vector::new();
        if exhaustive {
            samples.extend(0..count);
        } else {
            samples.extend([0, count / 3, count / 2, count.saturating_sub(1)]);
            if count > 141_232 {
                samples.extend([83_034, 141_232]);
            }
        }
        for cluster in samples {
            if archive
                .cluster_info(cluster)
                .expect("inspect sampled cluster")
                .compression
                != mrml_zim::Compression::Zstd
            {
                continue;
            }
            let compressed = archive
                .read_cluster_payload(cluster)
                .expect("read sampled cluster");
            let decoded = mrml_zstd::decode(&compressed).expect("decode sampled Zstandard cluster");
            assert!(!decoded.is_empty());
            if mrml_runtime::environment_variable("MRML_TEST_ZIM_HASHES").as_deref() == Some("1") {
                let hash = decoded.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                    (hash ^ *byte as u64).wrapping_mul(0x100_0000_01b3)
                });
                mrml_runtime::mrml_println!(
                    "cluster={cluster} bytes={} fnv64={hash:016x}",
                    decoded.len()
                );
            }
            decoded_clusters += 1;
        }
        assert!(decoded_clusters > 0);
    }

    #[test]
    fn extracts_external_wikipedia_article_when_configured() {
        let Some(path) = mrml_runtime::environment_variable("MRML_TEST_ZIM") else {
            return;
        };
        let mut archive = Archive::open(&path).expect("open configured ZIM archive");
        let count = archive.header().entry_count;
        for index in 0..count {
            let entry = archive.entry(index).expect("read directory entry");
            if is_article(&entry) && archive.mime_type(entry.mime_type) == Some("text/html") {
                let text = article_text(&mut archive, index).expect("extract Wikipedia article");
                if text.len() > 100 && text.split_whitespace().count() > 20 {
                    assert!(!text.contains("<html"));
                    return;
                }
            }
        }
        panic!("configured ZIM contains no HTML Wikipedia article");
    }

    #[test]
    fn streams_external_articles_when_configured() {
        let Some(path) = mrml_runtime::environment_variable("MRML_TEST_ZIM") else {
            return;
        };
        let mut reader = ArticleReader::open(&path).expect("open Wikipedia article stream");
        let mut previous = None;
        let mut count = 0;
        while count < 100 {
            let article = match reader.next_article() {
                Ok(article) => article,
                Err(error) => {
                    let index = reader.next_index.saturating_sub(1);
                    let entry = reader.archive.entry(index).expect("reread failing entry");
                    let (cluster, detail) =
                        if let EntryLocation::Blob { cluster, .. } = entry.location {
                            let payload = reader
                                .archive
                                .read_cluster_payload(cluster)
                                .expect("read failing cluster");
                            (Some(cluster), mrml_zstd::decode(&payload).err())
                        } else {
                            (None, None)
                        };
                    panic!(
                        "stream Wikipedia article at directory index {index}, cluster={cluster:?}: {error:?}, codec={detail:?}"
                    )
                }
            };
            let Some(article) = article else { break };
            if let Some(index) = previous {
                assert!(article.index > index);
            }
            assert!(!article.path.is_empty());
            previous = Some(article.index);
            count += 1;
        }
        assert!(count >= 10);
    }
}
