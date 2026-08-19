//! Dependency-free workspace traversal for search and browser discovery.
use mrml_runtime::Vector;
use std::path::{Path, PathBuf};

pub struct Paths {
    pending: Vector<PathBuf>,
}

pub fn paths(root: impl AsRef<Path>) -> Paths {
    Paths {
        pending: [root.as_ref().to_path_buf()].into_iter().collect(),
    }
}

impl Iterator for Paths {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.pending.pop()?;
        if path
            .to_str()
            .is_some_and(mrml_runtime::path_is_directory)
        {
            if let Some(path_text) = path.to_str() {
                if let Ok(entries) = mrml_runtime::read_directory(path_text) {
                    self.pending.extend(
                        entries
                            .into_iter()
                            .filter(|entry| !entry.is_symlink)
                            .map(|entry| path.join(entry.name.as_str())),
                    );
                }
            }
        }
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn includes_root_and_skips_missing_children() {
        let root = std::env::temp_dir().join(format!("mrml-walk-{}", std::process::id()));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("file.txt"), b"test").unwrap();

        let found = super::paths(&root).collect::<Vec<_>>();
        assert!(found.contains(&root));
        assert!(found.contains(&nested.join("file.txt")));

        std::fs::remove_dir_all(root).unwrap();
    }
}
