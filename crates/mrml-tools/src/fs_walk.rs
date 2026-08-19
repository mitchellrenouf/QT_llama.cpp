//! Dependency-free workspace traversal for search and browser discovery.
use mrml_runtime::Vector;
use mrml_runtime::Text;

pub struct Paths {
    pending: Vector<Text>,
}

pub fn paths(root: &str) -> Paths {
    Paths {
        pending: [Text::from(root)].into_iter().collect(),
    }
}

impl Iterator for Paths {
    type Item = Text;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.pending.pop()?;
        if mrml_runtime::path_is_directory(&path) {
            if let Ok(entries) = mrml_runtime::read_directory(&path) {
                self.pending.extend(entries.into_iter().filter(|entry| !entry.is_symlink).map(
                    |entry| {
                        let mut child = path.clone();
                        if !child.ends_with(['/', '\\']) {
                            child.push(if cfg!(windows) { '\\' } else { '/' });
                        }
                        child.push_str(&entry.name);
                        child
                    },
                ));
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

        let found = super::paths(root.to_str().unwrap()).collect::<Vec<_>>();
        assert!(found.iter().any(|path| path == root.to_str().unwrap()));
        assert!(found
            .iter()
            .any(|path| path == nested.join("file.txt").to_str().unwrap()));

        std::fs::remove_dir_all(root).unwrap();
    }
}
