use std::fs;
use std::path::{Path, PathBuf};

pub struct Paths {
    pending: Vec<PathBuf>,
}

pub fn paths(root: impl AsRef<Path>) -> Paths {
    Paths { pending: vec![root.as_ref().to_path_buf()] }
}

impl Iterator for Paths {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        let path = self.pending.pop()?;
        if path.is_dir() {
            if let Ok(entries) = fs::read_dir(&path) {
                self.pending.extend(entries.filter_map(Result::ok).filter_map(|entry| {
                    entry.file_type().ok().and_then(|kind| {
                        (!kind.is_symlink()).then(|| entry.path())
                    })
                }));
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
