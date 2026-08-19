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
    use mrml_runtime::{Vector, join_path, mrml_format as format, process_id, temporary_directory};

    #[test]
    fn includes_root_and_skips_missing_children() {
        let root = join_path(&temporary_directory(), &format!("mrml-walk-{}", process_id()));
        let nested = join_path(&root, "nested");
        mrml_runtime::create_dir_all(&nested).unwrap();
        let file = join_path(&nested, "file.txt");
        mrml_runtime::write_file(&file, b"test").unwrap();

        let found = super::paths(&root).collect::<Vector<_>>();
        assert!(found.iter().any(|path| path == &root));
        assert!(found.iter().any(|path| path == &file));

        mrml_runtime::remove_dir_all(&root).unwrap();
    }
}
