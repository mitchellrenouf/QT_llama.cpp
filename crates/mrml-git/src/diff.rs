use mrml_runtime::{Text, Vector};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileDiff {
    pub path: Text,
    pub old: Option<Vector<u8>>,
    pub new: Option<Vector<u8>>,
}

impl FileDiff {
    pub fn unified(&self) -> Text {
        let mut output = mrml_runtime::mrml_format!("diff --mrml a/{0} b/{0}\n", self.path);
        match (&self.old, &self.new) {
            (None, Some(_)) => output.push_str("new file mode 100644\n"),
            (Some(_), None) => output.push_str("deleted file mode 100644\n"),
            _ => {}
        }
        let old = self.old.as_deref().unwrap_or(&[]);
        let new = self.new.as_deref().unwrap_or(&[]);
        if old.contains(&0) || new.contains(&0) || core::str::from_utf8(old).is_err() || core::str::from_utf8(new).is_err() {
            output.push_str(&mrml_runtime::mrml_format!("Binary files a/{0} and b/{0} differ\n", self.path));
            return output;
        }
        let old_text = core::str::from_utf8(old).unwrap_or("");
        let new_text = core::str::from_utf8(new).unwrap_or("");
        let old_lines = lines(old_text);
        let new_lines = lines(new_text);
        output.push_str(&mrml_runtime::mrml_format!("--- {}\n+++ {}\n", if self.old.is_some() { mrml_runtime::mrml_format!("a/{}", self.path) } else { Text::from("/dev/null") }, if self.new.is_some() { mrml_runtime::mrml_format!("b/{}", self.path) } else { Text::from("/dev/null") }));
        output.push_str(&mrml_runtime::mrml_format!("@@ -1,{} +1,{} @@\n", old_lines.len(), new_lines.len()));
        let script = edit_script(&old_lines, &new_lines);
        for (kind, line) in script { output.push(kind); output.push_str(line); if !line.ends_with('\n') { output.push('\n'); } }
        output
    }
}

fn lines(text: &str) -> Vector<&str> {
    if text.is_empty() { return Vector::new(); }
    text.split_inclusive('\n').collect()
}

fn edit_script<'a>(old: &[&'a str], new: &[&'a str]) -> Vector<(char, &'a str)> {
    const MAX_CELLS: usize = 4_000_000;
    if old.len().checked_mul(new.len()).is_none_or(|cells| cells > MAX_CELLS) {
        let mut result = Vector::new();
        result.extend(old.iter().map(|line| ('-', *line)));
        result.extend(new.iter().map(|line| ('+', *line)));
        return result;
    }
    let width = new.len() + 1;
    let mut table = Vector::new(); table.resize((old.len() + 1) * width, 0u32);
    for i in (0..old.len()).rev() { for j in (0..new.len()).rev() {
        table[i * width + j] = if old[i] == new[j] { table[(i + 1) * width + j + 1] + 1 } else { table[(i + 1) * width + j].max(table[i * width + j + 1]) };
    }}
    let (mut i, mut j) = (0, 0); let mut result = Vector::new();
    while i < old.len() || j < new.len() {
        if i < old.len() && j < new.len() && old[i] == new[j] { result.push((' ', old[i])); i += 1; j += 1; }
        else if j < new.len() && (i == old.len() || table[i * width + j + 1] >= table[(i + 1) * width + j]) { result.push(('+', new[j])); j += 1; }
        else { result.push(('-', old[i])); i += 1; }
    }
    result
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn emits_line_edits_and_binary_marker() { let patch=FileDiff{path:"a.txt".into(),old:Some(Vector::from(*b"one\ntwo\n")),new:Some(Vector::from(*b"one\nthree\n"))}.unified(); assert!(patch.contains(" one\n")); assert!(patch.contains("-two\n")); assert!(patch.contains("+three\n")); let binary=FileDiff{path:"b".into(),old:Some(Vector::from([0u8])),new:Some(Vector::from([1u8]))}.unified(); assert!(binary.contains("Binary files")); }
}
