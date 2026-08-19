use crate::{Text, Vector};
#[cfg(unix)]
use crate::File;

pub fn command_arguments() -> Vector<Text> {
    #[cfg(windows)]
    {
        parse_windows_command_line(mrml_windows::command_line_wide())
    }
    #[cfg(unix)]
    {
        let mut file = File::open("/proc/self/cmdline").expect("failed to open /proc/self/cmdline");
        let mut bytes = Vector::new();
        let mut chunk = [0u8; 4096];
        loop {
            let read = file.read(&mut chunk).expect("failed to read /proc/self/cmdline");
            if read == 0 {
                break;
            }
            bytes
                .try_extend_from_slice(&chunk[..read])
                .expect("MRML allocation failed");
        }
        let mut output = Vector::new();
        for argument in bytes.split(|byte| *byte == 0).filter(|argument| !argument.is_empty()) {
            output.push(
                Text::try_from_str(
                    core::str::from_utf8(argument).expect("command line contains invalid UTF-8"),
                )
                .expect("MRML allocation failed"),
            );
        }
        output
    }
}

#[cfg(windows)]
fn parse_windows_command_line(command_line: &[u16]) -> Vector<Text> {
    let mut arguments = Vector::new();
    let mut index = 0usize;
    while index < command_line.len() {
        while index < command_line.len() && matches!(command_line[index], 0x20 | 0x09) {
            index += 1;
        }
        if index == command_line.len() {
            break;
        }
        let mut units = Vector::new();
        let mut quoted = false;
        while index < command_line.len() {
            let mut slashes = 0usize;
            while index < command_line.len() && command_line[index] == b'\\' as u16 {
                slashes += 1;
                index += 1;
            }
            if index < command_line.len() && command_line[index] == b'"' as u16 {
                for _ in 0..slashes / 2 {
                    units.push(b'\\' as u16);
                }
                if slashes % 2 == 1 {
                    units.push(b'"' as u16);
                    index += 1;
                } else if quoted
                    && index + 1 < command_line.len()
                    && command_line[index + 1] == b'"' as u16
                {
                    units.push(b'"' as u16);
                    index += 2;
                } else {
                    quoted = !quoted;
                    index += 1;
                }
                continue;
            }
            for _ in 0..slashes {
                units.push(b'\\' as u16);
            }
            if index == command_line.len()
                || (!quoted && matches!(command_line[index], 0x20 | 0x09))
            {
                break;
            }
            units.push(command_line[index]);
            index += 1;
        }
        let mut argument = Text::new();
        for character in core::char::decode_utf16(units.iter().copied()) {
            argument.push(character.expect("command line contains invalid UTF-16"));
        }
        arguments.push(argument);
    }
    arguments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_arguments_match_test_harness_arguments() {
        let native = command_arguments();
        assert!(!native.is_empty());
        assert!(!native[0].is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn windows_parser_preserves_quotes_empty_values_and_backslashes() {
        let source = r#"app.exe "two words" plain "say \"hi\"" "" C:\models\gemma.gguf"#;
        let wide = source.encode_utf16().collect::<Vector<_>>();
        let parsed = parse_windows_command_line(&wide);
        let values = parsed.iter().map(Text::as_str).collect::<Vector<_>>();
        assert_eq!(
            values,
            &[
                "app.exe",
                "two words",
                "plain",
                "say \"hi\"",
                "",
                r"C:\models\gemma.gguf",
            ][..]
        );
    }
}
