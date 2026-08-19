use crate::{Text, Vector};

pub fn environment_variable(name: &str) -> Option<Text> {
    if name.as_bytes().contains(&0) {
        return None;
    }
    #[cfg(windows)]
    {
        let mut wide_name = Vector::with_capacity(name.len() + 1).ok()?;
        wide_name.extend(name.encode_utf16());
        wide_name.push(0);
        let mut buffer = Vector::with_capacity(128).ok()?;
        buffer.resize(128, 0);
        let length = match mrml_windows::environment_variable_wide(&wide_name, &mut buffer) {
            Ok(length) => length,
            Err(0) => return None,
            Err(required) => {
                buffer.resize(required, 0);
                mrml_windows::environment_variable_wide(&wide_name, &mut buffer).ok()?
            }
        };
        let mut output = Text::new();
        for character in core::char::decode_utf16(buffer[..length].iter().copied()) {
            output.push(character.ok()?);
        }
        Some(output)
    }
    #[cfg(unix)]
    {
        let mut encoded_name = Vector::with_capacity(name.len() + 1).ok()?;
        encoded_name.try_extend_from_slice(name.as_bytes()).ok()?;
        encoded_name.push(0);
        let name = core::ffi::CStr::from_bytes_with_nul(&encoded_name).ok()?;
        let mut buffer = Vector::with_capacity(128).ok()?;
        buffer.resize(128, 0);
        let length = match mrml_linux::environment_variable_bytes(name, &mut buffer) {
            Ok(length) => length,
            Err(0) => return None,
            Err(required) => {
                buffer.resize(required, 0);
                mrml_linux::environment_variable_bytes(name, &mut buffer).ok()?
            }
        };
        buffer.truncate(length);
        Text::try_from_utf8(buffer).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_environment_into_mrml_text() {
        let path = environment_variable("PATH").expect("PATH should be set");
        assert!(!path.is_empty());
        assert_eq!(environment_variable("MRML_ENV_TEST_DEFINITELY_MISSING"), None);
    }

    #[test]
    fn preserves_unicode_and_long_environment_values() {
        const NAME: &str = "MRML_RUNTIME_ENV_UNICODE_TEST";
        let expected = "observatory-λ-星".repeat(24);
        unsafe { std::env::set_var(NAME, &expected) };
        assert_eq!(environment_variable(NAME).as_deref(), Some(expected.as_str()));
        unsafe { std::env::remove_var(NAME) };
    }
}
