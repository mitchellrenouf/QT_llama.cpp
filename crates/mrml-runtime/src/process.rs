use crate::Text;

pub fn process_id() -> u32 {
    #[cfg(windows)]
    {
        mrml_windows::process_id()
    }
    #[cfg(unix)]
    {
        mrml_linux::process_id()
    }
}

pub fn temporary_directory() -> Text {
    #[cfg(windows)]
    {
        crate::environment_variable("TEMP")
            .or_else(|| crate::environment_variable("TMP"))
            .unwrap_or_else(|| ".".into())
    }
    #[cfg(unix)]
    {
        crate::environment_variable("TMPDIR").unwrap_or_else(|| "/tmp".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_live_process_and_temporary_directory() {
        assert_eq!(process_id(), std::process::id());
        assert!(crate::path_is_directory(&temporary_directory()));
    }
}
