use core::fmt::{self, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub const fn as_gemma(self) -> &'static str {
        match self {
            Self::System | Self::Developer => "system",
            Self::User => "user",
            Self::Assistant => "model",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BorrowedMessage<'a> {
    pub role: ChatRole,
    pub content: &'a str,
}

impl<'a> BorrowedMessage<'a> {
    pub const fn new(role: ChatRole, content: &'a str) -> Self {
        Self { role, content }
    }
}

pub fn render_gemma4_borrowed(
    output: &mut impl Write,
    messages: &[BorrowedMessage<'_>],
    generation_prompt: bool,
) -> fmt::Result {
    output.write_str("<bos>")?;
    for message in messages {
        write!(
            output,
            "<|turn>{}\n{}<turn|>\n",
            message.role.as_gemma(),
            message.content.trim()
        )?;
    }
    if generation_prompt {
        output.write_str("<|turn>model\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Buffer {
        bytes: [u8; 160],
        len: usize,
    }

    impl Write for Buffer {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            let end = self.len + value.len();
            if end > self.bytes.len() {
                return Err(fmt::Error);
            }
            self.bytes[self.len..end].copy_from_slice(value.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    #[test]
    fn renders_gemma_chat_into_caller_storage() {
        let messages = [
            BorrowedMessage::new(ChatRole::System, " Be concise. "),
            BorrowedMessage::new(ChatRole::User, " Hello "),
        ];
        let mut output = Buffer {
            bytes: [0; 160],
            len: 0,
        };
        render_gemma4_borrowed(&mut output, &messages, true).unwrap();
        assert_eq!(
            &output.bytes[..output.len],
            b"<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
        );
    }
}
