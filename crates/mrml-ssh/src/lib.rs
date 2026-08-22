#![no_std]

use mrml_runtime::Text;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SshRemote {
    pub user: Option<Text>,
    pub host: Text,
    pub port: Option<u16>,
    pub path: Text,
}

impl SshRemote {
    pub fn parse(source: &str) -> Result<Self, Error> {
        validate_text(source)?;
        if let Some(authority_and_path) = source.strip_prefix("ssh://") {
            Self::parse_uri(authority_and_path)
        } else if source.contains("://") {
            Err(Error::UnsupportedScheme)
        } else {
            Self::parse_scp(source)
        }
    }

    fn parse_uri(source: &str) -> Result<Self, Error> {
        let slash = source.find('/').ok_or(Error::MissingPath)?;
        let (authority, path) = source.split_at(slash);
        if authority.is_empty() || path.len() <= 1 {
            return Err(Error::MissingPath);
        }
        let (user, host_port) = split_user(authority)?;
        let (host, port) = split_host_port(host_port)?;
        validate_path(&path[1..])?;
        Ok(Self {
            user: user.map(Into::into),
            host: host.into(),
            port,
            path: path[1..].into(),
        })
    }

    fn parse_scp(source: &str) -> Result<Self, Error> {
        let (user, host_and_path) = split_user(source)?;
        let separator = if host_and_path.starts_with('[') {
            host_and_path.find("]:").map(|index| index + 1)
        } else {
            host_and_path.find(':')
        };
        let separator = separator.ok_or(Error::MissingPath)?;
        let host = &host_and_path[..separator];
        let path = &host_and_path[separator + 1..];
        let host = host.trim_matches(['[', ']']);
        validate_host(host)?;
        validate_path(path)?;
        Ok(Self {
            user: user.map(Into::into),
            host: host.into(),
            port: None,
            path: path.into(),
        })
    }

    pub fn destination(&self) -> Text {
        match &self.user {
            Some(user) => mrml_runtime::mrml_format!("{}@{}", user, self.host),
            None => self.host.clone(),
        }
    }
}

fn split_user(authority: &str) -> Result<(Option<&str>, &str), Error> {
    match authority.rsplit_once('@') {
        Some((user, host)) => {
            if user.is_empty() || host.is_empty() {
                return Err(Error::InvalidAuthority);
            }
            if user.contains(':') {
                return Err(Error::PasswordNotAllowed);
            }
            validate_user(user)?;
            Ok((Some(user), host))
        }
        None => Ok((None, authority)),
    }
}

fn split_host_port(authority: &str) -> Result<(&str, Option<u16>), Error> {
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']').ok_or(Error::InvalidAuthority)?;
        let host = &rest[..end];
        let suffix = &rest[end + 1..];
        let port = if suffix.is_empty() {
            None
        } else {
            let value = suffix
                .strip_prefix(':')
                .ok_or(Error::InvalidPort)?
                .parse()
                .map_err(|_| Error::InvalidPort)?;
            if value == 0 {
                return Err(Error::InvalidPort);
            }
            Some(value)
        };
        validate_host(host)?;
        Ok((host, port))
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        if host.contains(':') {
            return Err(Error::InvalidAuthority);
        }
        validate_host(host)?;
        let port = port.parse().map_err(|_| Error::InvalidPort)?;
        if port == 0 {
            return Err(Error::InvalidPort);
        }
        Ok((host, Some(port)))
    } else {
        validate_host(authority)?;
        Ok((authority, None))
    }
}

fn validate_text(value: &str) -> Result<(), Error> {
    if value.is_empty() || value.chars().any(|character| character.is_control()) {
        Err(Error::InvalidText)
    } else {
        Ok(())
    }
}

fn validate_host(host: &str) -> Result<(), Error> {
    if host.is_empty()
        || host.starts_with('-')
        || host.chars().any(|character| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | ':'))
        })
    {
        Err(Error::InvalidAuthority)
    } else {
        Ok(())
    }
}

fn validate_user(user: &str) -> Result<(), Error> {
    if user.starts_with('-')
        || user.chars().any(|character| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-'))
        })
    {
        Err(Error::InvalidAuthority)
    } else {
        Ok(())
    }
}

fn validate_path(path: &str) -> Result<(), Error> {
    if path.is_empty()
        || path.starts_with('-')
        || path.chars().any(|character| {
            !(character.is_ascii_alphanumeric() || matches!(character, '/' | '.' | '_' | '-' | '~'))
        })
    {
        Err(Error::InvalidPath)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidText,
    UnsupportedScheme,
    PasswordNotAllowed,
    InvalidAuthority,
    InvalidPort,
    MissingPath,
    InvalidPath,
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidText => "SSH remote is empty or contains control characters",
            Self::UnsupportedScheme => "remote is not an SSH URL",
            Self::PasswordNotAllowed => "passwords are not allowed in SSH URLs",
            Self::InvalidAuthority => "SSH user or host is invalid",
            Self::InvalidPort => "SSH port is invalid",
            Self::MissingPath => "SSH repository path is missing",
            Self::InvalidPath => "SSH repository path contains unsafe characters",
        })
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scp_style_git_remote() {
        let remote = SshRemote::parse("git@github.com:owner/project.git").unwrap();
        assert_eq!(remote.user.as_deref(), Some("git"));
        assert_eq!(remote.host, "github.com");
        assert_eq!(remote.path, "owner/project.git");
        assert_eq!(remote.destination(), "git@github.com");
    }

    #[test]
    fn parses_uri_with_ipv6_and_port() {
        let remote = SshRemote::parse("ssh://git@[2001:db8::1]:2222/owner/repo.git").unwrap();
        assert_eq!(remote.host, "2001:db8::1");
        assert_eq!(remote.port, Some(2222));
    }

    #[test]
    fn rejects_passwords_non_ssh_schemes_and_controls() {
        assert_eq!(
            SshRemote::parse("ssh://git:secret@example.com/repo").unwrap_err(),
            Error::PasswordNotAllowed
        );
        assert_eq!(
            SshRemote::parse("https://example.com/repo").unwrap_err(),
            Error::UnsupportedScheme
        );
        assert_eq!(
            SshRemote::parse("git@example.com:repo\ncommand").unwrap_err(),
            Error::InvalidText
        );
        assert_eq!(
            SshRemote::parse("git@example.com:repo;command").unwrap_err(),
            Error::InvalidPath
        );
        assert_eq!(
            SshRemote::parse("bad user@example.com:repo").unwrap_err(),
            Error::InvalidAuthority
        );
    }
}
