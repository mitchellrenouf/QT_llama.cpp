use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetError {
    InvalidAddress,
    BindFailed,
    ConnectFailed,
    AcceptFailed,
    ReadFailed,
    WriteFailed,
    TimeoutFailed,
}

impl fmt::Display for NetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidAddress => "invalid IPv4 address",
            Self::BindFailed => "failed to bind TCP listener",
            Self::ConnectFailed => "failed to connect TCP stream",
            Self::AcceptFailed => "failed to accept TCP connection",
            Self::ReadFailed => "failed to read TCP stream",
            Self::WriteFailed => "failed to write TCP stream",
            Self::TimeoutFailed => "failed to configure TCP timeout",
        })
    }
}

impl core::error::Error for NetError {}

pub struct TcpListener {
    #[cfg(windows)]
    native: mrml_windows::NativeTcpListener,
    #[cfg(unix)]
    native: mrml_linux::NativeTcpListener,
}

impl TcpListener {
    pub fn bind(ip: [u8; 4], port: u16) -> Result<Self, NetError> {
        #[cfg(windows)]
        let native = mrml_windows::NativeTcpListener::bind(ip, port);
        #[cfg(unix)]
        let native = mrml_linux::NativeTcpListener::bind(ip, port);
        native.map(|native| Self { native }).ok_or(NetError::BindFailed)
    }

    pub fn local_port(&self) -> Result<u16, NetError> {
        self.native.local_port().ok_or(NetError::BindFailed)
    }

    pub fn accept(&self) -> Result<TcpStream, NetError> {
        self.native.accept().map(|native| TcpStream { native }).ok_or(NetError::AcceptFailed)
    }
}

pub struct TcpStream {
    #[cfg(windows)]
    native: mrml_windows::NativeTcpStream,
    #[cfg(unix)]
    native: mrml_linux::NativeTcpStream,
}

impl TcpStream {
    pub fn connect(ip: [u8; 4], port: u16) -> Result<Self, NetError> {
        #[cfg(windows)]
        let native = mrml_windows::NativeTcpStream::connect(ip, port);
        #[cfg(unix)]
        let native = mrml_linux::NativeTcpStream::connect(ip, port);
        native.map(|native| Self { native }).ok_or(NetError::ConnectFailed)
    }

    pub fn read(&mut self, buffer: &mut [u8]) -> Result<usize, NetError> {
        self.native.read(buffer).ok_or(NetError::ReadFailed)
    }

    pub fn read_exact(&mut self, mut buffer: &mut [u8]) -> Result<(), NetError> {
        while !buffer.is_empty() {
            let read = self.read(buffer)?;
            if read == 0 { return Err(NetError::ReadFailed); }
            buffer = &mut buffer[read..];
        }
        Ok(())
    }

    pub fn write_all(&mut self, mut buffer: &[u8]) -> Result<(), NetError> {
        while !buffer.is_empty() {
            let written = self.native.write(buffer).ok_or(NetError::WriteFailed)?;
            if written == 0 { return Err(NetError::WriteFailed); }
            buffer = &buffer[written..];
        }
        Ok(())
    }

    pub fn set_read_timeout_millis(&self, milliseconds: u64) -> Result<(), NetError> {
        self.native.set_timeout_millis(true, milliseconds).then_some(()).ok_or(NetError::TimeoutFailed)
    }

    pub fn set_write_timeout_millis(&self, milliseconds: u64) -> Result<(), NetError> {
        self.native.set_timeout_millis(false, milliseconds).then_some(()).ok_or(NetError::TimeoutFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_listener_exchanges_bytes() {
        let listener = TcpListener::bind([127, 0, 0, 1], 0).unwrap();
        let port = listener.local_port().unwrap();
        assert!(crate::spawn_detached(move || {
            let mut stream = listener.accept().unwrap();
            let mut request = [0u8; 5];
            stream.read_exact(&mut request).unwrap();
            assert_eq!(&request, b"hello");
            stream.write_all(b"world").unwrap();
        }).is_ok());
        let mut stream = TcpStream::connect([127, 0, 0, 1], port).unwrap();
        stream.set_read_timeout_millis(1000).unwrap();
        stream.set_write_timeout_millis(1000).unwrap();
        stream.write_all(b"hello").unwrap();
        let mut response = [0u8; 5];
        stream.read_exact(&mut response).unwrap();
        assert_eq!(&response, b"world");
    }
}
