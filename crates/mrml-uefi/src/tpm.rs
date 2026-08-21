const TPM_ST_NO_SESSIONS: u16 = 0x8001;
const TPM_ST_SESSIONS: u16 = 0x8002;
const TPM_RS_PW: u32 = 0x4000_0009;
const TPM_CC_NV_INCREMENT: u32 = 0x0000_0137;
const TPM_CC_NV_READ: u32 = 0x0000_014e;
const TPM_RC_SUCCESS: u32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NvCounterError {
    InvalidIndex,
    Rollback,
    AdvanceLimit,
    Transport,
    MalformedResponse,
    Tpm(u32),
}

pub trait TpmTransport {
    fn submit(&mut self, command: &[u8], response: &mut [u8]) -> Result<usize, NvCounterError>;
}

pub fn enforce_version<T: TpmTransport>(
    transport: &mut T,
    index: u32,
    version: u64,
    maximum_advances: u64,
) -> Result<u64, NvCounterError> {
    if !(0x0100_0000..=0x01ff_ffff).contains(&index) || version == 0 {
        return Err(NvCounterError::InvalidIndex);
    }
    let current = read_counter(transport, index)?;
    if current > version {
        return Err(NvCounterError::Rollback);
    }
    let advances = version - current;
    if advances > maximum_advances {
        return Err(NvCounterError::AdvanceLimit);
    }
    for _ in 0..advances {
        increment_counter(transport, index)?;
    }
    Ok(version)
}

pub fn read_counter<T: TpmTransport>(transport: &mut T, index: u32) -> Result<u64, NvCounterError> {
    let mut command = [0u8; 35];
    sessions_header(&mut command, TPM_CC_NV_READ);
    command[10..14].copy_from_slice(&index.to_be_bytes());
    command[14..18].copy_from_slice(&index.to_be_bytes());
    command[18..22].copy_from_slice(&9u32.to_be_bytes());
    password_session(&mut command[22..31]);
    command[31..33].copy_from_slice(&8u16.to_be_bytes());
    let mut response = [0u8; 64];
    let length = transport.submit(&command, &mut response)?;
    parse_counter_response(&response[..length])
}

pub fn increment_counter<T: TpmTransport>(
    transport: &mut T,
    index: u32,
) -> Result<(), NvCounterError> {
    let mut command = [0u8; 31];
    sessions_header(&mut command, TPM_CC_NV_INCREMENT);
    command[10..14].copy_from_slice(&index.to_be_bytes());
    command[14..18].copy_from_slice(&index.to_be_bytes());
    command[18..22].copy_from_slice(&9u32.to_be_bytes());
    password_session(&mut command[22..31]);
    let mut response = [0u8; 32];
    let length = transport.submit(&command, &mut response)?;
    parse_empty_response(&response[..length])
}

fn sessions_header(command: &mut [u8], code: u32) {
    let length = command.len() as u32;
    command[..2].copy_from_slice(&TPM_ST_SESSIONS.to_be_bytes());
    command[2..6].copy_from_slice(&length.to_be_bytes());
    command[6..10].copy_from_slice(&code.to_be_bytes());
}

fn password_session(output: &mut [u8]) {
    output[..4].copy_from_slice(&TPM_RS_PW.to_be_bytes());
    output[4..6].copy_from_slice(&0u16.to_be_bytes());
    output[6] = 0;
    output[7..9].copy_from_slice(&0u16.to_be_bytes());
}

fn response_header(response: &[u8]) -> Result<(u16, usize), NvCounterError> {
    if response.len() < 10 {
        return Err(NvCounterError::MalformedResponse);
    }
    let tag = u16::from_be_bytes([response[0], response[1]]);
    let declared = u32::from_be_bytes(response[2..6].try_into().unwrap()) as usize;
    let code = u32::from_be_bytes(response[6..10].try_into().unwrap());
    if declared != response.len() || !matches!(tag, TPM_ST_NO_SESSIONS | TPM_ST_SESSIONS) {
        return Err(NvCounterError::MalformedResponse);
    }
    if code != TPM_RC_SUCCESS {
        return Err(NvCounterError::Tpm(code));
    }
    Ok((tag, declared))
}

fn parse_counter_response(response: &[u8]) -> Result<u64, NvCounterError> {
    let (tag, _) = response_header(response)?;
    if tag != TPM_ST_SESSIONS || response.len() < 24 {
        return Err(NvCounterError::MalformedResponse);
    }
    let parameter_size = u32::from_be_bytes(response[10..14].try_into().unwrap()) as usize;
    if parameter_size != 10 || response.len() != 14 + parameter_size + 5 {
        return Err(NvCounterError::MalformedResponse);
    }
    if u16::from_be_bytes(response[14..16].try_into().unwrap()) != 8 {
        return Err(NvCounterError::MalformedResponse);
    }
    parse_empty_auth(&response[24..])?;
    Ok(u64::from_be_bytes(response[16..24].try_into().unwrap()))
}

fn parse_empty_response(response: &[u8]) -> Result<(), NvCounterError> {
    let (tag, _) = response_header(response)?;
    if tag != TPM_ST_SESSIONS || response.len() != 19 {
        return Err(NvCounterError::MalformedResponse);
    }
    let parameter_size = u32::from_be_bytes(response[10..14].try_into().unwrap());
    if parameter_size != 0 {
        return Err(NvCounterError::MalformedResponse);
    }
    parse_empty_auth(&response[14..])
}

fn parse_empty_auth(response: &[u8]) -> Result<(), NvCounterError> {
    if response == [0, 0, 0, 0, 0] {
        Ok(())
    } else {
        Err(NvCounterError::MalformedResponse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        value: u64,
        calls: usize,
    }

    impl TpmTransport for Counter {
        fn submit(&mut self, command: &[u8], response: &mut [u8]) -> Result<usize, NvCounterError> {
            self.calls += 1;
            let code = u32::from_be_bytes(command[6..10].try_into().unwrap());
            if code == TPM_CC_NV_INCREMENT {
                self.value += 1;
                response[..19].copy_from_slice(&[
                    0x80, 0x02, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]);
                return Ok(19);
            }
            response[..29].copy_from_slice(&[
                0x80, 0x02, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 10, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]);
            response[16..24].copy_from_slice(&self.value.to_be_bytes());
            Ok(29)
        }
    }

    #[test]
    fn advances_monotonically_and_rejects_rollback() {
        let mut counter = Counter { value: 3, calls: 0 };
        assert_eq!(enforce_version(&mut counter, 0x0180_0001, 5, 8), Ok(5));
        assert_eq!(counter.value, 5);
        assert_eq!(counter.calls, 3);
        assert_eq!(
            enforce_version(&mut counter, 0x0180_0001, 4, 8),
            Err(NvCounterError::Rollback)
        );
    }

    #[test]
    fn bounds_advancement_and_rejects_malformed_responses() {
        let mut counter = Counter { value: 1, calls: 0 };
        assert_eq!(
            enforce_version(&mut counter, 0x0180_0001, 10, 4),
            Err(NvCounterError::AdvanceLimit)
        );
        assert_eq!(
            response_header(&[0; 9]),
            Err(NvCounterError::MalformedResponse)
        );
    }
}
