use crate::{
    ObjectId, Repository, extract_pack_response, fetch_request, parse_advertisement,
    parse_push_response_mode, push_request_mode,
};
use core::fmt;
use mrml_runtime::{Text, Vector};
use mrml_ssh::{AuthenticatedSsh, ChannelEvent, GitChannel, GitService, RsaPrivateKey, SshRemote};

const MAX_WIRE: usize = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchResult {
    pub objects: Vector<ObjectId>,
    pub branches: Vector<(Text, ObjectId)>,
    pub default_branch: Option<Text>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FetchError {
    Ssh,
    Protocol,
    Remote,
    Pack,
    Repository,
    TooLarge,
    NoReferences,
}

pub fn fetch_ssh(
    repository: &Repository,
    remote_name: &str,
    remote: &SshRemote,
    key: &RsaPrivateKey,
    host_key: &[u8],
) -> Result<FetchResult, FetchError> {
    let user = remote.user.as_deref().unwrap_or("git");
    let authenticated =
        AuthenticatedSsh::connect(&remote.host, remote.port.unwrap_or(22), user, key, host_key)
            .map_err(|_| FetchError::Ssh)?;
    let mut channel = GitChannel::open(authenticated, GitService::UploadPack, &remote.path)
        .map_err(|_| FetchError::Ssh)?;
    let advertisement_bytes = collect_until_flush(&mut channel)?;
    let advertisement =
        parse_advertisement(&advertisement_bytes).map_err(|_| FetchError::Protocol)?;
    let mut wants = Vector::new();
    for reference in &advertisement.refs {
        if reference.name.starts_with("refs/heads/") && !wants.contains(&reference.id) {
            wants.push(reference.id);
        }
    }
    if wants.is_empty() {
        channel.send_all(b"0000").map_err(|_| FetchError::Ssh)?;
        channel.send_eof().map_err(|_| FetchError::Ssh)?;
        collect_to_close(&mut channel)?;
        repository
            .prune_remote_refs(remote_name, &[])
            .map_err(|_| FetchError::Repository)?;
        return Ok(FetchResult {
            objects: Vector::new(),
            branches: Vector::new(),
            default_branch: None,
        });
    }
    let side_band = advertisement
        .capabilities
        .iter()
        .any(|capability| capability == "side-band-64k");
    let mut capabilities = Vector::new();
    if side_band {
        capabilities.push("side-band-64k");
    }
    if advertisement
        .capabilities
        .iter()
        .any(|capability| capability == "ofs-delta")
    {
        capabilities.push("ofs-delta");
    }
    let request = fetch_request(&wants, &[], &capabilities).map_err(|_| FetchError::Protocol)?;
    channel.send_all(&request).map_err(|_| FetchError::Ssh)?;
    channel.send_eof().map_err(|_| FetchError::Ssh)?;
    let response = collect_to_close(&mut channel)?;
    let pack = extract_pack_response(&response, side_band).map_err(|_| FetchError::Protocol)?;
    let objects = repository
        .import_pack(&pack)
        .map_err(|_| FetchError::Pack)?;
    let default_branch = advertisement.capabilities.iter().find_map(|capability| {
        capability
            .strip_prefix("symref=HEAD:refs/heads/")
            .map(Into::into)
    });
    let mut branches: Vector<(Text, ObjectId)> = Vector::new();
    for reference in advertisement.refs {
        if let Some(branch) = reference.name.strip_prefix("refs/heads/") {
            repository
                .update_remote_ref(remote_name, &reference.name, reference.id)
                .map_err(|_| FetchError::Repository)?;
            branches.push((branch.into(), reference.id));
        }
    }
    let live: Vector<Text> = branches.iter().map(|(branch, _)| branch.clone()).collect();
    repository
        .prune_remote_refs(remote_name, &live)
        .map_err(|_| FetchError::Repository)?;
    Ok(FetchResult {
        objects,
        branches,
        default_branch,
    })
}

fn append(output: &mut Vector<u8>, bytes: &[u8]) -> Result<(), FetchError> {
    if output
        .len()
        .checked_add(bytes.len())
        .is_none_or(|n| n > MAX_WIRE)
    {
        return Err(FetchError::TooLarge);
    }
    output.extend(bytes.iter().copied());
    Ok(())
}
fn collect_until_flush(channel: &mut GitChannel) -> Result<Vector<u8>, FetchError> {
    let mut output = Vector::new();
    loop {
        match channel.receive_event().map_err(|_| FetchError::Ssh)? {
            ChannelEvent::Data { bytes, .. } => {
                append(&mut output, &bytes)?;
                if output.windows(4).any(|part| part == b"0000") {
                    return Ok(output);
                }
            }
            ChannelEvent::ExtendedData { .. } => return Err(FetchError::Remote),
            ChannelEvent::Close(_) | ChannelEvent::Eof(_) => return Err(FetchError::Protocol),
            _ => {}
        }
    }
}
fn collect_to_close(channel: &mut GitChannel) -> Result<Vector<u8>, FetchError> {
    let mut output = Vector::new();
    let mut status = None;
    loop {
        match channel.receive_event().map_err(|_| FetchError::Ssh)? {
            ChannelEvent::Data { bytes, .. } => append(&mut output, &bytes)?,
            ChannelEvent::ExtendedData { .. } => return Err(FetchError::Remote),
            ChannelEvent::ExitStatus { status: value, .. } => status = Some(value),
            ChannelEvent::Close(_) => {
                return if status.is_none_or(|value| value == 0) {
                    Ok(output)
                } else {
                    Err(FetchError::Remote)
                };
            }
            _ => {}
        }
    }
}
impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ssh => "native SSH transport failed",
            Self::Protocol => "remote Git protocol response is invalid",
            Self::Remote => "remote Git service failed",
            Self::Pack => "received Git pack failed authentication",
            Self::Repository => "failed to update remote-tracking references",
            Self::TooLarge => "remote response exceeds limit",
            Self::NoReferences => "remote advertises no branch references",
        })
    }
}
impl core::error::Error for FetchError {}

pub fn check_ssh(
    remote: &SshRemote,
    key: &RsaPrivateKey,
    host_key: &[u8],
) -> Result<usize, FetchError> {
    let user = remote.user.as_deref().unwrap_or("git");
    let authenticated =
        AuthenticatedSsh::connect(&remote.host, remote.port.unwrap_or(22), user, key, host_key)
            .map_err(|_| FetchError::Ssh)?;
    let mut channel = GitChannel::open(authenticated, GitService::UploadPack, &remote.path)
        .map_err(|_| FetchError::Ssh)?;
    let bytes = collect_until_flush(&mut channel)?;
    Ok(parse_advertisement(&bytes)
        .map_err(|_| FetchError::Protocol)?
        .refs
        .len())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushResult {
    pub old: ObjectId,
    pub new: ObjectId,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushError {
    Ssh,
    Protocol,
    Remote,
    Repository,
    TooLarge,
    NonFastForward,
    NeedsFetch,
}
pub fn push_ssh(
    repository: &Repository,
    remote_name: &str,
    branch: &str,
    remote: &SshRemote,
    key: &RsaPrivateKey,
    host_key: &[u8],
) -> Result<PushResult, PushError> {
    let user = remote.user.as_deref().unwrap_or("git");
    let authenticated =
        AuthenticatedSsh::connect(&remote.host, remote.port.unwrap_or(22), user, key, host_key)
            .map_err(|_| PushError::Ssh)?;
    let mut channel = GitChannel::open(authenticated, GitService::ReceivePack, &remote.path)
        .map_err(|_| PushError::Ssh)?;
    let advertisement_bytes = collect_until_flush(&mut channel).map_err(map_fetch_push)?;
    let advertisement =
        parse_advertisement(&advertisement_bytes).map_err(|_| PushError::Protocol)?;
    if !advertisement
        .capabilities
        .iter()
        .any(|cap| cap == "report-status")
    {
        return Err(PushError::Protocol);
    }
    let side_band = advertisement
        .capabilities
        .iter()
        .any(|cap| cap == "side-band-64k");
    let reference = mrml_runtime::mrml_format!("refs/heads/{branch}");
    let old = advertisement
        .refs
        .iter()
        .find(|entry| entry.name == reference)
        .map(|entry| entry.id)
        .unwrap_or(ObjectId([0; 20]));
    let new = repository
        .resolve_revision(branch)
        .map_err(|_| PushError::Repository)?;
    if old == new {
        return Ok(PushResult { old, new });
    }
    if old != ObjectId([0; 20]) {
        match repository.is_ancestor(old, new) {
            Ok(true) => {}
            Ok(false) => return Err(PushError::NonFastForward),
            Err(_) => return Err(PushError::NeedsFetch),
        }
    }
    let pack = repository
        .pack_reachable(new)
        .map_err(|_| PushError::Repository)?;
    let request = push_request_mode(old, new, &reference, &pack, side_band)
        .map_err(|_| PushError::Protocol)?;
    channel.send_all(&request).map_err(|_| PushError::Ssh)?;
    channel.send_eof().map_err(|_| PushError::Ssh)?;
    let response = collect_to_close(&mut channel).map_err(map_fetch_push)?;
    parse_push_response_mode(&response, &reference, side_band).map_err(|_| PushError::Remote)?;
    repository
        .update_remote_ref(remote_name, &reference, new)
        .map_err(|_| PushError::Repository)?;
    Ok(PushResult { old, new })
}
fn map_fetch_push(error: FetchError) -> PushError {
    match error {
        FetchError::Ssh => PushError::Ssh,
        FetchError::TooLarge => PushError::TooLarge,
        FetchError::Remote => PushError::Remote,
        _ => PushError::Protocol,
    }
}
impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ssh => "native SSH transport failed",
            Self::Protocol => "remote receive-pack protocol is unsupported or invalid",
            Self::Remote => "remote rejected the pushed reference",
            Self::Repository => "failed to read or update repository state",
            Self::TooLarge => "push response exceeds limit",
            Self::NonFastForward => "push is not a fast-forward",
            Self::NeedsFetch => "remote tip is not available locally; fetch before pushing",
        })
    }
}
impl core::error::Error for PushError {}
