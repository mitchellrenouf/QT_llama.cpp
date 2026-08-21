#![no_std]
#![cfg_attr(not(test), no_main)]

use mrml_crypto::{
    LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, Sha3_512,
    lamport_public_key, lamport_sign, lamport_verify,
};
use mrml_error::{Result, anyhow};
use mrml_kernel::{
    ArtifactKind, PeImage, ReleaseManifest, SIGNED_ARTIFACT_HEADER_BYTES, artifact_statement,
    executable_image_limit,
};
use mrml_runtime::{Text, Vector, mrml_println as println};

const MAX_ARTIFACT: usize = 512 * 1024 * 1024;

fn application_main() -> Result<()> {
    let args = mrml_runtime::command_arguments();
    match args.get(1).map(|value| value.as_str()) {
        Some("keygen") if args.len() == 4 => keygen(&args[2], &args[3]),
        Some("sign") if args.len() == 7 => sign(
            parse_kind(&args[2])?,
            args[3]
                .parse()
                .map_err(|_| anyhow!("invalid artifact version"))?,
            &args[4],
            &args[5],
            &args[6],
        ),
        Some("sign-bundle") if args.len() == 7 => sign_bundle(
            parse_kind(&args[2])?,
            args[3]
                .parse()
                .map_err(|_| anyhow!("invalid artifact version"))?,
            &args[4],
            &args[5],
            &args[6],
        ),
        Some("key-digest") if args.len() == 3 => key_digest(&args[2]),
        Some("manifest") if args.len() == 10 => manifest(&args),
        _ => Err(anyhow!(
            "usage:\n  mrml-sign keygen PRIVATE PUBLIC\n  mrml-sign sign KIND VERSION ARTIFACT PRIVATE SIGNATURE\n  mrml-sign sign-bundle KIND VERSION ARTIFACT PRIVATE OUTPUT\n  mrml-sign key-digest PUBLIC\n  mrml-sign manifest VERSION OUTPUT NEXT_ROOT KERNEL VM SERVICE CUDA POLICY"
        )),
    }
}

fn sign_bundle(
    kind: ArtifactKind,
    version: u64,
    artifact_path: &str,
    private_path: &str,
    output_path: &str,
) -> Result<()> {
    if version == 0 || mrml_runtime::path_exists(output_path) {
        return Err(anyhow!("invalid version or output already exists"));
    }
    let artifact = mrml_runtime::read_file_bounded(artifact_path, MAX_ARTIFACT)?;
    if artifact.is_empty() {
        return Err(anyhow!("artifact must not be empty"));
    }
    if let Some(maximum) = executable_image_limit(kind) {
        PeImage::parse_with_limit(&artifact, maximum)
            .map_err(|_| anyhow!("OS executable violates its MRML PE32+ size or format policy"))?;
    }
    let mut private = mrml_runtime::read_file_bounded(private_path, LAMPORT_PRIVATE_KEY_BYTES)?;
    if private.len() != LAMPORT_PRIVATE_KEY_BYTES {
        return Err(anyhow!("invalid private-key length"));
    }
    let digest = Sha3_512::digest(&artifact);
    let statement = artifact_statement(kind, version, artifact.len() as u64, digest);
    let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
    let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
    lamport_public_key(&private, &mut public).map_err(|_| anyhow!("key derivation failed"))?;
    lamport_sign(&private, &statement, &mut signature).map_err(|_| anyhow!("signing failed"))?;
    for byte in private.iter_mut() {
        *byte = 0;
    }
    lamport_verify(&public, &statement, &signature)
        .map_err(|_| anyhow!("self-verification failed; bundle was not written"))?;

    let length = SIGNED_ARTIFACT_HEADER_BYTES
        .checked_add(LAMPORT_PUBLIC_KEY_BYTES)
        .and_then(|value| value.checked_add(LAMPORT_SIGNATURE_BYTES))
        .and_then(|value| value.checked_add(artifact.len()))
        .ok_or_else(|| anyhow!("signed bundle length overflow"))?;
    let mut bundle = Vector::with_capacity(length).map_err(|_| anyhow!("allocation failed"))?;
    let mut header = [0u8; SIGNED_ARTIFACT_HEADER_BYTES];
    header[..16].copy_from_slice(b"MRML-SIGNED-v1\0\0");
    header[16] = kind as u8;
    header[24..32].copy_from_slice(&version.to_le_bytes());
    header[32..40].copy_from_slice(&(artifact.len() as u64).to_le_bytes());
    header[40..104].copy_from_slice(&digest);
    bundle
        .try_extend_from_slice(&header)
        .and_then(|_| bundle.try_extend_from_slice(&public))
        .and_then(|_| bundle.try_extend_from_slice(&signature))
        .and_then(|_| bundle.try_extend_from_slice(&artifact))
        .map_err(|_| anyhow!("allocation failed"))?;
    mrml_runtime::write_file(output_path, &bundle)?;
    mrml_runtime::remove_file(private_path)?;
    println!(
        "wrote canonical signed {:?} version {} bundle; consumed one-time private key",
        kind, version
    );
    Ok(())
}

fn manifest(args: &[Text]) -> Result<()> {
    let version = args[2]
        .parse()
        .map_err(|_| anyhow!("invalid release version"))?;
    if mrml_runtime::path_exists(&args[3]) {
        return Err(anyhow!("refusing to overwrite release manifest"));
    }
    let next = parse_digest(&args[4])?;
    let roots = [
        parse_digest(&args[5])?,
        parse_digest(&args[6])?,
        parse_digest(&args[7])?,
        parse_digest(&args[8])?,
        parse_digest(&args[9])?,
    ];
    let encoded = ReleaseManifest::new(version, next, roots)
        .map_err(|_| anyhow!("invalid release manifest"))?
        .encode();
    mrml_runtime::write_file(&args[3], &encoded)?;
    println!("wrote canonical release {} manifest", version);
    Ok(())
}

fn parse_digest(value: &str) -> Result<[u8; 64]> {
    if value.len() != 128 {
        return Err(anyhow!(
            "trust-root digest must contain 128 hexadecimal characters"
        ));
    }
    let bytes = value.as_bytes();
    let mut digest = [0u8; 64];
    for index in 0..64 {
        digest[index] = (hex(bytes[index * 2])? << 4) | hex(bytes[index * 2 + 1])?;
    }
    Ok(digest)
}

fn hex(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(anyhow!("invalid hexadecimal trust-root digest")),
    }
}

fn keygen(private_path: &str, public_path: &str) -> Result<()> {
    if mrml_runtime::path_exists(private_path) || mrml_runtime::path_exists(public_path) {
        return Err(anyhow!("refusing to overwrite signing key material"));
    }
    let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
    mrml_runtime::fill_random(&mut private).map_err(|_| anyhow!("OS random generation failed"))?;
    let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
    lamport_public_key(&private, &mut public).map_err(|_| anyhow!("key derivation failed"))?;
    mrml_runtime::write_file(private_path, &private)?;
    mrml_runtime::write_file(public_path, &public)?;
    println!("generated one-time Lamport key; destroy PRIVATE after exactly one signature");
    Ok(())
}

fn sign(
    kind: ArtifactKind,
    version: u64,
    artifact_path: &str,
    private_path: &str,
    signature_path: &str,
) -> Result<()> {
    if mrml_runtime::path_exists(signature_path) {
        return Err(anyhow!("refusing to overwrite signature"));
    }
    let artifact = mrml_runtime::read_file_bounded(artifact_path, MAX_ARTIFACT)?;
    let mut private = mrml_runtime::read_file_bounded(private_path, LAMPORT_PRIVATE_KEY_BYTES)?;
    if private.len() != LAMPORT_PRIVATE_KEY_BYTES {
        return Err(anyhow!("invalid private-key length"));
    }
    let statement = artifact_statement(
        kind,
        version,
        artifact.len() as u64,
        Sha3_512::digest(&artifact),
    );
    let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
    lamport_sign(&private, &statement, &mut signature).map_err(|_| anyhow!("signing failed"))?;
    let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
    lamport_public_key(&private, &mut public).map_err(|_| anyhow!("key derivation failed"))?;
    for byte in private.iter_mut() {
        *byte = 0;
    }
    lamport_verify(&public, &statement, &signature)
        .map_err(|_| anyhow!("self-verification failed; signature was not written"))?;
    mrml_runtime::write_file(signature_path, &signature)?;
    mrml_runtime::remove_file(private_path)?;
    println!(
        "signed {:?} version {}; consumed and removed the one-time private key",
        kind, version
    );
    Ok(())
}

fn key_digest(path: &str) -> Result<()> {
    let public = mrml_runtime::read_file_bounded(path, LAMPORT_PUBLIC_KEY_BYTES)?;
    if public.len() != LAMPORT_PUBLIC_KEY_BYTES {
        return Err(anyhow!("invalid public-key length"));
    }
    let digest = Sha3_512::digest(&public);
    let mut output = Text::with_capacity(128).map_err(|_| anyhow!("allocation failed"))?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 15) as usize] as char);
    }
    println!("{}", output);
    Ok(())
}

fn parse_kind(value: &str) -> Result<ArtifactKind> {
    match value {
        "kernel" => Ok(ArtifactKind::Kernel),
        "vm" => Ok(ArtifactKind::VmImage),
        "service" => Ok(ArtifactKind::ServiceImage),
        "cuda" => Ok(ArtifactKind::CudaKernelBundle),
        "policy" => Ok(ArtifactKind::LaunchPolicy),
        _ => Err(anyhow!("kind must be kernel, vm, service, cuda, or policy")),
    }
}

mrml_runtime::mrml_entrypoint!(application_main);
