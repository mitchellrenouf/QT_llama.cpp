#![no_std]
#![cfg_attr(not(test), no_main)]

use mrml_crypto::{
    LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, Sha3_512,
    lamport_public_key, lamport_sign, lamport_verify,
};
use mrml_error::{Result, anyhow};
use mrml_kernel::{ArtifactKind, artifact_statement};
use mrml_runtime::{Text, mrml_println as println};

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
        Some("key-digest") if args.len() == 3 => key_digest(&args[2]),
        _ => Err(anyhow!(
            "usage:\n  mrml-sign keygen PRIVATE PUBLIC\n  mrml-sign sign KIND VERSION ARTIFACT PRIVATE SIGNATURE\n  mrml-sign key-digest PUBLIC"
        )),
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
