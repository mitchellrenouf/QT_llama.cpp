#![no_std]
#![cfg_attr(not(test), no_main)]

use mrml_crypto::{LAMPORT_PUBLIC_KEY_BYTES, Sha3_512};
use mrml_error::{Result, anyhow};
use mrml_kernel::{
    ArtifactKind, SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact, TrustRoot, VmBackend, VmExit,
};
use mrml_kvm::{KvmLaunchLayout, KvmSystem};
use mrml_runtime::mrml_println as println;

const MAX_KERNEL_BUNDLE: usize = SIGNED_ARTIFACT_OVERHEAD_BYTES + 16 * 1024 * 1024;
const FRAMEBUFFER: u64 = 0x00a0_0000;

fn application_main() -> Result<()> {
    let arguments = mrml_runtime::command_arguments();
    if arguments.len() != 3 {
        return Err(anyhow!("usage: mrml-kvm-run KERNEL.signed RELEASE.public"));
    }
    let bundle = mrml_runtime::read_file_bounded(&arguments[1], MAX_KERNEL_BUNDLE)?;
    let public = mrml_runtime::read_file_bounded(&arguments[2], LAMPORT_PUBLIC_KEY_BYTES)?;
    if public.len() != LAMPORT_PUBLIC_KEY_BYTES {
        return Err(anyhow!("invalid release public-key length"));
    }
    let root = TrustRoot::new(ArtifactKind::Kernel, Sha3_512::digest(&public), 1);
    let signed = SignedArtifact::decode(&bundle).map_err(|_| anyhow!("invalid signed bundle"))?;
    let executable = signed
        .verify_executable(&root, ArtifactKind::Kernel)
        .map_err(|_| anyhow!("kernel signature or PE policy rejected"))?;
    let handoff = boot_handoff(executable.artifact().version());
    let layout = KvmLaunchLayout::new(
        0x10_0000,
        32,
        0x20_0000,
        0xffff_8001_4000_0000,
        0x30_0000,
        0xffff_8001_5000_0000,
        0x40_0000,
        0xffff_8001_6000_0000,
        8,
        false,
    )
    .map_err(|_| anyhow!("invalid fixed kernel launch layout"))?;
    let system = KvmSystem::open()
        .map_err(|error| anyhow!("KVM is unavailable or incompatible: {:?}", error))?;
    let mut guest = system
        .prepare_kernel_guest::<5>(0, &executable, &handoff, layout)
        .map_err(|error| anyhow!("verified kernel launch preparation failed: {:?}", error))?;
    let exit = VmBackend::run(&mut guest, 0)
        .map_err(|error| anyhow!("KVM execution failed: {:?}", error))?;
    if exit != VmExit::Halted {
        return Err(anyhow!("kernel returned an unexpected VM exit"));
    }
    let mut marker = [0u8; 4];
    VmBackend::read_guest(&guest, FRAMEBUFFER, &mut marker)
        .map_err(|_| anyhow!("kernel framebuffer is unreadable"))?;
    if marker != [0x57, 0xc8, 0xff, 0] {
        return Err(anyhow!(
            "kernel did not paint its authenticated boot marker"
        ));
    }
    println!("verified kernel reached its framebuffer marker under KVM");
    Ok(())
}

fn boot_handoff(version: u64) -> [u8; 240] {
    let mut encoded = [0u8; 240];
    encoded[..16].copy_from_slice(b"MRML-HANDOFF-v1\0");
    encoded[16..20].copy_from_slice(&240u32.to_le_bytes());
    encoded[20..22].copy_from_slice(&3u16.to_le_bytes());
    encoded[22..24].copy_from_slice(&7u16.to_le_bytes());
    encoded[24..32].copy_from_slice(&version.to_le_bytes());
    encoded[32..64].fill(0xa5);
    encoded[64..128].copy_from_slice(&Sha3_512::digest(b"mrml-kvm-run measured boot"));
    encoded[128..136].copy_from_slice(&0x9000u64.to_le_bytes());
    encoded[136..144].copy_from_slice(&FRAMEBUFFER.to_le_bytes());
    encoded[144..152].copy_from_slice(&0x1000u64.to_le_bytes());
    encoded[152..156].copy_from_slice(&16u32.to_le_bytes());
    encoded[156..160].copy_from_slice(&16u32.to_le_bytes());
    encoded[160..164].copy_from_slice(&16u32.to_le_bytes());
    encoded[164] = 1;
    encoded[168..176].copy_from_slice(&0x1000u64.to_le_bytes());
    encoded[176..184].copy_from_slice(&2u64.to_le_bytes());
    encoded[192..200].copy_from_slice(&0x3000u64.to_le_bytes());
    encoded[200..208].copy_from_slice(&1u64.to_le_bytes());
    encoded[208] = 1;
    encoded[216..224].copy_from_slice(&FRAMEBUFFER.to_le_bytes());
    encoded[224..232].copy_from_slice(&1u64.to_le_bytes());
    encoded[232] = 3;
    encoded
}

mrml_runtime::mrml_entrypoint!(application_main);
