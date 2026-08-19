use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda_ptx/kernels.rs");

    #[cfg(feature = "cuda")]
    {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let nightly_rustc = env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .map(|home| {
                home.join(".rustup/toolchains/nightly-x86_64-pc-windows-msvc/bin/rustc.exe")
            })
            .filter(|path| path.is_file())
            .unwrap_or_else(|| PathBuf::from("rustc"));
        let rust_ptx = out_dir.join("rust_cuda_kernels.ptx");
        let rust_status = Command::new(&nightly_rustc)
            .args([
                "cuda_ptx/kernels.rs",
                "--crate-name",
                "mrml_cuda_ptx",
                "--crate-type",
                "cdylib",
                "--edition",
                "2024",
                "--target",
                "nvptx64-nvidia-cuda",
                "-O",
                "-C",
                "target-cpu=sm_120",
                "-C",
                "unsafe-allow-abi-mismatch=target-cpu",
                "--emit",
                "asm",
                "-o",
            ])
            .arg(&rust_ptx)
            .status()
            .expect("failed to run nightly Rust CUDA compiler");
        if !rust_status.success() {
            panic!("Rust CUDA PTX compilation failed");
        }
        println!(
            "cargo:warning=[mrml-tensor] Compiled Rust CUDA PTX with {}",
            nightly_rustc.display()
        );
    }
}
