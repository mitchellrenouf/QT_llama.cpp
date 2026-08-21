#![no_std]
#![no_main]

use mrml_runtime::mrml_println as println;
#[cfg(feature = "cuda")]
use mrml_runtime::{Command, Text, environment_variable, join_path};

fn application_main() -> Result<(), &'static str> {
    println!("cargo:rerun-if-changed=cuda_ptx/kernels.rs");

    #[cfg(feature = "cuda")]
    {
        let out_dir = environment_variable("OUT_DIR").ok_or("Cargo did not provide OUT_DIR")?;
        let nightly_rustc = environment_variable("RUSTC").unwrap_or_else(|| Text::from("rustc"));
        let rust_ptx = join_path(&out_dir, "rust_cuda_kernels.ptx");
        let mut command = Command::new(&nightly_rustc);
        command.args([
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
        ]);
        command.arg(&rust_ptx);
        let output = command
            .output()
            .map_err(|_| "failed to run nightly Rust CUDA compiler")?;
        if !output.status.success() {
            return Err("Rust CUDA PTX compilation failed");
        }
        println!(
            "cargo:warning=[mrml-tensor] Compiled Rust CUDA PTX with {}",
            nightly_rustc
        );
    }
    Ok(())
}

mrml_runtime::mrml_entrypoint!(application_main);
