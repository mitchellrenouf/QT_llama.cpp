use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda_ptx/kernels.rs");

    #[cfg(feature = "cuda")]
    {
        let is_windows = env::var("CARGO_CFG_TARGET_OS")
            .map(|s| s == "windows")
            .unwrap_or(false);
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

        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_TOOLKIT_ROOT_DIR"))
            .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3".into());
        let cuda_path = PathBuf::from(cuda_path);
        if is_windows {
            println!("cargo:rustc-link-search=native={}", cuda_path.join("lib").join("x64").display());
        } else {
            println!("cargo:rustc-link-search=native={}", cuda_path.join("lib64").display());
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
