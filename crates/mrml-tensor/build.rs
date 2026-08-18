use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
fn find_msvc_bin() -> Option<PathBuf> {
    if let Some(path) = env::var_os("VCToolsInstallDir") {
        let bin = PathBuf::from(path).join("bin").join("Hostx64").join("x64");
        if bin.join("cl.exe").is_file() {
            return Some(bin);
        }
    }

    let program_files = env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tools = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim())
        .join("VC")
        .join("Tools")
        .join("MSVC");
    let mut versions = std::fs::read_dir(tools)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    versions.sort_unstable();
    versions.into_iter().rev().find_map(|version| {
        let bin = version.join("bin").join("Hostx64").join("x64");
        bin.join("cl.exe").is_file().then_some(bin)
    })
}

fn main() {
    println!("cargo:rerun-if-changed=c_src/cuda_kernels.cu");
    println!("cargo:rerun-if-env-changed=MRML_CUDA_ARCHS");

    #[cfg(feature = "cuda")]
    {
        let is_windows = env::var("CARGO_CFG_TARGET_OS").map(|s| s == "windows").unwrap_or(false);
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let mut cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_TOOLKIT_ROOT_DIR"))
            .ok();

        if cuda_path.is_none() && is_windows {
            let default_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3";
            if Path::new(default_cuda).exists() {
                cuda_path = Some(default_cuda.to_string());
            }
        }

        if let Some(ref p_str) = cuda_path {
            let p = PathBuf::from(p_str);
            let nvcc_path = if is_windows {
                p.join("bin").join("nvcc.exe")
            } else {
                p.join("bin").join("nvcc")
            };

            if nvcc_path.exists() {
                println!("cargo:warning=[mrml-tensor] Compiling CUDA acceleration kernels with NVCC: {}", nvcc_path.display());

                let obj_out = if is_windows {
                    out_dir.join("cuda_kernels.obj")
                } else {
                    out_dir.join("cuda_kernels.o")
                };
                let lib_out = if is_windows {
                    out_dir.join("mrml_tensor_cuda_kernels.lib")
                } else {
                    out_dir.join("libmrml_tensor_cuda_kernels.a")
                };

                let mut cmd = Command::new(&nvcc_path);

                #[cfg(windows)]
                let msvc_bin = find_msvc_bin().expect("Visual Studio C++ x64 tools are required for CUDA builds");
                #[cfg(windows)]
                cmd.arg("-ccbin").arg(&msvc_bin);

                cmd.args(&[
                    "-c", "c_src/cuda_kernels.cu",
                    "-o", obj_out.to_str().unwrap(),
                    "-O3",
                    "--use_fast_math",
                ]);

                // Ship native cubins for the broad NVIDIA generations we support.
                // CUDA selects the best image at load time; the final PTX image
                // preserves forward compatibility with later architectures.
                let archs = env::var("MRML_CUDA_ARCHS")
                    // Turing; Ampere datacenter/consumer; Ada; Hopper; and
                    // Blackwell datacenter/consumer. Keep the lowest PTX as a
                    // forward-compatible fallback for intermediate GPUs.
                    .unwrap_or_else(|_| "75,80,86,89,90,100,103,120".to_string());
                let mut parsed_archs = Vec::new();
                for arch in archs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    if !arch.chars().all(|c| c.is_ascii_digit()) {
                        panic!("invalid CUDA architecture in MRML_CUDA_ARCHS: {arch}");
                    }
                    cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
                    parsed_archs.push(arch.to_string());
                }
                if let Some(lowest) = parsed_archs.first() {
                    cmd.arg(format!("-gencode=arch=compute_{lowest},code=compute_{lowest}"));
                }

                if is_windows {
                    cmd.args(&["-Xcompiler", "/MD"]);
                } else {
                    cmd.args(&["-Xcompiler", "-fPIC"]);
                }

                let status = cmd.status().expect("Failed to execute nvcc");
                if !status.success() {
                    panic!("nvcc compilation failed");
                }

                // Let NVCC drive the platform archiver as well. This uses the same
                // discovered host toolchain as compilation without a Rust wrapper.
                let mut lib_cmd = Command::new(&nvcc_path);
                #[cfg(windows)]
                lib_cmd.arg("-ccbin").arg(&msvc_bin);
                lib_cmd.arg("--lib").arg(&obj_out).arg("-o").arg(&lib_out);
                
                let lib_status = lib_cmd.status().expect("Failed to execute archiver");
                if !lib_status.success() {
                    panic!("Library archive creation failed");
                }

                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=mrml_tensor_cuda_kernels");

                if is_windows {
                    println!("cargo:rustc-link-search=native={}", p.join("lib").join("x64").display());
                } else {
                    println!("cargo:rustc-link-search=native={}", p.join("lib64").display());
                }
                println!("cargo:rustc-link-lib=dylib=cudart");
            }
        }
    }
}
