use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c_src/cuda_kernels.cu");

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
                println!("cargo:warning=[qtensor] Compiling CUDA acceleration kernels with NVCC: {}", nvcc_path.display());

                let obj_out = out_dir.join("cuda_kernels.obj");
                let lib_out = out_dir.join("qtensor_cuda_kernels.lib");

                let mut cmd = Command::new(&nvcc_path);
                
                if is_windows {
                    let cl_tool = cc::Build::new().get_compiler();
                    let cl_path = cl_tool.path();
                    if let Some(cl_dir) = cl_path.parent() {
                        cmd.arg("-ccbin").arg(cl_dir);
                    }
                }

                cmd.args(&[
                    "-c", "c_src/cuda_kernels.cu",
                    "-o", obj_out.to_str().unwrap(),
                    "-O3",
                    "--use_fast_math",
                    "-Xcompiler", "/MD",
                ]);

                let status = cmd.status().expect("Failed to execute nvcc");
                if !status.success() {
                    panic!("nvcc compilation failed");
                }

                // Create static library archive
                let lib_exe = if is_windows {
                    let cl_tool = cc::Build::new().get_compiler();
                    cl_tool.path().parent().map(|d| d.join("lib.exe")).unwrap_or_else(|| PathBuf::from("lib.exe"))
                } else {
                    PathBuf::from("ar")
                };

                let mut lib_cmd = Command::new(&lib_exe);
                if is_windows {
                    let out_arg = format!("/OUT:{}", lib_out.display());
                    lib_cmd.args(&[
                        out_arg.as_str(),
                        obj_out.to_str().unwrap(),
                    ]);
                } else {
                    lib_cmd.args(&[
                        "crus",
                        lib_out.to_str().unwrap(),
                        obj_out.to_str().unwrap(),
                    ]);
                }
                
                let lib_status = lib_cmd.status().expect("Failed to execute archiver");
                if !lib_status.success() {
                    panic!("Library archive creation failed");
                }

                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=qtensor_cuda_kernels");

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
