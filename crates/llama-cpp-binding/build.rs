use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.join("../../llama.cpp");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=c_src/bridge.cpp");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");
    println!("cargo:rerun-if-env-changed=LLAMA_HIPBLAS");
    println!("cargo:rerun-if-env-changed=LLAMA_SYCL");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";
    let is_macos = target_os == "macos";

    let is_cuda_feat = env::var("CARGO_FEATURE_CUDA").is_ok();
    let is_vulkan_feat = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let is_hip_feat = env::var("CARGO_FEATURE_HIPBLAS").is_ok();
    let is_sycl_feat = env::var("CARGO_FEATURE_SYCL").is_ok();
    let is_auto_feat = env::var("CARGO_FEATURE_AUTO").is_ok();

    // Check CUDA presence (Windows %CUDA_PATH%, /opt/cuda, /usr/local/cuda, /app/cuda, nvcc in PATH)
    let has_cuda = is_cuda_feat
        || env::var("LLAMA_CUDA").map(|v| v == "1" || v == "ON").unwrap_or(false)
        || (is_auto_feat && (
            env::var("CUDA_PATH").is_ok()
            || env::var("CUDA_TOOLKIT_ROOT_DIR").is_ok()
            || Path::new("/opt/cuda/bin/nvcc").exists()
            || Path::new("/usr/local/cuda/bin/nvcc").exists()
            || Path::new("/app/cuda/bin/nvcc").exists()
            || std::process::Command::new("which").arg("nvcc").output().map(|o| o.status.success()).unwrap_or(false)
            || (is_windows && std::process::Command::new("where").arg("nvcc").output().map(|o| o.status.success()).unwrap_or(false))
        ));

    // Check Vulkan presence
    let has_vulkan = is_vulkan_feat
        || env::var("LLAMA_VULKAN").map(|v| v == "1" || v == "ON").unwrap_or(false)
        || (is_auto_feat && (
            env::var("VULKAN_SDK").is_ok()
            || Path::new("/usr/include/vulkan/vulkan.h").exists()
            || Path::new("/usr/local/include/vulkan/vulkan.h").exists()
            || Path::new("/app/include/vulkan/vulkan.h").exists()
            || std::process::Command::new("pkg-config").args(["--exists", "vulkan"]).status().map(|s| s.success()).unwrap_or(false)
        ));

    let has_hipblas = is_hip_feat
        || env::var("LLAMA_HIPBLAS").map(|v| v == "1" || v == "ON").unwrap_or(false);

    let has_sycl = is_sycl_feat
        || env::var("LLAMA_SYCL").map(|v| v == "1" || v == "ON").unwrap_or(false);

    let mut cfg = cmake::Config::new(&llama_root);
    cfg.define("LLAMA_STANDALONE", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_COMMON", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    let mut prefix_paths = Vec::new();
    if Path::new("/app").exists() {
        prefix_paths.push("/app".to_string());
    }
    prefix_paths.push("/usr".to_string());

    if has_cuda {
        println!("cargo:warning=[llama-cpp-binding] Enabling CUDA GPU Acceleration (NVIDIA cuBLAS)...");
        cfg.define("GGML_CUDA", "ON");
        cfg.define("GGML_CUDA_NCCL", "OFF");

        if let Ok(cuda_dir) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_TOOLKIT_ROOT_DIR")) {
            let p = PathBuf::from(&cuda_dir);
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", &p);
            let nvcc_candidates = [
                p.join("bin").join("nvcc.exe"),
                p.join("bin").join("nvcc"),
            ];
            for nvcc in &nvcc_candidates {
                if nvcc.exists() {
                    cfg.define("CMAKE_CUDA_COMPILER", nvcc);
                    break;
                }
            }
            prefix_paths.push(p.display().to_string());
        } else if Path::new("/usr/lib/sdk/cuda").exists() {
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", "/usr/lib/sdk/cuda");
            cfg.define("CMAKE_CUDA_COMPILER", "/usr/lib/sdk/cuda/bin/nvcc");
            prefix_paths.push("/usr/lib/sdk/cuda".to_string());
        } else if Path::new("/app/cuda").exists() {
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", "/app/cuda");
            cfg.define("CMAKE_CUDA_COMPILER", "/app/cuda/bin/nvcc");
            prefix_paths.push("/app/cuda".to_string());
        } else if Path::new("/opt/cuda").exists() {
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", "/opt/cuda");
            cfg.define("CMAKE_CUDA_COMPILER", "/opt/cuda/bin/nvcc");
            prefix_paths.push("/opt/cuda".to_string());
        } else if Path::new("/usr/local/cuda").exists() {
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", "/usr/local/cuda");
            cfg.define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc");
            prefix_paths.push("/usr/local/cuda".to_string());
        }
    }

    if has_vulkan {
        println!("cargo:warning=[llama-cpp-binding] Enabling Vulkan GPU Acceleration...");
        cfg.define("GGML_VULKAN", "ON");
        if let Ok(sdk) = env::var("VULKAN_SDK") {
            prefix_paths.push(sdk);
        }
        if Path::new("/app/share/cmake/SPIRV-Headers").exists() {
            cfg.define("SPIRV-Headers_DIR", "/app/share/cmake/SPIRV-Headers");
        }
    }

    if has_hipblas {
        println!("cargo:warning=[llama-cpp-binding] Enabling AMD ROCm / HIP Acceleration...");
        cfg.define("GGML_HIP", "ON");
        cfg.define("GGML_HIPBLAS", "ON");
        if let Ok(rocm_dir) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
            let p = PathBuf::from(rocm_dir);
            cfg.define("ROCM_PATH", &p);
            let hipcc = p.join("bin/hipcc");
            if hipcc.exists() {
                cfg.define("CMAKE_CXX_COMPILER", hipcc);
            }
            prefix_paths.push(p.display().to_string());
        } else if Path::new("/usr/lib/sdk/rocm").exists() {
            cfg.define("ROCM_PATH", "/usr/lib/sdk/rocm");
            cfg.define("CMAKE_CXX_COMPILER", "/usr/lib/sdk/rocm/bin/hipcc");
            prefix_paths.push("/usr/lib/sdk/rocm".to_string());
        } else if Path::new("/app/rocm").exists() {
            cfg.define("ROCM_PATH", "/app/rocm");
            cfg.define("CMAKE_CXX_COMPILER", "/app/rocm/bin/hipcc");
            prefix_paths.push("/app/rocm".to_string());
        } else if Path::new("/opt/rocm").exists() {
            cfg.define("ROCM_PATH", "/opt/rocm");
            cfg.define("CMAKE_CXX_COMPILER", "/opt/rocm/bin/hipcc");
            prefix_paths.push("/opt/rocm".to_string());
        }
    }

    if has_sycl {
        println!("cargo:warning=[llama-cpp-binding] Enabling Intel SYCL GPU Acceleration...");
        cfg.define("GGML_SYCL", "ON");
        let oneapi_root = if let Ok(oneapi_dir) = env::var("ONEAPI_ROOT").or_else(|_| env::var("MKLROOT")) {
            PathBuf::from(oneapi_dir)
        } else if Path::new("/usr/lib/sdk/oneapi").exists() {
            PathBuf::from("/usr/lib/sdk/oneapi")
        } else if Path::new("/app/oneapi").exists() {
            PathBuf::from("/app/oneapi")
        } else if Path::new("/opt/intel/oneapi").exists() {
            PathBuf::from("/opt/intel/oneapi")
        } else {
            PathBuf::from("/app/oneapi")
        };

        cfg.define("ONEAPI_ROOT", &oneapi_root);
        let icpx = oneapi_root.join("compiler/latest/bin/icpx");
        if icpx.exists() {
            cfg.define("CMAKE_CXX_COMPILER", icpx);
        }
        let mkl_dir = oneapi_root.join("mkl/latest");
        if mkl_dir.exists() {
            cfg.define("MKLROOT", &mkl_dir);
            cfg.define("MKL_ROOT", &mkl_dir);
            cfg.define("MKL_DIR", mkl_dir.join("lib/cmake/mkl"));
            prefix_paths.push(mkl_dir.display().to_string());
            prefix_paths.push(mkl_dir.join("lib/cmake/mkl").display().to_string());
        }

        let tbb_dir = oneapi_root.join("tbb/latest");
        if tbb_dir.exists() {
            cfg.define("TBB_DIR", tbb_dir.join("lib/cmake/tbb"));
            cfg.define("TBB_ROOT", &tbb_dir);
            prefix_paths.push(tbb_dir.display().to_string());
            prefix_paths.push(tbb_dir.join("lib/cmake/tbb").display().to_string());
            prefix_paths.push(tbb_dir.join("lib/cmake/TBB").display().to_string());
            prefix_paths.push(tbb_dir.join("lib64/cmake/tbb").display().to_string());
            prefix_paths.push(tbb_dir.join("lib64/cmake/TBB").display().to_string());
        }

        let dnnl_dir = oneapi_root.join("dnnl/latest");
        if dnnl_dir.exists() {
            cfg.define("DNNL_DIR", dnnl_dir.join("lib/cmake/dnnl"));
            cfg.define("DNNL_ROOT", &dnnl_dir);
            prefix_paths.push(dnnl_dir.display().to_string());
            prefix_paths.push(dnnl_dir.join("lib/cmake/dnnl").display().to_string());
            prefix_paths.push(dnnl_dir.join("lib64/cmake/dnnl").display().to_string());
        }

        prefix_paths.push(oneapi_root.display().to_string());
        prefix_paths.push(oneapi_root.join("compiler/latest").display().to_string());
    }

    let joined_prefix = prefix_paths.join(";");
    cfg.define("CMAKE_PREFIX_PATH", &joined_prefix);

    let dst = cfg.build();

    // Dynamically search all build output directories for static/import libraries
    for entry in walkdir::WalkDir::new(&dst).into_iter().flatten() {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension() {
                if ext == "a" || ext == "lib" || ext == "so" || ext == "dylib" {
                    if let Some(parent) = entry.path().parent() {
                        println!("cargo:rustc-link-search=native={}", parent.display());
                    }
                }
            }
        }
    }

    // Compile bridge for common_fit_params
    let mut bridge_build = cc::Build::new();
    bridge_build
        .cpp(true)
        .std("c++17")
        .file("c_src/bridge.cpp")
        .include(llama_root.join("include"))
        .include(llama_root.join("ggml/include"))
        .include(llama_root.join("common"));
    bridge_build.compile("qt_llama_bridge");

    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        let mut cuda_lib_candidates = vec![
            "/opt/cuda/lib64".to_string(),
            "/opt/cuda/lib".to_string(),
            "/usr/local/cuda/lib64".to_string(),
            "/usr/local/cuda/lib".to_string(),
            "/app/cuda/lib64".to_string(),
            "/app/cuda/lib".to_string(),
            "/app/extensions/cuda/lib64".to_string(),
            "/app/extensions/cuda/lib".to_string(),
            "/usr/lib/sdk/cuda/lib64".to_string(),
            "/usr/lib/sdk/cuda/lib".to_string(),
            "/usr/lib/x86_64-linux-gnu".to_string(),
            "/usr/lib64".to_string(),
            "/usr/lib".to_string(),
        ];

        if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_TOOLKIT_ROOT_DIR")) {
            let p = PathBuf::from(&cuda_path);
            cuda_lib_candidates.push(p.join("lib").join("x64").display().to_string());
            cuda_lib_candidates.push(p.join("lib").join("Win32").display().to_string());
            cuda_lib_candidates.push(p.join("lib64").display().to_string());
            cuda_lib_candidates.push(p.join("lib").display().to_string());
        }

        for p in cuda_lib_candidates {
            if Path::new(&p).exists() {
                println!("cargo:rustc-link-search=native={}", p);
            }
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        if let Ok(sdk) = env::var("VULKAN_SDK") {
            let p = PathBuf::from(sdk);
            if is_windows {
                println!("cargo:rustc-link-search=native={}", p.join("Lib").display());
            } else {
                println!("cargo:rustc-link-search=native={}", p.join("lib").display());
            }
        }
        if is_windows {
            println!("cargo:rustc-link-lib=dylib=vulkan-1");
        } else {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    }

    if has_hipblas {
        println!("cargo:rustc-link-lib=static=ggml-hip");
        let rocm_lib_candidates = [
            "/opt/rocm/lib",
            "/opt/rocm/lib64",
            "/usr/lib/sdk/rocm/lib",
            "/usr/lib/sdk/rocm/lib64",
            "/app/rocm/lib",
            "/app/rocm/lib64",
            "/app/extensions/rocm/lib",
            "/app/extensions/rocm/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        ];
        for p in rocm_lib_candidates {
            if Path::new(p).exists() {
                println!("cargo:rustc-link-search=native={}", p);
            }
        }
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }

    if has_sycl {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
        let oneapi_lib_candidates = [
            "/opt/intel/oneapi/compiler/latest/lib",
            "/opt/intel/oneapi/mkl/latest/lib",
            "/opt/intel/oneapi/tbb/latest/lib",
            "/usr/lib/sdk/oneapi/compiler/latest/lib",
            "/app/oneapi/compiler/latest/lib",
            "/app/extensions/oneapi/compiler/latest/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        ];
        for p in oneapi_lib_candidates {
            if Path::new(p).exists() {
                println!("cargo:rustc-link-search=native={}", p);
            }
        }
        println!("cargo:rustc-link-lib=dylib=sycl");
        println!("cargo:rustc-link-lib=dylib=OpenCL");
    }

    // Platform-specific standard C++ & OpenMP runtime libraries
    if !is_windows {
        if is_macos {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}
