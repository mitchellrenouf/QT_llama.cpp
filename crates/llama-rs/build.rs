use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.join("../../llama.cpp");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=c_src/ggml_engine.h");
    println!("cargo:rerun-if-changed=c_src/ggml_engine.cpp");
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
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("GGML_NATIVE", "ON")
        .define("GGML_AVX", "ON")
        .define("GGML_AVX2", "ON")
        .define("GGML_FMA", "ON")
        .define("GGML_F16C", "ON")
        .define("GGML_BMI2", "ON")
        .define("GGML_CPU_REPACK", "ON");

    if is_windows {
        cfg.cflag("/O2").cflag("/Oi").cflag("/Ot");
        cfg.cxxflag("/O2").cxxflag("/Oi").cxxflag("/Ot");
    }

    let mut prefix_paths = Vec::new();
    if Path::new("/app").exists() {
        prefix_paths.push("/app".to_string());
    }
    prefix_paths.push("/usr".to_string());

    if has_cuda {
        println!("cargo:warning=[llama-rs] Enabling CUDA GPU Acceleration for pure GGML...");
        cfg.define("GGML_CUDA", "ON");
        cfg.define("GGML_CUDA_NCCL", "OFF");
        cfg.define("GGML_CUDA_FA", "ON");
        cfg.define("GGML_CUDA_GRAPHS", "ON");
        cfg.define("GGML_CUDA_PEER_MAX_BATCH_SIZE", "128");

        if let Ok(cuda_dir) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_TOOLKIT_ROOT_DIR")) {
            let p = PathBuf::from(&cuda_dir);
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", &p);
            let nvcc_candidates = [
                p.join("bin").join("nvcc.exe"),
                p.join("bin").join("nvcc"),
            ];
            for cand in &nvcc_candidates {
                if cand.exists() {
                    cfg.define("CMAKE_CUDA_COMPILER", cand);
                    break;
                }
            }
            prefix_paths.push(p.display().to_string());
        }

        let well_known_cuda_dirs = [
            "/opt/cuda",
            "/usr/local/cuda",
            "/app/cuda",
            "/app/extensions/cuda",
            "/usr/lib/sdk/cuda",
        ];
        for dir in &well_known_cuda_dirs {
            let p = Path::new(dir);
            if p.exists() {
                prefix_paths.push(p.display().to_string());
                let nvcc = p.join("bin/nvcc");
                if nvcc.exists() {
                    cfg.define("CMAKE_CUDA_COMPILER", &nvcc);
                }
            }
        }
    }

    if has_vulkan {
        println!("cargo:warning=[llama-rs] Enabling Vulkan GPU Acceleration for pure GGML...");
        cfg.define("GGML_VULKAN", "ON");
        if let Ok(sdk) = env::var("VULKAN_SDK") {
            let p = PathBuf::from(sdk);
            cfg.define("Vulkan_INCLUDE_DIR", p.join("Include"));
            cfg.define("Vulkan_LIBRARY", p.join("Lib").join("vulkan-1.lib"));
            prefix_paths.push(p.display().to_string());
        }
    }

    if has_hipblas {
        println!("cargo:warning=[llama-rs] Enabling AMD ROCm / HIP Acceleration for pure GGML...");
        cfg.define("GGML_HIPBLAS", "ON");
    }

    if has_sycl {
        println!("cargo:warning=[llama-rs] Enabling Intel SYCL GPU Acceleration for pure GGML...");
        cfg.define("GGML_SYCL", "ON");
    }

    let joined_prefix = prefix_paths.join(";");
    cfg.define("CMAKE_PREFIX_PATH", &joined_prefix);

    let dst = cfg.build();

    // Search build output directories for static/import libraries
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

    // Compile pure GGML engine C++ bridge
    let mut bridge_build = cc::Build::new();
    bridge_build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .flag_if_supported("/O2")
        .flag_if_supported("/Oi")
        .flag_if_supported("/Ot")
        .flag_if_supported("/fp:fast")
        .file("c_src/ggml_engine.cpp")
        .include(llama_root.join("ggml/include"))
        .include(llama_root.join("ggml/src"));

    if has_cuda {
        bridge_build.define("GGML_USE_CUDA", "1");
    }
    if has_vulkan {
        bridge_build.define("GGML_USE_VULKAN", "1");
    }
    if has_hipblas {
        bridge_build.define("GGML_USE_HIPBLAS", "1");
    }
    if has_sycl {
        bridge_build.define("GGML_USE_SYCL", "1");
    }

    bridge_build.compile("qt_llama_ggml_engine");

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
