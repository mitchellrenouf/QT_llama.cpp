use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_root = manifest_dir.join("../../llama.cpp");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");
    println!("cargo:rerun-if-env-changed=LLAMA_HIPBLAS");

    let is_cuda_feat = env::var("CARGO_FEATURE_CUDA").is_ok();
    let is_vulkan_feat = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let is_hip_feat = env::var("CARGO_FEATURE_HIPBLAS").is_ok();
    let is_sycl_feat = env::var("CARGO_FEATURE_SYCL").is_ok();
    let is_auto_feat = env::var("CARGO_FEATURE_AUTO").is_ok();

    // Check CUDA presence
    let has_cuda = is_cuda_feat
        || env::var("LLAMA_CUDA").map(|v| v == "1" || v == "ON").unwrap_or(false)
        || (is_auto_feat && (
            Path::new("/opt/cuda/bin/nvcc").exists()
            || Path::new("/usr/local/cuda/bin/nvcc").exists()
            || std::process::Command::new("which").arg("nvcc").output().map(|o| o.status.success()).unwrap_or(false)
        ));

    // Check Vulkan presence (including Flatpak SDK paths)
    let has_vulkan = is_vulkan_feat
        || env::var("LLAMA_VULKAN").map(|v| v == "1" || v == "ON").unwrap_or(false)
        || (is_auto_feat && (
            Path::new("/usr/include/vulkan/vulkan.h").exists()
            || Path::new("/usr/local/include/vulkan/vulkan.h").exists()
            || Path::new("/app/include/vulkan/vulkan.h").exists()
            || env::var("VULKAN_SDK").is_ok()
            // pkg-config check for Flatpak SDK vulkan-stack
            || std::process::Command::new("pkg-config").args(["--exists", "vulkan"]).status().map(|s| s.success()).unwrap_or(false)
        ));

    let has_hipblas = is_hip_feat
        || env::var("LLAMA_HIPBLAS").map(|v| v == "1" || v == "ON").unwrap_or(false);

    let has_sycl = is_sycl_feat
        || env::var("LLAMA_SYCL").map(|v| v == "1" || v == "ON").unwrap_or(false);

    let mut cfg = cmake::Config::new(&llama_root);
    cfg.define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    if has_cuda {
        println!("cargo:warning=[llama-cpp-binding] Enabling CUDA GPU Acceleration (NVIDIA cuBLAS)...");
        cfg.define("GGML_CUDA", "ON");
        cfg.define("GGML_CUDA_NCCL", "OFF");
        if Path::new("/opt/cuda").exists() {
            cfg.define("CUDA_TOOLKIT_ROOT_DIR", "/opt/cuda");
            cfg.define("CMAKE_CUDA_COMPILER", "/opt/cuda/bin/nvcc");
        }
    }

    if has_vulkan {
        println!("cargo:warning=[llama-cpp-binding] Enabling Vulkan GPU Acceleration...");
        cfg.define("GGML_VULKAN", "ON");
    }

    if has_hipblas {
        println!("cargo:warning=[llama-cpp-binding] Enabling AMD ROCm / HIP Acceleration...");
        cfg.define("GGML_HIPBLAS", "ON");
    }

    if has_sycl {
        println!("cargo:warning=[llama-cpp-binding] Enabling Intel SYCL GPU Acceleration...");
        cfg.define("GGML_SYCL", "ON");
    }

    let dst = cfg.build();

    // Dynamically search all build output directories for static libraries
    for entry in walkdir::WalkDir::new(&dst).into_iter().flatten() {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension() {
                if ext == "a" {
                    if let Some(parent) = entry.path().parent() {
                        println!("cargo:rustc-link-search=native={}", parent.display());
                    }
                }
            }
        }
    }

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        let cuda_lib_candidates = [
            "/opt/cuda/lib64",
            "/opt/cuda/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        ];
        for p in cuda_lib_candidates {
            if Path::new(p).exists() {
                println!("cargo:rustc-link-search=native={}", p);
            }
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    if has_hipblas {
        println!("cargo:rustc-link-lib=static=ggml-hip");
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
    }

    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
