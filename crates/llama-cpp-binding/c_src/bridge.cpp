#include "fit.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>

extern "C" {

void qt_llama_load_all_backends() {
    static bool s_loaded = false;
    if (s_loaded) return;
    s_loaded = true;

    // 1. Primary default discovery (binary dir and standard paths)
    ggml_backend_load_all();

    // 2. Custom environment backend search paths
    if (const char * env_path = std::getenv("GGML_BACKEND_PATH")) {
        ggml_backend_load_all_from_path(env_path);
    }
    if (const char * env_path = std::getenv("QT_LLAMA_BACKEND_PATH")) {
        ggml_backend_load_all_from_path(env_path);
    }

    // 3. Scan well-known Linux / Flatpak / extension directories
    const char * search_dirs[] = {
        "/app/lib",
        "/app/lib/ggml",
        "/app/extensions/cuda/lib",
        "/app/extensions/cuda/lib64",
        "/app/extensions/rocm/lib",
        "/app/extensions/rocm/lib64",
        "/app/extensions/oneapi/compiler/latest/lib",
        "/app/extensions/oneapi/lib",
        "/usr/local/lib",
        "/usr/lib",
        nullptr
    };

    for (int i = 0; search_dirs[i] != nullptr; ++i) {
        ggml_backend_load_all_from_path(search_dirs[i]);
    }
}

int qt_llama_fit_params_backend(
    const char * model_path,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    uint32_t min_ctx,
    const char * backend_choice
) {
    if (!model_path || !mparams || !cparams) {
        return (int)COMMON_PARAMS_FIT_STATUS_ERROR;
    }

    // Initialize & discover all dynamically available ggml backends
    qt_llama_load_all_backends();

    static thread_local std::vector<ggml_backend_dev_t> s_selected_devs;
    s_selected_devs.clear();

    std::string choice = backend_choice ? backend_choice : "auto";
    std::transform(choice.begin(), choice.end(), choice.begin(), ::tolower);

    if (choice == "cpu") {
        mparams->n_gpu_layers = 0;
        mparams->devices = nullptr;
        mparams->tensor_buft_overrides = nullptr;
        return (int)COMMON_PARAMS_FIT_STATUS_SUCCESS;
    }

    if (choice == "cuda" || choice == "vulkan" || choice == "sycl" || choice == "oneapi" || choice == "hip" || choice == "rocm") {
        std::string prefix;
        if (choice == "cuda") prefix = "CUDA";
        else if (choice == "vulkan") prefix = "Vulkan";
        else if (choice == "sycl" || choice == "oneapi") prefix = "SYCL";
        else if (choice == "hip" || choice == "rocm") prefix = "ROCm";

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            const char * name = ggml_backend_dev_name(dev);
            if (name && std::strncmp(name, prefix.c_str(), prefix.size()) == 0) {
                s_selected_devs.push_back(dev);
            }
        }
        if (!s_selected_devs.empty()) {
            s_selected_devs.push_back(nullptr); // NULL-terminated
            mparams->devices = s_selected_devs.data();
        }
    } else {
        // Auto mode: Prioritize high-performance native backends over Vulkan/CPU
        // Priority: CUDA -> ROCm/HIP -> SYCL -> Vulkan -> CPU
        const char * priority_prefixes[] = {"CUDA", "ROCm", "HIP", "SYCL", "Vulkan", nullptr};
        for (int p = 0; priority_prefixes[p] != nullptr; ++p) {
            const char * prefix = priority_prefixes[p];
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (!dev) continue;
                const char * name = ggml_backend_dev_name(dev);
                if (name && std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
                    s_selected_devs.push_back(dev);
                }
            }
            if (!s_selected_devs.empty()) {
                break; // Found preferred hardware acceleration devices
            }
        }

        if (!s_selected_devs.empty()) {
            s_selected_devs.push_back(nullptr);
            mparams->devices = s_selected_devs.data();
        } else {
            mparams->devices = nullptr; // fallback to all / CPU
        }
    }

    size_t max_dev = llama_max_devices();
    size_t max_overrides = llama_max_tensor_buft_overrides();

    static thread_local std::vector<float> s_tensor_split;
    static thread_local std::vector<llama_model_tensor_buft_override> s_overrides;
    static thread_local std::vector<size_t> s_margins;

    s_tensor_split.assign(max_dev, 0.0f);
    s_overrides.assign(max_overrides, llama_model_tensor_buft_override{nullptr, nullptr});
    s_margins.assign(max_dev, (size_t)384 * 1024 * 1024);

    mparams->tensor_buft_overrides = s_overrides.data();
    mparams->tensor_split = s_tensor_split.data();

    auto status = common_fit_params(
        model_path,
        mparams,
        cparams,
        s_tensor_split.data(),
        s_overrides.data(),
        s_margins.data(),
        min_ctx > 0 ? min_ctx : 4096,
        GGML_LOG_LEVEL_INFO
    );

    return (int)status;
}

int qt_llama_fit_params(
    const char * model_path,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    uint32_t min_ctx
) {
    return qt_llama_fit_params_backend(model_path, mparams, cparams, min_ctx, "auto");
}

}
