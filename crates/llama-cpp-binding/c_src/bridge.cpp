#include "fit.h"
#include "common.h"
#include "llama.h"
#include <vector>

extern "C" {

int qt_llama_fit_params(
    const char * model_path,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    uint32_t min_ctx
) {
    if (!model_path || !mparams || !cparams) {
        return (int)COMMON_PARAMS_FIT_STATUS_ERROR;
    }

    size_t max_dev = llama_max_devices();
    size_t max_overrides = llama_max_tensor_buft_overrides();

    static thread_local std::vector<float> s_tensor_split;
    static thread_local std::vector<llama_model_tensor_buft_override> s_overrides;
    static thread_local std::vector<size_t> s_margins;

    s_tensor_split.assign(max_dev, 0.0f);
    s_overrides.assign(max_overrides, llama_model_tensor_buft_override{nullptr, nullptr});
    // Set 384 MiB device margin: stable headroom for CUDA graphs and cuBLAS while maximizing GPU layer residency
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

}
