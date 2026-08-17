#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_engine_context ggml_engine_context_t;

// Initialize engine and load GGUF model directly on GGML
ggml_engine_context_t* ggml_engine_init(
    const char* model_path,
    int32_t n_gpu_layers,
    uint32_t ctx_size,
    const char* backend_name
);

// Free engine resources
void ggml_engine_free(ggml_engine_context_t* ctx);

// Tokenize text into tokens using GGUF vocabulary
int32_t ggml_engine_tokenize(
    ggml_engine_context_t* ctx,
    const char* text,
    int32_t* out_tokens,
    int32_t max_tokens,
    bool add_special,
    bool parse_special
);

// Convert single token to UTF-8 piece
int32_t ggml_engine_token_to_piece(
    ggml_engine_context_t* ctx,
    int32_t token,
    char* out_buf,
    int32_t max_len
);

// Check if token is EOG (end of generation)
bool ggml_engine_is_eog(ggml_engine_context_t* ctx, int32_t token);

// Evaluate prompt/tokens forward pass in GGML
int32_t ggml_engine_eval(
    ggml_engine_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_past,
    float* out_logits
);

// Sample next token from logits (Greedy, Top-P, Temperature)
int32_t ggml_engine_sample(
    ggml_engine_context_t* ctx,
    const float* logits,
    float temperature,
    float top_p
);

// Reset KV cache memory
void ggml_engine_kv_cache_clear(ggml_engine_context_t* ctx);

// Get vocabulary size
int32_t ggml_engine_get_n_vocab(ggml_engine_context_t* ctx);

#ifdef __cplusplus
}
#endif
