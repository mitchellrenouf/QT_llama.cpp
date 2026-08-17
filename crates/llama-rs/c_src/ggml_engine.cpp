#include "ggml_engine.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <random>

struct gemma_layer {
    struct ggml_tensor* attn_norm = nullptr;
    struct ggml_tensor* wq = nullptr;
    struct ggml_tensor* wk = nullptr;
    struct ggml_tensor* wv = nullptr;
    struct ggml_tensor* wo = nullptr;
    struct ggml_tensor* wq_norm = nullptr;
    struct ggml_tensor* wk_norm = nullptr;

    struct ggml_tensor* ffn_norm = nullptr;
    struct ggml_tensor* ffn_gate = nullptr;
    struct ggml_tensor* ffn_up = nullptr;
    struct ggml_tensor* ffn_down = nullptr;

    struct ggml_tensor* post_attn_norm = nullptr;
    struct ggml_tensor* post_ffw_norm = nullptr;
};

struct gemma_model {
    uint32_t n_vocab = 256000;
    uint32_t n_embd = 2048;
    uint32_t n_layer = 18;
    uint32_t n_head = 8;
    uint32_t n_head_kv = 1;
    uint32_t head_dim = 256;
    uint32_t n_ff = 16384;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    uint32_t sliding_window = 4096;

    struct ggml_tensor* tok_embeddings = nullptr;
    struct ggml_tensor* output_norm = nullptr;
    struct ggml_tensor* output = nullptr;

    std::vector<gemma_layer> layers;
};

struct gemma_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float> scores;
    std::vector<int32_t> special_eog_ids;
    int32_t bos_id = 2;
    int32_t eos_id = 1;
    int32_t eot_id = 107; // <end_of_turn>
};

struct ggml_engine_context {
    gemma_model model;
    gemma_vocab vocab;
    uint32_t ctx_size = 8192;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    struct ggml_context* model_ctx = nullptr;

    // KV Cache in GPU VRAM
    struct ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    struct ggml_tensor* k_cache = nullptr;
    struct ggml_tensor* v_cache = nullptr;

    // Memory planner
    ggml_gallocr_t galloc = nullptr;

    // Logits buffer for output
    std::vector<float> last_logits;
    std::mt19937 rng;

    ggml_engine_context() : rng(std::random_device{}()) {}
};

static void init_backends() {
    static bool initialized = false;
    if (!initialized) {
        ggml_backend_load_all();
        initialized = true;
    }
}

ggml_engine_context_t* ggml_engine_init(
    const char* model_path,
    int32_t n_gpu_layers,
    uint32_t ctx_size,
    const char* backend_name
) {
    (void)n_gpu_layers;
    init_backends();

    auto* engine = new ggml_engine_context();
    engine->ctx_size = (ctx_size > 0 && ctx_size <= 32768) ? ctx_size : 8192;

    struct gguf_init_params params = {
        /* no_alloc = */ true,
        /* ctx      = */ &engine->model_ctx,
    };

    struct gguf_context* gguf_ctx = gguf_init_from_file(model_path, params);
    if (!gguf_ctx) {
        std::cerr << "[ggml-engine] Failed to load GGUF metadata from " << model_path << std::endl;
        delete engine;
        return nullptr;
    }

    // 1. Parse Vocab
    int64_t tokens_key = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
    if (tokens_key >= 0) {
        size_t n_vocab = gguf_get_arr_n(gguf_ctx, tokens_key);
        engine->vocab.id_to_token.resize(n_vocab);
        engine->vocab.scores.resize(n_vocab, 0.0f);
        engine->model.n_vocab = (uint32_t)n_vocab;

        for (size_t i = 0; i < n_vocab; ++i) {
            const char* str = gguf_get_arr_str(gguf_ctx, tokens_key, i);
            engine->vocab.id_to_token[i] = str ? str : "";
            engine->vocab.token_to_id[engine->vocab.id_to_token[i]] = (int32_t)i;
        }
    }

    int64_t scores_key = gguf_find_key(gguf_ctx, "tokenizer.ggml.scores");
    if (scores_key >= 0) {
        size_t n_scores = gguf_get_arr_n(gguf_ctx, scores_key);
        const float* raw_scores = (const float*)gguf_get_arr_data(gguf_ctx, scores_key);
        if (raw_scores) {
            for (size_t i = 0; i < n_scores && i < engine->vocab.scores.size(); ++i) {
                engine->vocab.scores[i] = raw_scores[i];
            }
        }
    }

    // Special token IDs
    int64_t bos_key = gguf_find_key(gguf_ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) engine->vocab.bos_id = gguf_get_val_u32(gguf_ctx, bos_key);

    int64_t eos_key = gguf_find_key(gguf_ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        engine->vocab.eos_id = gguf_get_val_u32(gguf_ctx, eos_key);
        engine->vocab.special_eog_ids.push_back(engine->vocab.eos_id);
    }

    // Add Gemma specific EOG tokens
    auto add_eog = [&](const std::string& token_str) {
        auto it = engine->vocab.token_to_id.find(token_str);
        if (it != engine->vocab.token_to_id.end()) {
            engine->vocab.special_eog_ids.push_back(it->second);
        }
    };
    add_eog("<end_of_turn>");
    add_eog("<|turn_end|>");
    add_eog("<|im_end|>");
    add_eog("</s>");
    add_eog("<channel|>");

    // 2. Parse Model Architecture & Hyperparameters
    auto read_u32 = [&](const char* key, uint32_t def_val) -> uint32_t {
        int64_t k = gguf_find_key(gguf_ctx, key);
        return k >= 0 ? gguf_get_val_u32(gguf_ctx, k) : def_val;
    };
    auto read_f32 = [&](const char* key, float def_val) -> float {
        int64_t k = gguf_find_key(gguf_ctx, key);
        return k >= 0 ? gguf_get_val_f32(gguf_ctx, k) : def_val;
    };

    engine->model.n_embd = read_u32("gemma.embedding_length", read_u32("general.embedding_length", 2048));
    engine->model.n_layer = read_u32("gemma.block_count", read_u32("general.block_count", 18));
    engine->model.n_head = read_u32("gemma.attention.head_count", 8);
    engine->model.n_head_kv = read_u32("gemma.attention.head_count_kv", engine->model.n_head);
    engine->model.head_dim = read_u32("gemma.attention.key_length", engine->model.n_embd / engine->model.n_head);
    engine->model.n_ff = read_u32("gemma.feed_forward_length", 16384);
    engine->model.rms_norm_eps = read_f32("gemma.attention.layer_norm_rms_epsilon", 1e-6f);
    engine->model.rope_freq_base = read_f32("gemma.rope.freq_base", 10000.0f);
    engine->model.sliding_window = read_u32("gemma.attention.sliding_window", 4096);

    std::cout << "[ggml-engine] Initializing Gemma architecture:"
              << " layers=" << engine->model.n_layer
              << ", embd=" << engine->model.n_embd
              << ", heads=" << engine->model.n_head << "/" << engine->model.n_head_kv
              << ", head_dim=" << engine->model.head_dim
              << ", ctx_size=" << engine->ctx_size << std::endl;

    // 3. Initialize Backend (CUDA / CPU)
    std::string backend_str = backend_name ? backend_name : "auto";
    std::transform(backend_str.begin(), backend_str.end(), backend_str.begin(), ::tolower);

    if (backend_str != "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            const char* name = ggml_backend_dev_name(dev);
            if (name && (std::strstr(name, "CUDA") || std::strstr(name, "cuda"))) {
                engine->backend = ggml_backend_dev_init(dev, nullptr);
                if (engine->backend) {
                    std::cout << "[ggml-engine] Selected hardware compute backend: " << name << std::endl;
                    break;
                }
            }
        }
    }

    if (!engine->backend) {
        engine->backend = ggml_backend_init_best();
        std::cout << "[ggml-engine] Fallback compute backend: " << (engine->backend ? ggml_backend_name(engine->backend) : "CPU") << std::endl;
    }

    // 4. Allocate Weights Buffer & Load Tensors
    engine->weights_buffer = ggml_backend_alloc_ctx_tensors(engine->model_ctx, engine->backend);
    if (!engine->weights_buffer) {
        std::cerr << "[ggml-engine] Failed to allocate weight tensors in backend buffer" << std::endl;
        gguf_free(gguf_ctx);
        delete engine;
        return nullptr;
    }

    // Map tensors
    engine->model.tok_embeddings = ggml_get_tensor(engine->model_ctx, "token_embd.weight");
    engine->model.output_norm = ggml_get_tensor(engine->model_ctx, "output_norm.weight");
    engine->model.output = ggml_get_tensor(engine->model_ctx, "output.weight");
    if (!engine->model.output) {
        engine->model.output = engine->model.tok_embeddings; // Tied weights
    }

    engine->model.layers.resize(engine->model.n_layer);
    for (uint32_t i = 0; i < engine->model.n_layer; ++i) {
        auto& l = engine->model.layers[i];
        std::string pfx = "blk." + std::to_string(i) + ".";
        l.attn_norm = ggml_get_tensor(engine->model_ctx, (pfx + "attn_norm.weight").c_str());
        l.wq = ggml_get_tensor(engine->model_ctx, (pfx + "attn_q.weight").c_str());
        l.wk = ggml_get_tensor(engine->model_ctx, (pfx + "attn_k.weight").c_str());
        l.wv = ggml_get_tensor(engine->model_ctx, (pfx + "attn_v.weight").c_str());
        l.wo = ggml_get_tensor(engine->model_ctx, (pfx + "attn_output.weight").c_str());
        l.wq_norm = ggml_get_tensor(engine->model_ctx, (pfx + "attn_q_norm.weight").c_str());
        l.wk_norm = ggml_get_tensor(engine->model_ctx, (pfx + "attn_k_norm.weight").c_str());

        l.ffn_norm = ggml_get_tensor(engine->model_ctx, (pfx + "ffn_norm.weight").c_str());
        l.ffn_gate = ggml_get_tensor(engine->model_ctx, (pfx + "ffn_gate.weight").c_str());
        l.ffn_up = ggml_get_tensor(engine->model_ctx, (pfx + "ffn_up.weight").c_str());
        l.ffn_down = ggml_get_tensor(engine->model_ctx, (pfx + "ffn_down.weight").c_str());

        l.post_attn_norm = ggml_get_tensor(engine->model_ctx, (pfx + "post_attention_norm.weight").c_str());
        l.post_ffw_norm = ggml_get_tensor(engine->model_ctx, (pfx + "post_ffw_norm.weight").c_str());
    }

    // 5. Allocate KV Cache in GPU VRAM
    struct ggml_init_params kv_params = {
        /* mem_size   = */ (size_t)engine->model.n_layer * 2 * sizeof(struct ggml_tensor) + 1024 * 1024,
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    engine->kv_ctx = ggml_init(kv_params);

    const uint32_t head_dim = engine->model.head_dim;
    const uint32_t n_head_kv = engine->model.n_head_kv;
    const uint32_t n_layer = engine->model.n_layer;
    const uint32_t ctx_len = engine->ctx_size;

    engine->k_cache = ggml_new_tensor_4d(engine->kv_ctx, GGML_TYPE_F16, head_dim, n_head_kv, ctx_len, n_layer);
    engine->v_cache = ggml_new_tensor_4d(engine->kv_ctx, GGML_TYPE_F16, head_dim, n_head_kv, ctx_len, n_layer);

    engine->kv_buffer = ggml_backend_alloc_ctx_tensors(engine->kv_ctx, engine->backend);
    if (!engine->kv_buffer) {
        std::cerr << "[ggml-engine] Failed to allocate KV cache in GPU VRAM" << std::endl;
    }

    // 6. Initialize Graph Allocator
    engine->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(engine->backend));
    engine->last_logits.resize(engine->model.n_vocab, 0.0f);

    gguf_free(gguf_ctx);
    std::cout << "[ggml-engine] Pure GGML engine initialized successfully!" << std::endl;
    return engine;
}

void ggml_engine_free(ggml_engine_context_t* ctx) {
    if (!ctx) return;
    if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
    if (ctx->kv_buffer) ggml_backend_buffer_free(ctx->kv_buffer);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->weights_buffer) ggml_backend_buffer_free(ctx->weights_buffer);
    if (ctx->model_ctx) ggml_free(ctx->model_ctx);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int32_t ggml_engine_tokenize(
    ggml_engine_context_t* ctx,
    const char* text,
    int32_t* out_tokens,
    int32_t max_tokens,
    bool add_special,
    bool parse_special
) {
    (void)parse_special;
    if (!ctx || !text || !out_tokens || max_tokens <= 0) return 0;

    std::vector<int32_t> tokens;
    if (add_special && ctx->vocab.bos_id >= 0) {
        tokens.push_back(ctx->vocab.bos_id);
    }

    std::string s(text);
    size_t i = 0;
    while (i < s.size() && (int32_t)tokens.size() < max_tokens) {
        int best_len = 0;
        int32_t best_id = -1;

        for (size_t len = std::min((size_t)32, s.size() - i); len > 0; --len) {
            std::string sub = s.substr(i, len);
            auto it = ctx->vocab.token_to_id.find(sub);
            if (it != ctx->vocab.token_to_id.end()) {
                best_len = (int)len;
                best_id = it->second;
                break;
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            i += best_len;
        } else {
            uint8_t byte_val = (uint8_t)s[i];
            char hex_buf[8];
            snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", byte_val);
            auto it = ctx->vocab.token_to_id.find(hex_buf);
            if (it != ctx->vocab.token_to_id.end()) {
                tokens.push_back(it->second);
            }
            i += 1;
        }
    }

    int32_t count = std::min((int32_t)tokens.size(), max_tokens);
    for (int32_t j = 0; j < count; ++j) {
        out_tokens[j] = tokens[j];
    }
    return count;
}

int32_t ggml_engine_token_to_piece(
    ggml_engine_context_t* ctx,
    int32_t token,
    char* out_buf,
    int32_t max_len
) {
    if (!ctx || !out_buf || max_len <= 0) return 0;
    if (token < 0 || (size_t)token >= ctx->vocab.id_to_token.size()) {
        out_buf[0] = '\0';
        return 0;
    }

    const std::string& tok_str = ctx->vocab.id_to_token[token];
    if (tok_str.rfind("<0x", 0) == 0 && tok_str.size() == 6 && tok_str.back() == '>') {
        unsigned int byte_val = 0;
        if (sscanf(tok_str.c_str(), "<0x%02X>", &byte_val) == 1) {
            out_buf[0] = (char)byte_val;
            out_buf[1] = '\0';
            return 1;
        }
    }

    std::string res;
    for (size_t i = 0; i < tok_str.size();) {
        if ((uint8_t)tok_str[i] == 0xE2 && i + 2 < tok_str.size() &&
            (uint8_t)tok_str[i + 1] == 0x96 && (uint8_t)tok_str[i + 2] == 0x81) {
            res += ' ';
            i += 3;
        } else {
            res += tok_str[i];
            i += 1;
        }
    }

    strncpy(out_buf, res.c_str(), max_len - 1);
    out_buf[max_len - 1] = '\0';
    return (int32_t)strlen(out_buf);
}

bool ggml_engine_is_eog(ggml_engine_context_t* ctx, int32_t token) {
    if (!ctx) return true;
    for (int32_t eog : ctx->vocab.special_eog_ids) {
        if (token == eog) return true;
    }
    return false;
}

int32_t ggml_engine_eval(
    ggml_engine_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_past,
    float* out_logits
) {
    if (!ctx || !tokens || n_tokens <= 0) return -1;

    struct ggml_init_params params = {
        /* mem_size   = */ (size_t)128 * 1024 * 1024,
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    struct ggml_context* ctx0 = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(ctx0);

    const int32_t n_embd = ctx->model.n_embd;
    const int32_t n_layer = ctx->model.n_layer;
    const int32_t n_head = ctx->model.n_head;
    const int32_t n_head_kv = ctx->model.n_head_kv;
    const int32_t head_dim = ctx->model.head_dim;
    const float eps = ctx->model.rms_norm_eps;

    struct ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");

    struct ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");

    // 1. Embedding lookup + Gemma scaling (sqrt(d_embd))
    struct ggml_tensor* cur = ggml_get_rows(ctx0, ctx->model.tok_embeddings, inp_tokens);
    cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));

    // 2. Transformer layers
    for (int il = 0; il < n_layer; ++il) {
        auto& l = ctx->model.layers[il];
        struct ggml_tensor* residual = cur;

        // Attention Pre-Norm
        cur = ggml_rms_norm(ctx0, cur, eps);
        if (l.attn_norm) cur = ggml_mul(ctx0, cur, l.attn_norm);

        // Q, K, V Projections
        struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, l.wq, cur);
        struct ggml_tensor* Kcur = ggml_mul_mat(ctx0, l.wk, cur);
        struct ggml_tensor* Vcur = ggml_mul_mat(ctx0, l.wv, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_head_kv, n_tokens);

        if (l.wq_norm) Qcur = ggml_mul(ctx0, ggml_rms_norm(ctx0, Qcur, eps), l.wq_norm);
        if (l.wk_norm) Kcur = ggml_mul(ctx0, ggml_rms_norm(ctx0, Kcur, eps), l.wk_norm);

        // RoPE positional embeddings
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, 0, 0, ctx->model.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, 0, 0, ctx->model.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // KV cache update
        if (ctx->k_cache && ctx->v_cache) {
            struct ggml_tensor* k_cache_view = ggml_view_3d(
                ctx0, ctx->k_cache,
                head_dim, n_head_kv, n_tokens,
                ctx->k_cache->nb[1], ctx->k_cache->nb[2],
                il * ctx->k_cache->nb[3] + n_past * ctx->k_cache->nb[2]
            );
            struct ggml_tensor* v_cache_view = ggml_view_3d(
                ctx0, ctx->v_cache,
                head_dim, n_head_kv, n_tokens,
                ctx->v_cache->nb[1], ctx->v_cache->nb[2],
                il * ctx->v_cache->nb[3] + n_past * ctx->v_cache->nb[2]
            );

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        }

        // Attention compute
        struct ggml_tensor* K_all = ggml_view_3d(
            ctx0, ctx->k_cache,
            head_dim, n_head_kv, n_past + n_tokens,
            ctx->k_cache->nb[1], ctx->k_cache->nb[2],
            il * ctx->k_cache->nb[3]
        );
        struct ggml_tensor* V_all = ggml_view_3d(
            ctx0, ctx->v_cache,
            head_dim, n_head_kv, n_past + n_tokens,
            ctx->v_cache->nb[1], ctx->v_cache->nb[2],
            il * ctx->v_cache->nb[3]
        );

        float kq_scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Qcur, K_all, V_all, nullptr, kq_scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx0, attn_out, head_dim * n_head, n_tokens);
        attn_out = ggml_mul_mat(ctx0, l.wo, attn_out);

        if (l.post_attn_norm) attn_out = ggml_mul(ctx0, ggml_rms_norm(ctx0, attn_out, eps), l.post_attn_norm);
        cur = ggml_add(ctx0, residual, attn_out);

        // FFN Pre-Norm
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        if (l.ffn_norm) cur = ggml_mul(ctx0, cur, l.ffn_norm);

        // SwiGLU FFN
        struct ggml_tensor* ffn_gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        struct ggml_tensor* ffn_up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        struct ggml_tensor* ffn_out = ggml_mul(ctx0, ffn_gate, ffn_up);
        ffn_out = ggml_mul_mat(ctx0, l.ffn_down, ffn_out);

        if (l.post_ffw_norm) ffn_out = ggml_mul(ctx0, ggml_rms_norm(ctx0, ffn_out, eps), l.post_ffw_norm);
        cur = ggml_add(ctx0, residual, ffn_out);
    }

    // Final Output Norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    if (ctx->model.output_norm) cur = ggml_mul(ctx0, cur, ctx->model.output_norm);

    // LM Head Logits for last token
    struct ggml_tensor* last_cur = ggml_view_1d(ctx0, cur, n_embd, (size_t)(n_tokens - 1) * n_embd * sizeof(float));
    struct ggml_tensor* logits = ggml_mul_mat(ctx0, ctx->model.output, last_cur);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    // Allocate and compute graph on GPU/CPU backend
    ggml_gallocr_alloc_graph(ctx->galloc, gf);

    // Copy input tokens
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));

    // Fill position indices
    std::vector<int32_t> pos_vec(n_tokens);
    for (int32_t p = 0; p < n_tokens; ++p) {
        pos_vec[p] = n_past + p;
    }
    ggml_backend_tensor_set(inp_pos, pos_vec.data(), 0, n_tokens * sizeof(int32_t));

    // Execute graph
    ggml_backend_graph_compute(ctx->backend, gf);

    // Extract logits
    if (out_logits) {
        ggml_backend_tensor_get(logits, out_logits, 0, ctx->model.n_vocab * sizeof(float));
    }

    ggml_free(ctx0);
    return 0;
}

int32_t ggml_engine_sample(
    ggml_engine_context_t* ctx,
    const float* logits,
    float temperature,
    float top_p
) {
    if (!ctx || !logits) return 0;
    const int32_t n_vocab = (int32_t)ctx->model.n_vocab;

    if (temperature <= 0.0f) {
        int32_t best_id = 0;
        float max_l = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    std::vector<std::pair<float, int32_t>> logits_id(n_vocab);
    float max_l = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }

    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        float p = expf((logits[i] - max_l) / temperature);
        logits_id[i] = {p, i};
        sum += p;
    }

    for (int32_t i = 0; i < n_vocab; ++i) {
        logits_id[i].first /= (float)sum;
    }

    std::sort(logits_id.begin(), logits_id.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    float cum_p = 0.0f;
    size_t cutoff = logits_id.size();
    for (size_t i = 0; i < logits_id.size(); ++i) {
        cum_p += logits_id[i].first;
        if (cum_p >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, cum_p);
    float r = dist(ctx->rng);
    float acc = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        acc += logits_id[i].first;
        if (r <= acc) {
            return logits_id[i].second;
        }
    }

    return logits_id[0].second;
}

void ggml_engine_kv_cache_clear(ggml_engine_context_t* ctx) {
    if (ctx && ctx->kv_buffer) {
        ggml_backend_buffer_clear(ctx->kv_buffer, 0);
    }
}

int32_t ggml_engine_get_n_vocab(ggml_engine_context_t* ctx) {
    return ctx ? (int32_t)ctx->model.n_vocab : 0;
}
