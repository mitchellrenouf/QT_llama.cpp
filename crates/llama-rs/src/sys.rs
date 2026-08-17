#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_float};

pub type llama_token = i32;
pub const LLAMA_TOKEN_NULL: llama_token = -1;

#[repr(C)]
pub struct ggml_engine_context {
    _unused: [u8; 0],
}
pub type ggml_engine_context_t = ggml_engine_context;

extern "C" {
    pub fn ggml_engine_init(
        model_path: *const c_char,
        n_gpu_layers: i32,
        ctx_size: u32,
        backend_name: *const c_char,
    ) -> *mut ggml_engine_context_t;

    pub fn ggml_engine_free(ctx: *mut ggml_engine_context_t);

    pub fn ggml_engine_tokenize(
        ctx: *mut ggml_engine_context_t,
        text: *const c_char,
        out_tokens: *mut i32,
        max_tokens: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn ggml_engine_token_to_piece(
        ctx: *mut ggml_engine_context_t,
        token: i32,
        out_buf: *mut c_char,
        max_len: i32,
    ) -> i32;

    pub fn ggml_engine_is_eog(ctx: *mut ggml_engine_context_t, token: i32) -> bool;

    pub fn ggml_engine_eval(
        ctx: *mut ggml_engine_context_t,
        tokens: *const i32,
        n_tokens: i32,
        n_past: i32,
        out_logits: *mut c_float,
    ) -> i32;

    pub fn ggml_engine_sample(
        ctx: *mut ggml_engine_context_t,
        logits: *const c_float,
        temperature: c_float,
        top_p: c_float,
    ) -> i32;

    pub fn ggml_engine_kv_cache_clear(ctx: *mut ggml_engine_context_t);

    pub fn ggml_engine_get_n_vocab(ctx: *mut ggml_engine_context_t) -> i32;
}
