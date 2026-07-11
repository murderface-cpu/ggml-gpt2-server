// OpenAI-API-compatible HTTP server for the ggml GPT-2 example.
//
// Exposes:
//   GET  /health
//   GET  /v1/models
//   POST /v1/completions       (supports "stream": true)
//   POST /v1/chat/completions  (supports "stream": true)
//
// Concurrency model: model weights are loaded once and shared read-only.
// Each in-flight request is assigned an independent KV-cache "slot" from a
// fixed-size pool, so multiple requests keep separate generation state and
// can be streamed to their clients concurrently. Actual tensor compute is
// still serialized behind a single mutex (ggml's CPU backend / this example's
// graph-building code is not safe to call from multiple threads at once) —
// but the lock is held per generation *step*, not per request, so requests
// interleave fairly instead of queuing behind one another end-to-end.
//
// The model/graph/eval code below is adapted from main-backend.cpp — each
// example in this directory is a self-contained file by convention.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "common.h"
#include "common-ggml.h"

#define CPPHTTPLIB_NO_OPENSSL 1
#include "third_party/httplib.h"
#include "third_party/json.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

#define GPT2_MAX_NODES 4096

using json = nlohmann::json;

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t ftype   = 1;
    float   eps     = 1e-5f;
};

struct gpt2_layer {
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

// read-only weights, loaded once and shared by every slot
struct gpt2_weights {
    gpt2_hparams hparams;

    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;
    struct ggml_tensor * wpe;
    struct ggml_tensor * lm_head;

    std::vector<gpt2_layer> layers;

    struct ggml_context * ctx_w;
    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

// per-request generation state: KV cache + scratch graph-build buffer
struct gpt2_slot {
    struct ggml_context * ctx_kv = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;
    struct ggml_tensor * memory_k = nullptr;
    struct ggml_tensor * memory_v = nullptr;

    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> graph_buf;

    int n_ctx = 0;
};

static bool gpt2_load_weights(const std::string & fname, gpt2_weights & weights, gpt_vocab & vocab, ggml_backend_t backend) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    {
        auto & hparams = weights.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != weights.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, weights.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (weights.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), weights.hparams.ftype);
        return false;
    }

    ggml_log_set(ggml_log_callback_default, nullptr);

    auto & ctx = weights.ctx_w;

    {
        size_t n_tensors = 2 + 6 + 12*weights.hparams.n_layer;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    {
        const auto & hparams = weights.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx_f = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        weights.layers.resize(n_layer);

        weights.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        weights.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        weights.wte     = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        weights.wpe     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx_f);
        weights.lm_head = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        weights.tensors["model/ln_f/g"] = weights.ln_f_g;
        weights.tensors["model/ln_f/b"] = weights.ln_f_b;

        weights.tensors["model/wte"]     = weights.wte;
        weights.tensors["model/wpe"]     = weights.wpe;
        weights.tensors["model/lm_head"] = weights.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = weights.layers[i];

            layer.ln_1_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_2_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w  = ggml_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            weights.tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = layer.ln_1_g;
            weights.tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = layer.ln_1_b;

            weights.tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = layer.ln_2_g;
            weights.tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = layer.ln_2_b;

            weights.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            weights.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            weights.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            weights.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            weights.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            weights.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            weights.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w;
            weights.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
        }
    }

    weights.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, backend);

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__, ggml_backend_buffer_get_size(weights.buffer_w)/(1024.0*1024.0));

    {
        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (weights.tensors.find(name) == weights.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = weights.tensors[name];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(weights.buffer_w)) {
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            if (name == "model/wte" && has_lm_head == false) {
                weights.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}

// allocates a fresh KV cache + graph-build scratch buffer for one request slot
static bool gpt2_slot_init(gpt2_slot & slot, const gpt2_hparams & hparams, int n_ctx, ggml_backend_t backend) {
    slot.n_ctx = std::min(n_ctx, hparams.n_ctx); // wpe is only hparams.n_ctx rows long

    {
        size_t n_tensors = 2;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        slot.ctx_kv = ggml_init(params);
        if (!slot.ctx_kv) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_mem      = n_layer*slot.n_ctx;
    const int n_elements = n_embd*n_mem;

    slot.memory_k = ggml_new_tensor_1d(slot.ctx_kv, GGML_TYPE_F32, n_elements);
    slot.memory_v = ggml_new_tensor_1d(slot.ctx_kv, GGML_TYPE_F32, n_elements);

    slot.buffer_kv = ggml_backend_alloc_ctx_tensors(slot.ctx_kv, backend);

    slot.graph_buf.resize(ggml_tensor_overhead()*GPT2_MAX_NODES + ggml_graph_overhead_custom(GPT2_MAX_NODES, false));

    slot.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    return true;
}

static struct ggml_cgraph * gpt2_graph(
        const gpt2_weights & weights,
        const gpt2_slot & slot,
        const int n_past,
        const int n_tokens) {
    const int N = n_tokens;

    const auto & hparams = weights.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = slot.n_ctx;
    const int n_head  = hparams.n_head;

    struct ggml_init_params params = {
        /*.mem_size   =*/ slot.graph_buf.size(),
        /*.mem_buffer =*/ (void *) slot.graph_buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph_custom(ctx, GPT2_MAX_NODES, false);

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    struct ggml_tensor * inpL =
        ggml_add(ctx,
                ggml_get_rows(ctx, weights.wte, embd),
                ggml_get_rows(ctx, weights.wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        {
            cur = ggml_norm(ctx, inpL, hparams.eps);

            cur = ggml_add(ctx,
                    ggml_mul(ctx,
                        cur,
                        weights.layers[il].ln_1_g),
                    weights.layers[il].ln_1_b);
        }

        {
            cur = ggml_mul_mat(ctx,
                    weights.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    weights.layers[il].c_attn_attn_b);
        }

        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx, slot.memory_k, N*n_embd, (ggml_element_size(slot.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx, slot.memory_v, N*n_embd, (ggml_element_size(slot.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx,
                        ggml_cont_3d(ctx, Qcur, n_embd/n_head, n_head, N),
                        0, 2, 1, 3);

            struct ggml_tensor * K =
                ggml_permute(ctx,
                        ggml_reshape_3d(ctx,
                            ggml_view_1d(ctx, slot.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(slot.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);

            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx,
                        KQ,
                        1.0f/sqrtf(float(n_embd)/n_head));

            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx, KQ_scaled, n_past);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx, KQ_masked);

            struct ggml_tensor * V_trans =
                ggml_cont_3d(ctx,
                        ggml_permute(ctx,
                            ggml_reshape_3d(ctx,
                                ggml_view_1d(ctx, slot.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(slot.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                        n_past + N, n_embd/n_head, n_head);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx, V_trans, KQ_soft_max);

            struct ggml_tensor * KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);

            cur = ggml_cont_2d(ctx, KQV_merged, n_embd, N);
        }

        {
            cur = ggml_mul_mat(ctx,
                    weights.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    weights.layers[il].c_attn_proj_b);
        }

        cur = ggml_add(ctx, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        {
            {
                cur = ggml_norm(ctx, inpFF, hparams.eps);

                cur = ggml_add(ctx,
                        ggml_mul(ctx,
                            cur,
                            weights.layers[il].ln_2_g),
                        weights.layers[il].ln_2_b);
            }

            cur = ggml_mul_mat(ctx,
                    weights.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    weights.layers[il].c_mlp_fc_b);

            cur = ggml_gelu(ctx, cur);

            cur = ggml_mul_mat(ctx,
                    weights.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    weights.layers[il].c_mlp_proj_b);
        }

        inpL = ggml_add(ctx, cur, inpFF);
    }

    {
        inpL = ggml_norm(ctx, inpL, hparams.eps);

        inpL = ggml_add(ctx,
                ggml_mul(ctx,
                    inpL,
                    weights.ln_f_g),
                weights.ln_f_b);
    }

    inpL = ggml_mul_mat(ctx, weights.lm_head, inpL);
    ggml_set_name(inpL, "logits");
    ggml_set_output(inpL);

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx);

    return gf;
}

// serializes all tensor compute — ggml's CPU backend / this graph-building
// code is not safe to call from multiple threads concurrently, so every
// generation step (across every slot) takes turns through this lock. This
// still gives fair interleaving across concurrent requests since the lock is
// only held for the duration of one step, not one whole request.
static std::mutex g_compute_mtx;

static bool gpt2_eval(
        const gpt2_weights & weights,
        gpt2_slot & slot,
        ggml_backend_t backend,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w) {
    const int N = embd_inp.size();
    const int n_vocab = weights.hparams.n_vocab;

    std::lock_guard<std::mutex> lock(g_compute_mtx);

    struct ggml_cgraph * gf = gpt2_graph(weights, slot, n_past, N);

    ggml_gallocr_alloc_graph(slot.allocr, gf);

    struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, embd_inp.data(), 0, N*ggml_element_size(embd));

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
    for (int i = 0; i < N; ++i) {
        int32_t v = n_past + i;
        ggml_backend_tensor_set(position, &v, i*sizeof(int32_t), sizeof(v));
    }

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    ggml_backend_graph_compute(backend, gf);

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");

    embd_w.resize(n_vocab);
    ggml_backend_tensor_get(logits, embd_w.data(), (n_vocab*(N-1))*sizeof(float), sizeof(float)*n_vocab);

    return true;
}

// ---------------------------------------------------------------------------
// slot pool
// ---------------------------------------------------------------------------

struct slot_pool {
    std::vector<std::unique_ptr<gpt2_slot>> slots;
    std::vector<bool> busy;
    std::mutex mtx;
    std::condition_variable cv;

    // returns a slot index, or -1 if none became free within timeout_ms
    int acquire(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mtx);
        int idx = -1;
        cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
            for (size_t i = 0; i < busy.size(); i++) {
                if (!busy[i]) { idx = (int) i; return true; }
            }
            return false;
        });
        if (idx >= 0) busy[idx] = true;
        return idx;
    }

    void release(int idx) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            busy[idx] = false;
        }
        cv.notify_one();
    }
};

struct server_state {
    gpt_vocab    vocab;
    gpt2_weights weights;
    ggml_backend_t backend = nullptr;
    slot_pool    pool;
    int n_threads = 4;         // threads used per generation step
    int n_batch   = 32;
    int queue_timeout_ms = 30000;
};

struct slot_guard {
    slot_pool & pool;
    int idx;
    slot_guard(slot_pool & p, int i) : pool(p), idx(i) {}
    ~slot_guard() { if (idx >= 0) pool.release(idx); }
};

static std::string gen_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%llx%04llx", prefix,
            (unsigned long long) time(NULL),
            (unsigned long long) (counter.fetch_add(1) & 0xffff));
    return buf;
}

// generates text for one request using the given slot. If on_token is set,
// it is invoked with each generated piece as it becomes available (for
// streaming) and generation stops early if it returns false (client gone).
// Returns the full generated continuation (prompt is not echoed).
//
// Note: stop-sequence trimming happens on the accumulated result, so in
// streaming mode the trailing stop text may already have been flushed to the
// client before a match is detected — an accepted approximation for a
// simple example server.
static std::string generate(
        server_state & st,
        int slot_idx,
        const std::string & prompt,
        int n_predict,
        int top_k,
        float top_p,
        float temp,
        int32_t seed,
        const std::vector<std::string> & stop_seqs,
        int & out_prompt_tokens,
        int & out_completion_tokens,
        std::string & finish_reason,
        const std::function<bool(const std::string &)> & on_token) {
    gpt2_slot & slot = *st.pool.slots[slot_idx];

    std::mt19937 rng(seed < 0 ? (uint32_t) time(NULL) : (uint32_t) seed);

    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(st.vocab, prompt);

    const int n_ctx = slot.n_ctx;

    if ((int) embd_inp.size() >= n_ctx) {
        finish_reason = "error";
        out_prompt_tokens = (int) embd_inp.size();
        out_completion_tokens = 0;
        return "";
    }

    n_predict = std::min(n_predict, n_ctx - (int) embd_inp.size());
    if (n_predict < 0) n_predict = 0;

    out_prompt_tokens = (int) embd_inp.size();

    int n_past = 0;
    std::vector<float> logits;
    std::vector<gpt_vocab::id> embd;
    std::string result;
    finish_reason = "length";

    int n_generated = 0;

    for (size_t i = embd.size(); i < embd_inp.size() + (size_t) n_predict; i++) {
        if (embd.size() > 0) {
            if (!gpt2_eval(st.weights, slot, st.backend, st.n_threads, n_past, embd, logits)) {
                finish_reason = "error";
                break;
            }
        }

        n_past += embd.size();
        embd.clear();

        bool is_generating = i >= embd_inp.size();

        if (is_generating) {
            const int n_vocab = st.weights.hparams.n_vocab;
            gpt_vocab::id id = gpt_sample_top_k_top_p(st.vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
            embd.push_back(id);
        } else {
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if ((int32_t) embd.size() >= st.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        if (is_generating) {
            // exactly one token is sampled per generating step (see above)
            const gpt_vocab::id id = embd[0];

            if (id == 50256 /* endoftext */) {
                finish_reason = "stop";
                break;
            }

            const std::string & piece = st.vocab.id_to_token[id];
            result += piece;
            n_generated++;

            if (on_token && !on_token(piece)) {
                finish_reason = "cancelled";
                break;
            }

            bool stopped = false;
            for (const auto & s : stop_seqs) {
                if (!s.empty() && result.size() >= s.size() &&
                    result.compare(result.size() - s.size(), s.size(), s) == 0) {
                    result.resize(result.size() - s.size());
                    finish_reason = "stop";
                    stopped = true;
                    break;
                }
            }
            if (stopped) break;
        }
    }

    out_completion_tokens = n_generated;

    return result;
}

static json make_error(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{
        {"error", {
            {"message", message},
            {"type", type},
        }}
    };
}

static void extract_common_params(const json & body, int & max_tokens, float & temperature, float & top_p, int & top_k, int32_t & seed, std::vector<std::string> & stop_seqs, int default_max_tokens) {
    max_tokens  = body.value("max_tokens", default_max_tokens);
    temperature = body.value("temperature", 0.9f);
    top_p       = body.value("top_p", 0.9f);
    top_k       = body.value("top_k", 40);
    seed        = body.value("seed", -1);

    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) stop_seqs.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array()) for (auto & s : body["stop"]) stop_seqs.push_back(s.get<std::string>());
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string model_path = "models/gpt-2-117M/ggml-model.bin";
    std::string host       = "0.0.0.0";
    int         port       = 8080;
    int         n_ctx      = 1024;
    std::string api_key    = "";
    int         queue_timeout_ms = 30000;
    std::string chat_template = "alpaca"; // "alpaca" or "raw" — see /v1/chat/completions handler

    const int hw = std::max(1u, std::thread::hardware_concurrency());
    int n_slots   = std::max(1, std::min(4, hw / 2));
    int n_threads = 0; // computed below once n_slots is finalized

    if (const char * e = getenv("MODEL_PATH"))       model_path = e;
    if (const char * e = getenv("HOST"))             host       = e;
    if (const char * e = getenv("PORT"))             port       = std::atoi(e);
    if (const char * e = getenv("CTX_SIZE"))         n_ctx      = std::atoi(e);
    if (const char * e = getenv("API_KEY"))          api_key    = e;
    if (const char * e = getenv("SLOTS"))            n_slots    = std::atoi(e);
    if (const char * e = getenv("THREADS"))          n_threads  = std::atoi(e);
    if (const char * e = getenv("QUEUE_TIMEOUT_MS")) queue_timeout_ms = std::atoi(e);
    if (const char * e = getenv("CHAT_TEMPLATE"))    chat_template = e;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", arg.c_str()); exit(1); }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model")        model_path = next();
        else if (arg == "--host")                   host       = next();
        else if (arg == "--port")                   port       = std::stoi(next());
        else if (arg == "-t" || arg == "--threads")  n_threads  = std::stoi(next());
        else if (arg == "-c" || arg == "--ctx-size") n_ctx      = std::stoi(next());
        else if (arg == "--api-key")                 api_key    = next();
        else if (arg == "--slots")                   n_slots    = std::stoi(next());
        else if (arg == "--chat-template")            chat_template = next();
        else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    n_slots = std::max(1, n_slots);
    if (n_threads <= 0) n_threads = std::max(1, hw / n_slots);

    server_state st;
    st.n_threads = n_threads;
    st.queue_timeout_ms = queue_timeout_ms;

    st.backend = ggml_backend_cpu_init();
    if (!st.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return 1;
    }

    if (!gpt2_load_weights(model_path, st.weights, st.vocab, st.backend)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path.c_str());
        return 1;
    }

    st.pool.slots.resize(n_slots);
    st.pool.busy.assign(n_slots, false);
    for (int i = 0; i < n_slots; i++) {
        st.pool.slots[i] = std::make_unique<gpt2_slot>();
        if (!gpt2_slot_init(*st.pool.slots[i], st.weights.hparams, n_ctx, st.backend)) {
            fprintf(stderr, "%s: failed to init slot %d\n", __func__, i);
            return 1;
        }

        // pre-reserve the compute buffer for the worst-case graph so the
        // first request on this slot doesn't pay a slow allocation
        int n_tokens = std::min(st.pool.slots[i]->n_ctx, st.n_batch);
        int n_past   = st.pool.slots[i]->n_ctx - n_tokens;
        struct ggml_cgraph * gf = gpt2_graph(st.weights, *st.pool.slots[i], n_past, n_tokens);
        ggml_gallocr_reserve(st.pool.slots[i]->allocr, gf);
    }

    {
        size_t mem_size = ggml_gallocr_get_buffer_size(st.pool.slots[0]->allocr, 0);
        fprintf(stderr, "%s: compute buffer size per slot: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    const std::string model_name = "gpt-2";

    httplib::Server svr;
    svr.new_task_queue = [n_slots] { return new httplib::ThreadPool(std::max(16, n_slots * 4)); };

    if (!api_key.empty()) {
        svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
            if (req.path == "/health") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            const std::string expected = "Bearer " + api_key;
            if (req.get_header_value("Authorization") != expected) {
                res.status = 401;
                res.set_content(make_error("Incorrect API key provided.", "invalid_request_error").dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id", model_name},
                    {"object", "model"},
                    {"created", (int64_t) time(NULL)},
                    {"owned_by", "local"},
                }
            })}
        };
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(make_error("Invalid JSON in request body.").dump(), "application/json");
            return;
        }

        if (!body.contains("prompt")) {
            res.status = 400;
            res.set_content(make_error("Missing required field: prompt.").dump(), "application/json");
            return;
        }

        std::string prompt;
        if (body["prompt"].is_string()) {
            prompt = body["prompt"].get<std::string>();
        } else if (body["prompt"].is_array() && !body["prompt"].empty()) {
            prompt = body["prompt"][0].get<std::string>();
        }

        int max_tokens, top_k; float temperature, top_p; int32_t seed;
        std::vector<std::string> stop_seqs;
        extract_common_params(body, max_tokens, temperature, top_p, top_k, seed, stop_seqs, 16);

        const bool stream = body.value("stream", false);
        const std::string id = gen_id("cmpl");

        if (!stream) {
            int slot_idx = st.pool.acquire(st.queue_timeout_ms);
            if (slot_idx < 0) {
                res.status = 503;
                res.set_content(make_error("Server is busy, please retry.", "server_error").dump(), "application/json");
                return;
            }
            slot_guard guard(st.pool, slot_idx);

            int prompt_tokens = 0, completion_tokens = 0;
            std::string finish_reason;
            std::string text = generate(st, slot_idx, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, prompt_tokens, completion_tokens, finish_reason, nullptr);

            if (finish_reason == "error") {
                res.status = 400;
                res.set_content(make_error("Prompt is too long for this model's context size.").dump(), "application/json");
                return;
            }

            json j = {
                {"id", id},
                {"object", "text_completion"},
                {"created", (int64_t) time(NULL)},
                {"model", model_name},
                {"choices", json::array({
                    {
                        {"text", text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", finish_reason},
                    }
                })},
                {"usage", {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens},
                }},
            };

            res.set_content(j.dump(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&st, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, id, model_name]
            (size_t /*offset*/, httplib::DataSink & sink) -> bool {
                int slot_idx = st.pool.acquire(st.queue_timeout_ms);
                if (slot_idx < 0) {
                    std::string chunk = "data: " + make_error("Server is busy, please retry.", "server_error").dump() + "\n\n";
                    sink.write(chunk.data(), chunk.size());
                    sink.done();
                    return true;
                }
                slot_guard guard(st.pool, slot_idx);

                int prompt_tokens = 0, completion_tokens = 0;
                std::string finish_reason;

                auto on_token = [&](const std::string & piece) -> bool {
                    json j = {
                        {"id", id},
                        {"object", "text_completion.chunk"},
                        {"created", (int64_t) time(NULL)},
                        {"model", model_name},
                        {"choices", json::array({
                            {{"text", piece}, {"index", 0}, {"logprobs", nullptr}, {"finish_reason", nullptr}}
                        })},
                    };
                    std::string chunk = "data: " + j.dump() + "\n\n";
                    return sink.write(chunk.data(), chunk.size());
                };

                std::string text = generate(st, slot_idx, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, prompt_tokens, completion_tokens, finish_reason, on_token);
                (void) text;

                if (finish_reason == "error") {
                    std::string chunk = "data: " + make_error("Prompt is too long for this model's context size.").dump() + "\n\n";
                    sink.write(chunk.data(), chunk.size());
                    sink.done();
                    return true;
                }

                if (finish_reason != "cancelled") {
                    json jf = {
                        {"id", id},
                        {"object", "text_completion.chunk"},
                        {"created", (int64_t) time(NULL)},
                        {"model", model_name},
                        {"choices", json::array({
                            {{"text", ""}, {"index", 0}, {"logprobs", nullptr}, {"finish_reason", finish_reason}}
                        })},
                    };
                    std::string fchunk = "data: " + jf.dump() + "\n\n";
                    sink.write(fchunk.data(), fchunk.size());
                    sink.write("data: [DONE]\n\n", 15);
                }

                sink.done();
                return true;
            });
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(make_error("Invalid JSON in request body.").dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(make_error("Missing required field: messages.").dump(), "application/json");
            return;
        }

        // "raw" transcript flattening suits a base (non-instruction-tuned)
        // GPT-2; "alpaca" targets instruction-tuned checkpoints fine-tuned on
        // the standard Alpaca-style template (e.g. MBZUAI/LaMini-GPT-124M).
        // Alpaca prompting is inherently single-turn, so only the latest
        // user message is used as the instruction.
        std::string prompt;
        std::vector<std::string> stop_seqs;
        if (chat_template == "raw") {
            for (auto & m : body["messages"]) {
                std::string role    = m.value("role", "user");
                std::string content = m.value("content", "");
                prompt += role + ": " + content + "\n";
            }
            prompt += "assistant:";
            stop_seqs = {"\nuser:", "\nsystem:"};
        } else {
            std::string instruction;
            for (auto & m : body["messages"]) {
                if (m.value("role", "user") == "user") {
                    instruction = m.value("content", "");
                }
            }
            prompt =
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n" + instruction + "\n\n### Response:\n";
            stop_seqs = {"### Instruction:", "\n###"};
        }

        int max_tokens, top_k; float temperature, top_p; int32_t seed;
        extract_common_params(body, max_tokens, temperature, top_p, top_k, seed, stop_seqs, 64);

        const bool stream = body.value("stream", false);
        const std::string id = gen_id("chatcmpl");

        if (!stream) {
            int slot_idx = st.pool.acquire(st.queue_timeout_ms);
            if (slot_idx < 0) {
                res.status = 503;
                res.set_content(make_error("Server is busy, please retry.", "server_error").dump(), "application/json");
                return;
            }
            slot_guard guard(st.pool, slot_idx);

            int prompt_tokens = 0, completion_tokens = 0;
            std::string finish_reason;
            std::string text = generate(st, slot_idx, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, prompt_tokens, completion_tokens, finish_reason, nullptr);

            if (finish_reason == "error") {
                res.status = 400;
                res.set_content(make_error("Prompt is too long for this model's context size.").dump(), "application/json");
                return;
            }

            json j = {
                {"id", id},
                {"object", "chat.completion"},
                {"created", (int64_t) time(NULL)},
                {"model", model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", text},
                        }},
                        {"finish_reason", finish_reason},
                    }
                })},
                {"usage", {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens},
                }},
            };

            res.set_content(j.dump(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&st, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, id, model_name]
            (size_t /*offset*/, httplib::DataSink & sink) -> bool {
                int slot_idx = st.pool.acquire(st.queue_timeout_ms);
                if (slot_idx < 0) {
                    std::string chunk = "data: " + make_error("Server is busy, please retry.", "server_error").dump() + "\n\n";
                    sink.write(chunk.data(), chunk.size());
                    sink.done();
                    return true;
                }
                slot_guard guard(st.pool, slot_idx);

                {
                    json j = {
                        {"id", id}, {"object", "chat.completion.chunk"}, {"created", (int64_t) time(NULL)},
                        {"model", model_name},
                        {"choices", json::array({
                            {{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}
                        })},
                    };
                    std::string chunk = "data: " + j.dump() + "\n\n";
                    sink.write(chunk.data(), chunk.size());
                }

                int prompt_tokens = 0, completion_tokens = 0;
                std::string finish_reason;

                auto on_token = [&](const std::string & piece) -> bool {
                    json j = {
                        {"id", id}, {"object", "chat.completion.chunk"}, {"created", (int64_t) time(NULL)},
                        {"model", model_name},
                        {"choices", json::array({
                            {{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}
                        })},
                    };
                    std::string chunk = "data: " + j.dump() + "\n\n";
                    return sink.write(chunk.data(), chunk.size());
                };

                std::string text = generate(st, slot_idx, prompt, max_tokens, top_k, top_p, temperature, seed, stop_seqs, prompt_tokens, completion_tokens, finish_reason, on_token);
                (void) text;

                if (finish_reason == "error") {
                    std::string chunk = "data: " + make_error("Prompt is too long for this model's context size.").dump() + "\n\n";
                    sink.write(chunk.data(), chunk.size());
                    sink.done();
                    return true;
                }

                if (finish_reason != "cancelled") {
                    json jf = {
                        {"id", id}, {"object", "chat.completion.chunk"}, {"created", (int64_t) time(NULL)},
                        {"model", model_name},
                        {"choices", json::array({
                            {{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}
                        })},
                    };
                    std::string fchunk = "data: " + jf.dump() + "\n\n";
                    sink.write(fchunk.data(), fchunk.size());
                    sink.write("data: [DONE]\n\n", 15);
                }

                sink.done();
                return true;
            });
    });

    fprintf(stderr, "%s: model loaded, n_ctx = %d, slots = %d, threads/slot = %d\n", __func__, st.pool.slots[0]->n_ctx, n_slots, n_threads);
    fprintf(stderr, "%s: listening on http://%s:%d\n", __func__, host.c_str(), port);

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "%s: failed to listen on %s:%d\n", __func__, host.c_str(), port);
        return 1;
    }

    return 0;
}
