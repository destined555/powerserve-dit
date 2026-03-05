// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "encode/sd_text_encoder_runner.hpp"

#include "core/exception.hpp"
#include "core/logger.hpp"
#include "encode/sd_text_encoder_shape_utils.hpp"

#include "ggml.h"
#if __has_include("gguf.h")
#include "gguf.h"
#endif
#if __has_include("ggml-cpu.h")
#include "ggml-cpu.h"
#define POWERSERVE_GGML_SPLIT_HEADERS 1
#else
#define POWERSERVE_GGML_SPLIT_HEADERS 0
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace powerserve {

namespace {

// 运行器负责将 tokenizer 输出的 token 序列，转换为 SD3.5 所需条件向量。
// 该实现目标是与 sd.cpp 的数值路径尽量一致（层结构、激活、拼接方式）。

constexpr const char *kTag = "SDTextEncoderRunner";
// SD3.5 三条文本分支的固定维度契约。
constexpr size_t kClipLDim = 768;
constexpr size_t kClipGDim = 1280;
constexpr size_t kT5Dim = 4096;
constexpr size_t kCrossattnDim = 4096;
constexpr size_t kVectorDim = 2048;
constexpr float kClipLayerNormEps = 1e-5f;
constexpr float kT5LayerNormEps = 1e-6f;

// CLIP 模型结构探测结果（从权重形状动态推断）。
struct ClipModelSpec {
    std::string prefix;
    int64_t hidden_dim = 0;
    int64_t ff_dim = 0;
    int64_t n_heads = 0;
    int64_t n_layers = 0;
    bool use_gelu = false;
    bool has_text_projection = false;
};

// T5 模型结构探测结果（从权重形状动态推断）。
struct T5ModelSpec {
    std::string prefix;
    int64_t model_dim = 0;
    int64_t ff_dim = 0;
    int64_t n_heads = 0;
    int64_t n_layers = 0;
};

// F32 张量的轻量宿主缓存，用于跨上下文搬运中间结果。
struct TensorBlobF32 {
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
    std::vector<float> data;
};

// RAII 封装，确保 ggml_context 正常释放。
class ScopedGGMLContext : Noncopyable {
public:
    explicit ScopedGGMLContext(size_t mem_size) {
        ggml_init_params params{};
        params.mem_size = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        m_ctx = ggml_init(params);
        if (m_ctx == nullptr) {
            throw ConfigException(kTag, fmt::format("ggml_init failed for {} bytes", mem_size));
        }
    }

    ~ScopedGGMLContext() {
        if (m_ctx != nullptr) {
            ggml_free(m_ctx);
            m_ctx = nullptr;
        }
    }

    auto get() const -> ggml_context * {
        return m_ctx;
    }

private:
    ggml_context *m_ctx = nullptr;
};

// 默认线程数策略：优先硬件并发数。
auto default_n_threads() -> int {
    const auto hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

// Token 类型统一转 int32，便于写入 ggml I32 输入张量。
auto to_i32_tokens(const std::vector<Token> &tokens) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(tokens.size());
    for (const Token token : tokens) {
        out.push_back(static_cast<int32_t>(token));
    }
    return out;
}

// 取出第 chunk_idx 个 77-token 分块，不足部分补 0。
auto extract_chunk_tokens(const std::vector<Token> &tokens, size_t chunk_idx) -> std::vector<int32_t> {
    std::vector<int32_t> chunk(detail::kSD3TextChunkLen, 0);
    const size_t offset = chunk_idx * detail::kSD3TextChunkLen;
    for (size_t i = 0; i < detail::kSD3TextChunkLen; ++i) {
        const size_t index = offset + i;
        if (index < tokens.size()) {
            chunk[i] = static_cast<int32_t>(tokens[index]);
        }
    }
    return chunk;
}

// 取出第 chunk_idx 个 token weight 分块，不足部分补 1.0。
auto extract_chunk_weights(const std::vector<float> &weights, size_t chunk_idx) -> std::vector<float> {
    std::vector<float> chunk(detail::kSD3TextChunkLen, 1.0f);
    if (weights.empty()) {
        return chunk;
    }

    const size_t offset = chunk_idx * detail::kSD3TextChunkLen;
    for (size_t i = 0; i < detail::kSD3TextChunkLen; ++i) {
        const size_t index = offset + i;
        if (index < weights.size()) {
            chunk[i] = weights[index];
        }
    }
    return chunk;
}

// 必需 tensor 查询；缺失即抛异常。
auto require_tensor(const SDTextEncoderModelLoader &loader, std::string_view key) -> ggml_tensor * {
    ggml_tensor *tensor = loader.get_tensor(key);
    if (tensor == nullptr) {
        throw ConfigException(kTag, fmt::format("required tensor not found: {}", key));
    }
    return tensor;
}

// 拼接模型前缀与参数名，构造完整 tensor key。
auto make_tensor_key(std::string_view prefix, std::string_view name) -> std::string {
    std::string key;
    key.reserve(prefix.size() + name.size());
    key.append(prefix);
    key.append(name);
    return key;
}

// 兼容旧版 64-char 名称截断：.weight 可能被截成 .w。
auto require_t5_relative_attention_bias_tensor(const SDTextEncoderModelLoader &loader, std::string_view prefix) -> ggml_tensor * {
    const std::string key_weight =
        make_tensor_key(prefix, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
    if (ggml_tensor *tensor = loader.get_tensor(key_weight); tensor != nullptr) {
        return tensor;
    }
    // Legacy 64-char tensor-name builds truncate ".weight" to ".w".
    return require_tensor(loader, make_tensor_key(prefix, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.w"));
}

auto cast_f32(ggml_context *ctx, ggml_tensor *x) -> ggml_tensor * {
    if (x->type == GGML_TYPE_F32) {
        return x;
    }
    if (ggml_is_quantized(x->type)) {
        // 该代码树里 ggml_cast 不支持 quantized -> f32 直接转换。
        // 用 add_cast + 全零 f32 tensor 触发反量化路径。
        ggml_tensor *zeros = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne);
        std::memset(zeros->data, 0, ggml_nbytes(zeros));
        return ggml_add_cast(ctx, x, zeros, GGML_TYPE_F32);
    }
    return ggml_cast(ctx, x, GGML_TYPE_F32);
}

// 读取环境变量布尔开关（调试用途）。
auto env_flag_enabled(const char *name) -> bool {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0;
}

// 将量化类型反量化到 F32（兼容拆分/非拆分 ggml 头文件两种构建）。
void dequantize_to_f32_or_throw(enum ggml_type type, const void *src_data, float *dst_data, int64_t n) {
#if POWERSERVE_GGML_SPLIT_HEADERS
    const ggml_type_traits *traits = ggml_get_type_traits(type);
    POWERSERVE_ASSERT_CONFIG(traits != nullptr && traits->to_float != nullptr, kTag, "unsupported quantized tensor type {}", ggml_type_name(type));
    traits->to_float(src_data, dst_data, n);
#else
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    POWERSERVE_ASSERT_CONFIG(traits.to_float != nullptr, kTag, "unsupported quantized tensor type {}", ggml_type_name(type));
    traits.to_float(src_data, dst_data, n);
#endif
}

// 将任意支持类型 tensor 拷贝为 host 侧 F32 向量。
auto tensor_to_f32_data(ggml_tensor *src) -> std::vector<float> {
    const int64_t n = ggml_nelements(src);
    std::vector<float> out(static_cast<size_t>(n), 0.0f);

    if (src->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src->data, out.size() * sizeof(float));
        return out;
    }
    if (src->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(src->data), out.data(), n);
        return out;
    }
    if (src->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(src->data), out.data(), n);
        return out;
    }

    if (ggml_is_quantized(src->type)) {
        dequantize_to_f32_or_throw(src->type, src->data, out.data(), n);
        return out;
    }

    throw ConfigException(kTag, fmt::format("unsupported tensor type for f32 clone: {}", ggml_type_name(src->type)));
}

// 在指定上下文中克隆一个 F32 tensor（若原本已是 F32 则直接复用）。
auto clone_tensor_as_f32(ggml_context *ctx, ggml_tensor *src) -> ggml_tensor * {
    if (src->type == GGML_TYPE_F32) {
        return src;
    }
    const std::vector<float> data = tensor_to_f32_data(src);
    ggml_tensor *dst = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne);
    std::memcpy(dst->data, data.data(), data.size() * sizeof(float));
    return dst;
}

auto clip_debug_dump_dir() -> const std::string & {
    static const std::string dump_dir = []() -> std::string {
        const char *value = std::getenv("POWERSERVE_CLIP_DEBUG_DUMP_DIR");
        if (value == nullptr) {
            return "";
        }
        return std::string(value);
    }();
    return dump_dir;
}

auto clip_debug_target_layer() -> int {
    static const int layer = []() -> int {
        const char *value = std::getenv("POWERSERVE_CLIP_DEBUG_LAYER");
        if (value == nullptr || value[0] == '\0') {
            return 5;
        }
        return std::atoi(value);
    }();
    return layer;
}

// 调试导出：写入 f32 原始数据 + shape 元信息。
void dump_blob_f32(const TensorBlobF32 &blob, const std::string &path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path, std::ios::binary);
    POWERSERVE_ASSERT_CONFIG(ofs.is_open(), kTag, "failed to open dump path {}", path);
    ofs.write(reinterpret_cast<const char *>(blob.data.data()), static_cast<std::streamsize>(blob.data.size() * sizeof(float)));
    POWERSERVE_ASSERT_CONFIG(ofs.good(), kTag, "failed to write dump path {}", path);

    std::ofstream meta(path + ".shape.txt");
    POWERSERVE_ASSERT_CONFIG(meta.is_open(), kTag, "failed to open shape dump path {}", path + ".shape.txt");
    meta << "ne: " << blob.ne[0] << "," << blob.ne[1] << "," << blob.ne[2] << "," << blob.ne[3] << "\n";
}

// 构图并执行一次前向，失败时携带步骤名报错。
void compute_graph_or_throw(ggml_context *ctx, ggml_tensor *output, int n_threads, std::string_view step) {
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        throw ConfigException(kTag, fmt::format("{} compute failed (status={})", step, static_cast<int>(status)));
    }
}

// 将输出 materialize 为连续 F32 host 缓冲，便于跨上下文传递。
auto materialize_f32(ggml_context *ctx, ggml_tensor *output, int n_threads, std::string_view step) -> TensorBlobF32 {
    output = ggml_cont(ctx, cast_f32(ctx, output));
    compute_graph_or_throw(ctx, output, n_threads, step);

    TensorBlobF32 blob;
    blob.ne = {output->ne[0], output->ne[1], output->ne[2], output->ne[3]};
    blob.data.resize(static_cast<size_t>(ggml_nelements(output)));
    std::memcpy(blob.data.data(), output->data, blob.data.size() * sizeof(float));
    return blob;
}

// 以下是若干“把 host 数据写入 ggml tensor”的便捷函数。
auto new_tensor_2d_f32(ggml_context *ctx, int64_t ne0, int64_t ne1, const std::vector<float> &data) -> ggml_tensor * {
    ggml_tensor *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    const size_t expect = static_cast<size_t>(ne0 * ne1);
    POWERSERVE_ASSERT_CONFIG(data.size() == expect, kTag, "f32 tensor size mismatch: {} vs {}", data.size(), expect);
    std::memcpy(tensor->data, data.data(), data.size() * sizeof(float));
    return tensor;
}

auto new_tensor_1d_f32(ggml_context *ctx, int64_t ne0, const std::vector<float> &data) -> ggml_tensor * {
    ggml_tensor *tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    const size_t expect = static_cast<size_t>(ne0);
    POWERSERVE_ASSERT_CONFIG(data.size() == expect, kTag, "f32 tensor size mismatch: {} vs {}", data.size(), expect);
    std::memcpy(tensor->data, data.data(), data.size() * sizeof(float));
    return tensor;
}

auto new_tensor_2d_i32(ggml_context *ctx, int64_t ne0, int64_t ne1, const std::vector<int32_t> &data) -> ggml_tensor * {
    ggml_tensor *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne0, ne1);
    const size_t expect = static_cast<size_t>(ne0 * ne1);
    POWERSERVE_ASSERT_CONFIG(data.size() == expect, kTag, "i32 tensor size mismatch: {} vs {}", data.size(), expect);
    std::memcpy(tensor->data, data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

auto new_tensor_from_blob_f32(ggml_context *ctx, const TensorBlobF32 &blob) -> ggml_tensor * {
    ggml_tensor *tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, const_cast<int64_t *>(blob.ne.data()));
    const size_t expect = static_cast<size_t>(ggml_nelements(tensor));
    POWERSERVE_ASSERT_CONFIG(blob.data.size() == expect, kTag, "blob tensor size mismatch: {} vs {}", blob.data.size(), expect);
    std::memcpy(tensor->data, blob.data.data(), blob.data.size() * sizeof(float));
    return tensor;
}

// 线性层封装：out = W*x + b，可选缩放与 matmul 精度策略。
auto linear_forward(
    ggml_context *ctx,
    ggml_tensor *x,
    ggml_tensor *w,
    ggml_tensor *b = nullptr,
    bool force_prec_f32 = false,
    float scale = 1.0f
) -> ggml_tensor * {
    if (scale != 1.0f) {
        x = ggml_scale(ctx, x, scale);
    }

    ggml_tensor *out = ggml_mul_mat(ctx, w, x);
    if (force_prec_f32) {
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    }

    if (scale != 1.0f) {
        out = ggml_scale(ctx, out, 1.0f / scale);
    }

    if (b != nullptr) {
        out = ggml_add_inplace(ctx, out, cast_f32(ctx, b));
    }
    return out;
}

// CLIP 使用的 LayerNorm：norm -> 仿射。
auto layer_norm_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b, float eps) -> ggml_tensor * {
    ggml_tensor *out = ggml_norm(ctx, x, eps);
    out = ggml_mul_inplace(ctx, out, cast_f32(ctx, w));
    out = ggml_add_inplace(ctx, out, cast_f32(ctx, b));
    return out;
}

// T5 使用的 RMSNorm：rms_norm -> 乘权重（无 bias）。
auto rms_norm_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, float eps) -> ggml_tensor * {
    ggml_tensor *out = ggml_rms_norm(ctx, x, eps);
    out = ggml_mul(ctx, out, cast_f32(ctx, w));
    return out;
}

// 通用 embedding lookup，输入是 token id，输出 [dim, token, batch]。
auto embedding_forward(ggml_context *ctx, ggml_tensor *weight, ggml_tensor *input_ids) -> ggml_tensor * {
    const int64_t n = input_ids->ne[1];
    input_ids = ggml_reshape_1d(ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);
    input_ids = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);

    ggml_tensor *embedding = ggml_get_rows(ctx, weight, input_ids);
    embedding = ggml_reshape_3d(ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);
    return embedding;
}

// CLIP embedding：token embedding + positional embedding。
auto clip_embedding_forward(
    ggml_context *ctx,
    ggml_tensor *token_embed_weight,
    ggml_tensor *position_embed_weight,
    ggml_tensor *input_ids
) -> ggml_tensor * {
    input_ids = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
    ggml_tensor *token_embedding = ggml_get_rows(ctx, token_embed_weight, input_ids);
    token_embedding = ggml_reshape_3d(ctx, token_embedding, token_embedding->ne[0], token_embedding->ne[1], token_embedding->ne[3]);
    if (position_embed_weight->type == GGML_TYPE_F32) {
        return ggml_add(ctx, token_embedding, position_embed_weight);
    }
    if (ggml_is_quantized(position_embed_weight->type) || position_embed_weight->type == GGML_TYPE_F16 ||
        position_embed_weight->type == GGML_TYPE_BF16) {
        // 让 quantized/F16/BF16 的位置编码在 add_cast 中提升到 F32。
        return ggml_add_cast(ctx, position_embed_weight, token_embedding, GGML_TYPE_F32);
    }
    return ggml_add(ctx, cast_f32(ctx, token_embedding), cast_f32(ctx, position_embed_weight));
}

// 通用多头注意力前向（q/k/v 已是线性投影后的张量）。
auto attention_forward(
    ggml_context *ctx,
    ggml_tensor *q,
    ggml_tensor *k,
    ggml_tensor *v,
    int64_t n_heads,
    ggml_tensor *mask,
    float qk_scale = -1.0f
) -> ggml_tensor * {
    const int64_t lq = q->ne[1];
    const int64_t lk = k->ne[1];
    const int64_t c = q->ne[0];
    const int64_t n = q->ne[2];

    const int64_t d_head = c / n_heads;
    const int64_t n_kv_heads = k->ne[0] / d_head;

    q = ggml_reshape_4d(ctx, q, d_head, n_heads, lq, n);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    q = ggml_reshape_3d(ctx, q, d_head, lq, n_heads * n);

    k = ggml_reshape_4d(ctx, k, d_head, n_kv_heads, lk, n);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    k = ggml_reshape_3d(ctx, k, d_head, lk, n_kv_heads * n);

    v = ggml_reshape_4d(ctx, v, d_head, n_kv_heads, lk, n);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    v = ggml_reshape_3d(ctx, v, lk, d_head, n_kv_heads * n);

    const float scale = (qk_scale > 0.0f) ? qk_scale : (1.0f / std::sqrt(static_cast<float>(d_head)));

    ggml_tensor *kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_scale_inplace(ctx, kq, scale);
    if (mask != nullptr) {
        kq = ggml_add_inplace(ctx, kq, mask);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    ggml_tensor *kqv = ggml_mul_mat(ctx, v, kq);
    kqv = ggml_reshape_4d(ctx, kqv, d_head, lq, n_heads, n);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_heads, lq, n);
    return kqv;
}

// 构造 CLIP 自回归 causal mask（上三角 -inf）。
auto build_clip_causal_mask(ggml_context *ctx, int64_t n_tokens) -> ggml_tensor * {
    std::vector<float> mask(static_cast<size_t>(n_tokens * n_tokens), 0.0f);
    for (int64_t i0 = 0; i0 < n_tokens; ++i0) {
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            float value = 0.0f;
            if (i0 > i1) {
                value = -std::numeric_limits<float>::infinity();
            }
            mask[static_cast<size_t>(i1 * n_tokens + i0)] = value;
        }
    }
    return new_tensor_2d_f32(ctx, n_tokens, n_tokens, mask);
}

// 由 hidden_dim 推断 CLIP 头数（与 SD3 常见配置匹配）。
auto infer_clip_heads(int64_t hidden_dim) -> int64_t {
    if (hidden_dim == 768) {
        return 12;
    }
    if (hidden_dim == 1024) {
        return 16;
    }
    if (hidden_dim == 1280) {
        return 20;
    }
    throw ConfigException(kTag, fmt::format("unsupported CLIP hidden_dim: {}", hidden_dim));
}

// 通过第 0 层 q_proj 是否存在来探测层数。
auto detect_clip_layers(const SDTextEncoderModelLoader &loader, std::string_view prefix) -> int64_t {
    int64_t n_layers = 0;
    while (true) {
        const std::string q_key = make_tensor_key(
            prefix,
            fmt::format("text_model.encoder.layers.{}.self_attn.q_proj.weight", n_layers)
        );
        if (!loader.has_tensor(q_key)) {
            break;
        }
        ++n_layers;
    }
    if (n_layers == 0) {
        throw ConfigException(kTag, fmt::format("failed to detect clip layers for prefix {}", prefix));
    }
    return n_layers;
}

// 通过第 0 层 T5 SelfAttention.q 是否存在来探测层数。
auto detect_t5_layers(const SDTextEncoderModelLoader &loader, std::string_view prefix) -> int64_t {
    int64_t n_layers = 0;
    while (true) {
        const std::string q_key = make_tensor_key(
            prefix,
            fmt::format("encoder.block.{}.layer.0.SelfAttention.q.weight", n_layers)
        );
        if (!loader.has_tensor(q_key)) {
            break;
        }
        ++n_layers;
    }
    if (n_layers == 0) {
        throw ConfigException(kTag, fmt::format("failed to detect t5 layers for prefix {}", prefix));
    }
    return n_layers;
}

// 从加载器中探测 CLIP 模型结构，并做 projection 存在性约束。
auto detect_clip_spec(const SDTextEncoderModelLoader &loader, std::string prefix, bool expect_projection) -> ClipModelSpec {
    ClipModelSpec spec;
    spec.prefix = std::move(prefix);

    ggml_tensor *token_embedding = require_tensor(loader, make_tensor_key(spec.prefix, "text_model.embeddings.token_embedding.weight"));
    spec.hidden_dim = token_embedding->ne[0];
    spec.n_heads = infer_clip_heads(spec.hidden_dim);
    spec.n_layers = detect_clip_layers(loader, spec.prefix);

    ggml_tensor *fc1_bias = require_tensor(
        loader,
        make_tensor_key(spec.prefix, "text_model.encoder.layers.0.mlp.fc1.bias")
    );
    spec.ff_dim = fc1_bias->ne[0];

    // 与 sd.cpp 对齐：hidden=1024/1280 时使用 gelu，其它走 gelu_quick。
    spec.use_gelu = (spec.hidden_dim == 1024 || spec.hidden_dim == 1280);

    const std::string projection_key = make_tensor_key(spec.prefix, "text_projection.weight");
    spec.has_text_projection = loader.has_tensor(projection_key);
    if (expect_projection && !spec.has_text_projection) {
        throw ConfigException(kTag, fmt::format("missing clip projection tensor: {}", projection_key));
    }
    if (!expect_projection && spec.has_text_projection) {
        throw ConfigException(kTag, fmt::format("unexpected clip projection tensor: {}", projection_key));
    }

    return spec;
}

// 从加载器中探测 T5 模型结构。
auto detect_t5_spec(const SDTextEncoderModelLoader &loader, std::string prefix) -> T5ModelSpec {
    T5ModelSpec spec;
    spec.prefix = std::move(prefix);

    ggml_tensor *embed = require_tensor(loader, make_tensor_key(spec.prefix, "encoder.embed_tokens.weight"));
    spec.model_dim = embed->ne[0];
    spec.n_layers = detect_t5_layers(loader, spec.prefix);

    ggml_tensor *wi0 = require_tensor(
        loader,
        make_tensor_key(spec.prefix, "encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    );
    spec.ff_dim = wi0->ne[1];

    ggml_tensor *relative_bias = require_t5_relative_attention_bias_tensor(loader, spec.prefix);
    spec.n_heads = relative_bias->ne[0];

    return spec;
}

// SD3.5 约束校验：clip_l=768, clip_g=1280, t5=4096。
void validate_sd3_shape_contract(const ClipModelSpec &clip_l, const ClipModelSpec &clip_g, const T5ModelSpec &t5) {
    if (clip_l.hidden_dim != static_cast<int64_t>(kClipLDim)) {
        throw ConfigException(kTag, fmt::format("clip_l hidden_dim mismatch: expected {}, got {}", kClipLDim, clip_l.hidden_dim));
    }
    if (clip_g.hidden_dim != static_cast<int64_t>(kClipGDim)) {
        throw ConfigException(kTag, fmt::format("clip_g hidden_dim mismatch: expected {}, got {}", kClipGDim, clip_g.hidden_dim));
    }
    if (t5.model_dim != static_cast<int64_t>(kT5Dim)) {
        throw ConfigException(kTag, fmt::format("t5 hidden_dim mismatch: expected {}, got {}", kT5Dim, t5.model_dim));
    }
}

auto run_clip_hidden(
    const SDTextEncoderModelLoader &loader,
    const ClipModelSpec &spec,
    const std::vector<int32_t> &tokens,
    int clip_skip,
    bool apply_final_layer_norm,
    int n_threads
) -> TensorBlobF32 {
    // CLIP 主干前向：
    // embedding -> N 层 Transformer -> (可选) final layer norm。
    static std::atomic<int> dump_call_index{0};
    const size_t mem_size = (spec.hidden_dim >= 1280) ? (1024ull * 1024ull * 1024ull) : (512ull * 1024ull * 1024ull);
    ScopedGGMLContext scoped_ctx(mem_size);
    ggml_context *ctx = scoped_ctx.get();

    ggml_tensor *input_ids = new_tensor_2d_i32(ctx, static_cast<int64_t>(tokens.size()), 1, tokens);

    ggml_tensor *token_embedding = require_tensor(loader, make_tensor_key(spec.prefix, "text_model.embeddings.token_embedding.weight"));
    ggml_tensor *position_embedding = require_tensor(loader, make_tensor_key(spec.prefix, "text_model.embeddings.position_embedding.weight"));
    // 以下 debug 开关仅用于问题定位，不影响默认路径。
    const bool debug_force_affine_f32 = env_flag_enabled("POWERSERVE_DEBUG_FORCE_F32_AFFINE");
    const bool debug_force_layer5_linear_f32 = env_flag_enabled("POWERSERVE_DEBUG_FORCE_F32_LAYER5_LINEAR");
    const std::string &debug_dump_dir = clip_debug_dump_dir();
    const bool debug_dump_this_call = !debug_dump_dir.empty() && !apply_final_layer_norm &&
                                      spec.hidden_dim == static_cast<int64_t>(kClipGDim);
    const int debug_dump_call_idx = debug_dump_this_call ? dump_call_index.fetch_add(1) : -1;
    const int debug_layer_idx = clip_debug_target_layer();
    if (debug_force_affine_f32) {
        position_embedding = clone_tensor_as_f32(ctx, position_embedding);
    }

    ggml_tensor *x = clip_embedding_forward(ctx, token_embedding, position_embedding, input_ids);
    ggml_tensor *mask = build_clip_causal_mask(ctx, static_cast<int64_t>(tokens.size()));

    int effective_clip_skip = clip_skip;
    if (effective_clip_skip <= 0) {
        effective_clip_skip = 2;
    }
    int64_t layer_stop = spec.n_layers - 1;
    if (effective_clip_skip > 0 && effective_clip_skip <= spec.n_layers) {
        // clip_skip 与 A1111/sd.cpp 一致，决定停在倒数第几层隐藏状态。
        layer_stop = spec.n_layers - effective_clip_skip;
    }

    for (int64_t layer_idx = 0; layer_idx <= layer_stop; ++layer_idx) {
        const std::string base = fmt::format("text_model.encoder.layers.{}.", layer_idx);

        ggml_tensor *ln1_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer_norm1.weight"));
        ggml_tensor *ln1_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer_norm1.bias"));
        ggml_tensor *ln2_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer_norm2.weight"));
        ggml_tensor *ln2_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer_norm2.bias"));

        ggml_tensor *q_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.q_proj.weight"));
        ggml_tensor *q_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.q_proj.bias"));
        ggml_tensor *k_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.k_proj.weight"));
        ggml_tensor *k_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.k_proj.bias"));
        ggml_tensor *v_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.v_proj.weight"));
        ggml_tensor *v_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.v_proj.bias"));
        ggml_tensor *o_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.out_proj.weight"));
        ggml_tensor *o_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "self_attn.out_proj.bias"));

        ggml_tensor *fc1_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "mlp.fc1.weight"));
        ggml_tensor *fc1_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "mlp.fc1.bias"));
        ggml_tensor *fc2_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "mlp.fc2.weight"));
        ggml_tensor *fc2_b = require_tensor(loader, make_tensor_key(spec.prefix, base + "mlp.fc2.bias"));

        if (debug_force_affine_f32) {
            ln1_w = clone_tensor_as_f32(ctx, ln1_w);
            ln1_b = clone_tensor_as_f32(ctx, ln1_b);
            ln2_w = clone_tensor_as_f32(ctx, ln2_w);
            ln2_b = clone_tensor_as_f32(ctx, ln2_b);

            q_b = clone_tensor_as_f32(ctx, q_b);
            k_b = clone_tensor_as_f32(ctx, k_b);
            v_b = clone_tensor_as_f32(ctx, v_b);
            o_b = clone_tensor_as_f32(ctx, o_b);
            fc1_b = clone_tensor_as_f32(ctx, fc1_b);
            fc2_b = clone_tensor_as_f32(ctx, fc2_b);
        }
        if (debug_force_layer5_linear_f32 && spec.hidden_dim == static_cast<int64_t>(kClipGDim) && layer_idx == 5) {
            q_w = clone_tensor_as_f32(ctx, q_w);
            k_w = clone_tensor_as_f32(ctx, k_w);
            v_w = clone_tensor_as_f32(ctx, v_w);
            o_w = clone_tensor_as_f32(ctx, o_w);
            fc1_w = clone_tensor_as_f32(ctx, fc1_w);
            fc2_w = clone_tensor_as_f32(ctx, fc2_w);
        }

        ggml_tensor *attn_norm = layer_norm_forward(ctx, x, ln1_w, ln1_b, kClipLayerNormEps);

        ggml_tensor *q = linear_forward(ctx, attn_norm, q_w, q_b, false, 1.0f);
        ggml_tensor *k = linear_forward(ctx, attn_norm, k_w, k_b, false, 1.0f);
        ggml_tensor *v = linear_forward(ctx, attn_norm, v_w, v_b, false, 1.0f);
        if (debug_dump_this_call && layer_idx == debug_layer_idx) {
            const std::filesystem::path base = std::filesystem::path(debug_dump_dir) / fmt::format("powerserve_call{}", debug_dump_call_idx);
            const TensorBlobF32 q_blob = materialize_f32(ctx, q, n_threads, "clip debug q");
            dump_blob_f32(q_blob, (base.string() + "_q.bin"));
            q = new_tensor_from_blob_f32(ctx, q_blob);

            const TensorBlobF32 k_blob = materialize_f32(ctx, k, n_threads, "clip debug k");
            dump_blob_f32(k_blob, (base.string() + "_k.bin"));
            k = new_tensor_from_blob_f32(ctx, k_blob);

            const TensorBlobF32 v_blob = materialize_f32(ctx, v, n_threads, "clip debug v");
            dump_blob_f32(v_blob, (base.string() + "_v.bin"));
            v = new_tensor_from_blob_f32(ctx, v_blob);
        }

        ggml_tensor *attn = attention_forward(ctx, q, k, v, spec.n_heads, mask);
        if (debug_dump_this_call && layer_idx == debug_layer_idx) {
            const TensorBlobF32 attn_ctx = materialize_f32(ctx, attn, n_threads, "clip debug attn ctx");
            dump_blob_f32(
                attn_ctx,
                (std::filesystem::path(debug_dump_dir) /
                 fmt::format("powerserve_call{}_attn_ctx.bin", debug_dump_call_idx))
                    .string()
            );
            attn = new_tensor_from_blob_f32(ctx, attn_ctx);
        }
        attn = linear_forward(ctx, attn, o_w, o_b, false, 1.0f);
        if (debug_dump_this_call && layer_idx == debug_layer_idx) {
            const TensorBlobF32 attn_out = materialize_f32(ctx, attn, n_threads, "clip debug attn out");
            dump_blob_f32(
                attn_out,
                (std::filesystem::path(debug_dump_dir) /
                 fmt::format("powerserve_call{}_attn_out.bin", debug_dump_call_idx))
                    .string()
            );
            attn = new_tensor_from_blob_f32(ctx, attn_out);
        }
        x = ggml_add(ctx, cast_f32(ctx, x), cast_f32(ctx, attn));
        if (debug_dump_this_call && layer_idx == debug_layer_idx) {
            const TensorBlobF32 post_attn = materialize_f32(ctx, x, n_threads, "clip debug post attn");
            dump_blob_f32(
                post_attn,
                (std::filesystem::path(debug_dump_dir) /
                 fmt::format("powerserve_call{}_post_attn.bin", debug_dump_call_idx))
                    .string()
            );
            x = new_tensor_from_blob_f32(ctx, post_attn);
        }

        ggml_tensor *mlp_norm = layer_norm_forward(ctx, x, ln2_w, ln2_b, kClipLayerNormEps);
        ggml_tensor *mlp = linear_forward(ctx, mlp_norm, fc1_w, fc1_b, false, 1.0f);
        if (!ggml_is_contiguous(mlp)) {
            mlp = ggml_cont(ctx, mlp);
        }
        mlp = spec.use_gelu ? ggml_gelu_inplace(ctx, mlp) : ggml_gelu_quick_inplace(ctx, mlp);
        mlp = linear_forward(ctx, mlp, fc2_w, fc2_b, false, 1.0f);

        x = ggml_add(ctx, cast_f32(ctx, x), cast_f32(ctx, mlp));
        if (debug_dump_this_call && layer_idx == debug_layer_idx) {
            const TensorBlobF32 post_mlp = materialize_f32(ctx, x, n_threads, "clip debug post mlp");
            dump_blob_f32(
                post_mlp,
                (std::filesystem::path(debug_dump_dir) /
                 fmt::format("powerserve_call{}_post_mlp.bin", debug_dump_call_idx))
                    .string()
            );
            x = new_tensor_from_blob_f32(ctx, post_mlp);
        }
    }

    if (apply_final_layer_norm) {
        ggml_tensor *final_ln_w = require_tensor(loader, make_tensor_key(spec.prefix, "text_model.final_layer_norm.weight"));
        ggml_tensor *final_ln_b = require_tensor(loader, make_tensor_key(spec.prefix, "text_model.final_layer_norm.bias"));
        if (debug_force_affine_f32) {
            final_ln_w = clone_tensor_as_f32(ctx, final_ln_w);
            final_ln_b = clone_tensor_as_f32(ctx, final_ln_b);
        }
        x = layer_norm_forward(ctx, x, final_ln_w, final_ln_b, kClipLayerNormEps);
    }

    return materialize_f32(ctx, x, n_threads, "clip hidden");
}

auto run_clip_pooled(
    const SDTextEncoderModelLoader &loader,
    const ClipModelSpec &spec,
    const std::vector<int32_t> &tokens,
    int eos_token_id,
    int n_threads
) -> std::vector<float> {
    // pooled 向量取 eos 位置隐状态，再按需过 text_projection。
    const TensorBlobF32 full_hidden = run_clip_hidden(loader, spec, tokens, 1, true, n_threads);
    const int64_t dim = full_hidden.ne[0];
    const int64_t token_count = full_hidden.ne[1];

    size_t eos_index = tokens.size() - 1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == eos_token_id) {
            eos_index = i;
            break;
        }
    }
    eos_index = std::min<size_t>(eos_index, static_cast<size_t>(token_count - 1));

    std::vector<float> pooled(static_cast<size_t>(dim));
    const float *src = full_hidden.data.data() + eos_index * static_cast<size_t>(dim);
    std::memcpy(pooled.data(), src, pooled.size() * sizeof(float));

    if (!spec.has_text_projection) {
        return pooled;
    }

    ScopedGGMLContext scoped_ctx(256ull * 1024ull * 1024ull);
    ggml_context *ctx = scoped_ctx.get();
    ggml_tensor *pooled_tensor = new_tensor_1d_f32(ctx, dim, pooled);
    ggml_tensor *projection = require_tensor(loader, make_tensor_key(spec.prefix, "text_projection.weight"));
    projection = clone_tensor_as_f32(ctx, projection);
    ggml_tensor *projected = linear_forward(ctx, pooled_tensor, projection, nullptr);

    const TensorBlobF32 projected_blob = materialize_f32(ctx, projected, n_threads, "clip pooled projection");
    return projected_blob.data;
}

// T5 相对位置桶逻辑，保持与 sd.cpp 实现一致。
auto relative_position_bucket_one_row(const std::vector<int32_t> &relative_position) -> std::vector<int32_t> {
    int32_t num_buckets = 32;
    const int32_t max_distance = 128;

    std::vector<int32_t> relative_buckets(relative_position.size(), 0);
    std::vector<int32_t> abs_relative_position = relative_position;

    num_buckets = num_buckets / 2;
    for (size_t i = 0; i < relative_position.size(); ++i) {
        if (relative_position[i] > 0) {
            relative_buckets[i] += num_buckets;
        }
        abs_relative_position[i] = std::abs(relative_position[i]);
    }

    const int32_t max_exact = num_buckets / 2;
    for (size_t i = 0; i < relative_position.size(); ++i) {
        if (abs_relative_position[i] < max_exact) {
            relative_buckets[i] += abs_relative_position[i];
        } else {
            const float log_pos = std::log(static_cast<float>(abs_relative_position[i]) / static_cast<float>(max_exact));
            const float log_base = std::log(static_cast<float>(max_distance) / static_cast<float>(max_exact));
            int32_t relative_if_large =
                max_exact + static_cast<int32_t>((log_pos / log_base) * static_cast<float>(num_buckets - max_exact));
            relative_if_large = std::min<int32_t>(relative_if_large, num_buckets - 1);
            relative_buckets[i] += relative_if_large;
        }
    }

    return relative_buckets;
}

// 生成 [query_len, key_len] 的 bucket id 表。
auto compute_relative_position_bucket(int32_t query_length, int32_t key_length) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(query_length * key_length));

    for (int32_t i = 0; i < query_length; ++i) {
        std::vector<int32_t> row;
        row.reserve(static_cast<size_t>(key_length));
        for (int32_t j = 0; j < key_length; ++j) {
            row.push_back(j - i);
        }
        auto row_bucket = relative_position_bucket_one_row(row);
        out.insert(out.end(), row_bucket.begin(), row_bucket.end());
    }
    return out;
}

// 预计算 T5 注意力 past_bias（可在各层复用）。
auto compute_t5_past_bias(
    const SDTextEncoderModelLoader &loader,
    const T5ModelSpec &spec,
    int64_t seq_len,
    int n_threads
) -> TensorBlobF32 {
    ScopedGGMLContext scoped_ctx(256ull * 1024ull * 1024ull);
    ggml_context *ctx = scoped_ctx.get();

    const std::vector<int32_t> buckets = compute_relative_position_bucket(static_cast<int32_t>(seq_len), static_cast<int32_t>(seq_len));
    ggml_tensor *bucket_tensor = new_tensor_2d_i32(ctx, seq_len, seq_len, buckets);

    ggml_tensor *relative_bias_w = require_t5_relative_attention_bias_tensor(loader, spec.prefix);

    ggml_tensor *values = embedding_forward(ctx, relative_bias_w, bucket_tensor);
    ggml_tensor *past_bias = ggml_cont(ctx, ggml_permute(ctx, values, 2, 0, 1, 3));
    return materialize_f32(ctx, past_bias, n_threads, "t5 relative bias");
}

// 单层 T5 encoder 前向：
// rms_norm + self_attn + residual + ffn(gated-gelu) + residual
auto run_t5_layer(
    const SDTextEncoderModelLoader &loader,
    const T5ModelSpec &spec,
    int64_t layer_idx,
    const std::vector<float> &input,
    int64_t seq_len,
    const TensorBlobF32 &past_bias,
    int n_threads
) -> TensorBlobF32 {
    ScopedGGMLContext scoped_ctx(1024ull * 1024ull * 1024ull);
    ggml_context *ctx = scoped_ctx.get();

    ggml_tensor *x = new_tensor_2d_f32(ctx, spec.model_dim, seq_len, input);

    const std::string base = fmt::format("encoder.block.{}.", layer_idx);

    ggml_tensor *ln1_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.0.layer_norm.weight"));
    ggml_tensor *q_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.0.SelfAttention.q.weight"));
    ggml_tensor *k_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.0.SelfAttention.k.weight"));
    ggml_tensor *v_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.0.SelfAttention.v.weight"));
    ggml_tensor *o_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.0.SelfAttention.o.weight"));

    ggml_tensor *ln2_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.1.layer_norm.weight"));
    ggml_tensor *wi0_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.1.DenseReluDense.wi_0.weight"));
    ggml_tensor *wi1_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.1.DenseReluDense.wi_1.weight"));
    ggml_tensor *wo_w = require_tensor(loader, make_tensor_key(spec.prefix, base + "layer.1.DenseReluDense.wo.weight"));

    ggml_tensor *attn_norm = rms_norm_forward(ctx, x, ln1_w, kT5LayerNormEps);
    ggml_tensor *q = linear_forward(ctx, attn_norm, q_w, nullptr);
    ggml_tensor *k = linear_forward(ctx, attn_norm, k_w, nullptr);
    ggml_tensor *v = linear_forward(ctx, attn_norm, v_w, nullptr);
    const int64_t d_head = spec.model_dim / spec.n_heads;
    k = ggml_is_contiguous(k) ? k : ggml_cont(ctx, k);
    k = ggml_scale_inplace(ctx, k, std::sqrt(static_cast<float>(d_head)));

    ggml_tensor *past_bias_tensor = new_tensor_from_blob_f32(ctx, past_bias);
    ggml_tensor *attn = attention_forward(ctx, q, k, v, spec.n_heads, past_bias_tensor);
    attn = linear_forward(ctx, attn, o_w, nullptr);

    x = ggml_add_inplace(ctx, cast_f32(ctx, attn), cast_f32(ctx, x));

    ggml_tensor *ff_norm = rms_norm_forward(ctx, x, ln2_w, kT5LayerNormEps);
    ggml_tensor *hidden_gelu = ggml_gelu(ctx, linear_forward(ctx, ff_norm, wi0_w, nullptr));
    ggml_tensor *hidden_linear = linear_forward(ctx, ff_norm, wi1_w, nullptr);
    ggml_tensor *ff = ggml_mul_inplace(ctx, hidden_gelu, hidden_linear);
    ff = linear_forward(ctx, ff, wo_w, nullptr, false, 1.0f / 32.0f);

    x = ggml_add_inplace(ctx, cast_f32(ctx, ff), cast_f32(ctx, x));

    return materialize_f32(ctx, x, n_threads, "t5 layer");
}

auto run_t5_hidden(
    const SDTextEncoderModelLoader &loader,
    const T5ModelSpec &spec,
    const std::vector<int32_t> &tokens,
    int n_threads
) -> TensorBlobF32 {
    const int64_t seq_len = static_cast<int64_t>(tokens.size());

    // embedding lookup
    std::vector<float> x_data;
    {
        ScopedGGMLContext scoped_ctx(512ull * 1024ull * 1024ull);
        ggml_context *ctx = scoped_ctx.get();

        ggml_tensor *input_ids = new_tensor_2d_i32(ctx, seq_len, 1, tokens);
        ggml_tensor *embed_w = require_tensor(loader, make_tensor_key(spec.prefix, "encoder.embed_tokens.weight"));
        ggml_tensor *x = embedding_forward(ctx, embed_w, input_ids);

        const TensorBlobF32 embedding_blob = materialize_f32(ctx, x, n_threads, "t5 embedding");
        x_data = embedding_blob.data;
    }

    const TensorBlobF32 past_bias = compute_t5_past_bias(loader, spec, seq_len, n_threads);

    for (int64_t layer_idx = 0; layer_idx < spec.n_layers; ++layer_idx) {
        TensorBlobF32 layer_out = run_t5_layer(loader, spec, layer_idx, x_data, seq_len, past_bias, n_threads);
        x_data = std::move(layer_out.data);
    }

    // final layer norm
    {
        ScopedGGMLContext scoped_ctx(512ull * 1024ull * 1024ull);
        ggml_context *ctx = scoped_ctx.get();

        ggml_tensor *x = new_tensor_2d_f32(ctx, spec.model_dim, seq_len, x_data);
        ggml_tensor *final_ln = require_tensor(loader, make_tensor_key(spec.prefix, "encoder.final_layer_norm.weight"));
        x = rms_norm_forward(ctx, x, final_ln, kT5LayerNormEps);
        return materialize_f32(ctx, x, n_threads, "t5 final norm");
    }
}

// 与 sd.cpp ggml_ext_tensor_mean() 对齐的均值实现。
auto tensor_mean_sdcpp_style(const std::vector<float> &data) -> float {
    float mean = 0.0f;
    if (data.empty()) {
        return mean;
    }

    const float n = static_cast<float>(data.size());
    // 等价于 sd.cpp: mean += data[i] / n * 1.0f
    for (const float value : data) {
        mean += value / n * 1.0f;
    }
    return mean;
}

// 应用 token 权重后做均值回缩放，减小加权引入的整体量纲漂移。
void apply_token_weights_and_mean_rescale(TensorBlobF32 &hidden, const std::vector<float> &token_weights) {
    if (hidden.data.empty()) {
        return;
    }

    POWERSERVE_ASSERT_CONFIG(
        hidden.ne[1] == static_cast<int64_t>(token_weights.size()),
        kTag,
        "token weight length mismatch: hidden tokens={} vs weights={}",
        hidden.ne[1],
        token_weights.size()
    );

    const int64_t ne0 = hidden.ne[0];
    const int64_t ne1 = hidden.ne[1];
    const int64_t ne2 = hidden.ne[2];
    const int64_t ne3 = hidden.ne[3];

    const float original_mean = tensor_mean_sdcpp_style(hidden.data);

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const float w = token_weights[static_cast<size_t>(i1)];
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t idx = static_cast<size_t>(i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3)));
                    hidden.data[idx] *= w;
                }
            }
        }
    }

    const float new_mean = tensor_mean_sdcpp_style(hidden.data);
    if (std::isfinite(original_mean) && std::isfinite(new_mean) && std::abs(new_mean) > 1e-20f) {
        const float scale = original_mean / new_mean;
        if (std::isfinite(scale)) {
            for (float &value : hidden.data) {
                value *= scale;
            }
        }
    }
}

// 关键张量存在性检查，尽早暴露模型文件不匹配问题。
void validate_required_tensors(const SDTextEncoderModelLoader &loader) {
    (void)require_tensor(loader, "text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight");
    (void)require_tensor(loader, "text_encoders.clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight");
    (void)require_tensor(loader, "text_encoders.clip_g.transformer.text_model.embeddings.token_embedding.weight");
    (void)require_tensor(loader, "text_encoders.clip_g.transformer.text_projection.weight");
    (void)require_tensor(loader, "text_encoders.t5xxl.transformer.encoder.embed_tokens.weight");
    (void)require_t5_relative_attention_bias_tensor(loader, "text_encoders.t5xxl.transformer.");
}

} // namespace

SDTextEncoderRunner::SDTextEncoderRunner(
    const SDTextEncoderModelLoader &loader, const TextEncodeConfig &config, int n_threads
) :
    m_loader(&loader), m_config(config), m_n_threads(n_threads) {
    if (m_loader->tensor_count() == 0) {
        throw ConfigException(kTag, "text encoder tensors are empty");
    }
    if (m_config.clip_l.path.empty() || m_config.clip_g.path.empty() || m_config.t5xxl.path.empty()) {
        throw ConfigException(kTag, "text encoder config path is empty");
    }
    validate_required_tensors(*m_loader);
}

auto SDTextEncoderRunner::encode_prompt(const SDTokenIdPack &tokens) const -> SDTextEncoderEmbeddings {
    // chunk 数由三条 token 分支共同决定。
    const size_t chunk_count = detail::compute_sd3_chunk_count(tokens.clip_l.size(), tokens.clip_g.size(), tokens.t5.size());
    if (chunk_count == 0) {
        throw ConfigException(kTag, "token ids are empty");
    }

    if (tokens.clip_l.empty() || tokens.clip_g.empty() || tokens.t5.empty()) {
        throw ConfigException(kTag, "token ids are empty");
    }

    const int n_threads = m_n_threads > 0 ? m_n_threads : default_n_threads();
    const ClipModelSpec clip_l = detect_clip_spec(*m_loader, "text_encoders.clip_l.transformer.", false);
    const ClipModelSpec clip_g = detect_clip_spec(*m_loader, "text_encoders.clip_g.transformer.", true);
    const T5ModelSpec t5 = detect_t5_spec(*m_loader, "text_encoders.t5xxl.transformer.");
    validate_sd3_shape_contract(clip_l, clip_g, t5);

    int clip_skip = m_config.clip_skip;
    if (clip_skip <= 0) {
        clip_skip = 2;
    }

    SDTextEncoderEmbeddings out;
    out.crossattn_dim = kCrossattnDim;
    out.crossattn_tokens = detail::compute_sd3_crossattn_tokens(chunk_count);
    out.vector_dim = kVectorDim;
    out.crossattn.assign(out.crossattn_tokens * out.crossattn_dim, 0.0f);
    out.vector.assign(out.vector_dim, 0.0f);

    std::vector<float> pooled_l;
    std::vector<float> pooled_g;

    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        POWERSERVE_LOG_INFO("[{}] encode chunk {}/{}", kTag, chunk + 1, chunk_count);
        const std::vector<int32_t> clip_l_chunk = extract_chunk_tokens(tokens.clip_l, chunk);
        const std::vector<int32_t> clip_g_chunk = extract_chunk_tokens(tokens.clip_g, chunk);
        const std::vector<int32_t> t5_chunk = extract_chunk_tokens(tokens.t5, chunk);
        const std::vector<float> clip_l_chunk_weights = extract_chunk_weights(tokens.clip_l_weights, chunk);
        const std::vector<float> clip_g_chunk_weights = extract_chunk_weights(tokens.clip_g_weights, chunk);
        const std::vector<float> t5_chunk_weights = extract_chunk_weights(tokens.t5_weights, chunk);

        POWERSERVE_LOG_INFO("[{}] run clip_l hidden", kTag);
        TensorBlobF32 clip_l_hidden = run_clip_hidden(*m_loader, clip_l, clip_l_chunk, clip_skip, false, n_threads);
        apply_token_weights_and_mean_rescale(clip_l_hidden, clip_l_chunk_weights);
        POWERSERVE_LOG_INFO("[{}] run clip_g hidden", kTag);
        TensorBlobF32 clip_g_hidden = run_clip_hidden(*m_loader, clip_g, clip_g_chunk, clip_skip, false, n_threads);
        apply_token_weights_and_mean_rescale(clip_g_hidden, clip_g_chunk_weights);
        POWERSERVE_LOG_INFO("[{}] run t5 hidden", kTag);
        TensorBlobF32 t5_hidden = run_t5_hidden(*m_loader, t5, t5_chunk, n_threads);
        apply_token_weights_and_mean_rescale(t5_hidden, t5_chunk_weights);

        POWERSERVE_ASSERT_CONFIG(
            clip_l_hidden.ne[0] == static_cast<int64_t>(kClipLDim) && clip_l_hidden.ne[1] == static_cast<int64_t>(detail::kSD3TextChunkLen),
            kTag,
            "clip_l hidden shape mismatch: [{}, {}]",
            clip_l_hidden.ne[0],
            clip_l_hidden.ne[1]
        );
        POWERSERVE_ASSERT_CONFIG(
            clip_g_hidden.ne[0] == static_cast<int64_t>(kClipGDim) && clip_g_hidden.ne[1] == static_cast<int64_t>(detail::kSD3TextChunkLen),
            kTag,
            "clip_g hidden shape mismatch: [{}, {}]",
            clip_g_hidden.ne[0],
            clip_g_hidden.ne[1]
        );
        POWERSERVE_ASSERT_CONFIG(
            t5_hidden.ne[0] == static_cast<int64_t>(kT5Dim) && t5_hidden.ne[1] == static_cast<int64_t>(detail::kSD3TextChunkLen),
            kTag,
            "t5 hidden shape mismatch: [{}, {}]",
            t5_hidden.ne[0],
            t5_hidden.ne[1]
        );

        if (chunk == 0) {
            // pooled 只取第一块（与 sd.cpp 一致）。
            POWERSERVE_LOG_INFO("[{}] run clip pooled", kTag);
            pooled_l = run_clip_pooled(*m_loader, clip_l, clip_l_chunk, 49407, n_threads);
            pooled_g = run_clip_pooled(*m_loader, clip_g, clip_g_chunk, 49407, n_threads);
        }

        const size_t row_offset = chunk * detail::kSD3CrossattnTokensPerChunk;

        for (size_t t = 0; t < detail::kSD3TextChunkLen; ++t) {
            // crossattn 前 77 行写入 clip_l + clip_g（其余维度清 0）。
            float *dst_clip = out.crossattn.data() + (row_offset + t) * out.crossattn_dim;
            const float *src_l = clip_l_hidden.data.data() + t * kClipLDim;
            const float *src_g = clip_g_hidden.data.data() + t * kClipGDim;
            std::memset(dst_clip, 0, out.crossattn_dim * sizeof(float));
            std::memcpy(dst_clip, src_l, kClipLDim * sizeof(float));
            std::memcpy(dst_clip + kClipLDim, src_g, kClipGDim * sizeof(float));

            // crossattn 后 77 行写入 t5。
            float *dst_t5 = out.crossattn.data() + (row_offset + detail::kSD3TextChunkLen + t) * out.crossattn_dim;
            const float *src_t5 = t5_hidden.data.data() + t * kT5Dim;
            std::memcpy(dst_t5, src_t5, kT5Dim * sizeof(float));
        }
    }

    POWERSERVE_ASSERT_CONFIG(pooled_l.size() == kClipLDim, kTag, "clip_l pooled size mismatch: {}", pooled_l.size());
    POWERSERVE_ASSERT_CONFIG(pooled_g.size() == kClipGDim, kTag, "clip_g pooled size mismatch: {}", pooled_g.size());

    std::memcpy(out.vector.data(), pooled_l.data(), kClipLDim * sizeof(float));
    std::memcpy(out.vector.data() + kClipLDim, pooled_g.data(), kClipGDim * sizeof(float));

    return out;
}

auto SDTextEncoderRunner::encode_prompt_pair(const SDPromptTokenization &tokenized) const -> SDPromptPairEmbeddings {
    // 正负提示词分别独立编码。
    SDPromptPairEmbeddings out;
    out.prompt = encode_prompt(tokenized.prompt);
    out.negative_prompt = encode_prompt(tokenized.negative_prompt);
    return out;
}

} // namespace powerserve
