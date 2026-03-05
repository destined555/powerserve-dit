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

#include "vae/sd35_vae.hpp"

#include "core/exception.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <thread>

#if __has_include("ggml-cpu.h")
#include "ggml-cpu.h"
#endif
#if __has_include("ggml-alloc.h")
#include "ggml-alloc.h"
#endif

namespace powerserve {

namespace {

// 日志/异常统一标签，便于在混合模块日志中快速定位来源。
constexpr const char *kTag = "SD35VAE";
// 当前仅支持 gguf 格式 VAE。
constexpr std::string_view kGgufFormat = "gguf";
// SD3.5 latent 到 VAE decoder 输入的线性变换参数：
// decoder_in = latent / scale + shift
// 该处理与 sd.cpp 侧保持一致，用于对齐解码输入分布。
constexpr float kSD35LatentScale = 1.5305f;
constexpr float kSD35LatentShift = 0.0609f;

// 一个极小的 RAII 封装，确保 ggml_context 在异常路径上也能正确释放。
class ScopedGGMLContext : Noncopyable {
public:
    explicit ScopedGGMLContext(size_t mem_size, bool no_alloc = false) {
        ggml_init_params params{};
        params.mem_size = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc = no_alloc;
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

// 当调用方不指定线程数时，默认使用硬件并发数。
auto default_n_threads() -> int {
    const auto hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

// 解析组件路径：
// - 若配置里是相对路径，则相对 work_folder 展开；
// - 若是绝对路径，则直接使用。
auto resolve_component_path(const Path &work_folder, const ComponentConfig &component) -> Path {
    Path path(component.path);
    if (path.is_relative()) {
        path = work_folder / path;
    }
    return path;
}

// 必须存在的 tensor 查询工具。
// 找不到时直接抛异常，避免后续出现更隐晦的空指针错误。
auto require_tensor(const std::unordered_map<std::string, ggml_tensor *> &tensors, std::string_view name)
    -> ggml_tensor * {
    auto it = tensors.find(std::string(name));
    if (it == tensors.end()) {
        throw ConfigException(kTag, fmt::format("required tensor not found: {}", name));
    }
    return it->second;
}

// 可选 tensor 查询工具。
// 常用于 bias、shortcut、upsample 等“可能存在”的权重。
auto find_tensor(const std::unordered_map<std::string, ggml_tensor *> &tensors, std::string_view name)
    -> ggml_tensor * {
    auto it = tensors.find(std::string(name));
    if (it == tensors.end()) {
        return nullptr;
    }
    return it->second;
}

// 将任意输入 tensor 统一为 F32。
// 这里用于数值稳定和与 sd.cpp 路径对齐（例如 norm/softmax 前后）。
auto cast_f32(ggml_context *ctx, ggml_tensor *x) -> ggml_tensor * {
    if (x->type == GGML_TYPE_F32) {
        return x;
    }
    return ggml_cast(ctx, x, GGML_TYPE_F32);
}

// 仅用于错误日志的人类可读形状打印。
auto shape_str(const ggml_tensor *t) -> std::string {
    return fmt::format("[{}, {}, {}, {}]", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
}

// 后缀判断小工具（兼容 string_view，零额外分配）。
auto ends_with(std::string_view s, std::string_view suffix) -> bool {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

// 判定某个权重是否需要在加载阶段归一化为 F16。
// 约束策略：
// - 仅处理 4D tensor（卷积核）；
// - 名称以 ".weight" 结尾；
// - 当前类型不是 F16。
//
// 这么做的目标是让 conv2d 走与 sd.cpp 更一致的数据类型路径，
// 同时规避某些量化类型在 CPU 后端的算子兼容问题。
auto should_normalize_conv_weight_to_f16(std::string_view name, const ggml_tensor *tensor) -> bool {
    if (tensor == nullptr) {
        return false;
    }
    if (ggml_n_dims(tensor) != 4) {
        return false;
    }
    if (!ends_with(name, ".weight")) {
        return false;
    }
    return tensor->type != GGML_TYPE_F16;
}

// 带形状检查的逐元素加法：
// b 允许按 channel/batch 维广播（等于 1 时视为广播）。
// 这比直接 ggml_add 更早暴露维度问题。
auto add_checked(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b, std::string_view label) -> ggml_tensor * {
    const bool ch_ok = (b->ne[2] == 1 || b->ne[2] == a->ne[2]);
    const bool batch_ok = (b->ne[3] == 1 || b->ne[3] == a->ne[3]);
    if (!ch_ok || !batch_ok) {
        throw ConfigException(
            kTag, fmt::format("{}: add shape mismatch a={} b={}", label, shape_str(a), shape_str(b))
        );
    }
    return ggml_add(ctx, a, b);
}

// Conv2D 前向封装：
// - 负责必要的权重形状重解释（1x1 卷积特殊导出格式）；
// - 对输入通道一致性做显式检查；
// - 输出统一 cast 到 F32，并在存在 bias 时执行加法。
auto conv2d_forward(
    ggml_context *ctx,
    ggml_tensor *x,
    ggml_tensor *w,
    ggml_tensor *b,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    std::string_view label
) -> ggml_tensor * {
    // 某些导出流程会把 1x1 卷积存成 [out, in, 1, 1]，
    // 而 ggml_conv_2d 期望布局是 [kw, kh, in, out]。
    // 这里通过 reshape 仅改变视图，不重排底层数据。
    if (w->ne[2] == 1 && w->ne[3] == 1 && w->ne[0] == x->ne[2]) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if (w->ne[2] != x->ne[2]) {
        throw ConfigException(
            kTag,
            fmt::format(
                "{}: conv input channel mismatch x={} w={} (w.ne[2]={} != x.ne[2]={})",
                label,
                shape_str(x),
                shape_str(w),
                w->ne[2],
                x->ne[2]
            )
        );
    }

    ggml_tensor *out = ggml_conv_2d(ctx, w, x, stride_w, stride_h, pad_w, pad_h, 1, 1);
    out = cast_f32(ctx, out);
    if (b != nullptr) {
        ggml_tensor *bias = ggml_reshape_4d(ctx, cast_f32(ctx, b), 1, 1, b->ne[0], 1);
        out = add_checked(ctx, out, bias, label);
    }
    return out;
}

// GroupNorm32 前向：
// - 先将输入和参数统一为 F32；
// - 先执行 group_norm，再做仿射变换 x * weight + bias。
auto group_norm32_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) -> ggml_tensor * {
    x = cast_f32(ctx, x);
    x = ggml_group_norm(ctx, x, 32, 1e-6f);

    ggml_tensor *weight = ggml_reshape_4d(ctx, cast_f32(ctx, w), 1, 1, w->ne[0], 1);
    ggml_tensor *bias = ggml_reshape_4d(ctx, cast_f32(ctx, b), 1, 1, b->ne[0], 1);

    x = ggml_mul(ctx, x, weight);
    x = add_checked(ctx, x, bias, "group_norm32");
    return x;
}

// Self-Attention 前向（VAE 中间块使用）：
// - 输入 q/k/v 形状经过上游整理后进入；
// - 兼容 singleton-head 的特殊路径，确保与 sd.cpp 行为一致；
// - 注意力分数在 softmax 前按 d_head 缩放。
auto attention_forward(
    ggml_context *ctx, ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, int64_t n_heads, ggml_tensor *mask = nullptr
) -> ggml_tensor * {
    // 某些 ggml 算子要求输入连续内存；这里在必要时 materialize contiguous view。
    const auto cont_if_needed = [&](ggml_tensor *t) -> ggml_tensor * {
        return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    };

    const int64_t lq = q->ne[1];
    const int64_t lk = k->ne[1];
    const int64_t d_head = v->ne[0];
    const int64_t n = v->ne[3];
    const int64_t n_kv_heads = k->ne[2] / n;

    q = cast_f32(ctx, q);
    k = cast_f32(ctx, k);
    v = cast_f32(ctx, v);

    if (n_kv_heads == 1 && n == 1) {
        // 与 sd.cpp 对齐：当 kv/head 退化为单头单 batch 时，直接视图 reshape，
        // 不做 permute，以避免数值路径和内存访问方式偏离。
        v = ggml_reshape_3d(ctx, v, lk, d_head, 1);
    } else {
        // 常规多头路径：reshape -> permute -> contiguous -> 3D 视图
        v = ggml_reshape_4d(ctx, v, d_head, n_kv_heads, lk, n);
        v = cont_if_needed(ggml_permute(ctx, v, 1, 2, 0, 3));
        v = ggml_reshape_3d(ctx, v, lk, d_head, n_kv_heads * n);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

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
    kqv = cont_if_needed(kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_heads, lq, n);
    return kqv;
}

auto conv_named(
    ggml_context *ctx,
    const std::unordered_map<std::string, ggml_tensor *> &tensors,
    ggml_tensor *x,
    std::string_view weight_name,
    std::string_view bias_name,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) -> ggml_tensor * {
    // 通过统一命名规则查权重，避免在主流程里散落字符串访问逻辑。
    ggml_tensor *w = require_tensor(tensors, weight_name);
    ggml_tensor *b = find_tensor(tensors, bias_name);
    return conv2d_forward(ctx, x, w, b, stride_h, stride_w, pad_h, pad_w, weight_name);
}

auto linear_forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b, std::string_view label) -> ggml_tensor * {
    if (ggml_n_dims(w) == 4) {
        // 兼容“1x1 conv 形式导出的线性层权重”：
        // [1, 1, in, out] 仅在视图上重解释为 [in, out]。
        if (w->ne[0] != 1 || w->ne[1] != 1) {
            throw ConfigException(
                kTag,
                fmt::format("{}: unsupported 4D linear weight shape {}", label, shape_str(w))
            );
        }
        w = ggml_reshape_2d(ctx, w, w->ne[2], w->ne[3]);
    }

    if (ggml_n_dims(w) != 2) {
        throw ConfigException(
            kTag,
            fmt::format("{}: linear weight must be 2D/4D, got {}", label, shape_str(w))
        );
    }

    ggml_tensor *out = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        ggml_tensor *bias = cast_f32(ctx, b);
        out = ggml_add_inplace(ctx, out, bias);
    }
    return out;
}

// 通过权重名字执行线性层，统一 require/find 逻辑。
auto linear_named(
    ggml_context *ctx,
    const std::unordered_map<std::string, ggml_tensor *> &tensors,
    ggml_tensor *x,
    std::string_view weight_name,
    std::string_view bias_name
) -> ggml_tensor * {
    ggml_tensor *w = require_tensor(tensors, weight_name);
    ggml_tensor *b = find_tensor(tensors, bias_name);
    return linear_forward(ctx, x, w, b, weight_name);
}

// ResNet block（decoder 中最常见的基础结构）：
// norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 (+ optional shortcut) -> residual add
auto resnet_block_forward(
    ggml_context *ctx, const std::unordered_map<std::string, ggml_tensor *> &tensors, ggml_tensor *x, std::string_view prefix
) -> ggml_tensor * {
    const std::string p(prefix);
    ggml_tensor *h = group_norm32_forward(
        ctx, x, require_tensor(tensors, p + ".norm1.weight"), require_tensor(tensors, p + ".norm1.bias")
    );
    h = ggml_silu_inplace(ctx, h);
    h = conv_named(ctx, tensors, h, p + ".conv1.weight", p + ".conv1.bias", 1, 1, 1, 1);

    h = group_norm32_forward(
        ctx, h, require_tensor(tensors, p + ".norm2.weight"), require_tensor(tensors, p + ".norm2.bias")
    );
    h = ggml_silu_inplace(ctx, h);
    h = conv_named(ctx, tensors, h, p + ".conv2.weight", p + ".conv2.bias", 1, 1, 1, 1);

    if (find_tensor(tensors, p + ".conv_shortcut.weight") != nullptr) {
        x = conv_named(ctx, tensors, x, p + ".conv_shortcut.weight", p + ".conv_shortcut.bias", 1, 1, 0, 0);
    }

    return add_checked(ctx, h, x, p + ".residual");
}

// Attention block（SD3.5 decoder mid/up block 里的注意力子块）：
// 支持两种权重形态：
// 1) 线性层形式（2D 权重）；
// 2) 1x1 卷积形式（4D 权重）。
auto attention_block_forward(
    ggml_context *ctx, const std::unordered_map<std::string, ggml_tensor *> &tensors, ggml_tensor *x, std::string_view prefix
) -> ggml_tensor * {
    const std::string p(prefix);
    ggml_tensor *h = group_norm32_forward(
        ctx,
        x,
        require_tensor(tensors, p + ".group_norm.weight"),
        require_tensor(tensors, p + ".group_norm.bias")
    );

    const int64_t n = h->ne[3];
    const int64_t c = h->ne[2];
    const int64_t h_size = h->ne[1];
    const int64_t w_size = h->ne[0];

    // 通过 to_out 权重维度来判断该块是 linear 版本还是 conv 版本。
    ggml_tensor *to_out_w = require_tensor(tensors, p + ".to_out.0.weight");
    const bool use_linear = ggml_n_dims(to_out_w) == 2;

    ggml_tensor *q = nullptr;
    ggml_tensor *k = nullptr;
    ggml_tensor *v = nullptr;

    if (use_linear) {
        // [W,H,C,B] -> [C, HW, B]，便于 linear(qkv) + attention 计算
        h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 2, 0, 3));
        h = ggml_reshape_3d(ctx, h, c, h_size * w_size, n);

        q = linear_named(ctx, tensors, h, p + ".to_q.weight", p + ".to_q.bias");
        k = linear_named(ctx, tensors, h, p + ".to_k.weight", p + ".to_k.bias");
        v = linear_named(ctx, tensors, h, p + ".to_v.weight", p + ".to_v.bias");

        h = attention_forward(ctx, q, k, v, 1);
        h = linear_named(ctx, tensors, h, p + ".to_out.0.weight", p + ".to_out.0.bias");

        h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3));
        h = ggml_reshape_4d(ctx, h, w_size, h_size, c, n);
    } else {
        // conv 版本先在空间布局下做 q/k/v，再转到 [C, HW, B] 计算 attention。
        q = conv_named(ctx, tensors, h, p + ".to_q.weight", p + ".to_q.bias", 1, 1, 0, 0);
        k = conv_named(ctx, tensors, h, p + ".to_k.weight", p + ".to_k.bias", 1, 1, 0, 0);
        v = conv_named(ctx, tensors, h, p + ".to_v.weight", p + ".to_v.bias", 1, 1, 0, 0);

        q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));
        q = ggml_reshape_3d(ctx, q, c, h_size * w_size, n);

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));
        k = ggml_reshape_3d(ctx, k, c, h_size * w_size, n);

        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        v = ggml_reshape_3d(ctx, v, c, h_size * w_size, n);

        h = attention_forward(ctx, q, k, v, 1);
        h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3));
        h = ggml_reshape_4d(ctx, h, w_size, h_size, c, n);

        h = conv_named(ctx, tensors, h, p + ".to_out.0.weight", p + ".to_out.0.bias", 1, 1, 0, 0);
    }

    return add_checked(ctx, h, x, p + ".residual");
}

// 上采样模块：nearest x2 + 3x3 conv。
auto upsample_block_forward(
    ggml_context *ctx, const std::unordered_map<std::string, ggml_tensor *> &tensors, ggml_tensor *x, std::string_view prefix
) -> ggml_tensor * {
    const std::string p(prefix);
    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
    x = conv_named(ctx, tensors, x, p + ".conv.weight", p + ".conv.bias", 1, 1, 1, 1);
    return x;
}

// SD3.5 VAE Decoder 主干：
// 结构顺序与原始模型一致：
// conv_in -> mid(resnet-attn-resnet) -> 4 个 up block -> norm/silu/conv_out
auto decoder_forward(
    ggml_context *ctx,
    const std::unordered_map<std::string, ggml_tensor *> &tensors,
    ggml_tensor *z
) -> ggml_tensor * {
    ggml_tensor *x = conv_named(ctx, tensors, z, "decoder.conv_in.weight", "decoder.conv_in.bias", 1, 1, 1, 1);

    x = resnet_block_forward(ctx, tensors, x, "decoder.mid_block.resnets.0");
    x = attention_block_forward(ctx, tensors, x, "decoder.mid_block.attentions.0");
    x = resnet_block_forward(ctx, tensors, x, "decoder.mid_block.resnets.1");

    for (int up_idx = 0; up_idx < 4; ++up_idx) {
        for (int res_idx = 0; res_idx < 3; ++res_idx) {
            const std::string prefix = fmt::format("decoder.up_blocks.{}.resnets.{}", up_idx, res_idx);
            x = resnet_block_forward(ctx, tensors, x, prefix);
        }

        const std::string upsample_weight = fmt::format("decoder.up_blocks.{}.upsamplers.0.conv.weight", up_idx);
        if (find_tensor(tensors, upsample_weight) != nullptr) {
            const std::string upsample_prefix = fmt::format("decoder.up_blocks.{}.upsamplers.0", up_idx);
            x = upsample_block_forward(ctx, tensors, x, upsample_prefix);
        }
    }

    x = group_norm32_forward(
        ctx, x, require_tensor(tensors, "decoder.conv_norm_out.weight"), require_tensor(tensors, "decoder.conv_norm_out.bias")
    );
    x = ggml_silu_inplace(ctx, x);
    x = conv_named(ctx, tensors, x, "decoder.conv_out.weight", "decoder.conv_out.bias", 1, 1, 1, 1);

    return x;
}

// 将输入 latent 变换到 decoder 期望分布。
// 注：命名沿用历史（out），语义上是 decode 前处理。
void process_latent_out_inplace(std::vector<float> &latent) {
    for (float &value : latent) {
        value = (value / kSD35LatentScale) + kSD35LatentShift;
    }
}

// decoder 输出从 [-1, 1] 映射到 [0, 1]，并 clamp。
auto normalize_decoder_output(std::vector<float> image) -> std::vector<float> {
    for (float &value : image) {
        value = (value + 1.0f) * 0.5f;
        value = std::clamp(value, 0.0f, 1.0f);
    }
    return image;
}

// 选择图构建上下文的元数据内存大小：
// - 当前 decode 图在 no_alloc=true 下只占用图与 tensor 描述，不含真实算子缓冲；
// - 该估算用于避免过小导致 graph 构建失败，也避免过大造成浪费。
auto select_graph_mem_size(int64_t latent_w, int64_t latent_h, int64_t batch) -> size_t {
    const size_t min_mem = 128ull * 1024 * 1024;
    const size_t max_mem = 1024ull * 1024 * 1024;

    const size_t latent_area = static_cast<size_t>(latent_w * latent_h * batch);
    // no_alloc 图上下文可使用较保守估计值，不需要覆盖实际激活内存。
    const size_t estimate = latent_area * 512ull * sizeof(float) * 16ull;
    return std::clamp(estimate, min_mem, max_mem);
}

} // namespace

SD35VAE::~SD35VAE() {
    clear();
}

void SD35VAE::load(const Path &work_folder, const VAEConfig &config) {
    // 允许重复 load：先清理旧资源，避免泄漏和状态污染。
    clear();

    // 明确限制格式，尽早给出可读错误。
    if (!config.vae.format.empty() && config.vae.format != kGgufFormat) {
        throw ConfigException(
            kTag, fmt::format("unsupported vae format '{}', expected '{}'", config.vae.format, kGgufFormat)
        );
    }

    POWERSERVE_ASSERT_CONFIG(!config.vae.path.empty(), kTag, "vae component path is empty");

    const Path vae_path = resolve_component_path(work_folder, config.vae);
    POWERSERVE_ASSERT_CONFIG(std::filesystem::exists(vae_path), kTag, "vae path does not exist: {}", vae_path);

    // gguf_init_from_file 会同时构造 gguf_ctx 和 ggml_ctx：
    // - gguf_ctx 负责文件级元信息；
    // - ggml_ctx 持有 tensor 对象与数据。
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params{
        .no_alloc = false,
        .ctx = &ggml_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(vae_path.c_str(), params);

    if (gguf_ctx == nullptr || ggml_ctx == nullptr) {
        if (gguf_ctx != nullptr) {
            gguf_free(gguf_ctx);
        }
        throw ConfigException(kTag, fmt::format("failed to load VAE gguf from {}", vae_path));
    }

    // 建立 name -> tensor 索引，后续 forward 统一通过名字查权重。
    std::unordered_map<std::string, ggml_tensor *> tensors;
    for (ggml_tensor *tensor = ggml_get_first_tensor(ggml_ctx); tensor != nullptr;
         tensor = ggml_get_next_tensor(ggml_ctx, tensor)) {
        const char *name = ggml_get_name(tensor);
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        tensors.emplace(name, tensor);
    }

    // 统计需要归一化为 F16 的卷积核，并预估辅助上下文内存。
    // 目的：
    // 1) 对齐 sd.cpp 卷积路径；
    // 2) 避免某些非 F16/F32 权重在 CPU 卷积实现上触发兼容问题。
    size_t normalized_conv_weight_count = 0;
    size_t aux_mem_size = 0;
    for (const auto &kv : tensors) {
        const std::string &name = kv.first;
        const ggml_tensor *tensor = kv.second;
        if (!should_normalize_conv_weight_to_f16(name, tensor)) {
            continue;
        }
        normalized_conv_weight_count += 1;
        aux_mem_size += ggml_tensor_overhead();
        aux_mem_size += static_cast<size_t>(tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]) * sizeof(ggml_fp16_t);
    }

    ggml_context *aux_ctx = nullptr;
    if (normalized_conv_weight_count > 0) {
        // 给辅助上下文留少量冗余，避免边界情况下 tensor 元数据不足。
        ggml_init_params aux_params{};
        aux_params.mem_size = std::max(aux_mem_size + ggml_tensor_overhead() * 8, size_t(16ull * 1024 * 1024));
        aux_params.mem_buffer = nullptr;
        aux_params.no_alloc = false;
        aux_ctx = ggml_init(aux_params);
        if (aux_ctx == nullptr) {
            gguf_free(gguf_ctx);
            throw ConfigException(kTag, "failed to allocate aux context for conv weight normalization");
        }

        for (auto &kv : tensors) {
            const std::string &name = kv.first;
            ggml_tensor *src = kv.second;
            if (!should_normalize_conv_weight_to_f16(name, src)) {
                continue;
            }

            // 在 aux_ctx 中创建 F16 副本并替换索引：
            // m_tensors 最终将指向该副本，原始 tensor 仍由 gguf/ggml_ctx 管理。
            ggml_tensor *dst = ggml_new_tensor_4d(aux_ctx, GGML_TYPE_F16, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
            ggml_set_name(dst, name.c_str());

            // 使用标量读取 + fp16 写入，确保对任意源类型都可正确转换。
            const size_t n = static_cast<size_t>(ggml_nelements(src));
            auto *dst_data = static_cast<ggml_fp16_t *>(dst->data);
            for (size_t i = 0; i < n; ++i) {
                dst_data[i] = ggml_fp32_to_fp16(ggml_get_f32_1d(src, static_cast<int>(i)));
            }
            kv.second = dst;
        }
    }

    POWERSERVE_ASSERT_CONFIG(
        tensors.find("decoder.conv_in.weight") != tensors.end(), kTag, "decoder tensors not found in {}", vae_path
    );

    // 仅在所有步骤成功后再落盘到成员变量，避免半初始化状态。
    m_loaded.path = vae_path;
    m_loaded.ggml_ctx = ggml_ctx;
    m_loaded.aux_ctx = aux_ctx;
    m_loaded.gguf_ctx = gguf_ctx;
    m_tensors = std::move(tensors);

    POWERSERVE_LOG_INFO("[{}] loaded SD3.5 VAE from '{}' ({} tensors)", kTag, m_loaded.path.string(), m_tensors.size());

}

void SD35VAE::clear() {
    // 先清空索引，避免释放后悬挂访问。
    m_tensors.clear();
    if (m_loaded.aux_ctx != nullptr) {
        ggml_free(m_loaded.aux_ctx);
    }
    if (m_loaded.gguf_ctx != nullptr) {
        gguf_free(m_loaded.gguf_ctx);
    }
    m_loaded = LoadedModel{};
}

auto SD35VAE::is_loaded() const -> bool {
    return m_loaded.gguf_ctx != nullptr && m_loaded.ggml_ctx != nullptr;
}

auto SD35VAE::model_path() const -> const Path & {
    return m_loaded.path;
}

auto SD35VAE::decode_random_latent(int64_t latent_w, int64_t latent_h, int64_t batch, int64_t seed, int n_threads) const
    -> SD35VAEDecodeResult {
    POWERSERVE_ASSERT_CONFIG(latent_w > 0 && latent_h > 0 && batch > 0, kTag, "invalid latent shape");

    // latent 布局与 decode_latent 保持一致：[W, H, C=16, B]。
    const size_t element_count = static_cast<size_t>(latent_w * latent_h * kLatentChannels * batch);
    std::vector<float> latent(element_count, 0.0f);

    // 与扩散初始化一致的标准正态采样。
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (float &value : latent) {
        value = norm_dist(rng);
    }

    return decode_latent(std::move(latent), latent_w, latent_h, batch, n_threads);
}

auto SD35VAE::decode_latent(std::vector<float> latent, int64_t latent_w, int64_t latent_h, int64_t batch, int n_threads) const
    -> SD35VAEDecodeResult {
    POWERSERVE_ASSERT_CONFIG(is_loaded(), kTag, "decode requested before VAE loading");
    POWERSERVE_ASSERT_CONFIG(latent_w > 0 && latent_h > 0 && batch > 0, kTag, "invalid latent shape");

    // 严格校验输入长度，防止形状不一致导致越界或结果错位。
    const size_t expect = static_cast<size_t>(latent_w * latent_h * kLatentChannels * batch);
    POWERSERVE_ASSERT_CONFIG(
        latent.size() == expect,
        kTag,
        "latent size mismatch: got {}, expect {} (w={}, h={}, c={}, b={})",
        latent.size(),
        expect,
        latent_w,
        latent_h,
        kLatentChannels,
        batch
    );

    // 预处理 latent 到 decoder 输入域。
    process_latent_out_inplace(latent);

    // 图上下文仅存放 graph/tensor 描述，不直接分配算子数据缓冲。
    // 实际缓冲由 backend + gallocr 管理。
    const size_t mem_size = select_graph_mem_size(latent_w, latent_h, batch);
    ScopedGGMLContext scoped_ctx(mem_size, true);
    ggml_context *ctx = scoped_ctx.get();

    // 构造输入 tensor（F32）并标记为 graph 输入。
    ggml_tensor *latent_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, latent_w, latent_h, kLatentChannels, batch);
    ggml_set_input(latent_tensor);

    // 拼装 decoder 计算图，并将最终输出标记为 graph 输出。
    ggml_tensor *decoded = decoder_forward(ctx, m_tensors, latent_tensor);
    decoded = ggml_cont(ctx, cast_f32(ctx, decoded));
    ggml_set_output(decoded);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, decoded);

    const int effective_threads = n_threads > 0 ? n_threads : default_n_threads();
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        throw ConfigException(kTag, "failed to initialize ggml CPU backend");
    }

#if __has_include("ggml-cpu.h")
    // 若可用则显式设置线程数；否则使用后端默认值。
    ggml_backend_cpu_set_n_threads(backend, effective_threads);
#endif

    // gallocr 负责为 graph 中每个 tensor 规划并分配后端缓冲。
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (allocr == nullptr) {
        ggml_backend_free(backend);
        throw ConfigException(kTag, "failed to create ggml graph allocator");
    }

    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        throw ConfigException(kTag, "failed to allocate ggml graph buffers");
    }

    // 将 host 侧 latent 拷入 graph 输入 tensor。
    ggml_backend_tensor_set(latent_tensor, latent.data(), 0, latent.size() * sizeof(float));

    // 执行图计算，失败时返回明确状态码。
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        throw ConfigException(kTag, fmt::format("VAE decode compute failed (status={})", static_cast<int>(status)));
    }

    SD35VAEDecodeResult result;
    // 直接从输出 tensor 形状回填结果元数据，避免重复推导。
    result.width = decoded->ne[0];
    result.height = decoded->ne[1];
    result.channels = decoded->ne[2];
    result.batch = decoded->ne[3];
    result.data.resize(static_cast<size_t>(ggml_nelements(decoded)));
    ggml_backend_tensor_get(decoded, result.data.data(), 0, result.data.size() * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);

    // 最终输出规范化到 [0,1]，便于后续 PNG 导出和跨实现对比。
    result.data = normalize_decoder_output(std::move(result.data));

    return result;
}

} // namespace powerserve
