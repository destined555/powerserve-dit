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

#pragma once

#include "core/config-sd.hpp"
#include "core/typedefs.hpp"
#include "ggml.h"
#if __has_include("gguf.h")
#include "gguf.h"
#endif

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace powerserve {

// VAE 解码输出描述：
// - width/height/channels/batch 对应输出张量形状；
// - data 为 F32，当前实现输出布局为 [W, H, C, B]。
struct SD35VAEDecodeResult {
    int64_t width = 0;
    int64_t height = 0;
    int64_t channels = 0;
    int64_t batch = 0;
    std::vector<float> data;
};

// SD3.5 VAE 解码器：
// - 负责从 gguf 加载 VAE 权重；
// - 支持从随机 latent 或外部给定 latent 执行 decode；
// - 仅承担 VAE 阶段，不包含扩散采样流程。
class SD35VAE : Noncopyable {
public:
    // SD3.5 VAE 的空间缩放倍数：latent 分辨率 * 8 = 输出图像分辨率。
    static constexpr int64_t kVaeScaleFactor = 8;
    // SD3.5 latent 通道数固定为 16。
    static constexpr int64_t kLatentChannels = 16;

    SD35VAE() = default;
    ~SD35VAE();

    // 从配置加载 VAE。若已加载旧模型会先 clear()。
    // work_folder 与 config.vae.path 会组合成最终权重路径。
    void load(const Path &work_folder, const VAEConfig &config);

    // 释放所有与模型关联的 ggml/gguf 资源。
    void clear();

    auto is_loaded() const -> bool;
    auto model_path() const -> const Path &;

    // 生成高斯随机 latent 并解码。主要用于快速自检/冒烟。
    auto decode_random_latent(int64_t latent_w, int64_t latent_h, int64_t batch, int64_t seed, int n_threads = 0) const
        -> SD35VAEDecodeResult;

    // 解码外部输入 latent：
    // - latent_w/latent_h 为 latent 空间尺寸（不是最终图像尺寸）；
    // - latent.size 必须等于 latent_w * latent_h * 16 * batch；
    // - n_threads <= 0 时会自动使用硬件线程数。
    auto decode_latent(std::vector<float> latent, int64_t latent_w, int64_t latent_h, int64_t batch, int n_threads = 0) const
        -> SD35VAEDecodeResult;

private:
    // 一次加载后的资源句柄集合。
    // - ggml_ctx: gguf 载入时创建的主上下文（存放原始 tensor）；
    // - aux_ctx: 加载阶段额外创建的上下文（存放归一化后的 conv 权重副本）；
    // - gguf_ctx: gguf 文件上下文，管理元信息与生命周期。
    struct LoadedModel {
        Path path;
        ggml_context *ggml_ctx = nullptr;
        ggml_context *aux_ctx = nullptr;
        gguf_context *gguf_ctx = nullptr;
    };

    LoadedModel m_loaded;
    // 名称到 tensor 的索引表，供 decoder_forward 按名字检索权重。
    std::unordered_map<std::string, ggml_tensor *> m_tensors;
};

} // namespace powerserve
