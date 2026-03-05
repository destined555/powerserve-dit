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

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace powerserve {

namespace detail {

// 构造 text encoder tensor 的全名 key，并避免前缀重复拼接。
// 例如：
// prefix = "text_encoders.clip_l.transformer."
// tensor_name = "text_model.embeddings.token_embedding.weight"
// => "text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight"
auto build_text_encoder_tensor_key(std::string_view prefix, std::string_view tensor_name) -> std::string;

} // namespace detail

struct SDTextEncoderLoadSummary {
    // 逻辑组件名：clip_l / clip_g / t5xxl。
    std::string name;
    // 最终解析后的模型路径（绝对或相对展开后）。
    Path path;
    // 建立 tensor 索引时使用的名字前缀。
    std::string prefix;
    // 该组件 gguf 内的 tensor 数量。
    size_t n_tensors = 0;
};

class SDTextEncoderModelLoader : Noncopyable {
public:
    SDTextEncoderModelLoader() = default;
    ~SDTextEncoderModelLoader();

    // 从 TextEncodeConfig 加载全部 text encoder，并重建内部状态。
    void load(const Path &work_folder, const TextEncodeConfig &config);
    // 释放已加载的 gguf/ggml 资源并清空索引。
    void clear();

    // 按“完整前缀名”查询 tensor。不存在时返回 nullptr。
    auto get_tensor(std::string_view tensor_name) const -> ggml_tensor *;
    // 判断“完整前缀名”是否存在。
    auto has_tensor(std::string_view tensor_name) const -> bool;
    // 返回只读加载摘要，供日志或检查使用。
    auto summaries() const -> const std::vector<SDTextEncoderLoadSummary> &;
    // 当前所有 text encoder 合计索引到的 tensor 数。
    auto tensor_count() const -> size_t;
    // 统计 key 以给定前缀开头的 tensor 数。
    auto tensor_count_with_prefix(std::string_view tensor_prefix) const -> size_t;

private:
    struct LoadedPart {
        std::string name;
        std::string prefix;
        Path path;
        ggml_context *ggml_ctx = nullptr;
        gguf_context *gguf_ctx = nullptr;
    };

    std::vector<LoadedPart> m_loaded_parts;
    std::vector<SDTextEncoderLoadSummary> m_summaries;
    std::unordered_map<std::string, ggml_tensor *> m_tensors;

    // 将组件路径相对 work_folder 展开为实际路径。
    static auto resolve_component_path(const Path &work_folder, const ComponentConfig &component) -> Path;
    // 加载单个组件 gguf，并将其加入 loader 状态。
    void load_component(
        const Path &work_folder, const ComponentConfig &component, std::string name, std::string prefix
    );
    // 将一个组件内全部 tensor 以“前缀名”写入 m_tensors 索引。
    void index_component_tensors(const LoadedPart &part);
};

} // namespace powerserve
