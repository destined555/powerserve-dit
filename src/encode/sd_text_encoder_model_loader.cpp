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

#include "encode/sd_text_encoder_model_loader.hpp"

#include "core/exception.hpp"
#include "core/logger.hpp"

#include <filesystem>
#include <utility>

namespace powerserve {

namespace {

// 统一日志标签。
constexpr const char *kTag = "SDTextEncoderModelLoader";
// 当前 text encoder 仅支持 gguf。
constexpr std::string_view kGgufFormat = "gguf";

} // namespace

auto detail::build_text_encoder_tensor_key(std::string_view prefix, std::string_view tensor_name) -> std::string {
    // 无前缀时直接使用原名。
    if (prefix.empty()) {
        return std::string(tensor_name);
    }
    // 已经是完整前缀时不重复拼接。
    if (tensor_name.rfind(prefix, 0) == 0) {
        return std::string(tensor_name);
    }

    std::string full_name;
    full_name.reserve(prefix.size() + tensor_name.size());
    full_name.append(prefix);
    full_name.append(tensor_name);
    return full_name;
}

SDTextEncoderModelLoader::~SDTextEncoderModelLoader() {
    clear();
}

auto SDTextEncoderModelLoader::resolve_component_path(const Path &work_folder, const ComponentConfig &component)
    -> Path {
    Path path(component.path);
    if (path.is_relative()) {
        path = work_folder / path;
    }
    return path;
}

void SDTextEncoderModelLoader::load(const Path &work_folder, const TextEncodeConfig &config) {
    // 允许重复 load，先清空旧状态。
    clear();
    try {
        // 与 SD3 条件分支顺序保持一致：clip_l -> clip_g -> t5xxl。
        load_component(work_folder, config.clip_l, "clip_l", "text_encoders.clip_l.transformer.");
        load_component(work_folder, config.clip_g, "clip_g", "text_encoders.clip_g.transformer.");
        load_component(work_folder, config.t5xxl, "t5xxl", "text_encoders.t5xxl.transformer.");
    } catch (...) {
        // 任一组件加载失败时，回滚到干净状态，避免半加载对象继续使用。
        clear();
        throw;
    }
}

void SDTextEncoderModelLoader::load_component(
    const Path &work_folder, const ComponentConfig &component, std::string name, std::string prefix
) {
    if (component.path.empty()) {
        POWERSERVE_LOG_WARN("[{}] skip {} loading because component path is empty.", kTag, name);
        return;
    }

    if (!component.format.empty() && component.format != kGgufFormat) {
        throw ConfigException(
            kTag, fmt::format("unsupported {} format '{}', expected '{}'", name, component.format, kGgufFormat)
        );
    }

    const Path model_path = resolve_component_path(work_folder, component);
    POWERSERVE_ASSERT_CONFIG(std::filesystem::exists(model_path), kTag, "{} path does not exist: {}", name, model_path);

    // 为该组件构建 gguf + ggml 上下文，并通过 m_loaded_parts 持有生命周期。
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &ggml_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
    POWERSERVE_ASSERT_CONFIG(gguf_ctx != nullptr, kTag, "failed to load {} gguf from {}", name, model_path);
    POWERSERVE_ASSERT_CONFIG(
        ggml_ctx != nullptr, kTag, "failed to initialize ggml context while loading {} from {}", name, model_path
    );

    LoadedPart part;
    part.name = std::move(name);
    part.prefix = std::move(prefix);
    part.path = model_path;
    part.ggml_ctx = ggml_ctx;
    part.gguf_ctx = gguf_ctx;

    try {
        // 先做索引，成功后再真正加入成员容器。
        index_component_tensors(part);
    } catch (...) {
        gguf_free(gguf_ctx);
        throw;
    }

    SDTextEncoderLoadSummary summary;
    summary.name = part.name;
    summary.path = part.path;
    summary.prefix = part.prefix;
    summary.n_tensors = static_cast<size_t>(gguf_get_n_tensors(part.gguf_ctx));

    POWERSERVE_LOG_INFO(
        "[{}] loaded {} from '{}' ({} tensors).", kTag, part.name, part.path.string(), summary.n_tensors
    );

    m_summaries.push_back(std::move(summary));
    m_loaded_parts.push_back(std::move(part));
}

void SDTextEncoderModelLoader::index_component_tensors(const LoadedPart &part) {
    for (ggml_tensor *tensor = ggml_get_first_tensor(part.ggml_ctx); tensor != nullptr;
         tensor = ggml_get_next_tensor(part.ggml_ctx, tensor)) {
        const char *tensor_name = ggml_get_name(tensor);
        if (tensor_name == nullptr || tensor_name[0] == '\0') {
            continue;
        }

        // 使用“完整前缀名”做统一索引，例如：
        // text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight
        const std::string full_tensor_name = detail::build_text_encoder_tensor_key(part.prefix, tensor_name);
        const bool inserted = m_tensors.emplace(full_tensor_name, tensor).second;
        POWERSERVE_ASSERT_CONFIG(
            inserted, kTag, "duplicated tensor key while loading {}: {}", part.name, full_tensor_name
        );
    }
}

void SDTextEncoderModelLoader::clear() {
    // 注意：gguf_free 会连带释放其关联的 ggml 上下文内存。
    for (auto &part : m_loaded_parts) {
        if (part.gguf_ctx != nullptr) {
            gguf_free(part.gguf_ctx);
            part.gguf_ctx = nullptr;
            part.ggml_ctx = nullptr;
        }
    }

    m_loaded_parts.clear();
    m_summaries.clear();
    m_tensors.clear();
}

auto SDTextEncoderModelLoader::get_tensor(std::string_view tensor_name) const -> ggml_tensor * {
    auto it = m_tensors.find(std::string(tensor_name));
    if (it == m_tensors.end()) {
        return nullptr;
    }
    return it->second;
}

auto SDTextEncoderModelLoader::has_tensor(std::string_view tensor_name) const -> bool {
    return m_tensors.find(std::string(tensor_name)) != m_tensors.end();
}

auto SDTextEncoderModelLoader::summaries() const -> const std::vector<SDTextEncoderLoadSummary> & {
    return m_summaries;
}

auto SDTextEncoderModelLoader::tensor_count() const -> size_t {
    return m_tensors.size();
}

auto SDTextEncoderModelLoader::tensor_count_with_prefix(std::string_view tensor_prefix) const -> size_t {
    size_t count = 0;
    for (const auto &entry : m_tensors) {
        const auto &name = entry.first;
        if (name.rfind(tensor_prefix, 0) == 0) {
            count += 1;
        }
    }
    return count;
}

} // namespace powerserve
