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

#include "encode/sd_text_encoder_vocab_loader.hpp"

#include "core/exception.hpp"

#include "encode/vocab/clip_t5.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace powerserve {

namespace {

// 统一日志/异常标签。
constexpr const char *kTag = "SDTextEncoderVocabLoader";

auto load_embedded_utf8(const unsigned char *payload, size_t payload_size) -> std::string {
    // 内嵌资源为空时返回空字符串，由上层统一做配置断言。
    if (payload == nullptr || payload_size == 0) {
        return {};
    }

    // 兼容“C 字符串末尾带 '\0'”与“裸字节数组”两种资源形式。
    size_t used_size = payload_size;
    if (payload[payload_size - 1] == '\0') {
        used_size = payload_size - 1;
    }

    return std::string(reinterpret_cast<const char *>(payload), used_size);
}

} // namespace

auto SDTextEncoderVocabLoader::load_embedded() -> SDTextEncoderVocabPack {
    // clip/t5 词表都来自编译期内嵌资源（不依赖运行时外部文件）。
    SDTextEncoderVocabPack pack;
    pack.clip_merges = load_embedded_utf8(clip_merges_utf8_c_str, sizeof(clip_merges_utf8_c_str));
    pack.t5_tokenizer_json = load_embedded_utf8(t5_tokenizer_json_str, sizeof(t5_tokenizer_json_str));
    return pack;
}

auto SDTextEncoderVocabLoader::validate(const SDTextEncoderVocabPack &pack) -> SDTextEncoderVocabInfo {
    SDTextEncoderVocabInfo info;

    info.clip_merges_bytes = pack.clip_merges.size();
    info.t5_tokenizer_bytes = pack.t5_tokenizer_json.size();

    POWERSERVE_ASSERT_CONFIG(!pack.clip_merges.empty(), kTag, "clip merges payload is empty");
    POWERSERVE_ASSERT_CONFIG(!pack.t5_tokenizer_json.empty(), kTag, "t5 tokenizer json payload is empty");

    // merges 行数过小通常意味着资源被截断。
    info.clip_merges_lines =
        static_cast<size_t>(std::count(pack.clip_merges.begin(), pack.clip_merges.end(), '\n'));
    POWERSERVE_ASSERT_CONFIG(info.clip_merges_lines > 1000, kTag, "clip merges payload seems truncated");

    nlohmann::json tokenizer_json;
    try {
        tokenizer_json = nlohmann::json::parse(pack.t5_tokenizer_json);
    } catch (const std::exception &err) {
        throw ConfigException(kTag, fmt::format("failed to parse t5 tokenizer json: {}", err.what()));
    }

    // 做最小 JSON 结构约束，确保 tokenizer 初始化阶段不会因关键字段缺失而崩溃。
    POWERSERVE_ASSERT_CONFIG(tokenizer_json.contains("model"), kTag, "t5 tokenizer json missing key `model`");
    const auto &model_json = tokenizer_json.at("model");
    POWERSERVE_ASSERT_CONFIG(model_json.contains("vocab"), kTag, "t5 tokenizer json missing key `model.vocab`");
    POWERSERVE_ASSERT_CONFIG(
        model_json.at("vocab").is_array(), kTag, "t5 tokenizer json `model.vocab` should be an array"
    );

    info.t5_vocab_size = model_json.at("vocab").size();
    POWERSERVE_ASSERT_CONFIG(info.t5_vocab_size > 0, kTag, "t5 tokenizer vocabulary is empty");

    return info;
}

} // namespace powerserve
