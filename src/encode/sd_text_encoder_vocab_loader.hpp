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

#include <cstddef>
#include <string>

namespace powerserve {

// text encoder tokenizer 词表载荷：
// - clip_merges：CLIP BPE merges 文本；
// - t5_tokenizer_json：T5 tokenizer JSON 配置。
struct SDTextEncoderVocabPack {
    std::string clip_merges;
    std::string t5_tokenizer_json;
};

// 词表载荷校验后的摘要信息，便于日志打印与诊断。
struct SDTextEncoderVocabInfo {
    size_t clip_merges_bytes = 0;
    size_t clip_merges_lines = 0;
    size_t t5_tokenizer_bytes = 0;
    size_t t5_vocab_size = 0;
};

class SDTextEncoderVocabLoader {
public:
    // 加载内嵌的 CLIP/T5 tokenizer 资源（由 sd.cpp 迁移而来）。
    static auto load_embedded() -> SDTextEncoderVocabPack;
    // 校验资源完整性，并返回摘要信息。
    static auto validate(const SDTextEncoderVocabPack &pack) -> SDTextEncoderVocabInfo;
};

} // namespace powerserve
