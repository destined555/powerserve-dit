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

#include "core/typedefs.hpp"
#include "encode/sd_text_encoder_vocab_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace powerserve {

// 单个 prompt 的 token id 与 token 权重（用于 SD 风格强调语法）。
// 三条分支分别对应 SD3.5 条件网络中的 clip_l / clip_g / t5。
struct SDTokenIdPack {
    std::vector<Token> clip_l;
    std::vector<float> clip_l_weights;
    std::vector<Token> clip_g;
    std::vector<float> clip_g_weights;
    std::vector<Token> t5;
    std::vector<float> t5_weights;
};

// 正负提示词成对编码结果。
struct SDPromptTokenization {
    SDTokenIdPack prompt;
    SDTokenIdPack negative_prompt;
};

// SD prompt tokenizer 统一入口：
// - 内部封装 CLIP BPE + T5 Unigram 两套 tokenizer；
// - 输出 token id 与 attention 权重；
// - token 长度与 padding 策略与 sd.cpp 对齐。
class SDPromptTokenizer : Noncopyable {
public:
    explicit SDPromptTokenizer(const SDTextEncoderVocabPack &vocab_pack);
    ~SDPromptTokenizer();

    // 编码单个 prompt（同时生成 clip_l/clip_g/t5 三条分支）。
    auto encode_prompt(const std::string &prompt) const -> SDTokenIdPack;
    // 编码正负提示词对。
    auto encode_prompt_pair(const std::string &prompt, const std::string &negative_prompt) const -> SDPromptTokenization;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace powerserve
