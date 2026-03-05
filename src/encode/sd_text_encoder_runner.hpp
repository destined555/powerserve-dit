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
#include "encode/sd_prompt_tokenizer.hpp"
#include "encode/sd_text_encoder_embeddings.hpp"
#include "encode/sd_text_encoder_model_loader.hpp"

namespace powerserve {

// Text encoder 运行器：
// - 输入：tokenizer 产出的 token id（含正负提示词）；
// - 输出：SD3.5 需要的 crossattn / pooled vector；
// - 依赖：已加载的 text encoder gguf 权重（clip_l/clip_g/t5xxl）。
class SDTextEncoderRunner : Noncopyable {
public:
    SDTextEncoderRunner(const SDTextEncoderModelLoader &loader, const TextEncodeConfig &config, int n_threads = 0);

    // 编码单个 prompt token pack，得到一组 embedding。
    auto encode_prompt(const SDTokenIdPack &tokens) const -> SDTextEncoderEmbeddings;
    // 编码正负 token pack。
    auto encode_prompt_pair(const SDPromptTokenization &tokenized) const -> SDPromptPairEmbeddings;

private:
    // 仅保存指针引用，生命周期由外部 loader 负责。
    const SDTextEncoderModelLoader *m_loader = nullptr;
    TextEncodeConfig m_config;
    // <=0 时内部自动选择默认线程数。
    int m_n_threads = 0;
};

} // namespace powerserve
