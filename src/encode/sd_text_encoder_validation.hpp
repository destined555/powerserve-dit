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

// 作用：统一承接 run-sd 的 text encoder 校验与 parity 对比逻辑，保持主程序简洁。

#pragma once

#include "core/typedefs.hpp"
#include "encode/sd_prompt_tokenizer.hpp"
#include "encode/sd_text_encoder_embeddings.hpp"

#include <string>

namespace powerserve {

// parity 执行选项：
// - dump_embeddings：将当前结果导出到文件；
// - compare_embeddings：读取参考文件并比较；
// - threshold：当 >0 时启用数值阈值校验。
struct SDTextEncoderParityOptions {
    Path dump_embeddings;
    Path compare_embeddings;
    float max_abs_threshold = 0.0f;
    float rmse_threshold    = 0.0f;
};

// 校验 token pack 基本约束（BOS/EOS、长度、权重维度等）。
auto verify_token_pack(const std::string &tag, const SDTokenIdPack &pack) -> bool;
// 校验正负提示词 tokenization，不通过则抛异常。
void verify_prompt_pair_tokenization_or_throw(const SDPromptTokenization &tokenized);

// 校验 embedding 形状和大小一致性。
auto verify_embedding(const std::string &tag, const SDTextEncoderEmbeddings &embeddings) -> bool;
// 校验正负 embedding，不通过则抛异常。
void verify_prompt_pair_embeddings_or_throw(const SDPromptPairEmbeddings &embeddings);

// 按配置导出当前 embedding（可选）。
void maybe_dump_prompt_pair_embeddings(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings);
// 按配置读取参考 embedding 并比较（可选）。
void maybe_compare_prompt_pair_embeddings(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings);
// parity 主入口：先 dump（若配置），再 compare（若配置）。
void maybe_run_prompt_pair_parity(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings);

} // namespace powerserve
