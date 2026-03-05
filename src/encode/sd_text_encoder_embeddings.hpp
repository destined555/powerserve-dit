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
#include <vector>

namespace powerserve {

struct SDTextEncoderEmbeddings {
    // 交叉注意力条件张量，逻辑 shape 为 [crossattn_tokens, crossattn_dim]。
    // 扁平存储索引：index = token_idx * crossattn_dim + dim_idx。
    std::vector<float> crossattn;
    // pooled 向量，逻辑 shape 为 [vector_dim]
    // （单 prompt 语义下等价于 [1, vector_dim]）。
    std::vector<float> vector;

    // crossattn 每个 token 的维度（SD3.5 下固定为 4096）。
    size_t crossattn_dim = 0;

    // crossattn token 数（按 chunk 叠加后得到）。
    size_t crossattn_tokens = 0;

    // pooled 向量维度（SD3.5 下固定为 2048 = 768 + 1280）。
    size_t vector_dim = 0;
};

// 正负提示词对应的一组 embedding 输出。
struct SDPromptPairEmbeddings {
    SDTextEncoderEmbeddings prompt;
    SDTextEncoderEmbeddings negative_prompt;
};

} // namespace powerserve
