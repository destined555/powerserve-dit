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
#include "encode/sd_text_encoder_embeddings.hpp"

namespace powerserve {

// 向量级对比统计。
struct SDVectorParityStats {
    double max_abs_error    = 0.0;
    double mean_abs_error   = 0.0;
    double rmse             = 0.0;
    double cosine_similarity = 1.0;
};

// SD3 crossattn 分支拆解统计：
// clip_l / clip_g / t5 以及 clip_l+clip_g 之后的 padding 区域。
struct SDSD3CrossattnBranchParityReport {
    bool available = false;
    SDVectorParityStats clip_l;
    SDVectorParityStats clip_g;
    SDVectorParityStats t5;
    SDVectorParityStats clip_lg_padding;
};

// SD3 pooled vector 分支拆解统计：clip_l / clip_g。
struct SDSD3VectorBranchParityReport {
    bool available = false;
    SDVectorParityStats clip_l;
    SDVectorParityStats clip_g;
};

// 正负提示词整体对比报告。
struct SDPromptPairParityReport {
    SDVectorParityStats prompt_crossattn;
    SDVectorParityStats prompt_vector;
    SDVectorParityStats negative_prompt_crossattn;
    SDVectorParityStats negative_prompt_vector;
    SDSD3CrossattnBranchParityReport prompt_sd3_crossattn;
    SDSD3CrossattnBranchParityReport negative_prompt_sd3_crossattn;
    SDSD3VectorBranchParityReport prompt_sd3_vector;
    SDSD3VectorBranchParityReport negative_prompt_sd3_vector;
};

// 保存 embedding 到自定义二进制格式（用于离线对比）。
void save_prompt_pair_embeddings(const Path &path, const SDPromptPairEmbeddings &embeddings);
// 从自定义二进制格式读取 embedding。
auto load_prompt_pair_embeddings(const Path &path) -> SDPromptPairEmbeddings;

// 对比 reference 与 candidate 的正负 embedding。
auto compare_prompt_pair_embeddings(
    const SDPromptPairEmbeddings &reference,
    const SDPromptPairEmbeddings &candidate
) -> SDPromptPairParityReport;

} // namespace powerserve
