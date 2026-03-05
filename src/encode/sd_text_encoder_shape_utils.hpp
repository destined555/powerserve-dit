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

namespace powerserve::detail {

// SD3 文本分块长度（CLIP/T5 均按 77 token 一个 chunk）。
inline constexpr size_t kSD3TextChunkLen            = 77;
// 每个 chunk 的 crossattn 由两条分支拼接：clip(l+g) + t5。
inline constexpr size_t kSD3CrossattnBranches       = 2;
inline constexpr size_t kSD3CrossattnTokensPerChunk = kSD3TextChunkLen * kSD3CrossattnBranches;

// 根据三条分支 token 数，计算需要处理的 chunk 数。
// 要求 token 数为 77 的倍数（0 表示空输入）。
auto compute_sd3_chunk_count(size_t clip_l_tokens, size_t clip_g_tokens, size_t t5_tokens) -> size_t;
// 根据 chunk 数推导 crossattn token 总数。
auto compute_sd3_crossattn_tokens(size_t chunk_count) -> size_t;

} // namespace powerserve::detail
