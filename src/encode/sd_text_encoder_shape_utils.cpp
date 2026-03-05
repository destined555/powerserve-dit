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

#include "encode/sd_text_encoder_shape_utils.hpp"

#include "core/exception.hpp"

#include <algorithm>

namespace powerserve::detail {

namespace {

// 统一标签。
constexpr const char *kTag = "SDTextEncoderShapeUtils";

void assert_token_multiple(size_t token_count, const char *name) {
    // 允许空输入（由上层决定是否报错），非空则必须按 77 分块。
    if (token_count == 0) {
        return;
    }
    if (token_count % kSD3TextChunkLen != 0) {
        throw ConfigException(
            kTag,
            fmt::format("{} token count ({}) is not divisible by {}", name, token_count, kSD3TextChunkLen)
        );
    }
}

} // namespace

auto compute_sd3_chunk_count(size_t clip_l_tokens, size_t clip_g_tokens, size_t t5_tokens) -> size_t {
    // 三条分支都要求以 77 为块对齐。
    assert_token_multiple(clip_l_tokens, "clip_l");
    assert_token_multiple(clip_g_tokens, "clip_g");
    assert_token_multiple(t5_tokens, "t5");

    // chunk 数由最长分支决定，其余分支由 tokenizer 侧做 padding 对齐。
    const size_t max_tokens = std::max({clip_l_tokens, clip_g_tokens, t5_tokens});
    if (max_tokens == 0) {
        return 0;
    }
    return max_tokens / kSD3TextChunkLen;
}

auto compute_sd3_crossattn_tokens(size_t chunk_count) -> size_t {
    return chunk_count * kSD3CrossattnTokensPerChunk;
}

} // namespace powerserve::detail
