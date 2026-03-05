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

#include "encode/sd_text_encoder_parity.hpp"

#include "core/exception.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace powerserve {

namespace {

constexpr const char *kTag = "SDTextEncoderParity";
// 自定义 embedding dump 文件魔数（含版本号）。
constexpr std::array<char, 8> kMagic = {'P', 'S', 'E', 'M', 'B', '0', '0', '1'};
// SD3.5 文本分支的固定维度常量。
constexpr size_t kSD3ChunkLen = 77;
constexpr size_t kSD3CrossattnTokensPerChunk = 154;
constexpr size_t kSD3CrossattnDim = 4096;
constexpr size_t kSD3ClipLDim = 768;
constexpr size_t kSD3ClipGDim = 1280;
constexpr size_t kSD3ClipLGDim = kSD3ClipLDim + kSD3ClipGDim;

struct SD3CrossattnSlices {
    bool available = false;
    std::vector<float> clip_l;
    std::vector<float> clip_g;
    std::vector<float> t5;
    std::vector<float> clip_lg_padding;
};

struct SD3VectorSlices {
    bool available = false;
    std::vector<float> clip_l;
    std::vector<float> clip_g;
};

// 校验 embedding 的元数据与实际 buffer 大小一致。
void assert_embedding_shape(const SDTextEncoderEmbeddings &embeddings, std::string_view label) {
    const size_t expected_crossattn = embeddings.crossattn_tokens * embeddings.crossattn_dim;
    if (embeddings.crossattn.size() != expected_crossattn) {
        throw ConfigException(
            kTag,
            fmt::format(
                "{} crossattn size mismatch: got {}, expected {}",
                label,
                embeddings.crossattn.size(),
                expected_crossattn
            )
        );
    }
    if (embeddings.vector.size() != embeddings.vector_dim) {
        throw ConfigException(
            kTag,
            fmt::format(
                "{} vector size mismatch: got {}, expected {}",
                label,
                embeddings.vector.size(),
                embeddings.vector_dim
            )
        );
    }
}

// 二进制写辅助：统一错误处理。
void write_bytes(std::ofstream &ofs, const void *data, size_t size, std::string_view field_name) {
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!ofs.good()) {
        throw EnvironmentException(kTag, fmt::format("failed to write {} into dump", field_name));
    }
}

template<typename T>
void write_scalar(std::ofstream &ofs, T value, std::string_view field_name) {
    write_bytes(ofs, &value, sizeof(T), field_name);
}

// 按“元信息 + 原始 float 数据”写入单个 embedding。
void write_embedding(std::ofstream &ofs, const SDTextEncoderEmbeddings &embeddings, std::string_view label) {
    assert_embedding_shape(embeddings, label);

    const uint64_t crossattn_tokens = static_cast<uint64_t>(embeddings.crossattn_tokens);
    const uint64_t crossattn_dim    = static_cast<uint64_t>(embeddings.crossattn_dim);
    const uint64_t vector_dim       = static_cast<uint64_t>(embeddings.vector_dim);

    write_scalar(ofs, crossattn_tokens, fmt::format("{} crossattn_tokens", label));
    write_scalar(ofs, crossattn_dim, fmt::format("{} crossattn_dim", label));
    write_scalar(ofs, vector_dim, fmt::format("{} vector_dim", label));

    if (!embeddings.crossattn.empty()) {
        write_bytes(
            ofs,
            embeddings.crossattn.data(),
            embeddings.crossattn.size() * sizeof(float),
            fmt::format("{} crossattn data", label)
        );
    }
    if (!embeddings.vector.empty()) {
        write_bytes(
            ofs,
            embeddings.vector.data(),
            embeddings.vector.size() * sizeof(float),
            fmt::format("{} vector data", label)
        );
    }
}

// 二进制读辅助：统一错误处理。
void read_bytes(std::ifstream &ifs, void *data, size_t size, std::string_view field_name) {
    ifs.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(size));
    if (!ifs.good()) {
        throw EnvironmentException(kTag, fmt::format("failed to read {} from dump", field_name));
    }
}

template<typename T>
auto read_scalar(std::ifstream &ifs, std::string_view field_name) -> T {
    T value{};
    read_bytes(ifs, &value, sizeof(T), field_name);
    return value;
}

// 从二进制读取单个 embedding，并做一次形状一致性校验。
auto read_embedding(std::ifstream &ifs, std::string_view label) -> SDTextEncoderEmbeddings {
    SDTextEncoderEmbeddings embeddings;

    embeddings.crossattn_tokens = static_cast<size_t>(read_scalar<uint64_t>(ifs, fmt::format("{} crossattn_tokens", label)));
    embeddings.crossattn_dim    = static_cast<size_t>(read_scalar<uint64_t>(ifs, fmt::format("{} crossattn_dim", label)));
    embeddings.vector_dim       = static_cast<size_t>(read_scalar<uint64_t>(ifs, fmt::format("{} vector_dim", label)));

    const size_t crossattn_size = embeddings.crossattn_tokens * embeddings.crossattn_dim;
    embeddings.crossattn.resize(crossattn_size);
    embeddings.vector.resize(embeddings.vector_dim);

    if (!embeddings.crossattn.empty()) {
        read_bytes(
            ifs,
            embeddings.crossattn.data(),
            embeddings.crossattn.size() * sizeof(float),
            fmt::format("{} crossattn data", label)
        );
    }
    if (!embeddings.vector.empty()) {
        read_bytes(
            ifs,
            embeddings.vector.data(),
            embeddings.vector.size() * sizeof(float),
            fmt::format("{} vector data", label)
        );
    }

    assert_embedding_shape(embeddings, label);
    return embeddings;
}

// reference/candidate 形状必须完全一致，否则不允许比较。
void assert_same_shape(
    const SDTextEncoderEmbeddings &reference,
    const SDTextEncoderEmbeddings &candidate,
    std::string_view label
) {
    if (reference.crossattn_tokens != candidate.crossattn_tokens || reference.crossattn_dim != candidate.crossattn_dim ||
        reference.vector_dim != candidate.vector_dim) {
        throw ConfigException(
            kTag,
            fmt::format(
                "{} shape mismatch: ref(crossattn_tokens={}, crossattn_dim={}, vector_dim={}) "
                "vs candidate(crossattn_tokens={}, crossattn_dim={}, vector_dim={})",
                label,
                reference.crossattn_tokens,
                reference.crossattn_dim,
                reference.vector_dim,
                candidate.crossattn_tokens,
                candidate.crossattn_dim,
                candidate.vector_dim
            )
        );
    }
}

// 逐元素计算 max_abs / mean_abs / rmse / cosine。
auto compare_vector(const std::vector<float> &reference, const std::vector<float> &candidate, std::string_view label)
    -> SDVectorParityStats {
    if (reference.size() != candidate.size()) {
        throw ConfigException(
            kTag,
            fmt::format("{} size mismatch: ref={} vs candidate={}", label, reference.size(), candidate.size())
        );
    }

    SDVectorParityStats stats;
    if (reference.empty()) {
        return stats;
    }

    long double sum_abs      = 0.0;
    long double sum_sq       = 0.0;
    long double dot          = 0.0;
    long double ref_norm_sq  = 0.0;
    long double cand_norm_sq = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i];
        const double val = candidate[i];
        const double diff = val - ref;
        const double abs_diff = std::abs(diff);

        stats.max_abs_error = std::max(stats.max_abs_error, abs_diff);
        sum_abs += abs_diff;
        sum_sq += diff * diff;

        dot += ref * val;
        ref_norm_sq += ref * ref;
        cand_norm_sq += val * val;
    }

    const double n = static_cast<double>(reference.size());
    stats.mean_abs_error = static_cast<double>(sum_abs / n);
    stats.rmse           = std::sqrt(static_cast<double>(sum_sq / n));

    const double denom = std::sqrt(static_cast<double>(ref_norm_sq)) * std::sqrt(static_cast<double>(cand_norm_sq));
    if (denom == 0.0) {
        stats.cosine_similarity = 1.0;
    } else {
        stats.cosine_similarity = static_cast<double>(dot) / denom;
    }

    return stats;
}

// 将 SD3 crossattn 按语义拆为四段：
// 1) clip_l
// 2) clip_g
// 3) clip_l+clip_g 之后的 padding 区
// 4) t5
auto slice_sd3_crossattn(const SDTextEncoderEmbeddings &embeddings) -> SD3CrossattnSlices {
    SD3CrossattnSlices out;
    if (embeddings.crossattn_dim != kSD3CrossattnDim) {
        return out;
    }
    if (embeddings.crossattn_tokens == 0 || embeddings.crossattn_tokens % kSD3CrossattnTokensPerChunk != 0) {
        return out;
    }

    const size_t chunk_count = embeddings.crossattn_tokens / kSD3CrossattnTokensPerChunk;
    out.available = true;
    out.clip_l.reserve(chunk_count * kSD3ChunkLen * kSD3ClipLDim);
    out.clip_g.reserve(chunk_count * kSD3ChunkLen * kSD3ClipGDim);
    out.clip_lg_padding.reserve(chunk_count * kSD3ChunkLen * (kSD3CrossattnDim - kSD3ClipLGDim));
    out.t5.reserve(chunk_count * kSD3ChunkLen * kSD3CrossattnDim);

    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t chunk_base = chunk * kSD3CrossattnTokensPerChunk;

        for (size_t t = 0; t < kSD3ChunkLen; ++t) {
            const size_t row = chunk_base + t;
            const float *src = embeddings.crossattn.data() + row * kSD3CrossattnDim;
            out.clip_l.insert(out.clip_l.end(), src, src + kSD3ClipLDim);
            out.clip_g.insert(out.clip_g.end(), src + kSD3ClipLDim, src + kSD3ClipLGDim);
            out.clip_lg_padding.insert(out.clip_lg_padding.end(), src + kSD3ClipLGDim, src + kSD3CrossattnDim);
        }

        for (size_t t = 0; t < kSD3ChunkLen; ++t) {
            const size_t row = chunk_base + kSD3ChunkLen + t;
            const float *src = embeddings.crossattn.data() + row * kSD3CrossattnDim;
            out.t5.insert(out.t5.end(), src, src + kSD3CrossattnDim);
        }
    }

    return out;
}

// 对比 crossattn 各子分支误差。
auto compare_sd3_crossattn_branches(
    const SDTextEncoderEmbeddings &reference,
    const SDTextEncoderEmbeddings &candidate
) -> SDSD3CrossattnBranchParityReport {
    SDSD3CrossattnBranchParityReport out;
    const SD3CrossattnSlices ref = slice_sd3_crossattn(reference);
    const SD3CrossattnSlices cand = slice_sd3_crossattn(candidate);
    if (!ref.available || !cand.available) {
        return out;
    }

    out.available = true;
    out.clip_l = compare_vector(ref.clip_l, cand.clip_l, "sd3.clip_l");
    out.clip_g = compare_vector(ref.clip_g, cand.clip_g, "sd3.clip_g");
    out.t5 = compare_vector(ref.t5, cand.t5, "sd3.t5");
    out.clip_lg_padding = compare_vector(ref.clip_lg_padding, cand.clip_lg_padding, "sd3.clip_lg_padding");
    return out;
}

// 将 pooled vector 拆为 clip_l / clip_g 两段。
auto slice_sd3_vector(const SDTextEncoderEmbeddings &embeddings) -> SD3VectorSlices {
    SD3VectorSlices out;
    if (embeddings.vector_dim != kSD3ClipLGDim || embeddings.vector.size() != kSD3ClipLGDim) {
        return out;
    }

    out.available = true;
    out.clip_l.insert(out.clip_l.end(), embeddings.vector.begin(), embeddings.vector.begin() + kSD3ClipLDim);
    out.clip_g.insert(
        out.clip_g.end(),
        embeddings.vector.begin() + kSD3ClipLDim,
        embeddings.vector.end()
    );
    return out;
}

// 对比 pooled 各子分支误差。
auto compare_sd3_vector_branches(
    const SDTextEncoderEmbeddings &reference,
    const SDTextEncoderEmbeddings &candidate
) -> SDSD3VectorBranchParityReport {
    SDSD3VectorBranchParityReport out;
    const SD3VectorSlices ref = slice_sd3_vector(reference);
    const SD3VectorSlices cand = slice_sd3_vector(candidate);
    if (!ref.available || !cand.available) {
        return out;
    }

    out.available = true;
    out.clip_l = compare_vector(ref.clip_l, cand.clip_l, "sd3.vector.clip_l");
    out.clip_g = compare_vector(ref.clip_g, cand.clip_g, "sd3.vector.clip_g");
    return out;
}

} // namespace

void save_prompt_pair_embeddings(const Path &path, const SDPromptPairEmbeddings &embeddings) {
    // 文件格式：magic + prompt embedding + negative_prompt embedding。
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw EnvironmentException(kTag, fmt::format("failed to open dump file for write: {}", path.string()));
    }

    write_bytes(ofs, kMagic.data(), kMagic.size(), "magic");
    write_embedding(ofs, embeddings.prompt, "prompt");
    write_embedding(ofs, embeddings.negative_prompt, "negative_prompt");
}

auto load_prompt_pair_embeddings(const Path &path) -> SDPromptPairEmbeddings {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw EnvironmentException(kTag, fmt::format("failed to open dump file for read: {}", path.string()));
    }

    std::array<char, kMagic.size()> magic{};
    read_bytes(ifs, magic.data(), magic.size(), "magic");
    if (magic != kMagic) {
        throw ConfigException(kTag, fmt::format("invalid dump magic: {}", path.string()));
    }

    SDPromptPairEmbeddings out;
    out.prompt          = read_embedding(ifs, "prompt");
    out.negative_prompt = read_embedding(ifs, "negative_prompt");
    return out;
}

auto compare_prompt_pair_embeddings(
    const SDPromptPairEmbeddings &reference,
    const SDPromptPairEmbeddings &candidate
) -> SDPromptPairParityReport {
    // 先校验每个 embedding 自身形状正确，再校验双方形状一致。
    assert_embedding_shape(reference.prompt, "reference.prompt");
    assert_embedding_shape(candidate.prompt, "candidate.prompt");
    assert_embedding_shape(reference.negative_prompt, "reference.negative_prompt");
    assert_embedding_shape(candidate.negative_prompt, "candidate.negative_prompt");

    assert_same_shape(reference.prompt, candidate.prompt, "prompt");
    assert_same_shape(reference.negative_prompt, candidate.negative_prompt, "negative_prompt");

    SDPromptPairParityReport out;
    out.prompt_crossattn = compare_vector(reference.prompt.crossattn, candidate.prompt.crossattn, "prompt.crossattn");
    out.prompt_vector    = compare_vector(reference.prompt.vector, candidate.prompt.vector, "prompt.vector");
    out.negative_prompt_crossattn = compare_vector(
        reference.negative_prompt.crossattn,
        candidate.negative_prompt.crossattn,
        "negative_prompt.crossattn"
    );
    out.negative_prompt_vector =
        compare_vector(reference.negative_prompt.vector, candidate.negative_prompt.vector, "negative_prompt.vector");
    out.prompt_sd3_crossattn = compare_sd3_crossattn_branches(reference.prompt, candidate.prompt);
    out.negative_prompt_sd3_crossattn =
        compare_sd3_crossattn_branches(reference.negative_prompt, candidate.negative_prompt);
    out.prompt_sd3_vector = compare_sd3_vector_branches(reference.prompt, candidate.prompt);
    out.negative_prompt_sd3_vector = compare_sd3_vector_branches(reference.negative_prompt, candidate.negative_prompt);
    return out;
}

} // namespace powerserve
