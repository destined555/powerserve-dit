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

#include "encode/sd_text_encoder_validation.hpp"

#include "core/exception.hpp"
#include "core/logger.hpp"
#include "encode/sd_text_encoder_parity.hpp"

#include <algorithm>
#include <string>

namespace powerserve {
namespace {

// 统一打印一组数值对齐指标。
void log_parity_stats(const std::string &tag, const SDVectorParityStats &stats) {
    POWERSERVE_LOG_INFO(
        "[run-sd][parity] {}: max_abs={:.8f} mean_abs={:.8f} rmse={:.8f} cosine={:.8f}",
        tag,
        stats.max_abs_error,
        stats.mean_abs_error,
        stats.rmse,
        stats.cosine_similarity
    );
}

// 打印 SD3 crossattn 各子分支报告。
void log_sd3_crossattn_branch_report(const std::string &tag, const SDSD3CrossattnBranchParityReport &report) {
    if (!report.available) {
        POWERSERVE_LOG_INFO("[run-sd][parity] {}: sd3 branch report unavailable (non-SD3 shape)", tag);
        return;
    }
    log_parity_stats(tag + ".clip_l", report.clip_l);
    log_parity_stats(tag + ".clip_g", report.clip_g);
    log_parity_stats(tag + ".t5", report.t5);
    log_parity_stats(tag + ".clip_lg_padding", report.clip_lg_padding);
}

// 打印 SD3 pooled vector 各子分支报告。
void log_sd3_vector_branch_report(const std::string &tag, const SDSD3VectorBranchParityReport &report) {
    if (!report.available) {
        POWERSERVE_LOG_INFO("[run-sd][parity] {}: sd3 vector branch report unavailable (non-SD3 shape)", tag);
        return;
    }
    log_parity_stats(tag + ".clip_l", report.clip_l);
    log_parity_stats(tag + ".clip_g", report.clip_g);
}

// 在用户配置阈值时执行硬阈值校验（不配置即仅打印）。
void verify_parity_stats(const std::string &tag, const SDVectorParityStats &stats, float max_abs_threshold, float rmse_threshold) {
    if (max_abs_threshold > 0.0f && stats.max_abs_error > static_cast<double>(max_abs_threshold)) {
        throw ConfigException(
            "run-sd",
            fmt::format("{} parity max_abs_error={} exceeds threshold {}", tag, stats.max_abs_error, max_abs_threshold)
        );
    }

    if (rmse_threshold > 0.0f && stats.rmse > static_cast<double>(rmse_threshold)) {
        throw ConfigException("run-sd", fmt::format("{} parity rmse={} exceeds threshold {}", tag, stats.rmse, rmse_threshold));
    }
}

} // namespace

auto verify_token_pack(const std::string &tag, const SDTokenIdPack &pack) -> bool {
    // 该函数以“尽量多报错”为目标，不在首个错误处立即返回。
    bool pass = true;
    auto fail = [&](const std::string &message) {
        pass = false;
        POWERSERVE_LOG_ERROR("[run-sd][verify] {}: FAIL - {}", tag, message);
    };

    if (pack.clip_l.empty()) {
        fail("clip_l tokens are empty");
    }
    if (pack.clip_g.empty()) {
        fail("clip_g tokens are empty");
    }
    if (pack.t5.empty()) {
        fail("t5 tokens are empty");
    }

    if (pack.clip_l.size() != pack.clip_g.size()) {
        fail(fmt::format("clip_l/clip_g token count mismatch: {} vs {}", pack.clip_l.size(), pack.clip_g.size()));
    }

    if (!pack.clip_l_weights.empty() && pack.clip_l_weights.size() != pack.clip_l.size()) {
        fail(fmt::format("clip_l weight count mismatch: {} vs {}", pack.clip_l_weights.size(), pack.clip_l.size()));
    }
    if (!pack.clip_g_weights.empty() && pack.clip_g_weights.size() != pack.clip_g.size()) {
        fail(fmt::format("clip_g weight count mismatch: {} vs {}", pack.clip_g_weights.size(), pack.clip_g.size()));
    }
    if (!pack.t5_weights.empty() && pack.t5_weights.size() != pack.t5.size()) {
        fail(fmt::format("t5 weight count mismatch: {} vs {}", pack.t5_weights.size(), pack.t5.size()));
    }

    if (!pack.clip_l.empty() && pack.clip_l.front() != 49406) {
        fail(fmt::format("clip_l first token should be BOS(49406), got {}", pack.clip_l.front()));
    }
    if (!pack.clip_g.empty() && pack.clip_g.front() != 49406) {
        fail(fmt::format("clip_g first token should be BOS(49406), got {}", pack.clip_g.front()));
    }

    if (!pack.clip_l.empty() && pack.clip_l.size() % 77 != 0) {
        fail(fmt::format("clip_l length {} is not a multiple of 77", pack.clip_l.size()));
    }
    if (!pack.clip_g.empty() && pack.clip_g.size() % 77 != 0) {
        fail(fmt::format("clip_g length {} is not a multiple of 77", pack.clip_g.size()));
    }
    if (!pack.t5.empty() && pack.t5.size() % 77 != 0) {
        fail(fmt::format("t5 length {} is not a multiple of 77", pack.t5.size()));
    }

    const bool t5_has_eos = std::find(pack.t5.begin(), pack.t5.end(), 1) != pack.t5.end();
    if (!t5_has_eos) {
        fail("t5 does not contain EOS(1)");
    }

    if (!pack.clip_g.empty()) {
        auto it = std::find_if(pack.clip_g.rbegin(), pack.clip_g.rend(), [](Token t) { return t != 0; });
        if (it == pack.clip_g.rend()) {
            fail("clip_g is all padding zeros");
        } else if (*it != 49407) {
            fail(fmt::format("clip_g last non-pad token should be EOS(49407), got {}", *it));
        }
    }

    if (!pack.t5.empty()) {
        auto it = std::find_if(pack.t5.rbegin(), pack.t5.rend(), [](Token t) { return t != 0; });
        if (it == pack.t5.rend()) {
            fail("t5 is all padding zeros");
        } else if (*it != 1) {
            fail(fmt::format("t5 last non-pad token should be EOS(1), got {}", *it));
        }
    }

    if (!pack.clip_l.empty() && !pack.clip_g.empty()) {
        // clip_l 与 clip_g 在非 padding 区域应保持 token 一致。
        const size_t n = std::min(pack.clip_l.size(), pack.clip_g.size());
        for (size_t i = 0; i < n; ++i) {
            if (pack.clip_g[i] == 0) {
                continue;
            }
            if (pack.clip_l[i] != pack.clip_g[i]) {
                fail(fmt::format("clip_l/clip_g mismatch at index {}: {} vs {}", i, pack.clip_l[i], pack.clip_g[i]));
                break;
            }
        }
    }

    return pass;
}

void verify_prompt_pair_tokenization_or_throw(const SDPromptTokenization &tokenized) {
    // 正负提示词任一失败都视为整体失败。
    const bool prompt_ok  = verify_token_pack("prompt", tokenized.prompt);
    const bool nprompt_ok = verify_token_pack("negative prompt", tokenized.negative_prompt);
    if (!prompt_ok || !nprompt_ok) {
        throw ConfigException("run-sd", "prompt->tokenid verification failed");
    }
    POWERSERVE_LOG_INFO("[run-sd][verify] all prompt->tokenid checks passed.");
}

auto verify_embedding(const std::string &tag, const SDTextEncoderEmbeddings &embeddings) -> bool {
    // 与 SD3.5 固定契约做强校验，便于快速发现 shape 漂移。
    bool pass = true;
    auto fail = [&](const std::string &message) {
        pass = false;
        POWERSERVE_LOG_ERROR("[run-sd][verify] {} embedding: FAIL - {}", tag, message);
    };

    if (embeddings.crossattn_dim != 4096) {
        fail(fmt::format("crossattn_dim should be 4096, got {}", embeddings.crossattn_dim));
    }

    if (embeddings.vector_dim != 2048) {
        fail(fmt::format("vector_dim should be 2048, got {}", embeddings.vector_dim));
    }

    if (embeddings.crossattn_tokens == 0) {
        fail("crossattn_tokens is zero");
    }

    const size_t expect_crossattn_size = embeddings.crossattn_tokens * embeddings.crossattn_dim;
    if (embeddings.crossattn.size() != expect_crossattn_size) {
        fail(
            fmt::format(
                "crossattn size mismatch: got {}, expected {}",
                embeddings.crossattn.size(),
                expect_crossattn_size
            )
        );
    }

    if (embeddings.vector.size() != embeddings.vector_dim) {
        fail(fmt::format("vector size mismatch: got {}, expected {}", embeddings.vector.size(), embeddings.vector_dim));
    }

    return pass;
}

void verify_prompt_pair_embeddings_or_throw(const SDPromptPairEmbeddings &embeddings) {
    const bool prompt_ok  = verify_embedding("prompt", embeddings.prompt);
    const bool nprompt_ok = verify_embedding("negative prompt", embeddings.negative_prompt);
    if (!prompt_ok || !nprompt_ok) {
        throw ConfigException("run-sd", "tokenid->vector verification failed");
    }
    POWERSERVE_LOG_INFO("[run-sd][verify] all tokenid->vector checks passed.");
}

void maybe_dump_prompt_pair_embeddings(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings) {
    // 未配置输出路径时跳过。
    if (options.dump_embeddings.empty()) {
        return;
    }
    save_prompt_pair_embeddings(options.dump_embeddings, embeddings);
    POWERSERVE_LOG_INFO("[run-sd][parity] dumped embeddings to {}", options.dump_embeddings.string());
}

void maybe_compare_prompt_pair_embeddings(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings) {
    // 未配置参考路径时跳过。
    if (options.compare_embeddings.empty()) {
        return;
    }

    // 比对顺序：整体指标 -> SD3 子分支指标 -> 阈值校验。
    const auto reference = load_prompt_pair_embeddings(options.compare_embeddings);
    const auto report    = compare_prompt_pair_embeddings(reference, embeddings);

    POWERSERVE_LOG_INFO("[run-sd][parity] compare against {}", options.compare_embeddings.string());
    log_parity_stats("prompt.crossattn", report.prompt_crossattn);
    log_parity_stats("prompt.vector", report.prompt_vector);
    log_parity_stats("negative_prompt.crossattn", report.negative_prompt_crossattn);
    log_parity_stats("negative_prompt.vector", report.negative_prompt_vector);
    log_sd3_crossattn_branch_report("prompt.sd3_crossattn", report.prompt_sd3_crossattn);
    log_sd3_crossattn_branch_report("negative_prompt.sd3_crossattn", report.negative_prompt_sd3_crossattn);
    log_sd3_vector_branch_report("prompt.sd3_vector", report.prompt_sd3_vector);
    log_sd3_vector_branch_report("negative_prompt.sd3_vector", report.negative_prompt_sd3_vector);

    verify_parity_stats("prompt.crossattn", report.prompt_crossattn, options.max_abs_threshold, options.rmse_threshold);
    verify_parity_stats("prompt.vector", report.prompt_vector, options.max_abs_threshold, options.rmse_threshold);
    verify_parity_stats(
        "negative_prompt.crossattn",
        report.negative_prompt_crossattn,
        options.max_abs_threshold,
        options.rmse_threshold
    );
    verify_parity_stats(
        "negative_prompt.vector",
        report.negative_prompt_vector,
        options.max_abs_threshold,
        options.rmse_threshold
    );

    POWERSERVE_LOG_INFO("[run-sd][parity] compare finished.");
}

void maybe_run_prompt_pair_parity(const SDTextEncoderParityOptions &options, const SDPromptPairEmbeddings &embeddings) {
    // 先导出再对比，便于一次执行中同时保留当前结果。
    maybe_dump_prompt_pair_embeddings(options, embeddings);
    maybe_compare_prompt_pair_embeddings(options, embeddings);
}

} // namespace powerserve
