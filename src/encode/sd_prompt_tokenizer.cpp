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

#include "encode/sd_prompt_tokenizer.hpp"

#include "core/exception.hpp"
#include "core/logger.hpp"
#include "encode/thirdparty/darts.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <codecvt>
#include <cmath>
#include <locale>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace powerserve {

namespace {

constexpr const char *kTag = "SDPromptTokenizer";

// UTF-8 与 UTF-32 互转辅助，BPE 合并阶段使用 UTF-32 以便按 Unicode 码点处理。
auto utf8_to_utf32(const std::string &utf8_str) -> std::u32string {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

auto unicode_value_to_utf32(int unicode_value) -> std::u32string {
    return std::u32string{static_cast<char32_t>(unicode_value)};
}

auto str_to_lower(std::string text) -> std::string {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// 压缩连续空白并做 trim，尽量与 sd.cpp 预处理一致。
auto whitespace_clean(const std::string &text) -> std::string {
    auto out = std::regex_replace(text, std::regex(R"(\s+)") , " ");
    return trim(out);
}

// 解析 SD 风格强调语法，行为对齐 sd.cpp：
// - "(...)" 增强权重；
// - "[...]" 降低权重；
// - "(foo:1.2)" 使用显式倍率；
// - 转义符支持 \(\)\[\]\\；
// - 仅改变权重，不将语法符号混入 token 文本。
auto parse_prompt_attention(const std::string &text) -> std::vector<std::pair<std::string, float>> {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    const float round_bracket_multiplier  = 1.1f;
    const float square_bracket_multiplier = 1.0f / 1.1f;

    const std::regex re_attention(
        R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|\bBREAK\b|[^\\()\[\]:B]+|:|\bB)"
    );
    const std::regex re_break(R"(\s*\bBREAK\b\s*)");

    const auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < static_cast<int>(res.size()); ++p) {
            res[static_cast<size_t>(p)].second *= multiplier;
        }
    };

    std::smatch m;
    std::smatch m2;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        const std::string token = m[0];
        const std::string weight = m[1];

        if (token == "(") {
            round_brackets.push_back(static_cast<int>(res.size()));
        } else if (token == "[") {
            square_brackets.push_back(static_cast<int>(res.size()));
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (token == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (token == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (token == "\\(") {
            res.emplace_back(token.substr(1), 1.0f);
        } else if (std::regex_search(token, m2, re_break)) {
            res.emplace_back("BREAK", -1.0f);
        } else {
            res.emplace_back(token, 1.0f);
        }

        remaining_text = m.suffix();
    }

    for (const int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }
    for (const int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.emplace_back("", 1.0f);
    }

    size_t i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + static_cast<long>(i + 1));
        } else {
            i += 1;
        }
    }

    return res;
}

// 按 special token 切分文本，并保留 special token 本体作为独立片段。
auto split_with_special_tokens(const std::string &text, const std::vector<std::string> &special_tokens)
    -> std::vector<std::string> {
    std::vector<std::string> result;
    size_t pos      = 0;
    size_t text_len = text.size();

    while (pos < text_len) {
        size_t next_pos = text_len;
        std::string matched_token;

        for (const auto &token : special_tokens) {
            const size_t token_pos = text.find(token, pos);
            if (token_pos != std::string::npos && token_pos < next_pos) {
                next_pos = token_pos;
                matched_token = token;
            }
        }

        if (next_pos > pos) {
            result.push_back(text.substr(pos, next_pos - pos));
        }

        if (!matched_token.empty()) {
            result.push_back(matched_token);
            pos = next_pos + matched_token.size();
        } else {
            break;
        }
    }

    if (result.empty()) {
        result.push_back(text);
    }
    return result;
}

// 与 CLIP BPE 近似的词片切分正则。
auto token_split(const std::string &text) -> std::vector<std::string> {
    static const std::regex pat(
        R"('s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
        std::regex::icase
    );

    std::sregex_iterator iter(text.begin(), text.end(), pat);
    std::sregex_iterator end;

    std::vector<std::string> result;
    for (; iter != end; ++iter) {
        result.emplace_back(iter->str());
    }
    return result;
}

// CLIP BPE 的 bytes_to_unicode 映射构造。
auto bytes_to_unicode() -> std::vector<std::pair<int, std::u32string>> {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;

    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.emplace_back(b, unicode_value_to_utf32(b));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.emplace_back(b, unicode_value_to_utf32(b));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.emplace_back(b, unicode_value_to_utf32(b));
    }

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.emplace_back(b, unicode_value_to_utf32(n + 256));
            ++n;
        }
    }

    return byte_unicode_pairs;
}

// CLIP tokenizer（BPE）实现。
class CLIPBPETokenizer {
public:
    explicit CLIPBPETokenizer(std::string merges_utf8_str, Token pad_token_id = 49407) : m_pad_token_id(pad_token_id) {
        load_from_merges(merges_utf8_str);
        m_special_tokens.push_back("<|startoftext|>");
        m_special_tokens.push_back("<|endoftext|>");
    }

    auto encode_sd3_with_weights(const std::string &text, size_t max_length = 77, bool padding = true) const
        -> std::pair<std::vector<Token>, std::vector<float>> {
        std::vector<Token> tokens;
        std::vector<float> weights;
        for (const auto &item : parse_prompt_attention(text)) {
            auto curr = encode(item.first);
            tokens.insert(tokens.end(), curr.begin(), curr.end());
            weights.insert(weights.end(), curr.size(), item.second);
        }
        pad_tokens(tokens, weights, max_length, padding);
        return {std::move(tokens), std::move(weights)};
    }

    auto encode_sd3(const std::string &text, size_t max_length = 77, bool padding = true) const -> std::vector<Token> {
        return encode_sd3_with_weights(text, max_length, padding).first;
    }

private:
    Token m_unk_token_id = 49407;
    Token m_bos_token_id = 49406;
    Token m_eos_token_id = 49407;
    Token m_pad_token_id = 49407;

    std::map<int, std::u32string> m_byte_encoder;
    std::map<std::u32string, int> m_encoder;
    std::map<std::pair<std::u32string, std::u32string>, int> m_bpe_ranks;
    std::vector<std::string> m_special_tokens;

private:
    // 收集相邻子词对，用于 BPE merge 迭代。
    static auto get_pairs(const std::vector<std::u32string> &subwords)
        -> std::set<std::pair<std::u32string, std::u32string>> {
        std::set<std::pair<std::u32string, std::u32string>> pairs;
        if (subwords.empty()) {
            return pairs;
        }

        std::u32string prev_subword = subwords[0];
        for (size_t i = 1; i < subwords.size(); ++i) {
            const std::u32string &subword = subwords[i];
            pairs.insert(std::pair(prev_subword, subword));
            prev_subword = subword;
        }
        return pairs;
    }

    auto is_special_token(const std::string &token) const -> bool {
        return std::find(m_special_tokens.begin(), m_special_tokens.end(), token) != m_special_tokens.end();
    }

    void load_from_merges(const std::string &merges_utf8_str) {
        auto byte_unicode_pairs = bytes_to_unicode();
        m_byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());

        std::vector<std::u32string> merges;
        size_t start = 0;
        const std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
        while (true) {
            const size_t pos = merges_utf32_str.find(U'\n', start);
            if (pos == std::u32string::npos) {
                break;
            }
            merges.push_back(merges_utf32_str.substr(start, pos - start));
            start = pos + 1;
        }

        // clip merges 文件应包含 1 行头 + 48894 行 merge。
        POWERSERVE_ASSERT_CONFIG(merges.size() >= 48895, kTag, "invalid clip merges rows: {}", merges.size());

        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        merge_pairs.reserve(merges.size() - 1);
        for (size_t i = 1; i < merges.size(); ++i) {
            const auto &merge = merges[i];
            const size_t space_pos = merge.find(U' ');
            if (space_pos == std::u32string::npos) {
                continue;
            }
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
        }

        std::vector<std::u32string> vocab;
        vocab.reserve(byte_unicode_pairs.size() * 2 + merge_pairs.size() + 2);
        for (const auto &pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto &pair : byte_unicode_pairs) {
            vocab.push_back(pair.second + utf8_to_utf32("</w>"));
        }
        for (const auto &merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|startoftext|>"));
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));

        m_encoder.clear();
        for (size_t i = 0; i < vocab.size(); ++i) {
            m_encoder[vocab[i]] = static_cast<int>(i);
        }

        m_bpe_ranks.clear();
        for (size_t rank = 0; rank < merge_pairs.size(); ++rank) {
            m_bpe_ranks[merge_pairs[rank]] = static_cast<int>(rank);
        }
    }

    // 标准 BPE merge 主循环。
    auto bpe(const std::u32string &token) const -> std::u32string {
        std::vector<std::u32string> word;
        if (token.empty()) {
            return utf8_to_utf32("</w>");
        }

        for (size_t i = 0; i + 1 < token.size(); ++i) {
            word.emplace_back(1, token[i]);
        }
        word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

        auto pairs = get_pairs(word);
        if (pairs.empty()) {
            return token + utf8_to_utf32("</w>");
        }

        while (true) {
            auto min_pair_iter = std::min_element(
                pairs.begin(),
                pairs.end(),
                [&](const std::pair<std::u32string, std::u32string> &a,
                    const std::pair<std::u32string, std::u32string> &b) {
                    const auto a_it = m_bpe_ranks.find(a);
                    const auto b_it = m_bpe_ranks.find(b);
                    if (a_it == m_bpe_ranks.end()) {
                        return false;
                    }
                    if (b_it == m_bpe_ranks.end()) {
                        return true;
                    }
                    return a_it->second < b_it->second;
                }
            );

            const auto &bigram = *min_pair_iter;
            if (m_bpe_ranks.find(bigram) == m_bpe_ranks.end()) {
                break;
            }

            const std::u32string &first  = bigram.first;
            const std::u32string &second = bigram.second;
            std::vector<std::u32string> new_word;

            size_t i = 0;
            while (i < word.size()) {
                auto it = std::find(word.begin() + static_cast<long>(i), word.end(), first);
                if (it == word.end()) {
                    new_word.insert(new_word.end(), word.begin() + static_cast<long>(i), word.end());
                    break;
                }

                const size_t it_pos = static_cast<size_t>(std::distance(word.begin(), it));
                new_word.insert(new_word.end(), word.begin() + static_cast<long>(i), it);
                i = it_pos;

                if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) {
                    new_word.push_back(first + second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }

            word = std::move(new_word);
            if (word.size() == 1) {
                break;
            }
            pairs = get_pairs(word);
        }

        std::u32string result;
        for (size_t i = 0; i < word.size(); ++i) {
            result += word[i];
            if (i + 1 < word.size()) {
                result += utf8_to_utf32(" ");
            }
        }
        return result;
    }

    // 文本 -> CLIP token id 序列。
    auto encode(const std::string &text) const -> std::vector<Token> {
        std::vector<Token> bpe_tokens;

        const std::string lowered = str_to_lower(whitespace_clean(text));
        const std::vector<std::string> split_texts = split_with_special_tokens(lowered, m_special_tokens);

        for (const auto &part : split_texts) {
            if (part.empty()) {
                continue;
            }

            if (is_special_token(part)) {
                // 与 sd.cpp 对齐：默认流程跳过 special token，
                // 这类 token 在特定回调路径中另行处理。
                continue;
            }

            const auto pieces = token_split(part);
            for (const auto &piece : pieces) {
                std::u32string utf32_token;
                for (size_t i = 0; i < piece.size(); ++i) {
                    const unsigned char b = static_cast<unsigned char>(piece[i]);
                    utf32_token += m_byte_encoder.at(static_cast<int>(b));
                }

                const std::u32string bpe_strs = bpe(utf32_token);
                size_t start = 0;
                while (true) {
                    const size_t pos = bpe_strs.find(U' ', start);
                    const std::u32string token =
                        (pos == std::u32string::npos) ? bpe_strs.substr(start) : bpe_strs.substr(start, pos - start);

                    if (!token.empty()) {
                        const auto it = m_encoder.find(token);
                        bpe_tokens.push_back(it == m_encoder.end() ? m_unk_token_id : static_cast<Token>(it->second));
                    }

                    if (pos == std::u32string::npos) {
                        break;
                    }
                    start = pos + 1;
                }
            }
        }

        return bpe_tokens;
    }

    // 将 token/weight 按 SD chunk 规则补齐：
    // 每块长度 max_length，插入 BOS/EOS，并在末尾 padding。
    void pad_tokens(std::vector<Token> &tokens, std::vector<float> &weights, size_t max_length = 0, bool padding = false)
        const {
        if (max_length == 0 || !padding) {
            return;
        }

        POWERSERVE_ASSERT_CONFIG(
            tokens.size() == weights.size(),
            kTag,
            "clip token/weight size mismatch: {} vs {}",
            tokens.size(),
            weights.size()
        );

        size_t n = static_cast<size_t>(std::ceil(tokens.size() * 1.0 / (max_length - 2)));
        if (n == 0) {
            n = 1;
        }
        const size_t length = max_length * n;

        std::vector<Token> new_tokens;
        std::vector<float> new_weights;
        new_tokens.push_back(m_bos_token_id);
        new_weights.push_back(1.0f);
        size_t token_idx = 0;
        for (size_t i = 1; i < length; ++i) {
            if (token_idx >= tokens.size()) {
                break;
            }
            if (i % max_length == 0) {
                new_tokens.push_back(m_bos_token_id);
                new_weights.push_back(1.0f);
            } else if (i % max_length == max_length - 1) {
                new_tokens.push_back(m_eos_token_id);
                new_weights.push_back(1.0f);
            } else {
                new_tokens.push_back(tokens[token_idx]);
                new_weights.push_back(weights[token_idx]);
                token_idx += 1;
            }
        }

        new_tokens.push_back(m_eos_token_id);
        new_weights.push_back(1.0f);
        tokens = std::move(new_tokens);
        weights = std::move(new_weights);
        tokens.insert(tokens.end(), length - tokens.size(), m_pad_token_id);
        weights.insert(weights.end(), length - weights.size(), 1.0f);
    }
};

// T5 tokenizer 预分词：把空格转 metaspace 形式。
class MetaspacePreTokenizer {
public:
    explicit MetaspacePreTokenizer(std::string replacement = " ", bool add_prefix_space = true) :
        m_replacement(std::move(replacement)), m_add_prefix_space(add_prefix_space) {}

    auto tokenize(const std::string &input) const -> std::string {
        std::string tokens;
        std::stringstream ss(input);

        if (m_add_prefix_space) {
            tokens += m_replacement;
        }

        std::string token;
        bool first = true;
        while (std::getline(ss, token, ' ')) {
            if (!first) {
                tokens += m_replacement + token;
            } else {
                tokens += token;
            }
            first = false;
        }

        return tokens;
    }

private:
    std::string m_replacement;
    bool m_add_prefix_space = true;
};

// T5 Unigram tokenizer（基于 trie + 最优路径搜索）。
class T5UniGramTokenizer {
public:
    enum Status {
        OK,
        NO_PIECES_LOADED,
        NO_ENTRY_FOUND,
        BUILD_DOUBLE_ARRAY_FAILED,
        INVALID_JSON,
    };

    explicit T5UniGramTokenizer(const std::string &tokenizer_json) {
        initialize_pieces(tokenizer_json);

        m_min_score = FLT_MAX;
        m_max_score = FLT_MIN;

        std::vector<std::pair<std::string, int>> pieces;
        pieces.reserve(m_piece_score_pairs.size());
        for (size_t i = 0; i < m_piece_score_pairs.size(); ++i) {
            const auto &piece_score = m_piece_score_pairs[i];
            m_min_score = std::min(m_min_score, piece_score.second);
            m_max_score = std::max(m_max_score, piece_score.second);
            pieces.emplace_back(piece_score.first, static_cast<int>(i));
        }

        build_trie(&pieces);
        POWERSERVE_ASSERT_CONFIG(
            m_status == OK,
            kTag,
            "failed to initialize T5 tokenizer, status={}",
            static_cast<int>(m_status)
        );
    }

    // 文本编码主入口，可选自动追加 eos。
    auto encode(const std::string &input, bool append_eos_if_not_present = true) const -> std::vector<Token> {
        std::string normalized = normalize(input);
        normalized = m_pre_tokenizer.tokenize(normalized);
        auto encoded = encode_optimized(normalized);

        if (!encoded.empty() && append_eos_if_not_present) {
            const auto &last = encoded.back();
            if (last.first != m_eos_token) {
                encoded.emplace_back(m_eos_token, m_eos_id);
            }
        }

        std::vector<Token> tokens;
        tokens.reserve(encoded.size());
        for (const auto &item : encoded) {
            tokens.push_back(static_cast<Token>(item.second));
        }
        return tokens;
    }

    auto encode_sd3_with_weights(const std::string &input, size_t max_length = 77, bool padding = true) const
        -> std::pair<std::vector<Token>, std::vector<float>> {
        std::vector<Token> tokens;
        std::vector<float> weights;
        for (const auto &item : parse_prompt_attention(input)) {
            auto curr = encode(item.first, true);
            tokens.insert(tokens.end(), curr.begin(), curr.end());
            weights.insert(weights.end(), curr.size(), item.second);
        }
        pad_tokens(tokens, weights, max_length, padding);
        return {std::move(tokens), std::move(weights)};
    }

    auto encode_sd3(const std::string &input, size_t max_length = 77, bool padding = true) const -> std::vector<Token> {
        return encode_sd3_with_weights(input, max_length, padding).first;
    }

private:
    using EncodeResult = std::vector<std::pair<std::string, int>>;

    MetaspacePreTokenizer m_pre_tokenizer;
    std::vector<std::pair<std::string, float>> m_piece_score_pairs;

    float m_min_score = 0.0f;
    float m_max_score = 0.0f;
    std::unique_ptr<Darts::DoubleArray> m_trie;

    int m_trie_results_size = 0;
    int m_unk_id = 2;
    std::string m_eos_token = "</s>";
    int m_eos_id = 1;

    Status m_status = OK;
    float m_unk_penalty = 10.0f;

private:
    // 从 tokenizer JSON 中提取 vocab、unk_id、pre_tokenizer 配置。
    void initialize_pieces(const std::string &json_str) {
        nlohmann::json data;
        try {
            data = nlohmann::json::parse(json_str);
        } catch (const nlohmann::json::parse_error &) {
            m_status = INVALID_JSON;
            return;
        }

        if (!data.contains("model") || !data.at("model").contains("vocab")) {
            m_status = INVALID_JSON;
            return;
        }

        const auto &model = data.at("model");
        if (model.contains("unk_id") && model.at("unk_id").is_number_integer()) {
            m_unk_id = model.at("unk_id").get<int>();
        }

        std::string replacement = " ";
        bool add_prefix_space = true;
        if (data.contains("pre_tokenizer") && data.at("pre_tokenizer").is_object()) {
            const auto &pre_tokenizer = data.at("pre_tokenizer");
            replacement = pre_tokenizer.value("replacement", replacement);
            add_prefix_space = pre_tokenizer.value("add_prefix_space", add_prefix_space);
        }
        m_pre_tokenizer = MetaspacePreTokenizer(replacement, add_prefix_space);

        const auto &vocab = model.at("vocab");
        if (!vocab.is_array()) {
            m_status = INVALID_JSON;
            return;
        }

        m_piece_score_pairs.clear();
        m_piece_score_pairs.reserve(vocab.size());
        for (const auto &item : vocab) {
            if (!item.is_array() || item.size() != 2 || !item.at(0).is_string() || !item.at(1).is_number()) {
                m_status = INVALID_JSON;
                return;
            }

            std::string piece = item.at(0).get<std::string>();
            if (piece.empty()) {
                piece = "<empty_token>";
            }

            const float score = item.at(1).get<float>();
            m_piece_score_pairs.emplace_back(piece, score);
        }
    }

    // 构建 double-array trie，供前缀匹配加速。
    void build_trie(std::vector<std::pair<std::string, int>> *pieces) {
        if (m_status != OK) {
            return;
        }

        if (pieces->empty()) {
            m_status = NO_PIECES_LOADED;
            return;
        }

        std::sort(pieces->begin(), pieces->end());

        std::vector<const char *> key(pieces->size());
        std::vector<int> value(pieces->size());
        for (size_t i = 0; i < pieces->size(); ++i) {
            key[i] = (*pieces)[i].first.data();
            value[i] = (*pieces)[i].second;
        }

        m_trie = std::make_unique<Darts::DoubleArray>();
        if (m_trie->build(key.size(), const_cast<char **>(&key[0]), nullptr, &value[0]) != 0) {
            m_status = BUILD_DOUBLE_ARRAY_FAILED;
            return;
        }

        const int kMaxTrieResultsSize = 1024;
        std::vector<Darts::DoubleArray::result_pair_type> results(kMaxTrieResultsSize);
        m_trie_results_size = 0;
        for (const auto &piece : *pieces) {
            const size_t num_nodes =
                m_trie->commonPrefixSearch(piece.first.data(), results.data(), results.size(), piece.first.size());
            m_trie_results_size = std::max(m_trie_results_size, static_cast<int>(num_nodes));
        }

        if (m_trie_results_size == 0) {
            m_status = NO_ENTRY_FOUND;
        }
    }

    auto get_score(int id) const -> float {
        return m_piece_score_pairs[static_cast<size_t>(id)].second;
    }

    static auto one_char_len(const char *src) -> size_t {
        return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
    }

    // 通过动态规划搜索“总分最高”的子词切分路径。
    auto encode_optimized(const std::string &normalized) const -> EncodeResult {
        if (m_status != OK || normalized.empty()) {
            return {};
        }

        struct BestPathNode {
            int id = -1;
            float best_path_score = 0;
            int starts_at = -1;
        };

        const int size = static_cast<int>(normalized.size());
        const float unk_score = m_min_score - m_unk_penalty;
        std::vector<BestPathNode> best_path_ends_at(static_cast<size_t>(size + 1));

        int starts_at = 0;
        while (starts_at < size) {
            size_t node_pos = 0;
            size_t key_pos = static_cast<size_t>(starts_at);
            const auto best_path_score_till_here = best_path_ends_at[static_cast<size_t>(starts_at)].best_path_score;
            bool has_single_node = false;
            const int mblen = std::min<int>(static_cast<int>(one_char_len(normalized.data() + starts_at)), size - starts_at);

            while (key_pos < static_cast<size_t>(size)) {
                const int ret = m_trie->traverse(normalized.data(), node_pos, key_pos, key_pos + 1);
                if (ret == -2) {
                    break;
                }
                if (ret >= 0) {
                    auto &target = best_path_ends_at[key_pos];
                    const auto length = key_pos - static_cast<size_t>(starts_at);
                    const auto score = get_score(ret);
                    const auto candidate = score + best_path_score_till_here;
                    if (target.starts_at == -1 || candidate > target.best_path_score) {
                        target.best_path_score = static_cast<float>(candidate);
                        target.starts_at = starts_at;
                        target.id = ret;
                    }
                    if (!has_single_node && static_cast<int>(length) == mblen) {
                        has_single_node = true;
                    }
                }
            }

            if (!has_single_node) {
                auto &target = best_path_ends_at[static_cast<size_t>(starts_at + mblen)];
                const auto candidate = unk_score + best_path_score_till_here;
                if (target.starts_at == -1 || candidate > target.best_path_score) {
                    target.best_path_score = candidate;
                    target.starts_at = starts_at;
                    target.id = m_unk_id;
                }
            }

            starts_at += mblen;
        }

        EncodeResult results;
        int ends_at = size;
        while (ends_at > 0) {
            const auto &node = best_path_ends_at[static_cast<size_t>(ends_at)];
            results.emplace_back(normalized.substr(static_cast<size_t>(node.starts_at), static_cast<size_t>(ends_at - node.starts_at)), node.id);
            ends_at = node.starts_at;
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    static auto normalize(const std::string &input) -> std::string {
        return std::regex_replace(input, std::regex(" {2,}"), " ");
    }

    // T5 分块补齐规则：每块 max_length，块尾放 EOS，余量补 0。
    void pad_tokens(std::vector<Token> &tokens, std::vector<float> &weights, size_t max_length = 0, bool padding = false)
        const {
        if (max_length == 0 || !padding || tokens.empty()) {
            return;
        }

        POWERSERVE_ASSERT_CONFIG(
            tokens.size() == weights.size(),
            kTag,
            "t5 token/weight size mismatch: {} vs {}",
            tokens.size(),
            weights.size()
        );

        const size_t orig_token_num = tokens.size() - 1;
        size_t n = static_cast<size_t>(std::ceil(orig_token_num * 1.0 / (max_length - 1)));
        if (n == 0) {
            n = 1;
        }
        const size_t length = max_length * n;

        std::vector<Token> new_tokens;
        std::vector<float> new_weights;
        size_t token_idx = 0;
        for (size_t i = 0; i < length; ++i) {
            if (token_idx >= orig_token_num) {
                break;
            }
            if (i % max_length == max_length - 1) {
                new_tokens.push_back(static_cast<Token>(m_eos_id));
                new_weights.push_back(1.0f);
            } else {
                new_tokens.push_back(tokens[token_idx]);
                new_weights.push_back(weights[token_idx]);
                token_idx += 1;
            }
        }

        new_tokens.push_back(static_cast<Token>(m_eos_id));
        new_weights.push_back(1.0f);
        tokens = std::move(new_tokens);
        weights = std::move(new_weights);
        tokens.insert(tokens.end(), length - tokens.size(), 0);
        weights.insert(weights.end(), length - weights.size(), 1.0f);
    }
};

} // namespace

// SDPromptTokenizer 的 PImpl：
// - clip_l/clip_g 共用 merges 但 padding 约定不同；
// - t5 使用 unigram tokenizer。
struct SDPromptTokenizer::Impl {
    CLIPBPETokenizer clip_l_tokenizer;
    CLIPBPETokenizer clip_g_tokenizer;
    T5UniGramTokenizer t5_tokenizer;

    explicit Impl(const SDTextEncoderVocabPack &vocab_pack) :
        clip_l_tokenizer(vocab_pack.clip_merges, 49407),
        clip_g_tokenizer(vocab_pack.clip_merges, 0),
        t5_tokenizer(vocab_pack.t5_tokenizer_json) {}

    // 对单个 prompt 同时产出三条分支 token + weight。
    auto encode_prompt(const std::string &prompt) const -> SDTokenIdPack {
        SDTokenIdPack out;
        auto clip_l = clip_l_tokenizer.encode_sd3_with_weights(prompt, 77, true);
        out.clip_l = std::move(clip_l.first);
        out.clip_l_weights = std::move(clip_l.second);

        auto clip_g = clip_g_tokenizer.encode_sd3_with_weights(prompt, 77, true);
        out.clip_g = std::move(clip_g.first);
        out.clip_g_weights = std::move(clip_g.second);

        auto t5 = t5_tokenizer.encode_sd3_with_weights(prompt, 77, true);
        out.t5 = std::move(t5.first);
        out.t5_weights = std::move(t5.second);
        return out;
    }
};

SDPromptTokenizer::SDPromptTokenizer(const SDTextEncoderVocabPack &vocab_pack) :
    m_impl(std::make_unique<Impl>(vocab_pack)) {}

SDPromptTokenizer::~SDPromptTokenizer() = default;

auto SDPromptTokenizer::encode_prompt(const std::string &prompt) const -> SDTokenIdPack {
    POWERSERVE_ASSERT_CONFIG(m_impl != nullptr, kTag, "tokenizer impl is null");
    return m_impl->encode_prompt(prompt);
}

auto SDPromptTokenizer::encode_prompt_pair(const std::string &prompt, const std::string &negative_prompt) const
    -> SDPromptTokenization {
    // 正负提示词分别编码，保持彼此独立。
    SDPromptTokenization out;
    out.prompt = encode_prompt(prompt);
    out.negative_prompt = encode_prompt(negative_prompt);
    return out;
}

} // namespace powerserve
