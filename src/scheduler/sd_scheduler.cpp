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

#include "scheduler/sd_scheduler.hpp"

#include "core/exception.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <utility>

namespace powerserve {
namespace {

// 内部日志/常量配置。
constexpr const char *kTag      = "SDScheduler";
constexpr int kTimesteps        = 1000;
constexpr int kMaxDiscreteIndex = kTimesteps - 1;
constexpr float kPi             = 3.14159265358979323846f;

// 去除首尾空白，便于后续对字符串参数做鲁棒解析。
auto trim_copy(std::string text) -> std::string {
    auto is_space = [](const unsigned char ch) {
        return std::isspace(ch) != 0;
    };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [&](char ch) {
                   return !is_space(static_cast<unsigned char>(ch));
               }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [&](char ch) {
                   return !is_space(static_cast<unsigned char>(ch));
               }).base(),
               text.end());
    return text;
}

// 转小写，统一策略名比较方式。
auto lower_copy(std::string text) -> std::string {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

// 统一参数格式：trim + lower。
auto normalize(std::string text) -> std::string {
    return lower_copy(trim_copy(std::move(text)));
}

// 生成等间距序列（与 sd.cpp 中 linear_space 行为对齐）。
auto linear_space(float start, float end, size_t num_points) -> std::vector<float> {
    std::vector<float> result(num_points, start);
    if (num_points == 0) {
        return result;
    }
    if (num_points == 1) {
        result[0] = start;
        return result;
    }

    const float inc = (end - start) / static_cast<float>(num_points - 1);
    for (size_t i = 1; i < num_points; ++i) {
        result[i] = result[i - 1] + inc;
    }
    return result;
}

// SD3 Flow 使用的时间域非线性变换。
auto time_snr_shift(float alpha, float t) -> float {
    if (alpha == 1.0f) {
        return t;
    }
    return alpha * t / (1.0f + (alpha - 1.0f) * t);
}

// SD3 Flow 的 t -> sigma 映射，保持与 sd.cpp 完全对齐。
auto sd3_flow_t_to_sigma(float shift, float t) -> float {
    // 与 sd.cpp::DiscreteFlowDenoiser::t_to_sigma 保持一致。
    return time_snr_shift(shift, (t + 1.0f) / static_cast<float>(kTimesteps));
}

auto flow_sigma_min(float shift) -> float {
    // 对齐 DiscreteFlowDenoiser::set_parameters() 的首元素（i=1）。
    return sd3_flow_t_to_sigma(shift, 1.0f);
}

auto flow_sigma_max(float shift) -> float {
    // 对齐 DiscreteFlowDenoiser::set_parameters() 的末元素（i=1000）。
    return sd3_flow_t_to_sigma(shift, static_cast<float>(kTimesteps));
}

// -------- 各种 sigma 生成策略实现 --------

// discrete：在离散 timestep 上线性采样，再映射到 sigma。
auto build_discrete_flow_sigmas(uint32_t sample_steps, float shift) -> std::vector<float> {
    std::vector<float> result;
    if (sample_steps == 0) {
        return result;
    }

    if (sample_steps == 1) {
        result.push_back(sd3_flow_t_to_sigma(shift, static_cast<float>(kMaxDiscreteIndex)));
        result.push_back(0.0f);
        return result;
    }

    result.reserve(static_cast<size_t>(sample_steps) + 1);
    const float step = static_cast<float>(kMaxDiscreteIndex) / static_cast<float>(sample_steps - 1);
    for (uint32_t i = 0; i < sample_steps; ++i) {
        const float t = static_cast<float>(kMaxDiscreteIndex) - step * static_cast<float>(i);
        result.push_back(sd3_flow_t_to_sigma(shift, t));
    }
    result.push_back(0.0f);
    return result;
}

// exponential：在 log(sigma) 域线性采样。
auto build_exponential_sigmas(uint32_t sample_steps, float sigma_min, float sigma_max) -> std::vector<float> {
    std::vector<float> sigmas;
    // 与 sd.cpp::ExponentialScheduler 保持算术一致（包括边界行为）。
    float log_sigma_min = std::log(sigma_min);
    float log_sigma_max = std::log(sigma_max);
    float step          = (log_sigma_max - log_sigma_min) / (sample_steps - 1);

    for (uint32_t i = 0; i < sample_steps; ++i) {
        float sigma = std::exp(log_sigma_max - step * i);
        sigmas.push_back(sigma);
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

// karras：Karras 论文常用的 rho 曲线重参数化。
auto build_karras_sigmas(uint32_t sample_steps, float sigma_min, float sigma_max) -> std::vector<float> {
    // 与 sd.cpp::KarrasScheduler 保持算术一致（包括边界行为）。
    std::vector<float> result(sample_steps + 1);
    if (sigma_min <= 1e-6f) {
        sigma_min = 1e-6f;
    }

    float rho = 7.f;
    float min_inv_rho = pow(sigma_min, (1.f / rho));
    float max_inv_rho = pow(sigma_max, (1.f / rho));
    for (uint32_t i = 0; i < sample_steps; ++i) {
        result[i] = pow(
            max_inv_rho + (float)i / ((float)sample_steps - 1.f) * (min_inv_rho - max_inv_rho),
            rho
        );
    }
    result[sample_steps] = 0.0f;
    return result;
}

// sgm_uniform：在 timestep 域均匀采样 n+1 点，取前 n 点映射后再补 0。
auto build_sgm_uniform_sigmas(uint32_t sample_steps, float shift) -> std::vector<float> {
    std::vector<float> result;
    if (sample_steps == 0) {
        result.push_back(0.0f);
        return result;
    }

    result.reserve(static_cast<size_t>(sample_steps) + 1);
    const std::vector<float> timesteps = linear_space(static_cast<float>(kMaxDiscreteIndex), 0.0f, sample_steps + 1);
    for (uint32_t i = 0; i < sample_steps; ++i) {
        result.push_back(sd3_flow_t_to_sigma(shift, timesteps[i]));
    }
    result.push_back(0.0f);
    return result;
}

// simple：按固定步长从 1000 个训练步中稀疏采样。
auto build_simple_sigmas(uint32_t sample_steps, float shift) -> std::vector<float> {
    std::vector<float> result;
    if (sample_steps == 0) {
        return result;
    }

    result.reserve(static_cast<size_t>(sample_steps) + 1);
    const float step_factor = static_cast<float>(kTimesteps) / static_cast<float>(sample_steps);
    for (uint32_t i = 0; i < sample_steps; ++i) {
        int offset_from_start = static_cast<int>(static_cast<float>(i) * step_factor);
        int timestep_index    = kTimesteps - 1 - offset_from_start;
        if (timestep_index < 0) {
            timestep_index = 0;
        }
        result.push_back(sd3_flow_t_to_sigma(shift, static_cast<float>(timestep_index)));
    }
    result.push_back(0.0f);
    return result;
}

// smoothstep：先对采样位置做 smoothstep，再映射 timestep。
constexpr auto smoothstep(float x) -> float {
    return x * x * (3.0f - 2.0f * x);
}

auto build_smoothstep_sigmas(uint32_t sample_steps, float shift) -> std::vector<float> {
    std::vector<float> result;
    if (sample_steps == 0) {
        return result;
    }
    if (sample_steps == 1) {
        result.push_back(sd3_flow_t_to_sigma(shift, static_cast<float>(kMaxDiscreteIndex)));
        result.push_back(0.0f);
        return result;
    }

    result.reserve(static_cast<size_t>(sample_steps) + 1);
    for (uint32_t i = 0; i < sample_steps; ++i) {
        const float u = 1.0f - static_cast<float>(i) / static_cast<float>(sample_steps);
        result.push_back(sd3_flow_t_to_sigma(shift, std::round(smoothstep(u) * static_cast<float>(kMaxDiscreteIndex))));
    }
    result.push_back(0.0f);
    return result;
}

// lcm：按 LCM 训练步规则（50 基准步）回推 sigma。
auto build_lcm_sigmas(uint32_t sample_steps, float shift) -> std::vector<float> {
    std::vector<float> result;
    result.reserve(static_cast<size_t>(sample_steps) + 1);

    if (sample_steps == 0) {
        result.push_back(0.0f);
        return result;
    }

    const int original_steps = 50;
    const int k              = kTimesteps / original_steps;
    for (uint32_t i = 0; i < sample_steps; ++i) {
        const int index    = static_cast<int>((i * static_cast<uint32_t>(original_steps)) / sample_steps);
        const int timestep = (original_steps - index) * k - 1;
        result.push_back(sd3_flow_t_to_sigma(shift, static_cast<float>(timestep)));
    }
    result.push_back(0.0f);
    return result;
}

// kl_optimal：在 atan(sigma) 空间线性采样，再 tan 回 sigma。
auto build_kl_optimal_sigmas(uint32_t sample_steps, float sigma_min, float sigma_max) -> std::vector<float> {
    std::vector<float> sigmas;
    if (sample_steps == 0) {
        return sigmas;
    }
    if (sample_steps == 1) {
        sigmas.push_back(sigma_max);
        sigmas.push_back(0.0f);
        return sigmas;
    }

    if (sigma_min <= 1e-6f) {
        sigma_min = 1e-6f;
    }

    sigmas.reserve(static_cast<size_t>(sample_steps) + 1);
    const float alpha_min = std::atan(sigma_min);
    const float alpha_max = std::atan(sigma_max);

    for (uint32_t i = 0; i < sample_steps; ++i) {
        const float t     = static_cast<float>(i) / static_cast<float>(sample_steps - 1);
        const float angle = t * alpha_min + (1.0f - t) * alpha_max;
        sigmas.push_back(std::tan(angle));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

// bong_tangent 的分段曲线基础实现。
auto get_bong_tangent_sigmas(int steps, float slope, float pivot, float start, float end) -> std::vector<float> {
    std::vector<float> sigmas;
    if (steps <= 0) {
        return sigmas;
    }

    const float smax   = ((2.0f / kPi) * std::atan(-slope * (0.0f - pivot)) + 1.0f) * 0.5f;
    const float smin   = ((2.0f / kPi) * std::atan(-slope * ((static_cast<float>(steps) - 1.0f) - pivot)) + 1.0f) * 0.5f;
    const float srange = smax - smin;
    const float sscale = start - end;

    sigmas.reserve(static_cast<size_t>(steps));
    if (std::fabs(srange) < 1e-8f) {
        if (steps == 1) {
            sigmas.push_back(start);
            return sigmas;
        }
        for (int i = 0; i < steps; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(steps - 1);
            sigmas.push_back(start + (end - start) * t);
        }
        return sigmas;
    }

    const float inv_srange = 1.0f / srange;
    for (int x = 0; x < steps; ++x) {
        const float v     = ((2.0f / kPi) * std::atan(-slope * (static_cast<float>(x) - pivot)) + 1.0f) * 0.5f;
        const float sigma = ((v - smin) * inv_srange) * sscale + end;
        sigmas.push_back(sigma);
    }
    return sigmas;
}

// bong_tangent：两段切线曲线拼接后的 sigma 计划。
auto build_bong_tangent_sigmas(uint32_t sample_steps, float sigma_min, float sigma_max) -> std::vector<float> {
    std::vector<float> result;
    if (sample_steps == 0) {
        return result;
    }

    const float start  = sigma_max;
    const float end    = sigma_min;
    const float middle = sigma_min + (sigma_max - sigma_min) * 0.5f;

    const float pivot_1 = 0.6f;
    const float pivot_2 = 0.6f;
    float slope_1       = 0.2f;
    float slope_2       = 0.2f;

    const int steps     = static_cast<int>(sample_steps) + 2;
    const int midpoint  = static_cast<int>(((static_cast<float>(steps) * pivot_1) + (static_cast<float>(steps) * pivot_2)) * 0.5f);
    const int pivot_1_i = static_cast<int>(static_cast<float>(steps) * pivot_1);
    const int pivot_2_i = static_cast<int>(static_cast<float>(steps) * pivot_2);

    const float slope_scale = static_cast<float>(steps) / 40.0f;
    slope_1                 = slope_1 / slope_scale;
    slope_2                 = slope_2 / slope_scale;

    const int stage_2_len = steps - midpoint;
    const int stage_1_len = steps - stage_2_len;

    std::vector<float> sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, static_cast<float>(pivot_1_i), start, middle);
    std::vector<float> sigmas_2 =
        get_bong_tangent_sigmas(stage_2_len, slope_2, static_cast<float>(pivot_2_i - stage_1_len), middle, end);

    if (!sigmas_1.empty()) {
        sigmas_1.pop_back();
    }

    result.reserve(static_cast<size_t>(sample_steps) + 1);
    result.insert(result.end(), sigmas_1.begin(), sigmas_1.end());
    result.insert(result.end(), sigmas_2.begin(), sigmas_2.end());

    if (result.size() < static_cast<size_t>(sample_steps) + 1) {
        while (result.size() < static_cast<size_t>(sample_steps) + 1) {
            result.push_back(end);
        }
    } else if (result.size() > static_cast<size_t>(sample_steps) + 1) {
        result.resize(static_cast<size_t>(sample_steps) + 1);
    }

    result[sample_steps] = 0.0f;
    return result;
}

// 解析并校验 sample_method，未指定时默认 discrete。
auto resolve_sigma_method(const std::string &sample_method) -> std::string {
    std::string method = normalize(sample_method);
    if (method.empty()) {
        return "discrete";
    }
    if (method == "bong-tangent") {
        method = "bong_tangent";
    }

    static constexpr std::array<const char *, 9> kSupportedMethods = {
        "discrete",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "smoothstep",
        "kl_optimal",
        "lcm",
        "bong_tangent",
    };
    for (const char *name : kSupportedMethods) {
        if (method == name) {
            return method;
        }
    }

    POWERSERVE_ASSERT_CONFIG(
        false,
        kTag,
        "unsupported sample_method '{}'; supported values are: [discrete, karras, exponential, sgm_uniform, simple, "
        "smoothstep, kl_optimal, lcm, bong_tangent]",
        sample_method
    );
    return "discrete";
}

// 根据策略名分发到具体 sigma 生成函数。
auto build_sd3_flow_sigmas(const std::string &method, uint32_t sample_steps, float flow_shift) -> std::vector<float> {
    const float sigma_min = flow_sigma_min(flow_shift);
    const float sigma_max = flow_sigma_max(flow_shift);

    if (method == "discrete") {
        return build_discrete_flow_sigmas(sample_steps, flow_shift);
    }
    if (method == "karras") {
        return build_karras_sigmas(sample_steps, sigma_min, sigma_max);
    }
    if (method == "exponential") {
        return build_exponential_sigmas(sample_steps, sigma_min, sigma_max);
    }
    if (method == "sgm_uniform") {
        return build_sgm_uniform_sigmas(sample_steps, flow_shift);
    }
    if (method == "simple") {
        return build_simple_sigmas(sample_steps, flow_shift);
    }
    if (method == "smoothstep") {
        return build_smoothstep_sigmas(sample_steps, flow_shift);
    }
    if (method == "kl_optimal") {
        return build_kl_optimal_sigmas(sample_steps, sigma_min, sigma_max);
    }
    if (method == "lcm") {
        return build_lcm_sigmas(sample_steps, flow_shift);
    }
    if (method == "bong_tangent") {
        return build_bong_tangent_sigmas(sample_steps, sigma_min, sigma_max);
    }

    POWERSERVE_ASSERT_CONFIG(false, kTag, "internal error: unhandled sample_method '{}'", method);
    return {};
}

} // namespace

// 对外入口：完成参数规范化、默认值填充和策略分发，返回完整调度结果。
auto build_sd35_scheduler(const SDSchedulerRequest &request) -> SDSchedulerSchedule {
    const std::string model_arch = normalize(request.model_arch);
    POWERSERVE_ASSERT_CONFIG(
        model_arch == "sd3",
        kTag,
        "minimal scheduler migration only supports model_arch=sd3, but got '{}'",
        request.model_arch
    );

    const std::string sigma_method = resolve_sigma_method(request.sample_method);

    float flow_shift = request.flow_shift;
    if (!std::isfinite(flow_shift)) {
        flow_shift = 3.0f;
    }

    SDSchedulerSchedule schedule;
    schedule.resolved_prediction = "sd3_flow";
    schedule.resolved_scheduler  = sigma_method;
    schedule.resolved_flow_shift = flow_shift;
    schedule.sigmas              = build_sd3_flow_sigmas(sigma_method, request.sample_steps, flow_shift);
    return schedule;
}

} // namespace powerserve
