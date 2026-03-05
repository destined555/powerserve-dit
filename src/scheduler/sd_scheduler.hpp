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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace powerserve {

// Scheduler 输入参数：
// - model_arch: 目前仅支持 "sd3"
// - sample_steps: 采样步数，通常对应输出 sigma 数组长度约为 sample_steps + 1
// - sample_method: sigma 生成策略名（如 discrete/karras/exponential 等）
// - scheduler: 预留字段，当前逻辑主要以 sample_method 为准
// - flow_shift: SD3 flow 公式的 shift 参数，非有限值会在实现中回落为默认值
struct SDSchedulerRequest {
    std::string model_arch;
    uint32_t sample_steps = 20;

    std::string sample_method;
    std::string scheduler;

    float flow_shift = std::numeric_limits<float>::infinity();
};

// Scheduler 解析结果：
// - resolved_prediction: 最终预测类型（当前 SD3 路径固定为 "sd3_flow"）
// - resolved_scheduler: 最终生效的 sigma 策略名
// - resolved_flow_shift: 最终生效的 flow_shift
// - sigmas: 供后续去噪迭代使用的 sigma 序列
struct SDSchedulerSchedule {
    std::string resolved_prediction;
    std::string resolved_scheduler;
    float resolved_flow_shift = 0.0f;
    // 噪声序列
    std::vector<float> sigmas;
};

// 根据请求生成 SD3.5 调度结果（含策略解析与 sigma 序列）。
auto build_sd35_scheduler(const SDSchedulerRequest &request) -> SDSchedulerSchedule;

} // namespace powerserve
