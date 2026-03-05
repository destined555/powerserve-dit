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
#include "vae/sd35_vae.hpp"

#include <vector>

namespace powerserve {

// VAE 对比导出结果：
// 1) decoded_bin_path: 原始浮点输出（F32，按 [W, H, C, B] 平面布局）；
// 2) png_path: 可视化图片（仅在满足 batch=1 且 channels=3 时导出）。
struct SD35VAECompareOutputs {
    Path png_path;
    Path decoded_bin_path;
};

// 将 F32 张量原样写入二进制文件，不做任何重排/归一化。
// 该文件用于与 sd.cpp 的输出做逐元素对比（例如 bitwise_equal 检查）。
void write_f32_tensor_to_bin(const Path &path, const std::vector<float> &data);

// 将解码结果写成 PNG：
// - 输入要求：decoded.batch == 1 且 decoded.channels == 3；
// - 输入布局：decoded.data 为 [W, H, C, B] 的平面布局；
// - 输出布局：PNG 需要 HWC 交错 RGB，因此函数内部会做一次布局转换。
void write_decoded_png(const Path &output_path, const SD35VAEDecodeResult &decoded);

// 在一个目录下统一导出对比所需产物：
// - 必定导出 `powerserve_decoded.bin`；
// - 当形状满足可视化条件时额外导出 `run_sd_latent_vae.png`。
auto export_sd35_vae_compare_outputs(const Path &output_dir, const SD35VAEDecodeResult &decoded)
    -> SD35VAECompareOutputs;

} // namespace powerserve
