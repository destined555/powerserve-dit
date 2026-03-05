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

#include "vae/vae.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../libs/stb_headers/stb/stb_image_write.h"

namespace powerserve {

namespace {

// 将 [0, 1] 的浮点值映射到 [0, 255] 的 8-bit 像素。
// - 超出范围会被 clamp；
// - +0.5f 用于四舍五入，降低直接截断导致的系统偏差。
auto to_u8(float value) -> uint8_t {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
}

// 判断当前解码结果是否满足导出 PNG 的最小条件。
// 这里限制 batch=1, channels=3，是因为当前仅支持单张 RGB 图像导出。
auto can_export_png(const SD35VAEDecodeResult &decoded) -> bool {
    if (decoded.batch != 1 || decoded.channels != 3) {
        return false;
    }
    if (decoded.width <= 0 || decoded.height <= 0) {
        return false;
    }
    const size_t expect = static_cast<size_t>(decoded.width) * static_cast<size_t>(decoded.height) * 3;
    return decoded.data.size() == expect;
}

} // namespace

void write_f32_tensor_to_bin(const Path &path, const std::vector<float> &data) {
    // 自动创建父目录，避免调用方必须预先 mkdir。
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("failed to open output bin path: " + path.string());
    }
    fout.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!fout) {
        throw std::runtime_error("failed to write output bin payload: " + path.string());
    }
}

void write_decoded_png(const Path &output_path, const SD35VAEDecodeResult &decoded) {
    // PNG 导出只支持单 batch 的 RGB 图像，其他情况直接报错防止误用。
    if (decoded.batch != 1 || decoded.channels != 3) {
        throw std::runtime_error("decoded tensor must be batch=1, channels=3 for PNG export");
    }
    if (decoded.width <= 0 || decoded.height <= 0) {
        throw std::runtime_error("decoded tensor has invalid image size");
    }

    const int width = static_cast<int>(decoded.width);
    const int height = static_cast<int>(decoded.height);
    const size_t plane_size = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t expect = plane_size * 3;
    if (decoded.data.size() != expect) {
        throw std::runtime_error("decoded tensor data size mismatch for PNG export");
    }

    // decoded.data 的布局是平面形式 [W, H, C, B]：
    // - 每个通道 (R/G/B) 连续存储一个平面；
    // - PNG 写入接口要求 HWC 交错布局（RGBRGB...）。
    // 因此这里将三个平面重排为交错像素缓冲区。
    std::vector<uint8_t> pixels(expect, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pos = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            pixels[3 * pos + 0] = to_u8(decoded.data[pos + plane_size * 0]);
            pixels[3 * pos + 1] = to_u8(decoded.data[pos + plane_size * 1]);
            pixels[3 * pos + 2] = to_u8(decoded.data[pos + plane_size * 2]);
        }
    }

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    const int ok = stbi_write_png(output_path.string().c_str(), width, height, 3, pixels.data(), width * 3);
    if (ok == 0) {
        throw std::runtime_error("failed to write png");
    }
}

auto export_sd35_vae_compare_outputs(const Path &output_dir, const SD35VAEDecodeResult &decoded)
    -> SD35VAECompareOutputs {
    SD35VAECompareOutputs outputs;
    // 与对比脚本约定的固定文件名，便于自动化流程复用。
    outputs.decoded_bin_path = output_dir / "powerserve_decoded.bin";

    write_f32_tensor_to_bin(outputs.decoded_bin_path, decoded.data);
    // 不是所有输出都能写 PNG（例如 batch>1），因此按条件导出。
    if (can_export_png(decoded)) {
        outputs.png_path = output_dir / "run_sd_latent_vae.png";
        write_decoded_png(outputs.png_path, decoded);
    }
    return outputs;
}

} // namespace powerserve
