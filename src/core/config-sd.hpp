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

#include <cstdint>
#include <limits>
#include <string>

namespace powerserve {

struct ComponentConfig {
    std::string format;
    std::string path;
};

struct SDHyperParams {
    uint32_t n_threads   = 4;
    uint32_t batch_size  = 128;
    uint32_t width       = 256;
    uint32_t height      = 256;
    uint32_t sample_step = 20;

    float cfg_scale = 4.5f;
    float strength  = 0.75f;

    int64_t seed = 42;

    std::string sample_method = "euler";
    std::string scheduler     = "discrete";

    std::string prompt;
    std::string negative_prompt;

    SDHyperParams() = default;
    SDHyperParams(const Path &params_file);

    ~SDHyperParams() noexcept = default;
};

struct TextEncodeConfig {
    ComponentConfig clip_l;
    ComponentConfig clip_g;
    ComponentConfig t5xxl;

    bool keep_on_cpu = true;
    int clip_skip    = 1;

    TextEncodeConfig() = default;

    ~TextEncodeConfig() noexcept = default;
};

struct SDModelConfig {
    struct QuantizationConfig {
        std::string out_type;
    };

    struct QNNConfig {
        bool enabled          = false;
        std::string workspace = "qnn";
    };

    uint32_t version = 1;
    std::string model_id;
    std::string arch;
    std::string pipeline;

    ComponentConfig weights;

    bool offload_params_to_cpu = false;
    bool flash_attn            = false;
    bool diffusion_flash_attn  = false;
    bool diffusion_conv_direct = false;
    float flow_shift           = std::numeric_limits<float>::infinity();

    QuantizationConfig quantization;
    QNNConfig qnn;

    SDModelConfig() = default;

    ~SDModelConfig() noexcept = default;
};

struct VAEConfig {
    struct TilingConfig {
        bool enabled         = false;
        int tile_size_x      = 0;
        int tile_size_y      = 0;
        float target_overlap = 0.5f;
    };

    ComponentConfig vae;

    bool decode_only                = false;
    bool keep_on_cpu                = false;
    bool conv_direct                = false;
    bool force_sdxl_vae_conv_scale  = false;
    TilingConfig tiling;

    VAEConfig() = default;

    ~VAEConfig() noexcept = default;
};

struct SDConfig {
    SDHyperParams hyper_params;
    TextEncodeConfig text_encode_config;
    SDModelConfig sd_model_config;
    VAEConfig vae_config;

    SDConfig() = default;
    SDConfig(const Path &model_config_file);
    SDConfig(const Path &model_config_file, const Path &hparams_file);

    ~SDConfig() noexcept = default;
};

} // namespace powerserve
