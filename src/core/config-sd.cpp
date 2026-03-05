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

#include "core/config-sd.hpp"

#include "core/exception.hpp"
#include "nlohmann/json.hpp"

#include <fstream>
#include <string>

namespace powerserve {

namespace {
using nlohmann::json;

ComponentConfig parse_component(const json &component_j) {
    ComponentConfig component;
    component_j.at("format").get_to(component.format);
    component_j.at("path").get_to(component.path);
    return component;
}

ComponentConfig parse_component_from_components(const json &root, const char *name, const char *tag) {
    POWERSERVE_ASSERT_CONFIG(root.contains("components"), tag, "missing required key: components");
    auto const &components = root.at("components");
    POWERSERVE_ASSERT_CONFIG(components.contains(name), tag, "missing required component: {}", name);
    return parse_component(components.at(name));
}

json parse_json_file(const Path &json_file, const char *tag) {
    json j;
    std::ifstream file(json_file);
    POWERSERVE_ASSERT_CONFIG(file.good(), tag, "failed to open json config file {}", json_file);

    try {
        file >> j;
    } catch (const std::exception &err) {
        throw ConfigException(tag, fmt::format("failed parsing json config file {}:\n{}", json_file, err.what()));
    }
    return j;
}

void parse_hyper_params_from_json(SDHyperParams &hyper_params, const json &j) {
    hyper_params.n_threads  = j.value("n_threads", hyper_params.n_threads);
    hyper_params.batch_size = j.value("batch_size", hyper_params.batch_size);

    hyper_params.width  = j.value("width", hyper_params.width);
    hyper_params.height = j.value("height", hyper_params.height);

    hyper_params.sample_step = j.value("sample_step", hyper_params.sample_step);
    hyper_params.sample_step = j.value("sample_steps", hyper_params.sample_step);
    hyper_params.sample_step = j.value("steps", hyper_params.sample_step);

    hyper_params.cfg_scale = j.value("cfg_scale", hyper_params.cfg_scale);
    hyper_params.cfg_scale = j.value("guidance_scale", hyper_params.cfg_scale);

    hyper_params.strength = j.value("strength", hyper_params.strength);
    hyper_params.seed     = j.value("seed", hyper_params.seed);

    hyper_params.sample_method = j.value("sample_method", hyper_params.sample_method);
    hyper_params.sample_method = j.value("sampling_method", hyper_params.sample_method);
    hyper_params.scheduler     = j.value("scheduler", hyper_params.scheduler);

    hyper_params.prompt = j.value("prompt", hyper_params.prompt);
    hyper_params.prompt = j.value("prompts", hyper_params.prompt);
    hyper_params.prompt = j.value("positive_prompt", hyper_params.prompt);

    hyper_params.negative_prompt = j.value("negative_prompt", hyper_params.negative_prompt);
    hyper_params.negative_prompt = j.value("nprompt", hyper_params.negative_prompt);
    hyper_params.negative_prompt = j.value("nprompts", hyper_params.negative_prompt);
}

void parse_text_encode_config(TextEncodeConfig &text_encode_config, const json &root) {
    const bool has_text_cfg = root.contains("text_encode_config") && root.at("text_encode_config").is_object();
    auto const &text_cfg    = has_text_cfg ? root.at("text_encode_config") : json::object();

    std::string model_arch = root.value("model_arch", std::string());
    if (root.contains("sd_model_config") && root.at("sd_model_config").is_object()) {
        model_arch = root.at("sd_model_config").value("model_arch", model_arch);
    }
    if (model_arch == "sd3") {
        text_encode_config.clip_skip = 2;
    }

    text_encode_config.keep_on_cpu = text_cfg.value("keep_on_cpu", text_encode_config.keep_on_cpu);
    text_encode_config.keep_on_cpu = text_cfg.value("clip_on_cpu", text_encode_config.keep_on_cpu);
    text_encode_config.clip_skip   = text_cfg.value("clip_skip", text_encode_config.clip_skip);

    if (text_cfg.contains("clip_l")) {
        text_encode_config.clip_l = parse_component(text_cfg.at("clip_l"));
    } else {
        text_encode_config.clip_l = parse_component_from_components(root, "clip_l", "TextEncodeConfig");
    }

    if (text_cfg.contains("clip_g")) {
        text_encode_config.clip_g = parse_component(text_cfg.at("clip_g"));
    } else {
        text_encode_config.clip_g = parse_component_from_components(root, "clip_g", "TextEncodeConfig");
    }

    if (text_cfg.contains("t5xxl")) {
        text_encode_config.t5xxl = parse_component(text_cfg.at("t5xxl"));
    } else {
        text_encode_config.t5xxl = parse_component_from_components(root, "t5xxl", "TextEncodeConfig");
    }
}

void parse_sd_model_config(SDModelConfig &sd_model_config, const json &root) {
    const bool has_model_cfg = root.contains("sd_model_config") && root.at("sd_model_config").is_object();
    auto const &model_cfg    = has_model_cfg ? root.at("sd_model_config") : json::object();

    sd_model_config.version  = root.value("version", sd_model_config.version);
    sd_model_config.version  = model_cfg.value("version", sd_model_config.version);
    sd_model_config.model_id = root.value("model_id", sd_model_config.model_id);
    sd_model_config.model_id = model_cfg.value("model_id", sd_model_config.model_id);
    sd_model_config.arch     = root.value("model_arch", sd_model_config.arch);
    sd_model_config.arch     = model_cfg.value("model_arch", sd_model_config.arch);
    sd_model_config.pipeline = root.value("pipeline", sd_model_config.pipeline);
    sd_model_config.pipeline = model_cfg.value("pipeline", sd_model_config.pipeline);

    if (model_cfg.contains("weights")) {
        sd_model_config.weights = parse_component(model_cfg.at("weights"));
    } else {
        sd_model_config.weights = parse_component_from_components(root, "weights", "SDModelConfig");
    }

    sd_model_config.offload_params_to_cpu = model_cfg.value("offload_params_to_cpu", sd_model_config.offload_params_to_cpu);
    sd_model_config.offload_params_to_cpu = model_cfg.value("offload_to_cpu", sd_model_config.offload_params_to_cpu);
    sd_model_config.flash_attn            = model_cfg.value("flash_attn", sd_model_config.flash_attn);
    sd_model_config.diffusion_flash_attn  = model_cfg.value("diffusion_flash_attn", sd_model_config.diffusion_flash_attn);
    sd_model_config.diffusion_conv_direct = model_cfg.value("diffusion_conv_direct", sd_model_config.diffusion_conv_direct);
    sd_model_config.flow_shift            = model_cfg.value("flow_shift", sd_model_config.flow_shift);

    if (root.contains("quantization") && root.at("quantization").is_object()) {
        auto const &quantization_j          = root.at("quantization");
        sd_model_config.quantization.out_type = quantization_j.value("out_type", sd_model_config.quantization.out_type);
    }
    if (model_cfg.contains("quantization") && model_cfg.at("quantization").is_object()) {
        auto const &quantization_j          = model_cfg.at("quantization");
        sd_model_config.quantization.out_type = quantization_j.value("out_type", sd_model_config.quantization.out_type);
    }

    if (root.contains("qnn") && root.at("qnn").is_object()) {
        auto const &qnn_j           = root.at("qnn");
        sd_model_config.qnn.enabled = qnn_j.value("enabled", sd_model_config.qnn.enabled);
        sd_model_config.qnn.workspace = qnn_j.value("workspace", sd_model_config.qnn.workspace);
    }
    if (model_cfg.contains("qnn") && model_cfg.at("qnn").is_object()) {
        auto const &qnn_j           = model_cfg.at("qnn");
        sd_model_config.qnn.enabled = qnn_j.value("enabled", sd_model_config.qnn.enabled);
        sd_model_config.qnn.workspace = qnn_j.value("workspace", sd_model_config.qnn.workspace);
    }
}

void parse_vae_config(VAEConfig &vae_config, const json &root) {
    const bool has_vae_cfg = root.contains("vae_config") && root.at("vae_config").is_object();
    auto const &vae_cfg    = has_vae_cfg ? root.at("vae_config") : json::object();

    if (vae_cfg.contains("vae")) {
        vae_config.vae = parse_component(vae_cfg.at("vae"));
    } else {
        vae_config.vae = parse_component_from_components(root, "vae", "VAEConfig");
    }

    vae_config.decode_only = vae_cfg.value("decode_only", vae_config.decode_only);
    vae_config.decode_only = vae_cfg.value("vae_decode_only", vae_config.decode_only);
    vae_config.keep_on_cpu = vae_cfg.value("keep_on_cpu", vae_config.keep_on_cpu);
    vae_config.keep_on_cpu = vae_cfg.value("vae_on_cpu", vae_config.keep_on_cpu);
    vae_config.conv_direct = vae_cfg.value("conv_direct", vae_config.conv_direct);
    vae_config.conv_direct = vae_cfg.value("vae_conv_direct", vae_config.conv_direct);
    vae_config.force_sdxl_vae_conv_scale =
        vae_cfg.value("force_sdxl_vae_conv_scale", vae_config.force_sdxl_vae_conv_scale);

    if (vae_cfg.contains("tiling") && vae_cfg.at("tiling").is_object()) {
        auto const &tiling_j              = vae_cfg.at("tiling");
        vae_config.tiling.enabled         = tiling_j.value("enabled", vae_config.tiling.enabled);
        vae_config.tiling.tile_size_x     = tiling_j.value("tile_size_x", vae_config.tiling.tile_size_x);
        vae_config.tiling.tile_size_y     = tiling_j.value("tile_size_y", vae_config.tiling.tile_size_y);
        vae_config.tiling.target_overlap  = tiling_j.value("target_overlap", vae_config.tiling.target_overlap);
    }
}

} // namespace

SDHyperParams::SDHyperParams(const Path &params_file) {
    auto const j = parse_json_file(params_file, "SDHyperParams");
    try {
        parse_hyper_params_from_json(*this, j);
    } catch (const std::exception &err) {
        throw ConfigException(
            "SDHyperParams", fmt::format("failed parsing hyper param config file {}:\n{}", params_file, err.what())
        );
    }
}

SDConfig::SDConfig(const Path &model_config_file) {
    auto const j = parse_json_file(model_config_file, "SDConfig");

    try {
        if (j.contains("hyper_params") && j.at("hyper_params").is_object()) {
            parse_hyper_params_from_json(hyper_params, j.at("hyper_params"));
        }
        if (j.contains("generation") && j.at("generation").is_object()) {
            parse_hyper_params_from_json(hyper_params, j.at("generation"));
        }
        if (j.contains("inference") && j.at("inference").is_object()) {
            parse_hyper_params_from_json(hyper_params, j.at("inference"));
        }

        parse_text_encode_config(text_encode_config, j);
        parse_sd_model_config(sd_model_config, j);
        parse_vae_config(vae_config, j);
    } catch (const std::exception &err) {
        throw ConfigException("SDConfig", fmt::format("failed parsing sd config file {}:\n{}", model_config_file, err.what()));
    }
}

SDConfig::SDConfig(const Path &model_config_file, const Path &hparams_file) : SDConfig(model_config_file) {
    hyper_params = SDHyperParams(hparams_file);
}

} // namespace powerserve
