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

#include "core/config-sd.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace powerserve {

struct CommandLineArgumentSD {
    /// The model directory path
    std::string work_folder;

    /// Positive prompt
    std::string prompts;

    /// Negative prompt
    std::string nprompts;

    /// Output image height
    uint32_t height = 256;

    /// Output image width
    uint32_t width = 256;

    /// The number of thread for inference
    uint32_t num_thread = 0;

    /// The maximum number of tokens processed in one iteration
    uint32_t batch_size = 0;

    /// Sigma generation method for scheduler logic.
    std::string sample_method;

    /// Dump prompt embeddings into this path.
    std::string dump_embeddings;

    /// Compare prompt embeddings against this dump file.
    std::string compare_embeddings;

    /// Optional max-abs threshold for parity check; <= 0 means disabled.
    float compare_max_abs_threshold = 0.0f;

    /// Optional RMSE threshold for parity check; <= 0 means disabled.
    float compare_rmse_threshold = 0.0f;

    /// Override text encoder clip_skip; <= 0 keeps model default.
    int32_t clip_skip = 0;

    /// Input latent bin path (float32 raw, layout [W/8, H/8, 16, 1]).
    std::string latent_bin = "/home/frp/jinye/test-vae/latent.bin";

    /// Output directory for VAE comparison artifacts (PNG + decoded bin).
    std::string vae_compare_dir;
};

/*!
 * @brief Parse SD command arguments from the program input.
 */
CommandLineArgumentSD parse_command_line_sd(const std::string_view program_name, int argc, char **argv);

/*!
 * @brief Parse and generate SD config according to the command line arguments.
 * @param[in] args The command line argument to overwrite SD config
 */
SDConfig get_config_from_argument_sd(const CommandLineArgumentSD &args);

} // namespace powerserve
