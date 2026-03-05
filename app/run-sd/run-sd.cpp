// 2026-3 hjy
#include "cmdline-sd.hpp"
#include "core/logger.hpp"
#include "encode/sd_prompt_tokenizer.hpp"
#include "encode/sd_text_encoder_model_loader.hpp"
#include "encode/sd_text_encoder_runner.hpp"
#include "encode/sd_text_encoder_validation.hpp"
#include "encode/sd_text_encoder_vocab_loader.hpp"
#include "scheduler/sd_scheduler.hpp"
#include "vae/sd35_vae.hpp"
#include "vae/vae.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

auto load_f32_tensor_from_bin(const powerserve::Path &path, size_t expect_elements) -> std::vector<float> {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin.is_open()) {
        throw std::runtime_error("failed to open latent file: " + path.string());
    }

    const std::streamsize bytes = fin.tellg();
    if (bytes < 0) {
        throw std::runtime_error("failed to read latent file size: " + path.string());
    }
    if ((bytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        throw std::runtime_error("latent file size is not aligned to float32: " + path.string());
    }

    const size_t num_elements = static_cast<size_t>(bytes / static_cast<std::streamsize>(sizeof(float)));
    if (num_elements != expect_elements) {
        throw std::runtime_error(
            "latent element count mismatch: got " + std::to_string(num_elements) +
            ", expect " + std::to_string(expect_elements) + " (" + path.string() + ")"
        );
    }

    std::vector<float> data(num_elements, 0.0f);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(data.data()), bytes);
    if (!fin) {
        throw std::runtime_error("failed to read latent file payload: " + path.string());
    }

    return data;
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        // 阶段 1：解析命令行参数并生成运行配置。
        const powerserve::CommandLineArgumentSD args =
            powerserve::parse_command_line_sd("PowerServe SD CLI", argc, argv);
        const powerserve::SDConfig config = powerserve::get_config_from_argument_sd(args);

        // 阶段 2：解析模型工作目录并加载文本编码器权重（clip_l / clip_g / t5）。
        powerserve::Path work_folder(args.work_folder);
        if (!std::filesystem::is_directory(work_folder)) {
            work_folder = work_folder.parent_path();
        }

        powerserve::SDTextEncoderModelLoader text_encoder_loader;
        text_encoder_loader.load(work_folder, config.text_encode_config);

        // 阶段 3：加载并校验内置词表资源。
        const auto vocab_pack = powerserve::SDTextEncoderVocabLoader::load_embedded();
        powerserve::SDTextEncoderVocabLoader::validate(vocab_pack);

        // 阶段 4：将正向/反向 prompt 分词并编码成 token id。
        powerserve::SDPromptTokenizer prompt_tokenizer(vocab_pack);
        const auto tokenized = prompt_tokenizer.encode_prompt_pair(
            config.hyper_params.prompt,
            config.hyper_params.negative_prompt
        );

        // 阶段 5：校验 token 编码结果的结构合法性（长度、BOS/EOS、权重对齐等）。
        powerserve::verify_prompt_pair_tokenization_or_throw(tokenized);

        // 阶段 6：执行文本编码器前向计算，将 token id 转为条件向量。
        powerserve::SDTextEncoderRunner runner(
            text_encoder_loader,
            config.text_encode_config,
            static_cast<int>(config.hyper_params.n_threads)
        );
        const auto embeddings = runner.encode_prompt_pair(tokenized);

        // 阶段 7：校验编码器输出的 shape 与维度约束。
        powerserve::verify_prompt_pair_embeddings_or_throw(embeddings);

        // 阶段 8：按参数执行可选的 dump / parity compare。
        powerserve::SDTextEncoderParityOptions parity_options;
        parity_options.dump_embeddings    = args.dump_embeddings;
        parity_options.compare_embeddings = args.compare_embeddings;
        parity_options.max_abs_threshold  = args.compare_max_abs_threshold;
        parity_options.rmse_threshold     = args.compare_rmse_threshold;
        powerserve::maybe_run_prompt_pair_parity(parity_options, embeddings);

        // 阶段 10：根据 --sample_method 选择 sigma 生成策略（未指定时默认 discrete）。
        powerserve::SDSchedulerRequest scheduler_request;
        scheduler_request.model_arch    = config.sd_model_config.arch;
        scheduler_request.sample_steps  = config.hyper_params.sample_step;
        scheduler_request.sample_method = config.hyper_params.sample_method;
        scheduler_request.scheduler     = config.hyper_params.scheduler;
        scheduler_request.flow_shift    = config.sd_model_config.flow_shift;

        const auto schedule = powerserve::build_sd35_scheduler(scheduler_request);
        (void)schedule;

        // 阶段 11：加载 SD3.5 VAE 权重。
        powerserve::SD35VAE vae_runner;
        vae_runner.load(work_folder, config.vae_config);
        POWERSERVE_LOG_INFO("[run-sd] 阶段 11 完成：SD3.5 VAE 权重加载完成。");

        // 阶段 12：读取 latent 执行 VAE 计算并导出对比文件。
        if (config.hyper_params.width % powerserve::SD35VAE::kVaeScaleFactor != 0 ||
            config.hyper_params.height % powerserve::SD35VAE::kVaeScaleFactor != 0) {
            throw std::runtime_error(
                "width/height must be divisible by 8 for SD3.5 VAE decode"
            );
        }

        const int64_t latent_w = static_cast<int64_t>(config.hyper_params.width / powerserve::SD35VAE::kVaeScaleFactor);
        const int64_t latent_h = static_cast<int64_t>(config.hyper_params.height / powerserve::SD35VAE::kVaeScaleFactor);

        const size_t latent_elements = static_cast<size_t>(latent_w * latent_h * powerserve::SD35VAE::kLatentChannels);
        if (args.latent_bin.empty()) {
            throw std::runtime_error("latent_bin cannot be empty");
        }
        const powerserve::Path latent_path(args.latent_bin);
        auto latent = load_f32_tensor_from_bin(latent_path, latent_elements);

        const auto decoded = vae_runner.decode_latent(
            std::move(latent),
            latent_w,
            latent_h,
            1,
            static_cast<int>(config.hyper_params.n_threads)
        );

        const powerserve::Path compare_dir =
            args.vae_compare_dir.empty() ? work_folder : powerserve::Path(args.vae_compare_dir);
        const auto outputs = powerserve::export_sd35_vae_compare_outputs(compare_dir, decoded);

        POWERSERVE_LOG_INFO(
            "[run-sd] 阶段 12 完成：图片='{}'，对比文件='{}'",
            outputs.png_path.string(),
            outputs.decoded_bin_path.string()
        );
    } catch (const std::exception &err) {
        POWERSERVE_LOG_ERROR("run-sd failed: {}", err.what());
        return 1;
    }

    return 0;
}
