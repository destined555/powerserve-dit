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

#include "cmdline.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/spec_model.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>

#ifdef POWERSERVE_DUMP_SPEEDINFO
#include <iostream>
#endif //POWERSERVE_DUMP_SPEEDINFO

// hjy
#ifdef POWERSERVE_WITH_PERFETTO
#include "core/perfetto_trace.hpp"
#endif


// hjy
// 获取当前进程的rss
double get_process_rss() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    double rss = 0;
    while (std::getline(stat_file, line)) {
        if (line.find("VmRSS:") == 0) { // 找到 VmRSS 这一行
            // 提取数字部分
            std::stringstream ss(line.substr(6));
            ss >> rss; // 提取数字
            break;
        }
    }
    return rss / 1024.0 / 1024.0; // 返回 GB
}
// 获取当前进程的swap
double get_process_swap() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    long vmswap_kb = 0; // VmSwap 通常以 KB 为单位报告

    while (std::getline(stat_file, line)) {
        // 寻找 "VmSwap:" 这行
        if (line.compare(0, 7, "VmSwap:") == 0) {
            std::stringstream ss(line.substr(7)); // 跳过 "VmSwap:"
            ss >> vmswap_kb;
            break;
        }
    }
    return static_cast<double>(vmswap_kb) / 1024.0 / 1024.0; // 转换为 GB
}
// 获取当前进程的vmsize
double get_process_vmsize() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    long vmsize_kb = 0; // VmSize 通常以 KB 为单位报告

    while (std::getline(stat_file, line)) {
        // 寻找 "VmSize:" 这行
        if (line.compare(0, 7, "VmSize:") == 0) {
            std::stringstream ss(line.substr(7)); // 跳过 "VmSize:"
            ss >> vmsize_kb;
            break;
        }
    }
    return static_cast<double>(vmsize_kb) / 1024.0 / 1024.0; // 转换为 GB 返回
}
// 获取当前进程的rss_anon
double get_process_rss_anon() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    long rss_anon_kb = 0; // VmRSS 通常以 KB 为单位报告

    while (std::getline(stat_file, line)) {
        // 寻找 "RssAnon:" 这行
        if (line.compare(0, 8, "RssAnon:") == 0) { // "RssAnon:" 长度为 8
            std::stringstream ss(line.substr(8)); // 跳过 "RssAnon:"
            ss >> rss_anon_kb;
            break;
        }
    }
    return static_cast<double>(rss_anon_kb) / 1024.0 / 1024.0; // 转换为 GB 返回
}
// 获取当前进程的rss_file
double get_process_rss_file() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    long rss_file_kb = 0; // VmRSS 通常以 KB 为单位报告

    while (std::getline(stat_file, line)) {
        // 寻找 "RssFile:" 这行
        if (line.compare(0, 8, "RssFile:") == 0) { // "RssFile:" 长度为 8
            std::stringstream ss(line.substr(8)); // 跳过 "RssFile:"
            ss >> rss_file_kb;
            break;
        }
    }
    return static_cast<double>(rss_file_kb) / 1024.0 / 1024.0; // 转换为 GB 返回
}
// 定义一个全局或局部原子变量来控制线程生命周期
std::atomic<bool> stop_mem_thread(false);
// 内存监控线程函数
void mem_monitor_worker(int interval_ms) {
    while (!stop_mem_thread) {
        double rss = get_process_rss(); 
        double swap = get_process_swap();
        double vmsize = get_process_vmsize();
        double rss_anon = get_process_rss_anon();
        double rss_file = get_process_rss_file();
        // 使用你的 Perfetto 实例进行采样
        powerserve::PerfettoTrace::counter("mem_rss_gb", rss);
        powerserve::PerfettoTrace::counter("mem_swap_gb", swap);
        powerserve::PerfettoTrace::counter("mem_vmsize_gb", vmsize);
        powerserve::PerfettoTrace::counter("mem_rss_anon_gb", rss_anon);
        powerserve::PerfettoTrace::counter("mem_rss_file_gb", rss_file);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

// hjy
// 获取当前的cpu中的 kv cache 大小
double get_vector(const std::vector<std::vector<float>> &vec){
    double ret=0;
    for(const auto& inner_vec : vec){
        ret += inner_vec.capacity() * sizeof(float);
    }
    return static_cast<double>(ret);
}
double get_kv_cache_cpu(auto & platform){
    double ret=0;
    for(auto const& pair : platform->ggml_backends){
        const std::unique_ptr<powerserve::ggml::GGMLBackend>& backend_ptr = pair.second;
        if(!backend_ptr){
            return 0;
        }
        auto const & kv_cache_ptr = backend_ptr->m_kv;
        if(!kv_cache_ptr){
            return 0;
        }
        auto const & chunk = kv_cache_ptr->chunk;
        ret+=get_vector(chunk.key_buffer);
        ret+=get_vector(chunk.value_buffer);
        ret+=get_vector(chunk.current_k);
        ret+=get_vector(chunk.current_v);
    }
    return ret / 1024.0 / 1024.0 / 1024.0; //GB
}
#if defined(POWERSERVE_WITH_QNN)
// 获取当前的npu中的 kv cache 大小
double get_kv_cache_npu(auto & platform){
    size_t ret=0;
    auto const & backend_ptr=platform->qnn_backend;
    if(!backend_ptr){
        return 0;
    }
    for(auto const& model_ptr : backend_ptr->m_models){
        auto const& model_instance = model_ptr.second;
        if(!model_instance){
            continue;
        }
        auto const& max_chunks = model_instance->m_chunks_map[model_instance->m_gparams.max_batch_size];
        for (size_t k = 0; k < max_chunks.size(); k++) {
            auto const& max_chunk = max_chunks[k];
            if(!max_chunk){
                continue;
            }
            for (size_t i = 0; i < max_chunk->n_layers(); i++) {
                for (size_t j = 0; j < max_chunk->m_model_config.llm.n_kv_heads; j++) {
                    auto const& buffer_ptr1=max_chunk->m_buffers[fmt::format("layer_{}_key_t_cache_{}", max_chunk->m_config.start_layer_id + i, j)];
                    if(buffer_ptr1){
                        ret += buffer_ptr1->m_size;
                    }
                    auto const& buffer_ptr2=max_chunk->m_buffers[fmt::format("layer_{}_value_cache_{}", max_chunk->m_config.start_layer_id + i, j)];
                    if(buffer_ptr2){
                        ret += buffer_ptr2->m_size;
                    }
                    auto const& buffer_ptr3=max_chunk->m_buffers[fmt::format("layer_{}_key_{}", max_chunk->m_config.start_layer_id + i, j)];
                    if(buffer_ptr3){
                        ret += buffer_ptr3->m_size;
                    }
                    auto const& buffer_ptr4=max_chunk->m_buffers[fmt::format("layer_{}_value_{}", max_chunk->m_config.start_layer_id + i, j)];
                    if(buffer_ptr4){
                        ret += buffer_ptr4->m_size;
                    }
                }
            }
        }
    }
    return ret / 1024.0 / 1024.0 / 1024.0; //GB
}
#endif
// 定义一个全局或局部原子变量来控制线程生命周期
std::atomic<bool> stop_kv_thread(false);
// 定义一个全局变量来标记是否已经初始化了npu的kv cache
#if defined(POWERSERVE_WITH_QNN)
std::atomic<bool> flag_kv_npu(false);
#endif
// kv监控线程函数
void kv_monitor_worker(int interval_ms,std::shared_ptr<powerserve::Platform> & platform) {
    while (!stop_kv_thread && platform) {
        double cpu_kv=get_kv_cache_cpu(platform);
        // 使用你的 Perfetto 实例进行采样
        powerserve::PerfettoTrace::counter("kvcache_cpu_gb", cpu_kv);
#if defined(POWERSERVE_WITH_QNN)
        if(flag_kv_npu){
            double npu_kv=get_kv_cache_npu(platform);
            powerserve::PerfettoTrace::counter("kvcache_npu_gb", npu_kv);
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

int main(int argc, char *argv[]) {

    // hjy
    auto& tracer = powerserve::PerfettoTrace::instance();
    tracer.start_tracing(32 * 1024);
    tracer.enable();
    std::thread mem_thread(mem_monitor_worker, 50);

    const powerserve::CommandLineArgument args = powerserve::parse_command_line("PowerServe CLI", argc, argv);
    const powerserve::Config config            = powerserve::get_config_from_argument(args);
    
    // hjy
    powerserve::PerfettoTrace::begin("load_model_cpu");

    std::shared_ptr<powerserve::Model> main_model  = powerserve::load_model(config.main_model_dir);
    std::shared_ptr<powerserve::Model> draft_model = nullptr;
    if (args.use_spec) {
        draft_model = powerserve::load_model(config.draft_model_dir);
    }
    POWERSERVE_LOG_INFO("after model init: {}", powerserve::perf_get_mem_result());

    // hjy
    powerserve::PerfettoTrace::end();

    // hjy
    powerserve::PerfettoTrace::begin("init_backend_cpu");

    const auto [sampler_config, n_threads, batch_size] = config.hyper_params;
    main_model->m_platform                             = std::make_shared<powerserve::Platform>();
    auto &platform                                     = main_model->m_platform;

    platform->init_ggml_backend(main_model->m_config, config.hyper_params);

    if (args.use_spec) {
        draft_model->m_platform = platform;
        platform->init_ggml_backend(draft_model->m_config, config.hyper_params);
    }

    // hjy
    powerserve::PerfettoTrace::end();
    std::thread kv_thread(kv_monitor_worker, 50, std::ref(platform));
    
#if defined(POWERSERVE_WITH_QNN)
    // hjy
    powerserve::PerfettoTrace::begin("init_backend_npu");
    if (!args.no_qnn) {
        auto &qnn_backend = main_model->m_platform->qnn_backend;
        main_model->m_platform->init_qnn_backend(args.qnn_lib_folder);
        qnn_backend->load_model(config.main_model_dir / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, main_model->m_config);
        main_model->kv_cache = platform->qnn_backend->m_models[main_model->m_config->model_id]->kv_cache.get();

        if (args.use_spec) {
            qnn_backend->load_model(
                config.draft_model_dir / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, draft_model->m_config
            );
            draft_model->kv_cache = platform->qnn_backend->m_models[draft_model->m_config->model_id]->kv_cache.get();
        }
    }
    // hjy
    flag_kv_npu = true;
    powerserve::PerfettoTrace::end();
#endif
    POWERSERVE_LOG_INFO("after platform init: {}", powerserve::perf_get_mem_result());

    // hjy
    powerserve::PerfettoTrace::begin("init_attn");

    main_model->m_attn = std::make_shared<powerserve::NormAttention>(main_model->m_config->llm, main_model->m_weights);
    if (args.use_spec) {
        draft_model->m_attn =
            std::make_shared<powerserve::NormAttention>(draft_model->m_config->llm, draft_model->m_weights);
    }
    POWERSERVE_LOG_INFO("after attn init: {}", powerserve::perf_get_mem_result());

    // hjy
    powerserve::PerfettoTrace::end();

    // hjy
    powerserve::PerfettoTrace::begin("init_tokenizer_and_sampler");

    const std::string tokenizer_path = config.main_model_dir / powerserve::MODEL_VOCAB_FILENAME;
    powerserve::Tokenizer tokenizer(tokenizer_path);
    POWERSERVE_LOG_INFO("after tokenizer init: {}", powerserve::perf_get_mem_result());

    powerserve::SamplerChain sampler{sampler_config, tokenizer};
    POWERSERVE_LOG_INFO("after sampler init: {}", powerserve::perf_get_mem_result());

    // hjy
    powerserve::PerfettoTrace::end();

    {
        POWERSERVE_LOG_INFO("prompt      : {:?}", powerserve::abbreviation(args.prompt, 50));
        POWERSERVE_LOG_INFO("n_predicts  : {}", args.num_predict);
        POWERSERVE_LOG_INFO("model arch  : {}", main_model->m_config->arch);
        POWERSERVE_LOG_INFO("n_threads   : {}", n_threads);
        POWERSERVE_LOG_INFO("batch_size   : {}", batch_size);
    }

    // hjy
    powerserve::PerfettoTrace::begin("prefill");

    // generate
    long prefill_start = 0;
    long prefill_end   = 0;
    long decode_end    = 0;
    bool start         = false;
    int actual_predict = 0;
    for (const powerserve::Token prompt_token : tokenizer.tokenize(args.prompt, tokenizer.m_vocab.tokenizer_add_bos)) {
        fmt::print("{}", tokenizer.to_string(prompt_token, false));
    }
    prefill_start = powerserve::timestamp_ms();

    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (args.use_spec) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(main_model, draft_model, args.speculative_config);
        iter       = spec_model->generate(tokenizer, sampler, args.prompt, args.num_predict, batch_size);
    } else
#endif
    {
        iter = main_model->generate(tokenizer, sampler, args.prompt, args.num_predict, batch_size);
    }
    prefill_end = powerserve::timestamp_ms();

    // hjy
    powerserve::PerfettoTrace::end();

    // hjy
    powerserve::PerfettoTrace::begin("decode");

    while (!iter->end()) {
        auto next = iter->next();
        if (!start) {
            start = true;
            continue;
        }
        actual_predict += 1;
        if (next == tokenizer.bos_token()) {
            break;
        }
        if (tokenizer.should_stop(next)) {
            fmt::print("[end of text]");
            break;
        }
        fmt::print("{}", tokenizer.to_string(next, false));
        fflush(stdout);
    }
    fmt::println("");

    // hjy
    powerserve::PerfettoTrace::end();

    if (start) {
        decode_end               = powerserve::timestamp_ms();
        const size_t num_prefill = tokenizer.tokenize(args.prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
        POWERSERVE_LOG_INFO("prefill time: {} s", (double)(prefill_end - prefill_start) / 1000);
        POWERSERVE_LOG_INFO(
            "prefill speed ({} tokens): {} tokens/s",
            num_prefill,
            num_prefill / (double)(prefill_end - prefill_start) * 1000
        );
        POWERSERVE_LOG_INFO(
            "decode speed ({} tokens): {} tokens/s",
            actual_predict,
            actual_predict / (double)(decode_end - prefill_end) * 1000
        );
        POWERSERVE_LOG_INFO(
            "total speed: {} tokens/s", (num_prefill + actual_predict) / (double)(decode_end - prefill_start) * 1000
        );

#ifdef POWERSERVE_DUMP_SPEEDINFO
    if(!args.use_spec) {
        const char* env_dump_file = std::getenv("dump_file");
        if (env_dump_file) {
            std::string filename(env_dump_file);
            std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
            if (outFile.is_open()) {
                outFile << "{\"prefill_tokens\": " << num_prefill << ", \"prefill_time\": " << (double)(prefill_end - prefill_start) << ", \"decode_tokens\": " << actual_predict << ", \"decode_time\": " << (double)(decode_end - prefill_end) << "}";
                outFile.close();
            }
        }
    }
#endif //POWERSERVE_DUMP_SPEEDINFO
    }
    
#if defined(POWERSERVE_WITH_QNN)
    if (args.use_spec) {
        spec_model->print_stat();
    }
#endif
    // hjy
    stop_mem_thread = true;
    if (mem_thread.joinable()) {
        mem_thread.join();
    }
    stop_kv_thread = true;
    if (kv_thread.joinable()) {
        kv_thread.join();
    }
    tracer.stop_tracing("./trace.data");

    return 0;
}
