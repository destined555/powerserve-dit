
## 1. 如何运行

### 1.1 下载 SD3.5 的 hf 模型到本地

先将 SD3.5 模型下载到本地目录 `./sd3.5`

### 1.2 转化模型

执行：

```bash
python3 ./tools/gguf_export_sd.py \
  --model-dir ./sd3.5 \
  -o ./SD3.5 \
  --model-id sd3.5-medium \
  -t f16
```

转换后目录结构示例：

```text
SD3.5/
  model.json
  ggml/
    weight.gguf # 后续可去掉
    clip_l.gguf
    clip_g.gguf
    t5xxl.gguf
    vae.gguf
  qnn/  # 先占位
```

### 1.3 编译

Linux：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Android：

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-35 \
  -DGGMLOPENMP=OFF \
  -DPOWERSERVE_WITH_QNN=OFF
cmake --build build
```

### 1.4 运行

```bash
./build/bin/run-sd \
  --workfolder ./SD3.5 \
  --prompts "cat" \
  --nprompts "ugly" \
  --latent-bin ./sdcpp_x0_latent.bin
```

`./sdcpp_x0_latent.bin` 是从 `sd.cpp` 中导出的一个去噪结果，默认是生成一只猫

后续接入去噪后就不用使用  `--latent-bin`

## 2. 提供给 DiT 的接口

运行主程序：`./app/run-sd/run-sd.cpp`

### 2.1 encode

`run-sd.cpp` line 112 的 `embedding.` 结构如下：

```cpp
struct SDTextEncoderEmbeddings {
    // 交叉注意力条件张量，逻辑 shape 为 [crossattn_tokens, crossattn_dim]。
    // index = token_idx * crossattn_dim + dim_idx
    std::vector<float> crossattn;
    // pooled 向量，逻辑 shape 为 [vector_dim]
    std::vector<float> vector;

    size_t crossattn_dim   = 0;

    size_t crossattn_tokens = 0;

    size_t vector_dim      = 0;
};
struct SDPromptPairEmbeddings {
    SDTextEncoderEmbeddings prompt;
    SDTextEncoderEmbeddings negative_prompt;
};
```

### 2.2 scheduler

`run-sd.cpp` line 157 的 `schedule` 结构如下：

```cpp
struct SDSchedulerSchedule {
    std::string resolved_prediction;
    std::string resolved_scheduler;
    float resolved_flow_shift = 0.0f;
    // 噪声序列
    std::vector<float> sigmas;
};
```

### 2.3 vae

`run-sd.cpp` line 185 的 `latent`：

- `latent` 为 `std::vector<float>`。
