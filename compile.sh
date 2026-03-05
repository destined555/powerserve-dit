rm -rf build 
rm -rf models_cpu

#!/bin/bash

# 检查 ANDROID_NDK 环境变量
if [ -z "$ANDROID_NDK" ]; then
  echo "❌ 错误：ANDROID_NDK 环境变量未设置，请先配置 Android NDK 路径后重试。"
  exit 1
fi

# 步骤1：CMake 配置项目
echo "🔧 正在配置 CMake 项目..."
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-35 \
  -DGGMLOPENMP=OFF \
  -DPOWERSERVE_WITH_QNN=OFF

if [ $? -ne 0 ]; then
  echo "❌ CMake 配置失败，脚本终止。"
  exit 1
fi

# 步骤2：构建项目
echo "🔨 正在构建项目..."
cmake --build build

if [ $? -ne 0 ]; then
  echo "❌ 构建失败，脚本终止。"
  exit 1
fi

# 步骤3：执行 powerserve create
echo "🚀 正在执行 powerserve create..."
./powerserve create -m ./SD3.5/ --exe-path ./build/out -o ./models_cpu/

if [ $? -ne 0 ]; then
  echo "❌ powerserve create 执行失败。"
  exit 1
fi

echo "✅ 所有步骤执行完成！"