# VAE 子模块当前包含：
# 1) SD3.5 VAE 核心解码实现（sd35_vae.cpp）
# 2) 对比导出工具（vae.cpp）
target_sources(powerserve PRIVATE
    "sd35_vae.cpp"
    "vae.cpp"
)
