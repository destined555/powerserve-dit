# 文本编码子模块：
# 1) 模型加载（clip_l/clip_g/t5）
# 2) prompt tokenization（CLIP BPE + T5 Unigram）
# 3) text encoder 前向执行
# 4) parity/validation 辅助
target_sources(powerserve PRIVATE
    "sd_text_encoder_model_loader.cpp"
    "sd_prompt_tokenizer.cpp"
    "sd_text_encoder_vocab_loader.cpp"
    "sd_text_encoder_shape_utils.cpp"
    "sd_text_encoder_runner.cpp"
    "sd_text_encoder_parity.cpp"
    "sd_text_encoder_validation.cpp"
)
