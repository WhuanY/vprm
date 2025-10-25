python judge.py \
    --input_file data/RealWorldQA_inferenced_qwen_base-trm_real_global_step_140_huggingface_20251022_13.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/RealWorldQA_jugde_qwen_base-trm_real_global_step_140_huggingface_20251022_13.jsonl > data/RealWorldQA_jugde_qwen_base-trm_real_global_step_140_huggingface_20251022_13.log 2>&1 &