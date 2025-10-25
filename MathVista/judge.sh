# export CUDA_VISIBLE_DEVICES="0,1"
# export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_USE_TRITON_FLASH_ATTN=True


python judge.py \
    --input_file data/MathVista_inferenced_qwen25vl3b-inst_cot.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/MathVista_judged_qwen25vl3b-inst_cot.jsonl 2>&1 | tee data/judge_qwen25vl3b-inst_cot.log