python judge.py \
    --input_file data/RealWorldQA_inferenced_qwen25vl3b_api0.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxx" \
    --output_file data/RealWorldQA_judged_qwen25vl3b_api0.jsonl > data/judge_qwen25vl3b_api0.log 2>&1 &