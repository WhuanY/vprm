python judge.py \
    --input_file data/MME-RealWorld-Lite_inferenced_qwen25vl3b-inst_load0.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxx" \
    --output_file data/MME-RealWorld-Lite_judged_qwen25vl3b-inst_load0.jsonl > data/judge_qwen25vl3b-inst_load0.log 2>&1 &