python judge.py \
    --input_file data/MME_RealWorld_lastx-inferenced-qwen25vl3b-inst_cot.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/MME_RealWorld_lastx-judge_results-qwen25vl3b-inst_cot.jsonl > data/judge_qwen25vl3b-inst_cot.log 2>&1 &