python judge.py \
    --input_file data/MME-RealWorld-Lite_inferenced_qwen25vl3b-inst_cot.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/MME-RealWorld-Lite_judge_results-qwen25vl3b-inst_cot.jsonl > data/judge_qwen25vl3b-inst_cot.log 2>&1 &