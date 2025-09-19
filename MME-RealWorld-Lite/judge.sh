python judge.py \
    --input_file data/MME-RealWorld-Lite_inferenced_qwen2vl7b-inst.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxx" \
    --output_file data/MME-RealWorld-Lite_judge_results-qwen2vl7b-inst.jsonl > data/judge_qwen2vl7b.log 2>&1 &