python judge.py \
    --input_file data/RealWorldQA_inferenced_qwen2-vl7b.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxx" \
    --output_file data/RealWorldQA_judged.jsonl > data/judge_qwen2vl7b.log 2>&1 &