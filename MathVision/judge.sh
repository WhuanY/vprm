python judge.py \
    --input_file data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxx" \
    --output_file data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl > data/judge_qwen2vl7b.log 2>&1 &