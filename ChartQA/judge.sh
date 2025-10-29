python judge.py \
    --input_file data/chartQA_test_inferenced_qwen25vl7b_cot.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/chartQA_test_judged_qwen25vl7b_cot.jsonl > data/chartQA_test_judge_qwen25vl7b_cot.log 2>&1 &