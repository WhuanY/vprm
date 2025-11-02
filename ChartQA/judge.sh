python judge.py \
    --input_file data/chartQA_test_inferenced_qwen25vl3b_cot.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxx" \
    --use_relax_accuracy \
    --output_file data/chartQA_test_judged_qwen25vl3b_cot_relaxacc.jsonl > data/chartQA_test_judged_qwen25vl3b_cot_relaxacc.log 2>&1 &