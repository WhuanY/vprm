python judge.py \
    --input_file data/trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.jsonl \
    --judge_api "https://aigc.x-see.cn/v1" \
    --api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --output_file data/trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.jsonl > data/trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.log 2>&1 &