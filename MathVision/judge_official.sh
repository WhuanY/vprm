mkdir -p outputs

use_cot=1
# 把接下来要评估的推理文件拷贝到 output 目录下（注意替换）
cp data/trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.jsonl outputs/trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.jsonl
python evaluation/evaluate.py \
--eval_file "trm_140_base-rm_conflict-it1-trm_global_step_80-judge_res-full.jsonl" \
--use_cot "$use_cot" \
--judge_url "https://aigc.x-see.cn/v1" \
--api_key "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

