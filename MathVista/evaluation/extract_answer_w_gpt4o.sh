export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxx"

python extract_answer_w_gpt4o.py \
--results_file_path ../results/qwen2vl7b_api0/output_qwen2vl7b_api0.json > ../data/extract_answer_w_gpt4o_qwen2vl7b.log 2>&1 &