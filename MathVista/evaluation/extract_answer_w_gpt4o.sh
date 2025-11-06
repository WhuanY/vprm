export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxx"

num_threads=100  # Get num_threads from first argument, default to 1 if not provided
echo "Using $num_threads thread(s) for concurrent request processing"

python extract_answer_w_gpt4o.py \
--results_file_path ../results/qwen25vl3b_cot/output_qwen25vl3b_cot.json \
--rerun \
--num_threads $num_threads | tee extract_answer_w_gpt4o.log