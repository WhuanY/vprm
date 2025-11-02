export CUDA_VISIBLE_DEVICES="8,9"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


use_cot=0
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    after_prompt=""
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt=""
    after_prompt="Please try to answer the question with short words or phrases if possible."
    cot_suffix=""
fi


nohup python inference.py \
--model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-7B-Instruct \
--pre_prompt "$pre_prompt" \
--after_prompt "$after_prompt" \
--use_cot $use_cot \
--input_file data/chartQA_test.json \
--save_name data/chartQA_test_inferenced_qwen25vl7b${cot_suffix}_w_offical_prompt.jsonl \
--tp 2 \
--bz 1 \
--max_new_tokens 8000 > data/chartQA_test_inferenced_qwen25vl7b${cot_suffix}_w_offical_prompt.log 2>&1 &
