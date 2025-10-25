export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


use_cot=1
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt=""
    cot_suffix=""
fi

nohup python inference.py \
--model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--pre_prompt "$pre_prompt" \
--use_cot $use_cot \
--input_file data/MME-RealWorld-Lite_unified.json \
--save_name data/MME-RealWorld-Lite_inferenced_qwen25vl3b-inst$cot_suffix.jsonl \
--tp 4 \
--bz 1 \
--max_new_tokens 8000 2>&1 | tee data/MME-RealWorld-Lite_inferenced_qwen25vl3b-inst$cot_suffix.log

# nohup python inference.py \
# --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
# --pre_prompt "$pre_prompt" \
# --use_cot $use_cot \
# --input_file data/MME-RealWorld-Lite_sample1.json \
# --save_name data/MME-RealWorld-Lite_sample1_inferenced_qwen25vl3b-inst$cot_suffix.jsonl \
# --tp 4 \
# --bz 1 \
# --max_new_tokens 8000 2>&1 | tee data/MME-RealWorld-Lite_sample1_inferenced_qwen25vl3b-inst$cot_suffix.log