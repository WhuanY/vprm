export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


use_cot=1
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    after_prompt=""
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt=""
    after_prompt="Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:,"
    cot_suffix=""
fi

nohup python inference.py \
--model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--pre_prompt "$pre_prompt" \
--after_prompt "$after_prompt" \
--input_file data/MME_RealWorld_lastx.json \
--image_base_dir data/images \
--save_name data/MME_RealWorld_lastx-inferenced-qwen25vl3b-inst$cot_suffix.jsonl \
--tp 1 \
--bz 50 \
--max_new_tokens 8000 2>&1 | tee data/MME_RealWorld_lastx-inferenced-qwen25vl3b-inst$cot_suffix.log
