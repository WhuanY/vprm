#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"
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

while true; do
    nohup python inference.py \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/mmstar.json \
    --save_name data/mmstar_inferenced_qwen25vl3b$cot_suffix.jsonl \
    --pre_prompt "$pre_prompt" \
    --tp 1 \
    --bz 1 \
    --max_new_tokens 8000 > data/MMStar_inference_qwen25vl3b$cot_suffix.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "inference.py finished successfully, exiting loop."
        break
    else
        echo "inference.py failed with exit code $EXIT_CODE, retrying after 60 seconds..."
        sleep 60
    fi
done