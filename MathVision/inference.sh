#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="5,6,7,8"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

echo "Server is running, starting inference..."

use_cot=1
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt="Please solve the problem step by step and put your answer in one \"\\boxed{}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{}\"."
    cot_suffix=""
fi

# 创建输出目录（如果不存在）
mkdir -p data

# 运行推理
nohup python inference.py \
    --model_name_or_path /home/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVision_test.json \
    --pre_prompt "$pre_prompt" \
    --save_name data/MathVision-test_inferenced_qwen25vl3b-inst$cot_suffix.jsonl \
    --tp 4 \
    --bz 1 \
    --max_new_tokens 8000 > data/inference_test_qwen25vl3b-inst$cot_suffix.log &

# 获取后台进程的PID
PID=$!
echo "Inference started with PID: $PID"
echo "You can monitor the progress with: tail -f data/inference_qwen25vl3b-inst$cot_suffix.log"
echo "To stop the process: kill $PID"