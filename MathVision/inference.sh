#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="5,6,7,8"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

echo "Server is running, starting inference..."

# 创建输出目录（如果不存在）
mkdir -p data

# 运行推理
nohup python inference.py \
    --model_name_or_path /home/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVision_test.json \
    --save_name data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl \
    --tp 4 \
    --bz 1 \
    --max_new_tokens 8000 > data/inference_test_qwen25vl3b-inst.log &

# 获取后台进程的PID
PID=$!
echo "Inference started with PID: $PID"
echo "You can monitor the progress with: tail -f data/inference_qwen25vl3b-inst.log"
echo "To stop the process: kill $PID"