#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="8"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

# 创建输出目录（如果不存在）
mkdir -p data

# 运行推理
nohup python inference.py \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVision_test.json \
    --save_name data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl \
    --tp 1 \
    --bz 1 \
    --max_new_tokens 8000 >> data/inference_test_qwen25vl3b-inst.log &
 

