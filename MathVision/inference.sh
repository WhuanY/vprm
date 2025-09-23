#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

# 检查服务器是否运行
echo "Checking if vLLM server is running..."
if ! curl -s http://localhost:9753/health > /dev/null; then
    echo "Error: vLLM server is not running on port 9753"
    echo "Please start the server first"
    exit 1
fi

echo "Server is running, starting inference..."

# 创建输出目录（如果不存在）
mkdir -p data

# 运行推理
python inference.py \
    --inference_api http://localhost:9753/v1 \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVision_test.json \
    --save_name data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl \
    --tp 2 \
    --bz 20 \
    --max_new_tokens 8000 \
    2>&1 | tee data/inference_qwen25vl3b-inst.log &

# 获取后台进程的PID
PID=$!
echo "Inference started with PID: $PID"
echo "You can monitor the progress with: tail -f data/inference_qwen25vl3b-inst.log"
echo "To stop the process: kill $PID"