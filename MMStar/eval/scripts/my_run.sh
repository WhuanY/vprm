#!/bin/bash
set -x

export MASTER_PORT=29512
export NUM_GPUS=8

MODEL_PATH="/media/public/models/huggingface/Qwen/Qwen2.5-VL-3B-Instruct"

while true; do
    torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
        --verbose \
        --data MMStar \
        --model "$MODEL_PATH" \
        --max-new-tokens 32 \
        --gen-mode mm
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "torchrun succeeded, exiting loop."
        break
    else
        echo "torchrun failed with exit code $EXIT_CODE, retrying after 10 seconds..."
        sleep 10
    fi
done