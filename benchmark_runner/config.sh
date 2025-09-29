#!/bin/bash
# config.sh - Common configuration variables for all benchmark scripts

# GPU Configuration
export CUDA_VISIBLE_DEVICES="5,6,7,8"

# API Endpoints for evaluation
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxx"

# VLLM Configuration
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_INFERENCE_PORT=9753  # Default port, can be overridden

# Benchmark Configuration
export MATHVISION_SUBSET="test"  # "testmini" or "test"
export IMAGE_BASE_DIR_MME_REALWORLD_LITE="/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs"

# Model Checkpoint Path (modify this to your checkpoint path)
export CKPT_PATH="/path/to/your/vprm_checkpoint"

# Generate unique run ID
export CKPT_NAME=$(basename "$CKPT_PATH")
export INFERENCE_RUN_ID="${CKPT_NAME}_$(date +"%Y%m%d_%H%M%S")"

# Base directory (script location)
export BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd)

# Validate configuration
[[ "$MATHVISION_SUBSET" == "testmini" || "$MATHVISION_SUBSET" == "test" ]] || { 
    echo "Error: MATHVISION_SUBSET must be 'testmini' or 'test'"; 
    exit 1; 
}

echo "Configuration loaded:"
echo "Checkpoint: $CKPT_PATH"
echo "Run ID: $INFERENCE_RUN_ID"
echo "Base directory: $BASE_DIR"