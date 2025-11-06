#!/bin/bash
# config.sh - Common configuration variables for all benchmark scripts

# GPU Configuration
export CUDA_DEVICES_LIST="5,6,7,8,9"
IFS=',' read -ra CUDA_DEVICES_ARRAY <<< "$CUDA_DEVICES_LIST"

# API Endpoints for evaluation
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxx"

# VLLM Configuration
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=1

# Benchmark Configuration
export MATHVISION_SUBSET="testmini"  # "testmini" or "test"
export IMAGE_BASE_DIR_MME_REALWORLD_LITE="/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs"

# Model Checkpoint Path (modify this to your checkpoint path)
# Only set default if not already set as environment variable
if [ -z "$CKPT_PATH" ]; then
    export CKPT_PATH="/mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct"
    # export CKPT_PATH="/mnt/minyingqian/data/results/trm_140_base-rm_conflict-it1-trm/global_step_80/actor/huggingface"
fi

# Base directory (script location)
export BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit; pwd)

# =========================================================================
# Time stamp sharing logic
# =========================================================================
# If SHARED_TIMESTAMP is not set, generate a new one
# This ensures all scripts use the same timestamp when called from run_all.sh
if [ -z "$SHARED_TIMESTAMP" ]; then
    export SHARED_TIMESTAMP=$(date +"%Y%m%d_%H%M")
fi

# =========================================================================
# Checkpoint path parsing function
# =========================================================================
# Parse checkpoint path to extract: model_name, global_step, last_dir
# Example: /mnt/bn/.../results/trm_140_base-rm_conflict-it1-trm/global_step_80/actor/huggingface
# Returns: trm_140_base-rm_conflict-it1-trm-global_step_80-huggingface
parse_ckpt_path() {
    local ckpt_path="$1"
    
    # Extract model name (directory after "results/")
    local model_name=""
    if [[ "$ckpt_path" == *"/results/"* ]]; then
        # Get the part after "/results/"
        local after_results="${ckpt_path#*/results/}"
        # Get the first directory after results/
        model_name=$(echo "$after_results" | cut -d'/' -f1)
    else
        # Fallback: use basename if no results/ found
        model_name=$(basename "$ckpt_path")
    fi
    
    # Extract global_step_XX
    local global_step=""
    if [[ "$ckpt_path" == *"global_step_"* ]]; then
        global_step=$(echo "$ckpt_path" | grep -o "global_step_[0-9]*" | head -1)
    fi
    
    # Extract last directory
    local last_dir=$(basename "$ckpt_path")
    
    # Combine: model_name-global_step-last_dir
    if [ -n "$global_step" ]; then
        echo "${model_name}-${global_step}-${last_dir}"
    else
        # Fallback: if no global_step found, use model_name-last_dir
        echo "${model_name}-${last_dir}"
    fi
}

# =========================================================================
# Generate unified result directory and paths
# =========================================================================
# Parse checkpoint path to get unified directory name
# New structure: results/$UNIFIED_RESULT_DIR/benchmark_name/$SHARED_TIMESTAMP/
if [ -n "$CKPT_PATH" ]; then
    UNIFIED_RESULT_DIR=$(parse_ckpt_path "$CKPT_PATH")
    export UNIFIED_RESULT_DIR
    
    # Generate unified result base path (without timestamp, timestamp will be added per benchmark)
    export UNIFIED_RESULT_BASE="$BASE_DIR/results/$UNIFIED_RESULT_DIR"
    
    # Generate CKPT_NAME and INFERENCE_RUN_ID (for backward compatibility)
    export CKPT_NAME=$(basename "$CKPT_PATH")
    export INFERENCE_RUN_ID="${CKPT_NAME}_${SHARED_TIMESTAMP}"
else
    # Fallback if CKPT_PATH is not set
    export CKPT_NAME="unknown"
    export INFERENCE_RUN_ID="unknown_${SHARED_TIMESTAMP}"
    export UNIFIED_RESULT_DIR="unknown"
    export UNIFIED_RESULT_BASE="$BASE_DIR/results/unknown"
fi

# Validate configuration
[[ "$MATHVISION_SUBSET" == "testmini" || "$MATHVISION_SUBSET" == "test" ]] || { 
    echo "Error: MATHVISION_SUBSET must be 'testmini' or 'test'"; 
    exit 1; 
}

echo "Configuration loaded:"
echo "Checkpoint: $CKPT_PATH"
echo "Unified Result Dir: $UNIFIED_RESULT_DIR"
echo "Unified Result Base: $UNIFIED_RESULT_BASE"
echo "Run ID: $INFERENCE_RUN_ID"
echo "Shared Timestamp: $SHARED_TIMESTAMP"
echo "Base directory: $BASE_DIR"