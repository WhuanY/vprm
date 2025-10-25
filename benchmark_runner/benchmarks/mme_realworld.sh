#!/bin/bash
# mme_realworld.sh - Run MME-RealWorld-Lite benchmark

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"


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


run_mme_realworld() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local GPU_ID="$2" 
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "=============================="
    echo "Evaluating MME-RealWorld-Lite..."
    echo "=============================="
    
    cd "$BASE_DIR/MME-RealWorld-Lite" || { 
        echo "Error: MME-RealWorld-Lite directory not found"; 
        return 1; 
    }
    
    mkdir -p "data"
    
    echo "Data preprocessing for MME-RealWorld-Lite..."
    python unify_format_lite.py \
        --input_file "data/MME-RealWorld-Lite.json" \
        --output_file "data/MME-RealWorld-Lite_unified.json" \
        --image_base_dir "$IMAGE_BASE_DIR_MME_REALWORLD_LITE" > "$LOG_DIR/mme_preprocess.log" 2>&1
    
    echo "Inferencing for MME-RealWorld-Lite..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/MME-RealWorld-Lite_unified.json" \
        --save_name "data/MME-RealWorld-Lite_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --tp 1 \
        --bz 1 \
        --use_cot $use_cot \
        --pre_prompt "$pre_prompt" \
        --max_new_tokens 8000 > "$LOG_DIR/mme_inference$cot_suffix.log" 2>&1
    
    echo "Calculating scores for MME-RealWorld-Lite..."
    python judge.py \
        --input_file "data/MME-RealWorld-Lite_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --judge_api "$CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT" \
        --api_key "$CUSTOMIZED_REMOTE_OPENAI_API_KEY" \
        --output_file "data/MME-RealWorld-Lite_judged_$INFERENCE_RUN_ID$cot_suffix.jsonl" > "$LOG_DIR/mme_judge$cot_suffix.log" 2>&1
    
    echo "MME-RealWorld-Lite evaluation completed. Results in: $BASE_DIR/MME-RealWorld-Lite/data/MME-RealWorld-Lite_judged_$INFERENCE_RUN_ID$cot_suffix.jsonl"
    echo "MME-RealWorld-Lite DONE" > "$LOG_DIR/mme_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    GPU_ID="$2"  # 接收 GPU ID 参数
    run_mme_realworld "$PORT" "$GPU_ID"
fi