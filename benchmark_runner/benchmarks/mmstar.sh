#!/bin/bash
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


mmstar() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local GPU_ID="$2"
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "=============================="
    echo "Evaluating MMStar..."
    echo "=============================="
    
    cd "$BASE_DIR/MMStar" || { 
        echo "Error: MMStar directory not found"; 
        return 1; 
    }
    
    echo "Data preprocessing for MMStar..."
    mkdir -p "data"
    python parquet_to_json.py \
        --input_file "data/mmstar.parquet" \
        --output_file "data/mmstar.json" \
        --sample_ratio 1.0 > "$LOG_DIR/mmstar_preprocess.log" 2>&1
    
    echo "Generating responses for MMStar..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/mmstar.json" \
        --save_name "data/mmstar_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --pre_prompt "$pre_prompt" \
        --tp 1 \
        --bz 1 \
        --max_new_tokens 8000 > "$LOG_DIR/mmstar_inference.log" 2>&1
    
    echo "Evaluating MMStar responses..."
    python judge.py \
        --input_file "data/mmstar_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
        --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
        --output_file "data/mmstar_judged_$INFERENCE_RUN_ID$cot_suffix.jsonl" > "$LOG_DIR/mmstar_judge$cot_suffix.log" 2>&1
    
    echo "MMStar evaluation completed. Results in: $BASE_DIR/MMStar/data/mmstar_judged_$INFERENCE_RUN_ID.jsonl"
    echo "MMStar DONE" > "$LOG_DIR/mmstar_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    GPU_ID="$2"  # 接收 GPU ID 参数
    mmstar "$PORT" "$GPU_ID"
fi