#!/bin/bash
# realworldqa.sh - Run RealWorldQA benchmark

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


run_realworldqa() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local GPU_ID="$2"
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "=============================="
    echo "Evaluating RealWorldQA..."
    echo "=============================="
    
    cd "$BASE_DIR/realworldqa" || { 
        echo "Error: RealWorldQA directory not found"; 
        return 1; 
    }
    
    mkdir -p "data"
    
    echo "Data preprocessing for RealWorldQA..."
    python parquet_to_json.py \
        --input_files "data/test-00000-of-00002.parquet data/test-00001-of-00002.parquet" \
        --output_file "data/RealWorldQA.json" \
        --sample_ratio 1.0 > "$LOG_DIR/realworldqa_preprocess.log" 2>&1
    
    echo "Generating Responses for RealWorldQA..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/RealWorldQA.json" \
        --use_cot $use_cot \
        --pre_prompt "$pre_prompt" \
        --save_name "data/RealWorldQA_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --tp 1 \
        --bz 1 \
        --max_new_tokens 8000 > "$LOG_DIR/realworldqa_inference$cot_suffix.log" 2>&1
    
    echo "Calculating scores for RealWorldQA..."
    python judge.py \
        --input_file "data/RealWorldQA_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --judge_api "$CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT" \
        --api_key "$CUSTOMIZED_REMOTE_OPENAI_API_KEY" \
        --output_file "data/RealWorldQA_judged_$INFERENCE_RUN_ID$cot_suffix.jsonl" > "$LOG_DIR/realworldqa_judge$cot_suffix.log" 2>&1
    
    echo "RealWorldQA evaluation completed. Results in: $BASE_DIR/realworldqa/data/RealWorldQA_judged_$INFERENCE_RUN_ID.jsonl"
    echo "RealWorldQA DONE" > "$LOG_DIR/realworldqa_done.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    GPU_ID="$2"  # 接收 GPU ID 参数
    run_realworldqa "$PORT" "$GPU_ID"
fi