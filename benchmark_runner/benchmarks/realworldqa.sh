#!/bin/bash
# realworldqa.sh - Run RealWorldQA benchmark

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

run_realworldqa() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
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
    python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/RealWorldQA.json" \
        --save_name "data/RealWorldQA_inferenced_$INFERENCE_RUN_ID.jsonl" \
        --tp 1 \
        --bz 1 \
        --max_new_tokens 8000 > "$LOG_DIR/realworldqa_inference.log" 2>&1
    
    echo "Calculating scores for RealWorldQA..."
    python judge.py \
        --input_file "data/RealWorldQA_inferenced_$INFERENCE_RUN_ID.jsonl" \
        --judge_api "$CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT" \
        --api_key "$CUSTOMIZED_REMOTE_OPENAI_API_KEY" \
        --output_file "data/RealWorldQA_judged_$INFERENCE_RUN_ID.jsonl" > "$LOG_DIR/realworldqa_judge.log" 2>&1
    
    echo "RealWorldQA evaluation completed. Results in: $BASE_DIR/realworldqa/data/RealWorldQA_judged_$INFERENCE_RUN_ID.jsonl"
    echo "RealWorldQA DONE" > "$LOG_DIR/realworldqa_done.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    run_realworldqa "$PORT"
fi