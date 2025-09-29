#!/bin/bash
# mathvista.sh - Run MathVista benchmark

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

run_mathvista() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "================================"
    echo "Evaluating MathVista..."
    echo "=============================="
    
    cd "$BASE_DIR/MathVista/evaluation" || { 
        echo "Error: MathVista directory not found"; 
        return 1; 
    }
    
    echo "Generating responses for MathVista..."
    python local_generate_response.py \
        --inference_api "http://localhost:$PORT/v1" \
        --data_file_path "$BASE_DIR/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        --model_path "$CKPT_PATH" \
        --output_dir "../results/$INFERENCE_RUN_ID" \
        --output_file "output_$INFERENCE_RUN_ID.json" > "$LOG_DIR/mathvista_inference.log" 2>&1
    
    echo "Extracting answers for MathVista..."
    python extract_answer_w_gpt4o.py \
        --results_file_path "../results/$INFERENCE_RUN_ID/output_$INFERENCE_RUN_ID.json" > "$LOG_DIR/mathvista_extract.log" 2>&1
    
    echo "Calculating scores for MathVista..."
    python calculate_score.py \
        --data_file_path "$BASE_DIR/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        --output_dir "../results/$INFERENCE_RUN_ID" \
        --output_file "output_$INFERENCE_RUN_ID.json" \
        --score_file "scores_$INFERENCE_RUN_ID.json" > "$LOG_DIR/mathvista_scores.log" 2>&1
    
    echo "MathVista evaluation completed. Results in: $BASE_DIR/MathVista/results/$INFERENCE_RUN_ID/scores_$INFERENCE_RUN_ID.json"
    echo "MathVista DONE" > "$LOG_DIR/mathvista_done.flag" # TODO: Flag 是啥
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    run_mathvista "$PORT"
fi