#!/bin/bash
# mathvista.sh - Run MathVista benchmark

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
        --pre_prompt "$pre_prompt" \
        --model_path "$CKPT_PATH" \
        --output_dir "../results/$INFERENCE_RUN_ID$cot_suffix" \
        --output_file "output_$INFERENCE_RUN_ID$cot_suffix.json" > "$LOG_DIR/mathvista_inference$cot_suffix.log" 2>&1
    
    echo "Extracting answers for MathVista..."
    python extract_answer_w_gpt4o.py \
        --results_file_path "../results/$INFERENCE_RUN_ID$cot_suffix/output_$INFERENCE_RUN_ID$cot_suffix.json" > "$LOG_DIR/mathvista_extract$cot_suffix.log" 2>&1
    
    echo "Calculating scores for MathVista..."
    python calculate_score.py \
        --data_file_path "$BASE_DIR/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        --output_dir "../results/$INFERENCE_RUN_ID$cot_suffix" \
        --output_file "output_$INFERENCE_RUN_ID$cot_suffix.json" \
        --score_file "scores_$INFERENCE_RUN_ID$cot_suffix.json" > "$LOG_DIR/mathvista_scores$cot_suffix.log" 2>&1
    
    echo "MathVista evaluation completed. Results in: $BASE_DIR/MathVista/results/$INFERENCE_RUN_ID$cot_suffix/scores_$INFERENCE_RUN_ID$cot_suffix.json"
    echo "MathVista DONE" > "$LOG_DIR/mathvista_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    run_mathvista "$PORT"
fi