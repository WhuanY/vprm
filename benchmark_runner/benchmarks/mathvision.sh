#!/bin/bash
# mathvision.sh - Run MathVision benchmark

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

use_cot=1
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt="Please solve the problem step by step and put your answer in one \"\\boxed{}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{}\"."
    cot_suffix=""
fi

run_mathvision() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local GPU_ID="$2"  # 接收 GPU ID 参数
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "================================"
    echo "Evaluating MathVision..."
    echo "================================"
    
    cd "$BASE_DIR/MathVision" || { 
        echo "Error: MathVision directory not found"; 
        return 1; 
    }
    
    echo "Data preprocessing for MathVision..."
    mkdir -p "data"
    
    # Convert data formats
    python parquet_to_json.py \
        --input_file "data/test-00000-of-00001-3532b8d3f1b4047a.parquet" \
        --output_file "data/MathVision_test.json" > "$LOG_DIR/mathvision_preprocess_test.log" 2>&1
    
    python parquet_to_json.py \
        --input_file "data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" \
        --output_file "data/MathVision_testmini.json" > "$LOG_DIR/mathvision_preprocess_testmini.log" 2>&1
    
    echo "Generating responses for MathVision..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/MathVision_$MATHVISION_SUBSET.json" \
        --save_name "data/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
        --pre_prompt "$pre_prompt" \
        --tp 1 \
        --bz 1 \
        --max_new_tokens 8000 > "$LOG_DIR/mathvision_inference$cot_suffix.log" 2>&1 
    
    echo "Evaluating MathVision responses..."
    # Check if MathVision subdirectory exists
    if [ -d "MathVision" ]; then
        cd "MathVision" || exit
    else
        # Create outputs directory if not in MathVision subdirectory
        mkdir -p "outputs"
    fi
    
    # # Copy results to outputs directory
    # cp "$BASE_DIR/MathVision/data/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
    #    "$BASE_DIR/MathVision/outputs/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl"
    
    # # Evaluate
    # python "$BASE_DIR/MathVision/evaluation/evaluate.py" \
    #     --eval_file "MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" > "$LOG_DIR/mathvision_eval$cot_suffix.log" 2>&1
    python judge.py \
    --input_file "data/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID$cot_suffix.jsonl" \
    --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
    --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
    --output_file "data/MathVision-${MATHVISION_SUBSET}_judged_$INFERENCE_RUN_ID$cot_suffix.jsonl" > "$LOG_DIR/mathvision_judge$cot_suffix.log" 2>&1
    
    echo "MathVision DONE" > "$LOG_DIR/mathvision_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    GPU_ID="$2"  # 接收 GPU ID 参数
    run_mathvision "$PORT" "$GPU_ID"
fi