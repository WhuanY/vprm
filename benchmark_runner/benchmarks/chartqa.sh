#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

split="test"
use_cot=1
if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    pre_prompt="Please try to answer the question with short words or phrases if possible."
    cot_suffix=""
fi


run_chartqa() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local GPU_ID="$2"
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"

    echo "=============================="
    echo "Evaluating ChartQA on $split set..."
    echo "=============================="

    cd "$BASE_DIR/ChartQA" || { 
        echo "Error: ChartQA directory not found"; 
        exit 1; 
    }

    mkdir -p "data"

    echo "Data preprocessing for ChartQA..."
    python parquet_to_json.py --split "$split" --output_file "data/chartQA_${split}.json" --split_human_machine
    echo "Generating Responses for ChartQA..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --model_name_or_path "$CKPT_PATH" \
    --pre_prompt "$pre_prompt" \
    --use_cot $use_cot \
    --input_file "data/chartQA_${split}.json" \
    --save_name "data/chartQA_${split}_inferenced_${INFERENCE_RUN_ID}${cot_suffix}.jsonl" \
    --tp 1 \
    --bz 1 \
    --max_new_tokens 8000 > "$LOG_DIR/chartQA_${split}_inferenced_${INFERENCE_RUN_ID}${cot_suffix}.log" 2>&1

    echo "Calculating scores for ChartQA..."
    python judge.py \
        --input_file "data/chartQA_${split}_inferenced${cot_suffix}.jsonl" \
        --judge_api "$CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT" \
        --api_key "$CUSTOMIZED_REMOTE_OPENAI_API_KEY" \
        --use_relax_accuracy \
        --output_file "data/chartQA_${split}_judged_${INFERENCE_RUN_ID}${cot_suffix}.jsonl" > "$LOG_DIR/chartqa_judge${cot_suffix}.log" 2>&1
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then 
    PORT="$1"
    GPU_ID="$2"
    run_chartqa "$PORT" "$GPU_ID"
fi