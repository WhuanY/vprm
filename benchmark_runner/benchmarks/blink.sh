source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
task_name="all"
use_cot=1 

if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    after_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    after_prompt=""
    cot_suffix=""
fi


run_blink() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    mkdir -p "$LOG_DIR"
    
    echo "================================"
    echo "Evaluating BLINK..."
    echo "=============================="

    
    cd "$BASE_DIR/BLINK_Benchmark" || { 
        echo "Error: BLINK_Benchmark directory not found"; 
        return 1; 
    }

    cd $BASE_DIR/BLINK_Benchmark/eval
    export INFERENCE_ENDPOINT="http://localhost:$PORT/v1" # Make sure this var is set before running inference
    mkdir -p ../logs
    echo "Generating responses for BLINK..."
    python test_benchmark.py \
        --model_name_or_path $CKPT_PATH \
        --inference_api $INFERENCE_ENDPOINT \
        --dataset_local_path ../data \
        --task_name $task_name \
        --after_prompt "$after_prompt" \
        >> $LOG_DIR/blink_inference$cot_suffix.log 2>&1

    cd $BASE_DIR/BLINK_Benchmark/eval
    python evaluate.py --model_name_or_path $CKPT_PATH >> $LOG_DIR/blink_evaluation.log 2>&1
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    PORT="$1"
    run_blink "$PORT"
fi
