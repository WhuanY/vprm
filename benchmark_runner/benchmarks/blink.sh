#!/bin/bash
# blink.sh - Run BLINK benchmark

# 脚本将在遇到任何错误时立即退出
set -e

# =========================================================================
# 1. 设置和加载默认配置
# =========================================================================

# 首先，加载 config.sh 中的默认值
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh" || echo "Warning: config.sh not found or has errors, proceeding with script defaults."

# 然后，设置此脚本自身的默认值
use_cot=1
gpu_id=0
bz=50
task_name="all"

# =========================================================================
# 2. 解析命令行参数 (覆盖默认值)
# =========================================================================

# 帮助函数
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -k, --ckpt-path <path>     Path to the model checkpoint. Overrides the one in config.sh."
    echo "                             (Default from config.sh: $CKPT_PATH)"
    echo "  -g, --gpu-id <id>          Specify the GPU ID to use. (Default: $gpu_id)"
    echo "  -i, --run-id <id>          Manually specify a run ID. Overrides the auto-generated one."
    echo "  -c, --use-cot <0|1>        Whether to use Chain of Thought. 1 for yes, 0 for no. (Default: $use_cot)"
    echo "  -t, --task-name <name>     Task name for BLINK. (Default: $task_name)"
    echo "  -b, --bz <size>            Batch size for inference. (Default: $bz)"
    echo "  -h, --help                 Display this help message."
    echo ""
    echo "Example: $0 --ckpt-path /path/to/new/model --gpu-id 1 --use-cot 0"
}

# 创建一个临时变量来判断 run-id 是否被手动设置
manual_run_id_provided=0

# 解析循环
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -k|--ckpt-path)
        CKPT_PATH="$2"
        shift 2
        ;;
        -g|--gpu-id)
        gpu_id="$2"
        shift 2
        ;;
        -i|--run-id)
        INFERENCE_RUN_ID="$2"
        manual_run_id_provided=1 # 标记 run-id 是手动指定的
        shift 2
        ;;
        -c|--use-cot)
        use_cot="$2"
        shift 2
        ;;
        -t|--task-name)
        task_name="$2"
        shift 2
        ;;
        -b|--bz)
        bz="$2"
        shift 2
        ;;
        -h|--help)
        usage
        exit 0
        ;;
        *)    # 未知选项
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
done

# =========================================================================
# 3. Finalize Configuration and Environment
# =========================================================================

# 核心逻辑：如果 run-id 不是手动指定的，config.sh 会处理
if [ "$manual_run_id_provided" -eq 0 ]; then
    if [ -z "$CKPT_PATH" ]; then
        echo "Error: CKPT_PATH is not set. Please set it in config.sh or provide it with --ckpt-path."
        exit 1
    fi
    # Re-source config.sh to regenerate UNIFIED_RESULT_BASE with potentially updated CKPT_PATH
    source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
fi

# 根据最终的 gpu_id 设置 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpu_id

# 根据最终的 use_cot 值设置 prompt
if [ "$use_cot" -eq 1 ]; then
    cot_prompt_settings="Using CoT inference"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    cot_prompt_settings="Not using CoT inference"
    pre_prompt=""
    cot_suffix=""
fi

# 打印最终生效的配置
echo "================================================="
echo "Final Configuration for BLINK Run:"
echo "-------------------------------------------------"
echo "GPU ID in use (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Model Checkpoint Path: $CKPT_PATH"
echo "Inference Run ID: $INFERENCE_RUN_ID"
echo "Unified Result Base: $UNIFIED_RESULT_BASE"
echo "CoT Setting: $cot_prompt_settings"
echo "Task Name: $task_name"
echo "Batch Size (bz): $bz"
echo "================================================="


# =========================================================================
# 4. 主逻辑 (使用最终确定的变量值)
# =========================================================================

run_blink() {
    # New structure: results/$UNIFIED_RESULT_DIR/blink/$SHARED_TIMESTAMP/
    local BENCHMARK_NAME="blink${cot_suffix}"
    local BENCHMARK_DIR="$UNIFIED_RESULT_BASE/$BENCHMARK_NAME"
    local LOG_DIR="$BENCHMARK_DIR/$SHARED_TIMESTAMP"
    mkdir -p "$LOG_DIR"
    
    echo "================================"
    echo "Evaluating BLINK..."
    echo "=============================="

    
    cd "$BASE_DIR/BLINK_Benchmark" || { 
        echo "Error: BLINK_Benchmark directory not found"; 
        return 1; 
    }

    cd $BASE_DIR/BLINK_Benchmark/eval
    
    # Set unified output directories
    BLINK_OUTPUT_DIR="$BENCHMARK_DIR/$SHARED_TIMESTAMP"
    # Fixed image directory under BLINK_Benchmark to avoid space waste
    BLINK_IMAGE_DIR="$BASE_DIR/BLINK_Benchmark/images"
    
    # Ensure directories exist
    mkdir -p "$BLINK_OUTPUT_DIR"
    mkdir -p "$BLINK_IMAGE_DIR"
    
    echo "Generating responses for BLINK..."
    echo "Output directory: $BLINK_OUTPUT_DIR"
    python test_benchmark.py \
        --model_name_or_path "$CKPT_PATH" \
        --dataset_local_path ../data \
        --task_name "$task_name" \
        --pre_prompt "$pre_prompt" \
        --output_save_folder "$BLINK_OUTPUT_DIR" \
        --image_save_folder "$BLINK_IMAGE_DIR" \
        --batch_size $bz \
        --tp "$VLLM_TENSOR_PARALLEL_SIZE" \
        --regen \
        > "$LOG_DIR/blink_inference$cot_suffix.log" 2>&1

    echo "Evaluating BLINK predictions..."
    python evaluate.py \
        --model_name_or_path "$CKPT_PATH" \
        --output_save_folder "$BLINK_OUTPUT_DIR" \
        --prediction_output_dir "$BLINK_OUTPUT_DIR" \
        > "$LOG_DIR/blink_evaluation$cot_suffix.log" 2>&1
    
    echo "BLINK evaluation completed."
    echo "Logs are in: $LOG_DIR"
    echo "Results are in: $BLINK_OUTPUT_DIR"
    echo "  - Task outputs: $BLINK_OUTPUT_DIR/*.json"
    echo "  - Predictions: $BLINK_OUTPUT_DIR/val_predictions_*.json"
    echo "  - Results: $BLINK_OUTPUT_DIR/val_results_*.json"
    echo "BLINK DONE" > "$LOG_DIR/blink_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 直接调用函数，不再需要传递任何参数
    run_blink
fi
