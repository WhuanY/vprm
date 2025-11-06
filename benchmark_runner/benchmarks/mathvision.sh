#!/bin/bash
# mathvision.sh - Run MathVision benchmark

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
bz=100
# Use MATHVISION_SUBSET from config.sh as default, or "testmini" if not set
mathvision_subset="${MATHVISION_SUBSET:-testmini}"

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
    echo "  -p, --port <port>          Specify the VLLM inference port. (Default from config.sh: $VLLM_INFERENCE_PORT)"
    echo "  -s, --subset <subset>      Dataset subset: 'testmini' or 'test'. (Default: $mathvision_subset)"
    echo "  -b, --bz <size>            Batch size for inference. (Default: $bz)"
    echo "  -h, --help                 Display this help message."
    echo ""
    echo "Example: $0 --ckpt-path /path/to/new/model --gpu-id 1 --subset test"
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
        -p|--port)
        VLLM_INFERENCE_PORT="$2"
        shift 2
        ;;
        -s|--subset)
        mathvision_subset="$2"
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
    pre_prompt="Please solve the problem step by step and put your answer in one \"\\boxed{}\". If it is a multiple choice question, only one letter is allowed in the \"\\boxed{}\"."
    cot_suffix=""
fi

# Validate subset
if [[ "$mathvision_subset" != "testmini" && "$mathvision_subset" != "test" ]]; then
    echo "Error: MATHVISION_SUBSET must be 'testmini' or 'test'"
    exit 1
fi

# 确保统一输出目录存在
mkdir -p "$UNIFIED_RESULT_BASE"

# 打印最终生效的配置
echo "================================================="
echo "Final Configuration for MathVision Run:"
echo "-------------------------------------------------"
echo "GPU ID in use (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Model Checkpoint Path: $CKPT_PATH"
echo "Inference Run ID: $INFERENCE_RUN_ID"
echo "Unified Result Base: $UNIFIED_RESULT_BASE"
echo "VLLM Inference Port: $VLLM_INFERENCE_PORT"
echo "CoT Setting: $cot_prompt_settings"
echo "Dataset Subset: $mathvision_subset"
echo "Batch Size (bz): $bz"
echo "================================================="


# =========================================================================
# 4. 主逻辑 (使用最终确定的变量值)
# =========================================================================

run_mathvision() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local LOG_DIR="$UNIFIED_RESULT_BASE"
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
    CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --input_file "data/MathVision_$mathvision_subset.json" \
        --save_name "$UNIFIED_RESULT_BASE/mathvision_${mathvision_subset}_inferenced${cot_suffix}.jsonl" \
        --pre_prompt "$pre_prompt" \
        --tp 1 \
        --bz "$bz" \
        --max_new_tokens 8000 > "$LOG_DIR/mathvision_inference$cot_suffix.log" 2>&1 
    
    echo "Evaluating MathVision responses..."
    python judge.py \
    --input_file "$UNIFIED_RESULT_BASE/mathvision_${mathvision_subset}_inferenced${cot_suffix}.jsonl" \
    --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
    --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
    --output_file "$UNIFIED_RESULT_BASE/mathvision_${mathvision_subset}_judged${cot_suffix}.jsonl" > "$LOG_DIR/mathvision_judge$cot_suffix.log" 2>&1
    
    echo "MathVision evaluation completed."
    echo "Logs are in: $LOG_DIR"
    echo "MathVision DONE" > "$LOG_DIR/mathvision_done$cot_suffix.flag"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 直接调用函数，不再需要传递任何参数
    run_mathvision
fi
