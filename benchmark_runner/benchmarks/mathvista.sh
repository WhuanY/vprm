#!/bin/bash
# mathvista.sh - Run MathVista benchmark

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
    echo "  -b, --bz <size>            Batch size (number of concurrent threads) for inference. (Default: $bz)"
    echo "  -h, --help                 Display this help message."
    echo ""
    echo "Example: $0 --ckpt-path /path/to/new/model --gpu-id 1 --use-cot 0 --bz 50"
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
# 但我们需要确保 UNIFIED_RESULT_BASE 已生成（config.sh 会处理）
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

# 确保统一输出目录存在
mkdir -p "$UNIFIED_RESULT_BASE/mathvista${cot_suffix}"

# 打印最终生效的配置
echo "================================================="
echo "Final Configuration for MathVista Run:"
echo "-------------------------------------------------"
echo "GPU ID in use (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Model Checkpoint Path: $CKPT_PATH"
echo "Inference Run ID: $INFERENCE_RUN_ID"
echo "Unified Result Base: $UNIFIED_RESULT_BASE"
echo "VLLM Inference Port: $VLLM_INFERENCE_PORT"
echo "CoT Setting: $cot_prompt_settings"
echo "Batch Size (bz): $bz"
echo "================================================="


# =========================================================================
# 4. VLLM Server Management Functions
# =========================================================================

# Function to start VLLM server
start_vllm_server() {
    local PORT="$1"
    local GPU_ID="$2"
    local LOG_DIR="$3"
    echo "[Port $PORT] Starting VLLM server on port $PORT using GPU $GPU_ID..."
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup vllm serve "$CKPT_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization 0.7 > "$LOG_DIR/vllm_server_$PORT.log" 2>&1 &
    
    local SERVER_PID=$!
    echo "[Port $PORT] VLLM server started with PID: $SERVER_PID on GPU $GPU_ID"
    
    # Wait for server to start
    echo "[Port $PORT] Waiting for VLLM server to initialize..."
    local max_wait=300  # 5 minutes
    local wait_interval=5
    local elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null | grep -q "200"; then
            echo "[Port $PORT] ✓ VLLM server is ready"
            return 0
        fi
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "[Port $PORT] Still waiting... (${elapsed}s/${max_wait}s)"
        fi
    done
    
    # Check if server process is still running
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "[Port $PORT] ✗ Error: VLLM server failed to start"
        cat "$LOG_DIR/vllm_server_$PORT.log"
        return 1
    fi
    
    echo "[Port $PORT] ✗ Error: VLLM server did not become healthy within ${max_wait}s"
    return 1
}

# Function to stop VLLM server
stop_vllm_server() {
    local PORT="$1"
    if [ "$PORT" == "0000" ]; then
        return 0
    fi
    echo "[Port $PORT] Stopping VLLM server..."
    pkill -f "vllm serve.*--port $PORT" || true
    sleep 2
}

# =========================================================================
# 5. 主逻辑 (使用最终确定的变量值)
# =========================================================================

run_mathvista() {
    local PORT="${1:-$VLLM_INFERENCE_PORT}"
    local LOG_DIR="$UNIFIED_RESULT_BASE"
    mkdir -p "$LOG_DIR"
    
    echo "================================"
    echo "Evaluating MathVista..."
    echo "=============================="
    
    cd "$BASE_DIR/MathVista/evaluation" || { 
        echo "Error: MathVista directory not found"; 
        return 1; 
    }

    # Start VLLM server
    echo "Starting VLLM server for MathVista..."
    if ! start_vllm_server "$PORT" "$gpu_id" "$LOG_DIR"; then
        echo "ERROR: Failed to start VLLM server"
        return 1
    fi
    
    # Set inference endpoint
    export INFERENCE_ENDPOINT="http://localhost:$PORT/v1"
    
    echo "Generating responses for MathVista..."
    python local_generate_response.py \
        --inference_api "http://localhost:$PORT/v1" \
        --data_file_path "$BASE_DIR/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        --rerun \
        --pre_prompt "$pre_prompt" \
        --model_path "$CKPT_PATH" \
        --output_dir "$UNIFIED_RESULT_BASE/mathvista${cot_suffix}" \
        --num_threads $bz \
        --output_file "mathvista_output${cot_suffix}.json" > "$LOG_DIR/mathvista_inference$cot_suffix.log" 2>&1
    
    echo "Extracting answers for MathVista..."
    python extract_answer_w_gpt4o.py \
        --rerun \
        --num_threads $bz \
        --results_file_path "$UNIFIED_RESULT_BASE/mathvista$cot_suffix/mathvista_output$cot_suffix.json" > "$LOG_DIR/mathvista_extract$cot_suffix.log" 2>&1
    
    echo "Calculating scores for MathVista..."
    python calculate_score.py \
        --data_file_path "$BASE_DIR/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
        --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
        --output_dir "$UNIFIED_RESULT_BASE/mathvista$cot_suffix" \
        --output_file "mathvista_output$cot_suffix.json" \
        --score_file "mathvista_scores$cot_suffix.json" > "$LOG_DIR/mathvista_scores$cot_suffix.log" 2>&1
    
    echo "MathVista evaluation completed. Results in: $UNIFIED_RESULT_BASE/mathvista$cot_suffix/mathvista_scores$cot_suffix.json"
    echo "MathVista DONE" > "$LOG_DIR/mathvista_done$cot_suffix.flag"
    
    # Stop VLLM server
    echo "Stopping VLLM server..."
    stop_vllm_server "$PORT"
}

# Run the benchmark if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 直接调用函数，不再需要传递任何参数
    run_mathvista
fi
