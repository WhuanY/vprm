#!/bin/bash
# run_all.sh - Main script to run all benchmarks in parallel

set -e

# =========================================================================
# 1. 设置和加载默认配置
# =========================================================================

# Generate shared timestamp BEFORE sourcing config.sh
# This ensures all benchmark scripts use the same timestamp
if [ -z "$SHARED_TIMESTAMP" ]; then
    export SHARED_TIMESTAMP=$(date +"%Y%m%d_%H%M")
    echo "Generated shared timestamp: $SHARED_TIMESTAMP"
fi

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# 检查原始数据文件
source "$(dirname "${BASH_SOURCE[0]}")/check_raw_files.sh"
check_all_raw_files
if [ $? -ne 0 ]; then
    echo "❌ Raw data integrity check failed. Please make sure all required files are downloaded before running this script."
    exit 1
fi

# =========================================================================
# 2. 解析命令行参数
# =========================================================================

# 帮助函数
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -k, --ckpt-path <path>     Path to the model checkpoint. Overrides the one in config.sh."
    echo "  -i, --run-id <id>          Manually specify a run ID. Overrides the auto-generated one."
    echo "  -b, --benchmarks <list>    Comma-separated list of benchmarks to run."
    echo "                             Available: mathvista,mathvision,mme_realworld,realworldqa,blink,chartqa,mmstar"
    echo "                             Default: all benchmarks"
    echo "  -h, --help                 Display this help message."
    echo ""
    echo "Example: $0 --ckpt-path /path/to/new/model --benchmarks mathvista,blink"
}

# 定义所有可用的benchmark
ALL_BENCHMARKS=("mathvista" "mathvision" "mme_realworld" "realworldqa" "blink" "chartqa" "mmstar")

# 解析命令行参数
SELECTED_BENCHMARKS=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -k|--ckpt-path)
        CKPT_PATH="$2"
        shift 2
        ;;
        -i|--run-id)
        INFERENCE_RUN_ID="$2"
        shift 2
        ;;
        -b|--benchmarks)
        SELECTED_BENCHMARKS="$2"
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

# 如果 CKPT_PATH 被修改，重新解析路径并生成统一目录
# Note: config.sh will handle the parsing and UNIFIED_RESULT_BASE generation
# when it's sourced. We just need to ensure CKPT_PATH is set before sourcing.
# Since we already sourced config.sh above, we need to re-source it if CKPT_PATH changed
if [ -n "$CKPT_PATH" ]; then
    # Re-source config.sh to regenerate UNIFIED_RESULT_BASE with new CKPT_PATH
    source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
fi

# 处理选择的benchmarks
if [ -z "$SELECTED_BENCHMARKS" ]; then
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
else
    # 解析逗号分隔的列表
    IFS=',' read -ra SELECTED_ARRAY <<< "$SELECTED_BENCHMARKS"
    BENCHMARKS=()
    for bench in "${SELECTED_ARRAY[@]}"; do
        bench=$(echo "$bench" | xargs)  # trim whitespace
        # 验证benchmark是否有效
        valid=0
        for valid_bench in "${ALL_BENCHMARKS[@]}"; do
            if [ "$bench" == "$valid_bench" ]; then
                valid=1
                break
            fi
        done
        if [ "$valid" -eq 1 ]; then
            BENCHMARKS+=("$bench")
        else
            echo "Warning: Unknown benchmark '$bench', skipping..."
        fi
    done
fi

# 检查是否有有效的benchmark
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    echo "Error: No valid benchmarks selected."
    exit 1
fi

# 验证GPU数量是否足够
NUM_BENCHMARKS=${#BENCHMARKS[@]}
NUM_GPUS=${#CUDA_DEVICES_ARRAY[@]}
if [ $NUM_BENCHMARKS -gt $NUM_GPUS ]; then
    echo "Error: Not enough GPUs configured!"
    echo "  Required: $NUM_BENCHMARKS GPUs (one per benchmark)"
    echo "  Configured: $NUM_GPUS GPUs in CUDA_DEVICES_LIST"
    echo ""
    echo "Please update config.sh to include at least $NUM_BENCHMARKS GPUs:"
    echo "  export CUDA_DEVICES_LIST=\"4,5,6,7,8,9,10\"  # Example for 7 GPUs"
    exit 1
fi

# Create logs directory
mkdir -p "$BASE_DIR/logs/$INFERENCE_RUN_ID"

echo "================================================="
echo "Running All Benchmarks in Parallel"
echo "-------------------------------------------------"
echo "Checkpoint: $CKPT_PATH"
echo "Run ID: $INFERENCE_RUN_ID"
echo "Selected Benchmarks (${#BENCHMARKS[@]}): ${BENCHMARKS[*]}"
echo "Available GPUs (${NUM_GPUS}): ${CUDA_DEVICES_LIST}"
echo "Base directory: $BASE_DIR"
echo "================================================="

# =========================================================================
# 3. VLLM 服务器管理函数
# =========================================================================

start_vllm_server() {
    local PORT="$1"
    local GPU_ID="$2"
    echo "[Port $PORT] Starting VLLM server on port $PORT using GPU $GPU_ID..."
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup vllm serve "$CKPT_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization 0.7 > "$BASE_DIR/logs/$INFERENCE_RUN_ID/vllm_server_$PORT.log" 2>&1 &
    
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
        cat "$BASE_DIR/logs/$INFERENCE_RUN_ID/vllm_server_$PORT.log"
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
# 4. Benchmark 配置和运行函数
# =========================================================================

# 定义每个benchmark是否依赖VLLM serve
# true = 需要VLLM serve, false = 不需要
declare -A BENCHMARK_NEEDS_SERVE=(
    ["mathvista"]="true"
    ["mathvision"]="false"
    ["mme_realworld"]="false"
    ["realworldqa"]="false"
    ["blink"]="true"
    ["chartqa"]="false"
    ["mmstar"]="false"
)

# 定义每个benchmark使用的端口（如果需要serve）
declare -A BENCHMARK_PORTS=(
    ["mathvista"]="9753"
    ["mathvision"]="0000"
    ["mme_realworld"]="0000"
    ["realworldqa"]="0000"
    ["blink"]="9751"
    ["chartqa"]="0000"
    ["mmstar"]="0000"
)

run_benchmark_parallel() {
    local BENCHMARK="$1"
    local PORT="$2"
    local GPU_ID="$3"
    
    local LOG_DIR="$BASE_DIR/logs/$INFERENCE_RUN_ID"
    local BENCHMARK_LOG="$LOG_DIR/${BENCHMARK}_run.log"
    
    echo "[$BENCHMARK] Starting benchmark on port $PORT (GPU $GPU_ID)..." | tee -a "$BENCHMARK_LOG"
    
    # 构建基准测试脚本的命令行参数
    local BENCHMARK_ARGS=()
    BENCHMARK_ARGS+=("--ckpt-path" "$CKPT_PATH")
    BENCHMARK_ARGS+=("--gpu-id" "$GPU_ID")
    BENCHMARK_ARGS+=("--run-id" "$INFERENCE_RUN_ID")
    
    if [ "$PORT" != "0000" ]; then
        BENCHMARK_ARGS+=("--port" "$PORT")
        
        # 启动 VLLM 服务器
        start_vllm_server "$PORT" "$GPU_ID" || {
            echo "[$BENCHMARK] ✗ Failed to start VLLM server" | tee -a "$BENCHMARK_LOG"
            return 1
        }
    else
        echo "[$BENCHMARK] Does not require VLLM server (using direct inference)" | tee -a "$BENCHMARK_LOG"
    fi
    
    # 运行 benchmark 脚本
    echo "[$BENCHMARK] Running benchmark evaluation..." | tee -a "$BENCHMARK_LOG"
    if bash "$BASE_DIR/benchmark_runner/benchmarks/${BENCHMARK}.sh" "${BENCHMARK_ARGS[@]}" >> "$BENCHMARK_LOG" 2>&1; then
        echo "[$BENCHMARK] ✓ Benchmark completed successfully" | tee -a "$BENCHMARK_LOG"
        BENCHMARK_RESULT=0
    else
        echo "[$BENCHMARK] ✗ Benchmark failed" | tee -a "$BENCHMARK_LOG"
        BENCHMARK_RESULT=1
    fi
    
    # 停止 VLLM 服务器（如果需要）
    if [ "$PORT" != "0000" ]; then
        stop_vllm_server "$PORT"
    fi
    
    return $BENCHMARK_RESULT
}

# =========================================================================
# 5. 自动分配GPU和端口
# =========================================================================

# 动态分配GPU和端口
# 为每个benchmark分配一个独立的GPU
# 为需要serve的benchmark分配不同的端口
declare -A ASSIGNED_GPUS
declare -A ASSIGNED_PORTS

PORT_BASE=9750
port_counter=1
gpu_counter=0

for BENCHMARK in "${BENCHMARKS[@]}"; do
    # 分配GPU（每个benchmark一个）
    ASSIGNED_GPUS["$BENCHMARK"]="${CUDA_DEVICES_ARRAY[$gpu_counter]}"
    gpu_counter=$((gpu_counter + 1))
    
    # 分配端口（仅对需要serve的benchmark）
    if [ "${BENCHMARK_NEEDS_SERVE[$BENCHMARK]}" == "true" ]; then
        ASSIGNED_PORTS["$BENCHMARK"]="$((PORT_BASE + port_counter))"
        port_counter=$((port_counter + 1))
    else
        ASSIGNED_PORTS["$BENCHMARK"]="0000"
    fi
    
    echo "[$BENCHMARK] Assigned GPU: ${ASSIGNED_GPUS[$BENCHMARK]}, Port: ${ASSIGNED_PORTS[$BENCHMARK]}"
done

# =========================================================================
# 6. 主执行逻辑 - 并行运行所有基准测试
# =========================================================================

echo ""
echo "Starting all benchmarks in parallel..."
echo ""

# 启动所有基准测试作为后台进程
declare -A PIDS
for BENCHMARK in "${BENCHMARKS[@]}"; do
    PORT="${ASSIGNED_PORTS[$BENCHMARK]}"
    GPU_ID="${ASSIGNED_GPUS[$BENCHMARK]}"
    
    run_benchmark_parallel "$BENCHMARK" "$PORT" "$GPU_ID" &
    PIDS["$BENCHMARK"]=$!
    echo "[$BENCHMARK] Started with PID: ${PIDS[$BENCHMARK]} (GPU: $GPU_ID, Port: $PORT)"
done

echo ""
echo "All benchmarks started. Waiting for completion..."
echo ""

# 等待所有基准测试完成并收集结果
FAILED_BENCHMARKS=()
for BENCHMARK in "${BENCHMARKS[@]}"; do
    PID="${PIDS[$BENCHMARK]}"
    if wait $PID; then
        echo "[$BENCHMARK] ✓ Completed successfully"
    else
        echo "[$BENCHMARK] ✗ Failed"
        FAILED_BENCHMARKS+=("$BENCHMARK")
    fi
done

# =========================================================================
# 7. 清理和总结
# =========================================================================

# 确保所有 VLLM 服务器都已停止
echo ""
echo "Stopping all VLLM servers..."
for BENCHMARK in "${BENCHMARKS[@]}"; do
    PORT="${ASSIGNED_PORTS[$BENCHMARK]}"
    if [ "$PORT" != "0000" ]; then
        stop_vllm_server "$PORT"
    fi
done

echo ""
echo "================================================="
if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo "✓ All benchmarks completed successfully!"
else
    echo "✗ Some benchmarks failed:"
    for BENCHMARK in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $BENCHMARK"
    done
fi
echo "================================================="
echo ""
echo "Results are available in the following locations:"
echo "- MathVista: $BASE_DIR/MathVista/results/$INFERENCE_RUN_ID/"
echo "- MathVision: $BASE_DIR/MathVision/data/"
echo "- MME-RealWorld-Lite: $BASE_DIR/MME-RealWorld-Lite/data/"
echo "- RealWorldQA: $BASE_DIR/realworldqa/data/"
echo "- BLINK: $BASE_DIR/BLINK_Benchmark/eval/results/"
echo "- ChartQA: $BASE_DIR/ChartQA/data/"
echo "- MMStar: $BASE_DIR/MMStar/data/"
echo ""
echo "Logs are available in: $BASE_DIR/logs/$INFERENCE_RUN_ID/"
echo ""

if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
    exit 1
fi
