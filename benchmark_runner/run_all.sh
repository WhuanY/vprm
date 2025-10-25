#!/bin/bash
# run_all.sh - Main script to run all benchmarks in parallel

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Check for required files
source "$(dirname "${BASH_SOURCE[0]}")/check_raw_files.sh"
check_all_raw_files
if [ $? -ne 0 ]; then
    echo "❌ Raw data integrity check failed. Please make sure all required files are downloaded before running this script."
    exit 1
fi

# Create logs directory
mkdir -p "$BASE_DIR/logs/$INFERENCE_RUN_ID"

start_vllm_server() {
    local PORT="$1"
    local GPU_ID="$2"  # 接收 GPU ID 参数
    echo "Starting VLLM server on port $PORT using GPU $GPU_ID..."
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup vllm serve "$CKPT_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" 
        --gpu_memory_utilization 0.7 > "$BASE_DIR/logs/$INFERENCE_RUN_ID/vllm_server_$PORT.log" 2>&1 &
    
    local SERVER_PID=$!
    echo "VLLM server started with PID: $SERVER_PID on port $PORT using GPU $GPU_ID"
    
    # Wait for server to start
    echo "Waiting for VLLM server to initialize (30 seconds)..."
    sleep 40
    
    # Check if server is running
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Error: VLLM server failed to start on port $PORT"
        cat "$BASE_DIR/logs/$INFERENCE_RUN_ID/vllm_server_$PORT.log"
        return 1
    fi
    
    echo "VLLM server ready on port $PORT"
    return 0
}

# Function to stop VLLM server
stop_vllm_server() {
    local PORT="$1"
    echo "Stopping VLLM server on port $PORT..."
    if [ "$PORT" == "0000" ]; then
        echo "No VLLM server to stop for port $PORT."
        return 0
    fi
    pkill -f "vllm serve.*--port $PORT" || true
}

# Function to run a benchmark in parallel
run_benchmark_parallel() {
    local BENCHMARK="$1"
    local PORT="$2"
    local GPU_ID="$3"  # 接收 GPU ID 参数
    
    # Start VLLM server
    if [ "$PORT" == "0000" ]; then
    echo "$BENCHMARK does not require VLLM server. Skipping server startup."
    bash "$BASE_DIR/benchmark_runner/benchmarks/${BENCHMARK}.sh" "$PORT" "$GPU_ID"
    else
        echo "$BENCHMARK requires VLLM server. Starting server..."

        # 检查 GPU_ID 是否设置
        if [ -z "$GPU_ID" ]; then
            echo "Error: GPU_ID is not set. Cannot start VLLM server."
            exit 1
        fi

        # 启动 VLLM 服务器
        start_vllm_server "$PORT" "$GPU_ID" || {
            echo "Error: Failed to start VLLM server on port $PORT with GPU $GPU_ID."
            exit 1
        }

        # 在这里之前，确保server已经启动成功
        sleep 300

        # 确认server已经启动成功    
        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; then
            echo "ERROR: VLLM server is not ready after 5 minutes wait." Try extend the sleep time or check the server logs.
            stop_vllm_server "$PORT"
            exit 1
        fi
        bash "$BASE_DIR/benchmark_runner/benchmarks/${BENCHMARK}.sh" "$PORT" "$GPU_ID"

        echo "VLLM server started successfully on port $PORT with GPU $GPU_ID."
        
        # Run benchmark
        echo "Starting $BENCHMARK benchmark on port $PORT with GPU $GPU_ID..."
    fi
    
    
    # Stop VLLM server
    stop_vllm_server "$PORT"
    
    echo "$BENCHMARK benchmark completed on port $PORT"
}

# Run all benchmarks in parallel
echo "Starting all benchmarks in parallel..."

# Define ports for each benchmark to avoid conflicts
MATHVISTA_PORT=9753
MATHVISION_PORT=0000 # MATHVISION推理不用vllm, 目前这个比较慢
MME_REALWORLD_PORT=0000 # MME-RealWorld-Lite推理不用vllm，比较慢
REALWORLDQA_PORT=0000 # RealWorldQA推理不用vllm，比较慢
BLINK_PORT=9751

# Run benchmarks in parallel
run_benchmark_parallel "mathvista" "$MATHVISTA_PORT" "${CUDA_DEVICES_ARRAY[0]}"&
run_benchmark_parallel "mathvision" "$MATHVISION_PORT" "${CUDA_DEVICES_ARRAY[1]}"&
run_benchmark_parallel "mme_realworld" "$MME_REALWORLD_PORT" "${CUDA_DEVICES_ARRAY[2]}"&
run_benchmark_parallel "realworldqa" "$REALWORLDQA_PORT" "${CUDA_DEVICES_ARRAY[3]}"&
run_benchmark_parallel "blink" "$BLINK_PORT" "${CUDA_DEVICES_ARRAY[4]}"&

# Wait for all benchmarks to complete
wait

echo "=============================="
echo "All benchmarks completed."
echo "Results are available in the following locations:"
echo "- MathVista: $BASE_DIR/MathVista/results/$INFERENCE_RUN_ID/scores_$INFERENCE_RUN_ID.json"
echo "- MathVision: $BASE_DIR/MathVision/outputs/evaluation_results.json"
echo "- MME-RealWorld-Lite: $BASE_DIR/MME-RealWorld-Lite/data/MME-RealWorld-Lite_judged_$INFERENCE_RUN_ID.jsonl"
echo "- RealWorldQA: $BASE_DIR/realworldqa/data/RealWorldQA_judged_$INFERENCE_RUN_ID.jsonl"
echo "Logs are available in: $BASE_DIR/logs/$INFERENCE_RUN_ID/"
