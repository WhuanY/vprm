#!/bin/bash
# run_all.sh - Main script to run all benchmarks in parallel

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Check for required files
source "$(dirname "${BASH_SOURCE[0]}")/check_raw_files.sh"
check_all_raw_files

# Create logs directory
mkdir -p "$BASE_DIR/logs/$INFERENCE_RUN_ID"

# Function to start VLLM server with a specific port
start_vllm_server() {
    local PORT="$1"
    echo "Starting VLLM server on port $PORT..."
    
    nohup vllm serve "$CKPT_PATH" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" > "$BASE_DIR/logs/$INFERENCE_RUN_ID/vllm_server_$PORT.log" 2>&1 &
    
    local SERVER_PID=$!
    echo "VLLM server started with PID: $SERVER_PID on port $PORT"
    
    # Wait for server to start
    echo "Waiting for VLLM server to initialize (30 seconds)..."
    sleep 30
    
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
    pkill -f "vllm serve.*--port $PORT" || true
}

# Function to run a benchmark in parallel
run_benchmark_parallel() {
    local BENCHMARK="$1"
    local PORT="$2"
    
    # Start VLLM server
    start_vllm_server "$PORT" || return 1
    
    # Run benchmark
    echo "Starting $BENCHMARK benchmark on port $PORT..."
    bash "$BASE_DIR/benchmarks/${BENCHMARK}.sh" "$PORT"
    
    # Stop VLLM server
    stop_vllm_server "$PORT"
    
    echo "$BENCHMARK benchmark completed on port $PORT"
}

# Run all benchmarks in parallel
echo "Starting all benchmarks in parallel..."

# Define ports for each benchmark to avoid conflicts
MATHVISTA_PORT=9753
MATHVISION_PORT=9754
MME_REALWORLD_PORT=9755
REALWORLDQA_PORT=9756

# Run benchmarks in parallel
run_benchmark_parallel "mathvista" "$MATHVISTA_PORT" &
run_benchmark_parallel "mathvision" "$MATHVISION_PORT" &
run_benchmark_parallel "mme_realworld" "$MME_REALWORLD_PORT" &
run_benchmark_parallel "realworldqa" "$REALWORLDQA_PORT" &

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