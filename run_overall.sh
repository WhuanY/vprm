# 一个可以把全部benchmark都过一遍的脚本

# 需要预先安装好下面的包
# pip install latex2sympy2

export CUDA_VISIBLE_DEVICES="5,6,7,8"
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1" # judge, MathVista依赖这个环境变量
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxx" # judge_api， MathVista依赖这个环境变量

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=2

VLLM_INFERENCE_PORT=9753 # VLLM推理服务端口
MATHVISION_SUBSET="test" # "testmini" or "test"
IMAGE_BASE_DIR_MME_REALWORLD_LITE="/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs" # MME-REALWORLD-LITE数据集的图片存放路径 
assert() { eval "[[ $1 ]]" || { echo "Assertion failed: $1" >&2; exit 1; }; }




CKPT_PATH="/path/to/your/vprm_checkpoint"
MME_REALWORLD_LITE_IMAGE_DIR="/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs" # MME-REALWORLD-LITE数据集的图片存放路径

now() { date +"%Y%m%d_%H%M%S"; } # YYYYMMDD_HHMMSS

# 获取CKPT_PATH的文件名
CKPT_NAME=$(basename "$CKPT_PATH")
INFERENCE_RUN_ID="${CKPT_NAME}_$(now)"  
echo "Checkpoint name: $CKPT_NAME"
echo "Inference run ID: $INFERENCE_RUN_ID"

# 获取这个脚本所在的目录
BASE_DIR=$(cd "$(dirname "$0")" || exit; pwd)

echo "Using checkpoint: $CKPT_PATH"

check_raw_files() {
    MATHVISION_DIR="$BASE_DIR/MathVision"
    # 检查 MathVision 数据
    mkdir -p "$MATHVISION_DIR/data"  # 创建data目录
    if [ ! -f "$MATHVISION_DIR/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" ]; then
        echo "MathVision testmini data not found. Downloading..."
        wget -O "$MATHVISION_DIR/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" \
        "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet"
    fi

    mkdir -p "$MATHVISION_DIR/data"  # 创建data目录
    if [ ! -f "$MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet" ]; then
        echo "MathVision test data not found. Downloading..."
        wget -O "$MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet" \
        "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet"

    fi

    mkdir -p "$MATHVISION_DIR/data"  # 创建data目录
    if [ ! -f "$MATHVISION_DIR/data/test.jsonl" ]; then
        echo "MathVision evaluation script data dependency not found. Downloading..."
        # 这里直接给出下载连接，然后退出脚本
        wget -O "$MATHVISION_DIR/data/test.jsonl" https://github.com/mathllm/MATH-V/raw/main/data/test.jsonl
    fi

    if [ $(find "$MATHVISION_DIR/images" -type f | wc -l) -ne 3040 ]; then
        echo "MathVision images count is not 3040."
    fi

    echo "MathVision data check completed."
    MATHVISTA_DIR="$BASE_DIR/MathVista"
    # 检查 MathVista 数据
    if [ ! -f "$MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" ]; then
        echo "MathVista testmini data not found. Downloading..."
        wget -O "$MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        "https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
    fi

    if [ $(find "$MATHVISTA_DIR/images" -type f | wc -l) -ne 6141 ]; then
        echo "MathVista images count is not 6141."
    fi

    echo "MathVista data check completed."

    MME_REALWORLD_LITE_DIR="$BASE_DIR/MME-RealWorld-Lite"
    
    # 检查 MME-RealWorld-Lite 数据
    if [ ! -f "$MME_REALWORLD_LITE_DIR/data/MME-RealWorld-Lite.json" ]; then
        echo "MME-RealWorld-Lite JSON data not found."
    fi

    if [ $(find "$MME_REALWORLD_LITE_DIR/images" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l) -lt 1543 ]; then
        echo "MME-RealWorld-Lite images count is less than 1543."
    fi

    echo "MME-RealWorld-Lite data check completed."

    REALWORLDQA_DIR="$BASE_DIR/realworldqa"
    
    # 检查 RealWorldQA 数据
    if [ ! -f "$REALWORLDQA_DIR/data/test-00000-of-00002.parquet" ]; then
        echo "RealWorldQA test-00000-of-00002 data not found. Downloading..."
        wget -O "$REALWORLDQA_DIR/data/test-00000-of-00002.parquet" \
        "https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00000-of-00002.parquet"
    fi

    if [ ! -f "$REALWORLDQA_DIR/data/test-00001-of-00002.parquet" ]; then
        echo "RealWorldQA test-00001-of-00002 data not found. Downloading..."
        wget -O "$REALWORLDQA_DIR/data/test-00001-of-00002.parquet" \
        "https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00001-of-00002.parquet"
    fi

    echo "RealWorldQA data check completed."
    mkdir -p "$REALWORLDQA_DIR/data/images"  # 创建图片存放目录
}

echo "==============================="
echo "Checking raw data integrity..."

check_raw_files

echo "================================"
echo "Staring VLLM Server..."
vllm serve $CKPT_PATH --port $VLLM_INFERENCE_PORT --host 0.0.0.0 --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE & # 后台运行，避免堵塞


echo "Raw data integrity check completed."

echo "================================"
echo "Evaluating Mathvista..."
echo "=============================="
cd "$BASE_DIR/MathVista" || exit
cd evaluation || exit
echo "Generating ckpt responses for MathVista..."
nohup python local_generate_response.py \
--inference_api "http://localhost:$VLLM_INFERENCE_PORT/v1" \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--model_path $CKPT_PATH \
--output_dir ../results/$INFERENCE_RUN_ID \
--output_file output_$INFERENCE_RUN_ID.json > ../data/inference_$INFERENCE_RUN_ID.log 2>&1 # 在这里会阻塞

echo "Genearting ckpt responses for MathVista completed."
echo "Extracting answers for MathVista..."
python extract_answer_w_gpt4o.py \
--results_file_path ../results/$INFERENCE_RUN_ID/output_$INFERENCE_RUN_ID.json > ../data/extract_answer_w_gpt4o_$INFERENCE_RUN_ID.log 2>&1 # 在这里会阻塞
echo "Extracting answers for MathVista completed."
echo "Calculating scores for MathVista..."
python calculate_score.py \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--output_dir ../results/$INFERENCE_RUN_ID \
--output_file output_$INFERENCE_RUN_ID.json \
--score_file scores_$INFERENCE_RUN_ID.json
echo "Calculating scores for MathVista completed. See scores in ../results/$INFERENCE_RUN_ID/scores_$INFERENCE_RUN_ID.json"
echo "Mathvista evaluation completed."

echo "================================"
echo "Evaluating MathVision..."
echo "================================"
cd "$BASE_DIR/MathVision" || exit
echo "Data preprocessing for MathVision..."

#TODO: 这里可以选择跑testmini还是test
python parquet_to_json.py --input_file "data/test-00000-of-00001-3532b8d3f1b4047a.parquet" --output_file "data/MathVision_test.json"
python parquet_to_json.py --input_file "data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" --output_file "data/MathVision_testmini.json"
echo "Generating Responses for MathVision..."

mkdir -p data
nohup python inference.py \
    --model_name_or_path $CKPT_PATH \
    --input_file data/MathVision_$MATHVISION_SUBSET.json \
    --save_name data/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID.jsonl \
    --tp 1 \
    --bz 1 \
    --max_new_tokens 8000 >> data/inference_test_$INFERENCE_RUN_ID.log # 在这里会阻塞

echo "Generating Responses for MathVision completed."
cd "$BASE_DIR/MathVision" || exit
mkdir -p outputs
cp data/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID.jsonl outputs/MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID.jsonl
python evaluation/evaluate.py --eval_file MathVision-${MATHVISION_SUBSET}_inferenced_$INFERENCE_RUN_ID.jsonl
echo "Calculating scores for MathVision completed. See scores in outputs/evaluation_results.json"
echo "MathVision evaluation completed."

echo "=============================="
echo "Evaluating MME-RealWorld-Lite..."
echo "=============================="
cd "$BASE_DIR/MME-RealWorld-Lite" || exit

echo "Data preprocessing for MME-RealWorld-Lite..."
python unify_format_lite.py \
--input_file data/MME-RealWorld-Lite.json \
--output_file data/MME-RealWorld-Lite_unified.json \
--image_base_dir $IMAGE_BASE_DIR_MME_REALWORLD_LITE > data/unifyfmt.log 2>&1 # 在这里会阻塞
echo "Inferencing for MME-RealWorld-Lite..."
python inference.py \
--model_name_or_path $CKPT_PATH \
--input_file data/MME-RealWorld-Lite_unified.json \
--save_name data/MME-RealWorld-Lite_inferenced_$INFERENCE_RUN_ID.jsonl \
--tp 1 \
--bz 1 \
--max_new_tokens 8000 > data/inference_$INFERENCE_RUN_ID.log 2>&1 # 在这里会阻塞
echo "Genearting Responses for MME-RealWorld-Lite completed."
echo "Calculating scores for MME-RealWorld-Lite..."
python judge.py \
    --input_file data/MME-RealWorld-Lite_inferenced_$INFERENCE_RUN_ID.jsonl \
    --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
    --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
    --output_file data/MME-RealWorld-Lite_judged_$INFERENCE_RUN_ID.jsonl > data/judge_$INFERENCE_RUN_ID.log 2>&1 &
echo "Calculating scores for MME-RealWorld-Lite completed. See results in the printed output file"
echo "MME-RealWorld-Lite evaluation completed."

echo "=============================="
echo "Evaluating RealWorldQA..."
echo "=============================="
cd "$BASE_DIR/realworldqa" || exit
echo "Data preprocessing for RealWorldQA..."
INPUT_FILES="data/test-00000-of-00002.parquet data/test-00001-of-00002.parquet"
OUTPUT_FILE="data/RealWorldQA.json"
SAMPLE_RATIO=1.0
python parquet_to_json.py --input_files "${INPUT_FILES}" --output_file "$OUTPUT_FILE" --sample_ratio "$SAMPLE_RATIO"

echo "Generating Responses for RealWorldQA..."
python inference.py \
--model_name_or_path $CKPT_PATH \
--input_file data/RealWorldQA.json \
--save_name data/RealWorldQA_inferenced_$INFERENCE_RUN_ID.jsonl \
--tp 1 \
--bz 1 \
--max_new_tokens 8000 2>&1 | tee data/inference_$INFERENCE_RUN_ID.log
echo "Generating Responses for RealWorldQA completed."
echo "Calculating scores for RealWorldQA..."
python judge.py \
    --input_file data/RealWorldQA_inferenced_$INFERENCE_RUN_ID.jsonl \
    --judge_api $CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT \
    --api_key $CUSTOMIZED_REMOTE_OPENAI_API_KEY \
    --output_file data/RealWorldQA_judged_$INFERENCE_RUN_ID.jsonl > data/judge_$INFERENCE_RUN_ID.log 2>&1

echo "RealWorldQA evaluation completed."
echo "=============================="

echo "Shutting down VLLM server..."
pkill -f "vllm serve"
echo "All evaluations completed."
