export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
MODEL_PATH="/home/minyingqian/"

python inference.py \
    --model_name_or_path /home/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file MathVision_testmini.json \
    --save_name MathVision_inferenced.jsonl \
    --tp 4 \
    --bz 20 \
    --max_new_tokens 8000 > inference.log 2>&1 &
