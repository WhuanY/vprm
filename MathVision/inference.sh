export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
MODEL_PATH="/home/minyingqian/"

python inference.py \
    --model_name_or_path /home/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVision_testmini.json \
    --save_name data/MathVision_inferenced_qwen25vl3b-inst.jsonl \
    --tp 2 \
    --bz 20 \
    --max_new_tokens 8000 2>&1 | tee data/nference_qwen25vl3b-inst.log 2>&1 &
