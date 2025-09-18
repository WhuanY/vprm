export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True



nohup python inference.py \
    --inference_api "http://localhost:9753/v1" \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVista_testmini.json \
    --save_name data/MathVista_inferenced_qwen25vl3b-inst_api.jsonl \
    --tp 2 \
    --bz 5 \
    --max_new_tokens 8000 2>&1 | tee data/inference_qwen25vl3b-inst_api.log
