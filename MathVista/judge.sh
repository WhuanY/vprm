export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

nohup python judge.py \
    --input_file MathVista_inferenced.jsonl \
    --output_file MathVista_judge_results.jsonl \
    --tp 4 \
    --bz 20 \
    --has_images 1 > judge.log 2>&1 &