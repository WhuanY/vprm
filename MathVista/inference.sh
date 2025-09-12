export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True




python inference.py \
    --input_file MathVista_testmini.json \
    --save_name MathVista_inferenced.jsonl \
    --tp 4 \
    --bz 20 \
    --max_new_tokens 8000 > inference_1.log 2>&1 &
