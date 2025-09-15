export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

nohup python judge.py \
    --input_file data/MathVision_inferenced_qwen25vl3b-inst.jsonl \
    --output_file data/MathVision_judge_qwen25vl3b-instresults.jsonl \
    --tp 2 \
    --bz 20 \
    --has_images 1 > judge.log 2>&1 &