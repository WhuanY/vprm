export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

nohup python judge.py \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-7B-Instruct \
    --input_file data/MathVista_inferenced_qwen25vl3b-inst.jsonl \
    --output_file data/MathVista_judge_results_qwen25vl3b-inst.jsonl \
    --tp 2 \
    --bz 40 \
    --has_images 1 2>&1 | tee data/judge_qwen25vl3b-inst.log 2>&1 &