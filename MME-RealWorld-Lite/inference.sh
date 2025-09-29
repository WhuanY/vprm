export CUDA_VISIBLE_DEVICES="5,6"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


nohup python inference.py \
--model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--input_file data/MME-RealWorld-Lite_unified.json \
--save_name data/MME-RealWorld-Lite_inferenced_qwen25vl3b-inst_load0.jsonl \
--tp 2 \
--bz 1 \
--max_new_tokens 8000 > data/inference_qwen25vl3b-inst_load0.log 2>&1 &