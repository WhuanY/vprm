export CUDA_VISIBLE_DEVICES="6,7,8,9"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


nohup python inference.py \
--model_name_or_path /path/to/your/local_ckpt \
--input_file data/MME-RealWorld-Lite_unified.json \
--save_name data/MME-RealWorld-Lite_inferenced_qwen2vl7b-inst_api.jsonl \
--tp 4 \
--bz 1 \
--max_new_tokens 8000 2>&1 | tee data/inference_0.log