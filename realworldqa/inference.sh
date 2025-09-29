export CUDA_VISIBLE_DEVICES="5,6,7,8"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


nohup python inference.py \
--inference_api "http://localhost:9753/v1" \
--model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--input_file data/RealWorldQA.json \
--save_name data/RealWorldQA_inferenced_qwen25vl3b_api0.jsonl \
--tp 4 \
--bz 1 \
--max_new_tokens 8000 2>&1 | tee data/inference_qwen25vl3b_load0.log
