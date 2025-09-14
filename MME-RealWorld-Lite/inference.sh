export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True


python inference.py \
--input_file MME-RealWorld-Lite_unified.json \
--save_name MME-RealWorld-Lite_inferenced.jsonl \
--tp 4 \
--bz 1 \
--max_new_tokens 8000 2>&1 | tee inference_0.log