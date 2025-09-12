export CUDA_VISIBLE_DEVICES="1,2,3,4"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

python extract_answer.py > extract.txt 2>&1 &