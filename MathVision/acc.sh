export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

python calc_acc.py > acc.log 2>&1 &