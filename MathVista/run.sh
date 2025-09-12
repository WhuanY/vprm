export CUDA_VISIBLE_DEVICES="5,6,7,8"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True

python eval_script_mathvista.py --input_file MathVista_testmini.json --tp 4 --bz 20 --max_new_tokens 8000 > log_mathvista_tp4.txt 2>&1 &
