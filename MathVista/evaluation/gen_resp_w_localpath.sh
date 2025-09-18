export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=2



python local_generate_response.py \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--inference_api http://localhost:9753/v1 \
--model_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--output_dir ../results/qwen25vl3b \
--output_file output_qwen25vl3b.json