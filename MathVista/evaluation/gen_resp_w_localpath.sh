export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=2

nohup python local_generate_response.py \
--inference_api "http://localhost:9753/v1" \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--model_path /mnt/minyingqian/models/Qwen2-VL-7B-Instruct \
--output_dir ../results/qwen2vl7b_api0 \
--output_file output_qwen2vl7b_api0.json > ../data/inference_qwen2vl7b_api0.log 2>&1 &