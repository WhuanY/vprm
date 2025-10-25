export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=2
use_cot=1
mkdir -p logs

if [ "$use_cot" = "1" ]; then
    echo "use_cot"
    cot_suffix="_cot"
    pre_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
else
    pre_prompt=""
    cot_suffix=""
fi

python local_generate_response.py \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--inference_api http://localhost:9753/v1 \
--model_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
--output_dir ../results/qwen25vl3b$cot_suffix \
--output_file output_qwen25vl3b$cot_suffix.json \
--pre_prompt "$pre_prompt" 2>&1 | tee logs/gen_resp_qwen25vl3b$cot_suffix.log