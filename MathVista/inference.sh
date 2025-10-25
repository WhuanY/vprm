export CUDA_VISIBLE_DEVICES="6,7"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
use_cot=0

if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    after_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    after_prompt=""
    cot_suffix=""
fi


nohup python inference.py \
    --model_name_or_path /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct \
    --input_file data/MathVista_testmini.json \
    --save_name data/MathVista_inferenced_qwen25vl3b-inst$cot_suffix.jsonl \
    --after_prompt "$after_prompt" \
    --tp 2 \
    --bz 1 \
    --use_cot $use_cot \
    --max_new_tokens 8000 2>&1 | tee data/inference_qwen25vl3b-inst$cot_suffix.log
