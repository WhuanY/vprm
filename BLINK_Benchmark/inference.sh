cd eval
export INFERENCE_ENDPOINT="http://localhost:9753/v1" # Make sure this var is set before running inference
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # the inference script is running while evaluation. Please make sure the endpoint and api key are correct.


task_name="all"
use_cot=1 

if [ $use_cot -eq 1 ]; then
    echo "Using CoT inference"
    after_prompt="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags."
    cot_suffix="_cot"
else
    echo "Not using CoT inference"
    after_prompt=""
    cot_suffix=""
fi

mkdir -p ../logs
CKPT_PATH="/mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct"
python test_benchmark.py \
    --model_name_or_path $CKPT_PATH \
    --inference_api $INFERENCE_ENDPOINT \
    --dataset_local_path ../data \
    --after_prompt "$after_prompt" \
    --task_name $task_name 2>&1 | tee ../logs/eval_blink_qwen25vl3b_$task_name$cot_suffix.log