cd eval
# Default values
CKPT_PATH="${1:-/mnt/minyingqian/Qwen2.5-VL-3B-Instruct}"
output_save_folder="${2:-outputs_qwen25vl3b}"
prediction_output_dir="${3:-judge_outputs_qwen25vl3b}"

if [ -z "$prediction_output_dir" ]; then
    python evaluate.py \
        --model_name_or_path "$CKPT_PATH" \
        --output_save_folder "$output_save_folder"
else
    python evaluate.py \
        --model_name_or_path "$CKPT_PATH" \
        --output_save_folder "$output_save_folder" \
        --prediction_output_dir "$prediction_output_dir"
fi