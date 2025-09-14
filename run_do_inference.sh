GPU_LIST="0 1"

python do_inference.py \
    --gpus $GPU_LIST \
    --mem-gb 23.12 \
    --utilization 80