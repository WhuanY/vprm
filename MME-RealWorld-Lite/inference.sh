export CUDA_VISIBLE_DEVICES="6,7,8,9"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True




# 重复运行4次，防止中间出现OOM等偶然失败，因为inference.py有保存中间结果，所以下一次跑可以不用重复上一次的结果。
python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_0.log 2>&1
echo “Finished 1st run”



python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_1.log 2>&1
echo “Finished 2nd run”



python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_2.log 2>&1
echo “Finished 3rd run”

python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_3.log 2>&1
echo “Finished 4th run”


python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_4.log 2>&1
echo “Finished 5th run”


python inference.py \
    --input_file MME-RealWorld-Lite_unified.json \
    --save_name MME-RealWorld-Lite_inferenced.jsonl \
    --tp 4 \
    --bz 5 \
    --max_new_tokens 8000 > inference_5.log 2>&1
echo “Finished 6th run”