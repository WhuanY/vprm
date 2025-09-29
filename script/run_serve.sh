# step1--start local inference engine
export CUDA_VISIBLE_DEVICES="5,6,7,8"

vllm serve /mnt/minyingqian/models/Qwen2-VL-7B-Instruct --port 9753 --host 0.0.0.0 --tensor-parallel-size 4