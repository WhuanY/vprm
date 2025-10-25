# step1--start local inference engine
export CUDA_VISIBLE_DEVICES="7"

vllm serve /mnt/minyingqian/models/Qwen2.5-VL-3B-Instruct --port 9751 --host 0.0.0.0 --tensor-parallel-size 1 --gpu_memory_utilization 0.7