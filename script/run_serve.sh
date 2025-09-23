# step1--start local inference engine
export CUDA_VISIBLE_DEVICES="6,7,8,9"

vllm serve /path/to/your/local/ckpt --port 9753 --host 0.0.0.0 --tensor-parallel-size 4 