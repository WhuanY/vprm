cd MathVision # 确保在 MathVision 目录下
mkdir -p outputs
# 把接下来要评估的推理文件拷贝到 output 目录下
cp data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl outputs/MathVision-test_inferenced_qwen25vl3b-inst.jsonl
python evaluation/evaluate.py --eval_file MathVision-test_inferenced_qwen25vl3b-inst.jsonl
