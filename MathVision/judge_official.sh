mkdir -p outputs
# 把接下来要评估的推理文件拷贝到 output 目录下（注意替换）
cp data/MathVision-test_inferenced_qwen25vl3b-inst.jsonl outputs/MathVision-test_inferenced_qwen25vl3b-inst.jsonl
python evaluation/evaluate.py
