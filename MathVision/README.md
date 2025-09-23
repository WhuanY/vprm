# MathVision Evaluation

This directory contains the evaluation setup for MathVision benchmark.


## QuickStart

### Step 1: 数据准备

从 MathVision 官方仓库或对应的 benchmark 源下载图片数据包：
```bash
cd vprm # 根目录
cd MathVision # bench目录

# 从官方提供的路径下载图片包
wget https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/images.zip 
unzip images.zip
ls images/ | wc -l # 检查图片数量，这一步应该输出3040
mkdir -p data
cd data
wget https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet
wget https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet
cd .. # 返回上层目录
```

解压后应该看到 `images/` 目录包含所有测试图片。

### Step 2: 格式转换
参考脚本`parquet_to_json.sh`

### Step 3: 推理
推理脚本支持使用api推理和直接加载VLLM引擎进行推理。如果带`inference_api`参数则是前者，不带则是后者。
请确保`model_name_or_path`是待评估的ckpt路径
```bash 
bash inference.sh 
```

### Step 4: 算分数
算分数的流程最小化复现了官方的整个流程。官方的评测是基于复杂的规则复现的。
```sh
# 首先，把这个链接对应的文件https://github.com/mathllm/MATH-V/blob/main/data/test.jsonl放到data目录下
judge_official.sh # 使用官方的脚本进行评测。在judge_official.sh目录中更改要评估的推理文件。
```

# Evaluation Result (Run Locally)
[本地测试] MathVista 子集 qwen25vl3b 21.9
[官方结果] Official  全集 qwen25vl3b 21.2 

