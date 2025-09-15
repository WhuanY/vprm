# MathVision Evaluation

This directory contains the evaluation setup for MathVision benchmark.

## 文件结构
```
MathVision/
├── README.md                    # 本文档
├── data/                        # 数据目录
│   ├── MathVision_testmini.json          # 测试数据集
│   ├── MathVision_inferenced.jsonl      # 推理结果文件
│   ├── MathVision_judge_results.jsonl   # 评判结果文件
│   ├── MathVision_judge_results_metrics.json # 评测指标
│   ├── test-00000-of-00001-*.parquet     # 原始parquet文件
│   └── testmini-00000-of-00001-*.parquet # 原始parquet文件
├── images/                      # 图片文件夹 (需要从官方下载)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── parquet_to_json.py           # 数据格式转换脚本
├── inference.py                 # 推理脚本
├── inference.sh                 # 推理执行脚本
├── judge.py                     # 评判脚本
├── judge.sh                     # 评判执行脚本
├── judge.log                    # 评判日志
├── see_neg.py                   # 查看负样本脚本
└── see_pos.py                   # 查看正样本脚本
```

## QuickStart

### Step 1: 准备 images.zip
从 MathVision 官方仓库或对应的 benchmark 源下载图片数据包：
```bash
cd vprm # 根目录
cd MathVision
wget ..../images.zip # 从官方提供的路径下载图片包
```

### Step 2: 解压图片文件
```bash
cd MathVision
unzip images.zip
```

解压后应该看到 `images/` 目录包含所有测试图片。

### Step 3: 验证数据完整性
```bash
# 检查图片数量
ls images/ | wc -l
```

### Step 4: 原始数据转换 
⚠️在跑这个文件之前，
```bash
python parquet_to_json.py # 这个脚本将原始数据集统一换成json
```


### Step 5: 推理
```bash 
bash inference.sh 
```
记得修改里面的输入文件和输出文件。


### Step 6: 算分数
```bash 
bash judge.sh
```

## 数据说明
- `MathVision_testmini.json`: 包含测试样本，每个样本包含问题、答案和对应的图片路径。这个json从parquet转换而来
- `images/`: 存放所有测试图片，文件名对应 JSON 中的图片 ID

## 注意事项
1. 图片文件较大，已在 `.gitignore` 中忽略，需要手动下载
2. 确保图片路径与 JSON 文件中的路径对应
3. inference流程首先会去*_inference.jsonl文件里找已经推理过的id并且排除掉。
