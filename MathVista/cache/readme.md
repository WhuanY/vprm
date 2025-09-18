# MathVista Evaluation

This directory contains the evaluation setup for MathVista benchmark.

## 文件结构

```
MathVista/
├── readme.md                    # 本文档
├── MathVista_testmini.json      # 测试数据集
├── images/                      # 图片文件夹 (需要从官方下载)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── eval_scripts/                # 评测脚本 (如果有)
```

## QuickStart

### Step 1: 准备 images.zip
从 MathVista 官方仓库或对应的 benchmark 源下载图片数据包：
```bash
cd vprm # 根目录
cd MathVista
wget https://github.com/lupantech/MathVista/releases/download/v1.0/images.zip
```

### Step 2: 解压图片文件
```bash
cd MathVista
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
- `MathVista_testmini.json`: 包含测试样本，每个样本包含问题、答案和对应的图片路径。这个json从parquet转换而来
- `images/`: 存放所有测试图片，文件名对应 JSON 中的图片 ID

## 注意事项
1. 图片文件较大，已在 `.gitignore` 中忽略，需要手动下载
2. 确保图片路径与 JSON 文件中的路径对应
3. 评测时请确保模型能够处理多模态输入（文本+图像）
