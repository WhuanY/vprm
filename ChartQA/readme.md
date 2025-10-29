## General Guide for Evaluating ChartQA with Qwen2.5-VL

### Step 1: Download raw files
Follow common practices, we evaluate the test set of ChartQA
```sh
mkdir -p data
cd data
wget https://huggingface.co/datasets/AI4Math/ChartQA/resolve/main/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet
```

### Step 2: Data Preparation
You can run `raw_to_json.sh`, selecting the data split(default to `test`)
```sh
split="${1:-test}"
python parquet_to_json.py --split "$split" --output_file data/chartQA_$split.json
```

### Step 3: Inference
```sh
bash inference.sh
```

### Step 4: Judge
```sh
bash judge.sh
```

## Result
Qwen2.5-VL-3B_cot: 75.23

Qwen2.5-VL-7B_cot: 82.71 
