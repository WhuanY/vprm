# Evalution Steps
## Step 1 Prepare data
```sh
mkdir data
wget https://huggingface.co/datasets/visheratin/realworldqa/resolve/main/data/test-00000-of-00002.parquet
wget 

cd ..
bash raw_to_json.sh
```

## Step 2 Inference
```sh
bash inference.sh
```

## Step 3 Judge
```sh
bash judge.sh
```

# Evaluation Result (Run Locally)
Offical evaluation result for qwen2-vl-7b:   70.1 
Locally run evaluation result for qwen2vl7b: 67.0
