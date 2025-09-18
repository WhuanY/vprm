## Evaluation Pipeline
To execute the evaluation scripts in the paper, ensure your `data` folder has the following structure:

```
├── query.json
├── test.json
├── testmini.json
├── images
    ├── 1.jpg
    ├── 2.jpg
    └── ...
└── texts
    ├── captions_bard.json
    └── ocrs_easyocr.json
```
the original data can be downloaded from https://github.com/lupantech/MathVista/tree/main/data. The image should be downloaded from https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/images.zip?download=true
### Step 1 Inference
Generate the response on the **testmini** subset:

```sh
# step1--start local inference engine
export CUDA_VISIBLE_DEVICES="6,7,8,9"

vllm serve /path/to/your/local/checkpoint-version-xxx --port xxxx --host 0.0.0.0 --tensor-parallel-size 4 

# step2--evaluate
cd evaluation

# bash gen_resp_w_localpath.sh
# or:
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_TRITON_FLASH_ATTN=True
export VLLM_TENSOR_PARALLEL_SIZE=2

python local_generate_response.py \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--inference_api http://localhost:[port_num]/v1 \
--model_path /path/to/your/local/checkpoint-version-xxx \
--output_dir ../results/local_ckpt-version-xxx \
--output_file output_ckpt_version-xxx.json
```
where the `--data_file_path` should be the original parquet file from official repo for evaluation.

### Step 2 Extract Answer
Extract the short answer text for score calculation on the **testmini** subset:

```sh
# bash extract_answer_w_gpt4o.sh
# or:
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://aigc.x-see.cn/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="sk-xxxxxx"

python extract_answer_w_gpt4o.py \
--results_file_path ../results/local_ckpt-version-xxx
```

### Step 3 Calculate the score
```sh
# bash calculate_score.sh
# or
python calculate_score.py \
--data_file_path /home/minyingqian/vprm/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
--output_dir ../results/local_ckpt-version-xxx \
--output_file output_local_ckpt-version-xxx.json \
--score_file scores_local_ckpt-version-xxx.json
```


