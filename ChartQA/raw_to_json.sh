#!/bin/bash


# split can be passed as first arg; defaults to "test"
split="${1:-test}"

# 运行转换脚本，使用 --split 推断输入文件并生成默认输出路径
python parquet_to_json.py --split "$split" --output_file data/chartQA_$split.json --split_human_machine