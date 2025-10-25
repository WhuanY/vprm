#!/bin/bash
INPUT_FILES="data/mmstar.parquet"
OUTPUT_FILE="data/mmstar.json"
SAMPLE_RATIO=1.0


# 运行转换脚本，传入所有文件路径
python parquet_to_json.py --input_files "${INPUT_FILES}" --output_file "$OUTPUT_FILE" --sample_ratio "$SAMPLE_RATIO"
