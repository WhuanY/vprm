#!/bin/bash

# MathVista Parquet to JSON Conversion Script

# 设置输入和输出文件路径
INPUT_FILE="data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
OUTPUT_FILE="MathVista_testmini.json"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist!"
    echo "Please make sure the parquet file is in the correct location."
    exit 1
fi

echo "Converting MathVista parquet to JSON..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# 运行转换脚本
python parquet_to_json.py --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"

# 检查转换是否成功
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    echo "Output file saved as: $OUTPUT_FILE"
    
    # 显示输出文件信息
    if [ -f "$OUTPUT_FILE" ]; then
        echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "Number of records: $(python -c "import json; data=json.load(open('$OUTPUT_FILE')); print(len(data))")"
    fi
else
    echo "Conversion failed!"
    exit 1
fi