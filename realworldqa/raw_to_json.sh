#!/bin/bash

# RealWorldQA Parquet to JSON Conversion Script with Sampling Support

# 默认参数
INPUT_FILE="data/test-00000-of-00002.parquet"
OUTPUT_FILE="data/RealWorldQA_test-02.json"
SAMPLE_RATIO=1.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --sample_ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--input_file FILE] [--output_file FILE] [--sample_ratio RATIO]"
            echo "  --input_file    Path to input parquet file (default: data/test-00000-of-00002.parquet)"
            echo "  --output_file   Path to output JSON file (default: RealWorldQA_test.json)"
            echo "  --sample_ratio  Ratio of data to sample 0.0-1.0 (default: 1.0)"
            echo ""
            echo "Examples:"
            echo "  $0 --sample_ratio 0.1                    # Sample 10% of default input"
            echo "  $0 --input_file data/test.parquet --sample_ratio 0.5"
            echo "  $0 --input_file data/test.parquet --output_file output.json --sample_ratio 0.2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# 验证sample_ratio范围
if (( $(echo "$SAMPLE_RATIO <= 0.0" | bc -l) )) || (( $(echo "$SAMPLE_RATIO > 1.0" | bc -l) )); then
    echo "Error: sample_ratio must be between 0.0 and 1.0, got $SAMPLE_RATIO"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist!"
    echo "Please make sure the parquet file is in the correct location."
    exit 1
fi

echo "Converting RealWorldQA parquet to JSON..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Sample ratio: $SAMPLE_RATIO"

# 运行转换脚本
python parquet_to_json.py --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" --sample_ratio "$SAMPLE_RATIO"

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