#!/bin/bash
# check_raw_files.sh - Check and download required benchmark data files

# Source configuration
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

check_mathvision() {
    local MATHVISION_DIR="$BASE_DIR/MathVision"
    
    # Create data directory
    mkdir -p "$MATHVISION_DIR/data"
    
    # Check/download testmini data
    if [ ! -f "$MATHVISION_DIR/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" ]; then
        echo "MathVision testmini data not found. Please Download it via:"
        echo "wget -O $MATHVISION_DIR/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet"
        return 1
    fi

    # Check/download test data
    if [ ! -f "$MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet" ]; then
        echo "MathVision test data not found. Please Download it via:"
        echo "wget -O $MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet"
        return 1
    fi

    # Check/download evaluation script data
    if [ ! -f "$MATHVISION_DIR/data/test.jsonl" ]; then
        echo "MathVision evaluation script data dependency not found. Please Download it via:"
        echo "wget -O $MATHVISION_DIR/data/test.jsonl https://github.com/mathllm/MathVision/raw/main/data/test.jsonl"
        return 1
    fi

    # Check image count
    if [ $(find "$MATHVISION_DIR/images" -type f 2>/dev/null | wc -l) -ne 3040 ]; then
        echo "Warning: MathVision images count is not 3040."
        echo "Run: wget -O $MATHVISION_DIR/images.zip https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/images.zip then unzip it."
        return 1
    fi

    echo "MathVision data check completed."
}

check_mathvista() {
    local MATHVISTA_DIR="$BASE_DIR/MathVista"
    
    # Create data directory
    mkdir -p "$MATHVISTA_DIR/data"
    
    # Check/download testmini data
    if [ ! -f "$MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" ]; then
        echo "MathVista testmini data not found, please download it via:"
        echo "wget -O $MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
        return 1
    fi

    # Check image count
    if [ $(find "$MATHVISTA_DIR/images" -type f 2>/dev/null | wc -l) -ne 6141 ]; then
        echo "Warning: MathVista images count is not 6141."
        echo "Run: wget -O $MATHVISTA_DIR/images.zip https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/images.zip then unzip it."
        return 1
    fi

    echo "MathVista data check completed."
}

check_mme_realworld_lite() {
    local MME_REALWORLD_LITE_DIR="$BASE_DIR/MME-RealWorld-Lite"
    
    # Create data directory
    mkdir -p "$MME_REALWORLD_LITE_DIR/data"
    
    # Check JSON data
    if [ ! -f "$MME_REALWORLD_LITE_DIR/data/MME-RealWorld-Lite.json" ]; then
        echo "Warning: MME-RealWorld-Lite JSON data not found."
        echo "Run: wget -O <IMAGE_BASE_DIR_MME_REALWORLD_LITE> https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite/resolve/main/data.zip then unzip it. MME-RealWorld-Lite.json is inside the zip file."
        return 1
    fi

    # Check image count
    if [ $(find "$IMAGE_BASE_DIR_MME_REALWORLD_LITE" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) 2>/dev/null | wc -l) -lt 1543 ]; then
        echo "Warning: MME-RealWorld-Lite images count is less than 1543."
        echo "Run: wget -O <IMAGE_BASE_DIR_MME_REALWORLD_LITE> https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite/resolve/main/data.zip then unzip it."
        return 1
    fi

    echo "MME-RealWorld-Lite data check completed."
}

check_realworldqa() {
    local REALWORLDQA_DIR="$BASE_DIR/realworldqa"
    
    # Create data and images directories
    mkdir -p "$REALWORLDQA_DIR/data/images"
    
    # Check/download test data part 1
    if [ ! -f "$REALWORLDQA_DIR/data/test-00000-of-00002.parquet" ]; then
        echo "RealWorldQA test-00000-of-00002 data not found. Please download it via:"
        echo "wget -O $REALWORLDQA_DIR/data/test-00000-of-00002.parquet https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00000-of-00002.parquet"
        return 1
    fi

    # Check/download test data part 2
    if [ ! -f "$REALWORLDQA_DIR/data/test-00001-of-00002.parquet" ]; then
        echo "RealWorldQA test-00001-of-00002 data not found. Please download it via:"
        echo "wget -O $REALWORLDQA_DIR/data/test-00001-of-00002.parquet https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00001-of-00002.parquet"
        return 1
    fi

    echo "RealWorldQA data check completed."
}

check_blink() {
    local BLINK_DIR="$BASE_DIR/BLINK_Benchmark"
    
    # 创建数据目录
    mkdir -p "$BLINK_DIR/data"
    
    # 定义需要检查的文件夹列表
    FOLDERS=(
        "Art_Style"
        "Counting"
        "Forensic_Detection"
        "Functional_Correspondence"
        "IQ_Test"
        "Jigsaw"
        "Multi-view_Reasoning"
        "Object_Localization"
        "Relative_Depth"
        "Relative_Reflectance"
        "Semantic_Correspondence"
        "Spatial_Relation"
        "Visual_Correspondence"
        "Visual_Similarity"
    )

    # 循环检查文件夹是否存在
    for FOLDER in "${FOLDERS[@]}"; do
        if [ ! -d "$BLINK_DIR/data/$FOLDER" ]; then
            echo "❌ Directory missing: $BLINK_DIR/data/$FOLDER"
            echo "⚠️  Warn: Raw data for BLINK benchmark is missing."
            echo "   Please download all files from: https://huggingface.co/datasets/BLINK-Benchmark/BLINK/tree/main"
            echo "   and place them under: $BLINK_DIR/data"
            return 1
        fi
    done

    echo "BLINK data directory check completed."
}

check_MMStar() {
    local MMSTAR_DIR="$BASE_DIR/MMStar"
    
    # Create data directory
    mkdir -p "$MMSTAR_DIR/data"
    
    # Check/download test data
    if [ ! -f "$MMSTAR_DIR/data/mmstar.parquet" ]; then
        echo "MMStar data not found. Please download it via:"
        echo "wget -O $MMSTAR_DIR/data/mmstar.parquet https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/main/mmstar.parquet"
        return 1
    fi

    echo "MMStar data check completed."
}

check_chartqa() {
    local CHARTQA_DIR="$BASE_DIR/ChartQA"

    mkdir -p $CHARTQA_DIR/data

    if [ ! -f "$CHARTQA_DIR/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"]; then 
        echo "ChartQA test data not found. Please download it via:"
        echo "wget -O $CHARTQA_DIR/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet https://huggingface.co/datasets/AI4Math/ChartQA/resolve/main/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
        return 1
    fi  

    echo "ChartQA data check completed."
}

# Main function to check all raw files
check_all_raw_files() {
    echo "==============================="
    echo "Checking raw data integrity..."
    
    check_mathvision || return 1
    check_mathvista || return 1
    check_mme_realworld_lite || return 1
    check_realworldqa || return 1
    check_blink || return 1
    check_MMstar || return 1
    check_chartqa || return 1
    
    echo "Raw data integrity check completed."
    echo "==============================="
}

# Run the check if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_all_raw_files
fi