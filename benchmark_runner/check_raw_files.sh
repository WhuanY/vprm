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
        echo "MathVision testmini data not found. Downloading..."
        wget -O "$MATHVISION_DIR/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet" \
        "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet"
    fi

    # Check/download test data
    if [ ! -f "$MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet" ]; then
        echo "MathVision test data not found. Downloading..."
        wget -O "$MATHVISION_DIR/data/test-00000-of-00001-3532b8d3f1b4047a.parquet" \
        "https://huggingface.co/datasets/MathLLMs/MathVision/resolve/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet"
    fi

    # Check/download evaluation script data
    if [ ! -f "$MATHVISION_DIR/data/test.jsonl" ]; then
        echo "MathVision evaluation script data dependency not found. Downloading..."
        wget -O "$MATHVISION_DIR/data/test.jsonl" \
        "https://github.com/mathllm/MATH-V/raw/main/data/test.jsonl"
    fi

    # Check image count
    if [ $(find "$MATHVISION_DIR/images" -type f 2>/dev/null | wc -l) -ne 3040 ]; then
        echo "Warning: MathVision images count is not 3040."
    fi

    echo "MathVision data check completed."
}

check_mathvista() {
    local MATHVISTA_DIR="$BASE_DIR/MathVista"
    
    # Create data directory
    mkdir -p "$MATHVISTA_DIR/data"
    
    # Check/download testmini data
    if [ ! -f "$MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" ]; then
        echo "MathVista testmini data not found. Downloading..."
        wget -O "$MATHVISTA_DIR/data/testmini-00000-of-00001-725687bf7a18d64b.parquet" \
        "https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
    fi

    # Check image count
    if [ $(find "$MATHVISTA_DIR/images" -type f 2>/dev/null | wc -l) -ne 6141 ]; then
        echo "Warning: MathVista images count is not 6141."
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
    fi

    # Check image count
    if [ $(find "$MME_REALWORLD_LITE_DIR/images" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) 2>/dev/null | wc -l) -lt 1543 ]; then
        echo "Warning: MME-RealWorld-Lite images count is less than 1543."
    fi

    echo "MME-RealWorld-Lite data check completed."
}

check_realworldqa() {
    local REALWORLDQA_DIR="$BASE_DIR/realworldqa"
    
    # Create data and images directories
    mkdir -p "$REALWORLDQA_DIR/data/images"
    
    # Check/download test data part 1
    if [ ! -f "$REALWORLDQA_DIR/data/test-00000-of-00002.parquet" ]; then
        echo "RealWorldQA test-00000-of-00002 data not found. Downloading..."
        wget -O "$REALWORLDQA_DIR/data/test-00000-of-00002.parquet" \
        "https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00000-of-00002.parquet"
    fi

    # Check/download test data part 2
    if [ ! -f "$REALWORLDQA_DIR/data/test-00001-of-00002.parquet" ]; then
        echo "RealWorldQA test-00001-of-00002 data not found. Downloading..."
        wget -O "$REALWORLDQA_DIR/data/test-00001-of-00002.parquet" \
        "https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00001-of-00002.parquet"
    fi

    echo "RealWorldQA data check completed."
}

# Main function to check all raw files
check_all_raw_files() {
    echo "==============================="
    echo "Checking raw data integrity..."
    
    check_mathvision
    check_mathvista
    check_mme_realworld_lite
    check_realworldqa
    
    echo "Raw data integrity check completed."
    echo "==============================="
}

# Run the check if this script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_all_raw_files
fi