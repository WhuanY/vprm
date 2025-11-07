#!/bin/bash
set -e
cd $(dirname $0)

pwd=$(pwd)
echo "Current directory: $pwd"

hf_endpoint="https://hf-mirror.com"
# hf_endpoint="https://huggingface.co"

merge_extract_clean() {
    local first_part="$1"
    local base_name="${first_part%.tar.gz.part_aa}"

    echo "Merging and extracting ${base_name}..."
    cat "${base_name}".tar.gz.part_* > "${base_name}.tar.gz"
    tar -xzf "${base_name}.tar.gz"
    rm -f "${base_name}".tar.gz.part_*
    rm -f "${base_name}.tar.gz"
}

extract_single() {
    local archive="$1"
    echo "Extracting ${archive}..."
    tar -xzf "${archive}"
    rm -f "${archive}"
}

# Autonomous Driving (single archive)
echo "Autonomous Driving"
wget -nc $hf_endpoint/datasets/yifanzhang114/MME-RealWorld/resolve/main/AutonomousDriving.tar.gz
extract_single "AutonomousDriving.tar.gz"

# Diagram and Table (split archive with only part_aa provided)
echo "Diagram and Table"
wget -nc $hf_endpoint/datasets/yifanzhang114/MME-RealWorld/resolve/main/diagram_and_table.tar.gz.part_aa
merge_extract_clean "diagram_and_table.tar.gz.part_aa"

# Monitoring Images (single split currently one part)
echo "Monitoring Images"
wget -nc $hf_endpoint/datasets/yifanzhang114/MME-RealWorld/resolve/main/monitoring_images.tar.gz.part_aa
merge_extract_clean "monitoring_images.tar.gz.part_aa"

# OCR CC (multiple parts)
echo "OCR CC"
for part in aa ab ac ad ae; do
    wget -nc $hf_endpoint/datasets/yifanzhang114/MME-RealWorld/resolve/main/ocr_cc.tar.gz.part_$part
done
merge_extract_clean "ocr_cc.tar.gz.part_aa"

# Remote Sensing (multiple parts)
echo "Remote Sensing"
for part in aa ab ac ad ae af ag ah ai aj ak al; do
    wget -nc $hf_endpoint/datasets/yifanzhang114/MME-RealWorld/resolve/main/remote_sensing.tar.gz.part_$part
done
merge_extract_clean "remote_sensing.tar.gz.part_aa"

echo "All downloads and extractions completed."