"""RealWorldQA parquet to JSON conversion script."""

import io
import os
import json
import argparse
from PIL import Image
import numpy as np
import random
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def save_image_from_bytes(image_bytes, filename):
    """Save image bytes to file - direct binary write approach"""
    try:
        # 检查是否为WEBP格式并调整文件名
        if image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            #print(f"Detected WEBP format for {filename}")
            filename = filename.replace('.jpg', '.webp')
        elif image_bytes.startswith(b'\xFF\xD8\xFF'):
            # JPEG格式
            filename = filename.replace('.webp', '.jpg')
        elif image_bytes.startswith(b'\x89PNG'):
            # PNG格式
            filename = filename.replace('.jpg', '.png').replace('.webp', '.png')
        
        # 创建目录
        image_path = f"images/{filename}"
        os.makedirs("images", exist_ok=True)
        
        # 直接写入字节数据
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        #print(f"Successfully saved {filename} ({len(image_bytes)} bytes)")
        return image_path
        
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return None
        
        return None

def single_record(record: dict, index: int):
    """
    Convert a single record to the desired template format.
    realworldqa format example: 
    question:In which direction is the front wheel of the c...
    answer:                                                     C
    image:       {'bytes': b'RIFF^\xa9\x0c\x00WEBPVP8LQ\xa9\x0c...
    """
    converted_template = {
        "id": "",
        "problem": "",
        "problem_w_choices": "",
        "answer": "",
        "answer_w_choices": "",
        "image": [],  # List of image paths
    }
    
    # Manually assign ID using index
    converted_template['id'] = str(index)
    
    is_multi_choice = False

    if len(record['answer'].strip().upper()) == 1 and record['answer'].strip().upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        is_multi_choice = True

    if is_multi_choice:
        converted_template['problem_w_choices'] = record['question']
        converted_template['answer_w_choices'] = record['answer']
    else:
        converted_template['problem'] = record['question']
        converted_template['answer'] = record['answer']
    
    # Handle image
    if 'image' in record and record['image']:
        if isinstance(record['image'], dict) and 'bytes' in record['image']:
            # Save image from bytes
            image_filename = f"realworldqa_{converted_template['id']}.jpg"
            image_path = save_image_from_bytes(record['image']['bytes'], image_filename)
            if image_path:
                converted_template['image'] = [image_path]
        elif isinstance(record['image'], str):
            # Image path already provided
            converted_template['image'] = [record['image']]
    
    return converted_template


def main():
    parser = argparse.ArgumentParser(description='Convert RealWorldQA parquet file to JSON format')
    parser.add_argument('--input_files', type=str, required=True, 
                       help='Space-separated list of input parquet files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output JSON file')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Ratio of data to sample (0.0-1.0, default: 1.0 for all data)')
    
    args = parser.parse_args()
    
    # Validate sample_ratio
    if not 0.0 < args.sample_ratio <= 1.0:
        print(f"Error: sample_ratio must be between 0.0 and 1.0, got {args.sample_ratio}")
        return
    
    # Parse input files
    input_files = args.input_files.split()
    
    # Check if input files exist
    valid_files = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} does not exist! Skipping.")
        else:
            valid_files.append(file_path)
    
    if not valid_files:
        print("Error: No valid input files found!")
        return
    
    # Process each parquet file and combine results
    all_results = []
    total_index = 0
    
    for file_path in valid_files:
        print(f"\nProcessing parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"File size: {len(df)} records")
        
        # Sample data if sample_ratio < 1.0
        if args.sample_ratio < 1.0:
            sample_size = int(len(df) * args.sample_ratio)
            print(f"Sampling {sample_size} records (ratio: {args.sample_ratio})")
            
            # Set random seed for reproducibility
            random.seed(42 + valid_files.index(file_path))  # Different seed for each file
            np.random.seed(42 + valid_files.index(file_path))
            
            # Random sampling
            df = df.sample(n=sample_size, random_state=42 + valid_files.index(file_path)).reset_index(drop=True)
            print(f"Sampled dataset size: {len(df)}")
        
        # Convert dataframe to list of dictionaries
        processed_lsts = [row.to_dict() for i, row in df.iterrows()]
        print(f"Processing {len(processed_lsts)} records...")

        # Process records using single thread with continuous ID assignment
        file_results = []
        for record in tqdm(processed_lsts, desc=f"Processing {os.path.basename(file_path)}"):
            result = single_record(record, total_index)
            file_results.append(result)
            total_index += 1  # Increment global index counter
        
        all_results.extend(file_results)
        print(f"Finished processing {file_path}. Total records so far: {len(all_results)}")
    
    # Save combined results to JSON file
    print(f'\nSaving combined results to JSON file: {args.output_file}')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed! Output saved to {args.output_file}")
    print(f"Final output contains {len(all_results)} records from {len(valid_files)} files")


if __name__ == "__main__":
    main()