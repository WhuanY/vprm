"""ChartQA parquet to JSON conversion script."""

import io
import os
import json
import argparse
from PIL import Image
import numpy as np
import pandas as pd
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
        

def single_record(split: str, idx: int, record: dict ):
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
        "human_or_machine": record.get('human_or_machine', -1)  # 0 for human, 1 for machine, -1 if unknown
    }
    
    # Manually assign ID using index
    converted_template['id'] = str(idx)
    # ground truth
    gt = record['label']
    assert len(gt) == 1
    gt = gt[0]
    
    # question
    question = record['query']
    
    is_multi_choice = False

    if len(gt.strip().upper()) == 1 and gt.strip().upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        is_multi_choice = True

    if is_multi_choice:
        print("is_multiple_choice!!")
        converted_template['problem_w_choices'] = question
        converted_template['answer_w_choices'] = gt
    else:
        converted_template['problem'] = question
        converted_template['answer'] = gt
    
    
    # Handle image
    if 'image' in record and record['image']:
        if isinstance(record['image'], dict) and 'bytes' in record['image']:
            # Save image from bytes
            image_filename = f"chartqa_{split}_{converted_template['id']}.jpg"
            image_path = save_image_from_bytes(record['image']['bytes'], image_filename)
            if image_path:
                converted_template['image'] = [image_path]
        elif isinstance(record['image'], str):
            # Image path already provided
            converted_template['image'] = [record['image']]
    
    return converted_template


def main():
    parser = argparse.ArgumentParser(description='Convert RealWorldQA parquet file to JSON format')
    parser.add_argument('--input_files', type=str, required=False, default="",
                       help='Space-separated list of input parquet files. If omitted, will be inferred from --split.')
    parser.add_argument('--output_file', type=str, required=False, default="",
                       help='Path to output JSON file. If omitted, defaults to data/chartQA_<split>.json')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test',
                       help='Dataset split to use when --input_files is not provided')
    parser.add_argument('--split_human_machine', action='store_true',
                       help='Whether to split human and machine questions separately')

    args = parser.parse_args()


    # Determine input files
    if args.input_files.strip():
        input_files = args.input_files.split()
    else:
        if args.split == 'test':
            input_files = ["data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"]
        elif args.split == 'val':
            input_files = ["data/val-00000-of-00001-0f11003c77497969.parquet"]
        else:
            print(f"Error: Unknown split {args.split}")
            return

    # Determine output file
    output_file = args.output_file if args.output_file.strip() else f"data/chartQA_{args.split}.json"

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
    
    df = pd.read_parquet(file_path)


    
    results = []
    for idx, row in tqdm(df.iterrows()):
        record = row.to_dict()
        result = single_record(args.split, idx, record)
        results.append(result)

    
    # write
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    if args.split_human_machine:
        res_human = []
        res_machine = []
        for item in results:
            item['label'] = item.pop('answer') if item['answer'] else item.pop('answer_w_choices')
            item['imgname'] = item['image'][0].split('/')[-1]
            item['query'] = item.pop('problem') if item['problem'] else item.pop('problem_w_choices')
            if item['human_or_machine'] == 0:
                res_human.append(item)
            elif item['human_or_machine'] == 1:
                res_machine.append(item)
            else:
                raise ValueError(f"Unknown human_or_machine value {item['human_or_machine']}")
        with open(output_file.replace('.json', '_human.json'), 'w', encoding='utf-8') as f:
            json.dump(res_human, f, ensure_ascii=False, indent=4)
        with open(output_file.replace('.json', '_machine.json'), 'w', encoding='utf-8') as f:
            json.dump(res_machine, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()