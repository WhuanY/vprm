"""In MathVista, the prompt can be directly get from 'query' field."""

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


def answer2choiceLetter(answer: str, options: list) -> str:
    """
    Convert answer to choice letter, e.g., "A", "B", "C", "D"
    eg: 
    options = ['Yes', 'No'], answer = 'No' -> 'B'
    options = [7.5,8,8.5,17], answer = 8.5 -> 'C'
    ...
    """
    assert len(options) > 1
    for idx, opt in enumerate(options):
        if str(opt).strip().lower() == str(answer).strip().lower():
            return chr(ord('A') + idx)
    return answer
    

def single_record(record: dict):
    """
    Convert a single record to the desired template format.
    """
    converted_template = {
        "id":"",
        "problem": "",
        "problem_w_choices": "",
        "answer": "",
        "answer_w_choices": "",
        "image": [], # List of image path
    }
    # Map fields
    converted_template['id'] = record['pid']
    # 判断是不是多选题
    if record['question_type'] != 'multi_choice': # 不是多选题
        converted_template['problem'] = record['query']
        converted_template['answer'] = record['answer']
    else: # 是多选题
        converted_template['problem_w_choices'] = record['query']
        converted_template['answer_w_choices'] = answer2choiceLetter(record['answer'], list(record['choices']))
    # image
    converted_template['image'] = [record['image']]

    return converted_template


def main():
    parser = argparse.ArgumentParser(description='Convert MathVista parquet file to JSON format')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to input parquet file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output JSON file')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Ratio of data to sample (0.0-1.0, default: 1.0 for all data)')
    
    args = parser.parse_args()
    
    # Validate sample_ratio
    if not 0.0 < args.sample_ratio <= 1.0:
        print(f"Error: sample_ratio must be between 0.0 and 1.0, got {args.sample_ratio}")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist!")
        return
    
    print(f"Reading parquet file: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"Original dataset size: {len(df)}")
    
    # Sample data if sample_ratio < 1.0
    if args.sample_ratio < 1.0:
        sample_size = int(len(df) * args.sample_ratio)
        print(f"Sampling {sample_size} records (ratio: {args.sample_ratio})")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Random sampling
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset size: {len(df)}")
    
    # Convert dataframe to list of dictionaries
    processed_lsts = [row.to_dict() for i, row in df.iterrows()]
    print(f"Total {len(processed_lsts)} records to process.")

    # Process records using multiprocessing
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.map(single_record, processed_lsts), total=len(processed_lsts)))

    # Save results to JSON file
    print(f'Saving results to JSON file: {args.output_file}')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed! Output saved to {args.output_file}")
    print(f"Final output contains {len(results)} records")


if __name__ == "__main__":
    main()