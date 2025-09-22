import pandas as pd
import json
import argparse
from tqdm import tqdm

def convert_options(options: list) -> str:
    """
    convert options list to string like "A. xxx B. xxx C. xxx D. xxx"
    """
    _ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_str = ' '.join([f"{_i}. {opt}" for _i, opt in zip(_, options)])
    print(option_str)
    return option_str

def single_record(record:dict):
    record.pop("decoded_image", None)  # Use None as default to avoid KeyError if key doesn't exist
    print(record)
    converted_template = {
            "id":"",
            "problem": "",
            "problem_w_choices": "",
            "answer": "",
            "answer_w_choices": "",
            "image": [], # List of image path
        }
    # Map fields
    # 判断是不是多选题
    converted_template['id'] = record['id']
    problem = record['question']
    options = list(record['options'])
    if options and len(options) > 1: # 如果是多选题
        converted_template['problem_w_choices'] = problem + "\n" + convert_options(options)
        
        converted_template['answer_w_choices'] = record['answer']
    else: #如果不是多选题
        converted_template['problem'] = problem
        converted_template['answer'] = record['answer'] 
    # Save image to disk
    converted_template['image'] = [record['image']]
    # converted_template['image'].append(
    #     save_image_from_bytes(record['decoded_image']['bytes'], f"MathVista-{record['pid']}-{record['metadata']['task']}.png")
    # )

    return converted_template

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert Parquet file to JSON for MathVision dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    
    args = parser.parse_args()
    
    # Read parquet file
    print(f"Reading parquet file from: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    
    processed_lsts = [row.to_dict() for i, row in df.iterrows()]
    print(f"Total {len(processed_lsts)} records to process.")
    
    results = []
    for lst in tqdm(processed_lsts, desc="Processing records"):
        results.append(single_record(lst))
    
    # Save results to JSON file
    print(f'------- Saving results to JSON file: {args.output_file} -------')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed. {len(results)} records saved to {args.output_file}")

if __name__ == "__main__":
    main()