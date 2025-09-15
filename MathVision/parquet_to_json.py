import pandas as pd


# read parquet file
path = "data/test-00000-of-00001-3532b8d3f1b4047a.parquet"

df = pd.read_parquet(path)

def convert_options(options: list) -> str:
    """
    convert options list to string like "A. xxx B. xxx C. xxx D. xxx"
    """
    _ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_str = ' '.join([f"{_i}. {opt}" for _i, opt in zip(_, options)])
    print(option_str)
    return option_str

def single_record(record:dict):
    record.pop("decoded_image")
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

processed_lsts = [row.to_dict() for i, row in df.iterrows()]
print(f"Total {len(processed_lsts)} records to process.")
from tqdm import tqdm

results = []
for lst in processed_lsts:
    results.append(single_record(lst))


# Save results to jsonl file
import json
print('------- Saving results to JSON file -------')
output_json = "MathVision_test.json"
# save results to json file
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
# with open(output_jsonl, 'w', encoding='utf-8') as f:
#     for item in results:
#         f.write(json.dumps(item) + '\n')
