"""In MathVista, the prompt can be directly get from 'query' field."""

import io
from PIL import Image
import matplotlib.pyplot as plt


path = "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# read parquet file
import pandas as pd
df = pd.read_parquet(path)
print(df.columns)
filtered_df = df


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
    example record:
    {'pid': '20',
 'question': 'Is the sum of smallest two bar is greater then the largest bar?',
 'image': 'images/20.jpg',
 'decoded_image': {'bytes': b'...'
  'path': '20.jpg'},
 'choices': array(['Yes', 'No'], dtype=object),
 'unit': None,
 'precision': nan,
 'answer': 'No',
 'question_type': 'multi_choice',
 'answer_type': 'text',
 'metadata': {'category': 'general-vqa',
  'context': 'bar chart',
  'grade': 'daily life',
  'img_height': 600,
  'img_width': 850,
  'language': 'english',
  'skills': array(['statistical reasoning'], dtype=object),
  'source': 'ChartQA',
  'split': 'testmini',
  'task': 'figure question answering'},
  'query': 'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: Is the sum of smallest two bar is greater then the largest bar?\nChoices:\n(A) Yes\n(B) No'}
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

if __name__ == "__main__":
    # Test the function with the first record
    df = filtered_df

    # multiprocessing for all records
    processed_lsts = [row.to_dict() for i, row in df.iterrows()]

    print(f"Total {len(processed_lsts)} records to process.")

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.map(single_record, processed_lsts), total=len(processed_lsts)))

    # Save results to jsonl file
    import json
    print('------- Saving results to JSON file -------')
    output_json = "MathVista_testmini.json"
    # save results to json file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # with open(output_jsonl, 'w', encoding='utf-8') as f:
    #     for item in results:
    #         f.write(json.dumps(item) + '\n')
    
