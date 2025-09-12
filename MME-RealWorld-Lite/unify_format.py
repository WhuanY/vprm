"""
Target Format:
{
        "id": "725",
        "problem": "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: In Figure, suppose that Barbara's velocity relative to Alex is a constant $v_{B A}=52 \\mathrm{~km} / \\mathrm{h}$ and car $P$ is moving in the negative direction of the $x$ axis.\r\n(a) If Alex measures a constant $v_{P A}=-78 \\mathrm{~km} / \\mathrm{h}$ for car $P$, what velocity $v_{P B}$ will Barbara measure?",
        "problem_w_choices": "",
        "answer": "-130",
        "answer_w_choices": "",
        "image": [
            "images/725.jpg"
        ]
},
Original Format:
{
        "Question_id": "perception/ocr_cc/book_map_poster/0913",
        "Image": "TRAIN_BICUBIC_HQ_50K_14456.png",
        "Text": "What dish does the Chinese chart correspond to in this picture?",
        "Question Type": "Multiple Choice",
        "Answer choices": [
            "(A) Nutty Sate Ribs",
            "(B) Sweet ＆ Sour Haw Flakes",
            "(C) Chili Grab Ribs",
            "(D) Inasal Baboy Ribs",
            "(E) The image does not feature the content."
        ],
        "Ground truth": "B",
        "Category": "book_map_poster",
        "Subtask": "OCR with Complex Context",
        "Task": "Perception",
        "Image_new": [
            "/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs/TRAIN_BICUBIC_HQ_50K_14456.png"
        ]
    }
Note: Question_id -> id
problem -> "" # This benchmark has no standalone problem statement
problem_w_choices -> Text + "\n" + Convert(Answer Choices)
answer -> "" # This benchmark has no standalone problem statement
answer_w_choices -> Ground truth # This benchmark has no standalone problem statement
Image_new -> image
"""

input_file = "MME-RealWorld-Lite_new.json"
output_file = "MME-RealWorld-Lite_unified.json"

import json
from tqdm import tqdm

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in tqdm(data):
        item['id'] = item.pop('Question_id')
        item['problem'] = ""
        options = item.pop('Answer choices', [])
        if options and len(options) > 1: # 如果是多选题
            item['problem_w_choices'] = item.pop('Text') + "\n" + ' '.join(options)
            item['answer_w_choices'] = item.pop('Ground truth')
            item['answer'] = ""
        else: #如果不是多选题
            item['problem'] = item.pop('Text')
            item['answer'] = item.pop('Ground truth')
            item['problem_w_choices'] = ""
            item['answer_w_choices'] = ""
        item['image'] = item.pop('Image_new', [])
        # 删除不需要的字段
        for key in ['Question Type', 'Category', 'Subtask', 'Task', 'Image']:
            if key in item:
                item.pop(key)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved unified data to {output_file}")

