# 更改format，并且和官方的prompt构建实现对齐。

SYS = {
        'MME-RealWorld': 'Select the best answer to the above multiple-choice question based on the image. \
            Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:',
        'MME-RealWorld-CN': '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n 最佳答案为：',
    }

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

import json
import os
from tqdm import tqdm
import argparse

def process_data(input_file, output_file, image_base_dir=None):
    """处理数据转换"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    processed_data = []
    for item in tqdm(data, desc="处理数据项"):
        processed_item = {
            "id": item.get('Question_id', ''),
            "problem": "",
            "problem_w_choices": "",
            "answer": "",
            "answer_w_choices": "",
            "image": []
        }
        
        # 处理图片路径
        image_file = item.get('Image', '')
        if image_file:
            if image_base_dir:
                # 构建完整路径
                full_path = os.path.join(image_base_dir, image_file)
                processed_item['image'] = [full_path]
            else:
                # 如果没有指定基础目录，直接使用原始路径
                processed_item['image'] = [image_file]
        
        # 检查是否有已经处理过的图像路径
        if 'Image_new' in item and item['Image_new']:
            processed_item['image'] = item['Image_new']
            
        # 处理文本和选项
        options = item.get('Answer choices', [])
        text = item.get('Text', '')
        
        # 检测是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in item.get('Ground truth', ''))
        subset = 'MME-RealWorld-CN' if has_chinese else 'MME-RealWorld'
        
        # 多选题拼接
        if options and len(options) > 1:
            options_text = '\n'.join(options)
            choice_prompt = '选项如下所示:\n' if has_chinese else 'The choices are listed below:\n'
            
            processed_item['problem_w_choices'] = text + '\n' + choice_prompt + options_text + '\n' + SYS[subset]
            processed_item['answer_w_choices'] = item.get('Ground truth', '')
        else:
            print("[WARNING] Found a non-multiple-choice question, which is unexpected.")
            processed_item['problem'] = text
            processed_item['answer'] = item.get('Ground truth', '')
        
        processed_data.append(processed_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved unified data to {output_file}")
    print(f"Processed {len(processed_data)} items")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Convert JSON file format for MME-RealWorld-Lite.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--image_base_dir', type=str, help="Base directory for image files.")
    args = parser.parse_args()
    
    # 处理数据
    process_data(args.input_file, args.output_file, args.image_base_dir)

if __name__ == "__main__":
    main()