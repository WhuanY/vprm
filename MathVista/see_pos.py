"""
Usage: python see_pos.py # 记得更改jsonl_file和output_file变量
"""


import json
import re

def extract_answer_from_response(response_list):
    """从response列表中提取<answer></answer>标签内的内容"""
    if not response_list:
        return "No response"
    
    # 取第一个response
    response_text = response_list[0] if isinstance(response_list, list) else str(response_list)
    
    # 使用正则表达式提取<answer></answer>之间的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    
    if answer_match:
        return answer_match.group(1).strip()
    else:
        return "No <answer> tag found"

def analyze_judgment_1(jsonl_file):
    """分析judgement为1的样本"""
    judgment_1_samples = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                if data.get('judgment', 1) == 1:  # 只看judgment为0的
                    standard_answer = data.get('standard_answer', 'N/A')
                    response = data.get('response', [])
                    extracted_answer = extract_answer_from_response(response)
                    
                    judgment_1_samples.append({
                        'line_num': line_num,
                        'id': data.get('id', 'N/A'),
                        'standard_answer': standard_answer,
                        'extracted_answer': extracted_answer,
                        'question': data.get('question', 'N/A')[:100] + '...' if len(data.get('question', '')) > 100 else data.get('question', 'N/A')
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return judgment_1_samples

def print_analysis(samples):
    """打印分析结果"""
    print(f"Found {len(samples)} samples with judgment = 1\n")
    print("="*100)
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i} (Line {sample['line_num']}, ID: {sample['id']})")
        print(f"Question: {sample['question']}")
        print(f"Standard Answer: '{sample['standard_answer']}'")
        print(f"Model Answer: '{sample['extracted_answer']}'")
        print("-"*100)
        
        if i % 10 == 0:  # 每10个样本暂停一下
            input("Press Enter to continue...")

def save_to_csv(samples, output_file):
    """保存结果到CSV文件"""
    import csv
    
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['line_num', 'id', 'standard_answer', 'extracted_answer', 'question'])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    jsonl_file = "data/MathVista_judge_results_qwen25vl3b-inst.jsonl"  # 替换为你的文件名
    output_file = "data/judgement_1_analysis_qwen25vl3b-inst.csv"
    
    # 分析judgement为1的样本
    samples = analyze_judgment_1(jsonl_file)
    
    # 打印结果
    print_analysis(samples)
    
    # 保存到CSV文件
    save_to_csv(samples, output_file=output_file)
    
    # 简单统计
    print(f"\nSummary:")
    print(f"Total judgment=1 samples: {len(samples)}")
    
    # 统计一些常见的答案模式
    standard_answers = [s['standard_answer'] for s in samples]
    extracted_answers = [s['extracted_answer'] for s in samples]
    
    print(f"Unique standard answers: {len(set(standard_answers))}")
    print(f"Unique extracted answers: {len(set(extracted_answers))}")
    
