import re
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_answer_from_response(response):
    """
    从response中提取单个字母答案
    如果strip后只剩单个字母(A-E)，则返回该字母
    如果找不到明确答案，返回None
    """
    has_answer_tag = bool("<answer>" in response.lower() or "</answer>" in response.lower())

    if not has_answer_tag:
        if not response:
            return None
        
        # 清理响应文本
        cleaned_response = response.replace("(", "").replace(")", "").strip()
        
        # 检查是否只有一个字符且是A-E
        if len(cleaned_response) == 1 and cleaned_response.upper() in "ABCDE":
            return cleaned_response.upper()
        
        # 如果没有找到明确答案，则返回None
        return None
    else: 
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if answer_match:
            model_answer = answer_match.group(1).strip()
            if len(model_answer) == 1 and model_answer.upper() in "ABCDE":
                return model_answer.upper()
        return None

# judge_with_gpt4o(response, golden_ans, question, id_field, client)
def judge_with_gpt4o(response, golden_ans, question, id_field, client):
    retries = 3
    for attempt in range(retries):
        try:
            prompt = f"""Judge whether the model's answer to the multiple-choice question is correct.
        question_id {id_field}
        question: {question}
        gloden answer:{golden_ans}
        model response: 
        {response}
        Please judge and response strictly, respond with only 0 or 1, where 1 means the model response aligns with gloden answer, and 0 means it is not aligned."""
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert judge. Respond with only 0 or 1."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0
            )
            
            # 提取结果
            result = completion.choices[0].message.content.strip()
            
            # 标准化结果
            if "1" in result:
                return 1
            else:
                return 0
        except Exception as e:
            print(f"[ERROR] GPT-4o-mini调用失败 (尝试 {attempt + 1}/{retries}): {e}")
            time.sleep(3)  # 等待后重试

    assert False, "GPT-4o-mini调用失败，超过最大重试次数"


def judge_single_line(line:dict, client):
    response = line.get('response', '')[0]  # 取第一个response
    golden_ans = line.get('answer_w_choices', '')
    question = line.get('problem_w_choices', '')
    id_field = line.get('id', '')
    
    # print(f"ID: {id_field}, Method: {method}, Subcategory: {subcategory}")
    
    # 步骤1：尝试提取答案
    gen_answer = extract_answer_from_response(response)
    
    # 判断逻辑
    is_correct = False
    if gen_answer is not None:
        print(f"直接提取答案: {gen_answer}, 标准答案: {golden_ans}")
        is_correct = gen_answer == golden_ans
    else:
        print(f"无法直接提取答案，使用GPT-4o-mini判断...")
        judge_result = judge_with_gpt4o(response, golden_ans, question, id_field, client)
        is_correct = judge_result == 1
        if is_correct:
            print(f"GPT-4o-mini判断结果: 正确. {line=}")

    if is_correct:
        line['judgment'] = 1
    else:
        line['judgment'] = 0
    
    return line


def main(args):
    """
    MMERealWorld-Lite评测流程
    1. 尝试从响应中提取答案
    2. 如果提取成功，直接比较；否则使用GPT-4o-mini判断
    """
    input_file = args.input_file
    id2category = json.load(open("data/mmstar_id2category.json", 'r', encoding='utf-8'))
    client = OpenAI(
        base_url = args.judge_api,
        api_key = args.api_key,
    )
    
    # 总体统计
    total_correct = 0
    total_wrong = 0
    
    category_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
    
    # 一次性加载所有数据
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(json.loads(line))

    res = []
    tasks = []

    max_workers = 30
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有judge_single_line任务
        for line in lines:
            tasks.append(executor.submit(judge_single_line, line, client))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Judging"):
            judged_line = future.result()
            res.append(judged_line)


    # 逐行判断子类别情况
    for res_line in res:
        id_field = res_line.get('id', '')


        
        category = id2category[id_field]['category']

        assert category != '' , f"No category found for id: {id_field}"

        is_correct = res_line.get('judgment', 0) == 1
        
        if is_correct:
            total_correct += 1
            category_stats[category]['correct'] += 1
        else:
            total_wrong += 1
            category_stats[category]['wrong'] += 1

        category_stats[category]['total'] += 1
  

    
    save_file = args.output_file
    # write res to jsonl
    with open(save_file, 'w', encoding='utf-8') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # calculate accuracy     

    # overall acc
    overall_accuracy = total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0.0

    # detailed metrics
    for category in category_stats:
        stats = category_stats[category]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    acc_metrics = {
        'overall_accuracy': str(overall_accuracy) + f"({total_correct}/{total_correct + total_wrong})",
        # 'category_stats': {category: str(stat['accuracy']) + f"{stat['correct']}/{stat['total']}" for category, stat in category_stats.items()}
        'category_stats': {category: f"{stat['accuracy']:.2f} {stat['correct']}/{stat['total']}"
    for category, stat in category_stats.items()}
    }
    
    
    print("\n=== Accuracy Metrics ===")
    print(json.dumps(acc_metrics, indent=4, ensure_ascii=False))
    
    # save detailed metrics to json
    with open(save_file.replace('.jsonl', '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(acc_metrics, f, ensure_ascii=False, indent=4)
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Judge evaluation script for MME-RealWorld-Lite")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/MME-RealWorld-Lite_inferenced_model.jsonl",
        help="Path to the JSONL file containing inference results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/MME-RealWorld-Lite_judge_results.jsonl"
    )
    parser.add_argument(
        "--judge_api",
        type=str,
        default="https://aigc.x-see.cn/v1",
        help="API endpoint for judgment (not used in current implementation)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxxxxx",
        help="API key for the judgment API (not used in current implementation)",
    )
    args = parser.parse_args()
    main(args)