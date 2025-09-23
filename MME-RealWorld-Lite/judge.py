import re
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
import openai
import time

def extract_answer_from_response(response):
    """
    从response中提取单个字母答案
    如果strip后只剩单个字母(A-E)，则返回该字母
    如果找不到明确答案，返回None
    """
    if not response:
        return None
    
    # 清理响应文本
    cleaned_response = response.replace("(", "").replace(")", "").strip()
    
    # 检查是否只有一个字符且是A-E
    if len(cleaned_response) == 1 and cleaned_response.upper() in "ABCDE":
        return cleaned_response.upper()
    
    # 如果没有找到明确答案，则返回None
    return None


def judge_with_gpt4o(response, golden_ans, question, id_field, client):
    """
    使用GPT-4o-mini判断模型答案是否正确
    """
    # 设置OpenAI API密钥
    openai.api_key = args.api_key

    if not openai.api_key:
        print(f"警告: 未找到OpenAI API密钥，无法使用GPT-4o-mini判断问题 {id_field}")
        return 0  # 默认为错误
    
    try:
        # 构建提示
        prompt = f"""Judge whether the model's answer to the multiple-choice question is correct.

question_id {id_field}

question: {question}

gloden answer:{golden_ans}

model response: 
{response}

Please judge and response strictly, respond with only 0 or 1, where 1 means the model response aligns with gloden answer, and 0 means it is not aligned."""
        
        # 调用OpenAI API
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
        print(f"GPT-4o调用错误 ({id_field}): {e}")
        time.sleep(2)  # 遇到错误等待一下再重试
        try:
            # 简化提示再试一次
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "判断答案是否正确。只回复0或1。"},
                    {"role": "user", "content": f"标准答案: {golden_ans}\n模型答案: {response}\n正确输出1，错误输出0:"}
                ],
                max_tokens=5,
                temperature=0
            )
            result = completion.choices[0].message.content.strip()
            if "1" in result:
                return 1
            else:
                return 0
        except:
            print("[WARNING] GPT-4o-mini重试失败，默认为错误")
            return 0  # 重试失败，默认为错误


def extract_subcategory(id_field):
    """
    从id字段提取子类别
    格式: "<method>/<task_split>/<ignore>/<id_number>"
    返回: "<method>/<task_split>"
    """
    parts = id_field.split('/')
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    print("[WARNING] ID格式异常，无法提取子类别:", id_field)
    return "unknown"


def extract_method(id_field):
    """
    从id字段提取方法（domain）
    格式: "<method>/<task_split>/<ignore>/<id_number>"
    返回: "<method>"
    """
    parts = id_field.split('/')
    if len(parts) >= 1:
        return parts[0].lower()  # 转换为小写以保持一致性
    print("[WARNING] ID格式异常，无法提取方法:", id_field)
    return "unknown"


def calculate_weighted_average(subcategory_results, total_questions):
    """
    计算加权平均准确率 (Avg)
    权重 = 每个子类别的题目数量 / 总题目数量
    """
    if total_questions == 0:
        return 0.0
    
    weighted_sum = 0.0
    for subcategory, stats in subcategory_results.items():
        weight = stats['total'] / total_questions
        weighted_sum += stats['accuracy'] * weight
    
    return weighted_sum


def calculate_unweighted_average(subcategory_results):
    """
    计算非加权平均准确率 (Avg-C)
    简单平均所有子类别的准确率
    """
    if not subcategory_results:
        return 0.0
    
    total_accuracy = sum(stats['accuracy'] for stats in subcategory_results.values())
    return total_accuracy / len(subcategory_results)


def main(args):
    """
    MMERealWorld-Lite评测流程
    1. 尝试从响应中提取答案
    2. 如果提取成功，直接比较；否则使用GPT-4o-mini判断
    """
    input_file = args.input_file
    
    # 总体统计
    total_correct = 0
    total_wrong = 0
    
    # 按子类别统计
    subcategory_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
    subcategory_questions = defaultdict(int)
    
    # 按方法（domain）统计
    method_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
    method_subcategories = defaultdict(dict)  # 存储每个方法下的子类别
    
    res = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            response = d.get('response', '')[0]  # 取第一个response
            golden_ans = d.get('answer_w_choices', '')
            question = d.get('problem_w_choices', '')
            id_field = d.get('id', '')
            
            # 提取子类别和方法
            subcategory = extract_subcategory(id_field)
            method = extract_method(id_field)
            
            subcategory_questions[subcategory] += 1
            
            print(f"ID: {id_field}, Method: {method}, Subcategory: {subcategory}")
            
            # 步骤1：尝试提取答案
            gen_answer = extract_answer_from_response(response)
            
            # 判断逻辑
            is_correct = False
            if gen_answer is not None:
                # 直接比较答案
                print(f"直接提取答案: {gen_answer}, 标准答案: {golden_ans}")
                is_correct = gen_answer == golden_ans
            else:
                # 使用GPT-4o-mini判断
                print(f"无法直接提取答案，使用GPT-4o-mini判断...")
                judge_result = judge_with_gpt4o(response, golden_ans, question, id_field)
                is_correct = judge_result == 1
                print(f"GPT-4o-mini判断结果: {'正确' if is_correct else '错误'}")
            
            # 更新统计数据
            if is_correct:
                total_correct += 1
                subcategory_stats[subcategory]['correct'] += 1
                method_stats[method]['correct'] += 1
                d['judgment'] = 1
            else:
                total_wrong += 1
                subcategory_stats[subcategory]['wrong'] += 1
                method_stats[method]['wrong'] += 1
                d['judgment'] = 0
            
            subcategory_stats[subcategory]['total'] += 1
            method_stats[method]['total'] += 1
            
            # 记录每个方法下的子类别
            if subcategory not in method_subcategories[method]:
                method_subcategories[method][subcategory] = {'correct': 0, 'wrong': 0, 'total': 0}
            
            method_subcategories[method][subcategory]['total'] += 1
            if is_correct:
                method_subcategories[method][subcategory]['correct'] += 1
            else:
                method_subcategories[method][subcategory]['wrong'] += 1
            
            res.append(d)
    
    # 计算各子类别的准确率
    for subcategory in subcategory_stats:
        stats = subcategory_stats[subcategory]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    # 计算各方法的准确率和子类别准确率
    method_results = {}
    for method in method_stats:
        stats = method_stats[method]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # 计算该方法下所有子类别的准确率
        method_subcategory_results = {}
        for subcategory in method_subcategories[method]:
            sub_stats = method_subcategories[method][subcategory]
            sub_stats['accuracy'] = sub_stats['correct'] / sub_stats['total'] if sub_stats['total'] > 0 else 0.0
            method_subcategory_results[subcategory] = sub_stats
        
        # 计算Avg（加权平均）和Avg-C（非加权平均）
        method_avg = calculate_weighted_average(method_subcategory_results, stats['total'])
        method_avg_c = calculate_unweighted_average(method_subcategory_results)
        
        method_results[method] = {
            'total_questions': stats['total'],
            'correct': stats['correct'],
            'wrong': stats['wrong'],
            'accuracy': stats['accuracy'],
            'avg': method_avg,
            'avg_c': method_avg_c,
            'subcategories': method_subcategory_results
        }
    
    save_file = args.output_file
    # write res to jsonl
    with open(save_file, 'w', encoding='utf-8') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # 构建详细的metrics结果
    detailed_metrics = {
        'summary': {
            'total_questions': total_correct + total_wrong,
            'total_correct': total_correct,
            'total_wrong': total_wrong,
            'overall_accuracy': total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0.0
        },
        'subcategory_question_counts': dict(subcategory_questions),
        'subcategory_results': {},
        'method_results': method_results
    }
    
    # 添加每个子类别的详细结果
    for subcategory, stats in subcategory_stats.items():
        detailed_metrics['subcategory_results'][subcategory] = {
            'total_questions': stats['total'],
            'correct': stats['correct'],
            'wrong': stats['wrong'],
            'accuracy': stats['accuracy']
        }
    
    # 按准确率排序子类别
    sorted_subcategories = sorted(
        detailed_metrics['subcategory_results'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # 按准确率排序方法
    sorted_methods = sorted(
        method_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # 打印结果摘要
    print(f"\n=== Overall Results ===")
    print(f"Total Questions: {detailed_metrics['summary']['total_questions']}")
    print(f"Correct: {detailed_metrics['summary']['total_correct']}")
    print(f"Wrong: {detailed_metrics['summary']['total_wrong']}")
    print(f"Overall Accuracy: {detailed_metrics['summary']['overall_accuracy']:.4f}")
    
    print(f"\n=== Method Results ===")
    for method, stats in sorted_methods:
        print(f"{method}: {stats['correct']}/{stats['total_questions']} = {stats['accuracy']:.4f} (Avg: {stats['avg']:.4f}, Avg-C: {stats['avg_c']:.4f})")
    
    print(f"\n=== Subcategory Results ===")
    for subcategory, stats in sorted_subcategories:
        print(f"{subcategory}: {stats['correct']}/{stats['total_questions']} = {stats['accuracy']:.4f}")
    
    # save detailed metrics to json
    with open(save_file.replace('.jsonl', '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_metrics, f, ensure_ascii=False, indent=4)
    
    print(f"\nDetailed metrics saved to: {save_file.replace('.jsonl', '_metrics.json')}")


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