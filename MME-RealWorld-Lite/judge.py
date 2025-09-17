import re
import json
from argparse import ArgumentParser
from collections import defaultdict

def extract_answer_from_response(response):
    """
    提取response中最后一个<answer>标签的内容
    对于单选题和多选题都适用
    """
    ans_pattern = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    ans = ""
    if not response:
        return None
    
    # 检查是否包含unfinished标记
    if "unfinished" in response.lower():
        return None
    
    # 使用正则表达式找到所有<answer></answer>标签
    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    if answer_matches:
        # 返回最后一个匹配的内容，去除首尾空白
        last_answer = answer_matches[-1].strip()
        ans = last_answer if last_answer else "No <answer> tag found"
    else:
        ans = "No <answer> tag found"
    
    if ans == "No <answer> tag found":
        
        print(f"Warning: {ans} in response: {response}")
        # 尝试从response中提取选项，如果extracted_answer中包含**单一**选项，则返回该选项
        # 防止模型输出多个选项仍然判断正确，
        extracted_options = [opt for opt in ans_pattern if opt in response]
        if len(extracted_options) == 1:
            ans = extracted_options[0]
            # mapping: (A) -> A, (B) -> B, ...
            ans = ans.replace("(", "").replace(")", "")
            print("Warning: Extracted single option from response:", ans)
            assert ans in ["A", "B", "C", "D", "E"]
        else:
            ans = "No <answer> tag found"
       
    return ans


def extract_subcategory(id_field):
    """
    从id字段提取子类别
    格式: "<method>/<task_split>/<ignore>/<id_number>"
    返回: "<method>/<task_split>"
    """
    parts = id_field.split('/')
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
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
        weight = stats['total'] / total_questions  # 修改：使用 'total' 而不是 'total_questions'
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
    MMERealWorld-Lite全部都是多选题，所以只对模糊选项使用LLM做判断
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
            response = d.get('response', '')[0] # 取第一个response
            gen_answer = extract_answer_from_response(response)
            golden_ans = d.get('answer_w_choices', '')
            
            # 提取子类别和方法
            subcategory = extract_subcategory(d.get('id', ''))
            method = extract_method(d.get('id', ''))
            
            subcategory_questions[subcategory] += 1
            
            print(f"ID: {d.get('id', '')}, Method: {method}, Subcategory: {subcategory}")
            print(f"Generated Answer: {gen_answer}, Golden Answer: {golden_ans}")
            
            is_correct = gen_answer == golden_ans
            
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
    
    # 按准确率排序子类别（可选）
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
    parser = ArgumentParser(description="Judge evaluation script for MathVision inference results")
    parser.add_argument(
        "--input_file",
        type=str,
        default="MathVision_judge_results.jsonl",
        help="Path to the JSONL file containing inference results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="MMEE-RealWorld-Lite_judge_results.jsonl"
    )
    args = parser.parse_args()
    main(args)