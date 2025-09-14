import re
import json
from argparse import ArgumentParser


def extract_answer_from_response(response):
    """
    提取response中最后一个<answer>标签的内容
    对于单选题和多选题都适用
    """
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
    
    return ans


def main(args):
    """
    MMERealWorld-Lite全部都是多选题，所以就不用模型判断了
    """
    input_file = args.input_file
    correct_cnt = 0
    wrong_cnt = 0
    res = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            response = d.get('response', '')[0] # 取第一个response
            gen_answer = extract_answer_from_response(response)
            golden_ans = d.get('answer_w_choices', '')
            print(f"Generated Answer: {gen_answer}, Golden Answer: {golden_ans}")
            if gen_answer == golden_ans:
                correct_cnt += 1
                d['judgment'] = 1
            else:
                d['judgment'] = 0
                wrong_cnt += 1
    
    save_file = args.output_file
    # write res to jsonl
    with open(save_file, 'w', encoding='utf-8') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    metrics = {
        'total': correct_cnt + wrong_cnt,
        'correct': correct_cnt,
        'wrong': wrong_cnt,
        'accuracy': correct_cnt / (correct_cnt + wrong_cnt) if (correct_cnt + wrong_cnt) > 0 else 0.0
    }

    # save metrics to json
    with open(save_file.replace('.jsonl', '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
              

    


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