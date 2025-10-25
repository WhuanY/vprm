import argparse
from posix import kill
import re
from tqdm import tqdm
import time
import json
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from openai import OpenAI

id_raw = {example['id']: example for example in load_jsonl("./data/test.jsonl")}

def judge_with_gpt4o(response, golden_ans, question, id_field, client):
    """
    使用GPT-4o-mini判断模型答案是否正确
    """
    prompt = r"""Judge whether the model's answer to the question is correct.
question_id {id_field}
question: {question}
golden answer:{golden_ans}
model response: 
{response}
Please judge and response strictly, respond with ONLY 0 or 1, where 1 means the model response aligns with golden answer, and 0 means it is not aligned. 
Note that the model response may try to express the answer in a different format, the answer is still correct. 
For example: 
- model response: "The answer is 28 degrees"; golden answer:  "28°". -> aligned
- model response: "<think>\...nThus, the correct measure of \\( x \\) is:\n\n\\[\nx = \\frac{360^\\circ}{7} \\approx 51.43^\\circ\n\\]\n\nThis is the correct measure of angle \\( x \\) in degrees.\n</think>\n<answer>\nThe measure of \\( x \\) is \\( 51.43^\\circ \\).\n</answer>"; golden_answer:"\\frac{360}7" -> aligned
"""
    
    # 调用OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert judge. Respond with only 0 or 1."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0,
    )
    
    # 提取结果
    result = completion.choices[0].message.content.strip()
    
    # 标准化结果
    if "1" in result:
        # print(f"[INFO] GPT-4o-mini Jugde Right, {result=}, {golden_ans=}, {response=}")
        return 1
    else:
        # print(f"[INFO] GPT-4o-mini Judge Wrong, {result=}, {golden_ans=}, {response=}")
        return 0
            

def evaluate(answer_file, regen_answer=True, use_cot="1"):
    lines = load_jsonl(answer_file)
    for line in tqdm(lines, desc='gen_correct'):
        raw_exampe = id_raw[line['id']]

        gt_answer = str(raw_exampe['answer'])
        if len(raw_exampe['options']) > 0:
            gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')]
        else:
            gt_answer_value = ''

        if 'model_answer' not in line or regen_answer:
            model_answer = line['response'][0].strip() # NOTE: 这里我修改了，只取第一个response
            # we first try to extract the final answer from "<answer>...</answer>"
            # 提取 <answer> 标签内的内容
            if use_cot == "1":
                answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', model_answer, re.DOTALL)
                if answer_match:
                    model_answer = answer_match.group(1).strip()
            

                for c in 'ABCDE':
                    if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                        model_answer = c
                    elif f"({c})".upper() in model_answer:
                        model_answer = c
                    elif f"{c})".upper() in model_answer:
                        model_answer = c
                    else:
                        pass
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                print("[is_number]")
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                print("[not boxed]")
                for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                
            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    save_jsonl(answer_file, lines, t_stamp=False)


# def evaluate(answer_file, regen_answer=False, use_cot="0", args=None):
#     client = OpenAI(
#         api_key=args.api_key,
#         base_url=args.judge_url
#     )
#     lines = load_jsonl(answer_file)
#     for line in tqdm(lines, desc='gen_correct'):
#         raw_exampe = id_raw[line['id']]

#         gt_answer = str(raw_exampe['answer'])
#         if len(raw_exampe['options']) > 0:
#             gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')]
#         else:
#             gt_answer_value = ''

#         if 'model_answer' not in line or regen_answer:
#             # Get raw response
#             if isinstance(line['response'], list):
#                 raw_response = line['response'][0].strip()
#             else:
#                 raw_response = line['response'].strip()
            
#             if use_cot == "1":
#                 # Extract answer from <answer></answer> tags
#                 answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_response, re.DOTALL | re.IGNORECASE)
#                 if answer_match:
#                     model_answer = answer_match.group(1).strip()
#                 else:
#                     # Fallback: if no <answer> tags, use full response
#                     model_answer = raw_response
#             else:
#                 # Official mode: use full response
#                 model_answer = raw_response
            
#             # Original official logic for answer extraction
#             for c in 'ABCDE':
#                 if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
#                     model_answer = c
#             if is_number(model_answer.split('is ')[-1].rstrip('.')):
#                 model_answer = model_answer.split('is ')[-1].rstrip('.')
#             if 'oxed{' not in model_answer and use_cot == "0":
#                 for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
#                     raw_model_answer = model_answer
#                     model_answer = model_answer.split(flag)[-1].strip()
#                     if flag in raw_model_answer:
#                         model_answer = model_answer.split('\n')[0].split('. ')[0]
#                     flag = flag.replace('the', 'The')
#                     raw_model_answer = model_answer
#                     model_answer = model_answer.split(flag)[-1].strip()
#                     if flag in raw_model_answer:
#                         model_answer = model_answer.split('\n')[0].split('. ')[0]
#             elif model_answer.count('oxed{') > 1:
#                 model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                
#             model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
#             line['model_answer'] = model_answer
#         else:
#             model_answer = line['model_answer']
#         is_rule_correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
#         if not is_rule_correct and len(model_answer) > 10: # if the model answer is too short, it can be judged rule-based
#             is_lasj_correct = judge_with_gpt4o(raw_response, gt_answer, raw_exampe['question'], raw_exampe["id"], client)
#         else:
#             is_lasj_correct = is_rule_correct
#         line['correct'] = is_rule_correct or is_lasj_correct
#         # line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
#     save_jsonl(answer_file, lines, t_stamp=False)


def process_single_line(line, id_raw, use_cot, client, args):
    """Process a single line for evaluation"""
    raw_exampe = id_raw[line['id']]

    gt_answer = str(raw_exampe['answer'])
    if len(raw_exampe['options']) > 0:
        gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')]
    else:
        gt_answer_value = ''

    # Get raw response
    if isinstance(line['response'], list):
        raw_response = line['response'][0].strip()
    else:
        raw_response = line['response'].strip()
    
    if use_cot == "1":
        # Extract answer from <answer></answer> tags
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            model_answer = answer_match.group(1).strip()
        else:
            # Fallback: if no <answer> tags, use full response
            model_answer = raw_response
    else:
        # Official mode: use full response
        model_answer = raw_response
    
    # Original official logic for answer extraction
    for c in 'ABCDE':
        if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
            model_answer = c
    if is_number(model_answer.split('is ')[-1].rstrip('.')):
        model_answer = model_answer.split('is ')[-1].rstrip('.')
    if 'oxed{' not in model_answer and use_cot == "0":
        for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split('\n')[0].split('. ')[0]
            flag = flag.replace('the', 'The')
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split('\n')[0].split('. ')[0]
    elif model_answer.count('oxed{') > 1:
        model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
        
    model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
    
    line['model_answer'] = model_answer
    
    # Rule-based judgment
    is_rule_correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    
    # GPT-4o judgment for long answers
    if not is_rule_correct:
        is_lasj_correct = judge_with_gpt4o(raw_response, gt_answer, raw_exampe['question'], raw_exampe["id"], client)
    else:
        is_lasj_correct = is_rule_correct
    
    line['correct'] = is_rule_correct or is_lasj_correct
    
    return line


def evaluate(answer_file, regen_answer=False, use_cot="0", args=None, max_workers=8):
    """
    Evaluate answers with multi-threading support
    
    Args:
        answer_file: Path to the answer file
        regen_answer: Whether to regenerate answers
        use_cot: "1" for COT mode, "0" for official mode
        args: Arguments containing api_key and judge_url
        max_workers: Number of parallel threads (default: 8)
    """
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.judge_url
    )
    
    lines = load_jsonl(answer_file)
    
    # Filter lines that need processing
    lines_to_process = []
    for line in lines:
        if 'model_answer' not in line or regen_answer:
            lines_to_process.append(line)
        else:
            # Already processed, just add correct judgment
            model_answer = line['model_answer']
            raw_exampe = id_raw[line['id']]
            gt_answer = str(raw_exampe['answer'])
            if len(raw_exampe['options']) > 0:
                gt_answer_value = raw_exampe['options'][ord(gt_answer)-ord('A')]
            else:
                gt_answer_value = ''
            line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    
    if len(lines_to_process) == 0:
        print("No lines to process. All answers already generated.")
        save_jsonl(answer_file, lines, t_stamp=False)
        return
    
    # Multi-threaded processing
    processed_lines = []
    write_lock = Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(process_single_line, line, id_raw, use_cot, client, args): line 
            for line in lines_to_process
        }
        
        # Process results with progress bar
        with tqdm(total=len(lines_to_process), desc='gen_correct') as pbar:
            for future in as_completed(future_to_line):
                try:
                    result = future.result()
                    with write_lock:
                        processed_lines.append(result)
                except Exception as exc:
                    original_line = future_to_line[future]
                    print(f"\nError processing line {original_line.get('id', 'unknown')}: {exc}")
                    with write_lock:
                        processed_lines.append(original_line)
                finally:
                    pbar.update(1)
    
    # Merge processed lines back (maintain original order by id)
    processed_dict = {line['id']: line for line in processed_lines}
    for i, line in enumerate(lines):
        if line['id'] in processed_dict:
            lines[i] = processed_dict[line['id']]
    
    save_jsonl(answer_file, lines, t_stamp=False)
    
    # Print summary
    correct_count = sum(1 for line in lines if line.get('correct', False))
    total_count = len(lines)
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"\n{'='*50}")
    print(f"Evaluation completed!")
    print(f"Correct: {correct_count}/{total_count} ({accuracy:.2f}%)")
    print(f"{'='*50}")



def math_level_subject_acc(answer_file):
    print(answer_file)
    lines = load_jsonl(answer_file)
    
    results_dict = {}
    for line in tqdm(lines, desc='math_level_subject_acc'):
        correct = line['correct']
        raw_exampe = id_raw[line['id']]
        subject = raw_exampe['subject']
        level = raw_exampe['level']
        for key in [
            '-all', 
            f'-level{level}', 
            f'{subject}', 
            f'{subject}_level{level}', 
            f'-level{level}_{subject}'
            ]:
            if key not in results_dict:
                results_dict[key] = [0,0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1


    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(answer_file), ':\t', results_dict['-all'])
    json.dump(results_dict, open(answer_file.replace('.jsonl', '_result.json'), 'w'), indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default=None, help='The name of the file to evaluate in the outputs directory.')
    parser.add_argument('--use_cot', type=str, default="0", help="whether the inference file is infered with cot format.")
    parser.add_argument('--api_key', type=str, default="sk-xxxx")
    parser.add_argument('--judge_url', type=str, default="https://aigc.x-see.cn/v1")
    args = parser.parse_args()

    
    output_dir = './outputs/'

    if args.eval_file:
        file_path = os.path.join(output_dir, args.eval_file)
        if os.path.exists(file_path):
            print(f"Evaluating single file: {file_path}")
            evaluate(file_path, regen_answer=True, use_cot=args.use_cot, args=args)
            math_level_subject_acc(file_path)
        else:
            print(f"Error: File not found at {file_path}")
    else:
        print("Evaluating all .jsonl files in the outputs directory...")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    evaluate(file_path, regen_answer=True, use_cot=args.use_cot, args=args)
                    math_level_subject_acc(file_path)
    # for root, dirs, files in os.walk('./outputs/'):
    #     for file in files:
    #         if file.endswith('.jsonl'):
    #             evaluate(os.path.join(root, file), True)
    #             math_level_subject_acc(os.path.join(root, file))