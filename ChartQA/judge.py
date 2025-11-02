import json
import re
import time
from openai import OpenAI
import tqdm
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser

from utils import (
    is_exact_match,
    relaxed_correctness,
)

def extract_answer_from_response(response: str):
    """
    从response中提取模型的最终答案
    """
    if "<answer>" in response.lower() and "</answer>" in response.lower(): # use COT
        answer_match = response.split("<answer>")[-1].split("</answer>")[0]
        return answer_match.strip()
    elif r'boxed{' in response:
        # \boxed{...}
        answer_match = re.findall(r'\\boxed\{(.*?)\}', response)
        extracted_answer = ""
        if answer_match:
            extracted_answer = answer_match[-1].strip()
            return extracted_answer
        else: # boxed{...}
            answer_match = re.findall(r'boxed\{(.*?)\}', response)
            if answer_match:
                extracted_answer = answer_match[-1].strip()
            return extracted_answer
    else:
        return response.strip() # fallback to full response


def judge_with_gpt4o(response, golden_ans, question, id_field, client):
    """
    使用GPT-4o-mini判断模型答案是否正确
    """
    try:
        # 构建提示
        prompt = f"""Judge whether the model's answer to the question is correct.

question_id {id_field}

question: {question}

golden answer:{golden_ans}

model response: 
{response}

Please judge and response strictly, respond with only 0 or 1, where 1 means the model response aligns with golden answer, and 0 means it is not aligned."""
        
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
            print(f"[INFO] GPT-4o-mini Jugde Right, {result=}, {golden_ans=}, {response=}")
            return 1
        else:
            print(f"[INFO] GPT-4o-mini Judge Wrong, {result=}, {golden_ans=}, {response=}")
            return 0
            
    except Exception as e:
        print(f"GPT-4o调用错误 ({id_field}): {e}")
        time.sleep(2)  # 遇到错误等待一下再重试
        try:
            # 简化提示再试一次
            completion = client.chat.completions.create(
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


def load_inference_results(input_file):
    """Load inference results from jsonl file"""
    data = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def process_single_item(idx, data, client, args):
    """处理单个数据项"""
    # Extract necessary information
    problem = data.get('problem', '')
    problem_w_choices = data.get('problem_w_choices', '')
    answer = data.get('answer', '')
    answer_w_choices = data.get('answer_w_choices', '')
    
    # Determine question and standard answer
    if problem_w_choices and not problem:
        question = problem_w_choices
        standard_answer = answer_w_choices
    elif problem and not problem_w_choices:
        question = problem
        standard_answer = answer
    else:
        # Handle edge cases
        question = problem_w_choices if problem_w_choices else problem
        standard_answer = answer_w_choices if answer_w_choices else answer

    # Get model response (first element if it's a list)
    model_response = data.get('response', [''])[0] if isinstance(data.get('response'), list) else data.get('response', '')
    
    # Get ID
    id_field = data.get('id', 'unknown')
    
    # First we try exact match
    extracted_answer = extract_answer_from_response(model_response)
    normalized_model_answer = extracted_answer.strip().lower()
    normalized_std_answer = standard_answer.strip().lower()
    if not args.use_relax_accuracy:
        exact_match = is_exact_match(normalized_model_answer, normalized_std_answer)
    elif args.use_relax_accuracy:
        exact_match = relaxed_correctness(standard_answer, extracted_answer)
    
    if exact_match:
        print(f"idx: {idx} exact match. {normalized_model_answer=}, {normalized_std_answer=}")
        judgment = 1
    else:
        # If we cannot judge via exact match, we use gpt4o to judge
        print("Using GPT-4o-mini for judgment...")
        judgment = judge_with_gpt4o(model_response, standard_answer, question, id_field, client)
    
    result = {
        "id": id_field,
        "question": question,
        "standard_answer": standard_answer,
        "model_response": model_response,
        "judgment": judgment
    }
    
    # Add original data fields
    for key, value in data.items():
        if key not in result:
            result[key] = value
    
    return idx, result


def run_judge_evaluation(inference_data, client, args):
    """Run judge evaluation using GPT-4o-mini with multithreading"""
    print(f"Running judge evaluation on {len(inference_data)} samples...")
    
    results = {}
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_idx = {
            executor.submit(process_single_item, idx, data, client, args): idx 
            for idx, data in enumerate(inference_data)
        }

        # 创建进度条
        pbar = tqdm.tqdm(total=len(future_to_idx), desc="Processing")
        
        for future in as_completed(future_to_idx):
            original_idx = future_to_idx[future]
            try:
                # 获取处理结果
                idx, result = future.result()
                results[idx] = result
                
            except Exception as e:
                error_count += 1
                print(f"\n[ERROR] Processing idx {original_idx} failed: {e}")
                # 失败时创建默认结果
                data = inference_data[original_idx]
                result = {
                    "id": data.get('id', 'unknown'),
                    "question": data.get('problem_w_choices', '') or data.get('problem', ''),
                    "standard_answer": data.get('answer_w_choices', '') or data.get('answer', ''),
                    "model_response": data.get('response', [''])[0] if isinstance(data.get('response'), list) else data.get('response', ''),
                    "judgment": 0
                }
                # Add original data fields
                for key, value in data.items():
                    if key not in result:
                        result[key] = value
                results[original_idx] = result
            
            finally:
                pbar.update(1)
                # 实时显示准确率
                if len(results) > 0:
                    current_acc = sum(1 for r in results.values() if r.get("judgment") == 1) / len(results)
                    pbar.set_postfix({
                        'acc': f'{current_acc:.4f}',
                        'errors': error_count
                    })
        
        pbar.close()
    
    # 按索引排序，转换为列表
    results_list = [results[i] for i in sorted(results.keys())]
    
    return results_list


def calculate_metrics(results):
    """Calculate evaluation metrics"""
    total_samples = len(results)
    correct_judgments = sum(1 for r in results if r["judgment"] == 1)

    # Calculate total accuracy
    acc = correct_judgments / total_samples if total_samples > 0 else 0

    # Calculate human accuracy
    human_samples = [r for r in results if r['human_or_machine'] == 0]
    correct_human_judgments = sum(1 for r in human_samples if r["judgment"] == 1)
    human_acc = correct_human_judgments / len(human_samples) if human_samples else 0

    # Calculate machine accuracy
    machine_samples = [r for r in results if r['human_or_machine'] == 1]
    correct_machine_judgments = sum(1 for r in machine_samples if r["judgment"] == 1)
    machine_acc = correct_machine_judgments / len(machine_samples) if machine_samples else 0

    print(f"\n=== Judge Evaluation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct judgments: {correct_judgments}")
    print(f"Total Accuracy (acc): {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Human Accuracy: {human_acc:.4f} ({human_acc * 100:.2f}%)")
    print(f"Machine Accuracy: {machine_acc:.4f} ({machine_acc * 100:.2f}%)")

    return {
        "total_samples": total_samples,
        "correct_judgments": correct_judgments,
        "acc": acc,
        "human_acc": human_acc,
        "machine_acc": machine_acc
    }


def main(args):
    # Initialize OpenAI client
    if not args.api_key:
        raise ValueError("API key is required for GPT-4o-mini")
    
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.judge_api
    )
    
    print(f"Loading inference results from: {args.input_file}")
    
    # Load inference results
    inference_data = load_inference_results(args.input_file)
    print(f"Loaded {len(inference_data)} inference results")
    
    # Run judge evaluation
    print("Running judge evaluation with GPT-4o-mini...")
    results = run_judge_evaluation(inference_data, client, args)
    
    # Save results
    print(f"Saving judge results to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Judge results saved to: {args.output_file}")
    
    # Calculate and display metrics
    metrics = calculate_metrics(results)
    
    # Save metrics
    metrics_file = args.output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Judge evaluation script using GPT-4o-mini")
    
    parser.add_argument("--input_file", type=str, 
                       required=True,
                       help="Input file containing inference results")
    parser.add_argument("--output_file", type=str, 
                       required=True,
                       help="Output file for judge results")
    parser.add_argument("--use_relax_accuracy", action='store_true',
                       help="Whether to use relaxed accuracy for numeric answers")
    parser.add_argument("--judge_api", type=str, 
                       required=True,
                       help="API endpoint for GPT-4o-mini")
    parser.add_argument("--api_key", type=str, 
                       required=True,
                       help="API key for GPT-4o-mini")
    
    args = parser.parse_args()
    
    print("GPT-4o-mini Judge Evaluation Script")
    print("=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Judge API: {args.judge_api}")
    
    main(args)
