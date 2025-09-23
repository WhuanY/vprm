import json
import time
from openai import OpenAI
import tqdm
from argparse import ArgumentParser


def extract_answer_from_response(response: str) -> str:
    """
    从模型回答中提取最终答案，假设答案在最后一个\boxed{}中
    """
    import re
    # 使用正则表达式查找所有\boxed{...}模式
    matches = re.findall(r'\\boxed\{(.*?)\}', response)
    if matches:
        # 返回最后一个匹配的内容作为答案
        return matches[-1].strip()
    else:
        # 如果没有找到，返回整个响应的简化版本
        return response.strip().split('\n')[-1].strip()


def judge_with_gpt4o(response, golden_ans, question, id_field, client):
    """
    使用GPT-4o-mini判断模型答案是否正确
    """
    try:
        # 构建提示
        prompt = f"""Judge whether the model's answer to the multiple-choice question is correct.

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
            temperature=0
        )
        
        # 提取结果
        result = completion.choices[0].message.content.strip()
        
        # 标准化结果
        # 在这里打印判断输入，以免需要的时候无所参考。
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


def run_judge_evaluation(inference_data, client, args):
    """Run judge evaluation using GPT-4o-mini"""
    
    results = []
    
    print(f"Running judge evaluation on {len(inference_data)} samples...")
    
    for idx, data in enumerate(tqdm.tqdm(inference_data)):
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
        
        # Judge with GPT-4o-mini
        # first we try exact match
        normalized_model_resp = model_response.strip().lower()
        normalized_std_answer = standard_answer.strip().lower()
        if normalized_model_resp == normalized_std_answer:
            print(f"idx: {idx} exact match. {normalized_model_resp=}, {normalized_std_answer=}")
            judgment = 1
        else:
            # if we cannot judge via exact match, we use gpt4o to judge
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
        
        results.append(result)
    
    return results


def calculate_metrics(results):
    """Calculate evaluation metrics"""
    total_samples = len(results)
    correct_judgments = sum(1 for r in results if r["judgment"] == 1)
    acc = correct_judgments / total_samples if total_samples > 0 else 0
    
    print(f"\n=== Judge Evaluation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct judgments: {correct_judgments}")
    print(f"Accuracy (acc): {acc:.4f} ({acc*100:.2f}%)")
    
    return {
        "total_samples": total_samples,
        "correct_judgments": correct_judgments,
        "acc": acc
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