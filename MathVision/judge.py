judge_prompt = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
[Question]: {question}
[Standard Answer]: {standard_answer}
[Model_answer] : {model_answer}
"""



import json
import re
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import tqdm
from vllm import LLM, SamplingParams
from argparse import ArgumentParser


def extract_answer_from_response(response):
    """
    提取response中最后一个<answer>标签的内容
    对于单选题和多选题都适用
    """
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
        return last_answer if last_answer else "No <answer> tag found"
    else:
        return "No <answer> tag found"
    
    
def load_inference_results(input_file):
    """Load inference results from jsonl file"""
    data = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def prepare_judge_inputs(inference_data, processor, args):
    """Prepare inputs for judge model"""
    inputs = []
    
    print(f"Preparing judge inputs for {len(inference_data)} samples...")
    
    for idx, data in enumerate(tqdm.tqdm(inference_data)):
        # Extract necessary information
        problem = data['problem']
        problem_w_choices = data['problem_w_choices']
        answer = data['answer']
        answer_w_choices = data['answer_w_choices']
        question = ""
        standard_answer = ""
        if problem_w_choices and not problem:
            question = problem_w_choices
            assert answer == "", f'problem_w_choices exists, answer should be empty, {data}'
            standard_answer = answer_w_choices
        elif problem and not problem_w_choices:
            question = problem
            assert answer_w_choices == "", f'problem exists, answer_w_choices should be empty, {data}'
            standard_answer = answer
        else:
            raise ValueError(f"Both problem and problem_w_choices are present or both are empty for id {data['id']}, {data}")

        
        model_response = data['response']
        has_image = "image" in data and data["image"] and args.has_images
        
        # Create judge prompt
        judge_prompt = f"""Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.

[Question]: {question}
[Standard Answer]: {standard_answer}
[Model_answer]: {model_response}"""

        # Prepare content based on whether image exists
        if has_image:
            # For multimodal input with image
            content = [
                {"type": "image", "image": data["image"][0]},  # Use first image
                {"type": "text", "text": judge_prompt}
            ]
        else:
            # For text-only input
            content = [{"type": "text", "text": judge_prompt}]
        
        # Create messages
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Create input item
        if has_image:
            image_data, _ = process_vision_info(messages)
            input_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_data},
                "original_data": data
            }
        else:
            input_item = {
                "prompt": prompt,
                "original_data": data
            }
        
        inputs.append(input_item)
    
    return inputs

def run_judge_evaluation(inputs, llm, tokenizer, args):
    """Run judge evaluation using the LLM"""
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Use deterministic generation for consistent judgments
        top_p=1.0,
        top_k=50,
        n=1,
        max_tokens=20,  # Judge only outputs 0 or 1
        stop_token_ids=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
    )
    
    results = []
    batch_size = args.bz if args.bz else len(inputs)
    
    for idx in range(0, len(inputs), batch_size):
        batch_inputs = inputs[idx : idx + batch_size]
        
        # Prepare batch for inference
        batch_prompts = []
        for inp in batch_inputs:
            if "multi_modal_data" in inp:
                batch_prompts.append({
                    "prompt": inp["prompt"],
                    "multi_modal_data": inp["multi_modal_data"]
                })
            else:
                batch_prompts.append({"prompt": inp["prompt"]})
        
        # Generate judgments
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        
        # Process outputs
        for i, output in enumerate(outputs):
            original_idx = idx + i
            judgment_text = output.outputs[0].text.strip()
            
            # Extract judgment (0 or 1)
            try:
                if "1" in judgment_text:
                    judgment = 1
                elif "0" in judgment_text:
                    judgment = 0
                else:
                    # If neither 0 nor 1 is found, default to 0 (inconsistent)
                    print(f"[WARNING] Error parsing judgment, defaulting to 0")
                    print(f"Sample ID: {inputs[original_idx]['original_data'].get('id', 'unknown')}")
                    print(f"Judgment text: '{judgment_text}'")
                    print(f"Question: {inputs[original_idx]['original_data'].get('problem', '')[:200]}...")
                    print(f"Standard answer: {inputs[original_idx]['original_data'].get('answer', '')}")
                    print(f"Model response: {inputs[original_idx]['original_data'].get('response', [''])[0][:200]}...")
                    print("-" * 80)
                    # 简单规则重新处理一下
                    rule_base_extracted =  extract_answer_from_response(inputs[original_idx]["original_data"].get("response", [""])[0]), # 从 model_response 中提取答案
                    if rule_base_extracted == inputs[original_idx]["original_data"].get("answer", "") or rule_base_extracted == inputs[original_idx]["original_data"].get("answer_w_choices", ""):
                        print("[INFO] Rule-based extraction matches standard answer, changing setting judgment to 1")
                        judgment = 1
                    else:
                        judgment = 0
            except Exception as e:
                print(f"[WARNING] Exception parsing judgment, defaulting to 0")
                print(f"Exception: {str(e)}")
                print(f"Sample ID: {inputs[original_idx]['original_data'].get('id', 'unknown')}")
                print(f"Judgment text: '{judgment_text}'")
                print(f"Question: {inputs[original_idx]['original_data'].get('problem', '')[:200]}...")
                print(f"Standard answer: {inputs[original_idx]['original_data'].get('answer', '')}")
                print(f"Model response: {inputs[original_idx]['original_data'].get('response', [''])[0][:200]}...")
                print("-" * 80)
                judgment = 0

            
            result = {
                "id": inputs[original_idx]["original_data"]["id"],
                # question 还需要判断一下是否是多选题根据inputs
                # "question": inputs[original_idx]["original_data"].get("problem", ""),
                "question": inputs[original_idx]["original_data"].get("problem", "") if inputs[original_idx]["original_data"].get("problem", "") else inputs[original_idx]["original_data"].get("problem_w_choices", ""),
                # "standard_answer": inputs[original_idx]["original_data"].get("answer", ""),
                "standard_answer": inputs[original_idx]["original_data"].get("answer", "") if inputs[original_idx]["original_data"].get("answer", "") else inputs[original_idx]["original_data"].get("answer_w_choices", ""),
                "model_response": inputs[original_idx]["original_data"].get("response", [""])[0],
                "model_response_ans_extracted": extract_answer_from_response(inputs[original_idx]["original_data"].get("response", [""])[0]), # 从 model_response 中提取答案
                "judgment": judgment,
                "judgment_text": judgment_text,
                "prompt": inputs[original_idx]["prompt"]
            }

            
            # Add original data fields
            for key, value in inputs[original_idx]["original_data"].items():
                if key not in result:
                    result[key] = value
            
            results.append(result)
    
    return results

# def calculate_metrics(results):
#     """Calculate evaluation metrics"""
#     total_samples = len(results)
#     correct_judgments = sum(1 for r in results if r["judgment"] == 1)
#     accuracy = correct_judgments / total_samples if total_samples > 0 else 0
    
#     print(f"\n=== Judge Evaluation Results ===")
#     print(f"Total samples: {total_samples}")
#     print(f"Correct judgments: {correct_judgments}")
#     print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
#     return {
#         "total_samples": total_samples,
#         "correct_judgments": correct_judgments,
#         "accuracy": accuracy
#     }

def calculate_metrics(results):
    """Calculate evaluation metrics with rule-based correction"""
    total_samples = len(results)
    
    # 原始准确率（基于judge模型的判断）
    correct_judgments_llm = sum(1 for r in results if r["judgment"] == 1)
    llm_acc = correct_judgments_llm / total_samples if total_samples > 0 else 0
    
    # 修正后的准确率（对judgment=0的情况进行规则修正）
    corrected_judgments = 0
    rule_corrections = 0
    
    def normalize_answer(answer):
        """标准化答案格式"""
        if not answer:
            return ""
        answer = str(answer).strip().lower()
        # 移除常见的标点符号
        answer = answer.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        return answer
    
    # def is_choice_match(extracted, standard):
    #     """检查选择题答案是否匹配"""
    #     # 提取字母选项 A, B, C, D
    #     import re
    #     extracted_choice = re.search(r'\b([A-Za-z])\b', extracted)
    #     standard_choice = re.search(r'\b([A-Za-z])\b', standard)
        
    #     if extracted_choice and standard_choice:
    #         return extracted_choice.group(1).upper() == standard_choice.group(1).upper()
    #     return False
    
    def is_numeric_match(extracted, standard):
        # """检查数值答案是否匹配"""
        # import re
        # try:
        #     # 提取数字
        #     extracted_nums = re.findall(r'-?\d+\.?\d*', extracted)
        #     standard_nums = re.findall(r'-?\d+\.?\d*', standard)
            
        #     if extracted_nums and standard_nums:
        #         return float(extracted_nums[-1]) == float(standard_nums[-1])
        # except:
        #     pass
        # return False
        """检查数值答案是否匹配"""
        import re
        try:
            # 提取所有数字
            extracted_nums = re.findall(r'-?\d+\.?\d*', extracted)
            standard_nums = re.findall(r'-?\d+\.?\d*', standard)
            
            if not extracted_nums or not standard_nums:
                return False
            
            # 转换为浮点数列表
            extracted_floats = [float(num) for num in extracted_nums]
            standard_floats = [float(num) for num in standard_nums]
            
            # 情况1: 如果只有一个数字，直接比较
            if len(extracted_floats) == 1 and len(standard_floats) == 1:
                return abs(extracted_floats[0] - standard_floats[0]) < 1e-6
            
            # 情况2: 如果数字个数不同，不匹配
            if len(extracted_floats) != len(standard_floats):
                return False
            
            # 情况3: 如果有多个数字，需要所有数字都匹配
            for ext_num, std_num in zip(extracted_floats, standard_floats):
                if abs(ext_num - std_num) >= 1e-6:  # 允许微小的浮点误差
                    return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    for r in results:
        if r["judgment"] == 1:
            # 原本就正确的
            corrected_judgments += 1
        else:
            # judgment=0的情况，进行规则修正
            model_extracted = r.get("model_response_ans_extracted", "")
            standard_answer = r.get("standard_answer", "")
            
            is_correct = False
            
            # 规则1: 完全匹配（标准化后）
            if normalize_answer(model_extracted) == normalize_answer(standard_answer):
                print("-"*100)
                print("[INFO] judgement = 0 but exact match found.")
                print("model_extracted:", model_extracted)
                print("standard_answer:", standard_answer)
                print("-"*100)
                is_correct = True
            
            # # 规则2: 选择题字母匹配
            # elif is_choice_match(model_extracted, standard_answer):
            #     print("-"*100)
            #     print("[INFO] judgement = 0 but choice match found.")
            #     print("model_extracted:", model_extracted)
            #     print("standard_answer:", standard_answer)
            #     print("-"*100)
            #     is_correct = True
            
            # 规则3: 数值匹配
            elif is_numeric_match(model_extracted, standard_answer):
                print("-"*100)
                print("[INFO] judgement = 0 but numeric match found.")
                print("model_extracted:", model_extracted)
                print("standard_answer:", standard_answer)
                print("-"*100)
                is_correct = True
            
            if is_correct:
                corrected_judgments += 1
                rule_corrections += 1
    
    acc = corrected_judgments / total_samples if total_samples > 0 else 0
    
    print(f"\n=== Judge Evaluation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"LLM Judge correct judgments: {correct_judgments_llm}")
    print(f"LLM Judge accuracy (llm_acc): {llm_acc:.4f} ({llm_acc*100:.2f}%)")
    print(f"Rule corrections applied: {rule_corrections}")
    print(f"Corrected judgments: {corrected_judgments}")
    print(f"Final accuracy (acc): {acc:.4f} ({acc*100:.2f}%)")
    print(f"Improvement: {(acc - llm_acc)*100:.2f} percentage points")
    
    return {
        "total_samples": total_samples,
        "correct_judgments_llm": correct_judgments_llm,
        "llm_acc": llm_acc,
        "rule_corrections": rule_corrections,
        "corrected_judgments": corrected_judgments,
        "acc": acc,
        "improvement": acc - llm_acc
    }


def main(args):
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Initialize LLM
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"image": 10, "video": 2},
        gpu_memory_utilization=0.9,
    )
    
    print(f"Loading inference results from: {args.input_file}")
    
    # Load inference results
    inference_data = load_inference_results(args.input_file)
    print(f"Loaded {len(inference_data)} inference results")
    
    # Prepare judge inputs
    judge_inputs = prepare_judge_inputs(inference_data, processor, args)
    
    # Run judge evaluation
    print("Running judge evaluation...")
    results = run_judge_evaluation(judge_inputs, llm, tokenizer, args)
    
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
    parser = ArgumentParser(description="Judge evaluation script for MathVision inference results")
    
    parser.add_argument("--model_name_or_path", type=str, 
                       default="/home/minyingqian/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to the judge model")
    parser.add_argument("--input_file", type=str, 
                       default="MathVision_inferenced.jsonl",
                       help="Input file containing inference results")
    parser.add_argument("--output_file", type=str, 
                       default="MathVision_judge_results.jsonl",
                       help="Output file for judge results")
    parser.add_argument("--tp", type=int, default=4,
                       help="Tensor parallel size")
    parser.add_argument("--bz", type=int, default=20,
                       help="Batch size for inference")
    parser.add_argument("--has_images", type=int, default=1,
                       help="Whether the dataset contains images")
    
    args = parser.parse_args()
    
    print("MathVision Judge Evaluation Script")
    print("=" * 50)
    print(f"Judge model: {args.model_name_or_path}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Tensor parallel size: {args.tp}")
    print(f"Batch size: {args.bz}")
    print(f"Has images: {bool(args.has_images)}")
    
    main(args)

