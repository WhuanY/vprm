import json
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI


def extract_answer_from_response(response):
    """
    try to extract the answer from the response, if the response contains <answer> tags, return the answer inside the tags.
    if the response does not contain <answer> tags, we guess the answer from the response.
    when return None, it means the answer is not found. 
    """
    if not response:
        return None
    
    # First try to extract from <answer> tags
    if "<answer>" in response and "</answer>" in response:
        try:
            extracted_answer = response.split("<answer>")[1].split("</answer>")[0].strip()
            
            # Try to match single letter options
            for option in "ABCDE":
                if extracted_answer.upper() in [option, f"({option})", f"({option})", f"{option}.", f"{option})"]:
                    return option
            
            # Try to extract from full answer text like "(A) Some text"
            match = re.match(r'\(?([A-E])\)?', extracted_answer.upper())
            if match:
                return match.group(1)
            
            # Handle text answers like "All the above answers are wrong."
            if "all" in extracted_answer.lower() and "wrong" in extracted_answer.lower():
                return "E"
            
            return extracted_answer
        except:
            return None
    
    # Fallback: try to find answer in the entire response
    # Look for patterns like "(A)", "A)", "A.", or standalone "A"
    matches = re.findall(r'\(?([A-E])\)?\.?(?:\s|$)', response.upper())
    if matches:
        # Return the last match (usually the final answer)
        return matches[-1]
    
    # Last resort: check for text indicators
    if "all" in response.lower() and "wrong" in response.lower():
        return "E"
    
    return None


def judge_with_model(response, golden_ans, question, identifier, client, api_key, model="gpt-4o-mini"):
    if not api_key:
        print(f"[WARNING] Missing API key, skip LLM judgment for {identifier}, mark as incorrect.")
        return 0

    if client is None:
        client = OpenAI(api_key=api_key)

    prompt = (
        "Judge whether the model's answer to the multiple-choice question is correct.\n\n"
        f"question_id: {identifier}\n\n"
        f"question: {question}\n\n"
        f"golden answer: {golden_ans}\n\n"
        "model response:\n"
        f"{response}\n\n"
        "Please respond with only 0 or 1 where 1 means the model response aligns with the golden answer, and 0 means it does not align."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert judge. Respond with only 0 or 1."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0,
        )
        result = completion.choices[0].message.content.strip()
        if "1" in result:
            print(f"LLM judgment result: {golden_ans=} {response=} {result=}", flush=True)
            return 1
        else:
            print(f"LLM judgment result: {golden_ans=} {response=} {result=}", flush=True)
        return 0
    except Exception as exc:
        print(f"[WARNING] LLM judgment failed for {identifier}: {exc}")
        time.sleep(2)
        print("FUCK")
        return 0


def extract_subcategory(identifier):
    if not identifier:
        return "unknown"
    parts = identifier.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    print(f"[WARNING] Unable to extract subcategory from id: {identifier}")
    return "unknown"


def extract_method(identifier):
    if not identifier:
        return "unknown"
    parts = identifier.split("/")
    if len(parts) >= 1 and parts[0]:
        return parts[0].lower()
    print(f"[WARNING] Unable to extract method from id: {identifier}")
    return "unknown"


def calculate_weighted_average(subcategory_results, total_questions):
    if total_questions == 0:
        return 0.0
    weighted_sum = 0.0
    for stats in subcategory_results.values():
        weight = stats["total"] / total_questions
        weighted_sum += stats["accuracy"] * weight
    return weighted_sum


def calculate_unweighted_average(subcategory_results):
    if not subcategory_results:
        return 0.0
    total_accuracy = sum(stats["accuracy"] for stats in subcategory_results.values())
    return total_accuracy / len(subcategory_results)


def build_question_text(record):
    question = record.get("problem_w_choices") or record.get("problem") or record.get("Text") or ""
    if question:
        return question
    text = record.get("Text") or record.get("question") or ""
    choices = record.get("Answer choices") or record.get("answer_choices") or []
    if choices:
        choices_str = "\n".join(str(choice) for choice in choices)
        return f"{text}\n{choices_str}".strip()
    return str(text)


def judge_single_record(record, api_key, judge_api):
    """
    Judge a single record and return the updated record with judgment result.
    This function will be executed in parallel.
    """
    # Create a new client for each thread
    client = None
    if api_key:
        client = OpenAI(api_key=api_key, base_url=judge_api)
    
    response = record.get("response", "")
    golden_ans = record.get("Ground truth")
    identifier = record.get("Question_id")
    question = build_question_text(record)

    subcategory = extract_subcategory(identifier)
    method = extract_method(identifier)

    print(f"Judging {identifier} | method={method} | subcategory={subcategory}")

    extracted_answer = extract_answer_from_response(response)
    is_correct = False

    # judge process: extract by rule and try exact match. If not match, we use LLM to judge for more inclusive cases.
    if extracted_answer is not None and golden_ans:
        is_correct = extracted_answer == golden_ans.upper()
    
    if not is_correct and extracted_answer in ["ABCDE"]: 
        # if the extracted answer is not options and wrong, by far we cannot make sure if the answer is correct or not. 
        # so we use LLM to judge for more inclusive cases.
        judge_result = judge_with_model(response, golden_ans, question, identifier, client, api_key)
        is_correct = judge_result == 1
    
    # Add judgment result to record
    record["judgment"] = 1 if is_correct else 0
    record["subcategory"] = subcategory
    record["method"] = method
    
    return record


def main(args):
    # Read all records
    records = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            records.append(json.loads(line))
    
    print(f"Total records to judge: {len(records)}")
    
    # Parallel judgment
    judged_records = []
    print(f"\n=== Starting parallel judgment with {args.max_workers} workers ===")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_record = {
            executor.submit(judge_single_record, record, args.api_key, args.judge_api): record 
            for record in records
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_record):
            try:
                judged_record = future.result()
                judged_records.append(judged_record)
            except Exception as exc:
                original_record = future_to_record[future]
                identifier = original_record.get("Question_id", "unknown")
                print(f"[ERROR] Record {identifier} generated an exception: {exc}")
                # Mark as incorrect if judgment fails
                original_record["judgment"] = 0
                original_record["subcategory"] = extract_subcategory(identifier)
                original_record["method"] = extract_method(identifier)
                judged_records.append(original_record)
    
    print(f"\n=== Judgment complete, processing {len(judged_records)} records ===")
    
    # Serial statistics calculation
    total_correct = 0
    total_wrong = 0
    subcategory_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    subcategory_questions = defaultdict(int)
    method_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    method_subcategories = defaultdict(dict)
    
    for record in judged_records:
        is_correct = record.get("judgment", 0) == 1
        subcategory = record.get("subcategory", "unknown")
        method = record.get("method", "unknown")
        
        subcategory_questions[subcategory] += 1
        
        # Update statistics
        if is_correct:
            total_correct += 1
            subcategory_stats[subcategory]["correct"] += 1
            method_stats[method]["correct"] += 1
        else:
            total_wrong += 1
            subcategory_stats[subcategory]["wrong"] += 1
            method_stats[method]["wrong"] += 1
        
        subcategory_stats[subcategory]["total"] += 1
        method_stats[method]["total"] += 1
        
        if subcategory not in method_subcategories[method]:
            method_subcategories[method][subcategory] = {"correct": 0, "wrong": 0, "total": 0}
        
        method_subcategories[method][subcategory]["total"] += 1
        if is_correct:
            method_subcategories[method][subcategory]["correct"] += 1
        else:
            method_subcategories[method][subcategory]["wrong"] += 1
    
    # Calculate accuracies
    for stats in subcategory_stats.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    method_results = {}
    for method, stats in method_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        method_subcategory_results = {}
        for subcategory, sub_stats in method_subcategories[method].items():
            sub_stats["accuracy"] = sub_stats["correct"] / sub_stats["total"] if sub_stats["total"] > 0 else 0.0
            method_subcategory_results[subcategory] = sub_stats
        
        method_avg = calculate_weighted_average(method_subcategory_results, stats["total"])
        method_avg_c = calculate_unweighted_average(method_subcategory_results)
        
        method_results[method] = {
            "total_questions": stats["total"],
            "correct": stats["correct"],
            "wrong": stats["wrong"],
            "accuracy": stats["accuracy"],
            "avg": method_avg,
            "avg_c": method_avg_c,
            "subcategories": method_subcategory_results,
        }
    
    # Write output
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for record in judged_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Prepare detailed metrics
    detailed_metrics = {
        "summary": {
            "total_questions": total_correct + total_wrong,
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "overall_accuracy": total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0.0,
        },
        "subcategory_question_counts": dict(subcategory_questions),
        "subcategory_results": {},
        "method_results": method_results,
    }
    
    for subcategory, stats in subcategory_stats.items():
        detailed_metrics["subcategory_results"][subcategory] = {
            "total_questions": stats["total"],
            "correct": stats["correct"],
            "wrong": stats["wrong"],
            "accuracy": stats["accuracy"],
        }
    
    sorted_subcategories = sorted(
        detailed_metrics["subcategory_results"].items(),
        key=lambda item: item[1]["accuracy"],
        reverse=True,
    )
    
    sorted_methods = sorted(
        method_results.items(),
        key=lambda item: item[1]["accuracy"],
        reverse=True,
    )
    
    print("\n=== Overall Results ===")
    print(f"Total Questions: {detailed_metrics['summary']['total_questions']}")
    print(f"Correct: {detailed_metrics['summary']['total_correct']}")
    print(f"Wrong: {detailed_metrics['summary']['total_wrong']}")
    print(f"Overall Accuracy: {detailed_metrics['summary']['overall_accuracy']:.4f}")
    
    print("\n=== Method Results ===")
    for method, stats in sorted_methods:
        print(
            f"{method}: {stats['correct']}/{stats['total_questions']} = {stats['accuracy']:.4f} "
            f"(Avg: {stats['avg']:.4f}, Avg-C: {stats['avg_c']:.4f})"
        )
    
    print("\n=== Subcategory Results ===")
    for subcategory, stats in sorted_subcategories:
        print(
            f"{subcategory}: {stats['correct']}/{stats['total_questions']} = {stats['accuracy']:.4f}"
        )
    
    metrics_path = args.output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(detailed_metrics, metrics_file, ensure_ascii=False, indent=4)
    
    print(f"\nDetailed metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Judge evaluation script for MME-RealWorld")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/MME_RealWorld_first5-inferenced.jsonl",
        help="Path to the JSONL file containing inference results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/MME_RealWorld_first5-judge_results.jsonl",
    )
    parser.add_argument(
        "--judge_api",
        type=str,
        default="https://aigc.x-see.cn/v1",
        help="API endpoint for judgment",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the judgment API",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=100,
        help="Maximum number of parallel workers for judgment",
    )
    args = parser.parse_args()
    main(args)
    # resp = "<think>\nTo determine the correct answer, let's analyze the image step by step:\n\n1. **Cars**: The image shows multiple cars parked on both sides of the street.\n2. **Pedestrians**: There are a few pedestrians visible on the sidewalk.\n3. **Trucks**: There is one truck visible on the street.\n\nNow, let's evaluate each option:\n\n(A) There are many cars, three pedestrians, and one trailer.\n- This option includes three pedestrians, which is correct.\n\n(B) There is one car, three trucks, and one trailer.\n- This option includes three trucks, which is incorrect.\n\n(C) There are many pedestrians, many cars, and two trucks.\n- This option includes many pedestrians, which is incorrect.\n\n(D) There are many cars, one pedestrian, and one truck.\n- This option includes one pedestrian, which is correct.\n\n(E) All the above answers are wrong.\n- This option is incorrect because options (A) and (D) are correct.\n\nBased on the analysis, the correct answer is either (A) or (D). However, since (A) includes three pedestrians, which is more than one pedestrian, it is the more comprehensive option.\n\nTherefore, the correct answer is (A).\n\n</think>\n<answer>\n(A)\n</answer>"

    # print(extract_answer_from_response(resp))