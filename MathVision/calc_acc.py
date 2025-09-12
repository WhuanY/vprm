import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

model_path = "/home/minyingqian/models/Qwen2.5-7B-Instruct"
inference_output_file = "judge.jsonl" # (id, gt, ex, judge)
metric_file = "acc.txt" # acc score
judge_prompt = """Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and **don't output anything else.**\n\n
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement: 
"""

if __name__ == "__main__":
    # get ground_truth and extracted_answer
    ground_truth_path = "MathVision_testmini.json"
    extracted_answer_path = "MathVision_extracted.jsonl"
    pk = "id"
    
    # load ground_truth
    id2q_gt = {} # (id: (question, ground_truth))
    # with open(ground_truth_path, "r", encoding="utf-8") as f:
    #     gt_lines = f.readlines()
    #     print(len(gt_lines))
    # note that ground_truth_path is a json file, not jsonl
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_lines = json.load(f)
        print(len(gt_lines)) # -> 304

    for gt_line in gt_lines:
        d = gt_line
        ans = d['answer'] if d['answer'] else d['answer_w_choices']
        print("gt: ", '\t', ans)
        problem = d['problem_w_choices'] if d['problem_w_choices'] else d['problem']
        id2q_gt[d[pk]] = (problem, ans)

    # load extracted_answer
    id2q_et = {} # (id: (question, extracted_answer))
    with open(extracted_answer_path, "r", encoding="utf-8") as f:
        ext_lines = f.readlines()
        print(len(ext_lines)) # -> 303?

    for ext_line in ext_lines:
        d = json.loads(ext_line)
        ans = d['extracted_answer']
        problem = d['problem_w_choices'] if d['problem_w_choices'] else d['problem']
        id2q_et[d[pk]] = (problem, ans)
    
    # safety check
    try:
        assert len(ext_lines) == len(gt_lines)
        assert len(id2q_et) == len(id2q_gt)
    except AssertionError as e:
        print(f"[ERROR] {e}")
        # 找出不匹配的 id
        missing_ids = set(id2q_gt.keys()) - set(id2q_et.keys())
        for idx in missing_ids:
            print(f"[ERROR] id {idx} in original data not in extracted answers")

    # link gt and ex based on id
    # link gt and ex based on id
    id_linked = {} # (id: (question, ground_truth, extracted_answer))
    for idx, tp in id2q_gt.items():
        q = tp[0]
        gt = tp[1]
        ea_tuple = id2q_et.get(idx, None)  # 这是一个元组 (question, extracted_answer)
        
        if ea_tuple is None:
            print(f"[ERROR] id {idx} in original data not in extracted answers")
            continue
        
        ea = ea_tuple[1] 
        id_linked[idx] = (q, gt, ea)  
    
    print("===="*50)
    for idx, value in id_linked.items():
        print(value[1], value[2])
    print("===="*50)
    kill

    # load llm
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=50 
    )

    # evaluate
    bz = 20 # batch_size
    all_ids = list(id_linked.keys())
    all_results = []
    
    with open(inference_output_file, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(all_ids), bz), desc="Evaluating batches"):
            batch_ids = all_ids[i:i+bz]
            batch_prompts = []
            
            # prepare batch prompts
            for idx in batch_ids:
                question, ground_truth, extracted_answer = id_linked[idx]
                prompt = judge_prompt.format(
                    question=question,
                    ground_truth=ground_truth,
                    predict_str=extracted_answer
                )
                
                # Format for Qwen chat template
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = llm.get_tokenizer().apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                batch_prompts.append(formatted_prompt)
            
            # generate judgments
            outputs = llm.generate(batch_prompts, sampling_params)
            
            # process results
            for j, output in enumerate(outputs):
                idx = batch_ids[j]
                question, ground_truth, extracted_answer = id_linked[idx]
                
                judgment_text = output.outputs[0].text.strip()
                
                # parse judgment
                judgment = 0
                if "1" in judgment_text:
                    print(judgment_text)
                    judgment = 1
                elif "0" in judgment_text:
                    judgment = 0
                else:
                    print(f"[WARNING] Unexpected judgment format for id {idx}: '{judgment_text}', defaulting to 0")
                
                # save result
                result = {
                    "id": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "extracted_answer": extracted_answer,
                    "judgment": judgment,
                    "judgment_text": judgment_text
                }
                
                all_results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    # calculate accuracy
    correct_count = sum(1 for result in all_results if result["judgment"] == 1)
    total_count = len(all_results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print(f"\nEvaluation Results:")
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # save accuracy to file
    with open(metric_file, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Correct: {correct_count}/{total_count}\n")
        f.write(f"Percentage: {accuracy*100:.2f}%\n")
    
    print(f"\nResults saved to:")
    print(f"- Detailed judgments: {inference_output_file}")
    print(f"- Accuracy metrics: {metric_file}")
    