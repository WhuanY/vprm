import json
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import os
from datetime import datetime

from utils import (
    load_jsonl, 
    load_json, 
    encode_image_to_base64, 
    LocalLLMClient
)

final_prompt = "Select the best answer to the above multiple-choice question based on the image. \
Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:',"

def load_dataset(
    raw_dataset, 
    processor, 
    modality, 
    system_prompt, 
    pre_prompt, 
    hdfs, 
    args
):
    inputs = []
    print(f"Loading {len(raw_dataset)} examples...")
    
    for idx, data in enumerate(tqdm(raw_dataset)):
        # 检查是否存在images字段
        has_images = "image" in data and data["image"] and args.has_images
        
        # 根据record内容自动判断问题类型和选择问题字段
        is_multiple_choice = False
        problem_key = "problem_w_choices"  # 默认使用problem
        
        if data['problem_w_choices'] != "":
            is_multiple_choice = True
            assert data['problem'] == ""
            problem_key = "problem_w_choices"
        
        # 获取原始问题
        if is_multiple_choice:
            problem = data["problem_w_choices"]
        else:
            problem = data["problem"]

        problem = problem + "\n" + final_prompt
        
        if args.pre_prompt:
            problem = args.pre_prompt + "\n" + problem

        if args.after_prompt:
            problem = problem + "\n" + args.after_prompt

        if not args.inference_api and args.model_name_or_path:
            # 但需要添加<image>标记用于多模态处理
            if '<image>' not in problem and has_images:
                problem = '<image>\n' + problem
            
            # 构建消息内容
            if has_images:
                # 处理包含图片的情况
                text_parts = problem.split("<image>")
                content = []
                
                for i in range(len(data["image"])):
                    if i < len(text_parts):
                        if text_parts[i].strip():
                            content.append({"type": "text", "text": text_parts[i].strip()})
                    content.append({"type": "image", "image": data["image"][i]})
                
                # 添加最后一段文本（如果存在）
                if len(text_parts) > len(data["image"]) and text_parts[-1].strip():
                    content.append({"type": "text", "text": text_parts[-1].strip()})
            else:
                # 处理纯文本情况
                content = [{"type": "text", "text": problem}]

            messages = [{"role": "user", "content": content}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            if has_images:
                image_data, _ = process_vision_info(messages)
                inputs.append({"prompt": prompt, "multi_modal_data": {modality: image_data}})
            else:
                inputs.append({"prompt": prompt})
        elif args.inference_api:
            content = []
            if has_images:
                for image_path in data['image']:
                    base64_uri = encode_image_to_base64(image_path)
                    if base64_uri:
                        content.append({"type": "image_url","image_url": {"url": base64_uri}})
            content.append({ "type": "text","text": problem})
            messages = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            inputs.append({"prompt": prompt, "messages":messages})

        else:
            raise ValueError("Either model_name_or_path or inference_api should be provided.")
    return inputs


def check_generated(args, raw_dataset):
    filtered_data = []
    keys_set = set()
    if os.path.exists(args.save_name):
        with open(args.save_name, 'r') as f:
            for _ in f.readlines():
                each_data = json.loads(_)
                keys_set.add(each_data[args.primary_key])
    
    for _ in raw_dataset:
        if _[args.primary_key] not in keys_set:
            filtered_data.append(_)
    print(f"After filtering generated examples: {len(filtered_data)} examples left...")
    return filtered_data

def main(args):
    if args.input_file.endswith('.jsonl'):
        raw_dataset = load_jsonl(args.input_file)[args.start : args.end]
    else:
        raw_dataset = load_json(args.input_file)[args.start : args.end]
    
    if os.path.exists(args.save_name):
        raw_dataset = check_generated(args, raw_dataset)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    inputs = load_dataset(
        raw_dataset,
        processor,
        args.modality,
        args.system_prompt,
        args.pre_prompt,
        args.hdfs,
        args,
    )
    
    if not args.inference_api:
        llm = LLM(
            model=args.model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            limit_mm_per_prompt={"image": 10, "video": 2},
            gpu_memory_utilization=0.9,
            # enforce_eager=True,
            # mm_processor_kwargs={
            #     "min_pixels": 28 * 28,
            #     "max_pixels": 1024 * 1024,
            # },
        )
    elif args.inference_api:
        llm = LocalLLMClient(
            model = args.model_name_or_path,
            inference_api = args.inference_api,
        )
    else:
        raise ValueError("Either model_name_or_path or inference_api should be provided.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
        + tokenizer.additional_special_tokens_ids,
    )

    if not args.bz:
        bz = len(raw_dataset)
    else:
        bz = args.bz

    for idx in tqdm(range(0, len(inputs), bz), 
                desc="Inferencing", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"):
        batch_inputs = inputs[idx : idx + bz]
        outputs = llm.generate(batch_inputs, sampling_params)
        with open(args.save_name, "a", encoding="utf-8") as f:
            for i in range(len(outputs)):
                original_idx = idx + i
                new_dict = {
                    "id": raw_dataset[original_idx]["id"],
                    "problem": raw_dataset[original_idx].get("problem", ""),
                    "problem_w_choices": raw_dataset[original_idx].get("problem_w_choices",""),
                    "answer": raw_dataset[original_idx].get("answer", ""),  
                    "answer_w_choices": raw_dataset[original_idx].get("answer_w_choices", ""),
                    "prompt": inputs[original_idx]["prompt"],
                }
                if not args.inference_api and args.model_name_or_path:
                    new_dict["response"] = [
                        outputs[i].outputs[j].text
                        for j in range(len(outputs[i].outputs))
                    ]
                elif args.inference_api:
                    new_dict["response"] = [
                        outputs[i].choices[j].message.content
                        for j in range(len(outputs[i].choices))
                    ]
                else:  
                    raise ValueError("Either model_name_or_path or inference_api should be provided.")

                # 把raw_dataset中的字段加入到new_dict中
                for key in raw_dataset[original_idx]:
                    if key not in new_dict:
                        new_dict[key] = raw_dataset[original_idx][key]
                f.write(json.dumps(new_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="MathVista evaluation script with Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--inference_api", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, 
                    default="/home/minyingqian/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--input_file", type=str, default="MATH-V_testmini.json")
    parser.add_argument("--save_name", type=str, default="MATH-V_testmini_output.jsonl")
    parser.add_argument("--modality", type=str, default="image")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--pre_prompt", type=str, default="")
    parser.add_argument("--after_prompt", type=str, default="")
    parser.add_argument("--hdfs", type=int, default=0)
    parser.add_argument("--bz", type=int, default=20)
    parser.add_argument("--has_images", type=int, default=1)
    parser.add_argument("--primary_key", type=str, default="id")

    args = parser.parse_args()
    
    main(args)
    print("Inference script finished. output saved to ", args.save_name)
