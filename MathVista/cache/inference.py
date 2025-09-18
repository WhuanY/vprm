import json
from typing import List
import base64
from openai import OpenAI
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import tqdm
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import os
from datetime import datetime

class LocalServerClient:
    def __init__(self, model_path, api_url="http://localhost:9753/v1"):
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require real API key
            base_url=api_url
        )
        self.model_path = model_path

    def generate(self, batch_inputs:List[dict], sampling_param:SamplingParams):
        """
        do batch inference by submitting batch request to local vLLM server
        """
        outputs = []
        for inp in batch_inputs:
            # print(inp)
            # print("*"*10)
            chat_completion_from_base64 = self.client.chat.completions.create(
                messages=inp,
                model = self.model_path,
                max_tokens=sampling_param.max_tokens,
                temperature=sampling_param.temperature,
            )
        outputs.append(chat_completion_from_base64)
        return outputs
            

def encode_base64_content_from_file(img_path):
    """
    Encode image file to base64 string
    """
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_dataset_openai_fmt(
    raw_dataset,
    system_prompt,
    pre_prompt,
    args
):
    inputs = []
    print(f"Loading {len(raw_dataset)} examples...")
    
    for idx, data in enumerate(tqdm.tqdm(raw_dataset)):
        # 检查是否存在images字段
        has_images = "image" in data and data["image"] and args.has_images
        
        # 根据record内容自动判断问题类型和选择问题字段
        is_multiple_choice = False
        
        if data['problem_w_choices'] != "":
            is_multiple_choice = True
            assert data['problem'] == ""
        # 获取原始问题
        if is_multiple_choice:
            problem = data["problem_w_choices"]
        else:
            problem = data["problem"]
        
        # print("problem with hint: ", problem)
        
        if args.pre_prompt:
            problem = args.pre_prompt + "\n" + problem

        if args.after_prompt:
            problem = problem + "\n" + args.after_prompt

        msg = [
            {
                "role": "user",
                "content": [
                    {"type":"text", "text": problem},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_base64_content_from_file(data['image'][0])}"
                        } 
                    }
                ]
            }
        ]
        inputs.append(msg)
        # # save msg to json 
        # with open(f"debug_msg_{idx}.json", "w") as f:
        #     json.dump(inputs, f, indent=4)
    return inputs

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
    
    for idx, data in enumerate(tqdm.tqdm(raw_dataset)):
        # 检查是否存在images字段
        has_images = "image" in data and data["image"] and args.has_images
        
        # 根据record内容自动判断问题类型和选择问题字段
        is_multiple_choice = False
        problem_key = "problem"  # 默认使用problem
        
        if data['problem_w_choices'] != "":
            is_multiple_choice = True
            assert data['problem'] == ""
            problem_key = "problem_w_choices"

        
        # 获取原始问题
        if is_multiple_choice:
            problem = data["problem_w_choices"]
        else:
            problem = data["problem"]
        
        print("problem with hint: ", problem)
        
        if args.pre_prompt:
            problem = args.pre_prompt + "\n" + problem

        if args.after_prompt:
            problem = problem + "\n" + args.after_prompt

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

        # 构建消息
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 详细记录构建的messages结构
        # print(messages)
        
        # 注释掉模型相关的处理，但保留结构
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if has_images:
            image_data, _ = process_vision_info(messages)
            inputs.append(
                {"prompt": prompt, "multi_modal_data": {modality: image_data}}
            )
        else:
            inputs.append({"prompt": prompt})
        
        # 保存messages用于后续处理
        # input_item = {
        #     "messages": messages,
        #     "question_type": "multiple_choice" if is_multiple_choice else "numerical",
        #     "problem_key_used": problem_key,
        #     "data_id": data.get('id', 'unknown')
        # }
        
        # inputs.append(input_item)
        # print(f"添加到inputs列表，当前总数: {len(inputs)}")
        

    # 统计问题类型分布
    mc_count = sum(1 for inp in inputs if inp.get("question_type") == "multiple_choice")
    num_count = len(inputs) - mc_count

    # print("----inputs----"*40)
    # print(inputs[0])
    
    return inputs



def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for _ in f.readlines():
            data.append(json.loads(_))
    return data


def load_json(path):
    """
    加载json格式的数据文件
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


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
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    print(f"开始处理MathVista数据集: {args.input_file}")
    (f"是否包含图片: {bool(args.has_images)}")
    
    # 根据文件扩展名选择加载方式
    if args.input_file.endswith('.jsonl'):
        raw_dataset = load_jsonl(args.input_file)[args.start : args.end]
    else:
        raw_dataset = load_json(args.input_file)[args.start : args.end]
    
    print(f"成功加载 {len(raw_dataset)} 条原始数据")
    
    # 显示数据集样本结构
    if raw_dataset:
        print("数据集字段结构:")
        sample = raw_dataset[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {type(value).__name__} (长度: {len(value)})")
            else:
                print(f"  {key}: {value}")
    
    if os.path.exists(args.save_name):
        raw_dataset = check_generated(args, raw_dataset)
    
    if args.inference_api: # 兼容OpenAI客户端的接口
        inputs = load_dataset_openai_fmt(
            raw_dataset,
            args.system_prompt,
            args.pre_prompt,
            args
        )

    else: # 使用本地加载的数据集
        inputs = load_dataset(
            raw_dataset,
            processor,
            args.modality,
            args.system_prompt,
            args.pre_prompt,
            args.hdfs,
            args,
        )
    
    # 注释掉模型推理部分，但保留结构
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
    else: # 使用API方式
        llm = LocalServerClient(model_path=args.model_name_or_path, api_url=args.inference_api)

    if not args.bz:
        bz = len(raw_dataset)
    else:
        bz = args.bz

    unfinish_cnt = 0 # 统计未生成完成的样本数量（通常是超过max_new_tokens)
    for idx in range(0, len(inputs), bz):
        batch_inputs = inputs[idx : idx + bz]
        outputs = llm.generate(batch_inputs, sampling_param=sampling_params)

        with open(args.save_name, "a", encoding="utf-8") as f:
            for i in range(len(outputs)):
                unfinished = False
                original_idx = idx + i

                # 检查是否因为max_tokens而未完成
                if any(output.finish_reason == 'length' for output in outputs[i].outputs):
                    unfinished = True
                    unfinish_cnt += 1
                new_dict = {
                    "id": raw_dataset[original_idx]["id"],
                    "problem": raw_dataset[original_idx].get("problem", ""),
                    "problem_w_choices": raw_dataset[original_idx].get("problem_w_choices",""),
                    "answer": raw_dataset[original_idx].get("answer", ""),  
                    "answer_w_choices": raw_dataset[original_idx].get("answer_w_choices", ""),
                    "response": [
                        outputs[i].outputs[j].text
                        for j in range(len(outputs[i].outputs))
                    ],
                    "prompt": inputs[original_idx]["prompt"],
                    "unfinished": unfinished
                }
                # 把raw_dataset中的字段加入到new_dict中
                for key in raw_dataset[original_idx]:
                    if key not in new_dict:
                        new_dict[key] = raw_dataset[original_idx][key]
                f.write(json.dumps(new_dict, ensure_ascii=False) + "\n")

    print(f"Inference Finished. Saved to:  {args.save_name}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="MathVista evaluation script with Qwen2.5-VL-7B-Instruct"
    )
    
    parser.add_argument("--model_name_or_path", type=str, 
                    default="/home/minyingqian/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--inference_api", type=str, default="") # 如果为空，则使用本地模型推理
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
    parser.add_argument("--after_prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.")
    parser.add_argument("--hdfs", type=int, default=0)
    parser.add_argument("--bz", type=int, default=20)
    parser.add_argument("--has_images", type=int, default=1)
    parser.add_argument("--primary_key", type=str, default="id")

    args = parser.parse_args()
    
    print("MathVista 数据集评测脚本")
    print("=" * 50)
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.save_name}")
    print("问题类型: 自动判断（多选题/数值题）")
    
    main(args)
    print("Inference script finished.")