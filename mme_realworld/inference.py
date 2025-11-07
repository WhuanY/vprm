import json
import os

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    load_jsonl,
    load_json,
    encode_image_to_base64,
    LocalLLMClient,
)

def _normalize_image_field(image_field):
    """Normalize image field from dataset into a list of cleaned image paths."""
    if not image_field:
        return []
    if isinstance(image_field, list):
        iterable = image_field
    else:
        iterable = [image_field]

    normalized = []
    for item in iterable:
        if not item:
            continue
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
        else:
            normalized.append(item)
    return normalized


def convert_record_to_internal_format(record, idx, args):
    """Convert original dataset record to the internal format expected by preprocess helper."""
    raw_text = record.get("Text")
    if raw_text is None:
        raw_text = record.get("question")
    if raw_text is None:
        raw_text = record.get("Problem")
    if raw_text is None:
        raw_text = ""
    text = str(raw_text).strip()

    choices = record.get("Answer choices") or record.get("answer_choices") or []

    if choices:
        formatted_choices = "\n".join(str(choice).strip() for choice in choices if str(choice).strip())
        problem_w_choices = f"{text}\n{formatted_choices}".strip()
    else:
        problem_w_choices = ""

    image_candidates = (
        record.get("Image")
        or record.get("Images")
        or record.get("image")
        or record.get("images")
    )
    normalized_images = _normalize_image_field(image_candidates)
    resolved_images = []
    active_root = getattr(args, "image_root", "") or getattr(args, "image_base_dir", "")
    for image_path in normalized_images:
        if isinstance(image_path, str) and active_root and not os.path.isabs(image_path):
            resolved_images.append(os.path.join(active_root, image_path))
        else:
            resolved_images.append(image_path)

    internal = {
        "id": record.get("Question_id", str(idx)),
        "problem": text,
        "problem_w_choices": problem_w_choices,
        "answer": record.get("Ground truth", record.get("answer", "")),
        "answer_w_choices": choices,
        "image": resolved_images,
    }
    return internal

def _process_single_item_helper(data_with_args):
    """处理单个数据项的辅助函数"""
    data_with_idx, processor, modality, system_prompt, _unused_pre_prompt, args = data_with_args
    idx, data = data_with_idx

    raw_images = data.get("image") or []
    resolved_images = []
    for image_path in raw_images:
        if not image_path:
            continue
        if isinstance(image_path, str):
            resolved_images.append(image_path.strip())
        else:
            resolved_images.append(image_path)

    has_images = bool(resolved_images) and bool(args.has_images)

    is_multiple_choice = bool(data.get("problem_w_choices"))
    base_problem = data["problem_w_choices"].strip() if is_multiple_choice else data.get("problem", "").strip()

    text_segments = []
    if args.pre_prompt:
        text_segments.append(args.pre_prompt.strip())
    if base_problem:
        text_segments.append(base_problem)
    if args.after_prompt:
        text_segments.append(args.after_prompt.strip())

    problem = "\n\n".join(segment for segment in text_segments if segment)
    if not problem:
        problem = base_problem

    result = {}

    if not args.inference_api and args.model_name_or_path:
        if "<image>" not in problem and has_images:
            problem = "<image>\n" + problem

        if has_images:
            text_parts = problem.split("<image>")
            content = []

            for i, image_path in enumerate(resolved_images):
                if i < len(text_parts):
                    text_segment = text_parts[i].strip()
                    if text_segment:
                        content.append({"type": "text", "text": text_segment})
                content.append({"type": "image", "image": image_path})

            if len(text_parts) > len(resolved_images):
                trailing_text = text_parts[-1].strip()
                if trailing_text:
                    content.append({"type": "text", "text": trailing_text})
        else:
            content = [{"type": "text", "text": problem}]

        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if has_images:
            image_data, _ = process_vision_info(messages)
            result = {"prompt": prompt, "multi_modal_data": {modality: image_data}}
        else:
            result = {"prompt": prompt}
    elif args.inference_api and args.model_name_or_path:
        content = []
        if has_images:
            for image_path in resolved_images:
                base64_uri = encode_image_to_base64(image_path)
                if base64_uri:
                    content.append({"type": "image_url", "image_url": {"url": base64_uri}})
        content.append({"type": "text", "text": problem})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        result = {"prompt": prompt, "messages": messages}
    else:
        raise ValueError("You should provide model_path. inference_api is optional")

    return idx, result

def load_dataset(
    raw_dataset, 
    processor, 
    modality, 
    system_prompt, 
    pre_prompt, 
    hdfs, 
    args
):
    from multiprocessing import Pool, cpu_count
    import tqdm

    print(f"Loading {len(raw_dataset)} examples...")

    # 准备数据：为每个数据项添加索引
    indexed_data = list(enumerate(raw_dataset))

    # 准备传递给每个进程的参数
    process_args = [
        (data_item, processor, modality, system_prompt, pre_prompt, args)
        for data_item in indexed_data
    ]

    # 确定进程数量
    num_processes = min(cpu_count(), len(raw_dataset), 100)
    print(f"Using {num_processes} processes for parallel processing...")

    # 使用多进程处理
    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(_process_single_item_helper, process_args),
            total=len(process_args),
            desc="Processing items"
        ))

    # 按原始顺序排序结果
    results.sort(key=lambda x: x[0])

    # 提取处理后的数据
    inputs = [result[1] for result in results]
    return inputs


def check_generated(args, raw_dataset):
    filtered_data = []
    keys_set = set()
    if os.path.exists(args.save_name):
        with open(args.save_name, 'r') as f:
            for _ in f.readlines():
                each_data = json.loads(_)
                key_val = each_data.get(args.primary_key)
                if key_val is not None:
                    keys_set.add(key_val)
    
    for idx, _ in enumerate(raw_dataset):
        key_val = _.get(args.primary_key)
        fallback_key = key_val if key_val is not None else str(idx)
        if fallback_key not in keys_set:
            filtered_data.append(_)
    print(f"After filtering generated examples: {len(filtered_data)} examples left...")
    return filtered_data

def main(args):
    if args.input_file.endswith('.jsonl'):
        original_dataset = load_jsonl(args.input_file)
    else:
        original_dataset = load_json(args.input_file)

    slice_end = None if args.end == -1 else args.end
    original_dataset = original_dataset[args.start : slice_end]

    if os.path.exists(args.save_name):
        original_dataset = check_generated(args, original_dataset)

    processed_dataset = [
        convert_record_to_internal_format(record, idx, args)
        for idx, record in enumerate(original_dataset)
    ]

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    inputs = load_dataset(
        processed_dataset,
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
            gpu_memory_utilization=0.7,
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
        bz = len(processed_dataset)
    else:
        bz = args.bz

    for idx in tqdm(
        range(0, len(inputs), bz),
        desc="Inferencing",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]",
    ):
        batch_inputs = inputs[idx : idx + bz]
        outputs = llm.generate(batch_inputs, sampling_params)
        with open(args.save_name, "a", encoding="utf-8") as f:
            for i in range(len(outputs)):
                original_idx = idx + i
                if not args.inference_api and args.model_name_or_path:
                    response_payload = [
                        outputs[i].outputs[j].text
                        for j in range(len(outputs[i].outputs))
                    ]
                elif args.inference_api:
                    response_payload = [
                        outputs[i].choices[j].message.content
                        for j in range(len(outputs[i].choices))
                    ]
                else:
                    raise ValueError("Either model_name_or_path or inference_api should be provided.")

                first_response = ""
                if isinstance(response_payload, list) and response_payload:
                    first_response = response_payload[0]
                elif isinstance(response_payload, str):
                    first_response = response_payload

                if isinstance(first_response, str):
                    first_response = first_response.strip()

                output_record = dict(original_dataset[original_idx])
                output_record["response"] = first_response
                output_record["prompt"] = inputs[original_idx]["prompt"]

                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="MME-RealWorld inference script with Qwen2.5-VL"
    )
    parser.add_argument("--inference_api", type=str, default="")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/minyingqian/models/Qwen2.5-VL-7B-Instruct",
    )
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
    parser.add_argument("--primary_key", type=str, default="Question_id")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--image_base_dir", type=str, default="")

    args = parser.parse_args()

    if not args.image_root and args.image_base_dir:
        args.image_root = args.image_base_dir

    main(args)
    print("Inference script finished. Output saved to", args.save_name)