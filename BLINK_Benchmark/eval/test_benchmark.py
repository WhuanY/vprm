import json
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image
import os
import re
from multiple_choice import match_multiple_choice
import argparse
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib

import pandas as pd
import io

def load_relative_reflectance_data(data_files):
    """
    Load Relative_Reflectance dataset from parquet files using pandas,
    bypassing HuggingFace datasets library to avoid schema compatibility issues.
    
    Parameters:
    - data_files: dict with 'val' and 'test' keys pointing to parquet file paths
    
    Returns:
    - dict with 'val' and 'test' keys, each containing a list of data records
    """
    def convert_image_field(image_data):
        """Convert image data to PIL Image object"""
        if image_data is None:
            return None
        if isinstance(image_data, Image.Image):
            # Already a PIL Image
            return image_data
        elif isinstance(image_data, bytes):
            # Raw bytes, convert to PIL Image
            try:
                return Image.open(io.BytesIO(image_data))
            except Exception as e:
                print(f"Warning: Failed to convert image bytes to PIL Image: {e}")
                return None
        elif isinstance(image_data, dict):
            # Dictionary format, check for 'bytes' key
            if 'bytes' in image_data:
                try:
                    return Image.open(io.BytesIO(image_data['bytes']))
                except Exception as e:
                    print(f"Warning: Failed to convert image dict to PIL Image: {e}")
                    return None
            # Might have other formats, try to extract image
            return None
        else:
            # Unknown format
            return None
    
    def load_split(file_path):
        """Load a single split from parquet file"""
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, returning empty list")
            return []
        
        try:
            # Read parquet file using pandas
            df = pd.read_parquet(file_path)
            
            # Convert to list of dictionaries
            records = []
            for idx, row in df.iterrows():
                record = row.to_dict()
                
                # Convert image fields to PIL Image objects
                for img_key in ['image_1', 'image_2', 'image_3', 'image_4']:
                    if img_key in record:
                        record[img_key] = convert_image_field(record[img_key])
                
                records.append(record)
            
            return records
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    result = {
        "val": load_split(data_files.get('val', '')),
        "test": load_split(data_files.get('test', ''))
    }
    
    # print(f"Loaded {len(result['val'])} val records and {len(result['test'])} test records")
    
    return result

disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"

def analyze_answer(d, gpt_answer, all_choices):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - d : data, the data containing the question and choices.
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    # if "<answer>" in gpt_answer and "</answer>" in gpt_answer: # use cot
    #     gpt_answer = gpt_answer.split("<answer>")[1].split("</answer>")[0].strip() 
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', gpt_answer, re.DOTALL)
    if answer_match:
        # 判断是否是单个字母答案, 如果是，返回该字母的upper形式
        gpt_answer = answer_match.group(1).strip()
    try:
        intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
        intersect_last = list(set(all_choices).intersection(set(gpt_answer.split('\n\n')[-1].split())))
        if gpt_answer in ["A", "B", "C", "D", "E"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            prediction = gpt_answer
        elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
            choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
            options = '\n'.join([f'{choices[i]} {d["choices"][i]}' for i in range(len(d['choices']))])
            extracted_answer = match_multiple_choice(f"{d['question']}\nSelect from the following choices", options, gpt_answer)
            prediction = extracted_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                gpt_answer = gpt_answer.split('\n\n')[-1]
            prediction = intersect[0]
        return prediction
    except Exception as e:
        print(f"Error in analyze_answer: {gpt_answer}, {e}")
        return None


def _process_single_item_helper(data_with_args):
    """
    Process a single data item to prepare it for batch inference.
    
    Parameters:
    - data_with_args: tuple of (idx, data, task_name, image_folder, processor, args)
    
    Returns:
    - tuple of (idx, result_dict) where result_dict contains the processed input for VLLM
    """
    idx, data, task_name, image_folder, processor, args = data_with_args
    
    # Load prompt and images
    image_paths, prompt = load_prompt(task_name, data, image_folder, args)
    
    has_images = image_paths is not None and len(image_paths) > 0
    
    # Build messages following MME-RealWorld-Lite pattern
    if has_images:
        # Load images as PIL Image objects from paths
        images = [Image.open(img_path) for img_path in image_paths]
        
        # Add <image> markers to prompt (required for process_vision_info)
        if '<image>' not in prompt:
            image_markers = '\n'.join(['<image>'] * len(images))
            prompt_with_markers = image_markers + '\n' + prompt
        else:
            prompt_with_markers = prompt
        
        # Split by <image> to interleave text and images
        text_parts = prompt_with_markers.split("<image>")
        content = []
        
        # Build content: images first, then text (aligned with VLMEvalKit)
        for i in range(len(images)):
            if i < len(text_parts) and text_parts[i].strip():
                content.append({"type": "text", "text": text_parts[i].strip()})
            content.append({"type": "image", "image": images[i]})
        
        # Add remaining text after all images
        if len(text_parts) > len(images) and text_parts[-1].strip():
            content.append({"type": "text", "text": text_parts[-1].strip()})
    else:
        # Pure text case
        content = [{"type": "text", "text": prompt}]

    messages = [{"role": "user", "content": content}]
    if hasattr(args, 'system_prompt') and args.system_prompt:
        messages.insert(0, {"role": "system", "content": args.system_prompt})
    
    # Process with processor
    processed_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Prepare input for VLLM
    if has_images:
        image_data, _ = process_vision_info(messages)
        result = {"prompt": processed_prompt, "multi_modal_data": {"image": image_data}}
    else:
        result = {"prompt": processed_prompt}
    
    # Store original data for later use
    result["_original_data"] = data
    result["_image_paths"] = image_paths
    result["_original_prompt"] = prompt
    
    return idx, result


def query_model(task_name, args=None):
    """
    Loads the dataset, processes items in batches, and performs batch inference.
    Similar to MME-RealWorld-Lite's implementation.
    
    Parameters:
    - task_name: String, the name of the task to evaluate.
    - args: command line arguments including model name

    Returns:
    - outputs, The result is also saved to 'output_filename.json'.
    """
    eval_split = args.eval_split
    dataset_local_path = args.dataset_local_path
    output_save_folder = getattr(args, 'output_save_folder', 'outputs')
    image_save_folder = getattr(args, 'image_save_folder', 'images')
    batch_size = getattr(args, 'batch_size', 20)
    
    output_path = f'{output_save_folder}/{task_name.replace("_", " ")}.json'
    os.makedirs(output_save_folder, exist_ok=True)
    image_folder = f'{image_save_folder}/{task_name}_images'
    os.makedirs(image_folder, exist_ok=True)
    inferenced_idx = set()
    eval_splits = ['val'] if eval_split == 'val' else ['test'] if eval_split == 'test' else ['val', 'test']
    
    # If regen is True, clear existing outputs and start fresh
    if args.regen:
        outputs = {'val': [], 'test': []}
        print(f"Regenerating {task_name} on {eval_splits} set")
    elif not os.path.exists(output_path):
        outputs = {'val': [], 'test': []}
        print(f"Evaluating {task_name} on {eval_splits} set")
    else:
        print(f"Loading outputs from {output_path}")
        outputs = json.load(open(output_path, 'r'))
        inferenced_idx = set([d['idx'] for d in outputs[eval_split]]) # already inferenced idx
    
    # Use shared processor, llm, and sampling_params from args (initialized in main)
    # If not initialized, initialize them (for backward compatibility)
    if not hasattr(args, '_processor') or args._processor is None:
        print(f"Initializing processor, tokenizer, and LLM for {task_name}...")
        args._processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        args._tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        args._llm = init_vllm(args)
        args._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8000,
            stop_token_ids=[args._tokenizer.eos_token_id] + args._tokenizer.additional_special_tokens_ids,
        )
    
    processor = args._processor
    tokenizer = args._tokenizer
    llm = args._llm
    sampling_params = args._sampling_params
    
    for split in eval_splits:
        data_files = {
            'val': f'{dataset_local_path}/{task_name}/val-00000-of-00001.parquet',
            'test': f'{dataset_local_path}/{task_name}/test-00000-of-00001.parquet'
        }
        
        if task_name == 'Relative_Reflectance':
            test_data = load_relative_reflectance_data(data_files)[split]
        else:
            test_data = load_dataset('parquet', data_files=data_files)[split]
        
        # Filter out already processed items
        if args.regen:
            items_to_process = [orig_d for orig_d in test_data]
        else:
            items_to_process = [orig_d for orig_d in test_data if orig_d['idx'] not in inferenced_idx]
        
        if len(items_to_process) == 0:
            print(f"No items to process for {task_name} {split}")
            continue
        
        print(f"Processing {len(items_to_process)} items for {task_name} {split}...")
        
        # Prepare data for batch processing (serial processing to avoid pickle issues with processor)
        print("Preparing inputs (serial processing)...")
        prepared_results = []
        for idx, data in enumerate(tqdm(items_to_process, desc="Preparing inputs")):
            result = _process_single_item_helper((idx, data, task_name, image_folder, processor, args))
            prepared_results.append(result)
        
        # Extract inputs and metadata
        inputs = [result[1] for result in prepared_results]
        original_data_list = [result[1]["_original_data"] for result in prepared_results]
        image_paths_list = [result[1]["_image_paths"] for result in prepared_results]
        original_prompts = [result[1]["_original_prompt"] for result in prepared_results]
        
        # Remove metadata from inputs before passing to VLLM
        for inp in inputs:
            inp.pop("_original_data", None)
            inp.pop("_image_paths", None)
            inp.pop("_original_prompt", None)
        
        # Batch inference
        print(f"Running batch inference with batch_size={batch_size}...")
        all_responses = []
        for idx in tqdm(range(0, len(inputs), batch_size), 
                       desc="Inferencing",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"):
            batch_inputs = inputs[idx : idx + batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params)
            
            # Extract responses
            for i in range(len(batch_outputs)):
                response_text = batch_outputs[i].outputs[0].text
                all_responses.append(response_text)
        
        # Post-process: judge answers (using multithreading)
        print(f"Judging answers with 50 threads...")
        
        def judge_single_item(item_data):
            """Judge a single item"""
            orig_d, gpt_answer, prompt = item_data
            idx = orig_d['idx']
            gold_answer = orig_d['answer']
            all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(orig_d['choices'])]
            
            prediction = analyze_answer(orig_d, gpt_answer, all_choices)
            
            result = {
                'idx': idx,
                'answer': gold_answer,
                'full_prediction': gpt_answer,
                'prediction': prediction,
                'prompt': prompt
            }
            return result
        
        # Prepare data for threading
        judge_data = list(zip(original_data_list, all_responses, original_prompts))
        
        # Use ThreadPoolExecutor with 50 threads
        results_list = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_item = {
                executor.submit(judge_single_item, item_data): item_data
                for item_data in judge_data
            }
            
            for future in tqdm(as_completed(future_to_item), total=len(judge_data), desc="Judging"):
                try:
                    result = future.result()
                    results_list.append(result)
                except Exception as e:
                    item_data = future_to_item[future]
                    print(f"Error judging item {item_data[0].get('idx', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Sort results by idx to maintain order
        results_list.sort(key=lambda x: x['idx'])
        
        # Add to outputs
        for result in results_list:
            outputs[split].append(result)
        
        # Save after processing each split
        json.dump(outputs, open(output_path, 'w'), indent=4)
        print(f"Saved results to {output_path}")
        
    return outputs


def rescale_img(img, tgt=None):
    """
    Rescale image to target size while maintaining aspect ratio.
    
    Parameters:
    - img: PIL Image object
    - tgt: Tuple like (-1, 512) meaning height=512, width auto, or (512, -1) meaning width=512, height auto
    
    Returns:
    - Rescaled PIL Image
    """
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def concat_images_vlmeval(image_list, target_size=512, mode='h', save_path=None):
    """
    Concatenate multiple PIL Images horizontally (or vertically) with optional resizing.
    Aligned with VLMEvalKit's implementation for BLINK benchmark.
    
    Parameters:
    - image_list: List of PIL Image objects
    - target_size: Target size for resizing (height if mode='h', width if mode='v'). Use -1 to skip resizing.
    - mode: 'h' for horizontal concatenation, 'v' for vertical
    - save_path: Path to save the concatenated image. If None, generates a temporary path.
    
    Returns:
    - Path to the saved concatenated image
    """
    if not image_list:
        return None
    
    # Resize images if target_size is specified
    ims = image_list.copy()
    if target_size != -1:
        ims = [
            rescale_img(im, (-1, target_size) if mode == 'h' else (target_size, -1))
            for im in ims
        ]
    
    ws, hs = [x.width for x in ims], [x.height for x in ims]
    
    if mode == 'h':
        new_w, new_h = sum(ws), max(hs)
        dst = Image.new('RGB', (new_w, new_h))
        x_offset = 0
        for i, im in enumerate(ims):
            dst.paste(im, (x_offset, 0))
            x_offset += im.width
    elif mode == 'v':
        new_w, new_h = max(ws), sum(hs)
        dst = Image.new('RGB', (new_w, new_h))
        y_offset = 0
        for i, im in enumerate(ims):
            dst.paste(im, (0, y_offset))
            y_offset += im.height
    
    # Generate save path if not provided
    if save_path is None:
        # Generate MD5 hash from image paths for uniqueness
        image_str = '_'.join([str(id(img)) for img in image_list])
        str_md5 = hashlib.md5(image_str.encode()).hexdigest()
        save_path = os.path.join('/tmp', f'{str_md5}.jpg')
    
    dst.save(save_path)
    return save_path


def concat_images_horizontally_with_margin(image_filenames, output_filename, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - output_filename: String, the filename to save the concatenated image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - None. The result is saved to 'output_filename'.
    """
    images = [Image.open(filename) for filename in image_filenames]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    new_image.save(output_filename)  # Save the result


def load_prompt(task_name, d, image_folder, args):
    """
    Loads the prompt and images from huggingface data entry, saves the images to a folder, and returns a list of image paths, and the prompt.
    
    For BLINK benchmark, aligns with VLMEvalKit preprocessing:
    - Multiple images are concatenated horizontally
    - Images are resized to height 512 (maintaining aspect ratio)
    - Returns a single concatenated image path

    Parameters:
    - task_name: String, the name of the task.
    - d: data entry, the data dictionary containing the prompt and images.
    - image_folder: String, the folder to save the images.
    - args: command line arguments. 

    Returns:
    - image_paths: List of strings, the filepaths to the saved images (single concatenated image for BLINK).
    - prompt: String, the prompt text.
    """
    # Collect all images
    images = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in d and d[k]:
            images.append(d[k])
    
    if not images:
        image_paths = []
    elif len(images) > 1:
        # For BLINK: concatenate multiple images (aligned with VLMEvalKit)
        # Save concatenated image to image_folder
        concat_path = os.path.join(image_folder, f'{d["idx"]}_concat.jpg')
        concat_images_vlmeval(images, target_size=512, mode='h', save_path=concat_path)
        image_paths = [concat_path]
    else:
        # Single image: save normally
        image = images[0]
        image_path = f'{image_folder}/{d["idx"]}_1.jpg'
        image.save(image_path)
        image_paths = [image_path]


    # process prompt
    if 'question' in d.keys() and 'choices' in d.keys(): # VLMEvalKit format
        question = d['question']
        options = {}
        for i, choice in enumerate(d.get('choices', [])):
            if choice:  # 只添加非空的选项
                options[chr(65 + i)] = choice  # A, B, C, D, E
        
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        
        hint = d.get('hint', None)
        prompt = ''
        if hint is not None and not pd.isna(hint):
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
    else:
        prompt = d['prompt']
    
    if args.pre_prompt:
        prompt = args.pre_prompt + "\n" + prompt
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    # Note: removed 'blip' check as it's no longer needed
    if args.after_prompt:
        prompt = prompt + "\n" + args.after_prompt
    
    return image_paths, prompt


def normalize_answer(answer):
    """
    Normalize answer format to ensure consistent comparison.
    Converts "A" -> "(A)", "(A)" -> "(A)", etc.
    
    Parameters:
    - answer: String, the answer in various formats (e.g., "A", "(A)")
    
    Returns:
    - String, normalized answer in format "(A)" or original if not a letter
    """
    if not answer:
        return answer
    
    answer = str(answer).strip()
    
    # If it's already in format "(A)", return as is
    if answer.startswith('(') and answer.endswith(')'):
        return answer
    
    # If it's a single letter like "A", "B", "C", convert to "(A)", "(B)", "(C)"
    if len(answer) == 1 and answer.isalpha() and answer.upper() in ['A', 'B', 'C', 'D', 'E']:
        return f"({answer.upper()})"
    
    # Otherwise return original
    return answer


def init_vllm(args):
    """
    Initializes the VLLM model.
    """
    from vllm import LLM, SamplingParams

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
    return llm


def eval_task(task_name, args):
    outputs = query_model(task_name, args)
    accu = {'val': 0, 'test': 0}
    for split in ['val', 'test']:
        for d in outputs[split]:
            # Normalize both answer and prediction for consistent comparison
            normalized_answer = normalize_answer(d['answer'])
            normalized_prediction = normalize_answer(d['prediction'])
            if normalized_answer == normalized_prediction:
                accu[split] += 1
    
    print('-'*50)
    print(f'Task {task_name} Performance')
    for split in ['val']: # 脚本默认只评 val set的
        print(f'{split} accuracy: {round(accu[split]/len(outputs[split])*100, 2)}%')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="path to the model")
    parser.add_argument("--dataset_local_path", type=str, default='BLINK_fromhf', help="the local path to the dataset downloaded from huggingface")
    parser.add_argument("--task_name", type=str, default='Relative_Depth', help="select the task name")
    parser.add_argument("--eval_split", type=str, default='val', help="select the eval split")
    parser.add_argument("--pre_prompt", type=str, default='', help="the pre-prompt for the model")
    parser.add_argument("--after_prompt", type=str, default='', help="the after-prompt for the model")
    parser.add_argument("--output_save_folder", type=str, default='outputs', help="directory to save output JSON files")
    parser.add_argument("--image_save_folder", type=str, default='images', help="directory to save processed images")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size for inference (default: 20)")
    parser.add_argument("--system_prompt", type=str, default='', help="the system prompt for the model")
    parser.add_argument("--tp", type=int, default=1, help="the tensor parallel size for the model")
    parser.add_argument("--regen", action='store_true', help="regenerate the outputs")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name_or_path # model_path: 模型参数路径
    model_name = os.path.basename(model_path) # model_name: ckpt文件名
    print(f'Using model: {model_path}, model name: {model_name}')
    print(f'Batch size: {args.batch_size}')
    
    # Initialize processor, tokenizer, llm, and sampling_params once at program start
    # All tasks will share these instances
    print("Initializing LLM engine (shared across all tasks)...")
    args._processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    args._tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    args._llm = init_vllm(args)
    args._sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8000,
        stop_token_ids=[args._tokenizer.eos_token_id] + args._tokenizer.additional_special_tokens_ids,
    )
    print("LLM engine initialized successfully!")
    
    # need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']
    need_disclaimer_tasks = []
    if args.task_name == 'all': 
        subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 
                    'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test',  
                'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        eval_task(task_name, args)
    
    print("All tasks done!")
