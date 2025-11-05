import json
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from PIL import Image
import os
import re
from multiple_choice import match_multiple_choice
import argparse
from query_model import (query_gpt4v, query_local)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

import pandas as pd
from PIL import Image
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


def query_model(task_name, args=None):
    """
    loads the dataset from huggingface, query the GPT 4V model with the prompt and images, and saves the result to a json file with specific format.

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
    num_threads = getattr(args, 'num_threads', 1)
    
    output_path = f'{output_save_folder}/{task_name.replace("_", " ")}.json'
    os.makedirs(output_save_folder, exist_ok=True)
    image_folder = f'{image_save_folder}/{task_name}_images'
    os.makedirs(image_folder, exist_ok=True)
    inferenced_idx = set()
    eval_splits = ['val'] if eval_split == 'val' else ['test'] if eval_split == 'test' else ['val', 'test']
    
    # If regen is True, clear existing outputs and start fresh
    if args.regen:
        outputs = {'val': [], 'test': []}
        print(f"Regenerating {task_name} on {eval_splits} set (clearing existing outputs)")
    elif not os.path.exists(output_path):
        outputs = {'val': [], 'test': []}
        print(f"Evaluating {task_name} on {eval_splits} set")
    else:
        print(f"Loading outputs from {output_path}")
        outputs = json.load(open(output_path, 'r'))
        inferenced_idx = set([d['idx'] for d in outputs[eval_split]]) # already inferenced idx
    
    # Thread lock for thread-safe updates
    outputs_lock = threading.Lock()
    
    def process_item(orig_d, split):
        """Process a single data item"""
        idx = orig_d['idx']
        # Skip if already processed (unless regen is True)
        if not args.regen and idx in inferenced_idx:
            return None
        
        gold_answer = orig_d['answer']
        all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(orig_d['choices'])]
        image_paths, prompt = load_prompt(task_name, orig_d, image_folder, args)
        gpt_answer = query_local(image_paths, prompt, args) # NOTE: main inference process
        prediction = analyze_answer(orig_d, gpt_answer, all_choices)
        
        result = {'idx': idx, 'answer': gold_answer, 'full_prediction': gpt_answer, 'prediction': prediction, 'prompt': prompt}
        
        # Thread-safe update of outputs and file save
        with outputs_lock:
            outputs[split].append(result)
            json.dump(outputs, open(output_path, 'w'), indent=4)
        
        return result
    
    for split in eval_splits:
        data_files = {'val': f'{dataset_local_path}/{task_name}/val-00000-of-00001.parquet','test': f'{dataset_local_path}/{task_name}/test-00000-of-00001.parquet'}
        
        if task_name == 'Relative_Reflectance': # 这个数据集有点问题，需要单独处理
            test_data = load_relative_reflectance_data(data_files)[split]
        else:
            test_data = load_dataset('parquet', data_files=data_files)[split]
        
        # Filter out already processed items
        if args.regen:
            items_to_process = [orig_d for orig_d in test_data]
        else:
            items_to_process = [orig_d for orig_d in test_data if orig_d['idx'] not in inferenced_idx]
        
        if num_threads > 1 and len(items_to_process) > 0:
            # Use concurrent processing
            print(f"Processing {len(items_to_process)} items with {num_threads} threads...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(process_item, orig_d, split): orig_d 
                    for orig_d in items_to_process
                }
                
                # Process with progress bar
                for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc=f"Processing {task_name} {split}"):
                    orig_d = future_to_item[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing item {orig_d.get('idx', 'unknown')}: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            # Sequential processing (original behavior)
            for orig_d in tqdm(items_to_process, desc=f"Processing {task_name} {split}"):
                process_item(orig_d, split)
        
        # Final save
        json.dump(outputs, open(output_path, 'w'), indent=4)
        
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
    if 'blip' in model_name:
        prompt += '\nAnswer:'
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
    parser.add_argument("--model_name_or_path", type=str, default='GPT4V', help="select the model name")
    parser.add_argument("--inference_api", type=str, default='http://0.0.0.0:8080/v1', help="the inference api for local model serving")
    parser.add_argument("--dataset_local_path", type=str, default='BLINK_fromhf', help="the local path to the dataset downloaded from huggingface")
    parser.add_argument("--task_name", type=str, default='Relative_Depth', help="select the task name")
    parser.add_argument("--eval_split", type=str, default='val', help="select the eval split")
    parser.add_argument("--pre_prompt", type=str, default='', help="the pre-prompt for the model")
    parser.add_argument("--after_prompt", type=str, default='', help="the after-prompt for the model")
    parser.add_argument("--output_save_folder", type=str, default='outputs', help="directory to save output JSON files")
    parser.add_argument("--image_save_folder", type=str, default='images', help="directory to save processed images")
    parser.add_argument("--num_threads", type=int, default=1, help="number of concurrent threads for processing (default: 1, sequential)")
    parser.add_argument("--regen", action='store_true', help="regenerate the outputs")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name_or_path # model_path: 模型参数路径
    model_name = os.path.basename(model_path) # model_name: ckpt文件名
    print(f'Using ckpt: {model_path}, model name: {model_name}')

    model_generate_funcs = {
        'GPT4V': query_gpt4v,                     
    }

    model_generate_funcs.get(model_name, query_local) # should go query_local
    
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
    # 退出脚本
    import sys
    sys.exit(0)
# if __name__ == '__main__':
#     load_relative_reflectance_data({
#         "val": "data/Relative_Reflectance/val-00000-of-00001.parquet",
#         "test": "data/Relative_Reflectance/test-00000-of-00001.parquet"
#     })
