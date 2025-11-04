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
        pass
        print(gpt_answer, e)
        kill


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
    if not os.path.exists(output_path):
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
        if idx in inferenced_idx:
            return None
        
        gold_answer = orig_d['answer']
        all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(orig_d['choices'])]
        image_paths, prompt = load_prompt(task_name, orig_d, image_folder, args)
        gpt_answer = query_local(image_paths, prompt, args) # NOTE: main inference process
        prediction = analyze_answer(orig_d, gpt_answer, all_choices)
        
        result = {'idx': idx, 'answer': gold_answer, 'full_prediction': gpt_answer, 'prediction': prediction}
        
        # Thread-safe update of outputs and file save
        with outputs_lock:
            outputs[split].append(result)
            json.dump(outputs, open(output_path, 'w'), indent=4)
        
        return result
    
    for split in eval_splits:
        data_files = {'val': f'{dataset_local_path}/{task_name}/val-00000-of-00001.parquet','test': f'{dataset_local_path}/{task_name}/test-00000-of-00001.parquet'}
        try:
            test_data = load_dataset('parquet', data_files=data_files)[split]
        except:
            print(f"ERROR in loading dataset: {task_name}")
            import traceback
            traceback.print_exc()
            kill
        
        # Filter out already processed items
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

    Parameters:
    - task_name: String, the name of the task.
    - d: data entry, the data dictionary containing the prompt and images.
    - image_folder: String, the folder to save the images.
    - args: command line arguments. 

    Returns:
    - image_paths: List of strings, the filepaths to the saved images.
    - prompt: String, the prompt text.
    - d: Dictionary, the data dictionary with the image paths removed.
    """
    image_paths = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in d and d[k]:
            image = d[k]
            image_path = f'{image_folder}/{d["idx"]}_{k[-1]}.jpg'
            image.save(image_path)
            image_paths.append(image_path)

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


def eval_task(task_name, args):
    outputs = query_model(task_name, args)
    accu = {'val': 0, 'test': 0}
    for split in ['val', 'test']:
        for d in outputs[split]:
            if d['answer'] == d['prediction']:
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
    
    need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']
    if args.task_name == 'all': 
        subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 
                    #'Relative_Reflectance', # 这个数据集有点问题
                      'Visual_Correspondence', 'Counting', 
                      'IQ_Test',  
                      'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        eval_task(task_name, args)
    
    print("All tasks done!")
    # 退出脚本
    import sys
    sys.exit(0)
