import json
import os
import argparse


def get_prediction_file(split, model_name, output_save_folder, prediction_output_dir=''):
    """
    Combine the task-specific prediction files for a model on split into one single final-prediction json file.

    Parameters:
    - split: String, the split to evaluate on.
    - model_name: String, the name of the model.
    - output_save_folder: String, directory where output JSON files are saved.
    - prediction_output_dir: String, directory to save prediction files. If empty, uses default 'split_predictions'.

    Returns:
    - save_path, the path to the saved final prediction json file.
    """
    if prediction_output_dir:
        save_path = f'{prediction_output_dir}/{split}_predictions_{model_name}.json'
    else:
        save_path = f'{split}_predictions/{model_name}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    saved = {}
    for task_name in subtasks:
        output_path = f'{output_save_folder}/{task_name.replace("_", " ")}.json'
        outputs = json.load(open(output_path, 'r'))[split]
        for d in outputs:
            saved[d['idx']] = d['prediction']
    json.dump(saved, open(save_path, 'w'), indent=4)
    return save_path


def eval_prediction(split, model_name, prediction_output_dir=''):
    """
    Evaluate the model on the split and return the accuracy for all tasks and also total accuracy.

    Parameters:
    - split: String, the split to evaluate on.
    - model_name: String, the name of the model.
    - prediction_output_dir: String, directory where prediction files are saved. If empty, uses default 'split_predictions'.

    Returns:
    - accu_by_task, the accuracy for all tasks and also total accuracy (averaged over all subtasks).
    """
    accu_by_task = {}
    task_numbers = {}
    errors = {}
    for task_name in subtasks:
        accu_by_task[task_name] = 0
        task_numbers[task_name] = 0
        errors[task_name] = []
    answer_file_path = f'{split}_answers.json'
    if prediction_output_dir:
        prediction_file_path = f'{prediction_output_dir}/{split}_predictions_{model_name}.json'
    else:
        prediction_file_path = f'{split}_predictions/{model_name}.json'
    answers = json.load(open(answer_file_path, 'r'))
    predictions = json.load(open(prediction_file_path, 'r'))
    for idx, gold_answer in answers.items():
        task = '_'.join(idx.split(split)[1][1:].split('_')[:-1])
        # task_numbers[task] += 1
        task_numbers[task] = task_numbers.get(task, 0) + 1
        if idx in predictions and predictions[idx] == gold_answer:
            accu_by_task[task] += 1
        else:
            if task not in errors:
                errors[task] = []
            errors[task].append(idx)

    average_accu = 0
    for task in subtasks:
        accu_by_task[task] = accu_by_task[task] / task_numbers[task]
        average_accu += accu_by_task[task]
    average_accu = average_accu / len(subtasks)
    accu_by_task["Total"] = average_accu 
    print(f'Average Accuracy of model {model_name} on BLINK split {split} over all tasks is {round(100 * average_accu, 2)}%')
    
    # Format results for saving
    results = {
        "model_name": model_name,
        "split": split,
        "total_accuracy": round(100 * average_accu, 2),
        "task_accuracies": {task: round(100 * accu_by_task[task], 2) for task in subtasks},
        "task_numbers": task_numbers,
        "errors": errors
    }
    
    return accu_by_task, results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='GPT4V', help="select the model name")
    parser.add_argument("--inference_api", type=str, default='http://0.0.0.0:8080/v1', help="the inference api for local model serving")
    parser.add_argument("--dataset_local_path", type=str, default='BLINK_fromhf', help="the local path to the dataset downloaded from huggingface")
    parser.add_argument("--task_name", type=str, default='Relative_Depth', help="select the task name")
    parser.add_argument("--eval_split", type=str, default='val', help="select the eval split")
    parser.add_argument("--output_save_folder", type=str, default='outputs', help="directory where output JSON files are saved (should match inference output)")
    parser.add_argument("--prediction_output_dir", type=str, default='', help="directory to save prediction files. If empty, uses default 'split_predictions' directory")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':  
    dataset_name = 'BLINK-Benchmark/BLINK'
    arg = parse_args()
    
    # Use args values if provided, otherwise use defaults
    output_save_folder = arg.output_save_folder
    prediction_output_dir = arg.prediction_output_dir

    # # models that we experimented on
    # model_names = [
    #                 'MiniGPT-4-v2', 'flamingov2', 'instructblip_7b', 'instructblip_13b',
    #                 'llava-internlm2-7b', 'Yi_VL_6B', 'Yi_VL_34B',
    #                 'llava-v1.5-7b-xtuner', 'llava-v1.5-13b-xtuner', 'cogvlm-chat', 
    #                 'llava_v1.5_7b', 'llava_v1.5_13b', 'llava-v1.6-34b',
    #                 'QwenVLMax', 'GeminiProVision', 'GPT4V', 'OPUS'
    #                 ]
    # # save to a output path with model_name.json, replace with custom model name
    # model_name = model_names[-2]
    
    subtasks = [
        'Visual_Similarity', 'Counting', 'Relative_Depth', 'Jigsaw', 'Art_Style', 'Functional_Correspondence', 'Semantic_Correspondence', 'Spatial_Relation', 'Object_Localization', 'Visual_Correspondence', 'Multi-view_Reasoning', 
        # 'Relative_Reflectance', # 这个数据集有点问题
        'Forensic_Detection', 'IQ_Test'
    ]
    model_name = os.path.basename(arg.model_name_or_path)
    print(f"{model_name=}")
    print(f"output_save_folder: {output_save_folder}")
    print(f"prediction_output_dir: {prediction_output_dir if prediction_output_dir else 'default (split_predictions)'}")

    split = 'val'
    get_prediction_file(split, model_name, output_save_folder, prediction_output_dir)
    accu_by_task, results = eval_prediction(split, model_name, prediction_output_dir)
    
    # Save results to file
    if prediction_output_dir:
        results_file = f'{prediction_output_dir}/{split}_results_{model_name}.json'
    else:
        results_file = f'{split}_results/{model_name}.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    json.dump(results, open(results_file, 'w'), indent=4)
    print(f'\nResults saved to: {results_file}')
    print(f'Total Accuracy: {results["total_accuracy"]}%')
    print('\nTask Accuracies:')
    for task in subtasks:
        print(f'  {task}: {results["task_accuracies"][task]}%')
