import os
import base64
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


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


def encode_image_to_base64(filepath):
    """Encodes an image file to a base64 data URI."""
    try:
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # 根据文件扩展名确定 MIME 类型
            mime_type = "image/jpeg" # 默认为 jpeg
            if filepath.lower().endswith(".png"):
                mime_type = "image/png"
            elif filepath.lower().endswith(".gif"):
                mime_type = "image/gif"
            elif filepath.lower().endswith(".webp"):
                mime_type = "image/webp"
            
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

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
    
class LocalLLMClient:
    def __init__(self, model, inference_api):
        self.client =  OpenAI(
            api_key="EMPTY",
            base_url=inference_api
        )
        self.model = model
        self.inference_api = inference_api

    def _generate_single(self, single_input, sampling_params):
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=single_input,
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature
        )
        return chat_response

    def generate(self, batch_inputs, sampling_params):
        outputs = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_index = {
                executor.submit(self._generate_single, msg['messages'], sampling_params): i
                for i, msg in enumerate(batch_inputs)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    response = future.result()
                    outputs.append(response)
                except Exception as e:
                    print(f"Request for input {index} generated an exception: {e}")
                    traceback.print_exc()
                    assert False, "Error in API request"
        return outputs
    