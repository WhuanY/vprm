import os
import base64
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


import numpy as np
import re
from copy import deepcopy
import time  
from math import *

from tqdm import tqdm  # Assuming tqdm is imported to show progress bar

from openai import OpenAI

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

def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def save_jsonl(path: str, data: list, t_stamp=True) -> None:
    if t_stamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def is_exact_match(model_answer: str, standard_answer: str) -> bool:
    """
    判断模型答案和标准答案是否完全匹配。
    standard_answer有三种情况:整数、浮点数、极短文本。
    """
    # 预处理：去除首尾空白
    model_ans = model_answer.strip()
    standard_ans = standard_answer.strip()
    
    # 1. 直接字符串匹配（大小写不敏感）
    if model_ans.lower() == standard_ans.lower():
        return True
    
    # 2. 尝试作为数字匹配
    try:
        # 尝试解析标准答案为数字
        standard_num = float(standard_ans)
        
        # 尝试从模型答案中提取数字
        model_num = _extract_number(model_ans)
        
        if model_num is not None:
            # 判断是否为整数类型
            if _is_integer_string(standard_ans):
                # 整数比较：要求完全相等
                return int(model_num) == int(standard_num)
            else:
                # 浮点数比较：使用相对容差
                return _float_equal(model_num, standard_num)
    except (ValueError, TypeError):
        pass
    
    # 3. 极短文本匹配：去除标点和多余空格后比较
    normalized_model = _normalize_text(model_ans)
    normalized_standard = _normalize_text(standard_ans)
    
    return normalized_model == normalized_standard


def _extract_number(text: str) -> float | None:
    """从文本中提取数字，支持多种格式。"""
    import re
    
    # 移除常见的非数字前缀（如货币符号、单位等）
    text = text.strip()
    
    # 模式1: 纯数字（可能带正负号、小数点、科学计数法）
    patterns = [
        r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)$',  # 纯数字
        r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[%]?$',  # 数字+可选百分号
        r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',  # 提取第一个数字
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                num_str = match.group(1)
                return float(num_str)
            except (ValueError, IndexError):
                continue
    
    return None


def _is_integer_string(s: str) -> bool:
    """判断字符串是否表示整数。"""
    import re
    # 匹配整数格式（可带正负号，但不带小数点）
    return bool(re.match(r'^[+-]?\d+$', s.strip()))


def _float_equal(a: float, b: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> bool:
    """
    浮点数比较，使用相对和绝对容差。
    rel_tol: 相对容差（默认0.001%）
    abs_tol: 绝对容差（处理接近0的情况）
    """
    import math
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _normalize_text(text: str) -> str:
    """
    标准化文本：
    - 转小写
    - 移除标点符号
    - 合并多个空格为一个
    - 去除首尾空格
    """
    import re
    import string
    
    # 转小写
    text = text.lower()
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text


# ============ 测试用例 ============
if __name__ == "__main__":
    test_cases = [
        # (model_answer, standard_answer, expected_result)
        
        # 整数测试
        ("42", "42", True),
        ("42", "43", False),
        ("  42  ", "42", True),
        ("The answer is 42", "42", True),
        ("42.0", "42", True),  # 浮点表示的整数
        ("-15", "-15", True),
        
        # 浮点数测试
        ("3.14", "3.14", True),
        ("3.14159", "3.14159", True),
        ("3.14159", "3.14158", False),  # 超出容差
        ("3.140001", "3.14", True),  # 在容差内
        ("  2.718  ", "2.718", True),
        ("The value is 2.718", "2.718", True),
        ("-0.5", "-0.5", True),
        ("1e-5", "0.00001", True),  # 科学计数法
        ("1.5e2", "150", True),
        
        # 极短文本测试
        ("yes", "yes", True),
        ("Yes", "yes", True),
        ("YES", "yes", True),
        ("  yes  ", "yes", True),
        ("yes!", "yes", True),  # 忽略标点
        ("yes.", "yes", True),
        ("no", "yes", False),
        ("cat", "cat", True),
        ("cat", "dog", False),
        ("hello world", "hello world", True),
        ("Hello, World!", "hello world", True),
        ("a b c", "a  b  c", True),  # 多个空格
        
        # 边界情况
        ("", "", True),
        ("0", "0", True),
        ("0.0", "0", True),
        ("-0", "0", True),
        ("true", "True", True),
        ("False", "false", True),
    ]
    
    print("Running tests...")
    passed = 0
    failed = 0
    
    for model_ans, standard_ans, expected in test_cases:
        result = is_exact_match(model_ans, standard_ans)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"{status} FAILED: is_exact_match('{model_ans}', '{standard_ans}') = {result}, expected {expected}")
    
    print(f"\nResults: {passed}/{len(test_cases)} passed, {failed} failed")