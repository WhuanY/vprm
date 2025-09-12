
import re

def extract_answer_from_response(response):
    """
    提取response中最后一个<answer>标签的内容
    对于单选题和多选题都适用
    """
    if not response:
        return None
    
    # 检查是否包含unfinished标记
    if "unfinished" in response.lower():
        return None
    
    # 使用正则表达式找到所有<answer></answer>标签
    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    if answer_matches:
        # 返回最后一个匹配的内容，去除首尾空白
        last_answer = answer_matches[-1].strip()
        return last_answer if last_answer else "No <answer> tag found"
    else:
        return "No <answer> tag found"

    

if __name__ == "__main__":
    # 测试代码
    test_responses = [
        "The answer is <answer>42</answer>.",
        "<answer>Yes</answer>, I believe so.",
        "No answer provided.",
        "Here is my answer: <answer>3.14</answer> and some more text <answer>ignored</answer>.",
        "",
        None
    ]
    
    for resp in test_responses:
        extracted = extract_answer_from_response(resp)
        print(f"Response: {resp}\nExtracted Answer: {extracted}\n{'-'*40}")