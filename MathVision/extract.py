from vllm import LLM, SamplingParams
import re
from tqdm import tqdm
import json

demo_prompt = """
You are an expert at extracting final answers from mathematical solutions. Your task is to identify and extract the final answer from the model's response.

RULES:
1. Extract ONLY the final numerical value, letter choice, or short phrase that answers the question
2. Do NOT include explanations, units, or extra text
3. If the answer is in \\boxed{}, extract what's inside the box
4. If the answer is in <answer></answer> tags, extract what's inside
5. If no clear answer exists or the response is incomplete/repetitive, output "None"
6. For numerical answers: keep the exact format (integer, decimal, fraction)
7. For multiple choice: extract only the letter (A, B, C, D, etc.)

EXAMPLES:

Question: What is the area of the triangle formed by these two lines and the line x = -2?
Model response: ...Therefore, the area of triangle ABC is (1/2)(15)(6) = \\boxed{45}.
Extracted answer: 45

Question: How many individual cubes have exactly four red faces?
Model response: ...Therefore, the number of individual cubes that have exactly four red faces is 0. <answer>0.0</answer>
Extracted answer: 0.0

Question: Which of the following tables could Carl have created?
Model response: ...Based on the analysis, the only table that matches the pattern is Option D. <answer>D</answer>
Extracted answer: D

Question: What is the value of x?
Model response: Solving the equation: 2x + 3 = 7, we get x = 2. The final answer is \\boxed{2}.
Extracted answer: 2

Question: What is the probability?
Model response: ...The probability is 3/4 or 0.75. So the answer is \\boxed{\\frac{3}{4}}.
Extracted answer: 3/4

Question: Choose the correct option.
Model response: After analyzing all options, I believe the answer is B because... The correct choice is B.
Extracted answer: B

Question: Calculate the result.
Model response: Let me think step by step... step by step... step by step... [repeating endlessly]
Extracted answer: None

Now extract the answer from the following:

Question: {question}
Model response: {response}
Extracted answer:"""

def process(line_dict:dict):
    question = ""
    response = ""
    if line_dict['problem']:
        question = line_dict['problem']
    else:
        question = line_dict['problem_w_choices']
    
    response = line_dict['response'][0]
    
    outputs = llm.generate(
        [f"{demo_prompt}\nQuestion: {question}\nModel response: {response}\nExtracted answer:"],
        sampling_params= SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            top_k=1, 
            max_tokens=50,
            stop=["\n", "Question:", "Model response:"]
        ),
    )
    answer = outputs[0].outputs[0].text
    print('====='*40)
    print("Response:", response, "\nLLM Extract: ", answer)
    print('====='*40)
    return answer


if __name__ == "__main__":
    # load llm
    model_path = "/home/minyingqian/models/Qwen2.5-7B-Instruct"
    inference_output_file = "MathVision_inferenced.jsonl"
    extracted_output_file = "MathVision_extracted.jsonl"
    
    tp = 4
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    # extract answer for each line
    with open(inference_output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    outputs = []
    for line in tqdm(lines):
        line_dict = json.loads(line)
        answer = process(line_dict)
        line_dict['extracted_answer'] = answer # 增加extracted answer 字段
        outputs.append(line_dict)
    
    # save lines to new file
    with open(extracted_output_file, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print("Finished processing all lines.")


