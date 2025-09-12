# path = "MathVision_inferenced.json"
import json


# with open(path, "r") as f:
#     data = json.load(f)

path = "MathVision_inferenced.jsonl"

with open(path, "r") as f:
    data = f.readlines()


print("-----"*50)
for d in data:
    d = json.loads(d)
    # 检查每一行的 answer 字段和 answer_w_choices 字段
    answer = d['answer']
    answer_w_choices = d['answer_w_choices']

    problem = d['problem']
    problem_w_choices = d['problem_w_choices']

    # 打印出有效的答案
    print("ground_truth_as:", answer if answer else answer_w_choices)

    print("problem:        ", problem if problem else problem_w_choices)


    # 检查至少一个字段有值(answer)
    assert answer or answer_w_choices, f"Both answer and answer_w_choices are empty.,{d}"
    # 检测至少一个字段有值(problem)
    assert problem or problem_w_choices, f"Both problem and problem_w_choices are empty.,{d}"
    print("-----"*50)
