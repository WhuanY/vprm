# testfile = "MathVista_testmini_output.jsonl"
# import json

# with open(testfile, "r") as f:
#     for line in f:
#         # 检查一下answer 和 answer_w_choices这两个字段
#         d = json.loads(line)
#         print(d['answer'], "\t", d['answer_w_choices'])


import json

# JSON 文件路径
path = "MathVista_testmini.json"

# 打开并读取整个 JSON 文件
with open(path, "r") as f:
    data = json.load(f)  # 直接加载整个文件为一个列表

    for d in data:
        answer = d.get('answer')
        answer_w_choices = d.get('answer_w_choices')
        
        # 检查是否有且仅有一个字段为真值
        if bool(answer) != bool(answer_w_choices):  # 逻辑异或
            print("Valid entry:")
            print("answer:", answer)
            print("answer_w_choices:", answer_w_choices)
        else:
            print("Invalid entry (both are empty or both are filled):")
            print("answer:", answer)
            print("answer_w_choices:", answer_w_choices)
        