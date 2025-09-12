# input "/home/minyingqian/vprm/MathVista/MathVista_judge_results.jsonl"
# output "metrics"
# logic: for each line, get judgement_text, if "1", correct +1, else wrong +1
import json
fp = open("MathVista_judge_results.jsonl", "r", encoding="utf-8")
lines = fp.readlines()
fp.close()

# def extract_answer_from_response(response):
#     """从模型的response中提取答案，假设答案是单个字母A/B/C/D/E"""
#     if isinstance(response, list):
#         response_text = " ".join(response)
#     else:
#         response_text = response

#     # 简单的正则表达式匹配A/B/C/D/E
#     import re
#     match = re.search(r'\b([A-E])\b', response_text)
#     if match:
#         return match.group(1)
#     else:
#         return "N/A"

total = len(lines)
correct = 0
wrong = 0
for line in lines:
    data = json.loads(line)
    if data["judgment_text"] == "1":
        std_ans = data['standard_answer'] 
        extracted = data['model_response_ans_extracted']
        if std_ans != extracted:
            print("[INFO] Judgment is 1 but extracted answer does not exact match standard answer") # 这里可能有错正例
            print("id:", data['id'], '\t', "standard answer: ", data['standard_answer'], '\t',"Extracted: ", data['model_response_ans_extracted'] )
        correct += 1
        
    elif data["judgment_text"] == "0":
        if data['standard_answer'] == data['model_response_ans_extracted']:
            print("[INFO] Rule based extraction matches standard answer, counting as correct")
            correct += 1
        else:
            wrong += 1
            # print("id:", data['id'], '\t', "standard answer: ", data['standard_answer'], '\t',"Extracted: ", data['model_response_ans_extracted'], )
        
    else:
        print("Unknown judgment_text: ", data["judgment_text"])



print("Total: ", total)
print("Correct: ", correct)
print("Wrong: ", wrong)
