def answer2choiceLetter(answer: str, options: list) -> str:
    """
    Convert answer to choice letter, e.g., "A", "B", "C", "D"
    eg: 
    options = ['Yes', 'No'], answer = 'No' -> 'B'
    options = [7.5,8,8.5,17], answer = 8.5 -> 'C'
    ...
    """
    if not options or len(options) == 0:
        return answer
    for idx, opt in enumerate(options):
        if str(opt).strip().lower() == str(answer).strip().lower():
            return chr(ord('A') + idx)
    return answer


if __name__ == "__main__":
    # 测试代码
    options_lst = [
        ['Yes', 'No'],
        [7.5,8,8.5,17],
        ['Apple', 'Banana', 'Cherry'],
        ["3.85米","4.00米","4.40米","4.50米"]
    ]
    answers = ['No', 8.5, 'Banana', '4.40米']
    desire_output = ['B', 'C', 'B', 'C']
    for options, answer, desire in zip(options_lst, answers, desire_output):
        converted = answer2choiceLetter(answer, options)
        print(f"Options: {options}\nAnswer: {answer}\nConverted: {converted}\nDesire: {desire}\n{'-'*40}")

    

