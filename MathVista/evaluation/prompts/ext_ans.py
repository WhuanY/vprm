

# pids = 852,  104,  824,  506,  540

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

# 852,  104,  824,  506,  540 + 880, 619, 606
demo_prompt_w_cot = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
Example 1: 
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: Which number is missing?\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nSolution:

Model response: <think>\nTo solve this problem, let's look at the sequence of numbers provided:\n4, 5, 7, 10, 14, 19, 25, 32\n\nWe can observe that each number in the sequence is obtained by adding consecutive odd numbers to the previous number:\n- 4 + 1 = 5\n- 5 + 2 = 7\n- 7 + 3 = 10\n- 10 + 4 = 14\n- 14 + 5 = 19\n- 19 + 6 = 25\n- 25 + 7 = 32\n\nFollowing this pattern, the next number should be obtained by adding 8 to 32:\n32 + 8 = 40\n\nTherefore, the missing number is 40.\n</think>\n<answer>\n40\n</answer>

Extracted answer: 40
Example2: 
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: What is the fraction of females facing the camera?\nHint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.\nSolution:

Model response: <think>\nThere are five people in the image, and four of them are facing the camera. Therefore, the fraction of females facing the camera is 4/5, which is equivalent to 0.8 or 80% when expressed as a percentage.\n</think>\n<answer>\n0.8\n</answer>",

Extracted answer: 0.8
Example 3:
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)\nHint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.\nSolution:

Model response: <think>\nTo find out how much money Luca needs to buy a sour apple candy and a butterscotch candy, we need to add the prices of these two items together. \n\nThe price of a sour apple candy is $0.06.\nThe price of a butterscotch candy is also $0.07.\n\nSo, the total cost is:\n$0.06 + $0.07 = $0.13\n\nTherefore, Luca needs $0.13 to buy both items.\n</think>\n<answer>\n0.13\n</answer>

Extracted answer: 0.13
Example 4: 
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: Between which two years does the line  graph saw its maximum peak?\nHint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\nSolution:

Model response: <think>\nTo determine between which two years the line graph saw its maximum peak, I need to identify the highest point on the graph. The x-axis represents years from 2005 to 2016, and the y-axis represents percentages. The highest point on the graph is clearly around 2015-2016, where the line reaches its peak. Therefore, the years with the maximum peak are 2015 and 2016.\n</think>\n<answer>\n[2015, 2016]\n</answer>"

Extracted answer: [2015, 2016]
Example 5:
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5\nHint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nSolution:

Model response: <think>\nTo determine what fraction of the shape is blue, we need to count the number of blue sections and divide it by the total number of sections in the shape.\n\nThe shape is divided into 11 equal parts, and out of these, 3 parts are blue.\n\nSo, the fraction of the shape that is blue is \\( \\frac{3}{11} \\).\n\n</think>\n<answer>\nA\n</answer>,
        
Extracted answer: A
Example 6: 
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: When does the expenditure per student in Jamaica have the greatest increase?\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nSolution:

Model response: <think>\nTo determine when the expenditure per student in Jamaica has the greatest increase, I need to look for the steepest upward slope on the line graph. The graph shows that the expenditure per student in Jamaica starts at around 40% in 2004, peaks at approximately 65% in 2005, and then decreases slightly after that. Therefore, the greatest increase occurs between 2004 and 2005.\n</think>\n<answer>\n2004 to 2005\n</answer>

Extracted answer: 2004 to 2005
Example 7: 
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: What's the total add up value of largest and smallest bar?\nHint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.\nSolution:

Model response: <think>\nTo find the total add up value of the largest and smallest bars, I need to:\n1. Identify the largest bar (Global ocean)\n2. Identify the smallest bar (Mediterranean Sea)\n3. Add these two values together\n\nThe largest bar is the Global ocean, which has a value of 5.25 trillion.\nThe smallest bar is the Mediterranean Sea, which has a value of 247.4 billion.\n\nNow, I'll add these two values:\n5.25 trillion + 0.2474 billion = 5.252474 trillion\n\nRounding this to two decimal places gives us 5.25 trillion.\n<answer>\n5.25 trillion\n</answer>

Extracted answer: 5.25
Example 8:
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)\nHint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.\nSolution:

Model response: <think>\nTo find the value of \\( f(4) \\) for the function \\( f(x) = 2^x \\), we need to substitute \\( x = 4 \\) into the equation and calculate the result.\n</think>\n<answer>\nf(4) = 2^4 = 16\n</answer>

Extracted answer: 16
Example 9:
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: What is the perimeter of the rectangle?\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nSolution:

Model response: <think>\nTo find the perimeter of a rectangle, we use the formula:\nPerimeter = 2 * (length + width)\nGiven that the length is 3 cm and the width is 2 cm, we can substitute these values into the formula:\nPerimeter = 2 * (3 cm + 2 cm) = 2 * 5 cm = 10 cm\n</think>\n<answer>\n10 cm\n</answer>

Extracted answer: 10

You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\\n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags.\n\n\nQuestion: Chase wants to buy 4 kilograms of oval beads and 5 kilograms of star-shaped beads. How much will he spend? (Unit: $)\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nSolution:

Model response: <think>\nTo calculate the total cost, we need to multiply the price per kilogram by the number of kilograms for each type of bead and then add the costs together.\n\nFor oval beads:\nPrice per kilogram = $2\nNumber of kilograms = 4\nTotal cost for oval beads = Price per kilogram \u00d7 Number of kilograms\nTotal cost for oval beads = $2 \u00d7 4 = $8\n\nFor star-shaped beads:\nPrice per kilogram = $2\nNumber of kilograms = 5\nTotal cost for star-shaped beads = Price per kilogram \u00d7 Number of kilograms\nTotal cost for star-shaped beads = $2 \u00d7 5 = $10\n\nNow, we add the costs together to find the total amount Chase will spend:\nTotal cost = Total cost for oval beads + Total cost for star-shaped beads\nTotal cost = $8 + $10 = $18\n\nTherefore, Chase will spend $18 in total.\n</think>\n<answer>\n$18\n</answer>

Extracted answer: 18
"""

