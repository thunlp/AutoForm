import os
from agentverse.llms.openai import OpenAIChat
import json
import ast
import re
from string import Template
import math
import timeout_decorator
import subprocess

prompt = """Solve the problem presented below:
---
${task_description}
---

RESPONSE GUIDELINES:
1. Initial State Representation: Begin by providing a clear and detailed representation of the initial state or conditions of the problem. It could be code, pseudocode, JSON, markdown table, logical operators, or math equation.
2. Step-by-Step Solution Process: Progressively update the state representation as you work through each step of the solution. This should include all logical reasoning and calculations leading to the final answer.
3. Concluding with the Answer (Optional): End your response with "Answer: {answer}", where {answer} is the final result of your problem-solving process. The {answer} should be a single capital letter.
4. You can use tools to execute Python code to help you do some calculations. You should not finish the whole problem in the code.
5. Only when you end your response with "Answer: {answer}" will the dialogue end.
"""

kwargs = {
    "model": "gpt-4-1106",
    "temperature": 0,
}

result_file = "results-7.jsonl"


@timeout_decorator.timeout(5)
def timeout_exec(f):
    return f()

def check_answer(response, label):
    ans = re.findall(r'Answer: *(.+) *', response)
    if len(ans) == 0:
        return False
    ret_res = label == ans[-1].strip()[0]
    if not ret_res:
        print(f"Response: {response}")
        print(f"Label: {label}")
        print(f"Answer: {ans[-1]}")
    return ret_res


openai = OpenAIChat(max_retry=1000000, **kwargs)
res = []

if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), result_file)):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), result_file), "w") as f:
        f.write("")

cnt_skip = 0
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), result_file)) as f:
    for line in f:
        data = json.loads(line)
        r = check_answer(data["response"], data["label"])
        res.append(r)
        cnt_skip += 1
print(f"Skipping {cnt_skip} tasks.")

with open(os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqua", "test.jsonl")) as f:
    for idx, line in enumerate(f):
        try:
            print(f"Accuracy: {sum(res) / len(res)}, total: {len(res)}")
        except ZeroDivisionError:
            pass
        data = json.loads(line)
        history = []
        func_call_idx = 0
        if idx < cnt_skip:
            continue
        task_description = data["question"] + "\n\n" + "Options:\n" + "\n".join(data['options'])
        label = data["correct"]
        print("\n\n")
        # print(f"Task description: {task_description}")
        for _ in range(20):
            while True:
                try:
                    response = openai.generate_response(
                        prepend_prompt=Template(prompt).safe_substitute(
                            task_description=task_description,
                        ),
                        history=history,
                        functions=[
                            {
                                "name": "exec_python",
                                "description": "Execute a Python script.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "The Python code to execute. The code should print the result to stdout.",
                                        },
                                    },
                                    "required": ["code"],
                                },
                            }
                        ],
                    )
                except KeyError as e:
                    print(e)
                    continue
                break

            if response.function_name:
                new_msg = ""
                try:
                    exec_code = response.function_arguments['code']
                    with open("temp.py", "w") as f:
                        f.write(exec_code)
                    with open("temp.out", "w") as f:
                        sbpss = timeout_exec(lambda: subprocess.run(["python", "temp.py"], stdout=f, stderr=f))
                    with open("temp.out") as f:
                        x = f.read()
                    if sbpss.returncode != 0:
                        x = "Error: " + x + "\nPlease check your code, fix the error, and try again."
                except Exception as e:
                    x = "Error: " + str(e) + "\nPlease check your code, fix the error, and try again."
                history.append({
                    "role": "assistant",
                    "function_call": {
                        "name": response.function_name,
                        "arguments": json.dumps(response.function_arguments),
                    },
                    "content": response.content,
                })
                history.append({
                    "role": "function",
                    "name": response.function_name,
                    "content": x,
                })
                func_call_idx += 1
                print(f"Tool call {func_call_idx}: {response.function_name}({response.function_arguments})")
                print(f"Tool response: {x}")
                print(f"Assistant response: {response.content}")
            elif response.content:
                history.append({
                    "role": "assistant",
                    "content": response.content,
                })
                print(f"Assistant response: {response.content}")

            response = response.content
            if len(re.findall(r"Answer: ?(.+)", response)) > 0:
                break

        print(f"Response: {response}")
        print(f"Label: {label}")
        res.append(check_answer(response, label))
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), result_file), "a") as f1:
            json.dump(
                {
                    "input": task_description,
                    "label": label,
                    "response": response,
                    "history": history,
                },
                f1,
            )
            f1.write("\n")
print(f"Accuracy: {sum(res) / len(res)}")
