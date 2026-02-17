

from prompt_utils import get_coding_prompt
from agent_tools import run_code
from apis import generate_prompt
from agent_system_prompts import tool_system_prompt
PREV_RES = []

MAX_ATTEMPTS = 10
original_prompt = get_coding_prompt("Write hello w")
pythonCode = generate_prompt(original_prompt, tool_system_prompt)
print(pythonCode)
PREV_RES.append(pythonCode)
output,exit_code = run_code(pythonCode)

attempts = 0

while exit_code != 0 and attempts < MAX_ATTEMPTS:
        newCodingPrompt = get_coding_prompt(
            task=f"fix {PREV_RES[-1]}\n",
            libraries="request python library",
            context=f"{original_prompt} is the original prompt\n",
            errors=f"{output} fix error\n"
        )
        pythonCode = generate_prompt(newCodingPrompt)
        PREV_RES.append(pythonCode)
        output, exit_code = run_code(pythonCode)
        print(output)
        attempts+=1
