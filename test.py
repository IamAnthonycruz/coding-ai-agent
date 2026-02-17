import json
from pathlib import Path
from agent_tools import run_code, write_file
from apis.ollama_api import generate_prompt
import asyncio
from agent_system_prompts import tool_system_prompt,  instruction_system_prompt, coding_system_prompt
from prompt_utils import get_coding_prompt
from workers import code_worker
async def create_and_execute(prompt):
    myPrompt = await generate_prompt(prompt=prompt, system_prompt=tool_system_prompt, temperature=0 )
    return myPrompt




async def main():
    result = await create_and_execute(
       "Create a file called greet.py that prints hello world then run the file"
    )
    code_centered_prompt = await generate_plrompt(
    "USER Prompt: Create a file called greet.py that prints hello world then run the file. DO NOT GENERATE CODE",
    instruction_system_prompt
)
    
    print(f"code centered prompt: {code_centered_prompt}")
    myCode = await generate_prompt(code_centered_prompt, coding_system_prompt)
    print(myCode)
    my_code_output = await code_worker(myCode)
    print(my_code_output)
    myJson = json.loads(result)
    print(myJson)
    
    res = write_file(Path(myJson['parameters']['file_path']), myJson['parameters']['content'])
    print(res)
    step2 = await create_and_execute(f"Proceed with the next step. Heres the output {res} from the previous step. Here's the context Create a file called greet.py that prints hello world then run it")
    print(step2)
    
    step2Json = json.loads(step2)
    output = run_code(step2Json['parameters']['code'])
    print(output)
    
asyncio.run(main())