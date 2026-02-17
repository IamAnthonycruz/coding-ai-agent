import json
from pathlib import Path
from agent_tools import run_code, write_file
from apis.ollama_api import generate_prompt
import asyncio
from agent_system_prompts import tool_system_prompt,  instruction_system_prompt, coding_system_prompt
from prompt_utils import get_coding_prompt
from workers import code_worker

async def main():
    prompt =  "Create a file called greet.py that prints hello world then run the file"
    result = await generate_prompt(prompt=prompt, system_prompt=tool_system_prompt, temperature=0)
    my_code_output = await code_worker(prompt, max_retries=10)
    print(my_code_output)
    myJson = json.loads(result)
    
    res = write_file(Path(myJson['parameters']['file_path']), my_code_output)
    print(res)
    step_2_prompt = get_coding_prompt(f"Proceed with the next step. Heres the output {res} from the previous step.", context=prompt)
    step2 = await generate_prompt(step_2_prompt, tool_system_prompt , temperature=0)
    print(step2)
    step2Json = json.loads(step2)
    output = run_code(step2Json['parameters']['code'])
    print(output)
    
asyncio.run(main())