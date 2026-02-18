import asyncio
import json
from pathlib import Path
from agent_tools import list_files
from apis.ollama_api import generate_prompt
from agent_system_prompts import tool_system_prompt

async def agent_loop(prompt: str):
    tool_selector = await generate_prompt(prompt=prompt, system_prompt=tool_system_prompt, temperature=0)
    try:
        tool_selector_json = json.loads(tool_selector)
        tool, tool_params = tool_selector_json['tool'], tool_selector_json['parameters']
        print(tool_params)
        match tool:
            case "run_python":
                pass
            case "write_file":
                pass
            case "read_file":
                pass
            case "list_files":
                parent_dir = Path(__file__).resolve().parent
                folder = Path(tool_params['directory']).name
                file_path = parent_dir / folder
                output = list_files(file_path,recursive=True, include_folders=True) 
                for item in output:
                    print(item)
            case "run_shell":
                pass
            case _:
                raise ModuleNotFoundError
    except Exception as e:
        return e

async def main():
    prompts = ["List the files in the sandbox directory, then read any Python files you find and summarize what they do","Write a Python script that generates 100 random numbers, save it to sandbox/random.py, run it, and tell me the average", "Read sandbox/broken.py, find the bug, fix it, write the fixed version, and run it" ]
    for prompt in prompts:
        model_output = await agent_loop(prompt=prompt) 
        
asyncio.run(main())