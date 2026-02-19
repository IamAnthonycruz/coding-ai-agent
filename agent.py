import asyncio
import json
from pathlib import Path
from agent_tools import list_files, read_file, run_code, run_shell, write_file
from apis.ollama_api import generate_prompt
from agent_system_prompts import tool_system_prompt, instruction_system_prompt
from workers import code_worker

async def agent_loop(prompt: str):
    tool_selector = await generate_prompt(prompt=prompt, system_prompt=tool_system_prompt, temperature=0)
    try:
        tool_selector_json = json.loads(tool_selector)
        tool, tool_params = tool_selector_json['tool'], tool_selector_json['parameters']
        
        
        end = False
        while not end:
            match tool:
                case "run_python":
                    run_code(tool_params['code'])
                    end = True
                case "write_file":
                    code_prompt = await generate_prompt(prompt=prompt, system_prompt=instruction_system_prompt, temperature=0)
                    validated_code = await code_worker(code_prompt, max_retries=10)
                    
                    file_path = Path(tool_params['file_path'])
                    
                    write_file_output = write_file(file_path, validated_code)
                    return write_file_output
                
                case "read_file":
                    file_path = tool_params['filepath']
                    print(file_path)
                    read_file_output = read_file(file_path=file_path)
                    return read_file_output
                
                case "list_files":
                    parent_dir = Path(__file__).resolve().parent
                    folder = Path(tool_params['directory']).name
                    file_path = parent_dir / folder
                    list_files_output = list_files(file_path,recursive=True, include_folders=True) 
                    return list_files_output
                case "run_shell":
                    commands = tool_params['command']
                    run_shell_output = run_shell(commands)
                    return run_shell_output
                case _:
                    raise ModuleNotFoundError
    except Exception as e:
        return e

async def main():
    prompts = ["Create a file called greet.py that prints hello world then run the file", "List the files in the sandbox directory, then read any Python files you find and summarize what they do","Write a Python script that generates 100 random numbers, save it to sandbox/random.py, run it, and tell me the average", "Read sandbox/broken.py, find the bug, fix it, write the fixed version, and run it" ]
    for prompt in prompts:
        model_output = await agent_loop(prompt=prompt) 
        
asyncio.run(main())