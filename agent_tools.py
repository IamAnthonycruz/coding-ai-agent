from asyncio import sleep
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import List, Union

from apis.ollama_api import generate_prompt

from agent_system_prompts import coding_system_prompt

def run_code(pythonStr:str, is_code_fixer=False):
    folder_path = r"C:\Users\charl\Documents\coding-agent\sandbox"
    file_name = 'task.py'
    if is_code_fixer:
        file_name = 'test.py'
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "w") as file:
        file.write(pythonStr)
    result = subprocess.run(['python', file_path], cwd=folder_path, capture_output=True, text=True)
    if(result.returncode != 0):
        return [result.stderr, 1]
    output = [result.stdout, 0]
    return output

async def create_code(instructions: str):
    return await generate_prompt(instructions, system_prompt=coding_system_prompt)


def finish(result: str):
    return True, result

def list_files(path:Path,*, recursive=False, include_folders=False):
    if not path.is_dir():
        raise NotADirectoryError(path)
    if recursive:
        iterator = path.rglob("*")
    else:
        iterator = path.iterdir()
    if include_folders:
        items = [p for p in iterator if p.is_file() or p.is_dir()]
    else:
        items = [p for p in iterator if p.is_file()]
    items.sort(key=lambda p:(p.is_file(), p.name.lower()))
    return items


def read_file(file_path: Path, max_size=2*1024*1024):
    if not file_path.is_file():
        raise FileNotFoundError("File could not be found")
    if file_path.stat().st_size > max_size:
        raise ValueError("File too large for processing")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        raise

def write_file(file_name: str, content: str):
    # Hardcoded folder path
    folder = Path("C:/Users/charl/Documents/coding-agent/sandbox")  # <-- your fixed folder
    file_path = folder / file_name

    # Ensure the folder exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose mode
    mode = 'a' if file_path.exists() else 'w'

    # Write content
    with file_path.open(mode) as f:
        f.write(content)

    return f"Tool result: Wrote {len(content)} chars to {str(file_path)}"
def run_command(command: Union[str, List[str]], *, timeout: int=30, shell:bool =False):
    if isinstance(command, str) and not shell:
        command = shlex.split(command)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell
        )
    return result.stdout, result.stderr, result.returncode





