from asyncio import sleep
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import List, Union


def run_code(pythonStr:str):
    folder_path = r"C:\Users\charl\documents\coding-agent\sandbox"
    file_path = os.path.join(folder_path, "task.py")
    
    with open(file_path, "w") as file:
        file.write(pythonStr)
    result = subprocess.run(['python', file_path], cwd=folder_path, capture_output=True, text=True)
    if(result.returncode != 0):
        return [result.stderr, 1]
    output = [result.stdout, 0]
    return output

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

def write_file(file_path:Path, content:str):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        mode = 'a'
    else:
        mode = "w"
    with file_path.open(mode) as f:
        f.write(content)
        
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
TOOLS = {
    "run_python": {
        "description": "Execute Python code and return the output",
        "function_name": "run_code",
        "parameters": {"code": "string"},

    },
    "finish": {
        "description" : "Indicate that the required prompt has been completed",
        "function_name": "finish",
        "parameters": {"result": "string"},
        
    },
    
}



