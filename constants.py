TOOLS = {
    
    "run_python": {
        "description": "Execute Python code and return the output",
        "function_name": "run_code",
        "parameters": {"code": "exec(open(r'enter file name').read())"},
    },
    "write_file": {
        "description": "Create files and append python text to them",
        "function_name": "write_file",
        "parameters": {"file_path":"string", "content":"string"}
    }
    
}