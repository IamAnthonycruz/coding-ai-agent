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
    },
    "read_file": {
        "description": "Read the contents of a file",
        "parameters": {"filepath": "string"},
        "function": "read_file"
    },
    "list_files": {
        "description": "List files in a directory",
        "parameters": {"directory": "string"},
        "function": "list_files"
    },
    "run_shell": {
        "description": "Run a shell command and return output",
        "parameters": {"command": "string"},
        "function": "run_shell"
    },
   "search_files": {
        "description": "Search for a pattern in Python files and return matching lines with surrounding context. Use when you need to find where something is defined or used without reading entire files.",
        "parameters": {"pattern": "string", "directory": "string (default: sandbox)"},
        "function": "search_files"
    }
    
}