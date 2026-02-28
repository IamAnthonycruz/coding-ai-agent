import asyncio
import json
from pathlib import Path
from agent_tools import list_files, read_file, run_code, run_shell, write_file
from apis.ollama_api import generate_prompt
from agent_system_prompts import tool_system_prompt, instruction_system_prompt
from prompt_utils import get_coding_prompt
from workers import code_worker, planner_worker


MAX_HISTORY = 5
MAX_RESULT_CHARS = 500

async def create_plan(task:str) -> list[str]:
    planning_prompt = f"""You are a coding agent. Before taking any action, create a clear plan.
    Task: {task}
    Output a numbered paln with 3-5  specific steps. Be concrete - name files, functions, tools
    Do not write any code yet. just plan.
    Example format:
    1. REad the existing files to understand the codebase
    2. Write calculator.py with add, subtract, multiply, divide functions
    3. Write test_calculator.py with pytests test
    4. Run the tests and fix any failures
    5. Use finish tool to declare completion
    Your plan: """
    response = await generate_prompt(
        prompt=planning_prompt,
        system_prompt="You are a planning assistant. Output only a numbered list",
        temperature=0
    )
    return parse_plan(response)
def parse_plan(response:str) -> list[str]:
    lines = response.strip().split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            step = line.split(".", 1)[-1].strip()
            steps.append(step)
    return steps
def truncate_result(result, max_chars: int = MAX_RESULT_CHARS) -> str:
    text = str(result)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def search_files(pattern: str, directory: str = "sandbox", context: int = 3):
    results = []

    path_list = Path(directory).rglob("*.py")

    for file_path in path_list:
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if pattern in line:
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    results.extend([l.strip() for l in lines[start:end]])
        except (UnicodeDecodeError, PermissionError):
            continue
    return "\n".join(results)[:2000]


async def agent_loop(prompt: str, MAX_ATTEMPTS, plan: list[str]=None):
    history = []
    output = prompt
    current_attempt = 0

    while MAX_ATTEMPTS > current_attempt:
        print("\n" + "=" * 80)
        print(f"[AGENT LOOP] Attempt {current_attempt}")
        print("-" * 80)
        print("Current prompt/output:")
        print(output)
        print("=" * 80 + "\n")

        current_attempt += 1

        tool_selector = await generate_prompt(
            prompt=output,
            system_prompt=tool_system_prompt,
            temperature=0
        )

        try:
            tool_selector_json = json.loads(tool_selector)
            tool = tool_selector_json["tool"]
            tool_params = tool_selector_json["parameters"]

            print("\n[TOOL SELECTION]")
            print("-" * 80)
            print(json.dumps(tool_selector_json, indent=2))
            print("-" * 80)
            print(f"Selected tool: {tool}")
            print(f"Tool parameters: {tool_params}")
            print("-" * 80 + "\n")

            def build_prompt(latest_result):
                history_str = "\n".join(
                    f"Step {i+1}: Used '{h['tool']}' → {h['result']}"
                    for i, h in enumerate(history)
                )
                return get_coding_prompt(
                    f"Select the next tool. Previous tool output:\n{latest_result}\n\nHistory:\n{history_str}",
                    context=f"Original prompt:\n{prompt}"
                )

            def append_history(tool_name, result):
                history.append({"tool": tool_name, "result": truncate_result(result)})
                if len(history) > MAX_HISTORY:
                    history.pop(0)

            match tool:
                case "run_python":
                    print("[RUN PYTHON]")
                    code_output, exit_code = run_code(tool_params["code"])
                    print("Exit code:", exit_code)
                    print("Output:")
                    print(code_output)
                    print("-" * 80)
                    append_history("run_python", code_output)
                    output = build_prompt(code_output)

                case "write_file":
                    code_prompt = await generate_prompt(prompt=prompt, system_prompt=instruction_system_prompt, temperature=0)
                    validated_code = await code_worker(code_prompt, max_retries=10)

                    file_path = Path(tool_params["file_path"])
                    write_file_output = write_file(file_path, validated_code)

                    print("Write result:")
                    print(write_file_output)
                    print("-" * 80)
                    append_history("write_file", write_file_output)
                    output = build_prompt(write_file_output)

                case "read_file":
                    print("[READ FILE]")
                    file_path = tool_params["filepath"]
                    read_file_output = read_file(file_path=file_path)

                    print("File contents:")
                    print(read_file_output)
                    print("-" * 80)
                    append_history("read_file", read_file_output)
                    output = build_prompt(read_file_output)

                case "list_files":
                    print("[LIST FILES]")
                    parent_dir = Path(__file__).resolve().parent
                    folder = Path(tool_params["directory"]).name
                    file_path = parent_dir / folder

                    list_files_output = list_files(
                        file_path,
                        recursive=True,
                        include_folders=True
                    )

                    print("Files found:")
                    for f in list_files_output:
                        print(f"  - {f}")
                    print("-" * 80)
                    append_history("list_files", list_files_output)
                    output = build_prompt(list_files_output)

                case "run_shell":
                    print("[RUN SHELL]")
                    commands = tool_params["command"]
                    stdout, stderr, returncode = run_shell(commands)

                    print("Return code:", returncode)
                    print("STDOUT:")
                    print(stdout)
                    print("STDERR:")
                    print(stderr)
                    print("-" * 80)
                    append_history("run_shell", stdout)
                    output = build_prompt(stdout)

                case "search_files":
                    print("[SEARCH FILES]")
                    pattern = tool_params["pattern"]
                    directory = tool_params.get("directory", "sandbox")

                    search_output = search_files(pattern=pattern, directory=directory)

                    print("Search results:")
                    print(search_output)
                    print("-" * 80)
                    append_history("search_files", search_output)
                    output = build_prompt(search_output)

                case _:
                    raise ModuleNotFoundError(f"Unknown tool: {tool}")

        except Exception as e:
            print("\n[AGENT ERROR]")
            print("-" * 80)
            print(e)
            print("-" * 80)
            break


async def main():
    prompts = [
        "Create a file called greet.py that prints hello world then run the file",
        "List the files in the sandbox directory, then read any Python files you find and summarize what they do",
        "Write a Python script that generates 100 random numbers, save it to sandbox/random.py, run it, and tell me the average",
        "Read sandbox/broken.py, find the bug, fix it, write the fixed version, and run the fixed code"
    ]
    for prompt in prompts:
        plan = await planner_worker(prompt)
        model_output = await agent_loop(prompt=prompt, MAX_ATTEMPTS=10)
        print(model_output)
        print(plan)

asyncio.run(main())