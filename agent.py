import asyncio
import json
from pathlib import Path
from agent_tools import list_files, read_file, run_code, run_shell, write_file
from apis.ollama_api import generate_prompt
from agent_system_prompts import tool_system_prompt, instruction_system_prompt
from prompt_utils import get_coding_prompt
from workers import code_worker

async def agent_loop(prompt: str, MAX_ATTEMPTS):
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

            match tool:
                case "run_python":
                    print("[RUN PYTHON]")
                    code_output, exit_code = run_code(tool_params["code"])
                    print("Exit code:", exit_code)
                    print("Output:")
                    print(code_output)
                    print("-" * 80)
                    return code_output, exit_code

                case "write_file":
                    code_prompt = await generate_prompt(prompt=prompt, system_prompt=instruction_system_prompt, temperature=0)
                    validated_code = await code_worker(code_prompt, max_retries=10)

                    file_path = Path(tool_params["file_path"])
                    write_file_output = write_file(file_path, validated_code)

                    print("Write result:")
                    print(write_file_output)
                    print("-" * 80)

                    output = get_coding_prompt(
                        f"Select the next tool. Previous tool output:\n{write_file_output}",
                        context=f"Original prompt:\n{prompt}"
                    )

                case "read_file":
                    print("[READ FILE]")
                    file_path = tool_params["filepath"]
                    read_file_output = read_file(file_path=file_path)

                    print("File contents:")
                    print(read_file_output)
                    print("-" * 80)

                    output = get_coding_prompt(
                        f"Select the next tool. Previous tool output:\n{read_file_output}",
                        context=f"Original prompt:\n{prompt}"
                    )

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

                    output = get_coding_prompt(
                        f"Select the next tool. Previous tool output:\n{list_files_output}",
                        context=f"Original prompt:\n{prompt}"
                    )

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

                    output = get_coding_prompt(
                        f"Select the next tool. Previous tool output:\n{stdout}",
                        context=f"Original prompt:\n{prompt}"
                    )

                case _:
                    raise ModuleNotFoundError(f"Unknown tool: {tool}")

        except Exception as e:
            print("\n[AGENT ERROR]")
            print("-" * 80)
            print(e)
            print("-" * 80)
            break
async def main():
    prompts = ["Create a file called greet.py that prints hello world then run the file", "List the files in the sandbox directory, then read any Python files you find and summarize what they do","Write a Python script that generates 100 random numbers, save it to sandbox/random.py, run it, and tell me the average", "Read sandbox/broken.py, find the bug, fix it, write the fixed version, and run the fixed code" ]
    for prompt in prompts:
        model_output = await agent_loop(prompt=prompt, MAX_ATTEMPTS=10) 
        print(model_output)
asyncio.run(main())