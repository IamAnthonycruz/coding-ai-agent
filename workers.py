
from agent_tools import run_code
from apis.ollama_api import generate_prompt
from prompt_utils import get_coding_prompt
from agent_system_prompts import (
    coding_system_prompt,
    code_fix_prompt,
    instruction_system_prompt,
)

async def code_worker(prompt: str, max_retries: int = 100):
    code_centered_prompt = await generate_prompt(
        prompt,
        instruction_system_prompt
    )

    python_code = await generate_prompt(
        code_centered_prompt,
        coding_system_prompt
    )

    code = python_code
    output, res_code = run_code(code, is_code_fixer=True)

    retries = 0

    while res_code == 1 and retries < max_retries:
        new_prompt = get_coding_prompt(
            f"fix this python code:\n{code}\n",
            errors=f"errors:\n{output}"
        )

        code = await generate_prompt(new_prompt, code_fix_prompt)
        output, res_code = run_code(code, is_code_fixer=True)

        retries += 1

    return code
