
from agent_tools import run_code
from apis.ollama_api import generate_prompt
from prompt_utils import get_coding_prompt
from agent_system_prompts import coding_system_prompt, code_fix_prompt

async def code_worker(code: str, max_retries: int=10):
    output, res_code = run_code(code)
    retries = 0
    PREV = [code]
    while(res_code == 1 and retries < max_retries):
        new_prompt = get_coding_prompt(f"Fix this code {PREV[-1]}", None, None, f"{output} fix ")
        code  = await generate_prompt(new_prompt, code_fix_prompt)
        print(code)
        PREV.append(code)
        output, res_code = run_code(PREV[-1], is_code_fixer=True)
        print(output, res_code)
        retries+=1
    return output