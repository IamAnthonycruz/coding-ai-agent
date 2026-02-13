

from apis.ollama_api import generate_prompt

from system_prompts import tool_system_prompt
from tools import TOOLS


MAX_ATTEMPTS = 10
user_request = 'print()'
hi = tool_system_prompt(user_request=user_request)

pythonCode = generate_prompt("l", hi)

print(pythonCode)

