import requests
MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
STREAM = False
async def generate_prompt(prompt, system_prompt, temperature=None, model=MODEL, stream=STREAM):
    payload = {
    "model": model,
    "system": system_prompt,
    "prompt": prompt,
    "stream": stream, 
    "options": {
        "temperature": temperature
    }
    }
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
    pythonStr = response.json()['response']
    return pythonStr