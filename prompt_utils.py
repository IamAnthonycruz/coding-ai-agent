

def get_coding_prompt(task, libraries=None, context=None, errors=None):
    coding_prompt = f"Task: {task}\n"
    if libraries:
        coding_prompt += f"Using: {', '.join(libraries)}\n"
    if context:
        coding_prompt += f"Context: {context}\n"
    if errors:
        coding_prompt += f"Errors: {errors}\n"    
    return coding_prompt

def get_task_system_prompt(prompt:str):
    system_prompt = f"""Create a 3-5 step plan to accomplish this
Task: {prompt}"""
    return system_prompt