

def get_coding_prompt(task, libraries=None, context=None, errors=None):
    coding_prompt = f"Task: {task}\n"
    if libraries:
        coding_prompt += f"Using: {', '.join(libraries)}\n"
    if context:
        coding_prompt += f"Context: {context}\n"
    if errors:
        coding_prompt += f"Errors: {errors}\n"    
    return coding_prompt