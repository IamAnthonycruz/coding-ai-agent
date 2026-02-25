import random

def generate_random_numbers() -> list[int]:
    return [random.randint(1, 100) for _ in range(100)]