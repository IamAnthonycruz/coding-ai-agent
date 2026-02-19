import random

def generate_random_numbers() -> List[int]:
    numbers = []
    for _ in range(100):
        numbers.append(random.randint(1, 100))
    return numbers