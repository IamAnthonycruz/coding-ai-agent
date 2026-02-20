import random

def generate_random_numbers() -> list[int]:
    """Generate a list of 100 random numbers between 1 and 1000."""
    return [random.randint(1, 1000) for _ in range(100)]