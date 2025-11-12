import random
import numpy as np

def generate_string_data() -> str:
    """Generate synthetic string data (e.g., names)"""
    return random.choice(["John", "Alice", "Bob", "Charlie"])

def generate_integer_data() -> int:
    """Generate synthetic integer data (e.g., age)"""
    return random.randint(20, 80)

def generate_distribution_data(distribution_type: str, parameters: dict):
    """Generate data based on a specified distribution"""
    if distribution_type == "uniform":
        return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
    elif distribution_type == "normal":
        return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
    # Add more distribution types as needed
    return None
