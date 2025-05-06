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

# import random
# import numpy as np

# def generate_string_data(path: str = None) -> str:
#     """Generate synthetic string data (e.g., names, street, etc.)"""
#     sample_strings = ["John", "Alice", "Main Street", "Zebra Crossing", "Charlie"]
#     return random.choice(sample_strings)
# import random

# def generate_integer_data(path: str = None, min_val: int = 0, max_val: int = 100) -> int:
#     """Generate synthetic integer data (e.g., age, quantity)"""
#     return random.randint(min_val, max_val)

# def generate_distribution_data(distribution_type: str, parameters: dict, path: str = None):
#     """Generate data based on a specified distribution"""
#     if distribution_type == "uniform":
#         return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
#     elif distribution_type == "normal":
#         return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
#     return generate_integer_data(path)  # Fallback

# import random
# import numpy as np
# import datetime

# def generate_string_data(path: str = None) -> str:
#     """Generate synthetic string data (e.g., names, street, etc.)"""
#     sample_strings = ["John", "Alice", "Main Street", "Zebra Crossing", "Charlie"]
#     return random.choice(sample_strings)

# def generate_integer_data(path: str = None, min_val: int = 0, max_val: int = 100) -> int:
#     """Generate synthetic integer data (e.g., age, quantity)"""
#     return random.randint(min_val, max_val)

# def generate_boolean_data() -> bool:
#     """Generate random boolean value"""
#     return random.choice([True, False])

# def generate_date_data() -> str:
#     """Generate a random date string"""
#     return str(datetime.datetime.now().date())

# def generate_distribution_data(distribution_type: str, parameters: dict, path: str = None):
#     """Generate data based on a specified distribution"""
#     if distribution_type == "uniform":
#         return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
#     elif distribution_type == "normal":
#         return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
#     return generate_integer_data(path)  # Fallback
