# import numpy as np

# def generate_sample_from_distribution(distribution_type: str, parameters: dict):
#     """Generate synthetic sample based on the specified distribution"""
#     if distribution_type == "uniform":
#         return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
#     elif distribution_type == "normal":
#         return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
#     # Add more distributions as needed
#     return None


# def generate_sample_from_distribution(distribution_type: str, parameters: dict, path: str = None):
#     """Generate synthetic sample based on the specified distribution"""
#     if distribution_type == "uniform":
#         return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
#     elif distribution_type == "normal":
#         return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
#     return generate_integer_data(path)


# def generate_sample_from_distribution(distribution_type: str, parameters: dict, path: str = None):
#     """Generate synthetic sample based on the specified distribution"""
#     if distribution_type == "uniform":
#         return np.random.uniform(parameters.get("min", 0), parameters.get("max", 100))
#     elif distribution_type == "normal":
#         return np.random.normal(parameters.get("mean", 50), parameters.get("std_dev", 10))
#     return generate_integer_data(path)


import random
from typing import Dict, Any

def generate_normal_distribution(mean: float, stddev: float, num_values: int) -> list:
    return [random.gauss(mean, stddev) for _ in range(num_values)]

def generate_uniform_distribution(low: float, high: float, num_values: int) -> list:
    return [random.uniform(low, high) for _ in range(num_values)]

def generate_skewed_distribution(low: float, high: float, num_values: int, custom_param: str) -> list:
    values = [random.uniform(low, high) for _ in range(num_values)]
    # Skewness logic can be added based on custom_param
    return values

# Add other distribution functions as needed
