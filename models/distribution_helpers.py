import random
from typing import Dict, Any, List

def generate_normal_distribution(mean: float, stddev: float, num_values: int) -> list:
    return [random.gauss(mean, stddev) for _ in range(num_values)]

def generate_uniform_distribution(low: float, high: float, num_values: int) -> list:
    return [random.uniform(low, high) for _ in range(num_values)]

def generate_skewed_distribution(low: float, high: float, num_values: int, custom_param: str) -> list:
    values = [random.uniform(low, high) for _ in range(num_values)]
    # Skewness logic can be added based on custom_param
    return values

# Add other distribution functions as needed
DIST_NS = "http://example.org/distribution#"

def extract_distribution_info(constraints: List[Dict[str, str]]) -> Dict[str, any]:
    dist_info = {}
    for c in constraints:
        for key, val in c.items():
            if key.startswith(DIST_NS):
                short_key = key[len(DIST_NS):]  # e.g., "distribution", "categories", "mean"
                # For categories, you may want to parse the RDF list - for now store raw string
                dist_info[short_key] = val
    return dist_info
