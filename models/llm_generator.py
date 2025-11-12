# models/llm_generator.py
import random
import time
import requests
from config import GROQ_BASE_URL, GROQ_API_KEY
from collections import defaultdict, deque

# Simple in-memory caches
LLM_CACHE = {}  # Per prompt
RECENT_RESPONSES_HISTORY = deque(maxlen=5)  # Tracks the last 5 unique responses
RECENT_RESPONSES = set()  # Across prompts
RECENT_CACHE_LIMIT = 50  # How many recent outputs to track

# Datatype mappings for readability
DATATYPE_MAP = {
    "http://www.w3.org/2001/XMLSchema#string": "text",
    "http://www.w3.org/2001/XMLSchema#integer": "integer",
    "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
    "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
    "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
    "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
}

def simplify_key(path: str) -> str:
    """
    Simplify a path to get the last segment.
    """
    return path.split("/")[-1]

def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
    """
    Create a standard prompt for LLM.
    """
    return f"Please randomly generate an interesting realistic example of {readable_type} for the field '{field_name}'. {user_message}. only return the value and ignore has."

def fetch_from_llm(prompt: str) -> list:
    """
    Fetch multiple samples from the LLM for a given prompt using Groq API.
    """
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,  # Increased randomness
        "max_tokens": 60,
        "n": 1  # Request multiple variations
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        generated_values = []
        for choice in response_json.get("choices", []):
            value = choice["message"]["content"].strip().split("\n")[0].strip('"')
            generated_values.append(value)

        return generated_values

    except Exception as e:
        print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
        return ["ExampleValue"]

import time

def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
    """
    Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
    """
    # Wait for 2 seconds before running
    time.sleep(2)

    readable_type = DATATYPE_MAP.get(datatype, "text")
    field_name = simplify_key(path)
    prompt = generate_prompt(field_name, readable_type, user_interactive_message)

    if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
        # Fetch fresh values if cache is empty
        LLM_CACHE[prompt] = fetch_from_llm(prompt)

    attempt = 0
    while attempt < 5:  # Try a few times to get a unique value
        if not LLM_CACHE[prompt]:
            # Auto-refill if we run out
            LLM_CACHE[prompt] = fetch_from_llm(prompt)

        value = LLM_CACHE[prompt].pop()

        # Check if the value is not in the last 5 responses
        if value not in RECENT_RESPONSES and value not in RECENT_RESPONSES_HISTORY:
            # Update recent responses
            RECENT_RESPONSES.add(value)
            if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
                # Remove oldest item (not perfectly FIFO but simple)
                RECENT_RESPONSES.pop()

            # Add the value to the history of recent responses
            RECENT_RESPONSES_HISTORY.append(value)
            return value

        attempt += 1

    # If no unique value after several tries, return the last tried one
    return value
