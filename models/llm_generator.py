import random
import time
import requests
from collections import defaultdict, deque


LLM_CACHE = {}
RECENT_RESPONSES_HISTORY = deque(maxlen=5)
RECENT_RESPONSES = set()
RECENT_CACHE_LIMIT = 50


DATATYPE_MAP = {
    "http://www.w3.org/2001/XMLSchema#string": "text",
    "http://www.w3.org/2001/XMLSchema#integer": "integer",
    "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
    "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
    "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
    "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
    "http://www.w3.org/ns/shacl#IRI": "IRI (e.g., http://example.org/resource/123)"
    
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
    Fetch multiple samples from a local Ollama LLM.
    """
    
    url = "http://host.docker.internal:11434/api/generate"

    payload = {
        
        "model": "llama3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "num_predict": 60
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_json = response.json()

        text = response_json.get("response", "").strip()
        value = text.split("\n")[0].strip('"')

        return [value]

    except Exception as e:
        print(f"[LLM ERROR - OLLAMA] {e}")
        return ["ExampleValue"]

import time

def generate_llm_data(path: str, datatype: str, user_interactive_message: str = "") -> str:
    """
    Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
    """
    time.sleep(2)

    readable_type = DATATYPE_MAP.get(datatype, "text")
    field_name = simplify_key(path)
    prompt = generate_prompt(field_name, readable_type, user_interactive_message)

    if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
        LLM_CACHE[prompt] = fetch_from_llm(prompt)

    attempt = 0
    while attempt < 5:
        if not LLM_CACHE[prompt]:
            LLM_CACHE[prompt] = fetch_from_llm(prompt)

        value = LLM_CACHE[prompt].pop()

        if value not in RECENT_RESPONSES and value not in RECENT_RESPONSES_HISTORY:
            RECENT_RESPONSES.add(value)
            if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
                RECENT_RESPONSES.pop()

            RECENT_RESPONSES_HISTORY.append(value)
            return value

        attempt += 1

    
    return value
