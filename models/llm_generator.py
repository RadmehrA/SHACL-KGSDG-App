# # models/llm_generator.py
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY



# # Initialize OpenRouter client
# # client = OpenAI(
# #     base_url="https://openrouter.ai/api/v1",
# #     api_key="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",  # Optional: move to env later
# # )

# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # ... rest of your code ...


# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     return path.split("/")[-1]

# def generate_llm_data(path: str, datatype: str) -> str:
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)

#     prompt = f"Generate a realistic example of {readable_type} for the field '{field_name}'. Return only the value. Avoid repetitive values."

#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=60
#         )
#         content = response.choices[0].message.content.strip()
#         value = content.split("\n")[0].strip().strip('"')
#         return value
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate value for {path}: {e}")
#         return "ExampleValue"


# # models/llm_generator.py
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY


# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # ... rest of your code ...


# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     return path.split("/")[-1]

# # def generate_llm_data(path: str, datatype: str) -> str:
# #     readable_type = DATATYPE_MAP.get(datatype, "text")
# #     field_name = simplify_key(path)

# #     prompt = f"Generate a realistic example of {readable_type} for the field '{field_name}'. Return only the value."

# #     try:
# #         response = client.chat.completions.create(
# #             extra_headers={
# #                 "HTTP-Referer": "<YOUR_SITE_URL>",
# #                 "X-Title": "<YOUR_SITE_NAME>",
# #             },
# #             model="mistralai/mistral-7b-instruct",
# #             messages=[{"role": "user", "content": prompt}],
# #             temperature=0.7,
# #             max_tokens=60
# #         )
# #         content = response.choices[0].message.content.strip()
# #         value = content.split("\n")[0].strip().strip('"')
# #         return value
# #     except Exception as e:
# #         print(f"[LLM ERROR] Could not generate value for {path}: {e}")
# #         return "ExampleValue"


# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)

#     # Modify the prompt to include the user interactive message
#     prompt = f"Generate a realistic example of {readable_type} for the field '{field_name}'. Return only the value. {user_interactive_message}."

#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="openai/gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.6,
#             max_tokens=60,
#             n=5
#         )
#         content = response.choices[0].message.content.strip()
#         value = content.split("\n")[0].strip().strip('"')
#         return value
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate value for {path}: {e}")
#         return "ExampleValue"



# # models/llm_generator.py
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory cache
# LLM_CACHE = {}

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last key segment.
#     """
#     return path.split("/")[-1]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with caching.
#     """

#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)

#     # Create the prompt
#     prompt = f"Generate a realistic example of {readable_type} for the field '{field_name}'. Return only the value. {user_interactive_message}."

#     # --- Check if we already have cached results
#     if prompt in LLM_CACHE:
#         cached_values = LLM_CACHE[prompt]
#         if cached_values:
#             return cached_values.pop()

#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="openai/gpt-4",  
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.8,  # Higher temp = more randomness
#             max_tokens=60,
#             n=5  # Ask for 5 different samples
#         )

#         # Save all new responses into cache
#         generated_values = []
#         for choice in response.choices:
#             value = choice.message.content.strip().split("\n")[0].strip().strip('"')
#             generated_values.append(value)
        
#         # Store into cache
#         LLM_CACHE[prompt] = generated_values

#         return LLM_CACHE[prompt].pop()

#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate value for {path}: {e}")
#         return "ExampleValue"


# # models/llm_generator.py
# import random
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory cache
# LLM_CACHE = {}

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last segment.
#     """
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a slightly randomized prompt to encourage diverse outputs.
#     """
#     variations = [
#         f"Generate a realistic example of {readable_type} for the field '{field_name}'. {user_message}. Return only the value.",
#         f"Provide a realistic {readable_type} value for '{field_name}'. {user_message}. Output only the value.",
#         f"Give an example {readable_type} for '{field_name}'. {user_message}. No explanation, only the value.",
#     ]
#     return random.choice(variations)

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Fetch multiple samples from the LLM for a given prompt.
#     """
#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.9,  # Increased randomness
#             max_tokens=60,
#             n=5  # Request multiple variations
#         )

#         generated_values = []
#         for choice in response.choices:
#             value = choice.message.content.strip().split("\n")[0].strip('"')
#             generated_values.append(value)

#         return generated_values
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation and caching.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     # Pop one value
#     return LLM_CACHE[prompt].pop()



# import random
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory cache
# LLM_CACHE = {}

# # Datatype mappings
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a strongly guided prompt to enforce clean value outputs.
#     """
#     variations = [
#         f"Generate ONLY a realistic {readable_type} value for the field '{field_name}'. No full sentences. No lists. Output just the value itself. {user_message}",
#         f"Provide ONLY a {readable_type} value for '{field_name}'. NO explanation, NO list. Just a raw value. {user_message}",
#         f"Output one realistic {readable_type} for '{field_name}' without any sentence, only the value itself. {user_message}",
#     ]
#     return random.choice(variations)

# def clean_value(value: str) -> str:
#     """
#     Basic cleaning of LLM outputs to enforce clean values.
#     """
#     value = value.strip().strip('"').strip("'")
#     value = value.replace("The value is ", "")
#     value = value.replace("Here is a value:", "")
#     value = value.replace("Example:", "")
#     value = value.replace(".", "")
#     value = value.replace(" is", "")  # Fix "John hasAge 34."
#     value = value.replace("spoken in", "").strip()

#     if not value:
#         return "ExampleValue"

#     return value

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Call LLM API and return multiple diverse outputs.
#     """
#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=1.0,          # Higher diversity
#             max_tokens=60,
#             n=1                       # Get multiple completions
#         )
#         choices = [choice.message.content.strip() for choice in response.choices]
#         return choices
#     except Exception as e:
#         print(f"[LLM ERROR] Failed to generate for prompt '{prompt}': {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Main generator function: returns a clean value for the requested field.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     raw_value = LLM_CACHE[prompt].pop()
#     return clean_value(raw_value)


# # models/llm_generator.py
# import random
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory caches
# LLM_CACHE = {}  # Per prompt
# RECENT_RESPONSES = set()  # Across prompts
# RECENT_CACHE_LIMIT = 50  # How many recent outputs to track

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last segment.
#     """
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a slightly randomized prompt to encourage diverse outputs.
#     """
#     variations = [
#         f"Generate a realistic example of {readable_type} for the field '{field_name}'. {user_message}. Return only the value.",
#         f"Provide a realistic {readable_type} value for '{field_name}'. {user_message}. Output only the value.",
#         f"Give an example {readable_type} for '{field_name}'. {user_message}. No explanation, only the value.",
#     ]
#     return random.choice(variations)

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Fetch multiple samples from the LLM for a given prompt.
#     """
#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.9,  # Increased randomness
#             max_tokens=60,
#             n=10  # Request multiple variations
#         )

#         generated_values = []
#         for choice in response.choices:
#             value = choice.message.content.strip().split("\n")[0].strip('"')
#             generated_values.append(value)

#         return generated_values
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     attempt = 0
#     while attempt < 5:  # Try a few times to get a unique value
#         if not LLM_CACHE[prompt]:
#             # Auto-refill if we run out
#             LLM_CACHE[prompt] = fetch_from_llm(prompt)

#         value = LLM_CACHE[prompt].pop()

#         if value not in RECENT_RESPONSES:
#             # Update recent responses
#             RECENT_RESPONSES.add(value)
#             if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
#                 # Remove oldest item (not perfect but good enough)
#                 RECENT_RESPONSES.pop()
#             return value

#         attempt += 1

#     # If no unique value after several tries, return the last tried one
#     return value


# # models/llm_generator.py
# import random
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
# from collections import defaultdict, deque

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory caches
# LLM_CACHE = {}  # Per prompt
# RECENT_RESPONSES = set()  # Across prompts
# RECENT_CACHE_LIMIT = 50  # How many recent outputs to track

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last segment.
#     """
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a standard prompt for LLM.
#     """
#     return f"Generate a realistic example of {readable_type} for the field '{field_name}'. {user_message}. Return only the value."

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Fetch multiple samples from the LLM for a given prompt.
#     """
#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.9,  # Increased randomness
#             max_tokens=60,
#             n=5  # Request multiple variations
#         )

#         generated_values = []
#         for choice in response.choices:
#             value = choice.message.content.strip().split("\n")[0].strip('"')
#             generated_values.append(value)

#         return generated_values
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     attempt = 0
#     while attempt < 5:  # Try a few times to get a unique value
#         if not LLM_CACHE[prompt]:
#             # Auto-refill if we run out
#             LLM_CACHE[prompt] = fetch_from_llm(prompt)

#         value = LLM_CACHE[prompt].pop()

#         if value not in RECENT_RESPONSES:
#             # Update recent responses
#             RECENT_RESPONSES.add(value)
#             if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
#                 # Remove oldest item (not perfectly FIFO but simple)
#                 RECENT_RESPONSES.pop()
#             return value

#         attempt += 1

#     # If no unique value after several tries, return the last tried one
#     return value


# # models/llm_generator.py
# import random
# from openai import OpenAI
# from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
# from collections import defaultdict, deque

# # Initialize OpenAI client
# client = OpenAI(
#     base_url=OPENROUTER_BASE_URL,
#     api_key=OPENROUTER_API_KEY,
# )

# # Simple in-memory caches
# LLM_CACHE = {}  # Per prompt
# RECENT_RESPONSES_HISTORY = deque(maxlen=5)  # Tracks the last 5 unique responses
# RECENT_RESPONSES = set()  # Across prompts
# RECENT_CACHE_LIMIT = 50  # How many recent outputs to track

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last segment.
#     """
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a standard prompt for LLM.
#     """
#     return f"Generate a realistic example of {readable_type} for the field '{field_name}'. {user_message}. Return only the value."

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Fetch multiple samples from the LLM for a given prompt.
#     """
#     try:
#         response = client.chat.completions.create(
#             extra_headers={
#                 "HTTP-Referer": "<YOUR_SITE_URL>",
#                 "X-Title": "<YOUR_SITE_NAME>",
#             },
#             model="mistralai/mistral-7b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.9,  # Increased randomness
#             max_tokens=60,
#             n=3  # Request multiple variations
#         )

#         generated_values = []
#         for choice in response.choices:
#             value = choice.message.content.strip().split("\n")[0].strip('"')
#             generated_values.append(value)

#         return generated_values
#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     attempt = 0
#     while attempt < 5:  # Try a few times to get a unique value
#         if not LLM_CACHE[prompt]:
#             # Auto-refill if we run out
#             LLM_CACHE[prompt] = fetch_from_llm(prompt)

#         value = LLM_CACHE[prompt].pop()

#         # Check if the value is not in the last 5 responses
#         if value not in RECENT_RESPONSES and value not in RECENT_RESPONSES_HISTORY:
#             # Update recent responses
#             RECENT_RESPONSES.add(value)
#             if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
#                 # Remove oldest item (not perfectly FIFO but simple)
#                 RECENT_RESPONSES.pop()

#             # Add the value to the history of recent responses
#             RECENT_RESPONSES_HISTORY.append(value)
#             return value

#         attempt += 1

#     # If no unique value after several tries, return the last tried one
#     return value



# # models/llm_generator.py
# import random
# import requests
# from config import GROQ_BASE_URL, GROQ_API_KEY
# from collections import defaultdict, deque

# # Simple in-memory caches
# LLM_CACHE = {}  # Per prompt
# RECENT_RESPONSES_HISTORY = deque(maxlen=5)  # Tracks the last 5 unique responses
# RECENT_RESPONSES = set()  # Across prompts
# RECENT_CACHE_LIMIT = 50  # How many recent outputs to track

# # Datatype mappings for readability
# DATATYPE_MAP = {
#     "http://www.w3.org/2001/XMLSchema#string": "text",
#     "http://www.w3.org/2001/XMLSchema#integer": "integer",
#     "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
#     "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
#     "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
#     "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# }

# def simplify_key(path: str) -> str:
#     """
#     Simplify a path to get the last segment.
#     """
#     return path.split("/")[-1]

# def generate_prompt(field_name: str, readable_type: str, user_message: str) -> str:
#     """
#     Create a standard prompt for LLM.
#     """
#     return f"Please randomly generate an interesting realistic example of {readable_type} for the field '{field_name}'. {user_message}. only return the value no extra explanation."

# def fetch_from_llm(prompt: str) -> list:
#     """
#     Fetch multiple samples from the LLM for a given prompt using Groq API.
#     """
#     url = f"{GROQ_BASE_URL}/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": "meta-llama/llama-4-scout-17b-16e-instruct",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.9,  # Increased randomness
#         "max_tokens": 60,
#         "n": 1  # Request multiple variations
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         response_json = response.json()

#         generated_values = []
#         for choice in response_json.get("choices", []):
#             value = choice["message"]["content"].strip().split("\n")[0].strip('"')
#             generated_values.append(value)

#         return generated_values

#     except Exception as e:
#         print(f"[LLM ERROR] Could not generate for prompt {prompt}: {e}")
#         return ["ExampleValue"]

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     attempt = 0
#     while attempt < 5:  # Try a few times to get a unique value
#         if not LLM_CACHE[prompt]:
#             # Auto-refill if we run out
#             LLM_CACHE[prompt] = fetch_from_llm(prompt)

#         value = LLM_CACHE[prompt].pop()

#         # Check if the value is not in the last 5 responses
#         if value not in RECENT_RESPONSES and value not in RECENT_RESPONSES_HISTORY:
#             # Update recent responses
#             RECENT_RESPONSES.add(value)
#             if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
#                 # Remove oldest item (not perfectly FIFO but simple)
#                 RECENT_RESPONSES.pop()

#             # Add the value to the history of recent responses
#             RECENT_RESPONSES_HISTORY.append(value)
#             return value

#         attempt += 1

#     # If no unique value after several tries, return the last tried one
#     return value





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

# def generate_llm_data(path: str, datatype: str, user_interactive_message: str) -> str:
#     """
#     Generate synthetic data using LLM with dynamic prompt variation, caching, and duplicate checking.
#     """
#     readable_type = DATATYPE_MAP.get(datatype, "text")
#     field_name = simplify_key(path)
#     prompt = generate_prompt(field_name, readable_type, user_interactive_message)

#     if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
#         # Fetch fresh values if cache is empty
#         LLM_CACHE[prompt] = fetch_from_llm(prompt)

#     attempt = 0
#     while attempt < 5:  # Try a few times to get a unique value
#         if not LLM_CACHE[prompt]:
#             # Auto-refill if we run out
#             LLM_CACHE[prompt] = fetch_from_llm(prompt)

#         value = LLM_CACHE[prompt].pop()

#         # Check if the value is not in the last 5 responses
#         if value not in RECENT_RESPONSES and value not in RECENT_RESPONSES_HISTORY:
#             # Update recent responses
#             RECENT_RESPONSES.add(value)
#             if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
#                 # Remove oldest item (not perfectly FIFO but simple)
#                 RECENT_RESPONSES.pop()

#             # Add the value to the history of recent responses
#             RECENT_RESPONSES_HISTORY.append(value)
#             return value

#         attempt += 1

#     # If no unique value after several tries, return the last tried one
#     return value


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
