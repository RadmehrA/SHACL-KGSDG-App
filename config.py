# config.py
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
