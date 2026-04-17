"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Groq API key - get one free at https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Council members - list of Groq model identifiers (currently available in production)
COUNCIL_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "openai/gpt-oss-120b"

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
