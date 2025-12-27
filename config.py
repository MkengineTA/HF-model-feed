import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
DB_PATH = os.getenv("DB_PATH", "models.db")

# Parameter thresholds
MAX_PARAMS_BILLIONS = 10.0
