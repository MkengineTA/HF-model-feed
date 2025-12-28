import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_API_KEY = os.getenv("LLM_API_KEY") # Optional for local, required for OpenRouter
LLM_ENABLE_REASONING = os.getenv("LLM_ENABLE_REASONING", "False").lower() == "true"

# OpenRouter Specific Headers (Optional)
OR_SITE_URL = os.getenv("OR_SITE_URL", "https://github.com/EdgeAIScout")
OR_APP_NAME = os.getenv("OR_APP_NAME", "Edge AI Scout")

DB_PATH = os.getenv("DB_PATH", "models.db")

# Parameter thresholds
MAX_PARAMS_BILLIONS = 10.0
