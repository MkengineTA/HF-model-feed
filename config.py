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

# Email Configuration
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
RECEIVER_MAIL = os.getenv("RECEIVER_MAIL")

# Parameter thresholds
MAX_PARAMS_BILLIONS = 40.0

# Explicit Exclusions (Deprecated by new namespace_policy but kept for backward compatibility if needed)
EXCLUDED_NAMESPACES = {"thireus"}

# Reporting Filters
MIN_SPECIALIST_SCORE = 6
EXCLUDE_REVIEW_REQUIRED = True

# --- Namespace Lists ---

BLACKLIST_NAMESPACES = {
    "ubergarm",
    "unsloth",
    "mradermacher",
    "aaryank",
    "bartowski",
    "mlx-community",
    "noctrex",
    "onnxruntime",
    "lmstudio-community",
    "ggml-org",
    "devquasar",
    "thireus" # Merged old exclusion
}

WHITELIST_NAMESPACES = {
    "jan-hq",
    "janhq",
}

def _parse_csv_set(val):
    if not val:
        return set()
    return {x.strip().lower() for x in val.split(",") if x.strip()}

# ENV Overrides
BLACKLIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_BLACKLIST"))
WHITELIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_WHITELIST"))
