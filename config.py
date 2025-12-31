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
    "AIDC-AI",
    "ai21labs",
    "Alibaba-NLP",
    "allenai",
    "apple",
    "arcee-ai",
    "BAAI",
    "baidu",
    "black-forest-labs",
    "briaai",
    "browser-use",
    "ByteDance",
    "ChatDOC",
    "codelion",
    "CohereLabs",
    "coqui",
    "datalab-to",
    "deepseek-ai",
    "eagerworks",
    "echo840",
    "EssentialAI",
    "facebook",
    "Falconsai",
    "FunAudioLLM",
    "google",
    "GritLM",
    "hexgrad",
    "HuggingFaceTB",
    "ibm-granite",
    "ibm-research",
    "inclusionAI",
    "IndexTeam",
    "intfloat",
    "IQuestLab",
    "jinaai",
    "Lajavaness",
    "lightonai",
    "Linq-AI-Research",
    "LiquidAI",
    "Maincode",
    "marin-community",
    "meta-llama",
    "microsoft",
    "MiniMaxAI",
    "mistralai",
    "moondream",
    "nanonets",
    "nomic-ai",
    "NousResearch",
    "nvidia",
    "openai",
    "openbmb",
    "opendatalab",
    "OpenGVLab",
    "PaddlePaddle",
    "pyannote",
    "Qwen",
    "rednote-hilab",
    "reducto",
    "ResembleAI",
    "Salesforce",
    "sentence-transformers",
    "ServiceNow-AI",
    "skt",
    "Skywork",
    "stepfun-ai",
    "tanaos",
    "tencent",
    "TomoroAI",
    "Tongyi-MAI",
    "Tongyi-Zhiwen",
    "utter-project",
    "vidore",
    "Wan-AI",
    "XiaomiMiMo",
    "zai-org"
}

def _parse_csv_set(val):
    if not val:
        return set()
    return {x.strip().lower() for x in val.split(",") if x.strip()}

# ENV Overrides
BLACKLIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_BLACKLIST"))
WHITELIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_WHITELIST"))
