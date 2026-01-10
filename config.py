# config.py
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("EdgeAIScout")

HF_TOKEN = os.getenv("HF_TOKEN")

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_API_KEY = os.getenv("LLM_API_KEY")  # Optional for local, required for OpenRouter
LLM_ENABLE_REASONING = os.getenv("LLM_ENABLE_REASONING", "False").lower() == "true"

# OpenRouter Specific Headers (Optional)
OR_SITE_URL = os.getenv("OR_SITE_URL", "https://github.com/EdgeAIScout")
OR_APP_NAME = os.getenv("OR_APP_NAME", "Edge AI Scout")

DB_PATH = os.getenv("DB_PATH", "models.db")

# Email Configuration
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
RECEIVER_MAIL = os.getenv("RECEIVER_MAIL")

# --- Parameter thresholds ---
# Skip if EITHER threshold exceeded.
MAX_TOTAL_PARAMS_BILLIONS = float(os.getenv("MAX_TOTAL_PARAMS_BILLIONS", "40.0"))
MAX_ACTIVE_PARAMS_BILLIONS = float(os.getenv("MAX_ACTIVE_PARAMS_BILLIONS", "40.0"))

# If True: try safetensors metadata first (exact when available at repo root)
PARAMS_PREFER_SAFETENSORS_META = os.getenv("PARAMS_PREFER_SAFETENSORS_META", "True").lower() == "true"

# Fallback bytes/param when estimating from file size (common for fp16/bf16 weights)
BYTES_PER_PARAM_FALLBACK = float(os.getenv("BYTES_PER_PARAM_FALLBACK", "2.0"))

# --- Security behavior ---
# Never skip; only warn. You decide.
SECURITY_WARN_ON_HF_SCAN_FLAGS = os.getenv("SECURITY_WARN_ON_HF_SCAN_FLAGS", "True").lower() == "true"
SECURITY_WARN_ON_EXECUTABLES = os.getenv("SECURITY_WARN_ON_EXECUTABLES", "True").lower() == "true"
SECURITY_WARN_ON_SCRIPTS = os.getenv("SECURITY_WARN_ON_SCRIPTS", "False").lower() == "true"

# Explicit Exclusions (Deprecated by new namespace_policy but kept for backward compatibility if needed)
EXCLUDED_NAMESPACES = {"thireus"}

# Reporting Filters
MIN_SPECIALIST_SCORE = int(os.getenv("MIN_SPECIALIST_SCORE", "0"))
EXCLUDE_REVIEW_REQUIRED = os.getenv("EXCLUDE_REVIEW_REQUIRED", "True").lower() == "true"
MODEL_NAME_DUPLICATE_BLOCK_LIMIT = int(os.getenv("MODEL_NAME_DUPLICATE_BLOCK_LIMIT", "3"))

# Evidence usage
# Keep evidence internally to validate LLM claims, but don't show in report by default.
LLM_REQUIRE_EVIDENCE = os.getenv("LLM_REQUIRE_EVIDENCE", "True").lower() == "true"
REPORT_INCLUDE_EVIDENCE = os.getenv("REPORT_INCLUDE_EVIDENCE", "False").lower() == "true"

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
    "thireus",
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
    "zai-org",
    "HIT-TMG",
    "codefuse-ai",
    "upstage",
    "meituan-longcat",
    "PrimeIntellect",
    "CalmState",
    "naver-hyperclovax",
    "NC-AI-consortium-VAETKI",
    "LGAI-EXAONE",
}

def _parse_csv_set(val: str | None) -> set[str]:
    if not val:
        return set()
    return {x.strip().lower() for x in val.split(",") if x.strip()}

# ENV Overrides
BLACKLIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_BLACKLIST"))
WHITELIST_NAMESPACES |= _parse_csv_set(os.getenv("HF_WHITELIST"))

# --- Dynamic blacklist thresholds ---
DYNAMIC_BLACKLIST_NO_README_MIN = int(os.getenv("DYNAMIC_BLACKLIST_NO_README_MIN", "20"))

# --- Newsletter Multi-Recipient Configuration ---
NEWSLETTER_TIMEZONE = os.getenv("NEWSLETTER_TIMEZONE", "Europe/Berlin")

# Supported languages for newsletters
SUPPORTED_LANGUAGES = {"en", "de"}

# Supported recipient types
SUPPORTED_RECIPIENT_TYPES = {"normal", "debug"}

# Valid send days
VALID_SEND_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


@dataclass
class NewsletterSubscriber:
    """Configuration for a newsletter recipient."""
    email: str
    type: Literal["normal", "debug"] = "normal"
    language: Literal["en", "de"] = "en"
    send_days: List[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    default_window_hours: int = 24
    window_hours_by_day: Dict[str, int] = field(default_factory=dict)
    
    def get_window_hours_for_day(self, day: str) -> int:
        """Get the window hours for a specific day, falling back to default."""
        return self.window_hours_by_day.get(day.lower(), self.default_window_hours)


def _parse_subscribers_json(json_str: str | None) -> List[NewsletterSubscriber]:
    """Parse NEWSLETTER_SUBSCRIBERS_JSON into list of NewsletterSubscriber objects."""
    if not json_str:
        return []
    
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            logger.warning("NEWSLETTER_SUBSCRIBERS_JSON must be a JSON array; ignoring.")
            return []
        
        subscribers = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(f"Subscriber {i} is not an object; skipping.")
                continue
            
            email = item.get("email")
            if not email or not isinstance(email, str):
                logger.warning(f"Subscriber {i} missing or invalid 'email'; skipping.")
                continue
            
            # Validate type
            sub_type = item.get("type", "normal")
            if sub_type not in SUPPORTED_RECIPIENT_TYPES:
                logger.warning(f"Subscriber {email}: invalid type '{sub_type}'; defaulting to 'normal'.")
                sub_type = "normal"
            
            # Validate language
            language = item.get("language", "en")
            if language not in SUPPORTED_LANGUAGES:
                logger.warning(f"Subscriber {email}: invalid language '{language}'; defaulting to 'en'.")
                language = "en"
            
            # Validate send_days
            send_days_raw = item.get("send_days", ["mon", "tue", "wed", "thu", "fri"])
            if not isinstance(send_days_raw, list):
                logger.warning(f"Subscriber {email}: 'send_days' must be a list; using default.")
                send_days_raw = ["mon", "tue", "wed", "thu", "fri"]
            send_days = [d.lower() for d in send_days_raw if isinstance(d, str) and d.lower() in VALID_SEND_DAYS]
            if not send_days:
                logger.warning(f"Subscriber {email}: no valid send_days; using default weekdays.")
                send_days = ["mon", "tue", "wed", "thu", "fri"]
            
            # Validate default_window_hours
            default_window_hours = item.get("default_window_hours", 24)
            if not isinstance(default_window_hours, int) or default_window_hours < 1:
                logger.warning(f"Subscriber {email}: invalid default_window_hours; using 24.")
                default_window_hours = 24
            
            # Validate window_hours_by_day
            window_hours_by_day_raw = item.get("window_hours_by_day", {})
            if not isinstance(window_hours_by_day_raw, dict):
                logger.warning(f"Subscriber {email}: 'window_hours_by_day' must be an object; ignoring.")
                window_hours_by_day_raw = {}
            window_hours_by_day = {}
            for day, hours in window_hours_by_day_raw.items():
                day_lower = day.lower()
                if day_lower not in VALID_SEND_DAYS:
                    logger.warning(f"Subscriber {email}: invalid day '{day}' in window_hours_by_day; skipping.")
                    continue
                if not isinstance(hours, int) or hours < 1:
                    logger.warning(f"Subscriber {email}: invalid hours for day '{day}'; skipping.")
                    continue
                window_hours_by_day[day_lower] = hours
            
            subscribers.append(NewsletterSubscriber(
                email=email,
                type=sub_type,
                language=language,
                send_days=send_days,
                default_window_hours=default_window_hours,
                window_hours_by_day=window_hours_by_day,
            ))
        
        return subscribers
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse NEWSLETTER_SUBSCRIBERS_JSON: {e}")
        return []


def get_newsletter_subscribers() -> List[NewsletterSubscriber]:
    """
    Get newsletter subscribers from configuration.
    
    If NEWSLETTER_SUBSCRIBERS_JSON is set, parse and return those subscribers.
    Otherwise, fall back to legacy single-recipient configuration (RECEIVER_MAIL).
    """
    subscribers_json = os.getenv("NEWSLETTER_SUBSCRIBERS_JSON")
    
    if subscribers_json:
        subscribers = _parse_subscribers_json(subscribers_json)
        if subscribers:
            return subscribers
        # If parsing failed or resulted in empty list, fall back to legacy
    
    # Legacy fallback: use RECEIVER_MAIL as a single debug recipient
    if RECEIVER_MAIL:
        return [NewsletterSubscriber(
            email=RECEIVER_MAIL,
            type="debug",
            language="de",  # Legacy behavior was German
            send_days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            default_window_hours=24,
        )]
    
    return []
