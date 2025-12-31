from __future__ import annotations
from typing import Optional, Tuple

from config import BLACKLIST_NAMESPACES as _BL, WHITELIST_NAMESPACES as _WL

def normalize_namespace(ns: str | None) -> str:
    s = (ns or "").strip().lower()

    # URL -> path
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1].strip("/")

    # repo-id -> namespace
    if "/" in s:
        s = s.split("/", 1)[0].strip()

    return s

# Pre-normalized sets for fast lookup
BLACKLIST = {normalize_namespace(x) for x in _BL}
WHITELIST = {normalize_namespace(x) for x in _WL}

def classify_namespace(ns: str) -> Tuple[str, Optional[str]]:
    """
    Returns (decision, reason)
    decision:
      - "allow"  -> proceed
      - "allow_whitelist" -> proceed, whitelisted
      - "deny_blacklist"  -> skip
    """
    key = normalize_namespace(ns)

    if key in WHITELIST:
        return ("allow_whitelist", "allow:whitelisted_namespace")

    if key in BLACKLIST:
        return ("deny_blacklist", "skip:blacklisted_namespace")

    return ("allow", None)
