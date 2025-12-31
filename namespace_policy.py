from __future__ import annotations
from typing import Optional, Tuple

from config import BLACKLIST_NAMESPACES, WHITELIST_NAMESPACES

def normalize_namespace(ns: str | None) -> str:
    return (ns or "").strip().lower()

def classify_namespace(ns: str) -> Tuple[str, Optional[str]]:
    """
    Returns (decision, reason)
    decision:
      - "allow"  -> proceed
      - "allow_whitelist" -> proceed, whitelisted
      - "deny_blacklist"  -> skip
    """
    key = normalize_namespace(ns)

    if key in WHITELIST_NAMESPACES:
        return ("allow_whitelist", None)

    if key in BLACKLIST_NAMESPACES:
        return ("deny_blacklist", "skip:blacklisted_namespace")

    return ("allow", None)
