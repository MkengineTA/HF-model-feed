from __future__ import annotations
import threading
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
BASE_BLACKLIST = frozenset(normalize_namespace(x) for x in _BL)
BASE_WHITELIST = frozenset(normalize_namespace(x) for x in _WL)
DYNAMIC_BLACKLIST: set[str] = set()
DYNAMIC_WHITELIST: set[str] = set()

BLACKLIST = set(BASE_BLACKLIST)
WHITELIST = set(BASE_WHITELIST)
_BLACKLIST_LOCK = threading.Lock()
_WHITELIST_LOCK = threading.Lock()


def set_dynamic_blacklist(namespaces: set[str] | list[str] | None) -> None:
    global DYNAMIC_BLACKLIST, BLACKLIST
    normalized: set[str] = set()
    for ns in namespaces or []:
        key = normalize_namespace(ns)
        if key:
            normalized.add(key)
    with _BLACKLIST_LOCK:
        DYNAMIC_BLACKLIST = normalized
        BLACKLIST = BASE_BLACKLIST | DYNAMIC_BLACKLIST


def get_dynamic_blacklist() -> set[str]:
    with _BLACKLIST_LOCK:
        return set(DYNAMIC_BLACKLIST)


def get_blacklist() -> set[str]:
    with _BLACKLIST_LOCK:
        return set(BLACKLIST)


def get_whitelist() -> set[str]:
    with _WHITELIST_LOCK:
        return set(WHITELIST)


def get_base_blacklist() -> set[str]:
    return set(BASE_BLACKLIST)


def set_dynamic_whitelist(namespaces: set[str] | list[str] | None) -> None:
    global DYNAMIC_WHITELIST, WHITELIST
    normalized: set[str] = set()
    for ns in namespaces or []:
        key = normalize_namespace(ns)
        if key:
            normalized.add(key)
    with _WHITELIST_LOCK:
        DYNAMIC_WHITELIST = normalized
        WHITELIST = BASE_WHITELIST | DYNAMIC_WHITELIST


def get_dynamic_whitelist() -> set[str]:
    with _WHITELIST_LOCK:
        return set(DYNAMIC_WHITELIST)


def classify_namespace(ns: str) -> Tuple[str, Optional[str]]:
    """
    Returns (decision, reason)
    decision:
      - "allow"  -> proceed
      - "allow_whitelist" -> proceed, whitelisted
      - "deny_blacklist"  -> skip
    
    Note: Blacklist wins if a namespace is in both blacklist and whitelist.
    """
    key = normalize_namespace(ns)

    # Check blacklist FIRST (blacklist wins over whitelist)
    with _BLACKLIST_LOCK:
        if key in BLACKLIST:
            return ("deny_blacklist", "skip:blacklisted_namespace")

    # Check whitelist (base + dynamic)
    with _WHITELIST_LOCK:
        if key in WHITELIST:
            return ("allow_whitelist", "allow:whitelisted_namespace")

    return ("allow", None)
