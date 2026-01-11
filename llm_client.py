# llm_client.py
from __future__ import annotations

import json
import logging
import re
import random
import time
import requests
import unicodedata
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("EdgeAIScout")

def extract_json_from_text(text: str) -> Optional[dict]:
    def clean_json(json_text: str) -> str:
        return re.sub(r",\s*([\]}])", r"\1", json_text)

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(clean_json(m.group(1)))
        except json.JSONDecodeError:
            pass

    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        return json.loads(clean_json(text[start_idx : end_idx + 1]))
    except json.JSONDecodeError:
        return None

def _coerce_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: list[str] = []
        for it in x:
            if it is None:
                continue
            out.append(str(it))
        return out
    return [str(x)]

def _split_dash_block(s: str) -> list[str]:
    """
    Fixes cases like: "Was ist neu? - A - B - C"
    or a single string containing multiple '- ' bullets.
    """
    if not s:
        return []
    t = " ".join(s.split()).strip()
    # Remove common prefixes
    t = re.sub(r"^(was ist neu\??|warum relevant\??|use cases\??)\s*:\s*", "", t, flags=re.IGNORECASE).strip()

    # If it looks like "X - A - B - C" split on " - " when multiple parts exist
    parts = [p.strip() for p in re.split(r"\s-\s", t) if p.strip()]
    if len(parts) >= 2:
        # If first part is clearly a label, drop it
        if re.match(r"^(was ist neu|warum relevant|use cases)$", parts[0], re.IGNORECASE):
            parts = parts[1:]
        return parts

    # If it contains newline bullets
    if "\n-" in s or s.strip().startswith("-"):
        lines = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            lines.append(line)
        return lines

    return [t]

def _normalize_bilingual_field(value: Any, fallback_lang: str = "de") -> dict[str, Any]:
    """
    Normalize a field that may be bilingual (dict with de/en) or monolingual.
    
    If the value is already a dict with language keys, return it.
    If it's a string/list (legacy format), put it under the fallback_lang key.
    """
    if isinstance(value, dict) and ("de" in value or "en" in value):
        return value
    # Legacy format - assume fallback language
    return {fallback_lang: value}


def _normalize_bilingual_list(value: Any, fallback_lang: str = "de") -> dict[str, list[str]]:
    """Normalize a list field that may be bilingual."""
    if isinstance(value, dict) and ("de" in value or "en" in value):
        result = {}
        for lang in ["de", "en"]:
            raw = value.get(lang, [])
            normalized = []
            for item in _coerce_list(raw):
                normalized.extend(_split_dash_block(item))
            result[lang] = normalized
        return result
    # Legacy format
    normalized = []
    for item in _coerce_list(value):
        normalized.extend(_split_dash_block(item))
    return {fallback_lang: normalized}


def normalize_llm_output(analysis: dict) -> dict:
    """
    Postprocess the LLM JSON so reporter gets clean lists.
    
    Supports both legacy (monolingual German) format and new bilingual format.
    For bilingual fields, the structure is: {"de": ..., "en": ...}
    """
    if not isinstance(analysis, dict):
        return analysis

    # newsletter_blurb: can be string or {"de": "...", "en": "..."}
    blurb = analysis.get("newsletter_blurb")
    analysis["newsletter_blurb"] = _normalize_bilingual_field(blurb, "de")

    # key_facts: can be list or {"de": [...], "en": [...]}
    kf_raw = analysis.get("key_facts")
    kf_bilingual = _normalize_bilingual_list(kf_raw, "de")
    # Apply truncation to each language
    for lang in kf_bilingual:
        kf_bilingual[lang] = [x[:140] for x in kf_bilingual[lang] if x][:3]
    analysis["key_facts"] = kf_bilingual

    # delta: can be legacy or bilingual per-field
    delta = analysis.get("delta") or {}
    wc_bilingual = _normalize_bilingual_list(delta.get("what_changed"), "de")
    wm_bilingual = _normalize_bilingual_list(delta.get("why_it_matters"), "de")
    # Apply limits
    for lang in wc_bilingual:
        wc_bilingual[lang] = [x for x in wc_bilingual[lang] if x][:3]
    for lang in wm_bilingual:
        wm_bilingual[lang] = [x for x in wm_bilingual[lang] if x][:3]
    analysis["delta"] = {
        "what_changed": wc_bilingual,
        "why_it_matters": wm_bilingual,
    }

    # manufacturing.use_cases: can be list or bilingual
    manu = analysis.get("manufacturing") or {}
    uc_bilingual = _normalize_bilingual_list(manu.get("use_cases"), "de")
    for lang in uc_bilingual:
        uc_bilingual[lang] = [x for x in uc_bilingual[lang] if x][:6]
    manu["use_cases"] = uc_bilingual
    manu.pop("risks", None)
    analysis["manufacturing"] = manu

    # edge: remove deployment_notes
    edge = analysis.get("edge") or {}
    edge.pop("deployment_notes", None)
    analysis["edge"] = edge

    # evidence optional
    if not config.LLM_REQUIRE_EVIDENCE:
        analysis.pop("evidence", None)

    return analysis

class LLMClient:
    def __init__(self, api_url: str, model: str, api_key: str | None = None, site_url: str | None = None, app_name: str | None = None, enable_reasoning: bool = False):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name
        self.enable_reasoning = enable_reasoning

    def _request_with_backoff(self, payload: dict, headers: dict) -> requests.Response:
        """
        Make an API request with robust retry logic.
        
        - For 429 (rate limit) errors: retry indefinitely with exponential backoff
        - For 5xx (server) errors: retry up to 10 times
        - For other errors: raise immediately
        
        Returns the successful response object.
        Raises an exception if unrecoverable error or max retries exceeded.
        """
        base_wait_time = 10  # Initial wait time in seconds
        max_wait_time = 3600  # Maximum wait time per sleep cycle (1 hour)
        max_5xx_retries = 10
        
        attempt = 0
        server_error_count = 0
        
        while True:
            attempt += 1
            
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
                
                # Success - return the response
                if response.status_code == 200:
                    return response
                
                # Rate limit (429) - wait and retry indefinitely
                if response.status_code == 429:
                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    
                    if retry_after:
                        try:
                            # Retry-After can be in seconds (integer) or HTTP-date
                            wait_time = int(retry_after)
                            # Add small buffer
                            wait_time += random.uniform(1, 5)
                        except ValueError:
                            # If it's a date string, fall back to exponential backoff
                            wait_time = min(base_wait_time * (2 ** (attempt - 1)), max_wait_time)
                            # Add jitter
                            wait_time += random.uniform(0, wait_time * 0.1)
                    else:
                        # Exponential backoff with jitter
                        wait_time = min(base_wait_time * (2 ** (attempt - 1)), max_wait_time)
                        # Add jitter to avoid thundering herd
                        wait_time += random.uniform(0, wait_time * 0.1)
                    
                    logger.warning(f"Rate limit hit (429). Sleeping for {wait_time:.1f}s before retry (attempt {attempt})...")
                    time.sleep(wait_time)
                    continue
                
                # Server errors (5xx) - limited retry
                if 500 <= response.status_code < 600:
                    server_error_count += 1
                    
                    if server_error_count >= max_5xx_retries:
                        logger.error(f"Server error {response.status_code} persisted after {max_5xx_retries} retries. Giving up.")
                        response.raise_for_status()
                    
                    # Exponential backoff for server errors
                    wait_time = min(base_wait_time * (2 ** (server_error_count - 1)), max_wait_time)
                    wait_time += random.uniform(0, wait_time * 0.1)
                    
                    logger.warning(f"Server error {response.status_code}. Sleeping for {wait_time:.1f}s before retry ({server_error_count}/{max_5xx_retries})...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors (4xx except 429, etc.) - raise immediately
                response.raise_for_status()
                
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout: {e}")
                raise
            except requests.exceptions.RequestException as e:
                # Connection errors, etc.
                logger.error(f"Request error: {e}")
                raise

    def analyze_model(self, readme_content: str, tags: list[str], yaml_meta: dict | None = None, file_summary: list[dict] | None = None) -> Optional[dict]:
        files_ctx = "Unknown"
        if file_summary:
            exts: dict[str, int] = {}
            for f in file_summary:
                p = f.get("path", "")
                ext = p.split(".")[-1] if "." in p else "bin"
                exts[ext] = exts.get(ext, 0) + 1
            files_ctx = ", ".join([f"{k}: {v}" for k, v in exts.items()])

        system_prompt = (
            "You are a strict analyst for Edge-AI and Manufacturing models. "
            "You MUST NOT invent anything. If information is not clearly documented (README/YAML), "
            "set it to null and list it under unknowns. Confidence decreases with missing evidence. "
            "Output ONLY valid JSON. Provide content in BOTH German (de) and English (en) for text fields."
        )

        evidence_block = ""
        evidence_rules = ""
        if config.LLM_REQUIRE_EVIDENCE:
            evidence_block = """
          "evidence": [
            { "claim": "Short claim", "quote": "Exact quote from README" }
          ],
            """
            evidence_rules = """
        - Evidence requirement: For claims about domain/performance, a quote MUST exist.
        - evidence: exactly 2-4 entries.
            """

        user_prompt = f"""
        Analyze the HuggingFace model.

        METADATA:
        - Tags: {', '.join(tags) if tags else 'None'}
        - YAML Headers: {json.dumps(yaml_meta, indent=2) if yaml_meta else 'None'}
        - Files Summary: {files_ctx}

        README (truncated):
        {readme_content[:32000]}

        LENGTH LIMITS (strict):
        - newsletter_blurb: Concise summary (approx. 80-100 words) in BOTH languages.
        - key_facts: 3 entries, max 140 characters each, in BOTH languages.
        - delta.what_changed / why_it_matters: 3 bullets each, in BOTH languages.
        - manufacturing.use_cases: in BOTH languages.
        {("- evidence: 2-4 entries." if config.LLM_REQUIRE_EVIDENCE else "")}

        RULES:
        - Provide text content in BOTH German ("de") and English ("en").
        - params_m / min_vram_gb: Only if explicitly stated.
        - quantization: Also check filenames (gguf/onnx) if not mentioned in text.
        {evidence_rules}

        JSON OUTPUT FORMAT (bilingual fields use {{"de": "...", "en": "..."}} structure):
        {{
          "model_type": "Base Model" | "LoRA Adapter" | "Finetune",
          "base_model": null,
          "params_m": null,
          "modality": "Text" | "Vision" | "Diffusion" | "Multimodal" | "Other",
          "category": "Inspection" | "VQA" | "Code" | "Extraction" | "Reasoning" | "Architecture" | "Other",
          "newsletter_blurb": {{"de": "Deutsche Zusammenfassung...", "en": "English summary..."}},
          "key_facts": {{"de": ["Fakt 1", "Fakt 2", "Fakt 3"], "en": ["Fact 1", "Fact 2", "Fact 3"]}},
          "delta": {{
            "what_changed": {{"de": ["Änderung 1"], "en": ["Change 1"]}},
            "why_it_matters": {{"de": ["Relevanz 1"], "en": ["Relevance 1"]}}
          }},
          "edge": {{
            "edge_ready": false,
            "min_vram_gb": null,
            "quantization": ["none"]
          }},
          "manufacturing": {{
            "manufacturing_fit_score": 1,
            "use_cases": {{"de": ["Anwendungsfall 1"], "en": ["Use case 1"]}}
          }},
          {evidence_block}
          "specialist_score": 1,
          "confidence": "low" | "medium" | "high",
          "unknowns": ["..."]
        }}
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "stream": False,
        }

        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        try:
            response = self._request_with_backoff(payload, headers)
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            analysis = extract_json_from_text(content)
            if not analysis:
                logger.error(f"JSON Parse Error. Raw:\n{content[:500]}...")
                return None
            return normalize_llm_output(analysis)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None
