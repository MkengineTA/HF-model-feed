# llm_client.py
from __future__ import annotations

import json
import logging
import re
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

def normalize_llm_output(analysis: dict) -> dict:
    """
    Postprocess the LLM JSON so reporter gets clean lists.
    """
    if not isinstance(analysis, dict):
        return analysis

    # key_facts must be list[str] (max 3)
    kf = []
    for item in _coerce_list(analysis.get("key_facts")):
        kf.extend(_split_dash_block(item))
    analysis["key_facts"] = [x[:140] for x in kf if x][:3]

    # delta lists
    delta = analysis.get("delta") or {}
    wc = []
    for item in _coerce_list(delta.get("what_changed")):
        wc.extend(_split_dash_block(item))
    wm = []
    for item in _coerce_list(delta.get("why_it_matters")):
        wm.extend(_split_dash_block(item))
    analysis["delta"] = {
        "what_changed": [x for x in wc if x][:3],
        "why_it_matters": [x for x in wm if x][:3],
    }

    # manufacturing.use_cases
    manu = analysis.get("manufacturing") or {}
    uc = []
    for item in _coerce_list(manu.get("use_cases")):
        uc.extend(_split_dash_block(item))
    manu["use_cases"] = [x for x in uc if x][:6]
    # risks removed
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
            "Du bist ein strenger Analyst für Edge-AI- und Manufacturing-Modelle. "
            "Du darfst NICHTS erfinden. Wenn Informationen nicht klar belegt sind (README/YAML), "
            "setze sie auf null und liste sie unter unknowns. Confidence sinkt bei fehlenden Belegen. "
            "Gib AUSSCHLIESSLICH valides JSON aus."
        )

        evidence_block = ""
        evidence_rules = ""
        if config.LLM_REQUIRE_EVIDENCE:
            evidence_block = """
          "evidence": [
            { "claim": "Kurzer Claim", "quote": "Exaktes Zitat aus dem README" }
          ],
            """
            evidence_rules = """
        - Evidence-Pflicht: Für Claims bzgl. Domain/Performance MUSS ein Zitat ("quote") existieren.
        - evidence: genau 2-4 Einträge.
            """

        user_prompt = f"""
        Analysiere das HuggingFace-Modell.

        METADATEN:
        - Tags: {', '.join(tags) if tags else 'None'}
        - YAML Headers: {json.dumps(yaml_meta, indent=2) if yaml_meta else 'None'}
        - Files Summary: {files_ctx}

        README (gekürzt):
        {readme_content[:32000]}

        LÄNGENLIMITS (hart):
        - newsletter_blurb: Prägnante Zusammenfassung (ca. 80-100 Wörter) auf Deutsch.
        - key_facts: 3 Einträge, max 140 Zeichen.
        - delta.what_changed / why_it_matters: je 3 Bullets.
        {("- evidence: 2-4 Einträge." if config.LLM_REQUIRE_EVIDENCE else "")}

        REGELN:
        - Sprache: DEUTSCH (außer Enum-Werte).
        - params_m / min_vram_gb: Nur wenn explizit genannt.
        - quantization: Prüfe auch Filenamen (gguf/onnx) falls im Text nicht genannt.
        {evidence_rules}

        JSON OUTPUT FORMAT:
        {{
          "model_type": "Base Model" | "LoRA Adapter" | "Finetune",
          "base_model": null,
          "params_m": null,
          "modality": "Text" | "Vision" | "Diffusion" | "Multimodal" | "Other",
          "category": "Inspection" | "VQA" | "Code" | "Extraction" | "Reasoning" | "Architecture" | "Other",
          "newsletter_blurb": "...",
          "key_facts": ["...", "...", "..."],
          "delta": {{
            "what_changed": ["..."],
            "why_it_matters": ["..."]
          }},
          "edge": {{
            "edge_ready": false,
            "min_vram_gb": null,
            "quantization": ["none"]
          }},
          "manufacturing": {{
            "manufacturing_fit_score": 1,
            "use_cases": ["..."]
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
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
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
