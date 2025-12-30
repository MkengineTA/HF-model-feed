import json
import re
import requests
import logging

logger = logging.getLogger("EdgeAIScout")

def extract_json_from_text(text):
    def clean_json(json_text):
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
        return json_text

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(clean_json(m.group(1)))
        except json.JSONDecodeError:
            pass

    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        json_str = text[start_idx:end_idx+1]
        return json.loads(clean_json(json_str))
    except json.JSONDecodeError:
        return None

class LLMClient:
    def __init__(self, api_url, model, api_key=None, site_url=None, app_name=None, enable_reasoning=False):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name
        self.enable_reasoning = enable_reasoning

    def analyze_model(self, readme_content, tags, yaml_meta=None, file_summary=None):
        """
        Analyzes the model README, tags, and metadata using the LLM.
        """

        # Summarize files for context
        files_ctx = "Unknown"
        if file_summary:
            exts = {}
            for f in file_summary:
                p = f.get('path', '')
                ext = p.split('.')[-1] if '.' in p else 'bin'
                exts[ext] = exts.get(ext, 0) + 1
            files_ctx = ", ".join([f"{k}: {v}" for k,v in exts.items()])

        system_prompt = (
            "Du bist ein strenger Analyst für Edge-AI- und Manufacturing-Modelle. "
            "Du darfst NICHTS erfinden. Wenn Informationen nicht klar belegt sind (README/YAML), "
            "setze sie auf null und liste sie unter unknowns. Confidence sinkt bei fehlenden Belegen. "
            "Gib AUSSCHLIESSLICH valides JSON aus."
        )

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
        - evidence: genau 2-4 Einträge.

        REGELN:
        - Sprache: DEUTSCH (außer Enum-Werte).
        - params_m / min_vram_gb: Nur wenn explizit genannt.
        - quantization: Prüfe auch Filenamen (gguf/onnx) falls im Text nicht genannt.
        - Evidence-Pflicht: Für Claims bzgl. Domain/Performance MUSS ein Zitat ("quote") existieren.

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
            "quantization": ["none"],
            "deployment_notes": ["..."]
          }},
          "manufacturing": {{
            "manufacturing_fit_score": 1,
            "use_cases": ["..."],
            "risks": ["..."]
          }},
          "evidence": [
            {{ "claim": "Kurzer Claim", "quote": "Exaktes Zitat aus dem README" }}
          ],
          "specialist_score": 1,
          "confidence": "low" | "medium" | "high",
          "unknowns": ["..."]
        }}
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "stream": False
        }

        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

            analysis = extract_json_from_text(content)
            if not analysis:
                logger.error(f"JSON Parse Error. Raw:\n{content[:500]}...")
                return None
            return analysis

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None
