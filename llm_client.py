import json
import re
import requests
import logging

logger = logging.getLogger("EdgeAIScout")

def extract_json_from_text(text):
    """
    Extracts JSON object from a string. Handles Markdown code blocks and raw text.
    Returns None if parsing fails.
    """
    # Helper to clean common JSON errors
    def clean_json(json_text):
        # Remove trailing commas in objects/arrays (simple regex)
        # Match , followed by whitespace and } or ]
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)
        return json_text

    # 1. Try fenced code block ```json ... ``` or just ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(clean_json(m.group(1)))
        except json.JSONDecodeError:
            pass # Fallback to method 2

    # 2. Fallback: find first '{' and last '}'
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

    def analyze_model(self, readme_content, tags):
        """
        Analyzes the model README and tags using the LLM.
        Returns a dictionary with the analysis results.
        """

        system_prompt = (
            "Du bist ein strenger Analyst für Edge-AI- und Manufacturing-Modelle. "
            "Du darfst NICHTS erfinden. Wenn Informationen im README/Tags nicht klar belegt sind, "
            "setze sie auf null und liste sie unter unknowns; confidence dann 'low' oder 'medium'. "
            "Gib AUSSCHLIESSLICH valides JSON aus (kein Markdown, kein Text außerhalb JSON). "
            "Halte Längenlimits strikt ein."
        )

        user_prompt = f"""
        Analysiere das HuggingFace-Modell anhand README + Tags.

        HF Tags: {', '.join(tags) if tags else 'None'}

        README (gekürzt):
        {readme_content[:32000]}

        LÄNGENLIMITS (hart):
        - newsletter_blurb: max 180 Zeichen, genau 1 Satz.
        - key_facts: genau 3 Einträge, je max 140 Zeichen.
        - delta.what_changed: max 3 Bullets, je max 140 Zeichen.
        - delta.why_it_matters: max 3 Bullets, je max 140 Zeichen.
        - edge.deployment_notes: max 3 Bullets, je max 140 Zeichen.
        - manufacturing.use_cases: max 3 Bullets, je max 140 Zeichen.
        - manufacturing.risks: max 2 Bullets, je max 140 Zeichen.
        - unknowns: max 3 Bullets, je max 120 Zeichen.

        REGELN:
        - Alles in Deutsch außer enum-Werte.
        - params_m nur setzen, wenn im README/Modellkarte klar genannt; sonst null.
        - min_vram_gb nur setzen, wenn klar genannt; sonst null.
        - quantization: nur nennen, wenn explizit erwähnt oder offensichtlich aus Artefakten (z.B. gguf-Dateien) ableitbar; sonst "none".
        - model_type sauber erkennen: Base Model vs Finetune vs LoRA Adapter.
        - confidence:
          - high: Base Model + Training/Delta klar beschrieben
          - medium: teilweise klar
          - low: viele unknowns / dünnes README

        Gib JSON in diesem Format zurück:

        {{
          "model_type": "Base Model" | "LoRA Adapter" | "Finetune",
          "base_model": null,
          "params_m": null,
          "modality": "Text" | "Vision" | "Diffusion" | "Multimodal" | "Other",
          "category": "Vision" | "Code" | "Extraction" | "Reasoning" | "Architecture" | "Other",
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
          "specialist_score": 1,
          "confidence": "low",
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
            # max_tokens removed to allow full reasoning
            "stream": False
        }

        if self.enable_reasoning:
            # Add OpenRouter specific "reasoning" block
            payload["reasoning"] = {"enabled": True}

        # Build Headers
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

            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            analysis = extract_json_from_text(content)
            if not analysis:
                logger.error(f"Failed to parse JSON from LLM response. Raw Content:\n{content[:500]}...\n[Truncated]")
                return None

            return analysis

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            return None
