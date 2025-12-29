import json
import re
import requests
import logging

logger = logging.getLogger("EdgeAIScout")

def extract_json_from_text(text):
    """
    Extracts JSON object from a string. Handles Markdown code blocks.
    Returns None if parsing fails.
    """
    try:
        # Find first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = text[start_idx:end_idx+1]
        return json.loads(json_str)
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
            "You are an expert AI researcher specializing in Edge AI and Manufacturing Specialist Models. "
            "Your task is to analyze a Hugging Face model based on its README and tags. "
            "You must provide a deep technical analysis, focusing on the 'Delta' (what changed from the base model). "
            "Output valid JSON only."
        )

        user_prompt = f"""
        Analyze the following model deeply.

        HF Tags: {', '.join(tags) if tags else 'None'}

        README Content (truncated if too long):
        {readme_content[:32000]}

        ## Requirements:
        1. **Language**: The `technical_summary` and `delta_explanation` MUST be written in **German**.
        2. **Technical Summary**: No short USPs. Describe the technical core. What architecture? What dataset? What training method? Use proper Markdown formatting (lists with `- `) if listing features.
        3. **Delta Analysis**: If this is an Adapter/Finetune:
           - What is the Base Model?
           - What specifically changed (Dataset, Objective, Quantization)?
           - Value Proposition: Why use this over the base model?
           - **Structure**: Use Markdown lists (`- `) for points. Do NOT use `(1)`, `(2)` or `•` inline. Use newlines for lists.
        4. **Categorization**: Determine if it's a Base Model, LoRA Adapter, or Finetune.
        5. **Scoring**: Score 1-10 on specialization for Edge/Manufacturing.

        ## Output Format (JSON):
        {{
            "model_type": "Base Model" | "LoRA Adapter" | "Finetune",
            "base_model": "<name of base model or null>",
            "technical_summary": "<detailed technical description in German, use Markdown lists if needed>",
            "delta_explanation": "<explanation of changes and value prop in German, use Markdown lists for points>",
            "category": "Vision" | "Code" | "Extraction" | "Reasoning" | "Architecture" | "Other",
            "specialist_score": <int 1-10>,
            "manufacturing_potential": <boolean>
        }}
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
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
                logger.error(f"Failed to parse JSON from LLM response: {content[:100]}...")
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
