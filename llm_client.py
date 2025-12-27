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
    def __init__(self, api_url, model):
        self.api_url = api_url
        self.model = model

    def analyze_model(self, readme_content, tags):
        """
        Analyzes the model README and tags using the local LLM.
        Returns a dictionary with the analysis results.
        """

        system_prompt = (
            "You are an expert AI researcher specializing in Edge AI and Manufacturing Specialist Models. "
            "Your task is to analyze a Hugging Face model based on its README and tags. "
            "You must output valid JSON only."
        )

        user_prompt = f"""
        Analyze the following model.

        HF Tags: {', '.join(tags) if tags else 'None'}

        README Content (truncated if too long):
        {readme_content[:32000]}

        Provide the output in the following JSON structure:
        {{
            "category": "Vision" | "Code" | "Extraction" | "Reasoning" | "Architecture" | "Other",
            "specialist_score": <int 1-10, where 10 is highly specialized/niche, 1 is generic chat>,
            "manufacturing_potential": <boolean, true if useful for manufacturing tasks like defect detection, pdf extraction, tool use>,
            "summary": "<concise sentence describing the USP>",
            "manufacturing_use_cases": ["<list of potential use cases if any>"]
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

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
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
            return None
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            return None
