from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import logging
import requests

logger = logging.getLogger("EdgeAIScout")

class HFClient:
    def __init__(self, token=None):
        self.api = HfApi(token=token)

    def fetch_new_models(self, limit=200):
        """
        Fetches the latest models from Hugging Face, sorted by creation date.
        """
        try:
            models = self.api.list_models(
                sort="created",
                direction="-1",
                limit=limit,
                full=False, # We don't need full info yet, but tags are useful
                fetch_config=True # Try to get config for params if possible
            )
            return list(models)
        except Exception as e:
            logger.error(f"Error fetching models from HF: {e}")
            return []

    def get_model_readme(self, model_id):
        """
        Downloads the README.md content for a given model.
        Returns the content as a string, or None if not found/error.
        """
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except (RepositoryNotFoundError, RevisionNotFoundError):
            logger.warning(f"README not found for {model_id}")
            return None
        except Exception as e:
            logger.error(f"Error downloading README for {model_id}: {e}")
            return None
