from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import logging
import requests

logger = logging.getLogger("EdgeAIScout")

class HFClient:
    def __init__(self, token=None):
        self.api = HfApi(token=token)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.base_url = "https://huggingface.co/api"

    def fetch_new_models(self, limit=200):
        """
        Fetches the latest models from Hugging Face, sorted by creation date.
        """
        try:
            models = self.api.list_models(
                sort="createdAt",
                direction="-1",
                limit=limit,
                full=False, # We don't need full info yet, but tags are useful
                fetch_config=True # Try to get config for params if possible
            )
            return list(models)
        except Exception as e:
            logger.error(f"Error fetching models from HF: {e}")
            return []

    def fetch_trending_models(self, limit=10):
        """
        Fetches trending models.
        """
        try:
            url = f"{self.base_url}/trending"
            params = {"type": "model", "limit": limit}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Map structure to a simple object similar to list_models result
            models = []
            for item in data.get("recentlyTrending", []):
                repo_data = item.get("repoData", {})
                if repo_data:
                     # Create a pseudo object or dict
                     # We need an object with .id, .tags, .created_at, .safetensors
                     # The trending API response structure for 'repoData' matches the model info partially
                     # We might need to wrap it. For simplicity, let's use a dict wrapper or SimpleNamespace in main.
                     # Or better: just return the raw data and handle normalization in main.
                     # To keep interface consistent, let's try to return objects compatible with what fetch_new_models returns.
                     # list_models returns ModelInfo objects.
                     # We can fetch the actual ModelInfo for these IDs or construct a dummy.
                     # Fetching actual info is safer to get tags/config uniformly.
                     models.append(repo_data.get("id"))

            # If we have IDs, let's fetch their full details to match the pipeline expectation
            if models:
                 # HfApi list_models can filter by model (not by list of IDs efficiently without loop)
                 # But list_models doesn't take a list of IDs.
                 # We will return the IDs and let main fetch details or we loop here.
                 # Let's return a list of model IDs and handle fetching details in main/utils.
                 return models
            return []
        except Exception as e:
            logger.error(f"Error fetching trending models: {e}")
            return []

    def fetch_daily_papers(self, limit=20):
        """
        Fetches daily papers and extracts potential model IDs.
        """
        try:
            url = f"{self.base_url}/daily_papers"
            params = {"limit": limit}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            papers = response.json()

            model_ids = []
            for paper in papers:
                paper_info = paper.get("paper", {})
                # Check projectPage
                project_page = paper_info.get("projectPage", "")
                if project_page and "huggingface.co/" in project_page:
                    # Extract ID: https://huggingface.co/org/repo -> org/repo
                    # Remove base url
                    part = project_page.split("huggingface.co/")[-1]
                    # Clean up (remove query params, etc)
                    part = part.split("?")[0].strip("/")
                    # Validate it looks like a model ID (has /)
                    if "/" in part and len(part.split("/")) >= 2:
                         # It might be a space or dataset, but we'll assume model and verify later
                         model_ids.append(part)

            return list(set(model_ids))
        except Exception as e:
            logger.error(f"Error fetching daily papers: {e}")
            return []

    def get_model_file_details(self, model_id):
        """
        Fetches the file tree to get sizes and security status.
        """
        try:
            url = f"{self.base_url}/models/{model_id}/tree/main"
            params = {"recursive": "true", "expand": "true"}
            response = requests.get(url, headers=self.headers, params=params, timeout=20)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching file details for {model_id}: {e}")
            return None

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

    def get_model_info(self, model_id):
        """
        Fetches single model info to normalize data from different sources (trending/papers).
        """
        try:
            # We use model_info from HfApi
            return self.api.model_info(model_id, files_metadata=True)
        except Exception as e:
            logger.warning(f"Could not fetch info for {model_id}: {e}")
            return None
