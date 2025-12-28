from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import logging
import requests
from datetime import datetime

logger = logging.getLogger("EdgeAIScout")

class HFClient:
    def __init__(self, token=None):
        self.api = HfApi(token=token)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.base_url = "https://huggingface.co/api"

    def fetch_new_models(self, since=None, limit=1000):
        """
        Fetches the latest models (Created).
        Iterates until models are older than `since` OR `limit` is reached.
        """
        try:
            # Note: `limit` in list_models behaves like a page size or total limit?
            # HfApi list_models returns an iterator. We can iterate it.
            # Passing a large limit to list_models might be needed if the API truncates.
            # But normally it yields.
            # The API itself has a hard limit per page, but the client handles pagination if we iterate?
            # Actually, `limit` in `list_models` usually restricts the TOTAL number returned.
            # So we pass a high limit (e.g. 2000) to ensure we get enough candidates,
            # then we stop early if we hit the timestamp.

            models_iter = self.api.list_models(
                sort="createdAt",
                direction="-1",
                limit=limit,
                full=False,
                fetch_config=True
            )

            candidates = []
            for m in models_iter:
                # Check timestamp
                if since and m.created_at:
                    if m.created_at <= since:
                        # Found a model older than last run -> Stop
                        break
                candidates.append(m)

            return candidates
        except Exception as e:
            logger.error(f"Error fetching new models from HF: {e}")
            return []

    def fetch_recently_updated_models(self, since=None, limit=1000):
        """
        Fetches models that were recently updated (lastModified).
        """
        try:
            models_iter = self.api.list_models(
                sort="lastModified",
                direction="-1",
                limit=limit,
                full=False,
                fetch_config=True
            )

            candidates = []
            for m in models_iter:
                if since and m.lastModified:
                    if m.lastModified <= since:
                        break
                candidates.append(m)

            return candidates
        except Exception as e:
            logger.error(f"Error fetching updated models from HF: {e}")
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

            models = []
            for item in data.get("recentlyTrending", []):
                repo_data = item.get("repoData", {})
                if repo_data:
                     models.append(repo_data.get("id"))
            return models
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
                project_page = paper_info.get("projectPage", "")
                if project_page and "huggingface.co/" in project_page:
                    part = project_page.split("huggingface.co/")[-1]
                    part = part.split("?")[0].strip("/")
                    if "/" in part and len(part.split("/")) >= 2:
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
        Downloads the README.md content.
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
        Fetches single model info.
        """
        try:
            return self.api.model_info(model_id, files_metadata=True)
        except Exception as e:
            logger.warning(f"Could not fetch info for {model_id}: {e}")
            return None
