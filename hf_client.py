from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import logging
import requests
from datetime import datetime, timezone
import time

logger = logging.getLogger("EdgeAIScout")

class HFClient:
    def __init__(self, token=None):
        self.api = HfApi(token=token)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.base_url = "https://huggingface.co/api"

    def _make_request(self, method, url, params=None, max_retries=3, headers=None):
        """
        Helper to make requests with retry logic for 429 errors.
        """
        req_headers = headers if headers is not None else self.headers
        for i in range(max_retries):
            try:
                response = requests.request(method, url, headers=req_headers, params=params, timeout=20)
                if response.status_code == 429:
                    wait_time = int(response.headers.get("Retry-After", 10 * (i + 1)))
                    logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                # We handle status codes in callers mostly, but raise_for_status handles 4xx/5xx
                # For tri-state logic (404 check), callers need the response object even if 404.
                # So we return response object here.
                return response
            except requests.exceptions.RequestException as e:
                # requests.request doesn't raise unless network error. HTTP status doesn't raise unless raise_for_status called.
                # But if network error:
                if i < max_retries - 1:
                    logger.warning(f"Request failed: {e}. Retrying...")
                    time.sleep(2)
                    continue
                # If final attempt fails or it's a hard error
                return None # Return None on network failure
        return None

    def fetch_new_models(self, since=None, limit=1000):
        try:
            models_iter = self.api.list_models(
                sort="createdAt",
                direction="-1",
                limit=limit,
                full=False,
                fetch_config=True
            )
            candidates = []
            for m in models_iter:
                if since and m.created_at and m.created_at <= since:
                    break
                candidates.append(m)
            return candidates
        except Exception as e:
            logger.error(f"Error fetching new models from HF: {e}")
            return []

    def fetch_recently_updated_models(self, since=None, limit=1000):
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
                if since and m.lastModified and m.lastModified <= since:
                    break
                candidates.append(m)
            return candidates
        except Exception as e:
            logger.error(f"Error fetching updated models from HF: {e}")
            return []

    def fetch_trending_models(self, limit=10):
        try:
            url = f"{self.base_url}/trending"
            params = {"type": "model", "limit": limit}
            response = self._make_request("GET", url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                models = []
                for item in data.get("recentlyTrending", []):
                    repo_data = item.get("repoData", {})
                    if repo_data:
                         models.append(repo_data.get("id"))
                return models
            return []
        except Exception as e:
            logger.error(f"Error fetching trending models: {e}")
            return []

    def fetch_daily_papers(self, limit=20):
        try:
            url = f"{self.base_url}/daily_papers"
            params = {"limit": limit}
            response = self._make_request("GET", url, params=params)
            if response and response.status_code == 200:
                papers = response.json()
                model_ids = []
                for paper in papers:
                    paper_info = paper.get("paper", {})
                    project_page = paper_info.get("projectPage", "")
                    if project_page and "huggingface.co/" in project_page:
                        part = project_page.split("huggingface.co/")[-1].split("?")[0].strip("/")
                        if "/" in part and len(part.split("/")) >= 2:
                             model_ids.append(part)
                return list(set(model_ids))
            return []
        except Exception as e:
            logger.error(f"Error fetching daily papers: {e}")
            return []

    def get_model_file_details(self, model_id):
        try:
            url = f"{self.base_url}/models/{model_id}/tree/main"
            params = {"recursive": "true", "expand": "true"}
            response = self._make_request("GET", url, params=params)
            if response:
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    return None
                # Other status codes might be auth or 429
                response.raise_for_status()
            return None
        except Exception as e:
            logger.error(f"Error fetching file details for {model_id}: {e}")
            return None

    def get_model_readme(self, model_id):
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except (RepositoryNotFoundError, RevisionNotFoundError):
            return None
        except Exception as e:
            logger.error(f"Error downloading README for {model_id}: {e}")
            return None

    def get_model_info(self, model_id):
        try:
            return self.api.model_info(model_id, files_metadata=True)
        except Exception as e:
            logger.warning(f"Could not fetch info for {model_id}: {e}")
            return None

    # --- New Author/Org Methods (Tri-State) ---

    def get_org_details(self, namespace):
        """
        Returns:
        - dict: if Org exists (200)
        - {}: if Org not found (404) -> 'Not an Org'
        - None: if error/transient (do not cache)
        """
        url = f"{self.base_url}/organizations/{namespace}"
        try:
            # Request without Auth headers to match public view behavior if needed,
            # or keep auth if user token has access?
            # User instruction: "optional headers override (damit du Org/User ohne Token abfragen kannst)"
            # And usage: "self._make_request('GET', url, headers={})"
            resp = self._make_request("GET", url, headers={})
            if resp is None:
                return None
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return {} # Explicit empty dict = Not Found

            logger.warning(f"Org lookup unexpected status {resp.status_code} for {namespace}")
            return None
        except Exception as e:
            logger.warning(f"Org lookup failed for {namespace}: {e}")
            return None

    def get_user_overview(self, namespace):
        """
        Returns:
        - dict: if User exists (200)
        - {}: if User not found (404) -> 'Not a User'
        - None: if error/transient
        """
        url = f"{self.base_url}/users/{namespace}/overview"
        try:
            resp = self._make_request("GET", url, headers={}) # Without Auth
            if resp is None:
                return None
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return {}

            logger.warning(f"User lookup unexpected status {resp.status_code} for {namespace}")
            return None
        except Exception as e:
            logger.warning(f"User lookup failed for {namespace}: {e}")
            return None
