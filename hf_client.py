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
                return response
            except requests.exceptions.RequestException as e:
                if i < max_retries - 1:
                    logger.warning(f"Request failed: {e}. Retrying...")
                    time.sleep(2)
                    continue
                return None
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

    # --- New Author/Org Methods (Tri-State with Avatar Check) ---

    def get_org_details(self, namespace):
        """
        Returns:
        - dict: if Org exists (200) - using avatar endpoint as proxy
        - {}: if Org not found (404)
        - None: if error/transient
        """
        url = f"{self.base_url}/organizations/{namespace}/avatar"
        try:
            # Check avatar without redirects if possible, or just checking 200
            # Adding redirect=null param might stop redirect if API supports it,
            # otherwise it redirects to CDN which returns 200 (image).
            # We just need to know if the endpoint accepts the org name.
            resp = self._make_request("GET", url, headers={})
            if resp is None:
                return None

            logger.debug(f"Org check {namespace}: {resp.status_code}")

            if resp.status_code == 200:
                # It exists. Return dummy dict or parse if JSON
                # Avatar endpoint usually returns image bytes if existing?
                # Or JSON if using API?
                # User said: "response.json() # e.g. {'avatarUrl': ...}"
                # Let's try .json(), if fails assume image -> exists.
                try:
                    return resp.json()
                except:
                    return {"exists": True}

            if resp.status_code == 404:
                return {} # Definite not found

            logger.warning(f"Org check unexpected status {resp.status_code} for {namespace}")
            return None
        except Exception as e:
            logger.warning(f"Org check failed for {namespace}: {e}")
            return None

    def get_user_overview(self, namespace):
        """
        Returns:
        - dict: if User exists (200)
        - {}: if User not found (404)
        - None: if error/transient
        """
        url = f"{self.base_url}/users/{namespace}/overview"
        try:
            resp = self._make_request("GET", url, headers={})
            if resp is None:
                return None

            logger.debug(f"User check {namespace}: {resp.status_code}")

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return {}

            logger.warning(f"User check unexpected status {resp.status_code} for {namespace}")
            return None
        except Exception as e:
            logger.warning(f"User check failed for {namespace}: {e}")
            return None
