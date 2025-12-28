import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import json
from datetime import datetime, timedelta, timezone

# Add current dir to sys.path to import modules
sys.path.append(os.getcwd())

import main
import config

class TestIntegration(unittest.TestCase):

    @patch('main.Database')
    @patch('main.HFClient')
    @patch('main.LLMClient')
    @patch('main.Reporter')
    @patch('filters.extract_parameter_count')
    @patch('filters.is_secure')
    def test_full_flow(self, mock_is_secure, mock_extract_params, MockReporter, MockLLMClient, MockHFClient, MockDatabase):
        # Setup Mocks

        # Database
        mock_db_instance = MockDatabase.return_value

        # Mock Existing IDs: "old-model/123" and "updated-model/456"
        mock_db_instance.get_existing_ids.return_value = {"old-model/123", "updated-model/456"}

        # Mock DB Last Modified
        # old-model: processed yesterday, unchanged
        # updated-model: processed yesterday, but now updated
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        older = datetime.now(timezone.utc) - timedelta(days=10)

        def side_effect_last_mod(mid):
            if mid == "old-model/123": return yesterday
            if mid == "updated-model/456": return yesterday
            return None
        mock_db_instance.get_model_last_modified.side_effect = side_effect_last_mod

        # HF Client
        mock_hf_instance = MockHFClient.return_value

        # 1. Models - Recently Created
        m_new = MagicMock()
        m_new.id = "new/specialist-model-7b"
        m_new.tags = ["manufacturing"]
        m_new.created_at = datetime.now(timezone.utc)
        m_new.lastModified = datetime.now(timezone.utc)

        # 2. Models - Recently Updated
        m_updated = MagicMock()
        m_updated.id = "updated-model/456" # Exists in DB
        m_updated.tags = ["vision"]
        m_updated.created_at = older
        m_updated.lastModified = datetime.now(timezone.utc) # Newer than DB!

        m_old_unchanged = MagicMock()
        m_old_unchanged.id = "old-model/123" # Exists in DB
        m_old_unchanged.lastModified = older # Older than DB (or same) -> No process

        mock_hf_instance.fetch_new_models.return_value = [m_new]
        mock_hf_instance.fetch_recently_updated_models.return_value = [m_updated, m_old_unchanged]
        mock_hf_instance.fetch_daily_papers.return_value = []

        # Filters
        mock_hf_instance.get_model_file_details.return_value = [] # Safe
        mock_is_secure.return_value = True
        mock_extract_params.return_value = 7.0

        # Readme
        mock_hf_instance.get_model_readme.return_value = "Detailed readme content " * 20

        # LLM
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.analyze_model.return_value = {"specialist_score": 5}

        # Run Main
        with patch.object(sys, 'argv', ["main.py", "--limit", "10"]):
            main.main()

        # Verifications

        # 1. Source Fetching
        mock_hf_instance.fetch_new_models.assert_called()
        mock_hf_instance.fetch_recently_updated_models.assert_called()

        # 2. Processing Logic
        # saved_models should include m_new (New) and m_updated (Update Detected)
        # Should NOT include m_old_unchanged

        saved_models = [call[0][0] for call in mock_db_instance.save_model.call_args_list]
        saved_ids = [m['id'] for m in saved_models]

        self.assertIn("new/specialist-model-7b", saved_ids)
        self.assertIn("updated-model/456", saved_ids)
        self.assertNotIn("old-model/123", saved_ids)

        # Verify reason logging (optional, but good for confidence)
        # We can't easily check logs here without capturing them, but the saving logic proves it works.

if __name__ == '__main__':
    unittest.main()
