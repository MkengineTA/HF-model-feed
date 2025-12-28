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
    @patch('main.Mailer')
    @patch('filters.extract_parameter_count')
    @patch('filters.is_secure')
    @patch('builtins.open', new_callable=mock_open, read_data="# Report Content")
    def test_full_flow(self, mock_file, mock_is_secure, mock_extract_params, MockMailer, MockReporter, MockLLMClient, MockHFClient, MockDatabase):
        # Setup Mocks

        # Database
        mock_db_instance = MockDatabase.return_value
        mock_db_instance.get_existing_ids.return_value = {"old-model/123", "updated-model/456"}

        # Last run 24h ago
        last_run = datetime.now(timezone.utc) - timedelta(hours=24)
        mock_db_instance.get_last_run_timestamp.return_value = last_run

        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        older = datetime.now(timezone.utc) - timedelta(days=10)

        def side_effect_last_mod(mid):
            if mid == "old-model/123": return yesterday
            if mid == "updated-model/456": return yesterday
            return None
        mock_db_instance.get_model_last_modified.side_effect = side_effect_last_mod

        # HF Client
        mock_hf_instance = MockHFClient.return_value

        # 1. Models
        # M1: Created just now (should be fetched)
        m_new = MagicMock()
        m_new.id = "new/specialist-model-7b"
        m_new.tags = ["manufacturing"]
        m_new.created_at = datetime.now(timezone.utc)
        m_new.lastModified = datetime.now(timezone.utc)

        # M2: Updated just now (should be fetched)
        m_updated = MagicMock()
        m_updated.id = "updated-model/456"
        m_updated.tags = ["vision"]
        m_updated.created_at = older
        m_updated.lastModified = datetime.now(timezone.utc)

        # M3: Trending (New Discovery)
        m_trending = MagicMock()
        m_trending.id = "trending/hot-model"

        # Info for trending
        m_trending_info = MagicMock()
        m_trending_info.id = "trending/hot-model"
        m_trending_info.tags = ["hot"]
        m_trending_info.created_at = older
        m_trending_info.lastModified = older

        def side_effect_info(mid):
            if mid == "trending/hot-model": return m_trending_info
            return None
        mock_hf_instance.get_model_info.side_effect = side_effect_info

        mock_hf_instance.fetch_new_models.return_value = [m_new]
        mock_hf_instance.fetch_recently_updated_models.return_value = [m_updated]
        mock_hf_instance.fetch_trending_models.return_value = ["trending/hot-model"]
        mock_hf_instance.fetch_daily_papers.return_value = []

        mock_hf_instance.get_model_file_details.return_value = []
        mock_is_secure.return_value = True
        mock_extract_params.return_value = 7.0
        mock_hf_instance.get_model_readme.return_value = "Detailed readme content " * 20

        # LLM
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.analyze_model.return_value = {
            "model_type": "Finetune",
            "base_model": "llama-2",
            "technical_summary": "Tech summary",
            "delta_explanation": "Delta details",
            "specialist_score": 8,
            "manufacturing_potential": True
        }

        # Run Main
        with patch.object(sys, 'argv', ["main.py", "--limit", "10"]):
            main.main()

        # Verifications
        # 1. Check fetch calls passed timestamp
        mock_hf_instance.fetch_new_models.assert_called_with(since=last_run, limit=10)
        mock_hf_instance.fetch_recently_updated_models.assert_called_with(since=last_run, limit=10)

        # 2. Check saved models
        saved_models = [call[0][0] for call in mock_db_instance.save_model.call_args_list]
        saved_ids = [m['id'] for m in saved_models]

        self.assertIn("new/specialist-model-7b", saved_ids)
        self.assertIn("updated-model/456", saved_ids)
        self.assertIn("trending/hot-model", saved_ids)

        # 3. Check State Update
        mock_db_instance.set_last_run_timestamp.assert_called()

        # 4. Check Mailer
        MockMailer.return_value.send_report.assert_called()

if __name__ == '__main__':
    unittest.main()
