import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import json
from datetime import datetime

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
        mock_db_instance.get_existing_ids.return_value = {"old-model/123"}

        # HF Client
        mock_hf_instance = MockHFClient.return_value

        # Models
        m1 = MagicMock()
        m1.id = "new/specialist-model-7b"
        m1.tags = ["manufacturing", "vision"]
        m1.created_at = datetime.now()

        m2 = MagicMock()
        m2.id = "big/giant-model-70b"
        m2.tags = ["nlp"]
        m2.created_at = datetime.now()

        # Model from Trending (needs info fetch)
        m_trending_id = "trending/cool-model"

        # Return lists
        mock_hf_instance.fetch_new_models.return_value = [m1, m2]
        mock_hf_instance.fetch_trending_models.return_value = [m_trending_id]
        mock_hf_instance.fetch_daily_papers.return_value = []

        # Mock get_model_info for trending
        m_trending_info = MagicMock()
        m_trending_info.id = m_trending_id
        m_trending_info.tags = ["trending"]
        m_trending_info.created_at = datetime.now()

        def side_effect_info(mid):
            if mid == m_trending_id: return m_trending_info
            return None
        mock_hf_instance.get_model_info.side_effect = side_effect_info

        # Mock File Details (Security & Size)
        # m1: Safe, Normal Size
        # m2: Safe, Big Size
        # m_trending: Safe, Normal Size
        mock_hf_instance.get_model_file_details.return_value = [] # Default empty list (safe)

        # Security Filter: Always True for now
        mock_is_secure.return_value = True

        # Extract Params
        def side_effect_params(info, files):
            if "70b" in info.id: return 70.0
            if "7b" in info.id: return 7.0
            return None
        mock_extract_params.side_effect = side_effect_params

        # Readme
        mock_hf_instance.get_model_readme.return_value = "Detailed readme content " * 20 # > 300 chars

        # LLM
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.analyze_model.return_value = {"specialist_score": 5}

        # Run Main
        with patch.object(sys, 'argv', ["main.py", "--limit", "10"]):
            main.main()

        # Verifications

        # 1. Fetching sources
        mock_hf_instance.fetch_new_models.assert_called()
        mock_hf_instance.fetch_trending_models.assert_called()
        mock_hf_instance.fetch_daily_papers.assert_called()

        # 2. Info fetching for extra IDs
        mock_hf_instance.get_model_info.assert_called_with(m_trending_id)

        # 3. Security Check Call
        # Should be called for m1, m2, m_trending
        self.assertTrue(mock_hf_instance.get_model_file_details.called)

        # 4. Saving
        # m1 saved
        # m_trending saved
        # m2 skipped (params)
        saved_models = [call[0][0] for call in mock_db_instance.save_model.call_args_list]
        saved_ids = [m['id'] for m in saved_models]

        self.assertIn("new/specialist-model-7b", saved_ids)
        self.assertIn("trending/cool-model", saved_ids)
        self.assertNotIn("big/giant-model-70b", saved_ids)

if __name__ == '__main__':
    unittest.main()
