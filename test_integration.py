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
import model_filters as filters
from namespace_policy import classify_namespace
from run_stats import RunStats
from param_estimator import ParamEstimate

class TestIntegration(unittest.TestCase):

    @patch('main.Database')
    @patch('main.HFClient')
    @patch('main.LLMClient')
    @patch('main.Reporter')
    @patch('main.Mailer')
    @patch('main.estimate_parameters')
    @patch('main.security_warnings')
    @patch('builtins.open', new_callable=mock_open, read_data="# Report Content")
    def test_full_flow(self, mock_file, mock_security_warnings, mock_estimate_params, MockMailer, MockReporter, MockLLMClient, MockHFClient, MockDatabase):
        # Setup Mocks

        # Database
        mock_db_instance = MockDatabase.return_value
        mock_db_instance.get_existing_ids.return_value = {"old-model/123", "updated-model/456"}
        last_run = datetime.now(timezone.utc) - timedelta(hours=24)
        mock_db_instance.get_last_run_timestamp.return_value = last_run

        # Mock Last Modified for Updated Model
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        def side_effect_last_mod(mid):
            if mid == "updated-model/456": return yesterday.isoformat()
            if mid == "old-model/123": return yesterday.isoformat()
            return None
        mock_db_instance.get_model_last_modified.side_effect = side_effect_last_mod

        # Author cache mock
        def side_effect_get_author(ns):
            if ns == "updated-model":
                return {'kind': 'user', 'last_checked': datetime.now(timezone.utc)}
            return None
        mock_db_instance.get_author.side_effect = side_effect_get_author

        # HF Client
        mock_hf_instance = MockHFClient.return_value

        # Author details
        mock_hf_instance.get_org_details.side_effect = lambda ns: {'id': 'org1'} if ns == 'trending' else {} if ns == 'new' else {} if ns == 'updated-model' else None
        mock_hf_instance.get_user_overview.side_effect = lambda ns: {'numFollowers': 50, 'isPro': False} if ns == 'new' else {'numFollowers': 100} if ns == 'updated-model' else None

        # Models
        m_new = MagicMock()
        m_new.id = "new/specialist-model-7b"
        m_new.tags = ["manufacturing"]
        m_new.created_at = datetime.now(timezone.utc)
        m_new.lastModified = datetime.now(timezone.utc)

        m_updated = MagicMock()
        m_updated.id = "updated-model/456"
        m_updated.tags = ["vision"]
        m_updated.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        m_updated.lastModified = datetime.now(timezone.utc) + timedelta(hours=1)

        m_trending = MagicMock()
        m_trending.id = "trending/hot-model"

        m_trending_info = MagicMock()
        m_trending_info.id = "trending/hot-model"
        m_trending_info.tags = ["hot"]
        m_trending_info.created_at = datetime.now(timezone.utc) - timedelta(days=5)
        m_trending_info.lastModified = datetime.now(timezone.utc) - timedelta(days=5)

        mock_hf_instance.get_model_info.side_effect = lambda mid: m_trending_info if mid == "trending/hot-model" else None

        mock_hf_instance.fetch_new_models.return_value = [m_new]
        mock_hf_instance.fetch_recently_updated_models.return_value = [m_updated]
        mock_hf_instance.fetch_trending_models.return_value = ["trending/hot-model"]
        mock_hf_instance.fetch_daily_papers.return_value = []

        mock_hf_instance.get_model_file_details.return_value = []

        # Security warnings (empty list = secure)
        mock_security_warnings.return_value = []

        # Param Estimation using new Dataclass
        mock_estimate_params.return_value = ParamEstimate(
            total_params=7000000000,
            active_params=7000000000,
            total_b=7.0,
            active_b=7.0,
            source="test",
            dtype_breakdown=None,
            is_moe=False,
            experts="Dense",
            notes=[]
        )

        # Unique readmes to avoid duplicate signature skipping
        def side_effect_readme(mid):
            base = "Detailed readme content with base_model: llama, dataset: c4, license: mit. " * 20
            return f"{base} ID: {mid}" # Append ID to make unique

        mock_hf_instance.get_model_readme.side_effect = side_effect_readme

        # LLM - New Comprehensive Structure with Evidence
        mock_llm_instance = MockLLMClient.return_value
        # NOTE: The evidence quote must match what is in the README!
        # Our side_effect_readme adds the ID at the end. The base text is constant.
        # So we should use the base text or a part of it for the quote.

        mock_llm_instance.analyze_model.return_value = {
            "model_type": "Finetune",
            "base_model": "llama-2",
            "params_m": 7000,
            "modality": "Text",
            "category": "Vision",
            "newsletter_blurb": "A great manufacturing model.",
            "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
            "delta": {
                "what_changed": ["Changed dataset", "New objective"],
                "why_it_matters": ["Better accuracy", "Faster"]
            },
            "edge": {
                "edge_ready": True,
                "min_vram_gb": 8,
                "quantization": ["int8"]
            },
            "manufacturing": {
                "manufacturing_fit_score": 9,
                "use_cases": ["Defect detection"]
            },
            "specialist_score": 9,
            "confidence": "high",
            "unknowns": [],
            "evidence": [
                {"claim": "test", "quote": "Detailed readme content"}
            ]
        }

        # Mock Reporter returning path
        MockReporter.return_value.generate_full_report.return_value = MagicMock(read_text=MagicMock(return_value="Report Content"))

        # Run Main
        with patch.object(sys, 'argv', ["main.py", "--limit", "10", "--force-email"]):
            main.main()

        # Verifications
        saved_models = [call[0][0] for call in mock_db_instance.save_model.call_args_list]
        saved_ids = [m['id'] for m in saved_models]

        self.assertIn("new/specialist-model-7b", saved_ids)
        self.assertIn("updated-model/456", saved_ids)
        self.assertIn("trending/hot-model", saved_ids)

        # Check if Authors upserted
        self.assertTrue(mock_db_instance.upsert_author.called)

        mock_db_instance.set_last_run_timestamp.assert_called()
        MockMailer.return_value.send_report.assert_called()
        MockReporter.return_value.generate_full_report.assert_called()

if __name__ == '__main__':
    unittest.main()
