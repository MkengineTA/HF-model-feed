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
    def test_full_flow(self, mock_extract_params, MockReporter, MockLLMClient, MockHFClient, MockDatabase):
        # Setup Mocks

        # Database
        mock_db_instance = MockDatabase.return_value
        mock_db_instance.get_existing_ids.return_value = {"old-model/123"}

        # HF Client
        mock_hf_instance = MockHFClient.return_value

        # Create some mock models
        # Model 1: Valid, specialist
        m1 = MagicMock()
        m1.id = "new/specialist-model-7b"
        m1.tags = ["manufacturing", "vision"]
        m1.created_at = datetime.now()

        # Model 2: Too large
        m2 = MagicMock()
        m2.id = "big/giant-model-70b"
        m2.tags = ["nlp"]
        m2.created_at = datetime.now()

        # Model 3: Already exists (should be filtered) -> actually handled by main logic before loop
        # We'll simulate fetching it
        m3 = MagicMock()
        m3.id = "old-model/123"
        m3.tags = []

        # Model 4: Short README, no links
        m4 = MagicMock()
        m4.id = "trash/bad-model"
        m4.tags = []

        # Model 5: Short README, has links
        m5 = MagicMock()
        m5.id = "review/linked-model"
        m5.tags = []

        mock_hf_instance.fetch_new_models.return_value = [m1, m2, m3, m4, m5]

        # Mocking filter logic behavior
        # extract_parameter_count side effects based on input model_info
        def side_effect_params(info):
            if "70b" in info.id: return 70.0
            if "7b" in info.id: return 7.0
            return None
        mock_extract_params.side_effect = side_effect_params

        # Mocking README downloads
        def side_effect_readme(model_id):
            if model_id == "new/specialist-model-7b":
                return "This is a very detailed readme about manufacturing defect detection using vision transformers..." * 10 # > 300 chars
            if model_id == "trash/bad-model":
                return "Too short."
            if model_id == "review/linked-model":
                return "Short but check https://github.com/repo"
            return ""
        mock_hf_instance.get_model_readme.side_effect = side_effect_readme

        # LLM Client
        mock_llm_instance = MockLLMClient.return_value
        mock_llm_instance.analyze_model.return_value = {
            "category": "Vision",
            "specialist_score": 9,
            "manufacturing_potential": True,
            "summary": "Great model",
            "manufacturing_use_cases": ["defect detection"]
        }

        # Reporter
        mock_reporter_instance = MockReporter.return_value

        # Run Main
        # We need to bypass argparse or mock it, but we can just call main() and rely on defaults
        # provided we patch sys.argv or use default args in main() if we refactored.
        # Since main() parses args, let's patch sys.argv
        with patch.object(sys, 'argv', ["main.py", "--limit", "10"]):
            main.main()

        # --- Verifications ---

        # 1. Deduplication: m3 should not be processed
        # calls to get_model_readme should verify this
        mock_hf_instance.get_model_readme.assert_any_call("new/specialist-model-7b")
        try:
            mock_hf_instance.get_model_readme.assert_any_call("old-model/123")
            assert False, "Should not have processed old-model/123"
        except AssertionError:
            pass # Good

        # 2. Parameter Filter: m2 should be skipped (70B > 10B)
        # Should not call get_model_readme for m2
        try:
            mock_hf_instance.get_model_readme.assert_any_call("big/giant-model-70b")
            assert False, "Should not have processed giant model"
        except AssertionError:
            pass

        # 3. Short README: m4 skipped, m5 saved as review_required
        # m4:
        # Should call get_model_readme
        mock_hf_instance.get_model_readme.assert_any_call("trash/bad-model")
        # Should NOT save m4
        # Verify save_model calls
        saved_models = [call[0][0] for call in mock_db_instance.save_model.call_args_list]
        saved_ids = [m['id'] for m in saved_models]

        self.assertNotIn("trash/bad-model", saved_ids)

        # m5: saved as review_required
        self.assertIn("review/linked-model", saved_ids)
        m5_data = next(m for m in saved_models if m['id'] == "review/linked-model")
        self.assertEqual(m5_data['status'], 'review_required')

        # 4. Valid Model: m1 processed and saved
        self.assertIn("new/specialist-model-7b", saved_ids)
        m1_data = next(m for m in saved_models if m['id'] == "new/specialist-model-7b")
        self.assertEqual(m1_data['status'], 'processed')
        self.assertEqual(m1_data['llm_analysis']['specialist_score'], 9)

        # 5. Reporting
        mock_reporter_instance.generate_markdown_report.assert_called_once()
        mock_reporter_instance.export_csv.assert_called_once()

if __name__ == '__main__':
    unittest.main()
