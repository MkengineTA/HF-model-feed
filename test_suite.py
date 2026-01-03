import unittest
from unittest.mock import MagicMock
import model_filters as filters
import param_estimator

class TestFilters(unittest.TestCase):
    def test_estimate_parameters(self):
        # This replaces the old test_extract_parameter_count
        mock_info = MagicMock()
        mock_info.safetensors = None

        # Test estimation from file size
        # 14GB file -> ~7B params
        files = [{'path': 'model.safetensors', 'size': 14_000_000_000}]

        api = MagicMock() # Mock HF API

        pe = param_estimator.estimate_parameters(api, "test/model", files)
        self.assertIsNotNone(pe.total_params)
        # Roughly 7B
        self.assertTrue(6000000000 < pe.total_params < 8000000000)
        self.assertEqual(pe.source, "filesize_fallback")

    def test_security_warnings(self):
        # Replaces test_is_secure
        # Safe file
        warns = param_estimator.security_warnings([{'path': 'model.safetensors'}])
        self.assertEqual(len(warns), 0)

        # Unsafe extension
        warns = param_estimator.security_warnings([{'path': 'virus.exe'}])
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0][0], "warn:executable_file_present")

        # HF Flagged
        warns = param_estimator.security_warnings([{'path': 'model.safetensors', 'securityFileStatus': 'unsafe'}])
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0][0], "warn:security_scan_flagged")

    def test_is_generative_visual(self):
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-to-image"
        self.assertTrue(filters.is_generative_visual(mock_info, []))

        mock_info.pipeline_tag = "text-classification"
        self.assertFalse(filters.is_generative_visual(mock_info, []))

    def test_is_excluded_content(self):
        self.assertTrue(filters.is_excluded_content("test", ["nsfw"]))
        self.assertFalse(filters.is_excluded_content("test", ["safe"]))

    def test_is_robotics_but_keep_vqa(self):
        mock_info = MagicMock()
        mock_info.pipeline_tag = "robotics"
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, ["robot"]))

        # VQA exception
        mock_info.pipeline_tag = "visual-question-answering"
        self.assertFalse(filters.is_robotics_but_keep_vqa(mock_info, ["robot"]))

    def test_is_robotics_reinforcement_learning_pipeline(self):
        """Test that reinforcement-learning pipeline is correctly identified as robotics."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "reinforcement-learning"
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], None))

    def test_is_robotics_common_words_not_triggering(self):
        """Test that common ML words like 'control' and 'policy' don't trigger robotics filter."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        # These words were removed from robotics keywords as they're too common
        readme_with_common_words = "This model provides control over generation and follows a policy."
        self.assertFalse(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_common_words))

    def test_llm_analysis_contains_robotics(self):
        """Test secondary filter for robotics terms in LLM-generated content."""
        # Test with robotics content
        robotics_analysis = {
            "newsletter_blurb": "Dieses Modell ist für Robotik-Steuerung geeignet.",
            "key_facts": ["Trainiert für Roboterarm-Bewegungen"],
            "delta": {"what_changed": [], "why_it_matters": []},
            "manufacturing": {"use_cases": ["Roboter-Steuerung"]}
        }
        self.assertTrue(filters.llm_analysis_contains_robotics(robotics_analysis))

        # Test without robotics content
        non_robotics_analysis = {
            "newsletter_blurb": "Ein Sprachmodell für Text-Generierung.",
            "key_facts": ["Unterstützt mehrere Sprachen"],
            "delta": {"what_changed": [], "why_it_matters": []},
            "manufacturing": {"use_cases": ["Dokumentenanalyse"]}
        }
        self.assertFalse(filters.llm_analysis_contains_robotics(non_robotics_analysis))

        # Test with None
        self.assertFalse(filters.llm_analysis_contains_robotics(None))

        # Test with empty dict
        self.assertFalse(filters.llm_analysis_contains_robotics({}))

    def test_is_export_or_conversion(self):
        self.assertTrue(filters.is_export_or_conversion("model-gguf", ["gguf"], []))
        self.assertFalse(filters.is_export_or_conversion("model-base", [], []))

    def test_compute_info_score_accepts_yaml_none(self):
        score = filters.compute_info_score(
            readme="Some README text without YAML frontmatter.",
            yaml_meta=None,
            tags=[],
            links_present=False,
        )
        self.assertIsInstance(score, int)


if __name__ == '__main__':
    unittest.main()
