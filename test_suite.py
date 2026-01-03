import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
import os
import model_filters as filters
import param_estimator
from main import should_block_model_name

class TestFilters(unittest.TestCase):
    @patch("param_estimator.hf_hub_download")
    def test_estimate_parameters(self, mock_download):
        # This replaces the old test_extract_parameter_count
        # Mock hf_hub_download to avoid network access
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock config.json
            cfg_path = os.path.join(tmpdir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "intermediate_size": 11008,
                    "vocab_size": 32000,
                }, f)
            
            # Return the config path only when config.json is requested, raise for others
            def mock_download_side_effect(repo_id, filename):
                if filename == "config.json":
                    return cfg_path
                raise FileNotFoundError(f"Mock: {filename} not found")
            
            mock_download.side_effect = mock_download_side_effect

            files = [{'path': 'model.safetensors', 'size': 14_000_000_000}]

            api = MagicMock() # Mock HF API
            # Ensure safetensors metadata does not try network
            api.get_safetensors_metadata.side_effect = Exception("offline test")

            pe = param_estimator.estimate_parameters(api, "test/model", files)
            self.assertIsNotNone(pe.total_params)
            # Since we have a config.json, source should be config_heuristic
            self.assertEqual(pe.source, "config_heuristic")

    @patch("param_estimator.hf_hub_download")
    def test_estimate_parameters_filesize_fallback(self, mock_download):
        # Test that filesize fallback works when config is not available
        # Mock hf_hub_download to raise FileNotFoundError for both config.json and params.json
        def mock_download_side_effect(repo_id, filename):
            raise FileNotFoundError(f"Mock: {filename} not found")
        
        mock_download.side_effect = mock_download_side_effect

        files = [{'path': 'model.safetensors', 'size': 14_000_000_000}]
        api = MagicMock()
        api.get_safetensors_metadata.side_effect = Exception("offline test")

        pe = param_estimator.estimate_parameters(api, "test/model", files)
        self.assertIsNotNone(pe.total_params)
        # Should fall back to filesize estimate
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

    def test_short_keywords_word_boundary_matching(self):
        """Test that short keywords like 'rl' and 'ros' use word boundaries to avoid false positives."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        
        # "rl" should NOT match in "world" or "real-world"
        readme_with_world = "This model achieves real-world performance in various tasks."
        self.assertFalse(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_world))
        
        # "ros" should NOT match in "across" or "micros"
        readme_with_across = "The model performs across multiple benchmarks and microservices."
        self.assertFalse(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_across))
        
        # But standalone "rl" SHOULD match
        readme_with_rl = "This is an RL agent trained for the task."
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_rl))
        
        # And "ros" as standalone SHOULD match
        readme_with_ros = "This model integrates with ROS for robot control."
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_ros))
        
        # "ros2" SHOULD match
        readme_with_ros2 = "Compatible with ROS2 middleware."
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], readme_with_ros2))

    def test_robotics_specific_terms_still_match(self):
        """Test that robotics-specific terms like mujoco, pybullet still trigger the filter."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        
        readme_mujoco = "Trained in MuJoCo simulation environment."
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], readme_mujoco))
        
        readme_pybullet = "Uses PyBullet for physics simulation."
        self.assertTrue(filters.is_robotics_but_keep_vqa(mock_info, [], readme_pybullet))

    def test_llm_analysis_contains_robotics(self):
        """Test secondary filter for robotics terms in LLM-generated content."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "reinforcement-learning"  # Evidence: robotics pipeline
        
        # Test with robotics content AND pipeline evidence
        robotics_analysis = {
            "newsletter_blurb": "Dieses Modell ist für Robotik-Steuerung geeignet.",
            "key_facts": ["Trainiert für Roboterarm-Bewegungen"],
            "delta": {"what_changed": [], "why_it_matters": []},
            "manufacturing": {"use_cases": ["Roboter-Steuerung"]}
        }
        is_robotics, matched = filters.llm_analysis_contains_robotics(
            robotics_analysis, model_info=mock_info, tags=[], readme_text=None
        )
        self.assertTrue(is_robotics)
        self.assertIsNotNone(matched)  # Should return matched keyword for debuggability

        # Test without robotics content
        mock_info.pipeline_tag = "text-generation"
        non_robotics_analysis = {
            "newsletter_blurb": "Ein Sprachmodell für Text-Generierung.",
            "key_facts": ["Unterstützt mehrere Sprachen"],
            "delta": {"what_changed": [], "why_it_matters": []},
            "manufacturing": {"use_cases": ["Dokumentenanalyse"]}
        }
        is_robotics, matched = filters.llm_analysis_contains_robotics(
            non_robotics_analysis, model_info=mock_info, tags=[], readme_text=None
        )
        self.assertFalse(is_robotics)
        self.assertIsNone(matched)

        # Test with None: exercise early-return path when no LLM analysis is available.
        # Other parameters are intentionally omitted because they are not used in this case.
        is_robotics, matched = filters.llm_analysis_contains_robotics(None)
        self.assertFalse(is_robotics)

        # Test with empty dict: same early-return behavior as None/absent analysis.
        # Again, additional parameters are irrelevant and omitted for clarity.
        is_robotics, matched = filters.llm_analysis_contains_robotics({})
        self.assertFalse(is_robotics)

    def test_llm_analysis_requires_evidence(self):
        """Test that secondary filter only triggers when evidence exists in README/tags/pipeline."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"  # NOT a robotics pipeline
        
        # LLM mentions robotics but no evidence in README/tags/pipeline
        robotics_analysis = {
            "newsletter_blurb": "This model can be used for robot control.",
            "key_facts": ["Supports robotics applications"],
            "delta": {"what_changed": [], "why_it_matters": []},
            "manufacturing": {"use_cases": []}
        }
        
        # No evidence in README or tags - should NOT trigger
        is_robotics, matched = filters.llm_analysis_contains_robotics(
            robotics_analysis, model_info=mock_info, tags=[], readme_text="A simple language model."
        )
        self.assertFalse(is_robotics)
        
        # WITH evidence in README - should trigger
        is_robotics, matched = filters.llm_analysis_contains_robotics(
            robotics_analysis, model_info=mock_info, tags=[], readme_text="This is a robot control model."
        )
        self.assertTrue(is_robotics)
        self.assertIsNotNone(matched)

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

    def test_should_block_model_name(self):
        counts = {}
        blocked = set()

        should_block, occurrences = should_block_model_name("foo", counts, blocked, 3)
        self.assertFalse(should_block)
        self.assertEqual(occurrences, 1)

        should_block, occurrences = should_block_model_name("FOO", counts, blocked, 3)
        self.assertFalse(should_block)
        self.assertEqual(occurrences, 2)

        should_block, occurrences = should_block_model_name("foo", counts, blocked, 3)
        self.assertTrue(should_block)
        self.assertEqual(occurrences, 3)

        should_block, occurrences = should_block_model_name("foo", counts, blocked, 3)
        self.assertTrue(should_block)
        self.assertEqual(occurrences, 4)


if __name__ == '__main__':
    unittest.main()
