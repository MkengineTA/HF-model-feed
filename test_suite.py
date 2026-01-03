import unittest
from unittest.mock import MagicMock
import model_filters as filters
import param_estimator
from main import should_block_model_name

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
        # Tag-based evidence is strong - should return True
        self.assertTrue(filters.is_export_or_conversion("model-gguf", ["gguf"], []))
        self.assertTrue(filters.is_export_or_conversion("some-model", ["onnx"], []))
        self.assertTrue(filters.is_export_or_conversion("some-model", ["gptq"], []))
        self.assertTrue(filters.is_export_or_conversion("some-model", ["awq"], []))
        
        # No evidence - should return False
        self.assertFalse(filters.is_export_or_conversion("model-base", [], []))
        
        # Name-only markers without file/tag evidence now return False (suspected, not strong)
        # These are suspected but NOT strong evidence
        self.assertFalse(filters.is_export_or_conversion("ryanli123/onnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("user/model-onnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("user/onnx-model", [], []))
        self.assertFalse(filters.is_export_or_conversion("model-onnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("model_onnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("onnx-model", [], []))
        
        # Models that should NOT be filtered (no match at all)
        self.assertFalse(filters.is_export_or_conversion("myonnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("onnxmodel", [], []))
        
        # Namespace should NOT affect filtering (only repo name matters)
        self.assertFalse(filters.is_export_or_conversion("onnx-community/model", [], []))
        self.assertFalse(filters.is_export_or_conversion("gguf-user/model", [], []))
        self.assertFalse(filters.is_export_or_conversion("myonnx/normalmodel", [], []))

    def test_classify_export_conversion_evidence_file_extensions(self):
        """Test that *.onnx, *.gguf, *.ggml files produce strong evidence."""
        # ONNX file -> strong evidence
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "model.onnx"}]
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "onnx")
        self.assertEqual(result["evidence"]["matched_file"], "model.onnx")
        
        # GGUF file -> strong evidence
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "model-Q4_K_M.gguf"}]
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "gguf")
        self.assertIn(".gguf", result["evidence"]["matched_file"])
        
        # GGML file -> strong evidence
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "weights.ggml"}]
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "ggml")
        
        # Nested path with ONNX file
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "models/fp16/model.onnx"}]
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "onnx")

    def test_classify_export_conversion_evidence_tags(self):
        """Test that tag-based evidence is strong."""
        # GGUF tag -> strong
        result = filters.classify_export_conversion_evidence(
            "user/model", ["gguf"], []
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "gguf")
        self.assertEqual(result["evidence"]["matched_tag"], "gguf")
        
        # ONNX tag -> strong
        result = filters.classify_export_conversion_evidence(
            "user/model", ["onnx", "transformers"], []
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "onnx")
        
        # GPTQ tag -> strong
        result = filters.classify_export_conversion_evidence(
            "user/model", ["GPTQ"], []  # Case insensitive
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "gptq")
        
        # AWQ tag -> strong
        result = filters.classify_export_conversion_evidence(
            "user/model", ["awq"], []
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "awq")

    def test_classify_export_conversion_evidence_config_files(self):
        """Test that config files produce suspected evidence (not strong alone)."""
        # gptq_config.json alone -> suspected (needs corroboration for strong)
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "gptq_config.json"}]
        )
        self.assertEqual(result["level"], "suspected")
        self.assertEqual(result["format"], "gptq")
        
        # awq_config.json alone -> suspected
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "awq_config.json"}]
        )
        self.assertEqual(result["level"], "suspected")
        self.assertEqual(result["format"], "awq")
        
        # Config + matching name pattern = strong (corroborated)
        result = filters.classify_export_conversion_evidence(
            "user/model-gptq", [], [{"path": "gptq_config.json"}]
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "gptq")
        
        # Config + README confirmation = strong
        result = filters.classify_export_conversion_evidence(
            "user/model", [], [{"path": "gptq_config.json"}],
            readme_text="This model was quantized to GPTQ format."
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "gptq")

    def test_classify_export_conversion_evidence_name_only_suspected(self):
        """Test that export format name markers produce suspected evidence (not strong)."""
        # Model name with GPTQ but no tags/files -> suspected
        result = filters.classify_export_conversion_evidence(
            "user/model-gptq", [], []
        )
        self.assertEqual(result["level"], "suspected")
        self.assertIsNotNone(result["evidence"]["matched_name"])
        
        # Model name with GGUF but no tags/files -> suspected
        result = filters.classify_export_conversion_evidence(
            "user/model-GGUF", [], []
        )
        self.assertEqual(result["level"], "suspected")
        self.assertEqual(result["format"], "gguf")
        
        # Model name with AWQ but no tags/files -> suspected
        result = filters.classify_export_conversion_evidence(
            "user/model-awq", [], []
        )
        self.assertEqual(result["level"], "suspected")
    
    def test_classify_export_conversion_evidence_generic_dtype_no_warning(self):
        """Test that generic dtype patterns (fp16/bf16/int8) do NOT trigger warnings."""
        # fp16 alone -> none (not suspected, no warning)
        result = filters.classify_export_conversion_evidence(
            "user/model-fp16", [], []
        )
        self.assertEqual(result["level"], "none")
        
        # bf16 alone -> none
        result = filters.classify_export_conversion_evidence(
            "user/model-bf16", [], []
        )
        self.assertEqual(result["level"], "none")
        
        # int8 alone -> none
        result = filters.classify_export_conversion_evidence(
            "user/model-int8", [], []
        )
        self.assertEqual(result["level"], "none")
        
        # Q4_K_M style patterns -> none
        result = filters.classify_export_conversion_evidence(
            "user/model-Q4_K_M", [], []
        )
        self.assertEqual(result["level"], "none")

    def test_classify_export_conversion_evidence_none(self):
        """Test that models without any evidence return none."""
        result = filters.classify_export_conversion_evidence(
            "user/normal-model", [], []
        )
        self.assertEqual(result["level"], "none")
        self.assertIsNone(result["format"])
        
        result = filters.classify_export_conversion_evidence(
            "user/normal-model", ["transformers", "pytorch"], [{"path": "model.safetensors"}]
        )
        self.assertEqual(result["level"], "none")

    def test_classify_export_conversion_evidence_readme_confirmation(self):
        """Test that README keywords can upgrade suspected to strong."""
        # Name-only match with README confirmation -> strong
        result = filters.classify_export_conversion_evidence(
            "user/model-onnx", [], [],
            readme_text="This model was converted to ONNX format for deployment."
        )
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["format"], "onnx")
        self.assertIsNotNone(result["evidence"]["matched_readme_keyword"])
        
        # Name-only match without README confirmation -> suspected
        result = filters.classify_export_conversion_evidence(
            "user/model-gptq", [], [],
            readme_text="This is a great language model for text generation."
        )
        self.assertEqual(result["level"], "suspected")
        
        # Name-only match with DIFFERENT format README should NOT upgrade
        # (e.g., gptq name but onnx keywords in README stays suspected)
        result = filters.classify_export_conversion_evidence(
            "user/model-gptq", [], [],
            readme_text="This model was converted to ONNX format for deployment."
        )
        self.assertEqual(result["level"], "suspected")
        self.assertEqual(result["format"], "gptq")  # Original format preserved

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
