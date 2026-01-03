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

    def test_is_export_or_conversion(self):
        self.assertTrue(filters.is_export_or_conversion("model-gguf", ["gguf"], []))
        self.assertFalse(filters.is_export_or_conversion("model-base", [], []))
        
        # Test ONNX filtering - checks repo name only, not namespace
        self.assertTrue(filters.is_export_or_conversion("ryanli123/onnx", [], []))
        self.assertTrue(filters.is_export_or_conversion("user/model-onnx", [], []))
        self.assertTrue(filters.is_export_or_conversion("user/onnx-model", [], []))
        
        # Test ONNX with other delimiters
        self.assertTrue(filters.is_export_or_conversion("model-onnx", [], []))
        self.assertTrue(filters.is_export_or_conversion("model_onnx", [], []))
        self.assertTrue(filters.is_export_or_conversion("onnx-model", [], []))
        
        # Test ONNX as tag still works
        self.assertTrue(filters.is_export_or_conversion("some-model", ["onnx"], []))
        
        # Test models that should NOT be filtered
        self.assertFalse(filters.is_export_or_conversion("myonnx", [], []))
        self.assertFalse(filters.is_export_or_conversion("onnxmodel", [], []))
        
        # Namespace should NOT affect filtering (only repo name matters)
        self.assertFalse(filters.is_export_or_conversion("onnx-community/model", [], []))
        self.assertFalse(filters.is_export_or_conversion("gguf-user/model", [], []))
        self.assertFalse(filters.is_export_or_conversion("myonnx/normalmodel", [], []))

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
