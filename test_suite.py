import unittest
from unittest.mock import MagicMock
import model_filters as filters
from llm_client import LLMClient, extract_json_from_text

class TestFilters(unittest.TestCase):

    def test_extract_parameter_count(self):
        mock_info = MagicMock()
        mock_info.safetensors = MagicMock()
        mock_info.safetensors.total = 7000000000
        self.assertEqual(filters.extract_parameter_count(mock_info), 7.0)

        self.assertEqual(filters.extract_parameter_count(MagicMock(id="mistralai/Mistral-7B-v0.1", safetensors=None)), 7.0)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/MyModel-1.5b", safetensors=None)), 1.5)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/Tiny-270m", safetensors=None)), 0.27)

        mock_files = [{'path': 'model.safetensors', 'size': 21474836480}]
        self.assertAlmostEqual(filters.extract_parameter_count(MagicMock(id="user/unk", safetensors=None), mock_files), 10.0, places=1)

    def test_is_generative_visual(self):
        m = MagicMock()
        m.pipeline_tag = "text-to-image"
        self.assertTrue(filters.is_generative_visual(m, []))

        m.pipeline_tag = None
        self.assertTrue(filters.is_generative_visual(m, ["diffusers"]))

        self.assertTrue(filters.is_generative_visual(m, [], "This is a ComfyUI workflow."))
        self.assertFalse(filters.is_generative_visual(m, [], "Just a classification model."))

    def test_robotics_vqa_logic(self):
        m = MagicMock()
        m.pipeline_tag = "visual-question-answering"
        # Should NOT filter VQA even if "robot" mentioned?
        # The function returns TRUE if it IS robotics/VLA.
        # But if VQA, it returns FALSE (keep).
        self.assertFalse(filters.is_robotics_but_keep_vqa(m, ["robot"]))

        m.pipeline_tag = "reinforcement-learning"
        self.assertTrue(filters.is_robotics_but_keep_vqa(m, ["robot"]))

        m.pipeline_tag = None
        # Text mentions robot -> True
        self.assertTrue(filters.is_robotics_but_keep_vqa(m, [], "This controls a robot arm."))
        # Text mentions robot BUT also VQA -> False
        self.assertFalse(filters.is_robotics_but_keep_vqa(m, [], "Visual question answering for robot data."))

    def test_is_export(self):
        self.assertTrue(filters.is_export_or_conversion("model-onnx", []))
        self.assertTrue(filters.is_export_or_conversion("model", ["gguf"]))
        self.assertFalse(filters.is_export_or_conversion("clean-model", []))

    def test_is_merge(self):
        self.assertTrue(filters.is_merge("user/merged-model", ""))
        self.assertTrue(filters.is_merge("user/model", "Created with mergekit"))

    def test_is_secure(self):
        self.assertTrue(filters.is_secure([{'securityFileStatus': {'status': 'innocuous'}}]))
        self.assertFalse(filters.is_secure([{'securityFileStatus': {'status': 'unsafe'}}]))

class TestLLMClient(unittest.TestCase):

    def test_extract_json_robustness(self):
        valid_json = '{"key": "value"}'
        self.assertEqual(extract_json_from_text(valid_json), {"key": "value"})

        markdown_json = 'Here is the JSON:\n```json\n{"key": "value"}\n```'
        self.assertEqual(extract_json_from_text(markdown_json), {"key": "value"})

        trailing = '{"key": "value", }'
        self.assertEqual(extract_json_from_text(trailing), {"key": "value"})

        broken_json = '{"key": "val'
        self.assertIsNone(extract_json_from_text(broken_json))

if __name__ == '__main__':
    unittest.main()
