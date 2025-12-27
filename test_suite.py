import unittest
from unittest.mock import MagicMock
import filters
from llm_client import LLMClient

class TestFilters(unittest.TestCase):

    def test_extract_parameter_count(self):
        # Case 1: Metadata present (safetensors)
        mock_info = MagicMock()
        mock_info.safetensors = {"parameters": {"F32": 7000000000}} # 7B
        # This implementation depends on how huggingface_hub returns this.
        # Assuming we might need to mock differently or implement the logic to handle this structure.
        # But let's test the Regex fallback which is more critical to implement correctly.

        # Case 2: Regex in name
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="mistralai/Mistral-7B-v0.1", safetensors=None)), 7.0)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/MyModel-1.5b", safetensors=None)), 1.5)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/Model-8b-chat", safetensors=None)), 8.0)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/Huge-Model-70B", safetensors=None)), 70.0)

        # Case 3: No info
        self.assertIsNone(filters.extract_parameter_count(MagicMock(id="user/bert-base-uncased", safetensors=None)))

    def test_is_quantized(self):
        self.assertTrue(filters.is_quantized("TheBloke/Llama-2-7B-GGUF"))
        self.assertTrue(filters.is_quantized("user/model-awq"))
        self.assertFalse(filters.is_quantized("mistralai/Mistral-7B-v0.1"))

    def test_has_external_links(self):
        self.assertTrue(filters.has_external_links("Check out https://github.com/org/repo"))
        self.assertTrue(filters.has_external_links("Paper: arxiv.org/abs/1234.5678"))
        self.assertFalse(filters.has_external_links("This is a simple model description without links."))

    def test_is_excluded_content(self):
        self.assertTrue(filters.is_excluded_content("user/nsfw-roleplay-model", ["roleplay"]))
        self.assertTrue(filters.is_excluded_content("user/normal-model", ["uncensored", "nlp"]))
        self.assertFalse(filters.is_excluded_content("user/manufacturing-bert", ["manufacturing", "vision"]))

class TestLLMClient(unittest.TestCase):

    def test_parse_response_robustness(self):
        # We are testing the private method or logic inside analyze_model really,
        # but since analyze_model involves network, we might mock the network part
        # and see if it returns the expected dict.

        client = LLMClient("http://fake", "model")

        # Mocking the _send_request method if we had one, or requests.post
        # For now, let's assume we can test a helper method parse_json if we expose it,
        # or we just rely on integration tests later.
        # To strictly follow TDD "parse LLM response", I will assume a helper method exists or
        # I'll test the logic that cleans the string.

        # Let's verify JSON extraction logic.
        from llm_client import extract_json_from_text

        valid_json = '{"category": "Vision", "score": 8}'
        self.assertEqual(extract_json_from_text(valid_json), {"category": "Vision", "score": 8})

        markdown_json = 'Here is the JSON:\n```json\n{"category": "Code"}\n```'
        self.assertEqual(extract_json_from_text(markdown_json), {"category": "Code"})

        broken_json = '{"category": "Other", "summary": "Unfinished' # Should probably return None or raise error
        # Depending on implementation, let's expect None or Error.
        self.assertIsNone(extract_json_from_text(broken_json))

if __name__ == '__main__':
    unittest.main()
