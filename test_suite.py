import unittest
from unittest.mock import MagicMock
import filters
from llm_client import LLMClient, extract_json_from_text

class TestFilters(unittest.TestCase):

    def test_extract_parameter_count(self):
        mock_info = MagicMock()
        mock_info.safetensors = MagicMock()
        mock_info.safetensors.total = 7000000000
        self.assertEqual(filters.extract_parameter_count(mock_info), 7.0)

        self.assertEqual(filters.extract_parameter_count(MagicMock(id="mistralai/Mistral-7B-v0.1", safetensors=None)), 7.0)
        self.assertEqual(filters.extract_parameter_count(MagicMock(id="user/MyModel-1.5b", safetensors=None)), 1.5)

        mock_files = [{'path': 'model.safetensors', 'size': 21474836480}]
        self.assertAlmostEqual(filters.extract_parameter_count(MagicMock(id="user/unk", safetensors=None), mock_files), 10.0, places=1)

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

    def test_is_secure(self):
        self.assertTrue(filters.is_secure([{'securityFileStatus': {'status': 'innocuous'}}]))
        self.assertFalse(filters.is_secure([{'securityFileStatus': {'status': 'unsafe'}}]))
        self.assertFalse(filters.is_secure([{'securityFileStatus': {'virusTotalScan': {'status': 'infected'}}}]))


class TestLLMClient(unittest.TestCase):

    def test_extract_json_robustness(self):
        # 1. Plain JSON
        valid_json = '{"key": "value"}'
        self.assertEqual(extract_json_from_text(valid_json), {"key": "value"})

        # 2. Markdown Block
        markdown_json = 'Here is the JSON:\n```json\n{"key": "value"}\n```'
        self.assertEqual(extract_json_from_text(markdown_json), {"key": "value"})

        # 3. Trailing comma
        trailing = '{"key": "value", }'
        # The simple regex `re.sub(r',\s*([\]}])', r'\1', json_text)` handles this
        self.assertEqual(extract_json_from_text(trailing), {"key": "value"})

        # 4. Broken
        broken_json = '{"key": "val'
        self.assertIsNone(extract_json_from_text(broken_json))

    def test_headers(self):
        client = LLMClient("url", "model", api_key="sk-123", site_url="my-site", app_name="my-app", enable_reasoning=True)
        with unittest.mock.patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {}
            mock_post.return_value.status_code = 200

            client.analyze_model("content", [])

            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs['headers']
            payload = call_kwargs['json']

            self.assertEqual(headers['Authorization'], 'Bearer sk-123')
            self.assertEqual(headers['HTTP-Referer'], 'my-site')
            self.assertEqual(headers['X-Title'], 'my-app')
            self.assertEqual(payload['reasoning']['enabled'], True)

if __name__ == '__main__':
    unittest.main()
