import unittest
from unittest.mock import MagicMock, patch, call
import time
import requests

from llm_client import LLMClient


class TestLLMClientBackoff(unittest.TestCase):
    """Tests for the _request_with_backoff method in LLMClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = LLMClient(
            api_url="http://test.example.com/api",
            model="test-model",
            api_key="test-key"
        )
        self.payload = {"model": "test-model", "messages": []}
        self.headers = {"Authorization": "Bearer test-key"}
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_successful_request_on_first_attempt(self, mock_sleep, mock_post):
        """Test that successful requests return immediately without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
        mock_post.return_value = mock_response
        
        result = self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(result, mock_response)
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 0)
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_rate_limit_with_retry_after_header(self, mock_sleep, mock_post):
        """Test that 429 with Retry-After header uses the specified wait time."""
        # First call returns 429, second succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '30'}
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        
        mock_post.side_effect = [mock_response_429, mock_response_200]
        
        result = self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(result, mock_response_200)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)
        
        # Check that sleep was called with approximately 30 seconds + small buffer
        sleep_time = mock_sleep.call_args[0][0]
        self.assertGreater(sleep_time, 30)
        self.assertLess(sleep_time, 36)  # 30 + max buffer of 5 + jitter
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_rate_limit_exponential_backoff(self, mock_sleep, mock_post):
        """Test that 429 without Retry-After uses exponential backoff."""
        # First three calls return 429, fourth succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        
        mock_post.side_effect = [
            mock_response_429,
            mock_response_429,
            mock_response_429,
            mock_response_200
        ]
        
        result = self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(result, mock_response_200)
        self.assertEqual(mock_post.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
        
        # Verify exponential backoff: 10s, 20s, 40s (with jitter)
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        
        # First sleep should be around 10s
        self.assertGreater(sleep_calls[0], 9)
        self.assertLess(sleep_calls[0], 12)
        
        # Second sleep should be around 20s
        self.assertGreater(sleep_calls[1], 18)
        self.assertLess(sleep_calls[1], 24)
        
        # Third sleep should be around 40s
        self.assertGreater(sleep_calls[2], 36)
        self.assertLess(sleep_calls[2], 48)
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_rate_limit_max_wait_cap(self, mock_sleep, mock_post):
        """Test that wait time is capped at max_wait_time (1 hour)."""
        # Simulate many 429 responses to exceed the cap
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {}
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        
        # Create enough 429s to trigger the cap (2^n > 3600)
        # After 8 attempts: 10 * 2^7 = 1280s, after 9: 2560s, after 10: 5120s > 3600
        mock_post.side_effect = [mock_response_429] * 10 + [mock_response_200]
        
        result = self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(result, mock_response_200)
        
        # Check that later sleep calls are capped at 3600s
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        
        # The last few should be capped at 3600 + jitter
        for sleep_time in sleep_calls[-3:]:
            self.assertLessEqual(sleep_time, 3600 * 1.1 + 1)  # Cap + 10% jitter + margin
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_server_error_limited_retry(self, mock_sleep, mock_post):
        """Test that 5xx errors retry up to 10 times then fail."""
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        # All calls return 500
        mock_post.return_value = mock_response_500
        
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._request_with_backoff(self.payload, self.headers)
        
        # Should try 10 times before giving up
        self.assertEqual(mock_post.call_count, 10)
        # Should sleep 9 times (no sleep after the 10th and final attempt)
        self.assertEqual(mock_sleep.call_count, 9)
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_server_error_recovery(self, mock_sleep, mock_post):
        """Test that server errors can recover within retry limit."""
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        
        # First 3 calls return 500, fourth succeeds
        mock_post.side_effect = [
            mock_response_500,
            mock_response_500,
            mock_response_500,
            mock_response_200
        ]
        
        result = self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(result, mock_response_200)
        self.assertEqual(mock_post.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
    
    @patch('llm_client.requests.post')
    def test_client_error_immediate_failure(self, mock_post):
        """Test that 4xx errors (except 429) fail immediately."""
        mock_response_401 = MagicMock()
        mock_response_401.status_code = 401
        mock_response_401.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        
        mock_post.return_value = mock_response_401
        
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._request_with_backoff(self.payload, self.headers)
        
        # Should only try once for auth errors
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('llm_client.requests.post')
    def test_timeout_error_immediate_failure(self, mock_post):
        """Test that timeout errors fail immediately."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with self.assertRaises(requests.exceptions.Timeout):
            self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('llm_client.requests.post')
    def test_connection_error_immediate_failure(self, mock_post):
        """Test that connection errors fail immediately."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client._request_with_backoff(self.payload, self.headers)
        
        self.assertEqual(mock_post.call_count, 1)


class TestLLMClientAnalyzeModelIntegration(unittest.TestCase):
    """Integration tests for analyze_model using _request_with_backoff."""
    
    @patch('llm_client.requests.post')
    @patch('llm_client.time.sleep')
    def test_analyze_model_with_rate_limit_recovery(self, mock_sleep, mock_post):
        """Test that analyze_model can recover from rate limits."""
        # First call returns 429, second succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '5'}
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "choices": [{
                "message": {
                    "content": '```json\n{"model_type": "Base Model", "newsletter_blurb": {"de": "Test", "en": "Test"}, "key_facts": {"de": [], "en": []}, "delta": {"what_changed": {"de": [], "en": []}, "why_it_matters": {"de": [], "en": []}}, "manufacturing": {"use_cases": {"de": [], "en": []}}, "edge": {}, "specialist_score": 5, "confidence": "medium", "unknowns": []}\n```'
                }
            }]
        }
        
        mock_post.side_effect = [mock_response_429, mock_response_200]
        
        client = LLMClient(
            api_url="http://test.example.com/api",
            model="test-model",
            api_key="test-key"
        )
        
        result = client.analyze_model("Test README", ["tag1"], yaml_meta={})
        
        self.assertIsNotNone(result)
        self.assertEqual(result.get("model_type"), "Base Model")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == '__main__':
    unittest.main()
