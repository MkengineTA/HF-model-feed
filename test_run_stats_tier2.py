import unittest
from datetime import datetime, timezone

from run_stats import RunStats


class TestRunStatsTier2(unittest.TestCase):
    def setUp(self):
        self.stats = RunStats()

    def test_tier2_candidates_empty_initially(self):
        # Initially, no tier2 candidates
        self.assertEqual(self.stats.tier2_candidates, {})

    def test_record_tier2_candidate_basic(self):
        # Record a single tier2 candidate
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        
        self.assertIn("test-user", self.stats.tier2_candidates)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["followers"], 250)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["is_pro"], False)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["count"], 1)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["model_ids"], ["test-user/model-1"])

    def test_record_tier2_candidate_multiple_models(self):
        # Record the same namespace multiple times with different models
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-2"
        )
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-3"
        )
        
        self.assertEqual(self.stats.tier2_candidates["test-user"]["count"], 3)
        self.assertEqual(len(self.stats.tier2_candidates["test-user"]["model_ids"]), 3)
        self.assertIn("test-user/model-1", self.stats.tier2_candidates["test-user"]["model_ids"])
        self.assertIn("test-user/model-2", self.stats.tier2_candidates["test-user"]["model_ids"])
        self.assertIn("test-user/model-3", self.stats.tier2_candidates["test-user"]["model_ids"])

    def test_record_tier2_candidate_duplicate_model(self):
        # Recording the same model_id twice should not duplicate
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        
        # Count should be 2 (two occurrences)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["count"], 2)
        # But model_ids should only have one entry
        self.assertEqual(len(self.stats.tier2_candidates["test-user"]["model_ids"]), 1)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["model_ids"], ["test-user/model-1"])

    def test_record_tier2_candidate_pro_user(self):
        # Record a PRO user
        self.stats.record_tier2_candidate(
            namespace="pro-user",
            followers=150,
            is_pro=True,
            model_id="pro-user/model-1"
        )
        
        self.assertEqual(self.stats.tier2_candidates["pro-user"]["is_pro"], True)
        self.assertEqual(self.stats.tier2_candidates["pro-user"]["followers"], 150)

    def test_record_tier2_candidate_no_model_id(self):
        # Record without model_id
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id=None
        )
        
        self.assertEqual(self.stats.tier2_candidates["test-user"]["count"], 1)
        self.assertEqual(self.stats.tier2_candidates["test-user"]["model_ids"], [])

    def test_record_tier2_candidate_multiple_namespaces(self):
        # Record multiple different namespaces
        self.stats.record_tier2_candidate(
            namespace="user1",
            followers=200,
            is_pro=False,
            model_id="user1/model-1"
        )
        self.stats.record_tier2_candidate(
            namespace="user2",
            followers=300,
            is_pro=True,
            model_id="user2/model-1"
        )
        self.stats.record_tier2_candidate(
            namespace="user3",
            followers=500,
            is_pro=False,
            model_id="user3/model-1"
        )
        
        self.assertEqual(len(self.stats.tier2_candidates), 3)
        self.assertIn("user1", self.stats.tier2_candidates)
        self.assertIn("user2", self.stats.tier2_candidates)
        self.assertIn("user3", self.stats.tier2_candidates)

    def test_record_tier2_candidate_no_followers(self):
        # Record without followers information
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=None,
            is_pro=True,
            model_id="test-user/model-1"
        )
        
        self.assertIsNone(self.stats.tier2_candidates["test-user"]["followers"])
        self.assertEqual(self.stats.tier2_candidates["test-user"]["is_pro"], True)


if __name__ == "__main__":
    unittest.main()
