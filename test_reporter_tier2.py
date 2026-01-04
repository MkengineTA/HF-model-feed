import unittest
import tempfile
import os
from pathlib import Path

from run_stats import RunStats
from reporter import Reporter
import config


class TestReporterTier2Section(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = Reporter(output_dir=self.temp_dir)
        self.stats = RunStats()

    def tearDown(self):
        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tier2_section_not_present_when_empty(self):
        # When there are no tier2 candidates, section should not be present
        path = self.reporter.write_markdown_report(
            stats=self.stats,
            processed_models=[],
            date_str="2024-01-01"
        )
        
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("## Tier 2 whitelist candidates", content)

    def test_tier2_section_not_present_when_disabled(self):
        # Add some tier2 candidates
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        
        # Temporarily disable the feature
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = False
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("## Tier 2 whitelist candidates", content)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_present_with_candidates(self):
        # Add some tier2 candidates
        self.stats.record_tier2_candidate(
            namespace="test-user",
            followers=250,
            is_pro=False,
            model_id="test-user/model-1"
        )
        
        # Ensure the feature is enabled
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            self.assertIn("## Tier 2 whitelist candidates (review)", content)
            self.assertIn("test-user", content)
            self.assertIn("250 followers", content)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_shows_pro_badge(self):
        # Add a PRO user
        self.stats.record_tier2_candidate(
            namespace="pro-user",
            followers=150,
            is_pro=True,
            model_id="pro-user/model-1"
        )
        
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            self.assertIn("pro-user", content)
            self.assertIn("PRO", content)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_shows_model_count(self):
        # Add multiple models for a namespace
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
        
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            self.assertIn("3 model(s) this run", content)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_shows_example_models(self):
        # Add models with links
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
        
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            self.assertIn("examples:", content)
            self.assertIn("https://huggingface.co/test-user/model-1", content)
            self.assertIn("https://huggingface.co/test-user/model-2", content)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_sorted_by_count(self):
        # Add multiple namespaces with different counts
        # user3 has 5 models
        for i in range(5):
            self.stats.record_tier2_candidate(
                namespace="user3",
                followers=200,
                is_pro=False,
                model_id=f"user3/model-{i}"
            )
        
        # user2 has 3 models
        for i in range(3):
            self.stats.record_tier2_candidate(
                namespace="user2",
                followers=300,
                is_pro=False,
                model_id=f"user2/model-{i}"
            )
        
        # user1 has 1 model
        self.stats.record_tier2_candidate(
            namespace="user1",
            followers=500,
            is_pro=False,
            model_id="user1/model-1"
        )
        
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            
            # Find positions of each namespace in the content
            pos_user3 = content.find("user3")
            pos_user2 = content.find("user2")
            pos_user1 = content.find("user1")
            
            # user3 should come before user2 (more models)
            self.assertLess(pos_user3, pos_user2)
            # user2 should come before user1 (more models)
            self.assertLess(pos_user2, pos_user1)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value

    def test_tier2_section_limits_max_items(self):
        # Add many tier2 candidates
        for i in range(50):
            self.stats.record_tier2_candidate(
                namespace=f"user{i}",
                followers=200 + i,
                is_pro=False,
                model_id=f"user{i}/model-1"
            )
        
        original_tier2_value = config.REPORT_INCLUDE_TIER2_REVIEW
        original_max_value = getattr(config, "TIER2_REVIEW_MAX_ITEMS", 30)
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        config.TIER2_REVIEW_MAX_ITEMS = 10  # Limit to 10
        
        try:
            path = self.reporter.write_markdown_report(
                stats=self.stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            
            # Count how many user entries are in the tier2 section
            # Extract just the tier2 section
            start = content.find("## Tier 2 whitelist candidates")
            end = content.find("## Processed models", start)
            tier2_section = content[start:end] if start != -1 and end != -1 else content[start:]
            
            # Count lines starting with "- **user" (indicating a namespace entry)
            user_lines = [line for line in tier2_section.split("\n") if line.strip().startswith("- **user")]
            
            # Should be at most 10
            self.assertLessEqual(len(user_lines), 10)
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_tier2_value
            config.TIER2_REVIEW_MAX_ITEMS = original_max_value


if __name__ == "__main__":
    unittest.main()
