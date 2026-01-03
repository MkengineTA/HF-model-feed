import unittest

from run_stats import RunStats


class TestRunStatsDynamic(unittest.TestCase):
    def test_prolific_skipped_uploaders_picks_top_no_readme(self):
        stats = RunStats()

        for i in range(6):
            stats.record_skip(f"model-{i}", "skip:no_readme", author="amax0416")
        stats.record_skip("model-extra", "skip:params_active_too_large", author="amax0416")

        for i in range(4):
            stats.record_skip(f"other-{i}", "skip:no_readme", author="someone_else")

        prolific = stats.prolific_skipped_uploaders("skip:no_readme", min_count=5)

        self.assertIn("amax0416", prolific)
        self.assertNotIn("someone_else", prolific)


if __name__ == "__main__":
    unittest.main()
