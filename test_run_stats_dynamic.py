import unittest
from tempfile import TemporaryDirectory
import os

import config
from database import Database
from main import apply_dynamic_blacklist
import namespace_policy
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

    def test_apply_dynamic_blacklist_persists_and_ignores_fetch_fail(self):
        stats = RunStats()
        threshold = config.DYNAMIC_BLACKLIST_NO_README_MIN

        for i in range(threshold):
            stats.record_skip(f"model-{i}", "skip:no_readme", author="amax0416")

        for i in range(threshold + 5):
            stats.record_skip(f"fetch-{i}", "skip:readme_fetch_failed", author="flaky_author")

        with TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "models.db")
            db = Database(db_path)

            apply_dynamic_blacklist(db, stats, dry_run=False)

            conn = db.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT namespace, count, reason FROM dynamic_blacklist WHERE namespace = ?", ("amax0416",))
            row = cur.fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row["namespace"], "amax0416")
            self.assertEqual(row["count"], threshold)
            self.assertEqual(row["reason"], "skip:no_readme")

            decision, _ = namespace_policy.classify_namespace("amax0416")
            self.assertEqual(decision, "deny_blacklist")

        namespace_policy.set_dynamic_blacklist(set())


if __name__ == "__main__":
    unittest.main()
