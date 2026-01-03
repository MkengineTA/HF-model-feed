import unittest
from tempfile import TemporaryDirectory
import os
from datetime import datetime, timedelta, timezone

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

    def test_dynamic_blacklist_prune_and_max_count(self):
        with TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "models.db")
            db = Database(db_path)

            now = datetime.now(timezone.utc)
            # Insert older entry
            db.upsert_dynamic_blacklist({"old_ns": 3}, reason="skip:no_readme")
            conn = db.get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE dynamic_blacklist SET last_seen = ? WHERE namespace = ?",
                ((now - timedelta(days=10)).isoformat(), "old_ns"),
            )
            conn.commit()

            # Insert fresh entry, then upsert with higher count to trigger MAX
            db.upsert_dynamic_blacklist({"fresh_ns": 2}, reason="skip:no_readme")
            db.upsert_dynamic_blacklist({"fresh_ns": 5}, reason="skip:no_readme")

            removed = db.prune_dynamic_blacklist(now - timedelta(days=7))
            self.assertIn("old_ns", removed)

            cur.execute("SELECT count FROM dynamic_blacklist WHERE namespace = ?", ("fresh_ns",))
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["count"], 5)


if __name__ == "__main__":
    unittest.main()
