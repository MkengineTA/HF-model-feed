import unittest
import tempfile
import os
from datetime import datetime, timezone, timedelta

from database import Database


class TestDatabaseDynamicWhitelist(unittest.TestCase):
    def setUp(self):
        # Create a temporary database for testing
        self.temp_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        self.db = Database(self.temp_db_path)

    def tearDown(self):
        # Clean up
        self.db.close()
        os.close(self.temp_fd)
        os.unlink(self.temp_db_path)

    def test_empty_dynamic_whitelist(self):
        # Initially, dynamic whitelist should be empty
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, set())

    def test_upsert_dynamic_whitelist_single(self):
        # Add a single namespace
        self.db.upsert_dynamic_whitelist({"test-user": 1}, reason="tier3_org")
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"test-user"})

    def test_upsert_dynamic_whitelist_multiple(self):
        # Add multiple namespaces
        additions = {
            "user1": 1,
            "user2": 2,
            "user3": 3,
        }
        self.db.upsert_dynamic_whitelist(additions, reason="tier3_org")
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"user1", "user2", "user3"})

    def test_upsert_dynamic_whitelist_update_existing(self):
        # Add a namespace
        self.db.upsert_dynamic_whitelist({"test-user": 1}, reason="tier3_org")
        
        # Update it with a higher count
        self.db.upsert_dynamic_whitelist({"test-user": 5}, reason="tier3_org_updated")
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"test-user"})
        
        # Check that count was updated (query the DB directly)
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT count, reason FROM dynamic_whitelist WHERE namespace = ?", ("test-user",))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["count"], 5)
        self.assertEqual(row["reason"], "tier3_org_updated")

    def test_remove_dynamic_whitelist(self):
        # Add some namespaces
        self.db.upsert_dynamic_whitelist({"user1": 1, "user2": 2, "user3": 3}, reason="test")
        
        # Remove one
        removed = self.db.remove_dynamic_whitelist({"user2"})
        self.assertEqual(removed, {"user2"})
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"user1", "user3"})

    def test_remove_dynamic_whitelist_multiple(self):
        # Add some namespaces
        self.db.upsert_dynamic_whitelist({"user1": 1, "user2": 2, "user3": 3, "user4": 4}, reason="test")
        
        # Remove multiple
        removed = self.db.remove_dynamic_whitelist({"user1", "user3"})
        self.assertEqual(removed, {"user1", "user3"})
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"user2", "user4"})

    def test_remove_dynamic_whitelist_empty_set(self):
        # Removing empty set should do nothing
        removed = self.db.remove_dynamic_whitelist(set())
        self.assertEqual(removed, set())

    def test_prune_dynamic_whitelist(self):
        # Add namespaces with different last_seen times
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(days=100)).isoformat()
        recent_time = (now - timedelta(days=5)).isoformat()
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Add one old entry
        cursor.execute(
            "INSERT INTO dynamic_whitelist (namespace, added_at, reason, count, last_seen) VALUES (?, ?, ?, ?, ?)",
            ("old-user", old_time, "test", 1, old_time),
        )
        
        # Add one recent entry
        cursor.execute(
            "INSERT INTO dynamic_whitelist (namespace, added_at, reason, count, last_seen) VALUES (?, ?, ?, ?, ?)",
            ("recent-user", recent_time, "test", 1, recent_time),
        )
        conn.commit()
        
        # Prune entries older than 30 days
        cutoff = now - timedelta(days=30)
        removed = self.db.prune_dynamic_whitelist(cutoff)
        
        self.assertEqual(removed, {"old-user"})
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"recent-user"})

    def test_prune_dynamic_whitelist_no_old_entries(self):
        # Add only recent entries
        self.db.upsert_dynamic_whitelist({"user1": 1, "user2": 2}, reason="test")
        
        # Prune old entries
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=30)
        removed = self.db.prune_dynamic_whitelist(cutoff)
        
        # Nothing should be removed
        self.assertEqual(removed, set())
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"user1", "user2"})

    def test_upsert_empty_additions(self):
        # Upsert with empty dict should do nothing
        self.db.upsert_dynamic_whitelist({}, reason="test")
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, set())

    def test_upsert_with_empty_namespace(self):
        # Namespaces with empty strings should be ignored
        self.db.upsert_dynamic_whitelist({"user1": 1, "": 2, "user2": 3}, reason="test")
        
        result = self.db.get_dynamic_whitelist()
        self.assertEqual(result, {"user1", "user2"})


if __name__ == "__main__":
    unittest.main()
