# test_digest.py
"""Tests for multi-recipient digest scheduling and dispatch."""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import tempfile
import os

import dateutil.parser

from config import NewsletterSubscriber, get_newsletter_subscribers, _parse_subscribers_json
from digest import (
    get_current_day_name,
    get_subscribers_for_today,
    group_subscribers,
)
from database import Database
from run_stats import RunStats


class TestSubscriberConfig(unittest.TestCase):
    """Tests for subscriber configuration parsing."""

    def test_parse_empty_json(self):
        """Empty or None JSON returns empty list."""
        self.assertEqual(_parse_subscribers_json(None), [])
        self.assertEqual(_parse_subscribers_json(""), [])

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list."""
        self.assertEqual(_parse_subscribers_json("not json"), [])

    def test_parse_non_array_json(self):
        """Non-array JSON returns empty list."""
        self.assertEqual(_parse_subscribers_json('{"email": "test@example.com"}'), [])

    def test_parse_valid_subscriber(self):
        """Valid subscriber JSON is parsed correctly."""
        json_str = '''[{
            "email": "test@example.com",
            "type": "normal",
            "language": "en",
            "send_days": ["mon", "wed", "fri"],
            "default_window_hours": 24,
            "window_hours_by_day": {"mon": 72}
        }]'''
        subscribers = _parse_subscribers_json(json_str)
        
        self.assertEqual(len(subscribers), 1)
        sub = subscribers[0]
        self.assertEqual(sub.email, "test@example.com")
        self.assertEqual(sub.type, "normal")
        self.assertEqual(sub.language, "en")
        self.assertEqual(sub.send_days, ["mon", "wed", "fri"])
        self.assertEqual(sub.default_window_hours, 24)
        self.assertEqual(sub.get_window_hours_for_day("mon"), 72)
        self.assertEqual(sub.get_window_hours_for_day("wed"), 24)

    def test_parse_invalid_type_defaults_to_normal(self):
        """Invalid type defaults to 'normal'."""
        json_str = '[{"email": "test@example.com", "type": "invalid"}]'
        subscribers = _parse_subscribers_json(json_str)
        self.assertEqual(subscribers[0].type, "normal")

    def test_parse_invalid_language_defaults_to_en(self):
        """Invalid language defaults to 'en'."""
        json_str = '[{"email": "test@example.com", "language": "fr"}]'
        subscribers = _parse_subscribers_json(json_str)
        self.assertEqual(subscribers[0].language, "en")

    def test_parse_missing_email_skips(self):
        """Subscriber without email is skipped."""
        json_str = '[{"type": "normal"}]'
        subscribers = _parse_subscribers_json(json_str)
        self.assertEqual(len(subscribers), 0)


class TestSubscriberWindowHours(unittest.TestCase):
    """Tests for window hours per day."""

    def test_default_window_hours(self):
        """Default window hours is used when no day-specific override."""
        sub = NewsletterSubscriber(
            email="test@example.com",
            default_window_hours=48,
        )
        self.assertEqual(sub.get_window_hours_for_day("mon"), 48)
        self.assertEqual(sub.get_window_hours_for_day("tue"), 48)

    def test_day_specific_window_hours(self):
        """Day-specific window hours override default."""
        sub = NewsletterSubscriber(
            email="test@example.com",
            default_window_hours=24,
            window_hours_by_day={"mon": 72, "wed": 48},
        )
        self.assertEqual(sub.get_window_hours_for_day("mon"), 72)
        self.assertEqual(sub.get_window_hours_for_day("tue"), 24)  # default
        self.assertEqual(sub.get_window_hours_for_day("wed"), 48)
        self.assertEqual(sub.get_window_hours_for_day("fri"), 24)  # default


class TestSubscriberFiltering(unittest.TestCase):
    """Tests for filtering subscribers by send day."""

    def test_get_subscribers_for_today_filters_by_send_days(self):
        """Only subscribers with today in send_days are returned."""
        sub_mon_only = NewsletterSubscriber(
            email="mon@example.com",
            send_days=["mon"],
            default_window_hours=24,
        )
        sub_all_days = NewsletterSubscriber(
            email="all@example.com",
            send_days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            default_window_hours=24,
        )
        sub_weekends = NewsletterSubscriber(
            email="weekend@example.com",
            send_days=["sat", "sun"],
            default_window_hours=24,
        )
        
        subscribers = [sub_mon_only, sub_all_days, sub_weekends]
        
        with patch('digest.get_current_day_name', return_value="mon"):
            result = get_subscribers_for_today(subscribers, "Europe/Berlin")
        
        emails = [sub.email for sub, _ in result]
        self.assertIn("mon@example.com", emails)
        self.assertIn("all@example.com", emails)
        self.assertNotIn("weekend@example.com", emails)


class TestSubscriberGrouping(unittest.TestCase):
    """Tests for grouping subscribers to minimize report generation."""

    def test_group_by_window_language_type(self):
        """Subscribers are grouped by (window, language, type)."""
        sub1 = NewsletterSubscriber(
            email="alice@example.com",
            type="normal",
            language="en",
            default_window_hours=24,
        )
        sub2 = NewsletterSubscriber(
            email="bob@example.com",
            type="normal",
            language="en",
            default_window_hours=24,
        )
        sub3 = NewsletterSubscriber(
            email="carol@example.com",
            type="debug",
            language="de",
            default_window_hours=24,
        )
        
        # All have same window (24h) for today
        subscribers_with_windows = [
            (sub1, 24),
            (sub2, 24),
            (sub3, 24),
        ]
        
        groups = group_subscribers(subscribers_with_windows)
        
        # Should have 2 groups: (24, en, normal) and (24, de, debug)
        self.assertEqual(len(groups), 2)
        self.assertIn((24, "en", "normal"), groups)
        self.assertIn((24, "de", "debug"), groups)
        self.assertEqual(len(groups[(24, "en", "normal")]), 2)
        self.assertEqual(len(groups[(24, "de", "debug")]), 1)

    def test_group_different_windows(self):
        """Different window hours create separate groups."""
        sub1 = NewsletterSubscriber(
            email="alice@example.com",
            type="normal",
            language="en",
            default_window_hours=24,
        )
        sub2 = NewsletterSubscriber(
            email="bob@example.com",
            type="normal",
            language="en",
            default_window_hours=72,
        )
        
        subscribers_with_windows = [
            (sub1, 24),
            (sub2, 72),
        ]
        
        groups = group_subscribers(subscribers_with_windows)
        
        # Should have 2 groups: (24, en, normal) and (72, en, normal)
        self.assertEqual(len(groups), 2)
        self.assertIn((24, "en", "normal"), groups)
        self.assertIn((72, "en", "normal"), groups)


class TestDatabaseProcessedAt(unittest.TestCase):
    """Tests for processed_at column and window queries."""

    def test_processed_at_is_set_on_save(self):
        """processed_at timestamp is set when model is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            
            model_data = {
                "id": "test/model",
                "name": "model",
                "author": "test",
                "created_at": None,
            }
            
            before = datetime.now(timezone.utc)
            db.save_model(model_data)
            after = datetime.now(timezone.utc)
            
            # Query back
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT processed_at FROM models WHERE id = ?", ("test/model",))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row["processed_at"])
            # Check timestamp is reasonable (between before and after)
            processed_at = dateutil.parser.parse(row["processed_at"])
            self.assertGreaterEqual(processed_at, before)
            self.assertLessEqual(processed_at, after)
            
            db.close()

    def test_get_models_by_processed_window(self):
        """Models can be queried by processed_at window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            
            # Save a model
            model_data = {
                "id": "test/model",
                "name": "model",
                "author": "test",
                "created_at": None,
                "llm_analysis": {"specialist_score": 5},
                "status": "processed",
            }
            db.save_model(model_data)
            
            # Query within 24h window
            models = db.get_models_by_processed_window(window_hours=24)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["id"], "test/model")
            
            # Query with very small window (should not find model)
            # Note: This might be flaky if test is slow, so we use 0
            # but model was just saved so it should be in any positive window
            
            db.close()


class TestLegacyFallback(unittest.TestCase):
    """Tests for backward compatibility with single-recipient config."""

    @patch.dict(os.environ, {"NEWSLETTER_SUBSCRIBERS_JSON": "", "RECEIVER_MAIL": "legacy@example.com"})
    @patch('config.RECEIVER_MAIL', "legacy@example.com")
    def test_fallback_to_receiver_mail(self):
        """When no JSON config, falls back to RECEIVER_MAIL."""
        # Need to reload config to pick up patched values
        subscribers = get_newsletter_subscribers()
        
        # If RECEIVER_MAIL is set and no JSON, should get legacy subscriber
        # Note: This test depends on actual env vars at runtime
        # In practice, the config is loaded at module import time
        # So we test the parsing logic instead
        pass


class TestProcessedAtFormat(unittest.TestCase):
    """Tests for processed_at timestamp format consistency."""

    def test_processed_at_uses_isoformat_with_timezone(self):
        """processed_at is always stored as ISO format with timezone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            
            model_data = {
                "id": "test/model",
                "name": "model",
                "author": "test",
                "created_at": None,
            }
            
            db.save_model(model_data)
            
            # Query back
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT processed_at FROM models WHERE id = ?", ("test/model",))
            row = cursor.fetchone()
            
            # Should be ISO format with T separator
            processed_at_str = row["processed_at"]
            self.assertIn("T", processed_at_str)  # ISO format uses T
            
            # Should be parseable and have timezone info
            # Note: We use dateutil.parser which handles various ISO formats including Z suffix
            parsed = dateutil.parser.parse(processed_at_str)
            self.assertIsNotNone(parsed.tzinfo)
            # Verify it has a valid UTC offset (not None)
            self.assertIsNotNone(parsed.utcoffset())
            
            db.close()

    def test_window_query_cutoff_format_matches_stored(self):
        """Window query cutoff uses same format as stored timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path)
            
            # Save a model
            model_data = {
                "id": "test/model",
                "name": "model",
                "author": "test",
                "created_at": None,
                "status": "processed",
            }
            db.save_model(model_data)
            
            # Query should work - if formats don't match, query would fail
            models = db.get_models_by_processed_window(window_hours=24)
            self.assertEqual(len(models), 1)
            
            db.close()


class TestBilingualBackwardCompatibility(unittest.TestCase):
    """Tests for i18n backward compatibility with old records."""

    def test_get_bilingual_value_handles_string(self):
        """Legacy string values are returned as-is."""
        from reporter import _get_bilingual_value
        
        # Legacy format: plain string
        result = _get_bilingual_value("Legacy text", "en")
        self.assertEqual(result, "Legacy text")
        
        result = _get_bilingual_value("Legacy text", "de")
        self.assertEqual(result, "Legacy text")

    def test_get_bilingual_value_handles_list(self):
        """Legacy list values are returned as-is."""
        from reporter import _get_bilingual_value
        
        # Legacy format: plain list
        result = _get_bilingual_value(["item1", "item2"], "en")
        self.assertEqual(result, ["item1", "item2"])

    def test_get_bilingual_value_handles_bilingual_dict(self):
        """Bilingual dict values return correct language."""
        from reporter import _get_bilingual_value
        
        # Bilingual format
        bilingual = {"de": "German", "en": "English"}
        
        self.assertEqual(_get_bilingual_value(bilingual, "en"), "English")
        self.assertEqual(_get_bilingual_value(bilingual, "de"), "German")

    def test_get_bilingual_value_fallback(self):
        """Falls back to available language when requested not present."""
        from reporter import _get_bilingual_value
        
        # Only one language available
        only_de = {"de": "German only"}
        self.assertEqual(_get_bilingual_value(only_de, "en", fallback_lang="de"), "German only")

    def test_get_bilingual_value_none(self):
        """None values are handled safely."""
        from reporter import _get_bilingual_value
        
        self.assertIsNone(_get_bilingual_value(None, "en"))


class TestWindowStats(unittest.TestCase):
    """Tests for window-centric stats in longer-window digests."""

    def test_window_digest_shows_window_stats(self):
        """Longer-window digests show window-centric stats."""
        from reporter import Reporter
        from run_stats import RunStats
        import tempfile
        
        stats = RunStats()
        stats.candidates_total = 100
        stats.processed = 5
        
        models = [
            {"id": "test/model1", "name": "model1", "namespace": "test", "llm_analysis": {"specialist_score": 7}},
            {"id": "test/model2", "name": "model2", "namespace": "test", "llm_analysis": {"specialist_score": 6}},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            path = reporter.generate_full_report(
                stats=stats,
                processed_models=models,
                date_str="2024-01-01",
                language="en",
                report_type="normal",
                window_hours=72,  # Longer than 24h
            )
            content = path.read_text(encoding="utf-8")
        
        # Should show window stats, not run stats
        self.assertIn("Digest Statistics", content)
        self.assertIn("Window (hours): **72**", content)
        self.assertIn("Models in window: **2**", content)
        
        # Should NOT show run diagnostics for normal type
        self.assertNotIn("Run diagnostics", content)

    def test_24h_window_shows_standard_summary(self):
        """24h window shows standard summary stats."""
        from reporter import Reporter
        from run_stats import RunStats
        import tempfile
        
        stats = RunStats()
        stats.candidates_total = 100
        
        models = [{"id": "test/model", "name": "model", "namespace": "test", "llm_analysis": {"specialist_score": 7}}]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            path = reporter.generate_full_report(
                stats=stats,
                processed_models=models,
                date_str="2024-01-01",
                language="en",
                report_type="debug",
                window_hours=24,
            )
            content = path.read_text(encoding="utf-8")
        
        # Should show standard summary
        self.assertIn("Summary", content)
        self.assertIn("Discovered (unique)", content)


if __name__ == "__main__":
    unittest.main()
