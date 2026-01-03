import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reporter import Reporter
from run_stats import RunStats


class TestReporter(unittest.TestCase):
    def test_underscores_are_escaped_in_model_links(self):
        stats = RunStats()
        model = {
            "id": "author/model_with_underscores",
            "name": "model_with_underscores",
            "namespace": "author",
            "llm_analysis": {"specialist_score": 8},
        }

        with TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            report_path = reporter.generate_full_report(stats, [model], date_str="2024-01-01")
            content = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("[model\\_with\\_underscores]", content)
        self.assertIn("[author/model\\_with\\_underscores]", content)


    def test_delta_lists_render_with_blank_lines_and_no_confidence_block(self):
        stats = RunStats()
        model = {
            "id": "author/sample_model",
            "name": "sample_model",
            "namespace": "author",
            "llm_analysis": {
                "specialist_score": 7,
                "delta": {
                    "what_changed": ["Changed dataset", "New objective"],
                    "why_it_matters": ["Better accuracy", "More stable"],
                },
                "manufacturing": {"use_cases": ["Text completion", "QA"]},
            },
        }

        with TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            report_path = reporter.generate_full_report(stats, [model], date_str="2024-01-01")
            content = Path(report_path).read_text(encoding="utf-8")

        # Default language is German
        self.assertIn("**Was ist neu?**\n\n- Changed dataset\n- New objective", content)
        self.assertIn("**Warum relevant?**\n\n- Better accuracy\n- More stable", content)
        self.assertIn("**Anwendungsfälle**\n\n- Text completion\n- QA", content)
        self.assertNotIn("Confidence:", content)
        self.assertNotIn("Unknowns:", content)

    def test_english_report_generation(self):
        """Test report generation with English language."""
        stats = RunStats()
        model = {
            "id": "author/sample_model",
            "name": "sample_model",
            "namespace": "author",
            "llm_analysis": {
                "specialist_score": 7,
                "delta": {
                    "what_changed": ["Changed dataset"],
                    "why_it_matters": ["Better accuracy"],
                },
                "manufacturing": {"use_cases": ["Text completion"]},
            },
        }

        with TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            report_path = reporter.generate_full_report(
                stats, [model], date_str="2024-01-01", language="en"
            )
            content = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("## Summary", content)
        self.assertIn("**What's new?**", content)
        self.assertIn("**Why it matters?**", content)
        self.assertIn("**Use cases**", content)

    def test_normal_report_hides_debug_sections(self):
        """Test that normal report type hides debug sections."""
        stats = RunStats()
        stats.record_skip("test/model", "skip:test_reason", author="test")
        stats.record_warn("test/model", "warn:test_reason", author="test")
        
        model = {
            "id": "author/sample_model",
            "name": "sample_model",
            "namespace": "author",
            "llm_analysis": {"specialist_score": 7},
        }

        with TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            
            # Debug report should include skip/warn sections
            debug_path = reporter.generate_full_report(
                stats, [model], date_str="2024-01-01", report_type="debug"
            )
            debug_content = Path(debug_path).read_text(encoding="utf-8")
            self.assertIn("Top Skip", debug_content)
            self.assertIn("Top Warnung", debug_content)
            
            # Normal report should NOT include skip/warn sections
            normal_path = reporter.generate_full_report(
                stats, [model], date_str="2024-01-01", report_type="normal"
            )
            normal_content = Path(normal_path).read_text(encoding="utf-8")
            self.assertNotIn("Top Skip", normal_content)
            self.assertNotIn("Top Warnung", normal_content)

    def test_format_params_b_boundaries(self):
        cases = {
            0: "0M",
            -0.5: "0M",
            0.0004: "0M",
            0.268: "270M",
            0.995: "1.0B",
            0.999: "1.0B",
            1.0: "1.0B",
            1.001: "1.0B",
            1.95: "2B",
            2.01: "2B",
            2.9: "3B",
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(Reporter._format_params_b(value), expected)


    def test_params_are_rounded_in_report(self):
        stats = RunStats()
        model_large = {
            "id": "author/large_model",
            "name": "large_model",
            "namespace": "author",
            "params_total_b": 9.343,
            "params_active_b": 1.244,
            "params_source": "test_source",
            "llm_analysis": {"specialist_score": 5},
        }
        model_small = {
            "id": "author/small_model",
            "name": "small_model",
            "namespace": "author",
            "params_active_b": 0.268,
            "params_source": "alt_source",
            "llm_analysis": {"specialist_score": 4},
        }

        with TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            report_path = reporter.generate_full_report(stats, [model_large, model_small], date_str="2024-01-01")
            content = Path(report_path).read_text(encoding="utf-8")

        self.assertIn("total=9B, active=1.2B (test_source)", content)
        self.assertIn("270M (alt_source)", content)


if __name__ == "__main__":
    unittest.main()
