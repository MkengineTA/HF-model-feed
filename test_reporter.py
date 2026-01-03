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

        self.assertIn("**Was ist neu?**\n\n- Changed dataset\n- New objective", content)
        self.assertIn("**Warum relevant?**\n\n- Better accuracy\n- More stable", content)
        self.assertIn("**Use cases**\n\n- Text completion\n- QA", content)
        self.assertNotIn("Confidence:", content)
        self.assertNotIn("Unknowns:", content)


if __name__ == "__main__":
    unittest.main()
