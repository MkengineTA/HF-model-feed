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


if __name__ == "__main__":
    unittest.main()
