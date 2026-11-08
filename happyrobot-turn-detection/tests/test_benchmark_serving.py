from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_serving import default_report_path, write_report


class BenchmarkReportTests(unittest.TestCase):
    def test_default_report_path_is_separate_from_original_benchmark(self) -> None:
        path = default_report_path()
        self.assertEqual(path.parent.name, "benchmark_runs")
        self.assertNotEqual(path, path.parents[1] / "performance.json")

    def test_write_report_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "report.json"
            write_report(path, {"methodology": {"server_worker_count": 8}})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"methodology": {"server_worker_count": 8}},
            )

    def test_write_report_refuses_to_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            path.write_text('{"original": true}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_report(path, {"replacement": True})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"original": true}\n')


if __name__ == "__main__":
    unittest.main()
