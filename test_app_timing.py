import tempfile
import unittest
from pathlib import Path
import json

import app


class JobTimingTest(unittest.TestCase):
    def test_format_elapsed_seconds(self):
        self.assertEqual(app.format_elapsed_seconds(0), "00:00:00")
        self.assertEqual(app.format_elapsed_seconds(65), "00:01:05")
        self.assertEqual(app.format_elapsed_seconds(3661), "01:01:01")

    def test_running_job_snapshot_reports_elapsed_and_timebox(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = app.Job(
                id="20260820-120000-abcdef",
                status="running",
                command=[],
                out_dir=Path(tmp),
                created_at=100.0,
                started_at=110.0,
                experiment_minutes=15,
            )

            snapshot = app.job_snapshot(job, now=410.0)

        self.assertEqual(snapshot["elapsed_seconds"], 300)
        self.assertEqual(snapshot["elapsed_label"], "00:05:00")
        self.assertEqual(snapshot["timebox_seconds"], 900)
        self.assertEqual(snapshot["timebox_label"], "15 分钟")
        self.assertEqual(snapshot["remaining_seconds"], 600)
        self.assertEqual(snapshot["remaining_label"], "00:10:00")
        self.assertFalse(snapshot["timebox_exceeded"])

    def test_finished_job_snapshot_uses_finished_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = app.Job(
                id="20260820-120000-abcdef",
                status="done",
                command=[],
                out_dir=Path(tmp),
                created_at=100.0,
                started_at=110.0,
                finished_at=200.0,
                experiment_minutes=1,
            )

            snapshot = app.job_snapshot(job, now=999.0)

        self.assertEqual(snapshot["elapsed_seconds"], 90)
        self.assertEqual(snapshot["elapsed_label"], "00:01:30")
        self.assertEqual(snapshot["remaining_seconds"], 0)
        self.assertTrue(snapshot["timebox_exceeded"])

    def test_timing_artifacts_are_downloadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = app.Job(
                id="20260820-120000-abcdef",
                status="done",
                command=["python", "competitor_harvester.py"],
                out_dir=Path(tmp),
                created_at=100.0,
                started_at=110.0,
                finished_at=170.0,
                returncode=0,
                experiment_minutes=15,
            )

            app.write_timing_artifacts(job)
            snapshot = app.job_snapshot(job, now=200.0)

            timing_json = Path(tmp) / "实验计时记录.json"
            timing_md = Path(tmp) / "实验计时记录.md"
            record = json.loads(timing_json.read_text(encoding="utf-8"))
            artifact_names = [item["name"] for item in snapshot["artifacts"]]

            self.assertTrue(timing_json.exists())
            self.assertTrue(timing_md.exists())
            self.assertEqual(record["elapsed_label"], "00:01:00")
            self.assertIn("实验计时记录.md", artifact_names)
            self.assertIn("实验计时记录.json", artifact_names)

    def test_load_login_required_reviews_from_chinese_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Demo,demo.example,Login,https://demo.example/login,https://demo.example/login,yes,requires_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["competitor"], "Demo")
        self.assertEqual(rows[0]["login_assist_url"], "https://demo.example/login")


if __name__ == "__main__":
    unittest.main()
