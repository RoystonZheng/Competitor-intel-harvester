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

    def test_record_login_open_request_dedupes_by_competitor_and_domain(self):
        original_runs_dir = app.RUNS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)

            try:
                first = app.record_login_open_request(job_id, "Demo", "https://demo.example/login")
                second = app.record_login_open_request(job_id, "Demo", "https://demo.example/account")
                self.assertEqual(first["marker_path"], second["marker_path"])
                self.assertTrue(Path(first["marker_path"]).exists())
            finally:
                app.RUNS_DIR = original_runs_dir

    def test_login_skip_request_hides_row_from_login_pool(self):
        original_runs_dir = app.RUNS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)
            queue = job_dir / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Demo,demo.example,Login,https://demo.example/login,https://demo.example/login,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            try:
                before = app.load_login_required_reviews(job_dir)
                app.record_login_skip_request(job_id, "Demo", "https://demo.example/login")
                after = app.load_login_required_reviews(job_dir)
            finally:
                app.RUNS_DIR = original_runs_dir

        self.assertEqual(len(before), 1)
        self.assertEqual(after, [])

    def test_login_pool_hides_weak_multi_word_competitor_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Apollo Go,apollo.io,Apollo.io login,https://apollo.io/login,https://apollo.io/login,yes,awaiting_user_login,请登录后继续\n"
                "Apollo Go,bitauto.com,Apollo Go by Baidu begins testing services,https://www.bitauto.com/news/1,https://www.bitauto.com/news/1,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "bitauto.com")

    def test_login_pool_hides_recruiting_account_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "WeRide,app.mokahr.com,文远知行WeRide - 校园招聘,https://app.mokahr.com/campus-recruitment/weride,https://app.mokahr.com/campus-recruitment/weride,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(rows, [])

    def test_train_local_filter_model_includes_current_job_training_sample(self):
        original_runs_dir = app.RUNS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)
            labels_path = root / "review_labels.csv"
            labels_path.write_text(
                "competitor,title,url,snippet,human_label,human_reason\n",
                encoding="utf-8",
            )
            sample_path = job_dir / "人工抽样标注表.csv"
            sample_path.write_text(
                "competitor,title,url,snippet,source_kind,page_role,human_label,human_reason\n"
                "Demo,Official Pricing,https://demo.example/pricing,pricing tiers api,official,pricing,include,official pricing page\n"
                "Demo,Login Shell,https://demo.example/login,email password sign in,account,auth_or_account_shell,exclude,login only\n"
                "Demo,Forum Rumor,https://forum.example/demo,unverified roadmap rumor,community,discussion,verify_later,needs source check\n",
                encoding="utf-8-sig",
            )

            try:
                report = app.train_local_filter_model(
                    {
                        "labels_path": str(labels_path),
                        "model_out": str(root / "models" / "filter_model.pt"),
                        "cards_dir": str(root / "search_cards"),
                        "min_labeled_rows": 3,
                        "job_id": job_id,
                        "include_problem_reviews": True,
                    }
                )
            finally:
                app.RUNS_DIR = original_runs_dir

            self.assertEqual(report["training_rows"], 3)
            self.assertIn(str(sample_path.resolve()), report["label_paths"])
            self.assertTrue((root / "models" / "filter_model.pt").exists())

    def test_model_status_for_ui_bootstraps_default_model_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_path = root / "bootstrap_labels.csv"
            labels_path.write_text(
                "url,title,snippet,source_kind,page_role,human_label\n"
                "https://demo.example/pricing,Demo pricing,Official pricing plans,official_core,pricing_packaging,include\n"
                "https://demo.example/login,Demo login,Sign in account,low_value_or_aggregator,auth_or_account_shell,exclude\n"
                "https://forum.example/demo,Demo forum,Unverified user rumor,community_or_social_signal,forum_or_community_discussion,verify_later\n",
                encoding="utf-8-sig",
            )
            model_path = root / "models" / "filter_model.pt"

            status = app.model_status_for_ui(
                model_path,
                bootstrap_label_paths=[labels_path],
                min_labeled_rows=3,
            )

            self.assertTrue(status["enabled"])
            self.assertTrue(status["bootstrap_created"])
            self.assertEqual(status["training_rows"], 3)
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
