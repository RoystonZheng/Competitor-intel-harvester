import csv
import threading
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
            self.assertNotIn("实验计时记录.json", artifact_names)

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

    def test_load_login_required_reviews_dedupes_same_keyword_root_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Demo Product,accounts.demo.example,Demo Product login,https://accounts.demo.example/login,https://accounts.demo.example/login,yes,awaiting_user_login,请登录后继续\n"
                "Demo Product,app.demo.example,Demo Product account,https://app.demo.example/account/login,https://app.demo.example/account/login,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "demo.example")
        self.assertEqual(rows[0]["queued_url_count"], "2")

    def test_load_login_required_reviews_allows_platform_login_assist(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,queued_urls,requires_user_login,review_reason,automated_review_status,next_step\n"
                "Atomic S9 FIS,douyin.com,ATOMIC S9FIS 到货 #滑雪,https://www.douyin.com/,https://www.douyin.com/,https://www.douyin.com/video/7170289680624192775,yes,platform_login_assist_user_action,requires_user_login,请点击登录池后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "douyin.com")
        self.assertIn("7170289680624192775", rows[0]["queued_urls"])

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

    def test_record_login_open_request_dedupes_subdomains_for_same_keyword(self):
        original_runs_dir = app.RUNS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)

            try:
                first = app.record_login_open_request(job_id, "Demo Product", "https://accounts.demo.example/login")
                second = app.record_login_open_request(job_id, "Demo Product", "https://app.demo.example/account")
                self.assertEqual(first["marker_path"], second["marker_path"])
                self.assertEqual(first["domain"], "demo.example")
            finally:
                app.RUNS_DIR = original_runs_dir

    def test_record_login_open_request_starts_post_run_capture_for_disk_job(self):
        original_runs_dir = app.RUNS_DIR
        original_runner = app.POST_RUN_LOGIN_CAPTURE_RUNNER
        finished = threading.Event()
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)
            (job_dir / "竞品分析报告_图片内嵌版.md").write_text("# done\n", encoding="utf-8")
            queue = job_dir / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,queued_urls,requires_user_login,review_reason,automated_review_status,next_step\n"
                "Atomic S9 FIS,douyin.com,ATOMIC S9FIS 到货 #滑雪,https://www.douyin.com/,https://www.douyin.com/,https://www.douyin.com/video/7170289680624192775,yes,platform_login_assist_user_action,requires_user_login,请点击登录池后继续\n",
                encoding="utf-8-sig",
            )

            def fake_runner(*args):
                calls.append(args)
                finished.set()

            app.POST_RUN_LOGIN_CAPTURE_RUNNER = fake_runner
            try:
                result = app.record_login_open_request(job_id, "Atomic S9 FIS", "https://www.douyin.com/")
                self.assertTrue(finished.wait(1.0))
            finally:
                app.RUNS_DIR = original_runs_dir
                app.POST_RUN_LOGIN_CAPTURE_RUNNER = original_runner
                app.POST_RUN_LOGIN_CAPTURE_THREADS.clear()

        self.assertFalse(result["active_job_consumer"])
        self.assertTrue(result["post_run_capture"]["started"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0][3]), 1)

    def test_login_pool_module_stays_visible_for_active_jobs(self):
        self.assertIn("登录等待池", app.INDEX_HTML)
        self.assertIn("const showPool = Boolean(job && job.id)", app.INDEX_HTML)

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
                "Apollo Go,apollo-go.example.com,Apollo Go account login,https://apollo-go.example.com/login,https://apollo-go.example.com/login,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "apollo-go.example.com")

    def test_login_pool_hides_public_display_pages_with_login_nav(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Apollo Go,bitauto.com,Apollo Go by Baidu begins testing services,https://www.bitauto.com/news/1,https://www.bitauto.com/news/1,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(rows, [])

    def test_login_pool_hides_articles_and_author_pages_about_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "需登录队列.csv"
            queue.write_text(
                "competitor,domain,title,url,login_assist_url,requires_user_login,automated_review_status,next_step\n"
                "Fathom,iconpolls.com,Fathom Review 2026: AI Meeting Assistant App Login Download,https://iconpolls.com/blogs/fathom-review-app-login-download,https://iconpolls.com/blogs/fathom-review-app-login-download,yes,awaiting_user_login,请登录后继续\n"
                "Fathom,lcfathom.com,L.C. Fathom | Author,https://www.lcfathom.com,https://www.lcfathom.com,yes,awaiting_user_login,请登录后继续\n"
                "Fathom,usefathom.com,Paul Jarvis author - Fathom Analytics,https://usefathom.com/author/paul-jarvis,https://usefathom.com/author/paul-jarvis,yes,awaiting_user_login,请登录后继续\n",
                encoding="utf-8-sig",
            )

            rows = app.load_login_required_reviews(Path(tmp))

        self.assertEqual(rows, [])

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

    def test_train_local_filter_model_uses_latest_job_sample_when_page_was_refreshed(self):
        original_runs_dir = app.RUNS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            old_job_dir = app.RUNS_DIR / "20260830-120000-111111"
            latest_job_dir = app.RUNS_DIR / "20260831-120000-abcdef"
            old_job_dir.mkdir(parents=True)
            latest_job_dir.mkdir(parents=True)
            labels_path = root / "review_labels.csv"
            labels_path.write_text(
                "competitor,title,url,snippet,human_label,human_reason\n",
                encoding="utf-8",
            )
            (old_job_dir / "人工抽样标注表.csv").write_text(
                "competitor,title,url,snippet,source_kind,page_role,human_label,human_reason\n"
                "Old,Old page,https://old.example,old content,official,overview,include,old run\n",
                encoding="utf-8-sig",
            )
            sample_path = latest_job_dir / "人工抽样标注表.csv"
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
                        "include_problem_reviews": True,
                    }
                )
            finally:
                app.RUNS_DIR = original_runs_dir

            self.assertEqual(report["training_rows"], 3)
            self.assertIn(str(sample_path.resolve()), report["label_paths"])
            self.assertNotIn(str((old_job_dir / "人工抽样标注表.csv").resolve()), report["label_paths"])

    def test_problem_review_training_button_reenables_after_request(self):
        self.assertIn("trainModelFromUi(includeCurrentJob, triggerButton)", app.INDEX_HTML)
        self.assertIn("trainModelFromUi(true, trainBtn)", app.INDEX_HTML)
        self.assertIn("currentJobId = job.id || currentJobId;", app.INDEX_HTML)
        self.assertIn("finally", app.INDEX_HTML)
        self.assertIn("triggerButton.disabled = false", app.INDEX_HTML)

    def test_feedback_review_samples_are_stratified_for_click_judgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "问题页面核验清单.csv").write_text(
                "competitor,priority,problem_type,source_queue,title,url,domain,status,source_kind,page_role,source_policy_tier,pending_verification,verification_reason,fact_type,increment_type,fact_group,reason,what_to_verify,data_entry_decision,suggested_human_label,human_label,human_reason\n"
                "Demo,P1,HTTP 403 / 反爬拦截,crawl4ai_page,Demo blocked,https://demo.example/blocked,demo.example,403,official_core,pricing_packaging,P0 官方核心来源,yes,blocked,pricing,pricing,,blocked,核验403,补证,verify_later,,\n"
                "Demo,P0-LOGIN,需登录/注册/账号权限,login_required_queue,Demo login,https://demo.example/login,demo.example,requires_user_login,official_core,auth_or_account_shell,Reject 登录/交易壳,yes,login,account,,,login,核验登录,登录后补证,verify_later,,\n",
                encoding="utf-8-sig",
            )
            (out_dir / "人工抽样标注表.csv").write_text(
                "competitor,title,url,domain,snippet,source_kind,page_role,source_policy_tier,decision_status,hard_gate,pending_verification,verification_reason,ml_confidence,ml_include_score,ml_exclude_score,suggested_label,human_label,human_reason\n"
                "Demo,Demo pricing,https://demo.example/pricing,demo.example,Official pricing tiers,official_core,pricing_packaging,P0 官方核心来源,selected,,no,,high,0.91,0.03,include,,\n"
                "Demo,Demo junk,https://spam.example/demo,spam.example,Download cracked app,low_value_or_aggregator,low_value_or_aggregator,Reject 低价值聚合,rejected,rejected_low_value,no,,high,0.02,0.95,exclude,,\n"
                "Demo,Demo forum,https://forum.example/demo,forum.example,Unverified rumor,community_or_social_signal,forum_or_community_discussion,P2 社区线索,signal,,yes,needs source,low,0.40,0.30,verify_later,,\n",
                encoding="utf-8-sig",
            )

            rows = app.build_feedback_review_samples(out_dir, per_category=1)

        categories = {row["feedback_category"] for row in rows}
        self.assertIn("失败/反爬/超时", categories)
        self.assertIn("待登录", categories)
        self.assertIn("内容不好放弃", categories)
        self.assertIn("主要内容来源", categories)
        self.assertIn("待核实线索", categories)
        for row in rows:
            self.assertTrue(row["model_conclusion"])
            self.assertIn(row["conclusion_label"], {"include", "exclude", "verify_later"})
            self.assertIn(row["counter_human_label"], {"include", "exclude", "verify_later"})
            self.assertEqual(row["feedback_status"], "pending")

    def test_feedback_review_click_writes_training_label(self):
        original_runs_dir = app.RUNS_DIR
        original_review_labels = app.DEFAULT_REVIEW_LABELS_PATH
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.RUNS_DIR = root / "runs"
            app.DEFAULT_REVIEW_LABELS_PATH = root / "training_data" / "review_labels.csv"
            job_id = "20260831-120000-abcdef"
            job_dir = app.RUNS_DIR / job_id
            job_dir.mkdir(parents=True)
            (job_dir / "人工抽样标注表.csv").write_text(
                "competitor,title,url,domain,snippet,source_kind,page_role,source_policy_tier,decision_status,hard_gate,pending_verification,verification_reason,ml_confidence,ml_include_score,ml_exclude_score,suggested_label,human_label,human_reason\n"
                "Demo,Demo pricing,https://demo.example/pricing,demo.example,Official pricing tiers,official_core,pricing_packaging,P0 官方核心来源,selected,,no,,high,0.91,0.03,include,,\n"
                "Demo,Demo junk,https://spam.example/demo,spam.example,Download cracked app,low_value_or_aggregator,low_value_or_aggregator,Reject 低价值聚合,rejected,rejected_low_value,no,,high,0.02,0.95,exclude,,\n",
                encoding="utf-8-sig",
            )

            try:
                qualified = app.record_feedback_review(
                    {
                        "job": job_id,
                        "competitor": "Demo",
                        "url": "https://demo.example/pricing",
                        "feedback_category": "主要内容来源",
                        "judgement": "qualified",
                    }
                )
                unqualified = app.record_feedback_review(
                    {
                        "job": job_id,
                        "competitor": "Demo",
                        "url": "https://spam.example/demo",
                        "feedback_category": "内容不好放弃",
                        "judgement": "unqualified",
                    }
                )
                with app.DEFAULT_REVIEW_LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
                    label_rows = list(csv.DictReader(handle))
                with (job_dir / "人工反馈标注.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                    job_rows = list(csv.DictReader(handle))
                empty_labels = root / "empty_labels.csv"
                empty_labels.write_text(
                    "competitor,title,url,snippet,human_label,human_reason\n",
                    encoding="utf-8",
                )
                train_report = app.train_local_filter_model(
                    {
                        "labels_path": str(empty_labels),
                        "model_out": str(root / "models" / "filter_model.pt"),
                        "cards_dir": str(root / "search_cards"),
                        "min_labeled_rows": 2,
                        "job_id": job_id,
                        "include_problem_reviews": True,
                    }
                )
            finally:
                app.RUNS_DIR = original_runs_dir
                app.DEFAULT_REVIEW_LABELS_PATH = original_review_labels

        self.assertEqual(qualified["human_label"], "include")
        self.assertEqual(unqualified["human_label"], "verify_later")
        self.assertEqual(len(label_rows), 2)
        self.assertEqual(label_rows[0]["feedback_judgement"], "qualified")
        self.assertEqual(label_rows[1]["feedback_judgement"], "unqualified")
        self.assertEqual(len(job_rows), 2)
        self.assertEqual(train_report["training_rows"], 2)
        self.assertIn(str((job_dir / "人工反馈标注.csv").resolve()), train_report["label_paths"])

    def test_feedback_review_ui_uses_click_judgement_buttons(self):
        self.assertIn("renderFeedbackReviews(job)", app.INDEX_HTML)
        self.assertIn("评判合格", app.INDEX_HTML)
        self.assertIn("评判不合格", app.INDEX_HTML)
        self.assertIn("/api/review/feedback", app.INDEX_HTML)
        self.assertIn("feedback_reviews", app.INDEX_HTML)
        self.assertIn("peerButton.disabled = false", app.INDEX_HTML)

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
