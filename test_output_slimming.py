import csv
import tempfile
import unittest
from pathlib import Path

import app
from competitor_harvester import (
    PageExtract,
    build_problem_review_rows,
    slim_output_directory,
    write_chinese_export_aliases,
    write_problem_review_outputs,
)


class OutputSlimmingTest(unittest.TestCase):
    def test_ui_shows_single_user_facing_format_for_duplicate_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for name in [
                "实验计时记录.md",
                "实验计时记录.json",
                "抓取前采集计划.md",
                "抓取前采集计划.json",
                "竞品分析报告_图片内嵌版.md",
                "final_analysis.md",
                "Codex分析报告.md",
                "analysis.md",
                "report.md",
                "收录过滤策略设计.md",
                "本地筛选模型状态.json",
                "人工复核队列.md",
                "需登录队列.md",
                "GUI自动复核结果.md",
                "问题页面核验清单.md",
                "问题页面核验清单.csv",
                "自动竞品发现.md",
                "自动竞品发现.csv",
                "自动竞品发现.json",
                "结构化事实.csv",
                "结构化事实.json",
                "事实聚类.md",
                "事实聚类.csv",
                "事实聚类.json",
                "人工抽样标注表.csv",
            ]:
                (out_dir / name).write_text("x", encoding="utf-8")
            job = app.Job(
                id="20260828-120000-abcdef",
                status="done",
                command=[],
                out_dir=out_dir,
                created_at=100.0,
                started_at=100.0,
                finished_at=120.0,
            )

            artifact_names = [row["name"] for row in app.job_snapshot(job)["artifacts"]]

        self.assertIn("竞品分析报告_图片内嵌版.md", artifact_names)
        self.assertIn("实验计时记录.md", artifact_names)
        self.assertIn("抓取前采集计划.md", artifact_names)
        self.assertIn("问题页面核验清单.csv", artifact_names)
        self.assertIn("自动竞品发现.csv", artifact_names)
        self.assertIn("结构化事实.csv", artifact_names)
        self.assertIn("事实聚类.csv", artifact_names)
        self.assertIn("人工抽样标注表.csv", artifact_names)
        self.assertNotIn("实验计时记录.json", artifact_names)
        self.assertNotIn("抓取前采集计划.json", artifact_names)
        self.assertNotIn("问题页面核验清单.md", artifact_names)
        self.assertNotIn("自动竞品发现.md", artifact_names)
        self.assertNotIn("自动竞品发现.json", artifact_names)
        self.assertNotIn("结构化事实.json", artifact_names)
        self.assertNotIn("事实聚类.md", artifact_names)
        self.assertNotIn("事实聚类.json", artifact_names)
        self.assertNotIn("收录过滤策略设计.md", artifact_names)
        self.assertNotIn("本地筛选模型状态.json", artifact_names)
        self.assertNotIn("final_analysis.md", artifact_names)
        self.assertNotIn("Codex分析报告.md", artifact_names)
        self.assertNotIn("analysis.md", artifact_names)
        self.assertNotIn("report.md", artifact_names)
        self.assertNotIn("人工复核队列.md", artifact_names)
        self.assertNotIn("需登录队列.md", artifact_names)
        self.assertNotIn("GUI自动复核结果.md", artifact_names)

    def test_output_directory_archives_internal_files_but_keeps_deliverables(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for name in [
                "实验计时记录.md",
                "实验计时记录.json",
                "竞品分析报告_图片内嵌版.md",
                "问题页面核验清单.md",
                "问题页面核验清单.csv",
                "所有采集来源.csv",
                "自动竞品发现.md",
                "自动竞品发现.csv",
                "自动竞品发现.json",
                "结构化事实.csv",
                "结构化事实.json",
                "事实聚类.md",
                "事实聚类.csv",
                "事实聚类.json",
                "本地筛选模型状态.json",
                "final_analysis.md",
                "raw.json",
                "pages.csv",
                "manual_review_queue.csv",
                "codex_input.md",
                "run.log",
            ]:
                (out_dir / name).write_text("x", encoding="utf-8")

            slim_output_directory(out_dir, keep_run_log=False)

            self.assertTrue((out_dir / "竞品分析报告_图片内嵌版.md").exists())
            self.assertTrue((out_dir / "问题页面核验清单.csv").exists())
            self.assertTrue((out_dir / "所有采集来源.csv").exists())
            self.assertTrue((out_dir / "自动竞品发现.csv").exists())
            self.assertTrue((out_dir / "结构化事实.csv").exists())
            self.assertTrue((out_dir / "事实聚类.csv").exists())
            self.assertFalse((out_dir / "实验计时记录.json").exists())
            self.assertFalse((out_dir / "问题页面核验清单.md").exists())
            self.assertFalse((out_dir / "自动竞品发现.md").exists())
            self.assertFalse((out_dir / "自动竞品发现.json").exists())
            self.assertFalse((out_dir / "结构化事实.json").exists())
            self.assertFalse((out_dir / "事实聚类.md").exists())
            self.assertFalse((out_dir / "事实聚类.json").exists())
            self.assertFalse((out_dir / "本地筛选模型状态.json").exists())
            self.assertFalse((out_dir / "final_analysis.md").exists())
            self.assertFalse((out_dir / "raw.json").exists())
            self.assertFalse((out_dir / "pages.csv").exists())
            self.assertFalse((out_dir / "manual_review_queue.csv").exists())
            self.assertFalse((out_dir / "run.log").exists())
            self.assertTrue((out_dir / "_internal" / "实验计时记录.json").exists())
            self.assertTrue((out_dir / "_internal" / "问题页面核验清单.md").exists())
            self.assertTrue((out_dir / "_internal" / "final_analysis.md").exists())
            self.assertTrue((out_dir / "_internal" / "raw.json").exists())
            self.assertTrue((out_dir / "_internal" / "run.log").exists())

    def test_chinese_aliases_are_created_only_for_primary_deliverables(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for name in [
                "all_sources.csv",
                "final_analysis_embedded.md",
                "problem_pages_review.csv",
                "problem_pages_review.md",
                "codex_input.md",
                "screening_strategy.md",
                "raw.json",
            ]:
                (out_dir / name).write_text("x", encoding="utf-8")

            write_chinese_export_aliases(out_dir)

            self.assertTrue((out_dir / "所有采集来源.csv").exists())
            self.assertTrue((out_dir / "竞品分析报告_图片内嵌版.md").exists())
            self.assertTrue((out_dir / "问题页面核验清单.csv").exists())
            self.assertFalse((out_dir / "问题页面核验清单.md").exists())
            self.assertFalse((out_dir / "Codex分析输入证据.md").exists())
            self.assertFalse((out_dir / "收录过滤策略设计.md").exists())
            self.assertFalse((out_dir / "原始数据.json").exists())

    def test_ui_artifact_lookup_finds_archived_internal_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            internal = out_dir / "_internal"
            internal.mkdir()
            (internal / "raw.json").write_text("{}", encoding="utf-8")
            (internal / "final_analysis_embedded.md").write_text("report", encoding="utf-8")

            self.assertEqual(app.artifact_path(out_dir, "raw.json"), internal / "raw.json")
            self.assertEqual(app.artifact_path(out_dir, "竞品分析报告_图片内嵌版.md"), internal / "final_analysis_embedded.md")

    def test_problem_review_rows_merge_antibot_login_timeout_and_video_issues(self):
        page = PageExtract(
            competitor="Demo",
            url="https://demo.example/pricing",
            title="Demo Pricing",
            markdown="",
            text_excerpt="pricing plans",
            links=[],
            image_urls=[],
            fields={},
            error="Blocked by anti-bot protection: HTTP 403 with HTML content",
        )
        manual_rows = [
            {
                "competitor": "Demo",
                "priority": "P0-LOGIN",
                "review_reason": "login_required_user_action",
                "title": "Login",
                "url": "https://demo.example/account",
                "domain": "demo.example",
                "requires_user_login": "yes",
                "suggested_next_step": "请登录后继续",
            }
        ]
        gui_rows = [
            {
                "competitor": "Demo",
                "priority": "P0-LOGIN",
                "review_reason": "login_required_user_action",
                "title": "Login",
                "url": "https://demo.example/account",
                "domain": "demo.example",
                "requires_user_login": "yes",
                "automated_review_status": "login_assist_timeout",
                "next_step": "等待登录超时",
            },
            {
                "competitor": "Demo",
                "priority": "P1",
                "review_reason": "public_source_adapter_candidate",
                "title": "Demo review video",
                "url": "https://www.youtube.com/watch?v=abc",
                "domain": "youtube.com",
                "requires_user_login": "no",
                "automated_review_status": "video_metadata_pending_timestamp",
                "needs_manual_video_timestamp": "yes",
                "next_step": "补时间点和截图",
            },
        ]

        rows = build_problem_review_rows([page], manual_rows, [], gui_rows, [], [])
        problem_types = {row["problem_type"] for row in rows}

        self.assertIn("HTTP 403 / 反爬拦截", problem_types)
        self.assertIn("超时未人工登录", problem_types)
        self.assertIn("视频缺时间点证据", problem_types)
        for row in rows:
            self.assertTrue(row["what_to_verify"])
            self.assertTrue(row["suggested_human_label"])
            self.assertIn(row["problem_type"], row["reason"])

    def test_problem_review_outputs_are_written_as_single_markdown_and_csv_pair(self):
        rows = [
            {
                "competitor": "Demo",
                "priority": "P1",
                "problem_type": "HTTP 403 / 反爬拦截",
                "title": "Demo Pricing",
                "url": "https://demo.example/pricing",
                "domain": "demo.example",
                "status": "anti_bot_or_access_block",
                "reason": "HTTP 403 / 反爬拦截：官方定价页疑似被拦截",
                "what_to_verify": "确认页面是否公开可见，是否包含定价、套餐或限制。",
                "data_entry_decision": "有证据则标 include；只需补证则标 verify_later；无价值或无法追溯则标 exclude。",
                "suggested_human_label": "verify_later",
                "human_label": "",
                "human_reason": "",
                "model_feedback_status": "pending_human_review",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            write_problem_review_outputs(Path(tmp), rows)
            md_text = (Path(tmp) / "problem_pages_review.md").read_text(encoding="utf-8")
            with (Path(tmp) / "problem_pages_review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertIn("问题页面核验清单", md_text)
        self.assertIn("HTTP 403 / 反爬拦截", md_text)
        self.assertEqual(csv_rows[0]["problem_type"], "HTTP 403 / 反爬拦截")


if __name__ == "__main__":
    unittest.main()
