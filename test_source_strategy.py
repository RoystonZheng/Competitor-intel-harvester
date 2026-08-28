import json
import tempfile
import unittest
from pathlib import Path

from competitor_harvester import (
    SearchResult,
    build_product_collection_plan,
    build_search_query_templates,
    rows_from_evidence_audit,
    rows_from_manual_review_queue,
    write_pre_crawl_plan,
)


class SourceStrategyTest(unittest.TestCase):
    def test_collection_plan_contains_traceable_source_and_competitor_discovery_rules(self):
        plan = build_product_collection_plan(
            [],
            own_product_name="AI 会议纪要工具",
            own_product_positioning="帮助团队自动整理会议记录并生成待办",
            own_product_context="希望先发现可替代竞品，再比较定价、集成、导出和隐私能力。",
        )

        source_names = [item.name for item in plan.source_strategies]
        discovery_names = [item.name for item in plan.competitor_discovery_strategies]

        self.assertIn("竞品官方来源", source_names)
        self.assertIn("论坛与社区", source_names)
        self.assertIn("App、社媒与视频", source_names)
        self.assertIn("海量搜索兜底", source_names)
        self.assertIn("用户输入竞品核验", discovery_names)
        self.assertIn("无竞品输入时的候选发现", discovery_names)
        self.assertTrue(all(item.traceability_rule for item in plan.source_strategies))
        self.assertTrue(all(item.traceability_rule for item in plan.competitor_discovery_strategies))

    def test_pre_crawl_plan_exports_source_and_competitor_strategy_sections(self):
        plan = build_product_collection_plan(["Fathom"], own_product_name="AI 会议纪要工具")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_pre_crawl_plan(out_dir, ["Fathom"], plan, own_product_name="AI 会议纪要工具")

            markdown = (out_dir / "pre_crawl_plan.md").read_text(encoding="utf-8")
            payload = json.loads((out_dir / "pre_crawl_plan.json").read_text(encoding="utf-8"))

        self.assertIn("## 数据源策略", markdown)
        self.assertIn("## 竞品发现策略", markdown)
        self.assertIn("App、社媒与视频", markdown)
        self.assertIn("公开视频时间点、截图或页面 URL", markdown)
        self.assertIn("source_strategies", payload["plan"])
        self.assertIn("competitor_discovery_strategies", payload["plan"])
        self.assertGreaterEqual(len(payload["plan"]["source_strategies"]), 4)
        self.assertGreaterEqual(len(payload["plan"]["competitor_discovery_strategies"]), 3)

    def test_directed_source_search_templates_route_platform_sources_through_searxng(self):
        plan = build_product_collection_plan(
            ["Fathom"],
            own_product_name="AI 会议纪要工具",
            own_product_positioning="自动整理会议记录、生成待办并同步到工作流",
        )

        templates = build_search_query_templates(plan, include_cn=True)
        joined = "\n".join(templates)

        self.assertIn("{name} site:youtube.com demo", joined)
        self.assertIn("{name} site:reddit.com review", joined)
        self.assertIn("{name} site:producthunt.com", joined)
        self.assertIn("{name} site:github.com", joined)
        self.assertIn("{name} site:zhihu.com 评价", joined)

    def test_evidence_audit_marks_valuable_video_for_gui_review_with_value_rules(self):
        plan = build_product_collection_plan(
            ["Gamma"],
            own_product_name="AI 演示文稿工具",
            own_product_positioning="帮助用户生成演示文稿并协作分享",
        )
        rows = rows_from_evidence_audit(
            [
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="Gamma site:youtube.com demo",
                    title="Gamma official demo pricing and workflow walkthrough",
                    url="https://www.youtube.com/watch?v=abc123",
                    snippet="Gamma demo shows presentation workflow, dashboard, pricing screen, and export options.",
                    engine="test",
                    score=8,
                )
            ],
            [],
            3,
            plan,
        )

        row = rows[0]
        self.assertEqual(row["page_role"], "video_or_social_content")
        self.assertEqual(row["pending_verification"], "yes")
        self.assertEqual(row["gui_review_candidate"], "yes")
        self.assertIn("竞品绑定", row["value_signals"])
        self.assertIn("决策相关", row["value_signals"])
        self.assertIn("信息增量", row["value_signals"])
        self.assertIn("视频", row["gui_review_value_reason"])

        manual_rows = rows_from_manual_review_queue([], rows)
        self.assertEqual(len(manual_rows), 1)
        self.assertEqual(manual_rows[0]["url"], "https://www.youtube.com/watch?v=abc123")

    def test_low_value_source_records_missing_value_principles(self):
        plan = build_product_collection_plan(["Gamma"], own_product_name="AI 演示文稿工具")
        rows = rows_from_evidence_audit(
            [
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="Gamma best tools",
                    title="Top 100 AI tools list",
                    url="https://example-directory.test/ai-tools",
                    snippet="A generic directory of many AI products with ads and no Gamma detail.",
                    engine="test",
                    score=1,
                )
            ],
            [],
            3,
            plan,
        )

        row = rows[0]
        self.assertEqual(row["value_verdict"], "low_value_or_noise")
        self.assertIn("决策相关", row["value_missing"])
        self.assertIn("信息增量", row["value_missing"])
        self.assertEqual(row["gui_review_candidate"], "no")


if __name__ == "__main__":
    unittest.main()
