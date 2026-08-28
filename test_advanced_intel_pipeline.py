import csv
import json
import tempfile
import unittest
from pathlib import Path

import competitor_harvester
from competitor_harvester import (
    PageExtract,
    SearchResult,
    build_product_collection_plan,
    cluster_structured_facts,
    dedupe_results,
    execute_gui_review_queue,
    extract_competitor_candidates_from_results,
    extract_structured_facts,
    main,
    page_quality_issue,
    login_required_queue_rows,
    rows_from_manual_review_queue,
)


class AdvancedIntelPipelineTest(unittest.TestCase):
    def test_no_competitor_input_discovers_traceable_candidates(self):
        plan = build_product_collection_plan(
            [],
            own_product_name="AI 演示文稿工具",
            own_product_positioning="帮助产品经理快速生成汇报和网页演示",
        )
        candidates = extract_competitor_candidates_from_results(
            [
                SearchResult(
                    competitor="DISCOVERY",
                    category="competitor_discovery",
                    query="AI presentation tool competitors",
                    title="Gamma - AI Presentation Maker",
                    url="https://gamma.app/",
                    snippet="Create presentations and webpages with AI.",
                    engine="test",
                ),
                SearchResult(
                    competitor="DISCOVERY",
                    category="competitor_discovery",
                    query="AI presentation tool alternatives",
                    title="Best Gamma alternatives: Beautiful.ai and Tome",
                    url="https://example.com/gamma-alternatives",
                    snippet="Beautiful.ai and Tome are often compared for AI decks.",
                    engine="test",
                ),
                SearchResult(
                    competitor="DISCOVERY",
                    category="competitor_discovery",
                    query="AI 演示文稿工具 竞品",
                    title="目前有哪些主流的AI？ - 知乎",
                    url="https://www.zhihu.com/question/591009674",
                    snippet="1、AI 写作类 ①ChatGPT：chat.openai.com 知名度最高的AI。",
                    engine="test",
                ),
                SearchResult(
                    competitor="DISCOVERY",
                    category="competitor_discovery",
                    query="AI PPT tool competitors",
                    title="秒篇AIPPT - AI一键生成PPT - 智能PPT制作网站",
                    url="https://aippt.example/",
                    snippet="支持一键生成PPT大纲、Word转PPT、自动排版美化PPT。",
                    engine="test",
                ),
            ],
            plan,
            own_product_name="AI 演示文稿工具",
            own_product_positioning="帮助产品经理快速生成汇报和网页演示",
            own_product_context="",
            max_candidates=3,
        )

        names = [candidate.name for candidate in candidates]
        gamma = next(candidate for candidate in candidates if candidate.name == "Gamma")

        self.assertIn("Gamma", names)
        self.assertIn("Beautiful.ai", names)
        self.assertIn("秒篇AIPPT", names)
        self.assertNotIn("Zhihu", names)
        self.assertEqual(gamma.official_url, "https://gamma.app/")
        self.assertEqual(gamma.confidence, "高信心")
        self.assertIn("AI presentation tool competitors", gamma.discovered_query)

    def test_main_accepts_own_product_only_and_writes_discovery_artifacts(self):
        original_check = competitor_harvester.check_searxng
        original_categories = competitor_harvester.searxng_categories
        original_search = competitor_harvester.searxng_search
        try:
            competitor_harvester.check_searxng = lambda *_args, **_kwargs: (True, "")
            competitor_harvester.searxng_categories = lambda *_args, **_kwargs: []

            def fake_search(_base_url, query, category, language, limit, timeout, proxy_url=""):
                if category != "general":
                    return []
                if "competitors" in query.lower() or "alternatives" in query.lower() or "替代" in query:
                    return [
                        {
                            "title": "Gamma - AI Presentation Maker",
                            "url": "https://gamma.app/",
                            "content": "AI presentation tool for teams.",
                            "engine": "test",
                        },
                        {
                            "title": "Beautiful.ai Presentation Software",
                            "url": "https://www.beautiful.ai/",
                            "content": "Presentation software often compared with Gamma.",
                            "engine": "test",
                        },
                    ][:limit]
                if "gamma" in query.lower():
                    return [
                        {
                            "title": "Gamma pricing",
                            "url": "https://gamma.app/pricing",
                            "content": "Official pricing plans.",
                            "engine": "test",
                        }
                    ]
                return []

            competitor_harvester.searxng_search = fake_search
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "run"
                code = main(
                    [
                        "--own-product-name",
                        "AI 演示文稿工具",
                        "--own-product-positioning",
                        "帮助产品经理快速生成汇报和网页演示",
                        "--searxng-url",
                        "http://fake-searxng.local",
                        "--out",
                        str(out_dir),
                        "--skip-crawl",
                        "--skip-images",
                        "--max-discovered-competitors",
                        "2",
                    ]
                )
                with (out_dir / "_internal" / "competitor_discovery.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                    discovery_rows = list(csv.DictReader(handle))
                raw = json.loads((out_dir / "_internal" / "raw.json").read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertGreaterEqual(len(discovery_rows), 2)
            self.assertIn("Gamma", raw["competitors"])
            self.assertTrue(raw["competitor_discovery"]["used_for_collection"])
        finally:
            competitor_harvester.check_searxng = original_check
            competitor_harvester.searxng_categories = original_categories
            competitor_harvester.searxng_search = original_search

    def test_main_crawls_explicit_competitor_url_when_search_is_unavailable(self):
        original_check = competitor_harvester.check_searxng
        original_crawl = competitor_harvester.crawl_with_crawl4ai
        try:
            competitor_harvester.check_searxng = lambda *_args, **_kwargs: (False, "search offline")

            async def fake_crawl(urls_by_competitor, *_args, **_kwargs):
                return [
                    PageExtract(
                        competitor=urls_by_competitor[0][0],
                        url=urls_by_competitor[0][1],
                        title="Gamma official",
                        markdown="Gamma official pricing starts at $20 per month.",
                        text_excerpt="Gamma official pricing starts at $20 per month.",
                        links=[],
                        image_urls=[],
                        fields={"pricing": "$20 per month"},
                    )
                ]

            competitor_harvester.crawl_with_crawl4ai = fake_crawl
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "run"
                code = main(
                    [
                        "https://gamma.app",
                        "--searxng-url",
                        "http://offline-searxng.local",
                        "--out",
                        str(out_dir),
                        "--skip-images",
                        "--skip-gui-review",
                        "--max-pages",
                        "1",
                    ]
                )
                with (out_dir / "_internal" / "pages.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                    pages = list(csv.DictReader(handle))

            self.assertEqual(code, 0)
            self.assertEqual(pages[0]["url"], "https://gamma.app")
        finally:
            competitor_harvester.check_searxng = original_check
            competitor_harvester.crawl_with_crawl4ai = original_crawl

    def test_gui_review_queue_captures_public_page_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "pricing.html"
            page.write_text(
                "<html><title>Demo pricing</title><body>Public pricing starts at $19 per month.</body></html>",
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "out"
            rows = execute_gui_review_queue(
                [
                    {
                        "competitor": "Demo",
                        "priority": "P1",
                        "review_reason": "official_core_low_text_or_possible_js_shell",
                        "title": "Demo pricing",
                        "url": page.as_uri(),
                        "domain": "",
                        "gui_review_url": page.as_uri(),
                    }
                ],
                out_dir,
                max_items=1,
                enable_browser=False,
            )

            self.assertEqual(rows[0]["automated_review_status"], "captured_public_snapshot")
            self.assertTrue(Path(rows[0]["text_snapshot_path"]).exists())
            self.assertTrue((out_dir / "gui_review_results.csv").exists())
            self.assertIn("Public pricing", Path(rows[0]["text_snapshot_path"]).read_text(encoding="utf-8"))

    def test_video_review_uses_public_metadata_but_requires_timestamp_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = execute_gui_review_queue(
                [
                    {
                        "competitor": "Gamma",
                        "priority": "P2",
                        "review_reason": "pre_crawl_value_candidate",
                        "title": "Gamma official demo",
                        "url": "https://www.youtube.com/watch?v=abc123",
                        "domain": "youtube.com",
                        "gui_review_url": "https://www.youtube.com/watch?v=abc123",
                    }
                ],
                Path(tmp),
                max_items=1,
                enable_browser=False,
            )

            self.assertEqual(rows[0]["automated_review_status"], "video_metadata_pending_timestamp")
            self.assertEqual(rows[0]["needs_manual_video_timestamp"], "yes")

    def test_login_page_enters_login_required_queue(self):
        page = PageExtract(
            competitor="Demo",
            url="https://demo.example/login",
            title="Sign in to Demo",
            markdown="",
            text_excerpt="Email Password Sign in Forgot password",
            links=[],
            image_urls=[],
            fields={},
            error="rejected_auth_or_transaction_shell:login",
        )

        manual_rows = rows_from_manual_review_queue([page])
        with tempfile.TemporaryDirectory() as tmp:
            gui_rows = execute_gui_review_queue(
                manual_rows,
                Path(tmp),
                max_items=1,
                enable_browser=False,
                login_assist=False,
            )
        login_rows = login_required_queue_rows(manual_rows, gui_rows)

        self.assertEqual(manual_rows[0]["review_reason"], "login_required_user_action")
        self.assertEqual(manual_rows[0]["requires_user_login"], "yes")
        self.assertEqual(gui_rows[0]["automated_review_status"], "requires_user_login")
        self.assertEqual(login_rows[0]["login_assist_url"], "https://demo.example/login")

    def test_pre_crawl_login_result_is_not_lost(self):
        audit_row = {
            "competitor": "Demo",
            "url": "https://demo.example/account/login",
            "title": "Login",
            "domain": "demo.example",
            "hard_gate": "rejected_auth_or_transaction_shell",
            "page_role": "auth_or_account_shell",
            "reason": "page_role: auth_or_account_shell",
        }

        manual_rows = rows_from_manual_review_queue([], [audit_row])

        self.assertEqual(len(manual_rows), 1)
        self.assertEqual(manual_rows[0]["review_reason"], "login_required_user_action")
        self.assertEqual(manual_rows[0]["requires_user_login"], "yes")

    def test_pages_export_structured_facts_for_prices_specs_and_certifications(self):
        plan = build_product_collection_plan(["Oakley MOD5"], own_product_name="双板全盔")
        page = PageExtract(
            competitor="Oakley MOD5",
            url="https://oakley.example/mod5",
            title="Oakley MOD5 specs",
            markdown=(
                "# Oakley MOD5\n"
                "Price starts at $270. Weight: 450 g. Sizes: S, M, L. "
                "The helmet uses MIPS and meets ASTM F2040 and CE EN1077 certification."
            ),
            text_excerpt="",
            links=[],
            image_urls=[],
            fields={"pricing": "Price starts at $270.", "weight": "Weight: 450 g."},
        )

        facts = extract_structured_facts([page], plan)
        signatures = {(fact["field_key"], fact["value"]) for fact in facts}

        self.assertIn(("pricing", "$270"), signatures)
        self.assertIn(("weight", "450 g"), signatures)
        self.assertIn(("certification", "ASTM F2040"), signatures)
        self.assertTrue(all(fact["source_url"] == "https://oakley.example/mod5" for fact in facts))

    def test_structured_fact_extraction_covers_physical_and_ai_fields(self):
        helmet_plan = build_product_collection_plan(["Demo Helmet"], own_product_name="双板全盔")
        helmet_page = PageExtract(
            competitor="Demo Helmet",
            url="https://demo.example/helmet",
            title="Demo Helmet official specs",
            markdown=(
                "Shell material: ABS hardshell with EPS liner. "
                "Dimensions: 55-59 cm. Color options: matte black, white."
            ),
            text_excerpt="",
            links=[],
            image_urls=[],
            fields={},
        )
        ai_plan = build_product_collection_plan(["Demo AI"], own_product_name="AI agent platform")
        ai_page = PageExtract(
            competitor="Demo AI",
            url="https://demo.example/docs",
            title="Demo AI docs",
            markdown=(
                "Developers can use the REST API and Webhook endpoints. "
                "Quota: 10,000 tokens per month. Security: SSO, SOC 2, GDPR."
            ),
            text_excerpt="",
            links=[],
            image_urls=[],
            fields={},
        )

        physical_signatures = {
            (fact["field_key"], fact["value"]) for fact in extract_structured_facts([helmet_page], helmet_plan)
        }
        ai_signatures = {
            (fact["field_key"], fact["value"]) for fact in extract_structured_facts([ai_page], ai_plan)
        }

        self.assertIn(("material_construction", "ABS hardshell"), physical_signatures)
        self.assertIn(("dimensions", "55-59 cm"), physical_signatures)
        self.assertIn(("color_variants", "matte black, white"), physical_signatures)
        self.assertIn(("api_sdk_webhook", "REST API"), ai_signatures)
        self.assertIn(("usage_quota_limits", "10,000 tokens per month"), ai_signatures)
        self.assertIn(("certification", "SOC2"), ai_signatures)

    def test_structured_facts_cluster_same_fact_with_primary_evidence(self):
        facts = [
            {
                "competitor": "Gamma",
                "dimension": "pricing",
                "field_key": "pricing",
                "value": "$20/mo",
                "source_url": "https://gamma.app/pricing",
                "source_title": "Gamma pricing",
                "source_policy_tier": "P0 官方核心来源",
                "confidence": "高信心",
            },
            {
                "competitor": "Gamma",
                "dimension": "pricing",
                "field_key": "pricing",
                "value": "$20 per month",
                "source_url": "https://blog.example.com/gamma-price",
                "source_title": "Gamma price overview",
                "source_policy_tier": "P2 第三方验证来源",
                "confidence": "中信心",
            },
            {
                "competitor": "Gamma",
                "dimension": "pricing",
                "field_key": "pricing",
                "value": "$99/mo",
                "source_url": "https://gamma.app/enterprise",
                "source_title": "Gamma enterprise",
                "source_policy_tier": "P0 官方核心来源",
                "confidence": "高信心",
            },
        ]

        clusters = cluster_structured_facts(facts)
        twenty = next(cluster for cluster in clusters if cluster["normalized_value"] == "20 monthly")

        self.assertEqual(twenty["source_count"], 2)
        self.assertEqual(twenty["primary_source_url"], "https://gamma.app/pricing")
        self.assertIn("https://blog.example.com/gamma-price", twenty["supporting_source_urls"])

    def test_search_dedupe_merges_tracking_url_variants(self):
        rows = dedupe_results(
            [
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="gamma pricing",
                    title="Gamma pricing",
                    url="https://gamma.app/pricing?utm_source=newsletter#plans",
                    snippet="Official pricing.",
                    engine="a",
                ),
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="gamma plans",
                    title="Gamma plans",
                    url="https://gamma.app/pricing/",
                    snippet="Plans and limits.",
                    engine="b",
                ),
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].url, "https://gamma.app/pricing")

    def test_fact_cluster_normalizes_equivalent_weight_units(self):
        clusters = cluster_structured_facts(
            [
                {
                    "competitor": "Demo Helmet",
                    "dimension": "product_specs",
                    "field_key": "weight",
                    "field_label": "重量",
                    "value": "450 g",
                    "source_url": "https://official.example/helmet",
                    "source_policy_tier": "P0 官方核心来源",
                    "confidence": "高信心",
                },
                {
                    "competitor": "Demo Helmet",
                    "dimension": "product_specs",
                    "field_key": "weight",
                    "field_label": "重量",
                    "value": "0.45 kg",
                    "source_url": "https://review.example/helmet",
                    "source_policy_tier": "P2 第三方验证来源",
                    "confidence": "中信心",
                },
            ]
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["normalized_value"], "450 g")
        self.assertEqual(clusters[0]["source_count"], 2)

    def test_product_homepage_with_login_link_is_not_auth_shell(self):
        plan = build_product_collection_plan(["Gamma"], own_product_name="AI presentation maker")
        issue = page_quality_issue(
            competitor="Gamma",
            url="https://gamma.app",
            title="Effortless AI design for presentations, websites, and more",
            markdown=(
                "Login Start for free. Products: Presentations, Websites, Documents, API, "
                "Graphics and Integrations. Turn any idea into a polished slide deck. "
                "Export to PPT, PDF, and more. Pricing plans are available for teams."
            ),
            collection_plan=plan,
        )

        self.assertEqual(issue, "")


if __name__ == "__main__":
    unittest.main()
