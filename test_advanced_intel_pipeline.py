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
    download_searxng_image_results,
    dedupe_login_review_rows,
    execute_gui_review_queue,
    extract_competitor_candidates_from_results,
    extract_structured_facts,
    main,
    page_quality_issue,
    page_extracts_from_gui_review_rows,
    login_required_queue_rows,
    run_codex_review,
    rows_from_manual_review_queue,
    rows_from_images,
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

    def test_searxng_image_results_can_be_downloaded_as_local_visual_evidence(self):
        def fake_image_fetch(url, timeout=20, proxy_url=""):
            return b"\x89PNG\r\n\x1a\nfake", "image/png"

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            image_result = SearchResult(
                competitor="Gamma",
                category="images",
                query="Gamma product screenshots",
                title="Gamma UI screenshot",
                url="https://images.example/gamma-ui.png",
                snippet="SearXNG image result",
                engine="bing",
            )

            downloaded = download_searxng_image_results(
                [image_result],
                out_dir,
                max_images_per_competitor=2,
                fetcher=fake_image_fetch,
            )
            rows = rows_from_images([image_result], [], downloaded)
            downloaded_file_exists = Path(downloaded[0]["file"]).exists()

        self.assertEqual(len(downloaded), 1)
        self.assertEqual(downloaded[0]["source"], "searxng_image_download")
        self.assertTrue(downloaded_file_exists)
        self.assertIn("searxng", Path(downloaded[0]["file"]).parts)
        self.assertTrue(any(row["source"] == "searxng_image_download" for row in rows))

    def test_codex_review_missing_cli_writes_fallback_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "analysis.md").write_text(
                "# 基线竞品分析报告\n\n- 结论：官方定价页已抓取。",
                encoding="utf-8",
            )
            (out_dir / "codex_input.md").write_text("evidence bundle", encoding="utf-8")

            ok = run_codex_review(
                out_dir,
                ["Demo"],
                codex_command="definitely-missing-codex-command",
                model="",
                timeout=1,
            )

            self.assertTrue(ok)
            self.assertTrue((out_dir / "codex_analysis.md").exists())
            self.assertTrue((out_dir / "codex_review.json").exists())
            self.assertIn("Codex CLI 未找到", (out_dir / "codex_analysis.md").read_text(encoding="utf-8"))
            self.assertIn("Codex command not found", (out_dir / "codex_run.log").read_text(encoding="utf-8"))

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
        original_adapter = competitor_harvester.collect_adapter_snapshot

        def fake_adapter(url, **_kwargs):
            return {
                "adapter_name": "youtube",
                "source_family": "video_social",
                "platform": "YouTube",
                "canonical_url": url,
                "automated_review_status": "video_metadata_pending_timestamp",
                "metadata_path": "",
                "text_snapshot_path": "",
                "screenshot_path": "",
                "transcript_path": "",
                "evidence_markers_path": "",
                "needs_manual_video_timestamp": "yes",
                "text_snapshot_excerpt": "",
                "adapter_next_step": "需要补充观点出现的时间点、截图或公开字幕后，才可进入正式事实证据。",
            }

        competitor_harvester.collect_adapter_snapshot = fake_adapter
        try:
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
        finally:
            competitor_harvester.collect_adapter_snapshot = original_adapter

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

    def test_display_page_with_login_nav_does_not_enter_login_queue(self):
        page = PageExtract(
            competitor="Apollo Go",
            url="https://www.bitauto.com/news/100199303395.html",
            title="Apollo Go by Baidu begins testing driverless ride services",
            markdown="",
            text_excerpt=(
                "Sign in Register Apollo Go by Baidu begins testing driverless ride services. "
                "The public article covers robotaxi launch cities and service availability."
            ),
            links=[],
            image_urls=[],
            fields={},
            error="rejected_auth_or_transaction_shell: nav contains sign in/register",
        )

        manual_rows = rows_from_manual_review_queue([page])

        self.assertEqual(manual_rows, [])

    def test_login_queue_prefers_original_login_url_over_stale_browser_url(self):
        opened_urls = []
        original_login_snapshot = competitor_harvester.login_assisted_browser_snapshot

        def fake_login_snapshot(url, *_args, **_kwargs):
            opened_urls.append(url)
            return "", "", "requires_user_login", "simulated login still needed"

        competitor_harvester.login_assisted_browser_snapshot = fake_login_snapshot
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows = execute_gui_review_queue(
                    [
                        {
                            "competitor": "Demo",
                            "priority": "P0-LOGIN",
                            "review_reason": "login_required_user_action",
                            "requires_user_login": "yes",
                            "title": "Demo account page",
                            "url": "https://www.52pojie.cn/thread-2124330-1-1.html",
                            "gui_review_url": "",
                            "login_assist_url": "https://demo.example/login",
                        }
                    ],
                    Path(tmp),
                    max_items=1,
                    enable_browser=False,
                    login_assist=True,
                )
        finally:
            competitor_harvester.login_assisted_browser_snapshot = original_login_snapshot

        self.assertEqual(opened_urls, [])
        self.assertEqual(rows[0]["url"], "https://demo.example/login")
        self.assertEqual(rows[0]["login_assist_url"], "https://demo.example/login")
        self.assertEqual(rows[0]["automated_review_status"], "requires_user_login")

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

    def test_pre_crawl_display_page_with_login_nav_not_login_queue(self):
        audit_row = {
            "competitor": "Apollo Go",
            "url": "https://www.bitauto.com/news/100199303395.html",
            "title": "Apollo Go by Baidu begins testing driverless ride services",
            "domain": "bitauto.com",
            "hard_gate": "rejected_auth_or_transaction_shell",
            "page_role": "auth_or_account_shell",
            "source_kind": "third_party_verification_source",
            "reason": "contains public launch and service availability evidence",
            "value_signals": "决策相关,信息增量",
            "matched_fields": "launch_city,service_scope",
            "pm_value_score": "3",
            "category_fit_score": "2",
            "cleaned_excerpt_sample": "Sign in Register Apollo Go public launch article.",
        }

        manual_rows = rows_from_manual_review_queue([], [audit_row])

        self.assertEqual(manual_rows, [])

    def test_pre_crawl_article_about_login_is_not_login_queue(self):
        audit_row = {
            "competitor": "Fathom",
            "url": "https://iconpolls.com/blogs/fathom-review-2026-ai-meeting-assistant-app-login-download-meeting-experience-and-faqs",
            "title": "Fathom Review 2026: AI Meeting Assistant, App, Login, Download, Meeting Experience and FAQs",
            "domain": "iconpolls.com",
            "hard_gate": "rejected_auth_or_transaction_shell",
            "page_role": "auth_or_account_shell",
            "source_kind": "third_party_verification_source",
            "reason": "blog article title mentions login as a topic",
            "cleaned_excerpt_sample": "A review article about Fathom features, app login, download, meeting experience and FAQs.",
        }

        manual_rows = rows_from_manual_review_queue([], [audit_row])

        self.assertEqual(manual_rows, [])

    def test_pre_crawl_login_queue_rejects_weak_competitor_binding(self):
        unrelated_login = {
            "competitor": "Apollo Go",
            "url": "https://apollo.io/login",
            "title": "Apollo.io login",
            "domain": "apollo.io",
            "hard_gate": "rejected_auth_or_transaction_shell",
            "page_role": "auth_or_account_shell",
            "source_kind": "official_candidate",
            "reason": "brand_match: apollo",
            "value_signals": "",
            "matched_fields": "",
            "pm_value_score": "0",
            "category_fit_score": "0",
        }
        related_login = {
            **unrelated_login,
            "url": "https://apollo-go.example.com/login",
            "title": "Apollo Go robotaxi account login",
            "domain": "apollo-go.example.com",
        }

        manual_rows = rows_from_manual_review_queue([], [unrelated_login, related_login])

        self.assertEqual(len(manual_rows), 1)
        self.assertEqual(manual_rows[0]["login_assist_url"], "https://apollo-go.example.com/login")

    def test_platform_competitor_content_enters_login_pool_for_user_click(self):
        audit_row = {
            "competitor": "Atomic S9 FIS",
            "url": "https://www.douyin.com/video/7170289680624192775",
            "title": "ATOMIC S9FIS 到货 #滑雪",
            "domain": "douyin.com",
            "query": "atomic s9 fis site:douyin.com review",
            "decision_status": "accepted",
            "page_role": "video_or_social_content",
            "source_kind": "community_or_social_signal",
            "gui_review_candidate": "yes",
            "gui_review_value_reason": "视频/社媒内容命中竞品和决策问题",
            "reason": "brand_match: atomic; page_role: video_or_social_content",
        }

        manual_rows = rows_from_manual_review_queue([], [audit_row])
        login_rows = login_required_queue_rows(manual_rows, [])

        self.assertEqual(len(login_rows), 1)
        self.assertEqual(login_rows[0]["domain"], "douyin.com")
        self.assertEqual(login_rows[0]["login_assist_url"], "https://www.douyin.com/")
        self.assertIn("https://www.douyin.com/video/7170289680624192775", login_rows[0]["queued_urls"])

    def test_platform_login_pool_rejects_cross_domain_search_drift(self):
        audit_row = {
            "competitor": "Stockli Laser SL FIS",
            "url": "https://v25.chaoxing.com",
            "title": "登录",
            "domain": "v25.chaoxing.com",
            "query": "stockli laser sl fis site:reddit.com problem",
            "hard_gate": "rejected_auth_or_transaction_shell",
            "page_role": "auth_or_account_shell",
            "source_kind": "public_web",
            "reason": "brand_match: no distinctive competitor evidence in title/url/snippet",
        }

        manual_rows = rows_from_manual_review_queue([], [audit_row])

        self.assertEqual(manual_rows, [])

    def test_login_assisted_snapshot_becomes_page_extract_for_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "login-assisted.txt"
            snapshot_path.write_text(
                "Title: Demo pricing\n"
                "Demo Pro plan costs $19 per user per month. API access and SSO are included.",
                encoding="utf-8",
            )
            rows = [
                {
                    "competitor": "Demo",
                    "url": "https://demo.example/account/pricing",
                    "title": "Demo pricing",
                    "canonical_url": "https://demo.example/account/pricing",
                    "automated_review_status": "login_assisted_snapshot_captured",
                    "text_snapshot_path": str(snapshot_path),
                    "screenshot_path": str(Path(tmp) / "login-assisted.png"),
                }
            ]

            pages = page_extracts_from_gui_review_rows(rows)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].competitor, "Demo")
        self.assertEqual(pages[0].url, "https://demo.example/account/pricing")
        self.assertIn("Demo Pro plan costs $19", pages[0].markdown)
        self.assertEqual(pages[0].error, "")

    def test_login_assist_runs_before_crawl_and_exports_snapshot_page(self):
        events = []
        originals = {
            "check_searxng": competitor_harvester.check_searxng,
            "searxng_categories": competitor_harvester.searxng_categories,
            "run_searches": competitor_harvester.run_searches,
            "crawl_with_crawl4ai": competitor_harvester.crawl_with_crawl4ai,
            "LoginAssistSession": competitor_harvester.LoginAssistSession,
        }

        def fake_check_searxng(*_args, **_kwargs):
            return True, ""

        def fake_categories(*_args, **_kwargs):
            return []

        def fake_run_searches(*_args, **_kwargs):
            return (
                [
                    SearchResult(
                        competitor="Demo",
                        category="web",
                        query="Demo login",
                        title="Sign in to Demo",
                        url="https://demo.example/login",
                        snippet="Email Password Sign in",
                        engine="test",
                        score=10,
                    ),
                    SearchResult(
                        competitor="Demo",
                        category="web",
                        query="Demo pricing",
                        title="Demo Pricing",
                        url="https://demo.example/pricing",
                        snippet="Official pricing page. Pro plan costs $29 per month.",
                        engine="test",
                        score=9,
                    ),
                ],
                [],
            )

        async def fake_crawl(urls_by_competitor, *_args, **_kwargs):
            events.append(("crawl", [url for _competitor, url in urls_by_competitor]))
            return [
                PageExtract(
                    competitor="Demo",
                    url="https://demo.example/pricing",
                    title="Demo Pricing",
                    markdown="Demo Pro plan costs $29 per month.",
                    text_excerpt="Demo Pro plan costs $29 per month.",
                    links=[],
                    image_urls=[],
                    fields={"pricing": "$29 per month"},
                )
            ]

        class FakeLoginAssistSession:
            def __init__(self, out_dir, *_args, **_kwargs):
                self.out_dir = Path(out_dir)
                self.rows = []

            def add_rows(self, rows):
                deduped = dedupe_login_review_rows(rows)
                if deduped:
                    events.append(("queue", [competitor_harvester.review_target_url(row) for row in deduped]))
                self.rows.extend(deduped)
                return len(deduped)

            def queue_rows(self):
                return [
                    {
                        "competitor": "Demo",
                        "priority": "P0-LOGIN",
                        "review_reason": "login_required_user_action",
                        "title": "Demo account page",
                        "url": "https://demo.example/login",
                        "domain": "demo.example",
                        "queued_url_count": "1",
                        "login_assist_url": "https://demo.example/login",
                        "automated_review_status": "awaiting_user_login",
                        "text_snapshot_path": "",
                        "screenshot_path": "",
                        "text_snapshot_excerpt": "",
                        "next_step": "waiting",
                        "allowed_boundary": "",
                    }
                ]

            def capture_all(self, *_args, **_kwargs):
                events.append(("capture", [competitor_harvester.review_target_url(row) for row in self.rows]))
                snapshot_dir = self.out_dir / "gui_review_snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / "login-1.txt"
                snapshot_path.write_text(
                    "Title: Demo account pricing\n"
                    "After login, Demo Team plan costs $49 per month and includes SSO.",
                    encoding="utf-8",
                )
                return [
                    {
                        "competitor": "Demo",
                        "priority": "P0-LOGIN",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "title": "Demo account pricing",
                        "url": "https://demo.example/login",
                        "domain": "demo.example",
                        "canonical_url": "https://demo.example/login",
                        "automated_review_status": "login_assisted_snapshot_captured",
                        "text_snapshot_path": str(snapshot_path),
                        "screenshot_path": "",
                        "text_snapshot_excerpt": "After login, Demo Team plan costs $49 per month and includes SSO.",
                        "login_assist_url": "https://demo.example/login",
                        "next_step": "captured",
                    }
                ]

            def close(self):
                events.append(("close", []))

        competitor_harvester.check_searxng = fake_check_searxng
        competitor_harvester.searxng_categories = fake_categories
        competitor_harvester.run_searches = fake_run_searches
        competitor_harvester.crawl_with_crawl4ai = fake_crawl
        competitor_harvester.LoginAssistSession = FakeLoginAssistSession
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = main(
                    [
                        "Demo",
                        "--searxng-url",
                        "http://localhost:8888",
                        "--out",
                        tmp,
                        "--per-query",
                        "1",
                        "--max-pages",
                        "2",
                        "--login-assist",
                        "--login-assist-wait",
                        "1",
                        "--skip-images",
                    ]
                )
                pages_csv = Path(tmp) / "页面抓取结果.csv"
                if not pages_csv.exists():
                    pages_csv = Path(tmp) / "_internal" / "pages.csv"
                with pages_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                    page_rows = list(csv.DictReader(handle))
        finally:
            for name, original in originals.items():
                setattr(competitor_harvester, name, original)

        self.assertEqual(code, 0)
        self.assertEqual(events[0][0], "queue")
        self.assertEqual(events[1][0], "crawl")
        self.assertEqual(events[2][0], "queue")
        self.assertEqual(events[3][0], "capture")
        self.assertTrue(any(row["url"] == "https://demo.example/login" for row in page_rows))

    def test_login_required_queue_dedupes_by_competitor_and_domain(self):
        rows = dedupe_login_review_rows(
            [
                {
                    "competitor": "Demo",
                    "review_reason": "login_required_user_action",
                    "requires_user_login": "yes",
                    "url": "https://demo.example/login",
                    "login_assist_url": "https://demo.example/login",
                },
                {
                    "competitor": "Demo",
                    "review_reason": "login_required_user_action",
                    "requires_user_login": "yes",
                    "url": "https://demo.example/account",
                    "login_assist_url": "https://demo.example/account",
                },
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queued_url_count"], "2")

    def test_login_required_queue_dedupes_same_keyword_root_domain(self):
        rows = dedupe_login_review_rows(
            [
                {
                    "competitor": "Demo Product",
                    "review_reason": "login_required_user_action",
                    "requires_user_login": "yes",
                    "url": "https://accounts.demo.example/login",
                    "login_assist_url": "https://accounts.demo.example/login",
                },
                {
                    "competitor": "Demo Product",
                    "review_reason": "login_required_user_action",
                    "requires_user_login": "yes",
                    "url": "https://app.demo.example/account",
                    "login_assist_url": "https://app.demo.example/account",
                },
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queued_url_count"], "2")

    def test_login_assist_session_queues_without_opening_browser_pages(self):
        events = []

        class FakePage:
            url = "about:blank"

            def is_closed(self):
                return False

            def goto(self, url, **_kwargs):
                events.append(("goto", url))

        class FakeContext:
            pages = [FakePage()]

            def new_page(self):
                events.append(("new_page", ""))
                return FakePage()

        with tempfile.TemporaryDirectory() as tmp:
            session = competitor_harvester.LoginAssistSession(Path(tmp))
            session.context = FakeContext()
            added = session.add_rows(
                [
                    {
                        "competitor": "Demo",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "url": "https://demo.example/login",
                        "login_assist_url": "https://demo.example/login",
                    },
                    {
                        "competitor": "Demo",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "url": "https://demo.example/account",
                        "login_assist_url": "https://demo.example/account",
                    },
                ]
            )
            queue = session.queue_rows()

        self.assertEqual(added, 1)
        self.assertEqual(queue[0]["queued_url_count"], "2")
        self.assertEqual(events, [])

    def test_login_assist_session_click_reuses_profile_for_same_domain_urls(self):
        events = []

        class FakeLocator:
            def inner_text(self, **_kwargs):
                return "Demo Product pricing specs release notes " * 20

        class FakePage:
            def __init__(self):
                self.url = "https://demo.example/dashboard"

            def is_closed(self):
                return False

            def goto(self, url, **_kwargs):
                events.append(("goto", url))
                self.url = url

            def title(self):
                return "Demo Product"

            def locator(self, _selector):
                return FakeLocator()

            def screenshot(self, path, **_kwargs):
                Path(path).write_bytes(b"fake image")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            session = competitor_harvester.LoginAssistSession(out_dir)
            session.add_rows(
                [
                    {
                        "competitor": "Demo Product",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "url": "https://accounts.demo.example/login",
                        "login_assist_url": "https://accounts.demo.example/login",
                    },
                    {
                        "competitor": "Demo Product",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "url": "https://app.demo.example/pricing",
                        "login_assist_url": "https://app.demo.example/pricing",
                    },
                ]
            )
            key = next(iter(session.rows_by_key))
            session.pages_by_key[key] = FakePage()
            click_path = competitor_harvester.login_click_marker_path(
                out_dir,
                "Demo Product",
                "https://accounts.demo.example/login",
            )
            click_path.parent.mkdir(parents=True)
            click_path.write_text("{}", encoding="utf-8")

            rows = session.capture_all(wait_seconds=0)

        self.assertEqual([row["automated_review_status"] for row in rows], ["login_assisted_snapshot_captured", "login_assisted_snapshot_captured"])
        self.assertEqual(
            events,
            [
                ("goto", "https://accounts.demo.example/login"),
                ("goto", "https://app.demo.example/pricing"),
            ],
        )

    def test_login_assist_session_honors_skip_marker_without_opening_browser_pages(self):
        events = []

        class FakeContext:
            pages = []

            def new_page(self):
                events.append(("new_page", ""))
                return None

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            session = competitor_harvester.LoginAssistSession(out_dir)
            session.context = FakeContext()
            session.add_rows(
                [
                    {
                        "competitor": "Demo",
                        "review_reason": "login_required_user_action",
                        "requires_user_login": "yes",
                        "url": "https://demo.example/login",
                        "login_assist_url": "https://demo.example/login",
                    }
                ]
            )
            skip_path = competitor_harvester.login_skip_marker_path(out_dir, "Demo", "https://demo.example/login")
            skip_path.parent.mkdir(parents=True)
            skip_path.write_text("{}", encoding="utf-8")
            rows = session.capture_all(wait_seconds=0)

        self.assertEqual(events, [])
        self.assertEqual(rows[0]["automated_review_status"], "login_skipped_by_user")

    def test_execute_gui_review_queue_does_not_auto_open_login_pages(self):
        original = competitor_harvester.login_assisted_browser_snapshot
        calls = []

        def fake_login_snapshot(*args, **kwargs):
            calls.append((args, kwargs))
            return "", "", "login_assisted_snapshot_captured", ""

        competitor_harvester.login_assisted_browser_snapshot = fake_login_snapshot
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows = competitor_harvester.execute_gui_review_queue(
                    [
                        {
                            "competitor": "Demo",
                            "priority": "P0-LOGIN",
                            "review_reason": "login_required_user_action",
                            "requires_user_login": "yes",
                            "url": "https://demo.example/login",
                            "login_assist_url": "https://demo.example/login",
                        }
                    ],
                    Path(tmp),
                    max_items=1,
                    login_assist=True,
                )
        finally:
            competitor_harvester.login_assisted_browser_snapshot = original

        self.assertEqual(calls, [])
        self.assertEqual(rows[0]["automated_review_status"], "requires_user_login")

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
