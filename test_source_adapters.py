import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from competitor_harvester import execute_gui_review_queue

from source_adapters import (
    adapter_search_templates,
    classify_source_url,
    collect_adapter_snapshot,
    extract_video_evidence_markers,
)


class SourceAdaptersTest(unittest.TestCase):
    def test_directed_templates_cover_common_public_sources(self):
        templates = adapter_search_templates("Gamma", category="ai_software")

        self.assertIn("Gamma site:youtube.com demo review", templates)
        self.assertIn("Gamma site:apps.apple.com reviews screenshots", templates)
        self.assertIn("Gamma site:github.com docs API", templates)
        self.assertIn("Gamma site:producthunt.com launch", templates)

    def test_classifies_youtube_and_app_store_urls(self):
        youtube = classify_source_url("https://www.youtube.com/watch?v=abc123")
        app_store = classify_source_url("https://apps.apple.com/us/app/chatgpt/id6448311069")

        self.assertEqual(youtube["adapter_name"], "youtube")
        self.assertEqual(youtube["source_family"], "video_social")
        self.assertEqual(app_store["adapter_name"], "apple_app_store")
        self.assertEqual(app_store["source_family"], "app_store")

    def test_youtube_snapshot_saves_metadata_and_timestamp_markers(self):
        def fake_fetch(url, timeout=12, proxy_url=""):
            return (
                200,
                "application/json",
                json.dumps({"title": "Gamma demo", "author_name": "Gamma", "thumbnail_url": "https://img.example/1.jpg"}),
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = collect_adapter_snapshot(
                url="https://www.youtube.com/watch?v=abc123&t=75s",
                title="Gamma walkthrough",
                snippet="At 01:15 pricing and export flow are shown.",
                out_dir=Path(tmp),
                slug="yt",
                fetcher=fake_fetch,
            )
            markers = json.loads(Path(result["evidence_markers_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["adapter_name"], "youtube")
        self.assertEqual(result["automated_review_status"], "adapter_metadata_captured")
        self.assertEqual(result["needs_manual_video_timestamp"], "no")
        self.assertEqual(markers[0]["timestamp_seconds"], 75)
        self.assertIn("Gamma demo", result["text_snapshot_excerpt"])

    def test_app_store_snapshot_uses_public_lookup_metadata(self):
        def fake_fetch(url, timeout=12, proxy_url=""):
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "resultCount": 1,
                        "results": [
                            {
                                "trackName": "ChatGPT",
                                "sellerName": "OpenAI",
                                "version": "2.0.1",
                                "averageUserRating": 4.9,
                                "userRatingCount": 12000,
                                "description": "Official app with voice, image, and text conversations.",
                                "screenshotUrls": ["https://img.example/screen1.png"],
                            }
                        ],
                    }
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = collect_adapter_snapshot(
                url="https://apps.apple.com/us/app/chatgpt/id6448311069",
                title="ChatGPT on the App Store",
                snippet="",
                out_dir=Path(tmp),
                slug="app",
                fetcher=fake_fetch,
            )
            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["adapter_name"], "apple_app_store")
        self.assertEqual(result["automated_review_status"], "adapter_metadata_captured")
        self.assertIn("ChatGPT", result["text_snapshot_excerpt"])
        self.assertEqual(metadata["fields"]["version"], "2.0.1")

    def test_video_evidence_markers_parse_timestamps_from_context(self):
        markers = extract_video_evidence_markers("00:32 onboarding flow. 1:45 pricing limits. 01:02:03 admin setup.")

        self.assertEqual([row["timestamp_seconds"] for row in markers], [32, 105, 3723])

    def test_gui_review_queue_routes_known_public_sources_to_adapters(self):
        def fake_adapter(url, **kwargs):
            snapshot_dir = Path(kwargs["out_dir"]) / "gui_review_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = snapshot_dir / "github-adapter-metadata.json"
            text_path = snapshot_dir / "github-adapter-snapshot.txt"
            metadata_path.write_text('{"adapter_name":"github"}', encoding="utf-8")
            text_path.write_text("GitHub public repository metadata", encoding="utf-8")
            return {
                "adapter_name": "github",
                "source_family": "developer_source",
                "platform": "GitHub",
                "canonical_url": url,
                "automated_review_status": "adapter_metadata_captured",
                "metadata_path": str(metadata_path),
                "text_snapshot_path": str(text_path),
                "screenshot_path": "",
                "transcript_path": "",
                "evidence_markers_path": "",
                "needs_manual_video_timestamp": "no",
                "text_snapshot_excerpt": "GitHub public repository metadata",
                "adapter_next_step": "已保存 GitHub 公开仓库元数据。",
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch("competitor_harvester.collect_adapter_snapshot", side_effect=fake_adapter) as mocked:
                rows = execute_gui_review_queue(
                    [
                        {
                            "competitor": "Demo",
                            "priority": "P1",
                            "review_reason": "public_source_adapter_candidate",
                            "title": "Demo GitHub repo",
                            "url": "https://github.com/demo/product",
                            "domain": "github.com",
                            "gui_review_url": "https://github.com/demo/product",
                        }
                    ],
                    Path(tmp),
                    max_items=1,
                    enable_browser=False,
                )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(rows[0]["adapter_name"], "github")
        self.assertEqual(rows[0]["platform"], "GitHub")
        self.assertEqual(rows[0]["automated_review_status"], "adapter_metadata_captured")
        self.assertIn("GitHub public repository metadata", rows[0]["text_snapshot_excerpt"])


if __name__ == "__main__":
    unittest.main()
