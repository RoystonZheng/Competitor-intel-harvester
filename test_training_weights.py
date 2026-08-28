import json
import tempfile
import unittest
from pathlib import Path

from filter_training import save_model_weights, train_filter_model


class TrainingWeightsTest(unittest.TestCase):
    def test_save_model_weights_exports_human_readable_feature_weights(self):
        rows = [
            {
                "human_label": "include",
                "source_kind": "official_core",
                "page_role": "pricing_packaging",
                "title": "Official pricing",
                "url": "https://demo.example/pricing",
                "matched_fields": "pricing",
                "reason": "official pricing page",
            },
            {
                "human_label": "include",
                "source_kind": "official_core",
                "page_role": "docs_api_or_developer",
                "title": "API docs",
                "url": "https://demo.example/docs/api",
                "matched_fields": "api_sdk_webhook",
                "reason": "official API documentation",
            },
            {
                "human_label": "exclude",
                "source_kind": "low_value_or_aggregator",
                "page_role": "auth_or_account_shell",
                "title": "Login",
                "url": "https://demo.example/login",
                "hard_gate": "rejected_auth_or_transaction_shell",
                "reason": "login screen",
            },
            {
                "human_label": "exclude",
                "source_kind": "low_value_or_aggregator",
                "page_role": "review_or_comparison",
                "title": "SEO alternatives",
                "url": "https://seo.example/alternatives",
                "reason": "generic SEO list",
            },
            {
                "human_label": "verify_later",
                "source_kind": "community_or_social_signal",
                "page_role": "video_or_social_content",
                "title": "Demo review video",
                "url": "https://www.youtube.com/watch?v=abc",
                "reason": "needs timestamp evidence",
            },
        ]

        model = train_filter_model(rows, min_labeled_rows=3)
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "filter_weights.json"
            save_model_weights(model, weights_path)
            weights = json.loads(weights_path.read_text(encoding="utf-8"))

        self.assertEqual(weights["format_version"], "filter-weights-v1")
        self.assertEqual(weights["model_version"], model.model_version)
        self.assertIn("codex", weights["compatible_hosts"])
        self.assertIn("deepseek", weights["compatible_hosts"])
        self.assertIn("include", weights["feature_weights"])
        self.assertGreater(weights["feature_weights"]["include"]["source_kind=official_core"], 0)
        self.assertLess(weights["feature_weights"]["exclude"]["source_kind=official_core"], 0)


if __name__ == "__main__":
    unittest.main()
