import csv
import json
import tempfile
import unittest
from pathlib import Path

from filter_training import (
    apply_ml_prediction_to_decision,
    build_training_rows,
    load_filter_model,
    save_filter_model,
    train_filter_model,
)


class FilterTrainingTest(unittest.TestCase):
    def sample_rows(self):
        return [
            {
                "url": "https://gamma.app/pricing",
                "title": "Gamma pricing plans",
                "snippet": "Official pricing, plans, credits, enterprise limits",
                "source_kind": "official_core",
                "page_role": "pricing_packaging",
                "source_policy_tier": "P0 官方核心来源",
                "fact_type": "pricing_packaging",
                "increment_type": "新增价格/套餐/限制",
                "decision_status": "selected",
                "human_label": "include",
            },
            {
                "url": "https://gamma.app/docs/api",
                "title": "Gamma API docs",
                "snippet": "Official API documentation, authentication, rate limits",
                "source_kind": "official_core",
                "page_role": "docs_api_or_developer",
                "source_policy_tier": "P0 官方核心来源",
                "fact_type": "api_docs_limits",
                "increment_type": "新增 API/接口/额度/限制",
                "decision_status": "selected",
                "human_label": "include",
            },
            {
                "url": "https://example.com/login",
                "title": "Please login",
                "snippet": "Sign in account center shopping cart",
                "source_kind": "low_value_or_aggregator",
                "page_role": "auth_or_account_shell",
                "source_policy_tier": "Reject 登录/交易壳",
                "fact_type": "general_product_signal",
                "increment_type": "一般线索，需人工判断增量",
                "decision_status": "rejected",
                "human_label": "exclude",
            },
            {
                "url": "https://example.com/cart",
                "title": "Cart checkout coupon",
                "snippet": "Shopping cart seller center account checkout",
                "source_kind": "low_value_or_aggregator",
                "page_role": "transaction_or_marketplace_shell",
                "source_policy_tier": "Reject 登录/交易壳",
                "fact_type": "general_product_signal",
                "increment_type": "一般线索，需人工判断增量",
                "decision_status": "rejected",
                "human_label": "exclude",
            },
            {
                "url": "https://forum.example.com/gamma-review",
                "title": "Gamma user review",
                "snippet": "One user says quality and limits need verification",
                "source_kind": "community_or_social_signal",
                "page_role": "review_or_comparison",
                "source_policy_tier": "P3 低置信线索",
                "fact_type": "review_quality_perception",
                "increment_type": "新增用户反馈/质量/口碑",
                "decision_status": "signal",
                "human_label": "verify_later",
            },
        ]

    def test_trained_model_scores_human_preferred_sources_higher_than_noise(self):
        model = train_filter_model(self.sample_rows(), min_labeled_rows=3)

        include_prediction = model.predict(
            {
                "url": "https://gamma.app/pricing",
                "title": "Gamma pricing",
                "snippet": "Official plans pricing enterprise credits",
                "source_kind": "official_core",
                "page_role": "pricing_packaging",
                "source_policy_tier": "P0 官方核心来源",
                "fact_type": "pricing_packaging",
                "increment_type": "新增价格/套餐/限制",
            }
        )
        exclude_prediction = model.predict(
            {
                "url": "https://shop.example.com/cart",
                "title": "Cart checkout",
                "snippet": "Please login account shopping cart",
                "source_kind": "low_value_or_aggregator",
                "page_role": "transaction_or_marketplace_shell",
                "source_policy_tier": "Reject 登录/交易壳",
            }
        )

        self.assertEqual(include_prediction.label, "include")
        self.assertGreater(include_prediction.include_score, 0.60)
        self.assertEqual(exclude_prediction.label, "exclude")
        self.assertLess(exclude_prediction.include_score, 0.40)

    def test_model_can_be_saved_loaded_and_applied_to_rule_decision(self):
        model = train_filter_model(self.sample_rows(), min_labeled_rows=3)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "filter_model.json"
            save_filter_model(model, model_path)
            loaded = load_filter_model(model_path)

        prediction = loaded.predict(
            {
                "url": "https://gamma.app/docs/api",
                "title": "Gamma API documentation",
                "snippet": "Authentication SDK API rate limits",
                "source_kind": "official_core",
                "page_role": "docs_api_or_developer",
                "source_policy_tier": "P0 官方核心来源",
                "fact_type": "api_docs_limits",
            }
        )
        adjusted = apply_ml_prediction_to_decision(
            {
                "decision_status": "signal",
                "hard_gate": "needs_more_context",
                "reason": "rule filter had weak context",
            },
            prediction,
        )

        self.assertEqual(loaded.model_version, model.model_version)
        self.assertEqual(prediction.label, "include")
        self.assertEqual(adjusted["ml_label"], "include")
        self.assertEqual(adjusted["decision_status"], "accepted")
        self.assertIn("ml_include_score", adjusted)

    def test_training_rows_can_be_built_from_review_csv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_path = Path(tmp) / "human_labels.csv"
            with review_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["url", "title", "human_label", "human_reason"])
                writer.writeheader()
                writer.writerow(
                    {
                        "url": "https://gamma.app/pricing",
                        "title": "Gamma pricing",
                        "human_label": "include",
                        "human_reason": "official pricing page",
                    }
                )

            rows = build_training_rows([review_path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["human_label"], "include")
        self.assertEqual(rows[0]["human_reason"], "official pricing page")


if __name__ == "__main__":
    unittest.main()
