import unittest

from competitor_harvester import (
    SearchResult,
    build_training_review_sample,
    build_product_collection_plan,
    rows_from_evidence_audit,
)
from filter_training import train_filter_model


class HarvesterMlIntegrationTest(unittest.TestCase):
    def trained_model(self):
        return train_filter_model(
            [
                {
                    "url": "https://gamma.app/pricing",
                    "title": "Gamma pricing",
                    "snippet": "Official pricing plans credits",
                    "source_kind": "official_core",
                    "page_role": "pricing_packaging",
                    "source_policy_tier": "P0 官方核心来源",
                    "fact_type": "pricing_packaging",
                    "decision_status": "selected",
                    "human_label": "include",
                },
                {
                    "url": "https://gamma.app/docs/api",
                    "title": "Gamma API docs",
                    "snippet": "Official docs SDK API limits",
                    "source_kind": "official_core",
                    "page_role": "docs_api_or_developer",
                    "source_policy_tier": "P0 官方核心来源",
                    "fact_type": "api_docs_limits",
                    "decision_status": "selected",
                    "human_label": "include",
                },
                {
                    "url": "https://shop.example.com/login",
                    "title": "Please login",
                    "snippet": "Account center sign in",
                    "source_kind": "low_value_or_aggregator",
                    "page_role": "auth_or_account_shell",
                    "source_policy_tier": "Reject 登录/交易壳",
                    "decision_status": "rejected",
                    "human_label": "exclude",
                },
                {
                    "url": "https://shop.example.com/cart",
                    "title": "Shopping cart",
                    "snippet": "Checkout seller center coupon",
                    "source_kind": "low_value_or_aggregator",
                    "page_role": "transaction_or_marketplace_shell",
                    "source_policy_tier": "Reject 登录/交易壳",
                    "decision_status": "rejected",
                    "human_label": "exclude",
                },
            ],
            min_labeled_rows=4,
        )

    def test_evidence_audit_exports_ml_scores_without_overriding_rule_hard_rejects(self):
        plan = build_product_collection_plan(
            ["Gamma"],
            own_product_name="竞品情报工具",
            own_product_positioning="面向 PM 的竞品分析工具",
        )
        model = self.trained_model()
        rows = rows_from_evidence_audit(
            [
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="Gamma pricing",
                    title="Gamma pricing plans",
                    url="https://gamma.app/pricing",
                    snippet="Official pricing plans credits",
                    engine="test",
                    score=10,
                ),
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="Gamma login",
                    title="Gamma login",
                    url="https://gamma.app/login",
                    snippet="Sign in account center",
                    engine="test",
                    score=9,
                ),
            ],
            [("Gamma", "https://gamma.app/pricing")],
            3,
            plan,
            ml_model=model,
        )

        pricing = next(row for row in rows if row["url"] == "https://gamma.app/pricing")
        login = next(row for row in rows if row["url"] == "https://gamma.app/login")

        self.assertEqual(pricing["ml_label"], "include")
        self.assertGreater(float(pricing["ml_include_score"]), 0.60)
        self.assertEqual(login["hard_gate"], "rejected_auth_or_transaction_shell")
        self.assertEqual(login["ml_adjustment"], "annotated_only_rule_protected")

    def test_training_review_sample_prefers_uncertain_and_model_boundary_rows(self):
        rows = build_training_review_sample(
            [
                {
                    "competitor": "Gamma",
                    "url": "https://gamma.app/pricing",
                    "title": "Gamma pricing",
                    "decision_status": "selected",
                    "pending_verification": "no",
                    "ml_confidence": "high",
                    "ml_include_score": "0.95",
                    "ml_exclude_score": "0.02",
                    "reason": "official pricing",
                },
                {
                    "competitor": "Gamma",
                    "url": "https://forum.example.com/gamma",
                    "title": "Gamma review",
                    "decision_status": "signal",
                    "pending_verification": "yes",
                    "ml_confidence": "low",
                    "ml_include_score": "0.45",
                    "ml_exclude_score": "0.30",
                    "reason": "needs verification",
                },
            ],
            sample_size=1,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], "https://forum.example.com/gamma")
        self.assertIn("human_label", rows[0])
        self.assertIn("human_reason", rows[0])


if __name__ == "__main__":
    unittest.main()
