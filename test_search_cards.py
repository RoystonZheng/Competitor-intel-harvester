import csv
import json
import tempfile
import unittest
from pathlib import Path

from app import train_local_filter_model
from competitor_harvester import (
    SearchResult,
    apply_search_cards_to_collection_plan,
    build_product_collection_plan,
    build_search_query_templates,
    build_training_review_sample,
    rows_from_evidence_audit,
)
from search_cards import build_search_cards, load_search_cards, write_search_cards


class SearchCardsTest(unittest.TestCase):
    def labeled_rows(self):
        return [
            {
                "product_category": "snow_helmet",
                "product_type_key": "snow_helmet",
                "product_type_label": "滑雪头盔/双板全盔",
                "competitor": "Oakley MOD5",
                "url": "https://www.oakley.com/en-us/product/FOS901055",
                "domain": "oakley.com",
                "query": "Oakley MOD5 official size chart weight MIPS",
                "title": "Oakley MOD5 size chart and safety technology",
                "source_policy_tier": "P0 官方核心来源",
                "page_role": "product_specs_or_parameters",
                "fact_type": "physical_specs",
                "matched_fields": "尺码/尺寸, 重量, 安全认证",
                "matched_include_keywords": "size chart, weight, MIPS, certification",
                "human_label": "include",
                "human_reason": "官网参数页能证明尺码、重量和防护技术。",
            },
            {
                "product_category": "snow_helmet",
                "product_type_key": "snow_helmet",
                "product_type_label": "滑雪头盔/双板全盔",
                "competitor": "Giro Range Mips",
                "url": "https://www.skimag.com/gear/giro-range-mips-review/",
                "domain": "skimag.com",
                "query": "Giro Range MIPS review durability quality",
                "title": "Giro Range MIPS review durability and fit",
                "source_policy_tier": "P2 第三方验证来源",
                "page_role": "review_or_comparison",
                "fact_type": "review_quality_perception",
                "matched_fields": "质量/口碑, 使用场景",
                "matched_include_keywords": "durability, fit, review",
                "human_label": "include",
                "human_reason": "专业测评站能补充质量和佩戴反馈。",
            },
            {
                "product_category": "snow_helmet",
                "product_type_key": "snow_helmet",
                "product_type_label": "滑雪头盔/双板全盔",
                "competitor": "POC Skull Dura",
                "url": "https://www.youtube.com/watch?v=snowhelmetdemo",
                "domain": "youtube.com",
                "query": "POC Skull Dura YouTube review fit ventilation",
                "title": "POC Skull Dura review with fit and ventilation walkthrough",
                "source_policy_tier": "P3 低置信线索",
                "page_role": "video_or_social_content",
                "fact_type": "visual_product_evidence",
                "matched_fields": "通风/舒适性, 视觉证据",
                "matched_include_keywords": "fit, ventilation, review",
                "human_label": "verify_later",
                "human_reason": "视频有佩戴演示，但需要时间点和截图补证。",
            },
            {
                "product_category": "snow_helmet",
                "product_type_key": "snow_helmet",
                "product_type_label": "滑雪头盔/双板全盔",
                "competitor": "Oakley MOD5",
                "url": "https://bike.example.com/oakley-bike-helmet",
                "domain": "bike.example.com",
                "query": "Oakley bike helmet review",
                "title": "Oakley bike helmet review",
                "source_policy_tier": "Reject 同名无关",
                "page_role": "unrelated_same_name",
                "fact_type": "general_product_signal",
                "matched_exclude_keywords": "bike helmet",
                "human_label": "exclude",
                "human_reason": "自行车头盔不是本轮滑雪头盔竞品。",
            },
        ]

    def test_human_reviewed_rows_generate_reusable_search_card_for_product_type(self):
        cards = build_search_cards(self.labeled_rows(), min_labeled_rows=3)

        card = cards["snow_helmet"]

        self.assertEqual(card["product_type_label"], "滑雪头盔/双板全盔")
        self.assertEqual(card["training_rows"], 4)
        self.assertIn("size chart", card["search_terms"])
        self.assertIn("weight", card["search_terms"])
        self.assertIn("bike helmet", card["exclude_keywords"])
        self.assertIn("oakley.com", card["source_domains"])
        self.assertIn("skimag.com", card["reusable_source_domains"])
        self.assertIn("{name} site:skimag.com review", card["directed_source_search_templates"])

    def test_rows_without_product_type_do_not_create_mixed_general_card(self):
        cards = build_search_cards(
            [
                {
                    "competitor": "Gamma",
                    "url": "https://gamma.app/pricing",
                    "title": "Gamma pricing",
                    "matched_include_keywords": "pricing",
                    "human_label": "include",
                },
                {
                    "competitor": "Oakley MOD5",
                    "url": "https://oakley.com/helmet",
                    "title": "Oakley helmet specs",
                    "matched_include_keywords": "size chart",
                    "human_label": "include",
                },
            ],
            min_labeled_rows=2,
        )

        self.assertEqual(cards, {})

    def test_saved_search_card_is_loaded_into_next_collection_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards_dir = Path(tmp) / "search_cards"
            write_search_cards(build_search_cards(self.labeled_rows(), min_labeled_rows=3), cards_dir)
            loaded_cards = load_search_cards(cards_dir, product_category="snow_helmet")

        plan = build_product_collection_plan(["Atomic Revent"], own_product_name="双板全盔")
        enhanced = apply_search_cards_to_collection_plan(plan, loaded_cards)
        templates = build_search_query_templates(enhanced, include_cn=False)

        self.assertTrue(any(template == "{name} size chart" for template in templates))
        self.assertTrue(any(template == "{name} site:skimag.com review" for template in templates))
        self.assertIn("bike helmet", enhanced.dynamic_exclude_keywords)
        self.assertTrue(any("搜索卡片" in note for note in enhanced.source_policy_notes))

    def test_training_review_sample_carries_product_type_metadata_for_card_generation(self):
        plan = build_product_collection_plan(["Gamma"], own_product_name="AI 演示文稿工具")
        rows = rows_from_evidence_audit(
            [
                SearchResult(
                    competitor="Gamma",
                    category="general",
                    query="Gamma pricing",
                    title="Gamma pricing",
                    url="https://gamma.app/pricing",
                    snippet="Official pricing plans credits",
                    engine="test",
                    score=10,
                )
            ],
            [("Gamma", "https://gamma.app/pricing")],
            3,
            plan,
        )

        sample = build_training_review_sample(
            rows,
            sample_size=1,
            product_category=plan.category,
            product_type_key=plan.category,
            product_type_label=plan.category_label,
            own_product_name="AI 演示文稿工具",
        )

        self.assertEqual(sample[0]["product_category"], "ai_software")
        self.assertEqual(sample[0]["product_type_key"], "ai_software")
        self.assertEqual(sample[0]["product_type_label"], "AI/软件工具")
        self.assertEqual(sample[0]["search_card_candidate"], "yes")

    def test_ui_training_generates_search_cards_after_human_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            labels_path = tmp_dir / "review_labels.csv"
            fieldnames = sorted({key for row in self.labeled_rows() for key in row.keys()})
            with labels_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.labeled_rows())

            report = train_local_filter_model(
                {
                    "labels_path": str(labels_path),
                    "model_out": str(tmp_dir / "models" / "filter_model.json"),
                    "cards_dir": str(tmp_dir / "search_cards"),
                    "min_labeled_rows": 3,
                    "min_card_labeled_rows": 3,
                }
            )

            card_path = tmp_dir / "search_cards" / "snow_helmet.json"
            index = json.loads((tmp_dir / "search_cards" / "index.json").read_text(encoding="utf-8"))
            card_exists = card_path.exists()

        self.assertTrue(card_exists)
        self.assertEqual(report["search_cards"]["written_cards"], 1)
        self.assertIn("snow_helmet", index["cards"])


if __name__ == "__main__":
    unittest.main()
