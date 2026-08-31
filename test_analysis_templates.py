import json
import tempfile
import unittest
from pathlib import Path

from analysis_templates import select_analysis_template
from competitor_harvester import (
    SearchResult,
    build_product_collection_plan,
    rows_from_evidence_audit,
    write_pre_crawl_plan,
)
from structured_extractor import build_extraction_schema, extract_structured_facts_from_text


class AnalysisTemplatesTest(unittest.TestCase):
    def test_autonomous_vehicle_template_is_selected_from_category_and_terms(self):
        template, score = select_analysis_template(
            "autonomous_vehicle_robotaxi",
            ["Apollo Go", "Pony.ai"],
            own_product_name="无人车 Robotaxi 产品",
            own_product_positioning="自动驾驶出租车运营与乘坐体验分析",
        )

        self.assertIsNotNone(template)
        self.assertGreaterEqual(score, 100)
        self.assertEqual(template["product_type_key"], "autonomous_vehicle_robotaxi")

    def test_collection_plan_loads_template_fields_and_exports_them(self):
        plan = build_product_collection_plan(
            ["Apollo Go"],
            own_product_name="无人车竞品分析",
            own_product_positioning="比较 Robotaxi 的运营、安全和乘坐体验",
        )

        field_keys = {field.key for field in plan.fields}
        self.assertEqual(plan.category, "autonomous_vehicle_robotaxi")
        self.assertEqual(plan.analysis_template_key, "autonomous_vehicle_robotaxi")
        self.assertIn("av_market_operations", field_keys)
        self.assertIn("av_autonomous_system", field_keys)
        self.assertIn("监管许可", plan.evidence_keywords)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_pre_crawl_plan(out_dir, ["Apollo Go"], plan, own_product_name="无人车竞品分析")
            markdown = (out_dir / "pre_crawl_plan.md").read_text(encoding="utf-8")
            payload = json.loads((out_dir / "pre_crawl_plan.json").read_text(encoding="utf-8"))

        self.assertIn("## 已加载分析模板", markdown)
        self.assertIn("市场与运营", markdown)
        self.assertIn("analysis_dimensions", payload["plan"])

    def test_autonomous_vehicle_search_result_is_a_core_decision_signal(self):
        plan = build_product_collection_plan(["Pony.ai"], own_product_name="Robotaxi 运营分析")
        rows = rows_from_evidence_audit(
            [
                SearchResult(
                    competitor="Pony.ai",
                    category="general",
                    query="Pony.ai official robotaxi service area safety report",
                    title="Pony.ai Robotaxi service area and autonomous vehicle safety",
                    url="https://www.pony.ai/robotaxi",
                    snippet="Official Robotaxi page mentions service area, fleet operations, sensor suite and safety.",
                    engine="test",
                    score=9,
                )
            ],
            [("Pony.ai", "https://www.pony.ai")],
            3,
            plan,
        )

        row = rows[0]
        self.assertEqual(row["page_role"], "autonomous_vehicle_detail")
        self.assertIn(row["decision_status"], {"selected", "accepted"})
        self.assertIn("新增无人车参数/运营/安全/体验证据", row["increment_type"])

    def test_structured_extractor_has_autonomous_vehicle_defaults(self):
        schema = build_extraction_schema("autonomous_vehicle_robotaxi")
        text = (
            "该 Robotaxi 服务范围覆盖城区核心区域，平均等待时间约 8 分钟。"
            "车辆采用激光雷达、摄像头和冗余设计，已获得自动驾驶监管许可。"
        )
        facts = extract_structured_facts_from_text(
            competitor="Demo Robotaxi",
            source_url="https://demo.example/robotaxi",
            source_title="Robotaxi service and safety",
            text=text,
            schema=schema,
        )
        keys = {fact["field_key"] for fact in facts}

        self.assertIn("av_market_operations", keys)
        self.assertIn("av_autonomous_system", keys)
        self.assertIn("av_safety_compliance", keys)


if __name__ == "__main__":
    unittest.main()
