import unittest

from structured_extractor import (
    build_extraction_schema,
    extract_structured_facts_from_text,
    normalize_fact_value,
)


class StructuredExtractorTest(unittest.TestCase):
    def test_schema_includes_required_evidence_fields(self):
        schema = build_extraction_schema(
            "ai_software",
            [
                {
                    "key": "api_sdk_webhook",
                    "label": "API/SDK/Webhook",
                    "description": "接口能力和接入方式",
                }
            ],
        )

        self.assertIn("api_sdk_webhook", schema["fields"])
        self.assertEqual(schema["fields"]["api_sdk_webhook"]["evidence_required"], ["value", "source_sentence", "source_url"])
        self.assertIn("source_title", schema["traceability_required"])

    def test_extracts_ai_fields_with_traceable_evidence(self):
        schema = build_extraction_schema("ai_software")
        text = (
            "The Pro plan is $20 per month. "
            "Developers can use the REST API, SDK and webhooks. "
            "Rate limit: 10,000 requests/month. Security includes SSO, SOC 2 and GDPR."
        )

        facts = extract_structured_facts_from_text(
            competitor="Demo AI",
            source_url="https://demo.example/pricing",
            source_title="Demo pricing and docs",
            text=text,
            schema=schema,
        )
        values = {(row["field_key"], row["value"]) for row in facts}

        self.assertIn(("pricing", "$20 per month"), values)
        self.assertIn(("api_sdk_webhook", "REST API"), values)
        self.assertIn(("usage_quota_limits", "10,000 requests/month"), values)
        self.assertIn(("security_privacy_deployment", "SOC 2"), values)
        self.assertTrue(all(row["source_url"] == "https://demo.example/pricing" for row in facts))
        self.assertTrue(all(row["extraction_method"] == "schema_extractor_v1" for row in facts))

    def test_extracts_physical_product_specs_without_navigation_noise(self):
        schema = build_extraction_schema("snow_helmet")
        text = (
            "Shell material: ABS hardshell with EPS liner. "
            "Weight: 0.45 kg. Size chart: S, M, L. "
            "Color options: matte black, white. Certifications: ASTM F2040 and CE EN1077. "
            "Login Start for free Cookie settings."
        )

        facts = extract_structured_facts_from_text(
            competitor="Demo Helmet",
            source_url="https://demo.example/helmet",
            source_title="Demo Helmet specs",
            text=text,
            schema=schema,
        )
        values = {(row["field_key"], row["value"]) for row in facts}

        self.assertIn(("material_construction", "ABS hardshell"), values)
        self.assertIn(("weight", "0.45 kg"), values)
        self.assertIn(("color_variants", "matte black, white"), values)
        self.assertIn(("certification", "ASTM F2040"), values)
        self.assertNotIn(("navigation", "Login Start for free"), values)

    def test_normalizes_equivalent_fact_values(self):
        self.assertEqual(normalize_fact_value("0.45 kg", "weight"), "450 g")
        self.assertEqual(normalize_fact_value("$20 per month", "pricing"), "20 monthly")
        self.assertEqual(normalize_fact_value("SOC 2", "certification"), "SOC2")


if __name__ == "__main__":
    unittest.main()
