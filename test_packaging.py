import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class PackagingConfigTest(unittest.TestCase):
    def test_pyproject_exposes_cli_and_ui_entrypoints(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('competitor-intel-harvester = "competitor_harvester:main"', text)
        self.assertIn('competitor-intel-ui = "app:main"', text)
        self.assertIn('competitor-intel-train = "train_filter_model:main"', text)

    def test_pyproject_includes_split_modules(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for module in ["source_adapters", "structured_extractor", "filter_training", "search_cards", "analysis_templates"]:
            self.assertRegex(text, rf'"{re.escape(module)}"')

    def test_runtime_requirements_include_browser_adapter_dependency(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("crawl4ai", requirements)
        self.assertIn("icrawler", requirements)
        self.assertIn("playwright", requirements)
        self.assertIn("PyYAML", requirements)
        self.assertIn("yt-dlp", requirements)

    def test_env_example_documents_local_model_and_source_services(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("SEARXNG_URL=", text)
        self.assertIn("CODEX_COMMAND=", text)
        self.assertIn("ML_FILTER_MODEL=", text)
        self.assertIn("SEARCH_CARDS_DIR=", text)


if __name__ == "__main__":
    unittest.main()
