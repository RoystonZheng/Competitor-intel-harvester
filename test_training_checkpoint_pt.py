import tempfile
import unittest
from pathlib import Path

from filter_training import load_filter_model, save_model_checkpoint_pt, train_filter_model


class TrainingCheckpointPtTest(unittest.TestCase):
    def test_training_checkpoint_pt_can_be_saved_and_loaded_by_the_program(self):
        rows = [
            {
                "human_label": "include",
                "source_kind": "official_core",
                "page_role": "pricing_packaging",
                "title": "Official pricing",
                "url": "https://demo.example/pricing",
            },
            {
                "human_label": "include",
                "source_kind": "official_core",
                "page_role": "docs_api_or_developer",
                "title": "Official API docs",
                "url": "https://demo.example/docs/api",
            },
            {
                "human_label": "exclude",
                "source_kind": "low_value_or_aggregator",
                "page_role": "auth_or_account_shell",
                "title": "Login",
                "url": "https://demo.example/login",
            },
            {
                "human_label": "verify_later",
                "source_kind": "community_or_social_signal",
                "page_role": "video_or_social_content",
                "title": "Needs timestamp",
                "url": "https://youtube.com/watch?v=abc",
            },
        ]
        model = train_filter_model(rows, min_labeled_rows=4)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "filter_model.pt"
            payload = save_model_checkpoint_pt(model, checkpoint_path)
            checkpoint_exists = checkpoint_path.exists()
            loaded = load_filter_model(checkpoint_path)

        self.assertTrue(checkpoint_exists)
        self.assertEqual(payload["format_version"], "filter-checkpoint-pt-v1")
        self.assertEqual(payload["model_state"]["model_version"], model.model_version)
        self.assertEqual(loaded.training_rows, 4)
        self.assertEqual(loaded.label_counts["include"], 2)


if __name__ == "__main__":
    unittest.main()
