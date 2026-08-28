#!/usr/bin/env python3
"""Train the local competitor evidence filter model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from filter_training import build_training_rows, save_filter_model, save_model_checkpoint_pt, save_model_weights, train_filter_model
from search_cards import build_search_cards, write_search_cards


APP_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS_PATH = APP_DIR / "training_data" / "review_labels.csv"
DEFAULT_MODEL_PATH = APP_DIR / "models" / "filter_model.pt"
DEFAULT_SEARCH_CARDS_DIR = APP_DIR / "search_cards"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train local filter model from human review labels.")
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        help="CSV file with human_label. Can repeat. Defaults to training_data/review_labels.csv.",
    )
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH), help="Output model path, default models/filter_model.pt.")
    parser.add_argument("--weights-out", default="", help="Output readable feature weights JSON path. Defaults to filter_weights.json next to the model.")
    parser.add_argument("--min-labeled-rows", type=int, default=10, help="Minimum labeled rows required.")
    parser.add_argument("--report-out", default="", help="Optional training report JSON path.")
    parser.add_argument("--cards-dir", default=str(DEFAULT_SEARCH_CARDS_DIR), help="Output directory for learned product search cards.")
    parser.add_argument("--min-card-labeled-rows", type=int, default=3, help="Minimum labeled rows required for one search card.")
    parser.add_argument("--skip-search-cards", action="store_true", help="Only train the filter model; do not write search cards.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    label_paths = [Path(item).expanduser() for item in args.labels] or [DEFAULT_LABELS_PATH]
    rows = build_training_rows(label_paths)
    model = train_filter_model(rows, min_labeled_rows=args.min_labeled_rows)
    model_path = Path(args.model_out).expanduser()
    if model_path.suffix.lower() == ".pt":
        checkpoint = save_model_checkpoint_pt(model, model_path)
    else:
        save_filter_model(model, model_path)
        checkpoint = save_model_checkpoint_pt(model, model_path.with_suffix(".pt"))
    weights_path = Path(args.weights_out).expanduser() if args.weights_out else model_path.with_name("filter_weights.json")
    weights = save_model_weights(model, weights_path)
    checkpoint_alias_path = model_path.with_name("本地筛选模型.pt")
    save_model_checkpoint_pt(model, checkpoint_alias_path)
    weights_alias_path = weights_path.with_name("本地筛选模型权重.json")
    weights_alias_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    search_card_report = {"enabled": False, "written_cards": 0, "card_keys": []}
    if not args.skip_search_cards:
        cards_dir = Path(args.cards_dir).expanduser()
        cards = build_search_cards(rows, min_labeled_rows=args.min_card_labeled_rows)
        card_index = write_search_cards(cards, cards_dir)
        search_card_report = {
            "enabled": True,
            "cards_dir": str(cards_dir.resolve()),
            "written_cards": len(cards),
            "card_keys": sorted(cards.keys()),
            "index_path": str((cards_dir / "index.json").resolve()),
            "index": card_index,
        }
    report = {
        "ok": True,
        "model_path": str(model_path.resolve()),
        "checkpoint_path": str((model_path if model_path.suffix.lower() == ".pt" else model_path.with_suffix(".pt")).resolve()),
        "checkpoint_alias_path": str(checkpoint_alias_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "weights_alias_path": str(weights_alias_path.resolve()),
        "checkpoint_format": checkpoint["format_version"],
        "model_version": model.model_version,
        "created_at": model.created_at,
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
        "labels": [str(path.resolve()) for path in label_paths],
        "search_cards": search_card_report,
        "weights": {
            "format_version": weights["format_version"],
            "compatible_hosts": weights["compatible_hosts"],
            "top_feature_count": {
                label: len(weights["feature_weights"].get(label, {}))
                for label in weights["feature_weights"]
            },
        },
    }
    report_path = Path(args.report_out).expanduser() if args.report_out else model_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
