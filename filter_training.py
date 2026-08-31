#!/usr/bin/env python3
"""Local trainable filter for competitor evidence screening.

The model is intentionally small and dependency-free so open-source users can
train it locally without installing a machine-learning stack. It learns from
human review labels and then produces scores that the rule filter can use as a
secondary signal.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


LABELS = ("include", "exclude", "verify_later")
MODEL_VERSION = "local-naive-bayes-v1"

LABEL_ALIASES = {
    "include": "include",
    "included": "include",
    "accept": "include",
    "accepted": "include",
    "yes": "include",
    "y": "include",
    "1": "include",
    "收录": "include",
    "保留": "include",
    "采用": "include",
    "exclude": "exclude",
    "excluded": "exclude",
    "reject": "exclude",
    "rejected": "exclude",
    "no": "exclude",
    "n": "exclude",
    "0": "exclude",
    "不收录": "exclude",
    "排除": "exclude",
    "删除": "exclude",
    "verify": "verify_later",
    "verify_later": "verify_later",
    "review": "verify_later",
    "manual_review": "verify_later",
    "later": "verify_later",
    "待核实": "verify_later",
    "人工复核": "verify_later",
    "待复核": "verify_later",
}

CATEGORICAL_FIELDS = (
    "source_stage",
    "source_type",
    "source_status",
    "source_kind",
    "page_role",
    "source_policy_tier",
    "fact_type",
    "increment_type",
    "decision",
    "decision_status",
    "gate_result",
    "hard_gate",
    "confidence",
    "pending_verification",
    "primary_evidence_candidate",
    "official_or_public",
    "selected_for_crawl",
)

TEXT_FIELDS = (
    "competitor",
    "title",
    "url",
    "url_or_path",
    "domain",
    "query",
    "engine",
    "snippet",
    "content_preview",
    "reason",
    "selection_note",
    "verification_reason",
    "primary_evidence_reason",
    "matched_fields",
    "matched_include_keywords",
    "matched_exclude_keywords",
    "human_reason",
)

PROTECTED_HARD_GATES = (
    "rejected_manual_exclude_keyword",
    "rejected_auth_or_transaction_shell",
    "rejected_non_html",
    "rejected_non_html_asset",
    "rejected_missing_url",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_label(value: Any) -> str:
    raw = textify(value).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(raw, "")


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", textify(value)).strip()


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]{1,48}|[\u4e00-\u9fff]{2,12}", text)
    cleaned: List[str] = []
    for token in tokens:
        token = token.strip("._-")
        if len(token) < 2:
            continue
        cleaned.append(token[:50])
    return cleaned


def row_url(row: Mapping[str, Any]) -> str:
    return compact_text(row.get("url") or row.get("url_or_path") or row.get("gui_review_url"))


def row_domain(row: Mapping[str, Any]) -> str:
    explicit = compact_text(row.get("domain"))
    if explicit:
        return explicit.lower()
    parsed = urlparse(row_url(row))
    return (parsed.netloc or "").lower().removeprefix("www.")


def extract_features(row: Mapping[str, Any]) -> List[str]:
    features: List[str] = []
    for field in CATEGORICAL_FIELDS:
        value = compact_text(row.get(field)).lower()
        if value:
            features.append(f"{field}={value[:80]}")
    domain = row_domain(row)
    if domain:
        features.append(f"domain={domain}")
        parts = [part for part in domain.split(".") if part]
        if parts:
            features.append(f"domain_root={parts[-2] if len(parts) >= 2 else parts[0]}")
    for field in TEXT_FIELDS:
        value = compact_text(row.get(field))
        if value:
            for token in tokenize(value):
                features.append(f"tok={token}")
    return sorted(set(features))


@dataclasses.dataclass
class FilterPrediction:
    label: str
    include_score: float
    exclude_score: float
    verify_later_score: float
    confidence: str
    reason: str
    model_version: str

    @property
    def scores(self) -> Dict[str, float]:
        return {
            "include": self.include_score,
            "exclude": self.exclude_score,
            "verify_later": self.verify_later_score,
        }


@dataclasses.dataclass
class LocalFilterModel:
    model_version: str
    created_at: str
    training_rows: int
    label_counts: Dict[str, int]
    feature_counts: Dict[str, Dict[str, int]]
    feature_totals: Dict[str, int]
    vocabulary: List[str]
    alpha: float = 1.0

    def predict(self, row: Mapping[str, Any]) -> FilterPrediction:
        features = extract_features(row)
        labels = [label for label in LABELS if self.label_counts.get(label, 0) > 0] or list(LABELS)
        vocab_size = max(1, len(self.vocabulary))
        total_rows = max(1, sum(self.label_counts.get(label, 0) for label in LABELS))
        log_scores: Dict[str, float] = {}
        for label in labels:
            prior = (self.label_counts.get(label, 0) + self.alpha) / (
                total_rows + self.alpha * len(LABELS)
            )
            log_score = math.log(prior)
            denom = self.feature_totals.get(label, 0) + self.alpha * vocab_size
            counts = self.feature_counts.get(label, {})
            for feature in features:
                log_score += math.log((counts.get(feature, 0) + self.alpha) / denom)
            log_scores[label] = log_score

        max_log = max(log_scores.values()) if log_scores else 0.0
        exps = {label: math.exp(value - max_log) for label, value in log_scores.items()}
        denom = sum(exps.values()) or 1.0
        probabilities = {label: exps.get(label, 0.0) / denom for label in LABELS}
        label = max(probabilities.items(), key=lambda item: item[1])[0]
        ordered = sorted(probabilities.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        if ordered[0] >= 0.85 and margin >= 0.35:
            confidence = "high"
        elif ordered[0] >= 0.65 and margin >= 0.18:
            confidence = "medium"
        else:
            confidence = "low"
        reason = self._prediction_reason(label, features)
        return FilterPrediction(
            label=label,
            include_score=round(probabilities.get("include", 0.0), 4),
            exclude_score=round(probabilities.get("exclude", 0.0), 4),
            verify_later_score=round(probabilities.get("verify_later", 0.0), 4),
            confidence=confidence,
            reason=reason,
            model_version=self.model_version,
        )

    def _prediction_reason(self, label: str, features: Sequence[str]) -> str:
        counts = self.feature_counts.get(label, {})
        matched = sorted(
            (feature for feature in features if counts.get(feature, 0) > 0),
            key=lambda feature: (-counts.get(feature, 0), feature),
        )[:6]
        if not matched:
            return "no strong learned feature matched; prediction uses learned label priors"
        return "matched learned features: " + ", ".join(matched)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "created_at": self.created_at,
            "training_rows": self.training_rows,
            "label_counts": self.label_counts,
            "feature_counts": self.feature_counts,
            "feature_totals": self.feature_totals,
            "vocabulary": self.vocabulary,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocalFilterModel":
        return cls(
            model_version=compact_text(payload.get("model_version")) or MODEL_VERSION,
            created_at=compact_text(payload.get("created_at")) or utc_stamp(),
            training_rows=int(payload.get("training_rows") or 0),
            label_counts={label: int((payload.get("label_counts") or {}).get(label, 0)) for label in LABELS},
            feature_counts={
                label: {str(key): int(value) for key, value in ((payload.get("feature_counts") or {}).get(label, {}) or {}).items()}
                for label in LABELS
            },
            feature_totals={label: int((payload.get("feature_totals") or {}).get(label, 0)) for label in LABELS},
            vocabulary=[str(item) for item in (payload.get("vocabulary") or [])],
            alpha=float(payload.get("alpha") or 1.0),
        )


def build_training_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = normalize_label(
                    row.get("human_label")
                    or row.get("人工标签")
                    or row.get("label")
                    or row.get("review_label")
                )
                if not label:
                    continue
                normalized = {str(key): compact_text(value) for key, value in row.items() if key is not None}
                normalized["human_label"] = label
                rows.append(normalized)
    return rows


def train_filter_model(
    rows: Sequence[Mapping[str, Any]],
    min_labeled_rows: int = 10,
    alpha: float = 1.0,
) -> LocalFilterModel:
    labeled_rows: List[Tuple[str, List[str]]] = []
    for row in rows:
        label = normalize_label(row.get("human_label") or row.get("label") or row.get("人工标签"))
        if not label:
            continue
        features = extract_features(row)
        if features:
            labeled_rows.append((label, features))
    if len(labeled_rows) < min_labeled_rows:
        raise ValueError(f"至少需要 {min_labeled_rows} 条已标注样本，当前只有 {len(labeled_rows)} 条。")

    label_counts: Counter[str] = Counter()
    feature_counts: Dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    vocabulary: set[str] = set()
    for label, features in labeled_rows:
        label_counts[label] += 1
        feature_counts[label].update(features)
        vocabulary.update(features)

    return LocalFilterModel(
        model_version=MODEL_VERSION,
        created_at=utc_stamp(),
        training_rows=len(labeled_rows),
        label_counts={label: int(label_counts.get(label, 0)) for label in LABELS},
        feature_counts={label: dict(feature_counts[label]) for label in LABELS},
        feature_totals={label: int(sum(feature_counts[label].values())) for label in LABELS},
        vocabulary=sorted(vocabulary),
        alpha=alpha,
    )


def save_filter_model(model: LocalFilterModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def model_feature_weights(model: LocalFilterModel, top_n: int = 500) -> Dict[str, Dict[str, float]]:
    vocabulary = list(model.vocabulary)
    vocab_size = max(1, len(vocabulary))
    weights: Dict[str, Dict[str, float]] = {}
    for label in LABELS:
        label_counts = model.feature_counts.get(label, {})
        label_total = model.feature_totals.get(label, 0)
        other_counts: Counter[str] = Counter()
        other_total = 0
        for other_label in LABELS:
            if other_label == label:
                continue
            other_counts.update(model.feature_counts.get(other_label, {}))
            other_total += model.feature_totals.get(other_label, 0)
        scored = []
        for feature in vocabulary:
            p_label = (label_counts.get(feature, 0) + model.alpha) / (label_total + model.alpha * vocab_size)
            p_other = (other_counts.get(feature, 0) + model.alpha) / (other_total + model.alpha * vocab_size)
            scored.append((feature, round(math.log(p_label / p_other), 6)))
        scored.sort(key=lambda item: (-abs(item[1]), item[0]))
        weights[label] = dict(scored[: max(1, top_n)])
    return weights


def save_model_weights(model: LocalFilterModel, path: Path, top_n: int = 500) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_weights = model_feature_weights(model, top_n=top_n)
    payload = {
        "format_version": "filter-weights-v1",
        "model_version": model.model_version,
        "created_at": utc_stamp(),
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
        "compatible_hosts": [
            "codex",
            "claude-code",
            "cursor",
            "deepseek",
            "openai-compatible",
            "local-python",
        ],
        "usage": (
            "本文件是本地筛选模型的可读权重摘要，用来解释哪些特征会提高 include、exclude 或 verify_later。"
            "不同代码助手可读取该文件辅助理解本地判断策略；运行时默认加载 filter_model.pt。"
        ),
        "feature_weights": feature_weights,
        "top_positive_features": {
            label: [
                {"feature": feature, "weight": weight}
                for feature, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:30]
                if weight > 0
            ]
            for label, weights in feature_weights.items()
        },
        "top_negative_features": {
            label: [
                {"feature": feature, "weight": weight}
                for feature, weight in sorted(weights.items(), key=lambda item: (item[1], item[0]))[:30]
                if weight < 0
            ]
            for label, weights in feature_weights.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def model_checkpoint_payload(model: LocalFilterModel, top_n: int = 500) -> Dict[str, Any]:
    return {
        "format_version": "filter-checkpoint-pt-v1",
        "created_at": utc_stamp(),
        "model_state": model.to_dict(),
        "feature_weights": model_feature_weights(model, top_n=top_n),
        "compatible_hosts": [
            "codex",
            "claude-code",
            "cursor",
            "deepseek",
            "openai-compatible",
            "local-python",
        ],
        "usage": (
            "This .pt checkpoint stores the local evidence-filter model used by the harvester. "
            "It is not a fine-tuned LLM; it improves local include/exclude/verify_later decisions."
        ),
    }


def save_model_checkpoint_pt(model: LocalFilterModel, path: Path, top_n: int = 500) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model_checkpoint_payload(model, top_n=top_n)
    if os.environ.get("HARVESTER_USE_TORCH_PT") == "1":
        import torch  # type: ignore

        payload["serialization"] = "torch.save"
        torch.save(payload, path)
        return payload
    payload["serialization"] = "pickle"
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_model_checkpoint_pt(path: Path) -> LocalFilterModel:
    path = Path(path)
    payload: Any
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        import torch  # type: ignore

        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("invalid .pt filter checkpoint")
    state = payload.get("model_state") or payload
    if not isinstance(state, Mapping):
        raise ValueError("invalid .pt filter checkpoint model_state")
    return LocalFilterModel.from_dict(state)


def load_filter_model(path: Path) -> LocalFilterModel:
    path = Path(path)
    if path.suffix.lower() == ".pt":
        return load_model_checkpoint_pt(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LocalFilterModel.from_dict(payload)


def protected_by_rule(decision_row: Mapping[str, Any]) -> bool:
    hard_gate = compact_text(decision_row.get("hard_gate")).lower()
    source_policy = compact_text(decision_row.get("source_policy_tier")).lower()
    if source_policy.startswith("reject"):
        return True
    return any(hard_gate.startswith(gate) for gate in PROTECTED_HARD_GATES)


def apply_ml_prediction_to_decision(
    decision_row: Mapping[str, Any],
    prediction: FilterPrediction,
    auto_include_threshold: float = 0.75,
    auto_exclude_threshold: float = 0.80,
    verify_later_threshold: float = 0.70,
) -> Dict[str, Any]:
    row = dict(decision_row)
    row.update(
        {
            "ml_label": prediction.label,
            "ml_include_score": f"{prediction.include_score:.4f}",
            "ml_exclude_score": f"{prediction.exclude_score:.4f}",
            "ml_verify_later_score": f"{prediction.verify_later_score:.4f}",
            "ml_confidence": prediction.confidence,
            "ml_reason": prediction.reason,
            "ml_model_version": prediction.model_version,
            "ml_adjustment": "none",
        }
    )
    if protected_by_rule(row):
        row["ml_adjustment"] = "annotated_only_rule_protected"
        return row

    status = compact_text(row.get("decision_status")).lower()
    if prediction.label == "include" and prediction.include_score >= auto_include_threshold:
        if status in {"signal", "rejected", "accepted"}:
            row["decision_status"] = "accepted"
            row["gate_result"] = "accepted_by_local_training_model"
            row["hard_gate"] = "ml_promoted_to_accepted"
            row["ml_adjustment"] = "promoted_to_accepted"
    elif prediction.label == "exclude" and prediction.exclude_score >= auto_exclude_threshold:
        if status in {"signal", "accepted"}:
            row["decision_status"] = "rejected"
            row["gate_result"] = "rejected_by_local_training_model"
            row["hard_gate"] = "ml_rejected_low_value"
            row["ml_adjustment"] = "demoted_to_rejected"
    elif prediction.label == "verify_later" and prediction.verify_later_score >= verify_later_threshold:
        row["pending_verification"] = "yes"
        row["verification_reason"] = row.get("verification_reason") or "本地训练模型判断需要人工复核"
        row["ml_adjustment"] = "marked_pending_verification"
    return row


def model_status(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"enabled": False, "path": str(path), "message": "model file not found"}
    try:
        model = load_filter_model(path)
        return {
            "enabled": True,
            "path": str(path),
            "model_version": model.model_version,
            "created_at": model.created_at,
            "training_rows": model.training_rows,
            "label_counts": model.label_counts,
        }
    except Exception as exc:
        return {"enabled": False, "path": str(path), "message": str(exc)}


def bootstrap_filter_model_if_missing(
    path: Path,
    label_paths: Sequence[Path],
    min_labeled_rows: int = 3,
) -> Dict[str, Any]:
    path = Path(path)
    if path.exists():
        status = model_status(path)
        status["created"] = False
        return status
    rows = build_training_rows([Path(item) for item in label_paths])
    model = train_filter_model(rows, min_labeled_rows=min_labeled_rows)
    checkpoint = save_model_checkpoint_pt(model, path)
    weights_path = path.with_name("filter_weights.json")
    save_model_weights(model, weights_path)
    return {
        "created": True,
        "enabled": True,
        "path": str(path),
        "checkpoint_path": str(path),
        "weights_path": str(weights_path),
        "checkpoint_format": checkpoint["format_version"],
        "model_version": model.model_version,
        "created_at": model.created_at,
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
    }
