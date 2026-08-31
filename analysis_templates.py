#!/usr/bin/env python3
"""Load product-specific analysis templates for crawl planning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


APP_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_DIR = APP_DIR / "analysis_dimensions"


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", textify(value)).strip()


def unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        text = compact_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def load_analysis_templates(template_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = Path(template_dir or DEFAULT_TEMPLATE_DIR)
    if not root.exists():
        return []
    templates: List[Dict[str, Any]] = []
    for path in sorted([*root.glob("*.yml"), *root.glob("*.yaml")]):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_source_path"] = str(path.resolve())
        templates.append(payload)
    return templates


def analysis_template_match_score(template: Mapping[str, Any], category: str, texts: Sequence[str]) -> int:
    haystack = " ".join(texts).lower()
    compact = re.sub(r"\s+", "", haystack)
    score = 0
    template_key = compact_text(template.get("product_type_key")).lower()
    template_label = compact_text(template.get("product_type_label")).lower()
    if template_key and template_key == category.lower():
        score += 100
    if template_key and template_key in haystack:
        score += 20
    if template_label and template_label in haystack:
        score += 20
    for keyword in unique_strings(template.get("match_keywords") or []):
        needle = keyword.lower()
        compact_needle = re.sub(r"\s+", "", needle)
        if needle in haystack or (compact_needle and compact_needle in compact):
            score += 8
    for dimension in template.get("dimensions") or []:
        if not isinstance(dimension, dict):
            continue
        for keyword in unique_strings([dimension.get("label"), *(dimension.get("required_evidence") or [])]):
            needle = keyword.lower()
            compact_needle = re.sub(r"\s+", "", needle)
            if needle in haystack or (compact_needle and compact_needle in compact):
                score += 2
    return score


def select_analysis_template(
    category: str,
    competitors: Sequence[str],
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
    template_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], int]:
    texts = [
        category,
        own_product_name,
        own_product_positioning,
        own_product_context,
        *competitors,
    ]
    best_template: Optional[Dict[str, Any]] = None
    best_score = 0
    for template in load_analysis_templates(template_dir):
        score = analysis_template_match_score(template, category, texts)
        if score > best_score:
            best_template = template
            best_score = score
    if best_template and best_score >= 8:
        return best_template, best_score
    return None, 0
