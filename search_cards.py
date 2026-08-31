#!/usr/bin/env python3
"""Generate reusable product search cards from human review labels."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from filter_training import compact_text, normalize_label


SEARCH_CARD_VERSION = "search-card-v1"

QUERY_SIGNAL_PHRASES = (
    "official website",
    "official specs",
    "official specifications",
    "product details",
    "size chart",
    "pricing",
    "plans",
    "features",
    "API docs",
    "SDK",
    "webhook",
    "integrations",
    "models",
    "rate limits",
    "quota",
    "security",
    "privacy",
    "SOC2",
    "SSO",
    "changelog",
    "release notes",
    "review",
    "reviews",
    "quality",
    "durability",
    "fit",
    "ventilation",
    "weight",
    "materials",
    "certification",
    "demo",
    "screenshot",
    "官网",
    "官方",
    "参数",
    "规格",
    "定价",
    "价格",
    "套餐",
    "功能",
    "文档",
    "接口",
    "集成",
    "模型",
    "额度",
    "限制",
    "安全",
    "隐私",
    "更新",
    "评测",
    "评价",
    "质量",
    "尺码",
    "重量",
    "材质",
    "认证",
)

FACT_TYPE_TERMS = {
    "pricing_packaging": ["pricing", "plans", "pricing plans"],
    "product_specs": ["official specs", "product details"],
    "physical_specs": ["official specs", "size chart", "weight", "materials"],
    "api_docs_limits": ["API docs", "SDK", "rate limits", "quota"],
    "app_store_metadata": ["app store", "reviews", "screenshots"],
    "visual_product_evidence": ["demo", "review", "screenshot"],
    "review_quality_perception": ["review", "quality", "durability"],
    "community_user_feedback": ["review", "problem", "user feedback"],
    "security_compliance": ["security", "privacy", "SOC2", "SSO"],
    "autonomous_vehicle_competitor_evidence": [
        "Robotaxi",
        "service area",
        "city coverage",
        "safety report",
        "sensor suite",
        "ride experience",
    ],
}

REUSABLE_PLATFORM_DOMAINS = {
    "youtube.com",
    "bilibili.com",
    "reddit.com",
    "zhihu.com",
    "v2ex.com",
    "producthunt.com",
    "g2.com",
    "capterra.com",
    "apps.apple.com",
    "play.google.com",
    "chromewebstore.google.com",
    "github.com",
    "npmjs.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "douyin.com",
    "xiaohongshu.com",
    "weixin.qq.com",
}

NOISE_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "localhost",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    value = compact_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "general_product"


def unique_strings(values: Iterable[Any], limit: int = 0) -> List[str]:
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
        if limit and len(output) >= limit:
            break
    return output


def split_terms(value: Any) -> List[str]:
    terms: List[str] = []
    for chunk in re.split(r"[\n,，;；|、/]+", compact_text(value)):
        item = re.sub(r"\s+", " ", chunk).strip()
        if item:
            terms.append(item)
    return terms


def row_domain(row: Mapping[str, Any]) -> str:
    explicit = compact_text(row.get("domain")).lower().removeprefix("www.")
    if explicit:
        return explicit
    url = compact_text(row.get("url") or row.get("url_or_path") or row.get("gui_review_url"))
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return host


def domain_root(domain: str) -> str:
    parts = [part for part in domain.split(".") if part]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def text_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9-]{1,40}|[\u4e00-\u9fff]{2,12}", value.lower()))


def competitor_tokens(row: Mapping[str, Any]) -> set[str]:
    values = [
        compact_text(row.get("competitor")),
        compact_text(row.get("product_type_label")),
        compact_text(row.get("product_category")),
    ]
    return set().union(*(text_tokens(value) for value in values if value))


def is_reusable_domain(domain: str, rows: Sequence[Mapping[str, Any]]) -> bool:
    domain = domain.lower().removeprefix("www.")
    if not domain or domain in NOISE_DOMAINS:
        return False
    if any(domain == item or domain.endswith("." + item) for item in REUSABLE_PLATFORM_DOMAINS):
        return True
    root = domain_root(domain)
    if not root:
        return False
    for row in rows:
        tokens = competitor_tokens(row)
        if root in tokens or any(root in token or token in root for token in tokens if len(token) >= 4):
            return False
    return True


def search_card_key(row: Mapping[str, Any]) -> str:
    return slugify(
        compact_text(row.get("product_type_key"))
        or compact_text(row.get("product_category"))
        or compact_text(row.get("category"))
        or compact_text(row.get("product_type_label"))
        or "general_product"
    )


def has_product_type_metadata(row: Mapping[str, Any]) -> bool:
    return any(
        compact_text(row.get(field))
        for field in ("product_type_key", "product_category", "product_type_label", "category_label")
    )


def product_type_label(rows: Sequence[Mapping[str, Any]], fallback: str) -> str:
    counts = Counter(
        compact_text(row.get("product_type_label") or row.get("category_label") or row.get("product_category"))
        for row in rows
    )
    counts.pop("", None)
    return counts.most_common(1)[0][0] if counts else fallback


def product_category(rows: Sequence[Mapping[str, Any]], fallback: str) -> str:
    counts = Counter(compact_text(row.get("product_category") or row.get("category")) for row in rows)
    counts.pop("", None)
    return counts.most_common(1)[0][0] if counts else fallback


def collect_positive_terms(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    terms: Counter[str] = Counter()
    for row in rows:
        label = normalize_label(row.get("human_label") or row.get("label"))
        if label not in {"include", "verify_later"}:
            continue
        weight = 2 if label == "include" else 1
        for term in split_terms(row.get("matched_include_keywords")):
            terms[term] += weight + 1
        for term in split_terms(row.get("matched_fields")):
            terms[term] += weight
        fact_type = compact_text(row.get("fact_type"))
        for term in FACT_TYPE_TERMS.get(fact_type, []):
            terms[term] += weight
        query = compact_text(row.get("query"))
        query_lower = query.lower()
        for phrase in QUERY_SIGNAL_PHRASES:
            if phrase.lower() in query_lower:
                terms[phrase] += weight
    return terms


def collect_exclude_terms(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    terms: Counter[str] = Counter()
    for row in rows:
        label = normalize_label(row.get("human_label") or row.get("label"))
        if label != "exclude":
            continue
        for term in split_terms(row.get("matched_exclude_keywords")):
            terms[term] += 3
        text = " ".join(
            compact_text(row.get(field))
            for field in ("title", "query", "human_reason")
            if compact_text(row.get(field))
        ).lower()
        for phrase in QUERY_SIGNAL_PHRASES:
            if phrase.lower() in text:
                terms[phrase] += 1
    return terms


def most_common_strings(counter: Counter[str], limit: int) -> List[str]:
    return [
        value
        for value, _count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:limit]
    ]


def source_suffix_for(domain: str, rows: Sequence[Mapping[str, Any]]) -> str:
    roles = Counter(compact_text(row.get("page_role")) for row in rows if row_domain(row) == domain)
    facts = Counter(compact_text(row.get("fact_type")) for row in rows if row_domain(row) == domain)
    joined = " ".join([*roles, *facts]).lower()
    if "video" in joined or domain in {"youtube.com", "bilibili.com", "tiktok.com", "douyin.com"}:
        return "review"
    if "app_store" in joined or domain in {"apps.apple.com", "play.google.com", "chromewebstore.google.com"}:
        return "reviews screenshots"
    if "docs" in joined or "api" in joined or domain in {"github.com", "npmjs.com"}:
        return "docs API"
    if "review" in joined or "quality" in joined:
        return "review"
    return "product"


def confidence_for(training_rows: int, included_rows: int, excluded_rows: int, verify_later_rows: int) -> str:
    if training_rows >= 20 and included_rows >= 5 and excluded_rows >= 3:
        return "high"
    if training_rows >= 8 and included_rows >= 3 and (excluded_rows + verify_later_rows) >= 2:
        return "medium"
    return "low"


def build_search_cards(
    rows: Sequence[Mapping[str, Any]],
    min_labeled_rows: int = 5,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = normalize_label(row.get("human_label") or row.get("label"))
        if not label:
            continue
        card_candidate = compact_text(row.get("search_card_candidate")).lower()
        if card_candidate in {"no", "false", "0", "否"}:
            continue
        if not has_product_type_metadata(row):
            continue
        grouped[search_card_key(row)].append(row)

    cards: Dict[str, Dict[str, Any]] = {}
    for key, card_rows in grouped.items():
        if len(card_rows) < min_labeled_rows:
            continue
        labels = Counter(normalize_label(row.get("human_label") or row.get("label")) for row in card_rows)
        positive_rows = [
            row for row in card_rows if normalize_label(row.get("human_label") or row.get("label")) in {"include", "verify_later"}
        ]
        include_rows = [row for row in card_rows if normalize_label(row.get("human_label") or row.get("label")) == "include"]
        domains = Counter(row_domain(row) for row in positive_rows if row_domain(row))
        reusable_domains = [
            domain
            for domain in most_common_strings(domains, 20)
            if is_reusable_domain(domain, positive_rows)
        ]
        search_terms = most_common_strings(collect_positive_terms(card_rows), 40)
        exclude_keywords = most_common_strings(collect_exclude_terms(card_rows), 30)
        page_roles = Counter(compact_text(row.get("page_role")) for row in positive_rows)
        page_roles.pop("", None)
        source_tiers = Counter(compact_text(row.get("source_policy_tier")) for row in positive_rows)
        source_tiers.pop("", None)
        evidence_fields = Counter()
        for row in include_rows:
            for field in split_terms(row.get("matched_fields")):
                evidence_fields[field] += 1
        source_templates = [f"{{name}} {term}" for term in search_terms[:15]]
        for domain in reusable_domains[:10]:
            source_templates.append(f"{{name}} site:{domain} {source_suffix_for(domain, positive_rows)}")

        category = product_category(card_rows, key)
        label = product_type_label(card_rows, category)
        cards[key] = {
            "card_version": SEARCH_CARD_VERSION,
            "generated_at": utc_stamp(),
            "product_type_key": key,
            "product_category": category,
            "product_type_label": label,
            "training_rows": len(card_rows),
            "included_rows": labels.get("include", 0),
            "excluded_rows": labels.get("exclude", 0),
            "verify_later_rows": labels.get("verify_later", 0),
            "confidence": confidence_for(
                len(card_rows),
                labels.get("include", 0),
                labels.get("exclude", 0),
                labels.get("verify_later", 0),
            ),
            "search_terms": unique_strings(search_terms, 40),
            "exclude_keywords": unique_strings(exclude_keywords, 30),
            "evidence_keywords": unique_strings([*search_terms, *most_common_strings(evidence_fields, 20)], 60),
            "source_domains": unique_strings(most_common_strings(domains, 30), 30),
            "reusable_source_domains": unique_strings(reusable_domains, 20),
            "preferred_page_roles": unique_strings(most_common_strings(page_roles, 20), 20),
            "preferred_source_tiers": unique_strings(most_common_strings(source_tiers, 20), 20),
            "evidence_fields": unique_strings(most_common_strings(evidence_fields, 25), 25),
            "directed_source_search_templates": unique_strings(source_templates, 40),
            "term_reasons": [
                {
                    "term": term,
                    "reason": "来自同类商品人工收录或待核实样本，下一轮优先作为搜索和证据命中线索。",
                }
                for term in unique_strings(search_terms, 25)
            ],
            "usage_policy": [
                "卡片只提供同类商品的搜索和筛选经验，不覆盖本轮抓取前分析。",
                "官方事实仍以当前竞品官网、官方文档和可追溯页面为准。",
                "登录、付费、验证码、私有接口和访问控制绕过仍由硬规则拦截。",
            ],
        }
    return cards


def write_search_card_markdown(path: Path, card: Mapping[str, Any]) -> None:
    lines = [
        f"# 搜索卡片：{card.get('product_type_label') or card.get('product_type_key')}",
        "",
        f"- **卡片版本:** {card.get('card_version')}",
        f"- **产品类型:** {card.get('product_type_key')}",
        f"- **训练样本:** {card.get('training_rows')} 条",
        f"- **置信度:** {card.get('confidence')}",
        f"- **收录/排除/待核实:** {card.get('included_rows')} / {card.get('excluded_rows')} / {card.get('verify_later_rows')}",
        "",
        "## 搜索词",
        "",
        ", ".join(card.get("search_terms") or []) or "无",
        "",
        "## 排除词",
        "",
        ", ".join(card.get("exclude_keywords") or []) or "无",
        "",
        "## 可复用来源",
        "",
        ", ".join(card.get("reusable_source_domains") or []) or "无",
        "",
        "## 定向搜索模板",
        "",
    ]
    for template in card.get("directed_source_search_templates") or []:
        lines.append(f"- `{template}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_search_cards(cards: Mapping[str, Mapping[str, Any]], cards_dir: Path) -> Dict[str, Any]:
    cards_dir = Path(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "generated_at": utc_stamp(),
        "card_version": SEARCH_CARD_VERSION,
        "cards": {},
    }
    for key, card in sorted(cards.items()):
        safe_key = slugify(key)
        payload = dict(card)
        payload["product_type_key"] = safe_key
        json_path = cards_dir / f"{safe_key}.json"
        md_path = cards_dir / f"{safe_key}.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_search_card_markdown(md_path, payload)
        index["cards"][safe_key] = {
            "product_type_label": payload.get("product_type_label", ""),
            "product_category": payload.get("product_category", ""),
            "confidence": payload.get("confidence", ""),
            "training_rows": payload.get("training_rows", 0),
            "json": json_path.name,
            "markdown": md_path.name,
        }
    (cards_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_search_cards(
    cards_dir: Path,
    product_category: str = "",
    product_type_key: str = "",
) -> List[Dict[str, Any]]:
    cards_dir = Path(cards_dir)
    if not cards_dir.exists():
        return []
    wanted_key = slugify(product_type_key or product_category) if (product_type_key or product_category) else ""
    wanted_category = compact_text(product_category).lower()
    cards: List[Dict[str, Any]] = []
    for path in sorted(cards_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        card_key = slugify(card.get("product_type_key") or path.stem)
        card_category = compact_text(card.get("product_category")).lower()
        if wanted_key and card_key != wanted_key and card_category != wanted_category:
            continue
        card["_source_path"] = str(path)
        cards.append(card)
    return cards
