#!/usr/bin/env python3
"""Schema-driven structured fact extraction for competitor evidence."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "product-evidence-schema-v1"
EXTRACTION_METHOD = "schema_extractor_v1"


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", textify(value)).strip()


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


def dimension_for_field(field_key: str) -> str:
    if field_key in {"pricing", "packaging_limits"}:
        return "pricing"
    if field_key in {
        "weight",
        "size",
        "size_fit",
        "dimensions",
        "materials",
        "material_construction",
        "color_variants",
        "certification",
        "safety_certification",
        "protection_technology",
        "ventilation_comfort",
        "visor_goggle_chinguard",
    }:
        return "product_specs"
    if field_key in {"api_sdk_webhook", "usage_quota_limits", "integrations_connectors"}:
        return "api_and_limits"
    if field_key in {"security_privacy_deployment"}:
        return "security"
    if field_key in {"customers", "gtm_channel"}:
        return "gtm_customer"
    if field_key in {"av_positioning", "av_market_operations", "av_release_tracking"}:
        return "autonomous_vehicle_market_operations"
    if field_key in {"av_pricing_commercialization"}:
        return "autonomous_vehicle_commercialization"
    if field_key in {"av_vehicle_platform", "av_autonomous_system"}:
        return "autonomous_vehicle_technology_specs"
    if field_key in {"av_safety_compliance"}:
        return "autonomous_vehicle_safety_compliance"
    if field_key in {"av_ride_experience", "av_edge_case_performance", "av_hmi_cabin"}:
        return "autonomous_vehicle_user_experience"
    if field_key in {"av_operations_maintenance"}:
        return "autonomous_vehicle_operations"
    if field_key in {"av_public_opinion"}:
        return "autonomous_vehicle_public_opinion"
    return "product_capability"


DEFAULT_FIELDS: Dict[str, Dict[str, Any]] = {
    "pricing": {
        "label": "定价",
        "description": "价格、订阅周期、套餐价格或单次购买价格。",
        "patterns": [
            r"([$€£¥￥]\s?\d+(?:[.,]\d+)?\s*(?:/|per)\s*(?:monthly|month|mo|annually|annual|year|yr))",
            r"([$€£¥￥]\s?\d+(?:[.,]\d+)?)",
        ],
    },
    "packaging_limits": {
        "label": "套餐/包装/限制",
        "description": "免费版、Pro、Enterprise、额度、席位、次数或功能限制。",
        "patterns": [
            r"\b(Free|Starter|Basic|Pro|Business|Team|Enterprise)\s+(?:plan|tier|套餐|版)\b",
            r"\b(\d[\d,.\s]*(?:credits?|tokens?|seats?|users?|projects?|exports?)(?:\s+per\s+(?:month|year|day))?)\b",
        ],
    },
    "api_sdk_webhook": {
        "label": "API/SDK/Webhook",
        "description": "API、SDK、Webhook、开发者接入方式。",
        "patterns": [
            r"\b(REST\s+API|GraphQL\s+API|API|SDK|webhooks?|developer\s+API)\b",
        ],
    },
    "usage_quota_limits": {
        "label": "额度/调用限制",
        "description": "API 调用、tokens、credits、requests、rate limit 等额度限制。",
        "patterns": [
            r"\b(?:rate\s+limit|quota|limits?|额度|限制)\s*[:：]?\s*(\d[\d,.\s]*(?:requests?|tokens?|credits?|calls?)(?:/[a-z]+|\s+per\s+(?:minute|hour|day|month|year))?)\b",
            r"\b(\d[\d,.\s]*(?:requests?|tokens?|credits?|calls?)(?:/[a-z]+|\s+per\s+(?:minute|hour|day|month|year)))\b",
        ],
    },
    "security_privacy_deployment": {
        "label": "安全/隐私/部署",
        "description": "SSO、安全认证、隐私合规、部署方式。",
        "patterns": [
            r"\b(SSO|SOC\s?2|GDPR|HIPAA|ISO\s?\d{3,6}|on[- ]premise|private cloud|data residency)\b",
        ],
    },
    "material_construction": {
        "label": "材质/结构",
        "description": "材料、壳体、内衬、结构工艺。",
        "patterns": [
            r"\b((?:ABS|PC|EPS|EPP|polycarbonate|carbon(?:\s+fiber)?|aluminum|steel|nylon|leather|silicone)\s+(?:hardshell|hard\s+shell|shell|liner|foam|construction|frame|body|material))\b",
        ],
    },
    "weight": {
        "label": "重量",
        "description": "产品重量。",
        "patterns": [
            r"\b(\d+(?:[.,]\d+)?)\s*(g|gram|grams|kg|oz|ounce|ounces|克|千克)\b",
        ],
    },
    "size_fit": {
        "label": "尺码/适配",
        "description": "尺码、头围、尺寸范围和适配方式。",
        "patterns": [
            r"\b(?:sizes?|size\s+chart|尺码|头围)\s*[:：]?\s*((?:XXS|XS|S|M|L|XL|XXL|[0-9]{2,3}(?:[-–][0-9]{2,3})?\s?cm)(?:\s*[,/、]\s*(?:XXS|XS|S|M|L|XL|XXL|[0-9]{2,3}(?:[-–][0-9]{2,3})?\s?cm))*)",
        ],
    },
    "dimensions": {
        "label": "尺寸",
        "description": "长宽高、头围、屏幕尺寸、型号尺寸。",
        "patterns": [
            r"\b(?:dimensions?|measurement|尺寸|头围)\s*[:：]?\s*((?:\d{2,4}(?:\.\d+)?\s?[-–x×]\s?\d{2,4}(?:\.\d+)?|\d{2,4}(?:\.\d+)?)\s?(?:cm|mm|in|inch|inches|厘米|毫米))\b",
        ],
    },
    "color_variants": {
        "label": "颜色",
        "description": "颜色、配色、SKU 颜色选项。",
        "patterns": [
            r"\b(?:color\s+options?|colou?r\s+variants?|colors?|colour|颜色|配色)\s*[:：]?\s*([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff0-9 ,/、-]{2,120})",
        ],
    },
    "certification": {
        "label": "认证",
        "description": "安全、质量、隐私、行业认证。",
        "patterns": [
            r"\b(ASTM\s?F2040|CE\s?EN\s?1077|EN\s?1077|FIS|MIPS|SOC\s?2|GDPR|HIPAA|ISO\s?\d{3,6})\b",
        ],
    },
    "av_positioning": {
        "label": "产品定位",
        "description": "目标用户、服务形态、运营场景或产品概念。",
        "patterns": [r"(目标用户|服务形态|运营场景|产品概念|Robotaxi|无人车|自动驾驶出租车)"],
    },
    "av_market_operations": {
        "label": "市场与运营",
        "description": "投放城市、运营范围、车辆规模、站点密度、等待时间或订单流转。",
        "patterns": [r"(投放城市|运营范围|服务范围|车辆规模|站点密度|等待时间|订单流转|service area|fleet|waiting time)"],
    },
    "av_pricing_commercialization": {
        "label": "价格与商业化",
        "description": "计费规则、优惠补贴、价格对比、盈利口径或合作方。",
        "patterns": [r"(计费规则|优惠补贴|价格对比|盈利口径|合作方|fare|pricing|subsidy|commercialization)"],
    },
    "av_vehicle_platform": {
        "label": "整车平台",
        "description": "基础车型、长宽高、轴距、重量、续航、能耗、座椅或补能方式。",
        "patterns": [r"(基础车型|长宽高|轴距|重量|续航|能耗|门型|座椅|补能方式|wheelbase|range|vehicle platform)"],
    },
    "av_autonomous_system": {
        "label": "自动驾驶系统",
        "description": "传感器、计算平台、算力、地图定位、清洁或冗余设计。",
        "patterns": [r"(传感器|激光雷达|毫米波雷达|摄像头|计算平台|算力|地图|定位|冗余|sensor|lidar|radar|camera|TOPS)"],
    },
    "av_safety_compliance": {
        "label": "安全与合规",
        "description": "远程监控、异常检测、事故处理、乘员保护、隐私保护或监管许可。",
        "patterns": [r"(远程监控|异常检测|碰撞处理|事故|乘员保护|隐私保护|监管许可|safety report|permit|remote assistance)"],
    },
    "av_ride_experience": {
        "label": "乘坐体验",
        "description": "上车、下车、路线、加速刹车、横向体感、车内屏、娱乐、温度或空间。",
        "patterns": [r"(上车|下车|路线|加速|刹车|横向体感|车内屏|音乐|温度|空间|ride experience|pickup|drop-off)"],
    },
    "av_edge_case_performance": {
        "label": "复杂场景表现",
        "description": "窄路掉头、锥桶施工、Cut in、无保护转弯、人流、交警指挥、临停避让或事故。",
        "patterns": [r"(窄路掉头|锥桶|施工|Cut in|无保护转弯|密集人流|交警指挥|临停避让|edge case|construction cone)"],
    },
    "av_hmi_cabin": {
        "label": "智舱与交互",
        "description": "车内外屏、语音、寻车、解锁、SOS、E-stop、摄像头提示或乘客控制项。",
        "patterns": [r"(车内外屏|语音|寻车|解锁|SOS|E-stop|摄像头提示|乘客控制|HMI|cabin screen)"],
    },
    "av_operations_maintenance": {
        "label": "运维效率",
        "description": "自动唤醒、自动出入库、OTA、远程启动、补能、清洁、故障诊断或线下救援。",
        "patterns": [r"(自动唤醒|自动出入库|OTA|远程启动|补能|清洁|故障诊断|线下救援|operations|maintenance)"],
    },
    "av_public_opinion": {
        "label": "社会接受度",
        "description": "热搜、指数、互动量、就业争议、隐私争议、安全争议或城市差异。",
        "patterns": [r"(热搜|指数|互动量|就业争议|隐私争议|安全争议|城市差异|public opinion|acceptance|privacy concern)"],
    },
    "av_release_tracking": {
        "label": "更新节奏",
        "description": "版本发布时间、功能更新、城市扩张、事故后改动或下一代车型。",
        "patterns": [r"(版本发布时间|功能更新|城市扩张|事故后改动|下一代车型|release|update|next generation)"],
    },
}


CATEGORY_FIELD_KEYS = {
    "ai_software": [
        "pricing",
        "packaging_limits",
        "api_sdk_webhook",
        "usage_quota_limits",
        "security_privacy_deployment",
    ],
    "snow_helmet": [
        "pricing",
        "material_construction",
        "weight",
        "size_fit",
        "dimensions",
        "color_variants",
        "certification",
    ],
    "physical_product": [
        "pricing",
        "material_construction",
        "weight",
        "size_fit",
        "dimensions",
        "color_variants",
        "certification",
    ],
    "general": ["pricing", "packaging_limits", "customers"],
    "autonomous_vehicle_robotaxi": [
        "pricing",
        "av_positioning",
        "av_market_operations",
        "av_pricing_commercialization",
        "av_vehicle_platform",
        "av_autonomous_system",
        "av_safety_compliance",
        "av_ride_experience",
        "av_edge_case_performance",
        "av_hmi_cabin",
        "av_operations_maintenance",
        "av_public_opinion",
        "av_release_tracking",
    ],
}


NAVIGATION_NOISE = (
    "login",
    "sign in",
    "start for free",
    "cookie",
    "privacy settings",
    "购物车",
    "登录",
    "注册",
    "导航",
)


def build_extraction_schema(category: str = "general", plan_fields: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    fields: Dict[str, Dict[str, Any]] = {}
    keys = unique_strings([*CATEGORY_FIELD_KEYS.get(category, []), *CATEGORY_FIELD_KEYS["general"]])
    for key in keys:
        if key in DEFAULT_FIELDS:
            fields[key] = {**DEFAULT_FIELDS[key], "evidence_required": ["value", "source_sentence", "source_url"]}
    for field in plan_fields or []:
        key = compact_text(field.get("key"))
        if not key:
            continue
        base = DEFAULT_FIELDS.get(key, {})
        fields[key] = {
            "label": compact_text(field.get("label")) or base.get("label") or key,
            "description": compact_text(field.get("description")) or base.get("description") or "",
            "patterns": list(base.get("patterns") or field.get("patterns") or []),
            "evidence_required": ["value", "source_sentence", "source_url"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "category": category or "general",
        "traceability_required": ["source_url", "source_title", "evidence_text", "extraction_method"],
        "fields": fields,
    }


def normalize_fact_value(value: Any, field_key: str = "") -> str:
    raw = compact_text(value)
    low = raw.lower()
    if field_key == "weight":
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(g|gram|grams|kg|oz|ounce|ounces|克|千克)", low)
        if match:
            number = float(match.group(1).replace(",", "."))
            unit = match.group(2)
            grams = number
            if unit in {"kg", "千克"}:
                grams = number * 1000
            elif unit in {"oz", "ounce", "ounces"}:
                grams = number * 28.3495
            return f"{int(round(grams))} g"
    if field_key == "pricing":
        match = re.search(r"([$€£¥￥])\s?(\d+(?:[.,]\d+)?)(?:\s*(?:/|per)\s*(mo|month|monthly|yr|year|annual|annually))?", raw, re.I)
        if match:
            amount = match.group(2).replace(",", ".")
            period = (match.group(3) or "").lower()
            if period in {"mo", "month", "monthly"}:
                return f"{amount} monthly"
            if period in {"yr", "year", "annual", "annually"}:
                return f"{amount} yearly"
            return amount
    if field_key in {"certification", "security_privacy_deployment"}:
        return re.sub(r"\s+", "", raw.upper())
    return low


def sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    left_candidates = [text.rfind(mark, 0, start) for mark in ".。！？!?\n"]
    right_candidates = [idx for mark in ".。！？!?\n" for idx in [text.find(mark, end)] if idx != -1]
    left = max(left_candidates) + 1 if left_candidates else 0
    right = min(right_candidates) + 1 if right_candidates else min(len(text), end + 220)
    return max(0, left), min(len(text), right)


def is_noise_sentence(sentence: str, value: str) -> bool:
    lower = sentence.lower()
    noise_hits = sum(1 for item in NAVIGATION_NOISE if item.lower() in lower)
    has_numbers_or_standards = bool(re.search(r"\d|ASTM|EN\s?1077|SOC\s?2|GDPR|MIPS|API|SDK", sentence, re.I))
    return noise_hits >= 2 and not has_numbers_or_standards and value.lower() in lower


def append_fact(
    facts: List[Dict[str, Any]],
    seen: set,
    field_key: str,
    field: Mapping[str, Any],
    competitor: str,
    source_url: str,
    source_title: str,
    text: str,
    match: re.Match,
    value: str,
) -> None:
    value = compact_text(value).strip(" .;；,，")
    if not value:
        return
    sent_start, sent_end = sentence_bounds(text, match.start(), match.end())
    evidence = compact_text(text[sent_start:sent_end])
    if is_noise_sentence(evidence, value):
        return
    key = (field_key, normalize_fact_value(value, field_key), source_url)
    if key in seen:
        return
    seen.add(key)
    facts.append(
        {
            "competitor": competitor,
            "dimension": dimension_for_field(field_key),
            "field_key": field_key,
            "field_label": compact_text(field.get("label")) or field_key,
            "value": value,
            "normalized_value": normalize_fact_value(value, field_key),
            "evidence_text": evidence,
            "source_url": source_url,
            "source_title": source_title,
            "confidence": "中信心",
            "needs_verification": "no" if source_url else "yes",
            "extraction_method": EXTRACTION_METHOD,
            "confidence_reason": "命中本品类结构化字段规则，并保留原文句子和来源 URL。",
            "evidence_start": sent_start,
            "evidence_end": sent_end,
            "schema_field_description": compact_text(field.get("description")),
            "fact_id": "",
        }
    )


def extract_structured_facts_from_text(
    competitor: str,
    source_url: str,
    source_title: str,
    text: str,
    schema: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    schema = schema or build_extraction_schema("general")
    source_text = textify(text)
    facts: List[Dict[str, Any]] = []
    seen = set()
    for field_key, field in (schema.get("fields") or {}).items():
        for pattern in field.get("patterns") or []:
            for match in re.finditer(pattern, source_text, re.I):
                if field_key == "weight" and match.lastindex and match.lastindex >= 2:
                    value = f"{match.group(1)} {match.group(2)}"
                else:
                    value = match.group(1) if match.lastindex else match.group(0)
                append_fact(
                    facts,
                    seen,
                    field_key,
                    field,
                    competitor,
                    source_url,
                    source_title,
                    source_text,
                    match,
                    value,
                )
    return facts
