#!/usr/bin/env python3
"""
Competitor Intel Harvester

Pipeline:
1. Query SearXNG for web/news/image results.
2. Crawl selected URLs with Crawl4AI.
3. Extract page text, links, image URLs, and rough PM fields.
4. Download extra keyword images with icrawler.
5. Export Markdown, CSV, and JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import dataclasses
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import ProxyHandler, Request, build_opener

from analysis_templates import select_analysis_template
from filter_training import (
    LocalFilterModel,
    apply_ml_prediction_to_decision,
    bootstrap_filter_model_if_missing,
    load_filter_model,
    model_status,
)
from search_cards import load_search_cards
from source_adapters import (
    adapter_search_templates as source_adapter_search_templates,
    classify_source_url,
    collect_adapter_snapshot,
)
from structured_extractor import (
    build_extraction_schema as build_structured_extraction_schema,
    extract_structured_facts_from_text as schema_extract_structured_facts_from_text,
    normalize_fact_value as schema_normalize_fact_value,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_FILTER_MODEL_PATH = APP_DIR / "models" / "filter_model.pt"
DEFAULT_BOOTSTRAP_LABELS_PATH = APP_DIR / "training_data" / "bootstrap_labels.csv"
DEFAULT_SEARCH_CARDS_DIR = APP_DIR / "search_cards"

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "spm",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


DEFAULT_GENERAL_QUERIES = [
    "{name} official website product features pricing",
    "{name} official site",
    "{name} app official website pricing",
    "{name} AI official website pricing",
    "{name} competitors alternatives product positioning",
    "{name} pricing plans features",
    "{name} launch funding product update",
]

DEFAULT_CN_QUERIES = [
    "{name} 官网 产品 功能 定价",
    "{name} 竞品 替代品 对比",
    "{name} 产品 更新 发布 融资",
]

DEFAULT_IMAGE_QUERIES = [
    "{name} product screenshots",
    "{name} app screenshots",
    "{name} UI screenshots",
    "{name} logo product images",
]

CHINESE_EXPORT_ALIASES = {
    "all_sources.csv": "所有采集来源.csv",
    "unfiltered_collection.md": "未经筛选的采集内容.md",
    "filtered_collection.md": "筛选后的采集内容.md",
    "final_analysis_embedded.md": "竞品分析报告_图片内嵌版.md",
    "collection_principles.md": "采集原则和筛选原则.md",
    "screening_strategy.md": "收录过滤策略设计.md",
    "ml_filter_status.json": "本地筛选模型状态.json",
    "filter_weights.json": "本地筛选模型权重.json",
    "problem_pages_review.md": "问题页面核验清单.md",
    "problem_pages_review.csv": "问题页面核验清单.csv",
    "manual_review_queue.md": "人工复核队列.md",
    "manual_review_queue.csv": "人工复核队列.csv",
    "login_required_queue.md": "需登录队列.md",
    "login_required_queue.csv": "需登录队列.csv",
    "training_review_sample.md": "人工抽样标注表.md",
    "training_review_sample.csv": "人工抽样标注表.csv",
    "competitor_discovery.md": "自动竞品发现.md",
    "competitor_discovery.csv": "自动竞品发现.csv",
    "gui_review_results.md": "GUI自动复核结果.md",
    "gui_review_results.csv": "GUI自动复核结果.csv",
    "structured_facts.csv": "结构化事实.csv",
    "fact_clusters.csv": "事实聚类.csv",
    "fact_clusters.md": "事实聚类.md",
    "anti_bot_strategy.md": "反爬处理策略.md",
    "codex_decisions.csv": "Codex收录决策.csv",
    "codex_review.json": "Codex结构化结果.json",
    "codex_input.md": "Codex分析输入证据.md",
    "pre_crawl_ai_strategy.json": "抓取前AI策略.json",
    "pre_crawl_ai_strategy.md": "抓取前AI策略.md",
    "pre_crawl_plan.md": "抓取前采集计划.md",
    "pre_crawl_plan.json": "抓取前采集计划.json",
    "competitors.csv": "竞品汇总.csv",
    "pages.csv": "页面抓取结果.csv",
    "images.csv": "图片索引.csv",
    "search_results.csv": "搜索结果.csv",
    "evidence_audit.csv": "证据筛选审计.csv",
    "raw.json": "原始数据.json",
    "run.log": "运行日志.log",
    "codex_run.log": "Codex运行日志.log",
}

INTERNAL_OUTPUT_DIR_NAME = "_internal"
EXTRA_BIN_DIRS = [
    Path.home() / ".local" / "bin",
    Path.home() / ".codex" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
]

ROOT_OUTPUT_FILES = {
    "实验计时记录.md",
    "抓取前采集计划.md",
    "采集原则和筛选原则.md",
    "所有采集来源.csv",
    "未经筛选的采集内容.md",
    "筛选后的采集内容.md",
    "竞品分析报告_图片内嵌版.md",
    "问题页面核验清单.csv",
    "自动竞品发现.csv",
    "结构化事实.csv",
    "事实聚类.csv",
    "人工抽样标注表.csv",
    "downloaded_images",
    "gui_review_snapshots",
}

def expanded_path_env() -> str:
    current = os.environ.get("PATH", "")
    extras = [str(path) for path in EXTRA_BIN_DIRS if path.exists()]
    return os.pathsep.join([*extras, current]) if extras else current


def resolve_executable_command(command: str) -> str:
    command = (command or "").strip() or "codex"
    expanded = Path(command).expanduser()
    if (os.sep in command or command.startswith("~")) and expanded.exists():
        return str(expanded)
    found = shutil.which(command, path=expanded_path_env())
    if found:
        return found
    if command == "codex":
        for directory in EXTRA_BIN_DIRS:
            candidate = directory / "codex"
            if candidate.exists():
                return str(candidate)
    return ""


FINAL_REPORT_FRAMEWORK = [
    ("0. 核心结论与决策建议", "直接回答 PM 应该关注什么、优先验证什么、下一步怎么做。"),
    ("1. 采集范围、来源与证据等级", "列出所有关键来源类型、证据强弱、Fact/Inference/Assumption 边界。"),
    ("2. 竞品概览与定位", "公司/产品简介、品类、目标市场、核心定位、主张、主要 URL。"),
    ("3. 产品能力与工作流对比", "按用户任务/JTBD 拆解核心能力、关键流程、AI/技术能力、集成和生态。"),
    ("4. 定价、套餐与商业化包装", "价格、套餐、免费/试用、额度、限制、Add-on、企业版和包装逻辑。"),
    ("5. 目标用户、市场与 GTM", "ICP、行业/地域、渠道、销售动作、合作伙伴、内容/PLG/企业销售线索。"),
    ("6. 客户体验、服务支持与产品质量", "上手路径、文档/支持、服务承诺、质量信号、用户评价或缺口。"),
    ("7. 视觉与产品证据", "把必要截图/图片放到对应分析位置，说明它证明了什么，不做图库堆砌。"),
    ("8. 牵引、更新节奏与战略方向", "用户数、ARR/融资/客户、发布节奏、招聘/路线图/战略动作。"),
    ("9. SWOT 与风险机会", "每个竞品的强弱项、机会、威胁，并给出对我方的含义。"),
    ("10. 横向对比矩阵", "用同一字段横向比较能力、价格、用户、GTM、证据质量和风险。"),
    ("11. 信息缺口与问题页面核验", "列出反爬、正文不足、登录超时、视频缺时间点和待核实来源，以及为什么值得或不值得补证。"),
    ("12. 建议下一步与监控计划", "沉淀监控清单、补采动作、后续分析/战卡/定价跟踪建议。"),
    ("13. 我方产品方向分析", "结合用户填写的我方产品定位，给出差异化方向、优先验证假设和路线建议；未填写时只列信息缺口。"),
]

FINAL_REPORT_FRAMEWORK_TEXT = "\n".join(
    f"- {title}: {description}" for title, description in FINAL_REPORT_FRAMEWORK
)

PAGE_PRIORITY_KEYWORDS = [
    "pricing",
    "price",
    "plans",
    "features",
    "product",
    "solutions",
    "customers",
    "blog",
    "changelog",
    "release",
    "news",
    "about",
]

GENERIC_COMPETITOR_TERMS = {
    "ai",
    "app",
    "apps",
    "map",
    "maps",
    "tool",
    "tools",
    "system",
    "platform",
    "software",
    "service",
    "services",
    "product",
    "api",
    "sdk",
    "地图",
    "导航",
    "系统",
    "平台",
    "软件",
    "服务",
    "产品",
}

NOISE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "youtube.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "wikipedia.org",
}

VIDEO_SOCIAL_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "douyin.com",
    "tiktok.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "xiaohongshu.com",
    "weixin.qq.com",
}

LOGIN_ASSIST_PLATFORM_TARGETS = {
    "douyin.com": {
        "login_url": "https://www.douyin.com/",
        "aliases": {"douyin.com", "www.douyin.com"},
    },
    "xiaohongshu.com": {
        "login_url": "https://www.xiaohongshu.com/explore",
        "aliases": {"xiaohongshu.com", "www.xiaohongshu.com", "xhslink.com"},
    },
    "tiktok.com": {
        "login_url": "https://www.tiktok.com/login",
        "aliases": {"tiktok.com", "www.tiktok.com"},
    },
    "instagram.com": {
        "login_url": "https://www.instagram.com/accounts/login/",
        "aliases": {"instagram.com", "www.instagram.com"},
    },
    "x.com": {
        "login_url": "https://x.com/i/flow/login",
        "aliases": {"x.com", "www.x.com"},
    },
    "twitter.com": {
        "login_url": "https://twitter.com/i/flow/login",
        "aliases": {"twitter.com", "www.twitter.com", "mobile.twitter.com"},
    },
    "zhihu.com": {
        "login_url": "https://www.zhihu.com/signin",
        "aliases": {"zhihu.com", "www.zhihu.com", "zhuanlan.zhihu.com"},
    },
    "reddit.com": {
        "login_url": "https://www.reddit.com/login/",
        "aliases": {"reddit.com", "www.reddit.com", "old.reddit.com"},
    },
    "bilibili.com": {
        "login_url": "https://passport.bilibili.com/login",
        "aliases": {"bilibili.com", "www.bilibili.com", "m.bilibili.com", "b23.tv"},
    },
    "weixin.qq.com": {
        "login_url": "https://mp.weixin.qq.com/",
        "aliases": {"mp.weixin.qq.com", "weixin.qq.com", "channels.weixin.qq.com", "web.wechat.com"},
    },
}

LOGIN_ASSIST_PLATFORM_ALIASES = {
    alias.lower().removeprefix("www."): canonical
    for canonical, config in LOGIN_ASSIST_PLATFORM_TARGETS.items()
    for alias in config["aliases"]
}

FORUM_COMMUNITY_DOMAINS = {
    "reddit.com",
    "zhihu.com",
    "v2ex.com",
    "news.ycombinator.com",
    "tieba.baidu.com",
}

APP_STORE_DOMAINS = {
    "apps.apple.com",
    "play.google.com",
    "chromewebstore.google.com",
    "microsoftedge.microsoft.com",
    "apps.shopify.com",
}

THIRD_PARTY_CRAWL_SKIP_DOMAINS = {
    "trustpilot.com",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
    "sourceforge.net",
    "csdn.net",
    "zhihu.com",
    "toutiao.com",
    "bilibili.com",
    "baijiahao.baidu.com",
    "qq.com",
    "163.com",
    "readhub.cn",
    "php.cn",
    "wishdown.com",
    "softwaresuggest.com",
    "amazon.com",
    "amazon.cn",
    "amazon.co.uk",
    "amazon.in",
    "ebay.com",
    "ebay.co.uk",
    "aliexpress.com",
    "mercadolivre.com.br",
    "walmart.com",
    "target.com",
}

LOW_VALUE_CRAWL_SKIP_DOMAINS = {
    "36dianping.com",
    "36kr.com",
    "4399.com",
    "aig123.com",
    "aigc.cn",
    "aieva.cn",
    "ai-bot.cn",
    "ainavpro.cn",
    "airukou.cn",
    "baike.baidu.com",
    "chinaz.com",
    "cloud.tencent.com",
    "cnblogs.com",
    "donews.com",
    "dongaigc.com",
    "douyin.com",
    "doyo.cn",
    "feizhuke.com",
    "gebidh.com",
    "guba.sina.com.cn",
    "guidegreat.cn",
    "jingyan.baidu.com",
    "jingzhunlink.com",
    "juejin.cn",
    "k.sina.com.cn",
    "maigoo.com",
    "m.guofenkong.com",
    "navtool.cn",
    "pooban.com",
    "sohu.com",
    "szxn.com",
    "tianyancha.com",
    "tool.p2hp.com",
    "tvmao.com",
    "wps.com",
    "xueqiu.com",
    "zhidao.baidu.com",
}

HIGH_VALUE_PUBLIC_CRAWL_DOMAINS = {
    "24slides.com",
    "businesswire.com",
    "producthunt.com",
    "prnewswire.com",
    "stripe.com",
    "techcrunch.com",
    "thegeneralist.com",
}

OFFICIAL_CORE_PATH_KEYWORDS = {
    "pricing",
    "plans",
    "price",
    "features",
    "product",
    "products",
    "specs",
    "specifications",
    "technology",
    "technologies",
    "materials",
    "material",
    "size",
    "size-guide",
    "size-chart",
    "solutions",
    "use-cases",
    "customers",
    "case-studies",
    "docs",
    "help",
    "support",
    "developers",
    "developer",
    "api",
    "integrations",
    "security",
    "trust",
    "legal",
    "privacy",
    "terms",
    "changelog",
    "release",
    "releases",
    "about",
    "定价",
    "价格",
    "套餐",
    "产品",
    "功能",
    "参数",
    "规格",
    "材质",
    "尺码",
    "尺寸",
    "颜色",
    "技术",
    "解决方案",
    "客户",
    "案例",
    "文档",
    "帮助",
    "开发者",
    "接口",
    "安全",
    "信任",
    "隐私",
    "条款",
    "更新",
    "发布",
    "关于",
}

OFFICIAL_CORE_PATHS = [
    "",
    "pricing",
    "features",
    "product",
    "products",
    "specs",
    "specifications",
    "technology",
    "materials",
    "size-guide",
    "size-chart",
    "solutions",
    "customers",
    "case-studies",
    "enterprise",
    "teams",
    "templates",
    "docs",
    "api",
    "help",
    "security",
    "privacy",
    "terms",
    "changelog",
    "blog",
]

OFFICIAL_PATH_PRIORITY = {
    "pricing": 0,
    "plans": 0,
    "": 1,
    "features": 2,
    "product": 3,
    "products": 3,
    "specs": 3,
    "specifications": 3,
    "technology": 4,
    "technologies": 4,
    "materials": 4,
    "size-guide": 4,
    "size-chart": 4,
    "solutions": 4,
    "customers": 5,
    "case-studies": 5,
    "docs": 6,
    "api": 6,
    "developers": 6,
    "help": 7,
    "security": 8,
    "trust": 8,
    "privacy": 9,
    "terms": 9,
    "changelog": 10,
    "blog": 11,
}

OFFICIAL_DISCOVERY_HINTS = {
    "official",
    "official website",
    "official site",
    "website",
    "homepage",
    "官网",
    "官方网站",
    "官网入口",
    "访问官网",
    "打开网站",
    "链接直达",
}

COMMON_DOMAIN_PARTS = {
    "www",
    "app",
    "apps",
    "ai",
    "co",
    "com",
    "cn",
    "io",
    "net",
    "org",
    "site",
    "so",
    "tech",
    "tools",
    "www2",
}

URL_IN_TEXT_RE = re.compile(r"https?://[A-Za-z0-9][A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*", re.I)

LOW_VALUE_DOMAINS = {
    "login.taobao.com",
    "passport.taobao.com",
    "cart.taobao.com",
    "i.taobao.com",
    "myseller.taobao.com",
    "seller.taobao.com",
    "bbs.taobao.com",
    "jianghu.taobao.com",
}

LOW_VALUE_PORTAL_DOMAINS = {
    "taobao.com",
    "tmall.com",
    "1688.com",
}

LOW_VALUE_URL_TOKENS = {
    "login",
    "log-in",
    "signin",
    "sign-in",
    "signup",
    "sign-up",
    "register",
    "auth",
    "oauth",
    "passport",
    "account",
    "member",
    "usercenter",
    "history",
    "search",
    "cart",
    "checkout",
    "basket",
    "wishlist",
    "forum",
    "bbs",
    "jianghu",
    "taojianghu",
    "reviews",
    "review",
    "alternatives",
}

LOGIN_POOL_EXCLUDE_HINTS = {
    "career",
    "careers",
    "job",
    "jobs",
    "hiring",
    "recruit",
    "recruitment",
    "campus",
    "talent",
    "mokahr",
    "greenhouse",
    "lever.co",
    "workday",
    "职位",
    "招聘",
    "校园招聘",
    "社招",
    "人才",
    "投递",
    "简历",
}

BOILERPLATE_PHRASES = {
    "亲，请登录",
    "亲请登录",
    "免费注册",
    "淘宝网首页",
    "已买到的宝贝",
    "我的淘宝",
    "我的足迹",
    "我的卡券包",
    "购物车",
    "收藏夹",
    "收藏的宝贝",
    "收藏的店铺",
    "免费开店",
    "淘宝开店",
    "天猫开店",
    "开直播店",
    "千牛卖家中心",
    "开店入驻",
    "已卖出的宝贝",
    "出售中的宝贝",
    "卖家服务市场",
    "卖家培训中心",
    "体检中心",
    "电商学习中心",
    "帮助中心",
    "官方客服",
    "商家客服",
    "消息中心",
    "意见反馈",
    "举报中心",
    "淘宝规则",
    "平台规则",
    "首页",
    "精选内容",
    "发布于淘江湖",
}

FORUM_NAV_PHRASES = {
    "茶馆",
    "闲唠八卦",
    "热点聚焦",
    "生活游记",
    "AI工具",
    "视频专区",
    "话题PK",
    "生意经",
    "淘宝教育",
    "淘宝问答",
    "聚财心法",
    "江湖反馈",
    "资产拍卖",
    "兴趣经验",
    "实用经验",
    "美食分享",
    "游戏交流",
    "黑板报",
    "种草笔记",
    "赚淘金币",
    "兑红包",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"亲[，,]?\s*请登录", re.I),
    re.compile(r"\b(log\s*in|sign\s*in|sign\s*up|create\s+account|shopping\s+cart|wishlist)\b", re.I),
    re.compile(r"(登录|注册|购物车|收藏夹|卖家中心|帮助中心|官方客服|商家客服)"),
]

BROKEN_PAGE_PATTERNS = [
    re.compile(r"\b404\b", re.I),
    re.compile(r"\b(page|route|resource)\s+(not\s+found|doesn['’]?t\s+exist)\b", re.I),
    re.compile(r"(something\s+went\s+wrong|you'?ve\s+wandered\s+off\s+the\s+map|oeps!?\s+er\s+ging\s+iets\s+mis)", re.I),
    re.compile(r"(页面不存在|找不到页面|出错了|访问的页面不存在)"),
]

ANTIBOT_ERROR_MARKERS = (
    "anti-bot",
    "antibot",
    "cloudflare",
    "datadome",
    "captcha",
    "http 403",
    "forbidden",
    "js challenge",
    "script_heavy_shell",
    "minimal_text",
    "acs-goto",
    "connection refused",
)

LOGIN_ASSIST_TERMS = (
    "login",
    "log in",
    "sign in",
    "signin",
    "sign-in",
    "signup",
    "sign up",
    "sign-up",
    "register",
    "registration",
    "auth",
    "oauth",
    "passport",
    "account",
    "password",
    "forgot password",
    "verification code",
    "captcha",
    "登录",
    "登陆",
    "注册",
    "账号",
    "账户",
    "密码",
    "验证码",
    "请登录",
)

LOGIN_FORM_TERMS = (
    "password",
    "forgot password",
    "verification code",
    "captcha",
    "one-time code",
    "请输入",
    "手机号",
    "密码",
    "验证码",
    "登录后",
    "注册账号",
)

AUTH_GATE_HOST_LABELS = {
    "account",
    "accounts",
    "auth",
    "id",
    "login",
    "member",
    "oauth",
    "passport",
    "sso",
}

AUTH_GATE_URL_TOKENS = (
    "login",
    "log-in",
    "signin",
    "sign-in",
    "signup",
    "sign-up",
    "register",
    "registration",
    "account",
    "accounts",
    "auth",
    "oauth",
    "passport",
    "member",
    "usercenter",
    "sso",
)

AUTH_GATE_URL_NOISE_TOKENS = {
    "article",
    "articles",
    "author",
    "authors",
    "blog",
    "blogs",
    "download",
    "downloads",
    "faq",
    "faqs",
    "guide",
    "guides",
    "news",
    "post",
    "posts",
    "press",
    "review",
    "reviews",
    "story",
    "stories",
    "alternatives",
    "comparison",
}

AUTH_GATE_TITLE_NOISE_TERMS = (
    "review",
    "reviews",
    "download",
    "downloads",
    "faq",
    "faqs",
    "guide",
    "guides",
    "article",
    "articles",
    "author",
    "authors",
    "blog",
    "blogs",
    "news",
    "comparison",
    "alternatives",
    "experience",
    "评测",
    "评价",
    "下载",
    "教程",
    "指南",
    "作者",
    "新闻",
    "文章",
)

AUTH_LOGIN_MARKER_RE = re.compile(
    r"(?<![a-z0-9])("
    r"log\s*in|sign\s*in|signin|sign-in|"
    r"sign\s*up|signup|sign-up|login|register|registration|"
    r"oauth|passport|account|accounts|password|captcha"
    r")(?![a-z0-9])",
    re.I,
)

AUTH_FORM_FIELD_TERMS = (
    "password",
    "forgot password",
    "verification code",
    "captcha",
    "one-time code",
    "email address",
    "手机号",
    "密码",
    "验证码",
    "登录后",
    "注册账号",
)

LOGIN_ASSIST_VALUE_TERMS = (
    "pricing",
    "plans",
    "features",
    "product",
    "docs",
    "documentation",
    "api",
    "integrations",
    "security",
    "customers",
    "specs",
    "parameter",
    "定价",
    "价格",
    "套餐",
    "功能",
    "产品",
    "文档",
    "接口",
    "参数",
    "规格",
    "客户",
)

MANUAL_REVIEW_FIELDS = [
    "competitor",
    "priority",
    "review_reason",
    "requires_user_login",
    "title",
    "url",
    "domain",
    "crawl_error",
    "cleaned_excerpt_sample",
    "gui_review_url",
    "login_assist_url",
    "queued_urls",
    "suggested_next_step",
    "allowed_boundary",
]

GUI_REVIEW_FIELDS = [
    "competitor",
    "priority",
    "review_reason",
    "requires_user_login",
    "title",
    "url",
    "domain",
    "adapter_name",
    "source_family",
    "platform",
    "canonical_url",
    "automated_review_status",
    "text_snapshot_path",
    "screenshot_path",
    "metadata_path",
    "transcript_path",
    "evidence_markers_path",
    "needs_manual_video_timestamp",
    "login_assist_url",
    "text_snapshot_excerpt",
    "allowed_boundary",
    "next_step",
]

LOGIN_REQUIRED_FIELDS = [
    "competitor",
    "priority",
    "review_reason",
    "title",
    "url",
    "domain",
    "queued_url_count",
    "login_assist_url",
    "queued_urls",
    "automated_review_status",
    "text_snapshot_path",
    "screenshot_path",
    "text_snapshot_excerpt",
    "next_step",
    "allowed_boundary",
]

PROBLEM_REVIEW_FIELDS = [
    "competitor",
    "priority",
    "problem_type",
    "source_queue",
    "title",
    "url",
    "domain",
    "status",
    "source_kind",
    "page_role",
    "source_policy_tier",
    "pending_verification",
    "verification_reason",
    "fact_type",
    "increment_type",
    "fact_group",
    "reason",
    "what_to_verify",
    "data_entry_decision",
    "suggested_human_label",
    "human_label",
    "human_reason",
    "use_as_primary_evidence",
    "reviewed_by",
    "reviewed_at",
    "model_feedback_status",
    "text_snapshot_path",
    "screenshot_path",
    "metadata_path",
    "evidence_markers_path",
    "allowed_boundary",
]


@dataclasses.dataclass
class SearchResult:
    competitor: str
    category: str
    query: str
    title: str
    url: str
    snippet: str
    engine: str = ""
    score: float = 0.0


@dataclasses.dataclass
class PageExtract:
    competitor: str
    url: str
    title: str
    markdown: str
    text_excerpt: str
    links: List[str]
    image_urls: List[str]
    fields: Dict[str, str]
    error: str = ""


@dataclasses.dataclass
class ProductCollectionField:
    key: str
    label: str
    description: str
    patterns: List[str]
    search_terms: List[str]


@dataclasses.dataclass
class SourceStrategyItem:
    name: str
    priority: str
    source_examples: List[str]
    selection_rule: str
    retrieval_rule: str
    traceability_rule: str
    evidence_role: str
    escalation_rule: str = ""
    legal_boundary: str = ""


@dataclasses.dataclass
class CompetitorDiscoveryStrategy:
    name: str
    stage: str
    trigger: str
    discovery_method: str
    acceptance_rule: str
    traceability_rule: str
    rejection_rule: str = ""


@dataclasses.dataclass
class ValueJudgmentRule:
    key: str
    label: str
    positive_rule: str
    negative_rule: str


@dataclasses.dataclass
class ProductCollectionPlan:
    category: str
    category_label: str
    rationale: str
    fields: List[ProductCollectionField]
    search_templates: List[str]
    cn_search_templates: List[str]
    image_terms: List[str]
    report_focus: List[str]
    generated_search_terms: List[str] = dataclasses.field(default_factory=list)
    fixed_exclude_keywords: List[str] = dataclasses.field(default_factory=list)
    dynamic_exclude_keywords: List[str] = dataclasses.field(default_factory=list)
    evidence_keywords: List[str] = dataclasses.field(default_factory=list)
    exclude_keywords: List[str] = dataclasses.field(default_factory=list)
    source_policy_notes: List[str] = dataclasses.field(default_factory=list)
    search_term_reasons: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    source_strategies: List[SourceStrategyItem] = dataclasses.field(default_factory=list)
    competitor_discovery_strategies: List[CompetitorDiscoveryStrategy] = dataclasses.field(default_factory=list)
    directed_source_search_templates: List[str] = dataclasses.field(default_factory=list)
    value_judgment_rules: List[ValueJudgmentRule] = dataclasses.field(default_factory=list)
    search_cards_applied: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    analysis_template_key: str = ""
    analysis_template_label: str = ""
    analysis_template_path: str = ""
    analysis_template_summary: str = ""
    analysis_template_match_score: int = 0
    analysis_dimensions: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    analysis_report_outline: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CompetitorCandidate:
    name: str
    candidate_type: str
    confidence: str
    status: str
    official_url: str
    official_domain: str
    discovered_query: str
    discovered_from_url: str
    evidence_title: str
    evidence_snippet: str
    overlap_reason: str
    source_count: int = 1


@dataclasses.dataclass
class EvidenceDecision:
    source_kind: str
    page_role: str
    decision_status: str
    gate_result: str
    hard_gate: str
    relevance_score: int
    evidence_score: int
    pm_value_score: int
    traceability_score: int
    category_fit_score: int
    confidence: str
    matched_fields: List[str]
    matched_include_keywords: List[str]
    matched_exclude_keywords: List[str]
    rejection_code: str
    reasons: List[str]
    pending_verification: bool = False
    verification_reason: str = ""
    source_policy_tier: str = ""
    fact_type: str = ""
    increment_type: str = ""
    fact_group: str = ""
    primary_evidence_candidate: bool = False
    primary_evidence_reason: str = ""
    ml_label: str = ""
    ml_include_score: str = ""
    ml_exclude_score: str = ""
    ml_verify_later_score: str = ""
    ml_confidence: str = ""
    ml_reason: str = ""
    ml_model_version: str = ""
    ml_adjustment: str = ""
    value_signals: List[str] = dataclasses.field(default_factory=list)
    value_missing: List[str] = dataclasses.field(default_factory=list)
    value_verdict: str = ""
    gui_review_candidate: bool = False
    gui_review_value_reason: str = ""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "competitor"


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.netloc in {"www.google.com", "google.com"} and parsed.path == "/url":
        q = parse_qs(parsed.query).get("q", [""])[0]
        if q:
            url = q
            parsed = urlparse(url)
    if not parsed.scheme and parsed.netloc:
        parsed = parsed._replace(scheme="https")
    elif not parsed.scheme and parsed.path:
        parsed = urlparse("https://" + url)
    if parsed.path == "/":
        parsed = parsed._replace(path="")
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def canonical_url_for_dedupe(url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    path = re.sub(r"/+$", "", parsed.path or "")
    filtered_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        low_key = key.lower()
        if low_key in TRACKING_QUERY_PARAMS or low_key.startswith("utm_"):
            continue
        filtered_query.append((key, value))
    query = urlencode(sorted(filtered_query))
    return urlunparse(
        parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=path,
            query=query,
            fragment="",
        )
    )


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


LOGIN_QUEUE_MULTI_LABEL_SUFFIXES = {
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "com.tw",
    "co.jp",
    "co.kr",
    "net.cn",
    "org.cn",
}

LOGIN_QUEUE_RESERVED_TEST_SUFFIXES = {"example.com", "example.net", "example.org"}


def site_domain_of(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host or host == "localhost" or re.fullmatch(r"\d+(?:\.\d+){3}", host):
        return host
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    suffix = ".".join(labels[-2:])
    if suffix in LOGIN_QUEUE_RESERVED_TEST_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    if suffix in LOGIN_QUEUE_MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def login_assist_platform_key_for_url_or_domain(value: str) -> str:
    raw = textify(value).strip().lower()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"https://{raw.lstrip('*.')}"
    host = (urlparse(candidate).hostname or "").lower().strip(".")
    if not host:
        return ""
    normalized_host = host.removeprefix("www.")
    if normalized_host in LOGIN_ASSIST_PLATFORM_ALIASES:
        return LOGIN_ASSIST_PLATFORM_ALIASES[normalized_host]
    domain = site_domain_of(f"https://{host}")
    if domain in LOGIN_ASSIST_PLATFORM_ALIASES:
        return LOGIN_ASSIST_PLATFORM_ALIASES[domain]
    if domain in LOGIN_ASSIST_PLATFORM_TARGETS:
        return domain
    return ""


def login_queue_domain_key_for_url(url: str) -> str:
    platform_key = login_assist_platform_key_for_url_or_domain(url)
    if platform_key:
        return platform_key
    return site_domain_of(url)


def normalize_login_queue_keyword(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def login_queue_key_for_values(competitor: str, url: str) -> Tuple[str, str]:
    site_domain = login_queue_domain_key_for_url(url)
    stable_target = site_domain or canonical_url_for_dedupe(url) or url
    return normalize_login_queue_keyword(competitor), stable_target.strip().lower()


def domain_matches(domain: str, domains: Iterable[str]) -> bool:
    return domain in domains or any(domain.endswith("." + item) for item in domains)


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    for attr in ("raw_markdown", "markdown", "fit_markdown", "markdown_with_citations"):
        nested = getattr(value, attr, None)
        if nested and nested is not value:
            text = textify(nested)
            if text:
                return text
    if isinstance(value, (list, tuple, set)):
        return "\n".join(textify(item) for item in value if item is not None)
    return str(value)


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return textify(value)


def field_spec(
    key: str,
    label: str,
    description: str,
    patterns: Sequence[str],
    search_terms: Sequence[str],
) -> ProductCollectionField:
    return ProductCollectionField(
        key=key,
        label=label,
        description=description,
        patterns=list(patterns),
        search_terms=list(search_terms),
    )


def source_strategy(
    name: str,
    priority: str,
    source_examples: Sequence[str],
    selection_rule: str,
    retrieval_rule: str,
    traceability_rule: str,
    evidence_role: str,
    escalation_rule: str = "",
    legal_boundary: str = "",
) -> SourceStrategyItem:
    return SourceStrategyItem(
        name=name,
        priority=priority,
        source_examples=list(source_examples),
        selection_rule=selection_rule,
        retrieval_rule=retrieval_rule,
        traceability_rule=traceability_rule,
        evidence_role=evidence_role,
        escalation_rule=escalation_rule,
        legal_boundary=legal_boundary,
    )


def competitor_discovery_strategy(
    name: str,
    stage: str,
    trigger: str,
    discovery_method: str,
    acceptance_rule: str,
    traceability_rule: str,
    rejection_rule: str = "",
) -> CompetitorDiscoveryStrategy:
    return CompetitorDiscoveryStrategy(
        name=name,
        stage=stage,
        trigger=trigger,
        discovery_method=discovery_method,
        acceptance_rule=acceptance_rule,
        traceability_rule=traceability_rule,
        rejection_rule=rejection_rule,
    )


def value_judgment_rule(
    key: str,
    label: str,
    positive_rule: str,
    negative_rule: str,
) -> ValueJudgmentRule:
    return ValueJudgmentRule(
        key=key,
        label=label,
        positive_rule=positive_rule,
        negative_rule=negative_rule,
    )


def build_value_judgment_rules() -> List[ValueJudgmentRule]:
    return [
        value_judgment_rule(
            "competitor_binding",
            "竞品绑定",
            "明确出现竞品名、官网域名、官方账号、产品型号、App 名或品牌别名。",
            "只出现泛品类词，看不出和目标竞品有关。",
        ),
        value_judgment_rule(
            "decision_relevance",
            "决策相关",
            "能回答功能、定价、参数、质量、目标用户、GTM、渠道、安全、集成、更新、用户痛点等问题。",
            "只有泛泛介绍、营销口号、排行榜堆砌。",
        ),
        value_judgment_rule(
            "information_increment",
            "信息增量",
            "提供此前没有的价格、参数、截图、流程、限制、评价、时间点、版本变化或竞品关系。",
            "只是重复已有官网信息，没有新事实或新视角。",
        ),
        value_judgment_rule(
            "source_credibility",
            "来源可信",
            "来自官网、官方账号、权威平台、垂直社区、真实用户讨论，或能回到原始出处。",
            "二创搬运、SEO 聚合、无来源总结、广告软文。",
        ),
        value_judgment_rule(
            "traceability",
            "可追溯",
            "能留下 URL、标题、发布时间、平台、作者展示名、视频时间点或截图。",
            "无链接、无时间、无出处、截图来源不明。",
        ),
        value_judgment_rule(
            "public_access",
            "可获取",
            "公开可访问，自动抓取或 GUI 复核都不需要登录、付费或绕过限制。",
            "需要登录、付费、验证码、私域权限或未授权接口。",
        ),
    ]


def build_directed_source_search_templates(category: str) -> List[str]:
    templates = [
        "{name} official website pricing docs",
        "{name} official product features changelog",
        "{name} site:producthunt.com",
        "{name} site:g2.com review",
        "{name} site:capterra.com review",
        "{name} site:apps.apple.com",
        "{name} site:play.google.com",
        "{name} site:reddit.com review",
        "{name} site:reddit.com problem",
        "{name} site:zhihu.com 评价",
        "{name} site:v2ex.com",
        "{name} site:youtube.com demo",
        "{name} site:youtube.com review",
        "{name} site:bilibili.com 评测",
        "{name} site:x.com product",
        "{name} site:tiktok.com review",
        "{name} site:douyin.com 评测",
        "{name} site:xiaohongshu.com 评价",
        "{name} site:weixin.qq.com",
    ]
    if category == "ai_software":
        templates.extend(
            [
                "{name} site:github.com API SDK",
                "{name} site:npmjs.com package",
                "{name} integrations marketplace",
                "{name} docs API rate limits",
            ]
        )
    elif category == "autonomous_vehicle_robotaxi":
        templates.extend(
            [
                "{name} official robotaxi autonomous vehicle specs",
                "{name} official safety report autonomous driving",
                "{name} official city operation service area fare",
                "{name} vehicle platform sensor compute lidar camera radar",
                "{name} regulator permit autonomous vehicle city",
                "{name} robotaxi ride experience waiting time review",
                "{name} site:youtube.com robotaxi ride review",
                "{name} site:bilibili.com 无人车 试乘 评测",
                "{name} site:douyin.com 无人车 试乘",
                "{name} site:xiaohongshu.com 无人车 体验",
                "{name} site:zhihu.com 无人车 评价",
                "{name} site:weixin.qq.com 无人车 运营 安全",
            ]
        )
    elif category in {"physical_product", "snow_helmet"}:
        templates.extend(
            [
                "{name} official manual specifications",
                "{name} official size chart warranty",
                "{name} certification safety standard",
                "{name} review durability quality",
            ]
        )
    templates.extend(source_adapter_search_templates("{name}", category))
    return unique_strings(templates)


def build_search_query_templates(
    plan: ProductCollectionPlan,
    manual_search_terms: Iterable[str] = (),
    include_cn: bool = True,
) -> List[str]:
    default_general_queries = DEFAULT_GENERAL_QUERIES
    if plan.category != "ai_software":
        default_general_queries = [
            template for template in DEFAULT_GENERAL_QUERIES if " AI " not in template
        ]
    return unique_strings(
        [
            *[f"{{name}} {term}" for term in normalize_keyword_inputs(manual_search_terms)],
            *plan.search_templates,
            *plan.directed_source_search_templates,
            *(plan.cn_search_templates if include_cn else []),
            *[f"{{name}} {term}" for term in plan.generated_search_terms[:40]],
            *default_general_queries,
        ]
    )


def build_source_strategies(category: str) -> List[SourceStrategyItem]:
    strategies = [
        source_strategy(
            "竞品官方来源",
            "P0",
            ["官网首页", "产品/功能页", "定价页", "官方文档/帮助中心", "API/开发者页", "安全/隐私/更新日志"],
            "先确认域名是否属于该竞品，再抓产品事实；官方页优先承载参数、价格、套餐、文档、接口、合规和发布时间。",
            "优先用 Crawl4AI 抓 HTML；识别官网后扩展 pricing、features、product、docs、api、security、changelog 等核心路径。",
            "记录竞品名、官方域名、完整 URL、页面标题、抓取时间和命中的页面角色；GUI 补证时保留公开页面 URL 和截图。",
            "可作为 Fact 的主证据；若是官方声称但未被使用体验验证，报告中写成“官方声称”。",
            "官方核心页遇到 403、JS 空壳或正文不足时进入问题页面核验清单，优先找同站 sitemap、帮助中心、公开静态页或人工公开摘录补证。",
            "只处理公开可访问内容，不破解验证码、登录、付费墙或访问控制。",
        ),
        source_strategy(
            "垂直品类与应用商店",
            "P1",
            ["App Store/Google Play/Chrome Web Store", "Product Hunt", "G2/Capterra", "行业测评站", "电商官方旗舰店或品牌授权页"],
            "只选择与本轮品类、目标用户和应采字段直接相关的垂直来源；平台评分、评论、榜单和类目只做验证，不覆盖官网事实。",
            "先通过搜索定位公开页面，再抓可读文本；应用商店、扩展商店和测评页保留评分、版本、评论摘要、截图和发布日期。",
            "记录平台名、页面 URL、标题、发布日期/版本号、评分或评论数量；无法确认原始来源的二次整理进入待核实。",
            "用于验证用户感知、市场热度、发布节奏、截图和质量反馈。",
            "若平台反爬但公开页有价值，进入 GUI 复核；只截取公开可见文本、评分、截图和视频时间点。",
            "不抓取登录后评论、不购买报告、不调用未授权平台接口。",
        ),
        source_strategy(
            "论坛与社区",
            "P2",
            ["Reddit", "知乎", "V2EX", "Hacker News", "贴吧/专业论坛", "小红书公开笔记评论"],
            "只有当帖子明确绑定竞品名、使用场景、问题、质量、价格或替代选择时才保留；泛闲聊和导航页降权。",
            "优先抓公开帖子正文、标题、发布时间和楼层摘要；社区内容默认不进入主事实，只进入口碑/问题线索。",
            "记录帖子 URL、平台、作者展示名、发布时间、楼层或评论位置；二次转述但无法找到原帖时列入待核实名单。",
            "用于发现痛点、购买顾虑、质量问题、使用场景和用户语言。",
            "登录墙、私域群、无公开 URL 或大量折叠评论不抓；疑似关键反馈可由人工 GUI 看公开页面截图。",
            "不进入私信、群聊、付费社群或需要身份伪装的区域。",
        ),
        source_strategy(
            "App、社媒与视频",
            "P2",
            ["YouTube", "Instagram", "X/Twitter", "TikTok/抖音", "微信公众号", "Bilibili", "小红书", "公开产品演示视频"],
            "优先选择官方账号、创始人/团队账号、产品演示、评测视频和带明确时间/版本的公开内容；转营销号的二创内容降权。",
            "文字类公开页面可直接抓取；视频类优先读取标题、简介、字幕/转写和关键片段，必要时模拟人工 GUI 观看并截取画面。",
            "记录平台、账号、帖子或视频 URL、发布日期、公开视频时间点、截图或页面 URL；观点必须标明出自哪个视频的哪个时间段。",
            "用于补充视觉证据、上手流程、真实演示、用户反馈和传播/GTM 信号。",
            "公开视频无法结构化读取但疑似有用时进入问题页面核验清单；截图只截公开可见画面并写入时间点。",
            "不绕过地区限制、登录限制、付费内容或平台访问控制。",
        ),
        source_strategy(
            "海量搜索兜底",
            "P3",
            ["SearXNG 多搜索引擎结果", "新闻页", "博客", "目录站", "SEO 聚合页"],
            "在官方、垂直来源和社区策略之后使用；只用于补齐遗漏来源、发现别名、发现新品类词和发现可能竞品。",
            "用竞品名加自动搜索补充词批量检索；结果先进入来源表和证据审计，再决定是否抓取正文。",
            "记录搜索词、搜索引擎、排名、URL、摘要和命中关键词；未进入抓取预算的来源仍保留排除原因。",
            "用于来源发现和缺口补齐，默认不直接写成报告事实。",
            "命中官方域、官方文档、重要垂直站或高价值新闻时提升优先级；低价值聚合页保留审计但不进正文。",
            "不把破解下载、搬运站、无来源二创或账号壳作为事实证据。",
        ),
    ]
    if category == "autonomous_vehicle_robotaxi":
        strategies.insert(
            1,
            source_strategy(
                "无人车官方、监管与运营来源",
                "P0/P1",
                ["官网产品页", "官方安全报告", "城市运营公告", "服务区/计价说明", "监管许可/事故通报", "官方 App 页面"],
                "无人车报告先确认产品实体、车辆平台、运营城市、服务范围、计价规则和安全合规来源；官方或监管来源优先承载事实。",
                "通过 SearXNG 定向检索官网、监管公告、城市运营信息和官方 App 页面；可读网页走 Crawl4AI，不可读但公开可见时进入 GUI 复核队列。",
                "记录 URL、标题、发布日期、城市/区域、车型/版本、政策主体和抓取时间；监管或官方材料可作为主证据。",
                "用于支撑产品定位、运营范围、商业化、安全合规、版本更新和核心参数判断。",
                "官网或监管页被反爬时优先查同站公开静态页、PDF、新闻稿、sitemap 或镜像摘要；登录/验证码页面只等待用户公开登录，不绕过访问控制。",
                "不调用未授权接口，不进入后台系统，不抓取非公开运营数据。",
            ),
        )
        strategies.insert(
            2,
            source_strategy(
                "无人车视频、实测与舆情来源",
                "P1/P2",
                ["汽车媒体试乘", "公开视频测评", "Bilibili/抖音/小红书体验", "知乎/Reddit 讨论", "新闻长测"],
                "只有视频或帖子明确出现竞品、城市、场景、乘坐体验、异常 case 或公众争议时才进入复核；泛流量短视频和搬运号降权。",
                "仍先用 SearXNG 定向检索公开视频和公开帖子；视频优先读取标题、简介、字幕和元数据，必要时 GUI 观看关键片段并截图。",
                "视频观点必须保留 URL、账号、发布日期、时间点、截图或转写片段；社区观点默认是待核实线索。",
                "用于补充真实场景、复杂路况表现、等待时间、用户情绪、社会接受度和可复测 case。",
                "如果视频明显展示竞品实车、公开道路、乘坐流程或异常处置，但无可读文本，进入 GUI/视频复核；无法补到时间点则不写成 Fact。",
                "不绕过登录、地区、会员、付费或隐私限制。",
            ),
        )
    elif category == "ai_software":
        strategies.insert(
            1,
            source_strategy(
                "开发者与生态来源",
                "P0/P1",
                ["官方 API 文档", "SDK/GitHub 组织", "插件市场", "集成目录", "状态页"],
                "AI/软件产品必须优先确认接口、模型、额度、集成、权限、安全和部署边界；生态来源要能回到官方文档或仓库。",
                "先抓官方 docs/api/integrations/security/status，再抓公开 GitHub README、release、issue 摘要和插件目录。",
                "记录仓库/文档 URL、版本号、更新时间、接口路径、限制说明和关联官方域；第三方 SDK 需标明非官方。",
                "用于支撑能力、集成、稳定性、安全和路线图判断。",
                "API 文档被 JS 渲染或反爬时进入 GUI 复核，优先查官方静态文档、OpenAPI 文件或 sitemap。",
                "不调用需要密钥、登录或越权的接口。",
            ),
        )
    elif category in {"physical_product", "snow_helmet"}:
        strategies.insert(
            1,
            source_strategy(
                "规格、零售与说明书来源",
                "P0/P1",
                ["品牌产品页", "官方说明书", "尺码表", "认证页", "授权零售商商品页"],
                "实物产品先找官方规格和说明书，再用授权零售页补充库存、SKU、颜色和用户评价；普通电商营销页默认降权。",
                "抓官方产品详情、说明书、尺码/颜色/认证页；零售页只抓与参数、价格、SKU、评价直接相关的公开信息。",
                "记录品牌域名、商品型号/SKU、页面 URL、价格币种、规格版本、截图或图片来源；授权关系不明时列入待核实。",
                "用于支撑参数、定价、版本、质量和售后判断。",
                "遇到电商登录壳、购物车、优惠券或论坛导购时排除；疑似官方旗舰店但不可确认时进入待核实。",
                "不抓登录后价格、订单页、购物车或个人账户信息。",
            ),
        )
    return strategies


def analysis_dimension_field_key(dimension: Mapping[str, Any]) -> str:
    raw = textify(dimension.get("id")) or textify(dimension.get("label")) or "dimension"
    key = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    if not key:
        key = slugify(raw).replace("-", "_") or "dimension"
    return key if key.startswith("av_") else f"av_{key}"


def regex_for_evidence_term(term: str) -> str:
    term = textify(term).strip()
    if not term:
        return ""
    if re.search(r"[\u4e00-\u9fff]", term):
        return re.escape(term)
    return r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"


def field_from_analysis_dimension(dimension: Mapping[str, Any]) -> ProductCollectionField:
    label = textify(dimension.get("label")) or textify(dimension.get("id")) or "分析维度"
    required_evidence = unique_strings(dimension.get("required_evidence") or [])
    search_terms = unique_strings([label, *required_evidence])
    patterns = unique_strings(regex_for_evidence_term(term) for term in search_terms)
    description = f"模板维度：围绕“{label}”收集可追溯证据，优先保留能支撑产品判断的具体事实。"
    return field_spec(
        analysis_dimension_field_key(dimension),
        label,
        description,
        patterns,
        search_terms,
    )


def apply_analysis_template_to_collection_plan(
    plan: ProductCollectionPlan,
    competitors: Sequence[str],
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> ProductCollectionPlan:
    template, match_score = select_analysis_template(
        plan.category,
        competitors,
        own_product_name,
        own_product_positioning,
        own_product_context,
    )
    if not template:
        return plan

    template_key = textify(template.get("product_type_key"))
    template_label = textify(template.get("product_type_label"))
    dimensions = [item for item in (template.get("dimensions") or []) if isinstance(item, dict)]
    report_outline = unique_strings(template.get("report_outline") or [])
    source_priority = [item for item in (template.get("source_priority") or []) if isinstance(item, dict)]

    if template_key:
        plan.analysis_template_key = template_key
    if template_label:
        plan.analysis_template_label = template_label
    plan.analysis_template_path = textify(template.get("_source_path"))
    plan.analysis_template_summary = textify(template.get("template_basis"))
    plan.analysis_template_match_score = match_score
    plan.analysis_dimensions = [
        {
            "id": textify(item.get("id")),
            "label": textify(item.get("label")),
            "required_evidence": unique_strings(item.get("required_evidence") or []),
        }
        for item in dimensions
    ]
    plan.analysis_report_outline = report_outline

    fields_by_key: Dict[str, ProductCollectionField] = {field.key: field for field in plan.fields}
    for dimension in dimensions:
        field = field_from_analysis_dimension(dimension)
        fields_by_key.setdefault(field.key, field)
        plan.generated_search_terms = unique_strings([*plan.generated_search_terms, *field.search_terms])
        plan.evidence_keywords = unique_strings([*plan.evidence_keywords, *field.search_terms])
    plan.fields = list(fields_by_key.values())

    if report_outline:
        plan.report_focus = unique_strings(
            [
                *plan.report_focus,
                f"已加载“{template_label or template_key}”分析模板，报告应覆盖：{', '.join(report_outline)}。",
            ]
        )
    if dimensions:
        plan.report_focus = unique_strings(
            [
                *plan.report_focus,
                "模板维度只决定优先证据，不会把未采到的信息写成事实；缺失维度进入信息缺口。",
            ]
        )
    for source in source_priority:
        tier = textify(source.get("tier"))
        name = textify(source.get("name"))
        use_for = "、".join(unique_strings(source.get("use_for") or []))
        if tier or name or use_for:
            plan.source_policy_notes.append(
                f"{tier} {name}: 优先用于 {use_for or '对应模板维度'}；事实必须保留来源 URL、标题和采集时间。"
            )
    plan.source_policy_notes = unique_strings(plan.source_policy_notes)
    return plan


def build_competitor_discovery_strategies(category_label: str) -> List[CompetitorDiscoveryStrategy]:
    return [
        competitor_discovery_strategy(
            "用户输入竞品核验",
            "S0",
            "用户已经输入竞品名或域名。",
            "用竞品名检索官网、官方社媒、官方文档和高可信第三方介绍；抽取别名、产品名、母公司、官网域名。",
            "至少确认一个官方域名或官方账号；若只有第三方提及，先进入待核实，不作为主竞品事实。",
            "保存原始输入、规范化竞品名、确认用 URL、确认时间、确认理由和别名。",
            "同名无关、缺少官方来源、只出现在聚合导航页或无法绑定产品实体的候选不进入核心竞品清单。",
        ),
        competitor_discovery_strategy(
            "无竞品输入时的候选发现",
            "S1",
            "用户只填写我方产品或品类，希望先发现竞品。",
            "从我方定位中抽取用户任务、目标人群、付费对象、核心能力和品类词，拼接 alternatives、competitors、top、best、替代品、对比 等搜索意图。",
            "候选必须同时满足：名称可识别、能找到官方入口、与我方用户任务或购买场景重叠；只出现一次且无官方入口的候选先放待核实。",
            "保存发现搜索词、出现页面、候选名称、官方入口、命中理由和证据数量。",
            "只因 SEO 榜单出现、品类不重叠、目标用户不同或没有公开产品入口的候选不进入核心清单。",
        ),
        competitor_discovery_strategy(
            "垂直来源扩展竞品",
            "S2",
            "已有一个或多个竞品，需要补齐同类、相邻和替代产品。",
            f"进入 {category_label} 相关目录、应用商店、论坛问答、评测榜单和官方对比页，寻找被同一批用户放在一起比较的产品。",
            "同类竞品优先；相邻产品和替代方案单独标注，不与直接竞品混排。至少保留来源类型和为什么相关。",
            "保存来源平台、类目/榜单/帖子 URL、候选所在位置、与我方/竞品的重叠点和置信等级。",
            "没有明确对比关系、只是广告位、只靠平台推荐且无产品证据的候选降权。",
        ),
        competitor_discovery_strategy(
            "候选竞品分层入池",
            "S3",
            "候选列表产生后，进入抓取前排序。",
            "按直接竞品、相邻竞品、替代方案和观察对象分层；优先抓直接竞品官方核心页，再抓相邻/替代的验证来源。",
            "直接竞品需要用户任务、目标用户和核心能力至少两项重叠；相邻/替代产品要说明差异，不强行并列比较。",
            "保存分层、入池理由、置信度、发现来源和是否需要人工确认。",
            "无法说明重叠点或只与泛品类词相关的候选不进入抓取预算。",
        ),
    ]


BASE_COLLECTION_FIELDS = [
    field_spec(
        "official_parameters",
        "官方参数/规格",
        "官网明确写出的规格、参数、配置、限制或产品 ID。",
        [
            r"\b(specs?|specifications?|technical details?|parameters?|item number|article ref|product id|model)\b",
            r"参数|规格|技术详情|型号|货号|商品编号|产品编号",
        ],
        ["specs", "specifications", "parameters", "technical details", "参数", "规格"],
    ),
    field_spec(
        "packaging_limits",
        "套餐/包装/限制",
        "价格、套餐、版本、额度、可选配件、企业版或购买限制。",
        [
            r"\b(price|pricing|plan|plans|package|bundle|limit|quota|free|pro|enterprise|subscription)\b",
            r"价格|定价|套餐|版本|额度|限制|免费|专业版|企业版|订阅",
        ],
        ["pricing", "plans", "limits", "packages", "定价", "套餐"],
    ),
]

PHYSICAL_PRODUCT_FIELDS = [
    field_spec(
        "material_construction",
        "材质与结构",
        "外壳、内衬、结构工艺、核心材料和耐用性信息。",
        [
            r"\b(material|materials|construction|shell|liner|EPS|EPP|ABS|PC|polycarbonate|carbon|in-?mold|hardshell|hybrid)\b",
            r"材质|材料|外壳|内衬|结构|工艺|聚碳酸酯|碳纤维|硬壳|一体成型",
        ],
        ["material", "construction", "shell", "liner", "材质", "结构"],
    ),
    field_spec(
        "weight",
        "重量",
        "官网或评测中给出的重量、尺码对应重量和轻量化说法。",
        [r"\b(weight|weighs|grams?|g\b|oz\b|ounces?)\b", r"重量|克|轻量|轻便"],
        ["weight", "grams", "重量"],
    ),
    field_spec(
        "size_fit",
        "尺码与适配",
        "尺码、头围/尺寸、调节系统、佩戴贴合和尺码表。",
        [
            r"\b(size|sizes|size chart|size guide|fit system|fitting|circumference|cm\b|BOA|dial|race lock|XS|XXS|XL|XXL|S/M|M/L)\b",
            r"尺码|尺寸|头围|大小|调节|适配|贴合|尺码表",
        ],
        ["size chart", "size guide", "fit system", "尺码", "头围"],
    ),
    field_spec(
        "color_variants",
        "颜色与 SKU",
        "颜色、配色、SKU、售罄状态和可选版本。",
        [r"\b(color|colour|colors|colours|variants?|SKU|matte|white|black|blue|pink)\b", r"颜色|配色|色号|SKU|款式|现货|售罄"],
        ["colors", "colour", "variants", "颜色", "配色"],
    ),
    field_spec(
        "quality_reviews",
        "质量与口碑",
        "耐用性、做工、质保、退换货、评分和用户反馈。",
        [
            r"\b(quality|durable|durability|warranty|returns?|reviews?|rating|stars?|verified buyer)\b",
            r"质量|做工|耐用|质保|保修|退换|评价|评分|口碑",
        ],
        ["quality", "durability", "warranty", "reviews", "质量", "口碑"],
    ),
]

SNOW_HELMET_FIELDS = [
    field_spec(
        "safety_certification",
        "安全认证",
        "滑雪头盔认证、FIS/ASTM/CE EN1077 等合规信息。",
        [
            r"\b(certification|certified|ASTM|FIS|CE\s*EN\s*1077|EN1077|EN 1077|safety standard)\b",
            r"认证|安全标准|合规|FIS|ASTM|EN1077|CE",
        ],
        ["certification", "ASTM", "CE EN1077", "FIS", "安全认证"],
    ),
    field_spec(
        "protection_technology",
        "防护技术",
        "MIPS、KOROYD、AMID、Holo Core、EPS4D、SPIN、EPP 等防护技术。",
        [
            r"\b(MIPS|Mips|KOROYD|Koroyd|AMID|Holo Core|EPS4D|SPIN|rotational|impact protection|multi-impact)\b",
            r"防护|冲击|旋转力|多重冲击|吸能|缓冲|MIPS",
        ],
        ["MIPS", "impact protection", "rotational", "防护技术"],
    ),
    field_spec(
        "ventilation_comfort",
        "通风与舒适性",
        "通风孔、气流、保暖、耳垫、内衬、全天佩戴舒适性。",
        [
            r"\b(ventilation|vents?|airflow|warmth|comfort|ear pads?|liner|washable|moisture)\b",
            r"通风|透气|保暖|舒适|耳垫|内衬|可拆洗",
        ],
        ["ventilation", "airflow", "comfort", "通风", "舒适"],
    ),
    field_spec(
        "visor_goggle_chinguard",
        "风镜/护颚兼容",
        "一体风镜、磁吸镜片、雪镜贴合、护颚、镜带固定等配套信息。",
        [
            r"\b(visor|goggle|goggles|lens|magnetic|chinguard|chin guard|goggle clip|seamless fit)\b",
            r"风镜|雪镜|镜片|磁吸|护颚|下巴保护|镜带|贴合",
        ],
        ["visor", "goggle compatibility", "chinguard", "风镜", "护颚"],
    ),
    field_spec(
        "skiing_use_case",
        "使用场景",
        "双板场景、竞速/回转/大回转/自由滑/全山地/道内等定位。",
        [
            r"\b(usage area|ski|skiing|snowboard|all[- ]?mountain|freeride|race|racing|slalom|giant slalom|GS|SL|DH|on[- ]?piste|piste)\b",
            r"双板|滑雪|单板|全山地|自由滑|竞速|回转|大回转|道内",
        ],
        ["ski", "race", "all mountain", "freeride", "双板", "竞速"],
    ),
]

AI_SOFTWARE_FIELDS = [
    field_spec(
        "api_sdk_webhook",
        "API/SDK/Webhook",
        "API 文档、SDK、Webhook、开发者入口、鉴权方式和调用限制。",
        [
            r"\b(API|SDK|webhook|developer|developers|endpoint|authentication|OAuth|REST|GraphQL)\b",
            r"接口|开发者|文档|鉴权|调用|开放平台",
        ],
        ["API docs", "SDK", "webhook", "developers", "接口文档"],
    ),
    field_spec(
        "integrations_connectors",
        "集成与配套",
        "第三方集成、插件、浏览器扩展、连接器、生态和导入导出。",
        [
            r"\b(integration|integrations|connector|connectors|plugin|plugins|extension|Zapier|Slack|Notion|Google|Microsoft)\b",
            r"集成|插件|扩展|连接器|生态|导入|导出|配套",
        ],
        ["integrations", "plugins", "connectors", "extensions", "集成", "插件"],
    ),
    field_spec(
        "models_capabilities",
        "模型与核心能力",
        "支持模型、AI 能力、上下文长度、多模态、自动化和 Agent 能力。",
        [
            r"\b(model|models|LLM|GPT|Claude|Gemini|context window|multimodal|agent|automation|workflow)\b",
            r"模型|大模型|多模态|上下文|智能体|自动化|工作流",
        ],
        ["models", "AI capabilities", "workflow", "agent", "模型", "智能体"],
    ),
    field_spec(
        "usage_quota_limits",
        "额度与限制",
        "额度、用量、限流、席位、文件大小、调用次数和套餐边界。",
        [
            r"\b(quota|usage|credits?|seats?|rate limit|limit|file size|storage|tokens?|monthly)\b",
            r"额度|用量|积分|席位|限流|调用次数|文件大小|存储|tokens",
        ],
        ["quota", "usage limits", "credits", "seats", "额度", "限制"],
    ),
    field_spec(
        "security_privacy_deployment",
        "安全、隐私与部署",
        "SSO、SOC2、数据保留、权限、企业部署、私有化和合规。",
        [
            r"\b(security|privacy|SOC ?2|SSO|SAML|GDPR|HIPAA|data retention|permissions?|deployment|on-prem|enterprise)\b",
            r"安全|隐私|权限|单点登录|合规|数据保留|私有化|部署|企业版",
        ],
        ["security", "privacy", "SSO", "SOC2", "deployment", "安全", "隐私"],
    ),
]


def unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        value = textify(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def infer_product_category(
    competitors: Sequence[str],
    own_product_name: str,
    own_product_positioning: str,
    own_product_context: str,
) -> Tuple[str, str, str]:
    haystack = " ".join([own_product_name, own_product_positioning, own_product_context, *competitors]).lower()
    compact = re.sub(r"\s+", "", haystack)
    autonomous_vehicle_tokens = [
        "robotaxi",
        "self-driving",
        "driverless",
        "autonomous vehicle",
        "autonomous driving",
        "waymo",
        "cruise",
        "zoox",
        "pony.ai",
        "weride",
        "apollo go",
        "rt6",
        "无人车",
        "自动驾驶",
        "自动驾驶出租车",
        "萝卜快跑",
        "小马智行",
        "文远知行",
        "无人驾驶",
        "车机",
        "自驶",
        "传感器布局",
        "远程监控",
    ]
    snow_helmet_tokens = [
        "helmet",
        "helmets",
        "ski",
        "snowboard",
        "mips",
        "slalom",
        "fis",
        "en1077",
        "滑雪",
        "双板",
        "单板",
        "雪盔",
        "头盔",
        "全盔",
        "护颚",
    ]
    ai_tokens = [
        " ai",
        "aigc",
        "llm",
        "api",
        "sdk",
        "agent",
        "copilot",
        "saas",
        "workflow",
        "automation",
        "model",
        "openai",
        "接口",
        "模型",
        "智能体",
        "插件",
        "集成",
        "自动化",
        "工作流",
    ]
    if any(token in haystack or token in compact for token in autonomous_vehicle_tokens):
        return (
            "autonomous_vehicle_robotaxi",
            "无人车/Robotaxi/自动驾驶整车",
            "我方定位或竞品名包含无人车、Robotaxi、自动驾驶、整车平台或运营场景信号，采集重点应覆盖官方参数、城市运营、商业化、安全合规、乘坐体验、复杂场景、运维和舆情证据。",
        )
    if any(token in haystack or token in compact for token in snow_helmet_tokens):
        return (
            "snow_helmet",
            "滑雪头盔/双板全盔",
            "我方定位或竞品名包含滑雪、双板、头盔、MIPS、FIS 等信号，采集重点应从通用功能扩展到材质、重量、尺码、颜色、安全认证和配套兼容。",
        )
    if re.search(r"(^|[^a-z0-9])ai([^a-z0-9]|$)", haystack) or any(token in haystack or token in compact for token in ai_tokens):
        return (
            "ai_software",
            "AI/软件工具",
            "我方定位或竞品名包含 AI、API、SDK、Agent、集成、自动化等信号，采集重点应覆盖接口、模型、额度、集成、安全和企业能力。",
        )
    if any(token in haystack for token in ["hardware", "device", "wearable", "gear", "equipment", "材质", "尺码", "颜色", "质量"]):
        return (
            "physical_product",
            "实物商品",
            "我方定位包含材质、尺码、颜色或装备类信号，采集重点应覆盖规格、材料、版本、质量和售后。",
        )
    return (
        "general_product",
        "通用产品/服务",
        "暂未识别到明确品类，先按通用 PM 框架补充参数、定价、限制、配套和质量证据。",
    )


def build_product_collection_plan(
    competitors: Sequence[str],
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> ProductCollectionPlan:
    category, category_label, rationale = infer_product_category(
        competitors,
        own_product_name,
        own_product_positioning,
        own_product_context,
    )
    fields = list(BASE_COLLECTION_FIELDS)
    search_templates = [
        "{name} official specifications parameters pricing",
        "{name} official product details specs",
        "{name} reviews quality limitations",
    ]
    cn_search_templates = [
        "{name} 官方 参数 规格 价格",
        "{name} 质量 评价 限制",
    ]
    image_terms = ["product details", "specs", "official product image"]
    report_focus = [
        "先判断哪些字段是本品类采购或选型的核心信息。",
        "官方事实和第三方口碑分开标注，不用评论覆盖官网参数。",
    ]
    generated_search_terms = [
        "official website",
        "official product details",
        "official specifications",
        "pricing plans",
        "features",
        "customers case studies",
        "docs help center",
        "security trust",
        "changelog release notes",
        "官网",
        "官方 产品",
        "官方 参数 规格",
        "价格 定价 套餐",
        "客户案例",
        "帮助文档",
    ]
    evidence_keywords = [
        "official",
        "pricing",
        "plans",
        "features",
        "product",
        "specs",
        "specifications",
        "parameters",
        "docs",
        "documentation",
        "support",
        "security",
        "customers",
        "case study",
        "changelog",
        "release notes",
        "官网",
        "官方",
        "参数",
        "规格",
        "价格",
        "定价",
        "套餐",
        "功能",
        "文档",
        "客户案例",
        "更新日志",
    ]
    fixed_exclude_keywords = [
        "login",
        "sign in",
        "signin",
        "register",
        "signup",
        "cart",
        "checkout",
        "coupon",
        "promo code",
        "jobs",
        "careers",
        "forum index",
        "site search",
        "亲，请登录",
        "免费注册",
        "购物车",
        "已买到的宝贝",
        "我的淘宝",
        "卖家中心",
        "开店",
        "优惠券",
        "招聘",
        "论坛导航",
        "站内搜索",
        "破解",
        "破解版",
        "免费下载",
        "网盘下载",
        "download crack",
        "apk download",
        "torrent",
        "广告",
        "促销",
        "课程促销",
        "直播预告",
        "会议报名",
        "抽奖",
        "标题党",
        "搬运",
        "转载",
        "SEO",
    ]
    dynamic_exclude_keywords: List[str] = []
    source_policy_notes = [
        "P0 官方核心页先过硬门禁：官网、产品页、规格/参数页、定价页、文档、API、Security/Trust、Changelog。",
        "P1 官方补充页作为背景：博客、新闻、客户案例、发布说明、帮助中心。",
        "P2 第三方高价值来源只做验证：专业评测、主流媒体、Product Hunt、G2/Capterra 等公开摘要，不覆盖官网事实。",
        "P3 社媒、论坛、目录站、SEO 聚合只做低置信线索，默认不进入正文证据池。",
    ]

    if category == "autonomous_vehicle_robotaxi":
        search_templates += [
            "{name} official robotaxi product specs service area pricing",
            "{name} autonomous vehicle sensor compute platform safety report",
            "{name} robotaxi city operation fare waiting time coverage",
            "{name} autonomous driving permit regulator safety incident",
            "{name} robotaxi ride experience review video",
            "{name} public opinion acceptance employment privacy safety",
        ]
        cn_search_templates += [
            "{name} 无人车 官网 参数 运营 城市 价格",
            "{name} 自动驾驶 传感器 算力 安全 合规",
            "{name} Robotaxi 试乘 等待时间 路线 体验",
            "{name} 无人车 舆情 事故 隐私 就业 争议",
            "{name} 车辆平台 续航 补能 运维 OTA",
        ]
        image_terms += [
            "robotaxi vehicle exterior",
            "robotaxi cabin screen",
            "autonomous vehicle sensor suite",
            "ride hailing app screenshot",
            "service area map",
            "无人车 外观",
            "车内屏",
            "传感器布局",
            "服务区地图",
        ]
        report_focus += [
            "横向比较产品定位、市场运营、价格商业化、整车平台、自动驾驶系统、安全合规、乘坐体验、复杂场景表现、智舱交互、运维效率、社会接受度和更新节奏。",
            "视频和社媒只在有竞品、场景、时间点和截图/转写证据时进入报告；否则只作为待核实线索。",
        ]
        generated_search_terms += [
            "robotaxi official specs",
            "autonomous vehicle safety report",
            "service area city coverage",
            "fare pricing subsidy",
            "sensor suite lidar camera radar",
            "compute platform TOPS",
            "remote assistance monitoring",
            "ride experience waiting time",
            "edge cases construction cone unprotected turn",
            "fleet size operations",
            "regulatory permit autonomous vehicle",
            "public opinion employment privacy safety",
            "无人车 官网 参数",
            "自动驾驶 传感器 算力",
            "Robotaxi 运营城市 服务范围",
            "计费规则 优惠补贴",
            "安全报告 监管许可",
            "试乘 等待时间 路线",
            "复杂场景 施工 锥桶 无保护转弯",
            "远程监控 运维 补能 OTA",
            "舆论 热搜 隐私 就业争议",
        ]
        evidence_keywords += [
            "robotaxi",
            "autonomous vehicle",
            "self-driving",
            "driverless",
            "service area",
            "city coverage",
            "fare",
            "fleet",
            "sensor",
            "lidar",
            "radar",
            "camera",
            "compute",
            "TOPS",
            "safety report",
            "permit",
            "remote assistance",
            "waiting time",
            "ride experience",
            "无人车",
            "自动驾驶",
            "自动驾驶出租车",
            "运营城市",
            "服务范围",
            "计费规则",
            "车辆规模",
            "传感器",
            "算力",
            "冗余",
            "安全报告",
            "监管许可",
            "远程监控",
            "试乘",
            "等待时间",
            "路线",
            "运维",
            "舆情",
        ]
        dynamic_exclude_keywords += [
            "遥控车",
            "玩具车",
            "模型车",
            "科幻电影",
            "游戏 MOD",
            "无人机",
            "agv 仓储",
            "叉车",
            "toy car",
            "rc car",
            "drone",
            "forklift",
        ]
        source_policy_notes += [
            "无人车/Robotaxi 模板会优先使用官方、监管、运营和实测视频证据；社媒观点必须保留公开视频时间点或截图。",
        ]
    elif category == "snow_helmet":
        fields += PHYSICAL_PRODUCT_FIELDS + SNOW_HELMET_FIELDS
        search_templates += [
            "{name} official helmet specs weight size colors certification",
            "{name} material construction MIPS safety certification ASTM CE EN1077 FIS",
            "{name} size chart colors weight ventilation fit system",
            "{name} visor goggle compatibility chinguard helmet",
            "{name} review quality durability ski helmet",
        ]
        cn_search_templates += [
            "{name} 滑雪头盔 材质 重量 尺码 颜色 认证",
            "{name} 双板 头盔 MIPS 防护 通风 护颚",
            "{name} 滑雪头盔 质量 做工 评价 尺码",
        ]
        image_terms += [
            "helmet side view",
            "size chart",
            "color options",
            "MIPS",
            "certification",
            "visor",
            "goggle compatibility",
            "chinguard",
            "滑雪头盔",
            "尺码表",
        ]
        report_focus += [
            "横向比较材质/结构、重量、尺码、颜色、安全认证、防护技术、通风舒适、雪镜/护颚兼容。",
            "把竞速、全山地、自由滑、道内等使用场景分开，不把普通雪盔和 FIS 竞速盔混在同一结论里。",
        ]
        generated_search_terms += [
            "ski helmet official specs",
            "helmet weight grams",
            "size chart circumference",
            "colors variants SKU",
            "material construction shell liner",
            "MIPS impact protection",
            "ASTM CE EN1077 FIS certification",
            "ventilation fit system",
            "goggle compatibility chinguard",
            "warranty reviews quality",
            "滑雪头盔 材质 重量",
            "滑雪头盔 尺码 头围",
            "滑雪头盔 颜色 SKU",
            "滑雪头盔 安全认证",
            "MIPS 防护技术",
            "雪镜兼容 护颚",
            "质量 质保 评价",
        ]
        evidence_keywords += [
            "helmet",
            "ski helmet",
            "weight",
            "size chart",
            "fit system",
            "color",
            "MIPS",
            "ASTM",
            "CE EN1077",
            "FIS",
            "safety certification",
            "shell",
            "liner",
            "ventilation",
            "goggle compatibility",
            "chinguard",
            "滑雪头盔",
            "双板",
            "重量",
            "尺码",
            "头围",
            "颜色",
            "材质",
            "安全认证",
            "防护技术",
            "通风",
            "雪镜兼容",
            "护颚",
        ]
        dynamic_exclude_keywords += [
            "bike helmet",
            "bicycle helmet",
            "motorcycle helmet",
            "football helmet",
            "hard hat",
            "helmet camera",
            "自行车头盔",
            "摩托车头盔",
            "安全帽",
            "橄榄球头盔",
            "头盔摄像机",
        ]
    elif category == "physical_product":
        fields += PHYSICAL_PRODUCT_FIELDS
        search_templates += [
            "{name} official material size colors weight warranty",
            "{name} product specs dimensions quality reviews",
        ]
        cn_search_templates += [
            "{name} 材质 尺码 颜色 重量 质保",
            "{name} 参数 尺寸 质量 评价",
        ]
        image_terms += ["size chart", "color options", "materials", "details", "尺码表", "颜色"]
        report_focus += [
            "横向比较材质、尺寸/尺码、颜色、重量、质保和用户评价。",
        ]
        generated_search_terms += [
            "official material size colors weight",
            "product specs dimensions",
            "warranty return policy",
            "quality durability reviews",
            "材质 尺码 颜色 重量",
            "参数 尺寸 质保",
            "质量 做工 评价",
        ]
        evidence_keywords += [
            "material",
            "materials",
            "size chart",
            "dimensions",
            "weight",
            "colors",
            "warranty",
            "reviews",
            "quality",
            "durability",
            "材质",
            "尺寸",
            "尺码",
            "重量",
            "颜色",
            "质保",
            "质量",
            "评价",
        ]
        dynamic_exclude_keywords += [
            "二手",
            "闲置",
            "代购",
            "优惠券",
            "清仓",
            "批发",
            "wholesale",
            "used",
            "second hand",
        ]
    elif category == "ai_software":
        fields += AI_SOFTWARE_FIELDS
        search_templates += [
            "{name} API documentation SDK webhook pricing limits",
            "{name} integrations connectors plugins extensions",
            "{name} supported models AI capabilities workflow automation",
            "{name} security privacy SSO SOC2 enterprise deployment",
            "{name} docs rate limits quota seats credits",
        ]
        cn_search_templates += [
            "{name} API 文档 SDK 接口 价格 限制",
            "{name} 集成 插件 模型 额度 企业版 安全",
            "{name} 工作流 自动化 配套 文档",
        ]
        image_terms += ["dashboard", "API docs", "integrations", "workflow", "settings", "pricing"]
        report_focus += [
            "横向比较 API/SDK、集成、支持模型、额度限制、安全隐私、部署方式和工作流。",
            "区分官网承诺、文档事实、套餐限制和第三方评价。",
        ]
        generated_search_terms += [
            "API documentation SDK webhook",
            "integrations connectors plugins",
            "supported models AI capabilities",
            "rate limits quota credits seats",
            "security privacy SOC2 SSO enterprise",
            "deployment workflow automation",
            "pricing enterprise plans",
            "docs limits changelog",
            "API 文档 SDK 接口",
            "集成 插件 连接器",
            "模型 支持能力",
            "额度 限制 seats credits",
            "企业版 安全 隐私 SSO",
            "部署 工作流 自动化",
        ]
        evidence_keywords += [
            "API",
            "SDK",
            "webhook",
            "docs",
            "documentation",
            "integrations",
            "connectors",
            "plugins",
            "extensions",
            "models",
            "rate limits",
            "quota",
            "credits",
            "seats",
            "SSO",
            "SOC2",
            "security",
            "privacy",
            "enterprise",
            "deployment",
            "workflow",
            "automation",
            "接口",
            "文档",
            "集成",
            "插件",
            "模型",
            "额度",
            "限制",
            "企业版",
            "安全",
            "隐私",
            "部署",
            "工作流",
            "自动化",
        ]
        dynamic_exclude_keywords += [
            "template tutorial",
            "prompt collection",
            "coupon",
            "lifetime deal",
            "破解版",
            "插件下载",
            "模板教程",
            "提示词合集",
            "优惠码",
            "限时优惠",
            "安装包下载",
        ]
    else:
        fields += AI_SOFTWARE_FIELDS[:2]
        search_templates += [
            "{name} integrations docs support security",
            "{name} product specs limitations reviews",
        ]
        cn_search_templates += [
            "{name} 配套 文档 安全 限制",
        ]
        report_focus += [
            "如果后续填写更具体的我方定位，系统会追加更细的品类字段。",
        ]

    field_by_key: Dict[str, ProductCollectionField] = {}
    for field in fields:
        field_by_key.setdefault(field.key, field)

    for field in field_by_key.values():
        evidence_keywords.extend(field.search_terms)
        generated_search_terms.extend(field.search_terms)

    exclude_keywords = unique_strings([*fixed_exclude_keywords, *dynamic_exclude_keywords])

    plan = ProductCollectionPlan(
        category=category,
        category_label=category_label,
        rationale=rationale,
        fields=list(field_by_key.values()),
        search_templates=unique_strings(search_templates),
        cn_search_templates=unique_strings(cn_search_templates),
        image_terms=unique_strings(image_terms),
        report_focus=unique_strings(report_focus),
        generated_search_terms=unique_strings(generated_search_terms),
        fixed_exclude_keywords=unique_strings(fixed_exclude_keywords),
        dynamic_exclude_keywords=unique_strings(dynamic_exclude_keywords),
        evidence_keywords=unique_strings(evidence_keywords),
        exclude_keywords=exclude_keywords,
        source_policy_notes=unique_strings(source_policy_notes),
        source_strategies=build_source_strategies(category),
        competitor_discovery_strategies=build_competitor_discovery_strategies(category_label),
        directed_source_search_templates=build_directed_source_search_templates(category),
        value_judgment_rules=build_value_judgment_rules(),
    )
    plan = apply_analysis_template_to_collection_plan(
        plan,
        competitors,
        own_product_name,
        own_product_positioning,
        own_product_context,
    )
    plan.search_term_reasons = search_term_reason_rows(plan, plan.generated_search_terms, "rule")
    return plan


def collection_plan_field_keys(plan: Optional[ProductCollectionPlan]) -> List[str]:
    if not plan:
        return []
    return [field.key for field in plan.fields]


def normalize_keyword_inputs(values: Iterable[str]) -> List[str]:
    chunks: List[str] = []
    for value in values or []:
        text = textify(value)
        for chunk in re.split(r"[\n,，;；]+", text):
            item = re.sub(r"\s+", " ", chunk).strip()
            if item:
                chunks.append(item)
    return unique_strings(chunks)


def keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    haystack = textify(text).lower()
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    hits = []
    for keyword in keywords or []:
        needle = textify(keyword).strip().lower()
        if not needle:
            continue
        compact_needle = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", needle)
        if needle in haystack or (compact_needle and compact_needle in compact):
            hits.append(keyword)
    return unique_strings(hits)


def collection_plan_evidence_keywords(
    plan: Optional[ProductCollectionPlan],
    manual_keywords: Iterable[str] = (),
) -> List[str]:
    keywords = list(PAGE_PRIORITY_KEYWORDS)
    if plan:
        keywords.extend(plan.evidence_keywords)
        for field in plan.fields:
            keywords.extend(field.search_terms)
    keywords.extend(manual_keywords or [])
    return unique_strings(keywords)


def collection_plan_exclude_keywords(
    plan: Optional[ProductCollectionPlan],
    manual_keywords: Iterable[str] = (),
) -> List[str]:
    keywords = []
    if plan:
        keywords.extend(plan.exclude_keywords)
    keywords.extend(manual_keywords or [])
    return unique_strings(keywords)


def product_field_hits(text: str, plan: Optional[ProductCollectionPlan]) -> List[str]:
    if not plan:
        return []
    hits: List[str] = []
    for field in plan.fields:
        matched = False
        for pattern in field.patterns:
            try:
                if re.search(pattern, text, re.I):
                    matched = True
                    break
            except re.error:
                continue
        if not matched and keyword_hits(text, field.search_terms):
            matched = True
        if matched:
            hits.append(field.label)
    return unique_strings(hits)


def source_policy_tier_for(source_kind: str, page_role: str = "") -> str:
    if source_kind == "official_core":
        return "P0 官方核心来源"
    if source_kind in {"official_expansion", "official_public"}:
        return "P1 官方补充来源"
    if page_role == "app_store_listing":
        return "P2 应用商店/垂直平台验证来源"
    if page_role in {"video_or_social_content", "forum_or_community_discussion"}:
        return "P3 社区/社媒/视频线索"
    if page_role == "autonomous_vehicle_detail":
        return "P1 无人车运营/参数/实测来源"
    if source_kind in {"trusted_public", "third_party_verification_source"}:
        return "P2 第三方验证来源"
    if source_kind == "community_or_social_signal":
        return "P3 低置信线索"
    if page_role in {"auth_or_account_shell", "transaction_or_marketplace_shell"}:
        return "Reject 登录/交易壳"
    if source_kind in {"low_value_or_aggregator", "non_html_asset", "missing_url"}:
        return "Reject 低价值或不可抓来源"
    return "P2 公开网页候选"


def fact_type_for(page_role: str, matched_fields: Sequence[str], text: str = "") -> str:
    haystack = textify(text).lower()
    field_text = " ".join(matched_fields)
    if page_role == "pricing_packaging" or keyword_hits(haystack, ["pricing", "price", "plans", "套餐", "定价", "价格"]):
        return "pricing_packaging"
    if page_role == "docs_api_or_developer" or keyword_hits(haystack + " " + field_text, ["api", "sdk", "webhook", "接口", "开发者"]):
        return "api_docs_limits"
    if page_role == "security_trust_or_legal" or keyword_hits(haystack + " " + field_text, ["security", "privacy", "soc2", "sso", "安全", "隐私", "认证"]):
        return "security_compliance"
    if page_role == "changelog_or_release" or keyword_hits(haystack, ["changelog", "release", "updates", "更新", "发布"]):
        return "release_update"
    if page_role == "customer_or_solution" or keyword_hits(haystack + " " + field_text, ["customer", "case", "客户", "案例", "场景"]):
        return "customer_case_gtm"
    if page_role in {"product_specs_or_features", "physical_product_detail", "ai_capability_detail"}:
        return "product_capability_specs"
    if page_role == "autonomous_vehicle_detail":
        return "autonomous_vehicle_competitor_evidence"
    if page_role == "app_store_listing":
        return "app_store_metadata"
    if page_role == "video_or_social_content":
        return "visual_product_evidence"
    if page_role == "forum_or_community_discussion":
        return "community_user_feedback"
    if page_role == "review_or_comparison" or keyword_hits(haystack, ["review", "comparison", "alternatives", "评价", "评测", "对比"]):
        return "review_quality_perception"
    if matched_fields:
        return "product_specific_field"
    return "general_product_signal"


def increment_type_for(page_role: str, matched_fields: Sequence[str], text: str = "") -> str:
    fact_type = fact_type_for(page_role, matched_fields, text)
    mapping = {
        "pricing_packaging": "新增价格/套餐/限制",
        "api_docs_limits": "新增 API/接口/额度/限制",
        "security_compliance": "新增安全/隐私/认证/合规",
        "release_update": "新增版本/功能/发布节奏",
        "customer_case_gtm": "新增客户/场景/GTM",
        "product_capability_specs": "新增参数/规格/能力",
        "autonomous_vehicle_competitor_evidence": "新增无人车参数/运营/安全/体验证据",
        "app_store_metadata": "新增应用商店/平台元数据",
        "visual_product_evidence": "新增视频/社媒/截图线索",
        "community_user_feedback": "新增社区/论坛用户线索",
        "review_quality_perception": "新增用户反馈/质量/口碑",
        "product_specific_field": "新增品类字段证据",
    }
    return mapping.get(fact_type, "一般线索，需人工判断增量")


def fact_group_for(
    competitor: str,
    fact_type: str,
    matched_fields: Sequence[str],
    page_role: str,
    url: str,
    title: str,
) -> str:
    field_key = slugify(matched_fields[0]) if matched_fields else slugify(page_role or fact_type)
    path = urlparse(url).path.lower().strip("/")
    path_tokens = [token for token in re.split(r"[/._-]+", path) if token]
    stable_path = "-".join(path_tokens[:3]) if path_tokens else slugify(title)[:40]
    return ":".join(filter(None, [slugify(competitor), fact_type, field_key, stable_path]))[:180]


def verification_policy_for(
    source_kind: str,
    page_role: str,
    decision_status: str,
    pre_crawl_reason: str,
    evidence_hits: Sequence[str],
    field_hits: Sequence[str],
    rejection_code: str = "",
) -> Tuple[bool, str]:
    has_value_signal = bool(evidence_hits or field_hits or page_role not in {"general_candidate", "auth_or_account_shell", "transaction_or_marketplace_shell"})
    if decision_status == "rejected" and (pre_crawl_reason or rejection_code):
        if has_value_signal and page_role not in {"auth_or_account_shell", "transaction_or_marketplace_shell"}:
            return True, f"被规则拒绝但存在价值线索，需要核实：{pre_crawl_reason or rejection_code}"
        return False, ""
    if source_kind in {"third_party_verification_source", "community_or_social_signal"}:
        return True, "非一手来源，只能做验证或线索，不能直接写成 Fact"
    if source_kind == "low_value_or_aggregator" and has_value_signal:
        return True, "聚合/二创来源可能含关键事实，需要追溯到原始来源"
    if page_role in {"review_or_comparison", "news_or_blog"} and not source_kind.startswith("official"):
        return True, "第三方评测/新闻需确认作者、日期、方法或原始引用"
    return False, ""


def primary_evidence_policy_for(
    source_kind: str,
    page_role: str,
    decision_status: str,
    pending_verification: bool,
) -> Tuple[bool, str]:
    if pending_verification:
        return False, "待核实内容不能作为主证据"
    if decision_status not in {"selected", "accepted"}:
        return False, "未进入 selected/accepted，不作为主证据"
    if source_kind == "official_core":
        return True, "官方核心页，可优先作为主证据"
    if source_kind in {"official_expansion", "official_public"} and page_role not in {"general_candidate", "news_or_blog"}:
        return True, "官方公开页且命中核心页面角色，可作为主证据"
    if source_kind == "trusted_public":
        return True, "可信第三方公开来源，可作为补充主证据候选"
    return False, "优先级低于官方或可信第三方，只做补充证据"


GENERIC_VALUE_EVIDENCE_TERMS = {
    "official",
    "official website",
    "official site",
    "product",
    "products",
    "features",
    "官网",
    "官方",
    "官方 产品",
    "功能",
}


def meaningful_evidence_hits(evidence_hits: Sequence[str]) -> List[str]:
    meaningful = []
    for hit in evidence_hits:
        normalized = textify(hit).strip().lower()
        if normalized and normalized not in GENERIC_VALUE_EVIDENCE_TERMS:
            meaningful.append(hit)
    return meaningful


def page_has_decision_signal(
    page_role: str,
    haystack: str,
    evidence_hits: Sequence[str],
    field_hits: Sequence[str],
    manual_include_hits: Sequence[str],
) -> bool:
    if meaningful_evidence_hits(evidence_hits) or field_hits or manual_include_hits:
        return True
    if page_role in {
        "pricing_packaging",
        "docs_api_or_developer",
        "security_trust_or_legal",
        "changelog_or_release",
        "customer_or_solution",
        "product_specs_or_features",
        "physical_product_detail",
        "ai_capability_detail",
        "autonomous_vehicle_detail",
        "app_store_listing",
        "video_or_social_content",
        "forum_or_community_discussion",
        "review_or_comparison",
    }:
        return True
    return bool(
        keyword_hits(
            haystack,
            [
                "pricing",
                "price",
                "plans",
                "demo",
                "review",
                "walkthrough",
                "workflow",
                "dashboard",
                "api",
                "docs",
                "integration",
                "quality",
                "problem",
                "robotaxi",
                "autonomous vehicle",
                "sensor",
                "fleet",
                "permit",
                "service area",
                "定价",
                "价格",
                "演示",
                "评测",
                "评价",
                "问题",
                "质量",
                "参数",
                "规格",
                "截图",
                "无人车",
                "自动驾驶",
                "运营城市",
                "服务范围",
                "传感器",
                "算力",
                "监管许可",
            ],
        )
    )


def page_has_increment_signal(
    page_role: str,
    haystack: str,
    evidence_hits: Sequence[str],
    field_hits: Sequence[str],
) -> bool:
    if meaningful_evidence_hits(evidence_hits) or field_hits:
        return True
    if page_role in {
        "pricing_packaging",
        "docs_api_or_developer",
        "security_trust_or_legal",
        "changelog_or_release",
        "customer_or_solution",
        "product_specs_or_features",
        "physical_product_detail",
        "ai_capability_detail",
        "autonomous_vehicle_detail",
        "app_store_listing",
        "video_or_social_content",
        "forum_or_community_discussion",
    }:
        return True
    return bool(
        keyword_hits(
            haystack,
            [
                "pricing",
                "quota",
                "rate limit",
                "size chart",
                "certification",
                "screenshot",
                "dashboard",
                "workflow",
                "setup",
                "comparison",
                "alternative",
                "release",
                "version",
                "sensor suite",
                "service area",
                "fleet size",
                "waiting time",
                "safety report",
                "价格",
                "额度",
                "尺码",
                "认证",
                "截图",
                "流程",
                "对比",
                "替代",
                "版本",
                "运营城市",
                "服务范围",
                "等待时间",
                "传感器",
                "安全报告",
            ],
        )
    )


def access_is_public_for_value(page_role: str, rejection_code: str, is_html: bool) -> bool:
    if not is_html:
        return False
    if page_role in {"auth_or_account_shell", "transaction_or_marketplace_shell"}:
        return False
    if any(token in rejection_code for token in ("login", "auth", "transaction", "private", "paid", "captcha")):
        return False
    return True


def value_judgment_for_result(
    item: SearchResult,
    source_kind: str,
    page_role: str,
    haystack: str,
    competitor_hits: Sequence[str],
    evidence_hits: Sequence[str],
    field_hits: Sequence[str],
    manual_include_hits: Sequence[str],
    rejection_code: str,
    is_html: bool,
) -> Tuple[List[str], List[str], str]:
    labels = {
        "competitor_binding": "竞品绑定",
        "decision_relevance": "决策相关",
        "information_increment": "信息增量",
        "source_credibility": "来源可信",
        "traceability": "可追溯",
        "public_access": "可获取",
    }
    checks = {
        "competitor_binding": bool(competitor_hits or source_kind.startswith("official") or manual_include_hits),
        "decision_relevance": page_has_decision_signal(page_role, haystack, evidence_hits, field_hits, manual_include_hits),
        "information_increment": page_has_increment_signal(page_role, haystack, evidence_hits, field_hits),
        "source_credibility": source_kind not in {"missing_url", "low_value_or_aggregator"} and page_role not in {"auth_or_account_shell", "transaction_or_marketplace_shell"},
        "traceability": bool(item.url and domain_of(item.url) and (item.title or item.query or item.snippet)),
        "public_access": access_is_public_for_value(page_role, rejection_code, is_html),
    }
    signals = [labels[key] for key, ok in checks.items() if ok]
    missing = [labels[key] for key, ok in checks.items() if not ok]
    if not checks["public_access"]:
        verdict = "reject_access_or_format"
    elif checks["competitor_binding"] and checks["decision_relevance"] and checks["information_increment"] and checks["traceability"]:
        verdict = "valuable_for_crawl_or_review"
    elif checks["decision_relevance"] and checks["traceability"]:
        verdict = "valuable_low_confidence_signal"
    else:
        verdict = "low_value_or_noise"
    return signals, missing, verdict


def gui_review_policy_for(
    page_role: str,
    value_verdict: str,
    source_kind: str,
    is_selected: bool,
    rejection_code: str,
) -> Tuple[bool, str]:
    if is_selected:
        return False, ""
    if value_verdict not in {"valuable_for_crawl_or_review", "valuable_low_confidence_signal"}:
        return False, ""
    if page_role == "video_or_social_content":
        return True, "视频/社媒内容命中竞品和决策问题；若自动文本不可读，GUI 复核需记录视频 URL、发布时间、观点时间点和公开截图。"
    if page_role == "forum_or_community_discussion":
        return True, "论坛/社区内容命中竞品和用户反馈问题；GUI 复核需确认帖子公开、楼层/发布时间可追溯，并只作为线索或验证。"
    if page_role == "app_store_listing":
        return True, "应用商店/平台页可能补充评分、版本、截图或评论；GUI 复核需保留平台 URL、版本/评分和截图。"
    if source_kind.startswith("official") and rejection_code:
        return True, "官方核心来源被规则拦截但可能有价值；GUI 复核需确认公开可见并补充同站公开证据。"
    return False, ""


def expected_source_for_term(term: str) -> str:
    low = term.lower()
    if keyword_hits(low, ["pricing", "price", "plans", "套餐", "定价", "价格"]):
        return "官网 pricing/plans/billing 页面"
    if keyword_hits(low, ["api", "sdk", "docs", "developer", "webhook", "接口", "文档"]):
        return "官方文档、API 或开发者页面"
    if keyword_hits(low, ["security", "privacy", "soc2", "sso", "trust", "安全", "隐私", "合规"]):
        return "官方 trust/security/privacy 页面或权威认证来源"
    if keyword_hits(low, ["customer", "case", "review", "quality", "客户", "案例", "评价", "质量"]):
        return "客户案例、评测、评论或第三方验证来源"
    if keyword_hits(low, ["changelog", "release", "updates", "发布", "更新"]):
        return "官方 changelog、release notes 或新闻稿"
    if keyword_hits(low, ["spec", "parameter", "material", "size", "weight", "规格", "参数", "材质", "尺码", "重量"]):
        return "官网产品详情、规格参数页或说明书"
    return "官方产品页或高可信公开来源"


def expected_evidence_for_term(term: str) -> str:
    low = term.lower()
    if keyword_hits(low, ["pricing", "price", "plans", "quota", "credits", "套餐", "定价", "额度"]):
        return "价格、套餐、额度、限制或企业版包装"
    if keyword_hits(low, ["api", "sdk", "webhook", "docs", "接口", "文档"]):
        return "接口能力、鉴权、调用限制、SDK 或接入方式"
    if keyword_hits(low, ["security", "privacy", "soc2", "sso", "安全", "隐私"]):
        return "安全、隐私、认证、SSO、数据保留或部署边界"
    if keyword_hits(low, ["material", "size", "weight", "color", "certification", "材质", "尺码", "重量", "颜色", "认证"]):
        return "官方参数、材质、尺码、颜色、认证或规格"
    if keyword_hits(low, ["customer", "case", "gtm", "客户", "案例", "场景"]):
        return "目标客户、使用场景、行业案例或 GTM 线索"
    if keyword_hits(low, ["review", "quality", "评价", "质量", "口碑"]):
        return "质量反馈、用户评价、限制或风险线索"
    return "能支撑报告字段的可追溯事实"


def source_field_for_term(term: str, plan: Optional[ProductCollectionPlan]) -> str:
    if not plan:
        return "通用竞品分析字段"
    term_text = term.lower()
    for field in plan.fields:
        if keyword_hits(term_text, field.search_terms):
            return f"{field.label} ({field.key})"
    if keyword_hits(term_text, ["pricing", "plans", "price", "定价", "价格", "套餐"]):
        return "套餐/包装/限制 (packaging_limits)"
    if keyword_hits(term_text, ["spec", "parameter", "参数", "规格"]):
        return "官方参数/规格 (official_parameters)"
    return "报告框架字段/来源角色模板"


def search_term_reason_rows(plan: Optional[ProductCollectionPlan], terms: Iterable[str], generated_by: str = "rule") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for term in unique_strings(terms):
        expected_source = expected_source_for_term(term)
        expected_evidence = expected_evidence_for_term(term)
        source_field = source_field_for_term(term, plan)
        noise_risk = "中：需绑定竞品名/官网/型号，避免教程、促销、下载站或同名无关内容"
        if keyword_hits(term, ["official", "官网", "docs", "pricing", "api", "security"]):
            noise_risk = "低：更容易命中官方或高价值来源"
        if keyword_hits(term, ["review", "评价", "alternatives", "对比", "quality", "质量"]):
            noise_risk = "中高：可能混入 SEO 聚合、泛评测或二手转述"
        priority = "P0" if any(token in expected_source for token in ["官网", "官方", "trust", "security", "API", "文档"]) else "P1"
        rows.append(
            {
                "term": term,
                "source_field": source_field,
                "expected_source": expected_source,
                "expected_evidence": expected_evidence,
                "must_bind_competitor": "yes",
                "noise_risk": noise_risk,
                "priority": priority,
                "generated_by": generated_by,
            }
        )
    return rows


def merge_search_term_reasons(existing: Sequence[Dict[str, str]], additions: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in list(additions) + list(existing):
        term = textify(row.get("term", "")).strip()
        if term and term not in merged:
            merged[term] = {str(key): textify(value) for key, value in row.items()}
    return list(merged.values())


def comparable_card_key(value: str) -> str:
    return re.sub(r"[-_]+", "-", slugify(value))


def search_card_matches_plan(card: Mapping[str, Any], plan: ProductCollectionPlan) -> bool:
    card_type = comparable_card_key(textify(card.get("product_type_key")) or textify(card.get("product_category")))
    card_category = textify(card.get("product_category")).lower()
    card_label = textify(card.get("product_type_label")).lower()
    return bool(
        card_type == comparable_card_key(plan.category)
        or card_category == plan.category.lower()
        or card_label == plan.category_label.lower()
    )


def apply_search_cards_to_collection_plan(
    plan: ProductCollectionPlan,
    cards: Sequence[Mapping[str, Any]],
) -> ProductCollectionPlan:
    applied: List[Dict[str, Any]] = []
    new_terms: List[str] = []
    new_reason_rows: List[Dict[str, str]] = []
    for card in cards or []:
        if not search_card_matches_plan(card, plan):
            continue
        label = textify(card.get("product_type_label")) or textify(card.get("product_type_key")) or plan.category_label
        terms = unique_strings(card.get("search_terms") or [])
        evidence_terms = unique_strings(card.get("evidence_keywords") or terms)
        exclude_terms = unique_strings(card.get("exclude_keywords") or [])
        templates = unique_strings(card.get("directed_source_search_templates") or [])
        plan.generated_search_terms = unique_strings([*plan.generated_search_terms, *terms])
        plan.evidence_keywords = unique_strings([*plan.evidence_keywords, *evidence_terms])
        plan.dynamic_exclude_keywords = unique_strings([*plan.dynamic_exclude_keywords, *exclude_terms])
        plan.exclude_keywords = unique_strings([*plan.fixed_exclude_keywords, *plan.dynamic_exclude_keywords])
        plan.directed_source_search_templates = unique_strings([*plan.directed_source_search_templates, *templates])
        new_terms.extend(terms)
        for row in card.get("term_reasons") or []:
            term = textify(row.get("term"))
            if not term:
                continue
            reason_row = {
                "term": term,
                "source_field": textify(row.get("source_field")) or source_field_for_term(term, plan),
                "expected_source": textify(row.get("expected_source")) or expected_source_for_term(term),
                "expected_evidence": textify(row.get("expected_evidence")) or expected_evidence_for_term(term),
                "must_bind_competitor": "yes",
                "noise_risk": textify(row.get("noise_risk")) or "中：来自历史核验卡片，仍需绑定当前竞品和可追溯来源",
                "priority": textify(row.get("priority")) or "P1",
                "generated_by": "search_card",
            }
            new_reason_rows.append(reason_row)
        applied.append(
            {
                "product_type_key": textify(card.get("product_type_key")),
                "product_type_label": label,
                "confidence": textify(card.get("confidence")),
                "training_rows": textify(card.get("training_rows")),
                "source_path": textify(card.get("_source_path")),
            }
        )

    if new_terms:
        fallback_reasons = search_term_reason_rows(plan, new_terms, "search_card")
        plan.search_term_reasons = merge_search_term_reasons(plan.search_term_reasons, [*new_reason_rows, *fallback_reasons])
    if applied:
        plan.search_cards_applied = unique_dicts_by_key([*plan.search_cards_applied, *applied], "product_type_key")
        plan.source_policy_notes = unique_strings(
            [
                *plan.source_policy_notes,
                f"已加载 {len(applied)} 张历史搜索卡片：用于补充同类产品的搜索词、可复用来源和排除词；当前竞品事实仍以本轮可追溯页面为准。",
            ]
        )
    return plan


def unique_dicts_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for row in rows:
        value = textify(row.get(key))
        if not value or value in seen:
            continue
        seen.add(value)
        output.append({str(item_key): json_safe(item_value) for item_key, item_value in row.items()})
    return output


def source_kind_for_result(
    item: SearchResult,
    official_domains: Dict[str, Dict[str, int]],
) -> str:
    domain = domain_of(item.url)
    if not item.url:
        return "missing_url"
    if not is_probably_html_page(item.url):
        return "non_html_asset"
    adapter_info = classify_source_url(item.url)
    if adapter_info.get("source_family") in {"app_store", "developer_source", "launch_database"}:
        return "trusted_public"
    if adapter_info.get("source_family") in {"video_social", "social_app", "forum_community"}:
        return "community_or_social_signal"
    if is_official_domain_for(item.competitor, item.url, official_domains):
        if item.category == "official_expansion":
            return "official_expansion"
        if official_core_signal(item.url, item.title, item.snippet):
            return "official_core"
        return "official_public"
    if domain_matches(domain, APP_STORE_DOMAINS):
        return "trusted_public"
    if domain_matches(domain, HIGH_VALUE_PUBLIC_CRAWL_DOMAINS):
        return "trusted_public"
    if domain_matches(domain, THIRD_PARTY_CRAWL_SKIP_DOMAINS):
        return "third_party_verification_source"
    if domain_matches(domain, LOW_VALUE_CRAWL_SKIP_DOMAINS | LOW_VALUE_DOMAINS | LOW_VALUE_PORTAL_DOMAINS):
        return "low_value_or_aggregator"
    if domain_matches(domain, NOISE_DOMAINS):
        return "community_or_social_signal"
    return "public_web"


def page_role_for_result(
    item: SearchResult,
    plan: Optional[ProductCollectionPlan] = None,
) -> str:
    haystack = f"{item.url} {item.title} {item.snippet} {item.query}".lower()
    path = urlparse(item.url).path.lower()
    domain = domain_of(item.url)
    if not is_probably_html_page(item.url):
        return "asset_or_download"
    if login_gate_confirmed_by_url_title_or_form(item.url, item.title, item.snippet):
        return "auth_or_account_shell"
    if any(token in haystack for token in ("cart", "checkout", "coupon", "已买到的宝贝", "购物车", "卖家中心", "开店")):
        return "transaction_or_marketplace_shell"
    if domain_matches(domain, APP_STORE_DOMAINS):
        return "app_store_listing"
    if domain_matches(domain, VIDEO_SOCIAL_DOMAINS) or keyword_hits(haystack, ["youtube", "tiktok", "douyin", "bilibili", "video", "demo", "walkthrough", "演示", "评测视频", "视频"]):
        return "video_or_social_content"
    if domain_matches(domain, FORUM_COMMUNITY_DOMAINS) or keyword_hits(haystack, ["reddit", "知乎", "v2ex", "hacker news", "论坛", "帖子", "问答"]):
        return "forum_or_community_discussion"
    if any(token in path for token in ("/pricing", "/price", "/plans")) or any(token in haystack for token in ("pricing", "plans", "定价", "价格", "套餐")):
        return "pricing_packaging"
    if any(token in path for token in ("/docs", "/documentation", "/api", "/developers", "/developer", "/sdk")) or any(token in haystack for token in ("api", "sdk", "documentation", "developer", "文档", "接口")):
        return "docs_api_or_developer"
    if any(token in path for token in ("/security", "/trust", "/privacy", "/legal", "/compliance")) or any(token in haystack for token in ("security", "trust", "soc2", "privacy", "安全", "隐私", "合规")):
        return "security_trust_or_legal"
    if any(token in path for token in ("/changelog", "/release", "/updates")) or any(token in haystack for token in ("changelog", "release notes", "updates", "更新日志", "发布说明")):
        return "changelog_or_release"
    if any(token in path for token in ("/customers", "/customer", "/case", "/stories", "/use-cases", "/solutions")) or any(token in haystack for token in ("case study", "customers", "solutions", "客户案例", "解决方案", "场景")):
        return "customer_or_solution"
    if any(token in path for token in ("/features", "/feature", "/product", "/products", "/specs", "/specifications", "/technology")) or any(token in haystack for token in ("features", "specifications", "specs", "product details", "功能", "规格", "参数")):
        return "product_specs_or_features"
    if plan and plan.category in {"snow_helmet", "physical_product"}:
        if keyword_hits(haystack, ["size chart", "weight", "material", "colors", "warranty", "certification", "尺码", "重量", "材质", "颜色", "认证", "质保"]):
            return "physical_product_detail"
    if plan and plan.category == "ai_software":
        if keyword_hits(haystack, ["integrations", "models", "workflow", "automation", "quota", "credits", "集成", "模型", "工作流", "额度"]):
            return "ai_capability_detail"
    if plan and plan.category == "autonomous_vehicle_robotaxi":
        if keyword_hits(
            haystack,
            [
                "robotaxi",
                "autonomous vehicle",
                "self-driving",
                "driverless",
                "service area",
                "city coverage",
                "sensor",
                "lidar",
                "compute",
                "fleet",
                "permit",
                "safety report",
                "无人车",
                "自动驾驶",
                "运营城市",
                "服务范围",
                "传感器",
                "算力",
                "安全报告",
                "监管许可",
                "试乘",
            ],
        ):
            return "autonomous_vehicle_detail"
    if any(token in haystack for token in ("review", "alternatives", "compare", "comparison", "评价", "评测", "替代品", "对比")):
        return "review_or_comparison"
    if any(token in haystack for token in ("blog", "news", "press", "funding", "launch", "融资", "发布", "新闻")):
        return "news_or_blog"
    return "general_candidate"


def evidence_decision_for_result(
    item: SearchResult,
    official_domains: Dict[str, Dict[str, int]],
    collection_plan: Optional[ProductCollectionPlan],
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
    selected: bool = False,
    rank: int = 0,
    max_pages_per_competitor: int = 0,
    ml_model: Optional[LocalFilterModel] = None,
    ml_auto_include_threshold: float = 0.75,
    ml_auto_exclude_threshold: float = 0.80,
) -> EvidenceDecision:
    haystack = f"{item.url} {item.title} {item.snippet} {item.query} {item.engine}"
    lower_haystack = haystack.lower()
    source_kind = source_kind_for_result(item, official_domains)
    page_role = page_role_for_result(item, collection_plan)
    is_html = is_probably_html_page(item.url)
    is_official_source = source_kind.startswith("official")
    pre_crawl_reason = pre_crawl_rejection_reason(item.url, item.title, item.snippet, item.competitor)
    manual_include_hits = keyword_hits(haystack, manual_include_keywords)
    manual_exclude_hits = keyword_hits(haystack, manual_exclude_keywords)
    plan_exclude_hits = keyword_hits(haystack, (collection_plan.exclude_keywords if collection_plan else []))
    evidence_hits = keyword_hits(haystack, collection_plan_evidence_keywords(collection_plan, manual_include_keywords))
    field_hits = product_field_hits(haystack, collection_plan)
    competitor_terms = competitor_relevance_terms(item.competitor)
    binding_haystack = f"{item.url} {item.title} {item.snippet} {item.engine}"
    competitor_hits = keyword_hits(binding_haystack, [item.competitor, *competitor_terms])

    relevance_score = 0
    evidence_score = 0
    pm_value_score = 0
    traceability_score = 0
    category_fit_score = 0
    reasons: List[str] = []
    rejection_code = ""

    if is_official_source:
        relevance_score += 4
        evidence_score += 4
        traceability_score += 5
        reasons.append("source: official-domain evidence candidate")
    elif source_kind == "trusted_public":
        relevance_score += 2
        evidence_score += 2
        traceability_score += 3
        reasons.append("source: trusted public validation source")
    elif source_kind == "third_party_verification_source":
        evidence_score += 1
        traceability_score += 2
        reasons.append("source: third-party verification source; use only as validation")
    elif source_kind == "community_or_social_signal":
        traceability_score += 1
        reasons.append("source: community/social signal; low-confidence only")
    elif source_kind == "low_value_or_aggregator":
        evidence_score -= 2
        reasons.append("source: low-value aggregator or portal")
    else:
        evidence_score += 1
        traceability_score += 1

    if competitor_hits:
        relevance_score += min(5, 2 + len(competitor_hits))
        reasons.append("brand_match: " + ", ".join(competitor_hits[:5]))
    elif not is_official_source:
        relevance_score -= 3
        reasons.append("brand_match: no distinctive competitor evidence in title/url/snippet")

    core_roles = {
        "pricing_packaging",
        "docs_api_or_developer",
        "security_trust_or_legal",
        "changelog_or_release",
        "customer_or_solution",
        "product_specs_or_features",
        "physical_product_detail",
        "ai_capability_detail",
        "autonomous_vehicle_detail",
        "app_store_listing",
    }
    signal_roles = {"video_or_social_content", "forum_or_community_discussion"}
    if page_role in core_roles:
        evidence_score += 2
        pm_value_score += 3
        reasons.append(f"page_role: {page_role}")
    elif page_role in signal_roles:
        evidence_score += 1
        pm_value_score += 2
        reasons.append(f"page_role: {page_role}; directed source signal, requires traceable review")
    elif page_role in {"review_or_comparison", "news_or_blog"}:
        evidence_score += 1
        pm_value_score += 1
        reasons.append(f"page_role: {page_role}; validate, do not replace official facts")
    elif page_role in {"auth_or_account_shell", "transaction_or_marketplace_shell"}:
        evidence_score -= 4
        pm_value_score -= 3
        reasons.append(f"page_role: {page_role}; likely shell/noise")
    else:
        reasons.append(f"page_role: {page_role}")

    if evidence_hits:
        pm_value_score += min(4, len(evidence_hits))
        evidence_score += min(3, len(evidence_hits))
        reasons.append("evidence_keyword_hits: " + ", ".join(evidence_hits[:8]))
    if field_hits:
        category_fit_score += min(5, len(field_hits) + 1)
        pm_value_score += min(4, len(field_hits))
        reasons.append("product_field_hits: " + "、".join(field_hits[:8]))
    if manual_include_hits:
        pm_value_score += 3
        category_fit_score += 2
        reasons.append("manual_include_hits: " + ", ".join(manual_include_hits[:8]))
    if manual_exclude_hits:
        reasons.append("manual_exclude_hits: " + ", ".join(manual_exclude_hits[:8]))
    elif plan_exclude_hits and page_role in {"auth_or_account_shell", "transaction_or_marketplace_shell", "general_candidate"}:
        reasons.append("preset_exclude_hits: " + ", ".join(plan_exclude_hits[:8]))

    if official_core_signal(item.url, item.title, item.snippet):
        evidence_score += 2
        pm_value_score += 2
        reasons.append("official_core_signal: URL/title/snippet looks like core PM evidence")
    if item.category == "official_expansion":
        traceability_score += 1
        reasons.append("official_expansion: generated from discovered official domain")
    if len(item.url) < 90:
        traceability_score += 1
    if pre_crawl_reason:
        reasons.append("pre_crawl_gate: " + pre_crawl_reason)

    directed_source_value_signal = (
        page_role in {"video_or_social_content", "forum_or_community_discussion", "app_store_listing"}
        and bool(competitor_hits or manual_include_hits)
        and page_has_decision_signal(page_role, lower_haystack, evidence_hits, field_hits, manual_include_hits)
    )

    if not item.url:
        rejection_code = "rejected_missing_url"
    elif not is_html:
        rejection_code = "rejected_non_html_asset_kept_as_source"
    elif manual_exclude_hits:
        rejection_code = "rejected_manual_exclude_keyword:" + ",".join(manual_exclude_hits[:5])
    elif page_role in {"auth_or_account_shell", "transaction_or_marketplace_shell"}:
        rejection_code = "rejected_auth_or_transaction_shell"
    elif pre_crawl_reason and not is_official_source and not manual_include_hits and not directed_source_value_signal:
        rejection_code = pre_crawl_reason
    elif source_kind in {"low_value_or_aggregator", "community_or_social_signal"} and not manual_include_hits and not directed_source_value_signal:
        rejection_code = "rejected_low_evidence_source_policy"
    elif relevance_score < 1 and not is_official_source and not manual_include_hits:
        rejection_code = "rejected_no_brand_or_product_evidence"
    elif evidence_score < 1 and not field_hits and not manual_include_hits:
        rejection_code = "rejected_no_extractable_pm_evidence"

    if selected:
        decision_status = "selected"
        gate_result = "crawl_selected"
        hard_gate = "passed_selected_for_crawl"
    elif rejection_code:
        decision_status = "rejected"
        gate_result = "hard_reject"
        hard_gate = rejection_code
    elif page_role in {"review_or_comparison", "news_or_blog", "video_or_social_content", "forum_or_community_discussion"} or source_kind in {"third_party_verification_source", "community_or_social_signal"}:
        decision_status = "signal"
        gate_result = "keep_as_low_or_mid_confidence_signal"
        hard_gate = "not_primary_evidence"
    elif is_official_source or field_hits or manual_include_hits or pm_value_score >= 4:
        decision_status = "accepted"
        gate_result = "accepted_not_crawled_or_over_budget"
        hard_gate = "passed_but_not_selected_by_budget"
    else:
        decision_status = "signal"
        gate_result = "weak_signal"
        hard_gate = "needs_more_context"

    total = relevance_score + evidence_score + pm_value_score + traceability_score + category_fit_score
    if decision_status == "rejected":
        confidence = "低信心/拒绝"
    elif is_official_source and total >= 13:
        confidence = "高信心"
    elif total >= 9:
        confidence = "中信心"
    else:
        confidence = "低信心/线索"

    if max_pages_per_competitor and rank > max_pages_per_competitor * 3 and decision_status not in {"selected", "accepted"}:
        reasons.append(f"rank_budget: rank {rank} is outside broad audit band for budget {max_pages_per_competitor}")

    fact_type = fact_type_for(page_role, field_hits, haystack)
    increment_type = increment_type_for(page_role, field_hits, haystack)
    value_signals, value_missing, value_verdict = value_judgment_for_result(
        item,
        source_kind,
        page_role,
        lower_haystack,
        competitor_hits,
        evidence_hits,
        field_hits,
        manual_include_hits,
        rejection_code,
        is_html,
    )
    gui_review_candidate, gui_review_value_reason = gui_review_policy_for(
        page_role,
        value_verdict,
        source_kind,
        selected,
        rejection_code,
    )
    if value_signals:
        reasons.append("value_signals: " + "、".join(value_signals))
    if value_missing and value_verdict == "low_value_or_noise":
        reasons.append("value_missing: " + "、".join(value_missing))
    if gui_review_candidate:
        reasons.append("gui_review_candidate: " + gui_review_value_reason)
    pending_verification, verification_reason = verification_policy_for(
        source_kind,
        page_role,
        decision_status,
        pre_crawl_reason,
        evidence_hits,
        field_hits,
        rejection_code,
    )
    if gui_review_candidate:
        pending_verification = True
        verification_reason = gui_review_value_reason
    source_policy_tier = source_policy_tier_for(source_kind, page_role)
    fact_group = fact_group_for(item.competitor, fact_type, field_hits, page_role, item.url, item.title)
    ml_label = ""
    ml_include_score = ""
    ml_exclude_score = ""
    ml_verify_later_score = ""
    ml_confidence = ""
    ml_reason = ""
    ml_model_version = ""
    ml_adjustment = ""
    if ml_model:
        prediction = ml_model.predict(
            {
                "competitor": item.competitor,
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "query": item.query,
                "engine": item.engine,
                "source_kind": source_kind,
                "page_role": page_role,
                "source_policy_tier": source_policy_tier,
                "fact_type": fact_type,
                "increment_type": increment_type,
                "decision_status": decision_status,
                "gate_result": gate_result,
                "hard_gate": hard_gate,
                "confidence": confidence,
                "pending_verification": "yes" if pending_verification else "no",
                "primary_evidence_candidate": "pending",
                "matched_fields": "、".join(field_hits),
                "matched_include_keywords": ", ".join(manual_include_hits or evidence_hits[:8]),
                "matched_exclude_keywords": ", ".join(manual_exclude_hits or plan_exclude_hits[:8]),
                "reason": " | ".join(reasons),
            }
        )
        adjusted = apply_ml_prediction_to_decision(
            {
                "decision_status": decision_status,
                "gate_result": gate_result,
                "hard_gate": hard_gate,
                "source_policy_tier": source_policy_tier,
                "pending_verification": "yes" if pending_verification else "no",
                "verification_reason": verification_reason,
            },
            prediction,
            auto_include_threshold=ml_auto_include_threshold,
            auto_exclude_threshold=ml_auto_exclude_threshold,
        )
        decision_status = textify(adjusted.get("decision_status") or decision_status)
        gate_result = textify(adjusted.get("gate_result") or gate_result)
        hard_gate = textify(adjusted.get("hard_gate") or hard_gate)
        if adjusted.get("pending_verification") == "yes":
            pending_verification = True
            verification_reason = textify(adjusted.get("verification_reason") or verification_reason)
        ml_label = textify(adjusted.get("ml_label"))
        ml_include_score = textify(adjusted.get("ml_include_score"))
        ml_exclude_score = textify(adjusted.get("ml_exclude_score"))
        ml_verify_later_score = textify(adjusted.get("ml_verify_later_score"))
        ml_confidence = textify(adjusted.get("ml_confidence"))
        ml_reason = textify(adjusted.get("ml_reason"))
        ml_model_version = textify(adjusted.get("ml_model_version"))
        ml_adjustment = textify(adjusted.get("ml_adjustment"))
        if ml_adjustment and ml_adjustment != "none":
            reasons.append(f"local_ml_filter: {ml_adjustment}; {ml_label}; include_score={ml_include_score}")

    primary_evidence_candidate, primary_evidence_reason = primary_evidence_policy_for(
        source_kind,
        page_role,
        decision_status,
        pending_verification,
    )

    return EvidenceDecision(
        source_kind=source_kind,
        page_role=page_role,
        decision_status=decision_status,
        gate_result=gate_result,
        hard_gate=hard_gate,
        relevance_score=max(-5, min(5, relevance_score)),
        evidence_score=max(-5, min(5, evidence_score)),
        pm_value_score=max(-5, min(5, pm_value_score)),
        traceability_score=max(0, min(5, traceability_score)),
        category_fit_score=max(0, min(5, category_fit_score)),
        confidence=confidence,
        matched_fields=field_hits,
        matched_include_keywords=manual_include_hits or evidence_hits[:8],
        matched_exclude_keywords=manual_exclude_hits or plan_exclude_hits[:8],
        rejection_code=rejection_code,
        reasons=unique_strings(reasons),
        pending_verification=pending_verification,
        verification_reason=verification_reason,
        source_policy_tier=source_policy_tier,
        fact_type=fact_type,
        increment_type=increment_type,
        fact_group=fact_group,
        primary_evidence_candidate=primary_evidence_candidate,
        primary_evidence_reason=primary_evidence_reason,
        ml_label=ml_label,
        ml_include_score=ml_include_score,
        ml_exclude_score=ml_exclude_score,
        ml_verify_later_score=ml_verify_later_score,
        ml_confidence=ml_confidence,
        ml_reason=ml_reason,
        ml_model_version=ml_model_version,
        ml_adjustment=ml_adjustment,
        value_signals=value_signals,
        value_missing=value_missing,
        value_verdict=value_verdict,
        gui_review_candidate=gui_review_candidate,
        gui_review_value_reason=gui_review_value_reason,
    )


def is_probably_html_page(url: str) -> bool:
    path = urlparse(url).path.lower()
    blocked_suffixes = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".pdf",
        ".zip",
        ".mp4",
        ".mov",
        ".avi",
        ".css",
        ".js",
    )
    return not path.endswith(blocked_suffixes)


def count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if phrase and phrase.lower() in text.lower())


def boilerplate_score(text: str) -> int:
    if not text:
        return 0
    phrase_hits = count_phrase_hits(text, BOILERPLATE_PHRASES) + count_phrase_hits(text, FORUM_NAV_PHRASES)
    pattern_hits = sum(1 for pattern in BOILERPLATE_PATTERNS if pattern.search(text))
    return phrase_hits + pattern_hits * 2


def strip_boilerplate_phrases(text: str) -> str:
    for phrase in sorted(BOILERPLATE_PHRASES | FORUM_NAV_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")
    return text


def is_boilerplate_line(line: str) -> bool:
    stripped = re.sub(r"\s+", "", line)
    if not stripped:
        return True
    score = boilerplate_score(line)
    if score >= 4:
        return True
    if len(stripped) <= 80 and score >= 2:
        return True
    if len(stripped) <= 30 and any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
        return True
    return False


def competitor_has_relevance(competitor: str, *values: str) -> bool:
    haystack = " ".join(values).lower()
    compact_haystack = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    compact_competitor = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", competitor.lower())
    if compact_competitor and compact_competitor in compact_haystack:
        return True
    return any(term in haystack or term in compact_haystack for term in competitor_relevance_terms(competitor))


def competitor_strong_binding(competitor: str, *values: str) -> bool:
    haystack = " ".join(values).lower()
    compact_haystack = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    compact_competitor = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", competitor.lower())
    if compact_competitor and compact_competitor in compact_haystack:
        return True
    if ".ai" in competitor.lower():
        return False
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", competitor.lower())
    semantic_tokens = [
        token
        for token in raw_tokens
        if len(token) > 1 and token not in GENERIC_COMPETITOR_TERMS
    ]
    if len(semantic_tokens) > 1:
        if any(len(token) <= 2 for token in semantic_tokens):
            return False
        return all(token in haystack or token in compact_haystack for token in semantic_tokens)
    return competitor_has_relevance(competitor, *values)


def low_value_url_reason(url: str, title: str = "", snippet: str = "", competitor: str = "") -> str:
    parsed = urlparse(url)
    domain = domain_of(url)
    haystack = f"{url} {title} {snippet}".lower()
    path_query = f"{parsed.path} {parsed.query}".lower()

    if domain_matches(domain, LOW_VALUE_DOMAINS):
        return "rejected_low_value_host"

    token_hits = sorted(token for token in LOW_VALUE_URL_TOKENS if token in path_query)
    if token_hits:
        return "rejected_low_value_url_token:" + ",".join(token_hits[:5])

    portal_boilerplate = domain_matches(domain, LOW_VALUE_PORTAL_DOMAINS) and boilerplate_score(f"{title} {snippet}") >= 3
    if portal_boilerplate and not competitor_has_relevance(competitor, title, snippet, url):
        return "rejected_portal_boilerplate_low_relevance"

    return ""


def official_core_signal(url: str, title: str = "", snippet: str = "") -> bool:
    parsed = urlparse(url)
    haystack = f"{parsed.path} {parsed.query} {title} {snippet}".lower()
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    return any(keyword in haystack or keyword in compact for keyword in OFFICIAL_CORE_PATH_KEYWORDS)


def pre_crawl_rejection_reason(url: str, title: str = "", snippet: str = "", competitor: str = "") -> str:
    low_value = low_value_url_reason(url, title, snippet, competitor)
    if low_value:
        return low_value
    domain = domain_of(url)
    if domain_matches(domain, THIRD_PARTY_CRAWL_SKIP_DOMAINS):
        return "rejected_third_party_non_official_source"
    if domain_matches(domain, LOW_VALUE_CRAWL_SKIP_DOMAINS):
        return "rejected_low_value_public_aggregator"
    return ""


def competitor_relevance_terms(competitor: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", competitor.lower())
    distinctive = [token for token in tokens if len(token) > 2 and token not in GENERIC_COMPETITOR_TERMS]
    return distinctive or [token for token in tokens if token not in GENERIC_COMPETITOR_TERMS]


def competitor_domain_terms(competitor: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", competitor.lower())
    terms = [
        token
        for token in tokens
        if len(token) > 2 and token not in GENERIC_COMPETITOR_TERMS and token not in COMMON_DOMAIN_PARTS
    ]
    return terms or competitor_relevance_terms(competitor)


def extract_inline_urls(text: str) -> List[str]:
    urls = []
    for match in URL_IN_TEXT_RE.findall(text or ""):
        cleaned = match.rstrip(".,;:!?)]}）】》\"'")
        normalized = normalize_url(cleaned)
        if normalized and is_probably_html_page(normalized):
            urls.append(normalized)
    return sorted(set(urls))


def domain_parts(domain: str) -> List[str]:
    return [part for part in domain.lower().split(".") if part and part not in COMMON_DOMAIN_PARTS]


def competitor_input_domain(competitor: str) -> str:
    value = competitor.strip()
    if not value or any(ch.isspace() for ch in value):
        inline = extract_inline_urls(value)
        return domain_of(inline[0]) if inline else ""
    if "." not in value:
        return ""
    normalized = normalize_url(value)
    domain = domain_of(normalized)
    return domain if "." in domain else ""


def guessed_official_domains(competitor: str) -> List[str]:
    terms = competitor_domain_terms(competitor)
    if not terms:
        return []
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", competitor.lower())
    if len(raw_tokens) > 2 or len(terms) > 2:
        return []
    base = "".join(terms)
    if len(base) < 3:
        return []
    return [f"{base}.com", f"{base}.ai", f"{base}.io", f"{base}.app"]


def official_domain_confidence(competitor: str, url: str, title: str = "", snippet: str = "") -> int:
    domain = domain_of(url)
    if not domain:
        return 0
    if domain_matches(domain, THIRD_PARTY_CRAWL_SKIP_DOMAINS | LOW_VALUE_CRAWL_SKIP_DOMAINS | NOISE_DOMAINS):
        return 0

    terms = competitor_domain_terms(competitor)
    parts = domain_parts(domain)
    first_part = parts[0] if parts else ""
    haystack = f"{url} {title} {snippet}".lower()
    compact_domain = re.sub(r"[^a-z0-9]+", "", domain)
    compact_competitor = re.sub(r"[^a-z0-9]+", "", competitor.lower())
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", competitor.lower())
    semantic_tokens = [
        token
        for token in raw_tokens
        if len(token) > 1 and token not in GENERIC_COMPETITOR_TERMS
    ]
    input_domain = competitor_input_domain(competitor)
    input_domain_match = bool(input_domain and domain_matches(domain, {input_domain}))
    domain_brand_match = any(term and any(term in part for part in parts) for term in terms)
    primary_brand_match = bool(terms and terms[0] and any(terms[0] in part for part in parts))
    if len(semantic_tokens) > 1 and not input_domain_match and not competitor_strong_binding(competitor, url, title, snippet):
        return 0
    if terms and not domain_brand_match and not input_domain_match and compact_competitor != compact_domain:
        return 0
    if len(terms) > 1 and terms[0] and not primary_brand_match and not input_domain_match:
        return 0

    score = 0
    if input_domain_match:
        score += 5
    if compact_competitor and compact_competitor == compact_domain:
        score += 5
    elif compact_competitor and compact_competitor in compact_domain and competitor_strong_binding(competitor, url, title, snippet):
        score += 2
    if first_part and first_part in terms:
        score += 4
    elif terms and terms[0] and terms[0] in first_part:
        score += 3
    elif any(term and term in first_part for term in terms):
        score += 1
    elif domain_brand_match:
        score += 3
    if official_core_signal(url, title, snippet):
        score += 1
    if any(hint in haystack for hint in OFFICIAL_DISCOVERY_HINTS):
        score += 2
    if len(domain.split(".")) <= 2:
        score += 1
    elif first_part in terms:
        score -= 1
    if "-" in first_part and first_part not in terms:
        score -= 2
    return max(score, 0)


def official_domains_by_competitor(
    competitors: Sequence[str],
    web_results: Sequence[SearchResult],
) -> Dict[str, Dict[str, int]]:
    domains: Dict[str, Dict[str, int]] = {competitor: {} for competitor in competitors}
    for competitor in competitors:
        input_domain = competitor_input_domain(competitor)
        if input_domain:
            domains.setdefault(competitor, {})[input_domain] = 6

    for item in web_results:
        if item.category == "official_expansion":
            continue
        candidates = [item.url, *extract_inline_urls(f"{item.title} {item.snippet}")]
        for url in candidates:
            confidence = official_domain_confidence(item.competitor, url, item.title, item.snippet)
            if confidence < 4:
                continue
            domain = domain_of(url)
            current = domains.setdefault(item.competitor, {}).get(domain, 0)
            domains[item.competitor][domain] = max(current, confidence)
    for competitor in competitors:
        if domains.get(competitor):
            continue
        guesses = guessed_official_domains(competitor)
        if guesses:
            domains.setdefault(competitor, {})[guesses[0]] = 4
    return domains


def is_official_domain_for(competitor: str, url: str, official_domains: Dict[str, Dict[str, int]]) -> bool:
    domain = domain_of(url)
    return bool(domain and domain_matches(domain, set(official_domains.get(competitor, {}).keys())))


def official_path_priority(url: str) -> int:
    path = urlparse(url).path.strip("/")
    first = path.split("/", 1)[0] if path else ""
    return OFFICIAL_PATH_PRIORITY.get(first, 50)


def official_path_priority_for_plan(url: str, collection_plan: Optional[ProductCollectionPlan] = None) -> int:
    if not collection_plan:
        return official_path_priority(url)
    path = urlparse(url).path.lower()
    first = path.strip("/").split("/", 1)[0]
    base = official_path_priority(url)
    if collection_plan.category in {"snow_helmet", "physical_product"}:
        if first in {"product", "products", "helmets", "helmet"} or any(token in path for token in ("/products/", "/product/", "helmet", "helmets")):
            return 0
        if first in {"specs", "specifications"} or any(token in path for token in ("specs", "specifications")):
            return 1
        if first in {"technology", "technologies", "materials", "material"} or any(token in path for token in ("technology", "technologies", "materials", "material")):
            return 2
        if first in {"size-guide", "size-chart", "sizeguide"} or any(token in path for token in ("size-guide", "size-chart", "sizeguide")):
            return 3
        if first in {"pricing", "price", "plans"} or any(token in path for token in ("pricing", "price", "plans")):
            return 4
        return base
    if collection_plan.category == "ai_software":
        if first in {"api", "docs", "developers", "developer", "sdk"} or any(token in path for token in ("api", "docs", "developers", "developer", "sdk")):
            return 0
        if first in {"integrations", "integration", "plugins", "connectors", "extensions"} or any(token in path for token in ("integrations", "integration", "plugins", "connectors", "extensions")):
            return 1
        if first in {"pricing", "price", "plans"} or any(token in path for token in ("pricing", "price", "plans")):
            return 2
        if first in {"security", "trust", "privacy"} or any(token in path for token in ("security", "trust", "privacy")):
            return 3
        return base
    return base


def enrich_web_results(
    competitors: Sequence[str],
    web_results: Sequence[SearchResult],
) -> List[SearchResult]:
    enriched = list(web_results)
    for item in web_results:
        for url in extract_inline_urls(f"{item.title} {item.snippet}"):
            enriched.append(
                SearchResult(
                    competitor=item.competitor,
                    category="discovered_url",
                    query=item.query,
                    title=f"Discovered URL from search result: {truncate_text(item.title, 120)}",
                    url=url,
                    snippet=f"Found inside source {item.url}. {truncate_text(item.snippet, 500)}",
                    engine=item.engine or "inline_url",
                )
            )

    enriched = dedupe_results(enriched)
    official_domains = official_domains_by_competitor(competitors, enriched)
    for competitor, domain_scores in official_domains.items():
        for domain, confidence in sorted(domain_scores.items(), key=lambda item: (-item[1], item[0])):
            scheme = "https"
            source_url = next((item.url for item in enriched if item.competitor == competitor and domain_of(item.url) == domain), "")
            if source_url and urlparse(source_url).scheme:
                scheme = urlparse(source_url).scheme
            base = f"{scheme}://{domain}"
            for path in OFFICIAL_CORE_PATHS:
                url = base if not path else f"{base}/{path}"
                label = "homepage" if not path else f"/{path}"
                enriched.append(
                    SearchResult(
                        competitor=competitor,
                        category="official_expansion",
                        query="official core page expansion",
                        title=f"{competitor} official core page: {label}",
                        url=url,
                        snippet=f"Official-domain expansion from {domain} (confidence {confidence}).",
                        engine="official_expander",
                    )
                )
    return dedupe_results(enriched)


def score_reasons(url: str, title: str = "", snippet: str = "", competitor: str = "") -> List[str]:
    haystack = f"{url} {title} {snippet}".lower()
    compact_haystack = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    domain = domain_of(url)
    compact_competitor = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", competitor.lower())
    terms = competitor_relevance_terms(competitor)
    reasons = []
    low_value_reason = low_value_url_reason(url, title, snippet, competitor)
    pre_crawl_reason = pre_crawl_rejection_reason(url, title, snippet, competitor)
    if domain_matches(domain, NOISE_DOMAINS):
        reasons.append("penalty: noisy social/wiki domain")
    if pre_crawl_reason:
        reasons.append("reject: " + pre_crawl_reason)
    if is_probably_html_page(url):
        reasons.append("positive: crawlable html-like page")
    else:
        reasons.append("reject: non-html asset")
    if compact_competitor and compact_competitor in compact_haystack:
        reasons.append("positive: exact competitor phrase match")
    if terms:
        hits = [term for term in terms if term in haystack or term in compact_haystack]
        if hits:
            reasons.append("positive: distinctive term match " + ", ".join(hits))
        else:
            reasons.append("penalty: no distinctive competitor term")
    priority_hits = [keyword for keyword in PAGE_PRIORITY_KEYWORDS if keyword in haystack]
    if priority_hits:
        reasons.append("positive: PM signal keyword " + ", ".join(priority_hits[:5]))
    if official_core_signal(url, title, snippet):
        reasons.append("positive: official/core page signal")
    if len(url) < 90:
        reasons.append("positive: concise URL")
    if any(token in haystack for token in ["login", "signin", "signup", "auth"]):
        reasons.append("penalty: auth/login-like page")
    return reasons


def score_url(url: str, title: str = "", snippet: str = "", competitor: str = "") -> float:
    haystack = f"{url} {title} {snippet}".lower()
    compact_haystack = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    score = 0.0
    domain = domain_of(url)
    if domain_matches(domain, NOISE_DOMAINS):
        score -= 3
    pre_crawl_reason = pre_crawl_rejection_reason(url, title, snippet, competitor)
    if pre_crawl_reason:
        score -= 8
        if pre_crawl_reason == "rejected_third_party_non_official_source":
            score -= 5
    if is_probably_html_page(url):
        score += 1
    compact_competitor = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", competitor.lower())
    terms = competitor_relevance_terms(competitor)
    if compact_competitor and compact_competitor in compact_haystack:
        score += 5
    if terms:
        hits = sum(1 for term in terms if term in haystack or term in compact_haystack)
        if hits:
            score += 2.5 * hits
        else:
            score -= 4
    for idx, keyword in enumerate(PAGE_PRIORITY_KEYWORDS):
        if keyword in haystack:
            score += max(1.0, 3.0 - idx * 0.1)
    if official_core_signal(url, title, snippet):
        score += 4
    if len(url) < 90:
        score += 0.5
    if any(token in haystack for token in ["login", "signin", "signup", "auth"]):
        score -= 1
    return score


def dedupe_results(results: Iterable[SearchResult]) -> List[SearchResult]:
    best_by_key: Dict[str, SearchResult] = {}
    for item in results:
        key = canonical_url_for_dedupe(item.url)
        if not key:
            continue
        item.url = key
        item.score = score_url(item.url, item.title, item.snippet, item.competitor)
        current = best_by_key.get(key)
        if current is None or item.score > current.score:
            best_by_key[key] = item
        elif current is not None and item.query and item.query not in current.query:
            current.query = unique_strings([current.query, item.query])[0]
    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped


def searxng_search(
    base_url: str,
    query: str,
    category: str,
    language: str,
    limit: int,
    timeout: int,
    proxy_url: str = "",
) -> List[Dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/search"
    params = {
        "q": query,
        "categories": category,
        "language": language,
        "format": "json",
    }
    request_url = endpoint + "?" + urlencode(params)
    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "competitor-intel-harvester/1.0",
        },
    )
    if proxy_url and not is_local_url(base_url):
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
        response_context = opener.open(request, timeout=timeout)
    else:
        opener = build_opener(ProxyHandler({}))
        response_context = opener.open(request, timeout=timeout)
    with response_context as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("results", []))[:limit]


def searxng_config(base_url: str, timeout: int, proxy_url: str = "") -> Dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + "/config",
        headers={
            "Accept": "application/json",
            "User-Agent": "competitor-intel-harvester/1.0",
        },
    )
    if proxy_url and not is_local_url(base_url):
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("SearXNG /config did not return a JSON object")
    return payload


def searxng_categories(base_url: str, timeout: int, proxy_url: str = "") -> List[str]:
    try:
        payload = searxng_config(base_url, timeout, proxy_url)
    except Exception:
        return []
    categories = payload.get("categories") or []
    return [str(item) for item in categories if item]


def check_searxng(base_url: str, timeout: int, proxy_url: str = "") -> Tuple[bool, str]:
    try:
        searxng_config(base_url, timeout, proxy_url)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def seed_results_from_competitor_inputs(competitors: Sequence[str]) -> List[SearchResult]:
    seeds: List[SearchResult] = []
    for competitor in competitors:
        urls = extract_inline_urls(competitor)
        if not urls and competitor_input_domain(competitor):
            urls = [normalize_url(competitor)]
        for url in urls:
            if not is_probably_html_page(url):
                continue
            seeds.append(
                SearchResult(
                    competitor=competitor,
                    category="input_url",
                    query="user provided competitor URL",
                    title=f"{competitor} user provided URL",
                    url=url,
                    snippet="User provided this URL directly; use it as a crawl seed when public and crawlable.",
                    engine="input_seed",
                )
            )
    return dedupe_results(seeds)


def run_searches(
    searxng_url: str,
    competitors: Sequence[str],
    general_queries: Sequence[str],
    image_queries: Sequence[str],
    include_cn: bool,
    per_query: int,
    timeout: int,
    proxy_url: str = "",
) -> Tuple[List[SearchResult], List[SearchResult]]:
    web_results: List[SearchResult] = []
    image_results: List[SearchResult] = []

    for competitor in competitors:
        queries = list(general_queries)
        if include_cn:
            queries += DEFAULT_CN_QUERIES
        for template in queries:
            query = template.format(name=competitor)
            try:
                rows = searxng_search(
                    searxng_url,
                    query=query,
                    category="general",
                    language="all",
                    limit=per_query,
                    timeout=timeout,
                    proxy_url=proxy_url,
                )
                for row in rows:
                    web_results.append(
                        SearchResult(
                            competitor=competitor,
                            category="general",
                            query=query,
                            title=str(row.get("title", "")),
                            url=str(row.get("url", "")),
                            snippet=str(row.get("content", row.get("snippet", ""))),
                            engine=str(row.get("engine", "")),
                        )
                    )
            except Exception as exc:
                print(f"[warn] SearXNG general query failed: {query} ({exc})", file=sys.stderr)

        for template in image_queries:
            query = template.format(name=competitor)
            try:
                rows = searxng_search(
                    searxng_url,
                    query=query,
                    category="images",
                    language="all",
                    limit=per_query,
                    timeout=timeout,
                    proxy_url=proxy_url,
                )
                for row in rows:
                    image_url = (
                        row.get("img_src")
                        or row.get("thumbnail_src")
                        or row.get("thumbnail")
                        or row.get("url")
                        or ""
                    )
                    image_results.append(
                        SearchResult(
                            competitor=competitor,
                            category="images",
                            query=query,
                            title=str(row.get("title", "")),
                            url=str(image_url),
                            snippet=str(row.get("content", row.get("source", ""))),
                            engine=str(row.get("engine", "")),
                        )
                    )
            except Exception as exc:
                print(f"[warn] SearXNG image query failed: {query} ({exc})", file=sys.stderr)

    return dedupe_results(web_results), dedupe_results(image_results)


DISCOVERY_SKIP_NAMES = {
    "ai",
    "api",
    "app",
    "apps",
    "best",
    "blog",
    "apr",
    "aippt",
    "aug",
    "dec",
    "feb",
    "for",
    "free",
    "gif",
    "google",
    "jan",
    "jul",
    "jun",
    "mar",
    "may",
    "nov",
    "oct",
    "ios",
    "maker",
    "official",
    "pricing",
    "product",
    "review",
    "reviews",
    "software",
    "tools",
    "top",
    "web",
    "website",
    "youtube",
    "zhihu",
    "知乎",
}

DISCOVERY_DESCRIPTOR_WORDS = DISCOVERY_SKIP_NAMES | GENERIC_COMPETITOR_TERMS | {
    "builder",
    "creator",
    "deck",
    "decks",
    "generator",
    "maker",
    "ppt",
    "presentation",
    "presentations",
    "slideshow",
    "slides",
}

DISCOVERY_SOURCE_DOMAINS = {
    "alternativeto.net",
    "capterra.com",
    "example.com",
    "g2.com",
    "getapp.com",
    "producthunt.com",
    "reddit.com",
    "saasworthy.com",
    "slant.co",
    "sourceforge.net",
    "weixin.qq.com",
    "xiaohongshu.com",
    "zhihu.com",
    "trustradius.com",
    "trustpilot.com",
    "wikipedia.org",
    "youtube.com",
}


def build_competitor_discovery_queries(
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> List[str]:
    base = " ".join(
        part
        for part in [
            own_product_name,
            own_product_positioning,
            plan.category_label,
        ]
        if part
    ).strip()
    if not base:
        base = plan.category_label or "product"
    focused = truncate_text(re.sub(r"\s+", " ", base), 120)
    focused_terms = discovery_seed_terms(plan, own_product_name, own_product_positioning, own_product_context)
    terms = [
        f"{focused} competitors",
        f"{focused} alternatives",
        f"{focused} best tools",
        f"{focused} top products",
        f"{focused} comparison",
        f"{focused} 替代品",
        f"{focused} 竞品",
        f"{focused} 对比",
        f"{focused} site:producthunt.com",
        f"{focused} site:g2.com",
    ]
    if plan.category == "ai_software":
        terms.extend(
            [
                f"{focused} AI tools alternatives",
                f"{focused} SaaS competitors API integrations",
            ]
        )
    elif plan.category in {"physical_product", "snow_helmet"}:
        terms.extend(
            [
                f"{focused} review best products",
                f"{focused} specs comparison",
            ]
        )
    for seed in focused_terms:
        terms.extend(
            [
                f"{seed} competitors",
                f"{seed} alternatives",
                f"{seed} comparison",
                f"{seed} 竞品",
                f"{seed} 替代品",
            ]
        )
    return unique_strings(terms)


def discovery_seed_terms(
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> List[str]:
    context = " ".join([own_product_name, own_product_positioning, own_product_context, plan.category_label]).lower()
    compact = re.sub(r"\s+", "", context)
    seeds: List[str] = []
    if keyword_hits(compact, ["演示", "文稿", "汇报", "ppt"]) or keyword_hits(context, ["presentation", "slide", "deck"]):
        seeds += ["AI presentation maker", "AI slide deck generator", "AI PPT tool"]
    if keyword_hits(compact, ["竞品", "情报", "竞对"]) or keyword_hits(context, ["competitive intelligence", "competitor research"]):
        seeds += ["competitive intelligence tool", "competitor research tool"]
    if keyword_hits(compact, ["滑雪", "头盔", "雪盔", "全盔", "双板"]) or keyword_hits(context, ["ski helmet", "snow helmet"]):
        seeds += ["ski helmet", "full face ski helmet", "MIPS ski helmet"]
    if not seeds:
        seeds.append(plan.category_label)
    return unique_strings(seeds)


def discovery_context_terms(
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> List[str]:
    context = " ".join([own_product_name, own_product_positioning, own_product_context, plan.category_label]).lower()
    compact = re.sub(r"\s+", "", context)
    terms: List[str] = []
    if keyword_hits(compact, ["演示", "文稿", "汇报", "ppt"]) or keyword_hits(context, ["presentation", "slide", "deck"]):
        terms += ["presentation", "presentations", "slide", "slides", "deck", "ppt", "演示", "文稿", "汇报"]
    if keyword_hits(compact, ["竞品", "情报", "竞对"]) or keyword_hits(context, ["competitive intelligence", "competitor research"]):
        terms += ["competitive", "competitor", "intelligence", "research", "竞品", "情报", "竞对"]
    if keyword_hits(compact, ["滑雪", "头盔", "雪盔", "全盔", "双板"]) or keyword_hits(context, ["ski helmet", "snow helmet"]):
        terms += ["ski", "snow", "helmet", "mips", "slalom", "滑雪", "头盔", "雪盔", "全盔", "双板"]
    if not terms and plan.category == "ai_software":
        terms += ["ai", "software", "tool", "automation", "workflow", "api", "智能", "工具", "自动化"]
    if not terms and plan.category in {"physical_product", "snow_helmet"}:
        terms += ["product", "specs", "review", "material", "size", "产品", "参数", "评价"]
    return unique_strings(terms)


def discovery_context_fit(
    text: str,
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
) -> bool:
    terms = discovery_context_terms(plan, own_product_name, own_product_positioning, own_product_context)
    if not terms:
        return True
    low = textify(text).lower()
    compact = re.sub(r"\s+", "", low)
    return any(term.lower() in low or term.lower() in compact for term in terms)


def clean_candidate_name(value: str, own_product_name: str = "") -> str:
    name = re.sub(r"\s+", " ", textify(value)).strip(" -–—:：,，|")
    name = re.split(r"\s[-–—:：|]\s", name)[0].strip()
    name = re.sub(r"^(best|top|official|review|reviews|compare|comparison|alternative|alternatives)\s+", "", name, flags=re.I)
    name = re.sub(r"\s+(competitors?|alternatives?|reviews?|pricing|official|website|software|tools?|apps?)$", "", name, flags=re.I)
    name = name.strip(" .,-–—:：|")
    if not name or len(name) < 2 or len(name) > 60:
        return ""
    lowered = name.lower()
    if lowered in DISCOVERY_SKIP_NAMES:
        return ""
    descriptor_tokens = re.findall(r"[a-z0-9]+", lowered)
    if descriptor_tokens and all(token in DISCOVERY_DESCRIPTOR_WORDS for token in descriptor_tokens) and not re.search(r"[\u4e00-\u9fff]", name):
        return ""
    if own_product_name and lowered == own_product_name.lower():
        return ""
    if re.fullmatch(r"\d{2,}", name):
        return ""
    return name


def candidate_name_from_domain(url: str) -> str:
    domain = domain_of(url)
    if not domain:
        return ""
    if domain in DISCOVERY_SOURCE_DOMAINS or domain_matches(domain, FORUM_COMMUNITY_DOMAINS | NOISE_DOMAINS):
        return ""
    parts = [part for part in domain.split(".") if part]
    if not parts:
        return ""
    root = parts[-2] if len(parts) >= 2 else parts[0]
    if root in COMMON_DOMAIN_PARTS or root in DISCOVERY_SKIP_NAMES:
        return ""
    return root.capitalize()


def candidate_names_from_text(text: str, own_product_name: str = "") -> List[str]:
    names: List[str] = []
    normalized_text = re.sub(r"\s+(and|or|vs\.?|versus)\s+", "\n", textify(text), flags=re.I)
    prefix_match = re.match(r"\s*([^|:：\-–—]{2,40})\s*[-–—:：|]", normalized_text)
    if prefix_match:
        prefix = prefix_match.group(1).strip()
        if not re.search(r"[?？]|哪些|如何|怎么|什么", prefix):
            name = clean_candidate_name(prefix, own_product_name)
            if name:
                names.append(name)
    for segment in re.split(r"[\n,;/]+", normalized_text):
        for match in re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9]*(?:\.[A-Za-z]{2,})?(?:[ \t]+[A-Z][A-Za-z0-9]*(?:\.[A-Za-z]{2,})?){0,2})(?![A-Za-z0-9])", segment):
            name = clean_candidate_name(match, own_product_name)
            if name:
                names.append(name)
    return unique_strings(names)


def candidate_name_matches_domain(name: str, domain_name: str) -> bool:
    name_slug = slugify(name)
    domain_slug = slugify(domain_name)
    if not name_slug or not domain_slug:
        return False
    name_root = slugify(textify(name).split(".")[0])
    return name_slug == domain_slug or name_root == domain_slug


def classify_competitor_candidate(candidate: Mapping[str, Any], plan: ProductCollectionPlan) -> str:
    evidence = " ".join(
        textify(candidate.get(key))
        for key in ("evidence_title", "evidence_snippet", "discovered_query")
    ).lower()
    if keyword_hits(evidence, ["alternative", "alternatives", "competitor", "competitors", "替代", "竞品", "对比"]):
        return "直接竞品候选"
    if keyword_hits(evidence, ["review", "best", "top", "评测", "评价", "榜单"]):
        return "相邻竞品/待核实候选"
    return "待核实候选"


def confidence_for_candidate(source_count: int, official_url: str, name: str, evidence_text: str) -> str:
    has_official = bool(official_url and domain_of(official_url))
    if has_official and source_count >= 1:
        return "高信心"
    if source_count >= 2:
        return "中信心"
    if name and keyword_hits(evidence_text, ["alternative", "competitor", "替代", "竞品"]):
        return "中信心"
    return "低信心"


def extract_competitor_candidates_from_results(
    results: Sequence[SearchResult],
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
    max_candidates: int = 8,
) -> List[CompetitorCandidate]:
    grouped: Dict[str, Dict[str, Any]] = {}
    own_tokens = set(re.findall(r"[a-z0-9][a-z0-9.-]{1,40}|[\u4e00-\u9fff]{2,12}", own_product_name.lower()))
    for item in results:
        text = f"{item.title} {item.snippet}"
        evidence_without_query = f"{item.title} {item.snippet} {item.url}"
        context_fit = discovery_context_fit(
            evidence_without_query,
            plan,
            own_product_name,
            own_product_positioning,
            own_product_context,
        )
        source_domain = domain_of(item.url)
        source_domain_is_directory = bool(
            source_domain
            and (
                source_domain in DISCOVERY_SOURCE_DOMAINS
                or domain_matches(source_domain, FORUM_COMMUNITY_DOMAINS | NOISE_DOMAINS | LOW_VALUE_CRAWL_SKIP_DOMAINS)
            )
        )
        text_names = candidate_names_from_text(text, own_product_name)
        primary_title_name = text_names[0] if text_names else ""
        names = list(text_names)
        domain_name = candidate_name_from_domain(item.url)
        if domain_name and (not primary_title_name or candidate_name_matches_domain(primary_title_name, domain_name)):
            names.insert(0, domain_name)
        for raw_name in unique_strings(names):
            name = clean_candidate_name(raw_name, own_product_name)
            if not name:
                continue
            lowered = name.lower()
            if lowered in own_tokens or lowered in DISCOVERY_SKIP_NAMES:
                continue
            key = slugify(name)
            row = grouped.setdefault(
                key,
                {
                    "name": name,
                    "queries": [],
                    "urls": [],
                    "titles": [],
                    "snippets": [],
                    "official_url": "",
                    "context_fit_count": 0,
                },
            )
            row["queries"].append(item.query)
            row["urls"].append(item.url)
            row["titles"].append(item.title)
            row["snippets"].append(item.snippet)
            if context_fit:
                row["context_fit_count"] += 1
            domain_candidate = candidate_name_from_domain(item.url)
            if domain_candidate and candidate_name_matches_domain(name, domain_candidate) and not row["official_url"]:
                row["official_url"] = item.url
            elif (
                context_fit
                and not source_domain_is_directory
                and primary_title_name
                and slugify(primary_title_name) == slugify(name)
                and not row["official_url"]
            ):
                row["official_url"] = item.url

    candidates: List[CompetitorCandidate] = []
    for row in grouped.values():
        query = unique_strings(row["queries"])[0] if row["queries"] else ""
        url = unique_strings(row["urls"])[0] if row["urls"] else ""
        title = unique_strings(row["titles"])[0] if row["titles"] else ""
        snippet = unique_strings(row["snippets"])[0] if row["snippets"] else ""
        official_url = row["official_url"]
        source_count = len(set(row["urls"])) or 1
        context_fit_count = int(row.get("context_fit_count") or 0)
        evidence_text = f"{query} {title} {snippet}"
        confidence = confidence_for_candidate(source_count, official_url, row["name"], evidence_text)
        if context_fit_count <= 0:
            confidence = "低信心"
        status = "accepted" if official_url and context_fit_count > 0 and confidence in {"高信心", "中信心"} else "pending_verification"
        candidate_payload = {
            "evidence_title": title,
            "evidence_snippet": snippet,
            "discovered_query": query,
        }
        candidates.append(
            CompetitorCandidate(
                name=row["name"],
                candidate_type=classify_competitor_candidate(candidate_payload, plan),
                confidence=confidence,
                status=status,
                official_url=official_url,
                official_domain=domain_of(official_url),
                discovered_query=query,
                discovered_from_url=url,
                evidence_title=title,
                evidence_snippet=snippet,
                overlap_reason=(
                    "候选与本轮产品任务词、竞品/替代品搜索意图放在同一上下文；"
                    + ("已找到疑似官方入口。" if official_url else "暂未找到官方入口，进入待核实。")
                ),
                source_count=source_count,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            {"高信心": 0, "中信心": 1, "低信心": 2}.get(candidate.confidence, 3),
            -candidate.source_count,
            candidate.name.lower(),
        )
    )
    return candidates[: max(0, max_candidates)]


def build_candidate_official_lookup_queries(candidate_name: str) -> List[str]:
    name = textify(candidate_name).strip()
    if not name:
        return []
    return unique_strings(
        [
            f"{name} official website",
            f"{name} official site",
            f"{name} 官网",
            f"{name} pricing",
        ]
    )


def run_competitor_discovery(
    searxng_url: str,
    plan: ProductCollectionPlan,
    own_product_name: str,
    own_product_positioning: str,
    own_product_context: str,
    per_query: int,
    timeout: int,
    proxy_url: str,
    max_candidates: int,
) -> Tuple[List[CompetitorCandidate], List[SearchResult]]:
    discovery_results: List[SearchResult] = []
    for query in build_competitor_discovery_queries(plan, own_product_name, own_product_positioning, own_product_context):
        try:
            rows = searxng_search(
                searxng_url,
                query=query,
                category="general",
                language="all",
                limit=per_query,
                timeout=timeout,
                proxy_url=proxy_url,
            )
        except Exception as exc:
            print(f"[warn] SearXNG competitor discovery failed: {query} ({exc})", file=sys.stderr)
            continue
        for row in rows:
            discovery_results.append(
                SearchResult(
                    competitor="DISCOVERY",
                    category="competitor_discovery",
                    query=query,
                    title=str(row.get("title", "")),
                    url=str(row.get("url", "")),
                    snippet=str(row.get("content", row.get("snippet", ""))),
                    engine=str(row.get("engine", "")),
                )
            )
    deduped_results = dedupe_results(discovery_results)
    candidates = extract_competitor_candidates_from_results(
        deduped_results,
        plan,
        own_product_name,
        own_product_positioning,
        own_product_context,
        max(max_candidates * 3, max_candidates),
    )
    accepted_count = len([candidate for candidate in candidates if candidate.status == "accepted"])
    if accepted_count < max_candidates:
        lookup_budget = max_candidates * 4
        lookups_run = 0
        for candidate in candidates:
            if candidate.official_url or candidate.confidence == "低信心":
                continue
            for query in build_candidate_official_lookup_queries(candidate.name):
                if lookups_run >= lookup_budget:
                    break
                lookups_run += 1
                try:
                    rows = searxng_search(
                        searxng_url,
                        query=query,
                        category="general",
                        language="all",
                        limit=max(2, min(per_query, 4)),
                        timeout=timeout,
                        proxy_url=proxy_url,
                    )
                except Exception as exc:
                    print(f"[warn] SearXNG official lookup failed: {query} ({exc})", file=sys.stderr)
                    continue
                for row in rows:
                    discovery_results.append(
                        SearchResult(
                            competitor="DISCOVERY",
                            category="competitor_official_lookup",
                            query=query,
                            title=str(row.get("title", "")),
                            url=str(row.get("url", "")),
                            snippet=str(row.get("content", row.get("snippet", ""))),
                            engine=str(row.get("engine", "")),
                        )
                    )
            if lookups_run >= lookup_budget:
                break
        deduped_results = dedupe_results(discovery_results)
        candidates = extract_competitor_candidates_from_results(
            deduped_results,
            plan,
            own_product_name,
            own_product_positioning,
            own_product_context,
            max_candidates,
        )
    else:
        candidates = candidates[: max(0, max_candidates)]

    return (
        candidates,
        deduped_results,
    )


def write_competitor_discovery(
    out_dir: Path,
    candidates: Sequence[CompetitorCandidate],
    discovery_results: Sequence[SearchResult],
    used_for_collection: bool,
) -> Dict[str, Any]:
    candidate_rows = [dataclasses.asdict(candidate) for candidate in candidates]
    write_csv(
        out_dir / "competitor_discovery.csv",
        candidate_rows,
        [
            "name",
            "candidate_type",
            "confidence",
            "status",
            "official_url",
            "official_domain",
            "discovered_query",
            "discovered_from_url",
            "evidence_title",
            "evidence_snippet",
            "overlap_reason",
            "source_count",
        ],
    )
    payload = {
        "generated_at": utc_stamp(),
        "used_for_collection": used_for_collection,
        "candidate_count": len(candidates),
        "search_result_count": len(discovery_results),
        "candidates": candidate_rows,
        "search_results": [dataclasses.asdict(result) for result in discovery_results],
    }
    (out_dir / "competitor_discovery.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 自动竞品发现",
        "",
        f"- **是否用于本轮采集:** {'是' if used_for_collection else '否'}",
        f"- **候选数量:** {len(candidates)}",
        f"- **发现搜索结果:** {len(discovery_results)}",
        "",
    ]
    if candidates:
        lines += [
            "| 候选竞品 | 类型 | 置信度 | 状态 | 官方入口 | 发现搜索词 | 发现来源 | 入池理由 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for candidate in candidates:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        candidate.name,
                        candidate.candidate_type,
                        candidate.confidence,
                        candidate.status,
                        candidate.official_url,
                        candidate.discovered_query,
                        candidate.discovered_from_url,
                        candidate.overlap_reason,
                    )
                )
                + " |"
            )
    else:
        lines.append("本轮没有发现可进入候选池的竞品。")
    lines.append("")
    (out_dir / "competitor_discovery.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def extract_title_from_markdown(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()[:180]
    return ""


def clean_text(markdown: str, limit: int = 1200) -> str:
    text = textify(markdown)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[#>*_`|~-]+", " ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        stripped_line = strip_boilerplate_phrases(line)
        stripped_line = re.sub(r"\s+", " ", stripped_line).strip()
        if is_boilerplate_line(line) and len(stripped_line) < 120:
            continue
        if stripped_line:
            cleaned_lines.append(stripped_line)
    text = " ".join(cleaned_lines)
    text = strip_boilerplate_phrases(text)
    text = re.sub(r"\b\d+\s*(浏览|回复|赞)\b", " ", text)
    text = re.sub(r"[#>*_`|~-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def auth_or_transaction_shell_dominates(
    url: str,
    title: str,
    evidence_haystack: str,
    cleaned_text: str,
    field_hits: Sequence[str],
    include_hits: Sequence[str],
) -> bool:
    low = evidence_haystack.lower()
    auth_hits = keyword_hits(
        low,
        ["login", "sign in", "signin", "register", "signup", "password", "auth", "验证码", "账号", "登录", "注册", "密码"],
    )
    transaction_hits = keyword_hits(
        low,
        ["cart", "checkout", "coupon", "seller center", "shopping cart", "购物车", "卖家中心", "开店", "已买到的宝贝"],
    )
    product_hits = keyword_hits(
        low,
        [
            "pricing",
            "plans",
            "features",
            "product",
            "docs",
            "api",
            "integrations",
            "customers",
            "presentations",
            "specs",
            "价格",
            "定价",
            "功能",
            "产品",
            "参数",
            "文档",
            "客户",
        ],
    )
    if transaction_hits and not field_hits and len(cleaned_text) < 2000:
        return True
    if auth_hits and not login_gate_confirmed_by_url_title_or_form(url, title, evidence_haystack):
        return False
    if auth_hits and re.search(r"(请输入|验证码|手机号|password|forgot password|登录后|注册账号)", low) and not field_hits:
        return True
    if auth_hits and not (field_hits or include_hits or len(product_hits) >= 2) and len(cleaned_text) < 1600:
        return True
    return False


def page_quality_issue(
    competitor: str,
    url: str,
    title: str,
    markdown: str,
    collection_plan: Optional[ProductCollectionPlan] = None,
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
) -> str:
    low_url = low_value_url_reason(url, title, "", competitor)
    if low_url:
        return low_url

    raw_text = textify(markdown)
    cleaned = clean_text(raw_text, limit=8000)
    domain = domain_of(url)
    broken_haystack = f"{title}\n{cleaned[:1200]}\n{raw_text[:1200]}"
    evidence_haystack = f"{url}\n{title}\n{cleaned[:4000]}\n{raw_text[:4000]}"
    manual_exclude_hits = keyword_hits(evidence_haystack, manual_exclude_keywords)
    plan_exclude_hits = keyword_hits(evidence_haystack, collection_plan.exclude_keywords if collection_plan else [])
    include_hits = keyword_hits(evidence_haystack, collection_plan_evidence_keywords(collection_plan, manual_include_keywords))
    field_hits = product_field_hits(evidence_haystack, collection_plan)

    if manual_exclude_hits:
        return "rejected_manual_exclude_keyword:" + ",".join(manual_exclude_hits[:5])
    if plan_exclude_hits and auth_or_transaction_shell_dominates(url, title, evidence_haystack, cleaned, field_hits, include_hits):
        return "rejected_auth_or_transaction_shell:" + ",".join(plan_exclude_hits[:5])
    if any(pattern.search(broken_haystack) for pattern in BROKEN_PAGE_PATTERNS):
        return "rejected_broken_or_404_page"
    if len(cleaned) < 140:
        if official_core_signal(url, title, raw_text) or field_hits or include_hits:
            return "rejected_core_page_no_extractable_text_possible_js_shell"
        return "rejected_no_extractable_content_after_boilerplate"

    score = boilerplate_score(raw_text[:12000])
    portal_page = domain_matches(domain, LOW_VALUE_PORTAL_DOMAINS)
    if portal_page and score >= 5:
        return "rejected_portal_boilerplate_or_login_page"
    if score >= 10 and len(cleaned) < 700:
        return "rejected_boilerplate_or_login_page"
    if score >= 16 and not competitor_has_relevance(competitor, title, url, cleaned):
        return "rejected_boilerplate_low_competitor_relevance"
    if (
        not competitor_has_relevance(competitor, title, url, cleaned)
        and not field_hits
        and not include_hits
        and len(cleaned) < 500
    ):
        return "rejected_no_brand_or_product_evidence_after_cleaning"
    return ""


def extract_urls_from_markdown(markdown: str) -> Tuple[List[str], List[str]]:
    image_urls = []
    links = []
    for alt, url in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", markdown):
        image_urls.append(normalize_url(url))
    for text, url in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", markdown):
        links.append(normalize_url(url))
    return sorted(set(filter(None, links))), sorted(set(filter(None, image_urls)))


def looks_like_navigation_fragment(text: str) -> bool:
    lower = text.lower()
    nav_hits = count_phrase_hits(
        lower,
        [
            "view all",
            "discover",
            "sale",
            "accessories",
            "women",
            "men",
            "youth",
            "add to cart",
            "shopping cart",
            "free shipping",
            "sign in",
            "login",
            "cookie",
        ],
    )
    if re.search(r"\ball [a-z0-9 /-]{0,60}\b(accessories|apparel)\b", lower):
        return True
    return nav_hits >= 4 or ("view all" in lower and nav_hits >= 2)


def pick_sentence(markdown: str, patterns: Sequence[str], max_chars: int = 420) -> str:
    regexes = [re.compile(pattern, re.I) for pattern in patterns]
    candidates: List[str] = []
    seen = set()

    for raw_line in markdown.splitlines():
        cleaned = clean_text(raw_line, limit=max_chars * 2)
        if cleaned and cleaned not in seen:
            candidates.append(cleaned)
            seen.add(cleaned)

    text = clean_text(markdown, limit=7000)
    for fragment in re.split(r"(?<=[。！？.!?])\s+|\s{2,}", text):
        cleaned = clean_text(fragment, limit=max_chars * 2)
        if cleaned and cleaned not in seen:
            candidates.append(cleaned)
            seen.add(cleaned)

    best = ""
    best_score = 0
    fact_signal = re.compile(
        r"(\d|[$€£¥]|XS|XXS|XL|XXL|ASTM|EN\s?1077|F2040|CE\b|MIPS|EPS|EPP|ABS|PC\b|SOC ?2|SSO|API|SDK)",
        re.I,
    )
    for candidate in candidates:
        matches = sum(1 for rx in regexes if rx.search(candidate))
        if not matches:
            continue
        navigation = looks_like_navigation_fragment(candidate)
        if navigation and matches == 1 and not fact_signal.search(candidate):
            continue
        score = matches * 10
        if 20 <= len(candidate) <= max_chars:
            score += 3
        if fact_signal.search(candidate):
            score += 6
        if re.search(r"[:：]| - |\|", candidate):
            score += 1
        if navigation:
            score -= 8
        if len(candidate) > max_chars:
            score -= 2
        if not re.search(r"\s", candidate) and len(candidate) > 80:
            score -= 4
        if score > best_score:
            best_score = score
            best = candidate[:max_chars]
    return best


def pick_labeled_fragment(markdown: str, label_pattern: str, value_pattern: str, max_chars: int = 420) -> str:
    text = clean_text(markdown, limit=9000)
    rx = re.compile(rf"((?:{label_pattern}).{{0,{max_chars}}}?(?:{value_pattern}).{{0,160}})", re.I)
    match = rx.search(text)
    if match:
        return match.group(1)[:max_chars]
    return ""


def extract_weight_field(markdown: str) -> str:
    return pick_labeled_fragment(
        markdown,
        r"\bweight\b|重量",
        r"\d+(?:[.,]\d+)?\s*(?:g|gram|grams|kg|oz|ounce|ounces|克|千克)\b",
    )


def extract_usage_area_field(markdown: str) -> str:
    return pick_labeled_fragment(
        markdown,
        r"\busage area\b|适用场景|使用场景|用途",
        r"\b(?:all[- ]?mountain|freeride|piste|race|racing|slalom|ski|snowboard)\b|全山地|自由滑|道内|竞速|滑雪|双板|单板",
    )


def infer_fields(markdown: str, collection_plan: Optional[ProductCollectionPlan] = None) -> Dict[str, str]:
    fields = {
        "positioning": pick_sentence(
            markdown,
            [
                r"\b(platform|software|tool|solution|workspace|copilot|agent|automation)\b",
                r"帮助|平台|工具|解决方案|自动化|智能",
            ],
        ),
        "pricing": pick_sentence(
            markdown,
            [
                r"\b(price|pricing|plan|free|pro|enterprise|month|annual|subscription)\b",
                r"价格|定价|套餐|免费|企业版|专业版|订阅",
            ],
        ),
        "features": pick_sentence(
            markdown,
            [
                r"\b(feature|capability|integrations?|workflow|dashboard|analytics|template)\b",
                r"功能|能力|集成|工作流|看板|分析|模板",
            ],
        ),
        "customers": pick_sentence(
            markdown,
            [
                r"\b(customer|team|enterprise|startup|business|developer|creator|marketer)\b",
                r"客户|团队|企业|开发者|创作者|营销|用户",
            ],
        ),
    }
    if collection_plan:
        for field in collection_plan.fields:
            if field.key in fields:
                continue
            if field.key == "weight":
                fields[field.key] = extract_weight_field(markdown) or pick_sentence(markdown, field.patterns)
            elif field.key == "skiing_use_case":
                fields[field.key] = extract_usage_area_field(markdown) or pick_sentence(markdown, field.patterns)
            else:
                fields[field.key] = pick_sentence(markdown, field.patterns)
    return fields


async def crawl_with_crawl4ai(
    urls_by_competitor: List[Tuple[str, str]],
    max_concurrency: int,
    proxy_url: str = "",
    collection_plan: Optional[ProductCollectionPlan] = None,
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
) -> List[PageExtract]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        try:
            from crawl4ai import ProxyConfig
        except Exception:
            ProxyConfig = None
        try:
            from crawl4ai import CrawlerRunConfig
        except Exception:
            CrawlerRunConfig = None
    except Exception as exc:
        raise RuntimeError(
            "crawl4ai is not installed or cannot be imported. Install it with: pip install -U crawl4ai"
        ) from exc

    try:
        if proxy_url and ProxyConfig is not None:
            browser_config = BrowserConfig(headless=True, proxy_config=ProxyConfig(server=proxy_url))
        else:
            browser_config = BrowserConfig(headless=True)
    except TypeError:
        browser_config = BrowserConfig(headless=True, proxy=proxy_url or None)
    run_config = None
    if CrawlerRunConfig is not None:
        try:
            run_config = CrawlerRunConfig(
                word_count_threshold=5,
                page_timeout=45000,
                wait_until="domcontentloaded",
            )
        except TypeError:
            try:
                run_config = CrawlerRunConfig(word_count_threshold=5)
            except TypeError:
                run_config = CrawlerRunConfig()
    semaphore = asyncio.Semaphore(max_concurrency)
    extracts: List[PageExtract] = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async def one(competitor: str, url: str) -> None:
            async with semaphore:
                try:
                    if run_config is not None:
                        result = await crawler.arun(url=url, config=run_config)
                    else:
                        result = await crawler.arun(url=url)
                    markdown = textify(getattr(result, "markdown", ""))
                    title = textify(getattr(result, "title", "")) or extract_title_from_markdown(markdown)
                    links, image_urls = extract_urls_from_markdown(markdown)
                    cleaned_excerpt = clean_text(markdown)
                    quality_issue = page_quality_issue(
                        competitor,
                        url,
                        title,
                        markdown,
                        collection_plan,
                        manual_include_keywords,
                        manual_exclude_keywords,
                    )
                    if quality_issue:
                        extracts.append(
                            PageExtract(
                                competitor=competitor,
                                url=url,
                                title=title,
                                markdown="",
                                text_excerpt=cleaned_excerpt,
                                links=[],
                                image_urls=[],
                                fields={},
                                error=quality_issue,
                            )
                        )
                        return
                    media = getattr(result, "media", None)
                    if isinstance(media, dict):
                        for image in media.get("images", []) or []:
                            if isinstance(image, dict):
                                src = image.get("src") or image.get("url")
                                if src:
                                    image_urls.append(normalize_url(textify(src)))
                            elif isinstance(image, str):
                                image_urls.append(normalize_url(image))
                    extracts.append(
                        PageExtract(
                            competitor=competitor,
                            url=url,
                            title=title,
                            markdown=markdown,
                            text_excerpt=cleaned_excerpt,
                            links=sorted(set(filter(None, links))),
                            image_urls=sorted(set(filter(None, image_urls))),
                            fields=infer_fields(markdown, collection_plan),
                        )
                    )
                except Exception as exc:
                    extracts.append(
                        PageExtract(
                            competitor=competitor,
                            url=url,
                            title="",
                            markdown="",
                            text_excerpt="",
                            links=[],
                            image_urls=[],
                            fields={},
                            error=str(exc),
                        )
                    )

        await asyncio.gather(*(one(competitor, url) for competitor, url in urls_by_competitor))

    return extracts


def crawl_keyword_images(
    competitors: Sequence[str],
    out_dir: Path,
    max_images: int,
    engine: str,
    extra_terms: Sequence[str],
    proxy_url: str = "",
) -> List[Dict[str, str]]:
    if max_images <= 0:
        return []
    try:
        from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
    except Exception as exc:
        print(f"[warn] icrawler is not installed; skip keyword image download ({exc})", file=sys.stderr)
        return []

    crawler_cls = {
        "bing": BingImageCrawler,
        "baidu": BaiduImageCrawler,
        "google": GoogleImageCrawler,
    }.get(engine.lower(), BingImageCrawler)

    downloaded: List[Dict[str, str]] = []
    image_root = out_dir / "downloaded_images"
    image_root.mkdir(parents=True, exist_ok=True)

    for competitor in competitors:
        keyword = " ".join([competitor, *extra_terms]).strip()
        comp_dir = image_root / slugify(competitor)
        before = set(comp_dir.glob("*")) if comp_dir.exists() else set()
        comp_dir.mkdir(parents=True, exist_ok=True)
        try:
            crawler = crawler_cls(storage={"root_dir": str(comp_dir)})
            crawler.session.trust_env = False
            if proxy_url:
                crawler.session.proxies.update({"http": proxy_url, "https": proxy_url})
            crawler.crawl(keyword=keyword, max_num=max_images)
            after = set(comp_dir.glob("*"))
            for path in sorted(after - before):
                if path.is_file():
                    downloaded.append(
                        {
                            "competitor": competitor,
                            "query": keyword,
                            "engine": engine,
                            "file": str(path),
                        }
                    )
        except Exception as exc:
            print(f"[warn] icrawler failed for {keyword}: {exc}", file=sys.stderr)

    return downloaded


def default_binary_fetch(url: str, timeout: int = 20, proxy_url: str = "") -> Tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "competitor-intel-harvester/1.0",
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*,*/*;q=0.8",
        },
    )
    if proxy_url and not is_local_url(url):
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        data = response.read(8_000_000)
    return data, content_type


def image_extension(content_type: str, url: str) -> str:
    parsed_suffix = Path(urlparse(url).path).suffix.lower()
    if parsed_suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"}:
        return ".jpg" if parsed_suffix == ".jpeg" else parsed_suffix
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed in {".jpe", ".jpeg"}:
        return ".jpg"
    if guessed in {".jpg", ".png", ".webp", ".gif", ".avif", ".bmp"}:
        return guessed
    return ".jpg"


def looks_like_image_bytes(data: bytes, content_type: str) -> bool:
    if content_type.startswith("image/"):
        return True
    signatures = (
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
        b"GIF87a",
        b"GIF89a",
        b"RIFF",
    )
    return any(data.startswith(signature) for signature in signatures)


def download_searxng_image_results(
    image_results: Sequence[SearchResult],
    out_dir: Path,
    max_images_per_competitor: int,
    proxy_url: str = "",
    timeout: int = 20,
    fetcher: Optional[Any] = None,
) -> List[Dict[str, str]]:
    if max_images_per_competitor <= 0:
        return []
    fetch = fetcher or default_binary_fetch
    downloaded: List[Dict[str, str]] = []
    seen_urls = set()
    counts: Dict[str, int] = {}
    image_root = out_dir / "downloaded_images"
    image_root.mkdir(parents=True, exist_ok=True)

    for item in image_results:
        if not item.url or item.url in seen_urls:
            continue
        competitor = item.competitor or "unknown"
        if counts.get(competitor, 0) >= max_images_per_competitor:
            continue
        seen_urls.add(item.url)
        comp_dir = image_root / slugify(competitor) / "searxng"
        comp_dir.mkdir(parents=True, exist_ok=True)
        try:
            data, content_type = fetch(item.url, timeout, proxy_url)
            if not data or not looks_like_image_bytes(data, content_type):
                continue
            counts[competitor] = counts.get(competitor, 0) + 1
            extension = image_extension(content_type, item.url)
            file_path = comp_dir / f"{counts[competitor]:06d}{extension}"
            file_path.write_bytes(data)
            downloaded.append(
                {
                    "competitor": competitor,
                    "query": item.query,
                    "engine": item.engine,
                    "file": str(file_path),
                    "source": "searxng_image_download",
                    "image_url": item.url,
                    "title": item.title,
                    "page_url": "",
                }
            )
        except Exception as exc:
            print(f"[warn] SearXNG image download failed for {item.url}: {exc}", file=sys.stderr)

    return downloaded


def choose_urls_to_crawl(
    web_results: Sequence[SearchResult],
    max_pages_per_competitor: int,
    collection_plan: Optional[ProductCollectionPlan] = None,
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
    ml_model: Optional[LocalFilterModel] = None,
    ml_auto_include_threshold: float = 0.75,
    ml_auto_exclude_threshold: float = 0.80,
) -> List[Tuple[str, str]]:
    competitors = list(dict.fromkeys(result.competitor for result in web_results))
    official_domains = official_domains_by_competitor(competitors, web_results)
    grouped: Dict[str, List[SearchResult]] = {}
    for result in web_results:
        if not result.url or not is_probably_html_page(result.url):
            continue
        decision = evidence_decision_for_result(
            result,
            official_domains,
            collection_plan,
            manual_include_keywords,
            manual_exclude_keywords,
            ml_model=ml_model,
            ml_auto_include_threshold=ml_auto_include_threshold,
            ml_auto_exclude_threshold=ml_auto_exclude_threshold,
        )
        if decision.decision_status == "rejected" and decision.hard_gate.startswith("rejected_manual_exclude_keyword"):
            continue
        if decision.hard_gate in {"rejected_auth_or_transaction_shell", "rejected_low_evidence_source_policy"}:
            continue
        if pre_crawl_rejection_reason(result.url, result.title, result.snippet, result.competitor):
            continue
        grouped.setdefault(result.competitor, []).append(result)

    chosen: List[Tuple[str, str]] = []
    for competitor, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x.score, reverse=True)
        official_rows = [
            row
            for row in rows
            if is_official_domain_for(competitor, row.url, official_domains)
        ]
        official_rows = sorted(
            official_rows,
            key=lambda row: (
                0 if row.category != "official_expansion" else 1,
                official_path_priority_for_plan(row.url, collection_plan),
                -evidence_decision_for_result(
                    row,
                    official_domains,
                    collection_plan,
                    manual_include_keywords,
                    manual_exclude_keywords,
                    ml_model=ml_model,
                    ml_auto_include_threshold=ml_auto_include_threshold,
                    ml_auto_exclude_threshold=ml_auto_exclude_threshold,
                ).pm_value_score,
                -float(
                    evidence_decision_for_result(
                        row,
                        official_domains,
                        collection_plan,
                        manual_include_keywords,
                        manual_exclude_keywords,
                        ml_model=ml_model,
                        ml_auto_include_threshold=ml_auto_include_threshold,
                        ml_auto_exclude_threshold=ml_auto_exclude_threshold,
                    ).ml_include_score
                    or 0
                ),
                -row.score,
                row.url,
            ),
        )
        high_value_public_rows = [
            row
            for row in rows
            if row not in official_rows
            and domain_matches(domain_of(row.url), HIGH_VALUE_PUBLIC_CRAWL_DOMAINS)
        ]
        high_value_public_rows = sorted(
            high_value_public_rows,
            key=lambda row: (
                -evidence_decision_for_result(
                    row,
                    official_domains,
                    collection_plan,
                    manual_include_keywords,
                    manual_exclude_keywords,
                    ml_model=ml_model,
                    ml_auto_include_threshold=ml_auto_include_threshold,
                    ml_auto_exclude_threshold=ml_auto_exclude_threshold,
                ).pm_value_score,
                -float(
                    evidence_decision_for_result(
                        row,
                        official_domains,
                        collection_plan,
                        manual_include_keywords,
                        manual_exclude_keywords,
                        ml_model=ml_model,
                        ml_auto_include_threshold=ml_auto_include_threshold,
                        ml_auto_exclude_threshold=ml_auto_exclude_threshold,
                    ).ml_include_score
                    or 0
                ),
                -row.score,
                row.url,
            ),
        )

        comp_chosen = []
        selected = set()
        for row in official_rows:
            if len(comp_chosen) >= max_pages_per_competitor:
                break
            if row.url in selected:
                continue
            comp_chosen.append((competitor, row.url))
            selected.add(row.url)

        public_budget = max_pages_per_competitor - len(comp_chosen)
        if official_rows:
            public_budget = min(public_budget, max(2, max_pages_per_competitor // 5))
        seen_public_domains = set()
        for row in high_value_public_rows:
            if public_budget <= 0 or len(comp_chosen) >= max_pages_per_competitor:
                break
            domain = domain_of(row.url)
            if row.url in selected or domain in seen_public_domains:
                continue
            seen_public_domains.add(domain)
            comp_chosen.append((competitor, row.url))
            selected.add(row.url)
            public_budget -= 1
        chosen.extend(comp_chosen)
    return chosen


def rows_from_evidence_audit(
    web_results: Sequence[SearchResult],
    chosen_urls: Sequence[Tuple[str, str]],
    max_pages_per_competitor: int,
    collection_plan: Optional[ProductCollectionPlan] = None,
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
    ml_model: Optional[LocalFilterModel] = None,
    ml_auto_include_threshold: float = 0.75,
    ml_auto_exclude_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    selected = {(competitor, url) for competitor, url in chosen_urls}
    grouped: Dict[str, List[SearchResult]] = {}
    for result in web_results:
        grouped.setdefault(result.competitor, []).append(result)
    official_domains = official_domains_by_competitor(list(grouped.keys()), web_results)

    rows: List[Dict[str, Any]] = []
    for competitor, results in grouped.items():
        sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
        selected_count = 0
        for rank, item in enumerate(sorted_results, start=1):
            is_html = is_probably_html_page(item.url)
            pre_crawl_reason = pre_crawl_rejection_reason(item.url, item.title, item.snippet, item.competitor)
            is_official_source = is_official_domain_for(item.competitor, item.url, official_domains)
            is_selected = (item.competitor, item.url) in selected
            structured = evidence_decision_for_result(
                item,
                official_domains,
                collection_plan,
                manual_include_keywords,
                manual_exclude_keywords,
                selected=is_selected,
                rank=rank,
                max_pages_per_competitor=max_pages_per_competitor,
                ml_model=ml_model,
                ml_auto_include_threshold=ml_auto_include_threshold,
                ml_auto_exclude_threshold=ml_auto_exclude_threshold,
            )
            if is_selected:
                selected_count += 1
            if is_selected:
                decision = "selected"
            elif not is_html:
                decision = "rejected_non_html"
            elif pre_crawl_reason and structured.decision_status == "rejected":
                decision = pre_crawl_reason
            elif max_pages_per_competitor <= 0:
                decision = "not_crawled_max_pages_zero"
            elif is_official_source:
                decision = "rejected_official_over_budget"
            elif rank > max_pages_per_competitor * 3:
                decision = "rejected_low_rank"
            else:
                decision = "rejected_lower_priority"
            reason = score_reasons(item.url, item.title, item.snippet, item.competitor)
            if is_official_source:
                reason.append("positive: discovered official-domain candidate")
            reason = unique_strings([*reason, *structured.reasons])
            rows.append(
                {
                    "competitor": item.competitor,
                    "decision": decision,
                    "decision_status": structured.decision_status,
                    "source_kind": structured.source_kind,
                    "page_role": structured.page_role,
                    "source_policy_tier": structured.source_policy_tier,
                    "gate_result": structured.gate_result,
                    "hard_gate": structured.hard_gate,
                    "confidence": structured.confidence,
                    "pending_verification": "yes" if structured.pending_verification else "no",
                    "verification_reason": structured.verification_reason,
                    "fact_type": structured.fact_type,
                    "increment_type": structured.increment_type,
                    "fact_group": structured.fact_group,
                    "primary_evidence_candidate": "yes" if structured.primary_evidence_candidate else "no",
                    "primary_evidence_reason": structured.primary_evidence_reason,
                    "value_signals": "、".join(structured.value_signals),
                    "value_missing": "、".join(structured.value_missing),
                    "value_verdict": structured.value_verdict,
                    "gui_review_candidate": "yes" if structured.gui_review_candidate else "no",
                    "gui_review_value_reason": structured.gui_review_value_reason,
                    "ml_label": structured.ml_label,
                    "ml_include_score": structured.ml_include_score,
                    "ml_exclude_score": structured.ml_exclude_score,
                    "ml_verify_later_score": structured.ml_verify_later_score,
                    "ml_confidence": structured.ml_confidence,
                    "ml_adjustment": structured.ml_adjustment,
                    "ml_reason": structured.ml_reason,
                    "ml_model_version": structured.ml_model_version,
                    "relevance_score": structured.relevance_score,
                    "evidence_score": structured.evidence_score,
                    "pm_value_score": structured.pm_value_score,
                    "traceability_score": structured.traceability_score,
                    "category_fit_score": structured.category_fit_score,
                    "matched_fields": "、".join(structured.matched_fields),
                    "matched_include_keywords": ", ".join(structured.matched_include_keywords),
                    "matched_exclude_keywords": ", ".join(structured.matched_exclude_keywords),
                    "rejection_code": structured.rejection_code,
                    "selected": "yes" if is_selected else "no",
                    "rank": rank,
                    "score": f"{item.score:.2f}",
                    "domain": domain_of(item.url),
                    "title": item.title,
                    "url": item.url,
                    "query": item.query,
                    "engine": item.engine,
                    "reason": "; ".join(reason),
                    "selection_note": (
                        "selected for Crawl4AI"
                        if is_selected
                        else (
                            f"excluded before Crawl4AI by hard gate: {structured.hard_gate}"
                            if structured.decision_status == "rejected"
                            else (
                                f"excluded before Crawl4AI by rule filter: {pre_crawl_reason}"
                                if pre_crawl_reason
                                else f"kept as {structured.decision_status} evidence but not crawled"
                            )
                        )
                    ),
                    "per_competitor_crawl_budget": max_pages_per_competitor,
                    "selected_count_so_far": selected_count,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_from_pages(
    pages: Sequence[PageExtract],
    collection_plan: Optional[ProductCollectionPlan] = None,
    evidence_audit_rows: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    rows = []
    planned_keys = collection_plan_field_keys(collection_plan)
    audit_by_url = {
        (row.get("competitor", ""), row.get("url", "")): row
        for row in evidence_audit_rows
    }
    for page in pages:
        audit = audit_by_url.get((page.competitor, page.url), {})
        page_role = audit.get("page_role") or page_role_for_result(
            SearchResult(page.competitor, "crawled_page", "", page.title, page.url, page.text_excerpt),
            collection_plan,
        )
        matched_fields = product_field_hits(f"{page.url} {page.title} {page.text_excerpt}", collection_plan)
        fact_type = audit.get("fact_type") or fact_type_for(page_role, matched_fields, page.text_excerpt)
        pending_verification = audit.get("pending_verification") or ("yes" if manual_review_reason(page) else "no")
        verification_reason = audit.get("verification_reason") or (manual_review_next_step(page) if manual_review_reason(page) else "")
        row = {
            "competitor": page.competitor,
            "url": page.url,
            "title": page.title,
            "source_policy_tier": audit.get("source_policy_tier", ""),
            "page_role": page_role,
            "pending_verification": pending_verification,
            "verification_reason": verification_reason,
            "fact_type": fact_type,
            "increment_type": audit.get("increment_type") or increment_type_for(page_role, matched_fields, page.text_excerpt),
            "fact_group": audit.get("fact_group") or fact_group_for(page.competitor, fact_type, matched_fields, page_role, page.url, page.title),
            "primary_evidence_candidate": audit.get("primary_evidence_candidate") or ("no" if page.error or pending_verification == "yes" else "yes"),
            "primary_evidence_reason": audit.get("primary_evidence_reason") or ("待核实或抓取失败，不能作为主证据" if page.error or pending_verification == "yes" else "抓取正文可用，可作为候选证据"),
            "value_signals": audit.get("value_signals", ""),
            "value_missing": audit.get("value_missing", ""),
            "value_verdict": audit.get("value_verdict", ""),
            "gui_review_candidate": audit.get("gui_review_candidate", ""),
            "gui_review_value_reason": audit.get("gui_review_value_reason", ""),
            "ml_label": audit.get("ml_label", ""),
            "ml_include_score": audit.get("ml_include_score", ""),
            "ml_exclude_score": audit.get("ml_exclude_score", ""),
            "ml_verify_later_score": audit.get("ml_verify_later_score", ""),
            "ml_confidence": audit.get("ml_confidence", ""),
            "ml_adjustment": audit.get("ml_adjustment", ""),
            "ml_reason": audit.get("ml_reason", ""),
            "ml_model_version": audit.get("ml_model_version", ""),
            "positioning": page.fields.get("positioning", ""),
            "pricing": page.fields.get("pricing", ""),
            "features": page.fields.get("features", ""),
            "customers": page.fields.get("customers", ""),
            "image_count": len(page.image_urls),
            "link_count": len(page.links),
            "text_excerpt": page.text_excerpt,
            "error": page.error,
        }
        for key in planned_keys:
            row[key] = page.fields.get(key, "")
        rows.append(row)
    return rows


def is_antibot_error(error: str) -> bool:
    low = (error or "").lower()
    return any(marker in low for marker in ANTIBOT_ERROR_MARKERS)


def looks_like_login_required_text(text: str) -> bool:
    low = textify(text).lower()
    return bool(
        AUTH_LOGIN_MARKER_RE.search(low)
        or any(token in low for token in ("登录", "登陆", "注册", "账号", "账户", "密码", "验证码", "请登录"))
    )


def url_has_auth_gate_path(url: str) -> bool:
    parsed = urlparse(textify(url))
    host = (parsed.hostname or "").lower()
    first_label = host.split(".", 1)[0] if host else ""
    if first_label in AUTH_GATE_HOST_LABELS:
        return True
    for segment in (parsed.path or "").lower().split("/"):
        compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", segment).strip()
        if not compact:
            continue
        tokens = set(compact.split())
        has_noise = bool(tokens & AUTH_GATE_URL_NOISE_TOKENS)
        if compact.replace(" ", "-") in AUTH_GATE_URL_TOKENS and not has_noise:
            return True
        if tokens & set(AUTH_GATE_URL_TOKENS) and len(tokens) <= 3 and not has_noise:
            return True
    return False


def title_is_auth_gate(title: str) -> bool:
    cleaned = re.sub(r"\s+", " ", textify(title)).strip()
    if not cleaned or len(cleaned) > 140:
        return False
    chunks = [cleaned, *re.split(r"[|:：/·\-–—]+", cleaned)]
    for chunk in chunks:
        low = chunk.strip().lower()
        if not low:
            continue
        if any(term in low for term in AUTH_GATE_TITLE_NOISE_TERMS):
            continue
        word_count = len(re.findall(r"[a-z0-9]+", low))
        has_english_login = bool(AUTH_LOGIN_MARKER_RE.search(low))
        has_chinese_login = any(token in low for token in ("登录", "登陆", "注册", "账号登录", "账户登录", "密码登录"))
        if has_english_login and 0 < word_count <= 5:
            return True
        if has_chinese_login and len(low) <= 30:
            return True
    return False


def login_gate_confirmed_by_url_title_or_form(
    url: str,
    title: str = "",
    text: str = "",
    error: str = "",
) -> bool:
    haystack = f"{url}\n{title}\n{text}\n{error}".lower()
    has_login_language = looks_like_login_required_text(haystack) or is_login_required_error(error)
    if not has_login_language:
        return False
    if url_has_auth_gate_path(url) or title_is_auth_gate(title):
        return True
    return bool(keyword_hits(haystack, AUTH_FORM_FIELD_TERMS))


def looks_like_login_form(url: str, title: str = "", text: str = "") -> bool:
    haystack = f"{url}\n{title}\n{text}".lower()
    login_hits = keyword_hits(haystack, LOGIN_ASSIST_TERMS)
    if not login_hits:
        return False
    value_hits = keyword_hits(haystack, LOGIN_ASSIST_VALUE_TERMS)
    form_hits = keyword_hits(haystack, LOGIN_FORM_TERMS)
    if keyword_hits(haystack, AUTH_FORM_FIELD_TERMS):
        return True
    if not (url_has_auth_gate_path(url) or title_is_auth_gate(title)):
        return False
    if form_hits:
        return not value_hits or len(clean_text(text, limit=2500)) < 1800
    return not value_hits and len(clean_text(text, limit=2500)) < 1800


def is_login_required_error(error: str) -> bool:
    low = textify(error).lower()
    if not low:
        return False
    if "rejected_auth_or_transaction_shell" in low:
        return True
    if any(token in low for token in ("low_value_url_token", "low_value_host")) and any(
        token in low
        for token in (
            "login",
            "signin",
            "signup",
            "register",
            "auth",
            "oauth",
            "passport",
            "account",
            "member",
        )
    ):
        return True
    return False


def page_requires_user_login(page: PageExtract) -> bool:
    text = f"{page.text_excerpt}\n{page.markdown}"
    return bool(page.error and login_gate_confirmed_by_url_title_or_form(page.url, page.title, text, page.error))


def audit_requires_user_login(row: Mapping[str, Any]) -> bool:
    haystack = "\n".join(
        textify(row.get(key))
        for key in (
            "url",
            "title",
            "domain",
            "reason",
            "selection_note",
            "hard_gate",
            "rejection_code",
            "page_role",
            "value_missing",
        )
    )
    page_role = textify(row.get("page_role"))
    hard_gate = textify(row.get("hard_gate") or row.get("rejection_code"))
    if page_role == "auth_or_account_shell":
        return login_gate_confirmed_by_url_title_or_form(
            textify(row.get("url")),
            textify(row.get("title")),
            haystack,
            hard_gate,
        )
    if is_login_required_error(hard_gate):
        return login_gate_confirmed_by_url_title_or_form(
            textify(row.get("url")),
            textify(row.get("title")),
            haystack,
            hard_gate,
        )
    return looks_like_login_form(textify(row.get("url")), textify(row.get("title")), haystack)


def login_pool_excluded_by_context(*values: str) -> bool:
    haystack = " ".join(textify(value) for value in values).lower()
    return bool(keyword_hits(haystack, LOGIN_POOL_EXCLUDE_HINTS))


def query_site_filters(query: str) -> List[str]:
    filters = []
    for match in re.finditer(r"\bsite:([A-Za-z0-9*_.-]+(?:\.[A-Za-z0-9*_.-]+)+)", textify(query), re.I):
        value = match.group(1).strip().lower().lstrip("*.").removeprefix("www.")
        if value:
            filters.append(value)
    return unique_strings(filters)


def query_site_filter_matches_url(query: str, url: str) -> bool:
    filters = query_site_filters(query)
    if not filters:
        return True
    url_host = (urlparse(textify(url)).hostname or "").lower().strip(".").removeprefix("www.")
    url_domain = site_domain_of(url)
    url_platform = login_assist_platform_key_for_url_or_domain(url)
    for site_filter in filters:
        filter_platform = login_assist_platform_key_for_url_or_domain(site_filter)
        filter_domain = site_domain_of(f"https://{site_filter}")
        if filter_platform and url_platform == filter_platform:
            return True
        if url_domain and filter_domain and url_domain == filter_domain:
            return True
        if url_host and (url_host == site_filter or url_host.endswith("." + site_filter)):
            return True
    return False


def platform_login_assist_target(row: Mapping[str, Any]) -> Dict[str, str]:
    url = textify(row.get("url") or row.get("gui_review_url") or row.get("login_assist_url"))
    platform_key = login_assist_platform_key_for_url_or_domain(url)
    if not platform_key:
        return {}
    if not query_site_filter_matches_url(textify(row.get("query")), url):
        return {}

    competitor = textify(row.get("competitor"))
    title = textify(row.get("title"))
    snippet = textify(row.get("cleaned_excerpt_sample") or row.get("snippet"))
    reason = textify(row.get("reason") or row.get("selection_note") or row.get("gui_review_value_reason"))
    if login_pool_excluded_by_context(url, title, snippet):
        return {}
    if not competitor_strong_binding(competitor, url, title, snippet, reason):
        return {}

    page_role = textify(row.get("page_role"))
    source_kind = textify(row.get("source_kind"))
    decision_status = textify(row.get("decision_status")).lower()
    gui_candidate = textify(row.get("gui_review_candidate")).lower() == "yes"
    platform_reviewable = (
        page_role in {"video_or_social_content", "forum_or_community_discussion"}
        or source_kind == "community_or_social_signal"
    )
    if not platform_reviewable:
        return {}
    if decision_status not in {"accepted", "signal"} and not gui_candidate:
        return {}

    config = LOGIN_ASSIST_PLATFORM_TARGETS.get(platform_key, {})
    login_url = textify(config.get("login_url"))
    if not login_url:
        return {}
    return {
        "platform_domain": platform_key,
        "login_url": login_url,
        "queued_url": url,
    }


def audit_login_queue_eligible(row: Mapping[str, Any]) -> bool:
    if not audit_requires_user_login(row):
        return False
    if textify(row.get("automated_review_status")).lower() == "login_skipped_by_user":
        return False
    competitor = textify(row.get("competitor"))
    url = textify(row.get("login_assist_url") or row.get("url") or row.get("gui_review_url"))
    title = textify(row.get("title"))
    snippet = textify(row.get("cleaned_excerpt_sample") or row.get("snippet"))
    reason = textify(row.get("reason") or row.get("suggested_next_step"))
    if login_pool_excluded_by_context(url, title, snippet):
        return False
    if competitor_strong_binding(competitor, url, title, snippet):
        return True
    return False


def row_requires_login_action(row: Mapping[str, Any]) -> bool:
    status = textify(row.get("automated_review_status")).lower()
    hard_gate = textify(row.get("hard_gate") or row.get("rejection_code")).lower()
    if status == "login_skipped_by_user":
        return False
    return (
        textify(row.get("requires_user_login")).lower() == "yes"
        or textify(row.get("review_reason")) == "login_required_user_action"
        or textify(row.get("page_role")) == "auth_or_account_shell"
        or status in {
            "requires_user_login",
            "awaiting_user_login",
            "login_assisted_snapshot_captured",
            "login_assist_still_requires_login",
        }
        or status.startswith("login_assist")
        or "auth_or_transaction_shell" in hard_gate
    )


def review_target_url(row: Mapping[str, Any]) -> str:
    keys = (
        ("login_assist_url", "gui_review_url", "url", "url_or_path")
        if row_requires_login_action(row)
        else ("gui_review_url", "url", "url_or_path", "login_assist_url")
    )
    for key in keys:
        value = textify(row.get(key))
        if value:
            return value
    return ""


def login_queue_key_for(row: Mapping[str, Any]) -> Tuple[str, str]:
    url = review_target_url(row)
    competitor = textify(row.get("competitor"))
    return login_queue_key_for_values(competitor, url)


def review_queue_key_for(row: Mapping[str, Any]) -> Tuple[str, str]:
    if row_requires_login_action(row):
        return login_queue_key_for(row)
    url = review_target_url(row)
    return textify(row.get("competitor")), canonical_url_for_dedupe(url) or url


def login_click_marker_id(competitor: str, url: str) -> str:
    keyword, stable_target = login_queue_key_for_values(competitor, url)
    raw = f"{keyword}::{stable_target}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def login_click_marker_path(out_dir: Path, competitor: str, url: str) -> Path:
    return Path(out_dir) / "login_click_requests" / f"{login_click_marker_id(competitor, url)}.json"


def login_skip_marker_path(out_dir: Path, competitor: str, url: str) -> Path:
    return Path(out_dir) / "login_skip_requests" / f"{login_click_marker_id(competitor, url)}.json"


def login_click_requested(out_dir: Path, row: Mapping[str, Any]) -> bool:
    url = review_target_url(row)
    if not url:
        return False
    return login_click_marker_path(out_dir, textify(row.get("competitor")), url).exists()


def login_skip_requested(out_dir: Path, row: Mapping[str, Any]) -> bool:
    url = review_target_url(row)
    if not url:
        return False
    return login_skip_marker_path(out_dir, textify(row.get("competitor")), url).exists()


def dedupe_login_review_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    queued_urls: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        if not row_requires_login_action(row):
            continue
        url = review_target_url(row)
        if not url:
            continue
        key = login_queue_key_for(row)
        if key not in by_key:
            by_key[key] = dict(row)
            queued_urls[key] = []
            deduped.append(by_key[key])
        row_queued_urls = [
            value.strip()
            for value in textify(row.get("queued_urls")).splitlines()
            if value.strip()
        ] or [textify(row.get("url")) or url]
        for queued_url in row_queued_urls:
            queued_domain = login_queue_domain_key_for_url(queued_url)
            if queued_domain and queued_domain != key[1]:
                continue
            if queued_url not in queued_urls[key]:
                queued_urls[key].append(queued_url)
    for row in deduped:
        key = login_queue_key_for(row)
        urls = queued_urls.get(key, [])
        row["queued_urls"] = "\n".join(urls)
        row["queued_url_count"] = str(len(urls) or 1)
    return deduped


def login_review_next_step() -> str:
    return (
        "系统会把同站点登录页合并到集中等待区，不会主动打开网页。"
        "需要处理时，请在 UI 登录池点击对应链接；工具收到点击信号后才打开该站点，并在公开页面采集结束后的等待期内尝试保存登录后快照。"
    )


def login_review_allowed_boundary() -> str:
    return (
        "仅限用户本人有权限访问的页面；不破解验证码、不绕过登录/付费/访问控制、不保存或导出账号凭据，"
        "不调用未授权私有接口。"
    )


def manual_review_reason(page: PageExtract) -> str:
    if not page.error:
        return ""
    if page_requires_user_login(page):
        return "login_required_user_action"
    if is_antibot_error(page.error):
        return "anti_bot_or_access_block"
    if page.error == "rejected_broken_or_404_page" and official_core_signal(page.url, page.title, page.text_excerpt):
        return "broken_official_core_path_needs_alternative"
    if page.error in {
        "rejected_low_text_content",
        "rejected_core_page_no_extractable_text_possible_js_shell",
        "rejected_no_extractable_content_after_boilerplate",
    } and official_core_signal(page.url, page.title, page.text_excerpt):
        return "official_core_low_text_or_possible_js_shell"
    if page.error in {
        "rejected_low_competitor_relevance",
        "rejected_boilerplate_low_competitor_relevance",
        "rejected_no_brand_or_product_evidence_after_cleaning",
    }:
        return "low_relevance_needs_gui_confirmation"
    return ""


def manual_review_priority(page: PageExtract) -> str:
    if page_requires_user_login(page):
        return "P0-LOGIN"
    path = urlparse(page.url).path.lower()
    if any(token in path for token in ("/pricing", "/plans", "/features", "/product", "/products", "/api", "/security")):
        return "P0"
    if is_antibot_error(page.error):
        return "P1"
    return "P2"


def manual_review_next_step(page: PageExtract) -> str:
    reason = manual_review_reason(page)
    if reason == "login_required_user_action":
        return login_review_next_step()
    if reason == "anti_bot_or_access_block":
        return (
            "先在 GUI/浏览器中打开原 URL 判断是否为公开且有价值内容；若有效，优先寻找同站公开替代入口、sitemap、"
            "官方帮助中心/API/文档/静态页面或允许访问的导出内容。不要破解验证码、不要使用未授权登录态。"
        )
    if reason == "official_core_low_text_or_possible_js_shell":
        return "先用浏览器打开确认是否为真实核心页；若是 JS 渲染页，可补充官网导航链接、sitemap 或人工摘录公开可见内容。"
    if reason == "broken_official_core_path_needs_alternative":
        return "先在浏览器确认是否确为 404；若官网导航中有等价页面，从首页、sitemap 或站内公开链接补采替代 URL。"
    return "用浏览器确认是否同名无关或低价值；只有能支持定位、能力、定价、客户、GTM 或战略判断时才补采。"


def manual_review_priority_from_evidence(row: Mapping[str, Any]) -> str:
    if textify(row.get("source_policy_tier")).startswith(("P0", "P1")):
        return "P0"
    if textify(row.get("page_role")) in {"video_or_social_content", "app_store_listing"}:
        return "P1"
    return "P2"


def rows_from_manual_review_queue(
    pages: Sequence[PageExtract],
    evidence_audit_rows: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for page in pages:
        seen.add((page.competitor, page.url))
        reason = manual_review_reason(page)
        if not reason:
            continue
        rows.append(
            {
                "competitor": page.competitor,
                "priority": manual_review_priority(page),
                "review_reason": reason,
                "requires_user_login": "yes" if page_requires_user_login(page) else "no",
                "title": page.title,
                "url": page.url,
                "domain": domain_of(page.url),
                "crawl_error": page.error,
                "cleaned_excerpt_sample": truncate_text(page.text_excerpt, 500),
                "gui_review_url": page.url,
                "login_assist_url": page.url if page_requires_user_login(page) else "",
                "queued_urls": page.url if page_requires_user_login(page) else "",
                "suggested_next_step": manual_review_next_step(page),
                "allowed_boundary": (
                    login_review_allowed_boundary()
                    if page_requires_user_login(page)
                    else "public content only; no captcha cracking, credential bypass, private APIs, or access-control circumvention"
                ),
            }
        )
    for row in evidence_audit_rows:
        url = textify(row.get("url"))
        adapter_info = classify_source_url(url)
        adapter_reviewable = (
            bool(adapter_info.get("adapter_name"))
            and textify(row.get("decision_status")).lower() in {"signal", "accepted"}
            and not textify(row.get("hard_gate")).startswith("rejected_auth_or_transaction_shell")
        )
        platform_target = platform_login_assist_target(row)
        login_required = audit_login_queue_eligible(row) or bool(platform_target)
        if textify(row.get("gui_review_candidate")).lower() != "yes" and not login_required and not adapter_reviewable:
            continue
        key = (textify(row.get("competitor")), url)
        if key in seen:
            continue
        seen.add(key)
        reason = (
            login_review_next_step()
            if login_required
            else (
                textify(row.get("gui_review_value_reason"))
                or (
                    f"{adapter_info.get('platform')} 公开来源可先抓元数据、截图/时间点线索和可追溯字段，再由人工核验是否写入强事实。"
                    if adapter_reviewable
                    else "预抓取阶段命中价值门槛，但自动文本抓取不稳定，需 GUI 复核。"
                )
            )
        )
        rows.append(
            {
                "competitor": textify(row.get("competitor")),
                "priority": "P0-LOGIN" if login_required else manual_review_priority_from_evidence(row),
                "review_reason": (
                    "platform_login_assist_user_action"
                    if platform_target
                    else (
                        "login_required_user_action"
                        if login_required
                        else ("public_source_adapter_candidate" if adapter_reviewable else "pre_crawl_value_candidate")
                    )
                ),
                "requires_user_login": "yes" if login_required else "no",
                "title": textify(row.get("title")),
                "url": key[1],
                "domain": platform_target.get("platform_domain") or textify(row.get("domain")) or domain_of(key[1]),
                "crawl_error": textify(row.get("hard_gate")) or textify(row.get("rejection_code")) or "not_crawled_pre_crawl_candidate",
                "cleaned_excerpt_sample": truncate_text(row.get("reason", ""), 500),
                "gui_review_url": key[1],
                "login_assist_url": platform_target.get("login_url") or (key[1] if login_required else ""),
                "queued_urls": platform_target.get("queued_url") or (key[1] if login_required else ""),
                "suggested_next_step": reason,
                "allowed_boundary": (
                    login_review_allowed_boundary()
                    if login_required
                    else "public content only; no captcha cracking, credential bypass, private APIs, or access-control circumvention"
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["priority"], row["competitor"], row["domain"], row["url"]))


def write_manual_review_queue(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# 人工复核队列",
        "",
        "本文件列出 Crawl4AI 遇到反爬、403、JS 壳、低文本、需登录或疑似核心页异常的 URL，也包括预抓取阶段命中价值门槛但不适合直接当事实的视频、社区、社媒或应用商店候选。",
        "",
        "处理原则：先在 GUI/浏览器中打开原网页，判断是否确实是公开且有产品情报价值的内容；确认有价值后，再通过合规方式补证，例如同站公开导航、sitemap、官方文档/API、帮助中心、公开静态页、可下载公开资料或人工摘录公开可见内容。",
        "",
        "遇到登录/注册页时，工具会先按竞品和域名去重放入登录等待区，公开页面继续采集；只有用户在 UI 登录池点击对应链接后，工具才会打开该站点并复用同一浏览器登录态保存快照，否则合并到问题页面核验清单。",
        "",
        "边界：不破解验证码，不绕过登录/付费/访问控制，不保存账号凭据，不调用未授权私有接口，不采集私密或违法内容。",
        "",
    ]
    if not rows:
        lines += ["本轮没有需要人工复核的反爬/异常页面。", ""]
    for idx, row in enumerate(rows, start=1):
        lines += [
            f"## {idx}. {row.get('priority')} | {row.get('competitor')} | {row.get('title') or row.get('url')}",
            "",
            f"- **URL:** {row.get('url')}",
            f"- **Domain:** {row.get('domain')}",
            f"- **Review reason:** {row.get('review_reason')}",
            f"- **Requires user login:** {row.get('requires_user_login') or 'no'}",
            f"- **Crawl error:** {row.get('crawl_error')}",
            f"- **GUI review:** {row.get('gui_review_url')}",
            f"- **Login assist:** {row.get('login_assist_url') or '无'}",
            f"- **Suggested next step:** {row.get('suggested_next_step')}",
            "",
            "**Cleaned excerpt sample:**",
            "",
            row.get("cleaned_excerpt_sample") or "无正文样本",
            "",
        ]
    path.write_text(normalize_confidence_lines("\n".join(lines)), encoding="utf-8")


def score_float(value: Any) -> float:
    try:
        return float(textify(value) or 0)
    except (TypeError, ValueError):
        return 0.0


def training_review_priority(row: Mapping[str, Any]) -> Tuple[int, str, str]:
    score = 0
    status = textify(row.get("decision_status")).lower()
    pending = textify(row.get("pending_verification")).lower()
    ml_confidence = textify(row.get("ml_confidence")).lower()
    ml_adjustment = textify(row.get("ml_adjustment")).lower()
    include_score = score_float(row.get("ml_include_score"))
    exclude_score = score_float(row.get("ml_exclude_score"))
    verify_score = score_float(row.get("ml_verify_later_score"))
    top_score = max(include_score, exclude_score, verify_score)
    if pending == "yes":
        score += 80
    if status in {"signal", "accepted"}:
        score += 35
    if ml_confidence in {"low", ""}:
        score += 30
    elif ml_confidence == "medium":
        score += 15
    if 0.35 <= include_score <= 0.75 or 0.35 <= exclude_score <= 0.75 or 0.35 <= verify_score <= 0.75:
        score += 30
    if top_score and top_score < 0.70:
        score += 25
    if ml_adjustment and ml_adjustment != "none":
        score += 25
    if textify(row.get("matched_fields")):
        score += 15
    if status == "selected":
        score += 10
    if textify(row.get("source_policy_tier")).startswith("P0"):
        score += 8
    return (-score, textify(row.get("competitor")), textify(row.get("url")))


def build_training_review_sample(
    evidence_audit_rows: Sequence[Dict[str, Any]],
    sample_size: int = 40,
    product_category: str = "",
    product_type_key: str = "",
    product_type_label: str = "",
    own_product_name: str = "",
) -> List[Dict[str, Any]]:
    seen = set()
    candidates: List[Dict[str, Any]] = []
    for row in sorted(evidence_audit_rows, key=training_review_priority):
        url = textify(row.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        candidate = dict(row)
        candidate.setdefault("human_label", "")
        candidate.setdefault("human_reason", "")
        candidate.setdefault("use_as_primary_evidence", "")
        candidate.setdefault("reviewed_by", "")
        candidate.setdefault("reviewed_at", "")
        candidate.setdefault("product_category", product_category)
        candidate.setdefault("product_type_key", product_type_key or product_category)
        candidate.setdefault("product_type_label", product_type_label or product_category)
        candidate.setdefault("own_product_name", own_product_name)
        candidate.setdefault("search_card_candidate", "yes" if (product_type_key or product_category) else "")
        candidate["suggested_label"] = candidate.get("ml_label") or candidate.get("decision_status") or ""
        candidate["review_hint"] = (
            "请判断该来源最终应为 include / exclude / verify_later。"
            "规则硬拒绝项通常不应改为 include；若改为 include，请在 human_reason 写清公开证据理由。"
        )
        candidates.append(candidate)
        if len(candidates) >= max(0, sample_size):
            break
    return candidates


def write_training_review_sample(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# 人工抽样标注表",
        "",
        "本文件来自本轮证据审计的抽样结果，用于给本地筛选模型积累训练数据。",
        "",
        "填写方式：在 CSV 中补 `human_label`，可选值为 `include`、`exclude`、`verify_later`；再补 `human_reason` 说明判断理由。",
        "",
    ]
    if not rows:
        lines += ["本轮没有可抽样的证据审计行。", ""]
    for idx, row in enumerate(rows, start=1):
        lines += [
            f"## {idx}. {row.get('competitor')} | {row.get('title') or row.get('url')}",
            "",
            f"- **URL:** {row.get('url')}",
            f"- **搜索卡片归属:** {row.get('product_type_label') or row.get('product_category') or '未标注'} (`{row.get('product_type_key') or row.get('product_category') or ''}`)",
            f"- **规则状态:** {row.get('decision_status')} / {row.get('hard_gate')}",
            f"- **待核实:** {row.get('pending_verification')}；{row.get('verification_reason')}",
            f"- **本地模型:** {row.get('ml_label') or '未启用'}；收录 {row.get('ml_include_score') or '-'} / 排除 {row.get('ml_exclude_score') or '-'} / 待核实 {row.get('ml_verify_later_score') or '-'}",
            f"- **建议标注:** {row.get('suggested_label') or '人工判断'}",
            f"- **理由:** {truncate_text(row.get('reason', ''), 700)}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def strip_html_snapshot(text: str, limit: int = 5000) -> Tuple[str, str]:
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    body = (
        body.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    body = re.sub(r"\s+", " ", body).strip()
    return title[:180], body[:limit]


def is_video_review_url(url: str) -> bool:
    domain = domain_of(url)
    return domain_matches(domain, VIDEO_SOCIAL_DOMAINS) or any(
        token in url.lower()
        for token in ("youtube.com/watch", "youtu.be/", "bilibili.com/video", "tiktok.com", "douyin.com")
    )


def browser_snapshot(
    url: str,
    out_dir: Path,
    slug: str,
    timeout: int = 12,
) -> Tuple[str, str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return "", "", "playwright_not_available"
    screenshot_path = out_dir / "gui_review_snapshots" / f"{slug}.png"
    text_path = out_dir / "gui_review_snapshots" / f"{slug}.txt"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1365, "height": 900})
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(1200)
            visible_text = page.locator("body").inner_text(timeout=3000)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
        text_path.write_text(clean_text(visible_text, limit=6000), encoding="utf-8")
        return str(screenshot_path), str(text_path), ""
    except Exception as exc:
        return "", "", f"browser_snapshot_failed:{exc}"


def public_text_snapshot(
    url: str,
    out_dir: Path,
    slug: str,
    timeout: int = 12,
    proxy_url: str = "",
) -> Tuple[str, str]:
    text_path = out_dir / "gui_review_snapshots" / f"{slug}.txt"
    try:
        request = Request(url, headers={"User-Agent": "competitor-intel-harvester/1.0", "Accept": "text/html,text/plain,*/*"})
        if proxy_url and not is_local_url(url):
            opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
        else:
            opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1_000_000)
            content_type = response.headers.get("content-type", "")
        encoding = "utf-8"
        enc_match = re.search(r"charset=([^;\s]+)", content_type, re.I)
        if enc_match:
            encoding = enc_match.group(1)
        decoded = raw.decode(encoding, errors="replace")
        title, text = strip_html_snapshot(decoded)
        snapshot = "\n".join(part for part in [f"Title: {title}" if title else "", text] if part).strip()
        if not snapshot:
            return "", "empty_public_snapshot"
        text_path.write_text(snapshot, encoding="utf-8")
        return str(text_path), ""
    except Exception as exc:
        return "", f"public_snapshot_failed:{exc}"


def read_snapshot_excerpt(path: str, limit: int = 700) -> str:
    if not path:
        return ""
    try:
        return truncate_text(Path(path).read_text(encoding="utf-8", errors="replace"), limit)
    except OSError:
        return ""


GUI_REVIEW_PAGE_STATUSES = {
    "browser_snapshot_captured",
    "captured_public_snapshot",
    "login_assisted_snapshot_captured",
    "adapter_metadata_captured",
}


def title_from_snapshot_text(text: str) -> str:
    match = re.search(r"^\s*Title:\s*(.+?)\s*$", textify(text), re.I | re.M)
    return truncate_text(match.group(1), 180) if match else ""


def page_extracts_from_gui_review_rows(
    gui_review_rows: Sequence[Mapping[str, Any]],
    collection_plan: Optional[ProductCollectionPlan] = None,
) -> List[PageExtract]:
    pages: List[PageExtract] = []
    seen = set()
    for row in gui_review_rows:
        status = textify(row.get("automated_review_status")).lower()
        if status not in GUI_REVIEW_PAGE_STATUSES:
            continue
        text_path = textify(row.get("text_snapshot_path"))
        if not text_path:
            continue
        try:
            raw_text = Path(text_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        markdown = clean_text(raw_text, limit=30000)
        if not markdown:
            continue
        url = textify(row.get("canonical_url")) or review_target_url(row)
        if not url:
            continue
        title = textify(row.get("title")) or title_from_snapshot_text(markdown) or url
        if looks_like_login_form(url, title, markdown):
            continue
        key = (textify(row.get("competitor")), url)
        if key in seen:
            continue
        seen.add(key)
        image_urls: List[str] = []
        screenshot_path = textify(row.get("screenshot_path"))
        if screenshot_path:
            image_urls.append(screenshot_path)
        fields = infer_fields(markdown, collection_plan)
        fields.update(
            {
                "snapshot_source": "gui_review",
                "snapshot_status": status,
                "screenshot_path": screenshot_path,
            }
        )
        pages.append(
            PageExtract(
                competitor=key[0],
                url=url,
                title=title,
                markdown=markdown,
                text_excerpt=truncate_text(markdown, 2000),
                links=[],
                image_urls=image_urls,
                fields=fields,
                error="",
            )
        )
    return pages


def merge_page_extracts(
    pages: Sequence[PageExtract],
    additions: Sequence[PageExtract],
) -> List[PageExtract]:
    merged = list(pages)
    seen = {(page.competitor, page.url) for page in merged}
    for page in additions:
        key = (page.competitor, page.url)
        if key in seen:
            continue
        seen.add(key)
        merged.append(page)
    return merged


def login_assisted_browser_snapshot(
    url: str,
    out_dir: Path,
    slug: str,
    wait_seconds: int = 120,
    timeout: int = 12,
    proxy_url: str = "",
) -> Tuple[str, str, str, str]:
    """Open a visible browser so the user can log in, then save an authorized snapshot if readable."""
    screenshot_path = out_dir / "gui_review_snapshots" / f"{slug}-login-assisted.png"
    text_path = out_dir / "gui_review_snapshots" / f"{slug}-login-assisted.txt"
    profile_dir = out_dir / "login_assist_profile"
    wait_seconds = max(10, int(wait_seconds or 0))
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        note_path = out_dir / "gui_review_snapshots" / f"{slug}-login-required.txt"
        note_path.write_text(
            "\n".join(
                [
                    "Playwright 不可用，无法在工具内等待登录后自动截取页面。",
                    "工具不会自动改用系统浏览器打开登录页；请回到 UI 登录池点击链接后再处理。",
                    f"URL: {url}",
                    "请用有权限的账号登录/注册后，重新运行该竞品或人工补充截图/摘录。",
                    f"错误: {exc}",
                ]
            ),
            encoding="utf-8",
        )
        return "", str(note_path), "requires_user_login", f"playwright_not_available:{exc}"

    print(f"[LOGIN] Opening browser for user login: {url}", flush=True)
    print(f"[LOGIN] Waiting up to {wait_seconds}s for the page to become readable after login/register.", flush=True)
    try:
        with sync_playwright() as p:
            launch_kwargs: Dict[str, Any] = {"headless": False}
            if proxy_url and not is_local_url(url):
                launch_kwargs["proxy"] = {"server": proxy_url}
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                viewport={"width": 1365, "height": 900},
                **launch_kwargs,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            deadline = time.time() + wait_seconds
            final_title = ""
            final_text = ""
            stable_readable_checks = 0
            while time.time() < deadline:
                page.wait_for_timeout(2500)
                try:
                    final_title = page.title()
                except Exception:
                    final_title = ""
                try:
                    final_text = page.locator("body").inner_text(timeout=4000)
                except Exception:
                    final_text = ""
                cleaned = clean_text(final_text, limit=8000)
                if cleaned and not looks_like_login_form(page.url, final_title, cleaned) and len(cleaned) >= 240:
                    stable_readable_checks += 1
                    if stable_readable_checks >= 2:
                        break
                else:
                    stable_readable_checks = 0
            screenshot_file = ""
            try:
                page.screenshot(path=str(screenshot_path), full_page=True)
                screenshot_file = str(screenshot_path)
            except Exception:
                screenshot_file = ""
            try:
                final_title = final_title or page.title()
                final_text = final_text or page.locator("body").inner_text(timeout=4000)
            except Exception:
                pass
            context.close()
        snapshot = clean_text("\n".join(part for part in [f"Title: {final_title}" if final_title else "", final_text] if part), limit=10000)
        if snapshot:
            text_path.write_text(snapshot, encoding="utf-8")
        if snapshot and not looks_like_login_form(url, final_title, snapshot) and len(snapshot) >= 240:
            return screenshot_file, str(text_path), "login_assisted_snapshot_captured", ""
        return (
            screenshot_file,
            str(text_path) if snapshot else "",
            "login_assist_still_requires_login",
            "page still looks like login/register/access-controlled content after waiting",
        )
    except Exception as exc:
        note_path = out_dir / "gui_review_snapshots" / f"{slug}-login-required.txt"
        note_path.write_text(
            "\n".join(
                [
                    "可见浏览器登录辅助失败。",
                    "工具不会自动改用系统浏览器打开登录页；请回到 UI 登录池点击链接后再处理。",
                    f"URL: {url}",
                    "请用有权限的账号登录/注册后，重新运行该竞品或人工补充截图/摘录。",
                    f"错误: {exc}",
                ]
            ),
            encoding="utf-8",
        )
        return "", str(note_path), "requires_user_login", f"login_assist_failed:{exc}"


class LoginAssistSession:
    def __init__(
        self,
        out_dir: Path,
        proxy_url: str = "",
        timeout: int = 12,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.proxy_url = proxy_url
        self.timeout = max(3, int(timeout or 12))
        self.profile_dir = self.out_dir / "login_assist_profile"
        self.snapshot_dir = self.out_dir / "gui_review_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.queued_urls_by_key: Dict[Tuple[str, str], List[str]] = {}
        self.pages_by_key: Dict[Tuple[str, str], Any] = {}
        self.results_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.playwright: Any = None
        self.context: Any = None
        self.start_error = ""

    def start(self) -> bool:
        if self.context is not None:
            return True
        try:
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright().start()
            launch_kwargs: Dict[str, Any] = {"headless": False}
            if self.proxy_url:
                launch_kwargs["proxy"] = {"server": self.proxy_url}
            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                viewport={"width": 1365, "height": 900},
                **launch_kwargs,
            )
            print("[LOGIN] Login assist browser opened once; queued pages will reuse this logged-in profile.", flush=True)
            return True
        except Exception as exc:
            self.start_error = f"login_assist_start_failed:{exc}"
            self.close()
            print(f"[LOGIN] Login assist browser unavailable: {exc}", flush=True)
            return False

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        added = 0
        for row in dedupe_login_review_rows(rows):
            url = review_target_url(row)
            if not url:
                continue
            key = login_queue_key_for(row)
            if key in self.rows_by_key:
                for queued_url in textify(row.get("queued_urls")).splitlines() or [url]:
                    if queued_url and queued_url not in self.queued_urls_by_key[key]:
                        self.queued_urls_by_key[key].append(queued_url)
                self.rows_by_key[key]["queued_url_count"] = str(len(self.queued_urls_by_key[key]))
                self.rows_by_key[key]["queued_urls"] = "\n".join(self.queued_urls_by_key[key])
                continue
            item = dict(row)
            queued_urls = [value for value in textify(row.get("queued_urls")).splitlines() if value] or [url]
            self.rows_by_key[key] = item
            self.queued_urls_by_key[key] = list(dict.fromkeys(queued_urls))
            item["queued_url_count"] = str(len(self.queued_urls_by_key[key]))
            item["queued_urls"] = "\n".join(self.queued_urls_by_key[key])
            added += 1
        if added:
            print(
                f"[LOGIN] Added {added} unique login-required site(s) to the login queue; "
                "browser pages open only after the UI login link is clicked.",
                flush=True,
            )
        return added

    def _open_or_reuse_page(self, key: Tuple[str, str], url: str) -> Any:
        if not self.context or not url:
            return None
        page = self.pages_by_key.get(key)
        should_navigate = False
        try:
            if page is None or page.is_closed():
                blank_pages = [candidate for candidate in self.context.pages if candidate.url == "about:blank"]
                page = blank_pages[0] if blank_pages else self.context.new_page()
                self.pages_by_key[key] = page
                should_navigate = True
            else:
                try:
                    should_navigate = textify(page.url) == "about:blank"
                except Exception:
                    should_navigate = True
            if should_navigate:
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
        except Exception as exc:
            self.results_by_key[key] = self._result_from_row(
                self.rows_by_key[key],
                "awaiting_user_login",
                next_step=f"登录页已进入等待区，页面打开可能较慢或被重定向：{exc}",
            )
        return page

    def _result_from_row(
        self,
        row: Mapping[str, Any],
        status: str,
        text_path: str = "",
        screenshot_path: str = "",
        excerpt: str = "",
        next_step: str = "",
    ) -> Dict[str, Any]:
        url = review_target_url(row)
        return {
            "competitor": textify(row.get("competitor")),
            "priority": textify(row.get("priority")) or "P0-LOGIN",
            "review_reason": textify(row.get("review_reason")) or "login_required_user_action",
            "requires_user_login": "yes",
            "title": textify(row.get("title")),
            "url": url,
            "domain": login_queue_domain_key_for_url(url) or domain_of(url) or textify(row.get("domain")),
            "adapter_name": "",
            "source_family": "",
            "platform": "",
            "canonical_url": url,
            "automated_review_status": status,
            "text_snapshot_path": text_path,
            "screenshot_path": screenshot_path,
            "metadata_path": "",
            "transcript_path": "",
            "evidence_markers_path": "",
            "needs_manual_video_timestamp": "no",
            "login_assist_url": url,
            "text_snapshot_excerpt": excerpt,
            "allowed_boundary": textify(row.get("allowed_boundary")) or login_review_allowed_boundary(),
            "next_step": next_step or login_review_next_step(),
            "queued_url_count": textify(row.get("queued_url_count")) or "1",
            "queued_urls": textify(row.get("queued_urls")) or url,
        }

    def queue_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, row in self.rows_by_key.items():
            if login_skip_requested(self.out_dir, row):
                continue
            result = self.results_by_key.get(key)
            if result and textify(result.get("automated_review_status")) in {"login_assisted_snapshot_captured", "login_skipped_by_user"}:
                continue
            pending = self._result_from_row(
                row,
                textify(result.get("automated_review_status")) if result else "awaiting_user_login",
                text_path=textify(result.get("text_snapshot_path")) if result else "",
                screenshot_path=textify(result.get("screenshot_path")) if result else "",
                excerpt=textify(result.get("text_snapshot_excerpt")) if result else "",
                next_step=textify(result.get("next_step")) if result else "已加入登录等待区；工具不会主动弹出网页，点击 UI 登录链接后才会打开该站点。",
            )
            pending["queued_url_count"] = str(len(self.queued_urls_by_key.get(key, [])) or 1)
            pending["queued_urls"] = "\n".join(self.queued_urls_by_key.get(key, [])) or pending["url"]
            rows.append(pending)
        return rows

    def capture_all(self, wait_seconds: int = 120) -> List[Dict[str, Any]]:
        if not self.rows_by_key:
            return []

        wait_seconds = max(0, int(wait_seconds or 0))
        print(
            f"[LOGIN] Waiting up to {wait_seconds}s after public crawling for UI-clicked login pages.",
            flush=True,
        )
        pending = set(self.rows_by_key)
        stable_checks: Dict[Tuple[str, str], int] = {key: 0 for key in pending}
        deadline = time.time() + wait_seconds
        while pending and time.time() < deadline:
            for key in list(pending):
                row = self.rows_by_key[key]
                if login_skip_requested(self.out_dir, row):
                    self.results_by_key[key] = self._result_from_row(
                        row,
                        "login_skipped_by_user",
                        next_step="用户已在 UI 登录池选择跳过；该站点不再进入登录等待。",
                    )
                    pending.remove(key)
                    continue
                if not login_click_requested(self.out_dir, row):
                    continue
                if not self.context and not self.start():
                    self.results_by_key[key] = self._result_from_row(
                        row,
                        "requires_user_login",
                        next_step=f"用户已点击登录池链接，但登录辅助浏览器不可用，已保留在需登录队列：{self.start_error}",
                    )
                    pending.remove(key)
                    continue
                page = self._open_or_reuse_page(key, review_target_url(row))
                readable, title, cleaned = self._readable_snapshot(page, row)
                if readable:
                    stable_checks[key] = stable_checks.get(key, 0) + 1
                    if stable_checks[key] >= 2:
                        pending.remove(key)
                else:
                    stable_checks[key] = 0
            if pending:
                time.sleep(2.5)

        rows: List[Dict[str, Any]] = []
        for index, (key, row) in enumerate(self.rows_by_key.items(), start=1):
            key_results = self._capture_key_results(index, key, row)
            if key_results:
                self.results_by_key[key] = key_results[0]
                rows.extend(key_results)
        return rows

    def _readable_snapshot(self, page: Any, row: Mapping[str, Any]) -> Tuple[bool, str, str]:
        if page is None:
            return False, "", ""
        try:
            title = textify(page.title())
        except Exception:
            title = ""
        try:
            body_text = textify(page.locator("body").inner_text(timeout=4000))
        except Exception:
            body_text = ""
        cleaned = clean_text(body_text, limit=10000)
        page_url = ""
        try:
            page_url = textify(page.url)
        except Exception:
            page_url = review_target_url(row)
        readable = bool(cleaned and not looks_like_login_form(page_url, title, cleaned) and len(cleaned) >= 240)
        return readable, title, cleaned

    def _capture_page(self, index: int, page: Any, row: Mapping[str, Any], ordinal: int = 0) -> Tuple[str, str, str, str, str]:
        url = review_target_url(row)
        index_label = f"{index:03d}" if ordinal <= 1 else f"{index:03d}-{ordinal:02d}"
        slug = f"{index_label}-{slugify(textify(row.get('competitor')) or site_domain_of(url) or domain_of(url) or 'login')}"
        screenshot_path = self.snapshot_dir / f"{slug}-login-assisted.png"
        text_path = self.snapshot_dir / f"{slug}-login-assisted.txt"
        if login_skip_requested(self.out_dir, row):
            return (
                "login_skipped_by_user",
                "",
                "",
                "",
                "用户已在 UI 登录池选择跳过；该站点不再进入登录等待。",
            )
        if not login_click_requested(self.out_dir, row):
            return (
                "awaiting_user_login",
                "",
                "",
                "",
                "用户尚未点击 UI 登录池链接；工具未主动打开该网页，已保留在需登录队列。",
            )
        readable, title, cleaned = self._readable_snapshot(page, row)
        screenshot_file = ""
        if page is not None:
            try:
                page.screenshot(path=str(screenshot_path), full_page=True)
                screenshot_file = str(screenshot_path)
            except Exception:
                screenshot_file = ""
        if cleaned:
            snapshot = "\n".join(part for part in [f"Title: {title}" if title else "", cleaned] if part)
            text_path.write_text(snapshot, encoding="utf-8")
            text_file = str(text_path)
        else:
            text_file = ""
        if readable:
            return (
                "login_assisted_snapshot_captured",
                text_file,
                screenshot_file,
                read_snapshot_excerpt(text_file),
                "登录后页面已保存文本快照和截图；已重新纳入分析候选。",
            )
        return (
            "login_assist_timeout",
            text_file,
            screenshot_file,
            read_snapshot_excerpt(text_file),
            "公开页面采集已结束并等待登录超时；该站点保留到问题页面核验清单。",
        )

    def _capture_key_results(self, index: int, key: Tuple[str, str], row: Mapping[str, Any]) -> List[Dict[str, Any]]:
        page = self.pages_by_key.get(key)
        clicked = login_click_requested(self.out_dir, row)
        skipped = login_skip_requested(self.out_dir, row)
        if skipped or not clicked:
            status, text_path, screenshot_path, excerpt, next_step = self._capture_page(index, page, row)
            return [
                self._result_from_row(
                    row,
                    status,
                    text_path=text_path,
                    screenshot_path=screenshot_path,
                    excerpt=excerpt,
                    next_step=next_step,
                )
            ]
        readable, _title, _cleaned = self._readable_snapshot(page, row)
        if not readable:
            status, text_path, screenshot_path, excerpt, next_step = self._capture_page(index, page, row)
            return [
                self._result_from_row(
                    row,
                    status,
                    text_path=text_path,
                    screenshot_path=screenshot_path,
                    excerpt=excerpt,
                    next_step=next_step,
                )
            ]

        queued_urls = self.queued_urls_by_key.get(key) or [review_target_url(row)]
        unique_urls = [url for url in dict.fromkeys(queued_urls) if url]
        if not unique_urls:
            unique_urls = [review_target_url(row)]
        results: List[Dict[str, Any]] = []
        for ordinal, url in enumerate(unique_urls, start=1):
            capture_row = dict(row)
            capture_row["url"] = url
            capture_row["login_assist_url"] = url
            capture_row["queued_url_count"] = str(len(unique_urls))
            capture_row["queued_urls"] = "\n".join(unique_urls)
            try:
                if page is not None and canonical_url_for_dedupe(textify(page.url)) != canonical_url_for_dedupe(url):
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
            except Exception as exc:
                result = self._result_from_row(
                    capture_row,
                    "login_assist_timeout",
                    next_step=f"登录态已复用，但访问同站点排队 URL 失败：{exc}",
                )
                results.append(result)
                continue
            status, text_path, screenshot_path, excerpt, next_step = self._capture_page(
                index,
                page,
                capture_row,
                ordinal=ordinal if len(unique_urls) > 1 else 0,
            )
            results.append(
                self._result_from_row(
                    capture_row,
                    status,
                    text_path=text_path,
                    screenshot_path=screenshot_path,
                    excerpt=excerpt,
                    next_step=next_step,
                )
            )
        return results

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
        except Exception:
            pass
        self.context = None
        try:
            if self.playwright is not None:
                self.playwright.stop()
        except Exception:
            pass
        self.playwright = None


def execute_gui_review_queue(
    manual_review_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    max_items: int = 8,
    enable_browser: bool = True,
    proxy_url: str = "",
    timeout: int = 12,
    login_assist: bool = False,
    login_assist_wait_seconds: int = 120,
) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    (out_dir / "gui_review_snapshots").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manual_review_rows[: max(0, max_items)], start=1):
        url = review_target_url(row)
        slug = f"{idx:03d}-{slugify(textify(row.get('competitor')) or domain_of(url) or 'review')}"
        requires_login = row_requires_login_action(row)
        result = {
            "competitor": textify(row.get("competitor")),
            "priority": textify(row.get("priority")),
            "review_reason": textify(row.get("review_reason")),
            "requires_user_login": "yes" if requires_login else "no",
            "title": textify(row.get("title")),
            "url": url,
            "domain": domain_of(url) or textify(row.get("domain")),
            "adapter_name": "",
            "source_family": "",
            "platform": "",
            "canonical_url": url,
            "automated_review_status": "",
            "text_snapshot_path": "",
            "screenshot_path": "",
            "metadata_path": "",
            "transcript_path": "",
            "evidence_markers_path": "",
            "needs_manual_video_timestamp": "no",
            "login_assist_url": url if requires_login else textify(row.get("login_assist_url")),
            "text_snapshot_excerpt": "",
            "allowed_boundary": login_review_allowed_boundary()
            if requires_login
            else "只处理公开可见内容；不破解验证码、不登录、不绕过付费或访问控制、不调用未授权私有接口。",
            "next_step": "",
        }
        if not url:
            result["automated_review_status"] = "missing_url"
            result["next_step"] = "缺少 URL，不能自动复核。"
            rows.append(result)
            continue
        if requires_login:
            result.update(
                {
                    "automated_review_status": "requires_user_login",
                    "next_step": login_review_next_step(),
                }
            )
            rows.append(result)
            continue
        adapter_info = classify_source_url(url)
        if adapter_info.get("adapter_name"):
            adapter_result = collect_adapter_snapshot(
                url,
                title=result["title"],
                snippet=textify(row.get("cleaned_excerpt_sample") or row.get("review_reason") or row.get("suggested_next_step")),
                out_dir=out_dir,
                slug=slug,
                timeout=timeout,
                proxy_url=proxy_url,
            )
            source_family = adapter_result.get("source_family", "")
            review_status = adapter_result.get("automated_review_status", "adapter_metadata_captured")
            needs_video_timestamp = adapter_result.get("needs_manual_video_timestamp") or "no"
            if source_family == "video_social" and review_status == "adapter_metadata_failed":
                needs_video_timestamp = "yes"
            if source_family == "video_social" and needs_video_timestamp == "yes":
                review_status = "video_metadata_pending_timestamp"
            result.update(
                {
                    "adapter_name": adapter_result.get("adapter_name", ""),
                    "source_family": source_family,
                    "platform": adapter_result.get("platform", ""),
                    "canonical_url": adapter_result.get("canonical_url") or url,
                    "automated_review_status": review_status,
                    "text_snapshot_path": adapter_result.get("text_snapshot_path", ""),
                    "screenshot_path": adapter_result.get("screenshot_path", ""),
                    "metadata_path": adapter_result.get("metadata_path", ""),
                    "transcript_path": adapter_result.get("transcript_path", ""),
                    "evidence_markers_path": adapter_result.get("evidence_markers_path", ""),
                    "needs_manual_video_timestamp": needs_video_timestamp,
                    "text_snapshot_excerpt": adapter_result.get("text_snapshot_excerpt", ""),
                    "next_step": adapter_result.get("adapter_next_step") or "已保存平台公开元数据；进入强事实前需人工核验。",
                }
            )
            rows.append(result)
            continue
        if is_video_review_url(url):
            metadata_path = out_dir / "gui_review_snapshots" / f"{slug}-metadata.json"
            metadata = {
                "url": url,
                "title": result["title"],
                "domain": result["domain"],
                "captured_at": utc_stamp(),
                "note": "公开视频/社媒内容先保存公开元数据；正式报告还需要时间点、截图或字幕证据。",
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            result.update(
                {
                    "automated_review_status": "video_metadata_pending_timestamp",
                    "metadata_path": str(metadata_path),
                    "needs_manual_video_timestamp": "yes",
                    "next_step": "需要补充观点出现的时间点、截图或公开字幕后，才可进入正式事实证据。",
                }
            )
            rows.append(result)
            continue

        browser_error = ""
        if enable_browser:
            screenshot_path, text_path, browser_error = browser_snapshot(url, out_dir, slug, timeout)
            if text_path:
                result.update(
                    {
                    "automated_review_status": "browser_snapshot_captured",
                    "text_snapshot_path": text_path,
                    "screenshot_path": screenshot_path,
                    "text_snapshot_excerpt": read_snapshot_excerpt(text_path),
                    "next_step": "已保存浏览器公开快照；人工只需核验其中是否有可引用证据。",
                }
            )
                rows.append(result)
                continue
        text_path, text_error = public_text_snapshot(url, out_dir, slug, timeout, proxy_url)
        if text_path:
            result.update(
                {
                    "automated_review_status": "captured_public_snapshot",
                    "text_snapshot_path": text_path,
                    "text_snapshot_excerpt": read_snapshot_excerpt(text_path),
                    "next_step": "已保存公开文本快照；若需要视觉证据，再补 GUI 截图。",
                }
            )
        else:
            result.update(
                {
                    "automated_review_status": "needs_manual_gui",
                    "next_step": f"自动公开快照失败，保留人工 GUI 复核：{browser_error or text_error}",
                }
            )
        rows.append(result)

    write_csv(
        out_dir / "gui_review_results.csv",
        rows,
        GUI_REVIEW_FIELDS,
    )
    write_gui_review_results_markdown(out_dir / "gui_review_results.md", rows)
    return rows


def write_gui_review_results_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# GUI 自动复核结果",
        "",
        "本文件记录问题页面核验清单的内部自动公开快照结果。公开页会直接保存文本或截图；登录/注册页会进入集中登录队列，用户授权后保存登录辅助快照。",
        "",
        "边界：不破解验证码，不绕过登录、付费墙或访问控制，不保存账号凭据，不做未授权私有接口逆向。",
        "",
    ]
    if not rows:
        lines += ["本轮没有需要自动复核的候选。", ""]
    for idx, row in enumerate(rows, start=1):
        lines += [
            f"## {idx}. {row.get('competitor')} | {row.get('title') or row.get('url')}",
            "",
            f"- **URL:** {row.get('url')}",
            f"- **状态:** {row.get('automated_review_status')}",
            f"- **平台适配器:** {row.get('platform') or '无'} / {row.get('adapter_name') or '无'}",
            f"- **来源类型:** {row.get('source_family') or '无'}",
            f"- **规范 URL:** {row.get('canonical_url') or row.get('url')}",
            f"- **需用户登录:** {row.get('requires_user_login') or 'no'}",
            f"- **登录辅助入口:** {row.get('login_assist_url') or '无'}",
            f"- **文本快照:** {row.get('text_snapshot_path') or '无'}",
            f"- **截图:** {row.get('screenshot_path') or '无'}",
            f"- **元数据:** {row.get('metadata_path') or '无'}",
            f"- **字幕/转写:** {row.get('transcript_path') or '无'}",
            f"- **时间点证据:** {row.get('evidence_markers_path') or '无'}",
            f"- **视频时间点:** {row.get('needs_manual_video_timestamp')}",
            f"- **下一步:** {row.get('next_step')}",
            "",
        ]
        if row.get("text_snapshot_excerpt"):
            lines += [
                "**快照摘要:**",
                "",
                truncate_text(row.get("text_snapshot_excerpt", ""), 700),
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def login_required_queue_rows(
    manual_review_rows: Sequence[Mapping[str, Any]],
    gui_review_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    gui_by_key = {
        login_queue_key_for(row): row
        for row in gui_review_rows
    }
    rows: List[Dict[str, Any]] = []
    seen = set()
    for row in dedupe_login_review_rows(manual_review_rows):
        if not row_requires_login_action(row):
            continue
        target_url = review_target_url(row)
        key = login_queue_key_for(row)
        if key in seen:
            continue
        seen.add(key)
        gui = gui_by_key.get(key, {})
        if textify(gui.get("automated_review_status")).lower() in {"login_assisted_snapshot_captured", "login_skipped_by_user"}:
            continue
        rows.append(
            {
                "competitor": key[0],
                "priority": textify(row.get("priority")) or "P0-LOGIN",
                "review_reason": textify(row.get("review_reason")) or "login_required_user_action",
                "title": textify(row.get("title")),
                "url": target_url,
                "domain": login_queue_domain_key_for_url(target_url) or domain_of(target_url) or textify(row.get("domain")),
                "queued_url_count": textify(row.get("queued_url_count")) or "1",
                "login_assist_url": target_url,
                "queued_urls": textify(row.get("queued_urls")) or textify(row.get("url")) or target_url,
                "automated_review_status": textify(gui.get("automated_review_status")) or "requires_user_login",
                "text_snapshot_path": textify(gui.get("text_snapshot_path")),
                "screenshot_path": textify(gui.get("screenshot_path")),
                "text_snapshot_excerpt": textify(gui.get("text_snapshot_excerpt")),
                "next_step": textify(gui.get("next_step")) or textify(row.get("suggested_next_step")) or login_review_next_step(),
                "allowed_boundary": textify(row.get("allowed_boundary")) or login_review_allowed_boundary(),
            }
        )
    return rows


def write_login_required_queue(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# 需登录队列",
        "",
        "这些页面疑似需要登录、注册、验证码或账号权限。工具会按竞品和域名去重，统一放入登录等待区；公开页面会继续采集。只有用户在 UI 登录池点击对应链接后，工具才会打开该站点并尝试保存登录后快照，仍不可读的会留在本队列。",
        "",
        "边界：只处理用户本人有权限访问的信息，不破解验证码，不绕过登录/付费/访问控制，不保存账号凭据。",
        "",
    ]
    if not rows:
        lines += ["本轮没有检测到需登录页面。", ""]
    else:
        lines += [
            "| 优先级 | 竞品 | 状态 | 同站点排队 URL | 登录入口 | 待补采 URL | 快照 | 截图 | 下一步 |",
            "|---|---|---|---:|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        row.get("priority"),
                        row.get("competitor"),
                        row.get("automated_review_status"),
                        row.get("queued_url_count") or "1",
                        row.get("url"),
                        row.get("queued_urls") or row.get("login_assist_url") or row.get("url"),
                        row.get("text_snapshot_path") or "无",
                        row.get("screenshot_path") or "无",
                        row.get("next_step"),
                    )
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def problem_type_for(status: str, reason: str = "", requires_login: str = "", needs_video_timestamp: str = "") -> str:
    haystack = f"{status} {reason}".lower()
    if "login_assist_timeout" in haystack or "timeout" in haystack and requires_login == "yes":
        return "超时未人工登录"
    if requires_login == "yes" or "login_required_user_action" in haystack:
        return "需登录/注册/账号权限"
    if "http 403" in haystack or " 403" in haystack or "forbidden" in haystack:
        return "HTTP 403 / 反爬拦截"
    if "cloudflare" in haystack or "js challenge" in haystack:
        return "Cloudflare / JS Challenge"
    if "datadome" in haystack or "captcha" in haystack or "验证码" in haystack:
        return "验证码 / DataDome"
    if needs_video_timestamp == "yes" or "video_metadata_pending_timestamp" in haystack:
        return "视频缺时间点证据"
    if "minimal_text" in haystack or "script_heavy_shell" in haystack or "low_text" in haystack or "正文不足" in haystack:
        return "JS 空壳 / 正文不足"
    if "pending" in haystack or "verify" in haystack or "待核实" in haystack:
        return "待核实来源"
    if "404" in haystack or "broken" in haystack:
        return "404 / 失效页面"
    if "timed out" in haystack or "timeout" in haystack or "超时" in haystack:
        return "请求超时"
    if "anti-bot" in haystack or "antibot" in haystack:
        return "反爬拦截"
    return "其他需核验问题"


def what_to_verify_for_problem(problem_type: str, page_role: str, source_kind: str, fact_type: str) -> str:
    if problem_type == "超时未人工登录":
        return "确认用户是否已经用有权限账号登录；登录后页面是否公开/授权可见；是否包含本轮需要的定价、参数、功能、接口、评价或截图证据。"
    if problem_type == "需登录/注册/账号权限":
        return "确认该页面是否必须登录；如果登录后只显示个人账户、订单、购物车或私密信息，应排除；如果是公开/授权可见产品资料，可人工补证。"
    if problem_type in {"HTTP 403 / 反爬拦截", "Cloudflare / JS Challenge", "验证码 / DataDome", "反爬拦截"}:
        return "先确认原页面是否公开且确有产品情报价值；优先找同站公开替代入口、sitemap、官方文档、帮助中心或静态页面补证。"
    if problem_type == "视频缺时间点证据":
        return "核验视频是否绑定目标竞品和本轮问题；补观点出现的时间点、公开截图或字幕摘录，才能进入正式事实。"
    if problem_type == "JS 空壳 / 正文不足":
        return "用浏览器确认页面是否真实承载产品信息；若只是导航、登录壳或广告页则排除，若是核心页则补截图、公开文本或同站替代页。"
    if problem_type == "待核实来源":
        return "追溯到原始来源；确认标题、URL、作者/平台、发布时间和证据句是否完整，二创不可追溯时保持待核实或排除。"
    if problem_type == "404 / 失效页面":
        return "确认是否真的失效；若它是官方核心路径，尝试从官网导航、sitemap 或搜索结果找到等价新地址。"
    if page_role in {"pricing_packaging", "docs_api_or_developer", "product_specs_or_features", "physical_product_detail", "ai_capability_detail"}:
        return "核验该页面是否能提供可入库的核心字段，并补齐来源、证据句、截图或字段值。"
    if source_kind in {"third_party_verification_source", "community_or_social_signal"}:
        return "确认它只能做第三方验证或用户线索；不要覆盖官网事实，必要时追溯原始出处。"
    if fact_type:
        return f"核验是否能支撑 `{fact_type}` 相关事实；能支撑则补证，不能支撑则排除。"
    return "确认是否绑定目标竞品、是否回答本轮决策问题、是否有信息增量、是否可追溯、是否可公开获取。"


def data_entry_decision_for_problem(problem_type: str) -> str:
    if problem_type in {"超时未人工登录", "需登录/注册/账号权限"}:
        return "登录后只保留用户有权限且可用于研究的内容；可引用则标 include，仍缺证据标 verify_later，涉及个人/私密/账号数据标 exclude。"
    if problem_type == "视频缺时间点证据":
        return "补齐 URL、时间点、截图或字幕后标 include；只有线索但证据不完整标 verify_later；无关视频标 exclude。"
    if problem_type in {"HTTP 403 / 反爬拦截", "Cloudflare / JS Challenge", "验证码 / DataDome", "反爬拦截"}:
        return "找到公开替代证据后标 include；暂时只能确认可能有价值标 verify_later；无法公开获取或无关标 exclude。"
    return "有可追溯证据且能进入分析标 include；需要补来源/截图/时间点标 verify_later；无价值、不可追溯或越界标 exclude。"


def suggested_label_for_problem(problem_type: str) -> str:
    if problem_type in {"需登录/注册/账号权限", "超时未人工登录", "视频缺时间点证据", "待核实来源"}:
        return "verify_later"
    if problem_type in {"404 / 失效页面"}:
        return "exclude"
    return "verify_later"


def build_problem_review_rows(
    pages: Sequence[PageExtract],
    manual_review_rows: Sequence[Mapping[str, Any]],
    login_required_rows: Sequence[Mapping[str, Any]],
    gui_review_rows: Sequence[Mapping[str, Any]],
    evidence_audit_rows: Sequence[Mapping[str, Any]],
    training_review_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    def add(source_queue: str, raw: Mapping[str, Any], status: str = "", reason: str = "") -> None:
        url = review_target_url(raw)
        competitor = textify(raw.get("competitor"))
        domain = domain_of(url) or textify(raw.get("domain"))
        requires_login = "yes" if row_requires_login_action(raw) else "no"
        needs_video_timestamp = "yes" if textify(raw.get("needs_manual_video_timestamp")).lower() == "yes" else "no"
        status_text = status or textify(raw.get("automated_review_status") or raw.get("review_reason") or raw.get("hard_gate") or raw.get("status"))
        reason_text = reason or textify(raw.get("reason") or raw.get("verification_reason") or raw.get("crawl_error") or raw.get("next_step") or raw.get("suggested_next_step"))
        problem_type = problem_type_for(status_text, reason_text, requires_login, needs_video_timestamp)
        key = (competitor, url, problem_type)
        if key in seen or not url:
            return
        seen.add(key)
        page_role = textify(raw.get("page_role"))
        source_kind = textify(raw.get("source_kind"))
        fact_type = textify(raw.get("fact_type"))
        what_to_verify = what_to_verify_for_problem(problem_type, page_role, source_kind, fact_type)
        rows.append(
            {
                "competitor": competitor,
                "priority": textify(raw.get("priority")) or manual_review_priority_from_evidence(raw),
                "problem_type": problem_type,
                "source_queue": source_queue,
                "title": textify(raw.get("title")),
                "url": url,
                "domain": domain,
                "status": status_text,
                "source_kind": source_kind,
                "page_role": page_role,
                "source_policy_tier": textify(raw.get("source_policy_tier")),
                "pending_verification": textify(raw.get("pending_verification")) or ("yes" if problem_type != "404 / 失效页面" else "no"),
                "verification_reason": textify(raw.get("verification_reason")) or reason_text,
                "fact_type": fact_type,
                "increment_type": textify(raw.get("increment_type")),
                "fact_group": textify(raw.get("fact_group")),
                "reason": f"{problem_type}：{reason_text or status_text or '需要人工核验'}",
                "what_to_verify": what_to_verify,
                "data_entry_decision": data_entry_decision_for_problem(problem_type),
                "suggested_human_label": textify(raw.get("suggested_label")) or suggested_label_for_problem(problem_type),
                "human_label": textify(raw.get("human_label")),
                "human_reason": textify(raw.get("human_reason")),
                "use_as_primary_evidence": textify(raw.get("use_as_primary_evidence")),
                "reviewed_by": textify(raw.get("reviewed_by")),
                "reviewed_at": textify(raw.get("reviewed_at")),
                "model_feedback_status": "ready_for_training" if textify(raw.get("human_label")) else "pending_human_review",
                "text_snapshot_path": textify(raw.get("text_snapshot_path")),
                "screenshot_path": textify(raw.get("screenshot_path")),
                "metadata_path": textify(raw.get("metadata_path")),
                "evidence_markers_path": textify(raw.get("evidence_markers_path")),
                "allowed_boundary": textify(raw.get("allowed_boundary")) or "只处理公开或用户有权限访问的内容；不破解验证码、不绕过登录/付费/访问控制、不保存账号凭据。",
            }
        )

    for page in pages:
        if not page.error:
            continue
        add(
            "crawl4ai_page",
            {
                "competitor": page.competitor,
                "title": page.title,
                "url": page.url,
                "domain": domain_of(page.url),
                "status": page.error,
                "reason": page.error,
                "page_role": page_role_for_result(SearchResult(page.competitor, "crawled_page", "", page.title, page.url, page.text_excerpt)),
                "text_snapshot_path": "",
                "screenshot_path": "",
            },
            status=page.error,
            reason=page.error,
        )
    for row in manual_review_rows:
        add("manual_review_queue", row)
    for row in login_required_rows:
        add("login_required_queue", row)
    for row in gui_review_rows:
        status = textify(row.get("automated_review_status")).lower()
        needs_video_timestamp = textify(row.get("needs_manual_video_timestamp")).lower() == "yes"
        if status in {"browser_snapshot_captured", "captured_public_snapshot", "adapter_metadata_captured", "login_assisted_snapshot_captured"} and not needs_video_timestamp:
            continue
        add("gui_review_results", row)
    for row in evidence_audit_rows:
        if textify(row.get("pending_verification")).lower() != "yes" and textify(row.get("gui_review_candidate")).lower() != "yes":
            continue
        add("evidence_audit", row)
    for row in training_review_rows:
        if textify(row.get("human_label")):
            add("training_review_sample", row)

    priority_order = {"P0-LOGIN": 0, "P0": 1, "P1": 2, "P2": 3}
    return sorted(
        rows,
        key=lambda row: (
            priority_order.get(textify(row.get("priority")), 9),
            textify(row.get("problem_type")),
            textify(row.get("competitor")),
            textify(row.get("url")),
        ),
    )


def write_problem_review_outputs(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_dir = Path(out_dir)
    write_csv(out_dir / "problem_pages_review.csv", [dict(row) for row in rows], PROBLEM_REVIEW_FIELDS)
    lines = [
        "# 问题页面核验清单",
        "",
        "这里合并所有需要人工确认的问题页面：反爬、403、Cloudflare、验证码、正文不足、视频缺时间点、登录超时、待核实来源等。",
        "",
        "核验后请在 CSV 中填写 `human_label` 和 `human_reason`。`include` 表示可入库，`verify_later` 表示需要补证，`exclude` 表示排除。训练时程序会读取这些人工标签，更新本地 `.pt` 筛选模型。",
        "",
        "边界：只处理公开或用户有权限访问的内容；不破解验证码，不绕过登录、付费墙或访问控制，不保存账号凭据，不调用未授权私有接口。",
        "",
    ]
    if not rows:
        lines += ["本轮没有需要人工核验的问题页面。", ""]
    else:
        lines += [
            "| 优先级 | 问题类型 | 竞品 | URL | 状态 | 需要核验什么 | 入库标注建议 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        row.get("priority"),
                        row.get("problem_type"),
                        row.get("competitor"),
                        row.get("url"),
                        row.get("status"),
                        row.get("what_to_verify"),
                        row.get("data_entry_decision"),
                    )
                )
                + " |"
            )
        lines.append("")
    (out_dir / "problem_pages_review.md").write_text("\n".join(lines), encoding="utf-8")

def field_label_map(plan: Optional[ProductCollectionPlan]) -> Dict[str, str]:
    labels = {
        "pricing": "定价",
        "features": "功能",
        "customers": "客户",
        "positioning": "定位",
        "weight": "重量",
        "size": "尺码",
        "size_fit": "尺码/适配",
        "dimensions": "尺寸",
        "material_construction": "材质与结构",
        "color_variants": "颜色",
        "certification": "认证",
        "api_sdk_webhook": "API/SDK/Webhook",
        "usage_quota_limits": "额度/调用限制",
        "integrations_connectors": "集成/连接器",
        "security_privacy_deployment": "安全、隐私与部署",
    }
    if plan:
        labels.update({field.key: field.label for field in plan.fields})
    return labels


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


PRICE_VALUE_RE = re.compile(
    r"([$€£¥￥]\s?\d+(?:[.,]\d+)?(?:\s*(?:/|per)\s*(?:monthly|month|mo|annually|annual|year|yr))?)",
    re.I,
)
WEIGHT_VALUE_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(g|gram|grams|kg|oz|ounce|ounces|克|千克)\b", re.I)
SIZE_VALUE_RE = re.compile(r"\b(?:sizes?|尺码|头围)\s*[:：]?\s*((?:XXS|XS|S|M|L|XL|XXL|[0-9]{2,3}(?:[-–][0-9]{2,3})?\s?cm)(?:\s*[,/、]\s*(?:XXS|XS|S|M|L|XL|XXL|[0-9]{2,3}(?:[-–][0-9]{2,3})?\s?cm))*)", re.I)
CERT_VALUE_RE = re.compile(r"\b(ASTM\s?F2040|CE\s?EN\s?1077|EN\s?1077|FIS|MIPS|SOC\s?2|GDPR|HIPAA|ISO\s?\d{3,6})\b", re.I)
MATERIAL_VALUE_RE = re.compile(
    r"\b((?:ABS|PC|EPS|EPP|polycarbonate|carbon(?:\s+fiber)?|aluminum|steel|nylon|leather|silicone)\s+(?:hardshell|hard\s+shell|shell|liner|foam|construction|frame|body|material))\b",
    re.I,
)
DIMENSION_VALUE_RE = re.compile(
    r"\b(?:dimensions?|size|head circumference|measurement|尺寸|头围)\s*[:：]?\s*((?:\d{2,4}(?:\.\d+)?\s?[-–x×]\s?\d{2,4}(?:\.\d+)?|\d{2,4}(?:\.\d+)?)\s?(?:cm|mm|in|inch|inches|厘米|毫米))\b",
    re.I,
)
COLOR_VALUE_RE = re.compile(r"\b(?:color options?|colou?r variants?|colou?rs?|配色|颜色)\s*[:：]\s*([^.;。；\n]{2,120})", re.I)
API_VALUE_RE = re.compile(r"\b(REST API|GraphQL API|Public API|API|SDK|Webhook|Webhooks|OAuth|SAML)\b", re.I)
QUOTA_VALUE_RE = re.compile(
    r"\b(?:quota|usage|credits?|tokens?|requests?|calls?|limit|limits?)\s*[:：]?\s*((?:\d[\d,]*(?:\.\d+)?)\s*(?:tokens?|credits?|requests?|calls?|files?|GB|MB)\s*(?:per|/)\s*(?:month|mo|day|minute|min|year|yr))\b",
    re.I,
)
SECURITY_VALUE_RE = re.compile(r"\b(SSO|SAML|SOC\s?2|GDPR|HIPAA|data retention|on-prem|on premise|private deployment)\b", re.I)
RAW_STRUCTURED_FIELD_KEYS = {
    "official_parameters",
    "packaging_limits",
    "material_construction",
    "weight",
    "size_fit",
    "color_variants",
    "quality_reviews",
    "safety_certification",
    "protection_technology",
    "ventilation_comfort",
    "visor_goggle_chinguard",
    "skiing_use_case",
    "api_sdk_webhook",
    "integrations_connectors",
    "models_capabilities",
    "usage_quota_limits",
    "security_privacy_deployment",
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
}


def should_keep_raw_structured_field(field_key: str, raw_value: str) -> bool:
    if field_key not in RAW_STRUCTURED_FIELD_KEYS:
        return False
    value = clean_text(raw_value, limit=500)
    if len(value) > 240:
        return False
    if keyword_hits(value, ["login", "sign in", "start for free", "cookie", "SEO", "导航", "登录", "注册"]):
        return False
    return bool(value)


def evidence_sentence_for_value(text: str, value: str) -> str:
    escaped = re.escape(value).replace("\\ ", r"\s+")
    for fragment in re.split(r"(?<=[。！？.!?])\s+|\n+", textify(text)):
        if re.search(escaped, fragment, re.I):
            return clean_text(fragment, limit=360)
    return clean_text(text, limit=360)


def append_structured_fact(
    facts: List[Dict[str, Any]],
    seen: set[Tuple[str, str, str, str]],
    page: PageExtract,
    field_key: str,
    value: str,
    evidence_text: str,
    plan: Optional[ProductCollectionPlan],
    audit: Mapping[str, Any],
) -> None:
    value = re.sub(r"\s+", " ", textify(value)).strip(" .;；,，")
    if not value:
        return
    key = (page.competitor, field_key, value.lower(), page.url)
    if key in seen:
        return
    seen.add(key)
    labels = field_label_map(plan)
    source_tier = textify(audit.get("source_policy_tier"))
    pending = textify(audit.get("pending_verification"))
    confidence = textify(audit.get("confidence")) or ("高信心" if source_tier.startswith("P0") else "中信心")
    facts.append(
        {
            "competitor": page.competitor,
            "dimension": dimension_for_field(field_key),
            "field_key": field_key,
            "field_label": labels.get(field_key, field_key),
            "value": value,
            "normalized_value": normalize_fact_value(field_key, value),
            "evidence_text": evidence_text or evidence_sentence_for_value(page.markdown or page.text_excerpt, value),
            "source_url": page.url,
            "source_title": page.title,
            "source_policy_tier": source_tier,
            "page_role": textify(audit.get("page_role")),
            "confidence": confidence,
            "confidence_reason": "规则命中页面字段或正文中的可引用事实。",
            "needs_verification": pending or ("yes" if page.error else "no"),
            "extraction_method": "legacy_regex_extractor",
            "evidence_start": "",
            "evidence_end": "",
            "schema_field_description": "",
            "fact_id": "",
        }
    )


def append_schema_structured_fact(
    facts: List[Dict[str, Any]],
    seen: set[Tuple[str, str, str, str]],
    page: PageExtract,
    fact: Mapping[str, Any],
    plan: Optional[ProductCollectionPlan],
    audit: Mapping[str, Any],
) -> None:
    field_key = textify(fact.get("field_key"))
    value = re.sub(r"\s+", " ", textify(fact.get("value"))).strip(" .;；,，")
    if not field_key or not value:
        return
    key = (page.competitor, field_key, value.lower(), page.url)
    if key in seen:
        return
    seen.add(key)
    labels = field_label_map(plan)
    source_tier = textify(audit.get("source_policy_tier"))
    pending = textify(audit.get("pending_verification"))
    confidence = textify(fact.get("confidence")) or textify(audit.get("confidence")) or ("高信心" if source_tier.startswith("P0") else "中信心")
    facts.append(
        {
            "competitor": page.competitor,
            "dimension": textify(fact.get("dimension")) or dimension_for_field(field_key),
            "field_key": field_key,
            "field_label": textify(fact.get("field_label")) or labels.get(field_key, field_key),
            "value": value,
            "normalized_value": textify(fact.get("normalized_value")) or schema_normalize_fact_value(value, field_key),
            "evidence_text": textify(fact.get("evidence_text")) or evidence_sentence_for_value(page.markdown or page.text_excerpt, value),
            "source_url": page.url,
            "source_title": page.title,
            "source_policy_tier": source_tier,
            "page_role": textify(audit.get("page_role")),
            "confidence": confidence,
            "confidence_reason": textify(fact.get("confidence_reason")) or "按品类 schema 命中字段、事实值和来源句。",
            "needs_verification": pending or textify(fact.get("needs_verification")) or ("yes" if page.error else "no"),
            "extraction_method": textify(fact.get("extraction_method")) or "schema_extractor_v1",
            "evidence_start": textify(fact.get("evidence_start")),
            "evidence_end": textify(fact.get("evidence_end")),
            "schema_field_description": textify(fact.get("schema_field_description")),
            "fact_id": "",
        }
    )


def extract_structured_facts(
    pages: Sequence[PageExtract],
    plan: Optional[ProductCollectionPlan] = None,
    evidence_audit_rows: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    audit_by_url = {textify(row.get("url")): row for row in evidence_audit_rows}
    schema = build_structured_extraction_schema(
        plan.category if plan else "general",
        [dataclasses.asdict(field) for field in plan.fields] if plan else [],
    )
    facts: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for page in pages:
        if page.error:
            continue
        audit = audit_by_url.get(page.url, {})
        text = page.markdown or page.text_excerpt or ""
        for schema_fact in schema_extract_structured_facts_from_text(page.competitor, page.url, page.title, text, schema):
            append_schema_structured_fact(facts, seen, page, schema_fact, plan, audit)

        for field_key, raw_value in (page.fields or {}).items():
            if not raw_value:
                continue
            if field_key == "pricing":
                for match in PRICE_VALUE_RE.finditer(raw_value):
                    append_structured_fact(facts, seen, page, "pricing", match.group(0), raw_value, plan, audit)
            elif field_key == "weight":
                for match in WEIGHT_VALUE_RE.finditer(raw_value):
                    append_structured_fact(facts, seen, page, "weight", f"{match.group(1)} {match.group(2)}", raw_value, plan, audit)
            elif should_keep_raw_structured_field(field_key, raw_value):
                append_structured_fact(facts, seen, page, field_key, raw_value[:300], raw_value, plan, audit)

        for match in PRICE_VALUE_RE.finditer(text):
            append_structured_fact(facts, seen, page, "pricing", match.group(0), evidence_sentence_for_value(text, match.group(0)), plan, audit)
        for match in WEIGHT_VALUE_RE.finditer(text):
            append_structured_fact(facts, seen, page, "weight", f"{match.group(1)} {match.group(2)}", evidence_sentence_for_value(text, match.group(0)), plan, audit)
        for match in SIZE_VALUE_RE.finditer(text):
            append_structured_fact(facts, seen, page, "size", match.group(1), evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in CERT_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1).upper()).replace("SOC 2", "SOC2")
            field_key = "certification"
            if value == "MIPS":
                field_key = "protection_technology"
            append_structured_fact(facts, seen, page, field_key, value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in MATERIAL_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            append_structured_fact(facts, seen, page, "material_construction", value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in DIMENSION_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).replace(" – ", "-").strip()
            append_structured_fact(facts, seen, page, "dimensions", value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in COLOR_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).strip(" .;；,，")
            append_structured_fact(facts, seen, page, "color_variants", value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in API_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            append_structured_fact(facts, seen, page, "api_sdk_webhook", value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in QUOTA_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            append_structured_fact(facts, seen, page, "usage_quota_limits", value, evidence_sentence_for_value(text, match.group(1)), plan, audit)
        for match in SECURITY_VALUE_RE.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1).upper()).replace("SOC 2", "SOC2")
            field_key = "certification" if value in {"SOC2", "GDPR", "HIPAA"} else "security_privacy_deployment"
            append_structured_fact(facts, seen, page, field_key, value, evidence_sentence_for_value(text, match.group(1)), plan, audit)

    for idx, fact in enumerate(facts, start=1):
        fact["fact_id"] = f"fact-{idx:04d}"
    return facts


def normalize_fact_value(field_key: str, value: str) -> str:
    raw = textify(value).lower()
    if field_key == "pricing":
        amount = re.search(r"[$€£¥￥]?\s*(\d+(?:[.,]\d+)?)", raw)
        if amount:
            number = amount.group(1).replace(",", "")
            if number.endswith(".0"):
                number = number[:-2]
            period = "monthly" if re.search(r"/\s*mo|per\s*month|monthly|/month", raw) else ""
            yearly = "yearly" if re.search(r"/\s*yr|per\s*year|annual|annually|/year", raw) else ""
            return " ".join(part for part in [number, period or yearly] if part)
    if field_key == "weight":
        match = WEIGHT_VALUE_RE.search(raw)
        if match:
            number = match.group(1).replace(",", ".")
            unit = match.group(2).lower()
            try:
                numeric = float(number)
            except ValueError:
                numeric = None
            if unit in {"gram", "grams", "克"}:
                unit = "g"
            if numeric is not None and unit in {"kg", "千克"}:
                numeric *= 1000
                unit = "g"
            if numeric is not None and unit in {"oz", "ounce", "ounces"}:
                numeric *= 28.3495
                unit = "g"
            if numeric is not None:
                if abs(numeric - round(numeric)) < 0.01:
                    number = str(int(round(numeric)))
                else:
                    number = f"{numeric:.2f}".rstrip("0").rstrip(".")
            return f"{number} {unit}"
    if field_key in {"certification", "protection_technology"}:
        return re.sub(r"\s+", " ", value.upper()).replace("SOC 2", "SOC2")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw).strip()


def cluster_structured_facts(facts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for fact in facts:
        normalized = textify(fact.get("normalized_value")) or normalize_fact_value(textify(fact.get("field_key")), textify(fact.get("value")))
        if not normalized:
            continue
        key = (
            slugify(textify(fact.get("competitor"))),
            textify(fact.get("dimension")),
            textify(fact.get("field_key")),
            normalized,
        )
        grouped.setdefault(key, []).append(fact)

    clusters: List[Dict[str, Any]] = []
    for idx, (key, rows) in enumerate(sorted(grouped.items()), start=1):
        primary = sorted(
            rows,
            key=lambda row: (
                0 if textify(row.get("source_policy_tier")).startswith("P0") else 1,
                0 if textify(row.get("confidence")).startswith("高") else 1,
                textify(row.get("source_url")),
            ),
        )[0]
        urls = unique_strings(row.get("source_url") for row in rows)
        supporting = [url for url in urls if url != primary.get("source_url")]
        clusters.append(
            {
                "cluster_id": f"cluster-{idx:04d}",
                "competitor": primary.get("competitor", ""),
                "dimension": primary.get("dimension", ""),
                "field_key": primary.get("field_key", ""),
                "field_label": primary.get("field_label", ""),
                "normalized_value": key[3],
                "display_value": primary.get("value", ""),
                "source_count": len(urls),
                "primary_source_url": primary.get("source_url", ""),
                "primary_source_title": primary.get("source_title", ""),
                "supporting_source_urls": "; ".join(supporting),
                "confidence": "高信心" if textify(primary.get("source_policy_tier")).startswith("P0") else textify(primary.get("confidence")),
                "needs_verification": "yes" if any(textify(row.get("needs_verification")).lower() == "yes" for row in rows) else "no",
                "evidence_text": primary.get("evidence_text", ""),
            }
        )
    return clusters


def write_fact_clusters_markdown(path: Path, clusters: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# 事实聚类",
        "",
        "同一竞品、同一字段、同一事实值会被合并为一个事实簇。官方来源优先作为主证据，第三方来源作为补充证据。",
        "",
    ]
    if not clusters:
        lines += ["本轮没有抽取到可聚类的结构化事实。", ""]
    else:
        lines += [
            "| 竞品 | 字段 | 事实值 | 来源数 | 主证据 | 补充证据 | 置信度 | 待核实 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cluster in clusters:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        cluster.get("competitor"),
                        cluster.get("field_label") or cluster.get("field_key"),
                        cluster.get("display_value"),
                        cluster.get("source_count"),
                        cluster.get("primary_source_url"),
                        cluster.get("supporting_source_urls"),
                        cluster.get("confidence"),
                        cluster.get("needs_verification"),
                    )
                )
                + " |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_structured_fact_outputs(
    out_dir: Path,
    facts: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(
        out_dir / "structured_facts.csv",
        list(facts),
        [
            "fact_id",
            "competitor",
            "dimension",
            "field_key",
            "field_label",
            "value",
            "normalized_value",
            "evidence_text",
            "source_url",
            "source_title",
            "source_policy_tier",
            "page_role",
            "confidence",
            "confidence_reason",
            "needs_verification",
            "extraction_method",
            "evidence_start",
            "evidence_end",
            "schema_field_description",
        ],
    )
    write_csv(
        out_dir / "fact_clusters.csv",
        list(clusters),
        [
            "cluster_id",
            "competitor",
            "dimension",
            "field_key",
            "field_label",
            "normalized_value",
            "display_value",
            "source_count",
            "primary_source_url",
            "primary_source_title",
            "supporting_source_urls",
            "confidence",
            "needs_verification",
            "evidence_text",
        ],
    )
    (out_dir / "structured_facts.json").write_text(
        json.dumps(json_safe(list(facts)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "fact_clusters.json").write_text(
        json.dumps(json_safe(list(clusters)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_fact_clusters_markdown(out_dir / "fact_clusters.md", clusters)


def rows_from_images(
    image_results: Sequence[SearchResult],
    pages: Sequence[PageExtract],
    downloaded: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in image_results:
        if not item.url or item.url in seen:
            continue
        seen.add(item.url)
        rows.append(
            {
                "competitor": item.competitor,
                "source": "searxng_images",
                "image_url": item.url,
                "page_url": "",
                "title": item.title,
                "query": item.query,
                "local_file": "",
            }
        )
    for page in pages:
        if page.error:
            continue
        for image_url in page.image_urls:
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            rows.append(
                {
                    "competitor": page.competitor,
                    "source": "crawl4ai_page",
                    "image_url": image_url,
                    "page_url": page.url,
                    "title": page.title,
                    "query": "",
                    "local_file": "",
                }
            )
    for item in downloaded:
        rows.append(
            {
                "competitor": item.get("competitor", ""),
                "source": item.get("source", "icrawler_download"),
                "image_url": item.get("image_url", ""),
                "page_url": item.get("page_url", ""),
                "title": item.get("title", ""),
                "query": item.get("query", ""),
                "local_file": item.get("file", ""),
            }
        )
    return rows


def rows_from_all_sources(
    web_results: Sequence[SearchResult],
    image_results: Sequence[SearchResult],
    chosen_urls: Sequence[Tuple[str, str]],
    pages: Sequence[PageExtract],
    downloaded: Sequence[Dict[str, str]],
    evidence_audit_rows: Sequence[Dict[str, Any]],
    collection_plan: Optional[ProductCollectionPlan] = None,
) -> List[Dict[str, Any]]:
    selected = {(competitor, url) for competitor, url in chosen_urls}
    audit_by_url = {
        (row.get("competitor", ""), row.get("url", "")): row
        for row in evidence_audit_rows
    }
    rows: List[Dict[str, Any]] = []

    for item in web_results:
        audit = audit_by_url.get((item.competitor, item.url), {})
        rows.append(
            {
                "competitor": item.competitor,
                "source_stage": "searxng_general",
                "source_type": audit.get("source_kind", "search_result"),
                "source_status": audit.get("decision_status") or audit.get("decision", "candidate"),
                "selected_for_crawl": "yes" if (item.competitor, item.url) in selected else "no",
                "official_or_public": audit.get("source_kind", "public_search_result"),
                "source_policy_tier": audit.get("source_policy_tier", ""),
                "page_role": audit.get("page_role", ""),
                "pending_verification": audit.get("pending_verification", ""),
                "verification_reason": audit.get("verification_reason", ""),
                "fact_type": audit.get("fact_type", ""),
                "increment_type": audit.get("increment_type", ""),
                "fact_group": audit.get("fact_group", ""),
                "primary_evidence_candidate": audit.get("primary_evidence_candidate", ""),
                "primary_evidence_reason": audit.get("primary_evidence_reason", ""),
                "value_signals": audit.get("value_signals", ""),
                "value_missing": audit.get("value_missing", ""),
                "value_verdict": audit.get("value_verdict", ""),
                "gui_review_candidate": audit.get("gui_review_candidate", ""),
                "gui_review_value_reason": audit.get("gui_review_value_reason", ""),
                "ml_label": audit.get("ml_label", ""),
                "ml_include_score": audit.get("ml_include_score", ""),
                "ml_exclude_score": audit.get("ml_exclude_score", ""),
                "ml_verify_later_score": audit.get("ml_verify_later_score", ""),
                "ml_confidence": audit.get("ml_confidence", ""),
                "ml_adjustment": audit.get("ml_adjustment", ""),
                "ml_reason": audit.get("ml_reason", ""),
                "ml_model_version": audit.get("ml_model_version", ""),
                "title": item.title,
                "url_or_path": item.url,
                "domain": domain_of(item.url),
                "query": item.query,
                "engine": item.engine,
                "score": f"{item.score:.2f}",
                "content_preview": truncate_text(item.snippet, 700),
                "reason": "; ".join(
                    unique_strings(
                        [
                            audit.get("hard_gate", ""),
                            audit.get("page_role", ""),
                            audit.get("confidence", ""),
                            audit.get("reason", ""),
                        ]
                    )
                ),
            }
        )

    for item in image_results:
        rows.append(
            {
                "competitor": item.competitor,
                "source_stage": "searxng_images",
                "source_type": "image_result",
                "source_status": "candidate",
                "selected_for_crawl": "no",
                "official_or_public": "public_image_result",
                "source_policy_tier": "P2 公开图片候选",
                "page_role": "visual_evidence_candidate",
                "pending_verification": "yes",
                "verification_reason": "图片来源需确认是否能支撑产品 UI、参数、价格或视觉事实",
                "fact_type": "visual_evidence",
                "increment_type": "新增视觉/UI/产品图片线索",
                "fact_group": fact_group_for(item.competitor, "visual_evidence", [], "visual_evidence_candidate", item.url, item.title),
                "primary_evidence_candidate": "no",
                "primary_evidence_reason": "图片需由 Codex/人工确认是否支撑具体事实",
                "value_signals": "可追溯、可获取",
                "value_missing": "竞品绑定、决策相关、信息增量、来源可信",
                "value_verdict": "valuable_low_confidence_signal",
                "gui_review_candidate": "no",
                "gui_review_value_reason": "",
                "ml_label": "",
                "ml_include_score": "",
                "ml_exclude_score": "",
                "ml_verify_later_score": "",
                "ml_confidence": "",
                "ml_adjustment": "",
                "ml_reason": "图片候选暂不进入文本筛选模型",
                "ml_model_version": "",
                "title": item.title,
                "url_or_path": item.url,
                "domain": domain_of(item.url),
                "query": item.query,
                "engine": item.engine,
                "score": f"{item.score:.2f}",
                "content_preview": truncate_text(item.snippet, 700),
                "reason": "image candidate from SearXNG image category",
            }
        )

    for page in pages:
        review_reason = manual_review_reason(page)
        audit = audit_by_url.get((page.competitor, page.url), {})
        page_result = SearchResult(page.competitor, "crawled_page", "", page.title, page.url, page.text_excerpt)
        page_role = page_role_for_result(page_result, collection_plan)
        field_hits = product_field_hits(f"{page.url} {page.title} {page.text_excerpt}", collection_plan)
        fact_type = fact_type_for(page_role, field_hits, page.text_excerpt)
        increment_type = increment_type_for(page_role, field_hits, page.text_excerpt)
        fact_group = fact_group_for(page.competitor, fact_type, field_hits, page_role, page.url, page.title)
        primary_candidate = "no" if review_reason or page.error else "yes"
        primary_reason = "待核实或抓取失败，不能作为主证据" if review_reason or page.error else "抓取正文可用，可作为候选证据"
        rows.append(
            {
                "competitor": page.competitor,
                "source_stage": "crawl4ai",
                "source_type": "crawled_page",
                "source_status": f"manual_review_required:{review_reason}" if review_reason else (page.error or "accepted_by_rule_filter"),
                "selected_for_crawl": "yes",
                "official_or_public": "public_web_page",
                "source_policy_tier": "manual_review_required" if review_reason else "",
                "page_role": page_role,
                "pending_verification": "yes" if review_reason else "no",
                "verification_reason": manual_review_next_step(page) if review_reason else "",
                "fact_type": fact_type,
                "increment_type": increment_type,
                "fact_group": fact_group,
                "primary_evidence_candidate": primary_candidate,
                "primary_evidence_reason": primary_reason,
                "value_signals": audit.get("value_signals", ""),
                "value_missing": audit.get("value_missing", ""),
                "value_verdict": audit.get("value_verdict", ""),
                "gui_review_candidate": audit.get("gui_review_candidate", ""),
                "gui_review_value_reason": audit.get("gui_review_value_reason", ""),
                "ml_label": audit.get("ml_label", ""),
                "ml_include_score": audit.get("ml_include_score", ""),
                "ml_exclude_score": audit.get("ml_exclude_score", ""),
                "ml_verify_later_score": audit.get("ml_verify_later_score", ""),
                "ml_confidence": audit.get("ml_confidence", ""),
                "ml_adjustment": audit.get("ml_adjustment", ""),
                "ml_reason": audit.get("ml_reason", ""),
                "ml_model_version": audit.get("ml_model_version", ""),
                "title": page.title,
                "url_or_path": page.url,
                "domain": domain_of(page.url),
                "query": "",
                "engine": "crawl4ai",
                "score": "",
                "content_preview": truncate_text(page.text_excerpt, 1000),
                "reason": manual_review_next_step(page) if review_reason else (page.error or "page passed rule filter and is eligible for Codex review"),
            }
        )

    for item in downloaded:
        local_file = item.get("file", "")
        source = item.get("source", "icrawler_download")
        rows.append(
            {
                "competitor": item.get("competitor", ""),
                "source_stage": "searxng_images" if source == "searxng_image_download" else "icrawler",
                "source_type": "downloaded_image",
                "source_status": "downloaded",
                "selected_for_crawl": "no",
                "official_or_public": "public_image_search_download",
                "source_policy_tier": "P2 公开图片候选",
                "page_role": "visual_evidence_candidate",
                "pending_verification": "yes",
                "verification_reason": "本地图片需在报告前确认是否为产品截图、UI、参数图或有效视觉证据",
                "fact_type": "visual_evidence",
                "increment_type": "新增视觉/UI/产品图片线索",
                "fact_group": fact_group_for(item.get("competitor", ""), "visual_evidence", [], "visual_evidence_candidate", local_file, item.get("query", "")),
                "primary_evidence_candidate": "no",
                "primary_evidence_reason": "图片需由 Codex/人工确认是否支撑具体事实",
                "value_signals": "可追溯、可获取",
                "value_missing": "竞品绑定、决策相关、信息增量、来源可信",
                "value_verdict": "valuable_low_confidence_signal",
                "gui_review_candidate": "no",
                "gui_review_value_reason": "",
                "ml_label": "",
                "ml_include_score": "",
                "ml_exclude_score": "",
                "ml_verify_later_score": "",
                "ml_confidence": "",
                "ml_adjustment": "",
                "ml_reason": "图片候选暂不进入文本筛选模型",
                "ml_model_version": "",
                "title": item.get("title", ""),
                "url_or_path": local_file,
                "domain": "",
                "query": item.get("query", ""),
                "engine": item.get("engine", ""),
                "score": "",
                "content_preview": Path(local_file).name if local_file else "",
                "reason": "downloaded from SearXNG image result" if source == "searxng_image_download" else "downloaded by icrawler keyword image search",
            }
        )

    return rows


def write_unfiltered_collection(
    path: Path,
    competitors: Sequence[str],
    web_results: Sequence[SearchResult],
    image_results: Sequence[SearchResult],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    searxng_url: str,
) -> None:
    lines = [
        "# 未经筛选的采集内容",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**SearXNG:** `{searxng_url}`",
        f"**Competitors:** {', '.join(competitors)}",
        "",
        "本文件保留本轮采集拿到的候选内容，不代表已被采纳为有效竞品证据。它用于复查漏筛、误筛和后续规则调优。",
        "",
        "## 1. SearXNG 网页候选",
        "",
    ]
    for idx, item in enumerate(web_results, start=1):
        lines += [
            f"### {idx}. {item.competitor} | {item.title or item.url}",
            f"- **URL:** {item.url}",
            f"- **Domain:** {domain_of(item.url)}",
            f"- **Query:** {item.query}",
            f"- **Engine:** {item.engine or 'unknown'}",
            f"- **Score:** {item.score:.2f}",
            f"- **Snippet:** {truncate_text(item.snippet, 1200) or '无摘要'}",
            "",
        ]

    lines += ["## 2. SearXNG 图片候选", ""]
    for idx, item in enumerate(image_results, start=1):
        lines += [
            f"### {idx}. {item.competitor} | {item.title or item.url}",
            f"- **Image URL:** {item.url}",
            f"- **Query:** {item.query}",
            f"- **Engine:** {item.engine or 'unknown'}",
            f"- **Snippet/source:** {truncate_text(item.snippet, 700) or '无'}",
            "",
        ]

    lines += ["## 3. Crawl4AI 抓取页面", ""]
    if not pages:
        lines += ["本轮未抓取页面，或 Crawl4AI 未返回页面。", ""]
    for idx, page in enumerate(pages, start=1):
        status = page.error or "accepted_by_rule_filter"
        lines += [
            f"### {idx}. {page.competitor} | {page.title or page.url}",
            f"- **URL:** {page.url}",
            f"- **Status:** {status}",
            f"- **Images found:** {len(page.image_urls)}",
            f"- **Links found:** {len(page.links)}",
            f"- **Positioning signal:** {truncate_text(page.fields.get('positioning', ''), 500) or '未抽取'}",
            f"- **Pricing signal:** {truncate_text(page.fields.get('pricing', ''), 500) or '未抽取'}",
            f"- **Feature signal:** {truncate_text(page.fields.get('features', ''), 500) or '未抽取'}",
            f"- **Customer signal:** {truncate_text(page.fields.get('customers', ''), 500) or '未抽取'}",
            "",
            "**Content excerpt:**",
            "",
            truncate_text(page.text_excerpt, 1800) or "无正文摘要",
            "",
        ]

    lines += ["## 4. 图片候选与本地下载", ""]
    if not image_rows:
        lines += ["本轮未得到图片候选或下载图片。", ""]
    for idx, row in enumerate(image_rows, start=1):
        src = row.get("local_file") or row.get("image_url") or ""
        lines += [
            f"### {idx}. {row.get('competitor', '')} | {row.get('source', '')}",
            f"- **Path/URL:** {src}",
            f"- **Page URL:** {row.get('page_url') or '无'}",
            f"- **Title/Query:** {truncate_text(row.get('title') or row.get('query') or '', 500) or '无'}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_filtered_collection(
    path: Path,
    competitors: Sequence[str],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    evidence_audit_rows: Sequence[Dict[str, Any]],
    collection_plan: Optional[ProductCollectionPlan] = None,
) -> None:
    report_dir = path.parent
    selected_rows = [row for row in evidence_audit_rows if row.get("selected") == "yes"]
    rejected_rows = [row for row in evidence_audit_rows if row.get("selected") != "yes"]
    accepted_pages = [page for page in pages if not page.error]
    rejected_pages = [page for page in pages if page.error]

    lines = [
        "# 筛选后的采集内容",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**Competitors:** {', '.join(competitors)}",
        "",
        "本文件是规则筛选后的候选证据池，供 Codex 进一步判断是否进入最终分析报告。",
        "",
        "## 0. 抓取前采集计划",
        "",
    ]
    if collection_plan:
        lines += [
            f"- **识别品类:** {collection_plan.category_label} (`{collection_plan.category}`)",
            f"- **判断原因:** {collection_plan.rationale}",
            "- **额外字段:** " + "、".join(field.label for field in collection_plan.fields),
            "",
        ]
    else:
        lines += ["未生成品类采集计划。", ""]

    lines += [
        "## 1. 已选择进入 Crawl4AI 的来源",
        "",
    ]
    if not selected_rows:
        lines += ["没有 URL 被选入 Crawl4AI。", ""]
    for idx, row in enumerate(selected_rows, start=1):
        lines += [
            f"### {idx}. {row.get('competitor')} | {row.get('title') or row.get('url')}",
            f"- **URL:** {row.get('url')}",
            f"- **Domain:** {row.get('domain')}",
            f"- **来源类型/页面角色:** {row.get('source_kind', 'unknown')} / {row.get('page_role', 'unknown')}",
            f"- **来源策略层级:** {row.get('source_policy_tier') or '未标注'}",
            f"- **事实类型 / 信息增量:** {row.get('fact_type') or '未标注'} / {row.get('increment_type') or '未标注'}",
            f"- **事实组:** `{row.get('fact_group') or '未生成'}`",
            f"- **主证据候选:** {row.get('primary_evidence_candidate') or 'no'}；{row.get('primary_evidence_reason') or '无'}",
            f"- **价值判断:** 命中 {row.get('value_signals') or '无'}；缺失 {row.get('value_missing') or '无'}；结论 {row.get('value_verdict') or '未标注'}",
            f"- **GUI复核候选:** {row.get('gui_review_candidate') or 'no'}；{row.get('gui_review_value_reason') or '无'}",
            f"- **待核实:** {row.get('pending_verification') or 'no'}；{row.get('verification_reason') or '无'}",
            f"- **本地模型判断:** {row.get('ml_label') or '未启用'}；收录 {row.get('ml_include_score') or '-'} / 排除 {row.get('ml_exclude_score') or '-'} / 待核实 {row.get('ml_verify_later_score') or '-'}；{row.get('ml_adjustment') or 'none'}",
            f"- **门禁结果:** {row.get('gate_result', '')} | {row.get('hard_gate', '')}",
            f"- **置信度:** {row.get('confidence', '')}",
            f"- **字段命中:** {row.get('matched_fields') or '无'}",
            f"- **分数:** 相关性 {row.get('relevance_score', '')} / 证据 {row.get('evidence_score', '')} / PM价值 {row.get('pm_value_score', '')} / 可追溯 {row.get('traceability_score', '')} / 品类匹配 {row.get('category_fit_score', '')}",
            f"- **Reason:** {truncate_text(row.get('reason', ''), 1000)}",
            "",
        ]

    signal_rows = [row for row in evidence_audit_rows if row.get("decision_status") in {"accepted", "signal"} and row.get("selected") != "yes"]
    lines += ["## 1.1 未抓取但保留的证据信号", ""]
    if not signal_rows:
        lines += ["没有额外保留的 accepted/signal 来源。", ""]
    for idx, row in enumerate(signal_rows[:80], start=1):
        lines += [
            f"### {idx}. {row.get('competitor')} | {row.get('title') or row.get('url')}",
            f"- **URL:** {row.get('url')}",
            f"- **状态:** {row.get('decision_status')} / {row.get('confidence')}",
            f"- **来源类型/页面角色:** {row.get('source_kind', 'unknown')} / {row.get('page_role', 'unknown')}",
            f"- **事实类型 / 信息增量:** {row.get('fact_type') or '未标注'} / {row.get('increment_type') or '未标注'}",
            f"- **事实组:** `{row.get('fact_group') or '未生成'}`",
            f"- **价值判断:** 命中 {row.get('value_signals') or '无'}；缺失 {row.get('value_missing') or '无'}；结论 {row.get('value_verdict') or '未标注'}",
            f"- **GUI复核候选:** {row.get('gui_review_candidate') or 'no'}；{row.get('gui_review_value_reason') or '无'}",
            f"- **待核实:** {row.get('pending_verification') or 'no'}；{row.get('verification_reason') or '无'}",
            f"- **本地模型判断:** {row.get('ml_label') or '未启用'}；收录 {row.get('ml_include_score') or '-'} / 排除 {row.get('ml_exclude_score') or '-'} / 待核实 {row.get('ml_verify_later_score') or '-'}；{row.get('ml_adjustment') or 'none'}",
            f"- **字段命中:** {row.get('matched_fields') or '无'}",
            f"- **保留原因:** {truncate_text(row.get('reason', ''), 700)}",
            "",
        ]

    lines += ["## 2. 通过规则筛选的页面内容", ""]
    if not accepted_pages:
        lines += ["没有页面通过规则筛选。", ""]
    for idx, page in enumerate(accepted_pages, start=1):
        lines += [
            f"### {idx}. {page.competitor} | {page.title or page.url}",
            f"- **URL:** {page.url}",
            f"- **Positioning:** {truncate_text(page.fields.get('positioning', ''), 500) or '未抽取'}",
            f"- **Pricing:** {truncate_text(page.fields.get('pricing', ''), 500) or '未抽取'}",
            f"- **Features:** {truncate_text(page.fields.get('features', ''), 500) or '未抽取'}",
            f"- **Customers:** {truncate_text(page.fields.get('customers', ''), 500) or '未抽取'}",
        ]
        if collection_plan:
            planned_lines = []
            for field in collection_plan.fields:
                value = truncate_text(page.fields.get(field.key, ""), 500)
                if value:
                    planned_lines.append(f"  - {field.label}: {value}")
            if planned_lines:
                lines += ["- **品类字段:**", *planned_lines]
            else:
                lines.append("- **品类字段:** 未抽取到稳定信号")
        lines += [
            "",
            truncate_text(page.text_excerpt, 1600) or "无正文摘要",
            "",
        ]

    lines += ["## 3. 候选图片证据", ""]
    total_selected_images = 0
    for competitor in competitors:
        selected_images = select_images_for_competitor(competitor, image_rows, report_dir)
        if not selected_images:
            continue
        total_selected_images += len(selected_images)
        lines += [f"### {competitor}", ""]
        for idx, image in enumerate(selected_images, start=1):
            alt = md_cell(f"{competitor} candidate visual {idx}")
            caption = md_cell(image.get("title") or image.get("source") or "image evidence")
            lines += [
                markdown_image_tag(alt, image["src"]),
                f"_Image {idx}: {caption}; source: {image.get('source') or 'unknown'}_",
                "",
            ]
    if total_selected_images == 0:
        lines += ["没有可直接嵌入的候选图片。", ""]

    lines += ["## 4. 已排除内容摘要", ""]
    lines += [
        f"- 规则排除/未选入 URL：{len(rejected_rows)} 条",
        f"- Crawl4AI 抓取后排除页面：{len(rejected_pages)} 页",
        "",
    ]
    for idx, page in enumerate(rejected_pages[:80], start=1):
        lines += [
            f"### {idx}. {page.competitor} | {page.title or page.url}",
            f"- **URL:** {page.url}",
            f"- **Reason:** {page.error}",
            f"- **Cleaned excerpt sample:** {truncate_text(page.text_excerpt, 700) or '无'}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_report_has_images(path: Path, competitors: Sequence[str], image_rows: Sequence[Dict[str, Any]]) -> None:
    if not path.exists() or not image_rows:
        return
    text = path.read_text(encoding="utf-8")
    if "![" in text:
        return
    report_dir = path.parent
    lines = [
        "",
        "## 图片证据",
        "",
        "Codex 最终报告正文未嵌入图片，以下补充本轮筛选后的候选视觉证据，供人工复核使用。",
        "",
    ]
    added = 0
    for competitor in competitors:
        selected_images = select_images_for_competitor(competitor, image_rows, report_dir)
        if not selected_images:
            continue
        lines += [f"### {competitor}", ""]
        for idx, image in enumerate(selected_images, start=1):
            alt = md_cell(f"{competitor} visual evidence {idx}")
            caption = md_cell(image.get("title") or image.get("source") or "image evidence")
            lines += [
                markdown_image_tag(alt, image["src"]),
                f"_Image {idx}: {caption}; source: {image.get('source') or 'unknown'}_",
                "",
            ]
            added += 1
    if added:
        path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def best_page_field(
    pages: Sequence[PageExtract],
    field: str,
    preferred_path_tokens: Sequence[str],
) -> str:
    candidates = [page for page in pages if not page.error and page.fields.get(field)]
    if not candidates:
        return ""

    def rank(page: PageExtract) -> Tuple[int, int, int]:
        path = urlparse(page.url).path.lower()
        token_rank = next((idx for idx, token in enumerate(preferred_path_tokens) if token in path), len(preferred_path_tokens))
        return (token_rank, official_path_priority(page.url), len(page.fields.get(field, "")))

    return sorted(candidates, key=rank)[0].fields.get(field, "")


def best_source_url(pages: Sequence[PageExtract], fallback_results: Sequence[SearchResult]) -> str:
    ok_pages = [page for page in pages if not page.error]
    if ok_pages:
        return sorted(ok_pages, key=lambda page: official_path_priority(page.url))[0].url
    if pages:
        return pages[0].url
    return fallback_results[0].url if fallback_results else ""


def summarize_competitors(
    competitors: Sequence[str],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    web_results: Sequence[SearchResult] = (),
) -> List[Dict[str, Any]]:
    summary = []
    for competitor in competitors:
        comp_pages = [p for p in pages if p.competitor == competitor]
        comp_images = [r for r in image_rows if r.get("competitor") == competitor]
        comp_search_results = [r for r in web_results if r.competitor == competitor]
        summary.append(
            {
                "competitor": competitor,
                "pages_crawled": len(comp_pages),
                "pages_ok": len([p for p in comp_pages if not p.error]),
                "images_found": len(comp_images),
                "positioning": best_page_field(comp_pages, "positioning", ("product", "features", "solutions", "about")),
                "pricing_signal": best_page_field(comp_pages, "pricing", ("pricing", "plans", "price")),
                "feature_signal": best_page_field(comp_pages, "features", ("features", "product", "products", "solutions")),
                "top_url": best_source_url(comp_pages, comp_search_results),
            }
        )
    return summary


def write_report(
    path: Path,
    competitors: Sequence[str],
    competitor_rows: Sequence[Dict[str, Any]],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    searxng_url: str,
) -> None:
    lines = [
        "# Competitor Intel Harvest Report",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**SearXNG:** `{searxng_url}`",
        f"**Competitors:** {', '.join(competitors)}",
        "",
        "## Summary",
        "",
        "| Competitor | Pages OK | Images | Top URL |",
        "|---|---:|---:|---|",
    ]
    for row in competitor_rows:
        lines.append(
            f"| {row['competitor']} | {row['pages_ok']}/{row['pages_crawled']} | {row['images_found']} | {row['top_url']} |"
        )

    lines += ["", "## PM Signals", ""]
    for row in competitor_rows:
        lines += [
            f"### {row['competitor']}",
            f"- **Positioning:** {row.get('positioning') or 'Not detected'}",
            f"- **Pricing signal:** {row.get('pricing_signal') or 'Not detected'}",
            f"- **Feature signal:** {row.get('feature_signal') or 'Not detected'}",
            "",
        ]

    lines += ["## Crawled Pages", ""]
    for page in pages:
        status = "error" if page.error else "ok"
        lines += [
            f"### {page.competitor} — {page.title or page.url}",
            f"- **Status:** {status}",
            f"- **URL:** {page.url}",
            f"- **Images found:** {len(page.image_urls)}",
        ]
        if page.error:
            lines.append(f"- **Error:** {page.error}")
        if page.text_excerpt:
            lines.append(f"- **Excerpt:** {page.text_excerpt[:500]}")
        lines.append("")

    lines += [
        "## Files",
        "",
        "- `all_sources.csv`: all collected sources",
        "- `pre_crawl_plan.md/json`: product-specific collection plan generated before search and crawl",
        "- `unfiltered_collection.md`: raw/unfiltered collected candidates",
        "- `filtered_collection.md`: rule-filtered candidate evidence pool",
        "- `final_analysis.md`: internal Codex analysis source used to build the embedded final report",
        "- `final_analysis_embedded.md`: formal final report with images embedded as base64",
        "- `collection_principles.md`: collection and filtering principles",
        "- Internal compatibility/debug files: `analysis.md`, `methodology.md`, `competitors.csv`, `pages.csv`, `images.csv`, `evidence_audit.csv`, `raw.json`, `downloaded_images/`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def md_cell(value: Any) -> str:
    text = textify(value).replace("\n", " ").replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def markdown_destination(src: str) -> str:
    src = textify(src).strip()
    if not src:
        return ""
    if src.startswith("<") and src.endswith(">"):
        return src
    if re.search(r"[\s()<>]", src):
        return f"<{src.replace('>', '%3E')}>"
    return src


def markdown_image_src(row: Dict[str, Any], report_dir: Path) -> str:
    local_file = row.get("local_file") or ""
    if local_file:
        local_path = Path(local_file).resolve()
        try:
            return local_path.relative_to(report_dir.resolve()).as_posix()
        except ValueError:
            return local_path.as_posix()
    return row.get("image_url") or ""


def markdown_image_tag(alt: str, src: str) -> str:
    return f"![{md_cell(alt)}]({markdown_destination(src)})"


def write_pre_crawl_plan(
    out_dir: Path,
    competitors: Sequence[str],
    plan: ProductCollectionPlan,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
    manual_search_terms: Iterable[str] = (),
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
) -> None:
    manual_search_terms = normalize_keyword_inputs(manual_search_terms)
    manual_include_keywords = normalize_keyword_inputs(manual_include_keywords)
    manual_exclude_keywords = normalize_keyword_inputs(manual_exclude_keywords)
    payload = {
        "generated_at": utc_stamp(),
        "own_product": {
            "name": own_product_name,
            "positioning": own_product_positioning,
            "context": own_product_context,
        },
        "competitors": list(competitors),
        "plan": dataclasses.asdict(plan),
        "manual_screening_overrides": {
            "manual_search_terms": manual_search_terms,
            "manual_include_keywords": manual_include_keywords,
            "manual_exclude_keywords": manual_exclude_keywords,
        },
    }
    (out_dir / "pre_crawl_plan.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 抓取前采集计划",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**我方产品:** {own_product_name or '未填写'}",
        f"**我方定位:** {own_product_positioning or '未填写'}",
        f"**补充背景:** {own_product_context or '未填写'}",
        f"**识别品类:** {plan.category_label} (`{plan.category}`)",
        "",
        "## 为什么这样抓",
        "",
        plan.rationale,
        "",
        "## 默认检索路径",
        "",
        "1. 由抓取前分析生成定向搜索词。",
        "2. 通过 SearXNG 定向检索官网、垂直平台、论坛、社媒、视频和图片入口。",
        "3. 对可读网页走自动抓取。",
        "4. 对疑似有价值但机器抓不到的公开页面，进入 GUI 复核。",
        "5. 最后才扩大到海量搜索兜底，用来补漏、发现别名和发现潜在竞品。",
        "",
        "## 本轮竞品",
        "",
    ]
    for competitor in competitors:
        lines.append(f"- {competitor}")

    if plan.analysis_template_key:
        lines += [
            "",
            "## 已加载分析模板",
            "",
            f"- **模板:** {plan.analysis_template_label or plan.analysis_template_key}",
            f"- **匹配分:** {plan.analysis_template_match_score}",
            f"- **路径:** `{plan.analysis_template_path}`",
        ]
        if plan.analysis_template_summary:
            lines.append(f"- **说明:** {plan.analysis_template_summary}")
        if plan.analysis_report_outline:
            lines += [
                "",
                "**报告参考结构:** " + "、".join(plan.analysis_report_outline),
            ]
        if plan.analysis_dimensions:
            lines += [
                "",
                "| 维度 | 必须寻找的证据 |",
                "|---|---|",
            ]
            for item in plan.analysis_dimensions:
                lines.append(
                    f"| {md_cell(item.get('label') or item.get('id'))} | {md_cell('、'.join(item.get('required_evidence') or []))} |"
                )

    if plan.search_cards_applied:
        lines += [
            "",
            "## 已加载搜索卡片",
            "",
            "| 产品类型 | 置信度 | 训练样本 | 来源 |",
            "|---|---|---|---|",
        ]
        for card in plan.search_cards_applied:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        card.get("product_type_label") or card.get("product_type_key"),
                        card.get("confidence"),
                        card.get("training_rows"),
                        card.get("source_path"),
                    )
                )
                + " |"
            )

    lines += [
        "",
        "## 竞品发现策略",
        "",
        "| 阶段 | 策略 | 触发条件 | 发现方式 | 入池规则 | 追溯规则 | 排除规则 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in plan.competitor_discovery_strategies:
        lines.append(
            "| "
            + " | ".join(
                md_cell(value)
                for value in (
                    item.stage,
                    item.name,
                    item.trigger,
                    item.discovery_method,
                    item.acceptance_rule,
                    item.traceability_rule,
                    item.rejection_rule,
                )
            )
            + " |"
        )

    lines += [
        "",
        "## 数据源策略",
        "",
        "| 优先级 | 数据源类别 | 典型来源 | 选择规则 | 获取方式 | 追溯规则 | 证据用途 | 升级/复核 | 合规边界 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in plan.source_strategies:
        lines.append(
            "| "
            + " | ".join(
                md_cell(value)
                for value in (
                    item.priority,
                    item.name,
                    "、".join(item.source_examples),
                    item.selection_rule,
                    item.retrieval_rule,
                    item.traceability_rule,
                    item.evidence_role,
                    item.escalation_rule,
                    item.legal_boundary,
                )
            )
            + " |"
        )

    lines += [
        "",
        "## 价值判断规则",
        "",
        "| 判断项 | 算有价值的情况 | 不算有价值的情况 |",
        "|---|---|---|",
    ]
    for rule in plan.value_judgment_rules:
        lines.append(f"| {md_cell(rule.label)} | {md_cell(rule.positive_rule)} | {md_cell(rule.negative_rule)} |")

    lines += [
        "",
        "## 需要额外抓取的字段",
        "",
        "| 字段 | 为什么重要 | 搜索/抽取线索 |",
        "|---|---|---|",
    ]
    for field in plan.fields:
        lines.append(
            f"| {md_cell(field.label)} (`{field.key}`) | {md_cell(field.description)} | {md_cell(', '.join(field.search_terms[:8]))} |"
        )

    lines += [
        "",
        "## 追加搜索模板",
        "",
    ]
    for template in plan.search_templates:
        lines.append(f"- `{template}`")
    if plan.directed_source_search_templates:
        lines += ["", "## 定向数据源搜索模板", ""]
        for template in plan.directed_source_search_templates:
            lines.append(f"- `{template}`")
    if plan.cn_search_templates:
        lines += ["", "## 追加中文搜索模板", ""]
        for template in plan.cn_search_templates:
            lines.append(f"- `{template}`")

    lines += [
        "",
        "## 证据关键词预设",
        "",
        ", ".join(plan.evidence_keywords[:120]) or "无",
        "",
        "## 预抓取分析自动生成的搜索补充词",
        "",
        ", ".join(plan.generated_search_terms[:140]) or "无",
        "",
        "## 自动搜索词生成理由",
        "",
        "| 搜索词 | 来自字段 | 预期来源 | 预期证据 | 噪声风险 | 优先级 | 生成方 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in plan.search_term_reasons[:160]:
        lines.append(
            "| "
            + " | ".join(
                md_cell(row.get(key, ""))
                for key in (
                    "term",
                    "source_field",
                    "expected_source",
                    "expected_evidence",
                    "noise_risk",
                    "priority",
                    "generated_by",
                )
            )
            + " |"
        )

    lines += [
        "",
        "## 固定排除/降权关键词",
        "",
        ", ".join(plan.fixed_exclude_keywords[:120]) or "无",
        "",
        "## 动态排除/降权关键词",
        "",
        ", ".join(plan.dynamic_exclude_keywords[:120]) or "无",
        "",
        "## 来源策略预设",
        "",
    ]
    for note in plan.source_policy_notes:
        lines.append(f"- {note}")

    if manual_search_terms or manual_include_keywords or manual_exclude_keywords:
        lines += [
            "",
            "## 本轮高级人工覆盖",
            "",
            "- **手动搜索补充词:** " + (", ".join(manual_search_terms) or "无。默认由预抓取分析自动生成，UI 不再要求填写。"),
            "- **人工重点关注词:** " + (", ".join(manual_include_keywords) or "无"),
            "- **手动排除词:** " + (", ".join(manual_exclude_keywords) or "无。默认由固定排除词和动态排除词共同决定。"),
        ]

    lines += [
        "",
        "## 追加图片关键词",
        "",
        ", ".join(plan.image_terms) or "无",
        "",
        "## 报告关注点",
        "",
    ]
    for focus in plan.report_focus:
        lines.append(f"- {focus}")
    lines.append("")
    (out_dir / "pre_crawl_plan.md").write_text(prepare_analysis_markdown("\n".join(lines), out_dir), encoding="utf-8")


def normalize_markdown_image_paths(text: str, report_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        raw_src = match.group(2).strip()
        src = raw_src[1:-1].strip() if raw_src.startswith("<") and raw_src.endswith(">") else raw_src
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https", "data"):
            return match.group(0)
        if parsed.scheme == "file":
            candidate = Path(parsed.path).resolve()
        elif src.startswith("/"):
            candidate = Path(src).resolve()
        else:
            relative_src = src[2:] if src.startswith("./") else src
            candidate = (report_dir / relative_src).resolve()
        if not candidate.exists():
            return match.group(0)
        try:
            normalized_src = candidate.relative_to(report_dir.resolve()).as_posix()
        except ValueError:
            normalized_src = candidate.as_posix()
        return markdown_image_tag(alt, normalized_src)

    return re.sub(r"!\[([^\]]*)\]\(([^)\n]+)\)", replace, text)


def embedded_markdown_image_paths(text: str, report_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        raw_src = match.group(2).strip()
        src = raw_src[1:-1].strip() if raw_src.startswith("<") and raw_src.endswith(">") else raw_src
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https", "data"):
            return match.group(0)
        if parsed.scheme == "file":
            candidate = Path(parsed.path).resolve()
        elif src.startswith("/"):
            candidate = Path(src).resolve()
        else:
            relative_src = src[2:] if src.startswith("./") else src
            candidate = (report_dir / relative_src).resolve()
        if not candidate.exists() or not candidate.is_file():
            return match.group(0)
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            return match.group(0)
        payload = base64.b64encode(candidate.read_bytes()).decode("ascii")
        return markdown_image_tag(alt, f"data:{mime};base64,{payload}")

    return re.sub(r"!\[([^\]]*)\]\(([^)\n]+)\)", replace, text)


def write_embedded_markdown_copy(src_path: Path, dst_path: Path) -> None:
    if not src_path.exists():
        return
    text = src_path.read_text(encoding="utf-8")
    text = normalize_markdown_image_paths(text, src_path.parent)
    text = embedded_markdown_image_paths(text, src_path.parent)
    dst_path.write_text(text, encoding="utf-8")


def write_chinese_export_aliases(out_dir: Path) -> None:
    for source_name, alias_name in CHINESE_EXPORT_ALIASES.items():
        if alias_name not in ROOT_OUTPUT_FILES:
            continue
        source = out_dir / source_name
        alias = out_dir / alias_name
        if source.exists():
            shutil.copyfile(source, alias)


def unique_archive_path(internal_dir: Path, name: str) -> Path:
    target = internal_dir / name
    if not target.exists():
        return target
    candidate = Path(name)
    stem = candidate.stem or candidate.name
    suffix = candidate.suffix
    counter = 2
    while True:
        next_target = internal_dir / f"{stem}_{counter}{suffix}"
        if not next_target.exists():
            return next_target
        counter += 1


def slim_output_directory(out_dir: Path, keep_run_log: bool = True) -> None:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    keep_names = set(ROOT_OUTPUT_FILES)
    keep_names.add(INTERNAL_OUTPUT_DIR_NAME)
    if keep_run_log:
        keep_names.add("run.log")
    internal_dir = out_dir / INTERNAL_OUTPUT_DIR_NAME
    moved_any = False
    for path in list(out_dir.iterdir()):
        if path.name in keep_names or path.name.startswith("."):
            continue
        if not path.is_file() and not path.is_dir():
            continue
        internal_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(unique_archive_path(internal_dir, path.name)))
        moved_any = True
    if moved_any:
        manifest = internal_dir / "README.md"
        if not manifest.exists():
            manifest.write_text(
                "# 内部文件\n\n"
                "这里保存中间结果、兼容旧版本的英文文件、调试日志和原始 JSON。"
                "根目录只保留单一格式的中文交付物：图片内嵌分析报告、采集来源、筛选前后内容、问题核验 CSV、抽样标注 CSV、结构化事实、事实聚类和必要原则文档。\n",
                encoding="utf-8",
            )


def write_screening_strategy_doc(out_dir: Path) -> None:
    source = Path(__file__).resolve().with_name("竞品信息收录过滤策略设计.md")
    if source.exists():
        (out_dir / "screening_strategy.md").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def normalize_confidence_lines(text: str) -> str:
    pattern = re.compile(
        r"^(\s*)-\s+(Fact|Inference|Assumption)\s*(?:[｜|]\s*)?"
        r"(高|中高|中|中低|低|待验证)?\s*(?:置信|信心)?\s*[:：]\s*(.+)$"
    )
    normalized: List[str] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            normalized.append(line)
            continue
        indent, evidence_type, confidence, content = match.groups()
        normalized.extend(
            [
                f"{indent}- 结论：{content.strip()}",
                f"{indent}  - 证据属性：{evidence_type}",
                f"{indent}  - 信心等级：{confidence or '未标注'}",
            ]
        )
    return "\n".join(normalized)


STOP_SLOP_ZH_PROMPT = """
Before finalizing analysis_markdown, run a stop-slop-zh pass:
- Start with conclusions. Do not use generic openings like "随着...发展", "众所周知", "值得注意的是", "综上所述".
- Keep evidence, numbers, URLs, tables, image syntax, and Fact/Inference/Assumption labels intact.
- Remove Chinese AI/report filler: 赋能、抓手、闭环、底层逻辑、组合拳、全方位、多维度、显著提升 unless a concrete fact follows.
- Avoid formulaic structures such as "不是 X 而是 Y", "不仅...而且", three-part parallel slogans, and elevated endings.
- Use specific subjects and actions. Write like a PM brief, not a generic marketing article.
- Do not reduce information density. Only improve wording and rhythm.
""".strip()

STOP_SLOP_ZH_PHRASE_REPLACEMENTS = [
    ("值得注意的是，", ""),
    ("值得注意的是,", ""),
    ("值得一提的是，", ""),
    ("值得一提的是,", ""),
    ("不得不说，", ""),
    ("不得不承认，", ""),
    ("换句话说，", ""),
    ("换言之，", ""),
    ("也就是说，", ""),
    ("简单来说，", ""),
    ("总的来说，", ""),
    ("客观来讲，", ""),
    ("坦白讲，", ""),
    ("与此同时，", ""),
    ("进一步来说，", ""),
    ("进一步而言，", ""),
    ("更重要的是，", "更关键的是，"),
    ("正因如此，", ""),
    ("正是因为如此，", ""),
    ("综上所述，", ""),
    ("总而言之，", ""),
    ("总之，", ""),
    ("由此可见，", ""),
    ("归根结底，", ""),
    ("进行优化", "优化"),
    ("进行了优化", "优化了"),
    ("进行分析", "分析"),
    ("进行了分析", "分析了"),
    ("进行判断", "判断"),
    ("进行了判断", "判断了"),
    ("进行验证", "验证"),
    ("进行了验证", "验证了"),
    ("加以改进", "改进"),
    ("予以解决", "解决"),
    ("起到了", "是"),
    ("赋能", "帮助"),
    ("抓手", "办法"),
    ("闭环", "走通"),
    ("对齐", "统一"),
    ("沉淀", "存下来"),
    ("颗粒度", "细度"),
    ("底层逻辑", "原理"),
    ("心智", "印象"),
    ("打法", "做法"),
    ("势能", "优势"),
    ("组合拳", "一套办法"),
    ("顶层设计", "整体规划"),
    ("保驾护航", "保障"),
    ("全方位", "多方面"),
    ("多维度", "多个角度"),
    ("整体来看，", ""),
    ("总体而言，", ""),
    ("综合判断，", ""),
]

STOP_SLOP_ZH_DROP_LINE_PATTERNS = [
    re.compile(r"^\s*(综上所述|综上|总而言之|总之|由此可见|可见|归根结底)[。！!，,：:；;]*\s*$"),
    re.compile(r"^\s*(展望未来|未来可期|充满无限可能|携手共进|共创未来).*$"),
]


def apply_stop_slop_zh(text: str) -> str:
    """Lightweight stop-slop-zh cleanup for Chinese report prose.

    The pass deliberately avoids tables, code fences, images, and raw URLs so the
    evidence layer stays unchanged.
    """
    cleaned: List[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            cleaned.append(line)
            continue
        if in_fence or stripped.startswith("|") or stripped.startswith("![") or re.match(r"^\s{0,3}[-*]\s+!\[", line):
            cleaned.append(line)
            continue
        if "http://" in line or "https://" in line or "data:image/" in line:
            cleaned.append(line)
            continue
        if any(pattern.match(line) for pattern in STOP_SLOP_ZH_DROP_LINE_PATTERNS):
            continue
        revised = line
        for source, target in STOP_SLOP_ZH_PHRASE_REPLACEMENTS:
            revised = revised.replace(source, target)
        revised = re.sub(r"^\s*-\s+(这)?不是([^，,。；;]+)[，,]\s*而是\s*", "- ", revised)
        revised = re.sub(r"^\s*(这)?不是([^，,。；;]+)[，,]\s*而是\s*", "", revised)
        revised = re.sub(r"(这)?不是([^，,。；;]+)[，,]\s*而是\s*", "", revised)
        revised = re.sub(r"不仅([^，,。；;]+)[，,]\s*而且", "", revised)
        revised = re.sub(r"既([^，,。；;]+)[，,]\s*又", "", revised)
        revised = re.sub(r"在当今[^。！？!?]{0,40}(时代背景下|背景下)[，,]?", "", revised)
        revised = re.sub(r"随着[^。！？!?]{1,40}(不断发展|飞速发展|日益普及)[，,]?", "", revised)
        revised = re.sub(r"进行了(全面的|深入的|系统的)?(优化|分析|判断|验证)", r"\2了", revised)
        revised = re.sub(r"进行(全面的|深入的|系统的)?(优化|分析|判断|验证)", r"\2", revised)
        revised = re.sub(r"(极其|非常|十分|充分地|深刻地|极大地)", "", revised)
        revised = re.sub(r"([。！？!?])\s*(综上所述|总而言之|总之|由此可见)[，,]?", r"\1", revised)
        cleaned.append(revised)
    return "\n".join(cleaned)


def prepare_analysis_markdown(text: str, report_dir: Optional[Path] = None) -> str:
    if report_dir is not None:
        text = normalize_markdown_image_paths(text, report_dir)
    text = normalize_confidence_lines(text)
    text = apply_stop_slop_zh(text)
    return text


def is_probably_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def image_priority(row: Dict[str, Any]) -> int:
    source = row.get("source", "")
    if row.get("local_file"):
        return 0
    if source == "searxng_images":
        return 1
    if source == "crawl4ai_page":
        return 2
    return 3


def select_images_for_competitor(
    competitor: str,
    image_rows: Sequence[Dict[str, Any]],
    report_dir: Path,
    limit: int = 3,
) -> List[Dict[str, str]]:
    selected = []
    seen = set()
    rows = [row for row in image_rows if row.get("competitor") == competitor]
    for row in sorted(rows, key=image_priority):
        src = markdown_image_src(row, report_dir)
        if not src or src in seen:
            continue
        if not row.get("local_file") and not is_probably_image_url(src):
            continue
        lower = src.lower()
        if lower.endswith(".svg") or lower.endswith(".gif"):
            continue
        seen.add(src)
        selected.append(
            {
                "src": src,
                "title": row.get("title") or row.get("query") or competitor,
                "source": row.get("source", ""),
                "page_url": row.get("page_url", ""),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def evidence_quality(row: Dict[str, Any]) -> str:
    pages_ok = int(row.get("pages_ok") or 0)
    images_found = int(row.get("images_found") or 0)
    detected = sum(1 for key in ("positioning", "pricing_signal", "feature_signal") if row.get(key))
    if pages_ok >= 3 and detected >= 2:
        return "High"
    if pages_ok >= 1 or images_found >= 3:
        return "Medium"
    return "Low"


def write_analysis_report(
    path: Path,
    competitors: Sequence[str],
    competitor_rows: Sequence[Dict[str, Any]],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    searxng_url: str,
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
    collection_plan: Optional[ProductCollectionPlan] = None,
    structured_facts: Sequence[Mapping[str, Any]] = (),
    fact_clusters: Sequence[Mapping[str, Any]] = (),
    gui_review_rows: Sequence[Mapping[str, Any]] = (),
    competitor_discovery: Optional[Mapping[str, Any]] = None,
    manual_review_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    report_dir = path.parent
    row_by_competitor = {row.get("competitor", ""): row for row in competitor_rows}
    lines = [
        "# 竞品分析报告",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**Competitors:** {', '.join(competitors)}",
        f"**我方产品:** {own_product_name or '未填写'}",
        f"**SearXNG:** `{searxng_url}`",
        "",
        "## 0. 核心结论与决策建议",
        "",
        "- Fact: 本报告基于本轮公开网页、搜索摘要、Crawl4AI 抽取内容、icrawler 图片和本地 Codex/规则分析生成。",
        "- Inference: 竞品定位、能力强弱和机会判断只代表当前证据池的推断，不等同于完整市场结论。",
        "- Assumption: 未抓到、被反爬或需要 GUI 复核的内容不代表竞品没有相关能力，需要进入补证流程。",
        "- Decision: 先用官方核心页确认定位、能力和定价，再用第三方证据验证牵引、口碑和战略方向。",
        "",
        "## 1. 采集范围、来源与证据等级",
        "",
        "| 竞品 | 抓取页 | 可用页 | 图片线索 | 证据等级 | Top URL |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in competitor_rows:
        lines.append(
            f"| {md_cell(row.get('competitor'))} | {row.get('pages_crawled', 0)} | {row.get('pages_ok', 0)} | {row.get('images_found', 0)} | {evidence_quality(row)} | {md_cell(row.get('top_url') or '')} |"
        )
    lines += [
        "",
        "**证据边界**",
        "",
        "- Fact: URL、抓取文本、搜索摘要、图片路径、公开页面中明确写出的价格/功能/客户/更新。",
        "- Inference: 基于多条证据组合得出的定位、战略方向、优劣势和风险机会。",
        "- Assumption: 证据缺失、反爬、低文本或只来自二手摘要时的待验证判断。",
        "",
        "### 抓取前产品分析与额外字段",
        "",
    ]
    if collection_plan:
        lines += [
            f"- **识别品类:** {collection_plan.category_label} (`{collection_plan.category}`)",
            f"- **判断原因:** {collection_plan.rationale}",
            "- **本品类应额外抓取:** " + "、".join(field.label for field in collection_plan.fields),
            "",
            "| 字段 | 说明 |",
            "|---|---|",
        ]
        for field in collection_plan.fields:
            lines.append(f"| {md_cell(field.label)} | {md_cell(field.description)} |")
        lines += ["", "**报告关注点**", ""]
        for focus in collection_plan.report_focus:
            lines.append(f"- {focus}")
        lines.append("")
    else:
        lines += ["未生成抓取前产品分析。", ""]

    if competitor_discovery and competitor_discovery.get("candidates"):
        lines += [
            "### 自动竞品发现记录",
            "",
            "| 候选竞品 | 置信度 | 状态 | 官方入口 | 发现来源 |",
            "|---|---|---|---|---|",
        ]
        for candidate in (competitor_discovery.get("candidates") or [])[:20]:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        candidate.get("name"),
                        candidate.get("confidence"),
                        candidate.get("status"),
                        candidate.get("official_url"),
                        candidate.get("discovered_from_url"),
                    )
                )
                + " |"
            )
        lines.append("")

    lines += [
        "## 2. 竞品概览与定位",
        "",
        "| 竞品 | 定位 | 目标用户/市场 | 关键来源 |",
        "|---|---|---|---|",
    ]
    for competitor in competitors:
        row = row_by_competitor.get(competitor, {})
        comp_pages = [page for page in pages if page.competitor == competitor]
        ok_pages = [page for page in comp_pages if not page.error]
        customer_signal = next((page.fields.get("customers") for page in ok_pages if page.fields.get("customers")), "")
        top_url = row.get("top_url") or (ok_pages[0].url if ok_pages else "")
        lines.append(
            f"| {md_cell(competitor)} | {md_cell(row.get('positioning') or '未稳定识别')} | {md_cell(customer_signal or '未稳定识别')} | {md_cell(top_url or '未获得稳定 URL')} |"
        )

    lines += [
        "",
        "## 3. 产品能力与工作流对比",
        "",
    ]
    for competitor in competitors:
        row = row_by_competitor.get(competitor, {})
        ok_pages = [page for page in pages if page.competitor == competitor and not page.error]
        capability = row.get("feature_signal") or next((page.fields.get("features") for page in ok_pages if page.fields.get("features")), "")
        workflow_sources = [page for page in ok_pages if any(token in page.url.lower() for token in ("product", "features", "solutions", "docs", "api"))]
        lines += [
            f"### {competitor}",
            f"- **核心能力:** {capability or '未稳定识别'} — {'Fact/Inference' if capability else 'Assumption'}",
            f"- **工作流证据:** {', '.join(page.url for page in workflow_sources[:4]) or '未抓到稳定功能/产品页'}",
            f"- **能力判断:** 当前证据支持的能力边界有限，需结合官网产品页、文档、截图和真实试用继续补证 — Inference",
            "",
        ]

    lines += [
        "## 4. 定价、套餐与商业化包装",
        "",
        "| 竞品 | 定价/套餐信号 | 证据来源 | 待补字段 |",
        "|---|---|---|---|",
    ]
    for competitor in competitors:
        ok_pages = [page for page in pages if page.competitor == competitor and not page.error]
        pricing_pages = [page for page in ok_pages if any(token in page.url.lower() for token in ("pricing", "price", "plans"))]
        pricing_signal = next((page.fields.get("pricing") for page in pricing_pages if page.fields.get("pricing")), "")
        if not pricing_signal:
            pricing_signal = next((page.fields.get("pricing") for page in ok_pages if page.fields.get("pricing")), "")
        source = pricing_pages[0].url if pricing_pages else (ok_pages[0].url if ok_pages else "")
        lines.append(
            f"| {md_cell(competitor)} | {md_cell(pricing_signal or '未稳定识别')} | {md_cell(source or '缺少官方定价页')} | 免费版、价格、额度、限制、企业版、Add-on |"
        )

    lines += [
        "",
        "## 5. 目标用户、市场与 GTM",
        "",
    ]
    for competitor in competitors:
        ok_pages = [page for page in pages if page.competitor == competitor and not page.error]
        customer_signal = next((page.fields.get("customers") for page in ok_pages if page.fields.get("customers")), "")
        gtm_pages = [page.url for page in ok_pages if any(token in page.url.lower() for token in ("customers", "case", "blog", "press", "about", "enterprise"))]
        lines += [
            f"### {competitor}",
            f"- **目标用户/ICP:** {customer_signal or '未稳定识别'} — {'Fact/Inference' if customer_signal else 'Assumption'}",
            f"- **GTM 信号:** {', '.join(gtm_pages[:5]) or '未抓到客户案例/新闻/企业页'}",
            "- **需要补证:** 渠道、销售模式、地域、本地化、合作伙伴、内容营销和 PLG/销售分工。",
            "",
        ]

    lines += [
        "## 6. 客户体验、服务支持与产品质量",
        "",
    ]
    for competitor in competitors:
        ok_pages = [page for page in pages if page.competitor == competitor and not page.error]
        support_pages = [page.url for page in ok_pages if any(token in page.url.lower() for token in ("docs", "help", "support", "security", "trust", "status"))]
        lines += [
            f"### {competitor}",
            f"- **支持/文档证据:** {', '.join(support_pages[:5]) or '未抓到稳定支持/文档/安全页'}",
            "- **质量判断:** 没有用户评价和真实试用时，不直接判断好坏；只标注公开支持与合规资料是否充分 — Inference",
            "",
        ]

    lines += [
        "## 7. 视觉与产品证据",
        "",
    ]
    for competitor in competitors:
        selected_images = select_images_for_competitor(competitor, image_rows, report_dir)
        lines += [f"### {competitor}", ""]
        if not selected_images:
            lines += ["未获得可直接嵌入的高价值图片证据。", ""]
            continue
        for idx, image in enumerate(selected_images, start=1):
            alt = md_cell(f"{competitor} visual evidence {idx}")
            caption = md_cell(image.get("title") or image.get("source") or "image evidence")
            lines += [
                markdown_image_tag(alt, image["src"]),
                f"_Image {idx}: {caption}; source: {image.get('source') or 'unknown'}_",
                "",
            ]

    lines += [
        "## 8. 牵引、更新节奏与战略方向",
        "",
    ]
    for competitor in competitors:
        ok_pages = [page for page in pages if page.competitor == competitor and not page.error]
        traction_pages = [page.url for page in ok_pages if any(token in page.url.lower() for token in ("customers", "case", "release", "changelog", "blog", "about"))]
        lines += [
            f"### {competitor}",
            f"- **牵引/更新证据:** {', '.join(traction_pages[:5]) or '本轮未抓到强牵引或更新节奏证据'}",
            "- **战略方向推断:** 需结合发布节奏、客户案例、融资/招聘/路线图进一步判断 — Assumption",
            "",
        ]

    lines += [
        "## 9. SWOT 与风险机会",
        "",
    ]
    for competitor in competitors:
        row = row_by_competitor.get(competitor, {})
        comp_pages = [page for page in pages if page.competitor == competitor]
        blocked = [page for page in comp_pages if page.error]
        lines += [
            f"### {competitor}",
            f"- **Strength:** {row.get('feature_signal') or row.get('positioning') or '公开证据不足'} — Inference",
            f"- **Weakness:** {'关键页面存在反爬/低文本，需要人工复核' if blocked else '缺少用户评价、真实试用和长期监控证据'} — Inference",
            "- **Opportunity:** 从其定价、核心功能、目标用户和支持资料中寻找差异化切入点 — Inference",
            "- **Threat:** 若其官方资料完整且搜索可见度高，会影响用户早期认知和采购短名单 — Inference",
            "",
        ]

    lines += [
        "## 10. 横向对比矩阵",
        "",
        "| Dimension | " + " | ".join(md_cell(c) for c in competitors) + " |",
        "|---|" + "|".join("---" for _ in competitors) + "|",
    ]

    dimensions = [
        ("定位", "positioning"),
        ("目标用户", "customers"),
        ("核心能力", "feature_signal"),
        ("定价信号", "pricing_signal"),
        ("主要缺口", "weakness"),
        ("证据等级", "evidence"),
    ]
    for label, key in dimensions:
        cells = []
        for competitor in competitors:
            row = row_by_competitor.get(competitor, {})
            comp_pages = [page for page in pages if page.competitor == competitor]
            if key == "customers":
                value = next((page.fields.get("customers") for page in comp_pages if page.fields.get("customers")), "")
            elif key == "weakness":
                value = "证据不足/反爬需复核" if any(page.error for page in comp_pages) else "待补充用户评价"
            elif key == "evidence":
                value = evidence_quality(row)
            else:
                value = row.get(key, "")
            cells.append(md_cell(value or "未识别"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    if fact_clusters:
        lines += [
            "",
            "### 结构化事实聚类",
            "",
            "| 竞品 | 字段 | 事实值 | 来源数 | 主证据 | 置信度 | 待核实 |",
            "|---|---|---:|---:|---|---|---|",
        ]
        for cluster in fact_clusters[:80]:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        cluster.get("competitor"),
                        cluster.get("field_label") or cluster.get("field_key"),
                        cluster.get("display_value"),
                        cluster.get("source_count"),
                        cluster.get("primary_source_url"),
                        cluster.get("confidence"),
                        cluster.get("needs_verification"),
                    )
                )
                + " |"
            )
    if structured_facts and not fact_clusters:
        lines += [
            "",
            "### 结构化事实",
            "",
            "| 竞品 | 字段 | 事实值 | 来源 |",
            "|---|---|---|---|",
        ]
        for fact in structured_facts[:80]:
            lines.append(
                f"| {md_cell(fact.get('competitor'))} | {md_cell(fact.get('field_label') or fact.get('field_key'))} | {md_cell(fact.get('value'))} | {md_cell(fact.get('source_url'))} |"
            )

    manual_rows = list(manual_review_rows) if manual_review_rows else rows_from_manual_review_queue(pages)
    lines += [
        "",
        "## 11. 信息缺口与问题页面核验",
        "",
    ]
    if manual_rows:
        lines += [
            "| Priority | 竞品 | URL | 问题 | 是否需登录 | 下一步 |",
            "|---|---|---|---|---|---|",
        ]
        for row in manual_rows[:40]:
            lines.append(
                f"| {row.get('priority')} | {md_cell(row.get('competitor'))} | {md_cell(row.get('url'))} | {md_cell(row.get('review_reason'))} | {md_cell(row.get('requires_user_login') or 'no')} | {md_cell(row.get('suggested_next_step'))} |"
            )
    else:
        lines += ["本轮没有反爬、登录超时、正文不足或视频缺时间点等问题页面。"]

    lines += [
        "",
        "## 12. 建议下一步与监控计划",
        "",
        "1. 先复核 `问题页面核验清单.md/csv` 中的 P0/P1 页面，确认是否公开、是否有价值、是否需要补证，再决定 include / verify_later / exclude。",
        "2. 固定每个竞品的官网、定价页、功能页、客户案例、文档/安全页、更新日志为监控清单。",
        "3. 对定价包装、能力矩阵、截图/UI、客户证据和战略动作分别做二次深挖，避免只用通用搜索摘要下结论。",
        "4. 后续可把高价值 URL 接入 ScopeHound 或定时任务做变化监控。",
        "",
        "## 13. 我方产品方向分析",
        "",
    ]
    if own_product_name or own_product_positioning or own_product_context:
        lines += [
            f"- **我方产品名称:** {own_product_name or '未填写'}",
            f"- **当前定位:** {own_product_positioning or '未填写'}",
            f"- **补充背景:** {own_product_context or '未填写'}",
            "",
            "### 可考虑的方向",
            "",
            "- **差异化定位:** 从竞品已公开强调的能力、定价、目标用户和 GTM 中寻找未被充分覆盖的细分场景。",
            "- **优先验证假设:** 先验证用户是否真的在意竞品弱项，而不是只基于功能缺口做路线判断。",
            "- **产品路线建议:** 把证据强的竞品能力列为 baseline，把反爬/缺证能力列为待验证，把我方已有优势转成可感知卖点。",
            "- **商业化建议:** 对比竞品套餐、额度、限制、企业版能力后，再决定我方免费版、专业版、团队版和企业版的边界。",
            "",
        ]
    else:
        lines += [
            "- **信息缺口:** 本轮未填写我方产品名称、定位或背景，因此不对我方方向做强推断。",
            "- **建议补充:** 在 UI 中填写我方产品定位、目标用户、核心差异化、当前阶段和希望验证的问题后，Codex 会在本节输出更具体的方向分析。",
            "",
        ]
    path.write_text(prepare_analysis_markdown("\n".join(lines), path.parent), encoding="utf-8")


def write_methodology(
    path: Path,
    max_pages_per_competitor: int,
    collection_plan: Optional[ProductCollectionPlan] = None,
    manual_include_keywords: Iterable[str] = (),
    manual_exclude_keywords: Iterable[str] = (),
    manual_search_terms: Iterable[str] = (),
) -> None:
    include_keywords = normalize_keyword_inputs(manual_include_keywords)
    exclude_keywords = normalize_keyword_inputs(manual_exclude_keywords)
    search_terms = normalize_keyword_inputs(manual_search_terms)
    lines = [
        "# 采集原则与筛选原则",
        "",
        "## 1. 目标",
        "",
        "本工具服务于 PM 竞品情报，不是通用网页剪藏。优先采集公开、合法、可回溯、能支持产品决策的信息，尤其是官网、定价页、产品页、功能页、客户/场景页、文档、API、Trust/Security、Legal、Changelog、Release notes 和产品截图。",
        "",
        "## 1.1 抓取前产品分析",
        "",
        "- 系统会先读取我方产品名称、定位和补充背景，识别本轮品类，再生成 `pre_crawl_plan.md/json`。",
        "- 这份计划会追加搜索模板、图片关键词和页面抽取字段。实物商品会补抓材质、重量、尺码、颜色、质量、售后等；AI/软件会补抓 API、SDK、集成、模型、额度、安全、部署等。",
        "- 官网、垂直平台、论坛社区、App/社媒/视频都先通过 SearXNG 定向搜索找到具体入口；GUI 只用于机器抓不到但疑似有价值的公开页面。",
        "- 计划只决定“应该多抓什么”，不直接替代证据。最终报告仍以公开页面、图片和 Codex 收录判断为准。",
        "",
        "## 1.2 本轮信息筛选策略",
        "",
        "筛选不是一个模糊相关性分数，而是按固定顺序过门禁：",
        "",
        "1. **来源门禁:** 判断来源是官方核心页、官方补充页、可信第三方、社区线索、目录/聚合、登录/交易壳或非 HTML 资产。",
        "2. **页面角色门禁:** 判断页面是否承担定价包装、产品规格、功能方案、API/文档、安全合规、客户案例、更新日志、评测对比等角色。",
        "3. **预抓取产品分析:** 系统先根据我方产品名称、定位、背景和竞品名识别品类，再自动生成搜索补充词、证据关键词和动态排除词。",
        "4. **产品字段命中:** 用本轮品类计划检查是否命中应采字段，例如 AI 产品的 API、模型、额度、集成、安全，或实物商品的材质、重量、尺码、颜色、认证、质保。",
        "5. **关键词覆盖:** 搜索补充词默认由预抓取分析自动生成；固定排除词处理登录、购物车、招聘、破解下载等跨品类噪声；动态排除词处理本品类容易混淆的内容。UI 只保留可选的人工重点关注词，用于临时提高某类证据的优先级。",
        "6. **证据强度判断:** 官方事实、可追溯 URL、结构化参数/定价/文档优先；第三方只验证口碑和市场感知；社区/论坛只保留为低置信线索。",
        "7. **信息增量判断:** 标记每条来源新增的是价格/套餐、参数/规格、API/额度、安全认证、客户场景、版本发布、用户反馈还是普通线索。只复述已有事实的内容折叠为补充来源。",
        "8. **事实组和主证据:** 同一价格、参数、功能、认证、客户案例或质量问题会进入同一个 `fact_group`。`primary_evidence_candidate=yes` 的来源优先进入报告引用。",
        "9. **待核实标记:** 二创聚合、社区单点反馈、反爬/JS 壳、缺少日期/来源的内容会标记 `pending_verification=yes`，不能直接写成 Fact。",
        "10. **预算选择:** Crawl4AI 抓取预算优先给官方核心页和高价值证据页。没进预算但有价值的内容会标记为 `accepted` 或 `signal`，不会被误删。",
        "11. **人工复核:** 反爬、403、JS 空壳、疑似核心页但正文不可抽取时进入 GUI 复核队列；只允许使用公开页面、官网导航、sitemap、公开文档或人工摘录公开可见内容补证。",
        "12. **本地训练模型:** 若存在 `models/filter_model.pt` 或通过 `--ml-model` 指定模型，系统会基于历史人工标注为候选来源追加 `ml_include_score`、`ml_exclude_score` 和 `ml_verify_later_score`。规则硬拒绝仍优先于模型。",
        "13. **搜索卡片复用:** 用户核验后重新训练时，系统会按 `product_type_key` 生成或更新 `search_cards/*.json`。下一次同类产品采集会自动加载卡片，补充搜索词、可复用来源和排除词。卡片只来自人工核验样本，不要求提前穷举所有商品类型。",
        "",
        "每条搜索结果会输出 `source_kind`、`page_role`、`source_policy_tier`、`decision_status`、`pending_verification`、`fact_type`、`increment_type`、`fact_group`、`primary_evidence_candidate`、`value_signals`、`value_missing`、`value_verdict`、`gui_review_candidate`、`gui_review_value_reason`、`ml_label`、`ml_include_score`、`ml_exclude_score`、`hard_gate`、`confidence`、字段命中、关键词命中和五个分项评分，避免只看到“相关性太低/正文太短”。",
        "",
        "## 1.2.1 价值判断原则",
        "",
        "候选来源不是只看相关性，而是逐项判断：",
        "",
        "| 判断项 | 算有价值的情况 | 不算有价值的情况 |",
        "|---|---|---|",
    ]
    for rule in (collection_plan.value_judgment_rules if collection_plan else build_value_judgment_rules()):
        lines.append(f"| {md_cell(rule.label)} | {md_cell(rule.positive_rule)} | {md_cell(rule.negative_rule)} |")
    lines += [
        "",
        "只有“竞品绑定、决策相关、信息增量、来源可信、可追溯、可获取”中至少命中关键项的来源，才会进入抓取、待核实或报告候选。视频/社媒内容如果没有可读文本，必须先满足竞品绑定、决策相关、信息增量和可追溯，才进入 GUI 复核。",
        "",
        "## 1.3 数据源获取策略",
        "",
        "数据源不是一起混抓，而是先分层：先找竞品官方来源，再找垂直品类和应用商店，再看论坛、社区、社媒和视频，最后才用海量搜索兜底。每类来源都会记录为什么选择、怎么获取、能证明什么、不能证明什么。",
        "",
        "- **官网和官方文档:** 用来证明参数、定价、套餐、功能、API、合规、更新节奏等一手事实。遇到反爬或页面空壳时，不直接丢弃，进入 GUI 复核队列，但只允许补充公开可见内容。",
        "- **垂直品类和应用商店:** 用来补充评分、版本、截图、评论、榜单、SKU、授权零售或品类排名。它可以验证市场感知，不能覆盖官网事实。",
        "- **论坛和社区:** 用来发现用户痛点、质量问题、购买顾虑和真实使用语言。单条帖子默认是低置信线索，必须保留 URL、发布时间、作者展示名或楼层位置。",
        "- **App、社媒和视频:** 文字类公开内容可直接抓取；视频类要记录视频 URL、发布时间、观点出现的时间点，必要时留下公开画面截图。没有时间点和截图的观点不能写成强事实。",
        "- **海量搜索:** 用来发现遗漏来源、别名、替代品和品类词。它是兜底入口，不是事实来源本身；进入报告前仍要回到官方、垂直来源或可追溯页面。",
        "",
        "## 1.4 竞品发现与入池策略",
        "",
        "如果用户已经输入竞品，系统先核验竞品实体：确认官网、官方账号、别名、母公司或产品系列，避免同名无关内容。如果用户没有完整竞品，系统应先从我方产品定位中抽取用户任务、目标人群、核心能力、购买场景和品类词，检索 alternatives、competitors、top、best、替代品、对比 等意图词，再生成候选竞品池。",
        "",
        "- **直接竞品:** 用户任务、目标用户和核心能力至少两项重叠，优先进入抓取预算。",
        "- **相邻竞品:** 目标用户或能力部分重叠，但购买场景不同，单独标注，不和直接竞品混成一类。",
        "- **替代方案:** 用户可能用它完成同一个任务，但产品形态不同，只用于方向判断或机会分析。",
        "- **待核实候选:** 只在榜单、二创文章或社区单点内容中出现，暂不作为核心竞品，需要补官网或官方账号证据。",
        "",
        "每个候选竞品都要保存发现搜索词、发现页面、候选名称、官方入口、入池理由和置信等级。只靠 SEO 榜单、广告位、同名泛词或无官方入口的候选，不进入核心竞品清单。",
        "",
        "## 2. 采集来源优先级",
        "",
        "- **P0 官方核心来源:** official website、pricing/plans、product/features、solutions/use cases、docs/help、API/developers、security/trust、legal、changelog/release notes。",
        "- **P1 官方补充来源:** blog、press、customer stories、case studies、demo videos、App Store/Chrome Store 官方页面、status page。",
        "- **P2 第三方验证来源:** G2、Capterra、主流新闻、融资数据库、分析文章、评测文章。只用于验证口碑、市场牵引和外部感知，不覆盖官方事实。",
        "- **P3 低置信线索:** 社媒、论坛、社区讨论、用户评论。只作为线索，不直接当作产品事实。",
        "- **禁用来源:** 登录页、注册页、购物车、账号中心、卖家中心、论坛导航、破解接口、私密内容、需要伪装身份或绕过访问控制的内容。",
        "",
        "## 3. URL 选择信号",
        "",
        "- **竞品相关性:** 精确竞品名和有区分度的品牌词权重最高。",
        "- **官方域发现:** 会从搜索结果 URL 和搜索摘要中抽取官网链接；例如第三方文章摘要里出现的官网直达链接，会被提升为官方候选。若搜索没有发现官网，会优先尝试 `{brand}.com`、`{brand}.app`、`{brand}.ai` 等官网探测，而不是抓同名无关页面。",
        "- **官方核心页扩展:** 识别到官网域名后，会尝试补抓 pricing、features、product、customers、docs、security、changelog 等核心路径。",
        "- **官方核心关键词:** pricing、plans、features、product、solutions、customers、docs、api、security、trust、legal、changelog、release、about 等优先。",
        "- **可抓取性:** Crawl4AI 优先抓 HTML 页面；PDF、视频、压缩包、脚本、原始图片保留在来源表，但不作为网页正文抓取目标。",
        "- **来源广度:** `max-pages` 是抓取上限，不再为了填满预算而抓低质量页面；有官网时，第三方公开资料只保留少量高价值验证源。",
        "- **噪声控制:** 社交媒体、百科、登录/注册、购物车、账号中心、市场导航、目录站、SEO 聚合、论坛导航、低相关页面会被降权或排除。",
        "- **反样本保留:** 被排除内容仍会进入 `all_sources.csv`、`unfiltered_collection.md` 或 `evidence_audit.csv`，便于追溯为什么不要。",
        "- **问题页面核验:** 反爬、403、Cloudflare、验证码、JS 壳、低文本、登录超时、视频缺时间点和待核实来源统一进入 `问题页面核验清单.md/csv`，先由人工确认是否公开、是否有价值、是否需要补证，再决定 include / verify_later / exclude。",
        "- **登录辅助:** 登录页、注册页和账号权限页会先进入 `需登录队列.md/csv`，按竞品和域名去重；开启 `--login-assist` 后，工具复用同一个可见浏览器登录态，公开页面采集结束后统一等待，登录后可读则保存快照，仍不可读则标记为超时未人工登录或需账号权限。",
        "- **合规边界:** 可使用站点公开导航、sitemap、官方文档/API、帮助中心、静态页、公开下载资料或人工摘录公开/授权可见内容；不破解验证码、不绕过登录/付费/访问控制、不保存账号凭据、不调用未授权私有接口。",
        "",
        "## 4. 页面内容筛选原则",
        "",
        "- **保留:** 能证明官方定位、目标用户、核心功能、定价包装、参数规格、套餐限制、API/集成、安全合规、客户案例、更新节奏、GTM 信号的内容。",
        "- **排除:** 登录页、注册页、纯导航页、站内搜索页、购物车、账号中心、论坛灌水、SEO 聚合、教程转载、同名无关产品、正文过短或反爬空白页。",
        "- **清洗:** 会剥离重复导航、登录提示、淘宝类菜单、论坛频道、浏览/回复/点赞等壳信息。",
        "- **置信度:** 官方页面最高；第三方评价用于感知验证；论坛/社媒只做低置信线索。",
        "- **事实边界:** 竞品官网内容是“官方声称”的事实；是否真实好用，需要评论、案例、试用或客户证据验证。",
        "",
        "## 5. Codex 收录原则",
        "",
        "- Codex 只在规则筛选后的候选证据池上做最终收录判断。",
        "- Codex 必须说明为什么收录或排除关键页面/图片。",
        "- 最终报告必须使用 Fact / Inference / Assumption 标注结论属性。",
        "- 若填写我方产品定位，Codex 只基于该定位和竞品证据做方向分析；未填写时只标注信息缺口，不编造我方方向。",
        "- 最终分析报告在导出前会应用 `stop-slop-zh` 话术清理：只改中文表达，不改证据、数字、表格、图片和来源 URL。",
        "- 图片只有在能支持产品能力、UI/UX、定价、截图、Logo 或关键视觉证据时才放入报告。",
        "",
        "## 6. 最终分析报告框架",
        "",
        FINAL_REPORT_FRAMEWORK_TEXT,
        "",
        "## 7. 正式导出文件",
        "",
        "- `all_sources.csv`: 所有采集来源，包括 SearXNG 网页候选、SearXNG 图片候选、Crawl4AI 页面、icrawler 下载图片。",
        "- `pre_crawl_plan.md/json`: 抓取前采集计划，说明我方产品分析、识别品类、追加字段、追加搜索词和报告关注点。",
        "- `competitor_discovery.md/csv/json`: 无竞品输入时的自动发现记录，说明候选从哪里来、为什么入池、是否用于采集。",
        "- `unfiltered_collection.md`: 未经筛选的采集内容，保留搜索摘要、抓取页面摘要、图片候选。",
        "- `filtered_collection.md`: 规则筛选后的候选证据池，供 Codex 分析使用。",
        "- `structured_facts.csv/json`: 从页面正文中抽取的价格、参数、材质、尺码、认证、API、额度、安全等事实。",
        "- `fact_clusters.md/csv/json`: 同一事实的聚类结果，官方来源优先作为主证据，第三方来源作为补充证据。",
        "- `问题页面核验清单.md/csv`: 统一收纳反爬、403、Cloudflare、验证码、正文不足、登录超时、视频缺时间点和待核实来源，并给出核验事项和入库标注建议。",
        "- `anti_bot_strategy.md`: 本轮反爬/异常类型统计、合规补证顺序和禁止动作。",
        "- `竞品分析报告_图片内嵌版.md`: 唯一正式分析报告，图片以内嵌 base64 形式保存，适合单独下载/转发时避免图片丢失。",
        "- `training_review_sample.md/csv`: 本轮抽样标注表，人工填写后可追加到 `training_data/review_labels.csv` 训练本地筛选模型。",
        "- `search_cards/*.json`: 人工核验后生成的品类搜索卡片，下一轮同类产品会自动加载。",
        "- `ml_filter_status.json`: 本轮本地训练模型加载状态、模型版本、训练样本数和自动判断阈值。",
        "- `人工抽样标注表.csv`: 本轮抽样标注表，人工填写后可追加到 `training_data/review_labels.csv` 训练本地 `.pt` 筛选模型。",
        "- `_internal/`: 旧版英文文件、原始 JSON、Codex 输入输出、分散复核队列、运行日志和图片下载目录，默认不作为交付物展示。",
        "",
        f"当前每个竞品 Crawl4AI 抓取预算：`{max_pages_per_competitor}` 页。",
        "",
    ]
    if collection_plan:
        lines += [
            "## 8. 本轮品类预设与人工覆盖",
            "",
            f"- **识别品类:** {collection_plan.category_label} (`{collection_plan.category}`)",
            f"- **判断原因:** {collection_plan.rationale}",
            "- **应采字段:** " + "、".join(field.label for field in collection_plan.fields),
            "- **自动搜索补充词:** " + (", ".join(collection_plan.generated_search_terms[:100]) or "无"),
            "- **证据关键词预设:** " + (", ".join(collection_plan.evidence_keywords[:100]) or "无"),
            "- **固定排除/降权关键词:** " + (", ".join(collection_plan.fixed_exclude_keywords[:100]) or "无"),
            "- **动态排除/降权关键词:** " + (", ".join(collection_plan.dynamic_exclude_keywords[:100]) or "无"),
            "- **已加载搜索卡片:** " + (", ".join(card.get("product_type_label") or card.get("product_type_key") for card in collection_plan.search_cards_applied) if collection_plan.search_cards_applied else "无"),
            "",
            "### 本轮竞品发现策略",
            "",
            "| 阶段 | 策略 | 触发条件 | 入池规则 | 追溯规则 |",
            "|---|---|---|---|---|",
        ]
        for item in collection_plan.competitor_discovery_strategies:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        item.stage,
                        item.name,
                        item.trigger,
                        item.acceptance_rule,
                        item.traceability_rule,
                    )
                )
                + " |"
            )
        lines += [
            "",
            "### 本轮数据源策略",
            "",
            "| 优先级 | 数据源类别 | 典型来源 | 选择规则 | 追溯规则 |",
            "|---|---|---|---|---|",
        ]
        for item in collection_plan.source_strategies:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(value)
                    for value in (
                        item.priority,
                        item.name,
                        "、".join(item.source_examples),
                        item.selection_rule,
                        item.traceability_rule,
                    )
                )
                + " |"
            )
        lines.append("")
        for note in collection_plan.source_policy_notes:
            lines.append(f"- {note}")
        lines.append("")
    if search_terms or include_keywords or exclude_keywords:
        lines += [
            "### 本轮人工覆盖",
            "",
            "- **手动搜索补充词:** " + (", ".join(search_terms) or "无。默认由预抓取分析自动生成。"),
            "- **人工重点关注词:** " + (", ".join(include_keywords) or "无"),
            "- **手动排除词:** " + (", ".join(exclude_keywords) or "无。默认由固定排除词和动态排除词共同决定。"),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def anti_bot_bucket(error: str) -> str:
    low = textify(error).lower()
    if "auth_or_transaction" in low or any(token in low for token in ("login", "signin", "signup", "register", "passport", "account")):
        return "登录/注册/账号权限"
    if "cloudflare" in low or "js challenge" in low:
        return "Cloudflare/JS challenge"
    if "datadome" in low or "captcha" in low:
        return "Captcha/DataDome"
    if "403" in low or "forbidden" in low:
        return "HTTP 403"
    if "script_heavy_shell" in low or "minimal_text" in low or "no_extractable" in low:
        return "JS 空壳/正文不足"
    if "connection refused" in low:
        return "连接失败"
    if "broken" in low or "404" in low:
        return "404/失效页面"
    return "其他异常"


def write_anti_bot_strategy_doc(
    path: Path,
    pages: Sequence[PageExtract],
    manual_review_rows: Sequence[Mapping[str, Any]],
    gui_review_rows: Sequence[Mapping[str, Any]],
) -> None:
    buckets: Dict[str, int] = {}
    for page in pages:
        if page.error:
            bucket = anti_bot_bucket(page.error)
            buckets[bucket] = buckets.get(bucket, 0) + 1
    lines = [
        "# 反爬与异常页面处理策略",
        "",
        "本策略优先处理公开可访问的信息源。遇到登录/注册/账号权限页面时，工具先进入集中登录队列，公开页面继续采集；网页抓取结束后统一等待用户授权并复用同一登录态保存快照。它的目标是减少误删有价值页面，而不是破解验证码或绕过站点访问控制。",
        "",
        "## 已观察到的问题类型",
        "",
    ]
    if buckets:
        lines += ["| 类型 | 数量 |", "|---|---|"]
        for bucket, count in sorted(buckets.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {md_cell(bucket)} | {count} |")
    else:
        lines.append("本轮没有记录到 Crawl4AI 异常页面。")
    lines += [
        "",
        "## 合规处理顺序",
        "",
        "1. **同站公开替代入口:** 优先找官网导航、`sitemap.xml`、帮助中心、文档页、静态产品页、公开 PDF 或公开下载资料。",
        "2. **搜索定向补证:** 用 SearXNG 针对同一竞品和同一字段重搜，例如 `site:官网域名 pricing`、`site:官网域名 docs API`。",
        "3. **公开元数据:** 视频和社媒先保留公开 URL、标题、作者展示名、发布时间、公开视频元数据；没有时间点、截图或字幕时不能写成强事实。",
        "4. **浏览器公开快照:** 自动打开公开页面并保存文本快照或截图；只记录用户无需登录即可看到的内容。",
        "5. **登录辅助复核:** 如果页面明确要求登录/注册，先按竞品和域名去重放入等待区；网页抓取结束后统一等待用户用有权限的账号登录，页面可读则保存快照，仍不可读则进入问题页面核验清单。",
        "6. **人工公开摘录:** 如果机器抓不到但页面公开或授权可见，可以人工摘录关键句，并保留 URL、截图、抓取时间和摘录人。",
        "",
        "## 禁止做的事",
        "",
        "- 不破解验证码。",
        "- 不绕过登录、付费墙、地理封锁或访问控制。",
        "- 不保存、导出或复用用户账号凭据。",
        "- 不调用未授权私有接口。",
        "- 不用私有 Cookie、账号权限或伪装身份批量抓取。",
        "- 不把二创、搬运、SEO 聚合内容当成 Fact；无法回到原始来源时进入待核实。",
        "",
        "## 自动复核结果",
        "",
        f"- 问题页面核验候选：{len(manual_review_rows)} 条",
        f"- 需登录/授权候选：{len(login_required_queue_rows(manual_review_rows, gui_review_rows))} 条",
        f"- 已自动公开复核：{len(gui_review_rows)} 条",
        "",
    ]
    if gui_review_rows:
        lines += ["| 竞品 | 状态 | URL | 下一步 |", "|---|---|---|---|"]
        for row in gui_review_rows[:60]:
            lines.append(
                f"| {md_cell(row.get('competitor'))} | {md_cell(row.get('automated_review_status'))} | {md_cell(row.get('url'))} | {md_cell(row.get('next_step'))} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def truncate_text(value: str, limit: int = 1200) -> str:
    value = re.sub(r"\s+", " ", textify(value)).strip()
    return value if len(value) <= limit else value[:limit] + "..."


def write_codex_input_bundle(
    path: Path,
    competitors: Sequence[str],
    competitor_rows: Sequence[Dict[str, Any]],
    web_results: Sequence[SearchResult],
    pages: Sequence[PageExtract],
    image_rows: Sequence[Dict[str, Any]],
    evidence_audit_rows: Sequence[Dict[str, Any]],
    manual_review_rows: Sequence[Dict[str, Any]],
    own_product_name: str = "",
    own_product_positioning: str = "",
    own_product_context: str = "",
    collection_plan: Optional[ProductCollectionPlan] = None,
    structured_facts: Sequence[Mapping[str, Any]] = (),
    fact_clusters: Sequence[Mapping[str, Any]] = (),
    gui_review_rows: Sequence[Mapping[str, Any]] = (),
    competitor_discovery: Optional[Mapping[str, Any]] = None,
) -> None:
    report_dir = path.parent
    lines = [
        "# Codex Review Input Bundle",
        "",
        "This file summarizes local crawl evidence for Codex AI inclusion review.",
        "Codex should use this file plus local CSV/JSON files in the same folder. Do not browse the web.",
        "",
        "## Required Final Report Framework",
        "",
        FINAL_REPORT_FRAMEWORK_TEXT,
        "",
        "## Own Product Context",
        "",
        f"- Product name: {own_product_name or 'NOT PROVIDED'}",
        f"- Positioning: {own_product_positioning or 'NOT PROVIDED'}",
        f"- Additional context / current direction: {own_product_context or 'NOT PROVIDED'}",
        "",
        "Use this own-product context only for section 13. If it is not provided, state the gap instead of inventing our product direction.",
        "",
        "## Pre-Crawl Product-Specific Collection Plan",
        "",
    ]
    if collection_plan:
        lines += [
            f"- Category: {collection_plan.category_label} ({collection_plan.category})",
            f"- Rationale: {collection_plan.rationale}",
            "- Required extra fields:",
        ]
        for field in collection_plan.fields:
            lines.append(f"  - {field.label} ({field.key}): {field.description}")
        lines += ["- Report focus:"]
        for focus in collection_plan.report_focus:
            lines.append(f"  - {focus}")
        if collection_plan.competitor_discovery_strategies:
            lines += ["- Competitor discovery strategy:"]
            for item in collection_plan.competitor_discovery_strategies:
                lines.append(
                    "  - "
                    + f"{item.stage} {item.name}: trigger={item.trigger}; "
                    + f"acceptance={item.acceptance_rule}; traceability={item.traceability_rule}"
                )
        if collection_plan.source_strategies:
            lines += ["- Source strategy:"]
            for item in collection_plan.source_strategies:
                lines.append(
                    "  - "
                    + f"{item.priority} {item.name}: sources={', '.join(item.source_examples)}; "
                    + f"selection={item.selection_rule}; traceability={item.traceability_rule}"
                )
        if collection_plan.search_term_reasons:
            lines += ["- Search term generation reasons:"]
            for row in collection_plan.search_term_reasons[:80]:
                lines.append(
                    "  - "
                    + f"{row.get('term')}: field={row.get('source_field')}; "
                    + f"source={row.get('expected_source')}; evidence={row.get('expected_evidence')}; "
                    + f"risk={row.get('noise_risk')}; by={row.get('generated_by')}"
                )
        lines.append("")
    else:
        lines += ["- NOT PROVIDED", ""]

    lines += [
        "## Competitors",
        "",
    ]
    for row in competitor_rows:
        lines.append(
            f"- {row.get('competitor')}: pages {row.get('pages_ok')}/{row.get('pages_crawled')}, images {row.get('images_found')}, top URL {row.get('top_url')}"
        )

    valid_pages = [page for page in pages if not page.error]
    excluded_pages = [page for page in pages if page.error]

    lines += ["", "## Crawled Pages Accepted By Rule Filter", ""]
    for page in valid_pages[:120]:
        lines += [
            f"### {page.competitor} | {page.title or page.url}",
            f"- URL: {page.url}",
            "- Status: ok",
            f"- Positioning signal: {truncate_text(page.fields.get('positioning', ''), 300)}",
            f"- Pricing signal: {truncate_text(page.fields.get('pricing', ''), 300)}",
            f"- Feature signal: {truncate_text(page.fields.get('features', ''), 300)}",
            f"- Customer signal: {truncate_text(page.fields.get('customers', ''), 300)}",
        ]
        if collection_plan:
            for field in collection_plan.fields:
                value = truncate_text(page.fields.get(field.key, ""), 260)
                if value:
                    lines.append(f"- Product-specific signal / {field.label}: {value}")
        lines += [
            f"- Excerpt: {truncate_text(page.text_excerpt, 900)}",
            "",
        ]

    lines += ["", "## Crawled Pages Excluded By Rule Filter", ""]
    for page in excluded_pages[:120]:
        lines += [
            f"- Competitor: {page.competitor}",
            f"  URL: {page.url}",
            f"  Title: {truncate_text(page.title, 180)}",
            f"  Exclusion reason: {truncate_text(page.error, 300)}",
            f"  Cleaned excerpt sample: {truncate_text(page.text_excerpt, 260)}",
        ]

    lines += ["", "## Top Search Evidence", ""]
    for result in sorted(web_results, key=lambda item: item.score, reverse=True)[:160]:
        lines += [
            f"- Competitor: {result.competitor}",
            f"  URL: {result.url}",
            f"  Title: {truncate_text(result.title, 180)}",
            f"  Score: {result.score:.2f}; Engine: {result.engine}; Query: {result.query}",
            f"  Snippet: {truncate_text(result.snippet, 360)}",
        ]

    lines += ["", "## Image Evidence", ""]
    for row in image_rows[:160]:
        local_file = row.get("local_file") or ""
        image_url = row.get("image_url") or ""
        src = markdown_image_src(row, report_dir)
        if not src:
            continue
        lines += [
            f"- Competitor: {row.get('competitor', '')}",
            f"  Source: {row.get('source', '')}",
            f"  Original Path/URL: {local_file or image_url}",
            f"  Markdown image path: {src}",
            f"  Markdown image syntax: {markdown_image_tag(row.get('title') or row.get('query') or row.get('competitor') or 'visual evidence', src)}",
            f"  Title/Query: {truncate_text(row.get('title') or row.get('query') or '', 220)}",
        ]

    if competitor_discovery:
        lines += ["", "## Automatic Competitor Discovery", ""]
        lines.append(f"- Used for collection: {competitor_discovery.get('used_for_collection')}")
        for candidate in (competitor_discovery.get("candidates") or [])[:40]:
            lines += [
                f"- {candidate.get('name')} | {candidate.get('candidate_type')} | {candidate.get('confidence')} | {candidate.get('status')}",
                f"  Official URL: {candidate.get('official_url')}",
                f"  Discovered query: {candidate.get('discovered_query')}",
                f"  Evidence URL: {candidate.get('discovered_from_url')}",
            ]

    lines += ["", "## Structured Facts", ""]
    if structured_facts:
        for fact in structured_facts[:160]:
            lines += [
                f"- {fact.get('competitor')} | {fact.get('field_label') or fact.get('field_key')}: {fact.get('value')}",
                f"  Source: {fact.get('source_url')}",
                f"  Evidence: {truncate_text(fact.get('evidence_text', ''), 300)}",
                f"  Confidence: {fact.get('confidence')}; needs_verification={fact.get('needs_verification')}",
            ]
    else:
        lines.append("- No structured facts extracted.")

    lines += ["", "## Fact Clusters", ""]
    if fact_clusters:
        for cluster in fact_clusters[:120]:
            lines += [
                f"- {cluster.get('competitor')} | {cluster.get('field_label') or cluster.get('field_key')}: {cluster.get('display_value')} | sources={cluster.get('source_count')}",
                f"  Primary source: {cluster.get('primary_source_url')}",
                f"  Supporting sources: {cluster.get('supporting_source_urls')}",
                f"  Confidence: {cluster.get('confidence')}; needs_verification={cluster.get('needs_verification')}",
            ]
    else:
        lines.append("- No fact clusters produced.")

    lines += ["", "## Evidence Audit Sample", ""]
    for row in evidence_audit_rows[:160]:
        lines += [
            f"- {row.get('competitor')} | {row.get('decision_status') or row.get('decision')} | {row.get('confidence')} | {row.get('url')}",
            f"  Source/PageRole/Policy/Gate: {row.get('source_kind')} / {row.get('page_role')} / {row.get('source_policy_tier')} / {row.get('hard_gate')}",
            f"  Fact strategy: type={row.get('fact_type')}; increment={row.get('increment_type')}; fact_group={row.get('fact_group')}",
            f"  Primary evidence: {row.get('primary_evidence_candidate')} ({row.get('primary_evidence_reason')})",
            f"  Pending verification: {row.get('pending_verification')} ({row.get('verification_reason')})",
            f"  Value judgment: signals={row.get('value_signals')}; missing={row.get('value_missing')}; verdict={row.get('value_verdict')}; gui_review={row.get('gui_review_candidate')} ({row.get('gui_review_value_reason')})",
            f"  Local ML filter: label={row.get('ml_label')}; include={row.get('ml_include_score')}; exclude={row.get('ml_exclude_score')}; verify_later={row.get('ml_verify_later_score')}; adjustment={row.get('ml_adjustment')}",
            f"  Scores: relevance {row.get('relevance_score')}, evidence {row.get('evidence_score')}, PM value {row.get('pm_value_score')}, traceability {row.get('traceability_score')}, category fit {row.get('category_fit_score')}",
            f"  Matched fields: {truncate_text(row.get('matched_fields', ''), 180)}",
            f"  Matched keywords: {truncate_text(row.get('matched_include_keywords', ''), 180)}",
            f"  Reason: {truncate_text(row.get('reason', ''), 360)}",
        ]

    lines += ["", "## Manual GUI Review Queue", ""]
    if manual_review_rows:
        for row in manual_review_rows[:120]:
            lines += [
                f"- {row.get('priority')} | {row.get('competitor')} | {row.get('review_reason')} | {row.get('url')}",
                f"  Requires user login: {row.get('requires_user_login')}; login assist URL: {row.get('login_assist_url')}",
                f"  Error: {truncate_text(row.get('crawl_error', ''), 260)}",
                f"  GUI review URL: {row.get('gui_review_url')}",
                f"  Suggested next step: {truncate_text(row.get('suggested_next_step', ''), 360)}",
            ]
    else:
        lines.append("- No manual GUI review candidates in this run.")

    lines += ["", "## Automated GUI Review Results", ""]
    if gui_review_rows:
        for row in gui_review_rows[:80]:
            lines += [
                f"- {row.get('competitor')} | {row.get('automated_review_status')} | {row.get('url')}",
                f"  Adapter/platform: {row.get('adapter_name')} / {row.get('platform')} / {row.get('source_family')}",
                f"  Requires user login: {row.get('requires_user_login')}; login assist URL: {row.get('login_assist_url')}",
                f"  Text snapshot: {row.get('text_snapshot_path')}",
                f"  Screenshot: {row.get('screenshot_path')}",
                f"  Metadata: {row.get('metadata_path')}",
                f"  Transcript: {row.get('transcript_path')}",
                f"  Evidence markers: {row.get('evidence_markers_path')}; needs_video_timestamp={row.get('needs_manual_video_timestamp')}",
                f"  Snapshot excerpt: {truncate_text(row.get('text_snapshot_excerpt', ''), 500)}",
                f"  Next step: {truncate_text(row.get('next_step', ''), 300)}",
            ]
    else:
        lines.append("- No automated GUI review results in this run.")

    lines += [
        "",
        "## Local Files Available",
        "",
        "- `all_sources.csv`: all collected sources across search, crawl, and image stages",
        "- `competitor_discovery.csv` / `competitor_discovery.md`: traceable automatic competitor discovery candidates",
        "- `unfiltered_collection.md`: raw/unfiltered collected candidates",
        "- `filtered_collection.md`: rule-filtered candidate evidence pool",
        "- `problem_pages_review.csv` / `problem_pages_review.md`: one merged list of anti-bot, 403, low-text, login-timeout, video-timestamp, and pending-verification issues",
        "- `structured_facts.csv`: structured facts extracted from pages",
        "- `fact_clusters.csv` / `fact_clusters.md`: clustered facts with primary and supporting evidence",
        "- `_internal/gui_review_results.csv` / `_internal/gui_review_results.md`: internal automated public review snapshots for GUI/video/social candidates",
        "- `anti_bot_strategy.md`: observed blocked-page types and compliant evidence recovery plan",
        "- `_internal/raw.json`: full structured evidence",
        "- `_internal/pages.csv`: crawled page extracts",
        "- `_internal/images.csv`: image index and local downloaded image paths",
        "- `_internal/evidence_audit.csv`: rule-based selected/rejected URL audit",
        "- `ml_filter_status.json`: local trained filter model status for this run",
        "- `training_review_sample.csv` / `_internal/training_review_sample.md`: sampled rows for human labels and future local model training",
        "- `_internal/search_results.csv`: normalized search results",
        "- `_internal/analysis.md`: rule-generated baseline analysis",
        "- `collection_principles.md`: collection and filtering principles",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def codex_review_schema() -> Dict[str, Any]:
    decision_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["competitor", "decision", "confidence", "url_or_path", "reason", "recommended_use"],
        "properties": {
            "competitor": {"type": "string"},
            "decision": {"type": "string"},
            "confidence": {"type": "string"},
            "url_or_path": {"type": "string"},
            "reason": {"type": "string"},
            "recommended_use": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "analysis_markdown",
            "included_pages",
            "excluded_pages",
            "included_images",
            "excluded_images",
            "quality_notes",
        ],
        "properties": {
            "summary": {"type": "string"},
            "analysis_markdown": {"type": "string"},
            "included_pages": {"type": "array", "items": decision_item},
            "excluded_pages": {"type": "array", "items": decision_item},
            "included_images": {"type": "array", "items": decision_item},
            "excluded_images": {"type": "array", "items": decision_item},
            "quality_notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def parse_json_payload(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def pre_crawl_ai_strategy_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "rationale",
            "generated_search_terms",
            "dynamic_exclude_keywords",
            "evidence_keywords",
            "report_focus",
            "source_policy_notes",
        ],
        "properties": {
            "rationale": {"type": "string"},
            "generated_search_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
            "dynamic_exclude_keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
            "evidence_keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 60},
            "report_focus": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            "source_policy_notes": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        },
    }


def merge_pre_crawl_ai_strategy(plan: ProductCollectionPlan, payload: Dict[str, Any]) -> ProductCollectionPlan:
    generated_search_terms = normalize_keyword_inputs(payload.get("generated_search_terms") or [])
    dynamic_exclude_keywords = normalize_keyword_inputs(payload.get("dynamic_exclude_keywords") or [])
    evidence_keywords = normalize_keyword_inputs(payload.get("evidence_keywords") or [])
    report_focus = normalize_keyword_inputs(payload.get("report_focus") or [])
    source_policy_notes = normalize_keyword_inputs(payload.get("source_policy_notes") or [])
    rationale = textify(payload.get("rationale", "")).strip()
    if rationale:
        plan.rationale = f"{plan.rationale} 预抓取 Codex 补充判断：{rationale}"
    plan.generated_search_terms = unique_strings([*generated_search_terms, *plan.generated_search_terms])
    plan.search_term_reasons = merge_search_term_reasons(
        plan.search_term_reasons,
        [
            *search_term_reason_rows(plan, generated_search_terms, "codex"),
            *search_term_reason_rows(plan, plan.generated_search_terms, "rule"),
        ],
    )
    plan.dynamic_exclude_keywords = unique_strings([*plan.dynamic_exclude_keywords, *dynamic_exclude_keywords])
    plan.exclude_keywords = unique_strings([*plan.fixed_exclude_keywords, *plan.dynamic_exclude_keywords])
    plan.evidence_keywords = unique_strings([*evidence_keywords, *plan.evidence_keywords])
    plan.report_focus = unique_strings([*plan.report_focus, *report_focus])
    plan.source_policy_notes = unique_strings([*plan.source_policy_notes, *source_policy_notes])
    return plan


def write_pre_crawl_ai_strategy_markdown(path: Path, payload: Dict[str, Any], ok: bool, message: str = "") -> None:
    lines = [
        "# 抓取前 AI 策略",
        "",
        f"**Generated:** {utc_stamp()}",
        f"**Status:** {'ok' if ok else 'fallback'}",
    ]
    if message:
        lines.append(f"**Message:** {message}")
    lines += ["", "## 策略理由", "", textify(payload.get("rationale", "")) or "本轮未获得 Codex 结构化策略，使用规则化预设。"]
    for title, key in [
        ("自动搜索补充词", "generated_search_terms"),
        ("动态排除/降权关键词", "dynamic_exclude_keywords"),
        ("证据关键词", "evidence_keywords"),
        ("报告关注点", "report_focus"),
        ("来源策略补充", "source_policy_notes"),
    ]:
        lines += ["", f"## {title}", ""]
        values = normalize_keyword_inputs(payload.get(key) or [])
        if not values:
            lines.append("无")
        for value in values:
            lines.append(f"- {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pre_crawl_ai_strategy(
    out_dir: Path,
    competitors: Sequence[str],
    plan: ProductCollectionPlan,
    own_product_name: str,
    own_product_positioning: str,
    own_product_context: str,
    codex_command: str,
    model: str,
    timeout: int,
) -> Tuple[ProductCollectionPlan, bool]:
    fallback_payload = {
        "rationale": "Codex 未运行或未返回有效结果，使用规则化品类预设。",
        "generated_search_terms": plan.generated_search_terms,
        "dynamic_exclude_keywords": plan.dynamic_exclude_keywords,
        "evidence_keywords": plan.evidence_keywords,
        "report_focus": plan.report_focus,
        "source_policy_notes": plan.source_policy_notes,
    }
    codex_path = resolve_executable_command(codex_command)
    if not codex_path:
        (out_dir / "pre_crawl_ai_strategy.json").write_text(
            json.dumps({"ok": False, "error": f"Codex command not found: {codex_command}", **fallback_payload}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_pre_crawl_ai_strategy_markdown(out_dir / "pre_crawl_ai_strategy.md", fallback_payload, False, "Codex command not found")
        return plan, False

    schema_path = out_dir / "pre_crawl_ai_strategy_schema.json"
    schema_path.write_text(json.dumps(pre_crawl_ai_strategy_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    output_path = out_dir / "pre_crawl_ai_strategy.json"
    log_path = out_dir / "pre_crawl_ai_strategy_run.log"
    prompt = f"""
你是一个产品经理和竞品情报分析助手。不要联网，不要编造具体竞品事实，只根据下面的我方产品上下文和规则化品类计划，生成本轮抓取前策略。

目标：
1. 生成搜索补充词：用于拼接“竞品名 + 搜索词”，帮助找到官方参数、定价、套餐、文档、接口、质量、客户、更新等核心证据。
2. 生成动态排除/降权关键词：只排除本品类明显混淆或无用的内容，保持保守，不要把可能有价值的官方页面排除。
3. 生成证据关键词：用于判断页面是否命中本轮应采字段。
4. 生成报告关注点和来源策略补充，尤其是本品类特有的官网子页、垂直网站、应用商店、论坛、社媒、视频或官方账号来源。

要求：
- 搜索补充词必须是短语，不要超过 8 个词。
- 不要输出泛词，例如“产品”“AI”“工具”这种单独词。
- 排除词必须保守，优先排登录、交易、破解、下载、促销、同名跨品类误命中。
- 如果我方产品信息不足，要围绕已识别品类生成通用但可执行的策略。
- 来源策略补充必须可追溯：说明推荐去哪些公开来源、这些来源能证明什么、哪些来源只能做线索或待核实。
- 如果需要视频/社媒证据，必须要求保留平台 URL、发布时间、视频时间点或公开截图，不能只写泛泛观点。
- 返回 JSON，必须匹配 schema。

竞品名单：
{json.dumps(list(competitors), ensure_ascii=False)}

我方产品：
- 名称：{own_product_name or "未填写"}
- 定位：{own_product_positioning or "未填写"}
- 背景：{own_product_context or "未填写"}

规则化品类计划：
{json.dumps(json_safe(dataclasses.asdict(plan)), ensure_ascii=False, indent=2)[:30000]}
"""
    cmd = [
        codex_path,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
    ]
    if model:
        cmd += ["-m", model]
    cmd.append("-")
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "CODEX_CI": "1", "PATH": expanded_path_env()},
            cwd=str(out_dir),
        )
    except Exception as exc:
        log_path.write_text(f"Failed to run pre-crawl Codex strategy: {exc}\n", encoding="utf-8")
        output_path.write_text(json.dumps({"ok": False, "error": str(exc), **fallback_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_crawl_ai_strategy_markdown(out_dir / "pre_crawl_ai_strategy.md", fallback_payload, False, str(exc))
        return plan, False

    log_path.write_text(
        "$ " + " ".join(cmd[:-1]) + " -\n\n"
        + "STDOUT:\n" + proc.stdout
        + "\nSTDERR:\n" + proc.stderr
        + f"\nReturn code: {proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0 or not output_path.exists():
        output_path.write_text(json.dumps({"ok": False, "error": f"returncode {proc.returncode}", **fallback_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_crawl_ai_strategy_markdown(out_dir / "pre_crawl_ai_strategy.md", fallback_payload, False, f"returncode {proc.returncode}")
        return plan, False
    try:
        payload = parse_json_payload(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        output_path.write_text(json.dumps({"ok": False, "error": f"parse failed: {exc}", **fallback_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_pre_crawl_ai_strategy_markdown(out_dir / "pre_crawl_ai_strategy.md", fallback_payload, False, f"parse failed: {exc}")
        return plan, False
    output_path.write_text(json.dumps(json_safe({"ok": True, **payload}), ensure_ascii=False, indent=2), encoding="utf-8")
    write_pre_crawl_ai_strategy_markdown(out_dir / "pre_crawl_ai_strategy.md", payload, True)
    return merge_pre_crawl_ai_strategy(plan, payload), True


def write_codex_decisions_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for group_name in ("included_pages", "excluded_pages", "included_images", "excluded_images"):
        for item in payload.get(group_name, []) or []:
            rows.append(
                {
                    "group": group_name,
                    "competitor": item.get("competitor", ""),
                    "decision": item.get("decision", ""),
                    "confidence": item.get("confidence", ""),
                    "url_or_path": item.get("url_or_path", ""),
                    "reason": item.get("reason", ""),
                    "recommended_use": item.get("recommended_use", ""),
                }
            )
    write_csv(
        path,
        rows,
        ["group", "competitor", "decision", "confidence", "url_or_path", "reason", "recommended_use"],
    )


def write_codex_cli_fallback_analysis(out_dir: Path, competitors: Sequence[str], codex_command: str) -> None:
    baseline_path = out_dir / "analysis.md"
    if baseline_path.exists():
        baseline = prepare_analysis_markdown(baseline_path.read_text(encoding="utf-8"), out_dir)
    else:
        baseline = "# 竞品分析报告\n\n本轮没有生成基线分析内容，请查看采集日志和证据文件。"
    notice = (
        "# Codex 分析报告（本地降级版）\n\n"
        f"> Codex CLI 未找到：`{codex_command}`。本轮先使用采集器规则化分析结果生成正式报告，"
        "避免交付物空缺；安装或配置 Codex CLI 后会自动切回 Codex 收录判断和分析。\n\n"
        f"> 本轮竞品：{', '.join(competitors) if competitors else '未填写'}。\n\n"
    )
    (out_dir / "codex_analysis.md").write_text(notice + baseline, encoding="utf-8")
    fallback_payload = {
        "ok": False,
        "fallback": True,
        "error": f"Codex command not found: {codex_command}",
        "analysis_markdown": notice + baseline,
        "included_pages": [],
        "excluded_pages": [],
        "included_images": [],
        "excluded_images": [],
    }
    (out_dir / "codex_review.json").write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_codex_decisions_csv(out_dir / "codex_decisions.csv", fallback_payload)


def run_codex_review(
    out_dir: Path,
    competitors: Sequence[str],
    codex_command: str,
    model: str,
    timeout: int,
) -> bool:
    codex_path = resolve_executable_command(codex_command)
    log_path = out_dir / "codex_run.log"
    if not codex_path:
        write_codex_cli_fallback_analysis(out_dir, competitors, codex_command)
        log_path.write_text(
            f"Codex command not found: {codex_command}\n"
            "Fallback analysis was written from the local rule-generated report.\n",
            encoding="utf-8",
        )
        return True

    schema_path = out_dir / "codex_review_schema.json"
    schema_path.write_text(json.dumps(codex_review_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    output_path = out_dir / "codex_review.json"
    input_bundle = (out_dir / "codex_input.md").read_text(encoding="utf-8") if (out_dir / "codex_input.md").exists() else ""
    prompt = f"""
You are a senior product manager and competitive intelligence analyst.

Use only the evidence included below and local file references from this run. Do not browse the web.

Task:
1. Decide which crawled pages and search evidence should be included in the final competitive analysis.
2. Decide which images are valuable enough to place in the report.
3. Exclude generic SEO pages, unrelated same-name products, low-signal tutorials, broken/anti-bot pages, and images that are decorative or unrelated.
4. Prefer evidence that reveals real product value: positioning, product capabilities, pricing/packaging, use cases, customer segments, GTM, traction, release notes, screenshots, or UI/UX clues.
5. Treat pages marked as excluded/error by the rule filter as rejected unless there is overwhelming useful evidence in the cleaned excerpt. Never use login screens, account centers, shopping carts, marketplace navigation, Taobao forum navigation, or boilerplate menu text as product evidence.
6. Do not include content just because the competitor name appears. A page/image earns inclusion only if it supports at least one PM decision lens: product capability, target customer/job, pricing/packaging, GTM/channel, traction, positioning/perception, customer pain/review, release/change signal, or strategic direction.
7. Label claims as Fact, Inference, or Assumption, and state confidence when drawing a conclusion.
8. In analysis_markdown, create a Chinese PM-facing report with conclusions first, not a clipping digest. You MUST use exactly the following H2 section framework and preserve the numbering:
{FINAL_REPORT_FRAMEWORK_TEXT}
9. The report must include official parameters, pricing, packaging, limits, target segments, JTBD, GTM, product quality/support, traction, SWOT, cross-competitor matrix, and evidence gaps when evidence exists. If evidence is missing, mark it as Assumption / 信息缺口 instead of inventing.
10. Place selected local images near the relevant competitor or the "7. 视觉与产品证据" section using the provided absolute "Markdown image syntax" when they are genuinely useful evidence.
11. Pages in `问题页面核验清单.md/csv` are not facts yet. Mention them in section 11 as verification candidates and explain what should be checked before inclusion.
12. Include why major evidence was included or excluded.
13. In section 13, analyze our own product direction only from the provided "Own Product Context" plus competitor evidence. If own-product context is missing, clearly say what is missing and do not invent our positioning.
14. Use the "Pre-Crawl Product-Specific Collection Plan" to decide what matters for this category. If an analysis template is loaded, treat its dimensions and report outline as the product-specific reference template inside the fixed H2 framework. For physical products, compare specs/materials/size/color/quality fields when present. For AI/software, compare API/integrations/models/limits/security/deployment fields when present. For autonomous vehicle/Robotaxi products, compare official parameters, city operation, commercialization, vehicle platform, autonomous driving system, safety/compliance, ride experience, complex scenarios, HMI/cabin, operations, public opinion, and release tracking when present. If a planned field is missing, mark it as 信息缺口.
15. Use the rule-generated fact strategy fields in the evidence audit:
   - `primary_evidence_candidate=yes` should be preferred as the cited evidence for a fact group.
   - `pending_verification=yes` cannot be written as Fact; put it in 信息缺口 / 待核实线索 unless independently supported by primary evidence.
   - `fact_group` means multiple URLs may support the same fact; avoid repeating duplicated evidence as separate conclusions.
   - `increment_type` explains what new information a source adds; prioritize sources with concrete pricing, parameters, API limits, security, customer, release, or quality increments.
   - `value_signals`, `value_missing`, and `value_verdict` explain why a source is useful or weak. Do not cite sources with missing traceability as Facts.
   - `gui_review_candidate=yes` means the source may be useful but needs public GUI/video review before it becomes evidence. Video/social conclusions require URL, timestamp, and screenshot or transcript evidence.
   - `ml_include_score`, `ml_exclude_score`, and `ml_verify_later_score` are local learned signals from prior human review. Use them to prioritize borderline evidence, but never let them override rule-protected login, cart, private, paid, captcha, or access-control exclusions.
16. Do not write compact claim prefixes like "- Fact 高置信: ...". Use this multi-line structure for important claims:
   - 结论：...
     - 证据属性：Fact / Inference / Assumption
     - 信心等级：高 / 中高 / 中 / 中低 / 低 / 待验证
     - 依据：...
17. Apply this stop-slop-zh writing pass before returning analysis_markdown:
{STOP_SLOP_ZH_PROMPT}

Competitors: {", ".join(competitors)}

Evidence bundle:
<evidence>
{input_bundle[:90000]}
</evidence>

Return JSON matching the provided schema exactly.
"""

    cmd = [
        codex_path,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
    ]
    if model:
        cmd += ["-m", model]
    cmd.append("-")

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "CODEX_CI": "1", "PATH": expanded_path_env()},
            cwd=str(out_dir),
        )
    except Exception as exc:
        log_path.write_text(f"Failed to run Codex: {exc}\n", encoding="utf-8")
        return False

    log_path.write_text(
        "$ " + " ".join(cmd[:-1]) + " -\n\n"
        + "STDOUT:\n" + proc.stdout
        + "\nSTDERR:\n" + proc.stderr
        + f"\nReturn code: {proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0 or not output_path.exists():
        return False

    try:
        payload = parse_json_payload(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log_path.write_text(log_path.read_text(encoding="utf-8") + f"\nFailed to parse Codex JSON: {exc}\n", encoding="utf-8")
        return False

    analysis_text = payload.get("analysis_markdown") or payload.get("summary") or json.dumps(payload, ensure_ascii=False, indent=2)
    analysis_text = prepare_analysis_markdown(analysis_text, out_dir)
    (out_dir / "codex_analysis.md").write_text(analysis_text, encoding="utf-8")
    write_codex_decisions_csv(out_dir / "codex_decisions.csv", payload)
    return True


def parse_competitors(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    if args.competitors:
        for item in args.competitors:
            values.extend([part.strip() for part in item.split(",") if part.strip()])
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line)
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch collect competitor information and images.")
    parser.add_argument("competitors", nargs="*", help="Competitor names or domains. Comma-separated values are accepted.")
    parser.add_argument("--file", help="Text file with one competitor per line.")
    parser.add_argument("--searxng-url", default=os.getenv("SEARXNG_URL", "http://localhost:8888"), help="SearXNG base URL.")
    parser.add_argument("--proxy-url", default=os.getenv("HARVESTER_PROXY_URL", ""), help="Optional HTTP proxy URL, for example http://127.0.0.1:7897.")
    parser.add_argument("--out", default="competitor_harvest_out", help="Output directory.")
    parser.add_argument("--per-query", type=int, default=8, help="SearXNG results per query.")
    parser.add_argument("--max-pages", type=int, default=8, help="Max pages to crawl per competitor.")
    parser.add_argument("--crawl-concurrency", type=int, default=3, help="Concurrent Crawl4AI page crawls.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for SearXNG.")
    parser.add_argument("--max-discovered-competitors", type=int, default=6, help="When no competitors are provided, collect at most this many discovered competitors.")
    parser.add_argument("--skip-crawl", action="store_true", help="Only search; skip Crawl4AI page crawling.")
    parser.add_argument("--skip-images", action="store_true", help="Skip icrawler keyword image downloads.")
    parser.add_argument("--skip-gui-review", action="store_true", help="Skip automated public snapshot pass for GUI review candidates.")
    parser.add_argument("--gui-review-max", type=int, default=8, help="Max GUI/manual-review candidates to snapshot automatically.")
    parser.add_argument("--disable-browser-gui", action="store_true", help="Use public text/metadata snapshots only; do not launch a browser for GUI review.")
    parser.add_argument("--login-assist", action="store_true", help="Queue login/register pages, reuse one visible browser profile, and capture authorized snapshots after public crawling.")
    parser.add_argument("--login-assist-wait", type=int, default=120, help="Seconds to wait after public page crawling for queued login pages before analysis continues.")
    parser.add_argument("--image-engine", choices=["bing", "baidu", "google"], default="bing", help="icrawler image engine.")
    parser.add_argument("--max-image-downloads", type=int, default=20, help="Downloaded images per competitor via icrawler.")
    parser.add_argument("--image-extra-term", action="append", default=[], help="Extra image search term. Can repeat.")
    parser.add_argument("--no-cn", action="store_true", help="Disable extra Chinese queries.")
    parser.add_argument("--codex-review", action="store_true", help="Run local Codex CLI after crawling to decide inclusion and generate final analysis.")
    parser.add_argument("--require-codex-review", action="store_true", help="Fail the run if Codex review cannot generate the final analysis.")
    parser.add_argument("--codex-command", default=os.getenv("CODEX_COMMAND", "codex"), help="Codex CLI command path/name.")
    parser.add_argument("--codex-model", default=os.getenv("CODEX_MODEL", ""), help="Optional Codex model name.")
    parser.add_argument("--codex-timeout", type=int, default=int(os.getenv("CODEX_TIMEOUT", "900")), help="Codex review timeout in seconds.")
    parser.add_argument("--skip-pre-crawl-ai", action="store_true", help="Skip the pre-crawl Codex strategy generation step.")
    parser.add_argument("--pre-crawl-ai-timeout", type=int, default=int(os.getenv("PRE_CRAWL_AI_TIMEOUT", "180")), help="Pre-crawl Codex strategy timeout in seconds.")
    parser.add_argument("--own-product-name", default="", help="Optional own product name for section 13.")
    parser.add_argument("--own-product-positioning", default="", help="Optional own product positioning for section 13.")
    parser.add_argument("--own-product-context", default="", help="Optional own product context, goals, or constraints for section 13.")
    parser.add_argument("--manual-search-term", action="append", default=[], help="Extra product-specific search term. Can repeat.")
    parser.add_argument("--manual-include-keyword", action="append", default=[], help="Keyword that should boost inclusion/PM value. Can repeat.")
    parser.add_argument("--manual-exclude-keyword", action="append", default=[], help="Keyword that should hard-exclude matching sources. Can repeat.")
    parser.add_argument(
        "--ml-model",
        default=os.getenv("HARVESTER_ML_MODEL") or os.getenv("ML_FILTER_MODEL") or str(DEFAULT_FILTER_MODEL_PATH),
        help="Local trained filter model path, default models/filter_model.pt.",
    )
    parser.add_argument("--disable-ml-filter", action="store_true", help="Disable local trained filter model even when a model file exists.")
    parser.add_argument("--ml-auto-include-threshold", type=float, default=float(os.getenv("HARVESTER_ML_AUTO_INCLUDE", "0.75")), help="ML include score needed to promote a weak source.")
    parser.add_argument("--ml-auto-exclude-threshold", type=float, default=float(os.getenv("HARVESTER_ML_AUTO_EXCLUDE", "0.80")), help="ML exclude score needed to demote a weak source.")
    parser.add_argument("--search-cards-dir", default=os.getenv("HARVESTER_SEARCH_CARDS_DIR", str(DEFAULT_SEARCH_CARDS_DIR)), help="Directory with learned product search cards.")
    parser.add_argument("--disable-search-cards", action="store_true", help="Disable learned search cards even when card files exist.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_codex_review:
        args.codex_review = True
    competitors = parse_competitors(args)
    has_own_product_context = bool(args.own_product_name or args.own_product_positioning or args.own_product_context)
    if not competitors and not has_own_product_context:
        print("Please provide competitors, or provide own product name/positioning/context for automatic competitor discovery.", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_search_terms = normalize_keyword_inputs(args.manual_search_term)
    manual_include_keywords = normalize_keyword_inputs(args.manual_include_keyword)
    manual_exclude_keywords = normalize_keyword_inputs(args.manual_exclude_keyword)
    ml_model: Optional[LocalFilterModel] = None
    ml_model_path = Path(args.ml_model).expanduser().resolve() if args.ml_model else DEFAULT_FILTER_MODEL_PATH
    if (
        not args.disable_ml_filter
        and ml_model_path == DEFAULT_FILTER_MODEL_PATH.resolve()
        and not ml_model_path.exists()
    ):
        try:
            bootstrap = bootstrap_filter_model_if_missing(
                ml_model_path,
                [DEFAULT_BOOTSTRAP_LABELS_PATH],
                min_labeled_rows=3,
            )
            if bootstrap.get("created"):
                print(
                    f"[0/5] Local training model bootstrapped: {ml_model_path} "
                    f"({bootstrap.get('training_rows', 0)} seed rows)"
                )
        except Exception as exc:
            print(f"[0/5] Local training model bootstrap skipped: {exc}")
    ml_status = model_status(ml_model_path)
    if args.disable_ml_filter:
        ml_status = {"enabled": False, "path": str(ml_model_path), "message": "disabled by --disable-ml-filter"}
    elif ml_status.get("enabled"):
        ml_model = load_filter_model(ml_model_path)
        print(
            f"[0/5] Local training model loaded: {ml_model_path} "
            f"({ml_model.training_rows} labeled rows)"
        )
    else:
        print(f"[0/5] Local training model: not loaded ({ml_status.get('message', 'not available')})")
    collection_plan = build_product_collection_plan(
        competitors,
        args.own_product_name,
        args.own_product_positioning,
        args.own_product_context,
    )
    competitor_discovery_payload: Dict[str, Any] = write_competitor_discovery(out_dir, [], [], False)
    if not competitors:
        print("[0.25/5] Discovering competitors from own product context ...")
        searxng_ok, searxng_error = check_searxng(args.searxng_url, min(args.timeout, 8), args.proxy_url)
        if not searxng_ok:
            print(f"[error] SearXNG is required for competitor discovery: {searxng_error}", file=sys.stderr)
            write_chinese_export_aliases(out_dir)
            return 3
        candidates, discovery_results = run_competitor_discovery(
            args.searxng_url,
            collection_plan,
            args.own_product_name,
            args.own_product_positioning,
            args.own_product_context,
            args.per_query,
            args.timeout,
            args.proxy_url,
            args.max_discovered_competitors,
        )
        accepted_candidates = [
            candidate
            for candidate in candidates
            if candidate.status == "accepted"
        ][: max(0, args.max_discovered_competitors)]
        competitors = unique_strings(candidate.name for candidate in accepted_candidates)
        competitor_discovery_payload = write_competitor_discovery(
            out_dir,
            candidates,
            discovery_results,
            bool(competitors),
        )
        print(f"      discovered competitors: {len(competitors)}")
        if not competitors:
            print("[error] No competitors could be discovered from the own-product context.", file=sys.stderr)
            write_chinese_export_aliases(out_dir)
            return 2
        collection_plan = build_product_collection_plan(
            competitors,
            args.own_product_name,
            args.own_product_positioning,
            args.own_product_context,
        )
    search_cards_dir = Path(args.search_cards_dir).expanduser()
    if not search_cards_dir.is_absolute():
        search_cards_dir = (APP_DIR / search_cards_dir).resolve()
    search_card_status: Dict[str, Any] = {
        "enabled": False,
        "cards_dir": str(search_cards_dir),
        "loaded_cards": 0,
        "card_keys": [],
    }
    if args.disable_search_cards:
        search_card_status["message"] = "disabled by --disable-search-cards"
    else:
        learned_cards = load_search_cards(
            search_cards_dir,
            product_category=collection_plan.category,
            product_type_key=collection_plan.category,
        )
        if learned_cards:
            collection_plan = apply_search_cards_to_collection_plan(collection_plan, learned_cards)
            search_card_status.update(
                {
                    "enabled": True,
                    "loaded_cards": len(learned_cards),
                    "card_keys": [textify(card.get("product_type_key")) for card in learned_cards],
                }
            )
            print(f"[0/5] Search cards loaded: {len(learned_cards)} from {search_cards_dir}")
        else:
            search_card_status["message"] = "no matching search card"
    pre_crawl_ai_ok = False
    if args.codex_review and not args.skip_pre_crawl_ai:
        print("[0.5/5] Running pre-crawl Codex strategy generation ...")
        collection_plan, pre_crawl_ai_ok = run_pre_crawl_ai_strategy(
            out_dir,
            competitors,
            collection_plan,
            args.own_product_name,
            args.own_product_positioning,
            args.own_product_context,
            args.codex_command,
            args.codex_model,
            args.pre_crawl_ai_timeout,
        )
        print(f"      pre-crawl Codex strategy: {'ok' if pre_crawl_ai_ok else 'fallback'}")
    else:
        write_pre_crawl_ai_strategy_markdown(
            out_dir / "pre_crawl_ai_strategy.md",
            {
                "rationale": "未开启 Codex 收录分析或显式跳过预抓取 AI 策略，使用规则化品类预设。",
                "generated_search_terms": collection_plan.generated_search_terms,
                "dynamic_exclude_keywords": collection_plan.dynamic_exclude_keywords,
                "evidence_keywords": collection_plan.evidence_keywords,
                "report_focus": collection_plan.report_focus,
                "source_policy_notes": collection_plan.source_policy_notes,
            },
            False,
            "pre-crawl Codex strategy skipped",
        )
        (out_dir / "pre_crawl_ai_strategy.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "skipped": True,
                    "generated_search_terms": collection_plan.generated_search_terms,
                    "dynamic_exclude_keywords": collection_plan.dynamic_exclude_keywords,
                    "evidence_keywords": collection_plan.evidence_keywords,
                    "report_focus": collection_plan.report_focus,
                    "source_policy_notes": collection_plan.source_policy_notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    write_pre_crawl_plan(
        out_dir,
        competitors,
        collection_plan,
        args.own_product_name,
        args.own_product_positioning,
        args.own_product_context,
        manual_search_terms,
        manual_include_keywords,
        manual_exclude_keywords,
    )
    print(f"[0/5] Pre-crawl product analysis: {collection_plan.category_label}")
    print(f"      extra fields: {len(collection_plan.fields)}, extra queries: {len(collection_plan.search_templates)}")

    print(f"[1/5] Searching SearXNG at {args.searxng_url} ...")
    if args.proxy_url:
        print(f"      proxy: {args.proxy_url}")
    fatal_after_export = False
    input_seed_results = seed_results_from_competitor_inputs(competitors)
    searxng_ok, searxng_error = check_searxng(args.searxng_url, min(args.timeout, 8), args.proxy_url)
    if not searxng_ok:
        print(f"[error] SearXNG is not reachable: {searxng_error}", file=sys.stderr)
        web_results, image_results = input_seed_results, []
        print(f"      web results: {len(web_results)} input URL seed(s), image results: 0")
        fatal_after_export = not bool(input_seed_results)
    else:
        categories = searxng_categories(args.searxng_url, min(args.timeout, 8), args.proxy_url)
        general_query_templates = build_search_query_templates(
            collection_plan,
            manual_search_terms=manual_search_terms,
            include_cn=not args.no_cn,
        )
        image_query_templates = unique_strings(
            [
                *DEFAULT_IMAGE_QUERIES,
                *[f"{{name}} {term}" for term in collection_plan.image_terms[:10]],
            ]
        )
        if categories and "images" not in categories:
            image_query_templates = []
            print("      SearXNG images category not enabled; image collection will use icrawler.")
        web_results, image_results = run_searches(
            searxng_url=args.searxng_url,
            competitors=competitors,
            general_queries=general_query_templates,
            image_queries=image_query_templates,
            include_cn=not args.no_cn,
            per_query=args.per_query,
            timeout=args.timeout,
            proxy_url=args.proxy_url,
        )
        web_results = enrich_web_results(competitors, [*web_results, *input_seed_results])
        print(f"      web results: {len(web_results)}, image results: {len(image_results)}")

    pages: List[PageExtract] = []
    urls_to_crawl = choose_urls_to_crawl(
        web_results,
        args.max_pages,
        collection_plan,
        manual_include_keywords,
        manual_exclude_keywords,
        ml_model,
        args.ml_auto_include_threshold,
        args.ml_auto_exclude_threshold,
    )
    evidence_audit_rows = rows_from_evidence_audit(
        web_results,
        urls_to_crawl,
        args.max_pages,
        collection_plan,
        manual_include_keywords,
        manual_exclude_keywords,
        ml_model,
        args.ml_auto_include_threshold,
        args.ml_auto_exclude_threshold,
    )
    gui_review_rows: List[Dict[str, Any]] = []
    login_session: Optional[LoginAssistSession] = None
    try:
        if args.login_assist and not args.skip_gui_review:
            pre_crawl_manual_review_rows = rows_from_manual_review_queue([], evidence_audit_rows)
            pre_crawl_login_rows = [
                row for row in pre_crawl_manual_review_rows
                if row_requires_login_action(row)
            ]
            if pre_crawl_login_rows:
                print(f"[1.5/5] Queuing {len(pre_crawl_login_rows)} login-required page(s) before Crawl4AI ...")
                login_session = LoginAssistSession(
                    out_dir,
                    proxy_url=args.proxy_url,
                    timeout=args.timeout,
                )
                login_session.add_rows(pre_crawl_login_rows)
                pre_crawl_login_required_rows = login_session.queue_rows()
                write_login_required_queue(out_dir / "login_required_queue.md", pre_crawl_login_required_rows)
                write_csv(out_dir / "login_required_queue.csv", pre_crawl_login_required_rows, LOGIN_REQUIRED_FIELDS)
                print(
                    f"      login queue: {len(pre_crawl_login_required_rows)} unique site(s); public crawling continues while you log in."
                )

        if args.skip_crawl:
            print("[2/5] Skipping Crawl4AI.")
        else:
            if not urls_to_crawl:
                print("[2/5] No pages to crawl; skipping Crawl4AI.")
            else:
                print(f"[2/5] Crawling {len(urls_to_crawl)} pages with Crawl4AI ...")
                pages = asyncio.run(
                    crawl_with_crawl4ai(
                        urls_to_crawl,
                        args.crawl_concurrency,
                        args.proxy_url,
                        collection_plan,
                        manual_include_keywords,
                        manual_exclude_keywords,
                    )
                )
                pages_ok = len([page for page in pages if not page.error])
                pages_rejected = len(pages) - pages_ok
                print(f"      crawled pages: {len(pages)} (usable candidates: {pages_ok}, rejected/blocked: {pages_rejected})")

        if args.login_assist and not args.skip_gui_review:
            post_crawl_manual_review_rows = rows_from_manual_review_queue(pages, evidence_audit_rows)
            post_crawl_login_rows = [
                row for row in post_crawl_manual_review_rows
                if row_requires_login_action(row)
            ]
            if post_crawl_login_rows:
                if login_session is None:
                    login_session = LoginAssistSession(
                        out_dir,
                        proxy_url=args.proxy_url,
                        timeout=args.timeout,
                    )
                added_login_sites = login_session.add_rows(post_crawl_login_rows)
                queued_rows = login_session.queue_rows()
                write_login_required_queue(out_dir / "login_required_queue.md", queued_rows)
                write_csv(out_dir / "login_required_queue.csv", queued_rows, LOGIN_REQUIRED_FIELDS)
                print(
                    f"[2.5/5] Login queue ready: {len(queued_rows)} unique site(s), {added_login_sites} newly added after crawl."
                )
                gui_review_rows.extend(login_session.capture_all(wait_seconds=args.login_assist_wait))
                write_csv(out_dir / "gui_review_results.csv", gui_review_rows, GUI_REVIEW_FIELDS)
                write_gui_review_results_markdown(out_dir / "gui_review_results.md", gui_review_rows)
    finally:
        if login_session is not None:
            login_session.close()

    login_snapshot_pages = page_extracts_from_gui_review_rows(gui_review_rows, collection_plan)
    if login_snapshot_pages:
        pages = merge_page_extracts(pages, login_snapshot_pages)
        print(f"      logged-in snapshots added to analysis pages: {len(login_snapshot_pages)}")

    downloaded_images: List[Dict[str, str]] = []
    if args.skip_images:
        print("[3/5] Skipping image downloads.")
    else:
        print("[3/5] Downloading visual evidence ...")
        searxng_downloaded = download_searxng_image_results(
            image_results,
            out_dir,
            max_images_per_competitor=args.max_image_downloads,
            proxy_url=args.proxy_url,
            timeout=args.timeout,
        )
        downloaded_images.extend(searxng_downloaded)
        print(f"      SearXNG downloaded images: {len(searxng_downloaded)}")
        print(f"      keyword images with icrawler ({args.image_engine}) ...")
        extra_terms = unique_strings(
            [
                *(args.image_extra_term or ["product", "screenshot", "UI"]),
                *collection_plan.image_terms,
                *manual_search_terms[:10],
                *manual_include_keywords[:10],
            ]
        )
        icrawler_images = crawl_keyword_images(
            competitors=competitors,
            out_dir=out_dir,
            max_images=args.max_image_downloads,
            engine=args.image_engine,
            extra_terms=extra_terms,
            proxy_url=args.proxy_url,
        )
        downloaded_images.extend(icrawler_images)
        print(f"      icrawler downloaded images: {len(icrawler_images)}")
        print(f"      downloaded images total: {len(downloaded_images)}")

    print("[4/5] Exporting CSV/JSON/Markdown ...")
    manual_review_rows = rows_from_manual_review_queue(pages, evidence_audit_rows)
    processed_gui_keys = {review_queue_key_for(row) for row in gui_review_rows}
    if args.skip_gui_review:
        if not gui_review_rows:
            write_gui_review_results_markdown(out_dir / "gui_review_results.md", gui_review_rows)
            write_csv(
                out_dir / "gui_review_results.csv",
                gui_review_rows,
                GUI_REVIEW_FIELDS,
            )
    else:
        remaining_manual_review_rows = [
            row for row in manual_review_rows
            if review_queue_key_for(row) not in processed_gui_keys
        ]
        if args.login_assist:
            remaining_login_rows = [
                row for row in remaining_manual_review_rows
                if row_requires_login_action(row)
            ]
            remaining_non_login_rows = [
                row for row in remaining_manual_review_rows
                if not row_requires_login_action(row)
            ]
            if remaining_login_rows:
                pending_login_rows = login_required_queue_rows(remaining_login_rows, gui_review_rows)
                if pending_login_rows:
                    write_login_required_queue(out_dir / "login_required_queue.md", pending_login_rows)
                    write_csv(out_dir / "login_required_queue.csv", pending_login_rows, LOGIN_REQUIRED_FIELDS)
            remaining_manual_review_rows = remaining_non_login_rows
            gui_review_max = args.gui_review_max
        else:
            gui_review_max = args.gui_review_max
        gui_review_rows.extend(
            execute_gui_review_queue(
                remaining_manual_review_rows,
                out_dir,
                max_items=gui_review_max,
                enable_browser=not args.disable_browser_gui,
                proxy_url=args.proxy_url,
                login_assist=args.login_assist,
                login_assist_wait_seconds=args.login_assist_wait,
            )
        )
    gui_snapshot_pages = page_extracts_from_gui_review_rows(gui_review_rows, collection_plan)
    if gui_snapshot_pages:
        pages = merge_page_extracts(pages, gui_snapshot_pages)
    manual_review_rows = rows_from_manual_review_queue(pages, evidence_audit_rows)
    write_csv(
        out_dir / "gui_review_results.csv",
        gui_review_rows,
        GUI_REVIEW_FIELDS,
    )
    write_gui_review_results_markdown(out_dir / "gui_review_results.md", gui_review_rows)
    image_rows = rows_from_images(image_results, pages, downloaded_images)
    training_review_rows = build_training_review_sample(
        evidence_audit_rows,
        product_category=collection_plan.category,
        product_type_key=collection_plan.category,
        product_type_label=collection_plan.category_label,
        own_product_name=args.own_product_name,
    )
    page_rows = rows_from_pages(pages, collection_plan, evidence_audit_rows)
    competitor_rows = summarize_competitors(competitors, pages, image_rows, web_results)
    login_required_rows = login_required_queue_rows(manual_review_rows, gui_review_rows)
    write_login_required_queue(out_dir / "login_required_queue.md", login_required_rows)
    write_csv(out_dir / "login_required_queue.csv", login_required_rows, LOGIN_REQUIRED_FIELDS)
    problem_review_rows = build_problem_review_rows(
        pages,
        manual_review_rows,
        login_required_rows,
        gui_review_rows,
        evidence_audit_rows,
        training_review_rows,
    )
    write_problem_review_outputs(out_dir, problem_review_rows)
    structured_facts = extract_structured_facts(pages, collection_plan, evidence_audit_rows)
    fact_clusters = cluster_structured_facts(structured_facts)
    write_structured_fact_outputs(out_dir, structured_facts, fact_clusters)
    all_source_rows = rows_from_all_sources(
        web_results=web_results,
        image_results=image_results,
        chosen_urls=urls_to_crawl,
        pages=pages,
        downloaded=downloaded_images,
        evidence_audit_rows=evidence_audit_rows,
        collection_plan=collection_plan,
    )

    write_csv(
        out_dir / "search_results.csv",
        [dataclasses.asdict(r) for r in web_results],
        ["competitor", "category", "query", "title", "url", "snippet", "engine", "score"],
    )
    write_csv(
        out_dir / "evidence_audit.csv",
        evidence_audit_rows,
        [
            "competitor",
            "decision",
            "decision_status",
            "source_kind",
            "page_role",
            "source_policy_tier",
            "gate_result",
            "hard_gate",
            "confidence",
            "pending_verification",
            "verification_reason",
            "fact_type",
            "increment_type",
            "fact_group",
            "primary_evidence_candidate",
            "primary_evidence_reason",
            "value_signals",
            "value_missing",
            "value_verdict",
            "gui_review_candidate",
            "gui_review_value_reason",
            "ml_label",
            "ml_include_score",
            "ml_exclude_score",
            "ml_verify_later_score",
            "ml_confidence",
            "ml_adjustment",
            "ml_reason",
            "ml_model_version",
            "relevance_score",
            "evidence_score",
            "pm_value_score",
            "traceability_score",
            "category_fit_score",
            "matched_fields",
            "matched_include_keywords",
            "matched_exclude_keywords",
            "rejection_code",
            "selected",
            "rank",
            "score",
            "domain",
            "title",
            "url",
            "query",
            "engine",
            "reason",
            "selection_note",
            "per_competitor_crawl_budget",
            "selected_count_so_far",
        ],
    )
    write_csv(
        out_dir / "competitors.csv",
        competitor_rows,
        [
            "competitor",
            "pages_crawled",
            "pages_ok",
            "images_found",
            "positioning",
            "pricing_signal",
            "feature_signal",
            "top_url",
        ],
    )
    write_csv(
        out_dir / "pages.csv",
        page_rows,
        [
            "competitor",
            "url",
            "title",
            "source_policy_tier",
            "page_role",
            "pending_verification",
            "verification_reason",
            "fact_type",
            "increment_type",
            "fact_group",
            "primary_evidence_candidate",
            "primary_evidence_reason",
            "value_signals",
            "value_missing",
            "value_verdict",
            "gui_review_candidate",
            "gui_review_value_reason",
            "ml_label",
            "ml_include_score",
            "ml_exclude_score",
            "ml_verify_later_score",
            "ml_confidence",
            "ml_adjustment",
            "ml_reason",
            "ml_model_version",
            "positioning",
            "pricing",
            "features",
            "customers",
            *collection_plan_field_keys(collection_plan),
            "image_count",
            "link_count",
            "text_excerpt",
            "error",
        ],
    )
    write_csv(
        out_dir / "manual_review_queue.csv",
        manual_review_rows,
        MANUAL_REVIEW_FIELDS,
    )
    write_csv(
        out_dir / "training_review_sample.csv",
        training_review_rows,
        [
            "competitor",
            "product_category",
            "product_type_key",
            "product_type_label",
            "own_product_name",
            "search_card_candidate",
            "url",
            "title",
            "domain",
            "decision_status",
            "source_kind",
            "page_role",
            "source_policy_tier",
            "hard_gate",
            "pending_verification",
            "verification_reason",
            "fact_type",
            "increment_type",
            "fact_group",
            "primary_evidence_candidate",
            "ml_label",
            "ml_include_score",
            "ml_exclude_score",
            "ml_verify_later_score",
            "ml_confidence",
            "ml_adjustment",
            "ml_reason",
            "matched_fields",
            "matched_include_keywords",
            "matched_exclude_keywords",
            "reason",
            "suggested_label",
            "human_label",
            "human_reason",
            "use_as_primary_evidence",
            "reviewed_by",
            "reviewed_at",
            "review_hint",
        ],
    )
    write_csv(
        out_dir / "images.csv",
        image_rows,
        ["competitor", "source", "image_url", "page_url", "title", "query", "local_file"],
    )
    write_csv(
        out_dir / "all_sources.csv",
        all_source_rows,
        [
            "competitor",
            "source_stage",
            "source_type",
            "source_status",
            "selected_for_crawl",
            "official_or_public",
            "source_policy_tier",
            "page_role",
            "pending_verification",
            "verification_reason",
            "fact_type",
            "increment_type",
            "fact_group",
            "primary_evidence_candidate",
            "primary_evidence_reason",
            "value_signals",
            "value_missing",
            "value_verdict",
            "gui_review_candidate",
            "gui_review_value_reason",
            "ml_label",
            "ml_include_score",
            "ml_exclude_score",
            "ml_verify_later_score",
            "ml_confidence",
            "ml_adjustment",
            "ml_reason",
            "ml_model_version",
            "title",
            "url_or_path",
            "domain",
            "query",
            "engine",
            "score",
            "content_preview",
            "reason",
        ],
    )
    write_unfiltered_collection(
        out_dir / "unfiltered_collection.md",
        competitors,
        web_results,
        image_results,
        pages,
        image_rows,
        args.searxng_url,
    )
    write_filtered_collection(
        out_dir / "filtered_collection.md",
        competitors,
        pages,
        image_rows,
        evidence_audit_rows,
        collection_plan,
    )
    write_manual_review_queue(out_dir / "manual_review_queue.md", manual_review_rows)
    write_training_review_sample(out_dir / "training_review_sample.md", training_review_rows)
    print(f"      manual review candidates: {len(manual_review_rows)}")
    print(f"      login-required candidates: {len(login_required_rows)}")
    print(f"      training review sample: {len(training_review_rows)}")
    print(f"      automated GUI review snapshots: {len(gui_review_rows)}")
    print(f"      problem review rows: {len(problem_review_rows)}")
    print(f"      structured facts: {len(structured_facts)}, fact clusters: {len(fact_clusters)}")
    write_report(out_dir / "report.md", competitors, competitor_rows, pages, image_rows, args.searxng_url)
    write_analysis_report(
        out_dir / "analysis.md",
        competitors,
        competitor_rows,
        pages,
        image_rows,
        args.searxng_url,
        args.own_product_name,
        args.own_product_positioning,
        args.own_product_context,
        collection_plan,
        structured_facts,
        fact_clusters,
        gui_review_rows,
        competitor_discovery_payload,
        manual_review_rows,
    )
    write_methodology(
        out_dir / "collection_principles.md",
        args.max_pages,
        collection_plan,
        manual_include_keywords,
        manual_exclude_keywords,
        manual_search_terms,
    )
    write_methodology(
        out_dir / "methodology.md",
        args.max_pages,
        collection_plan,
        manual_include_keywords,
        manual_exclude_keywords,
        manual_search_terms,
    )
    write_anti_bot_strategy_doc(out_dir / "anti_bot_strategy.md", pages, manual_review_rows, gui_review_rows)

    raw = {
        "generated_at": utc_stamp(),
        "searxng_url": args.searxng_url,
        "proxy_url": args.proxy_url,
        "competitors": competitors,
        "web_results": [dataclasses.asdict(r) for r in web_results],
        "image_results": [dataclasses.asdict(r) for r in image_results],
        "pages": [dataclasses.asdict(p) for p in pages],
        "downloaded_images": downloaded_images,
        "evidence_audit": evidence_audit_rows,
        "manual_review_queue": manual_review_rows,
        "login_required_queue": login_required_rows,
        "gui_review_results": gui_review_rows,
        "problem_review_queue": problem_review_rows,
        "structured_facts": structured_facts,
        "fact_clusters": fact_clusters,
        "training_review_sample": training_review_rows,
        "all_sources": all_source_rows,
        "competitor_discovery": competitor_discovery_payload,
        "own_product": {
            "name": args.own_product_name,
            "positioning": args.own_product_positioning,
            "context": args.own_product_context,
        },
        "pre_crawl_collection_plan": dataclasses.asdict(collection_plan),
        "pre_crawl_ai_strategy_ok": pre_crawl_ai_ok,
        "manual_screening_overrides": {
            "manual_search_terms": manual_search_terms,
            "manual_include_keywords": manual_include_keywords,
            "manual_exclude_keywords": manual_exclude_keywords,
        },
        "local_ml_filter": {
            **ml_status,
            "auto_include_threshold": args.ml_auto_include_threshold,
            "auto_exclude_threshold": args.ml_auto_exclude_threshold,
        },
        "search_cards": search_card_status,
    }
    (out_dir / "raw.json").write_text(json.dumps(json_safe(raw), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ml_filter_status.json").write_text(
        json.dumps(
            json_safe(
                {
                    **ml_status,
                    "auto_include_threshold": args.ml_auto_include_threshold,
                    "auto_exclude_threshold": args.ml_auto_exclude_threshold,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_codex_input_bundle(
        out_dir / "codex_input.md",
        competitors,
        competitor_rows,
        web_results,
        pages,
        image_rows,
        evidence_audit_rows,
        manual_review_rows,
        args.own_product_name,
        args.own_product_positioning,
        args.own_product_context,
        collection_plan,
        structured_facts,
        fact_clusters,
        gui_review_rows,
        competitor_discovery_payload,
    )

    codex_ok = False
    if args.codex_review and not fatal_after_export:
        print("[4.5/5] Running Codex AI inclusion review ...")
        codex_ok = run_codex_review(
            out_dir=out_dir,
            competitors=competitors,
            codex_command=args.codex_command,
            model=args.codex_model,
            timeout=args.codex_timeout,
        )
        print(f"      Codex review: {'ok' if codex_ok else 'skipped/failed'}")

    baseline_analysis = out_dir / "analysis.md"
    codex_analysis = out_dir / "codex_analysis.md"
    final_analysis = out_dir / "final_analysis.md"
    if codex_ok and codex_analysis.exists():
        text = prepare_analysis_markdown(codex_analysis.read_text(encoding="utf-8"), out_dir)
        final_analysis.write_text(text, encoding="utf-8")
        ensure_report_has_images(final_analysis, competitors, image_rows)
    elif args.require_codex_review:
        final_analysis.write_text(
            "# Codex 分析未生成\n\n"
            "本次任务要求最终报告必须经过 Codex AI 收录判断和分析，但 Codex 步骤失败或未返回有效结构化结果。"
            "请查看任务状态或内部 `codex_run.log` 后重试。\n",
            encoding="utf-8",
        )
        print("[error] Codex review is required but did not generate a valid final analysis.", file=sys.stderr)
        print("[5/5] Done with blocking errors.")
        print(f"Output: {out_dir}")
        return 4
    elif baseline_analysis.exists():
        text = prepare_analysis_markdown(baseline_analysis.read_text(encoding="utf-8"), out_dir)
        final_analysis.write_text(text, encoding="utf-8")
        ensure_report_has_images(final_analysis, competitors, image_rows)

    write_embedded_markdown_copy(final_analysis, out_dir / "final_analysis_embedded.md")
    write_screening_strategy_doc(out_dir)
    write_chinese_export_aliases(out_dir)
    slim_output_directory(out_dir, keep_run_log=False)

    if fatal_after_export:
        print("[5/5] Done with blocking errors.")
        print(f"Output: {out_dir}")
        return 3

    print("[5/5] Done.")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
