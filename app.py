#!/usr/bin/env python3
"""
Local Web UI for Competitor Intel Harvester.

Run:
    python3 app.py
Open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import csv
import hashlib
import html
import signal
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from filter_training import (
    bootstrap_filter_model_if_missing,
    build_training_rows,
    model_status,
    normalize_label,
    save_filter_model,
    save_model_checkpoint_pt,
    save_model_weights,
    train_filter_model,
)
from search_cards import build_search_cards, write_search_cards


APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
SCRIPT_PATH = APP_DIR / "competitor_harvester.py"
VENV_PYTHON = APP_DIR / ".venv" / "bin" / "python"
DEFAULT_FILTER_MODEL_PATH = APP_DIR / "models" / "filter_model.pt"
DEFAULT_BOOTSTRAP_LABELS_PATH = APP_DIR / "training_data" / "bootstrap_labels.csv"
DEFAULT_REVIEW_LABELS_PATH = APP_DIR / "training_data" / "review_labels.csv"
DEFAULT_SEARCH_CARDS_DIR = APP_DIR / "search_cards"
INTERNAL_OUTPUT_DIR_NAME = "_internal"
EXTRA_BIN_DIRS = [
    Path.home() / ".local" / "bin",
    Path.home() / ".codex" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
]


def worker_python() -> str:
    return str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))


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


@dataclass
class Job:
    id: str
    status: str
    command: List[str]
    out_dir: Path
    proxy_url: str = ""
    experiment_minutes: int = 0
    process_pid: Optional[int] = None
    terminate_requested: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    logs: List[str] = field(default_factory=list)


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

PRIMARY_ARTIFACTS = [
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
]

ROOT_OUTPUT_FILES = set(PRIMARY_ARTIFACTS) | {"downloaded_images", "gui_review_snapshots"}

FAILURE_ARTIFACTS = [
    "运行日志.log",
    "Codex运行日志.log",
    "run.log",
    "codex_run.log",
]

FEEDBACK_CATEGORY_ORDER = [
    "失败/反爬/超时",
    "待登录",
    "内容不好放弃",
    "主要内容来源",
    "待核实线索",
    "模型边界样本",
]

FEEDBACK_REVIEW_FIELDS = [
    "source_job_id",
    "feedback_category",
    "competitor",
    "title",
    "url",
    "domain",
    "snippet",
    "reason",
    "model_conclusion",
    "conclusion_label",
    "counter_human_label",
    "source_file",
    "source_queue",
    "problem_type",
    "decision_status",
    "hard_gate",
    "source_kind",
    "page_role",
    "source_policy_tier",
    "pending_verification",
    "verification_reason",
    "fact_type",
    "increment_type",
    "fact_group",
    "ml_confidence",
    "ml_include_score",
    "ml_exclude_score",
    "ml_verify_later_score",
    "feedback_judgement",
    "human_label",
    "human_reason",
    "feedback_status",
    "reviewed_at",
]

CHINESE_EXPORT_ALIASES = {
    "all_sources.csv": "所有采集来源.csv",
    "unfiltered_collection.md": "未经筛选的采集内容.md",
    "filtered_collection.md": "筛选后的采集内容.md",
    "final_analysis_embedded.md": "竞品分析报告_图片内嵌版.md",
    "collection_principles.md": "采集原则和筛选原则.md",
    "screening_strategy.md": "收录过滤策略设计.md",
    "ml_filter_status.json": "本地筛选模型状态.json",
    "filter_model.pt": "本地筛选模型.pt",
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
    "competitor_discovery.json": "自动竞品发现.json",
    "gui_review_results.md": "GUI自动复核结果.md",
    "gui_review_results.csv": "GUI自动复核结果.csv",
    "structured_facts.csv": "结构化事实.csv",
    "structured_facts.json": "结构化事实.json",
    "fact_clusters.md": "事实聚类.md",
    "fact_clusters.csv": "事实聚类.csv",
    "fact_clusters.json": "事实聚类.json",
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

INTERNAL_DOWNLOADABLE_ARTIFACTS = {
    "run.log",
    "all_sources.csv",
    "unfiltered_collection.md",
    "filtered_collection.md",
    "final_analysis.md",
    "final_analysis_embedded.md",
    "problem_pages_review.md",
    "problem_pages_review.csv",
    "collection_principles.md",
    "screening_strategy.md",
    "ml_filter_status.json",
    "filter_model.pt",
    "filter_weights.json",
    "codex_analysis.md",
    "codex_decisions.csv",
    "codex_review.json",
    "codex_input.md",
    "pre_crawl_ai_strategy.json",
    "pre_crawl_ai_strategy.md",
    "pre_crawl_plan.md",
    "pre_crawl_plan.json",
    "codex_run.log",
    "manual_review_queue.md",
    "manual_review_queue.csv",
    "login_required_queue.md",
    "login_required_queue.csv",
    "training_review_sample.md",
    "training_review_sample.csv",
    "competitor_discovery.md",
    "competitor_discovery.csv",
    "competitor_discovery.json",
    "gui_review_results.md",
    "gui_review_results.csv",
    "structured_facts.csv",
    "structured_facts.json",
    "fact_clusters.md",
    "fact_clusters.csv",
    "fact_clusters.json",
    "anti_bot_strategy.md",
    "report.md",
    "analysis.md",
    "methodology.md",
    "competitors.csv",
    "pages.csv",
    "images.csv",
    "search_results.csv",
    "evidence_audit.csv",
    "raw.json",
}

ALL_DOWNLOADABLE_ARTIFACTS = INTERNAL_DOWNLOADABLE_ARTIFACTS | set(CHINESE_EXPORT_ALIASES.values())
ALL_DOWNLOADABLE_ARTIFACTS |= {"实验计时记录.md", "实验计时记录.json"}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>竞品情报采集器</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d2430;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #1264a3;
      --accent-2: #0b7a5c;
      --danger: #b42318;
      --soft: #eef5fb;
      --shadow: 0 12px 30px rgba(16, 24, 40, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 19px; letter-spacing: 0; }
    .shell {
      display: grid;
      grid-template-columns: minmax(360px, 430px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .panel-head h2 { margin: 0; font-size: 16px; }
    form { padding: 18px; display: grid; gap: 16px; }
    label { display: grid; gap: 7px; font-size: 13px; color: var(--muted); font-weight: 600; }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 40px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: #fff;
      outline: none;
    }
    textarea { min-height: 128px; resize: vertical; line-height: 1.45; }
    input:focus, textarea:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(18, 100, 163, .12);
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      font-weight: 500;
    }
    .check input { width: 16px; min-height: 16px; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      min-height: 42px;
      padding: 0 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { background: #344054; }
    button.danger { background: var(--danger); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
      font-weight: 700;
      background: #f2f4f7;
      color: #344054;
    }
    .status.running { background: #e8f3fb; color: var(--accent); }
    .status.paused { background: #fff7e6; color: #9a5b00; }
    .status.done { background: #e7f6ef; color: var(--accent-2); }
    .status.stopping { background: #fff1f3; color: var(--danger); }
    .status.failed { background: #fee4e2; color: var(--danger); }
    .timer {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
      font-weight: 700;
    }
    .timer.over { color: var(--danger); background: #fff7f6; border-color: #fecdca; }
    .content { padding: 18px; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 14px;
    }
    .artifact-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .artifact {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      text-decoration: none;
      color: var(--ink);
      background: #fff;
    }
    .artifact span { color: var(--muted); font-size: 12px; }
    .login-review-banner {
      border: 1px solid #fdb022;
      border-left: 4px solid #f79009;
      border-radius: 8px;
      background: #fffaeb;
      color: #7a2e0e;
      padding: 12px;
      margin-bottom: 14px;
      display: grid;
      gap: 10px;
      font-size: 13px;
      line-height: 1.45;
    }
    .login-review-banner[hidden] { display: none; }
    .login-review-title { font-weight: 800; color: #7a2e0e; }
    .login-review-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .login-review-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 9px;
      border: 1px solid #fdb022;
      border-radius: 6px;
      background: #fff;
    }
    .login-review-meta {
      display: grid;
      gap: 3px;
      min-width: 0;
    }
    .login-review-meta strong,
    .login-review-meta span {
      overflow-wrap: anywhere;
    }
    .login-review-meta span {
      color: #9a5b00;
      font-size: 12px;
    }
    .login-review-buttons {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .login-review-actions a,
    .login-review-actions button {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      border: 1px solid #fdb022;
      color: #7a2e0e;
      text-decoration: none;
      font-weight: 700;
    }
    .login-review-actions button {
      color: #7a2e0e;
      cursor: pointer;
    }
    .login-review-actions button.skip {
      border-color: #fedf89;
      color: #93370d;
    }
    pre {
      min-height: 440px;
      max-height: calc(100vh - 310px);
      overflow: auto;
      margin: 0;
      padding: 14px;
      border-radius: 8px;
      background: #111827;
      color: #d1fae5;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 22px;
      color: var(--muted);
      background: #fbfcfd;
    }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.4; }
    .training-panel {
      border-top: 1px solid var(--line);
      padding: 18px;
      display: grid;
      gap: 12px;
    }
    .training-panel h3 { margin: 0; font-size: 15px; }
    .training-output {
      min-height: 120px;
      max-height: 220px;
    }
    .review-modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 30;
      display: grid;
      place-items: center;
      padding: 20px;
      background: rgba(15, 23, 42, .35);
    }
    .review-modal-backdrop[hidden] { display: none; }
    .review-modal {
      width: min(760px, 100%);
      max-height: min(720px, calc(100vh - 40px));
      overflow: auto;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      box-shadow: 0 24px 80px rgba(15, 23, 42, .22);
    }
    .review-modal-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    .review-modal-head h3 { margin: 0; font-size: 16px; }
    .review-modal-body {
      padding: 16px 18px;
      display: grid;
      gap: 12px;
      font-size: 13px;
      line-height: 1.5;
    }
    .review-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 6px;
      background: #fbfcfd;
    }
    .review-item strong { color: var(--ink); }
    @media (max-width: 920px) {
      .shell { grid-template-columns: 1fr; padding: 12px; }
      header { padding: 0 16px; }
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>竞品情报采集器</h1>
    <div class="hint">SearXNG → Crawl4AI → icrawler → Codex → 导出</div>
  </header>
  <main class="shell">
    <section>
      <div class="panel-head">
        <h2>采集配置</h2>
      </div>
      <form id="runForm">
        <label>
          竞品名单
          <textarea name="competitors" placeholder="每行一个，例如：&#10;Gamma&#10;Beautiful.ai&#10;Canva AI"></textarea>
        </label>
        <label>
          我方产品名称
          <input name="own_product_name" value="" placeholder="可选，例如：竞品情报采集器">
        </label>
        <label>
          我方产品定位
          <textarea name="own_product_positioning" placeholder="可选，例如：面向产品经理的一站式竞品情报采集、筛选与分析工具"></textarea>
        </label>
        <label>
          我方补充背景
          <textarea name="own_product_context" placeholder="可选，例如：目标用户、当前阶段、核心差异化、希望验证的问题"></textarea>
        </label>
        <label>
          人工重点关注词
          <textarea name="manual_include_keywords" placeholder="可选。默认搜索词由系统在抓取前自动生成；这里只填写本轮特别想提高优先级的证据，例如：&#10;MIPS&#10;CE EN1077&#10;enterprise deployment"></textarea>
        </label>
        <label>
          SearXNG 地址
          <input name="searxng_url" value="http://localhost:8888">
        </label>
        <label>
          代理地址
          <input name="proxy_url" value="" placeholder="可选，例如：http://127.0.0.1:7897">
        </label>
        <div class="grid-3">
          <label>
            每词结果
            <input name="per_query" type="number" min="1" max="100" value="20">
          </label>
          <label>
            每竞品抓页
            <input name="max_pages" type="number" min="0" max="100" value="30">
          </label>
          <label>
            抓取并发
            <input name="crawl_concurrency" type="number" min="1" max="8" value="3">
          </label>
        </div>
        <label>
          自动发现竞品上限
          <input name="max_discovered_competitors" type="number" min="1" max="20" value="6">
        </label>
        <label>
          登录页等待时间
          <input name="login_assist_wait" type="number" min="10" max="600" value="120">
        </label>
        <div class="grid-2">
          <label>
            图片引擎
            <select name="image_engine">
              <option value="bing">Bing</option>
              <option value="baidu">Baidu</option>
              <option value="google">Google</option>
            </select>
          </label>
          <label>
            每竞品下载图片
            <input name="max_image_downloads" type="number" min="0" max="200" value="40">
          </label>
        </div>
        <label>
          图片补充关键词
          <input name="image_terms" value="product, screenshot, UI">
        </label>
        <label>
          Codex 模型
          <input name="codex_model" value="" placeholder="可选，留空使用 Codex 默认模型">
        </label>
        <label>
          本地筛选模型路径
          <input name="ml_model_path" value="models/filter_model.pt" placeholder="可选，默认 models/filter_model.pt">
        </label>
        <label>
          实验限时
          <select name="experiment_minutes">
            <option value="0">不设限时，只记录全流程</option>
            <option value="15">15 分钟</option>
            <option value="30">30 分钟</option>
          </select>
        </label>
        <div class="checks">
          <label class="check"><input name="broad_crawl" type="checkbox" checked> 宽采集</label>
          <label class="check"><input name="codex_review" type="checkbox" checked> Codex AI 收录分析</label>
          <label class="check"><input name="use_ml_filter" type="checkbox" checked> 使用本地训练模型</label>
          <label class="check"><input name="login_assist" type="checkbox" checked> 登录页集中队列与登录态复用</label>
          <label class="check"><input name="skip_gui_review" type="checkbox"> 跳过公开快照复核</label>
          <label class="check"><input name="skip_crawl" type="checkbox"> 跳过网页抓取</label>
          <label class="check"><input name="skip_images" type="checkbox"> 跳过图片下载</label>
          <label class="check"><input name="no_cn" type="checkbox"> 不追加中文搜索</label>
        </div>
        <button id="runBtn" type="submit">开始采集</button>
        <div class="hint">首次运行 Crawl4AI 可能较慢；如果 SearXNG 禁用了 JSON，需要先在 SearXNG 配置里开启。</div>
      </form>
      <div class="training-panel">
        <h3>本地模型训练</h3>
        <label>
          人工标签 CSV
          <input id="labelsPath" value="training_data/review_labels.csv">
        </label>
        <div class="grid-2">
          <label>
            最少标注数
            <input id="minLabels" type="number" min="1" value="10">
          </label>
          <label>
            模型输出路径
            <input id="modelOutPath" value="models/filter_model.pt">
          </label>
          <label>
            搜索卡片目录
            <input id="cardsDirPath" value="search_cards">
          </label>
        </div>
        <div class="toolbar">
          <button id="modelStatusBtn" class="secondary" type="button">模型状态</button>
          <button id="trainModelBtn" type="button">训练筛选模型</button>
        </div>
        <div id="modelStatusText" class="hint">默认读取 training_data/review_labels.csv，训练后会生成 .pt 本地模型、权重摘要和同品类搜索卡片。</div>
        <pre id="trainingLogs" class="training-output" hidden></pre>
      </div>
    </section>
    <section>
      <div class="panel-head">
        <h2>运行状态</h2>
        <div id="status" class="status">待开始</div>
      </div>
      <div class="content">
        <div class="toolbar">
          <button id="refreshBtn" class="secondary" type="button">刷新</button>
          <button id="checkBtn" class="secondary" type="button">环境检查</button>
          <button id="pauseBtn" class="secondary" type="button" disabled>暂停</button>
          <button id="resumeBtn" class="secondary" type="button" disabled>继续</button>
          <button id="terminateBtn" class="danger" type="button" disabled>终止</button>
          <span id="timerMeta" class="timer">计时未开始</span>
          <span id="jobMeta" class="hint"></span>
        </div>
        <div id="loginReviewBanner" class="login-review-banner" hidden></div>
        <div id="artifacts" class="artifact-grid"></div>
        <div id="empty" class="empty">输入竞品，或只填写我方产品信息后点击开始采集；页面会显示关键进度和正式导出文件。</div>
        <pre id="logs" hidden></pre>
      </div>
    </section>
  </main>
  <div id="reviewModal" class="review-modal-backdrop" hidden>
    <div class="review-modal">
      <div class="review-modal-head">
        <h3>任务结束后的人工核验</h3>
        <button id="closeReviewModalBtn" class="secondary" type="button">关闭</button>
      </div>
      <div id="reviewModalBody" class="review-modal-body"></div>
    </div>
  </div>
  <script>
    const form = document.getElementById('runForm');
    const runBtn = document.getElementById('runBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const checkBtn = document.getElementById('checkBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resumeBtn = document.getElementById('resumeBtn');
    const terminateBtn = document.getElementById('terminateBtn');
    const modelStatusBtn = document.getElementById('modelStatusBtn');
    const trainModelBtn = document.getElementById('trainModelBtn');
    const labelsPathEl = document.getElementById('labelsPath');
    const minLabelsEl = document.getElementById('minLabels');
    const modelOutPathEl = document.getElementById('modelOutPath');
    const cardsDirPathEl = document.getElementById('cardsDirPath');
    const modelStatusTextEl = document.getElementById('modelStatusText');
    const trainingLogsEl = document.getElementById('trainingLogs');
    const statusEl = document.getElementById('status');
    const logsEl = document.getElementById('logs');
    const emptyEl = document.getElementById('empty');
    const artifactsEl = document.getElementById('artifacts');
    const jobMetaEl = document.getElementById('jobMeta');
    const timerMetaEl = document.getElementById('timerMeta');
    const loginReviewBannerEl = document.getElementById('loginReviewBanner');
    const reviewModalEl = document.getElementById('reviewModal');
    const reviewModalBodyEl = document.getElementById('reviewModalBody');
    const closeReviewModalBtn = document.getElementById('closeReviewModalBtn');
    let currentJobId = null;
    let pollTimer = null;
    const shownProblemReviewKeys = new Set();

    function statusClass(status) {
      if (status === 'running' || status === 'queued') return 'status running';
      if (status === 'paused') return 'status paused';
      if (status === 'stopping' || status === 'terminated') return 'status stopping';
      if (status === 'done') return 'status done';
      if (status === 'failed') return 'status failed';
      return 'status';
    }

    function formatDuration(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const hours = String(Math.floor(total / 3600)).padStart(2, '0');
      const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
      const secs = String(total % 60).padStart(2, '0');
      return `${hours}:${minutes}:${secs}`;
    }

    function renderTimer(job) {
      const elapsed = job.elapsed_label || formatDuration(job.elapsed_seconds || 0);
      let text = `已用 ${elapsed}`;
      if (job.timebox_seconds) {
        if (job.timebox_exceeded) {
          const overtime = (job.elapsed_seconds || 0) - job.timebox_seconds;
          text += ` · 已超时 ${formatDuration(overtime)} / ${job.timebox_label}`;
        } else {
          text += ` · 剩余 ${job.remaining_label || '00:00:00'} / ${job.timebox_label}`;
        }
      } else {
        text += ' · 未设限时';
      }
      timerMetaEl.className = job.timebox_exceeded ? 'timer over' : 'timer';
      timerMetaEl.textContent = text;
    }

    function clearChildren(node) {
      while (node.firstChild) node.removeChild(node.firstChild);
    }

    async function sendLoginRequest(job, row, action, button, statusNode) {
      if (!job.id) return;
      const url = row.login_assist_url || row.url || '';
      const endpoint = action === 'skip' ? '/api/login/skip' : '/api/login/open';
      button.disabled = true;
      statusNode.textContent = action === 'skip' ? '正在跳过' : '正在发送登录请求';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job: job.id,
          competitor: row.competitor || '',
          url
        })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        button.disabled = false;
        statusNode.textContent = `操作失败：${data.error || data.message || action}`;
        return;
      }
      statusNode.textContent = action === 'skip' ? '已跳过' : '已请求登录，等待工具浏览器接管';
      fetchJob();
    }

    function renderLoginReviews(job) {
      const rows = job.login_required_reviews || [];
      clearChildren(loginReviewBannerEl);
      if (!rows.length) {
        loginReviewBannerEl.hidden = true;
        return;
      }
      loginReviewBannerEl.hidden = false;
      const title = document.createElement('div');
      title.className = 'login-review-title';
      title.textContent = `发现 ${rows.length} 个需登录/注册页面`;
      const body = document.createElement('div');
      body.textContent = '这些站点已按竞品和域名去重。工具不会主动打开登录网页；只有点击下面的登录按钮，采集进程才会访问该站点。公开页面会继续采集，等待期结束仍未登录的会进入“问题页面核验清单”。';
      const batchActions = document.createElement('div');
      batchActions.className = 'login-review-actions';
      const skipAllBtn = document.createElement('button');
      skipAllBtn.type = 'button';
      skipAllBtn.className = 'skip';
      skipAllBtn.textContent = '全部跳过登录页';
      skipAllBtn.addEventListener('click', async () => {
        skipAllBtn.disabled = true;
        skipAllBtn.textContent = '正在跳过';
        await Promise.all(rows.map(row => fetch('/api/login/skip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job: job.id || '',
            competitor: row.competitor || '',
            url: row.login_assist_url || row.url || ''
          })
        }).catch(() => null)));
        skipAllBtn.textContent = '已跳过';
        fetchJob();
      });
      batchActions.appendChild(skipAllBtn);
      const actions = document.createElement('div');
      actions.className = 'login-review-actions';
      rows.forEach((row, index) => {
        const item = document.createElement('div');
        item.className = 'login-review-item';
        const url = row.login_assist_url || row.url;
        const count = row.queued_url_count ? ` · ${row.queued_url_count} 个URL` : '';
        const meta = document.createElement('div');
        meta.className = 'login-review-meta';
        const label = document.createElement('strong');
        label.textContent = `${index + 1}. ${row.competitor || '未知竞品'} · ${row.domain || '未知域名'}${count}`;
        const detail = document.createElement('span');
        detail.textContent = row.title || url || '无标题';
        const state = document.createElement('span');
        state.textContent = row.login_click_requested === 'yes' ? '已请求登录，等待工具浏览器读取登录态' : '等待操作';
        meta.appendChild(label);
        meta.appendChild(detail);
        meta.appendChild(state);

        const buttons = document.createElement('div');
        buttons.className = 'login-review-buttons';
        const openBtn = document.createElement('button');
        openBtn.type = 'button';
        openBtn.textContent = row.login_click_requested === 'yes' ? '已请求登录' : '登录并继续';
        openBtn.disabled = row.login_click_requested === 'yes';
        openBtn.addEventListener('click', () => sendLoginRequest(job, row, 'open', openBtn, state));
        const skipBtn = document.createElement('button');
        skipBtn.type = 'button';
        skipBtn.className = 'skip';
        skipBtn.textContent = '跳过';
        skipBtn.addEventListener('click', () => sendLoginRequest(job, row, 'skip', skipBtn, state));
        buttons.appendChild(openBtn);
        buttons.appendChild(skipBtn);

        item.appendChild(meta);
        item.appendChild(buttons);
        actions.appendChild(item);
      });
      loginReviewBannerEl.appendChild(title);
      loginReviewBannerEl.appendChild(body);
      loginReviewBannerEl.appendChild(batchActions);
      loginReviewBannerEl.appendChild(actions);
    }

    function renderProblemReviews(job) {
      const rows = job.problem_reviews || [];
      if (!rows.length || !(job.status === 'done' || job.status === 'failed')) return;
      const key = `${job.id || ''}:${rows.length}`;
      if (shownProblemReviewKeys.has(key)) return;
      shownProblemReviewKeys.add(key);
      clearChildren(reviewModalBodyEl);
      const intro = document.createElement('div');
      intro.textContent = `本轮有 ${rows.length} 个页面或来源需要人工核验。请先看“问题页面核验清单”，按每行的核验要求填写 human_label 和 human_reason；填完后可以直接训练本地 .pt 筛选模型。`;
      reviewModalBodyEl.appendChild(intro);

      const actions = document.createElement('div');
      actions.className = 'login-review-actions';
      [['问题页面核验清单.csv', '下载问题核验表'], ['人工抽样标注表.csv', '下载抽样标注表']].forEach(([file, label]) => {
        const link = document.createElement('a');
        link.href = `/download?job=${encodeURIComponent(job.id)}&file=${encodeURIComponent(file)}`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = label;
        actions.appendChild(link);
      });
      const trainBtn = document.createElement('button');
      trainBtn.type = 'button';
      trainBtn.textContent = '核验后训练模型';
      trainBtn.addEventListener('click', () => trainModelFromUi(true, trainBtn));
      actions.appendChild(trainBtn);
      reviewModalBodyEl.appendChild(actions);

      rows.slice(0, 8).forEach((row, index) => {
        const item = document.createElement('div');
        item.className = 'review-item';
        const title = document.createElement('strong');
        title.textContent = `${index + 1}. ${row.priority || ''} ${row.problem_type || '需核验'} · ${row.competitor || row.domain || ''}`;
        const url = document.createElement('a');
        url.href = row.url;
        url.target = '_blank';
        url.rel = 'noopener noreferrer';
        url.textContent = row.title || row.url;
        const verify = document.createElement('div');
        verify.textContent = `核验：${row.what_to_verify || '确认是否绑定竞品、是否有信息增量、是否可追溯。'}`;
        const decision = document.createElement('div');
        decision.textContent = `入库标注：${row.data_entry_decision || 'include / verify_later / exclude'}`;
        item.appendChild(title);
        item.appendChild(url);
        item.appendChild(verify);
        item.appendChild(decision);
        reviewModalBodyEl.appendChild(item);
      });
      reviewModalEl.hidden = false;
    }

    async function sendFeedbackReview(job, row, judgement, button, statusNode, peerButton) {
      if (!job.id) return;
      button.disabled = true;
      statusNode.textContent = judgement === 'qualified' ? '正在记录合格反馈' : '正在记录不合格反馈';
      try {
        const res = await fetch('/api/review/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job: job.id,
            competitor: row.competitor || '',
            url: row.url || '',
            feedback_category: row.feedback_category || '',
            judgement
          })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          button.disabled = false;
          if (peerButton) peerButton.disabled = false;
          statusNode.textContent = `反馈失败：${data.error || judgement}`;
          return;
        }
        row.feedback_status = 'ready_for_training';
        row.feedback_judgement = judgement;
        row.human_label = data.human_label || '';
        statusNode.textContent = `已反馈：${judgement === 'qualified' ? '结论合格' : '结论不合格'}；训练标签 ${row.human_label}`;
      } catch (error) {
        button.disabled = false;
        if (peerButton) peerButton.disabled = false;
        statusNode.textContent = `反馈失败：${String(error)}`;
      }
    }

    function renderFeedbackReviews(job) {
      const rows = job.feedback_reviews || [];
      if (!rows.length || !(job.status === 'done' || job.status === 'failed' || job.status === 'terminated')) {
        renderProblemReviews(job);
        return;
      }
      const key = `${job.id || ''}:feedback:${rows.length}`;
      if (shownProblemReviewKeys.has(key)) return;
      shownProblemReviewKeys.add(key);
      clearChildren(reviewModalBodyEl);
      const intro = document.createElement('div');
      intro.textContent = `本轮已从失败、待登录、内容不好放弃、主要内容来源、待核实线索和模型边界样本中分层抽样 ${rows.length} 条。请判断系统结论是否合格，点击后会写入本地训练数据；反馈完成后训练 .pt 模型。`;
      reviewModalBodyEl.appendChild(intro);

      const actions = document.createElement('div');
      actions.className = 'login-review-actions';
      const trainBtn = document.createElement('button');
      trainBtn.type = 'button';
      trainBtn.textContent = '反馈完成并训练模型';
      trainBtn.addEventListener('click', () => trainModelFromUi(true, trainBtn));
      actions.appendChild(trainBtn);
      [['问题页面核验清单.csv', '下载问题核验表'], ['人工抽样标注表.csv', '下载抽样标注表']].forEach(([file, label]) => {
        const link = document.createElement('a');
        link.href = `/download?job=${encodeURIComponent(job.id)}&file=${encodeURIComponent(file)}`;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = label;
        actions.appendChild(link);
      });
      reviewModalBodyEl.appendChild(actions);

      rows.forEach((row, index) => {
        const item = document.createElement('div');
        item.className = 'review-item';
        const title = document.createElement('strong');
        title.textContent = `${index + 1}. ${row.feedback_category || '抽样核验'} · ${row.competitor || row.domain || ''}`;
        const url = document.createElement('a');
        url.href = row.url;
        url.target = '_blank';
        url.rel = 'noopener noreferrer';
        url.textContent = row.title || row.url;
        const conclusion = document.createElement('div');
        conclusion.textContent = `系统结论：${row.model_conclusion || row.conclusion_label || '待判断'}`;
        const reason = document.createElement('div');
        reason.textContent = `依据：${row.reason || row.snippet || row.verification_reason || '无摘要'}`;
        const status = document.createElement('div');
        status.textContent = row.feedback_status === 'ready_for_training'
          ? `已反馈：${row.feedback_judgement || ''}；训练标签 ${row.human_label || ''}`
          : '等待人工评判';
        const buttons = document.createElement('div');
        buttons.className = 'login-review-buttons';
        const goodBtn = document.createElement('button');
        goodBtn.type = 'button';
        goodBtn.textContent = '评判合格';
        const badBtn = document.createElement('button');
        badBtn.type = 'button';
        badBtn.className = 'skip';
        badBtn.textContent = '评判不合格';
        if (row.feedback_status === 'ready_for_training') {
          goodBtn.disabled = true;
          badBtn.disabled = true;
        }
        goodBtn.addEventListener('click', () => {
          badBtn.disabled = true;
          sendFeedbackReview(job, row, 'qualified', goodBtn, status, badBtn);
        });
        badBtn.addEventListener('click', () => {
          goodBtn.disabled = true;
          sendFeedbackReview(job, row, 'unqualified', badBtn, status, goodBtn);
        });
        buttons.appendChild(goodBtn);
        buttons.appendChild(badBtn);
        item.appendChild(title);
        item.appendChild(url);
        item.appendChild(conclusion);
        item.appendChild(reason);
        item.appendChild(status);
        item.appendChild(buttons);
        reviewModalBodyEl.appendChild(item);
      });
      reviewModalEl.hidden = false;
    }

    function renderJob(job) {
      currentJobId = job.id || currentJobId;
      statusEl.className = statusClass(job.status);
      statusEl.textContent = job.status || 'unknown';
      jobMetaEl.textContent = job.id ? `任务 ${job.id} · ${job.out_dir || ''}` : '';
      renderTimer(job);
      logsEl.hidden = false;
      emptyEl.hidden = true;
      logsEl.textContent = (job.logs || []).join('');
      logsEl.scrollTop = logsEl.scrollHeight;
      renderLoginReviews(job);
      renderFeedbackReviews(job);
      artifactsEl.innerHTML = '';
      const files = job.artifacts || [];
      files.forEach(file => {
        const a = document.createElement('a');
        a.className = 'artifact';
        a.href = `/download?job=${encodeURIComponent(job.id)}&file=${encodeURIComponent(file.name)}`;
        a.target = '_blank';
        a.innerHTML = `<strong>${file.name}</strong><span>${file.size_label}</span>`;
        artifactsEl.appendChild(a);
      });
      const active = ['running', 'queued', 'paused', 'stopping'].includes(job.status);
      runBtn.disabled = active;
      pauseBtn.disabled = job.status !== 'running';
      resumeBtn.disabled = job.status !== 'paused';
      terminateBtn.disabled = !['running', 'queued', 'paused', 'stopping'].includes(job.status);
      if (job.status === 'done' || job.status === 'failed' || job.status === 'terminated') {
        clearInterval(pollTimer);
        pollTimer = null;
        runBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        terminateBtn.disabled = true;
      }
    }

    async function fetchJob() {
      if (!currentJobId) return;
      const res = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}`);
      const data = await res.json();
      if (!res.ok) {
        clearInterval(pollTimer);
        pollTimer = null;
        statusEl.className = 'status failed';
        statusEl.textContent = 'job not found';
        logsEl.hidden = false;
        emptyEl.hidden = true;
        logsEl.textContent = data.error || '任务不存在。可能是服务重启后旧任务状态丢失，或任务目录已被删除。';
        renderLoginReviews({login_required_reviews: []});
        timerMetaEl.className = 'timer over';
        timerMetaEl.textContent = '计时不可用';
        runBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        terminateBtn.disabled = true;
        return;
      }
      renderJob(data);
    }

    async function controlJob(action) {
      if (!currentJobId) return;
      const res = await fetch(`/api/jobs/${encodeURIComponent(currentJobId)}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      });
      const data = await res.json();
      if (!res.ok) {
        logsEl.hidden = false;
        emptyEl.hidden = true;
        logsEl.textContent += `\n[ui] 控制失败：${data.error || action}\n`;
        return;
      }
      renderJob(data);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      runBtn.disabled = true;
      statusEl.className = 'status running';
      statusEl.textContent = 'starting';
      timerMetaEl.className = 'timer';
      timerMetaEl.textContent = '计时启动中';
      const formData = new FormData(form);
      const payload = {
        competitors: formData.get('competitors'),
        own_product_name: formData.get('own_product_name'),
        own_product_positioning: formData.get('own_product_positioning'),
        own_product_context: formData.get('own_product_context'),
        manual_include_keywords: formData.get('manual_include_keywords'),
        searxng_url: formData.get('searxng_url'),
        proxy_url: formData.get('proxy_url'),
        per_query: Number(formData.get('per_query')),
        max_pages: Number(formData.get('max_pages')),
        crawl_concurrency: Number(formData.get('crawl_concurrency')),
        max_discovered_competitors: Number(formData.get('max_discovered_competitors')),
        login_assist_wait: Number(formData.get('login_assist_wait')),
        image_engine: formData.get('image_engine'),
        max_image_downloads: Number(formData.get('max_image_downloads')),
        image_terms: formData.get('image_terms'),
        experiment_minutes: Number(formData.get('experiment_minutes')),
        broad_crawl: formData.has('broad_crawl'),
        codex_review: formData.has('codex_review'),
        codex_model: formData.get('codex_model'),
        use_ml_filter: formData.has('use_ml_filter'),
        ml_model_path: formData.get('ml_model_path'),
        login_assist: formData.has('login_assist'),
        skip_gui_review: formData.has('skip_gui_review'),
        skip_crawl: formData.has('skip_crawl'),
        skip_images: formData.has('skip_images'),
        no_cn: formData.has('no_cn')
      };
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        statusEl.className = 'status failed';
        statusEl.textContent = 'failed';
        logsEl.hidden = false;
        emptyEl.hidden = true;
        logsEl.textContent = data.error || '启动失败';
        timerMetaEl.className = 'timer over';
        timerMetaEl.textContent = '计时未开始';
        runBtn.disabled = false;
        return;
      }
      currentJobId = data.id;
      renderJob(data);
      clearInterval(pollTimer);
      pollTimer = setInterval(fetchJob, 1200);
      fetchJob();
    });

    refreshBtn.addEventListener('click', fetchJob);
    pauseBtn.addEventListener('click', () => controlJob('pause'));
    resumeBtn.addEventListener('click', () => controlJob('resume'));
    terminateBtn.addEventListener('click', () => controlJob('terminate'));

    checkBtn.addEventListener('click', async () => {
      const formData = new FormData(form);
      logsEl.hidden = false;
      emptyEl.hidden = true;
      statusEl.className = 'status running';
      statusEl.textContent = 'checking';
      timerMetaEl.className = 'timer';
      timerMetaEl.textContent = '环境检查中';
      const url = `/api/check?searxng_url=${encodeURIComponent(formData.get('searxng_url'))}&proxy_url=${encodeURIComponent(formData.get('proxy_url'))}`;
      const res = await fetch(url);
      const data = await res.json();
      statusEl.className = data.ok ? 'status done' : 'status failed';
      statusEl.textContent = data.ok ? 'ready' : 'needs setup';
      timerMetaEl.className = 'timer';
      timerMetaEl.textContent = '计时未开始';
      logsEl.textContent = [
        `SearXNG: ${data.searxng.ok ? 'OK' : 'FAILED'} ${data.searxng.message || ''}`,
        `SearXNG search: ${data.searxng_search_probe && data.searxng_search_probe.ok ? 'OK' : 'FAILED'} ${data.searxng_search_probe ? data.searxng_search_probe.message || '' : ''}`,
        `Proxy: ${data.proxy.ok ? (data.proxy_url ? 'OK' : 'not set') : 'FAILED'} ${data.proxy.message || ''}`,
        `crawl4ai: ${data.python.crawl4ai ? 'OK' : 'MISSING'}`,
        `icrawler: ${data.python.icrawler ? 'OK' : 'MISSING'}`,
        `Codex CLI: ${data.codex.ok ? 'OK' : 'MISSING'} ${data.codex.message || ''}`,
        `Local filter model: ${data.local_filter_model && data.local_filter_model.enabled ? 'OK' : 'not trained'} ${(data.local_filter_model && (data.local_filter_model.training_rows || data.local_filter_model.message)) || ''}`,
        `Python: ${data.python.executable}`,
      ].join('\n');
    });

    async function refreshModelStatus() {
      const modelPath = modelOutPathEl.value || 'models/filter_model.pt';
      const res = await fetch(`/api/ml/status?model_path=${encodeURIComponent(modelPath)}`);
      const data = await res.json();
      trainingLogsEl.hidden = false;
      trainingLogsEl.textContent = JSON.stringify(data, null, 2);
      modelStatusTextEl.textContent = data.enabled
        ? `模型可用：${data.training_rows || 0} 条标注，${data.model_version || ''}`
        : `模型未启用：${data.message || 'model file not found'}。请先在本轮人工抽样标注表/问题页面核验清单填写 human_label，再训练生成 .pt 模型。`;
    }

    modelStatusBtn.addEventListener('click', refreshModelStatus);

    async function trainModelFromUi(includeCurrentJob, triggerButton) {
      trainModelBtn.disabled = true;
      if (triggerButton) triggerButton.disabled = true;
      modelStatusTextEl.textContent = '训练中';
      trainingLogsEl.hidden = false;
      trainingLogsEl.textContent = '';
      const payload = {
        labels_path: labelsPathEl.value || 'training_data/review_labels.csv',
        model_out: modelOutPathEl.value || 'models/filter_model.pt',
        cards_dir: cardsDirPathEl.value || 'search_cards',
        min_labeled_rows: Number(minLabelsEl.value || 10),
        min_card_labeled_rows: 3,
        job_id: currentJobId || '',
        include_problem_reviews: Boolean(includeCurrentJob)
      };
      try {
        const res = await fetch('/api/ml/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json().catch(() => ({ error: '服务返回内容不是 JSON' }));
        trainingLogsEl.textContent = JSON.stringify(data, null, 2);
        modelStatusTextEl.textContent = res.ok
          ? `训练完成：${data.training_rows || 0} 条标注，生成搜索卡片 ${data.search_cards ? data.search_cards.written_cards || 0 : 0} 张`
          : `训练失败：${data.error || 'unknown error'}。若提示样本不足，请先填写 human_label。`;
      } catch (error) {
        trainingLogsEl.textContent = JSON.stringify({ error: String(error) }, null, 2);
        modelStatusTextEl.textContent = `训练失败：${String(error)}`;
      } finally {
        trainModelBtn.disabled = false;
        if (triggerButton) triggerButton.disabled = false;
      }
    }

    trainModelBtn.addEventListener('click', () => trainModelFromUi(true, trainModelBtn));
    closeReviewModalBtn.addEventListener('click', () => { reviewModalEl.hidden = true; });
  </script>
</body>
</html>
"""


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, body: str, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def file_size_label(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


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


def archive_internal_outputs(out_dir: Path, keep_run_log: bool = False) -> None:
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
                "根目录只保留默认交付物。\n",
                encoding="utf-8",
            )


def artifact_candidate_paths(out_dir: Path, filename: str) -> List[Path]:
    internal_dir = out_dir / INTERNAL_OUTPUT_DIR_NAME
    names = [filename]
    if filename in CHINESE_EXPORT_ALIASES:
        names.append(CHINESE_EXPORT_ALIASES[filename])
    reverse_aliases = {alias: source for source, alias in CHINESE_EXPORT_ALIASES.items()}
    source_name = reverse_aliases.get(filename)
    if source_name:
        names.append(source_name)
    seen_names = []
    for name in names:
        if name and name not in seen_names:
            seen_names.append(name)
    paths: List[Path] = []
    for name in seen_names:
        paths.append(out_dir / name)
        paths.append(internal_dir / name)
    return paths


def artifact_path(out_dir: Path, filename: str) -> Optional[Path]:
    for path in artifact_candidate_paths(out_dir, filename):
        if path.exists():
            return path
    return None


def format_elapsed_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_local_time(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


LOGIN_GENERIC_COMPETITOR_TERMS = {
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

LOGIN_AUTH_HOST_LABELS = {
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

LOGIN_AUTH_URL_TOKENS = {
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
}

LOGIN_AUTH_URL_NOISE_TOKENS = {
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

LOGIN_AUTH_TITLE_TERMS = {
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
    "登录",
    "登陆",
    "注册",
    "账号",
    "账户",
    "密码",
    "验证码",
}

LOGIN_AUTH_TITLE_NOISE_TERMS = {
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
}

LOGIN_AUTH_MARKER_RE = re.compile(
    r"(?<![a-z0-9])("
    r"log\s*in|sign\s*in|signin|sign-in|"
    r"sign\s*up|signup|sign-up|login|register|registration|"
    r"oauth|passport|account|accounts|password|captcha"
    r")(?![a-z0-9])",
    re.I,
)

LOGIN_AUTH_FORM_TERMS = {
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
}


def login_url_has_auth_gate_path(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    first_label = host.split(".", 1)[0] if host else ""
    if first_label in LOGIN_AUTH_HOST_LABELS:
        return True
    for segment in (parsed.path or "").lower().split("/"):
        compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", segment).strip()
        if not compact:
            continue
        tokens = set(compact.split())
        has_noise = bool(tokens & LOGIN_AUTH_URL_NOISE_TOKENS)
        if compact.replace(" ", "-") in LOGIN_AUTH_URL_TOKENS and not has_noise:
            return True
        if tokens & LOGIN_AUTH_URL_TOKENS and len(tokens) <= 3 and not has_noise:
            return True
    return False


def login_title_is_auth_gate(title: str) -> bool:
    cleaned = re.sub(r"\s+", " ", title or "").strip()
    if not cleaned or len(cleaned) > 140:
        return False
    chunks = [cleaned, *re.split(r"[|:：/·\-–—]+", cleaned)]
    for chunk in chunks:
        low = chunk.strip().lower()
        if not low:
            continue
        if any(term in low for term in LOGIN_AUTH_TITLE_NOISE_TERMS):
            continue
        word_count = len(re.findall(r"[a-z0-9]+", low))
        has_english_login = bool(LOGIN_AUTH_MARKER_RE.search(low))
        has_chinese_login = any(token in low for token in ("登录", "登陆", "注册", "账号登录", "账户登录", "密码登录"))
        if has_english_login and 0 < word_count <= 5:
            return True
        if has_chinese_login and len(low) <= 30:
            return True
    return False


def login_row_is_actual_auth_gate(url: str, title: str = "", source_text: str = "") -> bool:
    haystack = f"{url}\n{title}\n{source_text}".lower()
    has_auth_marker = bool(
        LOGIN_AUTH_MARKER_RE.search(haystack)
        or any(token in haystack for token in ("登录", "登陆", "注册", "账号", "账户", "密码", "验证码", "请登录"))
    )
    if not has_auth_marker:
        return False
    if login_url_has_auth_gate_path(url) or login_title_is_auth_gate(title):
        return True
    return any(term in haystack for term in LOGIN_AUTH_FORM_TERMS)


def login_row_has_strong_binding(competitor: str, url: str, title: str = "") -> bool:
    competitor = (competitor or "").strip().lower()
    if not competitor:
        return True
    haystack = f"{url} {title}".lower()
    if any(hint in haystack for hint in LOGIN_POOL_EXCLUDE_HINTS):
        return False
    compact_haystack = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack)
    compact_competitor = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", competitor)
    if compact_competitor and compact_competitor in compact_haystack:
        return True
    if ".ai" in competitor:
        return False
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", competitor)
    semantic_tokens = [
        token
        for token in raw_tokens
        if len(token) > 1 and token not in LOGIN_GENERIC_COMPETITOR_TERMS
    ]
    if len(semantic_tokens) > 1:
        if any(len(token) <= 2 for token in semantic_tokens):
            return False
        return all(token in haystack or token in compact_haystack for token in semantic_tokens)
    if not semantic_tokens:
        return True
    return semantic_tokens[0] in haystack or semantic_tokens[0] in compact_haystack


def job_timing_snapshot(job: Job, now: Optional[float] = None) -> dict:
    end_time = job.finished_at if job.finished_at is not None else (now if now is not None else time.time())
    start_time = job.started_at or job.created_at
    elapsed_seconds = max(0, int(end_time - start_time))
    timebox_seconds = max(0, int(job.experiment_minutes or 0) * 60)
    remaining_seconds: Optional[int] = None
    timebox_exceeded = False
    if timebox_seconds:
        remaining_seconds = max(0, timebox_seconds - elapsed_seconds)
        timebox_exceeded = elapsed_seconds > timebox_seconds
    return {
        "started_at": job.started_at,
        "started_at_label": format_local_time(job.started_at),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_label": format_elapsed_seconds(elapsed_seconds),
        "experiment_minutes": int(job.experiment_minutes or 0),
        "timebox_seconds": timebox_seconds,
        "timebox_label": f"{job.experiment_minutes} 分钟" if job.experiment_minutes else "未设置",
        "remaining_seconds": remaining_seconds,
        "remaining_label": format_elapsed_seconds(remaining_seconds) if remaining_seconds is not None else "",
        "timebox_exceeded": timebox_exceeded,
    }


def load_login_required_reviews(out_dir: Path, limit: int = 200) -> List[dict]:
    candidate_names = [
        "需登录队列.csv",
        "login_required_queue.csv",
        "GUI自动复核结果.csv",
        "gui_review_results.csv",
        "人工复核队列.csv",
        "manual_review_queue.csv",
    ]
    rows: List[dict] = []
    seen = set()
    candidate_paths: List[Path] = []
    seen_paths = set()
    for name in candidate_names:
        for path in artifact_candidate_paths(out_dir, name):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidate_paths.append(path)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    requires_login = str(row.get("requires_user_login") or "").strip().lower() == "yes"
                    review_reason = str(row.get("review_reason") or "").strip()
                    status = str(row.get("automated_review_status") or "").strip()
                    if status.lower() == "login_assisted_snapshot_captured":
                        continue
                    if status.lower() == "login_skipped_by_user":
                        continue
                    if not requires_login and review_reason != "login_required_user_action" and "login" not in status.lower():
                        continue
                    url = row.get("login_assist_url") or row.get("url") or row.get("gui_review_url") or ""
                    domain = row.get("domain") or (urlparse(url).netloc or "").lower().removeprefix("www.")
                    key = (row.get("competitor") or "", domain or url)
                    if not url or key in seen:
                        continue
                    if login_request_marker_exists(out_dir, key[0], url, "skip"):
                        continue
                    if not login_row_has_strong_binding(key[0], url, row.get("title") or ""):
                        continue
                    source_text = "\n".join(
                        str(row.get(key) or "")
                        for key in ("crawl_error", "cleaned_excerpt_sample", "text_snapshot_excerpt", "reason")
                    )
                    if not login_row_is_actual_auth_gate(url, row.get("title") or "", source_text):
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "competitor": row.get("competitor") or "",
                            "domain": domain,
                            "title": row.get("title") or "",
                            "url": row.get("url") or url,
                            "login_assist_url": url,
                            "queued_url_count": row.get("queued_url_count") or "1",
                            "automated_review_status": status or "requires_user_login",
                            "login_click_requested": "yes" if login_request_marker_exists(out_dir, key[0], url, "click") else "no",
                            "login_skip_requested": "no",
                            "next_step": row.get("next_step") or row.get("suggested_next_step") or "",
                        }
                    )
                    if len(rows) >= limit:
                        return rows
        except OSError:
            continue
    return rows


def load_problem_reviews(out_dir: Path, limit: int = 50) -> List[dict]:
    candidate_names = ["问题页面核验清单.csv", "problem_pages_review.csv"]
    rows: List[dict] = []
    seen = set()
    candidate_paths: List[Path] = []
    seen_paths = set()
    for name in candidate_names:
        for path in artifact_candidate_paths(out_dir, name):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidate_paths.append(path)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    url = row.get("url") or ""
                    problem_type = row.get("problem_type") or ""
                    key = (row.get("competitor") or "", url, problem_type)
                    if not url or key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "competitor": row.get("competitor") or "",
                            "priority": row.get("priority") or "",
                            "problem_type": problem_type,
                            "title": row.get("title") or "",
                            "url": url,
                            "domain": row.get("domain") or "",
                            "status": row.get("status") or "",
                            "what_to_verify": row.get("what_to_verify") or "",
                            "data_entry_decision": row.get("data_entry_decision") or "",
                            "suggested_human_label": row.get("suggested_human_label") or "",
                        }
                    )
                    if len(rows) >= limit:
                        return rows
        except OSError:
            continue
        if rows:
            break
    return rows


def compact_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_csv_artifact_rows(out_dir: Path, names: Sequence[str], source_file_label: str = "") -> List[dict]:
    rows: List[dict] = []
    candidate_paths: List[Path] = []
    seen_paths = set()
    for name in names:
        for path in artifact_candidate_paths(out_dir, name):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidate_paths.append(path)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    normalized = {str(key): compact_cell(value) for key, value in row.items() if key is not None}
                    normalized["source_file"] = source_file_label or path.name
                    rows.append(normalized)
        except OSError:
            continue
        if rows:
            break
    return rows


def feedback_row_url(row: Mapping[str, Any]) -> str:
    return compact_cell(row.get("url") or row.get("login_assist_url") or row.get("url_or_path") or row.get("gui_review_url"))


def feedback_row_domain(row: Mapping[str, Any]) -> str:
    explicit = compact_cell(row.get("domain"))
    if explicit:
        return explicit.lower().removeprefix("www.")
    return (urlparse(feedback_row_url(row)).netloc or "").lower().removeprefix("www.")


def feedback_category_for_row(row: Mapping[str, Any]) -> str:
    problem_type = compact_cell(row.get("problem_type"))
    status = compact_cell(row.get("status") or row.get("automated_review_status") or row.get("decision_status"))
    hard_gate = compact_cell(row.get("hard_gate"))
    source_policy = compact_cell(row.get("source_policy_tier"))
    source_kind = compact_cell(row.get("source_kind"))
    pending = compact_cell(row.get("pending_verification")).lower()
    page_role = compact_cell(row.get("page_role"))
    haystack = " ".join([problem_type, status, hard_gate, source_policy, source_kind, page_role, compact_cell(row.get("reason"))]).lower()
    if "登录" in problem_type or "login" in haystack or page_role == "auth_or_account_shell":
        return "待登录"
    if any(token in haystack for token in ("403", "cloudflare", "captcha", "datadome", "反爬", "超时", "timeout", "404", "minimal_text", "正文不足", "js 空壳")):
        return "失败/反爬/超时"
    if status.lower() == "rejected" or hard_gate.startswith("rejected") or source_policy.lower().startswith("reject"):
        return "内容不好放弃"
    if source_policy.startswith("P0") or compact_cell(row.get("primary_evidence_candidate")).lower() == "yes" or (
        status.lower() in {"selected", "accepted"} and "official" in source_kind.lower()
    ):
        return "主要内容来源"
    if pending == "yes" or "待核实" in haystack or "verify" in haystack:
        return "待核实线索"
    include_score = feedback_score(row.get("ml_include_score"))
    exclude_score = feedback_score(row.get("ml_exclude_score"))
    verify_score = feedback_score(row.get("ml_verify_later_score"))
    scored = [score for score in (include_score, exclude_score, verify_score) if score > 0]
    confidence = compact_cell(row.get("ml_confidence")).lower()
    if confidence in {"low", "medium"} or (len(scored) >= 2 and max(scored) - sorted(scored)[-2] <= 0.25):
        return "模型边界样本"
    return ""


def feedback_score(value: Any) -> float:
    try:
        return float(compact_cell(value) or 0)
    except ValueError:
        return 0.0


def feedback_conclusion_label(row: Mapping[str, Any], category: str) -> str:
    label = normalize_label(row.get("suggested_human_label") or row.get("suggested_label") or row.get("ml_label"))
    if label:
        return label
    if category == "主要内容来源":
        return "include"
    if category == "内容不好放弃":
        return "exclude"
    if category == "失败/反爬/超时" and "404" in compact_cell(row.get("problem_type")):
        return "exclude"
    return "verify_later"


def feedback_counter_label(label: str, category: str) -> str:
    if label == "include":
        return "exclude"
    if label == "exclude":
        return "verify_later"
    if category == "主要内容来源":
        return "exclude"
    return "exclude"


def feedback_model_conclusion(label: str, category: str) -> str:
    if label == "include":
        return f"{category}：建议收录，可作为竞品分析证据。"
    if label == "exclude":
        return f"{category}：建议放弃，不进入本轮分析。"
    return f"{category}：建议待核实，补证后再决定是否入库。"


def build_feedback_card(row: Mapping[str, Any], category: str, job_id: str, feedback_status: Mapping[Tuple[str, str, str], Mapping[str, Any]]) -> dict:
    url = feedback_row_url(row)
    label = feedback_conclusion_label(row, category)
    key = (compact_cell(row.get("competitor")), url, category)
    prior = feedback_status.get(key, {})
    return {
        "source_job_id": job_id,
        "feedback_category": category,
        "competitor": compact_cell(row.get("competitor")),
        "title": compact_cell(row.get("title")),
        "url": url,
        "domain": feedback_row_domain(row),
        "snippet": compact_cell(row.get("snippet") or row.get("content_preview") or row.get("text_snapshot_excerpt")),
        "reason": compact_cell(row.get("reason") or row.get("verification_reason") or row.get("what_to_verify")),
        "model_conclusion": feedback_model_conclusion(label, category),
        "conclusion_label": label,
        "counter_human_label": feedback_counter_label(label, category),
        "source_file": compact_cell(row.get("source_file")),
        "source_queue": compact_cell(row.get("source_queue")),
        "problem_type": compact_cell(row.get("problem_type")),
        "decision_status": compact_cell(row.get("decision_status")),
        "hard_gate": compact_cell(row.get("hard_gate")),
        "source_kind": compact_cell(row.get("source_kind")),
        "page_role": compact_cell(row.get("page_role")),
        "source_policy_tier": compact_cell(row.get("source_policy_tier")),
        "pending_verification": compact_cell(row.get("pending_verification")),
        "verification_reason": compact_cell(row.get("verification_reason")),
        "fact_type": compact_cell(row.get("fact_type")),
        "increment_type": compact_cell(row.get("increment_type")),
        "fact_group": compact_cell(row.get("fact_group")),
        "ml_confidence": compact_cell(row.get("ml_confidence")),
        "ml_include_score": compact_cell(row.get("ml_include_score")),
        "ml_exclude_score": compact_cell(row.get("ml_exclude_score")),
        "ml_verify_later_score": compact_cell(row.get("ml_verify_later_score")),
        "feedback_judgement": compact_cell(prior.get("feedback_judgement")),
        "human_label": compact_cell(prior.get("human_label")),
        "human_reason": compact_cell(prior.get("human_reason")),
        "feedback_status": compact_cell(prior.get("feedback_status")) or "pending",
        "reviewed_at": compact_cell(prior.get("reviewed_at")),
    }


def read_feedback_status(out_dir: Path) -> Dict[Tuple[str, str, str], dict]:
    rows = read_csv_artifact_rows(out_dir, ["人工反馈标注.csv", "human_feedback_labels.csv"], "人工反馈标注.csv")
    status: Dict[Tuple[str, str, str], dict] = {}
    for row in rows:
        key = (compact_cell(row.get("competitor")), feedback_row_url(row), compact_cell(row.get("feedback_category")))
        if key[1] and key[2]:
            status[key] = row
    return status


def build_feedback_review_samples(out_dir: Path, per_category: int = 3) -> List[dict]:
    out_dir = Path(out_dir)
    source_rows: List[dict] = []
    source_rows.extend(read_csv_artifact_rows(out_dir, ["问题页面核验清单.csv", "problem_pages_review.csv"], "问题页面核验清单.csv"))
    source_rows.extend(read_csv_artifact_rows(out_dir, ["人工抽样标注表.csv", "training_review_sample.csv"], "人工抽样标注表.csv"))
    source_rows.extend(read_csv_artifact_rows(out_dir, ["evidence_audit.csv", "证据筛选审计.csv"], "evidence_audit.csv"))
    feedback_status = read_feedback_status(out_dir)
    by_category: Dict[str, List[dict]] = {category: [] for category in FEEDBACK_CATEGORY_ORDER}
    seen_source_keys = set()
    for row in source_rows:
        category = feedback_category_for_row(row)
        url = feedback_row_url(row)
        if not category or not url:
            continue
        key = (category, compact_cell(row.get("competitor")), url)
        if key in seen_source_keys:
            continue
        seen_source_keys.add(key)
        by_category.setdefault(category, []).append(row)

    samples: List[dict] = []
    used_urls = set()
    job_id = out_dir.name
    for category in FEEDBACK_CATEGORY_ORDER:
        candidates = by_category.get(category, [])
        seed = int(hashlib.sha1(f"{job_id}:{category}:{len(candidates)}".encode("utf-8")).hexdigest()[:10], 16)
        shuffled = list(candidates)
        random.Random(seed).shuffle(shuffled)
        picked = 0
        for row in shuffled:
            url = feedback_row_url(row)
            if url in used_urls:
                continue
            samples.append(build_feedback_card(row, category, job_id, feedback_status))
            used_urls.add(url)
            picked += 1
            if picked >= max(0, per_category):
                break
    return samples


def merge_csv_fields(existing_fields: Sequence[str], preferred_fields: Sequence[str], row: Mapping[str, Any]) -> List[str]:
    fields: List[str] = []
    for field in [*preferred_fields, *existing_fields, *[str(key) for key in row.keys()]]:
        if field and field not in fields:
            fields.append(field)
    return fields


def upsert_csv_row(path: Path, row: Mapping[str, Any], preferred_fields: Sequence[str], key_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[dict] = []
    existing_fields: List[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [dict(item) for item in reader]
    normalized_row = {str(key): compact_cell(value) for key, value in row.items()}
    fields = merge_csv_fields(existing_fields, preferred_fields, normalized_row)

    def row_key(item: Mapping[str, Any]) -> Tuple[str, ...]:
        return tuple(compact_cell(item.get(field)) for field in key_fields)

    target_key = row_key(normalized_row)
    replaced = False
    next_rows: List[dict] = []
    for existing in existing_rows:
        if row_key(existing) == target_key:
            next_rows.append({**existing, **normalized_row})
            replaced = True
        else:
            next_rows.append(existing)
    if not replaced:
        next_rows.append(normalized_row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in next_rows:
            writer.writerow(item)


def feedback_payload_judgement(value: str) -> str:
    normalized = compact_cell(value).lower()
    if normalized in {"qualified", "good", "pass", "合格", "正确"}:
        return "qualified"
    if normalized in {"unqualified", "bad", "fail", "不合格", "错误"}:
        return "unqualified"
    raise ValueError("judgement must be qualified or unqualified")


def record_feedback_review(payload: Mapping[str, Any]) -> dict:
    job_id = compact_cell(payload.get("job") or payload.get("job_id"))
    out_dir = safe_job_dir(job_id)
    if not out_dir:
        raise ValueError("job not found")
    url = compact_cell(payload.get("url"))
    competitor = compact_cell(payload.get("competitor"))
    category = compact_cell(payload.get("feedback_category"))
    judgement = feedback_payload_judgement(compact_cell(payload.get("judgement")))
    samples = build_feedback_review_samples(out_dir, per_category=10)
    match = None
    for row in samples:
        if feedback_row_url(row) != url:
            continue
        if competitor and compact_cell(row.get("competitor")) != competitor:
            continue
        if category and compact_cell(row.get("feedback_category")) != category:
            continue
        match = row
        break
    if not match:
        raise ValueError("feedback sample not found")
    human_label = match["conclusion_label"] if judgement == "qualified" else match["counter_human_label"]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    feedback_text = "结论合格" if judgement == "qualified" else "结论不合格"
    row = {
        **match,
        "source_job_id": job_id,
        "feedback_judgement": judgement,
        "human_label": human_label,
        "human_reason": f"{feedback_text}：{match.get('model_conclusion')}",
        "feedback_status": "ready_for_training",
        "reviewed_at": now,
    }
    key_fields = ["source_job_id", "feedback_category", "competitor", "url"]
    upsert_csv_row(out_dir / "人工反馈标注.csv", row, FEEDBACK_REVIEW_FIELDS, key_fields)
    upsert_csv_row(out_dir / "human_feedback_labels.csv", row, FEEDBACK_REVIEW_FIELDS, key_fields)
    upsert_csv_row(DEFAULT_REVIEW_LABELS_PATH, row, FEEDBACK_REVIEW_FIELDS, key_fields)
    return {
        "ok": True,
        "job": job_id,
        "feedback_category": row["feedback_category"],
        "url": row["url"],
        "judgement": judgement,
        "human_label": human_label,
        "review_labels_path": str(DEFAULT_REVIEW_LABELS_PATH),
        "job_feedback_path": str(out_dir / "人工反馈标注.csv"),
    }


def write_timing_artifacts(job: Job) -> None:
    timing = job_timing_snapshot(job)
    record = {
        "job_id": job.id,
        "status": job.status,
        "returncode": job.returncode,
        "created_at": job.created_at,
        "created_at_label": format_local_time(job.created_at),
        "started_at": job.started_at,
        "started_at_label": timing["started_at_label"],
        "finished_at": job.finished_at,
        "finished_at_label": format_local_time(job.finished_at),
        "elapsed_seconds": timing["elapsed_seconds"],
        "elapsed_label": timing["elapsed_label"],
        "experiment_minutes": timing["experiment_minutes"],
        "timebox_seconds": timing["timebox_seconds"],
        "timebox_label": timing["timebox_label"],
        "remaining_seconds": timing["remaining_seconds"],
        "remaining_label": timing["remaining_label"],
        "timebox_exceeded": timing["timebox_exceeded"],
        "command": job.command,
    }
    (job.out_dir / "实验计时记录.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    timebox_line = timing["timebox_label"]
    if timing["timebox_seconds"]:
        timebox_line += f"（{'已超时' if timing['timebox_exceeded'] else '未超时'}）"
    md = "\n".join(
        [
            "# 实验计时记录",
            "",
            f"- 任务 ID：{job.id}",
            f"- 任务状态：{job.status}",
            f"- 开始时间：{timing['started_at_label'] or format_local_time(job.created_at)}",
            f"- 结束时间：{format_local_time(job.finished_at) or '运行中'}",
            f"- 全流程耗时：{timing['elapsed_label']}",
            f"- 实验限时：{timebox_line}",
            f"- 退出码：{job.returncode if job.returncode is not None else ''}",
            "",
        ]
    )
    (job.out_dir / "实验计时记录.md").write_text(md, encoding="utf-8")


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def check_proxy_endpoint(proxy_url: str, timeout: float = 2.0) -> dict:
    proxy_url = (proxy_url or "").strip()
    if not proxy_url:
        return {"ok": True, "message": "not set"}
    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return {"ok": False, "message": "代理地址格式错误"}
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme in {"https", "socks5h"} else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return {"ok": True, "message": f"{parsed.hostname}:{port}"}
    except OSError as exc:
        return {"ok": False, "message": f"{parsed.hostname}:{port} 不可连接：{exc}"}


def job_snapshot(job: Job, now: Optional[float] = None) -> dict:
    artifacts = []
    visible_names = list(PRIMARY_ARTIFACTS)
    if job.status == "failed":
        visible_names += FAILURE_ARTIFACTS
    for name in visible_names:
        path = artifact_path(job.out_dir, name)
        if path:
            artifacts.append({"name": name, "size_label": file_size_label(path)})
    snapshot = {
        "id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "created_at_label": format_local_time(job.created_at),
        "finished_at": job.finished_at,
        "finished_at_label": format_local_time(job.finished_at),
        "returncode": job.returncode,
        "out_dir": str(job.out_dir),
        "command": job.command,
        "logs": job.logs[-300:],
        "artifacts": artifacts,
        "login_required_reviews": load_login_required_reviews(job.out_dir),
        "problem_reviews": load_problem_reviews(job.out_dir),
        "feedback_reviews": build_feedback_review_samples(job.out_dir) if job.status in {"done", "failed", "terminated"} else [],
    }
    snapshot.update(job_timing_snapshot(job, now=now))
    return snapshot


def safe_job_dir(job_id: str) -> Optional[Path]:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{6}", job_id or ""):
        return None
    out_dir = (RUNS_DIR / job_id).resolve()
    try:
        out_dir.relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return out_dir if out_dir.is_dir() else None


def latest_job_dir_with_training_reviews() -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    candidates: List[Path] = []
    for item in RUNS_DIR.iterdir():
        if not item.is_dir():
            continue
        if artifact_path(item, "人工抽样标注表.csv") or artifact_path(item, "问题页面核验清单.csv"):
            candidates.append(item)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def login_click_marker_id(competitor: str, url: str) -> str:
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower().removeprefix("www.")
    stable_target = domain or url
    raw = f"{(competitor or '').strip().lower()}::{stable_target.strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def login_request_marker_path(out_dir: Path, competitor: str, url: str, kind: str) -> Path:
    marker_dir = out_dir / f"login_{kind}_requests"
    return marker_dir / f"{login_click_marker_id(competitor, url)}.json"


def login_request_marker_exists(out_dir: Path, competitor: str, url: str, kind: str) -> bool:
    return login_request_marker_path(out_dir, competitor, url, kind).exists()


def record_login_request(job_id: str, competitor: str, url: str, kind: str) -> dict:
    out_dir = safe_job_dir(job_id)
    if not out_dir:
        raise ValueError("job not found")
    url = (url or "").strip()
    if not url or urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("invalid login url")
    if kind not in {"click", "skip"}:
        raise ValueError("invalid login action")
    clicked_at = time.time()
    marker_path = login_request_marker_path(out_dir, competitor, url, kind)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "competitor": competitor,
        "url": url,
        "domain": (urlparse(url).netloc or "").lower().removeprefix("www."),
        "action": kind,
        "requested_at": clicked_at,
        "requested_at_label": format_local_time(clicked_at),
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**payload, "marker_path": str(marker_path)}


def record_login_open_request(job_id: str, competitor: str, url: str) -> dict:
    return record_login_request(job_id, competitor, url, "click")


def record_login_skip_request(job_id: str, competitor: str, url: str) -> dict:
    return record_login_request(job_id, competitor, url, "skip")


def disk_job_snapshot(job_id: str) -> Optional[dict]:
    out_dir = safe_job_dir(job_id)
    if not out_dir:
        return None
    artifacts = []
    visible_names = list(PRIMARY_ARTIFACTS)
    for name in visible_names:
        path = artifact_path(out_dir, name)
        if path:
            artifacts.append({"name": name, "size_label": file_size_label(path)})
    run_log = artifact_path(out_dir, "run.log")
    logs = []
    if run_log:
        try:
            logs = run_log.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-300:]
        except OSError:
            logs = []
    finished = max((path.stat().st_mtime for path in out_dir.iterdir() if path.exists()), default=out_dir.stat().st_mtime)
    status = "done"
    done_markers = ["竞品分析报告_图片内嵌版.md", "final_analysis_embedded.md", "final_analysis.md", "analysis.md", "report.md"]
    if not any(artifact_path(out_dir, name) for name in done_markers):
        status = "failed"
    if status == "failed":
        known_artifacts = {row["name"] for row in artifacts}
        for name in FAILURE_ARTIFACTS:
            if name in known_artifacts:
                continue
            path = artifact_path(out_dir, name)
            if path:
                artifacts.append({"name": name, "size_label": file_size_label(path)})
    timing_path = out_dir / "实验计时记录.json"
    timing = {}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timing = {}
    created_at = timing.get("created_at", out_dir.stat().st_ctime)
    finished_at = timing.get("finished_at", finished)
    elapsed_seconds = int(timing.get("elapsed_seconds", max(0, finished_at - created_at)))
    return {
        "id": job_id,
        "status": status,
        "created_at": created_at,
        "created_at_label": timing.get("created_at_label", format_local_time(created_at)),
        "started_at": timing.get("started_at", created_at),
        "started_at_label": timing.get("started_at_label", format_local_time(created_at)),
        "finished_at": finished_at,
        "finished_at_label": timing.get("finished_at_label", format_local_time(finished_at)),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_label": timing.get("elapsed_label", format_elapsed_seconds(elapsed_seconds)),
        "experiment_minutes": int(timing.get("experiment_minutes", 0) or 0),
        "timebox_seconds": int(timing.get("timebox_seconds", 0) or 0),
        "timebox_label": timing.get("timebox_label", "未设置"),
        "remaining_seconds": timing.get("remaining_seconds"),
        "remaining_label": timing.get("remaining_label", ""),
        "timebox_exceeded": bool(timing.get("timebox_exceeded", False)),
        "returncode": None,
        "out_dir": str(out_dir),
        "command": [],
        "logs": logs,
        "artifacts": artifacts,
        "login_required_reviews": load_login_required_reviews(out_dir),
        "problem_reviews": load_problem_reviews(out_dir),
        "feedback_reviews": build_feedback_review_samples(out_dir) if status in {"done", "failed", "terminated"} else [],
        "restored_from_disk": True,
    }


def parse_competitors(raw: str) -> List[str]:
    values = []
    normalized = re.sub(r"[,，;；]+", "\n", raw)
    for chunk in normalized.splitlines():
        item = chunk.strip()
        if item:
            values.append(item)
    seen = set()
    out = []
    for item in values:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def int_payload(payload: dict, key: str, default: int) -> int:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return int(value)


def bool_payload(payload: dict, key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(bool_payload({"value": item}, "value", False) for item in value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "checked", "是", "开启", "启用"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "unchecked", "否", "关闭", "停用"}:
        return False
    return default


def resolve_app_path(value: str, default_path: Path) -> Path:
    raw = (value or "").strip()
    if not raw:
        return default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def model_status_for_ui(
    model_path: Path,
    bootstrap_label_paths: Optional[List[Path]] = None,
    min_labeled_rows: int = 3,
) -> dict:
    model_path = Path(model_path).expanduser().resolve()
    status = model_status(model_path)
    if status.get("enabled"):
        status["bootstrap_created"] = False
        return status

    bootstrap_paths = bootstrap_label_paths
    should_bootstrap_default = model_path == DEFAULT_FILTER_MODEL_PATH.resolve()
    if bootstrap_paths is None and should_bootstrap_default:
        bootstrap_paths = [DEFAULT_BOOTSTRAP_LABELS_PATH, DEFAULT_REVIEW_LABELS_PATH]
    if not bootstrap_paths:
        status["bootstrap_created"] = False
        return status

    try:
        bootstrap = bootstrap_filter_model_if_missing(
            model_path,
            bootstrap_paths,
            min_labeled_rows=min_labeled_rows,
        )
    except Exception as exc:
        status["bootstrap_created"] = False
        status["bootstrap_error"] = str(exc)
        return status

    refreshed = model_status(model_path)
    refreshed["bootstrap_created"] = bool(bootstrap.get("created"))
    refreshed["bootstrap_label_paths"] = [str(Path(path).expanduser().resolve()) for path in bootstrap_paths]
    refreshed["message"] = (
        "bootstrapped from local seed labels"
        if bootstrap.get("created")
        else refreshed.get("message", "")
    )
    return refreshed


def train_local_filter_model(payload: dict) -> dict:
    labels_path = resolve_app_path(str(payload.get("labels_path") or ""), DEFAULT_REVIEW_LABELS_PATH)
    model_out = resolve_app_path(str(payload.get("model_out") or ""), DEFAULT_FILTER_MODEL_PATH)
    cards_dir = resolve_app_path(str(payload.get("cards_dir") or ""), DEFAULT_SEARCH_CARDS_DIR)
    min_labeled_rows = int_payload(payload, "min_labeled_rows", 10)
    min_card_labeled_rows = int_payload(payload, "min_card_labeled_rows", 3)
    label_paths = [labels_path]
    if bool_payload(payload, "include_problem_reviews", False):
        job_id = str(payload.get("job_id") or "").strip()
        job_dir = safe_job_dir(job_id)
        if not job_dir:
            job_dir = latest_job_dir_with_training_reviews()
        if job_dir:
            for candidate_names in (
                ("人工反馈标注.csv", "human_feedback_labels.csv"),
                ("人工抽样标注表.csv", "training_review_sample.csv"),
                ("问题页面核验清单.csv", "problem_pages_review.csv"),
            ):
                for candidate_name in candidate_names:
                    candidate = artifact_path(job_dir, candidate_name) or (job_dir / candidate_name)
                    if candidate.exists() and candidate not in label_paths:
                        label_paths.append(candidate)
                        break
    rows = build_training_rows(label_paths)
    model = train_filter_model(rows, min_labeled_rows=min_labeled_rows)
    if model_out.suffix.lower() == ".pt":
        checkpoint = save_model_checkpoint_pt(model, model_out)
        checkpoint_path = model_out
    else:
        save_filter_model(model, model_out)
        checkpoint_path = model_out.with_suffix(".pt")
        checkpoint = save_model_checkpoint_pt(model, checkpoint_path)
    checkpoint_alias_path = checkpoint_path.with_name("本地筛选模型.pt")
    save_model_checkpoint_pt(model, checkpoint_alias_path)
    weights_path = model_out.with_name("filter_weights.json")
    weights = save_model_weights(model, weights_path)
    weights_alias_path = weights_path.with_name("本地筛选模型权重.json")
    weights_alias_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = build_search_cards(rows, min_labeled_rows=min_card_labeled_rows)
    card_index = write_search_cards(cards, cards_dir)
    report = {
        "ok": True,
        "labels_path": str(labels_path),
        "label_paths": [str(path.resolve()) for path in label_paths],
        "model_path": str(model_out),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_alias_path": str(checkpoint_alias_path),
        "weights_path": str(weights_path),
        "weights_alias_path": str(weights_alias_path),
        "checkpoint_format": checkpoint["format_version"],
        "model_version": model.model_version,
        "created_at": model.created_at,
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
        "search_cards": {
            "cards_dir": str(cards_dir),
            "written_cards": len(cards),
            "card_keys": sorted(cards.keys()),
            "index_path": str(cards_dir / "index.json"),
            "index": card_index,
        },
        "weights": {
            "format_version": weights["format_version"],
            "compatible_hosts": weights["compatible_hosts"],
            "top_feature_count": {
                label: len(weights["feature_weights"].get(label, {}))
                for label in weights["feature_weights"]
            },
        },
    }
    report_path = model_out.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def start_job(payload: dict) -> Job:
    competitors = parse_competitors(str(payload.get("competitors", "")))
    own_product_name = str(payload.get("own_product_name") or "").strip()
    own_product_positioning = str(payload.get("own_product_positioning") or "").strip()
    own_product_context = str(payload.get("own_product_context") or "").strip()
    if not competitors and not (own_product_name or own_product_positioning or own_product_context):
        raise ValueError("请至少输入一个竞品，或填写我方产品信息用于自动发现竞品。")
    proxy_url = str(payload.get("proxy_url") or "").strip()
    proxy = check_proxy_endpoint(proxy_url)
    if proxy_url and not proxy["ok"]:
        raise ValueError(f"代理不可用：{proxy['message']}。请启动代理，或清空代理地址后再开始采集。")

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = int_payload(payload, "per_query", 20)
    max_pages = int_payload(payload, "max_pages", 30)
    max_discovered_competitors = int_payload(payload, "max_discovered_competitors", 6)
    max_image_downloads = int_payload(payload, "max_image_downloads", 40)
    experiment_minutes = int_payload(payload, "experiment_minutes", 0)
    if experiment_minutes not in {0, 15, 30}:
        raise ValueError("实验限时只支持不设限、15 分钟或 30 分钟。")
    if bool_payload(payload, "broad_crawl"):
        per_query = max(per_query, 20)
        max_pages = max(max_pages, 30)
        max_image_downloads = max(max_image_downloads, 40)

    cmd = [
        worker_python(),
        str(SCRIPT_PATH),
        *competitors,
        "--searxng-url",
        str(payload.get("searxng_url") or "http://localhost:8888"),
        "--out",
        str(out_dir),
        "--per-query",
        str(per_query),
        "--max-pages",
        str(max_pages),
        "--crawl-concurrency",
        str(int_payload(payload, "crawl_concurrency", 3)),
        "--max-discovered-competitors",
        str(max_discovered_competitors),
        "--login-assist-wait",
        str(int_payload(payload, "login_assist_wait", 120)),
        "--image-engine",
        str(payload.get("image_engine") or "bing"),
        "--max-image-downloads",
        str(max_image_downloads),
    ]
    if proxy_url:
        cmd += ["--proxy-url", proxy_url]
    if own_product_name:
        cmd += ["--own-product-name", own_product_name]
    if own_product_positioning:
        cmd += ["--own-product-positioning", own_product_positioning]
    if own_product_context:
        cmd += ["--own-product-context", own_product_context]
    for term in parse_competitors(str(payload.get("manual_include_keywords") or "")):
        cmd += ["--manual-include-keyword", term]
    terms = str(payload.get("image_terms") or "").replace(",", "\n").splitlines()
    for term in [t.strip() for t in terms if t.strip()]:
        cmd += ["--image-extra-term", term]
    if bool_payload(payload, "skip_gui_review"):
        cmd.append("--skip-gui-review")
    if bool_payload(payload, "login_assist"):
        cmd.append("--login-assist")
    if bool_payload(payload, "skip_crawl"):
        cmd.append("--skip-crawl")
    if bool_payload(payload, "skip_images"):
        cmd.append("--skip-images")
    if bool_payload(payload, "no_cn"):
        cmd.append("--no-cn")
    if bool_payload(payload, "codex_review"):
        cmd.append("--codex-review")
        cmd.append("--require-codex-review")
        codex_command = resolve_executable_command(str(payload.get("codex_command") or os.getenv("CODEX_COMMAND", "codex")))
        if codex_command:
            cmd += ["--codex-command", codex_command]
        codex_model = str(payload.get("codex_model") or "").strip()
        if codex_model:
            cmd += ["--codex-model", codex_model]
    ml_model_path = str(payload.get("ml_model_path") or "").strip()
    if ml_model_path:
        cmd += ["--ml-model", ml_model_path]
    else:
        model_status_for_ui(DEFAULT_FILTER_MODEL_PATH)
    if not bool_payload(payload, "use_ml_filter", True):
        cmd.append("--disable-ml-filter")

    job = Job(
        id=job_id,
        status="queued",
        command=cmd,
        out_dir=out_dir,
        proxy_url=proxy_url,
        experiment_minutes=experiment_minutes,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(target=run_job, args=(job,), daemon=True)
    thread.start()
    return job


def check_environment(searxng_url: str, proxy_url: str = "") -> dict:
    import urllib.error
    import urllib.request

    proxy = check_proxy_endpoint(proxy_url)
    request_url = searxng_url.rstrip("/") + "/config"
    searxng = {"ok": False, "message": ""}
    try:
        req = urllib.request.Request(request_url, headers={"Accept": "application/json"})
        if proxy_url and not is_local_url(searxng_url):
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            )
            response_context = opener.open(req, timeout=5)
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            response_context = opener.open(req, timeout=5)
        with response_context as response:
            json.loads(response.read().decode("utf-8"))
        searxng = {"ok": True, "message": request_url}
    except Exception as exc:
        searxng = {"ok": False, "message": str(exc)}

    search_probe = {"ok": False, "message": "skipped because /config failed", "result_count": 0}
    if searxng["ok"]:
        probe_url = searxng_url.rstrip("/") + "/search?" + urlencode(
            {"q": "competitor pricing", "format": "json", "categories": "general", "language": "all"}
        )
        try:
            req = urllib.request.Request(probe_url, headers={"Accept": "application/json"})
            if proxy_url and not is_local_url(searxng_url):
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
                )
                response_context = opener.open(req, timeout=8)
            else:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                response_context = opener.open(req, timeout=8)
            with response_context as response:
                payload = json.loads(response.read().decode("utf-8"))
            result_count = len(payload.get("results") or [])
            search_probe = {
                "ok": result_count > 0,
                "message": f"{probe_url} returned {result_count} results",
                "result_count": result_count,
            }
        except Exception as exc:
            search_probe = {"ok": False, "message": str(exc), "result_count": 0}

    py = worker_python()
    check_code = (
        "import importlib.util,json,sys;"
        "print(json.dumps({"
        "'executable':sys.executable,"
        "'crawl4ai':importlib.util.find_spec('crawl4ai') is not None,"
        "'icrawler':importlib.util.find_spec('icrawler') is not None"
        "}))"
    )
    try:
        proc = subprocess.run([py, "-c", check_code], text=True, capture_output=True, timeout=10)
        python = json.loads(proc.stdout) if proc.returncode == 0 else {
            "executable": py,
            "crawl4ai": False,
            "icrawler": False,
            "error": proc.stderr,
        }
    except Exception as exc:
        python = {"executable": py, "crawl4ai": False, "icrawler": False, "error": str(exc)}
    codex_path = resolve_executable_command(os.getenv("CODEX_COMMAND", "codex"))
    if codex_path:
        try:
            proc = subprocess.run([codex_path, "--version"], text=True, capture_output=True, timeout=5, env={**os.environ, "PATH": expanded_path_env()})
            codex = {"ok": proc.returncode == 0, "message": (proc.stdout or proc.stderr).strip(), "path": codex_path}
        except Exception as exc:
            codex = {"ok": False, "message": str(exc), "path": codex_path}
    else:
        codex = {"ok": False, "message": "codex command not found", "path": ""}
    return {
        "ok": bool(searxng["ok"] and search_probe["ok"] and proxy["ok"] and python["crawl4ai"] and python["icrawler"] and codex["ok"]),
        "proxy_url": proxy_url,
        "proxy": proxy,
        "searxng": searxng,
        "searxng_search_probe": search_probe,
        "python": python,
        "codex": codex,
        "local_filter_model": model_status_for_ui(DEFAULT_FILTER_MODEL_PATH),
    }


def subprocess_env(proxy_url: str = "") -> dict:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["PATH"] = expanded_path_env()
    codex_command = resolve_executable_command(env.get("CODEX_COMMAND", "codex"))
    if codex_command:
        env["CODEX_COMMAND"] = codex_command
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    if proxy_url:
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["ALL_PROXY"] = proxy_url
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
        env["all_proxy"] = proxy_url
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def ui_log_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("$ "):
        return "$ 开始运行采集任务；完整命令和底层日志已写入 run.log。\n\n"
    if stripped.startswith(("[1/5]", "[1.5/5]", "[2/5]", "[2.5/5]", "[3/5]", "[4/5]", "[4.5/5]", "[5/5]", "[error]")):
        return line
    keep_fragments = (
        "web results:",
        "image results:",
        "crawled pages:",
        "downloaded images:",
        "manual review candidates:",
        "login-required candidates:",
        "training review sample:",
        "problem review rows:",
        "Codex review:",
        "SearXNG images category",
        "Output:",
        "proxy:",
        "login queue:",
        "[LOGIN]",
    )
    if any(fragment in stripped for fragment in keep_fragments):
        return line
    nonfatal_page_error_fragments = (
        "[ANTIBOT]",
        "Blocked by anti-bot protection",
        "Cloudflare JS challenge",
        "DataDome captcha",
        "HTTP 403",
        "Structural: minimal_text",
        "script_heavy_shell",
        "Failed on navigating ACS-GOTO",
    )
    if any(fragment in stripped for fragment in nonfatal_page_error_fragments):
        return None
    noisy_fragments = (
        "[FETCH]",
        "[SCRAPE]",
        "[COMPLETE]",
        "[INIT]",
        "icrawler.",
        "downloader -",
        "parser -",
        "feeder -",
        "urllib3.connectionpool",
        "DEBUG:",
        "INFO:",
    )
    if any(fragment in stripped for fragment in noisy_fragments):
        return None
    if "[warn]" in stripped and "icrawler" not in stripped:
        return line
    if "Traceback" in stripped or "Error" in stripped or "failed" in stripped.lower():
        return line
    return None


def signal_process_group(pid: int, sig: int) -> None:
    if not pid:
        raise ValueError("任务进程还没有启动。")
    if os.name == "posix":
        os.killpg(os.getpgid(pid), sig)
    else:
        os.kill(pid, sig)


def control_job(job_id: str, action: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise KeyError("job not found")
        if job.status in {"done", "failed", "terminated"}:
            return job_snapshot(job)
        pid = job.process_pid
        if not pid:
            if action == "terminate":
                job.terminate_requested = True
                job.status = "terminated"
                job.finished_at = time.time()
                job.logs.append("[ui] 任务已在启动前终止。\n")
                write_timing_artifacts(job)
                return job_snapshot(job)
            raise ValueError("任务进程还没有启动。")
        if action == "pause":
            if job.status != "running":
                return job_snapshot(job)
            if not hasattr(signal, "SIGSTOP"):
                raise ValueError("当前系统不支持暂停进程。")
            signal_process_group(pid, signal.SIGSTOP)
            job.status = "paused"
            job.logs.append("[ui] 任务已暂停。\n")
        elif action == "resume":
            if job.status != "paused":
                return job_snapshot(job)
            if not hasattr(signal, "SIGCONT"):
                raise ValueError("当前系统不支持继续进程。")
            signal_process_group(pid, signal.SIGCONT)
            job.status = "running"
            job.logs.append("[ui] 任务已继续。\n")
        elif action == "terminate":
            job.terminate_requested = True
            job.status = "stopping"
            job.logs.append("[ui] 正在终止任务。\n")
            try:
                signal_process_group(pid, signal.SIGTERM)
            finally:
                if os.name == "posix":
                    try:
                        signal_process_group(pid, signal.SIGCONT)
                    except OSError:
                        pass
        else:
            raise ValueError("unsupported job action")
        return job_snapshot(job)


def run_job(job: Job) -> None:
    with JOBS_LOCK:
        if job.terminate_requested:
            job.status = "terminated"
            job.finished_at = time.time()
            job.returncode = None
            write_timing_artifacts(job)
            return
        job.started_at = time.time()
        job.status = "running"
        job.logs.append("$ 开始运行采集任务；完整命令和底层日志已写入 run.log。\n\n")
    log_path = job.out_dir / "run.log"
    log_path.write_text("$ " + " ".join(job.command) + "\n\n", encoding="utf-8")
    try:
        process = subprocess.Popen(
            job.command,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=subprocess_env(job.proxy_url),
            start_new_session=True,
        )
        with JOBS_LOCK:
            job.process_pid = process.pid
        assert process.stdout is not None
        for line in process.stdout:
            display_line = ui_log_line(line)
            if display_line:
                with JOBS_LOCK:
                    job.logs.append(display_line)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        code = process.wait()
        with JOBS_LOCK:
            job.returncode = code
            job.finished_at = time.time()
            if job.terminate_requested or code in {-signal.SIGTERM, -signal.SIGKILL}:
                job.status = "terminated"
            else:
                job.status = "done" if code == 0 else "failed"
            elapsed_label = job_timing_snapshot(job)["elapsed_label"]
            job.logs.append(f"\n[ui] Process exited with code {code}.\n")
            job.logs.append(f"[ui] 全流程耗时：{elapsed_label}.\n")
        write_timing_artifacts(job)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ui] Process exited with code {code}.\n")
            handle.write(f"[ui] 全流程耗时：{elapsed_label}.\n")
        archive_internal_outputs(job.out_dir, keep_run_log=False)
    except Exception as exc:
        with JOBS_LOCK:
            job.status = "failed"
            job.finished_at = time.time()
            elapsed_label = job_timing_snapshot(job)["elapsed_label"]
            job.logs.append(f"\n[ui] Failed to run job: {exc}\n")
            job.logs.append(f"[ui] 全流程耗时：{elapsed_label}.\n")
        write_timing_artifacts(job)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ui] Failed to run job: {exc}\n")
            handle.write(f"[ui] 全流程耗时：{elapsed_label}.\n")
        archive_internal_outputs(job.out_dir, keep_run_log=False)


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            text_response(self, INDEX_HTML)
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = unquote(parsed.path.rsplit("/", 1)[-1])
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    snapshot = disk_job_snapshot(job_id)
                    if not snapshot:
                        json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                        return
                    json_response(self, snapshot)
                    return
                snapshot = job_snapshot(job)
            json_response(self, snapshot)
            return
        if parsed.path == "/download":
            params = parse_qs(parsed.query)
            job_id = params.get("job", [""])[0]
            filename = params.get("file", [""])[0]
            self.serve_download(job_id, filename)
            return
        if parsed.path == "/api/login/open":
            params = parse_qs(parsed.query)
            job_id = params.get("job", [""])[0]
            competitor = params.get("competitor", [""])[0]
            url = params.get("url", [""])[0]
            try:
                payload = record_login_open_request(job_id, competitor, url)
            except Exception as exc:
                text_response(self, f"登录请求失败：{exc}", "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST)
                return
            escaped_url = html.escape(payload["url"], quote=True)
            html_body = (
                "<!doctype html><meta charset='utf-8'>"
                "<title>登录请求已发送</title>"
                "<body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.6;padding:24px;'>"
                "<h2>登录请求已发送</h2>"
                "<p>已记录你的点击。采集进程只会对这个站点放行，并在等待期内打开工具浏览器。</p>"
                "<p>请在弹出的工具浏览器中使用你有权限的账号登录。未点击的登录站点不会被主动打开。</p>"
                "<p>如果任务已经结束，可以回到主页面重新运行，或手动补充公开截图和摘录。</p>"
                f"<p><a href='{escaped_url}' target='_blank' rel='noopener noreferrer'>手动打开原网页</a></p>"
                "</body>"
            )
            text_response(self, html_body)
            return
        if parsed.path == "/api/check":
            params = parse_qs(parsed.query)
            searxng_url = params.get("searxng_url", ["http://localhost:8888"])[0]
            proxy_url = params.get("proxy_url", [""])[0]
            json_response(self, check_environment(searxng_url, proxy_url))
            return
        if parsed.path == "/api/ml/status":
            params = parse_qs(parsed.query)
            model_path = resolve_app_path(params.get("model_path", [""])[0], DEFAULT_FILTER_MODEL_PATH)
            json_response(self, model_status_for_ui(model_path))
            return
        text_response(self, "Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        job_action_match = re.fullmatch(r"/api/jobs/([^/]+)/(pause|resume|terminate)", parsed.path)
        if job_action_match:
            job_id = unquote(job_action_match.group(1))
            action = job_action_match.group(2)
            try:
                json_response(self, control_job(job_id, action))
            except KeyError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/ml/train":
            try:
                payload = json.loads(body or "{}")
                json_response(self, train_local_filter_model(payload), HTTPStatus.CREATED)
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/review/feedback":
            try:
                payload = json.loads(body or "{}")
                json_response(self, record_feedback_review(payload), HTTPStatus.CREATED)
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path in {"/api/login/open", "/api/login/skip"}:
            try:
                payload = json.loads(body or "{}")
                job_id = payload.get("job") or payload.get("job_id") or ""
                competitor = payload.get("competitor") or ""
                url = payload.get("url") or ""
                if parsed.path.endswith("/skip"):
                    result = record_login_skip_request(job_id, competitor, url)
                    result["ok"] = True
                    result["message"] = "login skipped"
                else:
                    result = record_login_open_request(job_id, competitor, url)
                    result["ok"] = True
                    result["message"] = "login requested"
                json_response(self, result)
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path != "/api/jobs":
            json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = json.loads(body or "{}")
            job = start_job(payload)
            with JOBS_LOCK:
                snapshot = job_snapshot(job)
            json_response(self, snapshot, HTTPStatus.CREATED)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def serve_download(self, job_id: str, filename: str) -> None:
        if filename not in ALL_DOWNLOADABLE_ARTIFACTS:
            text_response(self, "Invalid file", "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST)
            return
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job:
            out_dir = job.out_dir
        else:
            out_dir = safe_job_dir(job_id)
            if not out_dir:
                text_response(self, "Job not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
        path = artifact_path(out_dir, filename)
        if not path:
            text_response(self, "File not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        content_type = "text/plain; charset=utf-8"
        if filename.endswith(".csv"):
            content_type = "text/csv; charset=utf-8"
        elif filename.endswith(".json"):
            content_type = "application/json; charset=utf-8"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        ascii_fallback = filename.encode("ascii", "ignore").decode("ascii") or "download"
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(filename)}',
        )
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[ui] " + fmt % args + "\n")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run local UI for Competitor Intel Harvester.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Competitor Intel Harvester UI: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
