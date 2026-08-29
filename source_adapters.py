#!/usr/bin/env python3
"""Public source adapters for video, social, app-store, and developer pages.

Adapters are deliberately conservative: they collect public metadata and
traceability hints. Login, paid, private, or captcha-gated content stays in the
review queues handled by the main harvester.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener


FetchFn = Callable[[str, int, str], Tuple[int, str, str]]
VideoMetadataFn = Callable[[str, int, str], Dict[str, Any]]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", textify(value)).strip()


def slugify(value: str) -> str:
    value = compact_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "source"


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def domain_matches(domain: str, patterns: Iterable[str]) -> bool:
    domain = domain.lower().removeprefix("www.")
    return any(domain == item or domain.endswith("." + item) for item in patterns)


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def default_fetch(url: str, timeout: int = 12, proxy_url: str = "") -> Tuple[int, str, str]:
    req = Request(
        url,
        headers={
            "User-Agent": "competitor-intel-harvester/1.0",
            "Accept": "application/json,text/html,text/plain,*/*",
        },
    )
    if proxy_url and not is_local_url(url):
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        opener = build_opener(ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(1_500_000)
        content_type = response.headers.get("content-type", "")
        status = getattr(response, "status", 200)
    encoding = "utf-8"
    enc_match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    if enc_match:
        encoding = enc_match.group(1)
    return int(status), content_type, raw.decode(encoding, errors="replace")


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


def adapter_search_templates(name: str, category: str = "") -> List[str]:
    base = [
        f"{name} site:youtube.com demo review",
        f"{name} site:bilibili.com 评测 演示",
        f"{name} site:reddit.com review problem",
        f"{name} site:zhihu.com 评价 对比",
        f"{name} site:producthunt.com launch",
        f"{name} site:apps.apple.com reviews screenshots",
        f"{name} site:play.google.com reviews screenshots",
        f"{name} site:chromewebstore.google.com reviews screenshots",
        f"{name} site:x.com product launch",
        f"{name} site:tiktok.com review",
        f"{name} site:douyin.com 评测",
        f"{name} site:xiaohongshu.com 评价",
        f"{name} site:weixin.qq.com 评测",
    ]
    if category == "ai_software":
        base.extend(
            [
                f"{name} site:github.com docs API",
                f"{name} site:npmjs.com package SDK",
                f"{name} site:github.com integrations",
            ]
        )
    if category in {"physical_product", "snow_helmet"}:
        base.extend(
            [
                f"{name} site:youtube.com fit review",
                f"{name} site:reddit.com durability quality",
            ]
        )
    return unique_strings(base)


def classify_source_url(url: str) -> Dict[str, str]:
    domain = domain_of(url)
    parsed = urlparse(url)
    path = parsed.path.lower()
    if domain_matches(domain, {"youtube.com", "youtu.be"}):
        return {"adapter_name": "youtube", "source_family": "video_social", "platform": "YouTube"}
    if domain_matches(domain, {"bilibili.com"}):
        return {"adapter_name": "bilibili", "source_family": "video_social", "platform": "Bilibili"}
    if domain_matches(domain, {"tiktok.com", "douyin.com"}):
        return {"adapter_name": "short_video", "source_family": "video_social", "platform": domain}
    if domain_matches(domain, {"instagram.com", "x.com", "twitter.com", "xiaohongshu.com", "weixin.qq.com"}):
        return {"adapter_name": "social_public", "source_family": "social_app", "platform": domain}
    if domain_matches(domain, {"apps.apple.com"}) and "/app/" in path:
        return {"adapter_name": "apple_app_store", "source_family": "app_store", "platform": "Apple App Store"}
    if domain_matches(domain, {"play.google.com"}):
        return {"adapter_name": "google_play", "source_family": "app_store", "platform": "Google Play"}
    if domain_matches(domain, {"chromewebstore.google.com"}):
        return {"adapter_name": "chrome_web_store", "source_family": "app_store", "platform": "Chrome Web Store"}
    if domain_matches(domain, {"github.com"}):
        return {"adapter_name": "github", "source_family": "developer_source", "platform": "GitHub"}
    if domain_matches(domain, {"producthunt.com"}):
        return {"adapter_name": "product_hunt", "source_family": "launch_database", "platform": "Product Hunt"}
    if domain_matches(domain, {"reddit.com", "zhihu.com", "v2ex.com", "news.ycombinator.com"}):
        return {"adapter_name": "community_forum", "source_family": "forum_community", "platform": domain}
    return {"adapter_name": "", "source_family": "", "platform": ""}


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    if domain_matches(domain_of(url), {"youtu.be"}):
        return parsed.path.strip("/").split("/")[0]
    return (parse_qs(parsed.query).get("v") or [""])[0]


def apple_app_id(url: str) -> str:
    match = re.search(r"/id(\d+)", urlparse(url).path)
    return match.group(1) if match else ""


def google_play_app_id(url: str) -> str:
    return (parse_qs(urlparse(url).query).get("id") or [""])[0]


def github_repo(url: str) -> Tuple[str, str]:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def strip_html_snapshot(raw: str, limit: int = 6000) -> Tuple[str, str, List[str]]:
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    if title_match:
        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
    images = []
    for match in re.findall(r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"'][^>]+content=[\"']([^\"']+)[\"']", raw, re.I):
        images.append(html.unescape(match))
    body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(re.sub(r"\s+", " ", body)).strip()
    return title[:220], body[:limit], unique_strings(images, 8)


def extract_open_graph(raw: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for attr, value in re.findall(r"<meta[^>]+(?:property|name)=[\"']([^\"']+)[\"'][^>]+content=[\"']([^\"']*)[\"'][^>]*>", raw, re.I):
        key = attr.lower().replace("og:", "")
        if key in {"title", "description", "image", "site_name", "type"}:
            fields[key] = html.unescape(value)
    return fields


def parse_timestamp(value: str) -> Optional[int]:
    value = value.strip().lower()
    url_t = re.fullmatch(r"(\d+)\s*s?", value)
    if url_t:
        return int(url_t.group(1))
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return None


def format_timestamp(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_video_evidence_markers(text: str) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    seen = set()
    for match in re.finditer(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)", textify(text)):
        stamp = match.group(0)
        seconds = parse_timestamp(stamp)
        if seconds is None or seconds in seen:
            continue
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        seen.add(seconds)
        markers.append(
            {
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
                "context": compact_text(text[start:end]),
            }
        )
    for match in re.finditer(r"[?&](?:t|start)=(\d+)s?", textify(text), re.I):
        seconds = int(match.group(1))
        if seconds in seen:
            continue
        seen.add(seconds)
        markers.insert(
            0,
            {
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
                "context": "timestamp from URL",
            },
        )
    return markers


def ytdlp_available() -> bool:
    try:
        import yt_dlp  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def extract_ytdlp_metadata(url: str, timeout: int = 20, proxy_url: str = "") -> Dict[str, Any]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        return {"yt_dlp_status": "not_installed", "yt_dlp_error": str(exc)}

    options: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "ignoreerrors": True,
        "extract_flat": False,
    }
    if proxy_url:
        options["proxy"] = proxy_url
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if not isinstance(info, dict):
            return {"yt_dlp_status": "empty", "yt_dlp_error": "yt-dlp returned no metadata"}
        info["yt_dlp_status"] = "ok"
        return info
    except Exception as exc:
        return {"yt_dlp_status": "failed", "yt_dlp_error": str(exc)}


def compact_ytdlp_fields(info: Mapping[str, Any], fallback_title: str = "") -> Dict[str, Any]:
    subtitles = info.get("subtitles") if isinstance(info.get("subtitles"), dict) else {}
    automatic_captions = info.get("automatic_captions") if isinstance(info.get("automatic_captions"), dict) else {}
    chapters = info.get("chapters") if isinstance(info.get("chapters"), list) else []
    return {
        "yt_dlp_status": info.get("yt_dlp_status", "ok"),
        "yt_dlp_error": info.get("yt_dlp_error", ""),
        "video_id": info.get("id", ""),
        "title": info.get("title") or fallback_title,
        "uploader": info.get("uploader") or info.get("channel") or "",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "duration": info.get("duration", ""),
        "upload_date": info.get("upload_date", ""),
        "view_count": info.get("view_count", ""),
        "like_count": info.get("like_count", ""),
        "webpage_url": info.get("webpage_url", ""),
        "thumbnail_url": info.get("thumbnail", ""),
        "description": compact_text(info.get("description", ""))[:2500],
        "chapters": chapters[:20],
        "tags": info.get("tags", [])[:20] if isinstance(info.get("tags"), list) else [],
        "categories": info.get("categories", [])[:10] if isinstance(info.get("categories"), list) else [],
        "subtitles": sorted(subtitles.keys())[:20],
        "automatic_captions": sorted(automatic_captions.keys())[:20],
    }


def chapter_markers(chapters: Any) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(chapters, list):
        return markers
    for chapter in chapters:
        if not isinstance(chapter, Mapping):
            continue
        start = chapter.get("start_time")
        if start is None:
            continue
        try:
            seconds = int(float(start))
        except (TypeError, ValueError):
            continue
        if seconds in seen:
            continue
        seen.add(seconds)
        markers.append(
            {
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
                "context": compact_text(chapter.get("title") or "chapter marker"),
            }
        )
    return markers


def merge_video_markers(*marker_groups: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in marker_groups:
        for marker in group:
            seconds = marker.get("timestamp_seconds")
            if seconds is None:
                continue
            try:
                key = int(seconds)
            except (TypeError, ValueError):
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "timestamp": textify(marker.get("timestamp")) or format_timestamp(key),
                    "timestamp_seconds": key,
                    "context": compact_text(marker.get("context")),
                }
            )
    return sorted(merged, key=lambda item: int(item.get("timestamp_seconds") or 0))


def write_video_evidence_file(path: Path, fields: Mapping[str, Any], markers: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Video evidence metadata",
        f"Title: {fields.get('title', '')}",
        f"Uploader: {fields.get('uploader', '')}",
        f"URL: {fields.get('webpage_url', '')}",
        f"Duration: {fields.get('duration', '')}",
        f"Thumbnail: {fields.get('thumbnail_url', '')}",
        "",
        "Description:",
        textify(fields.get("description", "")),
        "",
        "Markers:",
    ]
    for marker in markers:
        lines.append(f"- {marker.get('timestamp')}: {marker.get('context')}")
    lines += [
        "",
        "Subtitle availability:",
        f"- subtitles: {', '.join(fields.get('subtitles') or [])}",
        f"- automatic_captions: {', '.join(fields.get('automatic_captions') or [])}",
        "",
        "Note: this file stores public metadata only; it is not a downloaded video.",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def write_text_snapshot(path: Path, sections: Mapping[str, Any]) -> str:
    lines = []
    for key, value in sections.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {compact_text(value)}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8", errors="replace")


def base_result(url: str, title: str, snippet: str, out_dir: Path, slug: str) -> Dict[str, Any]:
    info = classify_source_url(url)
    return {
        "handled": "yes" if info["adapter_name"] else "no",
        "adapter_name": info["adapter_name"],
        "source_family": info["source_family"],
        "platform": info["platform"],
        "canonical_url": url,
        "automated_review_status": "",
        "metadata_path": "",
        "text_snapshot_path": "",
        "screenshot_path": "",
        "transcript_path": "",
        "evidence_markers_path": "",
        "needs_manual_video_timestamp": "no",
        "text_snapshot_excerpt": compact_text(f"{title} {snippet}")[:900],
        "adapter_next_step": "",
    }


def json_payload(content_type: str, body: str) -> Dict[str, Any]:
    if "json" not in content_type.lower():
        return {}
    try:
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def collect_adapter_snapshot(
    url: str,
    title: str = "",
    snippet: str = "",
    out_dir: Union[Path, str] = ".",
    slug: str = "source",
    fetcher: Optional[FetchFn] = None,
    video_metadata_extractor: Optional[VideoMetadataFn] = None,
    timeout: int = 12,
    proxy_url: str = "",
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    snapshot_dir = out_dir / "gui_review_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(slug)
    result = base_result(url, title, snippet, out_dir, slug)
    if result["handled"] != "yes":
        result["automated_review_status"] = "adapter_not_available"
        return result

    fetch = fetcher or default_fetch
    metadata: Dict[str, Any] = {
        "url": url,
        "title": title,
        "snippet": snippet,
        "adapter_name": result["adapter_name"],
        "source_family": result["source_family"],
        "platform": result["platform"],
        "captured_at": utc_stamp(),
        "fields": {},
    }
    text_sections: Dict[str, Any] = {
        "URL": url,
        "Platform": result["platform"],
        "Title": title,
        "Snippet": snippet,
    }

    try:
        adapter = result["adapter_name"]
        if adapter == "youtube":
            video_id = youtube_video_id(url)
            ytdlp_extract = video_metadata_extractor or extract_ytdlp_metadata
            ytdlp_info = ytdlp_extract(url, timeout, proxy_url)
            markers: List[Dict[str, Any]] = []
            if ytdlp_info.get("yt_dlp_status", "ok") == "ok":
                fields = compact_ytdlp_fields(ytdlp_info, title)
                fields["video_id"] = fields.get("video_id") or video_id
                metadata["fields"].update(fields)
                text_sections.update(fields)
                markers = merge_video_markers(
                    chapter_markers(ytdlp_info.get("chapters")),
                    extract_video_evidence_markers(" ".join([url, title, snippet, fields.get("title", ""), fields.get("description", "")])),
                )
                transcript_path = snapshot_dir / f"{slug}-video-evidence.txt"
                result["transcript_path"] = write_video_evidence_file(transcript_path, fields, markers)
            else:
                metadata["fields"].update(
                    {
                        "yt_dlp_status": ytdlp_info.get("yt_dlp_status", "failed"),
                        "yt_dlp_error": ytdlp_info.get("yt_dlp_error", ""),
                    }
                )
                oembed_url = "https://www.youtube.com/oembed?" + urlencode({"url": url, "format": "json"})
                status, content_type, body = fetch(oembed_url, timeout, proxy_url)
                payload = json_payload(content_type, body)
                metadata["fields"].update(
                    {
                        "video_id": video_id,
                        "oembed_status": status,
                        "title": payload.get("title") or title,
                        "author_name": payload.get("author_name", ""),
                        "thumbnail_url": payload.get("thumbnail_url", ""),
                    }
                )
                text_sections.update(metadata["fields"])
                marker_context = " ".join([url, title, snippet, payload.get("title", ""), payload.get("author_name", "")])
                markers = extract_video_evidence_markers(marker_context)
            if markers:
                markers_path = snapshot_dir / f"{slug}-video-markers.json"
                markers_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")
                result["evidence_markers_path"] = str(markers_path)
            result["needs_manual_video_timestamp"] = "no" if markers else "yes"
            result["adapter_next_step"] = (
                "已找到视频时间点线索；进入报告前仍建议核对公开画面或字幕。"
                if markers
                else "缺少观点时间点；需要人工补时间点、截图或公开字幕后再写入强事实。"
            )
        elif adapter == "apple_app_store":
            app_id = apple_app_id(url)
            lookup_url = "https://itunes.apple.com/lookup?" + urlencode({"id": app_id}) if app_id else url
            status, content_type, body = fetch(lookup_url, timeout, proxy_url)
            payload = json_payload(content_type, body)
            item = (payload.get("results") or [{}])[0] if payload else {}
            metadata["fields"].update(
                {
                    "app_id": app_id,
                    "lookup_status": status,
                    "track_name": item.get("trackName") or title,
                    "seller_name": item.get("sellerName", ""),
                    "version": item.get("version", ""),
                    "average_rating": item.get("averageUserRating", ""),
                    "rating_count": item.get("userRatingCount", ""),
                    "description": compact_text(item.get("description", ""))[:1200],
                    "screenshot_urls": item.get("screenshotUrls", [])[:8] if isinstance(item.get("screenshotUrls"), list) else [],
                }
            )
            text_sections.update(metadata["fields"])
            result["adapter_next_step"] = "已保存应用商店公开元数据；评分、版本、截图可作为 P1/P2 验证证据。"
        elif adapter == "github":
            owner, repo = github_repo(url)
            api_url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}" if owner and repo else url
            status, content_type, body = fetch(api_url, timeout, proxy_url)
            payload = json_payload(content_type, body)
            metadata["fields"].update(
                {
                    "repo": f"{owner}/{repo}".strip("/"),
                    "api_status": status,
                    "description": payload.get("description") or snippet,
                    "stars": payload.get("stargazers_count", ""),
                    "forks": payload.get("forks_count", ""),
                    "license": ((payload.get("license") or {}).get("spdx_id") if isinstance(payload.get("license"), dict) else ""),
                    "default_branch": payload.get("default_branch", ""),
                    "updated_at": payload.get("updated_at", ""),
                }
            )
            text_sections.update(metadata["fields"])
            result["adapter_next_step"] = "已保存 GitHub 公开仓库元数据；可用于开源活跃度、开发者生态和技术接口验证。"
        else:
            adapter_handled = False
            if result["source_family"] == "video_social":
                ytdlp_extract = video_metadata_extractor or extract_ytdlp_metadata
                ytdlp_info = ytdlp_extract(url, timeout, proxy_url)
                if ytdlp_info.get("yt_dlp_status", "ok") == "ok":
                    fields = compact_ytdlp_fields(ytdlp_info, title)
                    metadata["fields"].update(fields)
                    text_sections.update(fields)
                    markers = merge_video_markers(
                        chapter_markers(ytdlp_info.get("chapters")),
                        extract_video_evidence_markers(" ".join([url, title, snippet, fields.get("title", ""), fields.get("description", "")])),
                    )
                    if markers:
                        markers_path = snapshot_dir / f"{slug}-video-markers.json"
                        markers_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")
                        result["evidence_markers_path"] = str(markers_path)
                    transcript_path = snapshot_dir / f"{slug}-video-evidence.txt"
                    result["transcript_path"] = write_video_evidence_file(transcript_path, fields, markers)
                    result["needs_manual_video_timestamp"] = "no" if markers else "yes"
                    result["adapter_next_step"] = (
                        "yt-dlp 已保存公开视频元数据和时间点线索；进入报告前仍建议核对公开画面或字幕。"
                        if markers
                        else "yt-dlp 已保存公开视频元数据，但缺少观点时间点；需要人工补时间点、截图或公开字幕。"
                    )
                    result["automated_review_status"] = "adapter_metadata_captured"
                    adapter_handled = True
                else:
                    metadata["fields"].update(
                        {
                            "yt_dlp_status": ytdlp_info.get("yt_dlp_status", "failed"),
                            "yt_dlp_error": ytdlp_info.get("yt_dlp_error", ""),
                        }
                    )
            if not adapter_handled:
                status, content_type, body = fetch(url, timeout, proxy_url)
                payload = json_payload(content_type, body)
                og = extract_open_graph(body) if body else {}
                html_title, html_text, html_images = strip_html_snapshot(body) if body and "html" in content_type.lower() else ("", compact_text(body)[:6000], [])
                metadata["fields"].update(
                    {
                        "fetch_status": status,
                        "open_graph": og,
                        "html_title": html_title,
                        "image_urls": html_images,
                        "body_excerpt": html_text[:1200],
                        "json_payload_keys": sorted(payload.keys())[:40] if payload else [],
                    }
                )
                text_sections.update({"OpenGraph": og, "HTML title": html_title, "Body": html_text[:3000], "Images": html_images})
                if result["source_family"] == "video_social":
                    markers = extract_video_evidence_markers(" ".join([url, title, snippet, html_text]))
                    if markers:
                        markers_path = snapshot_dir / f"{slug}-video-markers.json"
                        markers_path.write_text(json.dumps(markers, ensure_ascii=False, indent=2), encoding="utf-8")
                        result["evidence_markers_path"] = str(markers_path)
                    result["needs_manual_video_timestamp"] = "no" if markers else "yes"
                result["adapter_next_step"] = "已保存该平台公开页面元数据；进入强事实前需核对原始公开页面和时间/作者/截图。"
    except Exception as exc:
        metadata["error"] = str(exc)
        result["automated_review_status"] = "adapter_metadata_failed"
        result["adapter_next_step"] = f"适配器抓取失败，保留人工 GUI 复核：{exc}"

    metadata_path = snapshot_dir / f"{slug}-adapter-metadata.json"
    text_path = snapshot_dir / f"{slug}-adapter-snapshot.txt"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot_text = write_text_snapshot(text_path, text_sections)
    result["metadata_path"] = str(metadata_path)
    result["text_snapshot_path"] = str(text_path)
    result["text_snapshot_excerpt"] = compact_text(snapshot_text)[:900]
    if not result["automated_review_status"]:
        result["automated_review_status"] = "adapter_metadata_captured"
    return result
