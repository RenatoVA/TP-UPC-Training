#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from legal_data.paths import PROCESSED_DIR, RAW_DIR
from legal_data.storage import write_json, write_text

INPUT_JSON = PROCESSED_DIR / "jurisprudence_entries_deduped.json"
OUTPUT_JSON = PROCESSED_DIR / "jurisprudence_entry_content.json"
RAW_HTML_DIR = RAW_DIR / "entry_detail_html_full"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SKIP_NON_HTML_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.td-post-content") or soup.body or soup
    lines = []
    for tag in content.find_all(["h2", "h3", "h4", "p", "li", "strong"], limit=500):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def _is_html_candidate(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.lower().split("?", 1)[0]
    return not any(lowered.endswith(ext) for ext in SKIP_NON_HTML_EXTENSIONS)


def main() -> None:
    items = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    results = []
    for item in items:
        entry_url = item.get("entry_url")
        if not _is_html_candidate(entry_url):
            results.append({**item, "content_status": "no_html_detail", "detail_raw_html_path": None, "detail_text": None, "detail_sha256": None, "best_source_type": "none", "combined_text": None})
            continue
        try:
            response = requests.get(entry_url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            html = response.text
            raw_path = RAW_HTML_DIR / f"{item['entry_id']}.html"
            write_text(raw_path, html)
            detail_text = _html_to_text(html)
            results.append({**item, "content_status": "success", "detail_raw_html_path": str(raw_path), "detail_text": detail_text, "detail_sha256": _sha256_text(html), "best_source_type": "detail_html", "combined_text": detail_text})
        except Exception as exc:
            results.append({**item, "content_status": f"error: {exc}", "detail_raw_html_path": None, "detail_text": None, "detail_sha256": None, "best_source_type": "none", "combined_text": None})
    write_json(OUTPUT_JSON, results)
    print(json.dumps({"entries_total": len(results), "resolved_success": sum(1 for x in results if x.get('content_status') == 'success'), "output_path": str(OUTPUT_JSON)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
