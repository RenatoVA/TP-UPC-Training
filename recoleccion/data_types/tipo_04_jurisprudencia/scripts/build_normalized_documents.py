#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from legal_data.paths import PROCESSED_DIR
from legal_data.storage import write_text

INPUT_JSON = PROCESSED_DIR / "jurisprudence_entry_detail_pages_10.json"
OUTPUT_JSON = PROCESSED_DIR / "normalized_documents.jsonl"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_title(item: dict[str, Any]) -> str:
    return item.get("detail_page_title") or item.get("entry_title_raw") or item.get("entry_id") or "sin_titulo"


def _build_topic_tags(item: dict[str, Any]) -> list[str]:
    tags = ["jurisprudencia", "lp_derecho"]
    article_id = item.get("article_id")
    entry_category = item.get("entry_category")
    if article_id:
        tags.append(f"articulo_{str(article_id).lower().replace(' ', '_')}")
    if entry_category:
        tags.append(str(entry_category).lower().replace(" ", "_"))
    return tags


def _normalized_doc(item: dict[str, Any]) -> dict[str, Any]:
    doc_id = f"tipo04_{item['entry_id']}"
    source_url = item.get("entry_url") or item.get("index_page_url")
    title = _normalize_title(item)
    markdown = item.get("detail_markdown")

    content_units = []
    if markdown:
        content_units.append(
            {
                "unit_id": f"{doc_id}_clean_markdown",
                "unit_type": "clean_markdown",
                "text": markdown,
                "order": 1,
            }
        )

    return {
        "doc_id": doc_id,
        "doc_type": "jurisprudencia_lp",
        "source_type": "tipo_04_jurisprudencia",
        "source_trace": {
            "source_url": source_url,
            "source_path": item.get("detail_raw_html_path"),
            "source_entity": "lpderecho",
            "index_page_id": item.get("index_page_id"),
            "index_page_url": item.get("index_page_url"),
            "entry_id": item.get("entry_id"),
            "entry_url": item.get("entry_url"),
            "pdf_url": None,
            "pdf_sha256": None,
            "collected_at": _now_iso(),
        },
        "document_metadata": {
            "title": title,
            "short_title": item.get("entry_title_raw"),
            "language": "es",
            "jurisdiction": "Perú",
            "issuing_body": item.get("entry_category"),
            "publication_date": None,
            "document_date_text": None,
            "resolution_reference": item.get("resolution_reference_raw"),
            "resolution_type": None,
            "article_id": item.get("article_id"),
            "article_heading": item.get("article_heading"),
            "chapter": None,
            "section": item.get("entry_category"),
            "topic_tags": _build_topic_tags(item),
        },
        "content_units": content_units,
        "normalization_signals": {
            "normalization_status": "success" if markdown else item.get("detail_fetch_status"),
            "summary_status": "pending",
            "chunking_ready": bool(markdown),
            "needs_review": item.get("detail_fetch_status") != "success" or not markdown,
            "notes": item.get("detail_fetch_status"),
        },
    }


def main() -> None:
    items = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    docs = [_normalized_doc(item) for item in items]
    lines = "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n"
    write_text(OUTPUT_JSON, lines)
    print(
        json.dumps(
            {
                "documents_total": len(docs),
                "chunking_ready": sum(1 for d in docs if d["normalization_signals"]["chunking_ready"]),
                "needs_review": sum(1 for d in docs if d["normalization_signals"]["needs_review"]),
                "output_path": str(OUTPUT_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
