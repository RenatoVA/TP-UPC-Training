#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from legal_data.paths import PROCESSED_DIR
from legal_data.storage import write_json

INPUT_JSON = PROCESSED_DIR / "jurisprudence_entry_content.json"
OUTPUT_JSON = PROCESSED_DIR / "jurisprudence_context_units.json"

SYSTEM_PROMPT = """Eres un analista jurídico. Recibirás el contenido textual de una publicación de jurisprudencia de LP Derecho. Debes sintetizarla para retrieval jurídico en contexto de violencia familiar y proceso penal. Responde SOLO en JSON válido con estas claves: title, document_kind, facts_summary, legal_issue, holding, reasoning_summary, outcome, applicable_norms, violence_family_relevance, procedural_relevance, practical_context_summary, keywords, llm_relevance_score, needs_review. No inventes hechos ni artículos no presentes o claramente inferibles del texto."""


def summarize_entry(client: OpenAI, model: str, item: dict) -> dict:
    payload = {
        "entry_title_raw": item.get("entry_title_raw"),
        "document_kind_hint": item.get("document_kind"),
        "index_page_title": item.get("index_page_title"),
        "combined_text": (item.get("combined_text") or "")[:18000],
        "provenance_sources": item.get("provenance_sources") or [],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        extra_headers={
            "HTTP-Referer": "https://openclaw.local",
            "X-Title": os.environ.get("APP_NAME", "peru-legal-rag-data"),
        },
    )
    content = response.choices[0].message.content if response.choices and response.choices[0].message else None
    if not content:
        raise RuntimeError("empty model response")
    parsed = json.loads(content)
    return {
        "entry_id": item.get("entry_id"),
        "index_page_id": item.get("index_page_id"),
        "entry_url": item.get("entry_url"),
        "pdf_url": item.get("pdf_url"),
        "provenance_sources": item.get("provenance_sources") or [],
        "context_unit": parsed,
    }


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
    items = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    outputs = []
    for item in items:
        if item.get("content_status") != "success" or not item.get("combined_text"):
            outputs.append({"entry_id": item.get("entry_id"), "index_page_id": item.get("index_page_id"), "entry_url": item.get("entry_url"), "pdf_url": item.get("pdf_url"), "provenance_sources": item.get("provenance_sources") or [], "context_unit": None, "status": "skipped"})
            continue
        try:
            outputs.append({**summarize_entry(client, model, item), "status": "success"})
        except Exception as exc:
            outputs.append({"entry_id": item.get("entry_id"), "index_page_id": item.get("index_page_id"), "entry_url": item.get("entry_url"), "pdf_url": item.get("pdf_url"), "provenance_sources": item.get("provenance_sources") or [], "context_unit": None, "status": f"error: {exc}"})
    write_json(OUTPUT_JSON, outputs)
    print(json.dumps({"entries_total": len(outputs), "summarized_success": sum(1 for x in outputs if x.get('status') == 'success'), "output_path": str(OUTPUT_JSON)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
