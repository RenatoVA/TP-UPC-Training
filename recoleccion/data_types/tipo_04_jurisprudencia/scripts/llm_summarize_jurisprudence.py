#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from legal_data.paths import PROCESSED_DIR
from legal_data.storage import write_json

INPUT_JSON = PROCESSED_DIR / "jurisprudence_entry_content.json"
OUTPUT_JSON = PROCESSED_DIR / "jurisprudence_context_units.json"


class ApplicableNorm(BaseModel):
    norm: str | None = None
    article: str | None = None


class AntecedentSignals(BaseModel):
    antecedent_relevance: Literal["yes", "no", "depends"]
    antecedent_richness: Literal["high", "medium", "low", "none"]
    has_factual_background: bool
    has_procedural_background: bool
    has_identifiable_parties: bool
    has_case_number_pattern: bool
    should_download_pdf: bool
    pdf_priority: Literal["high", "medium", "low", "none"]
    antecedent_reason: str


class JurisprudenceSummaryModel(BaseModel):
    title: str | None = None
    document_kind: str | None = None
    facts_summary: str | None = None
    legal_issue: str | None = None
    holding: str | None = None
    reasoning_summary: str | None = None
    outcome: str | None = None
    applicable_norms: list[ApplicableNorm] = Field(default_factory=list)
    violence_family_relevance: str | None = None
    procedural_relevance: str | None = None
    practical_context_summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    llm_relevance_score: float | None = None
    needs_review: bool = False
    detected_resolution_label: str | None = None
    detected_case_number: str | None = None
    pdf_url_mapped: str | None = None
    antecedent_signals: AntecedentSignals


SYSTEM_PROMPT = """Eres un analista jurídico. Recibirás el contenido textual de una publicación de jurisprudencia de LP Derecho. Debes sintetizarla para retrieval jurídico en contexto de violencia familiar y proceso penal y, al mismo tiempo, clasificar si vale la pena descargar/usar su PDF por riqueza de antecedentes fácticos o procesales. Regla clave: expedientes de salas superiores, juzgados mixtos/familia y juzgados penales suelen ser útiles para antecedentes; acuerdos plenarios y doctrina vinculante pura no. Casaciones y recursos de nulidad pueden depender del contenido real. Responde SOLO en JSON válido siguiendo exactamente el schema pedido. No inventes hechos, artículos, PDFs o números de caso."""


def summarize_entry(client: OpenAI, model: str, item: dict) -> dict:
    payload = {
        "entry_title_raw": item.get("entry_title_raw"),
        "document_kind_hint": item.get("document_kind"),
        "index_page_title": item.get("index_page_title"),
        "entry_url": item.get("entry_url"),
        "pdf_url_hint": item.get("pdf_url"),
        "combined_text": (item.get("combined_text") or "")[:18000],
        "provenance_sources": item.get("provenance_sources") or [],
        "classification_rules": {
            "likely_yes": [
                "Exp. 05935-2021-84 Sala Penal Superior",
                "Exp. 00201-2023-0-1601-JM-FT-01 Juzgado Mixto/Familia",
                "Exp. 01733-2019-0 Juzgado penal"
            ],
            "depends": [
                "Casación N° 1764-2022 Corte Suprema",
                "R.N. N° XXXX-20XX"
            ],
            "likely_no": [
                "Casación N° XXX/20XX-XX con doctrina vinculante pura",
                "Acuerdo Plenario N° X"
            ]
        }
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Devuelve SOLO JSON válido ajustado a este schema conceptual: "
                    "title, document_kind, facts_summary, legal_issue, holding, reasoning_summary, outcome, applicable_norms[], "
                    "violence_family_relevance, procedural_relevance, practical_context_summary, keywords[], llm_relevance_score, needs_review, "
                    "detected_resolution_label, detected_case_number, pdf_url_mapped, antecedent_signals{antecedent_relevance, antecedent_richness, has_factual_background, has_procedural_background, has_identifiable_parties, has_case_number_pattern, should_download_pdf, pdf_priority, antecedent_reason}.\n\n"
                    f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
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
    parsed = JurisprudenceSummaryModel.model_validate(json.loads(content))
    return {
        "entry_id": item.get("entry_id"),
        "index_page_id": item.get("index_page_id"),
        "entry_url": item.get("entry_url"),
        "pdf_url": item.get("pdf_url"),
        "provenance_sources": item.get("provenance_sources") or [],
        "context_unit": parsed.model_dump(mode="json"),
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
    print(json.dumps({
        "entries_total": len(outputs),
        "summarized_success": sum(1 for x in outputs if x.get('status') == 'success'),
        "antecedent_yes": sum(1 for x in outputs if (x.get('context_unit') or {}).get('antecedent_signals', {}).get('antecedent_relevance') == 'yes'),
        "pdf_download_recommended": sum(1 for x in outputs if (x.get('context_unit') or {}).get('antecedent_signals', {}).get('should_download_pdf') is True),
        "output_path": str(OUTPUT_JSON),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
