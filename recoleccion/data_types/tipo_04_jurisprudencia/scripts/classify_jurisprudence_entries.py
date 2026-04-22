#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

from legal_data.index_parser import parse_index_entries
from legal_data.storage import write_json

INPUT_JSON = Path("recoleccion/data_types/tipo_04_jurisprudencia/data/processed/input_indices_10.json")
ENTRIES_OUTPUT = Path("recoleccion/data_types/tipo_04_jurisprudencia/data/processed/jurisprudence_entries_llm_10.json")
FILTERED_OUTPUT = Path("recoleccion/data_types/tipo_04_jurisprudencia/data/processed/jurisprudence_entries_filtered_10.json")


def _predict_jurisprudence_type(text: str, category: str | None) -> str:
    value = f"{category or ''} {text}".lower()
    if any(k in value for k in ["juzgado mixto", "juzgado de familia", "jm-ft", "jr-ft"]):
        return "juzgado_familia_mixto"
    if any(k in value for k in ["juzgado penal", "juzgado unipersonal", "jip", "jpu"]):
        return "juzgado_penal"
    if any(k in value for k in ["sala penal superior", "sala superior", "corte superior"]):
        return "sala_superior"
    if "casación" in value or "casacion" in value:
        return "casacion"
    if any(k in value for k in ["r.n.", "rn ", "recurso de nulidad"]):
        return "recurso_nulidad"
    if "acuerdo plenario" in value:
        return "acuerdo_plenario"
    if any(k in value for k in ["doctrina vinculante", "doctrina legal"]):
        return "doctrina"
    if "exp." in value or "expediente" in value:
        return "expediente_no_clasificado"
    return "unknown"


def _classify_relevance(text: str, predicted_type: str) -> tuple[str, str, bool]:
    value = text.lower()

    if predicted_type in {"juzgado_familia_mixto", "juzgado_penal", "sala_superior"}:
        return "accepted", "órgano jurisdiccional con alta probabilidad de contener hechos y antecedentes del caso", False

    if predicted_type == "expediente_no_clasificado":
        return "accepted", "entry con referencia de expediente, probable caso concreto con contexto fáctico", False

    if predicted_type in {"casacion", "recurso_nulidad"}:
        return "conditional", "puede contener hechos, pero también puede ser mayormente doctrinal", True

    if predicted_type in {"acuerdo_plenario", "doctrina"}:
        return "rejected", "pieza doctrinal o plenaria sin garantía de antecedentes útiles para dataset", False

    if any(k in value for k in ["acuerdo plenario", "doctrina", "precedente vinculante"]):
        return "rejected", "texto con señales doctrinales no orientadas a caso concreto", False

    return "conditional", "clasificación incierta, requiere revisión posterior", True


def main() -> None:
    sampled_pages = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    parsed_pages = [
        {
            "id": item["index_page_id"],
            "jurisprudence_url": item["index_page_url"],
            "raw_html_path": item["raw_html_path"],
            "linked_articles": item.get("linked_articles", []),
        }
        for item in sampled_pages
    ]

    entries = parse_index_entries(parsed_pages)
    llm_like_entries = []
    filtered_entries = []

    for entry in entries:
        predicted_type = _predict_jurisprudence_type(
            entry.get("entry_title_candidate") or entry.get("entry_text") or "",
            entry.get("entry_category"),
        )
        relevance_status, relevance_reason, needs_llm_review = _classify_relevance(
            entry.get("entry_title_candidate") or entry.get("entry_text") or "",
            predicted_type,
        )

        llm_like = {
            "entry_id": entry["entry_id"],
            "index_page_id": entry["page_id"],
            "index_page_url": entry["page_source_url"],
            "article_id": entry.get("page_anchor_article_id"),
            "article_heading": entry.get("page_anchor_article_heading"),
            "entry_category": entry.get("entry_category"),
            "entry_title_raw": entry.get("entry_title_candidate") or entry.get("entry_text"),
            "entry_url": entry.get("lp_entry_url"),
            "resolution_reference_raw": entry.get("resolution_reference_raw"),
            "jurisprudence_type_hint": predicted_type,
            "parse_confidence": "high" if entry.get("lp_entry_url") else "medium",
        }
        llm_like_entries.append(llm_like)
        filtered_entries.append(
            {
                **llm_like,
                "jurisprudence_type_predicted": predicted_type,
                "relevance_status": relevance_status,
                "relevance_reason": relevance_reason,
                "needs_llm_review": needs_llm_review,
            }
        )

    write_json(ENTRIES_OUTPUT, llm_like_entries)
    write_json(FILTERED_OUTPUT, filtered_entries)
    summary = {
        "entries_total": len(filtered_entries),
        "accepted": sum(1 for x in filtered_entries if x["relevance_status"] == "accepted"),
        "conditional": sum(1 for x in filtered_entries if x["relevance_status"] == "conditional"),
        "rejected": sum(1 for x in filtered_entries if x["relevance_status"] == "rejected"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
