#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TYPES_DIR = ROOT / "data_types"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_JSONL = OUTPUT_DIR / "document_chunks_all.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "document_chunks_summary.json"

TYPE_PATHS = {
    "tipo_01_ley_30364": DATA_TYPES_DIR / "tipo_01_ley_30364" / "data" / "processed" / "document_chunks.jsonl",
    "tipo_02_codigo_penal": DATA_TYPES_DIR / "tipo_02_codigo_penal" / "data" / "processed" / "document_chunks.jsonl",
    "tipo_03_codigo_procesal_penal": DATA_TYPES_DIR / "tipo_03_codigo_procesal_penal" / "data" / "processed" / "document_chunks.jsonl",
    "tipo_04_jurisprudencia": DATA_TYPES_DIR / "tipo_04_jurisprudencia" / "data" / "processed" / "document_chunks.jsonl",
    "tipo_05_documentos_institucionales": DATA_TYPES_DIR / "tipo_05_documentos_institucionales" / "data" / "processed" / "document_chunks.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    per_type = {}
    missing = []

    for type_name, path in TYPE_PATHS.items():
        rows = read_jsonl(path)
        if not rows:
            missing.append(type_name)
        per_type[type_name] = len(rows)
        all_chunks.extend(rows)

    OUTPUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_chunks) + ("\n" if all_chunks else ""), encoding="utf-8")
    summary = {
        "chunks_total": len(all_chunks),
        "per_type": per_type,
        "missing_or_empty": missing,
        "output_path": str(OUTPUT_JSONL),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
