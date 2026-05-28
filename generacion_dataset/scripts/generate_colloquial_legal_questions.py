from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "input" / "document_chunks_all.jsonl"
DEFAULT_CASES_DIR = PROJECT_ROOT / "data" / "intermediate" / "resumeai_batch" / "cases"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "legal_questions_dataset"
DEFAULT_BM25_PATH = PROJECT_ROOT / "data" / "intermediate" / "bm25_document_chunks.pkl"

MODEL_NAME = "x-ai/grok-4.1-fast"


# ============================================================
# 2. SCHEMA DEL LLM
# ============================================================

LegalContextType = Literal[
    "ley_30364",
    "cp",
    "cpp",
    "jurisprudence",
    "institutional_protocol",
    "unknown",
]

BarrierType = Literal[
    "verguenza",
    "desconocimiento_derechos",
    "no_sabe_como_denunciar",
    "desconfianza_sistema",
    "miedo_agresor",
    "dependencia_economica",
    "preocupacion_hijos",
    "duda_si_es_delito",
    "otro",
]

QuestionTopic = Literal[
    "identificacion_delito",
    "canales_denuncia",
    "medidas_proteccion",
    "hijos_y_tenencia",
    "pruebas_y_certificados",
    "lesiones_secuelas_indemnizacion",
    "violencia_psicologica",
    "violencia_fisica",
    "violencia_economica",
    "violencia_sexual",
    "procedimiento_penal",
    "desconfianza_en_justicia",
    "reparacion_civil",
    "otro",
]


class PositiveContextRef(BaseModel):
    """
    Referencia que el LLM debe devolver para que el script pueda buscar
    los chunks positivos en document_chunks_all.jsonl.

    Para normas:
      - type: ley_30364, cp, cpp
      - articles: ["1", "5", "121-B", "122-A"]

    Para jurisprudencia:
      - el LLM no necesita proponerla.
      - el script la agregará determinísticamente usando resolution_reference_raw.
    """

    type: LegalContextType = Field(
        description="Tipo de fuente jurídica. Usar ley_30364, cp, cpp o institutional_protocol cuando corresponda."
    )
    articles: list[str] = Field(
        default_factory=list,
        description="Lista de artículos citados o inferidos. Ejemplo: ['121-B', '122-A']. Vacío si no aplica.",
    )
    reason: str = Field(
        description="Breve explicación de por qué esta referencia ayuda a responder la pregunta.",
    )


class GeneratedQuestion(BaseModel):
    query: str = Field(
        description=(
            "Pregunta coloquial que podría hacer una víctima de violencia familiar en Perú. "
            "Debe sonar natural, no técnica, y reflejar dudas, temor, vergüenza o desconfianza."
        )
    )
    normalized_query: str = Field(
        description="Versión corregida y más clara de la pregunta, manteniendo lenguaje ciudadano."
    )
    topic: QuestionTopic = Field(
        description="Tema principal de la pregunta."
    )
    barrier: BarrierType = Field(
        description="Barrera principal que explica por qué la persona no procede legalmente."
    )
    victim_voice_rationale: str = Field(
        description=(
            "Explicación breve de por qué esta pregunta es plausible para una víctima que duda "
            "en denunciar o proceder legalmente."
        )
    )
    positive_context_refs: list[PositiveContextRef] = Field(
        description=(
            "Referencias normativas que deberían resolver la pregunta. "
            "Usar artículos cuando sea posible. No incluir jurisprudencia del caso; "
            "esa será agregada por el script."
        )
    )
    expected_answer_focus: str = Field(
        description=(
            "Qué debería poder responder el sistema RAG usando los positivos. "
            "No responder la pregunta completa, solo indicar el foco esperado."
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confianza del LLM en que la pregunta y las referencias son pertinentes.",
    )


class CaseQuestionBatch(BaseModel):
    case_id: str = Field(description="Identificador del caso, normalmente derivado del PDF o archivo JSON.")
    case_summary_for_traceability: str = Field(
        description="Resumen breve y no sensacionalista del caso usado como base."
    )
    questions: list[GeneratedQuestion] = Field(
        min_length=1,
        max_length=12,
        description="Preguntas coloquiales generadas a partir del caso.",
    )


# ============================================================
# 3. UTILIDADES DE TEXTO
# ============================================================

def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9áéíóúñü\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [tok for tok in text.split() if len(tok) > 1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pydantic_schema_text(model_class: type[BaseModel]) -> str:
    """
    Compatibilidad Pydantic v1/v2.
    """
    if hasattr(model_class, "model_json_schema"):
        return json.dumps(model_class.model_json_schema(), ensure_ascii=False, indent=2)
    return json.dumps(model_class.schema(), ensure_ascii=False, indent=2)


# ============================================================
# 4. LECTURA DE CHUNKS Y BM25
# ============================================================

def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en línea {line_number}: {exc}") from exc

            chunks.append(chunk)

    return chunks


def build_index_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    topic_tags = metadata.get("topic_tags") or chunk.get("retrieval_tags") or []

    title = chunk.get("title") or ""
    text = chunk.get("text") or ""
    tags = " ".join(str(tag) for tag in topic_tags)

    return f"{title}. {text} {tags}".strip()


def build_or_load_bm25(
    chunks: list[dict[str, Any]],
    bm25_path: Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    if bm25_path.exists() and not rebuild:
        with bm25_path.open("rb") as f:
            return pickle.load(f)

    corpus_texts = [build_index_text(chunk) for chunk in chunks]
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    index = {
        "bm25": bm25,
        "chunks": chunks,
        "corpus_texts": corpus_texts,
        "tokenized_corpus": tokenized_corpus,
    }

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as f:
        pickle.dump(index, f)

    return index


def bm25_search(index: dict[str, Any], query: str, top_k: int = 8) -> list[dict[str, Any]]:
    bm25: BM25Okapi = index["bm25"]
    chunks: list[dict[str, Any]] = index["chunks"]

    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "source_type": chunk.get("source_type"),
                "title": chunk.get("title"),
                "article_id": (chunk.get("metadata") or {}).get("article_id"),
                "resolution_reference": (chunk.get("metadata") or {}).get("resolution_reference"),
            }
        )

    return results


# ============================================================
# 5. ÍNDICES DETERMINÍSTICOS PARA POSITIVE CONTEXT
# ============================================================

TYPE_TO_SOURCE_TYPE = {
    "ley_30364": "tipo_01_ley_30364",
    "cp": "tipo_02_codigo_penal",
    "cpp": "tipo_03_codigo_procesal_penal",
    "jurisprudence": "tipo_04_jurisprudencia",
    "institutional_protocol": "tipo_05_institutional_protocol",
}


def normalize_article_id(article_id: str | None) -> str:
    if not article_id:
        return ""

    value = str(article_id).strip()

    value = value.replace("Artículo", "")
    value = value.replace("artículo", "")
    value = value.replace("Art.", "")
    value = value.replace("art.", "")
    value = value.replace("°", "")
    value = value.replace("º", "")

    value = re.sub(r"\s+", "", value)
    value = value.upper()

    return value


def normalize_resolution_reference(value: str | None) -> str:
    if not value:
        return ""

    value = str(value).strip().upper()

    value = value.replace("Nº", "")
    value = value.replace("N°", "")
    value = value.replace("NO.", "")
    value = value.replace("EXP.", "")
    value = value.replace("RN", "")
    value = value.replace("R.N.", "")

    value = re.sub(r"\s+", "", value)
    value = value.strip("[]()")

    return value


def build_lookup_indexes(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Índices determinísticos basados estrictamente en:

    - source_type
    - metadata.article_id
    - metadata.resolution_reference

    No inferimos tipos. Usamos lo que ya viene en document_chunks_all.jsonl.
    """

    by_source_article: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_resolution_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_chunk_id: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        source_type = chunk.get("source_type")
        metadata = chunk.get("metadata") or {}

        if chunk_id:
            by_chunk_id[chunk_id] = chunk

        article_id = normalize_article_id(metadata.get("article_id"))
        if source_type and article_id:
            by_source_article[(source_type, article_id)].append(chunk)

        resolution_reference = normalize_resolution_reference(
            metadata.get("resolution_reference")
        )
        if source_type == "tipo_04_jurisprudencia" and resolution_reference:
            by_resolution_reference[resolution_reference].append(chunk)

    return {
        "by_source_article": by_source_article,
        "by_resolution_reference": by_resolution_reference,
        "by_chunk_id": by_chunk_id,
    }

def resolve_normative_refs(
    refs: list[PositiveContextRef],
    lookup: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    matched_chunks = []
    unresolved_refs = []

    by_source_article = lookup["by_source_article"]

    for ref in refs:
        if ref.type in {"jurisprudence", "unknown"}:
            continue

        source_type = TYPE_TO_SOURCE_TYPE.get(ref.type)

        if not source_type:
            unresolved_refs.append(
                {
                    "type": ref.type,
                    "articles": ref.articles,
                    "reason": ref.reason,
                    "error": "Tipo no mapeado a source_type.",
                }
            )
            continue

        if not ref.articles:
            unresolved_refs.append(
                {
                    "type": ref.type,
                    "source_type": source_type,
                    "articles": [],
                    "reason": ref.reason,
                    "error": "Referencia sin artículos.",
                }
            )
            continue

        for raw_article in ref.articles:
            article_id = normalize_article_id(raw_article)
            key = (source_type, article_id)

            chunks_for_ref = by_source_article.get(key, [])

            if chunks_for_ref:
                matched_chunks.extend(chunks_for_ref)
            else:
                unresolved_refs.append(
                    {
                        "type": ref.type,
                        "source_type": source_type,
                        "article": raw_article,
                        "normalized_article": article_id,
                        "reason": ref.reason,
                        "error": "No se encontró chunk con ese source_type y article_id.",
                    }
                )

    return deduplicate_chunks(matched_chunks), unresolved_refs

def resolve_case_jurisprudence(
    case_data: dict[str, Any],
    lookup: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Busca el chunk tipo_04_jurisprudencia usando la resolución del JSON resumido.

    Fuente:
    - source_trace.resolution_reference_raw
    - resumeai_run.resolution_reference_raw

    Match contra:
    - chunk["source_type"] == "tipo_04_jurisprudencia"
    - chunk["metadata"]["resolution_reference"]
    """

    source_trace = case_data.get("source_trace") or {}
    resumeai_run = case_data.get("resumeai_run") or {}

    raw_resolution = (
        source_trace.get("resolution_reference_raw")
        or resumeai_run.get("resolution_reference_raw")
        or ""
    )

    normalized_resolution = normalize_resolution_reference(raw_resolution)

    if not normalized_resolution:
        return [], {
            "raw_resolution_reference": raw_resolution,
            "normalized_resolution_reference": normalized_resolution,
            "status": "missing_resolution_reference",
        }

    chunks = lookup["by_resolution_reference"].get(normalized_resolution, [])

    if not chunks:
        return [], {
            "raw_resolution_reference": raw_resolution,
            "normalized_resolution_reference": normalized_resolution,
            "status": "not_found_in_document_chunks",
        }

    return chunks, {
        "raw_resolution_reference": raw_resolution,
        "normalized_resolution_reference": normalized_resolution,
        "status": "matched",
        "matched_chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
    }
def bm25_search_chunks(
    index: dict[str, Any],
    query: str,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    bm25: BM25Okapi = index["bm25"]
    chunks: list[dict[str, Any]] = index["chunks"]

    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]
        metadata = chunk.get("metadata") or {}

        results.append(
            {
                "rank": rank,
                "bm25_score": float(scores[idx]),
                "chunk_id": chunk.get("chunk_id"),
                "source_type": chunk.get("source_type"),
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title"),
                "article_id": metadata.get("article_id"),
                "resolution_reference": metadata.get("resolution_reference"),
                "text": chunk.get("text"),
            }
        )

    return results
def build_hard_negatives_from_bm25(
    bm25_index: dict[str, Any],
    query: str,
    positive_chunk_ids: set[str],
    top_k_search: int = 40,
    max_hard_negatives: int = 10,
) -> list[dict[str, Any]]:
    """
    Recupera chunks cercanos por BM25 y elimina los positivos.

    Resultado:
    - chunks lexicalmente relevantes,
    - pero no iguales a los positivos,
    - buenos candidatos para hard negative.
    """

    bm25_candidates = bm25_search_chunks(
        index=bm25_index,
        query=query,
        top_k=top_k_search,
    )

    hard_negatives = []

    for candidate in bm25_candidates:
        chunk_id = candidate.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in positive_chunk_ids:
            continue

        hard_negatives.append(candidate)

        if len(hard_negatives) >= max_hard_negatives:
            break

    return hard_negatives


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            continue

        seen.add(chunk_id)
        unique.append(chunk)

    return unique


def build_positive_context_text(chunks: list[dict[str, Any]]) -> str:
    blocks = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        block = {
            "chunk_id": chunk.get("chunk_id"),
            "source_type": chunk.get("source_type"),
            "title": chunk.get("title"),
            "article_id": metadata.get("article_id"),
            "resolution_reference": metadata.get("resolution_reference"),
            "text": chunk.get("text"),
        }
        blocks.append(json.dumps(block, ensure_ascii=False))

    return "\n\n".join(blocks)


# ============================================================
# 6. EXTRACCIÓN DEL RESUMEN DEL CASO
# ============================================================

def extract_case_payload_for_llm(case_data: dict[str, Any]) -> dict[str, Any]:
    source_trace = case_data.get("source_trace") or {}
    report = case_data.get("report") or {}
    legal_analysis = report.get("legal_analysis") or {}
    factual_density = case_data.get("factual_density_signals") or {}

    payload = {
        "source_trace": {
            "title": source_trace.get("title"),
            "resolution_reference_raw": source_trace.get("resolution_reference_raw"),
            "jurisprudence_type": source_trace.get("jurisprudence_type"),
            "pdf_filename": source_trace.get("pdf_filename"),
        },
        "executive_summary": report.get("executive_summary"),
        "legal_analysis": {
            "case_background": legal_analysis.get("case_background", []),
            "reported_acts": legal_analysis.get("reported_acts", []),
            "evidence_summary": legal_analysis.get("evidence_summary", []),
            "legal_reasoning": legal_analysis.get("legal_reasoning", []),
            "protective_measures_and_orders": legal_analysis.get("protective_measures_and_orders", []),
            "prosecutorial_or_procedural_outcomes": legal_analysis.get("prosecutorial_or_procedural_outcomes", []),
            "norms_cited": legal_analysis.get("norms_cited", []),
        },
        "factual_density_signals": factual_density,
    }

    return payload


def get_case_id(case_path: Path, case_data: dict[str, Any]) -> str:
    source_trace = case_data.get("source_trace") or {}
    pdf_filename = source_trace.get("pdf_filename")

    if pdf_filename:
        return Path(pdf_filename).stem

    return case_path.stem


# ============================================================
# 7. PROMPT Y LLAMADA AL LLM
# ============================================================

def build_system_prompt() -> str:
    schema = pydantic_schema_text(CaseQuestionBatch)

    return f"""
Eres un especialista en construcción de datasets para retrieval jurídico en Perú.

Tu tarea es generar preguntas coloquiales que podría hacer una víctima de violencia familiar
antes de decidir si denuncia o continúa un proceso legal.

El público objetivo son víctimas que dudan por:
- vergüenza,
- desconocimiento de sus derechos,
- no saber cómo denunciar,
- miedo al agresor,
- dependencia económica,
- preocupación por sus hijos,
- desconfianza en el sistema judicial.

Reglas:

1. Genera preguntas ciudadanas, coloquiales y naturales.
   No uses lenguaje jurídico como pregunta principal.

2. No inventes hechos fuera del caso resumido.

3. No respondas las preguntas.
   Solo genera preguntas y referencias normativas.

4. No generes chunk_id.
   El chunk_id será resuelto por el pipeline de manera determinística.

5. Para cada pregunta, incluye positive_context_refs con:
   - type
   - articles
   - reason
   - el objetivo de este campo es que mapees todos los articulos de la ley 30364, codigo penal y procesal penal que ayudan a responder la pregunta generada de manera eficaz
   - sigue estrictamente la nomenclatura

6. Los valores permitidos de type son:
   - ley_30364
   - cp
   - cpp
   - institutional_protocol
   - unknown

7. No uses source_type directamente.
   El pipeline mapeará:
   - ley_30364 → tipo_01_ley_30364
   - cp → tipo_02_codigo_penal
   - cpp → tipo_03_codigo_procesal_penal

8. Para articles usa solo IDs limpios:
   - "1"
   - "5"
   - "8"
   - "121-B"
   - "122-A"
   - "93"

9. No incluyas jurisprudencia en positive_context_refs.
   La jurisprudencia será agregada automáticamente usando resolution_reference del caso.

10. Si no tienes suficiente base para inferir un artículo, usa:
   type = "unknown"
   articles = []

11. Prioriza preguntas sobre:
   - si lo ocurrido es delito,
   - dónde denunciar,
   - qué pruebas sirven,
   - medidas de protección,
   - reparación civil o indemnización,
   - hijos, tenencia o protección familiar,
   - lesiones o secuelas,
   - violencia psicológica,
   - miedo o desconfianza en la justicia.

Responde estrictamente con este JSON Schema:

Ten mucho cuidado con los campos literal debes respetarlos al 100%

{schema}
""".strip()


def build_user_prompt(case_payload: dict[str, Any], n_questions: int) -> str:
    return f"""
Genera entre 6 y {n_questions} preguntas coloquiales basadas en este caso resumido.

Caso resumido en JSON:

{json.dumps(case_payload, ensure_ascii=False, indent=2)}
""".strip()


def call_llm_for_case(
    client: OpenAI,
    case_payload: dict[str, Any],
    n_questions: int,
    model: str,
) -> CaseQuestionBatch:
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(case_payload, n_questions)},
        ],
        stream=False,
        response_format={"type":"json_object"},
        extra_body={"reasoning": {"enabled": True}}
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("El modelo no devolvió contenido.")

    try:
        raw_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"El modelo no devolvió JSON válido. Content: {content[:500]}") from exc

    parsed = CaseQuestionBatch.model_validate(raw_json)

    return parsed


# ============================================================
# 8. CONSTRUCCIÓN DEL DATASET FINAL
# ============================================================

def build_dataset_records_for_case(
    case_path: Path,
    case_data: dict[str, Any],
    llm_output: CaseQuestionBatch,
    lookup: dict[str, Any],
    bm25_index: dict[str, Any],
) -> list[dict[str, Any]]:

    case_id = get_case_id(case_path, case_data)

    jurisprudence_chunks, jurisprudence_match_info = resolve_case_jurisprudence(
        case_data=case_data,
        lookup=lookup,
    )

    records = []

    for i, question in enumerate(llm_output.questions, start=1):
        normative_chunks, unresolved_refs = resolve_normative_refs(
            refs=question.positive_context_refs,
            lookup=lookup,
        )

        positive_chunks = deduplicate_chunks(
            normative_chunks + jurisprudence_chunks
        )

        positive_chunk_ids = {
            chunk["chunk_id"]
            for chunk in positive_chunks
            if chunk.get("chunk_id")
        }

        hard_negative_candidates = build_hard_negatives_from_bm25(
            bm25_index=bm25_index,
            query=question.query,
            positive_chunk_ids=positive_chunk_ids,
            top_k_search=40,
            max_hard_negatives=10,
        )

        record = {
            "dataset_id": f"{case_id}_q{i:03d}",
            "case_id": case_id,
            "source_file": str(case_path.relative_to(PROJECT_ROOT)),

            "query": question.query,
            "normalized_query": question.normalized_query,
            "topic": question.topic,
            "barrier": question.barrier,
            "victim_voice_rationale": question.victim_voice_rationale,
            "expected_answer_focus": question.expected_answer_focus,
            "confidence": question.confidence,

            "positive_context_refs": [
                ref.model_dump() if hasattr(ref, "model_dump") else ref.dict()
                for ref in question.positive_context_refs
            ],

            "positive_chunk_ids": sorted(positive_chunk_ids),

            "positive_chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_type": chunk.get("source_type"),
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "article_id": (chunk.get("metadata") or {}).get("article_id"),
                    "resolution_reference": (chunk.get("metadata") or {}).get("resolution_reference"),
                    "text": chunk.get("text"),
                }
                for chunk in positive_chunks
            ],

            "positive_context_text": build_positive_context_text(positive_chunks),
            "positive_context_count": len(positive_chunks),

            "hard_negative_chunk_ids": [
                candidate["chunk_id"]
                for candidate in hard_negative_candidates
            ],

            "hard_negative_chunks": hard_negative_candidates,

            "unresolved_positive_refs": unresolved_refs,
            "jurisprudence_match_info": jurisprudence_match_info,

            "source_trace": case_data.get("source_trace", {}),
        }

        records.append(record)

    return records
def explode_to_training_triples(
    grouped_records: list[dict[str, Any]],
    max_negatives_per_positive: int = 3,
) -> list[dict[str, Any]]:

    triples = []

    for record in grouped_records:
        query = record["query"]
        normalized_query = record["normalized_query"]

        positives = record.get("positive_chunks", [])
        negatives = record.get("hard_negative_chunks", [])

        if not positives or not negatives:
            continue

        for positive in positives:
            selected_negatives = negatives[:max_negatives_per_positive]

            for negative in selected_negatives:
                triples.append(
                    {
                        "dataset_id": record["dataset_id"],
                        "case_id": record["case_id"],
                        "query": query,
                        "normalized_query": normalized_query,
                        "topic": record["topic"],
                        "barrier": record["barrier"],

                        "positive_chunk_id": positive["chunk_id"],
                        "positive_source_type": positive["source_type"],
                        "positive_title": positive["title"],
                        "positive_text": positive["text"],

                        "hard_negative_chunk_id": negative["chunk_id"],
                        "hard_negative_source_type": negative["source_type"],
                        "hard_negative_title": negative["title"],
                        "hard_negative_text": negative["text"],
                        "hard_negative_bm25_score": negative["bm25_score"],
                    }
                )

    return triples


def process_cases(
    chunks_path: Path,
    cases_dir: Path,
    output_dir: Path,
    bm25_path: Path,
    model: str,
    n_questions: int,
    limit: int | None,
    rebuild_bm25: bool,
) -> None:

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"),base_url="https://openrouter.ai/api/v1")

    chunks = load_chunks(chunks_path)
    print(f"[INFO] Chunks cargados: {len(chunks)}")

    bm25_index = build_or_load_bm25(chunks, bm25_path=bm25_path, rebuild=rebuild_bm25)
    print(f"[INFO] BM25 listo: {bm25_path}")

    lookup = build_lookup_indexes(chunks)
    print("[INFO] Índices determinísticos listos.")

    case_paths = sorted(cases_dir.glob("*.json"))
    if limit is not None:
        case_paths = case_paths[:limit]

    print(f"[INFO] Casos a procesar: {len(case_paths)}")

    output_jsonl = output_dir / "questions_with_positive_context.jsonl"
    errors_jsonl = output_dir / "generation_errors.jsonl"

    if output_jsonl.exists():
        output_jsonl.unlink()

    if errors_jsonl.exists():
        errors_jsonl.unlink()

    total_records = 0

    for case_path in case_paths:
        print(f"\n[INFO] Procesando: {case_path.name}")

        try:
            case_data = read_json(case_path)

            factual_density = case_data.get("factual_density_signals") or {}
            if not factual_density.get("is_dataset_candidate", True):
                print(f"[WARN] Saltando caso no candidato: {case_path.name}")
                continue

            case_payload = extract_case_payload_for_llm(case_data)
            case_id = get_case_id(case_path, case_data)

            llm_output = call_llm_for_case(
                client=client,
                case_payload=case_payload,
                n_questions=n_questions,
                model=model,
            )

            # Reforzamos case_id correcto aunque el modelo devuelva otro.
            llm_output.case_id = case_id

            records = build_dataset_records_for_case(
            case_path=case_path,
            case_data=case_data,
            llm_output=llm_output,
            lookup=lookup,
            bm25_index=bm25_index,
            )

            for record in records:
                append_jsonl(record, output_jsonl)

            total_records += len(records)
            print(f"[OK] Preguntas generadas: {len(records)}")

        except (ValidationError, Exception) as exc:
            error_record = {
                "source_file": str(case_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            append_jsonl(error_record, errors_jsonl)
            print(f"[ERROR] Falló {case_path.name}: {exc}")

    print("\n============================================================")
    print("[RESUMEN]")
    print(f"Total de preguntas generadas: {total_records}")
    print(f"Dataset: {output_jsonl}")
    print(f"Errores: {errors_jsonl}")
    print("============================================================")


# ============================================================
# 9. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dataset de preguntas jurídicas coloquiales con positive context para contrastive learning."
    )

    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Ruta a document_chunks_all.jsonl.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=DEFAULT_CASES_DIR,
        help="Carpeta con JSONs resumidos por ResumeAI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida del dataset.",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=DEFAULT_BM25_PATH,
        help="Ruta donde se guarda/carga el índice BM25.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Modelo OpenAI a usar.",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=10,
        help="Máximo de preguntas por caso.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N casos. Útil para pruebas.",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="Reconstruye el índice BM25 aunque ya exista.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_cases(
        chunks_path=args.chunks_path,
        cases_dir=args.cases_dir,
        output_dir=args.output_dir,
        bm25_path=args.bm25_path,
        model=args.model,
        n_questions=args.n_questions,
        limit=args.limit,
        rebuild_bm25=args.rebuild_bm25,
    )