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

MODEL_NAME = "x-ai/grok-4.3"


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
    "sancion_penal_al_agresor",
    "doctrina_jurisprudencial",
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
    jurisprudencia_chunk_text: str = Field(
        description=(
            "Síntesis en 3-5 oraciones del criterio jurídico principal que establece esta resolución. "
            "NO menciones nombres propios, fechas específicas, ni hechos particulares del caso. "
            "Expresa el principio o estándar jurídico general que se establece, la decisión adoptada "
            "y su relevancia para casos de violencia familiar en Perú. "
            "Usa lenguaje jurídico conciso, no coloquial. "
            "Este texto reemplazará al chunk en la base vectorial RAG."
        )
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
    "institutional_protocol": "tipo_05_documentos_institucionales",
}

# Chunk designado como positivo principal para preguntas de canales_denuncia.
# El LLM referencia institutional_protocol con articles=[] y este chunk es el que se devuelve.
CANALES_ATENCION_CHUNK_ID = "chunk_tipo05_canales_atencion_vf_peru_01"


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
    by_source_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        source_type = chunk.get("source_type")
        metadata = chunk.get("metadata") or {}

        if chunk_id:
            by_chunk_id[chunk_id] = chunk

        if source_type:
            by_source_type[source_type].append(chunk)

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
        "by_source_type": by_source_type,
    }

def resolve_normative_refs(
    refs: list[PositiveContextRef],
    lookup: dict[str, Any],
    jurisprudence_chunks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    matched_chunks = []
    unresolved_refs = []

    by_source_article = lookup["by_source_article"]
    by_source_type = lookup["by_source_type"]

    for ref in refs:
        if ref.type == "unknown":
            continue

        # jurisprudence: usar los chunks pre-resueltos del caso (solo si el LLM lo pide explícitamente)
        if ref.type == "jurisprudence":
            if jurisprudence_chunks:
                matched_chunks.extend(jurisprudence_chunks)
            else:
                unresolved_refs.append(
                    {
                        "type": "jurisprudence",
                        "articles": [],
                        "reason": ref.reason,
                        "error": "Jurisprudencia no encontrada para este caso.",
                    }
                )
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

        # institutional_protocol con artículos vacíos: devolver el chunk de canales designado
        if not ref.articles:
            if ref.type == "institutional_protocol":
                by_chunk_id = lookup["by_chunk_id"]
                canales_chunk = by_chunk_id.get(CANALES_ATENCION_CHUNK_ID)
                if canales_chunk:
                    matched_chunks.append(canales_chunk)
                else:
                    # Fallback: todos los chunks del tipo si el designado no existe
                    chunks_for_type = by_source_type.get(source_type, [])
                    if chunks_for_type:
                        matched_chunks.extend(chunks_for_type)
                    else:
                        unresolved_refs.append(
                            {
                                "type": ref.type,
                                "source_type": source_type,
                                "articles": [],
                                "reason": ref.reason,
                                "error": "No se encontraron chunks institucionales.",
                            }
                        )
            else:
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

Tu tarea es:
A) Generar preguntas coloquiales que podría hacer una víctima de violencia familiar
   antes de decidir si denuncia o continúa un proceso legal.
B) Generar una síntesis jurídica abstracta del criterio de la resolución (campo jurisprudencia_chunk_text).

El público objetivo son víctimas que dudan por:
- vergüenza,
- desconocimiento de sus derechos,
- no saber cómo denunciar,
- miedo al agresor,
- dependencia económica,
- preocupación por sus hijos,
- desconfianza en el sistema judicial.

═══════════════════════════════════════════════════
PARTE A: REGLAS PARA LAS PREGUNTAS
═══════════════════════════════════════════════════

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
   Mapea todos los artículos de ley_30364, cp y cpp que responden la pregunta de manera eficaz.
   Sigue estrictamente la nomenclatura.

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
   - "1", "5", "8", "121-B", "122-A", "93", etc.

9. Puedes referenciar la jurisprudencia del caso explícitamente usando:
   type = "jurisprudence", articles = []
   El pipeline resolverá el chunk automáticamente. Úsalo SOLO cuando la respuesta a la
   pregunta proviene directamente de la doctrina de esta resolución (categoría doctrina_jurisprudencial).

10. Si no tienes suficiente base para inferir un artículo, usa:
    type = "unknown"
    articles = []

11. DISTRIBUCIÓN OBLIGATORIA DE PREGUNTAS

    Debes generar AL MENOS 1 pregunta de cada una de estas categorías obligatorias:

    ① identificacion_delito
       ¿Lo que me hicieron cuenta como violencia familiar? ¿Es delito aunque seamos pareja?
       → refs: artículos de ley_30364 (definición, tipos de violencia) y cp si aplica.

    ② canales_denuncia
       ¿Dónde puedo denunciar? ¿Cómo pido ayuda? ¿Hay un número al que llamar?
       → SIEMPRE incluye: {{"type": "institutional_protocol", "articles": [], "reason": "..."}}
         El sistema RAG tiene un chunk institucional con información sobre:
         Línea 100 (gratuita 24/7), Chat 100, Centros de Emergencia Mujer (CEM),
         comisarías PNP, Fiscalía de Familia, Juzgados de Familia.
         Puedes combinar con artículos de ley_30364 si corresponde (ej. art. 8).

    ③ medidas_proteccion
       ¿Pueden obligarle a salir de casa? ¿Qué medidas de protección puedo pedir?
       → refs: artículos de ley_30364 sobre medidas de protección.

    ④ sancion_penal_al_agresor
       ¿Qué le va a pasar a él si lo denuncio? ¿Va a ir preso? ¿Cuántos años de cárcel?
       → refs: artículos de cp (121-B, 122-B, 108-B, etc.) según el caso.

    ⑤ doctrina_jurisprudencial (OBLIGATORIA)
       Una pregunta coloquial cuya respuesta directa ES el principio jurídico que establece
       esta resolución. La pregunta debe sonar natural, como si la víctima hubiera vivido una
       situación similar y quisiera saber si el sistema la ampara.
       Inspírate en el Criterio principal y la Decisión o resultado del resumen del caso.
       Ejemplo: "¿Es verdad que si no me avisaron que cerraron la investigación, todavía puedo
       pedir ser parte del juicio?"
       → DEBES incluir: {{"type": "jurisprudence", "articles": [], "reason": "..."}}
       → NO uses esta categoría si la doctrina del caso es meramente procesal sin aplicación
         directa para una víctima de violencia familiar.

    Categorías contextuales (genera si el caso las justifica):
    ⑥ procedimiento_penal    — pasos y plazos del proceso judicial
    ⑦ pruebas_y_certificados — qué evidencia necesito presentar
    ⑧ hijos_y_tenencia       — si hay menores involucrados en el caso
    ⑨ reparacion_civil       — indemnización o reparación económica
    ⑩ violencia_psicologica  — si el caso incluye daño psicológico
    ⑪ desconfianza_en_justicia — miedos y dudas sobre el sistema

    Total esperado: entre 6 y 10 preguntas (el usuario puede indicar otro máximo en el prompt).

═══════════════════════════════════════════════════
PARTE B: REGLAS PARA jurisprudencia_chunk_text
═══════════════════════════════════════════════════

El campo jurisprudencia_chunk_text debe contener una síntesis de 3-5 oraciones con:
- El criterio o principio jurídico general que establece esta resolución.
- La decisión adoptada y su fundamento legal.
- Su relevancia o aplicación en contextos de violencia familiar.

PROHIBIDO en este campo:
- Nombres propios de personas (ni anonimizados).
- Fechas específicas del caso.
- Hechos particulares del expediente.

El texto debe ser abstracto, aplicable a casos similares futuros, en lenguaje jurídico conciso.

Ejemplo del nivel de abstracción esperado:
"La notificación de la disposición de conclusión de la investigación preparatoria es el hito
procesal que activa el plazo para que el agraviado solicite su constitución en actor civil.
Una solicitud presentada el mismo día de la notificación es oportuna y no puede declararse
extemporánea. Este criterio protege el derecho de defensa y el acceso a la justicia de las
víctimas de violencia familiar, garantizando que los plazos procesales no operen en su perjuicio."

═══════════════════════════════════════════════════

Responde estrictamente con este JSON Schema.
Ten mucho cuidado con los campos literal: debes respetarlos al 100%.

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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Retorna (records, chunk_update) donde chunk_update contiene la síntesis
    jurídica generada por el LLM para actualizar el text del chunk de jurisprudencia.
    """
    import datetime

    case_id = get_case_id(case_path, case_data)

    jurisprudence_chunks, jurisprudence_match_info = resolve_case_jurisprudence(
        case_data=case_data,
        lookup=lookup,
    )

    # Construir el chunk_update si el LLM generó síntesis y hay chunk matcheado
    chunk_update: dict[str, Any] = {}
    synthesis_text = (llm_output.jurisprudencia_chunk_text or "").strip()
    matched_jp_ids = jurisprudence_match_info.get("matched_chunk_ids", [])
    if synthesis_text and matched_jp_ids:
        for chunk_id in matched_jp_ids:
            chunk_update[chunk_id] = {
                "chunk_id": chunk_id,
                "new_text": synthesis_text,
                "case_id": case_id,
                "generated_at": datetime.datetime.utcnow().isoformat(),
            }

    records = []
    skipped_empty = 0

    for i, question in enumerate(llm_output.questions, start=1):
        normative_chunks, unresolved_refs = resolve_normative_refs(
            refs=question.positive_context_refs,
            lookup=lookup,
            jurisprudence_chunks=jurisprudence_chunks,
        )

        positive_chunks = deduplicate_chunks(normative_chunks)

        if not positive_chunks:
            skipped_empty += 1
            print(
                f"  [SKIP] {case_id}_q{i:03d} (topic={question.topic}): "
                "sin positivos resueltos — refs no encontradas en chunks."
            )
            continue

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

        # Campos EDA pre-calculados
        pos_source_types = list(dict.fromkeys(
            c.get("source_type", "") for c in positive_chunks if c.get("source_type")
        ))
        pos_articles = [
            (c.get("metadata") or {}).get("article_id")
            for c in positive_chunks
            if (c.get("metadata") or {}).get("article_id")
        ]

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

            # ── Campos EDA (pre-calculados para análisis directo con pandas/Counter) ──
            "query_length": len(question.query),
            "negative_count": len(hard_negative_candidates),
            "unresolved_count": len(unresolved_refs),
            "positive_source_types": pos_source_types,
            "positive_articles": pos_articles,
            "has_jurisprudence_positive": any(
                c.get("source_type") == "tipo_04_jurisprudencia" for c in positive_chunks
            ),
            "has_institutional_positive": any(
                c.get("source_type") == "tipo_05_documentos_institucionales" for c in positive_chunks
            ),
            "jurisprudence_matched": jurisprudence_match_info.get("status") == "matched",
            "neg_bm25_scores": [c.get("bm25_score", 0.0) for c in hard_negative_candidates],
            "negative_source_types": [c.get("source_type", "") for c in hard_negative_candidates],
        }

        records.append(record)

    if skipped_empty:
        print(f"  [INFO] Registros saltados por positivos vacíos: {skipped_empty}")

    return records, chunk_update
FINETUNING_PROMPT = "Represent this sentence for searching relevant passages: "


def build_training_records(
    records: list[dict[str, Any]],
    max_negatives: int = 10,
) -> list[dict[str, Any]]:
    """
    Genera el dataset de entrenamiento en el formato esperado por los frameworks
    de fine-tuning de embeddings (sentence-transformers, GTE, etc.).

    Una fila por query con pos y neg como listas de texto.
    """
    training = []

    for idx, record in enumerate(records):
        positives = record.get("positive_chunks", [])
        negatives = record.get("hard_negative_chunks", [])

        if not positives or not negatives:
            continue

        training.append(
            {
                "id": str(idx),
                "query": record["query"],
                "pos": [chunk["text"] for chunk in positives if chunk.get("text")],
                "neg": [
                    chunk["text"]
                    for chunk in negatives[:max_negatives]
                    if chunk.get("text")
                ],
                "prompt": FINETUNING_PROMPT,
                # Trazabilidad — ignorada por los trainers, útil para debug
                "dataset_id": record["dataset_id"],
                "case_id": record["case_id"],
                "topic": record["topic"],
                "barrier": record["barrier"],
            }
        )

    return training


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
    all_chunk_updates: dict[str, Any] = {}  # chunk_id → update entry

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

            records, chunk_update = build_dataset_records_for_case(
                case_path=case_path,
                case_data=case_data,
                llm_output=llm_output,
                lookup=lookup,
                bm25_index=bm25_index,
            )

            for record in records:
                append_jsonl(record, output_jsonl)

            all_chunk_updates.update(chunk_update)
            if chunk_update:
                print(f"[OK] Síntesis de chunk generada para: {list(chunk_update.keys())}")

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

    # Escribir archivo de actualizaciones de chunks de jurisprudencia
    updates_jsonl = PROJECT_ROOT / "data" / "intermediate" / "jurisprudencia_chunk_updates.jsonl"
    if all_chunk_updates:
        write_jsonl(list(all_chunk_updates.values()), updates_jsonl)
        print(f"\n[INFO] Síntesis de chunks escritas: {len(all_chunk_updates)} → {updates_jsonl}")
    else:
        print("\n[WARN] No se generaron síntesis de chunks (posiblemente ningún caso matcheó jurisprudencia).")

    # Generar archivo final de triples de entrenamiento (query / positive_text / negative_text)
    triples_jsonl = output_dir / "training_triples.jsonl"
    if output_jsonl.exists():
        with output_jsonl.open(encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]
        triples = build_training_records(all_records, max_negatives=10)
        write_jsonl(triples, triples_jsonl)
        print(f"[INFO] Triples de entrenamiento: {len(triples)} → {triples_jsonl}")
    else:
        print("[WARN] No se generaron triples: el dataset principal está vacío.")

    print("\n============================================================")
    print("[RESUMEN]")
    print(f"Total de preguntas generadas: {total_records}")
    print(f"Dataset: {output_jsonl}")
    print(f"Triples entrenamiento: {triples_jsonl}")
    print(f"Errores: {errors_jsonl}")
    print(f"Chunk updates: {updates_jsonl} ({len(all_chunk_updates)} entradas)")
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