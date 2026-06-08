"""
preprocesing_dataset.py
========================
Prepara el dataset de fine-tuning de embeddings a partir de training_triples.jsonl.

Estructura de entrada (training_triples.jsonl):
    {"id": str, "query": str, "pos": List[str] | str, "neg": List[str],
     "prompt": str, ...campos_extra}

Archivos de salida en entrenamiento/data/:
    training.json       → 90% de los datos (formato FlagEmbedding)
    test_queries.jsonl  → queries del 10% de test  {"id", "text"}
    corpus.jsonl        → todos los positivos       {"id", "text"}
    test_qrels.jsonl    → mapeo qid→docid           {"qid", "docid", "relevance"}

Uso:
    python entrenamiento/scripts/preprocesing_dataset.py \
        --input  generacion_dataset/output/legal_questions_dataset/training_triples.jsonl \
        --output entrenamiento/data \
        --test_size 0.1 \
        --seed 520
"""

import json
import random
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Carga un archivo .jsonl o .json (una lista o línea por objeto)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Intentar como JSON array primero
    if content.startswith("["):
        records = json.loads(content)
    else:
        # JSONL: una línea por objeto
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ {path}  ({len(records)} filas)")


def normalize_pos(pos_field) -> list[str]:
    """Garantiza que 'pos' sea siempre una lista de strings."""
    if isinstance(pos_field, str):
        return [pos_field]
    if isinstance(pos_field, list):
        # Aplanar si algún elemento es lista (raro pero posible)
        flat = []
        for item in pos_field:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [str(pos_field)]


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def main(input_path: Path, output_dir: Path, test_size: float, seed: int) -> None:

    print(f"\n{'='*55}")
    print(f"  Dataset preparation — FlagEmbedding fine-tuning")
    print(f"{'='*55}\n")

    # 1. Cargar dataset ──────────────────────────────────────
    print(f"[1/5] Cargando dataset desde: {input_path}")
    records = load_jsonl(input_path)
    print(f"      Total de filas: {len(records)}")

    # Normalizar 'pos' a lista en todos los registros
    for r in records:
        r["pos"] = normalize_pos(r["pos"])

    # 2. Split train / test ──────────────────────────────────
    print(f"\n[2/5] Dividiendo dataset (test_size={test_size}, seed={seed})")
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n_test  = max(1, int(len(shuffled) * test_size))
    n_train = len(shuffled) - n_test

    train_records = shuffled[:n_train]
    test_records  = shuffled[n_train:]

    print(f"      Train: {len(train_records)} filas")
    print(f"      Test : {len(test_records)}  filas")

    # 3. Guardar training.json ───────────────────────────────
    print(f"\n[3/5] Guardando training.json")

    # Campos esperados por FlagEmbedding; los extra se conservan
    training_out = output_dir / "training.json"
    training_out.parent.mkdir(parents=True, exist_ok=True)
    with open(training_out, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ {training_out}  ({len(train_records)} filas)")

    # 4. Construir corpus.jsonl ──────────────────────────────
    # El corpus usa TODOS los positivos del dataset completo
    # (igual que el notebook: corpus = ds.select_columns(["id","pos"]))
    print(f"\n[4/5] Construyendo corpus.jsonl (positivos de todo el dataset)")

    corpus_records = []
    seen_ids = set()
    for rec in records:
        rec_id = str(rec["id"])
        if rec_id in seen_ids:
            continue
        seen_ids.add(rec_id)

        # Un positivo por fila (el primero si hay varios)
        # Si necesitas indexar múltiples positivos, duplifica el id
        pos_texts = rec["pos"]
        if len(pos_texts) == 1:
            corpus_records.append({"id": rec_id, "text": pos_texts[0]})
        else:
            for i, txt in enumerate(pos_texts):
                corpus_records.append({"id": f"{rec_id}_{i}", "text": txt})

    write_jsonl(corpus_records, output_dir / "corpus.jsonl")

    # 5. Construir test_queries.jsonl y test_qrels.jsonl ─────
    print(f"\n[5/5] Construyendo archivos de evaluación")

    test_queries = []
    test_qrels   = []

    for rec in test_records:
        qid = str(rec["id"])
        test_queries.append({"id": qid, "text": rec["query"]})

        # qrels: cada query apunta a su positivo en el corpus
        pos_texts = rec["pos"]
        if len(pos_texts) == 1:
            test_qrels.append({"qid": qid, "docid": qid, "relevance": 1})
        else:
            for i in range(len(pos_texts)):
                test_qrels.append({"qid": qid, "docid": f"{qid}_{i}", "relevance": 1})

    write_jsonl(test_queries, output_dir / "test_queries.jsonl")
    write_jsonl(test_qrels,   output_dir / "test_qrels.jsonl")

    # ── Resumen ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Resumen final")
    print(f"{'='*55}")
    print(f"  training.json       → {len(train_records):>5} filas")
    print(f"  corpus.jsonl        → {len(corpus_records):>5} documentos")
    print(f"  test_queries.jsonl  → {len(test_queries):>5} queries")
    print(f"  test_qrels.jsonl    → {len(test_qrels):>5} relaciones")
    print(f"\n  Todos los archivos en: {output_dir.resolve()}\n")

    # Verificación rápida de integridad
    _verify(test_queries, test_qrels, corpus_records)


def _verify(queries: list, qrels: list, corpus: list) -> None:
    """Comprueba que todos los docids del qrel existen en el corpus."""
    corpus_ids = {r["id"] for r in corpus}
    missing = [q["docid"] for q in qrels if q["docid"] not in corpus_ids]
    if missing:
        print(f"  ⚠  ADVERTENCIA: {len(missing)} docids en qrels no encontrados en corpus.")
        print(f"     Primeros 5: {missing[:5]}")
    else:
        print(f"  ✓ Integridad OK: todos los docids del qrel están en el corpus.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara dataset para fine-tuning de modelos de embedding."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("generacion_dataset/data/output/legal_questions_dataset/training_triples.jsonl"),
        help="Ruta al archivo training_triples.jsonl de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("entrenamiento/data"),
        help="Directorio de salida para los archivos generados.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fracción del dataset para test (default: 0.1 → 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=520,
        help="Semilla para reproducibilidad del shuffle (default: 520).",
    )

    args = parser.parse_args()
    main(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        seed=args.seed,
    )