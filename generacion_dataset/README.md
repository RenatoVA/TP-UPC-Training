# Generación de dataset

Esta fase toma como insumos los artefactos curados de `recoleccion/` y prepara subconjuntos, candidatos y outputs para la construcción del dataset de entrenamiento.

## Estructura

- `data/input/` -> copias o snapshots de entrada desde recolección
- `data/intermediate/` -> subconjuntos filtrados y artefactos auxiliares
- `data/output/` -> datasets finales de entrenamiento/validación/prueba
- `scripts/` -> scripts de preparación y filtrado
- `docs/` -> notas metodológicas de esta fase

## Primer pipeline recomendado

```bash
python3 generacion_dataset/scripts/prepare_base_inputs.py
python3 generacion_dataset/scripts/filter_jurisprudence_candidates.py
```

## Artefactos esperados

- `data/input/jurisprudence_context_units.json`
- `data/input/document_chunks_all.jsonl`
- `data/intermediate/jurisprudence_vf_non_null.json`
- `data/intermediate/jurisprudence_vf_factrich.json`
- `data/intermediate/jurisprudence_pdf_candidates_217.json`
- `data/intermediate/document_chunks_tipo04_vf.jsonl`
- `data/intermediate/preparation_summary.json`
