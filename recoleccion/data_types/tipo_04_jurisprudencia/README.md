# Tipo de dato 04: Jurisprudencia relevante

Este subárbol contiene la recolección inicial de jurisprudencia relevante asociada a los artículos identificados como candidatos en otras fuentes normativas.

## Clasificación metodológica

Según `recoleccion/METODOLOGIA_DATOS.md`, este tipo de dato corresponde a:

- **jurisprudencia relevante**

## Función en el sistema

- aportar contexto interpretativo,
- reforzar criterios de aplicación práctica,
- asociar resoluciones relevantes a artículos normativos previamente seleccionados,
- servir como proxy de casos jurídicos reales para la futura construcción de dataset sintético orientado a retrieval.

## Alcance actual

Este subárbol ya no debe leerse solo como una recolección plana de URLs. La dirección actual del trabajo es:

1. recolectar páginas índice de jurisprudencia detectadas en tipos 02 y 03,
2. parsear sus entries de forma estructurada,
3. clasificar cuáles entries sí apuntan a casos con antecedentes útiles,
4. resolver y descargar sus PDFs,
5. recién después enriquecer esos PDFs con `ResumeAI-Legal`.

## Prueba actual prioritaria

El hito inmediato no es procesar todo el corpus, sino una prueba controlada con **10 índices jurisprudenciales**.

La prueba debe llegar hasta:
- parseo LLM de categorías + entries,
- aplanado de entradas,
- clasificación de relevancia,
- resolución de PDF,
- descarga local de PDFs relevantes.

Schema de entrada/salida de esta prueba:
- `docs/subpipeline-10-indices-schema.md`

## Ejecución local recomendada

Desde esta misma carpeta puede correrse la prueba completa de 10 índices con un único runner:

```bash
cd recoleccion/data_types/tipo_04_jurisprudencia
uv run python run_sample_10_pipeline.py
```

Esto ejecuta:
- sample de 10 índices,
- parseo LLM-first,
- clasificación de entries,
- resolución y descarga de PDFs.

## Scripts activos vs experimentales

### Scripts activos
Se consideran parte del flujo principal o reutilizable:
- `build_jurisprudence_corpus.py`
- `parse_jurisprudence_corpus.py`
- `build_index_entries.py`
- `show_index_entries_stats.py`
- `show_jurisprudence_stats.py`
- `show_parsed_jurisprudence_stats.py`
- `enrich_case_with_resumeai_api_sample.py`

### Scripts experimentales
Los scripts de pruebas LLM o samples ad hoc fueron movidos a:
- `scripts/experimental/`

La idea es separar mejor:
- pipeline reutilizable,
- pruebas exploratorias,
- experimentos de prompting.
