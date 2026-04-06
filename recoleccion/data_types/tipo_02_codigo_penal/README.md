# Tipo de dato 02: Código Penal peruano

Este subárbol contiene el trabajo de recolección orientado a identificar artículos del **Código Penal peruano** potencialmente relacionados con violencia familiar y violencia de género.

## Objetivo inmediato

No usar LLM todavía.
Primero se busca:

1. parsear el HTML de la fuente seleccionada,
2. extraer todos los artículos,
3. aplicar filtros heurísticos deterministas,
4. generar un subconjunto de candidatos para revisión humana.

## Fuente operativa actual

- `https://lpderecho.pe/codigo-penal-peruano-actualizado/`

## Flujo de esta etapa

- parsing HTML → `all_articles.json`
- scoring heurístico → `candidate_articles.json`
- revisión manual humana de sobrevivientes

## Nota metodológica

La fuente actual es operativa y útil para extracción estructurada, pero no sustituye por sí sola la fuente oficial del Estado. En esta etapa se prioriza eficiencia de recolección y reducción de candidatos.
