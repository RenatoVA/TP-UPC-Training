# Tipo de dato 03: Código Procesal Penal peruano

Este subárbol contiene el trabajo de recolección orientado a identificar artículos del **Código Procesal Penal peruano** potencialmente relevantes para violencia familiar y violencia de género.

## Objetivo inmediato

Sin usar LLM todavía:

1. parsear el HTML del CPP desde LP Derecho,
2. extraer todos los artículos,
3. aplicar heurísticas deterministas para reducir candidatos,
4. dejar sobrevivientes en JSON para revisión humana.

## Fuente operativa actual

- `https://lpderecho.pe/nuevo-codigo-procesal-penal-peruano-actualizado/`

## Flujo de esta etapa

- parsing HTML → `all_articles.json`
- scoring heurístico → `candidate_articles.json`
- revisión manual humana de sobrevivientes
