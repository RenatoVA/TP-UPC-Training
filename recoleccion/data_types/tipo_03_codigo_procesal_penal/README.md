# Tipo de dato 03: Código Procesal Penal peruano

Este subárbol contiene el trabajo de recolección orientado a identificar artículos del **Código Procesal Penal peruano** potencialmente relevantes para violencia familiar y violencia de género.

## Clasificación metodológica

Según `recoleccion/METODOLOGIA_DATOS.md`, este tipo de dato corresponde a:

- **normativa procesal penal complementaria**

## Función en el sistema

- explicar denuncia, investigación y actuación institucional,
- complementar la dimensión sustantiva con procedimiento,
- responder dudas sobre trámite penal aplicable.

## Objetivo inmediato

Sin usar LLM todavía:

1. parsear el HTML del CPP desde LP Derecho,
2. extraer todos los artículos,
3. aplicar heurísticas deterministas para reducir candidatos,
4. dejar sobrevivientes en JSON para revisión humana.

## Fuente operativa actual

- `https://lpderecho.pe/nuevo-codigo-procesal-penal-peruano-actualizado/`

## Criterio de inclusión

- artículos relacionados con víctima, agraviado, denuncia, medidas de protección, prueba anticipada, investigación preliminar o preparatoria, actuación fiscal y policial,
- selección mediante heurísticas deterministas,
- revisión humana posterior.

## Flujo de esta etapa

- parsing HTML → `all_articles.json`
- scoring heurístico → `candidate_articles.json`
- revisión manual humana de sobrevivientes
