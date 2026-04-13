# Tipo de dato 04: Jurisprudencia relevante

Este subárbol contiene la recolección inicial de jurisprudencia relevante asociada a los artículos identificados como candidatos en otras fuentes normativas.

## Clasificación metodológica

Según `recoleccion/METODOLOGIA_DATOS.md`, este tipo de dato corresponde a:

- **jurisprudencia relevante**

## Función en el sistema

- aportar contexto interpretativo,
- reforzar criterios de aplicación práctica,
- asociar resoluciones relevantes a artículos normativos previamente seleccionados.

## Alcance de esta primera implementación

En esta etapa no se clasifica jurisprudencia con LLM.
Se implementa un pipeline mínimo para:

1. recolectar URLs jurisprudenciales detectadas en tipos 02 y 03,
2. deduplicarlas,
3. descargar su HTML,
4. guardar metadata estructurada para revisión posterior.
