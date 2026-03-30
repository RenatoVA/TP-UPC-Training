# peru-legal-rag-data

Investigación y tooling para recolectar datos legales crudos de fuentes oficiales peruanas, con foco inicial en la Ley 30364 y sus modificaciones.

## Objetivo de esta fase

En esta primera fase **no** estamos usando LLMs para interpretar normas ni para consolidar versiones.
Estamos respondiendo algo más básico y más importante: **cómo extraer los datos crudos de manera confiable** desde las fuentes oficiales.

Fuentes iniciales bajo análisis:

- SPIJ libre: `https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/H1141065`
- Búsquedas El Peruano: `https://busquedas.elperuano.pe/`

## Hallazgos rápidos

- **SPIJ sí expone un backend reutilizable** con detalle normativo en JSON, pero requiere un token público/guest que la propia app web obtiene automáticamente.
- **El Peruano sí expone un endpoint GraphQL** reutilizable para búsquedas estructuradas.
- **Los PDFs del Peruano se pueden descargar directamente** desde URLs devueltas por su API.
- No vimos, en esta fase, un bloqueo serio tipo Cloudflare challenge que impida requests HTTP normales desde scripts.

## Estructura

- `docs/` → notas de investigación
- `scripts/` → scripts de exploración y prueba
- `data/` → muestras o salidas crudas cuando haga sentido

## Estado

Fase actual: **investigación de extracción cruda**.
Siguiente fase probable: pipeline incremental para detectar publicaciones relevantes y consolidar cambios sobre la Ley 30364.
