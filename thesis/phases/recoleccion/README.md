# Fase: Recolección

Esta fase agrupa todo el trabajo relacionado con la **adquisición, extracción, normalización y actualización de datos fuente** para la tesis.

## Objetivo

Construir pipelines confiables para obtener datos jurídicos peruanos desde fuentes oficiales, preservando trazabilidad y preparando insumos limpios para las fases posteriores de:

- generación de dataset
- entrenamiento

## Organización interna

Dentro de esta fase, cada tipo de dato debe vivir en su propio subárbol.

## Tipos de dato

- `data_types/tipo_01_ley_30364/` → flujo específico para Ley 30364 y sus cambios asociados

## Principio de diseño

La fase de recolección **no** debería mezclar objetivos de entrenamiento ni de generación de dataset. Aquí se resuelve principalmente:

- de dónde sale la data,
- cómo se descarga,
- cómo se limpia,
- cómo se mantiene actualizada,
- cómo se conserva en formatos humano/máquina.
