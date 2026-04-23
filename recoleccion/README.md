# Fase: Recolección

Esta fase agrupa todo el trabajo relacionado con la **adquisición, extracción, normalización y actualización de datos fuente** para la tesis.

## Objetivo

Construir pipelines confiables para obtener datos jurídicos peruanos y documentos procedimentales útiles para el dominio de violencia familiar, preservando trazabilidad y preparando insumos limpios para las fases posteriores de:

- generación de dataset
- entrenamiento

## Referencia metodológica principal

Antes de revisar los tipos de dato, conviene leer:

- `METODOLOGIA_DATOS.md`

Ese archivo define la clasificación funcional de fuentes usada en la fase de recolección.

## Organización interna

Dentro de esta fase, cada tipo de dato vive en su propio subárbol.

## Tipos de dato implementados

- `data_types/tipo_01_ley_30364/` → normativa principal del dominio
- `data_types/tipo_02_codigo_penal/` → normativa penal sustantiva complementaria
- `data_types/tipo_03_codigo_procesal_penal/` → normativa procesal penal complementaria

## Reproducción desde la raíz del repo

Desde `/` del repositorio:

```bash
uv sync
python3 run_recoleccion.py collect
```

O con fecha explícita:

```bash
python3 run_recoleccion.py collect 20250914
```

Para modo worker:

```bash
python3 run_recoleccion.py worker
```

## Pipeline base documental unificado

Desde `recoleccion/` o desde la raíz del repo se debe poder invocar un runner reproducible para la base documental.

Desde la raíz del repo:

```bash
python3 recoleccion/run_base_documental.py plan
```

Para ejecutar la muestra reproducible actual de jurisprudencia tipo 04:

```bash
python3 recoleccion/run_base_documental.py tipo04-sample10
```

Comandos disponibles actualmente para artefactos reales:

```bash
python3 recoleccion/run_base_documental.py tipo02-normalize
python3 recoleccion/run_base_documental.py tipo02-chunk
python3 recoleccion/run_base_documental.py tipo03-normalize
python3 recoleccion/run_base_documental.py tipo03-chunk
python3 recoleccion/run_base_documental.py tipo04-normalize
python3 recoleccion/run_base_documental.py tipo04-chunk
python3 recoleccion/run_base_documental.py tipo05-normalize
python3 recoleccion/run_base_documental.py tipo05-chunk
```

Además, sigue disponible la muestra reproducible actual de tipo 04:

```bash
python3 recoleccion/run_base_documental.py tipo04-sample10
```

## Documentación del schema unificado

Ver:

- `recoleccion/docs/unified-normalization-schema.md`

Ese documento define:
- schema unificado de `normalized_document`
- schema unificado de `document_chunk`
- mapeo de `tipo_01` a `tipo_05`
- orden del pipeline base documental

## Principio de diseño

La fase de recolección **no** debería mezclar objetivos de entrenamiento ni de generación de dataset. Aquí se resuelve principalmente:

- de dónde sale la data,
- cómo se descarga,
- cómo se limpia,
- cómo se mantiene actualizada,
- cómo se clasifica según su función jurídica,
- cómo se conserva en formatos humano/máquina.
