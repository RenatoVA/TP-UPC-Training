# Ampara Legal RAG

Repositorio de trabajo para la tesis, organizado por fases en la raíz del proyecto.

## Estructura principal

- `recoleccion/` → adquisición y mantenimiento de fuentes
- `generacion_dataset/` → fase futura
- `entrenamiento/` → fase futura

## Estado actual

Por ahora, la fase implementada es:

- `recoleccion/`

Y dentro de ella, el tipo de dato disponible es:

- `recoleccion/data_types/tipo_01_ley_30364/`

## Ejecución reproducible desde la raíz

Se agregó un entrypoint Python en la raíz del repo:

- `run_recoleccion.py`

Antes de ejecutar, sincroniza el entorno:

```bash
uv sync
```

### Modo 1: solo recolectar data

```bash
python3 run_recoleccion.py collect
```

O con fecha explícita:

```bash
python3 run_recoleccion.py collect 20250914
```

Este modo:
- hace bootstrap de Ley 30364 si aún no existe,
- corre el escaneo exploratorio de El Peruano,
- guarda la data dentro de la fase de recolección.

### Modo 2: worker

```bash
python3 run_recoleccion.py worker
```

Este modo inicia el worker en bucle para la fase de recolección del tipo de dato 01.
La configuración de horario y proveedor depende del `.env` en la raíz del repo.

## Punto de entrada recomendado

Lee primero:

- `recoleccion/README.md`
- `recoleccion/data_types/tipo_01_ley_30364/README.md`
