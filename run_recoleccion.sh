#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TYPE_DIR="$ROOT_DIR/recoleccion/data_types/tipo_01_ley_30364"
SRC_DIR="$TYPE_DIR/src"
SCRIPTS_DIR="$TYPE_DIR/scripts"

usage() {
  cat <<'EOF'
Uso:
  ./run_recoleccion.sh collect [fecha_yyyymmdd]
  ./run_recoleccion.sh worker

Modos:
  collect   Ejecuta la lógica de recolección para el tipo de dato 01 (Ley 30364).
            Hace bootstrap si aún no existe y luego corre el escaneo exploratorio.
            Si no se pasa fecha, usa la fecha UTC actual en formato YYYYMMDD.

  worker    Inicia el worker en bucle para el tipo de dato 01 (Ley 30364).
            El horario se controla por el .env en la raíz del repo.
EOF
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv no está instalado o no está en PATH." >&2
    exit 1
  fi
}

run_collect() {
  local scan_date="${1:-$(date -u +%Y%m%d)}"
  echo "[1/2] Bootstrap Ley 30364 desde SPIJ"
  PYTHONPATH="$SRC_DIR" uv run python "$SCRIPTS_DIR/bootstrap_law_30364.py"

  echo "[2/2] Escaneo exploratorio El Peruano para fecha: $scan_date"
  PYTHONPATH="$SRC_DIR" uv run python "$SCRIPTS_DIR/daily_elperuano_scan.py" "$scan_date"
}

run_worker() {
  echo "Iniciando worker para Ley 30364..."
  PYTHONPATH="$SRC_DIR" uv run python "$SCRIPTS_DIR/run_daily_worker_loop.py"
}

main() {
  ensure_uv

  local mode="${1:-}"
  case "$mode" in
    collect)
      shift || true
      run_collect "${1:-}"
      ;;
    worker)
      run_worker
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "Modo no reconocido: $mode" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
