#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RECOLECCION_DIR = ROOT_DIR / "recoleccion"


def run_python(script_path: Path, cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = ["uv", "run", "python", str(script_path)]
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def cmd_plan() -> None:
    print("Pipeline base documental unificado")
    print("[1] tipo_01_ley_30364 -> normalización por artículos")
    print("[2] tipo_02_codigo_penal -> normalización por artículos")
    print("[3] tipo_03_codigo_procesal_penal -> normalización por artículos")
    print("[4] tipo_04_jurisprudencia -> fetch detail + resumen base documental por entry")
    print("[5] tipo_05_documentos_institucionales -> normalización por sección/bloque")
    print("[6] chunking unificado + consolidación global")
    print("\nVer esquema: recoleccion/docs/unified-normalization-schema.md")


def cmd_tipo04_sample10() -> None:
    type_dir = RECOLECCION_DIR / "data_types" / "tipo_04_jurisprudencia"
    src_dir = type_dir / "src"
    script_path = type_dir / "run_sample_10_pipeline.py"
    run_python(script_path=script_path, cwd=type_dir, extra_env={"PYTHONPATH": str(src_dir)})


def cmd_tipo04_normalize() -> None:
    type_dir = RECOLECCION_DIR / "data_types" / "tipo_04_jurisprudencia"
    src_dir = type_dir / "src"
    script_path = type_dir / "scripts" / "build_normalized_documents.py"
    run_python(script_path=script_path, cwd=type_dir, extra_env={"PYTHONPATH": str(src_dir)})


def cmd_tipo04_chunk() -> None:
    type_dir = RECOLECCION_DIR / "data_types" / "tipo_04_jurisprudencia"
    src_dir = type_dir / "src"
    script_path = type_dir / "scripts" / "build_document_chunks.py"
    run_python(script_path=script_path, cwd=type_dir, extra_env={"PYTHONPATH": str(src_dir)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrypoint reproducible desde recoleccion/ para el pipeline base documental."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("plan", help="Muestra el orden del pipeline base documental unificado.")
    subparsers.add_parser(
        "tipo04-sample10",
        help="Ejecuta la muestra reproducible de 10 índices de tipo_04_jurisprudencia desde la raíz de recoleccion.",
    )
    subparsers.add_parser(
        "tipo04-normalize",
        help="Construye normalized_documents.jsonl para tipo_04 desde los detail pages ya recolectados.",
    )
    subparsers.add_parser(
        "tipo04-chunk",
        help="Construye document_chunks.jsonl para tipo_04 desde normalized_documents.jsonl.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        cmd_plan()
        return
    if args.command == "tipo04-sample10":
        cmd_tipo04_sample10()
        return
    if args.command == "tipo04-normalize":
        cmd_tipo04_normalize()
        return
    if args.command == "tipo04-chunk":
        cmd_tipo04_chunk()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
