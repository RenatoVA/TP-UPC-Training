#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TYPE_DIR = Path(__file__).resolve().parent
SRC_DIR = TYPE_DIR / "src"
SCRIPTS_DIR = TYPE_DIR / "scripts"


def run_step(script_name: str) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(SCRIPTS_DIR / script_name),
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=TYPE_DIR,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(SRC_DIR),
        },
    )


def main() -> None:
    print("[1/4] Sampling 10 index pages...")
    run_step("sample_index_pages.py")

    print("[2/4] LLM parsing sampled index pages...")
    run_step("llm_extract_index_pages_batch.py")

    print("[3/4] Classifying jurisprudence entries...")
    run_step("classify_jurisprudence_entries.py")

    print("[4/4] Resolving and downloading PDFs...")
    run_step("resolve_and_download_pdfs.py")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
